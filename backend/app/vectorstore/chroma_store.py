"""Chroma 向量库适配层（进程内持久化，余弦距离，零出网）。

对齐 base.py 的 VectorStore 契约：单一 collection（名称含嵌入模型指纹）、
元数据仅收标量(str/int/float/bool)、检索返回「块文本 + 元数据 + 相似度」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.models.schemas import BlockMetadata
from app.vectorstore.embedder import Embedder


def _serialize_metadata(metadata: BlockMetadata) -> dict[str, Any]:
    """BlockMetadata -> Chroma 元数据 dict。

    source_type 序列化为字符串；丢弃 None 值（Chroma 不收 None）；
    其余字段均为标量(str/int/float/bool)直传。
    """
    raw: dict[str, Any] = metadata.model_dump()
    raw["source_type"] = metadata.source_type.value
    return {key: value for key, value in raw.items() if value is not None}


class ChromaVectorStore:
    """Chroma 持久化向量库：upsert / delete / query，按 block_id 幂等。

    客户端显式禁用匿名遥测（隐私红线：默认零出网）。
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _collection_guard(self, operation: Any) -> Any:
        """collection 句柄失效（被外部删除/重建）时自愈重建，再执行 operation。

        崩溃根因回归：Chroma PersistentClient 被并发进程写入同一 persist_dir 时，
        collection 可能被删除重建（内部 UUID 变化），旧句柄随即失效并抛 NotFoundError。
        读操作应自愈为「空库」、写操作应自愈重建后重试，而不是把异常抛给上层让
        /api/status 等服务入口崩溃。
        """
        try:
            return operation(self._collection)
        except NotFoundError:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return operation(self._collection)

    def upsert(
        self,
        blocks: list[tuple[str, str, list[float], BlockMetadata]],
    ) -> None:
        """按 block_id 幂等写入「向量 + 块文本 + 元数据」（同 id 重复 = 更新）。"""
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for block_id, text, vector, metadata in blocks:
            if len(vector) != self._embedder.dim:
                raise ValueError(
                    f"vector dim {len(vector)} != embedder dim {self._embedder.dim} "
                    f"(block {block_id})"
                )
            ids.append(block_id)
            documents.append(text)
            embeddings.append(vector)
            metadatas.append(_serialize_metadata(metadata))
        self._collection_guard(
            lambda c: c.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        )

    def delete_by_ids(self, block_ids: list[str]) -> None:
        """按 block_id 删除（增量同步 / 重建索引用）。"""
        if block_ids:
            self._collection_guard(lambda c: c.delete(ids=block_ids))

    def delete_by_where(self, where: dict[str, Any]) -> None:
        """按元数据条件批量删除（如文件删除时按 source_file 级联清块）。"""
        if where:
            self._collection_guard(lambda c: c.delete(where=where))

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        """返回某来源文件的全部块：{block_id: text}（增量同步块级比对用）。

        按 source_file 元数据过滤，只取 documents 不取向量（读便宜）。
        """
        result = self._collection_guard(
            lambda c: c.get(where={"source_file": source_file}, include=["documents"])
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        return dict(zip(ids, documents))

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        """枚举全部块：[(block_id, text, metadata)]（混合检索建 BM25 索引用）。

        只取文本 + 元数据不取向量（读便宜）；空库返回 []；collection 失效自愈为空。
        """
        result = self._collection_guard(lambda c: c.get(include=["documents", "metadatas"]))
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            (str(block_id), str(text), dict(meta or {}))
            for block_id, text, meta in zip(ids, documents, metadatas, strict=True)
        ]

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """余弦检索 top_k；空向量 / top_k<=0 返回 []。

        余弦空间 distance = 1 - cosine_similarity，故 similarity = 1 - distance。
        """
        if not vector or top_k <= 0:
            return []
        result = self._collection_guard(
            lambda c: c.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where,
            )
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        hits: list[dict[str, Any]] = []
        for index, block_id in enumerate(ids):
            distance = float(distances[index])
            hits.append(
                {
                    "block_id": block_id,
                    "text": documents[index],
                    "metadata": metadatas[index] or {},
                    "similarity": 1.0 - distance,
                    "distance": distance,
                }
            )
        return hits

    def count(self) -> int:
        """返回库中块总数；0 = 尚未建立索引。"""
        return self._collection_guard(lambda c: c.count())

    def count_files(self) -> int:
        """返回库中去重后的源文件数（读取全量元数据按 source_file 去重）。

        MVP 文件级计数：单个 collection 规模为百级文档，全量 get 元数据可接受。
        """
        result = self._collection_guard(lambda c: c.get(include=["metadatas"]))
        files = {
            metadata.get("source_file")
            for metadata in (result.get("metadatas") or [])
            if isinstance(metadata, dict)
        }
        files.discard(None)
        return len(files)

    def count_images(self) -> int:
        """返回库中图片描述块数（source_type=image_description；/api/status 展示用）。"""
        result = self._collection_guard(
            lambda c: c.get(where={"source_type": "image_description"}, include=["metadatas"])
        )
        return len(result.get("ids") or [])


def create_vector_store(
    persist_dir: Path,
    collection_name: str | None = None,
    embedder: Embedder | None = None,
) -> ChromaVectorStore:
    """工厂：创建 ChromaVectorStore。

    collection_name 缺省为 chunks__{embedder.fingerprint}__v1（云端 = chunks__cloud-{model}__v1）。
    embedder 必填（云端嵌入）；未提供时抛错，避免无嵌入器误建库。
    """
    if embedder is None:
        raise ValueError("嵌入器未提供：请先配置云端嵌入（base_url + model + api_key）")
    if collection_name is None:
        collection_name = f"chunks__{embedder.fingerprint}__v1"
    return ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
