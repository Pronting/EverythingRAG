"""Chroma 向量库适配层（进程内持久化，余弦距离，零出网）。

对齐 base.py 的 VectorStore 契约：单一 collection（名称含嵌入模型指纹）、
元数据仅收标量(str/int/float/bool)、检索返回「块文本 + 元数据 + 相似度」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.schemas import BlockMetadata
from app.vectorstore.embedder import Embedder, FastEmbedEmbedder


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
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_ids(self, block_ids: list[str]) -> None:
        """按 block_id 删除（增量同步 / 重建索引用）。"""
        if block_ids:
            self._collection.delete(ids=block_ids)

    def delete_by_where(self, where: dict[str, Any]) -> None:
        """按元数据条件批量删除（如文件删除时按 source_file 级联清块）。"""
        if where:
            self._collection.delete(where=where)

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
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
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
        return self._collection.count()


def create_vector_store(
    persist_dir: Path,
    collection_name: str | None = None,
    embedder: Embedder | None = None,
) -> ChromaVectorStore:
    """工厂：创建 ChromaVectorStore。

    collection_name 缺省为 chunks__{embedder.fingerprint}__v1（真模型 = chunks__bge-m3__v1）；
    embedder 缺省 FastEmbedEmbedder（惰性，构造零加载零下载）。
    """
    if embedder is None:
        embedder = FastEmbedEmbedder()
    if collection_name is None:
        collection_name = f"chunks__{embedder.fingerprint}__v1"
    return ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
