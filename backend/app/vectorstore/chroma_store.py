"""Chroma 向量库适配层（进程内持久化，余弦距离，零出网）。

对齐 base.py 的 VectorStore 契约：单一 collection（名称含嵌入模型指纹）、
元数据仅收标量(str/int/float/bool)、检索返回「块文本 + 元数据 + 相似度」。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION
from app.models.schemas import BlockMetadata
from app.vectorstore.embedder import Embedder
from app.vectorstore.integrity import IndexIntegrityReport
from app.vectorstore.write_lock import exclusive_write_lock


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
        self._image_description_by_source: dict[str, tuple[str, str]] | None = None
        self._write_lock_path = self._persist_dir / ".chroma-write.lock"
        with exclusive_write_lock(self._write_lock_path):
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _collection_guard(self, operation: Any, *, write_lock_held: bool = False) -> Any:
        """collection 句柄失效（被外部删除/重建）时自愈重建，再执行 operation。

        崩溃根因回归：Chroma PersistentClient 被并发进程写入同一 persist_dir 时，
        collection 可能被删除重建（内部 UUID 变化），旧句柄随即失效并抛 NotFoundError。
        读操作应自愈为「空库」、写操作应自愈重建后重试，而不是把异常抛给上层让
        /api/status 等服务入口崩溃。读路径仅在需要自愈时获取跨进程写锁；写路径由
        调用方传入 ``write_lock_held=True``，复用外层锁，避免嵌套文件锁死锁。
        """
        try:
            return operation(self._collection)
        except NotFoundError:

            def recreate_and_retry() -> Any:
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return operation(self._collection)

            if write_lock_held:
                return recreate_and_retry()

            with exclusive_write_lock(self._write_lock_path):
                return recreate_and_retry()

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
        with exclusive_write_lock(self._write_lock_path):

            def write_and_verify(collection: Any) -> None:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                if not ids:
                    return
                top_k = min(max(1, collection.count()), 8)
                missing = _missing_ann_self_hits(
                    collection,
                    ids,
                    embeddings,
                    top_k=top_k,
                )
                # Chroma/HNSW can persist a logical record while omitting it from the ANN segment
                # after a bulk upsert. Re-upserting only the affected records repairs the segment
                # without re-embedding or touching unrelated blocks.
                for index in missing:
                    collection.upsert(
                        ids=[ids[index]],
                        documents=[documents[index]],
                        embeddings=[embeddings[index]],
                        metadatas=[metadatas[index]],
                    )
                if missing:
                    retry_ids = [ids[index] for index in missing]
                    retry_embeddings = [embeddings[index] for index in missing]
                    remaining = _missing_ann_self_hits(
                        collection,
                        retry_ids,
                        retry_embeddings,
                        top_k=min(max(1, collection.count()), 256),
                    )
                    if remaining:
                        raise RuntimeError("Chroma ANN 写入后仍有不可达记录")

            self._collection_guard(write_and_verify, write_lock_held=True)

    def delete_by_ids(self, block_ids: list[str]) -> None:
        """按 block_id 删除（增量同步 / 重建索引用）。"""
        if block_ids:
            with exclusive_write_lock(self._write_lock_path):
                self._collection_guard(
                    lambda c: c.delete(ids=block_ids),
                    write_lock_held=True,
                )

    def delete_by_where(self, where: dict[str, Any]) -> None:
        """按元数据条件批量删除（如文件删除时按 source_file 级联清块）。"""
        if where:
            with exclusive_write_lock(self._write_lock_path):
                self._collection_guard(
                    lambda c: c.delete(where=where),
                    write_lock_held=True,
                )

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

    def find_cached_image_description_by_source(self, src: str) -> tuple[str, str] | None:
        """Recover ``(content_hash, raw_description)`` from any retained collection.

        Collection version changes intentionally preserve old indexes. If an image URL later
        expires, its already-generated vision description can therefore be migrated into the
        active collection without downloading or analysing the image again.
        """

        if self._image_description_by_source is None:
            self._image_description_by_source = self._load_image_description_source_cache()
        return self._image_description_by_source.get(src)

    def _load_image_description_source_cache(self) -> dict[str, tuple[str, str]]:
        cache: dict[str, tuple[str, str]] = {}
        collections = self._client.list_collections()
        named: list[tuple[str, Any]] = []
        for item in collections:
            name = item if isinstance(item, str) else str(getattr(item, "name", ""))
            if not name:
                continue
            collection = self._collection if name == self._collection_name else item
            if isinstance(collection, str):
                collection = self._client.get_collection(name)
            named.append((name, collection))
        named.sort(key=lambda item: (item[0] != self._collection_name, item[0]))

        for _name, collection in named:
            try:
                result = collection.get(
                    where={"chunk_type": "image_description"},
                    include=["documents", "metadatas"],
                )
            except NotFoundError:
                continue
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for document, metadata in zip(documents, metadatas, strict=True):
                meta = dict(metadata or {})
                image_path = str(meta.get("image_path", "") or "")
                if not image_path or image_path in cache:
                    continue
                description = _raw_image_description(str(document or ""))
                if not description:
                    continue
                content_hash = str(meta.get("image_content_hash", "") or "")
                if not content_hash:
                    content_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
                cache[image_path] = (content_hash, description)
        return cache

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

    def repair_ann_index(self, *, max_rounds: int = 3) -> int:
        """补写逻辑记录存在、但生产 Top-8 ANN 路径不可达的节点。

        Chroma/HNSW 在连续多批 upsert 时，后续批次偶发使更早的节点脱离 ANN 路径；
        只校验当前批次不足以发现它。这里在一次导入/同步结束后扫描全库，并逐条
        upsert 不可达记录。每轮都会重新扫描全库，避免修复一个节点时挤掉另一个。
        返回本轮曾修复的去重记录数；达到轮数上限仍不可达则失败，不允许假健康。
        """

        if max_rounds <= 0:
            raise ValueError("max_rounds 必须大于 0")

        def repair(collection: Any) -> int:
            raw = collection.get(include=["documents", "embeddings", "metadatas"])
            ids = [str(block_id) for block_id in (raw.get("ids") or [])]
            documents = [str(text) for text in (raw.get("documents") or [])]
            raw_embeddings = raw.get("embeddings")
            embeddings = (
                [[float(value) for value in list(vector)] for vector in raw_embeddings]
                if raw_embeddings is not None
                else []
            )
            metadatas = [dict(meta or {}) for meta in (raw.get("metadatas") or [])]
            counts = {len(ids), len(documents), len(embeddings), len(metadatas)}
            if len(counts) != 1:
                raise RuntimeError("Chroma 逻辑记录字段数量不一致")
            if not ids:
                return 0

            repaired_ids: set[str] = set()
            for _ in range(max_rounds):
                missing = _missing_ann_self_hits(
                    collection,
                    ids,
                    embeddings,
                    top_k=min(len(ids), 8),
                )
                if not missing:
                    return len(repaired_ids)
                for index in missing:
                    collection.upsert(
                        ids=[ids[index]],
                        documents=[documents[index]],
                        embeddings=[embeddings[index]],
                        metadatas=[metadatas[index]],
                    )
                    repaired_ids.add(ids[index])

            remaining = _missing_ann_self_hits(
                collection,
                ids,
                embeddings,
                top_k=min(len(ids), 8),
            )
            if remaining:
                raise RuntimeError("Chroma ANN 全库修复后仍有不可达记录")
            return len(repaired_ids)

        with exclusive_write_lock(self._write_lock_path):
            return int(self._collection_guard(repair, write_lock_held=True))

    def inspect_integrity(self) -> IndexIntegrityReport:
        """执行离线完整性门禁；结果只含计数，不泄露块 ID、正文、路径或向量。

        此检查会读取全部 embedding，并用每条已存向量反查自身 ID。近似近邻索引
        不保证一次 ``n_results=count`` 能枚举全库，因此不能把“全库邻居少一条”误判
        为损坏；逐条自身命中才是与生产 top-k 查询一致的可达性契约。适合重建/发布
        验收，不应放在每次聊天请求或常规健康检查中。
        """

        logical_count = self.count()
        errors: list[str] = []
        try:
            raw = self._collection_guard(lambda c: c.get(include=["embeddings"]))
            ids = [str(block_id) for block_id in (raw.get("ids") or [])]
            raw_embeddings = raw.get("embeddings")
            embeddings = list(raw_embeddings) if raw_embeddings is not None else []
        except Exception as exc:  # noqa: BLE001 -- 完整性工具必须把损坏收敛为报告
            ids = []
            embeddings = []
            errors.append(type(exc).__name__)

        unique_id_count = len(set(ids))
        invalid_dimension_count = 0
        non_finite_embedding_count = 0
        if embeddings:
            try:
                expected_dim = self._embedder.dim
            except RuntimeError:
                # 新进程只读既有 collection 时 embedder 的惰性 dim 尚未解析；以首条
                # 已存向量为基准检查全库维度一致性，不为此制造完整性假失败。
                expected_dim = len(list(embeddings[0]))
            except Exception as exc:  # noqa: BLE001 -- 其它 dim 异常仍记类型
                expected_dim = -1
                errors.append(type(exc).__name__)
            for embedding in embeddings:
                vector = list(embedding)
                if expected_dim < 0 or len(vector) != expected_dim:
                    invalid_dimension_count += 1
                if any(not math.isfinite(float(value)) for value in vector):
                    non_finite_embedding_count += 1

        ann_ids: set[str] = set()
        logical_ids = set(ids)
        if logical_count and embeddings:
            pairs = list(zip(ids, embeddings, strict=False))
            missing: list[tuple[str, Any]] = []
            for block_id, embedding in pairs:
                try:
                    vector = [float(value) for value in list(embedding)]
                    ann = self._collection_guard(
                        lambda c, vector=vector: c.query(
                            query_embeddings=[vector],
                            n_results=min(logical_count, 8),
                            include=["distances"],
                        )
                    )
                    route_ids = [str(item) for item in (ann.get("ids") or [[]])[0]]
                    if len(route_ids) != len(set(route_ids)):
                        errors.append("ANNDuplicateIds")
                    if set(route_ids) - logical_ids:
                        errors.append("ANNUnexpectedIds")
                    if block_id in route_ids:
                        ann_ids.add(block_id)
                    else:
                        missing.append((block_id, embedding))
                except Exception as exc:  # noqa: BLE001 -- ANN 损坏收敛为报告
                    errors.append(type(exc).__name__)
            for block_id, embedding in missing:
                try:
                    vector = [float(value) for value in list(embedding)]
                    ann = self._collection_guard(
                        lambda c, vector=vector: c.query(
                            query_embeddings=[vector],
                            n_results=min(logical_count, 256),
                            include=["distances"],
                        )
                    )
                    route_ids = [str(item) for item in (ann.get("ids") or [[]])[0]]
                    if block_id in route_ids:
                        ann_ids.add(block_id)
                except Exception as exc:  # noqa: BLE001 -- ANN 损坏收敛为报告
                    errors.append(type(exc).__name__)
            if logical_ids - ann_ids:
                errors.append("ANNSelfProbeMissingIds")

        embedding_count = len(embeddings)
        unreadable = max(0, logical_count - embedding_count)
        ann_unique = len(ann_ids)
        healthy = (
            not errors
            and logical_count == unique_id_count == embedding_count
            and unreadable == 0
            and invalid_dimension_count == 0
            and non_finite_embedding_count == 0
            and ann_unique == logical_count
        )
        if logical_count == 0:
            errors.append("EmptyIndex")
            healthy = False
        return IndexIntegrityReport(
            healthy=healthy,
            logical_count=logical_count,
            unique_id_count=unique_id_count,
            embedding_count=embedding_count,
            unreadable_embedding_count=unreadable,
            invalid_dimension_count=invalid_dimension_count,
            non_finite_embedding_count=non_finite_embedding_count,
            ann_result_count=ann_unique,
            ann_unique_id_count=ann_unique,
            errors=tuple(dict.fromkeys(errors)),
        )


def _raw_image_description(document: str) -> str:
    """Strip retrieval-only heading/context prefixes from a stored image description block."""

    marker = "【图片描述】"
    if marker in document:
        return document.split(marker, 1)[1].strip()
    return document.strip()


def _missing_ann_self_hits(
    collection: Any,
    ids: list[str],
    embeddings: list[list[float]],
    *,
    top_k: int,
) -> list[int]:
    """Return input positions whose own vector cannot retrieve its ID; never expose the IDs."""

    missing: list[int] = []
    for index, (block_id, vector) in enumerate(zip(ids, embeddings, strict=True)):
        result = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["distances"],
        )
        route_ids = [str(item) for item in (result.get("ids") or [[]])[0]]
        if block_id not in route_ids:
            missing.append(index)
    return missing


def create_vector_store(
    persist_dir: Path,
    collection_name: str | None = None,
    embedder: Embedder | None = None,
) -> ChromaVectorStore:
    """工厂：创建 ChromaVectorStore。

    collection_name 缺省包含嵌入指纹和入库结构版本。解析/切片/向量表示升级会自动写入
    新 collection，避免继续读取旧 ANN 段；旧 collection 保留，可回滚且不会被原地破坏。
    embedder 必填（云端嵌入）；未提供时抛错，避免无嵌入器误建库。
    """
    if embedder is None:
        raise ValueError("嵌入器未提供：请先配置云端嵌入（base_url + model + api_key）")
    if collection_name is None:
        collection_name = f"chunks__{embedder.fingerprint}__v{INGESTION_SCHEMA_VERSION}"
    return ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
