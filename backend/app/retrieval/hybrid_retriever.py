"""混合检索器：dense（向量）+ BM25（词法，含标题注入）+ RRF 融合（优化③）。

解决纯向量检索的两类固有短板：
- 标题关键词（如「实时风控方案」「ApiSix 网关」）只在文件名、不在块文本时，
  BM25 把「文件名 + heading_path + 块文本」作为可搜索文本，词法直接命中。
- 近似文档对（同主题多文档）由 RRF 融合排序，结合两路信号而非单看余弦。

门禁：min_similarity 非 None 时，仅来自 dense 且相似度低于阈值的命中被过滤；
BM25 命中的块（词法强相关）不受该余弦阈值约束（否则标题救回会被门禁抵消）。
空查询 / 未建索引抛 RetrievalError；无匹配返回 []（非错误）。零出网。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk, _get_str
from app.vectorstore.base import VectorStore


class HybridRetriever:
    """dense top-candidate_k ∪ BM25 top-bm25_k -> RRF 融合 -> context_top_k。

    BM25 索引按需构建（首次 retrieve 时从 vectorstore 枚举全部块）；语料千级块
    建索引毫秒级，不影响服务启动。
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        candidate_k: int = 20,
        context_top_k: int = 5,
        bm25_k: int = 50,
        min_similarity: float | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._vectorstore = vectorstore
        self._candidate_k = candidate_k
        self._context_top_k = context_top_k
        self._bm25_k = bm25_k
        self._min_similarity = min_similarity
        self._rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._blocks: dict[str, tuple[str, dict[str, Any]]] = {}
        self._index_to_id: dict[int, str] = {}

    def retrieve(
        self,
        query_text: str,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索：dense + BM25（标题注入）RRF 融合取 top-context_k。"""
        self._reject_empty_query(query_vector)
        if self._vectorstore.count() == 0:
            raise RetrievalError("知识库尚未建立索引，请先导入并同步文档")

        dense_hits = self._vectorstore.query(query_vector, self._candidate_k, where=where)
        bm25 = self._ensure_bm25()
        bm25_hits = bm25.search(query_text or "", self._bm25_k)

        ordered, dense_sims = self._fuse(dense_hits, bm25_hits)
        results: list[RetrievedChunk] = []
        for block_id in ordered:
            if block_id not in self._blocks:
                continue
            text, metadata = self._blocks[block_id]
            dense_sim = dense_sims.get(block_id, 0.0)
            if self._min_similarity is not None and dense_sim < self._min_similarity:
                continue  # 相关度门禁：dense 相似度低于阈值即过滤（词法命中不豁免，堵负向泄漏）
            results.append(self._to_chunk(block_id, text, metadata, dense_sim))
            if len(results) >= self._context_top_k:
                break
        return results

    def _fuse(
        self,
        dense_hits: list[dict[str, Any]],
        bm25_hits: list[tuple[int, float]],
    ) -> tuple[list[str], dict[str, float]]:
        """RRF 融合：dense 排名 + BM25 排名各自 1/(k+rank) 累加，返回降序 block_id + dense 相似度。"""
        scores: dict[str, float] = {}
        dense_sims: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, 1):
            block_id = str(hit.get("block_id", ""))
            if not block_id:
                continue
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (self._rrf_k + rank)
            dense_sims[block_id] = float(hit.get("similarity", 0.0))
        for rank, (index, _score) in enumerate(bm25_hits, 1):
            block_id = self._index_to_id.get(index)
            if block_id is None:
                continue
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (self._rrf_k + rank)
        ordered = [block_id for block_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
        return ordered, dense_sims

    def _ensure_bm25(self) -> BM25Index:
        """按需构建 BM25 索引：块文本 + 标题注入（文件名 + heading_path）。"""
        if self._bm25 is None:
            texts: list[str] = []
            self._blocks = {}
            self._index_to_id = {}
            for index, (block_id, text, metadata) in enumerate(self._vectorstore.list_blocks()):
                title = _title_from_path(metadata.get("source_file", ""))
                heading = metadata.get("heading_path") or ""
                texts.append(f"{title} {heading} {text}".strip())
                self._blocks[block_id] = (text, metadata)
                self._index_to_id[index] = block_id
            self._bm25 = BM25Index(texts)
        return self._bm25

    @staticmethod
    def _reject_empty_query(query_vector: list[float]) -> None:
        """空列表 / 全零向量 -> 领域错误。"""
        if not query_vector or all(value == 0.0 for value in query_vector):
            raise RetrievalError("查询向量为空，请先对查询文本进行嵌入")

    @staticmethod
    def _to_chunk(
        block_id: str,
        text: str,
        metadata: dict[str, Any],
        similarity: float,
    ) -> RetrievedChunk:
        """元数据 + 相似度 -> RetrievedChunk（字段缺失取安全默认）。"""
        return RetrievedChunk(
            block_id=block_id,
            text=text,
            similarity=similarity,
            source_file=_get_str(metadata, "source_file", ""),
            platform=_get_str(metadata, "platform", "local"),
            chunk_type=_get_str(metadata, "chunk_type", "text"),
            anchor=_get_str(metadata, "anchor"),
            heading_path=_get_str(metadata, "heading_path"),
            metadata=dict(metadata),
        )


def _title_from_path(source_file: str) -> str:
    """从 source_file 提取文件名（去 .md 后缀）作为标题注入信号。"""
    name = Path(source_file).name
    return name[: -len(".md")] if name.lower().endswith(".md") else name
