"""混合检索器：dense + BM25（标题注入）+ RRF 融合（优化③）+ 文档级扩展。

- 标题关键词（如「实时风控方案」「ApiSix 网关」）只在文件名、不在块文本时，
  BM25 把「文件名 + heading_path + 块文本」作为可搜索文本，词法直接命中。
- 近似文档对（同主题多文档）由 RRF 融合排序，结合两路信号而非单看余弦。
- 门禁：min_similarity 非 None 时，dense 相似度低于阈值的命中被过滤（堵负向泄漏）。
- 文档级扩展：某文档有 1 块过门禁后，其同源兄弟块一并召回（锚点优先、兄弟补位），
  解决「问整篇文档只召回 1 块、上下文不全」的问题。
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
        candidate_k: int = 100,
        context_top_k: int = 8,
        bm25_k: int = 50,
        min_similarity: float | None = None,
        rrf_k: int = 60,
        expand_min_similarity: float = 0.68,
        lexical_floor: float = 0.45,
        bm25_rescue_k: int = 15,
        numeric_penalty: float = 0.10,
        dense_only_penalty: float = 0.05,
    ) -> None:
        self._vectorstore = vectorstore
        self._candidate_k = candidate_k
        self._context_top_k = context_top_k
        self._bm25_k = bm25_k
        self._min_similarity = min_similarity
        self._rrf_k = rrf_k
        self._expand_min_similarity = expand_min_similarity
        self._lexical_floor = lexical_floor
        self._bm25_rescue_k = bm25_rescue_k
        self._numeric_penalty = numeric_penalty
        self._dense_only_penalty = dense_only_penalty
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

        ordered, dense_sims, bm25_matched, bm25_hit = self._fuse(dense_hits, bm25_hits)
        return self._select_with_expansion(ordered, dense_sims, bm25_matched, bm25_hit)

    def _select_with_expansion(
        self,
        ordered: list[str],
        dense_sims: dict[str, float],
        bm25_matched: set[str],
        bm25_hit: set[str],
    ) -> list[RetrievedChunk]:
        """门控选出锚点块后，仅对「强锚点」同源文档做兄弟扩展。

        - 基础门禁（min_similarity）：相似度达标的块可入上下文；
        - 词法救援（lexical_floor）：短缩写（如 SPM/MQ/RFID）dense 相似度天然偏低，
          但 BM25 词法强命中，故对词法命中的块把阈值放宽到 lexical_floor；
        - 纯 dense 加罚（dense_only_penalty）：既无强词法、也无弱词法命中（仅 dense
          相似度高）的块，往往是与查询共享「业务/数据」等泛词的假阳性，需更高阈值；
        - 文档级扩展（expand_min_similarity）：只有相似度 ≥ 强阈值的块才触发
          同文档兄弟扩展，避免弱命中拖入整篇无关文档。
        """
        anchors: list[str] = []
        for block_id in ordered:
            if block_id not in self._blocks:
                continue
            dense_sim = dense_sims.get(block_id, 0.0)
            if self._min_similarity is not None:
                if block_id in bm25_matched and self._lexical_floor < self._min_similarity:
                    threshold = self._lexical_floor  # 强词法命中：救援
                elif block_id not in bm25_hit:
                    threshold = self._min_similarity + self._dense_only_penalty  # 纯 dense：加罚
                else:
                    threshold = self._min_similarity  # 弱词法命中：基础门槛
                # 数值噪声块（结构化定价/统计表，语义密度低）加罚：需更高相似度，
                # 避免「大模型价格表」这类 dense 噪声磁铁压过真正相关的块
                if _is_numeric_noise(self._blocks[block_id][0]):
                    threshold += self._numeric_penalty
                if dense_sim < threshold:
                    continue
            anchors.append(block_id)
        if not anchors:
            return []

        expansion_sections: set[tuple[str, str]] = set()
        for block_id in anchors:
            if dense_sims.get(block_id, 0.0) >= self._expand_min_similarity:
                expansion_sections.add(_section_key(self._blocks[block_id][1]))

        results: list[RetrievedChunk] = []
        seen: set[str] = set()

        def _append(block_id: str) -> None:
            text, metadata = self._blocks[block_id]
            results.append(self._to_chunk(block_id, text, metadata, dense_sims.get(block_id, 0.0)))
            seen.add(block_id)

        # 1. 锚点优先：真正相关（过门禁）的块先入上下文，避免被兄弟块挤出
        for block_id in anchors:
            if len(results) >= self._context_top_k:
                return results
            _append(block_id)

        # 2. 父子检索：仅强锚点所在「父节」的其余候选块（RRF 序）填满剩余名额
        if not expansion_sections:
            return results
        for block_id in ordered:
            if len(results) >= self._context_top_k:
                break
            if block_id in seen or block_id not in self._blocks:
                continue
            if _section_key(self._blocks[block_id][1]) not in expansion_sections:
                continue
            _append(block_id)
        return results

    def _fuse(
        self,
        dense_hits: list[dict[str, Any]],
        bm25_hits: list[tuple[int, float]],
    ) -> tuple[list[str], dict[str, float], set[str], set[str]]:
        """RRF 融合：dense 排名 + BM25 排名各自 1/(k+rank) 累加。

        返回降序 block_id + dense 相似度 + 强词法命中集（top-k）+ 全部词法命中集。
        """
        scores: dict[str, float] = {}
        dense_sims: dict[str, float] = {}
        bm25_matched: set[str] = set()
        bm25_hit: set[str] = set()
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
            bm25_hit.add(block_id)  # 全部词法命中（top-k）
            if rank <= self._bm25_rescue_k:
                bm25_matched.add(block_id)  # 仅强词法命中（top-k）触发词法救援
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (self._rrf_k + rank)
        ordered = [block_id for block_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
        return ordered, dense_sims, bm25_matched, bm25_hit

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


_NUMERIC_PUNCT = frozenset("；：，。、()（）$.,/[]{}:;|")


def _is_numeric_noise(text: str, ratio: float = 0.45, min_len: int = 50) -> bool:
    """是否为结构化数值噪声（定价/统计表）：数字+标点占比过高、语义密度低。

    这类块（如「大模型价格表」）是 dense 检索的噪声磁铁——对无关查询有虚高
    相似度，会压过真正相关的块。命中时在门控处加罚（需更高相似度）。
    """
    if not text or len(text) < min_len:
        return False
    n = sum(1 for c in text if c.isdigit() or c in _NUMERIC_PUNCT)
    return n / len(text) > ratio


def _section_key(metadata: dict[str, Any]) -> tuple[str, str]:
    """父子检索的「父节」键：命中子块时，返回同父节（同一上级标题）的上下文。

    ``heading_path`` 形如 ``A > B > C``，父节取去掉末段后的 ``A > B``；顶级/无标题
    父节为 ``""``（即整篇文档），与旧「同文档扩展」语义一致。
    """
    source = metadata.get("source_file", "") or ""
    heading = metadata.get("heading_path", "") or ""
    parts = heading.split(" > ")
    parent = " > ".join(parts[:-1]) if len(parts) > 1 else ""
    return (source, parent)
