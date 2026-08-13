"""HybridRetriever 混合检索测试：dense + BM25（标题注入）+ RRF 融合。

核心验收：标题关键词（实时风控方案 / ApiSix 网关）只在文件名、不在块文本时，
BM25 标题注入能把它救回 top-5；近似文档对由 RRF 融合排序。
全部 fake 依赖，零出网。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk


class FakeVectorStore:
    """实现 VectorStore Protocol：可控 list_blocks / query / count。"""

    def __init__(
        self,
        blocks: list[tuple[str, str, dict[str, Any]]] | None = None,
        query_result: list[dict[str, Any]] | None = None,
        count: int | None = None,
    ) -> None:
        self._blocks = blocks if blocks is not None else []
        self._query_result = query_result if query_result is not None else []
        self._count = count
        self.query_calls: list[tuple[list[float], int, dict[str, Any] | None]] = []

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        raise AssertionError("retrieval 不应触发 upsert")

    def delete_by_ids(self, block_ids: list[str]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_ids")

    def delete_by_where(self, where: dict[str, Any]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_where")

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        raise AssertionError("retrieval 不应触发 get_blocks_by_source")

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._blocks

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.query_calls.append((vector, top_k, where))
        return self._query_result

    def count(self) -> int:
        return self._count if self._count is not None else len(self._blocks)


def _block(block_id: str, text: str, source_file: str, heading: str | None = None) -> dict[str, Any]:
    """构造 list_blocks 的 (block_id, text, metadata)。"""
    metadata: dict[str, Any] = {
        "block_id": block_id,
        "doc_id": f"doc-{block_id}",
        "source_type": "document",
        "source_file": source_file,
        "platform": "local",
        "chunk_type": "text",
    }
    if heading is not None:
        metadata["heading_path"] = heading
    return (block_id, text, metadata)


def _hit(block_id: str, similarity: float, source_file: str = "a.md") -> dict[str, Any]:
    """构造 vectorstore.query 的命中 dict。"""
    return {
        "block_id": block_id,
        "text": f"text-{block_id}",
        "metadata": {"block_id": block_id, "source_file": source_file, "platform": "local"},
        "similarity": similarity,
        "distance": 1.0 - similarity,
    }


# 标题注入救回场景：目标文档「实时风控方案」的关键词只在文件名，块文本不含
_TITLE_BLOCKS = [
    _block("b1", "关于订单履约全链路的设计与流转", "订单履约全链路.md", "履约"),
    _block("b2", "统一缓存 sdk 的设计说明", "cider-基建2023.md", "基建"),
    _block("b3", "风控事件和拦截操作页面化，支持 userId 维度", "实时风控方案.md", "支持的策略"),
    _block("b4", "reCAPTCHA 功能调研", "Google调研.md"),
]
_TITLE_DENSE = [
    _hit("b1", 0.70, "订单履约全链路.md"),
    _hit("b4", 0.68, "Google调研.md"),
    _hit("b2", 0.65, "cider-基建2023.md"),
    _hit("b3", 0.55, "实时风控方案.md"),
]


def _retriever(
    store: FakeVectorStore,
    candidate_k: int = 20,
    context_top_k: int = 5,
    **kwargs: Any,
) -> HybridRetriever:
    return HybridRetriever(store, candidate_k=candidate_k, context_top_k=context_top_k, **kwargs)


# ---------------------------------------------------------------- 1. 标题注入救回


def test_title_injection_rescues_keyword_in_filename() -> None:
    """关键词只在文件名 -> BM25 标题注入把它带进 top-5（纯向量会排后面）。"""
    store = FakeVectorStore(blocks=_TITLE_BLOCKS, query_result=_TITLE_DENSE)
    retriever = _retriever(store)
    chunks = retriever.retrieve("实时风控方案支持哪些维度的限流策略", [1.0, 0.0, 0.0])

    ids = [c.block_id for c in chunks]
    assert "b3" in ids  # 实时风控方案.md 被救回
    assert ids[0] == "b3"  # RRF 融合后置顶


def test_bm25_lexical_rescue_exact_term() -> None:
    """查询含唯一术语，dense 未命中 -> BM25 词法命中带入。"""
    store = FakeVectorStore(
        blocks=[
            _block("x1", "普通 段落 内容", "a.md"),
            _block("x2", "唯一标识符 SPR-2024-0001 出现在此文档", "b.md"),
        ],
        query_result=[_hit("x1", 0.9, "a.md")],  # dense 只命中 x1
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("SPR-2024-0001", [1.0, 0.0, 0.0])
    assert any(c.block_id == "x2" for c in chunks)


# ---------------------------------------------------------------- 2. RRF 融合

def test_rrf_ranks_present_in_both_higher() -> None:
    """同时被 dense + BM25 命中的块 RRF 分数更高，排前面。"""
    store = FakeVectorStore(
        blocks=[
            _block("a", "埋点平台 生命周期 管理", "a.md"),
            _block("b", "缓存 设计", "b.md"),
            _block("c", "埋点 上报", "c.md"),
        ],
        query_result=[_hit("b", 0.95), _hit("a", 0.6), _hit("c", 0.5)],
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("埋点平台 埋点", [1.0, 0.0, 0.0])
    # a 同时 dense(rank2) + bm25 -> RRF 高于仅 dense rank1 的 b
    assert chunks[0].block_id == "a"


# ---------------------------------------------------------------- 3. 门禁协调


def test_min_similarity_filters_below_threshold_even_bm25_hit() -> None:
    """min_similarity=0.55：dense 相似度低于阈值的命中全部过滤（词法命中不豁免，堵负向泄漏）。"""
    store = FakeVectorStore(
        blocks=[_block("keep", "实时风控方案 限流维度", "实时风控方案.md")],
        query_result=[_hit("keep", 0.52, "实时风控方案.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    assert retriever.retrieve("实时风控方案 限流", [1.0, 0.0, 0.0]) == []


def test_bm25_rescue_kept_when_dense_sim_above_threshold() -> None:
    """BM25 把标题关键词块带进 top-5，且其 dense sim ≥ 阈值 -> 通过门禁保留。"""
    store = FakeVectorStore(
        blocks=[
            _block("rel", "实时风控方案 限流维度", "实时风控方案.md"),
            _block("noise", "缓存 设计 说明", "cider-基建.md"),
        ],
        query_result=[_hit("noise", 0.90, "cider-基建.md"), _hit("rel", 0.57, "实时风控方案.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    chunks = retriever.retrieve("实时风控方案 限流", [1.0, 0.0, 0.0])
    assert "rel" in [chunk.block_id for chunk in chunks]


def test_no_bm25_and_low_sim_returns_empty() -> None:
    """无关查询：无 BM25 命中 + dense 全低于阈值 -> 返回 []（门禁拦截）。"""
    store = FakeVectorStore(
        blocks=[_block("g", "量子 物理 科普", "q.md")],
        query_result=[_hit("g", 0.45, "q.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    assert retriever.retrieve("火星殖民地的生态系统", [1.0, 0.0, 0.0]) == []


# ---------------------------------------------------------------- 4. 守卫


@pytest.mark.parametrize("query_vector", [[], [0.0], [0.0, 0.0]])
def test_empty_query_raises_clear_error(query_vector: list[float]) -> None:
    store = FakeVectorStore(blocks=[_block("a", "内容", "a.md")])
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="空"):
        retriever.retrieve("问题", query_vector)
    assert store.query_calls == []


def test_unindexed_raises_clear_error() -> None:
    store = FakeVectorStore(blocks=[], count=0)
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="索引"):
        retriever.retrieve("问题", [1.0, 0.0, 0.0])
    assert store.query_calls == []


def test_no_hits_returns_empty() -> None:
    store = FakeVectorStore(blocks=[_block("a", "内容", "a.md")], query_result=[])
    retriever = _retriever(store)
    assert retriever.retrieve("问题", [1.0, 0.0, 0.0]) == []


def test_returns_retrieved_chunk_with_metadata() -> None:
    store = FakeVectorStore(
        blocks=[_block("b1", "文本 内容", "a.md", "标题")],
        query_result=[_hit("b1", 0.9, "a.md")],
    )
    retriever = _retriever(store)
    chunk = retriever.retrieve("文本", [1.0, 0.0, 0.0])[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.block_id == "b1"
    assert chunk.source_file == "a.md"
    assert chunk.heading_path == "标题"


# ---------------------------------------------------------------- 5. 零出网


def test_zero_outbound_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected outbound network call")

    monkeypatch.setattr(socket.socket, "connect", deny)
    store = FakeVectorStore(blocks=_TITLE_BLOCKS, query_result=_TITLE_DENSE)
    retriever = _retriever(store)
    chunks = retriever.retrieve("实时风控方案", [1.0, 0.0, 0.0])
    assert chunks
