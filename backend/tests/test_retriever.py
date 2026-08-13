"""VectorRetriever 检索管线测试（假 vectorstore 实现 VectorStore Protocol，零出网）。

隐私红线：不触碰真实 Chroma、不加载嵌入模型、无 socket/requests 调用；
另有阻断一切 socket 连接的用例实证全链路零出网。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk, VectorRetriever


class FakeVectorStore:
    """实现 VectorStore Protocol 的最小假实现：可配置块总数 / 查询结果 / 抛错。"""

    def __init__(
        self,
        count: int = 0,
        query_result: list[dict[str, Any]] | None = None,
    ) -> None:
        self._count = count
        self._query_result = query_result if query_result is not None else []
        # 记录每次 query 调用：(vector, top_k, where)，供断言 candidate_k / where 透传
        self.query_calls: list[tuple[list[float], int, dict[str, Any] | None]] = []

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        raise AssertionError("retrieval 不应触发 upsert")

    def delete_by_ids(self, block_ids: list[str]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_ids")

    def delete_by_where(self, where: dict[str, Any]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_where")

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        raise AssertionError("retrieval 不应触发 get_blocks_by_source")

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.query_calls.append((vector, top_k, where))
        return self._query_result

    def count(self) -> int:
        return self._count


def _block_dict(
    block_id: str,
    similarity: float,
    source_file: str = "a.md",
    platform: str = "kimi",
    chunk_type: str = "text",
    anchor: str | None = "anchor-x",
    heading_path: str | None = "s1/s2",
) -> dict[str, Any]:
    """构造符合 vectorstore.query 返回契约的块 dict（metadata 为标量 dict）。"""
    metadata: dict[str, Any] = {
        "block_id": block_id,
        "doc_id": f"doc-{block_id}",
        "source_type": "document",
        "source_file": source_file,
        "platform": platform,
        "chunk_type": chunk_type,
    }
    if anchor is not None:
        metadata["anchor"] = anchor
    if heading_path is not None:
        metadata["heading_path"] = heading_path
    return {
        "block_id": block_id,
        "text": f"text-{block_id}",
        "metadata": metadata,
        "similarity": similarity,
        "distance": 1.0 - similarity,
    }


def _retriever(
    store: FakeVectorStore,
    candidate_k: int = 20,
    context_top_k: int = 5,
    min_similarity: float | None = None,
) -> VectorRetriever:
    return VectorRetriever(
        store,
        candidate_k=candidate_k,
        context_top_k=context_top_k,
        min_similarity=min_similarity,
    )


# ---------------------------------------------------------------- 1. query -> top-k 带来源块


def test_query_returns_top_k_with_sources() -> None:
    """store 返回 >context_top_k 个块 -> retriever 裁剪到 context_top_k 个并完整提取来源。"""
    store = FakeVectorStore(
        count=6,
        query_result=[
            _block_dict("b1", 0.9),
            _block_dict("b2", 0.8),
            _block_dict("b3", 0.7),
            _block_dict("b4", 0.6),
            _block_dict("b5", 0.5),
            _block_dict("b6", 0.4),
        ],
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])

    assert len(chunks) == 5  # context_top_k 裁剪
    assert all(isinstance(chunk, RetrievedChunk) for chunk in chunks)
    first = chunks[0]
    assert first.block_id == "b1"
    assert first.text == "text-b1"
    assert first.similarity == pytest.approx(0.9)
    assert first.source_file == "a.md"
    assert first.platform == "kimi"
    assert first.chunk_type == "text"
    assert first.anchor == "anchor-x"
    assert first.heading_path == "s1/s2"
    assert isinstance(first.metadata, dict) and first.metadata["source_file"] == "a.md"


# ---------------------------------------------------------------- 2. candidate_k / context_top_k 语义


def test_candidate_k_and_context_top_k_semantics() -> None:
    """store 以 candidate_k=20 查询；结果按 similarity 降序裁剪到 context_top_k=5。"""
    # 输入乱序，验证重排
    store = FakeVectorStore(
        count=8,
        query_result=[
            _block_dict("b-low", 0.3),
            _block_dict("b-high", 0.98),
            _block_dict("b-mid", 0.6),
            _block_dict("b2", 0.85),
            _block_dict("b3", 0.8),
            _block_dict("b4", 0.75),
            _block_dict("b5", 0.7),
            _block_dict("b6", 0.65),
        ],
    )
    retriever = _retriever(store, candidate_k=20, context_top_k=5)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])

    assert store.query_calls[0][1] == 20  # candidate_k 传给 store
    assert [chunk.similarity for chunk in chunks] == [0.98, 0.85, 0.8, 0.75, 0.7]
    assert [chunk.block_id for chunk in chunks] == [
        "b-high",
        "b2",
        "b3",
        "b4",
        "b5",
    ]  # b-low 被裁剪


def test_tie_break_by_block_id_stable() -> None:
    """similarity 同分时按 block_id 字典序稳定排序。"""
    store = FakeVectorStore(
        count=3,
        query_result=[
            _block_dict("b-b", 0.7),
            _block_dict("b-a", 0.7),
            _block_dict("b-c", 0.7),
        ],
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["b-a", "b-b", "b-c"]


# ---------------------------------------------------------------- 3. 空查询 -> 清晰错误


@pytest.mark.parametrize("query_vector", [[], [0.0], [0.0, 0.0, 0.0], [0, 0]])
def test_empty_query_raises_clear_error(query_vector: list[float]) -> None:
    """空列表 / 全零向量 -> RetrievalError，消息可读，且不触发 store.query。"""
    store = FakeVectorStore(count=5, query_result=[_block_dict("b1", 0.9)])
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="空"):
        retriever.retrieve("查询文本",query_vector)
    assert store.query_calls == []


# ---------------------------------------------------------------- 4. 未建索引 -> 清晰错误


def test_unindexed_raises_clear_error() -> None:
    """store.count()==0 -> RetrievalError（含「未建索引」语义），且不触发 store.query。"""
    store = FakeVectorStore(count=0)
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="索引"):
        retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert store.query_calls == []


# ---------------------------------------------------------------- 5. 无匹配 -> []（非错误）


def test_no_match_returns_empty_list() -> None:
    """store 有数据但 query 返回空 -> 返回 []，不抛错。"""
    store = FakeVectorStore(count=5)  # query_result 缺省为空
    retriever = _retriever(store)
    assert retriever.retrieve("查询文本",[1.0, 0.0, 0.0]) == []
    assert store.query_calls[0][1] == 20


# ---------------------------------------------------------------- 6. where 透传


def test_where_passthrough() -> None:
    """retrieve 的 where 原样透传给 store.query。"""
    where = {"platform": "kimi", "source_file": "a.md"}
    store = FakeVectorStore(count=3, query_result=[_block_dict("b1", 0.9)])
    retriever = _retriever(store)
    retriever.retrieve("查询文本",[1.0, 0.0, 0.0], where=where)
    assert store.query_calls[0][2] == where


# ---------------------------------------------------------------- 6b. 相关度门控（min_similarity）


def test_min_similarity_drops_below_threshold() -> None:
    """min_similarity 设定后，相似度低于阈值的命中被过滤（不再进上下文）。"""
    store = FakeVectorStore(
        count=6,
        query_result=[
            _block_dict("b-high", 0.92),
            _block_dict("b-mid", 0.71),
            _block_dict("b-low", 0.42),
            _block_dict("b-lowest", 0.30),
        ],
    )
    retriever = _retriever(store, min_similarity=0.55)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["b-high", "b-mid"]
    assert store.query_calls[0][1] == 20  # 候选仍全量召回，门控发生在裁剪前


def test_min_similarity_all_below_returns_empty() -> None:
    """全部命中低于阈值 -> 返回 []（无相关上下文，调用方走常识路径）。"""
    store = FakeVectorStore(
        count=4,
        query_result=[_block_dict("b1", 0.45), _block_dict("b2", 0.38)],
    )
    retriever = _retriever(store, min_similarity=0.55)
    assert retriever.retrieve("查询文本",[1.0, 0.0, 0.0]) == []


def test_min_similarity_default_none_keeps_all() -> None:
    """min_similarity 缺省 None：不过滤（向后兼容，既有行为不变）。"""
    store = FakeVectorStore(
        count=3,
        query_result=[_block_dict("b1", 0.4), _block_dict("b2", 0.3)],
    )
    retriever = _retriever(store)  # 不传 min_similarity
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["b1", "b2"]


def test_min_similarity_boundary_inclusive() -> None:
    """恰等于阈值的命中保留（>= 语义）。"""
    store = FakeVectorStore(count=2, query_result=[_block_dict("b1", 0.55)])
    retriever = _retriever(store, min_similarity=0.55)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["b1"]


# ---------------------------------------------------------------- 7. 缺失字段安全默认 + 零出网


def test_missing_metadata_fields_get_safe_defaults() -> None:
    """metadata 缺字段时取安全默认：platform=local / chunk_type=text / anchor·heading_path=None。"""
    block = _block_dict("b1", 0.9)
    block["metadata"] = {"block_id": "b1", "source_file": "a.md"}  # 仅最小编码
    store = FakeVectorStore(count=2, query_result=[block])
    retriever = _retriever(store)
    chunk = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])[0]
    assert chunk.platform == "local"
    assert chunk.chunk_type == "text"
    assert chunk.anchor is None
    assert chunk.heading_path is None


def test_zero_outbound_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切 socket 连接，retrieve 全链路仍成功 => 零出网。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)

    store = FakeVectorStore(count=3, query_result=[_block_dict("b1", 0.9)])
    retriever = _retriever(store)
    chunks = retriever.retrieve("查询文本",[1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["b1"]


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
