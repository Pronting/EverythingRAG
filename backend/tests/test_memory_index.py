"""记忆索引与检索通路单测。

覆盖 v1.1.0 决策 D6 的核心风险：记忆通路必须**独立**于文档通路——
独立 collection、独立阈值、独立配额，且任何失败都不影响文档问答。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.memory.index_schema import (
    MEMORY_SCHEMA_VERSION,
    make_memory_fingerprint,
    memory_block_id,
    memory_collection_name,
)
from app.memory.indexer import MemoryIndexer, memory_index_text, memory_metadata
from app.memory.models import MemoryCategory, MemoryStatus
from app.memory.retriever import (
    DEFAULT_MEMORY_MIN_SIMILARITY,
    DEFAULT_MEMORY_TOP_K,
    MemoryRetriever,
    retrieve_memory_chunks,
)
from app.memory.store import MemoryStore
from tests.fakes import FakeEmbedder


class RecordingMemoryVectorStore:
    """记录写入/删除、可编程查询结果的向量库替身（记忆 collection 语义）。"""

    def __init__(self, hits: list[dict[str, Any]] | None = None, chunk_count: int | None = None) -> None:
        self.blocks: dict[str, tuple[str, list[float], dict[str, Any]]] = {}
        self.deleted: list[str] = []
        self._hits = hits or []
        # None = 由写入的块数推导；显式给值则覆盖（用于模拟"索引为空"等场景）。
        self._chunk_count = chunk_count
        self.fail_query = False

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        for block_id, text, vector, metadata in blocks:
            self.blocks[block_id] = (text, vector, dict(metadata))

    def delete_by_ids(self, block_ids: list[str]) -> None:
        self.deleted.extend(block_ids)
        for block_id in block_ids:
            self.blocks.pop(block_id, None)

    def delete_by_where(self, where: dict[str, Any]) -> None:
        for block_id in list(self.blocks):
            self.blocks.pop(block_id, None)
            self.deleted.append(block_id)

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {}

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [(key, value[0], value[2]) for key, value in self.blocks.items()]

    def query(
        self, vector: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.fail_query:
            raise RuntimeError("向量库暂时不可用")
        return self._hits[:top_k]

    def count(self) -> int:
        return len(self.blocks) if self._chunk_count is None else self._chunk_count

    def count_files(self) -> int:
        return len({value[2].get("source_file") for value in self.blocks.values()})

    def count_images(self) -> int:
        return 0

    def repair_ann_index(self) -> int:
        return 0


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def _entry(store: MemoryStore, statement: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    params: dict[str, Any] = {
        "statement": statement,
        "category": MemoryCategory.PREFERENCE,
        "confidence": 0.9,
        "source_conversation_id": "conv-1",
        "source_message_index": 4,
        "quote": "原话摘录",
    }
    params.update(kwargs)
    return store.add(**params)  # type: ignore[arg-type]


def _hit(entry_id: str, similarity: float, category: str = "preference") -> dict[str, Any]:
    return {
        "block_id": memory_block_id(entry_id),
        "text": "ignored",
        "similarity": similarity,
        "metadata": {"memory_id": entry_id, "category": category},
    }


# ---------------------------------------------------------------- 结构指纹


def test_fingerprint_is_composite_and_versioned() -> None:
    """记忆指纹 = 嵌入指纹 + 记忆结构版本（两者任一变化即换 collection）。"""

    fingerprint = make_memory_fingerprint("embed-fp")
    assert fingerprint == f"embed-fp::memory-v{MEMORY_SCHEMA_VERSION}"
    assert make_memory_fingerprint("other-fp") != fingerprint
    assert make_memory_fingerprint("embed-fp", schema_version=99) != fingerprint


def test_collection_name_is_separate_from_document_collection() -> None:
    """记忆 collection 前缀必须与文档 ``chunks__`` 区分开（同一 persist_dir）。"""

    name = memory_collection_name("embed-fp")
    assert name.startswith("memory__")
    assert "chunks__" not in name
    assert name.endswith(f"v{MEMORY_SCHEMA_VERSION}")


def test_memory_block_id_is_stable_and_prefixed() -> None:
    assert memory_block_id("abc") == memory_block_id("abc")
    assert memory_block_id("abc") != memory_block_id("abd")
    assert memory_block_id("abc").endswith("abc")


# ---------------------------------------------------------------- 索引文本与元数据


def test_index_text_is_statement_only(tmp_path: Path) -> None:
    """嵌入文本只含陈述，不拼类别前缀（避免类别词在全部记忆上重复稀释语义）。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    assert memory_index_text(entry) == "用户偏好 SQLite"
    assert entry.category.value not in memory_index_text(entry)


def test_metadata_is_scalar_and_carries_traceability(tmp_path: Path) -> None:
    """元数据必须全为 Chroma 可收的标量，并带回溯所需的源指针与摘录。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    metadata = memory_metadata(entry)

    assert all(
        isinstance(value, (str, int, float, bool)) for value in metadata.values()
    ), f"非标量元数据会污染记忆 collection: {metadata}"
    assert metadata["memory_id"] == entry.id
    assert metadata["source_ref"] == "conv-1#4"
    assert metadata["quote"] == "原话摘录"
    assert metadata["category"] == "preference"
    # 刻意不写 source_type：记忆不在 chunks collection，也不该扩展文档元数据 schema。
    assert "source_type" not in metadata


# ---------------------------------------------------------------- 索引写入


def test_upsert_indexes_retrievable_entries_only(tmp_path: Path) -> None:
    """只索引可检索条目：forgotten/pending 一律不进向量库。"""

    store = _store(tmp_path)
    active = _entry(store, "用户偏好 SQLite")
    forgotten = _entry(store, "用户偏好浅色主题")
    store.forget(forgotten.id)
    pending = _entry(store, "用户偏好 vim", confidence=0.2, min_confidence=0.6)
    assert pending.status is MemoryStatus.PENDING

    vectorstore = RecordingMemoryVectorStore()
    indexer = MemoryIndexer(FakeEmbedder(), vectorstore)

    indexed = indexer.upsert_entries([active, forgotten, pending], store=store)

    assert indexed == 1
    assert set(vectorstore.blocks) == {memory_block_id(active.id)}
    # 不可检索的条目被显式移出索引（而非静默留下旧向量）。
    assert memory_block_id(forgotten.id) in vectorstore.deleted
    assert memory_block_id(pending.id) in vectorstore.deleted
    text, _vector, metadata = vectorstore.blocks[memory_block_id(active.id)]
    assert text == "用户偏好 SQLite"
    assert metadata["memory_id"] == active.id


def test_upsert_without_store_trusts_passed_entries(tmp_path: Path) -> None:
    """不传 store 时按传入对象判定（调用方明确知道自己持有权威快照的场景）。"""

    store = _store(tmp_path)
    active = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore()
    indexer = MemoryIndexer(FakeEmbedder(), vectorstore)

    assert indexer.upsert_entries([active]) == 1
    assert set(vectorstore.blocks) == {memory_block_id(active.id)}


def test_upsert_with_store_ignores_stale_forgotten_snapshot(tmp_path: Path) -> None:
    """关键安全网：调用方持有的是"遗忘前"的旧对象时，传 store 仍必须拦下它。

    ``MemoryEntry`` 是不可变值对象，``forget()`` 返回新对象；若只按调用方手上的
    快照判定，已遗忘的记忆会被重新写回向量库。
    """

    store = _store(tmp_path)
    stale_snapshot = _entry(store, "用户偏好 SQLite")
    assert stale_snapshot.is_retrievable() is True

    store.forget(stale_snapshot.id)

    vectorstore = RecordingMemoryVectorStore()
    indexer = MemoryIndexer(FakeEmbedder(), vectorstore)
    indexed = indexer.upsert_entries([stale_snapshot], store=store)

    assert indexed == 0
    assert vectorstore.blocks == {}
    assert memory_block_id(stale_snapshot.id) in vectorstore.deleted


def test_upsert_empty_is_noop() -> None:
    vectorstore = RecordingMemoryVectorStore()
    assert MemoryIndexer(FakeEmbedder(), vectorstore).upsert_entries([]) == 0
    assert vectorstore.blocks == {}


def test_delete_entries_removes_prefixed_ids() -> None:
    vectorstore = RecordingMemoryVectorStore()
    MemoryIndexer(FakeEmbedder(), vectorstore).delete_entries(["a", "b"])
    assert vectorstore.deleted == [memory_block_id("a"), memory_block_id("b")]


def test_delete_entries_empty_is_noop() -> None:
    vectorstore = RecordingMemoryVectorStore()
    MemoryIndexer(FakeEmbedder(), vectorstore).delete_entries([])
    assert vectorstore.deleted == []


def test_reindex_entry_single(tmp_path: Path) -> None:
    """编辑陈述后单条重建索引（内容随之更新，块 ID 不变 -> upsert 幂等）。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore()
    indexer = MemoryIndexer(FakeEmbedder(), vectorstore)
    indexer.upsert_entries([entry])

    updated = store.update(entry.id, statement="用户偏好 SQLite 而非 Postgres")
    assert updated is not None
    assert indexer.reindex_entry(updated) == 1

    text, _vector, _metadata = vectorstore.blocks[memory_block_id(entry.id)]
    assert text == "用户偏好 SQLite 而非 Postgres"
    assert len(vectorstore.blocks) == 1


def test_sync_from_store_reports_indexed_and_removed(tmp_path: Path) -> None:
    """全量对齐：返回 {indexed, removed} 且失效条目被移出。"""

    store = _store(tmp_path)
    kept = _entry(store, "用户偏好 SQLite")
    dropped = _entry(store, "用户偏好浅色主题")
    store.forget(dropped.id)

    vectorstore = RecordingMemoryVectorStore()
    indexer = MemoryIndexer(FakeEmbedder(), vectorstore)
    # 先塞一条历史向量，模拟"索引早于遗忘"的滞后状态。
    indexer.upsert_entries([dropped])

    report = indexer.sync_from_store(store)

    assert report == {"indexed": 1, "removed": 1}
    assert set(vectorstore.blocks) == {memory_block_id(kept.id)}


# ---------------------------------------------------------------- 检索


def test_retrieve_returns_sql_authoritative_content(tmp_path: Path) -> None:
    """命中内容取自 SQL（权威源），而非向量库里的副本。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.8)])
    retriever = MemoryRetriever(vectorstore, store)

    matches = retriever.retrieve([1.0, 0.0])

    assert len(matches) == 1
    assert matches[0].statement == "用户偏好 SQLite"
    assert matches[0].similarity == 0.8
    assert matches[0].source_ref == "conv-1#4"
    assert matches[0].quote == "原话摘录"


def test_retrieve_filters_below_threshold(tmp_path: Path) -> None:
    """独立阈值生效：低于记忆阈值的命中被过滤。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.30)])
    retriever = MemoryRetriever(vectorstore, store, min_similarity=DEFAULT_MEMORY_MIN_SIMILARITY)
    assert retriever.retrieve([1.0, 0.0]) == []


def test_retrieve_threshold_is_independent_of_document_default() -> None:
    """记忆阈值不得与文档阈值耦合（D6）：两者是各自校准的常量。"""

    from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY

    assert DEFAULT_MEMORY_MIN_SIMILARITY != DEFAULT_MIN_SIMILARITY


def test_retrieve_respects_top_k_quota(tmp_path: Path) -> None:
    """配额生效：即使候选很多，返回条数不超过 top_k（记忆不挤占文档证据位）。"""

    store = _store(tmp_path)
    hits = []
    for index in range(6):
        entry = _entry(store, f"用户偏好第 {index} 项")
        hits.append(_hit(entry.id, 0.9 - index * 0.01))
    vectorstore = RecordingMemoryVectorStore(hits=hits)
    retriever = MemoryRetriever(vectorstore, store, top_k=2)

    assert len(retriever.retrieve([1.0, 0.0])) == 2
    assert DEFAULT_MEMORY_TOP_K < 5  # 必须小于文档 Top-K 缺省值


def test_retrieve_sorts_by_similarity_desc(tmp_path: Path) -> None:
    store = _store(tmp_path)
    low = _entry(store, "用户偏好 A")
    high = _entry(store, "用户偏好 B")
    vectorstore = RecordingMemoryVectorStore(
        hits=[_hit(low.id, 0.6), _hit(high.id, 0.95)]
    )
    retriever = MemoryRetriever(vectorstore, store)

    matches = retriever.retrieve([1.0, 0.0])
    assert [match.entry_id for match in matches] == [high.id, low.id]


def test_retrieve_excludes_forgotten_even_when_index_is_stale(tmp_path: Path) -> None:
    """关键安全网：索引滞后时，回表 SQL 仍必须拦下已遗忘的记忆。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.9)])
    retriever = MemoryRetriever(vectorstore, store)

    assert len(retriever.retrieve([1.0, 0.0])) == 1

    store.forget(entry.id)
    # 向量索引未更新（仍在 hits 里），但检索结果必须为空。
    assert retriever.retrieve([1.0, 0.0]) == []


def test_retrieve_excludes_orphaned_and_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    orphaned = _entry(store, "用户偏好 A")
    pending = _entry(store, "用户偏好 B", confidence=0.1, min_confidence=0.6)
    store.mark_source_orphaned("conv-1")

    vectorstore = RecordingMemoryVectorStore(
        hits=[_hit(orphaned.id, 0.9), _hit(pending.id, 0.9)]
    )
    assert MemoryRetriever(vectorstore, store).retrieve([1.0, 0.0]) == []


def test_retrieve_ignores_unknown_entry_ids(tmp_path: Path) -> None:
    """向量库里的孤儿向量（SQL 已无对应行）不得冒泡成来源。"""

    store = _store(tmp_path)
    vectorstore = RecordingMemoryVectorStore(hits=[_hit("ghost-id", 0.9)])
    assert MemoryRetriever(vectorstore, store).retrieve([1.0, 0.0]) == []


def test_retrieve_empty_vector_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit("x", 0.9)])
    retriever = MemoryRetriever(vectorstore, store)

    assert retriever.retrieve([]) == []
    assert retriever.retrieve([0.0, 0.0]) == []


def test_retrieve_empty_index_returns_empty(tmp_path: Path) -> None:
    """记忆库为空（无任何向量命中）时返回空列表。"""

    store = _store(tmp_path)
    vectorstore = RecordingMemoryVectorStore(hits=[], chunk_count=0)
    assert MemoryRetriever(vectorstore, store).retrieve([1.0, 0.0]) == []


def test_retrieve_swallows_vectorstore_failure(tmp_path: Path) -> None:
    """记忆通路失败必须静默降级：不得把异常抛给问答层。"""

    store = _store(tmp_path)
    _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore()
    vectorstore.fail_query = True
    assert MemoryRetriever(vectorstore, store).retrieve([1.0, 0.0]) == []


# ---------------------------------------------------------------- chunk 映射


def test_to_chunk_marks_memory_source_type(tmp_path: Path) -> None:
    """chunk_type='memory' 是前端渲染「提炼」徽章与回溯入口的唯一开关。"""

    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.9)])
    match = MemoryRetriever(vectorstore, store).retrieve([1.0, 0.0])[0]

    chunk = MemoryRetriever(vectorstore, store).to_chunk(match)

    assert chunk.chunk_type == "memory"
    assert chunk.match_type == "memory"
    assert chunk.text == "用户偏好 SQLite"
    assert chunk.anchor == "conv-1#4"
    assert chunk.source_file == "preference"
    assert chunk.metadata["source_ref"] == "conv-1#4"
    assert chunk.metadata["confidence"] == pytest.approx(0.9)


def test_retrieve_memory_chunks_without_retriever_returns_empty() -> None:
    """未配置记忆检索时（retriever=None）返回空列表，不报错。"""

    assert retrieve_memory_chunks(None, [1.0, 0.0]) == []


def test_retrieve_memory_chunks_maps_all_matches(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.9)])
    retriever = MemoryRetriever(vectorstore, store)

    chunks = retrieve_memory_chunks(retriever, [1.0, 0.0])
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "memory"


def test_retrieve_memory_chunks_never_raises(tmp_path: Path) -> None:
    """即便检索器内部抛出未预期异常，便捷入口也必须返回空列表。"""

    class Exploding:
        def retrieve(self, _vector: list[float]) -> list[Any]:
            raise RuntimeError("boom")

    assert retrieve_memory_chunks(Exploding(), [1.0, 0.0]) == []  # type: ignore[arg-type]
