from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import AgentKind, RunStatus
from app.agent.tools import (
    TOOL_FORBIDDEN_PATH_KEYS,
    ToolArgumentError,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolPermissionError,
    ToolStatus,
    ToolUnavailableError,
    build_default_registry,
    document_id_of_block,
    resolve_document_source,
)
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk
from tests.fakes import FakeEmbedder

ABS_SOURCE = "/abs/path/内部资料/报销制度.md"
ABS_IMAGE = "/abs/path/内部资料/截图.png"


# ---------------------------------------------------------------- 替身


class _BlockStore:
    """最小 ``VectorStore`` 替身：只实现工具层用到的方法。"""

    def __init__(
        self,
        blocks: list[tuple[str, str, dict[str, Any]]] | None = None,
        *,
        hits: list[dict[str, Any]] | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.blocks = list(blocks or [])
        self.hits = list(hits or [])
        self.list_error = list_error
        self.list_calls = 0
        self.queries: list[tuple[list[float], int, dict[str, Any] | None]] = []

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        pass

    def delete_by_ids(self, block_ids: list[str]) -> None:
        pass

    def delete_by_where(self, where: dict[str, Any]) -> None:
        pass

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {
            block_id: text
            for block_id, text, metadata in self.blocks
            if metadata.get("source_file") == source_file
        }

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return list(self.blocks)

    def query(
        self, vector: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append((vector, top_k, where))
        return self.hits[:top_k]

    def count(self) -> int:
        return len(self.blocks)

    def count_files(self) -> int:
        return len({meta.get("source_file") for *_rest, meta in self.blocks if meta.get("source_file")})

    def count_images(self) -> int:
        return sum(1 for *_rest, meta in self.blocks if meta.get("chunk_type") == "image_description")

    def repair_ann_index(self) -> int:
        return 0


class _SpyRetriever:
    """检索器替身：记录调用次数（A01 断言「本地正文检索次数为 0」）。"""

    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks or [])
        self.error = error
        self.calls: list[tuple[str, list[float]]] = []

    def retrieve(
        self, query_text: str, query_vector: list[float], where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query_text, query_vector))
        if self.error is not None:
            raise self.error
        return list(self.chunks)


@dataclass(frozen=True)
class _Area:
    """``WorkspaceKnowledgeArea`` 的最小同形状替身。"""

    context: Any
    retriever: Any
    vector_store: Any


class _StateStore:
    def __init__(self, count: int) -> None:
        self._count = count

    def load_all(self) -> dict[str, Any]:
        return {f"doc-{index}": object() for index in range(self._count)}


def _block(
    document_id: str,
    *,
    heading: str = "第一章",
    text: str = "正文",
    index: int = 0,
    source_file: str = ABS_SOURCE,
    chunk_type: str = "text",
    doc_revision: str = "rev-1",
    **extra: Any,
) -> tuple[str, str, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "source_file": source_file,
        "heading_path": heading,
        "anchor": heading,
        "chunk_type": chunk_type,
        "doc_id": doc_revision,
        "platform": "local",
    } | extra
    return (f"{document_id}::chunk-{index}", text, metadata)


def _chunk(
    document_id: str,
    text: str,
    *,
    similarity: float | None = 0.9,
    heading: str = "第一章",
    index: int = 0,
    chunk_type: str = "text",
    source_file: str = ABS_SOURCE,
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=f"{document_id}::chunk-{index}",
        text=text,
        similarity=similarity,
        source_file=source_file,
        platform="local",
        chunk_type=chunk_type,
        anchor=heading,
        heading_path=heading,
        metadata={
            "source_file": source_file,
            "heading_path": heading,
            "anchor": heading,
            "chunk_type": chunk_type,
            "doc_id": "rev-1",
            "platform": "local",
        },
    )


def _runtime_workspace(tmp_path: Path, *, workspace_id: str = "ws-a", name: str = "甲区") -> Any:
    store = WorkspaceStore(tmp_path)
    record = store.create(name, workspace_id=workspace_id)
    context = WorkspaceContextFactory(tmp_path, store).resolve(record.id)
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return context


def _context(
    workspace: Any,
    *,
    store: _BlockStore | None = None,
    retriever: Any = None,
    embedder: Any = None,
    app_store: AppStore | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    agent_kind: AgentKind | None = AgentKind.RAG,
    knowledge_area: Any = None,
    state_store: Any = None,
    image_reader: Any = None,
) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        app_store=app_store,
        embedder=embedder,
        retriever=retriever,
        knowledge_area=knowledge_area,
        vector_store=store,
        state_store=state_store,
        image_reader=image_reader,
        run_id=run_id,
        task_id=task_id,
        agent_kind=agent_kind,
    )


async def _seeded_store(workspace: Any, *, run_id: str = "run-a") -> AppStore:
    store = await AppStore.open_memory()
    await store.create_workspace(workspace.workspace_id, workspace.storage_key)
    conversation = await store.create_conversation(f"conv-{workspace.workspace_id}", workspace.workspace_id)
    await store.create_run(
        conversation_id=conversation.id,
        workspace_id=workspace.workspace_id,
        run_id=run_id,
        status=RunStatus.RUNNING,
    )
    return store


def _assert_no_local_paths(data: Any) -> None:
    """工具结果不得出现绝对路径与原图路径。"""

    text = json.dumps(data, ensure_ascii=False, default=str)
    assert "/abs/path" not in text
    assert ABS_SOURCE not in text
    assert ABS_IMAGE not in text
    if isinstance(data, dict):
        assert not (set(data) & TOOL_FORBIDDEN_PATH_KEYS)
        for value in data.values():
            _assert_no_local_paths(value)
    elif isinstance(data, list):
        for item in data:
            _assert_no_local_paths(item)


# =====================================================================
# 1. lookup_catalog
# =====================================================================


async def test_lookup_catalog_matches_documents_with_positions_and_paging(tmp_path: Path) -> None:
    registry = build_default_registry()
    store = _BlockStore(
        [
            _block("doc-a", heading="第一章 报销范围", index=0),
            _block("doc-a", heading="第二章 报销时限", index=1),
            _block("doc-b", heading="第一章 差旅标准", index=0, source_file="/abs/path/差旅标准.md"),
        ]
    )
    context = _context(_runtime_workspace(tmp_path), store=store)

    page1 = await registry.execute(ToolName.LOOKUP_CATALOG, {"query": "报销", "limit": 1}, context)

    assert page1.status is ToolStatus.OK
    assert page1.data["total"] == 1
    assert page1.data["next_cursor"] is None
    document = page1.data["documents"][0]
    assert document["document_id"] == "doc-a"
    assert document["name"] == "报销制度.md"  # 只暴露 basename
    assert document["chunk_count"] == 2
    assert "第一章 报销范围" in document["headings"]
    assert document["positions"][0]["heading_path"] == "第一章 报销范围"
    assert document["matched_terms"] == ["报销"]
    _assert_no_local_paths(page1.data)


async def test_lookup_catalog_paginates_with_opaque_cursor(tmp_path: Path) -> None:
    registry = build_default_registry()
    store = _BlockStore(
        [
            _block("doc-a", heading="报销范围", index=0),
            _block("doc-b", heading="报销时限", index=1),
            _block("doc-c", heading="报销流程", index=2),
        ]
    )
    context = _context(_runtime_workspace(tmp_path), store=store)

    first = await registry.execute(ToolName.LOOKUP_CATALOG, {"query": "报销", "limit": 2}, context)
    second = await registry.execute(
        ToolName.LOOKUP_CATALOG,
        {"query": "报销", "limit": 2, "cursor": first.data["next_cursor"]},
        context,
    )

    assert first.data["total"] == 3
    assert [item["document_id"] for item in first.data["documents"]] == ["doc-a", "doc-b"]
    assert first.data["next_cursor"] == "2"
    assert [item["document_id"] for item in second.data["documents"]] == ["doc-c"]
    assert second.data["next_cursor"] is None


async def test_lookup_catalog_returns_empty_status_without_match(tmp_path: Path) -> None:
    registry = build_default_registry()
    store = _BlockStore([_block("doc-a", heading="报销范围")])
    context = _context(_runtime_workspace(tmp_path), store=store)

    result = await registry.execute(
        ToolName.LOOKUP_CATALOG, {"query": "完全不相关的词"}, context
    )

    assert result.status is ToolStatus.EMPTY
    assert result.data["documents"] == []
    assert result.data["total"] == 0
    assert result.data["next_cursor"] is None


async def test_lookup_catalog_rejects_bad_cursor(tmp_path: Path) -> None:
    registry = build_default_registry()
    store = _BlockStore([_block("doc-a", heading="报销范围")])
    context = _context(_runtime_workspace(tmp_path), store=store)

    with pytest.raises(ToolArgumentError) as info:
        await registry.execute(
            ToolName.LOOKUP_CATALOG, {"query": "报销", "cursor": "not-a-cursor"}, context
        )

    assert info.value.code is ToolErrorCode.INVALID_ARGUMENTS


# =====================================================================
# 2. get_workspace_overview
# =====================================================================


async def test_get_workspace_overview_reports_counts_topics_and_images(tmp_path: Path) -> None:
    registry = build_default_registry()
    store = _BlockStore(
        [
            _block("doc-a", heading="报销范围", index=0),
            _block("doc-a", heading="报销时限", index=1),
            _block(
                "doc-img",
                heading="截图说明",
                index=0,
                chunk_type="image_description",
                image_content_hash="hash-img",
                image_path=ABS_IMAGE,
                source_file=ABS_IMAGE,
            ),
        ]
    )
    context = _context(
        _runtime_workspace(tmp_path), store=store, state_store=_StateStore(5)
    )

    result = await registry.execute(ToolName.GET_WORKSPACE_OVERVIEW, {}, context)

    assert result.status is ToolStatus.OK
    data = result.data
    assert data["workspace"]["workspace_id"] == "ws-a"
    assert data["knowledge"]["chunk_count"] == 3
    assert data["knowledge"]["image_chunk_count"] == 1
    assert data["documents"] == {"registered": 5, "indexed": 2}
    assert data["images"] == {"image_chunk_count": 1, "indexed_images": 1}
    assert data["tasks"]["available"] is False  # 没有 run 上下文时如实标注
    assert any(topic["name"] for topic in data["topics"])
    _assert_no_local_paths(data)


async def test_get_workspace_overview_categories_subset_and_task_status(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    store = _BlockStore([_block("doc-a", heading="报销范围")])
    app_store = await _seeded_store(workspace, run_id="run-a")
    try:
        await app_store.create_task(run_id="run-a", task_id="t-1", agent_kind=AgentKind.RAG)
        await app_store.create_task(run_id="run-a", task_id="t-2", agent_kind=AgentKind.WEB)
        registry = build_default_registry()
        context = _context(
            workspace, store=store, app_store=app_store, run_id="run-a"
        )

        result = await registry.execute(
            ToolName.GET_WORKSPACE_OVERVIEW, {"categories": ["tasks", "documents"]}, context
        )

        assert set(result.data) >= {"workspace", "categories", "tasks", "documents"}
        assert "knowledge" not in result.data and "topics" not in result.data
        assert result.data["tasks"]["total"] == 2
        assert result.data["tasks"]["by_status"] == {"queued": 2}
    finally:
        await app_store.close()


# =====================================================================
# 3. search_knowledge
# =====================================================================


async def test_search_knowledge_returns_chunks_scores_and_knowledge_version(tmp_path: Path) -> None:
    registry = build_default_registry()
    retriever = _SpyRetriever(
        [_chunk("doc-a", "报销上限 5000 元"), _chunk("doc-b", "差旅标准 800 元", similarity=0.72)]
    )
    store = _BlockStore([_block("doc-a"), _block("doc-b")])
    context = _context(
        _runtime_workspace(tmp_path), store=store, retriever=retriever, embedder=FakeEmbedder()
    )

    result = await registry.execute(ToolName.SEARCH_KNOWLEDGE, {"query": "报销上限"}, context)

    assert result.status is ToolStatus.OK
    assert result.data["count"] == 2
    first = result.data["chunks"][0]
    assert first["text"] == "报销上限 5000 元"
    assert first["similarity"] == 0.9
    assert first["document_id"] == "doc-a"
    assert first["document_name"] == "报销制度.md"
    assert first["heading_path"] == "第一章"
    assert result.data["knowledge_revision"] == 0
    assert len(retriever.calls) == 1
    _assert_no_local_paths(result.data)


async def test_search_knowledge_top_k_and_document_scope(tmp_path: Path) -> None:
    registry = build_default_registry()
    retriever = _SpyRetriever(
        [
            _chunk("doc-a", "甲", index=0),
            _chunk("doc-b", "乙", index=1),
            _chunk("doc-b", "丙", index=2),
        ]
    )
    store = _BlockStore([_block("doc-a"), _block("doc-b")])
    context = _context(
        _runtime_workspace(tmp_path), store=store, retriever=retriever, embedder=FakeEmbedder()
    )

    capped = await registry.execute(
        ToolName.SEARCH_KNOWLEDGE, {"query": "报销", "top_k": 1}, context
    )
    scoped = await registry.execute(
        ToolName.SEARCH_KNOWLEDGE, {"query": "报销", "document_scope": ["doc-b"]}, context
    )

    assert capped.data["count"] == 1
    assert capped.data["candidate_count"] == 3
    assert [chunk["document_id"] for chunk in scoped.data["chunks"]] == ["doc-b", "doc-b"]


async def test_search_knowledge_min_similarity_marks_irrelevant(tmp_path: Path) -> None:
    registry = build_default_registry()
    retriever = _SpyRetriever([_chunk("doc-a", "无关内容", similarity=0.2)])
    context = _context(
        _runtime_workspace(tmp_path),
        store=_BlockStore([_block("doc-a")]),
        retriever=retriever,
        embedder=FakeEmbedder(),
    )

    result = await registry.execute(
        ToolName.SEARCH_KNOWLEDGE, {"query": "报销", "min_similarity": 0.6}, context
    )

    assert result.status is ToolStatus.IRRELEVANT
    assert result.data["code"] == "irrelevant"
    assert result.data["candidate_count"] == 1
    assert result.data["chunks"] == []


async def test_search_knowledge_empty_results_and_empty_index(tmp_path: Path) -> None:
    registry = build_default_registry()
    no_hits = _context(
        _runtime_workspace(tmp_path),
        store=_BlockStore([_block("doc-a")]),
        retriever=_SpyRetriever([]),
        embedder=FakeEmbedder(),
    )
    empty_index = _context(
        _runtime_workspace(tmp_path, workspace_id="ws-empty", name="空区"),
        store=_BlockStore([]),
        retriever=_SpyRetriever([]),
        embedder=FakeEmbedder(),
    )

    no_match = await registry.execute(ToolName.SEARCH_KNOWLEDGE, {"query": "报销"}, no_hits)
    unindexed = await registry.execute(ToolName.SEARCH_KNOWLEDGE, {"query": "报销"}, empty_index)

    assert no_match.status is ToolStatus.EMPTY
    assert no_match.data["code"] == "no_results"
    assert unindexed.status is ToolStatus.EMPTY
    assert unindexed.data["code"] == "index_empty"


async def test_search_knowledge_requires_embedder_and_retriever(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)

    with pytest.raises(ToolUnavailableError):
        await registry.execute(
            ToolName.SEARCH_KNOWLEDGE,
            {"query": "报销"},
            _context(workspace, store=_BlockStore([_block("doc-a")]), retriever=_SpyRetriever()),
        )
    with pytest.raises(ToolUnavailableError):
        await registry.execute(
            ToolName.SEARCH_KNOWLEDGE,
            {"query": "报销"},
            _context(workspace, store=_BlockStore([_block("doc-a")]), embedder=FakeEmbedder()),
        )


async def test_search_knowledge_retrieval_error_is_service_error(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(
        _runtime_workspace(tmp_path),
        store=_BlockStore([_block("doc-a")]),
        retriever=_SpyRetriever(error=RetrievalError("索引暂不可用")),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(ToolExecutionError) as info:
        await registry.execute(ToolName.SEARCH_KNOWLEDGE, {"query": "报销"}, context)

    assert info.value.code is ToolErrorCode.SERVICE_ERROR


async def test_search_knowledge_rejects_foreign_knowledge_area(tmp_path: Path) -> None:
    """跨工作区检索面必须被拒绝，且**在检索之前**（本地正文检索次数为 0）。"""

    registry = build_default_registry()
    foreign_workspace = _runtime_workspace(tmp_path, workspace_id="ws-b", name="乙区")
    spy = _SpyRetriever([_chunk("doc-a", "乙区机密")])
    area = _Area(
        context=foreign_workspace, retriever=spy, vector_store=_BlockStore([_block("doc-a")])
    )
    context = _context(
        _runtime_workspace(tmp_path, workspace_id="ws-a"),
        store=_BlockStore([_block("doc-a")]),
        embedder=FakeEmbedder(),
        knowledge_area=area,
    )

    with pytest.raises(ToolPermissionError) as info:
        await registry.execute(ToolName.SEARCH_KNOWLEDGE, {"query": "机密"}, context)

    assert info.value.code is ToolErrorCode.WORKSPACE_MISMATCH
    assert spy.calls == []


async def test_search_knowledge_rejects_foreign_vector_store(tmp_path: Path) -> None:
    registry = build_default_registry()
    foreign_workspace = _runtime_workspace(tmp_path, workspace_id="ws-b", name="乙区")
    foreign_store = _BlockStore([_block("doc-b", text="乙区正文")])
    area = _Area(context=foreign_workspace, retriever=_SpyRetriever(), vector_store=foreign_store)
    context = _context(
        _runtime_workspace(tmp_path, workspace_id="ws-a"),
        embedder=FakeEmbedder(),
        knowledge_area=area,
    )

    with pytest.raises(ToolPermissionError):
        await registry.execute(ToolName.READ_DOCUMENT, {"document_id": "doc-b"}, context)


# =====================================================================
# 4. read_document
# =====================================================================


def test_document_identity_helpers_do_not_leak_paths(tmp_path: Path) -> None:
    store = _BlockStore([_block("doc-a")])
    assert document_id_of_block("doc-a::chunk-0") == "doc-a"
    assert document_id_of_block("plain-id") == "plain-id"
    assert resolve_document_source(store, "doc-a") == ABS_SOURCE
    assert resolve_document_source(store, "nope") == ""


# =====================================================================
# 5. verify_image
# =====================================================================


def _image_block(image_hash: str = "hash-img", source_file: str = ABS_IMAGE) -> tuple[str, str, dict[str, Any]]:
    return _block(
        "doc-img",
        heading="截图说明",
        text="描述：月度销售额",
        index=0,
        chunk_type="image_description",
        source_file=source_file,
        image_content_hash=image_hash,
        image_path=source_file,
    )


async def test_verify_image_reads_original_with_question(tmp_path: Path) -> None:
    registry = build_default_registry()
    seen: list[tuple[str, str]] = []

    def _reader(chunk: Any, question: str) -> str:
        seen.append((chunk.block_id, question))
        return "原图显示：1 月 120 万，2 月 150 万"

    context = _context(
        _runtime_workspace(tmp_path), store=_BlockStore([_image_block()]), image_reader=_reader
    )

    result = await registry.execute(
        ToolName.VERIFY_IMAGE, {"image_id": "hash-img", "question": "1 月多少？"}, context
    )

    assert result.status is ToolStatus.OK
    assert result.data["verification"].startswith("原图显示")
    assert result.data["image_content_hash"] == "hash-img"
    assert result.data["document_name"] == "截图.png"
    assert seen == [("doc-img::chunk-0", "1 月多少？")]
    _assert_no_local_paths(result.data)


async def test_verify_image_cross_workspace_is_not_found(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(
        _runtime_workspace(tmp_path, workspace_id="ws-b", name="乙区"),
        store=_BlockStore([_block("doc-a")]),
        image_reader=lambda chunk, question: "不该被调用",
    )

    result = await registry.execute(
        ToolName.VERIFY_IMAGE, {"image_id": "hash-img", "question": "多少？"}, context
    )

    assert result.status is ToolStatus.NOT_FOUND
    assert result.data["code"] == "image_not_found"


async def test_verify_image_requires_reader_and_sanitizes_failure(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)

    with pytest.raises(ToolUnavailableError):
        await registry.execute(
            ToolName.VERIFY_IMAGE,
            {"image_id": "hash-img", "question": "多少？"},
            _context(workspace, store=_BlockStore([_image_block()])),
        )

    def _broken(chunk: Any, question: str) -> str:
        raise ValueError(f"内部细节 {ABS_IMAGE}")

    with pytest.raises(ToolExecutionError) as info:
        await registry.execute(
            ToolName.VERIFY_IMAGE,
            {"image_id": "hash-img", "question": "多少？"},
            _context(
                workspace, store=_BlockStore([_image_block()]), image_reader=_broken
            ),
        )

    assert "ValueError" in str(info.value)
    assert ABS_IMAGE not in str(info.value)


async def test_verify_image_empty_verification_is_empty_status(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _context(
        _runtime_workspace(tmp_path),
        store=_BlockStore([_image_block()]),
        image_reader=lambda chunk, question: "   ",
    )

    result = await registry.execute(
        ToolName.VERIFY_IMAGE, {"image_id": "hash-img", "question": "多少？"}, context
    )

    assert result.status is ToolStatus.EMPTY
    assert result.data["code"] == "empty_verification"


# =====================================================================
# 6. ask_user
# =====================================================================


async def test_ask_user_creates_input_request(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    app_store = await _seeded_store(workspace, run_id="run-a")
    try:
        context = _context(
            workspace, app_store=app_store, run_id="run-a", task_id=None
        )

        result = await registry.execute(
            ToolName.ASK_USER,
            {"question": "要查哪个季度？", "options": ["Q1", "Q2"], "reason": "缺少时间范围"},
            context,
        )

        assert result.status is ToolStatus.WAITING_INPUT
        request_id = result.data["input_request_id"]
        assert request_id.startswith("input-")
        saved = await app_store.get_input_request(request_id)
        assert saved is not None
        assert saved.question == "要查哪个季度？"
        assert saved.options == ["Q1", "Q2"]
        assert saved.question_version == 1
        assert str(saved.status) == "pending"
        assert saved.run_id == "run-a"
    finally:
        await app_store.close()


async def test_ask_user_requires_run_context(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    app_store = await _seeded_store(workspace, run_id="run-a")
    try:
        with pytest.raises(ToolUnavailableError):
            await registry.execute(
                ToolName.ASK_USER,
                {"question": "要查哪个季度？"},
                _context(workspace, app_store=app_store),
            )
    finally:
        await app_store.close()


async def test_read_tools_fail_closed_without_workspace_services(tmp_path: Path) -> None:
    """没有绑定索引句柄时如实报「未配置」，而不是静默返回空结果。"""

    registry = build_default_registry()
    context = _context(_runtime_workspace(tmp_path))

    with pytest.raises(ToolUnavailableError):
        await registry.execute(ToolName.LOOKUP_CATALOG, {"query": "报销"}, context)
    with pytest.raises(ToolUnavailableError):
        await registry.execute(ToolName.READ_DOCUMENT, {"document_id": "doc-a"}, context)
