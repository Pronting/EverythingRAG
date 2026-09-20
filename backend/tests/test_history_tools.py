"""历史检索工具测试（PRD 7.1 / 5.4）。Owner: T04。

``search_history`` / ``read_history`` 只返回**本工作区**的会话消息：跨区会话与消息
等同不存在（与 T01 的「跨区 404」一致）。用真实 ``AppStore``（内存）+ T01 的 tmp
工作区上下文，零出网、零向量库。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import AgentKind
from app.agent.tools import (
    ToolArgumentError,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolStatus,
    ToolUnavailableError,
    build_default_registry,
)
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore


def _runtime_workspace(tmp_path: Path, *, workspace_id: str = "ws-a", name: str = "甲区") -> Any:
    store = WorkspaceStore(tmp_path)
    record = store.create(name, workspace_id=workspace_id)
    return WorkspaceContextFactory(tmp_path, store).resolve(record.id)


async def _app_store_with(*workspaces: Any, run_id: str = "run-a") -> AppStore:
    store = await AppStore.open_memory()
    for workspace in workspaces:
        await store.create_workspace(workspace.workspace_id, workspace.storage_key)
    owner = workspaces[0]
    conversation = await store.create_conversation(
        f"conv-{owner.workspace_id}", owner.workspace_id
    )
    await store.create_run(
        conversation_id=conversation.id, workspace_id=owner.workspace_id, run_id=run_id
    )
    return store


async def _add_conversation(
    store: AppStore,
    workspace: Any,
    conversation_id: str,
    messages: list[tuple[str, str, str]],
) -> None:
    """``messages`` = [(role, content, created_at)]。"""

    await store.create_conversation(conversation_id, workspace.workspace_id)
    for role, content, created_at in messages:
        await store.append_message(
            conversation_id=conversation_id, role=role, content=content, created_at=created_at
        )


def _context(
    workspace: Any, store: AppStore | None, *, run_id: str | None = "run-a", **overrides: Any
) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        app_store=store,
        run_id=run_id,
        agent_kind=AgentKind.RAG,
        **overrides,
    )


class _CountingStore:
    """``AppStore`` 计数代理：只统计 ``list_conversations`` 调用次数，其余全部委托。

    用来断言「接了 FTS 接缝时**不触发全表扫描**」。
    """

    def __init__(self, inner: AppStore) -> None:
        self._inner = inner
        self.list_conversations_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def list_conversations(self, *args: Any, **kwargs: Any) -> Any:
        self.list_conversations_calls += 1
        return await self._inner.list_conversations(*args, **kwargs)


class _RecordingHistorySeam:
    """FTS 接缝替身：记录 (query, request)，可返回 payload 或抛异常。"""

    def __init__(self, payload: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._payload = payload

    async def __call__(self, query: str, request: Mapping[str, Any]) -> Any:
        self.calls.append((query, dict(request)))
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


MESSAGE_KEYS = {
    "conversation_id",
    "conversation_title",
    "message_id",
    "sequence",
    "role",
    "created_at",
    "excerpt",
    "matched_terms",
}
DATA_KEYS_OK = {"messages", "total", "returned", "next_cursor", "conversation_count"}
DATA_KEYS_EMPTY = {"messages", "total", "next_cursor", "conversation_count"}


def _fts_payload(*, total: int | None = 7, next_cursor: str | None = "20") -> dict[str, Any]:
    """t8 形状的接缝返回（``HistoryHit.as_dict``：带 snippet/score，无 matched_terms）。"""

    return {
        "messages": [
            {
                "message_id": "m-1",
                "conversation_id": "conv-a",
                "role": "user",
                "content": "报销上限 5000 元",
                "created_at": "2026-09-18T08:00:00Z",
                "sequence": 1,
                "score": -1.23,
                "snippet": "…报销上限 5000 元…",
            }
        ],
        "next_cursor": next_cursor,
        "total": total,
        "conversation_count": 3,
    }


# =====================================================================
# search_history
# =====================================================================


async def test_search_history_returns_only_this_workspace_messages(tmp_path: Path) -> None:
    registry = build_default_registry()
    first = _runtime_workspace(tmp_path, workspace_id="ws-a", name="甲区")
    second = _runtime_workspace(tmp_path, workspace_id="ws-b", name="乙区")
    store = await _app_store_with(first, second)
    try:
        await _add_conversation(
            store,
            first,
            "conv-a",
            [
                ("user", "甲区讨论报销上限", "2026-09-18T08:00:00Z"),
                ("assistant", "甲区答复：报销上限 5000 元", "2026-09-18T08:00:05Z"),
            ],
        )
        await _add_conversation(
            store,
            second,
            "conv-b",
            [("user", "乙区也在讨论报销上限 9999", "2026-09-19T08:00:00Z")],
        )

        result = await registry.execute(ToolName.SEARCH_HISTORY, {"query": "报销上限"}, _context(first, store))

        assert result.status is ToolStatus.OK
        assert result.data["total"] == 2
        assert {item["conversation_id"] for item in result.data["messages"]} == {"conv-a"}
        assert all("乙区" not in item["excerpt"] for item in result.data["messages"])
        assert result.data["conversation_count"] == 2  # 本区 conv-a + 建 run 用的空会话；乙区不计入
        assert result.data["messages"][0]["role"] in {"user", "assistant"}
    finally:
        await store.close()


async def test_search_history_scores_multi_terms_and_skips_non_dialogue_roles(
    tmp_path: Path,
) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [
                ("user", "报销流程与时限", "2026-09-18T08:00:00Z"),
                ("user", "只有报销", "2026-09-18T09:00:00Z"),
                ("system", "报销 时限 系统提示", "2026-09-18T10:00:00Z"),
                ("assistant", "报销 时限 答复", "2026-09-18T11:00:00Z"),
            ],
        )

        result = await registry.execute(
            ToolName.SEARCH_HISTORY, {"query": "报销 时限"}, _context(workspace, store)
        )

        # 多词任一命中即候选（OR + 打分）；system 角色不进历史检索
        assert result.data["total"] == 3
        assert all(item["role"] in {"user", "assistant"} for item in result.data["messages"])
        assert [len(item["matched_terms"]) for item in result.data["messages"]] == [2, 2, 1]
    finally:
        await store.close()


async def test_search_history_date_range_filter(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [
                ("user", "报销 九月十七", "2026-09-17T08:00:00Z"),
                ("user", "报销 九月十八", "2026-09-18T08:00:00Z"),
                ("user", "报销 九月十九", "2026-09-19T08:00:00Z"),
            ],
        )

        result = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "start_date": "2026-09-18", "end_date": "2026-09-18"},
            _context(workspace, store),
        )

        assert result.data["total"] == 1
        assert "九月十八" in result.data["messages"][0]["excerpt"]
    finally:
        await store.close()


async def test_search_history_order_is_deterministic(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [
                ("user", "报销 时限 第一次", "2026-09-18T08:00:00Z"),
                ("user", "只有报销", "2026-09-18T09:00:00Z"),
            ],
        )
        await _add_conversation(
            store,
            workspace,
            "conv-b",
            [("user", "报销 时限 新会话", "2026-09-19T08:00:00Z")],
        )

        result = await registry.execute(
            ToolName.SEARCH_HISTORY, {"query": "报销 时限"}, _context(workspace, store)
        )

        assert [item["conversation_id"] for item in result.data["messages"]] == [
            "conv-b",
            "conv-a",
            "conv-a",
        ]
        assert [len(item["matched_terms"]) for item in result.data["messages"]] == [2, 2, 1]
    finally:
        await store.close()


async def test_search_history_paginates_with_cursor(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [("user", f"报销 第 {index} 条", f"2026-09-1{index}T08:00:00Z") for index in range(1, 4)],
        )

        first = await registry.execute(
            ToolName.SEARCH_HISTORY, {"query": "报销", "limit": 2}, _context(workspace, store)
        )
        second = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "limit": 2, "cursor": first.data["next_cursor"]},
            _context(workspace, store),
        )

        assert first.data["total"] == 3
        assert first.data["returned"] == 2
        assert first.data["next_cursor"] == "2"
        assert second.data["returned"] == 1
        assert second.data["next_cursor"] is None
        assert {item["message_id"] for item in first.data["messages"]} & {
            item["message_id"] for item in second.data["messages"]
        } == set()
    finally:
        await store.close()


async def test_search_history_empty_and_misconfigured(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store, workspace, "conv-a", [("user", "与关键词无关", "2026-09-18T08:00:00Z")]
        )

        empty = await registry.execute(
            ToolName.SEARCH_HISTORY, {"query": "完全没有的词"}, _context(workspace, store)
        )
        with pytest.raises(ToolUnavailableError):
            await registry.execute(
                ToolName.SEARCH_HISTORY, {"query": "报销"}, _context(workspace, None)
            )
        with pytest.raises(ToolArgumentError) as info:
            await registry.execute(
                ToolName.SEARCH_HISTORY,
                {"query": "报销", "cursor": "abc"},
                _context(workspace, store),
            )

        assert empty.status is ToolStatus.EMPTY
        assert empty.data["messages"] == []
        assert info.value.code is ToolErrorCode.INVALID_ARGUMENTS
    finally:
        await store.close()


# =====================================================================
# read_history
# =====================================================================


async def test_read_history_around_sequence_window_and_continuation(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [("user", f"第 {index} 条", f"2026-09-18T08:0{index}:00Z") for index in range(1, 7)],
        )

        window = await registry.execute(
            ToolName.READ_HISTORY,
            {"conversation_id": "conv-a", "around_sequence": 3, "limit": 4},
            _context(workspace, store),
        )
        following = await registry.execute(
            ToolName.READ_HISTORY,
            {"conversation_id": "conv-a", "after_sequence": window.data["next_after_sequence"]},
            _context(workspace, store),
        )

        assert [message["sequence"] for message in window.data["messages"]] == [1, 2, 3, 4]
        assert window.data["next_after_sequence"] == 4
        assert [message["sequence"] for message in following.data["messages"]] == [5, 6]
        assert following.data["next_after_sequence"] is None
    finally:
        await store.close()


async def test_read_history_cross_workspace_is_not_found(tmp_path: Path) -> None:
    registry = build_default_registry()
    first = _runtime_workspace(tmp_path, workspace_id="ws-a", name="甲区")
    second = _runtime_workspace(tmp_path, workspace_id="ws-b", name="乙区")
    store = await _app_store_with(first, second)
    try:
        await _add_conversation(
            store, second, "conv-b", [("user", "乙区机密", "2026-09-18T08:00:00Z")]
        )

        result = await registry.execute(
            ToolName.READ_HISTORY, {"conversation_id": "conv-b"}, _context(first, store)
        )

        assert result.status is ToolStatus.NOT_FOUND
        assert result.data["code"] == "conversation_not_found"
        assert "content" not in str(result.data)
    finally:
        await store.close()


async def test_read_history_unknown_conversation_and_end_of_history(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store, workspace, "conv-a", [("user", "唯一一条", "2026-09-18T08:00:00Z")]
        )

        unknown = await registry.execute(
            ToolName.READ_HISTORY, {"conversation_id": "conv-nope"}, _context(workspace, store)
        )
        exhausted = await registry.execute(
            ToolName.READ_HISTORY,
            {"conversation_id": "conv-a", "after_sequence": 100},
            _context(workspace, store),
        )

        assert unknown.status is ToolStatus.NOT_FOUND
        assert exhausted.status is ToolStatus.EMPTY
        assert exhausted.data["code"] == "no_more_messages"
    finally:
        await store.close()


def test_history_tools_are_local_only() -> None:
    registry = build_default_registry()
    for name in (ToolName.SEARCH_HISTORY, ToolName.READ_HISTORY):
        spec = registry.get(name)
        assert spec.networked is False
        assert spec.requires_approval is False
        assert spec.allows(AgentKind.RAG) is True


# =====================================================================
# search_history：FTS 接缝（T08 实现，T09 装配）
# =====================================================================


async def test_search_history_delegates_to_index_without_scanning(tmp_path: Path) -> None:
    """接了 FTS 接缝 → 委托被调用，且**不触发**全表扫描（list_conversations 次数为 0）。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        counting = _CountingStore(store)
        seam = _RecordingHistorySeam(_fts_payload())
        context = _context(workspace, counting, history_search=seam)

        result = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {
                "query": "报销 时限",
                "start_date": "2026-09-01",
                "end_date": "2026-09-30",
                "limit": 20,
                "cursor": "20",
            },
            context,
        )

        assert result.status is ToolStatus.OK
        assert len(seam.calls) == 1
        query, request = seam.calls[0]
        assert query == "报销 时限"
        assert request == {
            "workspace_id": "ws-a",
            "limit": 20,
            "cursor": "20",
            "start_date": "2026-09-01",
            "end_date": "2026-09-30",
        }
        assert counting.list_conversations_calls == 0  # 没有全表扫描
        assert result.data["total"] == 7
        assert result.data["next_cursor"] == "20"
        assert result.data["conversation_count"] == 3
    finally:
        await store.close()


async def test_search_history_delegated_mapping_keeps_schema(tmp_path: Path) -> None:
    """委托路径与回退路径的**键集完全一致**（新增/缺失键都会让该断言失败）。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await store.create_conversation("conv-a", workspace.workspace_id, title="报销答疑")
        seam = _RecordingHistorySeam(_fts_payload())
        context = _context(workspace, store, history_search=seam)

        result = await registry.execute(ToolName.SEARCH_HISTORY, {"query": "报销"}, context)

        assert set(result.data) == DATA_KEYS_OK
        message = result.data["messages"][0]
        assert set(message) == MESSAGE_KEYS
        assert message["excerpt"] == "…报销上限 5000 元…"  # 用接缝的 snippet
        assert message["matched_terms"] == ["报销"]  # 无 matched_terms 时按命中词补算
        assert message["conversation_title"] == "报销答疑"  # 按会话 ID 主键补标题
        assert "FTS 索引检索" in result.message
    finally:
        await store.close()


async def test_search_history_delegated_payload_variants(tmp_path: Path) -> None:
    """接缝可以只给命中列表；``total`` 未提供时如实为 None（不猜数字）。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    hits = _fts_payload()["messages"]

    bare = await registry.execute(
        ToolName.SEARCH_HISTORY,
        {"query": "报销"},
        _context(workspace, None, history_search=_RecordingHistorySeam(hits)),
    )
    no_total = await registry.execute(
        ToolName.SEARCH_HISTORY,
        {"query": "报销"},
        _context(workspace, None, history_search=_RecordingHistorySeam({"messages": hits})),
    )

    assert bare.status is ToolStatus.OK
    assert set(bare.data) == DATA_KEYS_OK
    assert bare.data["total"] is None
    assert no_total.data["total"] is None
    # 未绑 app.db 也能工作：委托路径不需要 store
    assert no_total.data["messages"][0]["conversation_title"] == ""


async def test_search_history_delegated_empty_and_failure(tmp_path: Path) -> None:
    """空结果 → EMPTY（键集仍一致）；接缝抛错 → SERVICE_ERROR（不静默回退）。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)

    empty = await registry.execute(
        ToolName.SEARCH_HISTORY,
        {"query": "没有"},
        _context(workspace, None, history_search=_RecordingHistorySeam({"messages": []})),
    )
    with pytest.raises(ToolExecutionError) as info:
        await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销"},
            _context(
                workspace,
                None,
                history_search=_RecordingHistorySeam(RuntimeError("索引未就绪")),
            ),
        )
    with pytest.raises(ToolExecutionError) as malformed:
        await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销"},
            _context(workspace, None, history_search=_RecordingHistorySeam({"messages": [1, 2]})),
        )

    assert empty.status is ToolStatus.EMPTY
    assert set(empty.data) == DATA_KEYS_EMPTY
    assert empty.data["messages"] == []
    assert info.value.code is ToolErrorCode.SERVICE_ERROR
    assert "索引未就绪" in str(info.value)
    assert malformed.value.code is ToolErrorCode.INTERNAL


# =====================================================================
# search_history：回退扫描的边界
# =====================================================================


async def test_search_history_fallback_is_bounded_across_conversations(tmp_path: Path) -> None:
    """未接接缝时回退扫描有**跨会话**上限：触顶即截断并在 message 中说明。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store, workspace, "conv-a", [("user", "报销 第一条", "2026-09-18T08:00:00Z")]
        )
        await _add_conversation(
            store, workspace, "conv-b", [("user", "报销 第二条", "2026-09-19T08:00:00Z")]
        )
        await _add_conversation(
            store, workspace, "conv-c", [("user", "报销 第三条", "2026-09-20T08:00:00Z")]
        )

        result = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "limit": 10},
            _context(workspace, store, history_max_conversations=1),
        )
        # 第二次调用显式 reuse=False：同 run + 同参数 → 同 operation_id，否则会命中
        # 第一次（带上限）的结果缓存，这属于重放复用特性而非本用例要断言的行为。
        uncapped = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "limit": 10},
            _context(workspace, store),
            reuse=False,
        )

        assert set(result.data) == DATA_KEYS_OK
        assert result.data["conversation_count"] == 1  # 只扫了一个会话
        assert result.data["total"] == 1
        assert "上限" in result.message and "不完整" in result.message
        assert uncapped.data["conversation_count"] >= 3
        assert uncapped.data["total"] == 3
        assert "上限" not in uncapped.message
    finally:
        await store.close()


async def test_search_history_fallback_bounds_messages_and_matches(tmp_path: Path) -> None:
    """消息总量上限与命中数上限同样生效（命中数触顶时命中集合被截断）。"""

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [
                ("user", f"报销 第 {index} 条", f"2026-09-1{index}T08:00:00Z")
                for index in range(1, 4)
            ],
        )

        by_message_cap = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "limit": 10},
            _context(workspace, store, history_max_messages=1),
        )
        by_match_cap = await registry.execute(
            ToolName.SEARCH_HISTORY,
            {"query": "报销", "limit": 10},
            _context(workspace, store, history_max_matches=2),
            reuse=False,  # 同上：同参数会命中上一次调用（带上限）的缓存
        )

        assert by_message_cap.data["total"] == 1
        assert "上限" in by_message_cap.message
        assert by_match_cap.data["total"] == 2
        assert "上限" in by_match_cap.message
    finally:
        await store.close()


# =====================================================================
# 端到端串联：search_history → read_history
# =====================================================================


async def test_search_history_then_read_history_expands_context(tmp_path: Path) -> None:
    """命中 → 展开上下文：search_history 给的 ``conversation_id`` + ``sequence``
    直接喂给 ``read_history(around_sequence=...)``。

    这是 T07/T09 实际要走的链路（先用 FTS/回退找到命中，再用位置读取展开上下文），
    因此它同时是「t8 的命中含 ``sequence``」这一衔接的可执行证据。
    """

    registry = build_default_registry()
    workspace = _runtime_workspace(tmp_path)
    store = await _app_store_with(workspace)
    try:
        await _add_conversation(
            store,
            workspace,
            "conv-a",
            [
                ("user", "开头寒暄", "2026-09-18T08:00:00Z"),
                ("assistant", "报销上限 5000 元", "2026-09-18T08:01:00Z"),
                ("user", "后续追问细节", "2026-09-18T08:02:00Z"),
            ],
        )

        found = await registry.execute(
            ToolName.SEARCH_HISTORY, {"query": "报销上限"}, _context(workspace, store)
        )
        hit = found.data["messages"][0]
        assert hit["sequence"] == 2
        assert hit["conversation_id"] == "conv-a"

        window = await registry.execute(
            ToolName.READ_HISTORY,
            {
                "conversation_id": hit["conversation_id"],
                "around_sequence": hit["sequence"],
                "limit": 3,
            },
            _context(workspace, store),
        )

        assert window.status is ToolStatus.OK
        assert [message["sequence"] for message in window.data["messages"]] == [1, 2, 3]
        assert window.data["messages"][1]["content"] == "报销上限 5000 元"
        assert window.data["messages"][1]["message_id"] == hit["message_id"]
    finally:
        await store.close()
