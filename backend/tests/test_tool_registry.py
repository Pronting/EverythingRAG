"""ToolRegistry 机制测试（PRD 7.1 / 12.1）。Owner: T04。

覆盖：标准 JSON 解析、Pydantic 校验、Agent 权限门控（含 A01 的 Web 门）、
稳定 ``operation_id``、``operations`` 登记与重放复用、artifacts 路径规则、
「已脱敏参数」入口。

全部零出网、零真实 Chroma：工作区用 T01 的真实 ``WorkspaceContext``（tmp 目录），
``app.db`` 用 ``AppStore`` 内存实例。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from app.agent.app_store import AppStore
from app.agent.contracts import AgentKind, ContractModel, OperationStatus
from app.agent.tools import (
    OPERATIONS_ARTIFACT_DIR,
    WEB_ARTIFACT_DIR,
    ToolArgumentError,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolPermissionError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolStatus,
    ToolUnavailableError,
    arguments_hash,
    build_default_registry,
    read_artifact,
    stable_operation_id,
    write_artifact,
)
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore

# ---------------------------------------------------------------- 回归用工具


class _EchoParams(ContractModel):
    """回显工具参数（与真实工具同基类，因此同样 ``extra="forbid"``）。"""

    text: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)


class _RecordingHandler:
    """记录每次调用的参数；可配置返回状态或抛出指定异常。"""

    def __init__(self, *, status: ToolStatus = ToolStatus.OK, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._status = status
        self._error = error

    async def __call__(self, params: Any, context: ToolContext) -> ToolResult:
        self.calls.append(params.model_dump(mode="json"))
        if self._error is not None:
            raise self._error
        return ToolResult(tool="", status=self._status, data={"echo": params.model_dump(mode="json")})


def _registry(
    handler: _RecordingHandler,
    *,
    name: ToolName = ToolName.WEB_SEARCH,
    allowed: frozenset[AgentKind] = frozenset({AgentKind.RAG}),
    networked: bool = True,
    knowledge_scoped: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            params_model=_EchoParams,
            handler=handler,
            description="回归用回显工具",
            returns="echo",
            networked=networked,
            knowledge_scoped=knowledge_scoped,
            allowed_agents=allowed,
        )
    )
    return registry


def _runtime_workspace(tmp_path: Path, *, workspace_id: str = "ws-a", name: str = "甲区") -> Any:
    """T01 的真实运行期上下文（tmp 目录，不触碰 ``~/.everything-rag``）。"""

    store = WorkspaceStore(tmp_path)
    record = store.create(name, workspace_id=workspace_id)
    context = WorkspaceContextFactory(tmp_path, store).resolve(record.id)
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return context


async def _seeded_store(workspace_id: str, *, run_id: str = "run-a") -> AppStore:
    """内存 app.db + 工作区 + 会话 + run（operations 的外键需要真实 run 行）。"""

    store = await AppStore.open_memory()
    await store.create_workspace(workspace_id, f"key-{workspace_id}")
    conversation = await store.create_conversation(f"conv-{workspace_id}", workspace_id)
    await store.create_run(
        conversation_id=conversation.id, workspace_id=workspace_id, run_id=run_id
    )
    return store


def _expected_operation_id(operation_id: str) -> str:
    return f"{OPERATIONS_ARTIFACT_DIR}/{operation_id}.json"


# ---------------------------------------------------------------- 1. 注册表内容


def test_default_registry_registers_prd_tools_in_order() -> None:
    registry = build_default_registry()
    assert registry.names() == tuple(str(name) for name in ToolName)
    assert len(registry.specs()) == 10
    for spec in registry.specs():
        assert spec.description.strip()
        assert spec.returns.strip()


def test_describe_exposes_parameters_and_source_scope() -> None:
    described = {item["name"]: item for item in build_default_registry().describe()}
    knowledge = described["search_knowledge"]
    assert knowledge["parameters"]["type"] == "object"
    assert "query" in knowledge["parameters"]["properties"]
    assert knowledge["networked"] is False
    web = described["web_search"]
    assert web["networked"] is True
    assert web["requires_approval"] is True
    assert described["ask_user"]["networked"] is False


def test_web_agent_is_restricted_to_web_tools() -> None:
    """A01：Web 路径拿不到任何本地知识正文工具（结构性保证）。"""

    registry = build_default_registry()
    allowed = {
        str(spec.name) for spec in registry.specs() if spec.allows(AgentKind.WEB)
    }
    assert allowed == {"web_search", "read_web_page", "ask_user"}
    for name in (
        ToolName.LOOKUP_CATALOG,
        ToolName.GET_WORKSPACE_OVERVIEW,
        ToolName.SEARCH_KNOWLEDGE,
        ToolName.READ_DOCUMENT,
        ToolName.SEARCH_HISTORY,
        ToolName.READ_HISTORY,
        ToolName.VERIFY_IMAGE,
    ):
        assert registry.get(name).allows(AgentKind.WEB) is False


def test_unknown_agent_kind_is_not_allowed() -> None:
    spec = build_default_registry().get(ToolName.SEARCH_KNOWLEDGE)
    assert spec.allows("不存在的职责") is False
    assert spec.allows(None) is False


# ---------------------------------------------------------------- 2. 参数解析


def test_unknown_tool_rejected() -> None:
    registry = build_default_registry()
    with pytest.raises(ToolArgumentError) as info:
        registry.get("no_such_tool")
    assert info.value.code is ToolErrorCode.UNKNOWN_TOOL


def test_invalid_json_rejected() -> None:
    registry = _registry(_RecordingHandler())
    with pytest.raises(ToolArgumentError) as info:
        registry.parse_arguments(ToolName.WEB_SEARCH, "{不是 JSON")
    assert info.value.code is ToolErrorCode.INVALID_JSON


def test_non_object_json_rejected() -> None:
    registry = _registry(_RecordingHandler())
    with pytest.raises(ToolArgumentError) as info:
        registry.parse_arguments(ToolName.WEB_SEARCH, "[1, 2]")
    assert info.value.code is ToolErrorCode.INVALID_JSON


def test_schema_violation_rejected() -> None:
    registry = _registry(_RecordingHandler())
    with pytest.raises(ToolArgumentError) as missing:
        registry.parse_arguments(ToolName.WEB_SEARCH, {})
    assert missing.value.code is ToolErrorCode.INVALID_ARGUMENTS
    assert "text" in str(missing.value)
    # extra="forbid"：多传未声明字段同样就地失败（PRD 12.5）
    with pytest.raises(ToolArgumentError) as extra:
        registry.parse_arguments(ToolName.WEB_SEARCH, {"text": "x", "unexpected": 1})
    assert extra.value.code is ToolErrorCode.INVALID_ARGUMENTS


async def test_execute_accepts_json_string_and_mapping(tmp_path: Path) -> None:
    handler = _RecordingHandler()
    registry = _registry(handler)
    context = ToolContext(workspace=_runtime_workspace(tmp_path))

    from_string = await registry.execute(ToolName.WEB_SEARCH, '{"text": "甲"}', context)
    from_mapping = await registry.execute(ToolName.WEB_SEARCH, {"text": "乙", "count": 3}, context)

    assert from_string.data["echo"] == {"text": "甲", "count": 1}
    assert from_mapping.data["echo"] == {"text": "乙", "count": 3}
    assert handler.calls == [{"text": "甲", "count": 1}, {"text": "乙", "count": 3}]


def test_empty_json_string_is_treated_as_empty_object() -> None:
    """模型偶发返回空串时按空对象处理，随后由 schema 校验如实拒绝。"""

    registry = _registry(_RecordingHandler())
    with pytest.raises(ToolArgumentError):
        registry.parse_arguments(ToolName.WEB_SEARCH, "")


# ---------------------------------------------------------------- 3. 权限门控


async def test_permission_gate_blocks_before_handler_runs(tmp_path: Path) -> None:
    handler = _RecordingHandler()
    registry = _registry(handler, allowed=frozenset({AgentKind.RAG}))
    context = ToolContext(workspace=_runtime_workspace(tmp_path), agent_kind=AgentKind.WEB)

    with pytest.raises(ToolPermissionError) as info:
        await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

    assert info.value.code is ToolErrorCode.NOT_PERMITTED
    assert info.value.tool == str(ToolName.WEB_SEARCH)
    assert handler.calls == []


async def test_permission_not_checked_without_agent_kind(tmp_path: Path) -> None:
    """未绑定 Agent 身份时不静默放行，而是如实标记 `permission_checked=False`。"""

    handler = _RecordingHandler()
    registry = _registry(handler)
    context = ToolContext(workspace=_runtime_workspace(tmp_path))

    result = await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

    assert result.permission_checked is False
    assert len(handler.calls) == 1


async def test_permission_checked_flag_true_for_allowed_agent(tmp_path: Path) -> None:
    handler = _RecordingHandler()
    registry = _registry(handler)
    context = ToolContext(workspace=_runtime_workspace(tmp_path), agent_kind=AgentKind.RAG)

    result = await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

    assert result.permission_checked is True


# ---------------------------------------------------------------- 4. operation_id


def test_stable_operation_id_is_deterministic_and_scoped() -> None:
    args = arguments_hash({"text": "x"})
    first = stable_operation_id("run-1", "task-1", "web_search", args)
    assert first == stable_operation_id("run-1", "task-1", "web_search", args)
    assert first.startswith("op-") and len(first) == 35
    assert first != stable_operation_id("run-2", "task-1", "web_search", args)
    assert first != stable_operation_id("run-1", "task-2", "web_search", args)
    assert first != stable_operation_id("run-1", "task-1", "read_document", args)
    assert first != stable_operation_id("run-1", "task-1", "web_search", arguments_hash({"text": "y"}))


def test_arguments_hash_is_order_insensitive() -> None:
    assert arguments_hash({"a": 1, "b": 2}) == arguments_hash({"b": 2, "a": 1})
    assert arguments_hash({"a": 1}) != arguments_hash({"a": "1"})


async def test_success_registers_and_finishes_operation(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    store = await _seeded_store(workspace.workspace_id, run_id="run-a")
    try:
        handler = _RecordingHandler()
        registry = _registry(handler)
        context = ToolContext(
            workspace=workspace, app_store=store, run_id="run-a", agent_kind=AgentKind.RAG
        )

        result = await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context)

        assert result.operation_id is not None
        record = await store.get_operation(result.operation_id)
        assert record is not None
        assert record.status is OperationStatus.SUCCEEDED
        assert record.attempts == 1
        assert record.tool_name == str(ToolName.WEB_SEARCH)
        assert record.args_hash == arguments_hash({"text": "你好", "count": 1})
        assert record.run_id == "run-a"
        assert record.result_ref == _expected_operation_id(result.operation_id)
        cache = workspace.artifacts_dir / _expected_operation_id(result.operation_id)
        assert cache.is_file()
    finally:
        await store.close()


async def test_replay_reuses_saved_result_without_second_handler_call(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    store = await _seeded_store(workspace.workspace_id, run_id="run-a")
    try:
        handler = _RecordingHandler()
        registry = _registry(handler)
        context = ToolContext(
            workspace=workspace, app_store=store, run_id="run-a", agent_kind=AgentKind.RAG
        )

        first = await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context)
        second = await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context)

        assert second.reused is True
        assert second.operation_id == first.operation_id
        assert second.data == first.data
        assert len(handler.calls) == 1  # 重放没有真的再执行一次
        record = await store.get_operation(first.operation_id)
        assert record is not None and record.attempts == 1
    finally:
        await store.close()


async def test_replay_reuse_can_be_disabled(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    handler = _RecordingHandler()
    registry = _registry(handler)
    context = ToolContext(workspace=workspace, agent_kind=AgentKind.RAG)

    await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context)
    forced = await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context, reuse=False)

    assert forced.reused is False
    assert len(handler.calls) == 2


async def test_replay_after_knowledge_revision_change_reexecutes(tmp_path: Path) -> None:
    """同一 run/参数但知识版本变了 → 不能复用旧结果（PRD 7.4「同一资料版本」）。"""

    workspace_store = WorkspaceStore(tmp_path)
    record = workspace_store.create("甲区", workspace_id="ws-a")
    factory = WorkspaceContextFactory(tmp_path, workspace_store)
    first_workspace = factory.resolve(record.id)
    first_workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    handler = _RecordingHandler()
    registry = _registry(handler, knowledge_scoped=True)
    context = ToolContext(workspace=first_workspace, agent_kind=AgentKind.RAG)

    await registry.execute(ToolName.WEB_SEARCH, {"text": "你好"}, context)
    assert len(handler.calls) == 1

    assert workspace_store.bump_knowledge_revision(record.id) == 1
    second_workspace = factory.resolve(record.id)
    assert second_workspace.knowledge_revision == 1

    refreshed = await registry.execute(
        ToolName.WEB_SEARCH,
        {"text": "你好"},
        ToolContext(workspace=second_workspace, agent_kind=AgentKind.RAG),
    )

    assert refreshed.reused is False
    assert len(handler.calls) == 2


async def test_tool_error_marks_operation_failed_and_propagates(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    store = await _seeded_store(workspace.workspace_id, run_id="run-a")
    try:
        handler = _RecordingHandler(
            error=ToolExecutionError("联网搜索失败（HTTP 500）", code=ToolErrorCode.SERVICE_ERROR)
        )
        registry = _registry(handler)
        context = ToolContext(
            workspace=workspace, app_store=store, run_id="run-a", agent_kind=AgentKind.RAG
        )
        operation_id = stable_operation_id(
            "run-a", None, str(ToolName.WEB_SEARCH), arguments_hash({"text": "x", "count": 1})
        )

        with pytest.raises(ToolExecutionError) as info:
            await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

        assert info.value.code is ToolErrorCode.SERVICE_ERROR
        assert info.value.tool == str(ToolName.WEB_SEARCH)
        record = await store.get_operation(operation_id)
        assert record is not None
        assert record.status is OperationStatus.FAILED
    finally:
        await store.close()


async def test_unexpected_exception_becomes_internal_error(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    handler = _RecordingHandler(error=RuntimeError("boom-internal-detail"))
    registry = _registry(handler)
    context = ToolContext(workspace=workspace, agent_kind=AgentKind.RAG)

    with pytest.raises(ToolExecutionError) as info:
        await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

    assert info.value.code is ToolErrorCode.INTERNAL
    assert "RuntimeError" in str(info.value)
    assert "boom-internal-detail" not in str(info.value)


async def test_permission_error_keeps_handler_and_operation_untouched(tmp_path: Path) -> None:
    workspace = _runtime_workspace(tmp_path)
    store = await _seeded_store(workspace.workspace_id, run_id="run-a")
    try:
        handler = _RecordingHandler()
        registry = _registry(handler, allowed=frozenset({AgentKind.RAG}))
        context = ToolContext(
            workspace=workspace, app_store=store, run_id="run-a", agent_kind=AgentKind.WEB
        )
        operation_id = stable_operation_id(
            "run-a", None, str(ToolName.WEB_SEARCH), arguments_hash({"text": "x", "count": 1})
        )

        with pytest.raises(ToolPermissionError):
            await registry.execute(ToolName.WEB_SEARCH, {"text": "x"}, context)

        assert handler.calls == []
        assert await store.get_operation(operation_id) is None
    finally:
        await store.close()


# ---------------------------------------------------------------- 5. 脱敏入口


def test_redact_arguments_entry_returns_redacted_payload(tmp_path: Path) -> None:
    registry = build_default_registry()
    seen: list[tuple[str, dict[str, Any]]] = []

    def _redactor(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        seen.append((tool, dict(arguments)))
        return {**arguments, "query": "[人名已脱敏]"}

    context = ToolContext(workspace=_runtime_workspace(tmp_path), redactor=_redactor)

    redacted = registry.redact_arguments(ToolName.WEB_SEARCH, {"query": "张三 的报销"}, context)

    assert redacted == {"query": "[人名已脱敏]", "max_results": 5, "topic": None,
                        "start_date": None, "end_date": None, "include_published_date": None,
                        "search_depth": None, "include_answer": None}
    assert seen[0][0] == "web_search"
    assert seen[0][1]["query"] == "张三 的报销"


def test_redact_arguments_without_redactor_returns_copy(tmp_path: Path) -> None:
    context = ToolContext(workspace=_runtime_workspace(tmp_path))
    original = {"query": "张三"}
    result = context.redact_arguments(ToolName.WEB_SEARCH, original)
    assert result == original
    assert result is not original


def test_redact_arguments_still_validates_schema(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = ToolContext(workspace=_runtime_workspace(tmp_path))
    with pytest.raises(ToolArgumentError):
        registry.redact_arguments(ToolName.WEB_SEARCH, {}, context)


# ---------------------------------------------------------------- 6. artifacts


def test_artifact_write_and_read_roundtrip(tmp_path: Path) -> None:
    context = ToolContext(workspace=_runtime_workspace(tmp_path))
    artifact_id = "a" * 32

    written = write_artifact(context, WEB_ARTIFACT_DIR, artifact_id, "网页正文", suffix=".md")

    assert written.ref == f"{WEB_ARTIFACT_DIR}/{artifact_id}.md"
    assert written.path == context.artifacts_dir / WEB_ARTIFACT_DIR / f"{artifact_id}.md"
    assert written.size == 4
    assert read_artifact(context, written.ref) == "网页正文"
    assert not written.path.with_name(written.path.name + ".tmp").exists()


def test_artifact_refuses_traversal_and_unknown_dirs(tmp_path: Path) -> None:
    context = ToolContext(workspace=_runtime_workspace(tmp_path))
    (context.artifacts_dir / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ToolExecutionError):
        read_artifact(context, "../secret.txt")
    with pytest.raises(ToolExecutionError):
        write_artifact(context, WEB_ARTIFACT_DIR, "../evil", "x")
    with pytest.raises(ToolExecutionError):
        write_artifact(context, "etc", "a" * 8, "x")
    with pytest.raises(ToolUnavailableError):
        read_artifact(context, f"{WEB_ARTIFACT_DIR}/{'b' * 32}.md")


def test_artifact_requires_workspace_directory(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class _NoArtifacts:
        workspace_id: str = "ws-a"
        knowledge_revision: int = 0

    context = ToolContext(workspace=_NoArtifacts())
    with pytest.raises(ToolUnavailableError):
        write_artifact(context, WEB_ARTIFACT_DIR, "a" * 8, "x")
