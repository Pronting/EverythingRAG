import asyncio
from pathlib import Path

import pytest

from app.agent.app_store import AppStore
from app.agent.approval import (
    ApprovalConflictError,
    build_approval_request,
    build_display,
)
from app.agent.contracts import AgentKind, ApprovalStatus, EventType, RunStatus
from app.agent.runtime import RunManager
from app.agent.tools import (
    ToolArgumentError,
    ToolContext,
    ToolName,
    build_default_registry,
)
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore


async def _seed_file_store(path: Path) -> AppStore:
    store = await AppStore.open(path)
    await store.create_workspace("ws-persist", "key-persist")
    conversation = await store.create_conversation("conv-persist", "ws-persist")
    await store.create_run(
        conversation_id=conversation.id,
        workspace_id="ws-persist",
        run_id="run-persist",
    )
    await store.create_task(
        run_id="run-persist",
        task_id="task-persist",
        agent_kind=AgentKind.RAG,
        question="读取工作区概览",
    )
    return store


async def test_research_state_restores_completed_operation_after_manager_restart(
    tmp_path: Path,
) -> None:
    workspace_store = WorkspaceStore(tmp_path / "workspaces")
    record = workspace_store.create("持久恢复", workspace_id="ws-persist")
    factory = WorkspaceContextFactory(tmp_path / "workspaces", workspace_store)
    workspace = factory.resolve(record.id)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "app.db"
    registry = build_default_registry()

    first_store = await _seed_file_store(db_path)
    first_manager = RunManager(store=first_store, context_factory=factory, registry=registry)
    first_context = ToolContext(
        workspace=workspace,
        app_store=first_store,
        run_id="run-persist",
        task_id="task-persist",
        agent_kind=AgentKind.RAG,
    )
    first_state = await first_manager.research_state_for("run-persist", first_context)
    arguments = {"question": "请确认研究范围"}
    first = await first_state.execute(registry, ToolName.ASK_USER, arguments, first_context)
    with pytest.raises(ToolArgumentError):
        await registry.execute(
            ToolName.ASK_USER,
            {"unexpected": True},
            first_context,
        )
    await first_manager.aclose()
    await first_store.close()

    second_store = await AppStore.open(db_path)
    second_manager = RunManager(store=second_store, context_factory=factory, registry=registry)
    second_context = ToolContext(
        workspace=workspace,
        app_store=second_store,
        run_id="run-persist",
        task_id="task-persist",
        agent_kind=AgentKind.RAG,
    )
    restored_state = await second_manager.research_state_for(
        "run-persist", second_context
    )
    restored = await restored_state.execute(
        registry, ToolName.ASK_USER, arguments, second_context
    )
    operation = (await second_store.list_operations("run-persist"))[0]

    assert restored.data == first.data
    assert restored.reused is True
    assert restored.reuse_reason
    assert operation.attempts == 1
    await second_manager.aclose()
    await second_store.close()


async def test_concurrent_duplicate_approval_is_idempotent_in_file_sqlite(
    tmp_path: Path,
) -> None:
    store = await _seed_file_store(tmp_path / "approval.db")
    manager = RunManager(store=store, registry=build_default_registry())
    request = build_approval_request(
        run_id="run-persist",
        task_id="task-persist",
        tool_name="web_search",
        operation_id="op-concurrent-approval",
        args_hash="args-concurrent",
        display=build_display(
            purpose="核查官方文档",
            tool_name="web_search",
            task_question="核查参数范围",
        ),
    )
    await store.request_approval(request)

    first, second = await asyncio.gather(
        manager.submit_approval(request.approval_id, decision_kind="allow_once"),
        manager.submit_approval(request.approval_id, decision_kind="allow_once"),
    )
    events = await manager.read_events("run-persist")

    assert first.status is ApprovalStatus.APPROVED
    assert second.status is ApprovalStatus.APPROVED
    assert sum(event.type is EventType.APPROVAL_RESOLVED for event in events) == 1

    conflict = await asyncio.gather(
        manager.submit_approval(request.approval_id, decision_kind="deny"),
        return_exceptions=True,
    )
    assert isinstance(conflict[0], ApprovalConflictError)
    await manager.aclose()
    await store.close()


async def test_concurrent_research_state_initialization_returns_one_instance(
    tmp_path: Path,
) -> None:
    workspace_store = WorkspaceStore(tmp_path / "workspaces")
    record = workspace_store.create("并发研究记录", workspace_id="ws-persist")
    factory = WorkspaceContextFactory(tmp_path / "workspaces", workspace_store)
    workspace = factory.resolve(record.id)
    store = await _seed_file_store(tmp_path / "research-state.db")
    manager = RunManager(
        store=store,
        context_factory=factory,
        registry=build_default_registry(),
    )
    context = ToolContext(
        workspace=workspace,
        app_store=store,
        run_id="run-persist",
        task_id="task-persist",
        agent_kind=AgentKind.RAG,
    )

    states = await asyncio.gather(
        *(manager.research_state_for("run-persist", context) for _ in range(4))
    )

    assert all(state is states[0] for state in states)
    await manager.aclose()
    await store.close()


async def test_restartable_runs_excludes_user_wait_states(tmp_path: Path) -> None:
    store = await _seed_file_store(tmp_path / "restartable.db")
    await store.update_run_status("run-persist", RunStatus.RUNNING)
    waiting = await store.create_conversation("conv-waiting", "ws-persist")
    await store.create_run(
        conversation_id=waiting.id,
        workspace_id="ws-persist",
        run_id="run-waiting",
        status=RunStatus.WAITING_APPROVAL,
    )

    runs = await store.list_restartable_runs()

    assert [run.id for run in runs] == ["run-persist"]
    await store.close()
