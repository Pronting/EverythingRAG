"""批准流程测试（PRD 10.2，验收 A14 / A15 / A18）。

覆盖：``ask`` 模式下联网工具执行前停顿 → 卡片落库 + ``approval.requested`` 事件 →
三种决定（允许本次 / 允许本轮同类 / 拒绝）→ 事务状态流转与幂等/冲突 409 →
``args_hash`` / ``context_epoch`` / ``rule_version`` 过期校验 → 拒绝后**工具不执行**。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.agent import contracts as c
from app.agent.approval import ApprovalConflictError, ApprovalDeniedError
from app.agent.contracts import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalScope,
    ApprovalStatus,
    RunStatus,
)
from app.api import deps as api_deps
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, build_registry, seed_conversation

RUNS = "/api/workspaces/ws-1/runs"


@pytest.fixture
async def approval_setup(app_store, tmp_path: Path):
    """ask 模式 + 需要批准的工具：跑到 waiting_approval 停下来。"""

    await seed_conversation(app_store)
    registry = build_registry()
    services = FakeServices(task_count=1, approve_tool="web_search")
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="张三最近怎么样"
    )
    await manager.run_to_completion(run.id)
    try:
        yield manager, services, registry, run
    finally:
        await manager.aclose()


async def test_ask_mode_suspends_with_saved_card(approval_setup) -> None:
    manager, _services, registry, run = approval_setup
    current = await manager.require_run(run.id)
    assert current.status is RunStatus.WAITING_APPROVAL
    cards = await manager.store.list_approvals(run.id)
    assert len(cards) == 1
    card = cards[0]
    assert card.status is ApprovalStatus.PENDING
    assert card.display.tool_name == "web_search"
    assert card.rule_version
    assert card.context_epoch == current.context_epoch
    # 卡片的参数已脱敏（PRD 10.4）：原始姓名不出现在卡片里
    assert "张三" not in str(card.display.model_dump())
    assert registry.call_log == []  # 执行前停顿：工具一次都没跑
    events = await manager.read_events(run.id)
    assert c.EventType.APPROVAL_REQUESTED in [event.type for event in events]
    requested = [event for event in events if event.type is c.EventType.APPROVAL_REQUESTED][-1]
    assert requested.payload["approval"]["approval_id"] == card.approval_id


async def test_allow_once_resumes_and_executes(approval_setup) -> None:
    manager, _services, registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    await manager.submit_approval(card.approval_id, decision_kind="allow_once")
    await manager.wait(run.id)
    assert (await manager.require_run(run.id)).status is RunStatus.COMPLETED
    assert registry.call_log == ["web_search"]  # 批准后才真正执行
    resolved = await manager.store.get_approval(card.approval_id)
    assert resolved.status is ApprovalStatus.APPROVED


async def test_allow_scope_grants_same_category_only(approval_setup) -> None:
    manager, _services, _registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    await manager.submit_approval(card.approval_id, decision_kind="allow_scope")
    await manager.wait(run.id)  # 等恢复的那次驱动收敛
    state = manager._states[run.id]
    grants = state.get("approval_grants") or []
    assert grants and grants[0]["tool_category"] == "web"
    scope = ApprovalScope.model_validate(grants[0])
    # 四项全等才算命中（PRD 10.2）
    assert scope.run_id == run.id
    assert scope.rule_version == card.rule_version
    assert scope.question_scope == "张三最近怎么样"


async def test_deny_stops_run_and_never_executes_tool(approval_setup) -> None:
    manager, _services, registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    await manager.submit_approval(card.approval_id, decision_kind="deny")
    await manager.wait(run.id)
    assert (await manager.require_run(run.id)).status is RunStatus.FAILED
    assert registry.call_log == []
    assert (await manager.store.get_approval(card.approval_id)).status is ApprovalStatus.DENIED


async def test_repeat_same_decision_is_idempotent_and_conflict_is_409(approval_setup) -> None:
    manager, _services, _registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    first = await manager.submit_approval(card.approval_id, decision_kind="allow_once")
    again = await manager.submit_approval(card.approval_id, decision_kind="allow_once")
    assert first.status is again.status is ApprovalStatus.APPROVED
    with pytest.raises(ApprovalConflictError):
        await manager.submit_approval(card.approval_id, decision_kind="deny")


async def test_stale_decision_is_rejected(approval_setup) -> None:
    """参数或上下文版本变化后，旧决定必须被拒（A15 过期决定）。"""

    manager, _services, _registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    decision = ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE, decided_at="")
    with pytest.raises(ApprovalConflictError):
        await manager.store.resolve_approval(
            card.approval_id, decision, args_hash="different-hash"
        )
    with pytest.raises(ApprovalConflictError):
        await manager.store.resolve_approval(
            card.approval_id, decision, context_epoch=int(card.context_epoch) + 1
        )
    with pytest.raises(ApprovalConflictError):
        await manager.store.resolve_approval(card.approval_id, decision, rule_version="v0")


async def test_allow_scope_without_scope_is_rejected_by_store(approval_setup) -> None:
    manager, _services, _registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    decision = ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_SCOPE, decided_at="")
    with pytest.raises(ApprovalConflictError):
        await manager.store.resolve_approval(card.approval_id, decision)


async def test_denied_tool_is_not_retried(approval_setup) -> None:
    """同一操作被拒后再次执行必须直接抛 :class:`ApprovalDeniedError`（PRD 10.2）。"""

    manager, _services, registry, run = approval_setup
    card = (await manager.store.list_approvals(run.id))[0]
    from app.agent.runtime import ApprovalGate
    from app.agent.tools import ToolContext

    context = ToolContext(
        workspace=manager._context("ws-1"),
        app_store=manager.store,
        run_id=run.id,
        task_id="t1",
        agent_kind=c.AgentKind.WEB,
    )
    gate = ApprovalGate(
        manager=manager,
        registry=registry,
        run=run,
        state={"question": "张三最近怎么样"},  # type: ignore[arg-type]
        question_scope="张三最近怎么样",
    )
    await manager.submit_approval(card.approval_id, decision_kind="deny")
    with pytest.raises(ApprovalDeniedError):
        await gate.execute("web_search", {"query": "张三 近况"}, context)


# =====================================================================
# API 层
# =====================================================================


@pytest.fixture
async def api_setup(approval_setup):
    manager, _services, _registry, run = approval_setup
    api_deps.register_app_store(manager.store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, manager, run
    app.state.run_manager = None
    api_deps.register_app_store(None)


async def test_approval_endpoint_returns_card(api_setup) -> None:
    client, manager, run = api_setup
    card = (await manager.store.list_approvals(run.id))[0]
    response = await client.post(
        f"/api/workspaces/ws-1/approvals/{card.approval_id}", json={"decision": "allow_once"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"


async def test_approval_endpoint_conflict_is_409(api_setup) -> None:
    client, manager, run = api_setup
    card = (await manager.store.list_approvals(run.id))[0]
    await client.post(
        f"/api/workspaces/ws-1/approvals/{card.approval_id}", json={"decision": "allow_once"}
    )
    conflict = await client.post(
        f"/api/workspaces/ws-1/approvals/{card.approval_id}", json={"decision": "deny"}
    )
    assert conflict.status_code == 409


async def test_approval_from_other_workspace_is_404(api_setup) -> None:
    client, manager, run = api_setup
    card = (await manager.store.list_approvals(run.id))[0]
    response = await client.post(
        f"/api/workspaces/ws-2/approvals/{card.approval_id}", json={"decision": "allow_once"}
    )
    assert response.status_code == 404
