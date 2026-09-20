"""状态机测试（PRD 12.3，验收 A13 / A14 / A24）。

两层都测：
1. **契约层**：11 个状态、迁移表、``can_transition_*`` 与 `resumable` 的唯一作用点；
2. **运行层**：RunManager 拒绝非法迁移、终态不可改写、失败收敛走两步合法迁移。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import contracts as c
from app.agent.contracts import RunStatus, TaskStatus
from app.agent.runtime import RunNotRunnableError
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, seed_conversation

# =====================================================================
# 契约层
# =====================================================================


def test_run_and_task_statuses_share_the_same_11_values() -> None:
    assert {item.value for item in RunStatus} == {item.value for item in TaskStatus}
    assert len(RunStatus) == 11
    assert set(c.RUN_STATUS_LABELS) == set(RunStatus)


def test_terminal_and_cancellable_sets_match_prd() -> None:
    assert c.TERMINAL_RUN_STATUSES == {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}
    for status in c.TERMINAL_RUN_STATUSES:
        assert c.RUN_STATUS_TRANSITIONS[status] == frozenset()
    assert RunStatus.COMPLETED not in c.CANCELLABLE_RUN_STATUSES
    assert RunStatus.RUNNING in c.CANCELLABLE_RUN_STATUSES


def test_resumable_only_affects_cancelled_to_running() -> None:
    assert c.can_transition_run(RunStatus.CANCELLED, RunStatus.RUNNING, resumable=True)
    assert not c.can_transition_run(RunStatus.CANCELLED, RunStatus.RUNNING, resumable=False)
    # 其它迁移完全不受 resumable 影响
    for status, targets in c.RUN_STATUS_TRANSITIONS.items():
        for target in targets:
            if (status, target) == (RunStatus.CANCELLED, RunStatus.RUNNING):
                continue
            assert c.can_transition_run(status, target, resumable=False)


def test_illegal_transitions_are_rejected() -> None:
    assert not c.can_transition_run(RunStatus.QUEUED, RunStatus.COMPLETED)
    assert not c.can_transition_run(RunStatus.COMPLETED, RunStatus.RUNNING)
    assert not c.can_transition_run(RunStatus.WAITING_CHILDREN, RunStatus.COMPLETED_WITH_GAPS)
    assert c.can_transition_task(TaskStatus.RUNNING, TaskStatus.COMPLETED)


# =====================================================================
# 运行层
# =====================================================================


@pytest.fixture
async def manager(app_store, tmp_path: Path):
    await seed_conversation(app_store)
    services = FakeServices(task_count=1)
    mgr = build_manager(app_store, services=services, root=tmp_path)
    try:
        yield mgr
    finally:
        await mgr.aclose()


async def test_illegal_transition_raises_and_keeps_state(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    assert run.status is RunStatus.QUEUED
    with pytest.raises(RunNotRunnableError):
        await manager._transition(run.id, RunStatus.COMPLETED)
    assert (await manager.require_run(run.id)).status is RunStatus.QUEUED


async def test_resume_rejects_non_resumable_states(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    with pytest.raises(RunNotRunnableError):
        await manager.resume(run.id)  # queued 不是可继续状态（PRD 12.3）


async def test_cancel_on_terminal_run_is_a_noop(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    done = await manager.require_run(run.id)
    assert done.status is RunStatus.COMPLETED
    after = await manager.cancel(run.id)
    assert after.status is RunStatus.COMPLETED  # 终态不被改写
    assert (await manager.require_run(run.id)).status is RunStatus.COMPLETED


async def test_failure_converges_through_legal_transitions(manager) -> None:
    """子任务失败：run 落 failed（不是永久 waiting_children），且带可继续标记。"""

    manager._services_factory = FakeServices(task_count=2, fail_tasks=(1,)).factory
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    failed = await manager.require_run(run.id)
    assert failed.status is RunStatus.FAILED
    assert failed.resumable is True
    events = await manager.read_events(run.id)
    assert events[-1].type is c.EventType.RUN_FAILED
    error = events[-1].payload["error"]
    assert error["error_code"] and error["node"] == "research"
    assert error["resumable"] is True
    assert error["required_action"]


async def test_completed_run_ends_with_run_completed_event(manager) -> None:
    """run.completed 必须是收尾帧（前端据此关闭订阅，PRD 13.3）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    events = await manager.read_events(run.id)
    assert events[-1].type in c.SUBSCRIPTION_CLOSING_EVENTS
    assert events[-1].type is c.EventType.RUN_COMPLETED
    types = [event.type for event in events]
    assert types[0] is c.EventType.RUN_STATUS
    assert c.EventType.ROUTE_DECIDED in types
    assert c.EventType.PLAN_UPDATED in types
    assert c.EventType.ANSWER_META in types
    assert c.EventType.ANSWER_DELTA in types
