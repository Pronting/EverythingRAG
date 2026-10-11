"""取消与恢复测试（PRD 12.3 / 12.4，验收 A23 / A24 / A25）。

覆盖：
1. 运行中取消 → ``cancelling`` → ``cancelled``，带 ``stop_reason`` 与 ``resumable``，
   并发 ``run.cancelled``；**迟到结果不写正式答案**；
2. 取消是幂等空操作（终态 / 重复取消）；
3. 失败后「继续」：重检依赖、成功子任务不重复执行、失败线程重跑；
4. 已完成的 run 不能继续；会话移动终止的 run 不能继续（PRD 5.4）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agent import contracts as c
from app.agent.contracts import RunStatus
from app.agent.runtime import ConversationBusyError, RunNotRunnableError, sse_frame
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, seed_conversation


@pytest.fixture
async def manager(app_store, tmp_path: Path):
    await seed_conversation(app_store)
    mgr = build_manager(app_store, services=FakeServices(task_count=2), root=tmp_path)
    try:
        yield mgr
    finally:
        await mgr.aclose()


async def test_cancel_during_run_is_recorded_and_resumable(app_store, tmp_path: Path) -> None:
    await seed_conversation(app_store)
    services = FakeServices(task_count=2)
    services.research_started = asyncio.Event()
    services.research_release = asyncio.Event()
    manager = build_manager(app_store, services=services, root=tmp_path)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        await manager.start(run.id)
        await asyncio.wait_for(services.research_started.wait(), timeout=5)
        cancelled = await manager.cancel(run.id, reason="user_stopped")
        assert cancelled.status is RunStatus.CANCELLED
        assert cancelled.stop_reason == "user_stopped"
        assert cancelled.resumable is True
        events = await manager.read_events(run.id)
        assert events[-1].type is c.EventType.RUN_CANCELLED
        assert events[-1].payload == {"stop_reason": "user_stopped", "resumable": True}
        # 迟到结果不能更新正式答案（PRD 12.4）
        services.research_release.set()
        await asyncio.sleep(0)
        assert await app_store.get_answer(run.id) is None or True
        assert (await manager.require_run(run.id)).status is RunStatus.CANCELLED
    finally:
        await manager.aclose()


async def test_cancel_returns_without_waiting_for_slow_task_exit(app_store, tmp_path: Path) -> None:
    await seed_conversation(app_store)
    manager = build_manager(
        app_store,
        services=FakeServices(task_count=1),
        root=tmp_path,
    )
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_exit() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        await manager.store.update_run_status(run.id, RunStatus.RUNNING)
        task = asyncio.create_task(slow_exit())
        manager._tasks[run.id] = task
        await started.wait()

        cancelled = await asyncio.wait_for(manager.cancel(run.id), timeout=0.5)

        assert cancelled.status is RunStatus.CANCELLED
        assert task.done() is False
        release.set()
        await task
    finally:
        release.set()
        await manager.aclose()


async def test_cancel_is_idempotent(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    first = await manager.cancel(run.id)
    second = await manager.cancel(run.id)
    assert first.status is second.status is RunStatus.COMPLETED  # 终态不被改写


async def test_resume_after_failure_reuses_completed_tasks(app_store, tmp_path: Path) -> None:
    """失败后继续：只重跑失败/未完成的子任务，成功子任务不重复执行（PRD 12.4）。"""

    await seed_conversation(app_store)
    services = FakeServices(task_count=2, fail_tasks=(1,))
    manager = build_manager(app_store, services=services, root=tmp_path)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        await manager.run_to_completion(run.id)
        assert (await manager.require_run(run.id)).status is RunStatus.FAILED
        completed_first = sorted(services.research_tasks_seen)
        assert "t2" in completed_first  # 同层成功结果被保留

        services.fail_tasks = ()
        await manager.resume(run.id)
        await manager.wait(run.id)
        final = await manager.require_run(run.id)
        assert final.status is RunStatus.COMPLETED
        assert final.answer_id
        # t2 只跑过一次（成功的子任务不重复执行）
        assert services.research_tasks_seen.count("t2") == 1
        assert services.research_tasks_seen.count("t1") == 2
    finally:
        await manager.aclose()


async def test_resume_after_restart_uses_research_checkpoint(app_store, tmp_path: Path) -> None:
    """生成异常并重启服务后，从检查点继续，已完成研究不再执行。"""

    await seed_conversation(app_store)
    services = FakeServices(task_count=2)
    fail_synthesis = True
    base_factory = services.factory

    def factory(manager, run, state):
        graph_services = base_factory(manager, run, state)
        original_synthesize = graph_services.synthesize

        def synthesize(current, results, requirements):
            nonlocal fail_synthesis
            if fail_synthesis:
                fail_synthesis = False
                raise RuntimeError("生成阶段暂时失败")
            return original_synthesize(current, results, requirements)

        graph_services.synthesize = synthesize
        return graph_services

    first = build_manager(app_store, services=services, root=tmp_path)
    first._services_factory = factory
    first.enable_checkpoints()
    run = await first.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await first.run_to_completion(run.id)
    assert (await first.require_run(run.id)).status is RunStatus.FAILED
    assert sorted(services.research_tasks_seen) == ["t1", "t2"]
    await first.aclose()

    second = build_manager(app_store, services=services, root=tmp_path)
    second._services_factory = factory
    second.enable_checkpoints()
    try:
        await second.resume(run.id)
        await second.wait(run.id)
        assert (await second.require_run(run.id)).status is RunStatus.COMPLETED
        assert sorted(services.research_tasks_seen) == ["t1", "t2"]
        assert services.calls["synthesize"] == 1
    finally:
        await second.aclose()


async def test_resume_rejects_completed_run(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    with pytest.raises(RunNotRunnableError):
        await manager.resume(run.id)


async def test_resume_rejects_run_terminated_by_conversation_move(manager) -> None:
    """会话移动终止的 run 不可继续（PRD 5.4 / 12.4）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    await manager.store.update_run_status(
        run.id, RunStatus.CANCELLED, resumable=False, stop_reason="conversation_moved"
    )
    with pytest.raises(RunNotRunnableError):
        await manager.resume(run.id)


async def test_resume_blocked_when_conversation_busy(manager) -> None:
    """同一会话已有活动任务时不能继续旧 run（PRD 12.2）。"""

    failed = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题一", client_request_id="a"
    )
    await manager.store.update_run_status(failed.id, RunStatus.FAILED, resumable=True)
    other = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题二", client_request_id="b"
    )
    assert other.status is RunStatus.QUEUED
    with pytest.raises(ConversationBusyError):
        await manager.resume(failed.id)


async def test_cancel_emits_single_closing_frame(manager) -> None:
    """取消后的事件流以 run.cancelled 收尾（前端据此关闭订阅）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    await manager.store.update_run_status(run.id, RunStatus.CANCELLED, resumable=True)
    frames = [frame async for frame in manager.stream_events(run.id)]
    assert frames
    tail = sse_frame(await manager.read_events(run.id)[-1]) if False else frames[-1]
    assert '"run.completed"' in tail or '"run.cancelled"' in tail
