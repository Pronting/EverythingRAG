"""会话移动（PRD 5.4 / 13.2；验收 A27 / A28）。Owner: T_a27。

逐条覆盖 t28 的交付要求：

1. API 新字段：``confirm``（确认标识）与 ``expected_context_epoch``（乐观并发 → 409）；
2. 静态移动：保留历史、读取**目标区**记忆快照、``context_epoch`` +1、响应与读入口都指向目标区；
3. 运行中移动：停止主任务及子任务 → **等待活动请求退出** → 才改归属（用探针断言顺序，
   不是只断终态）；原 run ``cancelled`` + ``stop_reason=conversation_moved`` + ``resumable=false``；
   原批准/澄清 → ``superseded``；原 run 的工作区归属不变；
4. 历史只读边界：移动后仍按**原工作区**读到 run 记录，且该入口不执行工具、不在目标区可见；
   因移动终止的 run 拒绝 ``resume`` / 拒绝提交批准与澄清；
5. 目标区继续：新 run 带 ``origin_run_id``，权限重新绑定，**不含**原区未发布的研究结果；
6. 后台提炼提交前再核对归属与 ``context_epoch``（不向错误工作区提交记忆）；
7. A28 并发：移动与提炼并发 → 更新不丢失、无跨区误提交；
另有跨两区锁序（PRD 11.2）与正向对照（没有移动时提炼照常提交，证明守卫非永真）。

测试只用 tmp 工作区与内存 app.db，全离线。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent import contracts as c
from app.agent.app_store import AppStore
from app.agent.contracts import (
    ApprovalDisplay,
    ApprovalRequest,
    ApprovalStatus,
    InputRequest,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
)
from app.agent.memory import extract_conversation_memory
from app.agent.runtime import RunManager, RunNotRunnableError
from app.api import deps as api_deps
from app.core.conversation_store import ConversationMessage, ConversationStore
from app.core.workspace_store import WorkspaceStore
from app.generation.base import ChatChunk
from app.main import app
from app.memory import locking as memory_locking
from app.memory.extractor import WorkspaceTurn
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.fakes_memory import extraction_reply, memory_for
from tests.fakes_run import FakeServices, build_manager, build_registry

SIDEBAR_DIR = "conv-data"
OLD_WORKSPACE = "ws-1"
NEW_WORKSPACE = "ws-2"
CONVERSATION = "conv-1"
QUESTION = "张三最近怎么样"


class BlockingExtractionModel:
    """提炼模型：进入 ``stream_chat`` 后先等事件，便于把「移动」插进提炼中间（A28）。"""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.started.set()
        await self.release.wait()
        yield ChatChunk("content", self._reply)


@dataclass
class MoveEnv:
    client: httpx.AsyncClient
    manager: RunManager
    workspaces: WorkspaceStore
    conversations: ConversationStore
    services: FakeServices
    tmp_path: Path


@pytest.fixture
async def env(app_store, tmp_path: Path) -> AsyncIterator[MoveEnv]:
    """两区环境：app.db（权威）与侧边栏存档在 id / storage_key 上一一对齐。"""

    await app_store.create_workspace(OLD_WORKSPACE, "sk-1", name="甲区")
    await app_store.create_workspace(NEW_WORKSPACE, "sk-2", name="乙区")
    await app_store.create_conversation(CONVERSATION, OLD_WORKSPACE)
    await app_store.append_message(
        conversation_id=CONVERSATION, role="user", content=QUESTION
    )

    workspaces = WorkspaceStore(tmp_path / SIDEBAR_DIR)
    for workspace_id, storage_key, name in (
        (OLD_WORKSPACE, "sk-1", "甲区"),
        (NEW_WORKSPACE, "sk-2", "乙区"),
    ):
        workspace = workspaces.create(name, workspace_id)
        workspace.storage_key = storage_key
        workspaces.adopt(workspace)
    conversations = ConversationStore(tmp_path / SIDEBAR_DIR)
    conversations.create(OLD_WORKSPACE, conversation_id=CONVERSATION)
    conversations.append_messages(
        CONVERSATION,
        [
            ConversationMessage(
                role="user",
                content=QUESTION,
                # 历史引用保存**原来的来源工作区**（移动后允许继续展示，PRD 5.4）
                sources=[{"label": "S1", "workspace_id": OLD_WORKSPACE, "snippet": "旧区片段"}],
            )
        ],
    )

    services = FakeServices(task_count=2)
    manager = build_manager(
        app_store, services=services, root=tmp_path, registry=build_registry()
    )
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield MoveEnv(
                client=client,
                manager=manager,
                workspaces=workspaces,
                conversations=conversations,
                services=services,
                tmp_path=tmp_path,
            )
    finally:
        app.state.run_manager = None
        api_deps.register_app_store(None)
        await manager.aclose()


def _move_url(conversation_id: str = CONVERSATION) -> str:
    return f"/api/conversations/{conversation_id}/workspace"


async def _move(env: MoveEnv, target: str = NEW_WORKSPACE, **extra: Any) -> httpx.Response:
    return await env.client.post(
        _move_url(), json={"workspace_id": target, "confirm": True, **extra}
    )


def _order(steps: list[str], *names: str) -> None:
    """断言这些步骤按给定先后出现（顺序证据，缺一个即失败）。"""

    indexes = [steps.index(name) for name in names]
    assert indexes == sorted(indexes), f"步骤顺序不对：{steps}"


async def _enable_memory(tmp_path: Path, storage_key: str, text: str) -> Any:
    memory = memory_for(tmp_path, storage_key)
    memory.set_enabled(True)
    memory.add_entry(text, "topic", user_pinned=True)
    return memory


# =====================================================================
# 1. 确认标识与乐观并发（PRD 13.2 末段）
# =====================================================================


async def test_confirmation_required_when_history_is_carried(env: MoveEnv, app_store) -> None:
    """带历史的会话必须先确认；未确认时**什么都不改**（不是先把历史搬走再报错）。"""

    unconfirmed = await env.client.post(
        _move_url(), json={"workspace_id": NEW_WORKSPACE}
    )
    assert unconfirmed.status_code == 409
    assert "确认" in unconfirmed.json()["detail"]

    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == OLD_WORKSPACE
    assert record.context_epoch == 1
    assert env.conversations.get(CONVERSATION).workspace_id == OLD_WORKSPACE

    moved = await _move(env)
    assert moved.status_code == 200, moved.text
    assert moved.json()["context_epoch"] == 2
    assert moved.json()["steps"][0] == "confirmed"
    assert (await app_store.get_conversation(CONVERSATION)).workspace_id == NEW_WORKSPACE


async def test_stale_expected_context_epoch_is_rejected(env: MoveEnv, app_store) -> None:
    """乐观并发：``expected_context_epoch`` 不一致 → 409，且不改任何东西。"""

    stale = await _move(env, expected_context_epoch=99)
    assert stale.status_code == 409
    assert "上下文版本" in stale.json()["detail"]
    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == OLD_WORKSPACE
    assert record.context_epoch == 1

    ok = await _move(env, expected_context_epoch=1)
    assert ok.status_code == 200, ok.text
    assert ok.json()["previous_context_epoch"] == 1
    assert ok.json()["context_epoch"] == 2


async def test_move_unknown_conversation_or_workspace_is_404(env: MoveEnv) -> None:
    assert (await env.client.post(_move_url("nope"), json={"workspace_id": NEW_WORKSPACE})).status_code == 404
    assert (await _move(env, "nope")).status_code == 404


# =====================================================================
# 2. 静态移动（无活动 run）
# =====================================================================


async def test_static_move_preserves_history_and_rebinds_target_snapshot(
    env: MoveEnv, app_store, tmp_path: Path
) -> None:
    """保留历史 + 读取目标区快照 + ``context_epoch`` +1 + 历史引用仍指向原工作区。"""

    target_memory = await _enable_memory(tmp_path, "sk-2", "目标区的一条记忆")
    src_memory = await _enable_memory(tmp_path, "sk-1", "原区的一条记忆")

    response = await _move(env)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["from_workspace_id"] == OLD_WORKSPACE
    assert payload["workspace_id"] == NEW_WORKSPACE
    assert payload["memory_snapshot_rebound"] is True
    _order(payload["steps"], "confirmed", "lock_acquired", "app_db_moved", "sidebar_moved", "snapshot_rebound", "done")

    # app.db 权威行：目标区 + 记录了来源区 + epoch +1 + 快照来自**目标区**
    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == NEW_WORKSPACE
    assert record.moved_from_workspace_id == OLD_WORKSPACE
    assert record.context_epoch == 2
    assert record.memory_snapshot_at
    assert "目标区" in record.memory_snapshot_text
    assert "原区" not in record.memory_snapshot_text
    # 目标区记忆本身没有被改写（移动只读它，不写它）
    assert target_memory.load().revision == 1
    assert src_memory.load().revision == 1

    # 历史消息与历史引用原样保留（引用仍指向原工作区）——按**用户可见**的读路径断言
    detail = (await env.client.get(f"/api/conversations/{CONVERSATION}")).json()
    assert detail["workspace_id"] == NEW_WORKSPACE
    assert [message["content"] for message in detail["messages"]] == [QUESTION]
    assert detail["messages"][0]["sources"][0]["workspace_id"] == OLD_WORKSPACE

    # 后续检索使用目标工作区：侧边栏与 app.db 都在新区
    assert [item["id"] for item in (await env.client.get(
        "/api/conversations", params={"workspace": NEW_WORKSPACE}
    )).json()] == [CONVERSATION]
    assert (await env.client.get(
        "/api/conversations", params={"workspace": OLD_WORKSPACE}
    )).json() == []


async def test_move_to_same_workspace_is_noop(env: MoveEnv, app_store) -> None:
    """同区「移动」是空操作：不递增 epoch、不清快照、不要求确认。"""

    await _enable_memory(env.tmp_path, "sk-1", "原区的一条记忆")
    await _move(env)  # 先真的移到新区并固定快照
    before = await app_store.get_conversation(CONVERSATION)
    assert before.memory_snapshot_at

    again = await _move(env, NEW_WORKSPACE, confirm=False)
    assert again.status_code == 200, again.text
    assert "noop_same_workspace" in again.json()["steps"]
    after = await app_store.get_conversation(CONVERSATION)
    assert after.context_epoch == before.context_epoch == 2
    assert after.memory_snapshot_at == before.memory_snapshot_at


# =====================================================================
# 3. 运行中移动（最容易做错的一半）
# =====================================================================


async def test_running_move_stops_runs_and_waits_for_exit_before_ownership_change(
    env: MoveEnv, app_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """顺序证据：改归属的那一刻，活动请求**已经**退出、run 已经 cancelled。"""

    env.services.research_started = asyncio.Event()
    env.services.research_release = asyncio.Event()
    created = await env.client.post(
        f"/api/workspaces/{OLD_WORKSPACE}/runs",
        json={"conversation_id": CONVERSATION, "message": QUESTION},
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    await asyncio.wait_for(env.services.research_started.wait(), timeout=5)
    assert env.manager.is_running(run_id)

    probes: list[dict[str, Any]] = []
    original = AppStore.move_conversation

    async def spy(self: AppStore, conversation_id: str, **kwargs: Any) -> Any:
        current = await self.get_run(run_id)
        probes.append(
            {
                "running": env.manager.is_running(run_id),
                "status": current.status,
                "resumable": bool(current.resumable),
                "active": [item.id for item in await self.list_active_runs(CONVERSATION)],
            }
        )
        return await original(self, conversation_id, **kwargs)

    monkeypatch.setattr(AppStore, "move_conversation", spy)

    response = await _move(env, expected_context_epoch=1)
    assert response.status_code == 200, response.text
    payload = response.json()
    _order(
        payload["steps"],
        "confirmed",
        "stop_requested",
        "runs_stopped",
        "activity_exited",
        "app_db_moved",
        "sidebar_moved",
        "done",
    )
    assert payload["stopped_run_ids"] == [run_id]

    # 探针（改归属的**那一刻**）：驱动任务已结束、run 已是 cancelled、会话已无活动 run
    assert probes, "必须观察到改归属那一刻的状态"
    assert probes[0]["running"] is False
    assert probes[0]["status"] is RunStatus.CANCELLED
    assert probes[0]["resumable"] is False
    assert probes[0]["active"] == []

    # 终态：因移动终止，归属不变
    run = await app_store.get_run(run_id)
    assert run.status is RunStatus.CANCELLED
    assert run.stop_reason == "conversation_moved"
    assert run.resumable is False
    assert run.workspace_id == OLD_WORKSPACE
    events = await env.manager.read_events(run_id)
    assert events[-1].type is c.EventType.RUN_CANCELLED
    assert events[-1].payload == {"stop_reason": "conversation_moved", "resumable": False}
    # 迟到结果不能改写终态（PRD 12.4）
    env.services.research_release.set()
    await asyncio.sleep(0)
    assert (await app_store.get_run(run_id)).status is RunStatus.CANCELLED


async def test_running_move_supersedes_pending_approval_and_clarification(
    env: MoveEnv, app_store
) -> None:
    """原批准与澄清请求转为 ``superseded``；响应给出可核对的计数。"""

    run = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question=QUESTION
    )
    await app_store.request_approval(
        ApprovalRequest(
            approval_id="ap-1",
            run_id=run.id,
            operation_id="op-ap-1",
            args_hash="hash-1",
            context_epoch=run.context_epoch,
            display=ApprovalDisplay(purpose="联网搜索", tool_name="web_search"),
        )
    )
    await app_store.request_input(
        InputRequest(input_request_id="in-1", run_id=run.id, question="范围？")
    )
    assert (await app_store.get_approval("ap-1")).status is ApprovalStatus.PENDING

    response = await _move(env)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["stopped_run_ids"] == [run.id]
    assert payload["superseded_approvals"] == 1
    assert payload["superseded_inputs"] == 1

    approval = await app_store.get_approval("ap-1")
    assert approval.status is ApprovalStatus.SUPERSEDED
    # 失效原因写明来源（PRD 5.4 要求说明会话移动）
    assert "conversation_moved" in approval.superseded_reason
    stored_input = await app_store.get_input_request("in-1")
    assert stored_input.status is c.InputRequestStatus.SUPERSEDED
    # 原 run 仍是原区归属，且因移动终止
    run_after = await app_store.get_run(run.id)
    assert run_after.workspace_id == OLD_WORKSPACE
    assert run_after.stop_reason == "conversation_moved"
    assert run_after.resumable is False


async def test_moved_run_rejects_resume_approval_and_clarification(
    env: MoveEnv, app_store
) -> None:
    """拒绝路径：因移动终止的 run 不能恢复、不能提交批准/澄清（明确拒绝，不是 200）。"""

    run = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question=QUESTION
    )
    await app_store.request_approval(
        ApprovalRequest(
            approval_id="ap-2",
            run_id=run.id,
            operation_id="op-ap-2",
            args_hash="hash-2",
            context_epoch=run.context_epoch,
            display=ApprovalDisplay(purpose="联网搜索", tool_name="web_search"),
        )
    )
    await app_store.request_input(
        InputRequest(input_request_id="in-2", run_id=run.id, question="范围？")
    )
    assert (await env.manager.require_run(run.id)).status is RunStatus.QUEUED  # 活动 run

    response = await _move(env)
    assert response.status_code == 200, response.text
    run_after = await app_store.get_run(run.id)
    assert run_after.stop_reason == "conversation_moved"
    assert run_after.resumable is False
    assert run_after.status is RunStatus.CANCELLED

    # 运行时守卫（三条路径都在运行层拒绝，不只依赖路由）
    with pytest.raises(RunNotRunnableError):
        await env.manager.resume(run.id)
    with pytest.raises(RunNotRunnableError):
        await env.manager.submit_approval("ap-2", decision_kind="deny")
    with pytest.raises(RunNotRunnableError):
        await env.manager.submit_input("in-2", answer="随便")

    # API 守卫：按 run 创建时的工作区访问，三条都不得 200
    base = f"/api/workspaces/{OLD_WORKSPACE}/runs/{run.id}"
    resume_response = await env.client.post(f"{base}/resume")
    assert resume_response.status_code == 409, resume_response.text
    approval_response = await env.client.post(
        f"{base}/approvals/ap-2", json={"decision": "deny"}
    )
    assert approval_response.status_code == 409, approval_response.text
    input_response = await env.client.post(
        f"{base}/inputs/in-2", json={"answer": "随便", "question_version": 1}
    )
    assert input_response.status_code == 409, input_response.text


# =====================================================================
# 4. 历史 run 的只读边界
# =====================================================================


async def test_historical_run_is_read_only_under_original_workspace(
    env: MoveEnv, app_store
) -> None:
    """移动后仍按**原工作区**读历史 run；该入口不执行工具，也不在目标区可见。"""

    # 活动 run：移动会把它终止（终止后的记录正是"历史 run"）
    run = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question=QUESTION
    )
    response = await _move(env)
    assert response.status_code == 200, response.text

    calls_before = dict(env.services.calls)
    detail = await env.client.get(f"/api/workspaces/{OLD_WORKSPACE}/runs/{run.id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["run_id"] == run.id
    assert body["workspace_id"] == OLD_WORKSPACE  # 归属未变
    assert body["resumable"] is False
    assert body["stop_reason"] == "conversation_moved"

    log = await env.client.get(f"/api/workspaces/{OLD_WORKSPACE}/runs/{run.id}/event-log")
    assert log.status_code == 200
    assert any(event["type"] == "run.cancelled" for event in log.json()["events"])

    # 只读入口不执行工具
    assert dict(env.services.calls) == calls_before

    # 目标区看不到这次历史 run（只读校验按 run 创建时的工作区）
    assert (await env.client.get(f"/api/workspaces/{NEW_WORKSPACE}/runs/{run.id}")).status_code == 404
    assert (
        await env.client.get(f"/api/workspaces/{NEW_WORKSPACE}/runs/{run.id}/event-log")
    ).status_code == 404
    # 原区其它资料不因此开放：run 详情里只有该 run 的记录，没有来源片段正文
    assert "旧区片段" not in detail.text
    for task in body.get("tasks", []):
        assert "snippet" not in task


# =====================================================================
# 5. 目标区继续当前问题
# =====================================================================


async def test_target_workspace_new_run_records_origin_and_carries_nothing_over(
    env: MoveEnv, app_store
) -> None:
    """新 run 带 ``origin_run_id``、绑定目标区；未执行前**不含**原区任何研究结果。"""

    old_run = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question=QUESTION
    )
    await env.manager.run_to_completion(old_run.id)
    old_tasks = await env.manager.list_tasks(old_run.id)
    assert old_tasks  # 原区确实有研究子任务（否则隔离断言永真）
    old_answer = (await app_store.get_run(old_run.id)).answer_id
    moved = await _move(env)
    assert moved.status_code == 200, moved.text
    conversation = await app_store.get_conversation(CONVERSATION)
    assert conversation.context_epoch == 2

    created = await env.client.post(
        f"/api/workspaces/{NEW_WORKSPACE}/runs",
        json={
            "conversation_id": CONVERSATION,
            "message": "继续问：后来呢",
            "origin_run_id": old_run.id,
        },
    )
    assert created.status_code == 202, created.text
    new_run = created.json()
    assert new_run["workspace_id"] == NEW_WORKSPACE
    assert new_run["context_epoch"] == 2
    stored = await app_store.get_run(new_run["run_id"])
    assert stored.origin_run_id == old_run.id

    # 让这次执行确定性收尾（后台驱动），再断言它有自己的答案、且不是原 run 的
    await env.manager.wait(new_run["run_id"])
    settled = await app_store.get_run(new_run["run_id"])
    assert settled.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}
    assert settled.answer_id and settled.answer_id != old_answer
    # 原 run 的答案与归属一字未动
    assert (await app_store.get_run(old_run.id)).workspace_id == OLD_WORKSPACE
    assert (await app_store.get_run(old_run.id)).answer_id == old_answer


async def test_target_area_fresh_run_starts_empty(env: MoveEnv, app_store) -> None:
    """未执行的第二个 run（目标区）：任务、答案、证据全部从零开始（A27 隔离）。"""

    old_run = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question=QUESTION
    )
    await env.manager.run_to_completion(old_run.id)
    old_tasks = await env.manager.list_tasks(old_run.id)
    old_snapshot = [(task.id, task.thread_id) for task in old_tasks]
    assert old_snapshot

    moved = await _move(env)
    assert moved.status_code == 200, moved.text

    # 刻意不 start：断言「新执行从零开始」，不受后台驱动时序影响
    fresh = await env.manager.create_run(
        workspace_id=NEW_WORKSPACE,
        conversation_id=CONVERSATION,
        question="继续问：后来呢",
        origin_run_id=old_run.id,
    )
    assert fresh.workspace_id == NEW_WORKSPACE
    assert fresh.origin_run_id == old_run.id
    assert fresh.context_epoch == 2
    assert await env.manager.list_tasks(fresh.id) == []
    assert await app_store.get_answer(fresh.id) is None
    # 原 run 的记录此时一字未动（原区未发布的研究结果没有进入新执行）
    assert [
        (task.id, task.thread_id) for task in await env.manager.list_tasks(old_run.id)
    ] == old_snapshot


async def test_second_run_task_records_are_run_scoped(env: MoveEnv, app_store) -> None:
    """同一会话连续多个 run 的任务记录各自归属、互不污染（v3 run 作用域身份）。

    这是 t30/t31 修掉的缺陷的**回归**：修前第二个 run 的 ``list_tasks`` 为空（id 撞车被
    静默丢弃），且第一个 run 的行会被改写（``thread_id`` 变成第二个 run 的）。
    """

    first = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question="第一问"
    )
    await env.manager.run_to_completion(first.id)
    first_tasks = await env.manager.list_tasks(first.id)
    assert first_tasks
    first_snapshot = {(task.id, task.thread_id, task.status) for task in first_tasks}

    second = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question="第二问"
    )
    await env.manager.run_to_completion(second.id)

    second_tasks = await env.manager.list_tasks(second.id)
    assert second_tasks, "第二个 run 必须有自己的任务记录"
    assert {task.run_id for task in second_tasks} == {second.id}
    # 同名任务在两个 run 里是**两条独立记录**（同一 task_id、不同 run）
    assert {task.id for task in second_tasks} == {task.id for task in first_tasks}
    # 历史 run 的记录不得被后继执行改写
    assert {
        (task.id, task.thread_id, task.status) for task in await env.manager.list_tasks(first.id)
    } == first_snapshot

    # 第三个 run 同样完整（不是「只有第二个特殊」）
    third = await env.manager.create_run(
        workspace_id=OLD_WORKSPACE, conversation_id=CONVERSATION, question="第三问"
    )
    await env.manager.run_to_completion(third.id)
    assert len(await env.manager.list_tasks(third.id)) == len(second_tasks)
    assert await env.manager.list_tasks(second.id)  # 第二个 run 的记录仍在


async def test_target_workspace_defaults_rebind_only_inherited_permission(
    env: MoveEnv, app_store, tmp_path: Path
) -> None:
    """PRD 5.4「重新绑定权限」：继承来的权限改绑目标区默认；用户显式设过的不动。"""

    target = env.workspaces.get(NEW_WORKSPACE)
    target.default_permission_mode = PermissionMode.FULL
    target.default_reasoning_level = ReasoningLevel.LOW
    env.workspaces.adopt(target)
    source_default = str(env.workspaces.get(OLD_WORKSPACE).default_permission_mode)
    before = await app_store.get_conversation(CONVERSATION)
    assert str(before.permission_mode) == source_default  # 继承自源区默认

    moved = await _move(env)
    assert moved.status_code == 200, moved.text
    assert moved.json()["permission_rebound"] is True
    after = await app_store.get_conversation(CONVERSATION)
    assert after.permission_mode is PermissionMode.FULL
    assert after.reasoning_level is ReasoningLevel.LOW
    # 新 run 绑定的是重新绑定后的权限（目标区默认），不是旧区的
    created = await env.client.post(
        f"/api/workspaces/{NEW_WORKSPACE}/runs",
        json={"conversation_id": CONVERSATION, "message": "继续"},
    )
    assert created.status_code == 202, created.text
    assert created.json()["permission_mode"] == PermissionMode.FULL

    # 用户显式改过的权限与强度：移动**不得**被目标区默认值吞掉
    await app_store.update_conversation(
        CONVERSATION, permission_mode=PermissionMode.AUTO, reasoning_level=ReasoningLevel.MAX
    )
    explicit = await _move(env, OLD_WORKSPACE)
    assert explicit.status_code == 200, explicit.text
    assert explicit.json()["permission_rebound"] is False
    kept = await app_store.get_conversation(CONVERSATION)
    assert kept.permission_mode is PermissionMode.AUTO
    assert kept.reasoning_level is ReasoningLevel.MAX


# =====================================================================
# 6./7. 后台提炼的一致性（A28）
# =====================================================================


def _turns() -> list[WorkspaceTurn]:
    return [WorkspaceTurn(role="user", content=QUESTION, message_id="m-1", sequence=1)]


def _reply() -> str:
    return extraction_reply(
        [
            {
                "kind": "topic",
                "text": "用户想了解张三的近况",
                "quote": QUESTION,
                "confidence": 0.9,
            }
        ]
    )


async def test_extraction_refuses_to_commit_after_conversation_moved(
    env: MoveEnv, app_store, tmp_path: Path
) -> None:
    """A28：提炼提交前再核对归属 → 移动后不得把记忆写进原工作区。"""

    src_memory = await _enable_memory(tmp_path, "sk-1", "原区的一条记忆")

    class Model(BlockingExtractionModel):
        pass

    model = Model(_reply())
    task = asyncio.create_task(
        extract_conversation_memory(
            memory=src_memory,
            chat_model=model,
            conversation_id=CONVERSATION,
            turns=_turns(),
            operation_id="op-1",
            cursor_message_id="m-1",
            app_store=app_store,
            expected_context_epoch=1,
        )
    )
    await asyncio.wait_for(model.started.wait(), timeout=5)

    # 提炼模型在跑的窗口里完成移动
    moved = await _move(env, expected_context_epoch=1)
    assert moved.status_code == 200, moved.text
    model.release.set()
    outcome = await asyncio.wait_for(task, timeout=5)

    assert outcome.applied is False, "移动后不得提交"
    assert "移动" in outcome.reason or "不提交" in outcome.reason
    # 原区记忆没有新增条目，目标区也没有被误写
    assert src_memory.load().revision == 1
    assert all("张三" not in entry.text for entry in src_memory.load().entries)
    target_memory = memory_for(tmp_path, "sk-2")
    assert all("张三" not in entry.text for entry in target_memory.load().entries)


async def test_extraction_still_commits_when_no_move_happened(
    env: MoveEnv, app_store, tmp_path: Path
) -> None:
    """正向对照：没有移动时同一调用照常提交（守卫不是永真拒绝）。"""

    src_memory = await _enable_memory(tmp_path, "sk-1", "原区的一条记忆")
    model = BlockingExtractionModel(_reply())
    model.release.set()
    outcome = await extract_conversation_memory(
        memory=src_memory,
        chat_model=model,
        conversation_id=CONVERSATION,
        turns=_turns(),
        operation_id="op-2",
        cursor_message_id="m-1",
        app_store=app_store,
        expected_context_epoch=1,
    )
    assert outcome.applied is True
    state = src_memory.load()
    assert state.revision == 2
    assert any("张三" in entry.text for entry in state.entries)


async def test_concurrent_move_and_extraction_lose_nothing(
    env: MoveEnv, app_store, tmp_path: Path
) -> None:
    """A28 并发：移动与提炼同时在飞 → 无异常、无丢失、无跨区误提交。"""

    src_memory = await _enable_memory(tmp_path, "sk-1", "原区的一条记忆")
    await _enable_memory(tmp_path, "sk-2", "目标区的一条记忆")

    model = BlockingExtractionModel(_reply())
    extraction = asyncio.create_task(
        extract_conversation_memory(
            memory=src_memory,
            chat_model=model,
            conversation_id=CONVERSATION,
            turns=_turns(),
            operation_id="op-3",
            cursor_message_id="m-1",
            app_store=app_store,
            expected_context_epoch=1,
        )
    )
    await asyncio.wait_for(model.started.wait(), timeout=5)

    # 真并发：移动在提炼未提交时发起（两者共用会话级 asyncio 锁）
    move_task = asyncio.create_task(_move(env, expected_context_epoch=1))
    await asyncio.sleep(0)
    assert not move_task.done() and not extraction.done()  # 两者确实同时在飞
    model.release.set()
    response, outcome = await asyncio.gather(move_task, extraction)
    assert response.status_code == 200, response.text

    # 无丢失：原区既有条目还在，最多新增一条，且没有重复 operation_id 造成的重复条目
    src_state = src_memory.load()
    assert any(entry.text == "原区的一条记忆" for entry in src_state.entries)
    assert src_state.revision in (1, 2)
    assert len([e for e in src_state.entries if "张三" in e.text]) == (
        1 if outcome.applied else 0
    )
    # 不向错误工作区提交：目标区永远收不到这次提炼
    target_state = memory_for(tmp_path, "sk-2").load()
    assert all("张三" not in entry.text for entry in target_state.entries)
    assert any(entry.text == "目标区的一条记忆" for entry in target_state.entries)

    # 移动本身仍然完成：epoch +1、快照来自目标区、run 归属未变
    conversation = await app_store.get_conversation(CONVERSATION)
    assert conversation.workspace_id == NEW_WORKSPACE
    assert conversation.context_epoch == 2
    assert "目标区" in conversation.memory_snapshot_text


# =====================================================================
# 跨两区锁序（PRD 11.2）
# =====================================================================


async def test_move_locks_both_workspaces_in_storage_key_order(
    env: MoveEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """跨两区按 ``storage_key`` 字典序取工作区锁；传参顺序不影响获取顺序。"""

    acquired: list[str] = []
    real = memory_locking.workspace_lock

    def spy(key: str) -> Any:
        acquired.append(key)
        return real(key)

    monkeypatch.setattr(memory_locking, "workspace_lock", spy)

    moved = await _move(env)
    assert moved.status_code == 200, moved.text
    assert acquired[:2] == ["sk-1", "sk-2"]  # 字典序（sk-1 < sk-2）

    # 反方向移动（新区 → 老区）仍然是同一获取顺序；先备好记忆再开始记录
    await _enable_memory(env.tmp_path, "sk-1", "老区记忆")
    acquired.clear()
    back = await _move(env, OLD_WORKSPACE)
    assert back.status_code == 200, back.text
    assert acquired[:2] == ["sk-1", "sk-2"]
