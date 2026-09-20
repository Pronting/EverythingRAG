"""第二条移动入口（侧边栏拖拽 ``POST /api/conversations/{id}/move``）必须走同一条状态机。

背景（`verifier2` 独立探针实测，captain 采信）：该入口早先**直接** ``store.move_in_sidebar``
改 JSON 存档归属 —— 实测返回 200，但**不停止原 run**（仍 ``waiting_approval``、
``stop_reason=None``）、**不递增 context_epoch**、**不 supersede 批准/澄清**、**不要求 confirm**、
**不写 app.db** ⇒ 侧边栏与 app.db 权威行分叉，PRD 5.4 的移动状态机在**最常用的拖拽路径**上被
整条绕过（而前端 ``App.tsx`` 拖拽正是在用这个入口）。

本文件的守卫分三层：
1. **行为**：该入口跨区时必须确认、走状态机、权威一致、失败不留中间态；
2. **非永真反向对照**：`_authority_diverged` 检测器必须能判死「只改存档」的旧行为（直接调用
   旧原语复现它），否则整套断言可能只是恒真；
3. **结构锁**：两个入口共用**唯一**实现 ``_move_via_state_machine``。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import httpx
import pytest

from app.agent.contracts import (
    ApprovalDisplay,
    ApprovalRequest,
    ApprovalStatus,
    InputRequest,
    InputRequestStatus,
    RunStatus,
)
from app.api.routes import conversations as conversations_routes
from tests.fakes_agent import app_store  # noqa: F401 -- 复用的 env 夹具依赖它
from tests.test_conversation_move import (  # noqa: F401 -- 复用 t28 的双区夹具
    CONVERSATION,
    NEW_WORKSPACE,
    OLD_WORKSPACE,
    QUESTION,
    SIDEBAR_DIR,
    MoveEnv,
    env,
)

DRAG_URL = f"/api/conversations/{CONVERSATION}/move"
WORKSPACE_URL = f"/api/conversations/{CONVERSATION}/workspace"


async def _drag(env: MoveEnv, target: str = NEW_WORKSPACE, **extra: Any) -> httpx.Response:
    return await env.client.post(DRAG_URL, json={"workspace_id": target, **extra})


def _sidebar_workspace(env: MoveEnv, conversation_id: str = CONVERSATION) -> str:
    """从**磁盘**重读侧边栏存档。

    ``ConversationStore`` 有内存态：路由用它自己的实例写入后，``env.conversations``
    那份（夹具构造时载入的）会读到旧值 —— 跨实例断言「存档内容」必须重读文件。
    """

    from app.core.conversation_store import ConversationStore

    store = ConversationStore(env.tmp_path / SIDEBAR_DIR)
    conversation = store.get(conversation_id)
    assert conversation is not None, "存档里必须有该会话（否则检测器本身失效）"
    return str(conversation.workspace_id)


async def _authority_diverged(env: MoveEnv, app_store: Any) -> bool:
    """权威一致性检测器：**存档归属 != app.db 归属** 即为分叉（t35 的缺陷判据）。"""

    record = await app_store.get_conversation(CONVERSATION)
    return str(record.workspace_id) != _sidebar_workspace(env)


async def _start_running_run(env: MoveEnv) -> str:
    """起一个**真的在跑**的 run（研究阶段被闸住），便于断言「移动停止了它」。"""

    env.services.research_started = asyncio.Event()
    env.services.research_release = asyncio.Event()
    created = await env.client.post(
        f"/api/workspaces/{OLD_WORKSPACE}/runs",
        json={"conversation_id": CONVERSATION, "message": QUESTION},
    )
    assert created.status_code == 202, created.text
    run_id = str(created.json()["run_id"])
    await asyncio.wait_for(env.services.research_started.wait(), timeout=5)
    assert env.manager.is_running(run_id)
    return run_id


# =====================================================================
# 1. 跨区拖拽必须确认（反向对照：旧行为会 200 且改存档）
# =====================================================================


async def test_cross_area_drag_requires_confirmation_and_changes_nothing(
    env: MoveEnv, app_store
) -> None:
    """未确认 → 409，且**什么都不改**。

    **这是本轮的非永真反向对照**：修复前该请求返回 200、只改存档归属、app.db 不动 ——
    本用例的每一条断言当时都会红。
    """

    run_id = await _start_running_run(env)
    response = await _drag(env)  # 没有 confirm
    assert response.status_code == 409, response.text
    assert "确认" in response.json()["detail"]

    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == OLD_WORKSPACE, "未确认不得改 app.db 归属"
    assert record.context_epoch == 1, "未确认不得改 context_epoch"
    assert _sidebar_workspace(env) == OLD_WORKSPACE, (
        "未确认不得改存档归属（旧行为正是在这里分叉的）"
    )
    assert await _authority_diverged(env, app_store) is False
    # run 必须原封不动（没被停止、没有中间态）
    current = await app_store.get_run(run_id)
    assert current.status is RunStatus.RUNNING
    assert current.stop_reason is None
    assert env.manager.is_running(run_id)
    env.services.research_release.set()


# =====================================================================
# 2. 确认后：走同一条状态机 + 权威一致
# =====================================================================


async def test_cross_area_drag_with_confirm_runs_the_full_state_machine(
    env: MoveEnv, app_store
) -> None:
    """确认后：停 run → 递增 epoch → supersede → app.db 权威 → 存档镜像。"""

    run_id = await _start_running_run(env)
    await app_store.request_approval(
        ApprovalRequest(
            approval_id="ap-drag",
            run_id=run_id,
            operation_id="op-drag",
            args_hash="hash-drag",
            context_epoch=1,
            display=ApprovalDisplay(purpose="联网搜索", tool_name="web_search"),
        )
    )
    await app_store.request_input(
        InputRequest(input_request_id="in-drag", run_id=run_id, question="范围？")
    )

    response = await _drag(env, confirm=True, expected_context_epoch=1)
    assert response.status_code == 200, response.text
    payload = response.json()

    steps = payload["steps"]
    indexes = [
        steps.index(name)
        for name in (
            "confirmed",
            "stop_requested",
            "runs_stopped",
            "activity_exited",
            "app_db_moved",
            "sidebar_moved",
            "done",
        )
    ]
    assert indexes == sorted(indexes), f"必须走完整状态机顺序：{steps}"
    assert payload["stopped_run_ids"] == [run_id]
    assert payload["superseded_approvals"] == 1
    assert payload["superseded_inputs"] == 1
    assert payload["previous_context_epoch"] == 1
    assert payload["context_epoch"] == 2

    # 原 run：已停止、不可恢复、归属仍在原区
    run = await app_store.get_run(run_id)
    assert run.status is RunStatus.CANCELLED
    assert run.stop_reason == "conversation_moved"
    assert run.resumable is False
    assert run.workspace_id == OLD_WORKSPACE
    assert (await app_store.get_approval("ap-drag")).status is ApprovalStatus.SUPERSEDED
    stored_input = await app_store.get_input_request("in-drag")
    assert stored_input.status is InputRequestStatus.SUPERSEDED

    # 权威单一：app.db == 存档 == 响应
    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == NEW_WORKSPACE
    assert _sidebar_workspace(env) == NEW_WORKSPACE
    assert payload["workspace_id"] == NEW_WORKSPACE
    assert await _authority_diverged(env, app_store) is False
    env.services.research_release.set()


# =====================================================================
# 3. 两个入口是同一条状态机（同输入同结果）
# =====================================================================


@pytest.mark.parametrize("url", [DRAG_URL, WORKSPACE_URL])
async def test_both_entries_require_confirmation_identically(
    env: MoveEnv, app_store, url: str
) -> None:
    response = await env.client.post(url, json={"workspace_id": NEW_WORKSPACE})
    assert response.status_code == 409, response.text
    assert await _authority_diverged(env, app_store) is False


@pytest.mark.parametrize("url", [DRAG_URL, WORKSPACE_URL])
async def test_both_entries_agree_after_confirmation(
    env: MoveEnv, app_store, url: str
) -> None:
    response = await env.client.post(
        url, json={"workspace_id": NEW_WORKSPACE, "confirm": True}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in (
        "workspace_id",
        "from_workspace_id",
        "context_epoch",
        "previous_context_epoch",
        "stopped_run_ids",
        "superseded_approvals",
        "superseded_inputs",
        "memory_snapshot_rebound",
        "permission_rebound",
        "steps",
    ):
        assert key in payload, f"两个入口的响应契约必须一致（缺 {key}）"
    assert payload["workspace_id"] == NEW_WORKSPACE
    assert payload["from_workspace_id"] == OLD_WORKSPACE
    assert payload["context_epoch"] == 2
    assert await _authority_diverged(env, app_store) is False


# =====================================================================
# 4. 失败路径不留中间态
# =====================================================================


async def test_drag_epoch_mismatch_leaves_no_partial_move(
    env: MoveEnv, app_store
) -> None:
    run_id = await _start_running_run(env)
    response = await _drag(env, confirm=True, expected_context_epoch=99)
    assert response.status_code == 409, response.text
    assert "上下文版本" in response.json()["detail"]

    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == OLD_WORKSPACE
    assert record.context_epoch == 1
    assert _sidebar_workspace(env) == OLD_WORKSPACE
    assert await _authority_diverged(env, app_store) is False
    current = await app_store.get_run(run_id)
    assert current.status is RunStatus.RUNNING  # 没被停掉
    assert current.stop_reason is None
    env.services.research_release.set()


# =====================================================================
# 5. 同区拖拽只是排序（不涉及归属，不触发状态机）
# =====================================================================


async def test_same_area_drag_is_ordering_only(env: MoveEnv, app_store) -> None:
    response = await env.client.post(
        DRAG_URL, json={"workspace_id": OLD_WORKSPACE, "target_index": 0}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "steps" not in payload, "同区排序不是跨区移动，不应触发状态机"
    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == OLD_WORKSPACE
    assert record.context_epoch == 1, "同区排序不得递增 context_epoch"
    assert _sidebar_workspace(env) == OLD_WORKSPACE
    assert await _authority_diverged(env, app_store) is False

    # 缺 target_index 的同区排序（拖到本区任意位置）同样只动存档顺序
    reordered = await env.client.post(DRAG_URL, json={"workspace_id": OLD_WORKSPACE})
    assert reordered.status_code == 200, reordered.text


# =====================================================================
# 6. 非永真证明：旧「只改存档」行为必须被判死
# =====================================================================


async def test_archive_only_move_is_caught_by_the_consistency_detector(
    env: MoveEnv, app_store
) -> None:
    """直接调用**旧原语**复现缺陷，检测器必须报分叉；修复后的入口必须报一致。"""

    assert await _authority_diverged(env, app_store) is False
    # 旧行为：只改存档归属（这正是 conversations.py:593 早先做的事）
    env.conversations.move_in_sidebar(CONVERSATION, workspace_id=NEW_WORKSPACE, target_index=0)
    assert await _authority_diverged(env, app_store) is True, (
        "检测器必须能判死「只改存档、app.db 不动」—— 否则第 1/3/4 条用例都是恒真"
    )

    # 走真实入口（确认）后必须回到一致，并且权威归属只由 app.db 说了算
    response = await _drag(env, confirm=True)
    assert response.status_code == 200, response.text
    assert response.json()["steps"], "分叉状态下的「跨区移动」不得被误判成同区排序"
    assert await _authority_diverged(env, app_store) is False
    record = await app_store.get_conversation(CONVERSATION)
    assert record.workspace_id == NEW_WORKSPACE
    assert (await app_store.get_conversation(CONVERSATION)).context_epoch == 2


# =====================================================================
# 7. 结构锁：唯一实现，不得再有第二条旁路
# =====================================================================


def test_both_entries_share_one_state_machine_implementation() -> None:
    drag = inspect.getsource(conversations_routes.move_conversation_in_sidebar)
    workspace = inspect.getsource(conversations_routes.move_conversation_workspace)
    assert "_move_via_state_machine(" in drag, "拖拽入口的跨区分支必须调用状态机"
    assert "_move_via_state_machine(" in workspace, "workspace 入口必须调用状态机"
    helper = inspect.getsource(conversations_routes._move_via_state_machine)
    assert "move_conversation(" in helper
    # 整个路由模块里只有这一处调用状态机服务（否则就是第二条旁路）
    module_source = inspect.getsource(conversations_routes)
    assert module_source.count("await move_conversation(") == 1
    assert module_source.count("store.move_in_sidebar(") == 2, (
        "只允许两处存档排序调用：同区分支 + 跨区落位（均不含归属变更）"
    )


async def test_drag_missing_conversation_is_404(env: MoveEnv) -> None:
    response = await env.client.post(
        "/api/conversations/nope/move", json={"workspace_id": NEW_WORKSPACE}
    )
    assert response.status_code == 404, response.text
    # 没有目标区、也不存在的会话：同样是 404（不得 500）
    no_target = await env.client.post("/api/conversations/nope/move", json={})
    assert no_target.status_code == 404, no_target.text
    unknown_workspace = await _drag(env, target="nope")
    assert unknown_workspace.status_code == 404, unknown_workspace.text
