"""会话移动编排（PRD 5.4 / 13.2；验收 A27 / A28）。Owner: T_a27。

本模块是「移动」这件事的**唯一实现**，路由只负责 HTTP 形状（``409`` / ``404`` 映射）。

PRD 5.4 的顺序契约（每条都在 :class:`MoveOutcome.steps` 里留下证据）::

    confirm → lock → stop_requested → runs_stopped → activity_exited
            → app_db_moved → permission_rebound → sidebar_moved → snapshot_rebound

要点与理由：

1. **先停止、再等待活动请求退出、最后才改归属**。``RunManager.stop_conversation_runs``
   会 ``await`` 掉驱动任务（``cancel`` 的既有语义），并用 ``ensure_conversation_idle``
   复核；本模块在改归属之前再显式调用一次 —— 顺序不是"约定"，是被检查过的。
2. **原 run 的工作区归属不变**：移动只改 ``conversations`` 行与侧边栏，``runs`` 行
   一行不动，因此历史记录仍按**创建时**的工作区可读（PRD 13.2 只读边界）。
3. **原 run 不可恢复、不可再提交批准/澄清**：``cancel(resumable=False)`` +
   ``stop_reason=conversation_moved``，由 ``contracts.can_transition_run`` 与
   ``RunManager`` 的守卫共同保证（PRD 12.3 / 12.4）。
4. **目标区快照**：``AppStore.move_conversation`` 在同一次写入里递增 ``context_epoch``、
   记录 ``moved_from_workspace_id`` 并**清空**会话快照，随后本模块用**目标工作区**的
   ``memory.json`` 重建快照（PRD 5.4「读取目标工作区的记忆快照」）。
5. **权限重新绑定（仅限"继承来的"值）**：会话的权限/强度若仍等于**源区**默认值，说明它
   只是继承来的，移动后改为**目标区**默认值；若用户显式改过（不等于源区默认），保持不动。
   这样既不吞掉用户选择，也不把旧区默认值带到新区。
6. **锁顺序（PRD 11.2）**：跨两区时按 ``storage_key`` 字典序取工作区写锁
   （``ordered_storage_keys``）；同一事件循环内的任务互斥由 ``conversation_async_lock``
   提供 —— ``threading.RLock`` 在 ``await`` 期间对**同线程**不生效，这点必须说清楚，
   否则"看起来加了锁"其实是空的（见 ``memory/locking.py`` 的说明）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.agent.contracts import DEFAULT_PERMISSION_MODE, DEFAULT_REASONING_LEVEL
from app.agent.memory import rebuild_conversation_snapshot
from app.agent.runtime import STOP_REASON_CONVERSATION_MOVED, RunManager
from app.core.conversation_store import Conversation, ConversationStore
from app.core.workspace_store import WorkspaceStore
from app.memory.locking import (
    conversation_async_lock,
    ordered_storage_keys,
    session_lock,
    workspace_write_locks,
)

logger = logging.getLogger(__name__)


class ConversationMoveError(RuntimeError):
    """移动被拒绝（路由映射 409）。"""


class ConversationMoveNotFound(ConversationMoveError):
    """会话或目标工作区不存在（路由映射 404）。"""


class MoveConfirmationRequired(ConversationMoveError):
    """会话带历史或活动任务：移动会携带历史内容，必须先由用户确认（PRD 5.4 首句）。"""


class ConversationMoveConflict(ConversationMoveError):
    """乐观并发校验失败（``expected_context_epoch`` 不匹配或无法核对）。"""


@dataclass(frozen=True, slots=True)
class MoveOutcome:
    """一次移动的结果（顺序证据 + 各项副作用计数，便于逐条断言）。"""

    conversation: Conversation | None
    record: Any | None
    from_workspace_id: str
    to_workspace_id: str
    previous_context_epoch: int | None
    context_epoch: int | None
    stopped_run_ids: tuple[str, ...]
    superseded_approvals: int
    superseded_inputs: int
    permission_rebound: bool
    memory_snapshot_rebound: bool
    locked_storage_keys: tuple[str, ...]
    steps: tuple[str, ...] = field(default=())

    @property
    def moved(self) -> bool:
        return "app_db_moved" in self.steps or "sidebar_moved" in self.steps


async def move_conversation(
    *,
    conversation_id: str,
    target_workspace_id: str,
    store: ConversationStore,
    workspaces: WorkspaceStore,
    manager: RunManager | None = None,
    app_store: Any = None,
    confirm: bool = False,
    expected_context_epoch: int | None = None,
    target_memory: Callable[[str], Any] | None = None,
) -> MoveOutcome:
    """把会话移动到目标工作区（PRD 5.4）。

    ``manager`` / ``app_store`` 允许为 ``None``（未启动 lifespan 的裸测试、未迁移安装）：
    此时「停止运行任务」「app.db 归属与 epoch」这部分如实跳过，只做侧边栏移动 ——
    这正是 D25 的过渡期双源语义，不是静默降级。
    """

    steps: list[str] = []
    sidebar = store.get(conversation_id)
    record = None
    if app_store is not None:
        record = await app_store.get_conversation(conversation_id)
    if sidebar is None and record is None:
        raise ConversationMoveNotFound("会话不存在")

    target_workspace = workspaces.get(target_workspace_id)
    if target_workspace is None:
        raise ConversationMoveNotFound("工作区不存在")

    source_workspace_id = (
        str(record.workspace_id) if record is not None else sidebar.workspace_id  # type: ignore[union-attr]
    )
    active_runs: list[Any] = []
    if manager is not None:
        async with manager.db():
            active_runs = await manager.store.list_active_runs(conversation_id)

    has_history = bool(sidebar is not None and sidebar.messages)
    if record is not None and not has_history:
        has_history = bool(await _message_count(app_store, conversation_id))
    same_workspace = source_workspace_id == target_workspace_id
    # 同区「移动」不带任何历史跨区，因此不需要确认；跨区且携带历史/活动任务时必须确认。
    if not same_workspace and (has_history or active_runs) and not confirm:
        raise MoveConfirmationRequired(
            "移动会话会携带历史内容，请确认后再移动（confirm=true，PRD 5.4）"
        )
    steps.append("confirmed")

    if expected_context_epoch is not None:
        if record is None:
            raise ConversationMoveConflict(
                "该会话尚无 app.db 归属记录，无法核对 expected_context_epoch（PRD 13.2）"
            )
        if int(record.context_epoch) != int(expected_context_epoch):
            raise ConversationMoveConflict(
                f"会话上下文版本已变化：期望 {expected_context_epoch}，"
                f"实际 {record.context_epoch}（PRD 13.2）"
            )

    if same_workspace:
        # 同区且 epoch 核对通过 → 空操作：不递增 epoch、不清快照、不停止任何 run。
        return MoveOutcome(
            conversation=sidebar,
            record=record,
            from_workspace_id=source_workspace_id,
            to_workspace_id=target_workspace_id,
            previous_context_epoch=None if record is None else int(record.context_epoch),
            context_epoch=None if record is None else int(record.context_epoch),
            stopped_run_ids=(),
            superseded_approvals=0,
            superseded_inputs=0,
            permission_rebound=False,
            memory_snapshot_rebound=False,
            locked_storage_keys=(),
            steps=(*steps, "noop_same_workspace"),
        )

    source_key = str(
        (workspaces.get(source_workspace_id).storage_key if workspaces.get(source_workspace_id) else "")
        or ""
    )
    target_key = str(target_workspace.storage_key or "")
    locked = ordered_storage_keys([source_key, target_key])

    stopped_ids: list[str] = []
    superseded_approvals = 0
    superseded_inputs = 0
    permission_rebound = False
    snapshot_rebound = False
    previous_epoch: int | None = int(record.context_epoch) if record is not None else None
    epoch: int | None = previous_epoch
    moved_record: Any | None = record
    moved_sidebar: Conversation | None = sidebar

    async with conversation_async_lock(conversation_id):
        steps.append("lock_acquired")
        with session_lock(conversation_id), workspace_write_locks(*locked):
            # ---- 1. 停止主任务及子任务 ----
            if manager is not None:
                for run in active_runs:
                    superseded_approvals += len(await manager.list_pending_approvals(run.id))
                    superseded_inputs += len(await manager.list_pending_inputs(run.id))
                steps.append("stop_requested")
                stopped = await manager.stop_conversation_runs(
                    conversation_id, reason=STOP_REASON_CONVERSATION_MOVED
                )
                stopped_ids = [run.id for run in stopped]
                steps.append("runs_stopped")
                # ---- 2. 等待活动请求退出（改归属前的最后一道闸） ----
                await manager.ensure_conversation_idle(conversation_id)
                steps.append("activity_exited")

            # ---- 3. 改归属：app.db 权威（含 context_epoch +1 与快照清空） ----
            if app_store is not None and record is not None:
                moved_record = await app_store.move_conversation(
                    conversation_id,
                    target_workspace_id=target_workspace_id,
                    expected_context_epoch=expected_context_epoch,
                )
                epoch = int(moved_record.context_epoch)
                steps.append("app_db_moved")

                # ---- 4. 权限/强度重新绑定（仅当当前值是"继承来的"） ----
                rebound = _rebind_defaults(
                    conversation_permission=record.permission_mode,
                    conversation_reasoning=record.reasoning_level,
                    source_workspace=workspaces.get(source_workspace_id),
                    target_workspace=target_workspace,
                )
                if rebound is not None:
                    permission, reasoning = rebound
                    moved_record = await app_store.update_conversation(
                        conversation_id, permission_mode=permission, reasoning_level=reasoning
                    )
                    permission_rebound = True
                    steps.append("permission_rebound")

            # ---- 5. 侧边栏（JSON 存档） ----
            if sidebar is not None:
                moved_sidebar = store.set_workspace(conversation_id, target_workspace_id)
                steps.append("sidebar_moved")

            # ---- 6. 目标区记忆快照 ----
            if app_store is not None and record is not None and target_memory is not None:
                memory = target_memory(target_workspace_id)
                if memory is not None:
                    moved_record, _snapshot = await rebuild_conversation_snapshot(
                        app_store=app_store,
                        memory=memory,
                        conversation_id=conversation_id,
                    )
                    snapshot_rebound = True
                    steps.append("snapshot_rebound")

    steps.append("done")
    return MoveOutcome(
        conversation=moved_sidebar,
        record=moved_record,
        from_workspace_id=source_workspace_id,
        to_workspace_id=target_workspace_id,
        previous_context_epoch=previous_epoch,
        context_epoch=epoch,
        stopped_run_ids=tuple(stopped_ids),
        superseded_approvals=superseded_approvals,
        superseded_inputs=superseded_inputs,
        permission_rebound=permission_rebound,
        memory_snapshot_rebound=snapshot_rebound,
        locked_storage_keys=locked,
        steps=tuple(steps),
    )


def _rebind_defaults(
    *,
    conversation_permission: Any,
    conversation_reasoning: Any,
    source_workspace: Any,
    target_workspace: Any,
) -> tuple[Any, Any] | None:
    """仅当会话权限/强度等于**源区默认值**时改绑**目标区默认值**（PRD 5.4）。

    返回 ``None`` 表示不需要改（用户显式设过，或目标区默认值与当前值相同）。
    """

    permission = conversation_permission
    reasoning = conversation_reasoning
    changed = False
    if source_workspace is not None and str(permission) == str(
        source_workspace.default_permission_mode or DEFAULT_PERMISSION_MODE
    ):
        target_permission = target_workspace.default_permission_mode or DEFAULT_PERMISSION_MODE
        if str(target_permission) != str(permission):
            permission = target_permission
            changed = True
    if source_workspace is not None and str(reasoning) == str(
        source_workspace.default_reasoning_level or DEFAULT_REASONING_LEVEL
    ):
        target_reasoning = target_workspace.default_reasoning_level or DEFAULT_REASONING_LEVEL
        if str(target_reasoning) != str(reasoning):
            reasoning = target_reasoning
            changed = True
    return (permission, reasoning) if changed else None


async def _message_count(app_store: Any, conversation_id: str) -> int:
    try:
        messages = await app_store.list_messages(conversation_id)
    except Exception:  # noqa: BLE001 - 计数只为"是否携带历史"，失败不阻断判定
        return 0
    return len(messages)


__all__ = [
    "STOP_REASON_CONVERSATION_MOVED",
    "ConversationMoveConflict",
    "ConversationMoveError",
    "ConversationMoveNotFound",
    "MoveConfirmationRequired",
    "MoveOutcome",
    "move_conversation",
]
