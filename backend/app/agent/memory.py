"""Memory Extractor 节点、固定快照与人工失效的运行时编排（PRD 11.1/11.3/11.4/11.5，Owner: T08）。

本模块是**记忆文件与 ``app.db`` 之间的唯一接缝**。``memory.json`` 自己不知道会话、
run 与批准的存在，``app.db`` 也不知道记忆条目；PRD 要求的两件事必须由有人同时看到两边
的代码来完成：

1. **固定会话快照**（11.3）：新会话的第一条消息把当时的工作区记忆定格存进
   ``conversations.memory_*``；此后**自动更新立即持久化，但当前会话快照保持不变**。
   "恢复会话 / 页面刷新 / 应用重启不得自动替换快照"这条不能靠调用方自觉 —— 所以
   ``ensure_conversation_snapshot`` 只在快照为空时写一次，重建需要显式调用另一个函数。
2. **人工失效即时生效**（11.4）：递增受影响会话的 ``context_epoch``、清空或重建快照、
   把已失效的批准转 ``superseded``、并只递增**受影响 Agent 自己**的
   ``agent_context_epoch``（A31：单个 Agent 的压缩不改变别人的上下文版本）。

出网约束：本模块唯一的出网是提炼时调用用户已配置的对话模型（经 ``ChatModel``，其底层
已由 ``OutboundClient`` 审计），且任何失败都静默降级。人名脱敏不在本模块（T05）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent.app_store import BackgroundJobSpec, ConversationRecord
from app.agent.contracts import (
    BackgroundJobStatus,
    MemoryFile,
    MemoryFileEntry,
)
from app.memory.extractor import (
    CandidateWorkspaceMemory,
    WorkspaceTurn,
    extract_workspace_candidates,
)
from app.memory.invalidation import HistoryMessageView, model_history_view
from app.memory.locking import conversation_async_lock, memory_maintenance_locks
from app.memory.workspace_memory import (
    MemoryMutationResult,
    MemorySnapshot,
    WorkspaceMemory,
    build_snapshot,
    snapshot_hash,
)

__all__ = [
    "MEMORY_EXTRACTION_JOB_KIND",
    "MEMORY_PRUNE_JOB_KIND",
    "AgentMemoryContext",
    "InvalidationRuntimeOutcome",
    "MemoryExtractionOutcome",
    "apply_invalidation_to_runtime",
    "blocked_source_conversations",
    "bump_agent_context_epoch",
    "compose_agent_memory_context",
    "ensure_conversation_snapshot",
    "extract_conversation_memory",
    "injectable_entries",
    "prune_source_memories",
    "rebuild_conversation_snapshot",
    "register_source_deletion",
    "snapshot_from_record",
]

#: 来源会话删除后更新各区记忆的持久后台任务（PRD 11.5）。
MEMORY_PRUNE_JOB_KIND = "memory_prune"
#: 单次记忆提炼任务（PRD 13.1 / 11.5）。
MEMORY_EXTRACTION_JOB_KIND = "memory_extraction"


# ===========================================================================
# 1. 固定会话快照（PRD 11.3）
# ===========================================================================


def snapshot_from_record(conversation: ConversationRecord) -> MemorySnapshot:
    """把会话上**已保存**的快照读回成 ``MemorySnapshot``（不重新计算）。"""

    text = conversation.memory_snapshot_text
    return MemorySnapshot(
        revision=conversation.memory_revision,
        invalidation_revision=conversation.memory_invalidation_revision,
        text=text,
        hash=conversation.memory_snapshot_hash,
        at=conversation.memory_snapshot_at,
    )


def _snapshot_present(conversation: ConversationRecord) -> bool:
    return bool(conversation.memory_snapshot_at or conversation.memory_snapshot_hash)


async def ensure_conversation_snapshot(
    *,
    app_store: Any,
    memory: WorkspaceMemory,
    conversation_id: str,
) -> tuple[ConversationRecord, MemorySnapshot]:
    """新会话首次使用时定格快照；已有快照时**原样返回**（PRD 11.3）。

    读取发生在会话锁 + 工作区写锁内，且 ``memory.json`` 是一次整体读 —— 因此并行创建
    会话与写入记忆时，快照必然对应**某一个完整版本**，不会读到"改了一半"的记忆
    （PRD 11.3 末段）。
    """

    conversation = await app_store.get_conversation(conversation_id)
    if conversation is None:
        raise LookupError(f"会话不存在：{conversation_id}")
    if _snapshot_present(conversation):
        return conversation, snapshot_from_record(conversation)

    # 关键：**临界区里不放 await**。锁是 ``threading.RLock``（同线程可重入，避免
    # 嵌套调用自锁），若在持锁期间 await，同一事件循环上的另一个任务会在"同线程"
    # 重新进入临界区，互斥就失效了。这里只做一次同步的整份文件读取。
    with memory_maintenance_locks(conversation_id, storage_keys=[memory.storage_key]):
        snapshot = memory.snapshot()

    current = await app_store.get_conversation(conversation_id)
    if current is None:  # pragma: no cover - 并发删除
        raise LookupError(f"会话不存在：{conversation_id}")
    if _snapshot_present(current):
        # 另一个执行体已经写过：固定语义优先，后来的不覆盖。
        return current, snapshot_from_record(current)
    updated = await app_store.save_memory_snapshot(
        conversation_id,
        memory_revision=snapshot.revision,
        memory_invalidation_revision=snapshot.invalidation_revision,
        memory_snapshot_text=snapshot.text,
        memory_snapshot_hash=snapshot.hash,
        memory_snapshot_at=snapshot.at,
    )
    return updated, snapshot


async def rebuild_conversation_snapshot(
    *,
    app_store: Any,
    memory: WorkspaceMemory,
    conversation_id: str,
) -> tuple[ConversationRecord, MemorySnapshot]:
    """**仅在人工失效之后**重建快照（PRD 11.4「重建上下文后…同步更新快照 hash」）。

    与 ``ensure_conversation_snapshot`` 分开是刻意的：把"重建"和"首次定格"混成一个
    函数，就等于允许任意代码路径在恢复会话时替换快照，直接违反 11.3。
    """

    current = await app_store.get_conversation(conversation_id)
    if current is None:
        raise LookupError(f"会话不存在：{conversation_id}")

    # 同样：临界区内只有同步的文件读取/构造（见 ensure_conversation_snapshot 的说明）。
    with memory_maintenance_locks(conversation_id, storage_keys=[memory.storage_key]):
        state = memory.load()
        snapshot = build_snapshot(state) if state.enabled else None

    if snapshot is None:
        updated = await app_store.clear_memory_snapshot(
            conversation_id, invalidation_revision=state.invalidation_revision
        )
        return updated, MemorySnapshot(
            revision=state.revision,
            invalidation_revision=state.invalidation_revision,
            text="",
            hash=snapshot_hash(""),
            at=updated.memory_snapshot_at,
        )
    updated = await app_store.save_memory_snapshot(
        conversation_id,
        memory_revision=snapshot.revision,
        memory_invalidation_revision=snapshot.invalidation_revision,
        memory_snapshot_text=snapshot.text,
        memory_snapshot_hash=snapshot.hash,
        memory_snapshot_at=snapshot.at,
    )
    return updated, snapshot


# ===========================================================================
# 2. 模型请求里的记忆段（PRD 9.3 / 11.3 / 11.4）
# ===========================================================================


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    """进入模型请求的记忆上下文（快照固定 + 历史按失效规则过滤）。

    ``messages`` 是可直接拼进 Chat Completions 的片段：第一段是固定快照，
    之后是**模型读取面**的历史（撤销的要求已移除，其他消息保留，PRD 11.4）。
    """

    revision: int
    invalidation_revision: int
    snapshot_text: str
    snapshot_hash: str
    messages: tuple[dict[str, str], ...] = ()

    @property
    def context_epoch_relevant(self) -> tuple[int, int]:
        return (self.revision, self.invalidation_revision)


async def compose_agent_memory_context(
    *,
    app_store: Any,
    memory: WorkspaceMemory,
    conversation_id: str,
    history: Sequence[Any] = (),
    ensure_snapshot: bool = True,
) -> AgentMemoryContext:
    """组装一次模型请求用的记忆上下文。

    快照取**会话上已保存的正文**（不是当下的 memory.json）：这正是"自动更新立即持久化，
    当前会话快照不变，新会话读取新版本"（PRD 11.3）的实现点。人工失效之后调用方应先
    ``rebuild_conversation_snapshot``，下一次请求就会带上新正文（A21）。
    """

    if ensure_snapshot:
        conversation, snapshot = await ensure_conversation_snapshot(
            app_store=app_store, memory=memory, conversation_id=conversation_id
        )
    else:
        conversation = await app_store.get_conversation(conversation_id)
        if conversation is None:
            raise LookupError(f"会话不存在：{conversation_id}")
        snapshot = snapshot_from_record(conversation)

    state = memory.load()
    views: list[HistoryMessageView] = model_history_view(history, state)

    messages: list[dict[str, str]] = []
    if snapshot.text:
        messages.append(
            {
                "role": "system",
                "content": "本会话固定的工作区记忆快照（内容自会话开始未变）：\n"
                + snapshot.text,
            }
        )
    for view in views:
        if not view.auto_carried:
            continue
        if view.role not in {"user", "assistant", "system"}:
            continue
        messages.append({"role": view.role, "content": view.content})

    return AgentMemoryContext(
        revision=snapshot.revision,
        invalidation_revision=snapshot.invalidation_revision,
        snapshot_text=snapshot.text,
        snapshot_hash=snapshot.hash,
        messages=tuple(messages),
    )


# ===========================================================================
# 3. 提炼（PRD 11.1 / 11.5）
# ===========================================================================


@dataclass(frozen=True, slots=True)
class MemoryExtractionOutcome:
    """一次记忆提炼的结果。"""

    applied: bool
    reason: str = ""
    operation_id: str = ""
    added_entry_ids: tuple[str, ...] = ()
    cursor: str = ""
    candidates: tuple[CandidateWorkspaceMemory, ...] = ()
    revision: int = 0
    invalidation_revision: int = 0

    @property
    def skipped(self) -> bool:
        return not self.applied


def build_turns(records: Sequence[Any]) -> list[WorkspaceTurn]:
    """把 ``MessageRecord`` 列表转成提炼用的轮次（带消息 ID 与序号）。"""

    return [
        WorkspaceTurn(
            role=str(getattr(record, "role", "") or ""),
            content=str(getattr(record, "content", "") or ""),
            message_id=str(getattr(record, "id", "") or ""),
            sequence=int(getattr(record, "sequence", 0) or 0),
        )
        for record in records
    ]


async def extract_conversation_memory(
    *,
    memory: WorkspaceMemory,
    chat_model: Any,
    conversation_id: str,
    turns: Sequence[WorkspaceTurn],
    operation_id: str,
    cursor_message_id: str = "",
    expected_revision: int | None = None,
    app_store: Any = None,
    expected_context_epoch: int | None = None,
) -> MemoryExtractionOutcome:
    """提炼一段对话并**原子提交**到 ``memory.json``（PRD 11.5）。

    - 开关关闭：直接跳过（"排队任务提交前检查开关"，PRD 11.4），不写文件也不报错。
    - 同一 ``operation_id`` 二次提交：幂等空操作，不产生重复条目、不移动游标之外的东西。
    - 提炼失败：返回空候选，但**游标仍然前进** —— 否则失败的对话会被无限重试。
    - **提交前再核对归属与 ``context_epoch``**（PRD 11.2 / 11.4 / A28，传 ``app_store`` 时
      生效）：本 handle 属于哪个工作区就只能提交到哪个工作区；会话若已移动到其它工作区或
      ``context_epoch`` 已变（移动、人工失效），本次提炼**不提交**并返回明确原因，绝不把
      记忆写进错误工作区。核对与提交在同一把 ``conversation_async_lock`` 内，因此「移动」
      不可能插在「核对通过」与「写文件」之间。
    """

    state = memory.load()
    if not state.enabled:
        return MemoryExtractionOutcome(
            False, reason="工作区记忆已关闭", operation_id=operation_id
        )
    candidates = await extract_workspace_candidates(chat_model, list(turns))

    if app_store is None:
        result: MemoryMutationResult = memory.apply_operations(
            operation_id=operation_id,
            conversation_id=conversation_id,
            cursor_message_id=cursor_message_id,
            candidates=[
                (candidate.text, candidate.kind, candidate.source_message_ids)
                for candidate in candidates
            ],
            expected_revision=expected_revision,
        )
    else:
        async with conversation_async_lock(conversation_id):
            blocked = await _memory_commit_blocker(
                app_store=app_store,
                memory=memory,
                conversation_id=conversation_id,
                expected_context_epoch=expected_context_epoch,
            )
            if blocked:
                return MemoryExtractionOutcome(
                    False, reason=blocked, operation_id=operation_id, candidates=tuple(candidates)
                )
            result = memory.apply_operations(
                operation_id=operation_id,
                conversation_id=conversation_id,
                cursor_message_id=cursor_message_id,
                candidates=[
                    (candidate.text, candidate.kind, candidate.source_message_ids)
                    for candidate in candidates
                ],
                expected_revision=expected_revision,
            )

    return MemoryExtractionOutcome(
        applied=result.applied,
        reason=result.reason,
        operation_id=operation_id,
        added_entry_ids=(() if result.entry is None else (result.entry.id,)),
        cursor=cursor_message_id,
        candidates=tuple(candidates),
        revision=result.memory.revision,
        invalidation_revision=result.memory.invalidation_revision,
    )


async def _memory_commit_blocker(
    *,
    app_store: Any,
    memory: WorkspaceMemory,
    conversation_id: str,
    expected_context_epoch: int | None,
) -> str:
    """提交前的归属/版本核对；返回**非空原因**表示这次提炼不得提交（A28）。"""

    owner = await app_store.get_workspace_by_storage_key(memory.storage_key)
    conversation = await app_store.get_conversation(conversation_id)
    if conversation is None:
        return "会话不存在，本次提炼不提交"
    if owner is None:
        return "记忆所属工作区无法验证，本次提炼不提交"
    if conversation.workspace_id != owner.id:
        return (
            f"会话已移动到工作区 {conversation.workspace_id}，"
            f"本次提炼不提交到 {owner.id}（PRD 5.4 / A28）"
        )
    if expected_context_epoch is not None and int(conversation.context_epoch) != int(
        expected_context_epoch
    ):
        return (
            "会话 context_epoch 已变化（移动或人工失效），"
            f"本次提炼不提交（期望 {expected_context_epoch}，实际 {conversation.context_epoch}）"
        )
    return ""


# ===========================================================================
# 4. 人工失效的运行时生效（PRD 11.4 / A21 / A31）
# ===========================================================================


@dataclass(frozen=True, slots=True)
class InvalidationRuntimeOutcome:
    """人工失效在 ``app.db`` 侧产生的全部后果（便于测试逐项断言）。"""

    affected_conversations: tuple[str, ...] = ()
    context_epochs: dict[str, int] = field(default_factory=dict)
    cleared_snapshots: tuple[str, ...] = ()
    rebuilt_snapshots: tuple[str, ...] = ()
    superseded_approvals: int = 0
    bumped_task_ids: tuple[str, ...] = ()
    disabled: bool = False


async def apply_invalidation_to_runtime(
    *,
    app_store: Any,
    memory: WorkspaceMemory,
    mutation: MemoryMutationResult,
    reason: str,
) -> InvalidationRuntimeOutcome:
    """人工纠正/删除/关闭之后，让变更在**下一次模型请求**里可见（PRD 11.4）。

    逐项对应 PRD 的句子：

    - "增加受影响会话的 ``context_epoch``" → ``bump_context_epoch``
    - "已经发送到服务商的请求无法撤回；它返回的受影响候选答案停止发布" → 由 RunManager
      依据 ``context_epoch`` 校验（T02/T09 侧），本函数只负责把版本推上去
    - "尚未发布的候选答案/批准按来源关系失效" → 待批准转 ``superseded``
    - "重建上下文后更新对应 ``agent_context_epoch``，仅重新执行受影响步骤" → 只递增
      受影响 run 下每个子任务**自己**的 ``agent_context_epoch``（A31）
    - "关闭记忆…受影响会话的记忆快照置空" → ``clear_memory_snapshot``
    - "纠正/删除…同步更新快照 hash" → ``rebuild_conversation_snapshot``
    """

    state = mutation.memory
    disabled = not state.enabled
    affected = tuple(dict.fromkeys(mutation.affected_conversation_ids))

    context_epochs: dict[str, int] = {}
    cleared: list[str] = []
    rebuilt: list[str] = []
    bumped_tasks: list[str] = []
    superseded = 0

    for conversation_id in affected:
        conversation = await app_store.get_conversation(conversation_id)
        if conversation is None:
            # 来源会话已删除（或从未存在）：条目已被标记 orphaned，不再需要版本推进。
            continue
        context_epochs[conversation_id] = await app_store.bump_context_epoch(conversation_id)

        if disabled:
            updated = await app_store.clear_memory_snapshot(
                conversation_id, invalidation_revision=state.invalidation_revision
            )
            if updated is not None:
                cleared.append(conversation_id)
        else:
            await rebuild_conversation_snapshot(
                app_store=app_store, memory=memory, conversation_id=conversation_id
            )
            rebuilt.append(conversation_id)

        for run in await app_store.list_active_runs(conversation_id):
            superseded += await app_store.supersede_pending_approvals(run.id, reason=reason)
            for task in await app_store.list_tasks(run.id):
                await bump_agent_context_epoch(app_store, run.id, task.id)
                bumped_tasks.append(task.id)

    return InvalidationRuntimeOutcome(
        affected_conversations=affected,
        context_epochs=context_epochs,
        cleared_snapshots=tuple(cleared),
        rebuilt_snapshots=tuple(rebuilt),
        superseded_approvals=superseded,
        bumped_task_ids=tuple(bumped_tasks),
        disabled=disabled,
    )


async def bump_agent_context_epoch(
    app_store: Any, run_id: str, task_id: str, *, delta: int = 1
) -> int:
    """只递增**这一个** Agent 的 ``agent_context_epoch``（PRD 9.3 / A31）。

    刻意不提供"按 run 批量递增"的接口：单个 Agent 压缩自己的上下文，其他 Agent
    必须继续使用各自有效的上下文版本。
    ``run_id`` 必填：任务身份是 run 作用域的（v3），只按 ``task_id`` 无法定位 —— 同一会话
    的不同 run 里 ``t1`` 是不同任务，全局解析会拿错对象。
    """

    task = await app_store.get_task(run_id, task_id)
    if task is None:
        raise LookupError(f"子任务不存在：run={run_id} task={task_id}")
    updated = await app_store.update_task_status(
        run_id,
        task_id,
        task.status,
        agent_context_epoch=task.agent_context_epoch + max(1, int(delta)),
    )
    return int(updated.agent_context_epoch)


# ===========================================================================
# 5. 来源会话删除（PRD 11.5）
# ===========================================================================


async def register_source_deletion(
    *,
    app_store: Any,
    conversation_id: str,
    workspace_ids: Sequence[str],
    job_id: str | None = None,
) -> Any:
    """删除事务里登记持久后台任务，并记录**各区处理状态**（PRD 11.5）。

    任务未完成期间，本区相关记忆通过 :func:`blocked_source_conversations` 被挡住，
    不会注入模型上下文 —— 否则"刚被删除的会话"还会以记忆的形式影响回答。
    """

    job = await app_store.create_background_job(
        BackgroundJobSpec(
            kind=MEMORY_PRUNE_JOB_KIND,
            conversation_id=conversation_id,
            workspace_id=workspace_ids[0] if workspace_ids else None,
            job_id=job_id,
        )
    )
    return await app_store.update_background_job(
        job.id,
        BackgroundJobStatus.PENDING,
        result={"workspaces": {workspace_id: "pending" for workspace_id in workspace_ids}},
    )


async def prune_source_memories(
    *,
    app_store: Any,
    job_id: str,
    memories: Mapping[str, WorkspaceMemory],
) -> dict[str, str]:
    """后台任务主体：按来源会话 ID 更新**各工作区**的记忆（PRD 11.5）。

    ``memories`` 是 ``storage_key -> WorkspaceMemory``。每个区的处理结果单独记录，
    某个区失败不影响其它区（局部失败要在报告里看得见，而不是整任务重跑）。
    """

    job = await app_store.get_background_job(job_id)
    if job is None:
        raise LookupError(f"后台任务不存在：{job_id}")
    conversation_id = job.conversation_id or ""
    if not conversation_id:
        raise ValueError(f"记忆清理任务缺少来源会话：{job_id}")

    await app_store.update_background_job(job_id, BackgroundJobStatus.RUNNING)
    statuses: dict[str, str] = {}
    for storage_key, memory in memories.items():
        try:
            result = memory.apply_source_deletion(
                conversation_id, operation_id=f"{job_id}:{storage_key}"
            )
            statuses[storage_key] = "updated" if result.applied else "noop"
        except Exception as exc:  # noqa: BLE001 -- 单区失败不拖垮整任务，但要留痕
            statuses[storage_key] = f"failed:{type(exc).__name__}"
    await app_store.update_background_job(
        job_id, BackgroundJobStatus.COMPLETED, result={"workspaces": statuses}
    )
    return statuses


async def blocked_source_conversations(
    app_store: Any, workspace_id: str
) -> frozenset[str]:
    """尚未完成清理的来源会话（这些来源的记忆**暂不允许注入**，PRD 11.5）。"""

    jobs = await app_store.list_background_jobs(kind=MEMORY_PRUNE_JOB_KIND)
    blocked: set[str] = set()
    for job in jobs:
        if job.status not in {BackgroundJobStatus.PENDING, BackgroundJobStatus.RUNNING}:
            continue
        if not job.conversation_id:
            continue
        workspaces = {}
        if isinstance(job.result, Mapping):
            raw = job.result.get("workspaces")
            if isinstance(raw, Mapping):
                workspaces = dict(raw)
        if job.workspace_id == workspace_id or workspace_id in workspaces:
            blocked.add(job.conversation_id)
    return frozenset(blocked)


def injectable_entries(
    memory: MemoryFile, blocked_sources: frozenset[str] | set[str]
) -> list[MemoryFileEntry]:
    """可注入模型上下文的记忆条目：``active`` 且来源未被删除标记挡住。"""

    from app.agent.contracts import MemoryEntryStatus

    return [
        entry
        for entry in memory.entries
        if entry.status is MemoryEntryStatus.ACTIVE
        and entry.source_conversation_id not in blocked_sources
    ]
