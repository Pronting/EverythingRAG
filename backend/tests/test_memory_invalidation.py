"""人工失效与即时生效（PRD 11.4 / 11.5；验收 A21、A28、A30、A31）。

PRD 11.4 要求人工纠正/删除/关闭**立即生效**，而"生效"不是改一个状态就完事：
它要同时推进受影响会话的 ``context_epoch``、重建或清空快照、让待批准操作失效、
并且只递增**受影响 Agent 自己**的 ``agent_context_epoch``。这里逐项验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import (
    ApprovalRequest,
    ApprovalStatus,
    MemoryEntryStatus,
    MemoryInvalidationKind,
    MemoryKind,
)
from app.agent.memory import (
    apply_invalidation_to_runtime,
    blocked_source_conversations,
    bump_agent_context_epoch,
    compose_agent_memory_context,
    ensure_conversation_snapshot,
    injectable_entries,
    prune_source_memories,
    register_source_deletion,
)
from app.memory.invalidation import (
    model_history_view,
    revoked_requirement_message_ids,
    user_history_view,
)
from app.memory.workspace_memory import WorkspaceMemory
from tests.fakes_agent import seed_run
from tests.fakes_memory import RecordingChatModel, memory_for


@pytest.fixture
async def store() -> AppStore:
    handle = await AppStore.open_memory()
    try:
        yield handle
    finally:
        await handle.close()


async def _seed(tmp_path: Path, store: AppStore):  # type: ignore[no-untyped-def]
    seeded = await seed_run(store, workspace_id="default", storage_key="default")
    memory = memory_for(tmp_path)
    entry = memory.add_entry(
        "回答一律用中文",
        MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id=seeded.conversation_id,
        source_message_ids=[seeded.user_message_id],
    ).entry
    return seeded, memory, entry


# ---------------------------------------------------------------- 人工纠正


async def test_correction_bumps_invalidation_revision_epoch_and_snapshot(
    tmp_path: Path, store: AppStore
) -> None:
    """纠正：``invalidation_revision`` + 受影响会话 ``context_epoch`` + 快照 hash 都变。"""

    seeded, memory, entry = await _seed(tmp_path, store)
    before = await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )
    frozen = before[1]

    mutation = memory.update_entry(entry.id, text="回答一律用中文，并且先给结论")
    outcome = await apply_invalidation_to_runtime(
        app_store=store, memory=memory, mutation=mutation, reason="用户纠正记忆"
    )

    assert mutation.memory.invalidation_revision == 1
    assert outcome.affected_conversations == (seeded.conversation_id,)
    assert outcome.context_epochs[seeded.conversation_id] == 2
    assert outcome.rebuilt_snapshots == (seeded.conversation_id,)
    assert outcome.cleared_snapshots == ()

    conversation = await store.get_conversation(seeded.conversation_id)
    assert conversation is not None
    assert conversation.context_epoch == 2
    assert conversation.memory_snapshot_hash != frozen.hash
    assert "先给结论" in conversation.memory_snapshot_text

    record = mutation.memory.invalidations[-1]
    assert record.kind is MemoryInvalidationKind.CORRECTED
    assert record.quote == "回答一律用中文"
    assert record.replacement == "回答一律用中文，并且先给结论"
    assert record.memory_id == entry.id
    assert record.source_message_ids == [seeded.user_message_id]


async def test_a21_next_model_request_carries_the_change(
    tmp_path: Path, store: AppStore
) -> None:
    """A21：人工纠正之后，**下一次模型请求**的完整上下文体现变更。"""

    seeded, memory, entry = await _seed(tmp_path, store)
    await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )
    model = RecordingChatModel()

    first = await compose_agent_memory_context(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )
    async for _chunk in model.stream_chat([*first.messages, {"role": "user", "content": "继续"}]):
        pass
    assert "回答一律用中文" in model.last_request_text()
    assert "先给结论" not in model.last_request_text()

    mutation = memory.update_entry(entry.id, text="回答一律用中文，并且先给结论")
    await apply_invalidation_to_runtime(
        app_store=store, memory=memory, mutation=mutation, reason="用户纠正记忆"
    )

    second = await compose_agent_memory_context(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )
    async for _chunk in model.stream_chat([*second.messages, {"role": "user", "content": "继续"}]):
        pass

    assert "先给结论" in model.last_request_text()
    assert second.snapshot_hash != first.snapshot_hash
    assert model.request_count() == 2


async def test_disabling_memory_clears_snapshot_and_bumps_epoch(
    tmp_path: Path, store: AppStore
) -> None:
    """关闭：``enabled=false`` + 失效版本同时提交，快照置空，epoch 推进（PRD 11.4）。"""

    seeded, memory, _entry = await _seed(tmp_path, store)
    await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )

    mutation = memory.set_enabled(False)
    outcome = await apply_invalidation_to_runtime(
        app_store=store, memory=memory, mutation=mutation, reason="用户关闭工作区记忆"
    )

    conversation = await store.get_conversation(seeded.conversation_id)
    assert conversation is not None
    assert conversation.memory_snapshot_text == ""
    assert conversation.memory_snapshot_hash == ""
    assert conversation.context_epoch == 2
    assert outcome.cleared_snapshots == (seeded.conversation_id,)
    assert outcome.rebuilt_snapshots == ()
    assert outcome.disabled is True

    # 关闭后的模型请求不再带任何记忆正文。
    context = await compose_agent_memory_context(
        app_store=store, memory=memory, conversation_id=seeded.conversation_id
    )
    assert context.snapshot_text == ""


async def test_stale_pending_approval_becomes_superseded(
    tmp_path: Path, store: AppStore
) -> None:
    """已经失效的批准转为 ``superseded``（PRD 11.4）。"""

    seeded, memory, entry = await _seed(tmp_path, store)
    await store.request_approval(
        ApprovalRequest(
            approval_id="appr-1",
            run_id=seeded.run_id,
            operation_id="op-1",
            args_hash="hash-1",
        )
    )

    mutation = memory.forget_entry(entry.id)
    outcome = await apply_invalidation_to_runtime(
        app_store=store, memory=memory, mutation=mutation, reason="用户遗忘记忆"
    )

    approval = await store.get_approval("appr-1")
    assert approval is not None
    assert approval.status is ApprovalStatus.SUPERSEDED
    assert "遗忘" in approval.superseded_reason
    assert outcome.superseded_approvals == 1


# ---------------------------------------------------------------- 历史视图


def test_revoked_requirement_leaves_model_view_but_stays_in_user_view(
    tmp_path: Path,
) -> None:
    """撤销的要求从自动携带指令中移除；用户视图保留原话并注明已撤销（PRD 11.4）。"""

    memory = memory_for(tmp_path)
    requirement = memory.add_entry(
        "回答一律用中文",
        MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id="conv-1",
        source_message_ids=["m-req"],
    ).entry
    memory.add_entry(
        "我们讨论了 FTS5 的选型",
        MemoryKind.PROGRESS,
        source_conversation_id="conv-1",
        source_message_ids=["m-topic"],
    )
    memory.forget_entry(requirement.id)

    messages = [
        {"id": "m-req", "conversation_id": "conv-1", "role": "user", "content": "以后都用中文回答我", "sequence": 1, "created_at": "2026-09-01T00:00:00Z"},
        {"id": "m-topic", "conversation_id": "conv-1", "role": "user", "content": "我们讨论一下 FTS5 的选型", "sequence": 2, "created_at": "2026-09-01T00:01:00Z"},
    ]
    state = memory.load()

    assert "m-req" in revoked_requirement_message_ids(state)
    model_views = {view.message_id: view for view in model_history_view(messages, state)}
    assert model_views["m-req"].auto_carried is False
    assert model_views["m-req"].reason
    # 其他消息内容保留。
    assert model_views["m-topic"].auto_carried is True

    user_views = {view.message_id: view for view in user_history_view(messages, state)}
    assert user_views["m-req"].content == "以后都用中文回答我"
    assert user_views["m-req"].note == "（已撤销）"
    assert user_views["m-topic"].note == ""


def test_corrected_conclusion_keeps_discussion_messages_but_notes_correction(
    tmp_path: Path,
) -> None:
    """结论被纠正时消息仍在（它不是"要求"），但用户视图标注了纠正内容。"""

    memory = memory_for(tmp_path)
    entry = memory.add_entry(
        "结论：采用 Chroma",
        MemoryKind.CONFIRMED_CONCLUSION,
        source_conversation_id="conv-1",
        source_message_ids=["m-1"],
    ).entry
    memory.update_entry(entry.id, text="结论：采用 SQLite FTS5")

    messages = [
        {"id": "m-1", "conversation_id": "conv-1", "role": "user", "content": "那就用 Chroma 吧", "sequence": 1},
    ]
    state = memory.load()

    assert revoked_requirement_message_ids(state) == {}
    assert model_history_view(messages, state)[0].auto_carried is True
    note = user_history_view(messages, state)[0].note
    assert "已纠正为" in note
    assert "SQLite FTS5" in note


# ---------------------------------------------------------------- A31


async def test_a31_compaction_bumps_only_its_own_agent_context_epoch(
    store: AppStore,
) -> None:
    """A31：单个 Agent 的上下文压缩只增加**自己**的 ``agent_context_epoch``。"""

    seeded = await seed_run(store, workspace_id="default", storage_key="default")
    await store.create_task(run_id=seeded.run_id, task_id="task-1", agent_kind="rag")
    await store.create_task(run_id=seeded.run_id, task_id="task-2", agent_kind="web")

    epoch = await bump_agent_context_epoch(store, seeded.run_id, "task-1")

    assert epoch == 2
    first = await store.get_task(seeded.run_id, "task-1")
    second = await store.get_task(seeded.run_id, "task-2")
    assert first is not None and second is not None
    assert first.agent_context_epoch == 2
    assert second.agent_context_epoch == 1


async def test_invalidation_only_touches_affected_conversations(
    tmp_path: Path, store: AppStore
) -> None:
    """工作区内全部会话读取更新记忆，其他工作区保持原有版本。"""

    await store.create_workspace("default", "default")
    await store.create_workspace("other", "other")
    await store.create_conversation("conv-1", "default")
    await store.create_conversation("conv-2", "default")
    await store.create_conversation("conv-other", "other")
    message = await store.append_message(
        conversation_id="conv-1", role="user", content="回答一律用中文"
    )
    memory = WorkspaceMemory(tmp_path / "memory.json", storage_key="default")
    entry = memory.add_entry(
        "回答一律用中文",
        MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id="conv-1",
        source_message_ids=[message.id],
    ).entry

    outcome = await apply_invalidation_to_runtime(
        app_store=store,
        memory=memory,
        mutation=memory.forget_entry(entry.id),
        reason="用户遗忘记忆",
    )

    assert outcome.affected_conversations == ("conv-1", "conv-2")
    for conversation_id in outcome.affected_conversations:
        current = await store.get_conversation(conversation_id)
        assert current is not None
        assert current.context_epoch == 2
    untouched = await store.get_conversation("conv-other")
    assert untouched is not None
    assert untouched.context_epoch == 1


# ---------------------------------------------------------------- 来源会话删除


async def test_source_deletion_blocks_injection_until_prune_completes(
    tmp_path: Path, store: AppStore
) -> None:
    """删除任务完成前用来源删除标记阻止注入；完成后条目转 orphaned（PRD 11.5）。"""

    seeded, memory, entry = await _seed(tmp_path, store)
    assert entry.status is MemoryEntryStatus.ACTIVE

    job = await register_source_deletion(
        app_store=store, conversation_id=seeded.conversation_id, workspace_ids=["default"]
    )
    assert job.result == {"workspaces": {"default": "pending"}}

    blocked = await blocked_source_conversations(store, "default")
    assert blocked == {seeded.conversation_id}
    injectable = injectable_entries(memory.load(), blocked)
    assert injectable == []

    statuses = await prune_source_memories(
        app_store=store, job_id=job.id, memories={"default": memory}
    )
    assert statuses == {"default": "updated"}

    state = memory.load()
    assert state.entries[0].status is MemoryEntryStatus.ORPHANED
    assert await blocked_source_conversations(store, "default") == frozenset()
    # 标记解除后，"可注入"为空是因为条目已 orphaned，而不是因为还被挡住。
    assert injectable_entries(state, frozenset()) == []


async def test_prune_records_per_workspace_status(tmp_path: Path, store: AppStore) -> None:
    """各区处理状态分别记录：一个区没有该来源时记 ``noop``，不影响另一个区。"""

    seeded, memory_default, _entry = await _seed(tmp_path, store)
    memory_other = memory_for(tmp_path, "other-key")
    memory_other.add_entry(
        "别的区也引用同一来源",
        MemoryKind.TOPIC,
        source_conversation_id=seeded.conversation_id,
    )

    job = await register_source_deletion(
        app_store=store,
        conversation_id=seeded.conversation_id,
        workspace_ids=["default", "other-key"],
    )
    statuses = await prune_source_memories(
        app_store=store,
        job_id=job.id,
        memories={"default": memory_default, "other-key": memory_other},
    )

    assert statuses == {"default": "updated", "other-key": "updated"}
    refreshed = await store.get_background_job(job.id)
    assert refreshed is not None
    assert refreshed.result == {"workspaces": {"default": "updated", "other-key": "updated"}}


def test_forgotten_entries_are_never_injectable(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    entry = memory.add_entry("要被遗忘的要求", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="c1").entry
    memory.forget_entry(entry.id)

    assert injectable_entries(memory.load(), frozenset()) == []
