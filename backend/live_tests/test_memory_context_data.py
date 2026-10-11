import asyncio
from uuid import uuid4

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import ApprovalRequest, ApprovalStatus, ConversationStatus, MemoryKind
from app.agent.memory import (
    apply_invalidation_to_runtime,
    compose_agent_memory_context,
    ensure_conversation_snapshot,
)
from app.memory.locking import workspace_async_lock
from app.memory.workspace_memory import WorkspaceMemory


@pytest.fixture
async def memory_store(tmp_path):
    store = await AppStore.open(tmp_path / "app.db")
    storage_key = f"memory-{uuid4().hex}"
    await store.create_workspace("memory-workspace", storage_key)
    await store.create_workspace("other-workspace", f"other-{uuid4().hex}")
    for conversation_id in ("source", "reader", "archived"):
        await store.create_conversation(conversation_id, "memory-workspace")
    await store.update_conversation("archived", status=ConversationStatus.ARCHIVED)
    await store.create_conversation("other", "other-workspace")
    memory = WorkspaceMemory(tmp_path / "memory.json", storage_key=storage_key)
    try:
        yield store, memory
    finally:
        await store.close()


async def snapshot_all(store, memory):
    for conversation_id in ("source", "reader", "archived"):
        await ensure_conversation_snapshot(
            app_store=store, memory=memory, conversation_id=conversation_id
        )


async def test_manual_correction_updates_every_workspace_reader(memory_store):
    store, memory = memory_store
    entry = memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE).entry
    await snapshot_all(store, memory)
    async with workspace_async_lock(memory.storage_key):
        mutation = memory.update_entry(entry.id, text="回答使用英文")
        outcome = await apply_invalidation_to_runtime(
            app_store=store, memory=memory, mutation=mutation, reason="用户修改回答要求"
        )
    assert set(outcome.affected_conversations) == {"source", "reader", "archived"}
    assert set(outcome.rebuilt_snapshots) == {"source", "reader", "archived"}
    for conversation_id in outcome.affected_conversations:
        current = await store.get_conversation(conversation_id)
        assert current.context_epoch == 2
        assert current.memory_invalidation_revision == mutation.memory.invalidation_revision
        assert "回答使用英文" in current.memory_snapshot_text
        assert "回答使用中文" not in current.memory_snapshot_text
    assert (await store.get_conversation("other")).context_epoch == 1


async def test_forget_is_propagated_once_and_filters_only_revoked_history(memory_store):
    store, memory = memory_store
    requirement = await store.append_message(
        conversation_id="source", role="user", content="以后回答使用英文"
    )
    discussion = await store.append_message(
        conversation_id="source", role="user", content="我们讨论了数据库索引"
    )
    entry = memory.add_entry(
        "以后回答使用英文", MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id="source", source_message_ids=[requirement.id],
    ).entry
    await snapshot_all(store, memory)
    async with workspace_async_lock(memory.storage_key):
        mutation = memory.forget_entry(entry.id)
        await apply_invalidation_to_runtime(
            app_store=store, memory=memory, mutation=mutation, reason="用户撤销回答要求"
        )
        repeated = await apply_invalidation_to_runtime(
            app_store=store, memory=memory, mutation=mutation, reason="重复同步"
        )
    assert repeated.context_epochs == {}
    context = await compose_agent_memory_context(
        app_store=store, memory=memory, conversation_id="source",
        history=await store.list_messages("source"),
    )
    assert all(
        requirement.content not in message["content"]
        for message in context.messages if message["role"] == "user"
    )
    assert context.messages[-1]["role"] == "system"
    assert "不得继续执行" in context.messages[-1]["content"]
    assert requirement.content in context.messages[-1]["content"]
    assert any(discussion.content == message["content"] for message in context.messages)
    assert (await store.get_message(requirement.id)).content == requirement.content
    assert (await store.get_conversation("reader")).context_epoch == 2


async def test_memory_switch_updates_snapshots_and_epochs(memory_store):
    store, memory = memory_store
    memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE)
    await snapshot_all(store, memory)
    for enabled, expected_epoch in ((False, 2), (True, 3)):
        async with workspace_async_lock(memory.storage_key):
            mutation = memory.set_enabled(enabled)
            await apply_invalidation_to_runtime(
                app_store=store, memory=memory, mutation=mutation, reason="用户修改开关"
            )
        conversation, snapshot = await ensure_conversation_snapshot(
            app_store=store, memory=memory, conversation_id="reader"
        )
        assert conversation.context_epoch == expected_epoch
        assert ("回答使用中文" in snapshot.text) == enabled
        assert snapshot.invalidation_revision == mutation.memory.invalidation_revision


async def test_snapshot_repairs_file_invalidation_before_runtime_sync(memory_store):
    store, memory = memory_store
    entry = memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE).entry
    await snapshot_all(store, memory)
    mutation = memory.update_entry(entry.id, text="回答使用英文")
    conversation, snapshot = await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id="reader"
    )
    assert conversation.context_epoch == 2
    assert snapshot.invalidation_revision == mutation.memory.invalidation_revision
    assert "回答使用英文" in snapshot.text
    async with workspace_async_lock(memory.storage_key):
        await apply_invalidation_to_runtime(
            app_store=store, memory=memory, mutation=mutation, reason="恢复同步"
        )
    assert (await store.get_conversation("reader")).context_epoch == 2
    assert (await store.get_conversation("source")).context_epoch == 2


async def test_automatic_addition_preserves_existing_snapshot(memory_store):
    store, memory = memory_store
    memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE)
    await snapshot_all(store, memory)
    before = await store.get_conversation("reader")
    memory.add_entry("用户正在学习数据库", MemoryKind.TOPIC)
    after, snapshot = await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id="reader"
    )
    assert after.memory_snapshot_hash == before.memory_snapshot_hash
    assert "用户正在学习数据库" not in snapshot.text
    assert after.context_epoch == before.context_epoch


async def test_new_conversation_after_manual_change_keeps_initial_epoch(memory_store):
    store, memory = memory_store
    entry = memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE).entry
    memory.update_entry(entry.id, text="回答使用英文")
    created = await store.create_conversation("new-reader", "memory-workspace")
    conversation, snapshot = await ensure_conversation_snapshot(
        app_store=store, memory=memory, conversation_id=created.id
    )
    assert conversation.context_epoch == created.context_epoch
    assert snapshot.invalidation_revision == memory.load().invalidation_revision
    assert "回答使用英文" in snapshot.text


async def test_snapshot_read_waits_for_memory_invalidation_commit(memory_store):
    store, memory = memory_store
    entry = memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE).entry
    await snapshot_all(store, memory)
    async with workspace_async_lock(memory.storage_key):
        mutation = memory.update_entry(entry.id, text="回答使用英文")
        reader = asyncio.create_task(ensure_conversation_snapshot(
            app_store=store, memory=memory, conversation_id="reader"
        ))
        await asyncio.sleep(0)
        assert not reader.done()
        await apply_invalidation_to_runtime(
            app_store=store, memory=memory, mutation=mutation, reason="用户修改回答要求"
        )
    conversation, snapshot = await reader
    assert conversation.context_epoch == 2
    assert "回答使用英文" in snapshot.text


@pytest.mark.parametrize("recover_from_file", [False, True])
async def test_manual_change_invalidates_active_tasks_and_approvals(
    memory_store, recover_from_file
):
    store, memory = memory_store
    entry = memory.add_entry("回答使用中文", MemoryKind.RESPONSE_PREFERENCE).entry
    await snapshot_all(store, memory)
    run = await store.create_run(
        conversation_id="reader", workspace_id="memory-workspace"
    )
    await store.create_task(run_id=run.id, task_id="general", agent_kind="general")
    approval = await store.request_approval(ApprovalRequest(
        approval_id="approval-memory", run_id=run.id, task_id="general",
        operation_id="memory-research", args_hash="memory-research-args",
    ))
    async with workspace_async_lock(memory.storage_key):
        mutation = memory.update_entry(entry.id, text="回答使用英文")
        if not recover_from_file:
            await apply_invalidation_to_runtime(
                app_store=store, memory=memory, mutation=mutation, reason="用户修改回答要求"
            )
    if recover_from_file:
        await ensure_conversation_snapshot(
            app_store=store, memory=memory, conversation_id="reader"
        )
    assert (await store.get_approval(approval.approval_id)).status == ApprovalStatus.SUPERSEDED
    assert (await store.get_task(run.id, "general")).agent_context_epoch == 2
