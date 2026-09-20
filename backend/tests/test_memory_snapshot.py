"""固定会话快照（PRD 11.3 / 9.3；验收 A20）。

A20 的原文要求是两句话：**会话 A 快照 hash 不变**，**新会话 B 读取更新版本**。
这里把两句话拆成互不重叠的断言，并额外锁死"恢复会话 / 刷新 / 重启不得自动替换快照"
这条 11.3 的硬约束 —— 它最容易被后续改动无意打破。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS, MemoryKind
from app.agent.memory import (
    compose_agent_memory_context,
    ensure_conversation_snapshot,
    rebuild_conversation_snapshot,
)
from app.memory.workspace_memory import build_snapshot
from tests.fakes_memory import RecordingChatModel, memory_for, seed_conversation


@pytest.fixture
async def app_store() -> AppStore:
    store = await AppStore.open_memory()
    await store.create_workspace("default", "default", name="默认")
    try:
        yield store
    finally:
        await store.close()


# ---------------------------------------------------------------- A20


async def test_conversation_a_snapshot_is_frozen_while_conversation_b_sees_new_version(
    tmp_path: Path, app_store: AppStore
) -> None:
    """A20：自动更新立即持久化；A 的快照 hash 不变，B 读到更新版本。"""

    memory = memory_for(tmp_path)
    memory.add_entry(
        "回答一律用中文",
        MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id="conv-A",
        source_message_ids=["m1"],
    )
    await seed_conversation(app_store, conversation_id="conv-A")
    await seed_conversation(app_store, conversation_id="conv-B")

    _record_a, snapshot_a = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-A"
    )
    assert "回答一律用中文" in snapshot_a.text
    assert snapshot_a.revision == 1

    # 自动记忆更新：立即持久化到 memory.json。
    memory.add_entry("新主题：FTS5 中文历史检索", MemoryKind.TOPIC, source_conversation_id="conv-B")
    assert memory.load().revision == 2

    # 恢复会话 A：快照必须还是那一份。
    record_a, snapshot_a_again = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-A"
    )
    assert snapshot_a_again.hash == snapshot_a.hash
    assert snapshot_a_again.text == snapshot_a.text
    assert "FTS5" not in snapshot_a_again.text
    assert record_a.memory_snapshot_hash == snapshot_a.hash
    assert record_a.memory_revision == snapshot_a.revision

    # 新会话 B：读更新版本。
    _record_b, snapshot_b = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-B"
    )
    assert snapshot_b.revision > snapshot_a.revision
    assert "FTS5" in snapshot_b.text
    assert snapshot_b.hash != snapshot_a.hash


async def test_restore_and_restart_never_replace_snapshot(
    tmp_path: Path, app_store: AppStore
) -> None:
    """重复调用（模拟页面刷新 / 应用重启）不得替换已保存的快照。"""

    memory = memory_for(tmp_path)
    memory.add_entry("第一条", MemoryKind.TOPIC, source_conversation_id="conv-1")
    await seed_conversation(app_store, conversation_id="conv-1")

    _record, first = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-1"
    )
    memory.add_entry("第二条", MemoryKind.TOPIC, source_conversation_id="conv-1")

    for _ in range(3):
        _record, again = await ensure_conversation_snapshot(
            app_store=app_store, memory=memory, conversation_id="conv-1"
        )
        assert again.hash == first.hash
        assert "第二条" not in again.text
    persisted = await app_store.get_conversation("conv-1")
    assert persisted is not None
    assert persisted.memory_snapshot_text == first.text
    assert persisted.memory_snapshot_hash == first.hash


async def test_only_the_explicit_rebuild_replaces_snapshot(
    tmp_path: Path, app_store: AppStore
) -> None:
    """人工失效路径之外没有任何"自动替换"入口：只有 rebuild 会换 hash。"""

    memory = memory_for(tmp_path)
    memory.add_entry("旧要求", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="conv-1")
    await seed_conversation(app_store, conversation_id="conv-1")
    _record, frozen = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-1"
    )

    memory.add_entry("新主题", MemoryKind.TOPIC, source_conversation_id="conv-1")
    _record, rebuilt = await rebuild_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-1"
    )

    assert rebuilt.hash != frozen.hash
    assert "新主题" in rebuilt.text
    assert rebuilt.revision == memory.load().revision


async def test_request_context_uses_the_saved_snapshot_not_live_memory(
    tmp_path: Path, app_store: AppStore
) -> None:
    """进入模型请求的是**会话上固定的正文**，不是当下的 memory.json（PRD 11.3 / 9.3）。"""

    memory = memory_for(tmp_path)
    memory.add_entry("回答一律用中文", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="conv-1")
    await seed_conversation(
        app_store,
        conversation_id="conv-1",
        messages=[("user", "我们开始吧")],
    )
    context = await compose_agent_memory_context(
        app_store=app_store, memory=memory, conversation_id="conv-1"
    )
    model = RecordingChatModel()
    async for _chunk in model.stream_chat([*context.messages, {"role": "user", "content": "继续"}]):
        pass
    assert "回答一律用中文" in model.last_request_text()

    # 工作区记忆变了，但本会话快照没变 → 下一次请求仍不带新内容。
    memory.add_entry("后来才有的结论", MemoryKind.CONFIRMED_CONCLUSION, source_conversation_id="conv-2")
    context2 = await compose_agent_memory_context(
        app_store=app_store, memory=memory, conversation_id="conv-1"
    )
    assert context2.snapshot_hash == context.snapshot_hash
    assert "后来才有的结论" not in context2.snapshot_text


# ---------------------------------------------------------------- 快照内容


def test_snapshot_orders_pinned_then_requirements_then_progress_then_topics(tmp_path: Path) -> None:
    """分组顺序：用户主动保存 → 回答要求 → 已确认结论 → 讨论进展 → 最近主题（PRD 11.2）。"""

    memory = memory_for(tmp_path)
    memory.add_entry("讨论中的题目", MemoryKind.TOPIC)
    memory.add_entry("进度：已读到第 3 章", MemoryKind.PROGRESS)
    memory.add_entry("结论：采用 FTS5", MemoryKind.CONFIRMED_CONCLUSION)
    memory.add_entry("回答请用中文", MemoryKind.RESPONSE_PREFERENCE)
    memory.add_entry("用户主动保存的要点", MemoryKind.TOPIC, user_pinned=True)

    text = build_snapshot(memory.load()).text
    order = [
        text.index("用户主动保存") ,
        text.index("回答要求"),
        text.index("已确认结论"),
        text.index("讨论进展"),
        text.index("最近主题"),
    ]
    assert order == sorted(order)
    assert "用户主动保存的要点" in text


def test_snapshot_excludes_inactive_entries(tmp_path: Path) -> None:
    """forgotten / superseded / orphaned 绝不进快照（读取侧也不得恢复）。"""

    memory = memory_for(tmp_path)
    keep = memory.add_entry("保留的结论", MemoryKind.CONFIRMED_CONCLUSION, source_conversation_id="c1").entry
    drop = memory.add_entry("要被遗忘的要求", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="c1").entry
    memory.forget_entry(drop.id)

    text = build_snapshot(memory.load()).text
    assert "保留的结论" in text
    assert "要被遗忘的要求" not in text
    assert keep.id


def test_snapshot_truncates_and_reports_overflow(tmp_path: Path) -> None:
    """超过 8000 字符时截断并回报，供历史查询/整理入口使用（PRD 11.2）。"""

    memory = memory_for(tmp_path)
    for index in range(100):
        memory.add_entry(f"条目 {index}：" + "内容" * 60, MemoryKind.PROGRESS)
    snapshot = build_snapshot(memory.load())

    assert snapshot.truncated is True
    assert snapshot.overflow_count > 0
    assert len(snapshot.text) <= DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS + 64


def test_disabled_memory_yields_empty_snapshot(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.add_entry("某条", MemoryKind.TOPIC, source_conversation_id="c1")
    memory.set_enabled(False)

    snapshot = build_snapshot(memory.load())
    assert snapshot.empty is True
    assert snapshot.hash == build_snapshot(memory.load()).hash


# ---------------------------------------------------------------- 并行


async def test_parallel_conversation_creation_always_matches_a_complete_version(
    tmp_path: Path, app_store: AppStore
) -> None:
    """同一工作区并行创建会话与写记忆时，快照必须对应**某一个完整版本**（PRD 11.3）。

    不变量：每条记忆恰好一个快照条目，且 ``revision`` 等于条目数 —— 所以"快照里的
    条目数"必须恒等于"它记录的 revision"。半成品（写到一半的文件）会让两者错位。
    """

    memory = memory_for(tmp_path)
    conversation_ids = [f"conv-{index}" for index in range(8)]
    for conversation_id in conversation_ids:
        await seed_conversation(app_store, conversation_id=conversation_id)

    stop = asyncio.Event()

    async def writer() -> None:
        index = 0
        while not stop.is_set():
            memory.add_entry(f"并行条目 {index}", MemoryKind.PROGRESS)
            index += 1
            if index >= 40:
                stop.set()

    async def snapshotter(conversation_id: str) -> None:
        while not stop.is_set():
            _record, snapshot = await ensure_conversation_snapshot(
                app_store=app_store, memory=memory, conversation_id=conversation_id
            )
            bullets = snapshot.text.count("\n- ")
            assert bullets == snapshot.revision, (
                f"快照条目数 {bullets} 与 revision {snapshot.revision} 不一致（读到了半成品）"
            )
            await asyncio.sleep(0)

    await asyncio.gather(writer(), *(snapshotter(cid) for cid in conversation_ids))


async def test_new_conversation_after_disable_gets_empty_snapshot(
    tmp_path: Path, app_store: AppStore
) -> None:
    """关闭记忆后新建会话拿到空快照；重新开启后按正常规则读取（PRD 11.4）。"""

    memory = memory_for(tmp_path)
    memory.add_entry("某条记忆", MemoryKind.TOPIC, source_conversation_id="c1")
    memory.set_enabled(False)
    await seed_conversation(app_store, conversation_id="conv-off")

    _record, snapshot = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-off"
    )
    assert snapshot.text == ""

    memory.set_enabled(True)
    memory.add_entry("重新开启后的新条目", MemoryKind.TOPIC, source_conversation_id="c1")
    await seed_conversation(app_store, conversation_id="conv-on")
    _record2, snapshot_on = await ensure_conversation_snapshot(
        app_store=app_store, memory=memory, conversation_id="conv-on"
    )
    assert "重新开启后的新条目" in snapshot_on.text
