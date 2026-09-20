"""``memory.json`` 落地、并发与中断恢复（PRD 11.2 / 11.5；验收 A28、A30）。

这里验证的是**文件级不变量**：结构与契约逐字一致、替换是原子的、``operation_id``
去重、锁顺序固定、并发不丢更新、中断后重放不产生重复条目。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.agent.contracts import (
    DEFAULT_MEMORY_MAX_CHARS,
    MEMORY_ENTRY_KEYS,
    MEMORY_FILE_KEYS,
    MemoryEntryStatus,
    MemoryKind,
)
from app.memory.locking import (
    AppWriteLock,
    MemoryWriteLockError,
    memory_maintenance_locks,
    workspace_write_locks,
)
from app.memory.workspace_memory import (
    MemoryDisabledError,
    MemoryFileError,
    MemoryRevisionConflict,
    WorkspaceMemory,
    build_snapshot,
)
from tests.fakes_memory import memory_for, workspace_paths_for

# ---------------------------------------------------------------- 结构契约


def test_memory_json_field_order_matches_prd_and_contract(tmp_path: Path) -> None:
    """落盘字段名与顺序逐字等于契约常量（PRD 11.2 的文件级 9 项、条目级 11 项）。"""

    memory = memory_for(tmp_path)
    memory.add_entry(
        "回答一律用中文",
        MemoryKind.RESPONSE_PREFERENCE,
        source_conversation_id="conv-1",
        source_message_ids=["m1"],
    )
    raw = json.loads(memory.path.read_text(encoding="utf-8"))

    assert list(raw) == list(MEMORY_FILE_KEYS)
    assert list(raw["entries"][0]) == list(MEMORY_ENTRY_KEYS)
    assert raw["schema_version"] == 1
    assert raw["enabled"] is True
    assert raw["revision"] == 1
    assert raw["invalidation_revision"] == 0
    assert raw["entries"][0]["kind"] == "response_preference"
    assert raw["entries"][0]["status"] == "active"


def test_corrupt_memory_file_raises_instead_of_resetting(tmp_path: Path) -> None:
    """损坏文件必须显式报错：静默重置等于把用户记忆清空，下次写入再覆盖掉原文。"""

    memory = memory_for(tmp_path)
    memory.path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(MemoryFileError):
        memory.load()


def test_missing_file_reads_as_empty_version(tmp_path: Path) -> None:
    """文件不存在（全新工作区）不是错误。"""

    memory = memory_for(tmp_path)
    state = memory.load()

    assert state.entries == []
    assert state.revision == 0
    assert state.enabled is True


# ---------------------------------------------------------------- 原子替换


def test_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.add_entry("主题 A", MemoryKind.TOPIC)

    leftovers = [path.name for path in memory.path.parent.glob("*.tmp")]
    assert leftovers == []


def test_failed_replace_keeps_previous_version_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """替换失败时磁盘上仍是**上一份完整版本**（PRD 11.2「原子文件替换」）。"""

    memory = memory_for(tmp_path)
    memory.add_entry("第一条", MemoryKind.TOPIC)
    before = memory.path.read_text(encoding="utf-8")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("磁盘满")

    monkeypatch.setattr("app.memory.workspace_memory.os.replace", boom)
    with pytest.raises(OSError):
        memory.add_entry("第二条", MemoryKind.TOPIC)

    assert memory.path.read_text(encoding="utf-8") == before
    monkeypatch.undo()
    assert memory.load().revision == 1


def test_revision_increases_only_when_content_changes(tmp_path: Path) -> None:
    """``revision`` 是内容版本而不是写入次数：幂等重放不涨版本。"""

    memory = memory_for(tmp_path)
    first = memory.add_entry("主题 A", MemoryKind.TOPIC, source_conversation_id="c1")
    assert first.memory.revision == 1

    again = memory.add_entry("主题 A", MemoryKind.TOPIC, source_conversation_id="c1")
    assert again.applied is False
    assert again.memory.revision == 1

    memory.add_entry("主题 B", MemoryKind.TOPIC, source_conversation_id="c1")
    assert memory.load().revision == 2


# ---------------------------------------------------------------- 条目规则


def test_auto_entry_is_truncated_but_user_pinned_is_kept(tmp_path: Path) -> None:
    """自动条目按 300 字符上限截断；用户主动保存的内容保留全文并要求整理（PRD 11.2）。"""

    memory = memory_for(tmp_path)
    long_text = "很长的陈述" * 100
    auto = memory.add_entry(long_text, MemoryKind.PROGRESS, auto=True)
    assert len(auto.entry.text) == DEFAULT_MEMORY_MAX_CHARS

    pinned = memory.add_entry(long_text + "用户版本", MemoryKind.TOPIC, user_pinned=True)
    assert len(pinned.entry.text) > DEFAULT_MEMORY_MAX_CHARS
    assert build_snapshot(memory.load()).needs_tidy is True


def test_same_content_does_not_create_second_entry(tmp_path: Path) -> None:
    """同 ``kind`` + 同归一化文本视为同一条（重复提炼不产生重复条目）。"""

    memory = memory_for(tmp_path)
    first = memory.add_entry("讨论 bge-m3 选型", MemoryKind.PROGRESS, source_message_ids=["m1"])
    second = memory.add_entry(
        "  讨论   bge-m3 选型 ", MemoryKind.PROGRESS, source_message_ids=["m2"]
    )

    state = memory.load()
    assert len(state.entries) == 1
    assert second.entry.id == first.entry.id
    assert state.entries[0].source_message_ids == ["m1", "m2"]


def test_forgotten_entry_is_never_revived_by_automation(tmp_path: Path) -> None:
    """自动流程不得恢复已遗忘条目（PRD 11.2 末段）。"""

    memory = memory_for(tmp_path)
    entry = memory.add_entry("用户要求英文回答", MemoryKind.RESPONSE_PREFERENCE).entry
    memory.forget_entry(entry.id)

    revived = memory.add_entry("用户要求英文回答", MemoryKind.RESPONSE_PREFERENCE)

    assert revived.applied is False
    assert memory.load().entries[0].status is MemoryEntryStatus.FORGOTTEN


def test_superseded_requires_existing_target(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    old = memory.add_entry("旧结论", MemoryKind.CONFIRMED_CONCLUSION).entry
    new = memory.add_entry("新结论", MemoryKind.CONFIRMED_CONCLUSION).entry

    result = memory.supersede_entry(old.id, new.id)

    assert result.applied is True
    assert result.memory.entries[0].status is MemoryEntryStatus.SUPERSEDED
    assert result.memory.entries[0].superseded_by == new.id
    with pytest.raises(MemoryFileError):
        memory.supersede_entry(old.id, "mem-does-not-exist")


# ---------------------------------------------------------------- 开关


def test_disabled_workspace_rejects_writes_and_empties_snapshot(tmp_path: Path) -> None:
    """``enabled=false`` 与失效版本同时提交，写入停止，快照为空（PRD 11.4）。"""

    memory = memory_for(tmp_path)
    memory.add_entry("回答一律用中文", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="c1")
    before = memory.load()

    result = memory.set_enabled(False)

    assert result.applied is True
    assert result.memory.enabled is False
    assert result.memory.revision == before.revision + 1
    assert result.memory.invalidation_revision == before.invalidation_revision + 1
    assert build_snapshot(memory.load()).empty is True
    assert "c1" in result.affected_conversation_ids

    with pytest.raises(MemoryDisabledError):
        memory.add_entry("关闭后仍可写？", MemoryKind.TOPIC)

    assert memory.set_enabled(False).applied is False
    assert memory.set_enabled(True).applied is True


def test_disabled_workspace_still_allows_explicit_reopen(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.set_enabled(False)
    reopened = memory.set_enabled(True)

    assert reopened.memory.enabled is True
    memory.add_entry("重新开启后可写", MemoryKind.TOPIC)
    assert len(memory.load().entries) == 1


def test_user_hygiene_still_works_while_memory_is_off(tmp_path: Path) -> None:
    """关闭只停**自动**写入：用户仍能纠正/遗忘自己的记忆（PRD 11.4 / 18.1）。

    如果这里也一并禁掉，用户关掉开关后就再也删不掉文件里的错误记忆 —— 数据被锁死。
    """

    memory = memory_for(tmp_path)
    requirement = memory.add_entry(
        "回答一律用英文", MemoryKind.RESPONSE_PREFERENCE, source_conversation_id="c1"
    ).entry
    conclusion = memory.add_entry(
        "旧结论", MemoryKind.CONFIRMED_CONCLUSION, source_conversation_id="c1"
    ).entry
    memory.set_enabled(False)

    # 自动路径：拒绝。
    with pytest.raises(MemoryDisabledError):
        memory.add_entry("自动写入", MemoryKind.TOPIC)
    with pytest.raises(MemoryDisabledError):
        memory.apply_operations(
            operation_id="op-off",
            conversation_id="c1",
            candidates=[("自动提炼", "progress")],
        )

    # 用户整理路径：允许。
    assert memory.forget_entry(requirement.id).applied is True
    assert memory.update_entry(conclusion.id, text="修正后的结论").applied is True
    assert memory.load().enabled is False
    assert memory.load().invalidations  # 失效记录照常登记


# ---------------------------------------------------------------- 乐观并发


def test_expected_revision_mismatch_is_rejected(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.add_entry("第一条", MemoryKind.TOPIC)

    with pytest.raises(MemoryRevisionConflict):
        memory.add_entry("并发写入", MemoryKind.TOPIC, expected_revision=0)

    assert len(memory.load().entries) == 1


# ---------------------------------------------------------------- 锁


def test_lock_order_is_session_then_workspace_sorted_by_storage_key() -> None:
    """跨两区按 ``storage_key`` 顺序取锁（PRD 11.2）：传入顺序不影响获取顺序。"""

    with workspace_write_locks("zzz-key", "aaa-key") as locks:
        assert locks.storage_keys == ("aaa-key", "zzz-key")

    # 会话锁在工作区锁之前获取；两者都可重入（同一线程嵌套调用不死锁）。
    with (
        memory_maintenance_locks("conv-1", storage_keys=["aaa-key", "zzz-key"]),
        memory_maintenance_locks("conv-1", storage_keys=["zzz-key"]),
    ):
        pass


def test_app_write_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    """重复启动写入进程返回明确错误（PRD 11.2），释放后可以重新获取。"""

    first = AppWriteLock(tmp_path)
    second = AppWriteLock(tmp_path)

    with first, pytest.raises(MemoryWriteLockError):
        second.acquire()
    with second:
        pass


def test_concurrent_writes_do_not_lose_updates(tmp_path: Path) -> None:
    """A28：并发记忆写入不丢失（每个线程用**独立实例**，证明锁不是实例本地的）。"""

    baseline = memory_for(tmp_path)
    baseline.add_entry("起点", MemoryKind.TOPIC)

    workers = 12
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        handle = memory_for(tmp_path)
        try:
            barrier.wait(timeout=10)
            handle.add_entry(f"并发条目 {index}", MemoryKind.TOPIC, operation_id=f"op-{index}")
        except BaseException as exc:  # noqa: BLE001 -- 收集后统一断言
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    state = memory_for(tmp_path).load()
    texts = {entry.text for entry in state.entries}
    assert texts == {"起点", *(f"并发条目 {index}" for index in range(workers))}
    assert state.revision == workers + 1


def test_concurrent_writes_never_land_in_another_workspace(tmp_path: Path) -> None:
    """A28：不向错误工作区提交记忆（每个 storage_key 一份文件、一把锁）。"""

    left = memory_for(tmp_path, "aaa-key")
    right = memory_for(tmp_path, "bbb-key")
    errors: list[BaseException] = []

    def writer(handle: WorkspaceMemory, label: str) -> None:
        try:
            for index in range(6):
                handle.add_entry(f"{label}-{index}", MemoryKind.TOPIC)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(left, "left")),
        threading.Thread(target=writer, args=(right, "right")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    left_texts = {entry.text for entry in memory_for(tmp_path, "aaa-key").load().entries}
    right_texts = {entry.text for entry in memory_for(tmp_path, "bbb-key").load().entries}
    assert left_texts == {f"left-{index}" for index in range(6)}
    assert right_texts == {f"right-{index}" for index in range(6)}
    assert left.path != right.path


# ---------------------------------------------------------------- A30 中断恢复


def test_interruption_after_commit_is_recovered_by_operation_id(tmp_path: Path) -> None:
    """A30：文件提交后进程中断 → 恢复确认已应用操作，条目与游标不重复。"""

    first = memory_for(tmp_path)
    committed = first.apply_operations(
        operation_id="op-round-1",
        conversation_id="conv-1",
        cursor_message_id="msg-9",
        candidates=[("讨论进展：确认使用 FTS5", "progress", ["msg-9"])],
    )
    assert committed.applied is True

    # 进程中断（内存对象丢弃）后重新打开同一个文件。
    reopened = memory_for(tmp_path)
    replayed = reopened.apply_operations(
        operation_id="op-round-1",
        conversation_id="conv-1",
        cursor_message_id="msg-9",
        candidates=[("讨论进展：确认使用 FTS5", "progress", ["msg-9"])],
    )

    state = reopened.load()
    assert replayed.applied is False
    assert state.revision == 1
    assert len(state.entries) == 1
    assert state.processed_cursors == {"conv-1": "msg-9"}
    assert state.applied_operations == ["op-round-1"]


def test_interruption_before_commit_keeps_old_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A30 的另一半：替换前中断 → 条目、游标、已应用操作**一个都不能留下**。"""

    memory = memory_for(tmp_path)
    memory.apply_operations(
        operation_id="op-round-1",
        conversation_id="conv-1",
        cursor_message_id="msg-3",
        candidates=[("第一轮", "progress", ["msg-3"])],
    )
    before = memory.load()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("替换失败")

    monkeypatch.setattr("app.memory.workspace_memory.os.replace", boom)
    with pytest.raises(OSError):
        memory.apply_operations(
            operation_id="op-round-2",
            conversation_id="conv-1",
            cursor_message_id="msg-7",
            candidates=[("第二轮", "progress", ["msg-7"])],
        )
    monkeypatch.undo()

    after = memory.load()
    assert after.revision == before.revision
    assert after.processed_cursors == before.processed_cursors
    assert after.applied_operations == before.applied_operations
    assert [entry.text for entry in after.entries] == [entry.text for entry in before.entries]


def test_source_deletion_orphans_only_active_entries(tmp_path: Path) -> None:
    """会话删除：有效派生记忆转 ``orphaned``；forgotten/superseded 保持原状态（PRD 11.5）。"""

    memory = memory_for(tmp_path)
    active = memory.add_entry("结论 A", MemoryKind.CONFIRMED_CONCLUSION, source_conversation_id="conv-x").entry
    forgotten = memory.add_entry("结论 B", MemoryKind.PROGRESS, source_conversation_id="conv-x").entry
    memory.forget_entry(forgotten.id)
    other = memory.add_entry("别的会话", MemoryKind.TOPIC, source_conversation_id="conv-y").entry

    result = memory.apply_source_deletion("conv-x", operation_id="prune-1")

    by_id = {entry.id: entry for entry in result.memory.entries}
    assert by_id[active.id].status is MemoryEntryStatus.ORPHANED
    assert by_id[forgotten.id].status is MemoryEntryStatus.FORGOTTEN
    assert by_id[other.id].status is MemoryEntryStatus.ACTIVE
    assert result.affected_conversation_ids == ("conv-x",)


def test_two_workspace_files_stay_separate_on_disk(tmp_path: Path) -> None:
    left_paths = workspace_paths_for(tmp_path, "aaa-key")
    right_paths = workspace_paths_for(tmp_path, "bbb-key")

    assert left_paths.memory_json != right_paths.memory_json
    assert left_paths.jieba_cache_dir != right_paths.jieba_cache_dir
    assert left_paths.memory_json.parent.name == "aaa-key"
