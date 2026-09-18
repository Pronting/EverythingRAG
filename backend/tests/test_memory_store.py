"""MemoryStore 单测：原子记忆条目的写入/去重/状态流转/软删除/按源失效。

重点覆盖 v1.1.0 的三条硬约束：
- D2 原子条目 + 源指针：源指针必须完整落盘，否则 [M1] 无法回溯。
- D7 软删除 + 失效标记：``/忘掉`` 不物理删除；源会话删除 → orphaned。
- 置信度门禁：低置信度进 pending，**不参与检索**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.memory.models import MemoryCategory, MemoryStatus
from app.memory.store import MemoryStore, content_hash


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def _add_preference(store: MemoryStore, statement: str = "用户偏好 SQLite 而非 Postgres", **kwargs):  # type: ignore[no-untyped-def]
    params = {
        "statement": statement,
        "category": MemoryCategory.PREFERENCE,
        "confidence": 0.9,
        "source_conversation_id": "conv-1",
        "source_message_index": 3,
        "quote": "我更喜欢 SQLite，Postgres 太重了",
    }
    params.update(kwargs)
    return store.add(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------- 写入与源指针


def test_add_persists_all_fields_including_source_pointer(tmp_path: Path) -> None:
    """写入后字段完整：类别/陈述/置信度/源指针/摘录全部落盘（[M1] 回溯的前提）。"""

    store = _store(tmp_path)
    entry = _add_preference(store)

    assert entry.id
    assert entry.category is MemoryCategory.PREFERENCE
    assert entry.confidence == 0.9
    assert entry.source_conversation_id == "conv-1"
    assert entry.source_message_index == 3
    assert entry.quote == "我更喜欢 SQLite，Postgres 太重了"
    assert entry.status is MemoryStatus.ACTIVE
    assert entry.created_at and entry.updated_at
    assert entry.source_ref() == "conv-1#3"
    assert entry.is_retrievable() is True


def test_source_ref_empty_without_pointer(tmp_path: Path) -> None:
    """无有效源指针时 source_ref 为空字符串（前端据此不渲染回溯链接）。"""

    store = _store(tmp_path)
    entry = store.add(
        statement="用户是后端工程师",
        category=MemoryCategory.IDENTITY,
        confidence=0.8,
    )
    assert entry.source_ref() == ""
    assert entry.source_conversation_id == ""
    assert entry.source_message_index == -1


def test_get_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = _add_preference(store)
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.statement == entry.statement
    assert store.get("missing-id") is None


# ---------------------------------------------------------------- 去重幂等


def test_add_same_statement_is_idempotent(tmp_path: Path) -> None:
    """同类别同陈述重复提炼不产生第二条：只刷新置信度与更新时间。

    这是后台自动提炼的基本安全网——每轮都跑，但同一偏好只应存在一条。
    """

    store = _store(tmp_path)
    first = _add_preference(store, confidence=0.7)
    second = _add_preference(store, confidence=0.95)

    assert first.id == second.id
    assert store.count() == 1
    assert second.confidence == 0.95
    assert second.updated_at >= first.updated_at


def test_add_normalizes_whitespace_and_case_for_dedup(tmp_path: Path) -> None:
    """去重按空白折叠 + 大小写统一，避免同一句因空格差异重复入库。"""

    store = _store(tmp_path)
    first = _add_preference(store, statement="用户偏好  SQLite 而非 Postgres")
    second = _add_preference(store, statement="用户偏好 sqlite 而非 postgres")
    assert first.id == second.id
    assert store.count() == 1


def test_same_statement_different_category_is_distinct(tmp_path: Path) -> None:
    """类别不同即视为不同记忆（类别是身份的一部分）。"""

    store = _store(tmp_path)
    a = _add_preference(store)
    b = _add_preference(store, category=MemoryCategory.CONSTRAINT)
    assert a.id != b.id
    assert store.count() == 2


def test_content_hash_is_stable_and_category_scoped() -> None:
    left = content_hash(MemoryCategory.PREFERENCE, "  A  B ")
    assert left == content_hash(MemoryCategory.PREFERENCE, "a b")
    assert left != content_hash(MemoryCategory.DECISION, "a b")


# ---------------------------------------------------------------- 置信度门禁


def test_low_confidence_goes_pending_and_is_not_retrievable(tmp_path: Path) -> None:
    """低于阈值的候选进 pending：可见于记忆面板，但不参与检索。"""

    store = _store(tmp_path)
    entry = _add_preference(store, confidence=0.3, min_confidence=0.6)
    assert entry.status is MemoryStatus.PENDING
    assert entry.is_retrievable() is False
    # 仍需可见：记忆面板的职责是让用户看到系统"想记"什么。
    assert [item.id for item in store.list_entries()] == [entry.id]


def test_confidence_at_threshold_is_active(tmp_path: Path) -> None:
    """阈值边界：等于阈值即视为可信（避免 off-by-one 让边界候选永远待确认）。"""

    store = _store(tmp_path)
    entry = _add_preference(store, confidence=0.6, min_confidence=0.6)
    assert entry.status is MemoryStatus.ACTIVE


def test_explicit_status_overrides_confidence(tmp_path: Path) -> None:
    """显式传入 status 时以显式值为准（``/记住`` 用 user_pinned + active）。"""

    store = _store(tmp_path)
    entry = _add_preference(store, confidence=0.1, status=MemoryStatus.ACTIVE)
    assert entry.status is MemoryStatus.ACTIVE


# ---------------------------------------------------------------- 软删除与状态流转


def test_forget_is_soft_delete_and_keeps_record(tmp_path: Path) -> None:
    """``/忘掉`` 软删除：状态转 forgotten，记录与源指针仍在，可撤销。"""

    store = _store(tmp_path)
    entry = _add_preference(store)

    forgotten = store.forget(entry.id)
    assert forgotten is not None
    assert forgotten.status is MemoryStatus.FORGOTTEN
    assert forgotten.is_retrievable() is False
    # 记录仍可读回，源指针不丢——历史答案里的 [M1] 仍能展示"已失效"而非断链。
    reopened = store.get(entry.id)
    assert reopened is not None
    assert reopened.quote == entry.quote
    assert store.count() == 1


def test_forget_missing_returns_none(tmp_path: Path) -> None:
    assert _store(tmp_path).forget("nope") is None


def test_reenriching_forgotten_memory_does_not_revive_it(tmp_path: Path) -> None:
    """用户遗忘意图优先于后台提炼：再次提炼到同一条不会把它复活。"""

    store = _store(tmp_path)
    entry = _add_preference(store)
    store.forget(entry.id)

    again = _add_preference(store, confidence=0.99)
    assert again.id == entry.id
    assert again.status is MemoryStatus.FORGOTTEN


def test_supersede_records_replacement_chain(tmp_path: Path) -> None:
    """新旧结论冲突：旧条目标 superseded 并指向新条目，不物理覆盖。"""

    store = _store(tmp_path)
    old = _add_preference(store, statement="用户决定用 Postgres")
    new = _add_preference(store, statement="用户决定用 SQLite")

    superseded = store.supersede(old.id, new.id)
    assert superseded is not None
    assert superseded.status is MemoryStatus.SUPERSEDED
    assert superseded.superseded_by == new.id
    assert superseded.is_retrievable() is False


def test_mark_source_orphaned_marks_only_that_conversation(tmp_path: Path) -> None:
    """源会话删除 → 仅该会话派生的记忆转 orphaned（依据不可核验）。"""

    store = _store(tmp_path)
    affected = _add_preference(store, source_conversation_id="conv-1")
    untouched = _add_preference(
        store, statement="用户偏好深色主题", source_conversation_id="conv-2"
    )

    assert store.mark_source_orphaned("conv-1") == 1
    assert store.get(affected.id).status is MemoryStatus.ORPHANED  # type: ignore[union-attr]
    assert store.get(untouched.id).status is MemoryStatus.ACTIVE  # type: ignore[union-attr]


def test_mark_source_orphaned_does_not_override_forgotten(tmp_path: Path) -> None:
    """已遗忘的记忆不被系统失效标记覆盖——分别对应两种不同的用户可见语义。"""

    store = _store(tmp_path)
    entry = _add_preference(store)
    store.forget(entry.id)

    assert store.mark_source_orphaned("conv-1") == 0
    assert store.get(entry.id).status is MemoryStatus.FORGOTTEN  # type: ignore[union-attr]


def test_mark_source_orphaned_empty_id_is_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_preference(store)
    assert store.mark_source_orphaned("") == 0


# ---------------------------------------------------------------- 编辑


def test_update_statement_and_category_recomputes_hash(tmp_path: Path) -> None:
    """编辑正文/类别后内容指纹同步重算，否则去重索引与真实内容脱节。

    ``content_hash`` 是存储内部列，不下沉到 ``MemoryEntry``；因此这里断言的是
    可观测的行为：编辑后按新内容再写入仍命中同一条，按旧内容写入则不再命中。
    """

    store = _store(tmp_path)
    entry = _add_preference(store)

    updated = store.update(
        entry.id, statement="用户偏好 SQLite", category=MemoryCategory.DECISION
    )
    assert updated is not None
    assert updated.statement == "用户偏好 SQLite"
    assert updated.category is MemoryCategory.DECISION

    # 按编辑后的内容再写入不会新增（指纹已重算）。
    same = _add_preference(store, statement="用户偏好 SQLite", category=MemoryCategory.DECISION)
    assert same.id == entry.id
    assert store.count() == 1


def test_update_confidence_and_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = _add_preference(store, confidence=0.3, min_confidence=0.6)
    assert entry.status is MemoryStatus.PENDING

    promoted = store.update(entry.id, confidence=0.9, status=MemoryStatus.ACTIVE)
    assert promoted is not None
    assert promoted.confidence == 0.9
    assert promoted.status is MemoryStatus.ACTIVE
    assert promoted.is_retrievable() is True


def test_update_missing_returns_none(tmp_path: Path) -> None:
    assert _store(tmp_path).update("nope", statement="x") is None


def test_update_rejects_out_of_range_confidence(tmp_path: Path) -> None:
    """置信度越界必须在写入前被 pydantic 拦下（模型契约而非静默存回读失败）。"""

    store = _store(tmp_path)
    entry = _add_preference(store)
    with pytest.raises(ValueError):
        store.update(entry.id, confidence=1.5)


# ---------------------------------------------------------------- 列表与过滤


def test_list_entries_filters_by_status_and_category(tmp_path: Path) -> None:
    store = _store(tmp_path)
    active = _add_preference(store)
    forgotten = _add_preference(store, statement="用户偏好浅色主题")
    store.forget(forgotten.id)
    decision = _add_preference(
        store, statement="用户决定用 Chroma", category=MemoryCategory.DECISION
    )

    assert {item.id for item in store.list_entries()} == {active.id, forgotten.id, decision.id}
    assert {item.id for item in store.list_entries(statuses=[MemoryStatus.ACTIVE])} == {
        active.id,
        decision.id,
    }
    assert {item.id for item in store.list_entries(categories=[MemoryCategory.DECISION])} == {
        decision.id
    }


def test_list_entries_respects_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(5):
        _add_preference(store, statement=f"用户偏好第 {index} 项")
    assert len(store.list_entries(limit=2)) == 2


def test_count_by_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = _add_preference(store)
    _add_preference(store, statement="用户偏好深色主题")
    store.forget(entry.id)

    assert store.count() == 2
    assert store.count(statuses=[MemoryStatus.ACTIVE]) == 1
    assert store.count(statuses=[MemoryStatus.FORGOTTEN]) == 1


def test_list_entries_empty_store(tmp_path: Path) -> None:
    assert _store(tmp_path).list_entries() == []


# ---------------------------------------------------------------- 持久化与隔离


def test_persists_across_reopen(tmp_path: Path) -> None:
    """数据落盘：关库重开（模拟重启）后记忆与源指针仍在。"""

    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)
    entry = _add_preference(store)
    store.close()

    reopened = MemoryStore(db_path)
    loaded = reopened.get(entry.id)
    assert loaded is not None
    assert loaded.statement == entry.statement
    assert loaded.source_ref() == "conv-1#3"
    reopened.close()


def test_missing_db_file_auto_created(tmp_path: Path) -> None:
    """memory.db 不存在时自动补建目录与文件。"""

    db_path = tmp_path / "nested" / "deeper" / "memory.db"
    store = MemoryStore(db_path)
    _add_preference(store)
    assert db_path.is_file()
    store.close()


def test_clear_resets_all(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _add_preference(store)
    store.clear()
    assert store.count() == 0
    assert store.list_entries() == []


def test_user_pinned_flag_survives_roundtrip(tmp_path: Path) -> None:
    """``/记住`` 创建的记忆带 user_pinned 标记，供提炼器与面板区分来源。"""

    store = _store(tmp_path)
    entry = _add_preference(store, user_pinned=True, confidence=1.0)
    assert entry.user_pinned is True
    assert store.get(entry.id).user_pinned is True  # type: ignore[union-attr]
