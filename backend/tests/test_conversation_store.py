"""ConversationStore 单测：CRUD / 追加 / 标题 / 持久化往返。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.conversation_store import ConversationMessage, ConversationStore


def test_create_and_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    conv = store.create()
    assert conv.title == "新对话"
    # 空会话不进侧边栏列表（避免「新对话」堆积），但本身可读
    assert store.list_summaries() == []
    assert store.get(conv.id) is not None

    store.append_messages(conv.id, [ConversationMessage(role="user", content="你好")])
    summaries = store.list_summaries()
    assert len(summaries) == 1
    assert summaries[0]["id"] == conv.id
    assert summaries[0]["message_count"] == 1


def test_append_messages_updates_count_and_time(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    conv = store.create()
    before = conv.updated_at
    updated = store.append_messages(
        conv.id,
        [
            ConversationMessage(role="user", content="你好"),
            ConversationMessage(role="assistant", content="你好！"),
        ],
    )
    assert updated is not None
    assert len(updated.messages) == 2
    assert updated.messages[0].content == "你好"
    assert store.get(conv.id) is not None
    assert store.list_summaries()[0]["message_count"] == 2
    assert updated.updated_at >= before


def test_set_title_and_get(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    conv = store.create()
    store.append_messages(conv.id, [ConversationMessage(role="user", content="如何导入文档")])
    store.set_title(conv.id, "导入文档指南")
    assert store.get(conv.id).title == "导入文档指南"


def test_delete(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    conv = store.create()
    assert store.delete(conv.id) is True
    assert store.get(conv.id) is None
    assert store.delete(conv.id) is False


def test_append_unknown_returns_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    assert store.append_messages("nope", [ConversationMessage(role="user", content="x")]) is None


def test_persists_across_instances(tmp_path: Path) -> None:
    """写入后新实例同目录可读到（磁盘持久化）。"""
    store1 = ConversationStore(tmp_path)
    conv = store1.create()
    store1.append_messages(conv.id, [ConversationMessage(role="user", content="持久化测试")])
    store1.set_title(conv.id, "持久化标题")

    store2 = ConversationStore(tmp_path)
    reloaded = store2.get(conv.id)
    assert reloaded is not None
    assert reloaded.title == "持久化标题"
    assert reloaded.messages[0].content == "持久化测试"


def _seeded(store: ConversationStore) -> None:
    """给 store 播两条带消息的会话（列表只返回非空会话）。"""

    first = store.create()
    store.append_messages(first.id, [ConversationMessage(role="user", content="第一条")])
    second = store.create()
    store.append_messages(second.id, [ConversationMessage(role="user", content="第二条")])


def test_archive_hides_from_active_list_and_keeps_archive_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _seeded(store)
    active = store.list_summaries()
    target = active[-1]["id"]

    archived = store.set_archived(target, True)
    assert archived is not None
    assert archived.archived is True
    assert archived.archived_at is not None
    assert target not in [item["id"] for item in store.list_summaries()]
    assert [item["id"] for item in store.list_archived_summaries()] == [target]

    restored = store.set_archived(target, False)
    assert restored is not None
    assert restored.archived is False
    assert restored.archived_at is None
    assert store.list_archived_summaries() == []
    assert target in [item["id"] for item in store.list_summaries()]


def test_archive_state_survives_reload(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _seeded(store)
    target = store.list_summaries()[0]["id"]
    store.set_archived(target, True)

    assert [item["id"] for item in ConversationStore(tmp_path).list_archived_summaries()] == [target]


def test_search_excludes_archived(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_messages(
        store.create().id,
        [ConversationMessage(role="user", content="唯一关键词 归档候选")],
    )
    target = store.list_summaries()[0]["id"]
    store.set_archived(target, True)
    assert store.search("唯一关键词") == []


def test_fork_deep_copies_messages_and_links_parent(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    parent = store.create()
    store.append_messages(
        parent.id,
        [ConversationMessage(role="user", content="原问题", sources=[{"source_id": "S1"}])],
    )
    store.set_title(parent.id, "原对话")

    branch = store.fork(parent.id)
    assert branch is not None
    assert branch.parent_id == parent.id
    assert branch.forked_from_title == "原对话"
    assert branch.title == "原对话 · 分支"
    assert [m.content for m in branch.messages] == ["原问题"]

    # 深拷贝：改父消息的 sources 不影响分支
    store.get(parent.id).messages[0].sources.append({"source_id": "S2"})  # type: ignore[union-attr]
    assert len(store.get(branch.id).messages[0].sources) == 1  # type: ignore[union-attr]


def test_fork_unknown_returns_none(tmp_path: Path) -> None:
    assert ConversationStore(tmp_path).fork("nope") is None


def test_delete_parent_promotes_children(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    parent = store.create()
    store.append_messages(parent.id, [ConversationMessage(role="user", content="原问题")])
    branch = store.fork(parent.id)
    assert branch is not None

    assert store.delete(parent.id) is True
    assert store.get(branch.id).parent_id is None  # type: ignore[union-attr]
    assert [item["id"] for item in store.list_summaries()] == [branch.id]


# ------------------------------------------------------------------ 工作区（对话分区）


def _conversation_in(store: ConversationStore, workspace_id: str, text: str) -> str:
    conv = store.create(workspace_id)
    store.append_messages(conv.id, [ConversationMessage(role="user", content=text)])
    return conv.id


def test_conversations_are_scoped_by_workspace(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    default_id = _conversation_in(store, "default", "默认区的对话")
    research_id = _conversation_in(store, "research", "研究区的对话")

    assert [item["id"] for item in store.list_summaries("default")] == [default_id]
    assert [item["id"] for item in store.list_summaries("research")] == [research_id]
    # 不传工作区 = 不过滤（跨分区列举）
    assert {item["id"] for item in store.list_summaries()} == {default_id, research_id}
    assert store.count_by_workspace() == {"default": 1, "research": 1}
    assert store.search("研究区", "default") == []
    assert [item["id"] for item in store.search("研究区", "research")] == [research_id]


def test_set_workspace_moves_conversation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conv_id = _conversation_in(store, "default", "待归类")

    moved = store.set_workspace(conv_id, "research")
    assert moved is not None
    assert moved.workspace_id == "research"
    assert store.list_summaries("default") == []
    assert store.get(conv_id).workspace_id == "research"  # type: ignore[union-attr]
    assert store.set_workspace("nope", "research") is None


def test_archive_workspace_archives_only_active_conversations(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "research", "第一条")
    second = _conversation_in(store, "research", "第二条")
    other = _conversation_in(store, "default", "别的分区")
    store.set_archived(second, True)

    assert store.archive_workspace("research") == 1
    assert store.list_summaries("research") == []
    archived_ids = {item["id"] for item in store.list_archived_summaries()}
    assert archived_ids == {first, second}
    # 已归档的那条不被重复计时；其它分区不受影响
    assert [item["id"] for item in store.list_summaries("default")] == [other]
    assert store.count_by_workspace() == {"default": 1}


def test_fork_stays_in_parent_workspace(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    parent_id = _conversation_in(store, "research", "原问题")
    branch = store.fork(parent_id)
    assert branch is not None
    assert branch.workspace_id == "research"


def test_sidebar_order_is_recency_only(tmp_path: Path) -> None:
    """侧边栏恒为「最近活动优先」：任何一次活动（含 move_in_sidebar）都把它顶到最前。

    2026-09-22（需求 2）起**没有**「手工拖拽过的排在前」的档位：同一工作区内怎么拖都不会
    改变显示顺序，否则拖过一次的旧会话会永远压在新会话之上（这正是用户报的缺陷）。
    """

    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "第一条")
    second = _conversation_in(store, "default", "第二条")
    third = _conversation_in(store, "default", "第三条")

    # 初始：最近更新的在前
    assert [item["id"] for item in store.list_summaries("default")] == [third, second, first]

    # 「第一条」再活动一次（等同同区拖拽 / 收到新消息）⇒ 它排到最前
    moved = store.move_in_sidebar(first)
    assert moved is not None
    assert [item["id"] for item in store.list_summaries("default")] == [first, third, second]
    assert "sort_index" not in store.list_summaries("default")[0]


def test_recency_order_is_stable_when_timestamps_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时间戳打平时顺序必须仍是「越新越靠前」，不能退化成创建顺序。

    冻结 ``_now()`` 精确复现全量测试下的真实条件（Windows 时钟粒度约 15.6ms，
    连续创建的三条会话常常落在同一时间戳）。修复前该断言会失败：
    侧边栏回退到「创建顺序」（最早创建的在最前），与 recency 语义相反。
    """

    import app.core.conversation_store as conversation_store_module

    frozen = "2026-09-19T00:00:00.000000+00:00"
    monkeypatch.setattr(conversation_store_module, "_now", lambda: frozen)

    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "第一条")
    second = _conversation_in(store, "default", "第二条")
    third = _conversation_in(store, "default", "第三条")

    # 打平的时间戳下仍须按「最近活动」倒序（第三条最后创建 → 排最前）
    assert [item["id"] for item in store.list_summaries("default")] == [third, second, first]

    # 让「第一条」再活动一次（时间戳仍打平）：靠活动序号兜底，它必须排到最前，
    # 另外两条的相对顺序不许退化成字典迭代顺序
    store.move_in_sidebar(first)
    ordered = [item["id"] for item in store.list_summaries("default")]
    assert ordered == [first, third, second]


def test_archived_order_is_stable_when_timestamps_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """归档列表同样不允许依赖字典迭代顺序：归档时间打平时按最近活动倒序。"""

    import app.core.conversation_store as conversation_store_module

    frozen = "2026-09-19T00:00:00.000000+00:00"
    monkeypatch.setattr(conversation_store_module, "_now", lambda: frozen)

    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "第一条")
    second = _conversation_in(store, "default", "第二条")
    third = _conversation_in(store, "default", "第三条")
    for conversation_id in (first, second, third):
        store.set_archived(conversation_id, True)

    assert [item["id"] for item in store.list_archived_summaries()] == [third, second, first]


def test_move_in_sidebar_within_workspace_keeps_recency_order(tmp_path: Path) -> None:
    """同区拖拽不再排序：只会把被拖的那条顶到最前（它刚「活动」过）。"""

    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "A")
    second = _conversation_in(store, "default", "B")
    third = _conversation_in(store, "default", "C")
    assert [item["id"] for item in store.list_summaries("default")] == [third, second, first]

    # 拖「A」到本区任意位置：没有位次语义，只剩「刷新了活动时间」这一效果
    store.move_in_sidebar(third)
    store.move_in_sidebar(first)
    assert [item["id"] for item in store.list_summaries("default")] == [first, third, second]


def test_move_in_sidebar_moves_to_another_workspace(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conv_id = _conversation_in(store, "default", "待搬")
    other = _conversation_in(store, "research", "目标区已有")

    moved = store.move_in_sidebar(conv_id, workspace_id="research")
    assert moved is not None
    assert moved.workspace_id == "research"
    assert store.list_summaries("default") == []
    assert [item["id"] for item in store.list_summaries("research")] == [conv_id, other]


def test_move_in_sidebar_unknown_returns_none(tmp_path: Path) -> None:
    assert ConversationStore(tmp_path).move_in_sidebar("nope") is None


def test_move_in_sidebar_order_survives_reload(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "A")
    _second = _conversation_in(store, "default", "B")
    store.move_in_sidebar(first)

    reloaded = ConversationStore(tmp_path)
    assert [item["id"] for item in reloaded.list_summaries("default")] == [first, _second]


def test_search_order_is_stable_when_timestamps_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """搜索结果同样不允许依赖字典迭代顺序：时间戳打平时按最近活动倒序。

    ``search`` 原先只按 ``updated_at`` 排序（``reverse=True``），打平后退化为
    **字典插入顺序**，即最早创建的最先返回，与「最近更新优先」的语义相反 ——
    与侧边栏/归档列表是同一类缺陷（t12 一并修）。
    """

    import app.core.conversation_store as conversation_store_module

    frozen = "2026-09-19T00:00:00.000000+00:00"
    monkeypatch.setattr(conversation_store_module, "_now", lambda: frozen)

    store = ConversationStore(tmp_path)
    first = _conversation_in(store, "default", "共同关键词 第一条")
    second = _conversation_in(store, "default", "共同关键词 第二条")
    third = _conversation_in(store, "default", "共同关键词 第三条")

    assert [item["id"] for item in store.search("共同关键词", "default")] == [
        third,
        second,
        first,
    ]

