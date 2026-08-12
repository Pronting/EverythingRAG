"""ConversationStore 单测：CRUD / 追加 / 标题 / 持久化往返。"""

from __future__ import annotations

from pathlib import Path

from app.core.conversation_store import ConversationMessage, ConversationStore


def test_create_and_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import tempfile

    store = ConversationStore(Path(tempfile.mkdtemp()))
    conv = store.create()
    assert conv.title == "新对话"
    summaries = store.list_summaries()
    assert len(summaries) == 1
    assert summaries[0]["id"] == conv.id
    assert summaries[0]["message_count"] == 0


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
