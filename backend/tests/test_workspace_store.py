"""WorkspaceStore 单测：默认分区 / 新建 / 改名 / 删除保护 / 持久化。"""

from __future__ import annotations

from pathlib import Path

from app.core.workspace_store import (
    DEFAULT_WORKSPACE_ID,
    DEFAULT_WORKSPACE_NAME,
    WorkspaceStore,
)


def test_default_workspace_is_created_and_listed_first(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    listed = store.list()
    assert [workspace.id for workspace in listed] == [DEFAULT_WORKSPACE_ID]
    assert listed[0].is_default is True
    assert listed[0].name == DEFAULT_WORKSPACE_NAME


def test_create_workspace_derives_slug_and_keeps_order(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("Work Notes")
    assert workspace.id == "work-notes"
    assert workspace.is_default is False
    # 默认分区始终在最前，新建的排在后面
    assert [item.id for item in store.list()] == [DEFAULT_WORKSPACE_ID, "work-notes"]


def test_create_workspace_with_chinese_name_falls_back_to_generated_id(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("读书笔记")
    assert workspace.name == "读书笔记"
    assert workspace.id == "workspace"
    # 第二个中文名分区不会撞 id
    assert store.create("工作").id == "workspace-2"
    # 中英混合名称也走同一兜底（不剥出无意义的 "a"）
    assert store.create("工作区 A").id == "workspace-3"


def test_create_workspace_truncates_name_and_defaults_empty(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    assert len(store.create("x" * 50).name) == 20
    assert store.create("   ").name == "新工作区"


def test_rename_workspace(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("临时")
    renamed = store.rename(workspace.id, "长期项目")
    assert renamed is not None
    assert renamed.name == "长期项目"
    # 默认分区可改名
    assert store.rename(DEFAULT_WORKSPACE_ID, "常用").name == "常用"  # type: ignore[union-attr]
    assert store.rename(workspace.id, "   ") is None
    assert store.rename("nope", "x") is None


def test_default_workspace_cannot_be_deleted(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    assert store.delete(DEFAULT_WORKSPACE_ID) is False
    assert store.get(DEFAULT_WORKSPACE_ID) is not None


def test_delete_workspace(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("待删")
    assert store.delete(workspace.id) is True
    assert store.get(workspace.id) is None
    assert store.delete(workspace.id) is False


def test_persists_across_instances(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    created = store.create("Work Notes")
    store.rename(DEFAULT_WORKSPACE_ID, "常用")

    reloaded = WorkspaceStore(tmp_path)
    names = {workspace.id: workspace.name for workspace in reloaded.list()}
    assert names == {DEFAULT_WORKSPACE_ID: "常用", created.id: "Work Notes"}


def test_ensure_default_restores_missing_default(tmp_path: Path) -> None:
    """workspaces.json 被手工改坏（缺少默认分区）时，ensure_default 兜底补回。"""

    (tmp_path / "workspaces.json").write_text("[]", encoding="utf-8")
    store = WorkspaceStore(tmp_path)
    workspace = store.ensure_default()
    assert workspace.id == DEFAULT_WORKSPACE_ID
    assert store.get(DEFAULT_WORKSPACE_ID) is not None


def test_pinned_workspaces_come_first(tmp_path: Path) -> None:
    """置顶项排在最前；取消置顶后回到原来的位次。"""

    store = WorkspaceStore(tmp_path)
    first = store.create("A")
    second = store.create("B")
    assert [item.id for item in store.list()] == [DEFAULT_WORKSPACE_ID, first.id, second.id]

    pinned = store.set_pinned(second.id, True)
    assert pinned is not None
    assert pinned.pinned is True
    assert [item.id for item in store.list()] == [second.id, DEFAULT_WORKSPACE_ID, first.id]

    store.set_pinned(second.id, False)
    assert [item.id for item in store.list()] == [DEFAULT_WORKSPACE_ID, first.id, second.id]
    assert store.set_pinned("nope", True) is None


def test_reorder_writes_manual_order_and_persists(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    first = store.create("A")
    second = store.create("B")

    ordered = store.reorder([second.id, DEFAULT_WORKSPACE_ID, first.id])
    assert [item.id for item in ordered] == [second.id, DEFAULT_WORKSPACE_ID, first.id]

    reloaded = WorkspaceStore(tmp_path)
    assert [item.id for item in reloaded.list()] == [second.id, DEFAULT_WORKSPACE_ID, first.id]


def test_pin_wins_over_manual_order(tmp_path: Path) -> None:
    """置顶优先于手工位次：置顶项永远在最前。"""

    store = WorkspaceStore(tmp_path)
    first = store.create("A")
    second = store.create("B")
    store.reorder([first.id, DEFAULT_WORKSPACE_ID, second.id])

    store.set_pinned(second.id, True)
    assert [item.id for item in store.list()] == [second.id, first.id, DEFAULT_WORKSPACE_ID]


def test_reorder_ignores_unknown_ids(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    created = store.create("A")
    store.reorder(["nope", created.id, "also-nope"])
    assert store.get(created.id).order_index == 1  # type: ignore[union-attr]


def test_new_workspace_goes_to_the_end(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    first = store.create("A")
    store.reorder([first.id, DEFAULT_WORKSPACE_ID])
    later = store.create("B")
    assert [item.id for item in store.list()][-1] == later.id
