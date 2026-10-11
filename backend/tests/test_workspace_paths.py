"""共享知识库的路径解析与 ``storage_key``（D-KB）。

覆盖：
- ``storage_key`` 服务端生成、稳定、唯一；历史数据确定性回填（私有目录不得相撞）；
- **知识路径全局共享**：任何工作区解析出的 ``documents_dir`` / ``state_db`` /
  ``image_state_db`` / ``chroma_dir`` 逐字段相同，且都在 ``data_dir`` 根；
- **私有路径按区**：``memory.json`` / ``checkpoints.db`` / ``artifacts`` / jieba 缓存
  落在 ``data_dir/workspaces/<storage_key>/``；
- ``storage.json`` 只表达「工作区记录迁移状态」（D25/D29），**改不动知识库布局**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.workspace_context import (
    SHARED_KNOWLEDGE_DIRS,
    LayoutMode,
    WorkspaceContextFactory,
    WorkspaceNotFoundError,
    commit_storage_version,
    committed_storage_version,
    ensure_layout_dirs,
    read_storage_marker,
)
from app.core.workspace_store import (
    DEFAULT_STORAGE_KEY,
    DEFAULT_WORKSPACE_ID,
    WORKSPACE_LAYOUT_VERSION,
    WORKSPACE_SUBDIRS,
    WorkspaceStore,
)


def _data_dir_with_knownledge(tmp_path: Path) -> Path:
    """造一个「根下已有知识数据」的数据目录（D-KB 里的常规形态）。"""

    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "chroma.sqlite3").write_bytes(b"chroma")
    (data_dir / "state.db").write_bytes(b"state")
    (data_dir / "documents").mkdir(exist_ok=True)
    (data_dir / "documents" / "a.md").write_text("既有资料", encoding="utf-8")
    return data_dir


def test_storage_key_is_generated_stable_and_unique(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    default = store.get(DEFAULT_WORKSPACE_ID)
    assert default is not None
    assert default.storage_key == DEFAULT_STORAGE_KEY

    first = store.create("研究")
    second = store.create("工作")
    assert first.storage_key and second.storage_key
    # 不同工作区的**私有**目录绝不能相撞（知识目录是共享的，不在此列）
    assert len({first.storage_key, second.storage_key, default.storage_key}) == 3

    # 重新加载后 key 不变（稳定 = 私有目录不会漂移）
    reloaded = WorkspaceStore(tmp_path)
    assert reloaded.get(first.id).storage_key == first.storage_key  # type: ignore[union-attr]
    assert reloaded.get_by_storage_key(first.storage_key).id == first.id  # type: ignore[union-attr]


def test_legacy_rows_without_storage_key_get_deterministic_backfill(tmp_path: Path) -> None:
    """历史 ``workspaces.json`` 没有 ``storage_key`` 时按 id 确定性派生并落盘。"""

    (tmp_path / "workspaces.json").write_text(
        '[{"id": "default", "name": "默认", "is_default": true},'
        ' {"id": "research", "name": "研究"}]',
        encoding="utf-8",
    )
    first = WorkspaceStore(tmp_path)
    keys = {item.id: item.storage_key for item in first.list()}
    assert keys[DEFAULT_WORKSPACE_ID] == DEFAULT_STORAGE_KEY
    assert keys["research"]

    second = WorkspaceStore(tmp_path)
    assert {item.id: item.storage_key for item in second.list()} == keys


def test_colliding_storage_keys_are_repaired(tmp_path: Path) -> None:
    """手工把两条记录写成同一个 key 时重新派生，绝不让两区共用私有目录。"""

    (tmp_path / "workspaces.json").write_text(
        '[{"id": "a", "name": "A", "storage_key": "same"},'
        ' {"id": "b", "name": "B", "storage_key": "same"}]',
        encoding="utf-8",
    )
    store = WorkspaceStore(tmp_path)
    keys = [item.storage_key for item in store.list()]
    assert len(set(keys)) == len(keys)
    assert {"a", "b"} <= {item.id for item in store.list()}


def test_paths_split_shared_knowledge_from_workspace_private(tmp_path: Path) -> None:
    """知识路径共享（所有区相同）＋私有件按区。"""

    data_dir = tmp_path / "erag-data"
    store = WorkspaceStore(data_dir)
    workspace = store.create("研究")

    root = store.ensure_dirs(data_dir, workspace.id)
    assert root == data_dir / "workspaces" / workspace.storage_key
    for relative in WORKSPACE_SUBDIRS:
        assert (root / relative).is_dir(), relative

    factory = WorkspaceContextFactory(data_dir, store)
    paths = factory.paths_for(workspace.id)
    assert paths.layout is LayoutMode.SHARED
    assert paths.root_dir == root

    # 知识：全部在 data_dir 根（共享）
    assert paths.documents_dir == data_dir / "documents"
    assert paths.state_db == data_dir / "state.db"
    assert paths.image_state_db == data_dir / "image_state.db"
    assert paths.chroma_dir == data_dir
    # 私有：全部在本区根下
    assert paths.memory_json == root / "memory.json"
    assert paths.checkpoints_db == root / "checkpoints.db"
    assert paths.artifacts_dir == root / "artifacts"
    assert paths.jieba_cache_dir == root / "cache" / "jieba"

    # 建目录：共享知识目录在 data_dir 根、私有目录在本区根（幂等）
    ensure_layout_dirs(paths)
    assert (data_dir / "documents").is_dir()
    assert (data_dir / "cache" / "jieba").is_dir()
    assert (root / "artifacts").is_dir()
    assert set(SHARED_KNOWLEDGE_DIRS) == {"documents", "cache/jieba"}


def test_every_workspace_resolves_the_same_shared_knowledge_paths(tmp_path: Path) -> None:
    """核心断言：三个工作区的知识路径逐字段相同，私有路径互不相同。"""

    data_dir = _data_dir_with_knownledge(tmp_path)
    store = WorkspaceStore(data_dir)
    factory = WorkspaceContextFactory(data_dir, store)
    contexts = [factory.resolve(store.create(name).id) for name in ("甲区", "乙区", "丙区")]

    for attribute in ("documents_dir", "state_db", "image_state_db", "chroma_dir"):
        values = {str(getattr(context, attribute)) for context in contexts}
        assert len(values) == 1, attribute
        assert next(iter(values)).startswith(str(data_dir)), attribute

    for attribute in ("memory_json", "checkpoints_db", "artifacts_dir"):
        values = {str(getattr(context, attribute)) for context in contexts}
        assert len(values) == len(contexts), attribute
    assert len({str(context.paths.jieba_cache_dir) for context in contexts}) == len(contexts)

    # 布局判定与工作区无关，且**不再**受 storage.json 影响
    assert {factory.layout_mode(context.workspace_id) for context in contexts} == {
        LayoutMode.SHARED
    }


def test_knowledge_paths_ignore_the_migration_marker(tmp_path: Path) -> None:
    """提交 ``storage_schema_version`` 前后，知识路径**完全不变**（D-KB）。"""

    data_dir = _data_dir_with_knownledge(tmp_path)
    store = WorkspaceStore(data_dir)
    factory = WorkspaceContextFactory(data_dir, store)
    workspace = store.create("研究")

    before = factory.paths_for(workspace.id)
    assert committed_storage_version(data_dir) == 0
    assert before.layout is LayoutMode.SHARED
    assert before.chroma_dir == data_dir
    assert before.state_db == data_dir / "state.db"

    # 写入新目录树、提交迁移标记都不改变解析结果
    store.ensure_dirs(data_dir, workspace.id)
    commit_storage_version(
        data_dir,
        version=WORKSPACE_LAYOUT_VERSION,
        extra={"migration_version": "v1.2.0"},
    )
    assert committed_storage_version(data_dir) == WORKSPACE_LAYOUT_VERSION
    marker = read_storage_marker(data_dir)
    assert marker["layout"] == "shared"
    assert marker["migration_version"] == "v1.2.0"

    after = factory.paths_for(workspace.id)
    assert after == before
    assert factory.layout_mode(workspace.id) is LayoutMode.SHARED


def test_fresh_install_uses_shared_knowledge_paths(tmp_path: Path) -> None:
    """全新安装（无任何既有产物）同样用共享知识路径。"""

    data_dir = tmp_path / "fresh"
    store = WorkspaceStore(data_dir)
    assert (data_dir / "workspaces.json").is_file()

    factory = WorkspaceContextFactory(data_dir, store)
    paths = factory.paths_for(DEFAULT_WORKSPACE_ID)
    assert factory.layout_mode(DEFAULT_WORKSPACE_ID) is LayoutMode.SHARED
    assert paths.chroma_dir == data_dir
    assert paths.documents_dir == data_dir / "documents"
    assert paths.memory_json == data_dir / "workspaces" / DEFAULT_STORAGE_KEY / "memory.json"


def test_loading_a_legacy_file_does_not_rewrite_it(tmp_path: Path) -> None:
    """构造 ``WorkspaceStore`` 不得改写用户文件。

    真实数据上 ``import app.api.deps`` 会在模块级构造 ``WorkspaceStore(settings.data_dir)``；
    若回填顺手落盘，就变成「仅仅 import 一次改写了用户的 workspaces.json」。
    确定性派生让回填无需落盘即可跨加载稳定。
    """

    path = tmp_path / "workspaces.json"
    original = (
        '[{"id": "default", "name": "默认", "is_default": true},\n'
        ' {"id": "research", "name": "研究"}]\n'
    )
    path.write_text(original, encoding="utf-8")

    store = WorkspaceStore(tmp_path)
    assert path.read_text(encoding="utf-8") == original
    assert store.get(DEFAULT_WORKSPACE_ID).storage_key == DEFAULT_STORAGE_KEY  # type: ignore[union-attr]
    first = store.get("research").storage_key  # type: ignore[union-attr]
    assert first
    # 不落盘也必须每次加载得到同一个 key
    assert WorkspaceStore(tmp_path).get("research").storage_key == first  # type: ignore[union-attr]


def test_unknown_workspace_and_storage_key_have_no_paths(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    factory = WorkspaceContextFactory(tmp_path, store)
    assert store.workspace_root(tmp_path, "nope") is None
    assert store.storage_key("nope") is None
    assert store.get_by_storage_key("nope") is None
    with pytest.raises(WorkspaceNotFoundError):
        factory.paths_for("nope")
