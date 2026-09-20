"""工作区写路径权威源：D25 过渡期双源 → D29 收口（app.db 唯一权威）。

覆盖（对应 t24 验收）：

① 迁移后（``storage_schema_version>=2``）：``POST`` / ``PUT`` / ``DELETE`` 工作区端点
   **落 app.db**，且**不再调用** JSON 权威写路径（``WorkspaceStore.create`` / ``rename``
   / ``delete`` 被换成「一调用即失败」，端点仍成功 = 非永真证据）；JSON 存档退化为
   app.db 行的**单向镜像**（D19/A3），D25 的唯一读取入口
   （``WorkspaceContextFactory`` → ``WorkspaceStore``）读到同一条记录。
② 迁移前（``storage_schema_version<2``）：JSON 仍是**活数据源**，app.db 不被写入。
③ 没有模块绕过 ``AppStore`` 直写 ``app.db.workspaces`` 表；``app/agent/app_store.py``
   不反向依赖 ``app.core``（D19 单向适配器）。
④ 排序语义与 t12 修复保持一致（同 tick 打平仍「最近更新在前」），工作区拖拽/置顶语义
   不变（本次只换权威源，不改规则）；**5/5** 写端点（含 ``pin`` / ``order``）迁移后落 app.db。
⑤ schema v1 → v2 就地升级：老库自动补列、**幂等**（连续两次 ``ensure_schema`` 无副作用）、
   升级后 pin/order 可写可读、既有行行为不变。
⑥ 存档**可再生**：迁移后 app.db 为权威，``workspaces.json`` 被删/损坏时读入口仍不丢工作区
   （解析前按需从 app.db 权威行水合）。
⑦ ``knowledge_revision_bumped``（PRD 5.3.4 缓存键 / 8.2 资料状态检查）迁移后落 app.db，
   并让读入口的版本与检索缓存键随之变化。

测试只用 tmp 工作区与内存 app.db（水合/schema 升级另用 tmp 文件库），全离线。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent.app_store import AppStore, upgrade_schema
from app.agent.contracts import APP_DB_SCHEMA_VERSION
from app.api import deps as api_deps
from app.core.config import settings
from app.core.conversation_store import ConversationMessage, ConversationStore
from app.core.workspace_context import STORAGE_MARKER_FILENAME, WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore, workspace_order_key
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）

#: conftest 把工作区存档隔离到 ``tmp_path/conv-data``（与路由 override 同一目录）。
ARCHIVE_DIR_NAME = "conv-data"
#: 绕过 AppStore 直写 app.db ``workspaces`` 表的源码特征。
_TABLE_WRITE_RE = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+workspaces\b", re.IGNORECASE
)
_CORE_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+app\.core\b", re.MULTILINE)


@pytest.fixture
async def client(app_store, _isolate_local_deps: None) -> AsyncIterator[httpx.AsyncClient]:
    """注册 app.db 单例的 API 客户端（真实路由 + 真实 aiosqlite 内存库）。"""

    api_deps.register_app_store(app_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as active:
        yield active
    api_deps.register_app_store(None)


def _archive(tmp_path: Path) -> WorkspaceStore:
    """重新加载 JSON 存档（模拟下一次请求/重启读取的就是盘上那份）。"""

    return WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME)


def _view(tmp_path: Path, workspace_id: str) -> Any:
    """D25 唯一读取入口：工厂 → 视图。"""

    return WorkspaceContextFactory(
        settings.data_dir, _archive(tmp_path)
    ).resolve(workspace_id).view


def _ids(payload: list[dict[str, Any]]) -> list[str]:
    return [item["id"] for item in payload]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


# =====================================================================
# ① 迁移后：写端点落 app.db，JSON 只做单向镜像
# =====================================================================


async def test_migrated_write_endpoints_land_in_app_db(
    client: httpx.AsyncClient, app_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移后 create / rename / delete 全部落 app.db；JSON 权威写路径零调用。"""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("迁移后不得再走 JSON 权威写路径（D29）")

    # 非永真证明：只要端点碰了这三条 JSON 权威写路径，用例立刻失败。
    monkeypatch.setattr(WorkspaceStore, "create", _forbidden)
    monkeypatch.setattr(WorkspaceStore, "rename", _forbidden)
    monkeypatch.setattr(WorkspaceStore, "delete", _forbidden)

    created = await client.post("/api/workspaces", json={"name": "工作笔记"})
    assert created.status_code == 201, created.text
    workspace_id = created.json()["id"]
    assert created.json()["name"] == "工作笔记"
    assert created.json()["status"] == "active"

    record = await app_store.get_workspace(workspace_id)
    assert record is not None, "新建工作区必须落 app.db（唯一权威）"
    assert record.name == "工作笔记"
    assert record.storage_key

    # 存档 = app.db 行的单向镜像（D19/A3）：字段逐项一致，不是另一份「活源」
    mirrored = _archive(tmp_path).get(workspace_id)
    assert mirrored is not None
    assert (mirrored.name, mirrored.storage_key, mirrored.status.value) == (
        record.name,
        record.storage_key,
        record.status.value,
    )
    # D25 唯一读取入口读到同一条记录
    view = _view(tmp_path, workspace_id)
    assert (view.workspace_id, view.name, view.storage_key) == (
        workspace_id,
        record.name,
        record.storage_key,
    )

    renamed = await client.put(f"/api/workspaces/{workspace_id}", json={"name": "长期项目"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "长期项目"
    assert (await app_store.get_workspace(workspace_id)).name == "长期项目"
    assert _archive(tmp_path).get(workspace_id).name == "长期项目"
    assert _view(tmp_path, workspace_id).name == "长期项目"

    deleted = await client.delete(f"/api/workspaces/{workspace_id}")
    assert deleted.status_code == 200, deleted.text
    archived_record = await app_store.get_workspace(workspace_id)
    assert archived_record is not None, "PRD 5.4：删除是可恢复归档，必须保留归属"
    assert archived_record.status.value == "archived"
    assert _archive(tmp_path).get(workspace_id).status.value == "archived"
    # 归档区不再出现在侧边栏列表
    assert workspace_id not in _ids((await client.get("/api/workspaces")).json())


async def test_migrated_delete_archives_conversations_and_keeps_default(
    client: httpx.AsyncClient, app_store
) -> None:
    """归档语义：本区 app.db 会话一并归档；默认工作区仍不可删。"""

    created = await client.post("/api/workspaces", json={"name": "临时"})
    workspace_id = created.json()["id"]
    await app_store.create_conversation("conv-d29", workspace_id)
    # 迁移安装的 app.db 必有默认区行；写端点也会幂等兜底补上（不会重复建）
    assert (await app_store.get_workspace("default")) is not None
    assert len(await app_store.list_workspaces()) == 2

    assert (await client.delete("/api/workspaces/default")).status_code == 400
    assert (await app_store.get_workspace("default")).status.value == "active"

    response = await client.delete(f"/api/workspaces/{workspace_id}")
    assert response.status_code == 200, response.text
    assert (await app_store.get_workspace(workspace_id)).status.value == "archived"
    conversation = await app_store.get_conversation("conv-d29")
    assert conversation.status.value == "archived"
    # 不存在的 id 仍 404（与过渡期一致）
    assert (await client.delete("/api/workspaces/nope")).status_code == 404


async def test_migrated_rename_and_pin_unknown_workspace_are_404(
    client: httpx.AsyncClient
) -> None:
    assert (await client.put("/api/workspaces/nope", json={"name": "x"})).status_code == 404
    assert (
        await client.put("/api/workspaces/nope/pin", json={"pinned": True})
    ).status_code == 404


# =====================================================================
# ② 迁移前：JSON 仍是活数据源（D25 过渡期双源）
# =====================================================================


async def test_unmigrated_writes_stay_in_json_archive(
    client: httpx.AsyncClient, app_store, tmp_path: Path
) -> None:
    """未提交迁移标记（storage_schema_version<2）时不得写 app.db。"""

    (settings.data_dir / STORAGE_MARKER_FILENAME).unlink()

    created = await client.post("/api/workspaces", json={"name": "过渡期"})
    assert created.status_code == 201, created.text
    workspace_id = created.json()["id"]

    assert await app_store.get_workspace(workspace_id) is None, "迁移前 app.db 不是权威源"
    assert _archive(tmp_path).get(workspace_id) is not None
    assert _ids((await client.get("/api/workspaces")).json()) == ["default", workspace_id]

    renamed = await client.put(f"/api/workspaces/{workspace_id}", json={"name": "改名"})
    assert renamed.status_code == 200
    assert _archive(tmp_path).get(workspace_id).name == "改名"
    assert await app_store.get_workspace(workspace_id) is None


# =====================================================================
# ③ 直写守卫：只有 AppStore 能写 workspaces 表；app_store 不依赖 core
# =====================================================================


def test_only_app_store_writes_the_workspaces_table() -> None:
    """D19/A3：任何模块都不得绕过 ``AppStore`` 直写 app.db 的 workspaces 表。"""

    # 守卫非永真：合成片段必须命中同一正则
    assert _TABLE_WRITE_RE.search("conn.execute('UPDATE workspaces SET name = ?')")
    assert _TABLE_WRITE_RE.search("INSERT INTO workspaces (id) VALUES (?)")

    app_root = _backend_root() / "app"
    offenders = [
        str(path.relative_to(app_root))
        for path in app_root.rglob("*.py")
        if path.name != "app_store.py"
        and _TABLE_WRITE_RE.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"绕过 AppStore 直写 workspaces 表：{offenders}"


def test_app_store_does_not_depend_on_app_core() -> None:
    """D19：依赖方向恒为 ``core → app_store``，``app_store`` 不反向 import ``app.core``。"""

    assert _CORE_IMPORT_RE.search("from app.core.workspace_store import WorkspaceStore")

    source = (_backend_root() / "app/agent/app_store.py").read_text(encoding="utf-8")
    assert _CORE_IMPORT_RE.search(source) is None


# =====================================================================
# ④ 排序语义：与 t12 修复一致，工作区拖拽/置顶规则不变（且落 app.db 权威列）
# =====================================================================


async def test_workspace_order_and_pin_semantics_unchanged(
    client: httpx.AsyncClient, app_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移后拖拽/置顶语义不变，落 app.db 的 ``pinned`` / ``order_index`` 权威列。"""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("迁移后不得再走 JSON 权威写路径（D29）")

    monkeypatch.setattr(WorkspaceStore, "set_pinned", _forbidden)
    monkeypatch.setattr(WorkspaceStore, "reorder", _forbidden)

    first = (await client.post("/api/workspaces", json={"name": "A"})).json()["id"]
    second = (await client.post("/api/workspaces", json={"name": "B"})).json()["id"]

    reordered = await client.put(
        "/api/workspaces/order", json={"ids": [second, "default", first]}
    )
    assert reordered.status_code == 200, reordered.text
    assert _ids(reordered.json()) == [second, "default", first]
    assert _ids((await client.get("/api/workspaces")).json()) == [second, "default", first]
    # 位次是 app.db 权威列，不是只在存档里
    assert (await app_store.get_workspace(second)).order_index == 0.0
    assert (await app_store.get_workspace("default")).order_index == 1.0
    assert (await app_store.get_workspace(first)).order_index == 2.0

    pinned = await client.put(f"/api/workspaces/{first}/pin", json={"pinned": True})
    assert pinned.status_code == 200
    assert pinned.json()["pinned"] is True
    assert (await app_store.get_workspace(first)).pinned is True
    assert _ids((await client.get("/api/workspaces")).json()) == [first, second, "default"]
    # 唯一排序规则（置顶 → 位次 → 创建时间）对 app.db 行同样成立
    records = await app_store.list_workspaces()
    assert [
        record.id for record in sorted(records, key=workspace_order_key)
    ][:3] == [first, second, "default"]

    await client.put(f"/api/workspaces/{first}/pin", json={"pinned": False})
    assert _ids((await client.get("/api/workspaces")).json()) == [second, "default", first]
    assert (await app_store.get_workspace(first)).pinned is False

    # 之后再走 app.db 权威写入（新建）：已拖拽的位次必须保留，新工作区排到末尾
    later = (await client.post("/api/workspaces", json={"name": "C"})).json()["id"]
    assert _ids((await client.get("/api/workspaces")).json()) == [
        second,
        "default",
        first,
        later,
    ]
    assert (await app_store.get_workspace(second)).order_index == 0.0
    assert _archive(tmp_path).get(second).order_index == 0


async def test_migrated_pin_unknown_workspace_is_404(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移后 pin 不存在的 id：先问 app.db 权威源（不得落到存档的写路径）。"""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("迁移后不得再走 JSON 权威写路径（D29）")

    monkeypatch.setattr(WorkspaceStore, "set_pinned", _forbidden)
    response = await client.put("/api/workspaces/nope/pin", json={"pinned": True})
    assert response.status_code == 404


def test_sidebar_recency_tie_break_matches_t12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """t12 修复未被触碰：时间戳打平时侧边栏仍「最近更新在前」。"""

    import app.core.conversation_store as conversation_store_module

    frozen = "2026-09-19T00:00:00.000000+00:00"
    monkeypatch.setattr(conversation_store_module, "_now", lambda: frozen)

    store = ConversationStore(tmp_path / "conv")
    created: list[str] = []
    for text in ("第一条", "第二条", "第三条"):
        conv = store.create("default")
        store.append_messages(conv.id, [ConversationMessage(role="user", content=text)])
        created.append(conv.id)

    assert _ids(store.list_summaries("default")) == list(reversed(created))


# =====================================================================
# ⑤ schema v1 → v2 就地升级（幂等、行为不变、升级后 pin/order 可写可读）
# =====================================================================

_V1_WORKSPACES_DDL = """
CREATE TABLE workspaces (
    id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    default_permission_mode TEXT NOT NULL DEFAULT 'ask',
    default_reasoning_level TEXT NOT NULL DEFAULT 'high',
    knowledge_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _make_v1_database(path: Path) -> None:
    """造一个真实的 v1 老库（``user_version=1``，没有 pinned / order_index 列）。"""

    connection = sqlite3.connect(path)
    try:
        connection.execute(_V1_WORKSPACES_DDL)
        connection.execute(
            "INSERT INTO workspaces (id, storage_key, name, status,"
            " default_permission_mode, default_reasoning_level, knowledge_revision,"
            " created_at, updated_at) VALUES (?, ?, ?, 'active', 'ask', 'high', 3, ?, ?)",
            ("legacy", "legacy-key", "老库工作区", "2026-09-19T00:00:00+00:00", "2026-09-19T00:00:00+00:00"),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()


async def test_schema_v1_upgrades_in_place_idempotently(tmp_path: Path) -> None:
    """老库补列：版本 1→当前（v2 加列 + v3 重建任务表）、既有行行为不变、幂等。"""

    path = tmp_path / "legacy-app.db"
    _make_v1_database(path)

    store = await AppStore.open(path)
    try:
        assert await store.user_version() == APP_DB_SCHEMA_VERSION
        assert APP_DB_SCHEMA_VERSION >= 2
        columns = {row["name"] for row in await store._fetch_all("PRAGMA table_info(workspaces)")}
        assert {"pinned", "order_index"} <= columns

        # 既有行：读得到、默认值等价于「未置顶、无手工位次」，其它字段一字未变
        legacy = await store.get_workspace("legacy")
        assert legacy is not None
        assert legacy.name == "老库工作区"
        assert legacy.knowledge_revision == 3
        assert legacy.pinned is False
        assert legacy.order_index is None
        assert legacy.created_at == "2026-09-19T00:00:00+00:00"

        # 升级后 pin/order 可写可读
        updated = await store.update_workspace("legacy", pinned=True)
        assert updated.pinned is True
        reordered = await store.reorder_workspaces(["legacy"])
        assert [record.id for record in reordered] == ["legacy"]
        assert (await store.get_workspace("legacy")).order_index == 0.0

        # 幂等：连续两次 ensure_schema（+ 显式 upgrade_schema）都没有副作用
        await store.ensure_schema()
        await store.ensure_schema()
        await upgrade_schema(store.connection)
        assert await store.user_version() == APP_DB_SCHEMA_VERSION
        again = await store.get_workspace("legacy")
        assert (again.pinned, again.order_index, again.knowledge_revision) == (True, 0.0, 3)
        assert len(await store.list_workspaces()) == 1
    finally:
        await store.close()


async def test_v1_archive_row_converts_with_defaults() -> None:
    """v1 老库的行（缺 pinned/order_index）转换成存档行时取默认值，不抛错。"""

    from app.core.workspace_store import workspace_from_record

    legacy_row = {
        "id": "legacy",
        "storage_key": "legacy-key",
        "name": "老库工作区",
        "status": "active",
        "default_permission_mode": "ask",
        "default_reasoning_level": "high",
        "knowledge_revision": 3,
        "created_at": "2026-09-19T00:00:00+00:00",
        "updated_at": "2026-09-19T00:00:00+00:00",
    }
    workspace = workspace_from_record(legacy_row)
    assert (workspace.pinned, workspace.order_index) == (False, None)
    assert workspace.knowledge_revision == 3
    assert workspace_order_key(workspace) == (1, float("inf"), "2026-09-19T00:00:00+00:00")


# =====================================================================
# ⑥ 存档可再生：JSON 被删/损坏时读入口仍从 app.db 拿得到工作区
# =====================================================================


async def _file_backed_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """注册**磁盘** app.db（``data_dir/app.db``，与生产同路径规则）的 API 客户端。"""

    monkeypatch.setattr(api_deps, "_app_store", None)
    store = await AppStore.open(settings.data_dir / "app.db")
    api_deps.register_app_store(store)
    return store


async def test_archive_is_regenerable_from_app_db_when_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移后 app.db 是权威：存档被删/损坏时，解析前按需水合，读入口不丢工作区。"""

    store = await _file_backed_client(tmp_path, monkeypatch)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/api/workspaces", json={"name": "权威区"})
            assert created.status_code == 201, created.text
            workspace_id = created.json()["id"]
            await client.put(f"/api/workspaces/{workspace_id}/pin", json={"pinned": True})

            archive_path = tmp_path / ARCHIVE_DIR_NAME / "workspaces.json"
            assert archive_path.is_file()

            # 存档被删：新实例（模拟重启）解析前必须从 app.db 水合
            archive_path.unlink()
            rebuilt = WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME)
            assert rebuilt.get(workspace_id) is None
            factory = WorkspaceContextFactory(
                settings.data_dir,
                rebuilt,
                app_db_reader=api_deps.read_app_db_workspaces,
            )
            view = factory.resolve(workspace_id).view
            assert (view.workspace_id, view.name) == (workspace_id, "权威区")
            assert rebuilt.get(workspace_id).pinned is True  # 权威值（含置顶）都回来了

            # 存档被损坏成非法 JSON：同样自愈
            archive_path.write_text("{ 这不是 JSON", encoding="utf-8")
            corrupted = WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME)
            assert corrupted.get(workspace_id) is None
            factory = WorkspaceContextFactory(
                settings.data_dir,
                corrupted,
                app_db_reader=api_deps.read_app_db_workspaces,
            )
            assert factory.resolve(workspace_id).view.name == "权威区"

            # 迁移前（未提交标记）不做水合：JSON 就是活数据源
            (settings.data_dir / STORAGE_MARKER_FILENAME).unlink()
            archive_path.write_text(json.dumps([{"id": "json-only", "name": "JSON 活源"}]), encoding="utf-8")
            legacy_store = WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME)
            legacy_factory = WorkspaceContextFactory(
                settings.data_dir,
                legacy_store,
                app_db_reader=api_deps.read_app_db_workspaces,
            )
            assert legacy_factory.resolve("json-only").view.name == "JSON 活源"
            assert legacy_store.get(workspace_id) is None  # 未被 app.db 覆盖
    finally:
        api_deps.register_app_store(None)
        await store.close()


async def test_read_entry_uses_authoritative_pin_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """水合把 app.db 的 ``pinned`` / ``order_index`` 权威列带进读入口（存档过期也能纠正）。"""

    store = await _file_backed_client(tmp_path, monkeypatch)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = (await client.post("/api/workspaces", json={"name": "A"})).json()["id"]
            second = (await client.post("/api/workspaces", json={"name": "B"})).json()["id"]
            await client.put("/api/workspaces/order", json={"ids": [second, first, "default"]})
            await client.put(f"/api/workspaces/{first}/pin", json={"pinned": True})

            # 人为把存档改回「未置顶、无位次」：读入口必须按 app.db 纠正回来
            archive_path = tmp_path / ARCHIVE_DIR_NAME / "workspaces.json"
            payload = json.loads(archive_path.read_text(encoding="utf-8"))
            for item in payload:
                item["pinned"] = False
                item["order_index"] = None
            archive_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            stale = WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME)
            assert stale.get(first).pinned is False
            factory = WorkspaceContextFactory(
                settings.data_dir,
                stale,
                app_db_reader=api_deps.read_app_db_workspaces,
            )
            factory.resolve(first)  # 触发水合
            assert stale.get(first).pinned is True
            assert stale.get(second).order_index == 0.0
            assert _ids([{"id": item.id} for item in stale.list()]) == [first, second, "default"]
    finally:
        api_deps.register_app_store(None)
        await store.close()


# =====================================================================
# ⑦ knowledge_revision：迁移后落 app.db，并且读入口与缓存键随之变化
# =====================================================================


async def test_knowledge_revision_bump_lands_in_app_db(
    client: httpx.AsyncClient, app_store, tmp_path: Path
) -> None:
    """``knowledge_revision_bumped`` 迁移后写 app.db（PRD 5.3.4 缓存键 / 8.2 资料状态）。"""

    created = await client.post("/api/workspaces", json={"name": "知识区"})
    workspace_id = created.json()["id"]
    before = await app_store.get_workspace(workspace_id)
    assert before.knowledge_revision == 0

    def _resolve() -> Any:
        return WorkspaceContextFactory(
            settings.data_dir,
            _archive(tmp_path),
            app_db_reader=api_deps.read_app_db_workspaces,
        ).resolve(workspace_id)

    stale = _resolve()
    assert stale.view.knowledge_revision == 0
    stale_key = stale.index_cache_key(index_fingerprint="f", generator_version="bm25-v1")

    revision = await api_deps.knowledge_revision_bumped(workspace_id)
    assert revision == 1

    after = await app_store.get_workspace(workspace_id)
    assert after.knowledge_revision == 1, "权威源（app.db）必须递增"
    # 读入口（工厂视图）与存档同步，检索缓存键随之变化（否则索引缓存不失效）
    fresh = _resolve()
    assert fresh.view.knowledge_revision == 1
    assert (
        fresh.index_cache_key(index_fingerprint="f", generator_version="bm25-v1") != stale_key
    )

    # 未知工作区返回 None（与过渡期一致）
    assert await api_deps.knowledge_revision_bumped("nope") is None


async def test_delete_knowledge_endpoint_bumps_app_db_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实调用方 ``DELETE /api/knowledge``（async）把版本递增落到 app.db（要求 ① 收口）。"""

    from app.ingestion.image_state_store import ImageStateStore
    from app.ingestion.state_store import DocumentStateStore
    from tests.fakes import FakeVectorStore

    store = await _file_backed_client(tmp_path, monkeypatch)
    state = DocumentStateStore(tmp_path / "state.db")
    images = ImageStateStore(tmp_path / "images.db")
    app.dependency_overrides[api_deps.get_workspace_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[api_deps.get_workspace_state_store] = lambda: state
    app.dependency_overrides[api_deps.get_workspace_image_state_store] = lambda: images
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/api/workspaces", json={"name": "清空区"})
            workspace_id = created.json()["id"]
            assert (await store.get_workspace(workspace_id)).knowledge_revision == 0

            response = await client.delete(
                "/api/knowledge", params={"workspace_id": workspace_id}
            )
            assert response.status_code == 200, response.text

        assert (await store.get_workspace(workspace_id)).knowledge_revision == 1
        # 读入口（工厂视图）同步：缓存键据此变化
        view = WorkspaceContextFactory(
            settings.data_dir,
            WorkspaceStore(tmp_path / ARCHIVE_DIR_NAME),
            app_db_reader=api_deps.read_app_db_workspaces,
        ).resolve(workspace_id).view
        assert view.knowledge_revision == 1
    finally:
        app.dependency_overrides.pop(api_deps.get_workspace_vector_store, None)
        app.dependency_overrides.pop(api_deps.get_workspace_state_store, None)
        app.dependency_overrides.pop(api_deps.get_workspace_image_state_store, None)
        state.close()
        images.close()
        api_deps.register_app_store(None)
        await store.close()
