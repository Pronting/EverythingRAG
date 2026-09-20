"""工作区默认权限 / 思考强度写端点测试（PRD 1.3「工作区默认值」+ 14.1「设置页」）。

两条权威路径都要测：

- **迁移后**（``storage_schema_version >= 2`` + app.db 已注册）：``app.db`` 是权威源，
  端点走 ``AppStore.update_workspace(default_permission_mode=…, default_reasoning_level=…)``
  并单向镜像回存档；
- **迁移前**（未提交 storage 版本 / app.db 未注册）：存档 ``workspaces.json`` 仍是活数据源，
  端点保持既有 JSON 语义。

非法取值一律 422（PRD 12.5：结构化内容不合法就地失败）；至少要给一个默认值，否则 400。
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.agent.app_store import AppStore
from app.agent.contracts import PermissionMode, ReasoningLevel
from app.api import deps as api_deps
from app.core.config import settings
from app.core.workspace_context import WORKSPACE_LAYOUT_VERSION, commit_storage_version
from app.core.workspace_store import WorkspaceStore
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（内存 app.db）


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _route_store() -> WorkspaceStore:
    """路由**实际使用**的存档实例（conftest 已隔离到 tmp）。

    不能自己 new 一个指向同目录的实例：内存副本会读到陈旧数据，断言会假失败。
    """

    from app.api.routes import conversations as conversations_mod

    return conversations_mod.get_workspace_store()


async def test_defaults_written_to_app_db_after_migration(
    app_store: AppStore, tmp_path: Path
) -> None:
    """迁移后：app.db 是权威源，写端点落库并可读回。"""

    commit_storage_version(settings.data_dir)  # >= WORKSPACE_LAYOUT_VERSION
    assert WORKSPACE_LAYOUT_VERSION >= 2
    await app_store.create_workspace("ws-1", "sk-ws-1", name="测试工作区")
    api_deps.register_app_store(app_store)
    _route_store().ensure_default()
    try:
        async with _client() as client:
            response = await client.put(
                "/api/workspaces/ws-1/defaults",
                json={"permission_mode": "auto", "reasoning_level": "low"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "ws-1"
        record = await app_store.get_workspace("ws-1")
        assert record is not None
        assert record.default_permission_mode is PermissionMode.AUTO
        assert record.default_reasoning_level is ReasoningLevel.LOW
        # 单向镜像：存档**文件**也能读到同一结果（D25 的唯一读入口）
        # 另起实例读盘，避免被另一种内存副本误导。
        mirrored = WorkspaceStore(_route_store().path.parent).get("ws-1")
        assert mirrored is not None
        assert mirrored.default_permission_mode is PermissionMode.AUTO
        assert mirrored.default_reasoning_level is ReasoningLevel.LOW
    finally:
        api_deps.register_app_store(None)


async def test_defaults_partial_update_keeps_other_value(
    app_store: AppStore, tmp_path: Path
) -> None:
    """只改一项时另一项保持现值（不传 ≠ 重置为默认）。"""

    commit_storage_version(settings.data_dir)
    await app_store.create_workspace("ws-1", "sk-ws-1", name="测试工作区")
    api_deps.register_app_store(app_store)
    _route_store().ensure_default()
    try:
        async with _client() as client:
            await client.put("/api/workspaces/ws-1/defaults", json={"permission_mode": "full"})
            await client.put("/api/workspaces/ws-1/defaults", json={"reasoning_level": "max"})
        record = await app_store.get_workspace("ws-1")
        assert record is not None
        assert record.default_permission_mode is PermissionMode.FULL
        assert record.default_reasoning_level is ReasoningLevel.MAX
    finally:
        api_deps.register_app_store(None)


async def test_defaults_reject_invalid_values(
    app_store: AppStore, tmp_path: Path
) -> None:
    """非法枚举值 → 422；一个值都不给 → 400。"""

    commit_storage_version(settings.data_dir)
    await app_store.create_workspace("ws-1", "sk-ws-1", name="测试工作区")
    api_deps.register_app_store(app_store)
    _route_store().ensure_default()
    try:
        async with _client() as client:
            invalid = await client.put(
                "/api/workspaces/ws-1/defaults", json={"permission_mode": "yolo"}
            )
            empty = await client.put("/api/workspaces/ws-1/defaults", json={})
            missing = await client.put(
                "/api/workspaces/ws-missing/defaults", json={"permission_mode": "ask"}
            )
        assert invalid.status_code == 422
        assert empty.status_code == 400
        assert missing.status_code == 404
    finally:
        api_deps.register_app_store(None)


async def test_defaults_use_json_authority_before_migration(tmp_path: Path) -> None:
    """迁移前（app.db 未注册 / 未提交 storage 版本）：存档是活数据源，保持 JSON 语义。"""

    workspace = _route_store().ensure_default()
    api_deps.register_app_store(None)  # 模拟 app.db 未注册的过渡期
    async with _client() as client:
        response = await client.put(
            f"/api/workspaces/{workspace.id}/defaults",
            json={"permission_mode": "full", "reasoning_level": "off"},
        )
    assert response.status_code == 200, response.text
    # 存档文件已落盘：**另起**一个实例（不共享内存副本）读回同一结果
    reloaded = WorkspaceStore(_route_store().path.parent).get(workspace.id)
    assert reloaded is not None
    assert reloaded.default_permission_mode is PermissionMode.FULL
    assert reloaded.default_reasoning_level is ReasoningLevel.OFF
