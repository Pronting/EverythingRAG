"""工作区默认权限/强度的三层优先级测试（PRD 1.3 + 14.1 + 18.1）。T_defaults。

两层都测：

1. **纯函数**（``app/api/settings_scope.py``）：三层 + 兜底的取舍、两个轴**各自独立**解析；
2. **API 行为**：新会话继承工作区默认值；新 run 按 ``请求显式 → 会话保存 → 工作区默认 →
   18.1 兜底`` 解析；改工作区默认值**不回溯**改写既有会话；过渡期（app.db 非权威）保持
   JSON 语义；以及 run 路径对「只在存档里的会话」自愈（实测到的真实缺口）。

只用 tmp 工作区与内存 app.db；不触真实 ``~/.everything-rag``。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent.contracts import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_REASONING_LEVEL,
    PermissionMode,
    ReasoningLevel,
    WorkspaceStatus,
)
from app.api import deps as api_deps
from app.api import settings_scope as scope
from app.api.workspace_bridge import ensure_workspace_row
from app.core.conversation_store import ConversationStore
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（内存 app.db）
from tests.fakes_run import FakeServices, build_manager

WORKSPACE = "ws-defaults"
RUNS = f"/api/workspaces/{WORKSPACE}/runs"


# =====================================================================
# 1. 纯函数
# =====================================================================


class _View:
    """工作区视图替身（只带本用例关心的两个默认值字段）。"""

    def __init__(self, permission: Any = None, reasoning: Any = None) -> None:
        self.default_permission_mode = permission
        self.default_reasoning_level = reasoning


def test_conversation_settings_use_workspace_defaults_then_fallback() -> None:
    resolved = scope.resolve_conversation_settings(
        _View(PermissionMode.FULL, ReasoningLevel.MAX)
    )
    assert resolved.permission_mode is PermissionMode.FULL
    assert resolved.reasoning_level is ReasoningLevel.MAX
    assert resolved.permission_source == scope.SOURCE_WORKSPACE
    assert resolved.reasoning_source == scope.SOURCE_WORKSPACE

    # 视图缺失 / 字段为空 / 取值非法 → 一律落到 PRD 18.1 兜底
    for view in (None, _View(), _View("", ""), _View("yolo", "turbo")):
        fallback = scope.resolve_conversation_settings(view)
        assert fallback.permission_mode is DEFAULT_PERMISSION_MODE
        assert fallback.reasoning_level is DEFAULT_REASONING_LEVEL
        assert fallback.permission_source == scope.SOURCE_FALLBACK


def test_run_settings_four_layers_per_axis() -> None:
    view = _View(PermissionMode.FULL, ReasoningLevel.MAX)
    requested = scope.resolve_run_settings(
        requested_permission_mode=PermissionMode.ASK,
        requested_reasoning_level=ReasoningLevel.OFF,
        conversation_permission_mode=PermissionMode.AUTO,
        conversation_reasoning_level=ReasoningLevel.LOW,
        view=view,
    )
    assert (requested.permission_mode, requested.reasoning_level) == (
        PermissionMode.ASK,
        ReasoningLevel.OFF,
    )
    assert requested.permission_source == requested.reasoning_source == scope.SOURCE_REQUEST

    saved = scope.resolve_run_settings(
        conversation_permission_mode=PermissionMode.AUTO,
        conversation_reasoning_level=ReasoningLevel.LOW,
        view=view,
    )
    assert (saved.permission_mode, saved.reasoning_level) == (
        PermissionMode.AUTO,
        ReasoningLevel.LOW,
    )
    assert saved.permission_source == scope.SOURCE_CONVERSATION

    from_workspace = scope.resolve_run_settings(view=view)
    assert (from_workspace.permission_mode, from_workspace.reasoning_level) == (
        PermissionMode.FULL,
        ReasoningLevel.MAX,
    )
    assert from_workspace.permission_source == scope.SOURCE_WORKSPACE

    fallback = scope.resolve_run_settings()
    assert (fallback.permission_mode, fallback.reasoning_level) == (
        DEFAULT_PERMISSION_MODE,
        DEFAULT_REASONING_LEVEL,
    )


def test_run_settings_axes_are_independent() -> None:
    """只显式给一个轴时，另一个轴仍按后面几层解析（不整体跳过）。"""

    resolved = scope.resolve_run_settings(
        requested_reasoning_level=ReasoningLevel.MAX,
        conversation_permission_mode=PermissionMode.AUTO,
        conversation_reasoning_level=ReasoningLevel.LOW,
        view=_View(PermissionMode.FULL, ReasoningLevel.HIGH),
    )
    assert resolved.reasoning_level is ReasoningLevel.MAX
    assert resolved.reasoning_source == scope.SOURCE_REQUEST
    assert resolved.permission_mode is PermissionMode.AUTO
    assert resolved.permission_source == scope.SOURCE_CONVERSATION


def test_pure_resolution_module_has_no_io() -> None:
    """解析模块保持纯函数（无 IO / 无 store 依赖），便于离线复核与单测。"""

    source = Path(scope.__file__).read_text(encoding="utf-8")
    for forbidden in ("await ", "async def", "sqlite", "AppStore", "Depends("):
        assert forbidden not in source, forbidden


# =====================================================================
# 2. API 行为
# =====================================================================


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    """注册**文件型** app.db + 假 RunManager；工作区行按 `full`/`max` 建好。

    刻意用文件型：工作区默认值的读取入口是 ``WorkspaceContextFactory`` 的视图，而视图
    按需从 ``app_db_path()`` 水合 —— 内存库读不到那条路径，会把「app.db 权威 → 视图」
    的真实链路测成假的。
    """

    from app.agent.app_store import AppStore
    from app.api.routes import conversations as conversations_mod

    app_store = await AppStore.open(api_deps.app_db_path())
    await ensure_workspace_row(
        app_store,
        api_deps.get_workspace_store(),
        WORKSPACE,
        allow_archive_missing=True,
        storage_key_fallback="sk-defaults",
        name_fallback="默认值测试区",
    )
    await app_store.update_workspace(
        WORKSPACE,
        default_permission_mode=PermissionMode.FULL,
        default_reasoning_level=ReasoningLevel.MAX,
    )
    manager = build_manager(app_store, services=FakeServices(task_count=1), root=tmp_path)
    legacy = ConversationStore(tmp_path / "legacy-conv")
    app.dependency_overrides[conversations_mod.get_conversation_store] = lambda: legacy
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "store": app_store,
            "manager": manager,
            "legacy": legacy,
        }
    app.state.run_manager = None
    api_deps.register_app_store(None)
    await manager.aclose()
    await app_store.close()


async def test_new_conversation_inherits_workspace_defaults(api: dict[str, Any]) -> None:
    """① 新会话继承工作区默认 `full`/`max`（不再是 18.1 的 ask/high）。"""

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    response = await client.post("/api/conversations", params={"workspace": WORKSPACE})
    assert response.status_code == 201, response.text
    conversation_id = response.json()["id"]

    record = await store.get_conversation(conversation_id)
    assert record is not None, "迁移后新会话必须落 app.db（run 的外键前提）"
    assert record.permission_mode is PermissionMode.FULL
    assert record.reasoning_level is ReasoningLevel.MAX


async def test_new_run_uses_conversation_then_workspace(api: dict[str, Any]) -> None:
    """② 新 run 未带参数：会话保存值优先于工作区默认值。"""

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    created = await client.post("/api/conversations", params={"workspace": WORKSPACE})
    conversation_id = created.json()["id"]
    await store.update_conversation(
        conversation_id,
        permission_mode=PermissionMode.AUTO,
        reasoning_level=ReasoningLevel.LOW,
    )
    response = await client.post(
        RUNS, json={"conversation_id": conversation_id, "message": "测试问题"}
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["permission_mode"] == PermissionMode.AUTO.value
    assert body["reasoning_level"] == ReasoningLevel.LOW.value
    await api["manager"].wait(body["run_id"])


async def test_run_falls_back_to_workspace_default(api: dict[str, Any]) -> None:
    """② 续：会话行按工作区默认值创建时，run 拿到的就是该默认值。"""

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    await store.create_workspace("ws-other", "sk-other", name="另一区")
    await store.update_workspace("ws-other", default_permission_mode=PermissionMode.FULL)
    await store.create_conversation("conv-other", "ws-other")
    response = await client.post(
        "/api/workspaces/ws-other/runs",
        json={"conversation_id": "conv-other", "message": "测试问题"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["permission_mode"] == PermissionMode.FULL.value
    await api["manager"].wait(response.json()["run_id"])


async def test_explicit_request_overrides_everything(api: dict[str, Any]) -> None:
    """③ 请求显式带参数时以请求为准（会话级菜单即时改强度要生效）。"""

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    conversation_id = (
        await client.post("/api/conversations", params={"workspace": WORKSPACE})
    ).json()["id"]
    await store.update_conversation(conversation_id, permission_mode=PermissionMode.AUTO)
    response = await client.post(
        RUNS,
        json={
            "conversation_id": conversation_id,
            "message": "测试问题",
            "permission_mode": "ask",
            "reasoning_level": "off",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["permission_mode"] == "ask"
    assert body["reasoning_level"] == "off"
    await api["manager"].wait(body["run_id"])


async def test_default_change_does_not_rewrite_existing_conversation(
    api: dict[str, Any],
) -> None:
    """④ 改工作区默认值不回溯改写既有会话的保存值（PRD 1.3 / 14.1 生效边界）。"""

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    old = (await client.post("/api/conversations", params={"workspace": WORKSPACE})).json()["id"]
    assert (await store.get_conversation(old)).permission_mode is PermissionMode.FULL

    changed = await client.put(
        f"/api/workspaces/{WORKSPACE}/defaults",
        json={"permission_mode": "ask", "reasoning_level": "high"},
    )
    assert changed.status_code == 200, changed.text

    kept = await store.get_conversation(old)
    assert kept.permission_mode is PermissionMode.FULL
    assert kept.reasoning_level is ReasoningLevel.MAX

    fresh = (await client.post("/api/conversations", params={"workspace": WORKSPACE})).json()["id"]
    created = await store.get_conversation(fresh)
    assert created.permission_mode is PermissionMode.ASK
    assert created.reasoning_level is ReasoningLevel.HIGH


async def test_run_self_heals_legacy_only_conversation(api: dict[str, Any]) -> None:
    """真实缺口回归：会话只在存档里时，run 路径按存档自愈**同一 id** 的行。

    实测过：``POST /api/conversations`` 早先只写存档，随后 ``POST /runs`` 因 app.db 缺会话行
    而失败。自愈后 run 正常创建，且会话行继承工作区默认值。
    """

    client: httpx.AsyncClient = api["client"]
    store = api["store"]
    legacy: ConversationStore = api["legacy"]
    conversation = legacy.create(WORKSPACE, conversation_id="legacy-conv-1")
    assert await store.get_conversation(conversation.id) is None  # 只存在存档里

    response = await client.post(
        RUNS, json={"conversation_id": conversation.id, "message": "测试问题"}
    )
    assert response.status_code == 202, response.text
    healed = await store.get_conversation(conversation.id)
    assert healed is not None and healed.id == conversation.id
    assert healed.permission_mode is PermissionMode.FULL
    await api["manager"].wait(response.json()["run_id"])


async def test_transition_period_keeps_json_semantics(app_store, tmp_path: Path) -> None:
    """⑤ 迁移前（storage 版本未提交）：建会话保持既有 JSON 语义，不写 app.db。"""

    from app.api.routes import conversations as conversations_mod

    legacy = ConversationStore(tmp_path / "legacy-conv")
    app.dependency_overrides[conversations_mod.get_conversation_store] = lambda: legacy
    original = conversations_mod.committed_storage_version
    conversations_mod.committed_storage_version = lambda _data_dir: 1  # type: ignore[assignment]
    api_deps.register_app_store(app_store)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/conversations", params={"workspace": WORKSPACE})
        assert response.status_code == 201, response.text
        conversation_id = response.json()["id"]
        assert await app_store.get_conversation(conversation_id) is None
        assert legacy.get(conversation_id) is not None
    finally:
        conversations_mod.committed_storage_version = original  # type: ignore[assignment]
        api_deps.register_app_store(None)


async def test_workspace_row_self_heals_with_archived_defaults(app_store) -> None:
    """工作区行自愈要把存档里的默认值带过来（否则用户设置被重置回兜底）。"""

    workspace_store = api_deps.get_workspace_store()
    workspace = workspace_store.ensure_default()
    workspace_store.adopt(
        workspace.model_copy(
            update={
                "default_permission_mode": PermissionMode.FULL,
                "default_reasoning_level": ReasoningLevel.MAX,
            }
        )
    )
    assert await app_store.get_workspace(workspace.id) is None

    record = await ensure_workspace_row(app_store, workspace_store, workspace.id)
    assert record.default_permission_mode is PermissionMode.FULL
    assert record.default_reasoning_level is ReasoningLevel.MAX
    assert record.status is WorkspaceStatus.ACTIVE
