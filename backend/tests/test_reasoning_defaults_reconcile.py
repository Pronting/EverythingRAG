"""工作区默认思考档位的一次性对账（用户实测复现的「默认还是深度思考」）。

**缺陷**：默认思考强度从 `high` 改为 `off` 后，新建对话读的是**工作区已落库的默认值**
（`frontend/src/App.tsx` → `workspace.default_reasoning_level ?? DEFAULT_REASONING_LEVEL`），
旧库里那批「按旧程序默认值写入的 `high`」继续覆盖代码默认值 ⇒ 界面仍显示「深度思考」。
只改代码常量/DDL 默认值**不足以**修复，必须把已落库的旧默认值对账掉。

本文件锁住五件事：
1. 旧默认值 `high` 的工作区被改写成当前默认值；
2. 用户/测试显式写过的档位（`low` / `max` / 已是 `off`）**一字不动**；
3. 幂等：同版本只跑一次，重复调用不再改写；
4. 存档（`workspaces.json`）随权威行镜像，不与 app.db 分叉；
5. **对账必须在事件循环里也能跑**（曾经用 `asyncio.run` ⇒ 重启后静默不执行）。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import DEFAULT_REASONING_LEVEL, ReasoningLevel
from app.core.reasoning_defaults import (
    WORKSPACE_REASONING_DEFAULTS_VERSION,
    reconcile_workspace_default_reasoning,
)
from app.core.settings_store import SettingsStore
from app.core.workspace_store import WorkspaceStore
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具


async def _seed_workspaces(app_store: object) -> None:
    """三个工作区：旧默认值 / 用户显式档位 / 已是当前默认值。"""

    await app_store.create_workspace(
        "ws-legacy", "sk-legacy", name="旧默认", default_reasoning_level=ReasoningLevel.HIGH
    )
    await app_store.create_workspace(
        "ws-explicit", "sk-explicit", name="显式", default_reasoning_level=ReasoningLevel.MAX
    )
    await app_store.create_workspace(
        "ws-current", "sk-current", name="已新", default_reasoning_level=DEFAULT_REASONING_LEVEL
    )


def test_reconcile_table_targets_the_current_default() -> None:
    """本版本要改写的旧默认值就是 `high`（防止对账表写成当前默认值而变成永真）。"""

    assert DEFAULT_REASONING_LEVEL is ReasoningLevel.OFF
    assert WORKSPACE_REASONING_DEFAULTS_VERSION == 1


async def test_reconcile_rewrites_only_the_legacy_default(
    app_store, tmp_path: Path
) -> None:
    await _seed_workspaces(app_store)
    settings = SettingsStore(tmp_path / "data")

    changed = await reconcile_workspace_default_reasoning(settings, app_store)

    assert changed == 1, "只有「旧程序默认值」那一个工作区允许被改写"
    legacy = await app_store.get_workspace("ws-legacy")
    assert legacy is not None and legacy.default_reasoning_level is DEFAULT_REASONING_LEVEL
    explicit = await app_store.get_workspace("ws-explicit")
    assert explicit is not None and explicit.default_reasoning_level is ReasoningLevel.MAX, (
        "用户显式写过的档位不得被对账覆盖"
    )
    current = await app_store.get_workspace("ws-current")
    assert current is not None and current.default_reasoning_level is DEFAULT_REASONING_LEVEL
    # 完成标记落盘：与 app.db 同一数据目录
    assert (
        settings.load().workspace_reasoning_preferences_version
        == WORKSPACE_REASONING_DEFAULTS_VERSION
    )


async def test_reconcile_is_idempotent(app_store, tmp_path: Path) -> None:
    await _seed_workspaces(app_store)
    settings = SettingsStore(tmp_path / "data")
    assert await reconcile_workspace_default_reasoning(settings, app_store) == 1

    # 用户随后显式改回 high（旧值）：对账已经跑过，不得再动它
    await app_store.update_workspace("ws-legacy", default_reasoning_level=ReasoningLevel.HIGH)
    assert await reconcile_workspace_default_reasoning(settings, app_store) == 0
    after = await app_store.get_workspace("ws-legacy")
    assert after is not None and after.default_reasoning_level is ReasoningLevel.HIGH


async def test_reconcile_mirrors_the_archive(app_store, tmp_path: Path) -> None:
    """存档行（workspaces.json）必须跟着权威行走，避免两份状态分叉。"""

    await _seed_workspaces(app_store)
    data_dir = tmp_path / "data"
    archive = WorkspaceStore(data_dir)
    archive.adopt_record(await app_store.get_workspace("ws-legacy"))

    await reconcile_workspace_default_reasoning(
        SettingsStore(data_dir), app_store, archive_store=archive
    )

    reloaded = WorkspaceStore(data_dir)
    assert reloaded.get("ws-legacy").default_reasoning_level is DEFAULT_REASONING_LEVEL


async def test_reconcile_result_reaches_conversation_defaults(
    app_store, tmp_path: Path
) -> None:
    """对账后的工作区默认值真的成为新会话默认值（PRD 1.3 三层优先级的工作区层）。"""

    from app.api.settings_scope import resolve_conversation_settings
    from app.core.workspace_context import WorkspaceContextFactory

    await _seed_workspaces(app_store)
    data_dir = tmp_path / "data"
    archive = WorkspaceStore(data_dir)
    for workspace_id in ("ws-legacy", "ws-explicit", "ws-current"):
        archive.adopt_record(await app_store.get_workspace(workspace_id))
    settings = SettingsStore(data_dir)
    await reconcile_workspace_default_reasoning(settings, app_store, archive_store=archive)

    factory = WorkspaceContextFactory(data_dir, archive)
    view = factory.view_for("ws-legacy")
    assert view.default_reasoning_level is DEFAULT_REASONING_LEVEL
    assert resolve_conversation_settings(view).reasoning_level is DEFAULT_REASONING_LEVEL


async def test_reconcile_never_writes_the_table_directly(tmp_path: Path) -> None:
    """D19/A3 守卫的补充：对账不得自带 SQL，写库必须走 AppStore。"""

    import inspect

    from app.core import reasoning_defaults as module

    source = inspect.getsource(module)
    assert "UPDATE workspaces" not in source, "写 workspaces 表必须走 AppStore（D19/A3）"
    assert "sqlite3" not in source
    assert "reconcile_default_reasoning_level(" in source


def test_lifespan_reconcile_is_awaited_inside_the_loop() -> None:
    """lifespan 装配处的对账必须是**被 await 的异步调用**（机器可核对的守门）。

    回归背景：早期版本在同步阶段用 ``asyncio.run`` ⇒ ``RuntimeError: asyncio.run() cannot
    be called from a running event loop``，被宽泛的 except 吞成一条 warning，重启后对账
    静默不执行、界面继续显示「深度思考」。
    """

    import ast
    import inspect

    from app import main as main_module

    source = inspect.getsource(main_module._reconcile_workspace_reasoning_defaults)
    assert inspect.iscoroutinefunction(main_module._reconcile_workspace_reasoning_defaults)
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "run" not in called, "不得在事件循环内 asyncio.run"
    # 真实的 lifespan 里必须真的 await 它
    assert "await _reconcile_workspace_reasoning_defaults(store)" in inspect.getsource(
        main_module.lifespan
    )
