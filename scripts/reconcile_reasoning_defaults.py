"""对真实数据跑一次工作区默认档位对账（先复制到临时目录，绝不直接改原件）。

用途：验证修复对用户真实 app.db 真的生效。dry-run 默认开启；加 ``--apply`` 才写回原件。
平时不需要手动运行 —— 服务启动时（``app.main.lifespan``）会自动做同一件事，本脚本只是
「在不启动服务的前提下核对结果」的离线入口。

用法：
    .venv/Scripts/python ../scripts/reconcile_reasoning_defaults.py            # dry-run
    .venv/Scripts/python ../scripts/reconcile_reasoning_defaults.py --apply    # 写回真实数据
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from app.agent.app_store import AppStore  # noqa: E402
from app.core.reasoning_defaults import (  # noqa: E402
    reconcile_workspace_default_reasoning,
)
from app.core.settings_store import SettingsStore  # noqa: E402
from app.core.workspace_store import WorkspaceStore  # noqa: E402

LIVE = Path.home() / ".everything-rag"


def _dump(app_store_records: list, label: str) -> None:
    print(f"--- {label} ---")
    for record in app_store_records:
        print(f"  {record.id:14s} {record.default_reasoning_level.value}")


async def main() -> int:
    apply_to_live = "--apply" in sys.argv
    target = LIVE if apply_to_live else Path(tempfile.mkdtemp(prefix="erag-reasoning-dryrun-"))
    if not apply_to_live:
        for name in ("app.db", "config.json", "workspaces.json"):
            src = LIVE / name
            if src.is_file():
                shutil.copy2(src, target / name)

    app_store = await AppStore.open(target / "app.db")
    archive = WorkspaceStore(target)
    settings = SettingsStore(target)
    try:
        before = await app_store.list_workspaces()
        _dump(before, f"BEFORE（{'真实数据' if apply_to_live else '副本'}）")
        # 与 lifespan 调用的是**同一个**函数（因此这里验证的就是启动时那条路径）
        changed = await reconcile_workspace_default_reasoning(
            settings, app_store, archive_store=archive
        )
        after = await app_store.list_workspaces()
        _dump(after, "AFTER")
        print(f"改写条数 = {changed}")
        print(f"标记版本 = {settings.load().workspace_reasoning_preferences_version}")
        print("归档行：", {w.id: w.default_reasoning_level.value for w in archive.list()})
        if not apply_to_live:
            print(f"（dry-run 副本目录：{target}）")
        return 0
    finally:
        await app_store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
