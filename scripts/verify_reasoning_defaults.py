"""对账链路的**取数级**复核（不碰真实数据）：真实 ``app.main.lifespan`` + 旧默认值工作区。

证明三件事：
1. `app.main.lifespan` 启动时真的调用对账，并把「旧默认档位 high」改写为当前默认值 off
   （app.db 权威行 = 启动后前端拿到的值）；
2. 完成标记（config.json）与 `workspaces.json` 存档同步，且**幂等**（再启动一次不改写）；
3. 用户显式写过的档位（max）不被误改。

为什么直接驱动 ``lifespan`` 而不是走 ``TestClient``：本仓库 ``deps.workspace_store`` 是
**导入时**建立的模块级单例，路由经 ``Depends(get_workspace_store)`` 拿的就是它；脚本里换
实例并不会让路由换实例，于是会得到一个与真实启动无关的假结论。直接 await 真实的
``lifespan`` 才是在复核生产那条路径。

用法：`.venv/Scripts/python ../scripts/verify_reasoning_defaults.py`（只需一次，针对临时数据目录）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


def _seed(data_dir: Path, *, legacy: Path | None = None) -> None:
    """在盘上造两个工作区：旧默认值 high + 用户显式 max（模拟用户既有安装）。"""

    from app.agent.app_store import AppStore
    from app.core.workspace_store import WorkspaceStore

    async def seed() -> None:
        db = legacy or (data_dir / "app.db")
        store = await AppStore.open(db)
        try:
            await store.create_workspace(
                "ws-legacy", "sk-legacy", name="旧默认", default_reasoning_level="high"
            )
            await store.create_workspace(
                "ws-explicit", "sk-explicit", name="显式", default_reasoning_level="max"
            )
            records = await store.list_workspaces()
        finally:
            await store.close()
        archive = WorkspaceStore(data_dir)
        for record in records:
            archive.adopt_record(record)

    asyncio.run(seed())


def _rows(data_dir: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{data_dir / 'app.db'}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT id, default_reasoning_level FROM workspaces"
            )
        }
    finally:
        connection.close()


def _archive(data_dir: Path) -> dict[str, str]:
    payload = json.loads((data_dir / "workspaces.json").read_text(encoding="utf-8"))
    return {row["id"]: row["default_reasoning_level"] for row in payload}


def _marker(data_dir: Path) -> int:
    path = data_dir / "config.json"
    if not path.is_file():
        return 0
    return int(
        json.loads(path.read_text(encoding="utf-8")).get(
            "workspace_reasoning_preferences_version", 0
        )
    )


async def _start(data_dir: Path) -> None:
    """真实启动一次服务（只跑 lifespan 的启动/关闭段）。"""

    from app.core.config import get_settings

    get_settings.cache_clear()
    assert get_settings().data_dir == data_dir, get_settings().data_dir

    from app.main import app, lifespan

    async with lifespan(app):
        print(f"  启动完成；app.db = {_rows(data_dir)}")


def main() -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="erag-verify-reasoning-"))
    os.environ["EVERYTHING_RAG_DATA_DIR"] = str(data_dir)
    print(f"data_dir = {data_dir}")

    _seed(data_dir)
    print(f"启动前 app.db = {_rows(data_dir)}（ws-legacy 应为 high）")
    assert _rows(data_dir) == {"ws-legacy": "high", "ws-explicit": "max"}
    assert _marker(data_dir) == 0

    print("第一次启动：")
    asyncio.run(_start(data_dir))
    after_first = _rows(data_dir)
    assert after_first["ws-legacy"] == "off", f"旧默认值未对账：{after_first}"
    assert after_first["ws-explicit"] == "max", f"显式档位被误改：{after_first}"
    assert _marker(data_dir) == 1, "完成标记未写入 config.json"
    assert _archive(data_dir)["ws-legacy"] == "off", "存档未同步"
    print(f"  对账后 app.db = {after_first}")
    print(f"  存档 = {_archive(data_dir)}；标记 = {_marker(data_dir)}")

    print("第二次启动（幂等性）：")
    asyncio.run(_start(data_dir))
    assert _rows(data_dir) == after_first, "重复启动不得再改写"
    print(f"  仍为 {_rows(data_dir)}")

    print("OK：真实 lifespan 对账链路全通（app.db → 存档 → 标记，且幂等）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
