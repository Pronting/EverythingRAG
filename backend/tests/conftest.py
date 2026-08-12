"""pytest 全局夹具：隔离真实数据目录（~/.everything-rag）。

/api/status 现在依赖向量库（get_vector_store）。默认给所有在进程内测试注入
空 FakeVectorStore + 清空的导入任务存储，避免测试触碰用户真实 Chroma（且与
运行中的服务抢同一 sqlite 锁）。需要真实行为或自定义计数的用例再自行 override。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api import deps
from app.main import app
from tests.fakes import FakeVectorStore


@pytest.fixture(autouse=True)
def _isolate_local_deps(tmp_path: Path) -> None:
    """每测试：注入空向量库替身 + 清空任务存储 + 隔离设置存储（tmp 目录）；
    结束后清空全部 overrides。"""
    deps.import_task_store.clear()
    app.dependency_overrides[deps.get_vector_store] = lambda: FakeVectorStore()
    from app.core import settings_store as settings_store_mod

    tmp_store = settings_store_mod.SettingsStore(tmp_path / "data")
    app.dependency_overrides[settings_store_mod.get_settings_store] = lambda: tmp_store
    yield
    app.dependency_overrides.clear()
