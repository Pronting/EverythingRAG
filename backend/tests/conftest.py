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
from tests.fakes import FakeEmbedder, FakeVectorStore


@pytest.fixture(autouse=True)
def _isolate_local_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """每测试：注入空向量库替身 + 清空任务存储 + 隔离设置/会话/增量状态存储（tmp 目录）；
    结束后清空全部 overrides。"""
    deps.import_task_store.clear()
    app.dependency_overrides[deps.get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
    from app.core import settings_store as settings_store_mod
    from app.core.config import settings

    # 隔离真实数据目录（~/.everything-rag）：上传规范根/状态库/会话都不触碰真实数据
    monkeypatch.setattr(settings, "data_dir", tmp_path / "erag-data")

    tmp_store = settings_store_mod.SettingsStore(tmp_path / "data")
    app.dependency_overrides[settings_store_mod.get_settings_store] = lambda: tmp_store
    from app.api.routes import conversations as conversations_mod
    from app.core.conversation_store import ConversationStore
    from app.core.workspace_store import WorkspaceStore

    app.dependency_overrides[conversations_mod.get_conversation_store] = lambda: ConversationStore(
        tmp_path / "conv-data"
    )
    # 对话分区（工作区）同样隔离到 tmp：否则测试会读写用户真实的 workspaces.json
    app.dependency_overrides[conversations_mod.get_workspace_store] = lambda: WorkspaceStore(
        tmp_path / "conv-data"
    )
    # 增量同步状态库也隔离到 tmp（避免触碰真实 ~/.everything-rag/state.db）
    from app.ingestion.state_store import DocumentStateStore

    app.dependency_overrides[deps.get_state_store] = lambda: DocumentStateStore(tmp_path / "state.db")

    # --- V1.2.0 工作区隔离（T01，append-only：不修改上面既有语义） ---
    # deps 新增的按区 provider（get_workspace_context / *_vector_store / *_state_store /
    # *_knowledge_area）也要隔离到 tmp。这里直接 monkeypatch 模块单例，而不仅
    # dependency_overrides：按区 provider 支持**非注入式直接调用**（迁移、后台任务、
    # 工具执行器），只 override FastAPI 依赖会漏掉那条路径并读到真实 ~/.everything-rag。
    from app.api import deps as deps_mod
    from app.core.workspace_context import commit_storage_version

    # 与 conversations 路由的 workspace_store 指向同一目录，避免两处状态分叉。
    monkeypatch.setattr(deps_mod, "workspace_store", WorkspaceStore(tmp_path / "conv-data"))
    # app.db 访问层由 lifespan 注册；测试里一律由用例显式注入（内存 AppStore）。
    monkeypatch.setattr(deps_mod, "_app_store", None)
    deps_mod.reset_workspace_scoped_caches()
    # 默认按**已迁移**（每工作区布局）解析：tmp 目录里没有旧布局产物，若不提交标记，
    # 一旦某用例造出 data_dir/state.db 之类的旧布局产物，解析结果会在用例内翻转。
    commit_storage_version(tmp_path / "erag-data")
    yield
    app.dependency_overrides.clear()


def _stop_chroma_systems() -> None:
    """先显式 ``stop()`` 每个存活 System，再清空 Chroma 的进程级缓存。

    只调 ``clear_system_cache()`` 是**不够**的：它丢掉引用但**不关闭**上一用例
    ``PersistentClient`` 持有的 HNSW 读写句柄。那些句柄仍挂在被清的 System 上，后续用例
    在同一 persist 根再建 client 时会拿到「半死」System —— 全量运行时表现为随机失败
    （captain 实测到 ``Error in compaction: Failed to get hnsw segment writer``，
    本工作流实测到 ``Error creating hnsw segment reader: Nothing found on disk``，
    同一根因的 writer / reader 两个面）。teardown 必须尽力而为，绝不掩盖真实失败。
    """

    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:  # pragma: no cover - chromadb 是硬依赖
        return
    systems = getattr(SharedSystemClient, "_identifier_to_system", {})
    for system in list(systems.values()):
        stop = getattr(system, "stop", None)
        if stop is None:
            continue
        try:
            stop()
        except Exception:  # noqa: BLE001, S110 -- 关闭失败不影响测试结果判定
            pass
    SharedSystemClient.clear_system_cache()


@pytest.fixture(autouse=True)
def _reset_chroma_system_cache() -> None:
    """每个用例前后**停止并清空** Chroma 的进程级 System 缓存（T01 append-only 补充）。

    Chroma 按 settings 标识缓存 ``PersistentClient`` 背后的 System；跨用例残留的写句柄
    会让后续用例拿到半死 System。见 :func:`_stop_chroma_systems`。
    """

    _stop_chroma_systems()
    yield
    _stop_chroma_systems()
