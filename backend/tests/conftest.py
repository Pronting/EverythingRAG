"""pytest 全局夹具：隔离真实数据目录（~/.everything-rag）+ 隔离真实凭证与出网。

/api/status 现在依赖向量库（get_vector_store）。默认给所有在进程内测试注入
空 FakeVectorStore + 清空的导入任务存储，避免测试触碰用户真实 Chroma（且与
运行中的服务抢同一 sqlite 锁）。需要真实行为或自定义计数的用例再自行 override。

另一条**默认全局**隔离见 :func:`_no_real_credentials_or_outbound`：``backend/.env`` 存有
真实明文 provider 凭证，``SettingsStore.load()`` 在 ``config.json`` 缺失时会回退 ``.env``，
未隔离的模型装配点因此会构造真实 provider 并**真的出网**（本会话实测两次）。详见
:mod:`tests.netguard`。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.api import deps
from app.main import app
from tests import netguard
from tests.fakes import FakeEmbedder, FakeVectorStore


def pytest_configure(config: pytest.Config) -> None:
    """注册守卫自证用例专用标记（改 pytest 配置要动 pyproject，这里就地注册）。"""

    config.addinivalue_line(
        "markers",
        f"{netguard.ALLOW_OUTBOUND_MARKER}: 允许该用例故意触发一次被阻断的非回环连接"
        "（仅 tests/test_network_guard.py 的守卫自证使用）",
    )


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


@pytest.fixture(autouse=True)
def _no_real_credentials_or_outbound(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    _isolate_local_deps: None,
) -> Iterator[None]:
    """默认把每个用例放进「零真实凭证 + 零真实出网」环境（CLAUDE.md §11）。

    三层隔离（设置存储 / 模型装配输入 / 非回环连接守卫）见 :mod:`tests.netguard`。
    刻意依赖 ``_isolate_local_deps``：那个夹具用 ``settings_store.get_settings_store``
    **函数对象**当 ``dependency_overrides`` 的键，必须先由它取到原函数，本夹具再去替换
    模块属性，否则 override 会挂到一个已经没人用的对象上。

    **live 豁免**：设 ``EVERYTHING_RAG_LIVE_EVAL=1``（或
    ``EVERYTHING_RAG_RUN_LIVE_EVAL=1``）时三层隔离**全部不装** —— live 用例是用户显式
    授权的真实配置 / 真实外呼路径。`tests/test_netguard_scope.py` 用同一个判据自证
    「隔离状态与 live 开关严格互补」，并在显式开关下实跑一次。
    """

    if not netguard.isolation_enabled():
        yield
        return

    netguard.install_isolation(monkeypatch)
    allows_outbound = (
        request.node.get_closest_marker(netguard.ALLOW_OUTBOUND_MARKER) is not None
    )
    yield
    attempts = netguard.blocked_attempts()
    netguard.reset_isolation()
    if attempts and not allows_outbound:
        detail = "\n".join(
            f"- {item['host']}:{item['port']} ← {item['where']}" for item in attempts
        )
        pytest.fail(
            f"本轮存在 {len(attempts)} 次非回环连接尝试（默认测试运行禁止真实外呼）：\n{detail}"
        )


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
