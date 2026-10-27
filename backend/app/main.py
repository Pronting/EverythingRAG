"""Everything RAG —— FastAPI 应用入口（本地 Web 服务）。

启动：uvicorn app.main:app --host 127.0.0.1 --port <随机端口>
MVP 阶段仅提供 /api/status 等基础端点，业务模块逐步接入。
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent.app_store import AppStore
from app.agent.runtime import register_run_manager
from app.api import deps as api_deps
from app.api.deps import (
    app_db_path,
    get_embedder,
    get_image_state_store,
    get_import_task_store,
    get_state_store,
    get_vector_store,
    register_app_store,
    resume_pending_image_tasks,
)
from app.api.knowledge_services import knowledge_service_for
from app.api.routes.approvals import router as approvals_router
from app.api.routes.audit import router as audit_router
from app.api.routes.avatars import router as avatars_router
from app.api.routes.blocks import router as blocks_router
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.imports import router as import_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.memory import router as memory_router
from app.api.routes.runs import router as runs_router
from app.api.routes.settings import router as settings_router
from app.api.run_services import build_run_manager, warm_retrieval_indexes
from app.core.config import settings
from app.core.outbound import outbound_client
from app.core.settings_store import get_settings_store, is_search_configured, is_vision_configured
from app.core.workspace_context import committed_storage_version
from app.core.workspace_store import WORKSPACE_LAYOUT_VERSION
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.index_schema import make_index_fingerprint
from app.ingestion.state_store import DocumentStateStore
from app.mcp_server.server import MCPMount, MCPRuntime
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)
mcp_mount = MCPMount()


async def _initialize_history_indexes(store: AppStore) -> None:
    factory = api_deps.get_workspace_context_factory()

    def index_for(workspace_id: str):
        context = factory.resolve(workspace_id)
        return api_deps.workspace_history_index(context, store)

    store.configure_history_index(index_for)
    workspaces = (
        await store.list_workspaces()
        if committed_storage_version(factory.data_dir) >= WORKSPACE_LAYOUT_VERSION
        else factory.workspace_store.list()
    )
    for workspace in workspaces:
        await index_for(workspace.id).ensure_fresh()


async def _reconcile_workspace_reasoning_defaults(store: AppStore) -> None:
    """启动时的**异步**工作区默认档位对账（旧默认值 → 当前程序默认值）。

    必须在事件循环内执行：早期版本在同步阶段用 ``asyncio.run`` ⇒ ``RuntimeError`` 被宽泛的
    except 吞掉 ⇒ 对账静默不执行、界面照旧显示「深度思考」（用户实测复现）。写库统一走
    :class:`AppStore`（D19/A3：不得绕过它直写 ``workspaces`` 表）。

    与续跑识图任务（``resume_pending_image_tasks``）同属「启动即做、失败不判死启动」的
    维护动作：对账失败只记类型，绝不让本地服务起不来。
    """

    from app.core.reasoning_defaults import reconcile_workspace_default_reasoning
    from app.core.settings_store import get_settings_store
    from app.core.workspace_store import WorkspaceStore

    try:
        await reconcile_workspace_default_reasoning(
            get_settings_store(),
            store,
            archive_store=WorkspaceStore(settings.data_dir),
        )
    except Exception as exc:  # noqa: BLE001 - 维护动作失败不阻断启动
        logger.warning(
            "工作区默认档位对账未完成：error=%s（步骤=reasoning_defaults）", type(exc).__name__
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """启动：先创建并注册 app.db 单例（D28/D20），再续跑未完成的识图任务。

    顺序不可颠倒：``resume_pending_image_tasks()`` 按区续跑时经
    ``get_workspace_context_factory()`` 取上下文，而上下文只**引用** app.db
    单例（D20），因此单例必须先注册。``AppStore.open`` 打开 data_dir 根下的
    ``app.db``（WAL + ``PRAGMA foreign_keys=ON`` + 幂等建表）。

    fail-fast（captain 裁示 1）：迁移后 app.db 是唯一权威源，打不开即启动失败。
    日志只记「步骤 + 异常类名 + sqlite 错误名」，沿用仓内既有脱敏风格
    （不打印 ``str(exc)``，见 `docs/tech/v1.2.0-architecture.md` D28）。

    关闭时先 ``register_app_store(None)`` 复位再 ``await store.close()``：
    顺序反了会让关闭后仍有人拿到旧句柄（D28）。
    """

    try:
        store = await AppStore.open(app_db_path())
    except Exception as exc:  # 统一转成可诊断的启动错误
        logger.error(
            "启动失败：app.db 初始化未成功（步骤=AppStore.open, error=%s, sqlite=%s）",
            type(exc).__name__,
            getattr(exc, "sqlite_errorname", None),
        )
        raise RuntimeError(
            f"app.db 初始化失败（步骤=AppStore.open, error={type(exc).__name__}, "
            f"sqlite={getattr(exc, 'sqlite_errorname', None)})"
        ) from exc
    register_app_store(store)
    await _initialize_history_indexes(store)
    await _reconcile_workspace_reasoning_defaults(store)
    manager = None
    try:
        # V1.2.0：RunManager（PRD 12.2）。lifespan 创建、退出关闭；每工作区检查点按需建。
        try:
            manager = build_run_manager(store)
            register_run_manager(manager)
            _app.state.run_manager = manager
            await manager.recover_after_restart()
        except Exception as exc:
            logger.error(
                "启动失败：RunManager 初始化未成功（步骤=build_run_manager, error=%s）",
                type(exc).__name__,
            )
            raise RuntimeError(
                f"RunManager 初始化失败（步骤=build_run_manager, error={type(exc).__name__}）"
            ) from exc
        resume_pending_image_tasks()
        # 后台预热词法索引：把 BM25 构建成本（4.7k 块约 2–3.4s）从第一问挪到启动期。
        warm_retrieval_indexes()
        if settings.mcp_enabled:
            runtime = MCPRuntime(settings, knowledge_service_for)
            async with runtime.server.session_manager.run():
                mcp_mount.runtime = runtime
                try:
                    yield
                finally:
                    mcp_mount.runtime = None
                    await runtime.aclose()
        else:
            yield
    finally:
        if manager is not None:
            try:
                await manager.aclose()
            except Exception as exc:  # noqa: BLE001 - 关闭失败不掩盖主异常
                logger.warning("RunManager 关闭失败：error=%s", type(exc).__name__)
            register_run_manager(None)
            _app.state.run_manager = None
        register_app_store(None)
        try:
            await store.close()
        except Exception as exc:  # noqa: BLE001 -- 关闭失败不掩盖主异常，仅记类型
            logger.warning("app.db 关闭失败：error=%s", type(exc).__name__)


app = FastAPI(
    title="Everything RAG",
    version=settings.app_version,
    description="个人知识第二大脑 —— 纯本地 RAG 检索问答",
    lifespan=lifespan,
)

# 业务路由
app.mount("/mcp", mcp_mount, name="mcp")


@app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def mcp_without_slash(request: Request):
    from starlette.responses import JSONResponse, RedirectResponse

    if not settings.mcp_enabled or mcp_mount.runtime is None:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    rejection = mcp_mount.runtime.app.rejection(request.scope)
    if rejection is not None:
        return rejection
    return RedirectResponse("/mcp/", status_code=307)


app.include_router(chat_router)
app.include_router(audit_router)
app.include_router(avatars_router)
app.include_router(blocks_router)
app.include_router(import_router)
app.include_router(knowledge_router)
app.include_router(settings_router)
app.include_router(conversations_router)
# V1.2.0：工作区记忆与历史检索（PRD 13.2，owner: T08）
app.include_router(memory_router)
# V1.2.0：run API / SSE / 批准澄清（PRD 13.2 / 13.3，owner: T09）
app.include_router(runs_router)
app.include_router(approvals_router)

# 同源部署（前端静态资源由本服务托管），MVP 无需跨域；保留可配置位
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/api/status")
async def status(
    workspace_id: str | None = None,
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
    task_store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    state_store: DocumentStateStore = Depends(get_state_store),  # noqa: B008
    embedder: Embedder = Depends(get_embedder),  # noqa: B008
) -> dict:
    """健康检查 + 基础状态：供前端判断启动完成 / 向导入口 / 空库判断。

    知识计数来自真实向量库（块数 / 去重文件数）与最近成功导入时间。

    **按区统计**（PRD 5.3 / 14.3）：带 ``?workspace_id=`` 时，块数、图片任务数与
    重建判定全部取自该工作区的向量库 / 图片库 / 状态库；不带参数时沿用默认工作区的
    全局句柄（V1.1.0 兼容路径，前端设置页会显式带上工作区）。跨区句柄绝不混用 ——
    否则 A 区的空库判断会被 B 区的数据掩盖。
    """

    image_store = get_image_state_store()
    active_workspace = workspace_id
    if workspace_id:
        context = api_deps.get_workspace_context(workspace_id)
        active_workspace = context.workspace_id
        vectorstore = api_deps.get_workspace_vector_store(context)
        state_store = api_deps.workspace_state_store(context)
        image_store = api_deps.workspace_image_state_store(context)
    last_sync = task_store.last_success_at
    _app_settings = get_settings_store().load()
    _search_config = _app_settings.search
    _vision = _app_settings.vision
    chunk_count = vectorstore.count()
    needs_rebuild = state_store.needs_rebuild(
        index_fingerprint=make_index_fingerprint(embedder.fingerprint),
        current_chunk_count=chunk_count,
    )
    return {
        "app": {
            "name": "everything-rag",
            "version": settings.app_version,
        },
        "workspace_id": active_workspace,
        "config": {
            "wizard_completed": settings.wizard_completed,
            # 嵌入：云端 OpenAI 兼容 /embeddings；base_url 与 model 都配好才算就绪
            "embed_configured": bool(settings.embed_base_url and settings.embed_model),
            # 对话：base_url 与 model 都配好才算就绪（key 走 SecretStr，不在此回显）
            "chat_configured": bool(settings.chat_base_url and settings.chat_model),
            "vision_configured": is_vision_configured(_vision),
            "vision_enabled": settings.vision_enabled,
            # 联网搜索：显式启用 + provider 凭证齐备才算可用（config.json 权威）
            "search_enabled": _search_config.enabled,
            "search_configured": is_search_configured(_search_config),
        },
        "knowledge": {
            "file_count": vectorstore.count_files(),
            "chunk_count": chunk_count,
            "image_count": vectorstore.count_images(),
            "image_tasks": image_store.count_status(),
            "last_sync_at": last_sync.isoformat() if last_sync is not None else None,
            "needs_rebuild": needs_rebuild,
        },
        "privacy": {
            "outbound_state": outbound_client.outbound_state,
            "providers": {},
        },
    }


#: 前端构建产物目录（对齐 CLAUDE.md「dist/ 由 FastAPI 托管」）。
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# 静态托管：挂在 API 路由之后，保证 /api/* 优先；hash 路由前端无需 SPA 回退。
# dist 不存在时不影响 /api（前端未构建，仅后端 API 可访问）。
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
