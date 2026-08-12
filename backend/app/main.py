"""Everything RAG —— FastAPI 应用入口（本地 Web 服务）。

启动：uvicorn app.main:app --host 127.0.0.1 --port <随机端口>
MVP 阶段仅提供 /api/status 等基础端点，业务模块逐步接入。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.deps import get_import_task_store, get_vector_store
from app.api.routes.audit import router as audit_router
from app.api.routes.chat import router as chat_router
from app.api.routes.imports import router as import_router
from app.core.config import settings
from app.core.outbound import outbound_client
from app.ingestion.import_task import ImportTaskStore
from app.vectorstore.base import VectorStore

app = FastAPI(
    title="Everything RAG",
    version=settings.app_version,
    description="个人知识第二大脑 —— 纯本地 RAG 检索问答",
)

# 业务路由
app.include_router(chat_router)
app.include_router(audit_router)
app.include_router(import_router)

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
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
    task_store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
) -> dict:
    """健康检查 + 基础状态：供前端判断启动完成 / 向导入口 / 空库判断。

    知识计数来自真实向量库（块数 / 去重文件数）与最近成功导入时间。
    """
    last_sync = task_store.last_success_at
    return {
        "app": {
            "name": "everything-rag",
            "version": settings.app_version,
        },
        "config": {
            "wizard_completed": settings.wizard_completed,
            # 嵌入：本地 bge-m3 自动可用（首次真实嵌入惰性下载），无需配置
            "embed_configured": True,
            # 对话：base_url 与 model 都配好才算就绪（key 走 SecretStr，不在此回显）
            "chat_configured": bool(settings.chat_base_url and settings.chat_model),
            "vision_configured": False,
            "vision_enabled": settings.vision_enabled,
        },
        "knowledge": {
            "file_count": vectorstore.count_files(),
            "chunk_count": vectorstore.count(),
            "image_count": 0,
            "last_sync_at": last_sync.isoformat() if last_sync is not None else None,
            "needs_rebuild": False,
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
