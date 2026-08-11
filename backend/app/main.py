"""Everything RAG —— FastAPI 应用入口（本地 Web 服务）。

启动：uvicorn app.main:app --host 127.0.0.1 --port <随机端口>
MVP 阶段仅提供 /api/status 等基础端点，业务模块逐步接入。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

app = FastAPI(
    title="Everything RAG",
    version=settings.app_version,
    description="个人知识第二大脑 —— 纯本地 RAG 检索问答",
)

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
async def status() -> dict:
    """健康检查 + 基础状态：供前端判断启动完成 / 向导入口 / 空库判断。"""
    return {
        "app": {
            "name": "everything-rag",
            "version": settings.app_version,
        },
        "config": {
            "wizard_completed": settings.wizard_completed,
            "embed_configured": False,
            "chat_configured": False,
            "vision_configured": False,
            "vision_enabled": settings.vision_enabled,
        },
        "knowledge": {
            "file_count": 0,
            "chunk_count": 0,
            "image_count": 0,
            "last_sync_at": None,
            "needs_rebuild": False,
        },
        "privacy": {
            "outbound_state": "local-only",
            "providers": {},
        },
    }
