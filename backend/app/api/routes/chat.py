"""SSE 问答端点：POST /api/chat（帧契约 meta/token/done/error）。

路由薄：请求校验 + 组装 StreamingResponse；业务在 ChatService。
依赖注入用 Depends(get_chat_service)；测试中用 app.dependency_overrides 替换为 fake。
provider 未配置时 get_chat_service 返回占位模型，错误延迟到 stream 阶段上报 error 帧。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_vector_store
from app.core.settings_store import SettingsStore, get_settings_store
from app.generation.chat_service import ChatService
from app.generation.providers import (
    ChatProviderError,
    UnconfiguredChatModel,
    create_chat_model_from_config,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import create_embedder_from_config

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def get_chat_service(
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> ChatService:
    """构建真实问答链路（嵌入 + Chroma 检索 + 对话模型 + 设置页系统提示词）。

    chat 配置来自设置（config.json 优先，env 兜底）；provider 未配置时返回占位模型，
    ChatProviderError 延迟到请求的 stream 阶段以 error 帧上报，不崩不 500。
    """
    app_settings = store.load()
    embedder = create_embedder_from_config(app_settings.embed)
    retriever = VectorRetriever(vectorstore)
    try:
        chat_model = create_chat_model_from_config(app_settings.chat)
    except ChatProviderError as exc:
        chat_model = UnconfiguredChatModel(exc)
    return ChatService(
        embedder=embedder,
        retriever=retriever,
        chat_model=chat_model,
        system_prompt=app_settings.system_prompt,
    )


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> StreamingResponse:
    """流式问答：SSE 帧 meta -> token(s) -> done / error（ensure_ascii=False）。"""

    async def event_stream() -> AsyncIterator[str]:
        async for frame in service.stream_answer(request.message):
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
