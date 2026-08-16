"""SSE 问答端点：POST /api/chat（帧契约 meta/token/done/error）。

路由薄：请求校验 + 组装 StreamingResponse；业务在 ChatService。
依赖注入用 Depends(get_chat_service)；测试中用 app.dependency_overrides 替换为 fake。
provider 未配置时 get_chat_service 返回占位模型，错误延迟到 stream 阶段上报 error 帧。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_embedder, get_vector_store
from app.core.outbound import outbound_client
from app.core.settings_store import (
    SettingsStore,
    get_settings_store,
    is_search_configured,
    is_vision_configured,
)
from app.generation.chat_service import ChatService
from app.generation.providers import (
    ChatProviderError,
    UnconfiguredChatModel,
    create_chat_model_from_config,
)
from app.generation.vision import VisionProviderError, create_vision_model_from_config
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY
from app.search.base import SearchError
from app.search.factory import create_search_provider_from_config
from app.vectorstore.base import VectorStore

router = APIRouter()


class ChatMessage(BaseModel):
    """多轮对话历史中的一条（role + content）。"""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    web_search: bool = False  # 「联网搜索」开关（每条消息独立 opt-in）
    #: 用户贴图：data URI（base64 内联）或 http(s) 图床 URL；纯文本模型走识图代理。
    images: list[str] = Field(default_factory=list, max_length=4)


def get_chat_service(
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> ChatService:
    """构建真实问答链路（嵌入 + Chroma 检索 + 对话模型 + 设置页系统提示词）。

    chat 配置来自设置（config.json 优先，env 兜底）；provider 未配置时返回占位模型，
    ChatProviderError 延迟到请求的 stream 阶段以 error 帧上报，不崩不 500。
    """
    app_settings = store.load()
    embedder = get_embedder()
    # 混合检索（dense + BM25 标题注入 + RRF）+ 相关度门控。
    # 注：cross-encoder 重排曾试（bge-reranker-base），实测 hit@5 84.6% -> 76.9%
    # 排序不准，已弃用；当前混合检索为实测最优。
    retriever = HybridRetriever(vectorstore, min_similarity=DEFAULT_MIN_SIMILARITY)
    try:
        chat_model = create_chat_model_from_config(app_settings.chat)
    except ChatProviderError as exc:
        chat_model = UnconfiguredChatModel(exc)
    # 联网搜索：启用且凭证齐备才构建 provider；否则 None（web_search 请求时上报友好错误）
    search_provider = None
    if is_search_configured(app_settings.search):
        try:
            search_provider = create_search_provider_from_config(app_settings.search)
        except SearchError:
            search_provider = None
    # 识图模型（可选）：纯文本对话模型 + 用户贴图时，识图代理复用它转文字。
    vision = None
    if is_vision_configured(app_settings.vision):
        try:
            vision = create_vision_model_from_config(app_settings.vision, outbound=outbound_client)
        except VisionProviderError:
            vision = None
    return ChatService(
        embedder=embedder,
        retriever=retriever,
        chat_model=chat_model,
        system_prompt=app_settings.system_prompt,
        search_provider=search_provider,
        search_max_results=app_settings.search.max_results,
        vision=vision,
    )


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),  # noqa: B008
) -> StreamingResponse:
    """流式问答：SSE 帧 meta -> token(s) -> done / error（ensure_ascii=False）。"""

    async def event_stream() -> AsyncIterator[str]:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        async for frame in service.stream_answer(
            request.message, history, web_search=request.web_search, images=request.images
        ):
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
