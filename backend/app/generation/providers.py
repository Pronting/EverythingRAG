"""对话 provider：OpenAI 兼容云端模型（流式），配置字段走环境变量占位。

隐私约束：API key 仅从 os.environ 读取，不落 config / 文件 / 日志；
import 本模块、create_chat_model 构造均零联网（AsyncOpenAI 客户端惰性，构造不建连）。
真实出网仅发生在 OpenAICompatChatModel.stream_chat 调用云端时（用户已授权）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.generation.base import ChatModel


class ChatProviderError(Exception):
    """对话 provider 配置缺失 / 调用失败等清晰错误（消息可读、不泄露密钥）。"""


class OpenAICompatChatModel:
    """OpenAI 兼容 chat/completions 流式模型（云端，用户已授权）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "local")

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式调用云端 chat/completions；逐 chunk 提取 delta.content，空内容跳过。"""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


class UnconfiguredChatModel:
    """配置缺失时的占位模型：stream 阶段抛出 ChatProviderError（构造零联网）。"""

    def __init__(self, error: ChatProviderError) -> None:
        self._error = error

    @property
    def supports_image_input(self) -> bool:
        return False

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        return _ErrorStream(self._error)


class _ErrorStream:
    """首次迭代即抛出给定异常的空 async 迭代器。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __aiter__(self) -> _ErrorStream:
        return self

    async def __anext__(self) -> str:
        raise self._error


def create_chat_model(settings: Settings) -> ChatModel:
    """从 settings 构造对话模型；配置缺失 raise ChatProviderError（零联网）。

    缺失项消息列出对应的环境变量名（base_url / model），便于用户补齐配置。
    """
    missing: list[str] = []
    if not settings.chat_base_url:
        missing.append("EVERYTHING_RAG_CHAT_BASE_URL")
    if not settings.chat_model:
        missing.append("EVERYTHING_RAG_CHAT_MODEL")
    if missing:
        raise ChatProviderError(f"对话 provider 未配置，缺少环境变量: {', '.join(missing)}")
    api_key = os.environ.get(settings.chat_api_key_env)
    return OpenAICompatChatModel(
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        api_key=api_key,
    )
