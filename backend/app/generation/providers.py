"""对话 provider：OpenAI 兼容云端模型（流式），配置字段走环境变量占位。

隐私约束：API key 仅从 os.environ 读取，不落 config / 文件 / 日志；
import 本模块、create_chat_model 构造均零联网（AsyncOpenAI 客户端惰性，构造不建连）。
真实出网仅发生在 OpenAICompatChatModel.stream_chat 调用云端时（用户已授权），
且必须经唯一 OutboundClient 记录审计事件（destination 只取 host，脱敏）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.outbound import OutboundClient, OutboundEvent, outbound_client
from app.core.settings_store import ChatModelConfig
from app.generation.base import ChatChunk, ChatModel


class ChatProviderError(Exception):
    """对话 provider 配置缺失 / 调用失败等清晰错误（消息可读、不泄露密钥）。"""


class OpenAICompatChatModel:
    """OpenAI 兼容 chat/completions 流式模型（云端，用户已授权）。

    outbound 为可选的 OutboundClient：注入后每次出网调用记录一条审计事件
    （成功 status=200 / 异常 status=None，异常照常向上抛）；缺省 None 时不
    记录，保持既有行为（零审计开销）。destination 只取 base_url 的 host，
    绝不含路径 / query / api key。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        outbound: OutboundClient | None = None,
        supports_image_input: bool = False,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._outbound = outbound
        self._destination = _destination_host(base_url)
        self._supports_image_input = supports_image_input
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "local")

    @property
    def supports_image_input(self) -> bool:
        return self._supports_image_input

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """流式调用云端 chat/completions；逐 chunk 提取正文与思维链增量。

        正文走 delta.content，思维链走 delta.reasoning_content（DeepSeek 推理模型
        等），分别产出 kind=content / reasoning 的 ChatChunk，前端分样式渲染。
        MEDIUM-1：收尾 usage-only chunk（choices 空）跳过；MEDIUM-3：空 choices 带
        error -> 抛 ChatProviderError。出网审计同前。
        """
        started_at = _utcnow() if self._outbound is not None else None
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for chunk in response:
                if not chunk.choices:  # usage-only 收尾 chunk（MEDIUM-1）
                    error = getattr(chunk, "error", None)
                    if error is not None:  # MEDIUM-3：流式中途报错
                        message = getattr(error, "message", None) or str(error)
                        raise ChatProviderError(f"对话服务流式中止: {message}")
                    continue
                delta = chunk.choices[0].delta
                token = delta.content or ""
                if token:
                    yield ChatChunk("content", token)
                reasoning = getattr(delta, "reasoning_content", None) or ""
                if reasoning:
                    yield ChatChunk("reasoning", reasoning)
        except BaseException:
            # 异常含 GeneratorExit（SSE 客户端断开触发 aclose）：已发生出网也要记审计
            self._record_outbound(started_at, status=None)
            raise
        else:
            self._record_outbound(started_at, status=200)

    def _record_outbound(self, started_at: datetime | None, status: int | None) -> None:
        """记录一条出网审计事件；未注入 outbound / 未开始计时则跳过。"""
        if self._outbound is None or started_at is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="chat",
                destination=self._destination,
                method="chat.completions",
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(started_at),
            )
        )


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


def _destination_host(base_url: str) -> str:
    """从 base_url 提取仅 host 的审计目的地（绝不含路径 / query / key）。"""
    hostname = urlsplit(base_url).hostname
    if hostname:
        return hostname
    head = base_url.split("/", 1)[0]
    return head.split("?", 1)[0].split("#", 1)[0] or "unknown"


def _utcnow() -> datetime:
    """当前 UTC 时间（审计 started_at 用，带时区可序列化）。"""
    return datetime.now(UTC)


def _elapsed_ms(started_at: datetime) -> float:
    """自 started_at 起经过的毫秒数（≥0）。"""
    elapsed = datetime.now(UTC) - started_at
    return max(0.0, elapsed.total_seconds() * 1000.0)


def create_chat_model(
    settings: Settings,
    outbound: OutboundClient | None = outbound_client,  # 缺省接全局单例，生产审计不静默失效
) -> ChatModel:
    """从 settings 构造对话模型；配置缺失 raise ChatProviderError（零联网）。

    缺失项消息列出对应的环境变量名（base_url / model），便于用户补齐配置。
    outbound 缺省为全局 outbound_client：真实出网自动进入网络审计（/api/audit、
    /api/status.privacy.outbound_state 才如实上报，而非恒为 local-only）。
    """
    missing: list[str] = []
    if not settings.chat_base_url:
        missing.append("EVERYTHING_RAG_CHAT_BASE_URL")
    if not settings.chat_model:
        missing.append("EVERYTHING_RAG_CHAT_MODEL")
    if missing:
        raise ChatProviderError(f"对话 provider 未配置，缺少环境变量: {', '.join(missing)}")
    api_key = _resolve_api_key(settings)
    return OpenAICompatChatModel(
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        api_key=api_key,
        outbound=outbound,
    )


def _resolve_api_key(settings: Settings) -> str | None:
    """解析 API key：优先 Settings.chat_api_key（.env / 环境变量，SecretStr）；
    其次回退 chat_api_key_env 指定的自定义环境变量名（真实 os.environ）。

    pydantic-settings 会把 .env 里的值填进 Settings 字段，但不会注入
    os.environ——因此 key 必须走 Settings 字段而非 os.environ.get，否则
    .env 里配置的 key 永远读不到（会用 "local" 占位导致云端 401）。
    """
    if settings.chat_api_key is not None:
        return settings.chat_api_key.get_secret_value()
    return os.environ.get(settings.chat_api_key_env)


def create_chat_model_from_config(
    chat: ChatModelConfig,
    outbound: OutboundClient | None = outbound_client,
) -> ChatModel:
    """从设置页 ChatModelConfig（config.json / env 合并后的有效值）构造对话模型。

    与 create_chat_model(env) 等价，但读取 dashboard 设置；配置缺失 raise
    ChatProviderError（缺失项列出字段名），outbound 缺省接全局审计单例。
    """
    missing: list[str] = []
    if not chat.base_url:
        missing.append("对话模型 Base URL")
    if not chat.model:
        missing.append("对话模型名")
    if missing:
        raise ChatProviderError(f"对话模型未配置，缺少: {', '.join(missing)}")
    api_key = chat.api_key.get_secret_value() if chat.api_key is not None else None
    return OpenAICompatChatModel(
        base_url=chat.base_url,
        model=chat.model,
        api_key=api_key,
        outbound=outbound,
        supports_image_input=chat.supports_image,
    )
