"""对话 provider：OpenAI 兼容云端模型（流式 + 完整响应），配置字段走环境变量占位。

隐私约束：API key 仅从 os.environ / Settings(SecretStr) 读取，不落 config / 文件 /
日志；import 本模块、create_chat_model 构造均零联网（AsyncOpenAI 客户端惰性，
构造不建连）。真实出网仅发生在 ``stream_chat`` / ``complete`` 调用云端时（用户已
授权），且必须经唯一 OutboundClient 记录审计事件（destination 只取 host，脱敏）。

V1.2.0（PRD 9.1 / 9.2 / 7.4）：

- 四档思考：``thinking.type=disabled|enabled`` + ``reasoning_effort=low|high|max``，
  ``thinking`` 经 OpenAI SDK 的 ``extra_body`` 传递，默认 ``high``；
- 完整响应：``content`` / ``reasoning_content`` / ``tool_calls`` / ``finish_reason`` /
  ``usage`` / 提供商请求 ID；未完整生成的内容不标成功；
- 带 tools 的历史协议：保留并回传此前 assistant 消息的 ``reasoning_content``
  （含未调用工具的轮次），工具结果使用对应 ``tool_call_id``；
- 独立子 Agent 上下文隔离：历史摘要作为参考数据，不伪造 assistant 工具调用；
- 关闭 SDK 隐式重试（``max_retries=0``），重试由用户决定（PRD 7.4）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI

from app.agent.contracts import ModelCapabilities, ReasoningLevel, ToolCall
from app.core.config import Settings
from app.core.outbound import OutboundClient, OutboundEvent, outbound_client
from app.core.settings_store import ChatModelConfig
from app.generation.base import (
    ChatChunk,
    ChatModel,
    MessageDict,
    ModelResponse,
    ResponseStatus,
    StreamResult,
    ToolCallAssembler,
    ToolCallFragment,
    classify_finish_reason,
    provider_request_id_from_headers,
    tool_calls_from_message,
    usage_from_payload,
)
from app.generation.capabilities import (
    HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS,
    CapabilityConfigurationError,
    ProviderReasoningParams,
    resolve_capabilities,
    resolve_reasoning_params,
)

__all__ = [
    "REASONING_REPLAY_TOOLS",
    "ChatProviderError",
    "OpenAICompatChatModel",
    "ProviderRequestError",
    "UnconfiguredChatModel",
    "create_chat_model",
    "create_chat_model_from_config",
    "features_from_request",
    "prepare_messages",
    "reference_only_messages",
    "replay_reasoning_content",
    "response_from_completion",
    "unavailable",
]

#: 全量重放推理内容的场景（带 tools 的会话，PRD 9.2 末段）。
REASONING_REPLAY_TOOLS = "tools"

#: 需要的显式能力声明（缺失时一次报全，不让用户逐个试错）。
_REQUIRED_DECLARATION = (
    "supports_tools",
    "supports_thinking_toggle",
    "reasoning_levels",
    "supports_image_input",
    "context_limit",
    "max_output_tokens",
    "history_protocol",
)


class ChatProviderError(Exception):
    """对话 provider 配置缺失 / 调用失败等清晰错误（消息可读、不泄露密钥）。"""


class ProviderRequestError(ChatProviderError):
    """Provider 提交前的配置错误：不支持所选功能、协议不满足（PRD 9.2）。"""


class OpenAICompatChatModel:
    """OpenAI 兼容 chat/completions 模型（云端，用户已授权）。

    outbound 为可选的 OutboundClient：注入后每次出网调用记录一条审计事件
    （成功 status=200 / 异常 status=None，异常照常向上抛）；缺省 None 时不
    记录，保持既有行为（零审计开销）。destination 只取 base_url 的 host，
    绝不含路径 / query / api key。

    capabilities 可显式注入（测试与显式声明的端点）；缺省时**惰性解析**：
    ``resolve_at_init=False``（默认）不在构造期解析能力（构造零联网，自定义端点也
    能构造模型），但任何需要能力的功能调用（``tools`` / 思考档位 / 预算）都会抛
    明确配置错误，绝不静默降级；``resolve_at_init=True`` 用于配置校验入口
    （``create_chat_model(strict=True)``），未知模型在设置阶段即失败。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        outbound: OutboundClient | None = None,
        supports_image_input: bool = False,
        capabilities: ModelCapabilities | None = None,
        *,
        timeout: float = 600.0,
        resolve_at_init: bool = False,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._outbound = outbound
        self._destination = _destination_host(base_url)
        self._capabilities = capabilities
        self._capabilities_error = ""
        if capabilities is None:
            if resolve_at_init:
                self._capabilities = resolve_capabilities(model, base_url)
            else:
                # 惰性解析：提前取一次错误消息，供 require_features / capabilities 使用。
                try:
                    self._capabilities = resolve_capabilities(model, base_url)
                except CapabilityConfigurationError as exc:
                    self._capabilities_error = str(exc)
        self._supports_image_input = supports_image_input or bool(
            self._capabilities is not None and self._capabilities.supports_image_input
        )
        self._timeout = timeout
        # PRD 7.4：关闭 SDK 隐式重试（重试由用户选择继续后发起）。
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "local", max_retries=0, timeout=timeout
        )

    # ------------------------------------------------------------ 能力

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> ModelCapabilities:
        """已解析的能力声明；未知模型抛明确配置错误（不猜能力、不静默降级）。"""
        return self.require_capabilities()

    def require_capabilities(self) -> ModelCapabilities:
        """取能力声明；未声明 / 未知模型时抛明确配置错误。"""
        if self._capabilities is None:
            raise ProviderRequestError(
                self._capabilities_error
                or f"未解析 Provider 能力（model={self._model!r}），请显式声明能力后重试"
            )
        return self._capabilities

    @property
    def capabilities_error(self) -> str:
        """惰性解析失败原因（空串表示能力已解析成功）。"""
        return self._capabilities_error

    @property
    def supports_tools(self) -> bool:
        return bool(self._capabilities and self._capabilities.supports_tools)

    @property
    def supports_thinking(self) -> bool:
        return bool(self._capabilities and self._capabilities.supports_thinking_toggle)

    @property
    def supports_image_input(self) -> bool:
        """能力位：是否支持多模态输入（v0.2 喂图增强用，MVP 恒 False）。"""
        return self._supports_image_input

    # ------------------------------------------------------------ 请求体

    def require_features(
        self,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        reasoning_level: ReasoningLevel | str | None = None,
        image_input: bool = False,
    ) -> ProviderReasoningParams | None:
        """校验所选功能是否被 Provider 支持；缺失一次报全（PRD 9.2 不静默降级）。

        未使用任何能力项时不做能力解析（保持 v0.1 的简单请求行为），返回 ``None``。
        """
        uses_tools = bool(tools)
        if not uses_tools and reasoning_level is None and not image_input:
            return None
        if self._capabilities is None:
            raise ProviderRequestError(
                self._capabilities_error
                or f"未解析 Provider 能力（model={self._model!r}），无法校验所用功能"
            )
        capabilities = self._capabilities
        missing: list[str] = []
        if uses_tools and not capabilities.supports_tools:
            missing.append(
                f"supports_tools=False（本会话需要 {len(list(tools or []))} 个工具定义）"
            )
        if image_input and not capabilities.supports_image_input:
            missing.append("supports_image_input=False（本请求含图片输入）")
        requested: ProviderReasoningParams | None = None
        if reasoning_level is not None:
            if not capabilities.supports_thinking_toggle:
                missing.append("supports_thinking_toggle=False（不能选择思考档位）")
            else:
                try:
                    # 档位选择错误由 resolve_reasoning_params 给逐条明确错误。
                    requested = resolve_reasoning_params(
                        capabilities, reasoning_level, model=self._model
                    )
                except CapabilityConfigurationError as exc:
                    missing.append(str(exc))
        if uses_tools and capabilities.history_protocol != HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS:
            missing.append(
                "history_protocol="
                f"{capabilities.history_protocol or '（空）'!r} 不满足带 tools 会话所需的"
                f" {HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS!r}（reasoning_content 全量回传）"
            )
        if missing:
            raise ProviderRequestError(self._declaration_hint(missing))
        return requested

    def build_request(
        self,
        messages: Sequence[MessageDict],
        *,
        stream: bool,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_level: ReasoningLevel | str | None = None,
        max_tokens: int | None = None,
        reasoning_replay: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构造完整 Chat Completions 请求体（纯函数，可直接单测）。

        ``reasoning_level`` 未显式传入时不改写请求：DeepSeek 官方默认即为
        ``thinking.type=enabled`` + ``reasoning_effort=high``（PRD 9.1 默认 ``high``），
        运行层把工作区/会话选择的档位传进来即可生效。
        """
        prepared = prepare_messages(
            messages,
            capabilities=self._capabilities,
            replay_reasoning=reasoning_replay or bool(tools),
        )
        request: dict[str, Any] = {
            "model": self._model,
            "messages": prepared,
            "stream": stream,
        }
        if tools:
            request["tools"] = [dict(item) for item in tools]
            if tool_choice is not None:
                request["tool_choice"] = tool_choice
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        params = self.require_features(tools=tools, reasoning_level=reasoning_level)
        if params is not None:
            request.update(params.request_params())
        request.update(kwargs)
        return request

    # ------------------------------------------------------------ 流式

    async def stream_chat(
        self,
        messages: list[MessageDict],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_level: ReasoningLevel | str | None = None,
        max_tokens: int | None = None,
        reasoning_replay: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """流式调用云端 chat/completions；逐 chunk 提取正文与思维链增量。

        正文走 delta.content，思维链走 delta.reasoning_content（DeepSeek 推理模型
        等），分别产出 kind=content / reasoning；ChatService 把 reasoning 作为
        独立 SSE 帧下发（前端默认折叠展示），不与正文混流。
        MEDIUM-1：收尾 usage-only chunk（choices 空）跳过；MEDIUM-3：空 choices 带
        error -> 抛 ChatProviderError。出网审计同前。

        未注入 ``tools`` 时不触发能力校验（保持 v0.1 既有行为）。
        """
        request = self.build_request(
            messages,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_level=reasoning_level if tools or reasoning_level is not None else None,
            max_tokens=max_tokens,
            reasoning_replay=reasoning_replay,
            **kwargs,
        )
        started_at = _utcnow() if self._outbound is not None else None
        try:
            response = await self._client.chat.completions.create(**request)
            async for chunk in response:
                if not chunk.choices:  # usage-only 收尾 chunk（MEDIUM-1）
                    error = getattr(chunk, "error", None)
                    if error is not None:  # MEDIUM-3：流式中途报错
                        message = getattr(error, "message", None) or str(error)
                        raise ChatProviderError(f"对话服务流式中止: {message}")
                    continue
                delta = chunk.choices[0].delta
                token = _attr(delta, "content") or ""
                if token:
                    yield ChatChunk("content", token)
                reasoning = _attr(delta, "reasoning_content") or ""
                if reasoning:
                    yield ChatChunk("reasoning", reasoning)
        except BaseException:
            # 异常含 GeneratorExit（SSE 客户端断开触发 aclose）：已发生出网也要记审计
            self._record_outbound(started_at, status=None)
            raise
        else:
            self._record_outbound(started_at, status=200)

    async def stream_result(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk | ToolCallFragment]:
        """流式 + 完整结果。

        先产出 ``ChatChunk``（正文/思维链增量）与 ``ToolCallFragment``（工具参数
        分片），**最后**产出一个 :class:`StreamResult`。分片不预先解析；完整 JSON
        到达并校验通过后才在结果里 ``complete=True``（PRD 9.2）。
        """
        request = self.build_request(messages, stream=True, **kwargs)
        assembler = ToolCallAssembler()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        usage_payload: Any = None
        provider_request_id = ""
        started_at = _utcnow() if self._outbound is not None else None
        try:
            response = await self._client.chat.completions.create(**request)
            provider_request_id = provider_request_id_from_headers(
                getattr(response, "headers", None)
            )
            async for chunk in response:
                if not chunk.choices:  # usage-only 收尾 chunk
                    error = getattr(chunk, "error", None)
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        usage_payload = usage
                    if error is not None:
                        message = getattr(error, "message", None) or str(error)
                        raise ChatProviderError(f"对话服务流式中止: {message}")
                    continue
                if not provider_request_id:
                    provider_request_id = str(getattr(chunk, "id", "") or "")
                choice = chunk.choices[0]
                delta = choice.delta
                token = _attr(delta, "content") or ""
                if token:
                    content_parts.append(token)
                    yield ChatChunk("content", token)
                reasoning = _attr(delta, "reasoning_content") or ""
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ChatChunk("reasoning", reasoning)
                for raw_call in _attr(delta, "tool_calls") or []:
                    fragment = ToolCallFragment.from_delta(raw_call)
                    assembler.add(fragment)
                    yield fragment
                if choice.finish_reason:
                    finish_reason = str(choice.finish_reason)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    usage_payload = usage
        except BaseException:
            self._record_outbound(started_at, status=None)
            raise
        else:
            self._record_outbound(started_at, status=200)
        response_obj = _assemble_response(
            model=self._model,
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=assembler.finish(),
            finish_reason=finish_reason,
            usage_payload=usage_payload,
            provider_request_id=provider_request_id,
            notes=assembler.failures(),
        )
        yield StreamResult.from_response(response_obj)

    # ------------------------------------------------------------ 非流式

    async def complete(
        self,
        messages: list[MessageDict],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_level: ReasoningLevel | str | None = None,
        max_tokens: int | None = None,
        reasoning_replay: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """非流式补全，返回完整响应对象（PRD 9.2）。"""
        request = self.build_request(
            messages,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_level=reasoning_level if tools or reasoning_level is not None else None,
            max_tokens=max_tokens,
            reasoning_replay=reasoning_replay,
            **kwargs,
        )
        started_at = _utcnow() if self._outbound is not None else None
        try:
            raw = await self._client.chat.completions.create(**request)
        except BaseException:
            self._record_outbound(started_at, status=None)
            raise
        self._record_outbound(started_at, status=200)
        return response_from_completion(self._model, raw)

    # ------------------------------------------------------------ 审计

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

    def _declaration_hint(self, missing: Sequence[str]) -> str:
        """拼装明确配置错误消息：问题 + 环境变量 + 可复制的能力声明模板。"""
        lines = [
            f"对话 Provider 不支持本次请求所需功能（model={self._model!r}，不静默降级）:"
        ]
        lines.extend(f"- {item}" for item in missing)
        lines.append(
            "若该端点确实支持，请显式声明能力：设置环境变量 EVERYTHING_RAG_CHAT_CAPABILITIES，"
            "包含 " + ", ".join(_REQUIRED_DECLARATION) + "；例如 "
            '{"supports_tools": true, "supports_thinking_toggle": false, "reasoning_levels": ["off"], '
            '"supports_image_input": false, "context_limit": 65536, "max_output_tokens": 8192, '
            '"history_protocol": "openai-chat-completions"}'
        )
        return "\n".join(lines)


def features_from_request(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """从请求关键字里判定本次用到的能力项（供能力校验与日志使用，纯函数）。"""
    tools = kwargs.get("tools")
    reasoning_level = kwargs.get("reasoning_level")
    return {
        "uses_tools": bool(tools),
        "tool_count": len(list(tools)) if tools else 0,
        "reasoning_level": (
            reasoning_level.value
            if isinstance(reasoning_level, ReasoningLevel)
            else reasoning_level
        ),
        "image_input": bool(kwargs.get("image_input")),
        "max_tokens": kwargs.get("max_tokens"),
    }


def replay_reasoning_content(
    capabilities: ModelCapabilities,
    *,
    has_tools: bool,
) -> bool:
    """是否需要全量回传此前 assistant 的 ``reasoning_content``（PRD 9.2 末段）。

    仅当请求带 ``tools`` 且该 Provider 声明了 DeepSeek 专用协议时为真；其他
    OpenAI 兼容端点的 ``reasoning_content`` 会被忽略，因此不携带。
    """
    return has_tools and capabilities.history_protocol == HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS


def prepare_messages(
    messages: Sequence[MessageDict],
    *,
    capabilities: ModelCapabilities,
    replay_reasoning: bool,
) -> list[MessageDict]:
    """按协议整理待发送历史（纯函数，不改原对象）。

    - 保留此前 assistant 消息的完整 ``reasoning_content``（含未调用工具的轮次）；
    - 工具结果消息必须带 ``tool_call_id``，否则抛明确错误（PRD 9.2）；
    - 独立子 Agent 把历史摘要当参考数据：只接受 ``user``/``system`` 角色，
      不得带 ``tool_calls``，避免制造未发生的 assistant 工具调用（PRD 9.2 末段）。
    """
    prepared: list[MessageDict] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        item: MessageDict = {key: value for key, value in message.items() if key != "self_authored"}
        if role == "tool":
            call_id = item.get("tool_call_id")
            if not call_id:
                raise ProviderRequestError(
                    f"第 {index} 条 tool 消息缺少 tool_call_id；每个工具结果必须使用对应调用的 id"
                )
        elif role == "assistant":
            if not replay_reasoning:
                item.pop("reasoning_content", None)
        elif role not in {"system", "user", "developer"}:
            raise ProviderRequestError(
                f"第 {index} 条消息 role={role!r} 不在 Provider 协议允许的范围内"
            )
        prepared.append(item)
    return prepared


def reference_only_messages(
    summary: str,
    *,
    role: str = "user",
) -> MessageDict:
    """把历史摘要包装成**参考数据**消息（独立子 Agent 上下文隔离，PRD 9.2 末段）。

    摘要只能是 user/system 文本，绝不包装成 assistant 工具调用。
    """
    if role not in {"user", "system"}:
        raise ProviderRequestError(f"历史摘要只能作为 user/system 参考数据，收到 role={role!r}")
    return {"role": role, "content": summary}


def response_from_completion(model: str, raw: Any) -> ModelResponse:
    """把非流式 SDK 响应映射为 :class:`ModelResponse`（缺字段不猜）。"""
    choices = _attr(raw, "choices") or []
    message = _attr(choices[0], "message") if choices else None
    finish_reason = _attr(choices[0], "finish_reason") if choices else None
    return _assemble_response(
        model=str(_attr(raw, "model") or model),
        content=str(_attr(message, "content") or ""),
        reasoning_content=str(_attr(message, "reasoning_content") or ""),
        tool_calls=tool_calls_from_message(message),
        finish_reason=str(finish_reason) if finish_reason else None,
        usage_payload=_attr(raw, "usage"),
        provider_request_id=str(_attr(raw, "id") or ""),
        notes=[],
    )


def _assemble_response(
    *,
    model: str,
    content: str,
    reasoning_content: str,
    tool_calls: list[ToolCall],
    finish_reason: str | None,
    usage_payload: Any,
    provider_request_id: str,
    notes: list[str],
) -> ModelResponse:
    """组装完整响应；工具参数不完整时降级为 ``INCOMPLETE`` 并记录原因（PRD 9.2）。"""
    status = classify_finish_reason(finish_reason)
    if any(not call.complete for call in tool_calls):
        status = ResponseStatus.INCOMPLETE
    return ModelResponse(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        status=status,
        usage=usage_from_payload(usage_payload),
        provider_request_id=provider_request_id,
        model=model,
        notes=list(notes),
    )


class UnconfiguredChatModel:
    """配置缺失时的占位模型：stream 阶段抛 ChatProviderError（构造零联网）。"""

    def __init__(self, error: ChatProviderError) -> None:
        self._error = error

    @property
    def capabilities(self) -> ModelCapabilities:
        return unavailable()

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def supports_thinking(self) -> bool:
        return False

    @property
    def supports_image_input(self) -> bool:
        return False

    def stream_chat(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        return _ErrorStream(self._error)

    def stream_result(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk | ToolCallFragment]:
        return _ErrorStream(self._error)

    async def complete(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> ModelResponse:
        raise self._error


def unavailable() -> ModelCapabilities:
    """未配置态的保守能力声明：不声明任何可用功能（fail-closed）。"""
    return ModelCapabilities(context_limit=1, max_output_tokens=1)


class _ErrorStream:
    """首次迭代即抛出给定异常的空 async 迭代器。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __aiter__(self) -> _ErrorStream:
        return self

    async def __anext__(self) -> Any:
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


def _attr(obj: Any, name: str) -> Any:
    """安全取属性；None / dict 都支持（SDK 对象与 dict 混用的兼容层）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def create_chat_model(
    settings: Settings,
    outbound: OutboundClient | None = outbound_client,  # 缺省接全局单例，生产审计不静默失效
) -> ChatModel:
    """从 settings 构造对话模型；配置缺失 raise ChatProviderError（零联网）。

    缺失项消息列出对应的环境变量名（base_url / model），便于用户补齐配置。
    outbound 缺省为全局 outbound_client：真实出网自动进入网络审计（/api/audit、
    /api/status.privacy.outbound_state 才如实上报，而非恒为 local-only）。
    未知模型的能力在构造时即要求显式声明（PRD 9.2，不给猜测值）。
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
