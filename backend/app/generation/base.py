"""问答生成层接口（对齐技术选型决策文档 T1/T3；V1.2.0 完整模型协议 PRD 9.2）。

对话模型走 OpenAI 兼容可插拔（默认本地 Ollama/llama.cpp，可选云端）；
流式输出 SSE：POST /api/chat + 前端 fetch/ReadableStream，帧契约
meta/reasoning/token/tool/done/error。
识图模型复用同一底层 client，见 VisionModel。

V1.2.0 在既有 ``ChatModel.stream_chat`` 之外补齐**完整模型响应**（PRD 9.2）：

- ``ModelResponse``：``content`` / ``reasoning_content`` / ``tool_calls`` /
  ``finish_reason`` + ``status`` / ``usage`` / ``provider_request_id``；
- ``ToolCallAssembler``：把流式 ``tool_calls`` 分片按调用 ``index`` 汇集，
  完整 JSON 到达并通过 Pydantic 校验后才标记 ``complete=True``；
- ``StreamResult``：一次流式调用的完整结果，含 ``tool_round_complete``
  （本组工具结果是否已完整返回，决定能否安排上下文整理，PRD 9.3）。

隐私约束：本模块只放纯数据结构与协议，不做任何出网。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, NamedTuple, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from app.agent.contracts import ModelCapabilities, ToolCall

__all__ = [
    "AUTHORED_BY_USER",
    "ChatChunk",
    "ChatModel",
    "CompletionUsage",
    "MessageDict",
    "ModelResponse",
    "ProviderMessageDict",
    "ResponseStatus",
    "StreamResult",
    "StreamStatus",
    "ToolCallAssembler",
    "ToolCallFragment",
    "VisionModel",
    "classify_finish_reason",
    "provider_request_id_from_headers",
    "tool_call_from_message",
    "tool_calls_from_message",
    "usage_from_payload",
]

#: 消息字典（OpenAI Chat Completions 形状，允许 ``reasoning_content`` 等 DeepSeek 扩展）。
MessageDict: TypeAlias = dict[str, Any]

#: Provider 协议消息（落 ``provider_messages.protocol`` 的形状，PRD 13.1）。
ProviderMessageDict: TypeAlias = dict[str, Any]

#: 消息来源标记：只允许由用户/工具真实产生，禁止伪造 assistant 工具调用（PRD 9.2）。
AUTHORED_BY_USER = "user"


class ChatChunk(NamedTuple):
    """流式产出单元：kind 区分正文与思维链，text 为增量文本。

    对齐 DeepSeek 推理模型：思维链在 delta.reasoning_content，正文在
    delta.content；ChatService 把 reasoning 单独下发为 SSE ``reasoning`` 帧
    （前端默认折叠展示），正文仍走引用白名单清洗。
    """

    kind: str  # "content" | "reasoning"
    text: str


class StreamStatus(StrEnum):
    """一次流式调用的真实状态（PRD 9.2：结束原因保存准确状态）。"""

    STREAMING = "streaming"  # 尚未结束
    WAITING_FOR_TOOLS = "waiting_for_tools"  # 模型要求调用工具，等待工具结果
    COMPLETED = "completed"  # 自然结束
    TRUNCATED = "truncated"  # finish_reason=length：未完整生成
    FILTERED = "filtered"  # finish_reason=content_filter
    ABORTED = "aborted"  # finish_reason=aborted
    RESOURCE_EXHAUSTED = "resource_exhausted"  # finish_reason=insufficient_system_resource
    FAILED = "failed"  # 流中断且无 finish_reason（内容不完整）


class ResponseStatus(StrEnum):
    """Provider 响应的完整度状态，`succeeded` 的唯一判据（PRD 9.2 末段）。"""

    COMPLETE = "complete"  # finish_reason=stop
    TOOL_CALLS = "tool_calls"  # finish_reason=tool_calls
    TRUNCATED = "length"  # finish_reason=length
    FILTERED = "content_filter"
    ABORTED = "aborted"
    RESOURCE_EXHAUSTED = "insufficient_system_resource"
    INCOMPLETE = "incomplete"  # 缺 finish_reason / 工具参数未完整到达

    @property
    def succeeded(self) -> bool:
        """未完整生成的内容不得标记成功（PRD 9.2）。"""
        return self in {ResponseStatus.COMPLETE, ResponseStatus.TOOL_CALLS}


_FINISH_REASON_STATUS: dict[str, ResponseStatus] = {
    "stop": ResponseStatus.COMPLETE,
    "tool_calls": ResponseStatus.TOOL_CALLS,
    "length": ResponseStatus.TRUNCATED,
    "content_filter": ResponseStatus.FILTERED,
    "aborted": ResponseStatus.ABORTED,
    "insufficient_system_resource": ResponseStatus.RESOURCE_EXHAUSTED,
}

_STREAM_STATUS_BY_RESPONSE: dict[ResponseStatus, StreamStatus] = {
    ResponseStatus.COMPLETE: StreamStatus.COMPLETED,
    ResponseStatus.TOOL_CALLS: StreamStatus.WAITING_FOR_TOOLS,
    ResponseStatus.TRUNCATED: StreamStatus.TRUNCATED,
    ResponseStatus.FILTERED: StreamStatus.FILTERED,
    ResponseStatus.ABORTED: StreamStatus.ABORTED,
    ResponseStatus.RESOURCE_EXHAUSTED: StreamStatus.RESOURCE_EXHAUSTED,
    ResponseStatus.INCOMPLETE: StreamStatus.FAILED,
}


def classify_finish_reason(finish_reason: str | None) -> ResponseStatus:
    """``finish_reason`` -> 状态；未知或缺失一律归为 ``INCOMPLETE``（不当作成功）。"""
    if not finish_reason:
        return ResponseStatus.INCOMPLETE
    return _FINISH_REASON_STATUS.get(str(finish_reason).strip().lower(), ResponseStatus.INCOMPLETE)


class CompletionUsage(BaseModel):
    """一次调用的 token 用量（PRD 9.3：含缓存命中与推理 token）。"""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    prompt_cache_hit_tokens: int = Field(default=0, ge=0)
    prompt_cache_miss_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)


def usage_from_payload(payload: Any) -> CompletionUsage:
    """从 SDK 的 ``usage`` 对象/dict 提取用量；缺失字段填 0，绝不估算。"""
    if payload is None:
        return CompletionUsage()
    prompt_tokens_details = _attr(payload, "prompt_tokens_details")
    completion_tokens_details = _attr(payload, "completion_tokens_details")
    hit = _attr(prompt_tokens_details, "prompt_cache_hit_tokens")
    if hit is None:
        hit = _attr(prompt_tokens_details, "cached_tokens")
    return CompletionUsage(
        prompt_tokens=_as_int(_attr(payload, "prompt_tokens")),
        completion_tokens=_as_int(_attr(payload, "completion_tokens")),
        total_tokens=_as_int(_attr(payload, "total_tokens")),
        prompt_cache_hit_tokens=_as_int(hit),
        prompt_cache_miss_tokens=_as_int(_attr(prompt_tokens_details, "prompt_cache_miss_tokens")),
        reasoning_tokens=_as_int(_attr(completion_tokens_details, "reasoning_tokens")),
    )


class ModelResponse(BaseModel):
    """一次模型调用的完整响应对象（PRD 9.2 / 契约 C10）。"""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    usage_available: bool = False
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    status: ResponseStatus = ResponseStatus.INCOMPLETE
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
    provider_request_id: str = ""
    model: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """只有完整生成（含完整工具参数）才算成功（PRD 9.2）。"""
        return self.status.succeeded and all(call.complete for call in self.tool_calls)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolCallFragment(BaseModel):
    """流式 ``tool_calls`` 的一个分片（``arguments`` 只是片段，不能当 JSON 用）。"""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(default=0, ge=0)
    call_id: str = ""
    name: str = ""
    arguments_fragment: str = ""

    @classmethod
    def from_delta(cls, delta: Any) -> ToolCallFragment:
        """从 SDK 的 stream delta tool_call 对象构造分片（缺失字段留空）。"""
        function = _attr(delta, "function")
        return cls(
            index=_as_int(_attr(delta, "index")),
            call_id=str(_attr(delta, "id") or ""),
            name=str(_attr(function, "name") or ""),
            arguments_fragment=str(_attr(function, "arguments") or ""),
        )


class ToolCallAssembler:
    """按调用 ``index`` 汇集流式工具参数，完整 JSON 校验通过后才算完成（PRD 9.2）。

    ``finish()`` 对每个调用的参数做 JSON 解析 + ``ToolCall``（Pydantic）校验；
    不完整的分片保留为 ``complete=False`` 并记录失败原因（不执行、不标成功）。
    """

    def __init__(self) -> None:
        self._order: list[int] = []
        self._buffers: dict[int, dict[str, str]] = {}
        self._failures: dict[int, str] = {}

    def add(self, fragment: ToolCallFragment) -> None:
        """追加一个分片；同名调用的 ``id``/``name`` 只取首个非空值。"""
        index = fragment.index
        buffer = self._buffers.get(index)
        if buffer is None:
            buffer = {"call_id": "", "name": "", "arguments": ""}
            self._buffers[index] = buffer
            self._order.append(index)
        if fragment.call_id and not buffer["call_id"]:
            buffer["call_id"] = fragment.call_id
        if fragment.name:
            buffer["name"] = fragment.name
        if fragment.arguments_fragment:
            buffer["arguments"] += fragment.arguments_fragment

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._order)

    def call_at(self, index: int) -> ToolCall:
        """当前已汇集到的调用（参数未完整时 ``complete=False``，并记录失败原因）。"""
        buffer = self._buffers[index]
        raw = buffer["arguments"]
        arguments: dict[str, Any] = {}
        complete = False
        failure = self._failures.get(index)
        if not raw.strip():
            failure = failure or "工具参数未完整到达（可能被截断）"
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                failure = failure or f"工具参数不是完整 JSON: {exc.msg}（位置 {exc.pos}）"
            else:
                if isinstance(parsed, dict):
                    arguments = parsed
                    complete = True
                else:
                    failure = failure or "工具参数 JSON 必须是对象（不接受数组或标量）"
        if not complete:
            self._failures[index] = failure or "工具参数未完整"
        return ToolCall(
            index=index,
            call_id=buffer["call_id"] or f"call_{index}",
            name=buffer["name"],
            arguments=arguments,
            raw_arguments=raw,
            complete=complete,
        )

    def finish(self) -> list[ToolCall]:
        """结束汇集：返回全部调用，未完整者为 ``complete=False``（调用方据此失败）。"""
        return [self.call_at(index) for index in self._order]

    def failures(self) -> list[str]:
        """未完整调用的失败原因（写入 ``ModelResponse.notes``，不泄露参数正文）。"""
        return [self._failures[index] for index in self._order if index in self._failures]


class StreamResult(BaseModel):
    """一次流式调用的完整结果（PRD 9.2 / 9.3）。"""

    model_config = ConfigDict(extra="forbid")

    response: ModelResponse = Field(default_factory=ModelResponse)
    stream_status: StreamStatus = StreamStatus.STREAMING
    content_complete: bool = False
    reasoning_complete: bool = False
    tool_calls_complete: bool = False

    @property
    def succeeded(self) -> bool:
        return self.content_complete and self.tool_calls_complete and self.response.succeeded

    @property
    def tool_round_complete(self) -> bool:
        """本组工具结果是否已完整返回（未完整时不得安排上下文整理，PRD 9.3）。

        只有「模型要求调工具**且**全部工具参数已完整到达」才为真：中断、截断、
        参数不完整都必须先补齐这一组工具调用与结果，再增加 ``agent_context_epoch``。
        """
        if not self.response.wants_tools:
            return False
        return self.tool_calls_complete and self.response.status is ResponseStatus.TOOL_CALLS

    @property
    def failure_reasons(self) -> list[str]:
        """未完整生成的原因（含工具参数问题），供运行层记录与展示。"""
        reasons = list(self.response.notes)
        if not self.tool_calls_complete and not reasons:
            reasons.append("工具调用参数未完整到达（不得执行、不得标成功）")
        if self.stream_status is StreamStatus.FAILED and not reasons:
            reasons.append("流式中断且未收到结束原因，内容不完整")
        return reasons

    @classmethod
    def from_response(cls, response: ModelResponse) -> StreamResult:
        stream_status = _STREAM_STATUS_BY_RESPONSE[response.status]
        succeeded = response.succeeded
        return cls(
            response=response,
            stream_status=stream_status,
            content_complete=succeeded or response.status is ResponseStatus.TOOL_CALLS,
            reasoning_complete=succeeded or response.status is ResponseStatus.TOOL_CALLS,
            tool_calls_complete=all(call.complete for call in response.tool_calls),
        )


def tool_call_from_message(raw: Any) -> ToolCall:
    """把非流式响应的一个 tool_call 转成契约对象（参数按官方文本原样解析）。"""
    function = _attr(raw, "function")
    raw_arguments = str(_attr(function, "arguments") or "")
    arguments: dict[str, Any] = {}
    complete = False
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        arguments = parsed
        complete = True
    return ToolCall(
        index=_as_int(_attr(raw, "index")),
        call_id=str(_attr(raw, "id") or ""),
        name=str(_attr(function, "name") or ""),
        arguments=arguments,
        raw_arguments=raw_arguments,
        complete=complete,
    )


def tool_calls_from_message(message: Any) -> list[ToolCall]:
    """非流式响应里的全部 tool_calls（缺失返回空列表）。"""
    raw_calls = _attr(message, "tool_calls")
    if not isinstance(raw_calls, (list, tuple)):
        return []
    return [tool_call_from_message(item) for item in raw_calls]


def provider_request_id_from_headers(headers: Any) -> str:
    """从响应头提取提供商请求 ID（``x-request-id`` / ``x-ds-request-id``）。"""
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    for key in ("x-request-id", "x-ds-request-id", "request-id"):
        try:
            value = getter(key)
        except Exception:  # noqa: BLE001 -- 头部形状不可控，缺失即视为无
            value = None
        if value:
            return str(value)
    return ""


class ChatModel(Protocol):
    """统一对话模型接口（支持流式；V1.2.0 增加完整响应与非流式补全）。"""

    @property
    def capabilities(self) -> ModelCapabilities:
        """能力声明（PRD 9.2）；不支持所选功能时调用方返回明确配置错误。"""
        ...

    @property
    def supports_tools(self) -> bool:
        ...

    @property
    def supports_thinking(self) -> bool:
        ...

    def stream_chat(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """provider 可产出正文与 reasoning；是否外发由上层安全策略决定。
        若上游为 OpenAI 兼容 chat/completions stream，在此层映射为 ChatChunk。
        """
        ...

    def stream_result(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk | ToolCallFragment]:
        """流式 + 完整结果：正文/思维链增量、工具分片，收尾给出 ``StreamResult``。"""
        ...

    async def complete(
        self,
        messages: list[MessageDict],
        **kwargs: Any,
    ) -> ModelResponse:
        """非流式请求，返回完整响应对象。"""
        ...

    @property
    def supports_image_input(self) -> bool:
        """能力位：是否支持多模态输入（v0.2 喂图增强用，MVP 恒 False）。"""
        ...


class VisionModel(Protocol):
    """识图模型接口（方案 B：图片 -> 文字描述，入库一次性、非流式）。"""

    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """把图片转为文字描述（走与 ChatModel 相同的底层 OpenAI 兼容 client）。"""
        ...


def _attr(obj: Any, name: str) -> Any:
    """安全取属性；None / dict 都支持（SDK 对象与 dict 混用的兼容层）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value: Any) -> int:
    """把 token 计数等转成非负 int；无法转换时给 0（不猜测、不伪造）。"""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
