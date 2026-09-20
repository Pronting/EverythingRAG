"""模型 provider V1.2.0 完整协议测试（PRD 9.1 / 9.2 / 7.4）。Owner: T03。

零外呼：全部注入假 OpenAI 兼容客户端（只记录请求体、返回构造好的响应对象），
并额外用 socket 阻断用例实证整条链路不发起真实网络请求。
"""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from app.generation import providers
from app.generation.base import (
    ResponseStatus,
    StreamResult,
    StreamStatus,
    ToolCallAssembler,
    ToolCallFragment,
    classify_finish_reason,
)
from app.generation.capabilities import (
    HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS,
    CapabilityConfigurationError,
    resolve_capabilities,
)
from app.generation.providers import (
    ChatProviderError,
    OpenAICompatChatModel,
    ProviderRequestError,
    prepare_messages,
    reference_only_messages,
    replay_reasoning_content,
)

OFFICIAL = "https://api.deepseek.com"
COMPAT = "https://compat.example.test/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在本工作区检索相关片段",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


# ---------------------------------------------------------------- 假客户端


class FakeDelta:
    def __init__(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, delta: FakeDelta | None, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(
        self,
        choices: list[FakeChoice],
        *,
        chunk_id: str = "",
        usage: Any = None,
        error: Any = None,
    ) -> None:
        self.choices = choices
        self.id = chunk_id
        self.usage = usage
        self.error = error


class FakeStream:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> FakeChunk:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class SpyCompletions:
    """记录全部请求体的假 ``chat.completions``（零网络）。"""

    def __init__(self, *, stream: FakeStream | None = None, completion: Any = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._stream = stream
        self._completion = completion

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            return self._stream or FakeStream([])
        return self._completion


class SpyClient:
    def __init__(self, completions: SpyCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> SpyCompletions:
    """把 AsyncOpenAI 换成间谍客户端，并记录构造参数（含 max_retries）。"""
    completions = SpyCompletions()
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _install(kwargs, completions))
    return completions


_CONSTRUCTED: dict[str, Any] = {}


def _install(kwargs: dict[str, Any], completions: SpyCompletions) -> SpyClient:
    _CONSTRUCTED.clear()
    _CONSTRUCTED.update(kwargs)
    return SpyClient(completions)


def _model(monkeypatch: pytest.MonkeyPatch, completions: SpyCompletions, **kwargs: Any):
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kw: _install(kw, completions))
    return OpenAICompatChatModel(
        base_url=kwargs.pop("base_url", OFFICIAL),
        model=kwargs.pop("model", "deepseek-flash"),
        **kwargs,
    )


def _tool_call_delta(index: int, call_id: str = "", name: str = "", arguments: str = "") -> Any:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------- 1. 四档请求体


@pytest.mark.parametrize(
    ("level", "expected_extra_body", "expected_effort"),
    [
        ("off", {"thinking": {"type": "disabled"}}, None),
        ("low", {"thinking": {"type": "enabled"}}, "low"),
        ("high", {"thinking": {"type": "enabled"}}, "high"),
        ("max", {"thinking": {"type": "enabled"}}, "max"),
    ],
)
async def test_four_reasoning_levels_in_request_body(
    monkeypatch: pytest.MonkeyPatch,
    level: str,
    expected_extra_body: dict[str, Any],
    expected_effort: str | None,
) -> None:
    """四档映射后的请求体：extra_body.thinking + reasoning_effort（PRD 9.1）。"""
    completions = SpyCompletions(stream=FakeStream([]))
    model = _model(monkeypatch, completions)
    async for _ in model.stream_chat(
        [{"role": "user", "content": "q"}], reasoning_level=level
    ):
        pass
    request = completions.requests[0]
    assert request["extra_body"] == expected_extra_body
    if expected_effort is None:
        assert "reasoning_effort" not in request
    else:
        assert request["reasoning_effort"] == expected_effort
    assert request["model"] == "deepseek-flash"
    assert request["stream"] is True


async def test_default_high_with_tools_keeps_official_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未传档位时不改写请求：官方默认即 thinking=enabled + effort=high（PRD 18.1）。"""
    completions = SpyCompletions(
        stream=FakeStream([FakeChunk([FakeChoice(FakeDelta("ok"), "stop")])])
    )
    model = _model(monkeypatch, completions)
    async for _ in model.stream_chat([{"role": "user", "content": "q"}], tools=TOOLS):
        pass
    request = completions.requests[0]
    assert "thinking" not in request
    assert "reasoning_effort" not in request
    assert request["tools"] == TOOLS


async def test_request_body_includes_tool_choice_and_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = SpyCompletions(stream=FakeStream([]))
    model = _model(monkeypatch, completions)
    async for _ in model.stream_chat(
        [{"role": "user", "content": "q"}],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=4096,
        reasoning_level="high",
    ):
        pass
    request = completions.requests[0]
    assert request["tool_choice"] == "auto"
    assert request["max_tokens"] == 4096
    assert request["reasoning_effort"] == "high"


def test_sdk_implicit_retry_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """PRD 7.4：SDK 隐式重试关闭（max_retries=0）。"""
    completions = SpyCompletions()
    _model(monkeypatch, completions)
    assert _CONSTRUCTED["max_retries"] == 0


# ---------------------------------------------------------------- 2. 流式工具分片汇集


async def test_stream_assembles_tool_call_fragments_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流式 tool_calls 按 index 汇集分片参数，完整 JSON 通过校验后才算完成（PRD 9.2）。"""
    stream = FakeStream(
        [
            FakeChunk([FakeChoice(FakeDelta(reasoning_content="想一想"), None)]),
            FakeChunk(
                [
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                _tool_call_delta(0, "call_a", "search_knowledge", '{"query": '),
                            ]
                        ),
                        None,
                    )
                ]
            ),
            FakeChunk(
                [
                    FakeChoice(
                        FakeDelta(tool_calls=[_tool_call_delta(0, "", "", '"退款"}')]),
                        None,
                    )
                ]
            ),
            FakeChunk([FakeChoice(FakeDelta(), "tool_calls")], chunk_id="req-1"),
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)

    fragments: list[ToolCallFragment] = []
    result = None
    async for item in model.stream_result(
        [{"role": "user", "content": "退款政策"}], tools=TOOLS
    ):
        if isinstance(item, ToolCallFragment):
            fragments.append(item)
        else:
            result = item

    assert result is not None
    assert [fragment.index for fragment in fragments] == [0, 0]
    calls = result.response.tool_calls
    assert len(calls) == 1
    assert calls[0].call_id == "call_a"
    assert calls[0].name == "search_knowledge"
    assert calls[0].arguments == {"query": "退款"}
    assert calls[0].raw_arguments == '{"query": "退款"}'
    assert calls[0].complete is True
    assert result.response.finish_reason == "tool_calls"
    assert result.response.status is ResponseStatus.TOOL_CALLS
    assert result.response.succeeded is True
    assert result.stream_status is StreamStatus.WAITING_FOR_TOOLS
    assert result.tool_round_complete is True
    assert result.response.provider_request_id == "req-1"


async def test_stream_assembles_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并行工具调用按各自 index 独立汇集（不串参数）。"""
    stream = FakeStream(
        [
            FakeChunk(
                [
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                _tool_call_delta(0, "call_a", "search_knowledge", '{"query": "a"}'),
                                _tool_call_delta(1, "call_b", "search_history", '{"query": "b"}'),
                            ]
                        ),
                        "tool_calls",
                    )
                ]
            )
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    result = None
    async for item in model.stream_result([{"role": "user", "content": "q"}], tools=TOOLS):
        if not isinstance(item, ToolCallFragment):
            result = item
    assert result is not None
    assert [call.name for call in result.response.tool_calls] == [
        "search_knowledge",
        "search_history",
    ]
    assert [call.arguments for call in result.response.tool_calls] == [
        {"query": "a"},
        {"query": "b"},
    ]
    assert all(call.complete for call in result.response.tool_calls)


async def test_incomplete_tool_arguments_are_not_marked_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数 JSON 未完整到达 -> complete=False、状态 INCOMPLETE、绝不标记成功（PRD 9.2）。"""
    stream = FakeStream(
        [
            FakeChunk(
                [
                    FakeChoice(
                        FakeDelta(tool_calls=[_tool_call_delta(0, "call_a", "search_knowledge", '{"que')]),
                        "tool_calls",
                    )
                ]
            )
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    result = None
    async for item in model.stream_result([{"role": "user", "content": "q"}], tools=TOOLS):
        if not isinstance(item, ToolCallFragment):
            result = item
    assert result is not None
    call = result.response.tool_calls[0]
    assert call.complete is False
    assert call.arguments == {}
    assert call.raw_arguments == '{"que'
    assert result.response.status is ResponseStatus.INCOMPLETE
    assert result.response.succeeded is False
    assert result.succeeded is False
    assert result.tool_calls_complete is False
    assert result.tool_round_complete is False
    assert result.failure_reasons and "JSON" in result.failure_reasons[0]
    assert result.response.notes and "JSON" in result.response.notes[0]


def test_tool_call_assembler_flags_non_object_arguments() -> None:
    """参数 JSON 是数组/标量 -> 不算完整（工具参数必须是对象）。"""
    assembler = ToolCallAssembler()
    assembler.add(ToolCallFragment(index=0, call_id="c", name="t", arguments_fragment="[1,2]"))
    calls = assembler.finish()
    assert calls[0].complete is False
    assert "对象" in assembler.failures()[0]


def test_tool_call_assembler_flags_truncated_arguments() -> None:
    """参数分片只到一半 -> 明确不完整（不得执行、不得标成功）。"""
    assembler = ToolCallAssembler()
    assembler.add(
        ToolCallFragment(index=0, call_id="c", name="t", arguments_fragment='{"query": "退')
    )
    calls = assembler.finish()
    assert calls[0].complete is False
    assert "JSON" in assembler.failures()[0]


def test_tool_call_assembler_keeps_first_call_id_and_full_name() -> None:
    """分片里重复出现 id/name 时只取首个非空值（官方按 index 汇集语义）。"""
    assembler = ToolCallAssembler()
    assembler.add(ToolCallFragment(index=0, call_id="call_1", name="search_knowledge"))
    assembler.add(
        ToolCallFragment(index=0, call_id="call_1", name="search_knowledge", arguments_fragment="{}")
    )
    call = assembler.finish()[0]
    assert call.call_id == "call_1"
    assert call.name == "search_knowledge"
    assert call.complete is True


# ---------------------------------------------------------------- 3. finish_reason 状态


@pytest.mark.parametrize(
    ("finish_reason", "status", "stream_status", "succeeded"),
    [
        ("stop", ResponseStatus.COMPLETE, StreamStatus.COMPLETED, True),
        ("tool_calls", ResponseStatus.TOOL_CALLS, StreamStatus.WAITING_FOR_TOOLS, True),
        ("length", ResponseStatus.TRUNCATED, StreamStatus.TRUNCATED, False),
        ("aborted", ResponseStatus.ABORTED, StreamStatus.ABORTED, False),
        ("content_filter", ResponseStatus.FILTERED, StreamStatus.FILTERED, False),
        (
            "insufficient_system_resource",
            ResponseStatus.RESOURCE_EXHAUSTED,
            StreamStatus.RESOURCE_EXHAUSTED,
            False,
        ),
        (None, ResponseStatus.INCOMPLETE, StreamStatus.FAILED, False),
        ("unknown_reason", ResponseStatus.INCOMPLETE, StreamStatus.FAILED, False),
    ],
)
async def test_finish_reason_status_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str | None,
    status: ResponseStatus,
    stream_status: StreamStatus,
    succeeded: bool,
) -> None:
    """``length`` / ``aborted`` 等结束原因保存准确状态，未完整内容不标成功（PRD 9.2）。"""
    stream = FakeStream(
        [FakeChunk([FakeChoice(FakeDelta("部分内容"), finish_reason)], chunk_id="req-9")]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    result = None
    async for item in model.stream_result([{"role": "user", "content": "q"}]):
        if not isinstance(item, ToolCallFragment):
            result = item
    assert result is not None
    assert result.response.status is status
    assert result.stream_status is stream_status
    assert result.response.succeeded is succeeded
    assert result.succeeded is succeeded
    assert result.response.finish_reason == finish_reason


def test_classify_finish_reason_is_case_insensitive_and_strict() -> None:
    assert classify_finish_reason(" STOP ") is ResponseStatus.COMPLETE
    assert classify_finish_reason("") is ResponseStatus.INCOMPLETE
    assert classify_finish_reason("weird") is ResponseStatus.INCOMPLETE


async def test_stream_records_usage_and_cache_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """usage 采集：输入/输出/推理 token + prompt_cache_hit/miss（PRD 9.3）。"""
    usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=340,
        total_tokens=1540,
        prompt_tokens_details=SimpleNamespace(
            prompt_cache_hit_tokens=1000, prompt_cache_miss_tokens=200
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=300),
    )
    stream = FakeStream(
        [
            FakeChunk([FakeChoice(FakeDelta("答"), None)]),
            FakeChunk([FakeChoice(FakeDelta(), "stop")], usage=usage),
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    result = None
    async for item in model.stream_result([{"role": "user", "content": "q"}]):
        if not isinstance(item, ToolCallFragment):
            result = item
    assert result is not None
    assert result.response.usage.prompt_tokens == 1200
    assert result.response.usage.completion_tokens == 340
    assert result.response.usage.prompt_cache_hit_tokens == 1000
    assert result.response.usage.prompt_cache_miss_tokens == 200
    assert result.response.usage.reasoning_tokens == 300


async def test_usage_only_chunk_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """收尾 chunk 带 error -> 抛 ChatProviderError（不吞错）。"""
    stream = FakeStream(
        [
            FakeChunk([FakeChoice(FakeDelta("a"), None)]),
            FakeChunk([], error=SimpleNamespace(message="upstream failed")),
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    with pytest.raises(ChatProviderError) as exc:
        async for _ in model.stream_result([{"role": "user", "content": "q"}]):
            pass
    assert "upstream failed" in str(exc.value)


# ---------------------------------------------------------------- 4. 非流式完整响应


async def test_complete_returns_full_response_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整响应保存 content/reasoning_content/tool_calls/finish_reason/usage/请求 ID。"""
    completion = SimpleNamespace(
        id="req-full-1",
        model="deepseek-flash",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="最终答案",
                    reasoning_content="推理过程",
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )
    completions = SpyCompletions(completion=completion)
    model = _model(monkeypatch, completions)
    response = await model.complete(
        [{"role": "user", "content": "q"}], reasoning_level="max", max_tokens=512
    )
    assert response.content == "最终答案"
    assert response.reasoning_content == "推理过程"
    assert response.finish_reason == "stop"
    assert response.status is ResponseStatus.COMPLETE
    assert response.provider_request_id == "req-full-1"
    assert response.usage.total_tokens == 30
    assert response.succeeded is True
    request = completions.requests[0]
    assert request["stream"] is False
    assert request["reasoning_effort"] == "max"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["max_tokens"] == 512


async def test_complete_tool_calls_invalid_json_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非流式工具参数不是合法 JSON -> complete=False 且状态 INCOMPLETE。"""
    completion = SimpleNamespace(
        id="req-2",
        model="deepseek-flash",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call_x",
                            function=SimpleNamespace(name="t", arguments="{bad json"),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    completions = SpyCompletions(completion=completion)
    model = _model(monkeypatch, completions)
    response = await model.complete([{"role": "user", "content": "q"}], tools=TOOLS)
    assert response.tool_calls[0].complete is False
    assert response.status is ResponseStatus.INCOMPLETE
    assert response.succeeded is False


# ---------------------------------------------------------------- 5. 能力缺失 -> 配置错误


async def test_tools_on_provider_without_tool_support_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 不支持工具 -> 明确配置错误，不静默降级（PRD 9.2）。"""
    caps = resolve_capabilities(
        "no-tools",
        COMPAT,
        declared={
            "supports_tools": False,
            "supports_thinking_toggle": False,
            "supports_image_input": False,
            "context_limit": 8192,
            "max_output_tokens": 1024,
            "history_protocol": "openai-chat-completions",
        },
    )
    completions = SpyCompletions()
    model = _model(
        monkeypatch, completions, base_url=COMPAT, model="no-tools", capabilities=caps
    )
    with pytest.raises(ProviderRequestError) as exc:
        async for _ in model.stream_chat([{"role": "user", "content": "q"}], tools=TOOLS):
            pass
    message = str(exc.value)
    assert "supports_tools" in message
    assert "EVERYTHING_RAG_CHAT_CAPABILITIES" in message
    assert completions.requests == []  # 未发出任何请求


async def test_reasoning_level_on_provider_without_toggle_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = resolve_capabilities(
        "no-think",
        COMPAT,
        declared={
            "supports_tools": True,
            "supports_thinking_toggle": False,
            "supports_image_input": False,
            "context_limit": 8192,
            "max_output_tokens": 1024,
            "history_protocol": "openai-chat-completions",
        },
    )
    completions = SpyCompletions()
    model = _model(monkeypatch, completions, base_url=COMPAT, model="no-think", capabilities=caps)
    with pytest.raises(ProviderRequestError) as exc:
        async for _ in model.stream_chat(
            [{"role": "user", "content": "q"}], reasoning_level="high"
        ):
            pass
    assert "supports_thinking_toggle" in str(exc.value)


async def test_tools_require_reasoning_replay_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端点声明 supports_tools 但协议不是 DeepSeek 全量回传 -> 明确配置错误。"""
    caps = resolve_capabilities(
        "plain-tools",
        COMPAT,
        declared={
            "supports_tools": True,
            "supports_thinking_toggle": False,
            "supports_image_input": False,
            "context_limit": 8192,
            "max_output_tokens": 1024,
            "history_protocol": "openai-chat-completions",
        },
    )
    completions = SpyCompletions()
    model = _model(
        monkeypatch, completions, base_url=COMPAT, model="plain-tools", capabilities=caps
    )
    with pytest.raises(ProviderRequestError) as exc:
        async for _ in model.stream_chat([{"role": "user", "content": "q"}], tools=TOOLS):
            pass
    assert "history_protocol" in str(exc.value)


def test_unknown_model_capabilities_access_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未知模型：模型可构造（零联网），但取能力/用能力时抛明确配置错误。"""
    completions = SpyCompletions()
    model = _model(monkeypatch, completions, base_url=COMPAT, model="mystery-llm")
    assert model.supports_tools is False
    assert "mystery-llm" in model.capabilities_error
    with pytest.raises(ProviderRequestError):
        _ = model.capabilities


def test_resolve_at_init_raises_for_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置校验入口（strict）：未知模型在构造期即明确失败。"""
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: SpyClient(SpyCompletions()))
    with pytest.raises(CapabilityConfigurationError):
        OpenAICompatChatModel(
            base_url=COMPAT, model="mystery-llm", resolve_at_init=True
        )


# ---------------------------------------------------------------- 6. 历史协议（reasoning 回传）


def test_tools_history_keeps_reasoning_content_for_every_turn() -> None:
    """带 tools 的会话保留此前 assistant 的完整 reasoning_content（含未调用工具的轮次）。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    assert replay_reasoning_content(caps, has_tools=True) is True
    history = [
        {"role": "user", "content": "退款政策？"},
        {
            "role": "assistant",
            "content": "先查资料",
            "reasoning_content": "用户问退款，需要检索内部文档",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "search_knowledge", "arguments": '{"query": "退款"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "退款 7 天"},
        {
            "role": "assistant",
            "content": "还需要确认时间口径",
            "reasoning_content": "未调用工具的轮次也必须保留",
        },
    ]
    prepared = prepare_messages(history, capabilities=caps, replay_reasoning=True)
    assert prepared[1]["reasoning_content"] == "用户问退款，需要检索内部文档"
    assert prepared[3]["reasoning_content"] == "未调用工具的轮次也必须保留"
    assert prepared[2]["tool_call_id"] == "call_a"
    assert prepared[1] is not history[1]  # 纯函数：不改原对象


def test_history_without_tools_drops_reasoning_content() -> None:
    """不带 tools 时 reasoning_content 会被忽略，因此不携带（PRD 9.2）。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    assert replay_reasoning_content(caps, has_tools=False) is False
    history = [{"role": "assistant", "content": "答案", "reasoning_content": "链"}]
    prepared = prepare_messages(history, capabilities=caps, replay_reasoning=False)
    assert "reasoning_content" not in prepared[0]


def test_tool_result_without_call_id_is_rejected() -> None:
    """工具结果必须使用对应 tool_call_id（缺失即失败，不静默丢关联）。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    with pytest.raises(ProviderRequestError) as exc:
        prepare_messages(
            [{"role": "tool", "content": "结果"}], capabilities=caps, replay_reasoning=True
        )
    assert "tool_call_id" in str(exc.value)


def test_unknown_role_is_rejected() -> None:
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    with pytest.raises(ProviderRequestError):
        prepare_messages(
            [{"role": "observer", "content": "x"}], capabilities=caps, replay_reasoning=True
        )


def test_reference_only_messages_never_create_assistant_tool_calls() -> None:
    """独立子 Agent 的历史摘要只作参考数据，不制造未发生的 assistant 工具调用。"""
    message = reference_only_messages("已核实：退款期为 7 天")
    assert message["role"] == "user"
    assert message["content"] == "已核实：退款期为 7 天"
    assert "tool_calls" not in message
    assert reference_only_messages("摘要", role="system")["role"] == "system"
    with pytest.raises(ProviderRequestError):
        reference_only_messages("伪造", role="assistant")


async def test_request_carries_reasoning_replay_for_tool_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """带 tools 的请求体里 assistant 历史保留 reasoning_content（端到端断言）。"""
    stream = FakeStream([FakeChunk([FakeChoice(FakeDelta("ok"), "stop")])])
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    history = [
        {"role": "assistant", "content": "查", "reasoning_content": "先检索"},
    ]
    async for _ in model.stream_chat(history, tools=TOOLS):
        pass
    assert completions.requests[0]["messages"][0]["reasoning_content"] == "先检索"


# ---------------------------------------------------------------- 7. 零出网


async def test_full_stream_chain_zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切 socket：流式 + 工具汇集 + 完整响应链路仍成功 => 零外呼。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)
    stream = FakeStream(
        [
            FakeChunk([FakeChoice(FakeDelta(reasoning_content="想"), None)]),
            FakeChunk(
                [
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[_tool_call_delta(0, "c", "t", '{"a": 1}')]
                        ),
                        "tool_calls",
                    )
                ]
            ),
        ]
    )
    completions = SpyCompletions(stream=stream)
    model = _model(monkeypatch, completions)
    kinds: list[str] = []
    result = None
    async for item in model.stream_result([{"role": "user", "content": "q"}], tools=TOOLS):
        if isinstance(item, ToolCallFragment):
            kinds.append("tool")
        elif isinstance(item, StreamResult):
            kinds.append("result")
            result = item
        else:
            kinds.append(item.kind)
    assert kinds == ["reasoning", "tool", "result"]
    assert result is not None and result.response.succeeded


def test_capability_sources_documented() -> None:
    """能力取值来源可复核（PRD 18.2：实现报告需写出测量来源）。"""
    from app.generation.capabilities import CAPABILITY_SOURCES

    assert "api-docs.deepseek.com" in "".join(CAPABILITY_SOURCES.values())
    assert HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS in json.dumps(
        {"history_protocol": HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS}
    )


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
