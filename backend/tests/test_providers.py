"""对话 provider 测试：配置校验 + 流式映射（fake/mock AsyncOpenAI，零出网）。

隐私红线：全部用 fake/mock 替代真实 AsyncOpenAI client，不触发任何真实网络；
另有阻断一切 socket 连接的用例实证零出网；API key 只从环境变量读取。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.settings_store import ChatModelConfig
from app.generation import providers
from app.generation.providers import (
    ChatProviderError,
    OpenAICompatChatModel,
    UnconfiguredChatModel,
    create_chat_model,
)


def _settings(**overrides: Any) -> Settings:
    """构造带对话 provider 字段的 Settings；overrides 优先。

    chat_api_key 默认显式 None：隔离磁盘 .env 的真实 key，让测试聚焦
    各自指定的来源（环境变量回退 / Settings 字段直传）。
    """
    base: dict[str, Any] = {
        "chat_provider_type": "openai_compatible",
        "chat_base_url": None,
        "chat_model": None,
        "chat_api_key": None,
        "chat_api_key_env": "EVERYTHING_RAG_CHAT_API_KEY",
    }
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------- 1. 配置缺失 -> ChatProviderError


def test_missing_base_url_raises_listing_env_name() -> None:
    """缺 base_url -> ChatProviderError，消息列出缺失的环境变量名。"""
    with pytest.raises(ChatProviderError) as exc:
        create_chat_model(_settings(chat_base_url=None, chat_model="gpt-4o-mini"))
    assert "EVERYTHING_RAG_CHAT_BASE_URL" in str(exc.value)


def test_missing_model_raises_listing_env_name() -> None:
    """缺 model -> ChatProviderError，消息列出缺失的环境变量名。"""
    with pytest.raises(ChatProviderError) as exc:
        create_chat_model(_settings(chat_base_url="http://localhost:9999/v1", chat_model=None))
    assert "EVERYTHING_RAG_CHAT_MODEL" in str(exc.value)


def test_missing_both_lists_all_env_names() -> None:
    """base_url 与 model 都缺 -> 消息同时列出两个环境变量名。"""
    with pytest.raises(ChatProviderError) as exc:
        create_chat_model(_settings(chat_base_url=None, chat_model=None))
    message = str(exc.value)
    assert "EVERYTHING_RAG_CHAT_BASE_URL" in message
    assert "EVERYTHING_RAG_CHAT_MODEL" in message


# ---------------------------------------------------------------- 2. 配置齐全 -> 构造成功（零联网）


def test_configured_returns_model_zero_outbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置齐全 -> 返回 OpenAICompatChatModel；阻断 socket 证明构造零联网。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)
    model = create_chat_model(
        _settings(chat_base_url="http://localhost:9999/v1", chat_model="gpt-4o-mini")
    )
    assert isinstance(model, OpenAICompatChatModel)
    assert model.supports_image_input is False


# ---------------------------------------------------------------- 3. API key 只从环境变量读


def test_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """key 从 os.environ 读取并传给 client；不落 Settings。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda base_url, api_key, **kwargs: captured.update(base_url=base_url, api_key=api_key),
    )
    monkeypatch.setenv("EVERYTHING_RAG_CHAT_API_KEY", "sk-123")
    create_chat_model(
        _settings(chat_base_url="http://localhost:9999/v1", chat_model="gpt-4o-mini")
    )
    assert captured["base_url"] == "http://localhost:9999/v1"
    assert captured["api_key"] == "sk-123"


def test_key_read_from_settings_chat_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """key 从 Settings.chat_api_key（.env 来源，SecretStr）读取，优先于 os.environ 回退。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda base_url, api_key, **kwargs: captured.update(base_url=base_url, api_key=api_key),
    )
    monkeypatch.delenv("EVERYTHING_RAG_CHAT_API_KEY", raising=False)  # .env 场景：os.environ 无此变量
    create_chat_model(
        _settings(
            chat_base_url="http://localhost:9999/v1",
            chat_model="gpt-4o-mini",
            chat_api_key=SecretStr("sk-from-dotenv"),
        )
    )
    assert captured["api_key"] == "sk-from-dotenv"


def test_settings_populates_chat_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic-settings 把 EVERYTHING_RAG_CHAT_API_KEY 填进 chat_api_key（SecretStr）。

    直接用 Settings() 而非 _settings()：后者显式传 chat_api_key=None 会屏蔽环境变量；
    此处验证「不显式传值」时环境变量被正确读入（env 优先于 .env 文件）。
    """
    monkeypatch.setenv("EVERYTHING_RAG_CHAT_API_KEY", "sk-123")
    s = Settings()
    assert s.chat_api_key is not None
    assert s.chat_api_key.get_secret_value() == "sk-123"


# ---------------------------------------------------------------- 5b. 从设置页 ChatModelConfig 构造


def test_create_from_config_uses_chat_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_chat_model_from_config 用 ChatModelConfig 的 base_url/model/api_key。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda base_url, api_key, **kwargs: captured.update(base_url=base_url, api_key=api_key),
    )
    chat = ChatModelConfig(base_url="http://cfg.test/v1", model="cfg-model", api_key=SecretStr("sk-cfg"))
    model = providers.create_chat_model_from_config(chat)
    assert isinstance(model, OpenAICompatChatModel)
    assert model.supports_image_input is False
    assert captured["base_url"] == "http://cfg.test/v1"
    assert captured["api_key"] == "sk-cfg"


def test_create_from_config_missing_raises_listing_fields() -> None:
    """缺 base_url/model -> ChatProviderError，消息列出字段名。"""
    with pytest.raises(ChatProviderError) as exc:
        providers.create_chat_model_from_config(ChatModelConfig())
    message = str(exc.value)
    assert "Base URL" in message
    assert "模型名" in message


def test_create_from_config_key_absent_uses_local_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChatModelConfig 无 api_key -> client 用 'local' 占位（不报错）。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda base_url, api_key, **kwargs: captured.update(base_url=base_url, api_key=api_key),
    )
    providers.create_chat_model_from_config(
        ChatModelConfig(base_url="http://cfg.test/v1", model="m")
    )
    assert captured["api_key"] == "local"


def test_key_absent_uses_local_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """key 未设置 -> client 用 'local' 占位（不落 config、不报错）。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda base_url, api_key, **kwargs: captured.update(base_url=base_url, api_key=api_key),
    )
    monkeypatch.delenv("EVERYTHING_RAG_CHAT_API_KEY", raising=False)
    create_chat_model(
        _settings(chat_base_url="http://localhost:9999/v1", chat_model="gpt-4o-mini")
    )
    assert captured["api_key"] == "local"


# ---------------------------------------------------------------- 4. 流式映射（mock client，零网络）


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _AsyncChunks:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> _FakeChunk:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCompletions:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, list[dict[str, Any]], bool]] = []

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        **kwargs: Any,
    ) -> _AsyncChunks:
        self.calls.append((model, messages, stream))
        return _AsyncChunks(self._chunks)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


async def test_stream_maps_chunks_and_skips_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chunk.delta.content 逐段映射为 AsyncIterator[str]；空 content chunk 跳过。"""
    chunks = [_FakeChunk("你"), _FakeChunk("好"), _FakeChunk(""), _FakeChunk("世界")]
    completions = _FakeCompletions(chunks)
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))

    model = OpenAICompatChatModel(base_url="http://localhost:9999/v1", model="gpt-4o-mini")
    messages = [{"role": "user", "content": "你好"}]
    tokens = [token async for token in model.stream_chat(messages)]

    assert tokens == ["你", "好", "世界"]  # 空 content chunk 被跳过
    called_model, called_messages, stream_arg = completions.calls[0]
    assert called_model == "gpt-4o-mini"
    assert called_messages == messages
    assert stream_arg is True


async def test_stream_zero_outbound_with_socket_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阻断一切 socket 连接，流式全链路仍成功 => 零出网。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)
    completions = _FakeCompletions([_FakeChunk("a"), _FakeChunk("b")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))

    model = OpenAICompatChatModel(base_url="http://localhost:9999/v1", model="m")
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "q"}])]
    assert tokens == ["a", "b"]


# ---------------------------------------------------------------- 5. 占位模型：stream 阶段才报错


async def test_unconfigured_model_raises_at_stream_time() -> None:
    """UnconfiguredChatModel 构造零副作用；stream 阶段抛 ChatProviderError。"""
    error = ChatProviderError("对话 provider 未配置，缺少环境变量: EVERYTHING_RAG_CHAT_MODEL")
    model = UnconfiguredChatModel(error)
    assert model.supports_image_input is False
    with pytest.raises(ChatProviderError):
        async for _token in model.stream_chat([{"role": "user", "content": "hi"}]):
            pass


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
