"""识图 provider 单测：describe 请求装配（base64 data URI）/ 结果映射 / 审计 / 工厂。

全部 mock OpenAI client，零出网；审计走真实 OutboundClient 断言记录。
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.outbound import OutboundClient
from app.core.settings_store import VisionModelConfig
from app.generation import vision as vision_mod
from app.generation.vision import (
    DESCRIBE_PROMPT,
    OpenAICompatibleVision,
    VisionProviderError,
    create_vision_model_from_config,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str = "描述") -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def _model(
    monkeypatch: pytest.MonkeyPatch,
    completions: _FakeCompletions,
    outbound: OutboundClient | None = None,
) -> OpenAICompatibleVision:
    monkeypatch.setattr(vision_mod, "OpenAI", lambda **kw: _FakeClient(completions))
    return OpenAICompatibleVision(
        base_url="http://localhost:11434/v1",
        model="qwen2-vl:7b",
        outbound=outbound,
    )


def test_describe_builds_image_url_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """describe 装配 user 消息：text=模板 + image_url=data URI（base64 内联）。"""
    completions = _FakeCompletions("画面主体：架构图")
    model = _model(monkeypatch, completions)
    image = b"\x89PNG fake"
    result = model.describe(image)
    assert result == "画面主体：架构图"
    call = completions.calls[0]
    assert call["model"] == "qwen2-vl:7b"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 800
    messages = call["messages"]
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": DESCRIBE_PROMPT}
    expected_uri = f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
    assert parts[1] == {"type": "image_url", "image_url": {"url": expected_uri}}


def test_describe_records_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """成功调用 -> 审计 provider=vision、status=200、destination 只取 host。"""
    outbound = OutboundClient()
    model = _model(monkeypatch, _FakeCompletions("x"), outbound=outbound)
    model.describe(b"img")
    events = outbound.entries()
    assert len(events) == 1
    assert events[0].provider == "vision"
    assert events[0].destination == "localhost"
    assert events[0].status == 200


def test_describe_error_raises_and_records_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """上游抛错 -> VisionProviderError，审计 status=None。"""

    class _BoomCompletions:
        def create(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    outbound = OutboundClient()
    model = _model(monkeypatch, _BoomCompletions(), outbound=outbound)
    with pytest.raises(VisionProviderError):
        model.describe(b"img")
    assert outbound.entries()[0].status is None


def test_describe_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """返回空描述 -> VisionProviderError。"""
    model = _model(monkeypatch, _FakeCompletions(""))
    with pytest.raises(VisionProviderError):
        model.describe(b"img")


def test_factory_missing_config_raises() -> None:
    """base_url/model 缺失 -> VisionProviderError。"""
    with pytest.raises(VisionProviderError) as exc:
        create_vision_model_from_config(VisionModelConfig())
    assert "Base URL" in str(exc.value)
    assert "模型名" in str(exc.value)


def test_factory_configured_returns_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """配置齐全 -> OpenAICompatibleVision，key 从 SecretStr 解析传入。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        vision_mod,
        "OpenAI",
        lambda base_url, api_key, **kw: captured.update(base_url=base_url, api_key=api_key),
    )
    model = create_vision_model_from_config(
        VisionModelConfig(base_url="http://v.test/v1", model="m", api_key=SecretStr("sk-v")),
    )
    assert isinstance(model, OpenAICompatibleVision)
    assert captured["base_url"] == "http://v.test/v1"
    assert captured["api_key"] == "sk-v"


def test_factory_key_absent_uses_local_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 api_key -> 'local' 占位（本地 Ollama 无需 key）。"""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        vision_mod,
        "OpenAI",
        lambda base_url, api_key, **kw: captured.update(base_url=base_url, api_key=api_key),
    )
    create_vision_model_from_config(VisionModelConfig(base_url="http://v.test/v1", model="m"))
    assert captured["api_key"] == "local"


class _ApiError(Exception):
    """模拟 openai APIStatusError（带 status_code）。"""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class _FlakyCompletions:
    """前 fail_times 次抛错，之后成功返回 content。"""

    def __init__(self, fail_times: int = 1, status_code: int = 500, content: str = "描述") -> None:
        self._fail_times = fail_times
        self._status = status_code
        self._content = content
        self.calls = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise _ApiError(self._status)
        return _FakeResponse(self._content)


def test_describe_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """瞬时 500 -> 指数退避重试后成功（不直接失败）。"""
    sleeps: list[float] = []
    monkeypatch.setattr(vision_mod.time, "sleep", lambda s: sleeps.append(s))
    completions = _FlakyCompletions(fail_times=1, status_code=500)
    monkeypatch.setattr(vision_mod, "OpenAI", lambda **kw: _FakeClient(completions))

    model = OpenAICompatibleVision(base_url="http://localhost:11434/v1", model="qwen2-vl:7b")
    assert model.describe(b"img") == "描述"
    assert completions.calls == 2  # 失败 1 次 + 重试成功 1 次
    assert sleeps == [2.0]


def test_describe_permanent_error_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """永久 400 -> 不重试，立即失败。"""
    sleeps: list[float] = []
    monkeypatch.setattr(vision_mod.time, "sleep", lambda s: sleeps.append(s))
    completions = _FlakyCompletions(fail_times=3, status_code=400)
    monkeypatch.setattr(vision_mod, "OpenAI", lambda **kw: _FakeClient(completions))

    model = OpenAICompatibleVision(base_url="http://localhost:11434/v1", model="qwen2-vl:7b")
    with pytest.raises(VisionProviderError):
        model.describe(b"img")
    assert completions.calls == 1  # 不重试
    assert sleeps == []


def test_describe_truncates_to_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """超长描述被硬截断到 MAX_DESCRIBE_CHARS（严格 ≤1000 字）。"""
    long_content = "字" * (vision_mod.MAX_DESCRIBE_CHARS + 500)
    model = _model(monkeypatch, _FakeCompletions(long_content))
    result = model.describe(b"img")
    assert len(result) == vision_mod.MAX_DESCRIBE_CHARS
    assert len(result) <= 1000
