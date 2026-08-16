"""联网搜索 provider 测试：Tavily（mock httpx，零出网）+ 工厂（配置校验）。

隐私红线：全部 fake/mock，无真实网络；出网审计经 OutboundClient 记录
（provider="search" / destination=host / method="web_search"），异常 status=None，
绝不泄露 key / 路径 / 正文。
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx
import pytest
from pydantic import SecretStr

from app.core.outbound import OutboundClient
from app.core.settings_store import SearchConfig
from app.search import tavily as tavily_mod
from app.search.base import SearchError
from app.search.factory import create_search_provider_from_config
from app.search.tavily import TavilySearchProvider


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"results": []}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"server error {self.status_code}", request=None, response=self
            )

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(
        self,
        response: _FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.posted: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self.posted.append((url, json))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _results_payload() -> dict[str, Any]:
    return {
        "results": [
            {"title": "结果一", "url": "https://example.com/1", "content": "正文一"},
            {"title": "结果二", "url": "https://example.com/2", "content": "正文二"},
        ]
    }


def _make_provider(outbound: OutboundClient | None) -> TavilySearchProvider:
    return TavilySearchProvider(api_key="sk-tavily", outbound=outbound)


# ---------------------------------------------------------------- 1. 成功路径


async def test_tavily_search_success_maps_results_and_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功搜索 -> 结果映射（title/url/content）+ 恰一条 provider="search" 审计事件。"""
    outbound = OutboundClient(max_entries=10)
    client = _FakeAsyncClient(response=_FakeResponse(200, _results_payload()))
    monkeypatch.setattr(tavily_mod.httpx, "AsyncClient", lambda **kw: client)
    provider = _make_provider(outbound)

    results = await provider.search("DeepSeek V4", max_results=5)

    assert [r.title for r in results] == ["结果一", "结果二"]
    assert results[0].url == "https://example.com/1"
    assert results[0].content == "正文一"

    # 请求体：query / max_results / api_key 齐全
    url, payload = client.posted[0]
    assert url == "https://api.tavily.com/search"
    assert payload["query"] == "DeepSeek V4"
    assert payload["max_results"] == 5
    assert payload["api_key"] == "sk-tavily"

    entries = outbound.entries()
    assert len(entries) == 1
    event = entries[0]
    assert event.provider == "search"
    assert event.destination == "api.tavily.com"  # 只记 host
    assert event.method == "web_search"
    assert event.status == 200
    assert event.duration_ms is not None


async def test_tavily_construction_zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造 provider 零联网：不创建 client、不记录审计。"""
    outbound = OutboundClient(max_entries=10)
    created: list[Any] = []
    monkeypatch.setattr(
        tavily_mod.httpx, "AsyncClient", lambda **kw: created.append(kw) or _FakeAsyncClient()
    )
    _make_provider(outbound)
    assert created == []  # 构造阶段未建 client
    assert outbound.entries() == []


# ---------------------------------------------------------------- 2. 失败路径


async def test_tavily_http_error_records_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 4xx/5xx -> SearchError（含状态码）+ 审计 status 记录真实状态码。"""
    outbound = OutboundClient(max_entries=10)
    client = _FakeAsyncClient(response=_FakeResponse(429, {"results": []}))
    monkeypatch.setattr(tavily_mod.httpx, "AsyncClient", lambda **kw: client)
    provider = _make_provider(outbound)

    with pytest.raises(SearchError, match="HTTP 429"):
        await provider.search("q")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status == 429
    assert entries[0].provider == "search"


async def test_tavily_network_error_records_none_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连接异常 -> SearchError（脱敏，只含异常类名）+ 审计 status=None。"""
    outbound = OutboundClient(max_entries=10)
    client = _FakeAsyncClient(error=ConnectionError("boom-internal-detail"))
    monkeypatch.setattr(tavily_mod.httpx, "AsyncClient", lambda **kw: client)
    provider = _make_provider(outbound)

    with pytest.raises(SearchError, match="联网搜索失败"):
        await provider.search("q")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status is None
    assert "boom-internal-detail" not in str(entries[0])


async def test_tavily_without_outbound_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """outbound=None -> 不记录审计，行为照常。"""
    client = _FakeAsyncClient(response=_FakeResponse(200, _results_payload()))
    monkeypatch.setattr(tavily_mod.httpx, "AsyncClient", lambda **kw: client)
    provider = TavilySearchProvider(api_key="sk-t", outbound=None)
    results = await provider.search("q")
    assert len(results) == 2


# ---------------------------------------------------------------- 3. 工厂


def test_factory_not_enabled_raises() -> None:
    """未启用 -> SearchError「未启用」。"""
    with pytest.raises(SearchError, match="未启用"):
        create_search_provider_from_config(SearchConfig(enabled=False))


def test_factory_tavily_missing_key_raises() -> None:
    """Tavily 缺 api_key -> SearchError「需要 API Key」。"""
    with pytest.raises(SearchError, match="需要 API Key"):
        create_search_provider_from_config(SearchConfig(enabled=True, api_key=None))


def test_factory_tavily_ok() -> None:
    """Tavily 配齐 -> 返回 TavilySearchProvider，destination 为官方 host。"""
    provider = create_search_provider_from_config(
        SearchConfig(enabled=True, api_key=SecretStr("sk-t")), outbound=None
    )
    assert isinstance(provider, TavilySearchProvider)
    assert provider._destination == "api.tavily.com"


def test_factory_searxng_not_implemented() -> None:
    """SearXNG 未实现 -> SearchError「尚未实现」（可插拔接口预留）。"""
    with pytest.raises(SearchError, match="尚未实现"):
        create_search_provider_from_config(
            SearchConfig(enabled=True, provider_type="searxng", base_url="http://searxng:8080"),
            outbound=None,
        )


def test_unknown_provider_rejected_by_schema() -> None:
    """未知 provider -> pydantic 在 SearchConfig 构造阶段即拒绝（Literal 约束）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SearchConfig(enabled=True, provider_type="bing")
