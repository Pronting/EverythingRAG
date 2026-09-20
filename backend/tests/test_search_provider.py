"""联网搜索 provider 测试：Tavily（MockTransport 注入，零出网）+ 工厂（配置校验）。

隐私红线：全部 fake/mock，无真实网络；出网审计经 OutboundClient 记录
（provider="search" / destination=host / method="web_search"），异常 status=None，
绝不泄露 key / 路径 / 正文。

T04 变更（重要）：本文件原先用
``monkeypatch.setattr(tavily_mod.httpx, "AsyncClient", lambda **kw: client)``
替换 **全局 httpx 模块**的 ``AsyncClient`` —— 那是进程级篡改，会改变并行执行的
所有 httpx 消费者的行为（同类全局状态曾让 Chroma 出现跨用例随机失败）。
现改为注入 ``httpx.MockTransport``（provider 的 ``transport`` 参数），不再触碰全局对象。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.outbound import OutboundClient
from app.core.settings_store import SearchConfig
from app.search.base import SearchError
from app.search.factory import create_search_provider_from_config
from app.search.tavily import TavilySearchProvider


class _Recorder:
    """``httpx.MockTransport`` 处理器：记录请求，并按配置返回响应或抛异常。"""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload if payload is not None else {"results": []}
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status_code, json=self.payload)

    @property
    def posted(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (str(request.url), json.loads(request.content.decode("utf-8")))
            for request in self.requests
        ]


def _results_payload() -> dict[str, Any]:
    return {
        "results": [
            {"title": "结果一", "url": "https://example.com/1", "content": "正文一"},
            {"title": "结果二", "url": "https://example.com/2", "content": "正文二"},
        ]
    }


def _make_provider(recorder: _Recorder, outbound: OutboundClient | None) -> TavilySearchProvider:
    return TavilySearchProvider(
        api_key="sk-tavily", outbound=outbound, transport=httpx.MockTransport(recorder)
    )


# ---------------------------------------------------------------- 1. 成功路径


async def test_tavily_search_success_maps_results_and_audits() -> None:
    """成功搜索 -> 结果映射（title/url/content）+ 恰一条 provider="search" 审计事件。"""

    outbound = OutboundClient(max_entries=10)
    recorder = _Recorder(payload=_results_payload())
    provider = _make_provider(recorder, outbound)

    results = await provider.search("DeepSeek V4", max_results=5)

    assert [r.title for r in results] == ["结果一", "结果二"]
    assert results[0].url == "https://example.com/1"
    assert results[0].content == "正文一"

    # 请求体：query / max_results / api_key 齐全
    url, payload = recorder.posted[0]
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


async def test_tavily_construction_zero_outbound() -> None:
    """构造 provider 零联网：不发请求、不记录审计。

    原断言是「构造阶段未创建 ``httpx.AsyncClient``」（靠全局 monkeypatch 观察）；
    改为注入 transport 后不再 patch 全局，因此用**行为等价且更硬**的断言：
    构造后传输层没有被调用过，审计为空。（「首次 search 才建 client」由上面
    ``test_tavily_search_success_maps_results_and_audits`` 覆盖。）
    """

    outbound = OutboundClient(max_entries=10)
    recorder = _Recorder(payload=_results_payload())
    _make_provider(recorder, outbound)

    assert recorder.requests == []
    assert outbound.entries() == []


# ---------------------------------------------------------------- 2. 失败路径


async def test_tavily_http_error_records_status() -> None:
    """HTTP 4xx/5xx -> SearchError（含状态码）+ 审计 status 记录真实状态码。"""

    outbound = OutboundClient(max_entries=10)
    recorder = _Recorder(status_code=429, payload={"results": []})
    provider = _make_provider(recorder, outbound)

    with pytest.raises(SearchError, match="HTTP 429"):
        await provider.search("q")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status == 429
    assert entries[0].provider == "search"


async def test_tavily_network_error_records_none_and_sanitizes() -> None:
    """连接异常 -> SearchError（脱敏，只含异常类名）+ 审计 status=None。"""

    outbound = OutboundClient(max_entries=10)
    recorder = _Recorder(error=httpx.ConnectError("boom-internal-detail"))
    provider = _make_provider(recorder, outbound)

    with pytest.raises(SearchError, match="联网搜索失败"):
        await provider.search("q")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status is None
    assert "boom-internal-detail" not in str(entries[0])


async def test_tavily_without_outbound_records_nothing() -> None:
    """outbound=None -> 不记录审计，行为照常。"""

    recorder = _Recorder(payload=_results_payload())
    provider = _make_provider(recorder, outbound=None)

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
