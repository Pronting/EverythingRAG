"""Tavily 联网 provider：Search（线索）+ Extract（原文），httpx async，经 OutboundClient 审计。

Tavily 专为 RAG/Agent 设计：POST ``{base}/search`` 返回结构化结果（title/url/content），
POST ``{base}/extract`` 返回网页正文（``results`` / ``failed_results``）。API key 仅运行期
从配置读取（SecretStr），绝不落日志；每次调用记录一条审计事件（``provider="search"`` /
``destination=host`` / ``method="web_search"`` 或 ``"web_extract"``），异常时 ``status=None``。

V1.2.0（T04）参数对齐 PRD 7.3：

- Search 支持 ``topic`` / ``start_date`` / ``end_date`` / ``include_published_date``，
  默认 ``search_depth="advanced"``、``include_answer=false``（复杂技术查询与新闻场景）。
- Extract 使用 ``format=markdown`` / ``extract_depth=advanced``；需要整页内容时省略
  ``query``；逐项检查 ``results`` 与 ``failed_results``——HTTP 成功不能替代内容检查。
- 纯 httpx + OutboundClient 审计，不使用任何框架自带网络客户端。

可测试性：``transport`` 为可选注入点（默认 ``None`` 时行为与 V1.1.0 完全一致）。
测试注入 ``httpx.MockTransport``，**不做**进程级 monkeypatch ``httpx.AsyncClient``
——那会篡改进程内所有 httpx 消费者，并行跑测试时产生跨用例串扰。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core.outbound import OutboundClient, OutboundEvent
from app.search.base import (
    ExtractedPage,
    ExtractFailure,
    ExtractResponse,
    SearchError,
    SearchResult,
    WebSearchResponse,
)

#: Tavily 官方 API 端点（base_url 可被自建代理覆盖）
DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"

#: PRD 7.3 默认：复杂技术查询与新闻用 advanced，且不要提供商生成的答案。
DEFAULT_SEARCH_DEPTH = "advanced"
DEFAULT_INCLUDE_ANSWER = False
DEFAULT_EXTRACT_DEPTH = "advanced"
DEFAULT_EXTRACT_FORMAT = "markdown"

#: 单次 Extract 的 URL 上限（PRD 7.3：URL 列表，但没有语义上的批量无上限需求）。
MAX_EXTRACT_URLS = 5

#: 逐项失败原因的最大长度（只保留提供商摘要，绝不放正文 / 密钥）。
_MAX_FAILURE_DETAIL = 200


class TavilySearchProvider:
    """Tavily Search + Extract：每次调用建临时 AsyncClient（无状态、零构造联网）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_TAVILY_BASE_URL,
        outbound: OutboundClient | None = None,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._outbound = outbound
        self._timeout = timeout
        self._transport = transport
        self._destination = _host(base_url)

    # ---------------------------------------------------------------- Search

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """POST /search；返回结果列表（V1.1.0 兼容签名），失败抛 SearchError。"""

        response = await self.search_web(query, max_results)
        return list(response.results)

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        *,
        topic: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_published_date: bool | None = None,
        search_depth: str = DEFAULT_SEARCH_DEPTH,
        include_answer: bool = DEFAULT_INCLUDE_ANSWER,
    ) -> WebSearchResponse:
        """联网搜索：返回线索 + 请求 ID；失败抛 SearchError（不泄露密钥 / 正文）。"""

        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": False,
        }
        if topic is not None:
            payload["topic"] = topic
        if start_date is not None:
            payload["start_date"] = start_date
        if end_date is not None:
            payload["end_date"] = end_date
        if include_published_date is not None:
            payload["include_published_date"] = include_published_date

        data = await self._post_json("/search", payload, method="web_search", what="联网搜索")
        results = tuple(_to_result(item) for item in _dict_items(data.get("results")))
        return WebSearchResponse(
            request_id=str(data.get("request_id") or ""),
            results=results,
            metadata=_search_metadata(data),
        )

    # ---------------------------------------------------------------- Extract

    async def extract(
        self,
        urls: Sequence[str],
        *,
        query: str | None = None,
        extract_depth: str = DEFAULT_EXTRACT_DEPTH,
        format: str = DEFAULT_EXTRACT_FORMAT,
    ) -> ExtractResponse:
        """POST /extract；返回逐项原文与逐项失败，失败抛 SearchError。

        ``query`` 省略（``None``）时读取整页内容（PRD 7.3）。
        """

        cleaned = [str(url).strip() for url in urls if str(url).strip()]
        if not cleaned:
            raise SearchError("网页读取失败：未提供 URL")
        if len(cleaned) > MAX_EXTRACT_URLS:
            raise SearchError(f"网页读取失败：单次最多 {MAX_EXTRACT_URLS} 个 URL")

        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "urls": cleaned,
            "extract_depth": extract_depth,
            "format": format,
        }
        if query is not None:
            payload["query"] = query

        data = await self._post_json("/extract", payload, method="web_extract", what="网页读取")
        pages = tuple(_to_page(item) for item in _dict_items(data.get("results")))
        failed = tuple(_to_failure(item) for item in _dict_items(data.get("failed_results")))
        return ExtractResponse(
            request_id=str(data.get("request_id") or ""),
            pages=pages,
            failed=failed,
        )

    # ---------------------------------------------------------------- HTTP

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        method: str,
        what: str,
    ) -> dict[str, Any]:
        """POST 并解析 JSON；无论成功失败都先记一条出网审计（唯一出网收敛点）。"""

        started_at = _utcnow()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(f"{self._base_url}{path}", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            self._record_outbound(started_at, exc.response.status_code, method)
            raise SearchError(f"{what}失败（HTTP {exc.response.status_code}）") from exc
        except Exception as exc:
            self._record_outbound(started_at, None, method)
            raise SearchError(f"{what}失败：{exc.__class__.__name__}") from exc
        self._record_outbound(started_at, response.status_code, method)
        if not isinstance(data, dict):
            raise SearchError(f"{what}失败：响应结构不是对象")
        return data

    def _record_outbound(self, started_at: datetime, status: int | None, method: str) -> None:
        """记录一条出网审计事件；未注入 outbound 则跳过。"""

        if self._outbound is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="search",
                destination=self._destination,
                method=method,
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(started_at),
            )
        )


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _to_result(item: dict[str, Any]) -> SearchResult:
    content = str(item.get("content") or "")
    return SearchResult(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=content,
        content=content,
        published_at=str(item.get("published_date") or ""),
        score=_optional_float(item.get("score")),
    )


def _to_page(item: dict[str, Any]) -> ExtractedPage:
    content = str(item.get("raw_content") or "")
    return ExtractedPage(
        url=str(item.get("url") or ""),
        final_url=str(item.get("final_url") or ""),
        title=str(item.get("title") or ""),
        published_at=str(item.get("published_date") or ""),
        content=content,
        # 提供商明确截断或返回空正文时标记不完整（PRD 7.3 末段）。
        content_complete=not bool(item.get("truncated")) and bool(content),
    )


def _to_failure(item: dict[str, Any]) -> ExtractFailure:
    detail = str(item.get("error") or "").strip()
    return ExtractFailure(url=str(item.get("url") or ""), error=detail[:_MAX_FAILURE_DETAIL])


def _search_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    response_time = _optional_float(data.get("response_time"))
    if response_time is not None:
        metadata["response_time"] = response_time
    return metadata


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _host(base_url: str) -> str:
    """从 base_url 提取仅 host 的审计目的地（绝不含路径 / query / key）。"""

    hostname = urlsplit(base_url).hostname
    if hostname:
        return hostname
    head = base_url.split("/", 1)[0]
    return head.split("?", 1)[0].split("#", 1)[0] or "unknown"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _elapsed_ms(started_at: datetime) -> float:
    return max(0.0, (datetime.now(UTC) - started_at).total_seconds() * 1000.0)
