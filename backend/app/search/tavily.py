"""Tavily 联网搜索 provider（httpx async，经 OutboundClient 审计）。

Tavily 专为 RAG/Agent 设计：POST {base}/search 返回结构化结果（title/url/content）。
API key 仅运行期从配置读取（SecretStr），绝不落日志；每次调用记录一条审计事件
（provider="search" / destination=host / method="web_search"），异常时 status=None。
出网即用户显式启用联网搜索的结果，属 opt-in。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core.outbound import OutboundClient, OutboundEvent
from app.search.base import SearchError, SearchResult

#: Tavily 官方 API 端点（base_url 可被 SearXNG 之外的自建代理覆盖）
DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"


class TavilySearchProvider:
    """Tavily 联网搜索：每次 search 建临时 AsyncClient（无状态、零构造联网）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_TAVILY_BASE_URL,
        outbound: OutboundClient | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._outbound = outbound
        self._timeout = timeout
        self._destination = _host(base_url)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """POST /search；成功返回结果列表，失败抛 SearchError（不泄露密钥/正文）。"""
        started_at = _utcnow()
        try:
            payload = {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/search", json=payload)
            response.raise_for_status()
            data = response.json()
            results = [_to_result(item) for item in data.get("results", [])]
        except httpx.HTTPStatusError as exc:
            self._record_outbound(started_at, exc.response.status_code)
            raise SearchError(f"联网搜索失败（HTTP {exc.response.status_code}）") from exc
        except Exception as exc:
            self._record_outbound(started_at, None)
            raise SearchError(f"联网搜索失败：{exc.__class__.__name__}") from exc
        else:
            self._record_outbound(started_at, response.status_code)
        return results

    def _record_outbound(self, started_at: datetime, status: int | None) -> None:
        """记录一条出网审计事件；未注入 outbound 则跳过。"""
        if self._outbound is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="search",
                destination=self._destination,
                method="web_search",
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(started_at),
            )
        )


def _to_result(item: dict[str, Any]) -> SearchResult:
    content = str(item.get("content") or "")
    return SearchResult(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=content,
        content=content,
    )


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
