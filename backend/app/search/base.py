"""联网搜索层接口：SearchProvider 协议 + SearchResult（对齐可插拔适配器路线）。

隐私约束：搜索结果只承载 title/url/snippet/content（用于组装上下文与来源展示），
不承载 API key；失败抛 SearchError，消息清晰且绝不泄露密钥 / 正文。

V1.2.0（T04）追加区：联网原文（Search 元信息 + Extract）。既有 ``SearchProvider``
一字未改，``SearchResult`` 只在末尾追加带默认值的 ``published_at`` / ``score``，
V1.1.0 问答链路与既有测试继续可用。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class SearchError(Exception):
    """联网搜索领域错误（未配置 / 未启用 / 调用失败），消息清晰可读、不泄露密钥。"""


@dataclass(frozen=True)
class SearchResult:
    """单条联网搜索结果：标题 + 链接 + 摘要 + 可选正文。"""

    title: str
    url: str
    snippet: str
    content: str = ""
    #: V1.2.0 追加：发布时间（缺失为空串）与来源相关性分数（缺失为 None）。
    published_at: str = ""
    score: float | None = None


class SearchProvider(Protocol):
    """联网搜索 provider 接口（可插拔：Tavily / SearXNG / Brave...）。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """执行一次联网搜索，返回结果列表；失败抛 SearchError。"""
        ...


# =====================================================================
# V1.2.0 追加区 · 联网原文（PRD 7.3）
# =====================================================================


@dataclass(frozen=True)
class WebSearchResponse:
    """一次 Search 的完整响应：结果列表 + 提供商请求 ID（PRD 7.3 要求保留）。

    ``search()`` 仍返回 ``list[SearchResult]``（V1.1.0 兼容）；需要请求 ID 与逐项
    元信息的调用方（Agent 工具层）用 :meth:`WebSearchProvider.search_web`。
    """

    request_id: str = ""
    results: tuple[SearchResult, ...] = ()
    #: 提供商返回的时间等标量元信息；不含密钥、正文与 URL。
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtractedPage:
    """Extract 的一页原文；缺失字段保持空值（PRD 7.3）。"""

    url: str
    final_url: str = ""
    title: str = ""
    published_at: str = ""
    content: str = ""
    #: 提供商返回部分内容 / 明确截断时为 False（PRD 7.3 末段）。
    content_complete: bool = True


@dataclass(frozen=True)
class ExtractFailure:
    """Extract 的逐项失败记录（HTTP 成功不能替代逐项内容检查，PRD 7.3）。"""

    url: str
    error: str = ""


@dataclass(frozen=True)
class ExtractResponse:
    """一次 Extract 的完整响应：成功页 + 逐项失败 + 请求 ID。"""

    request_id: str = ""
    pages: tuple[ExtractedPage, ...] = ()
    failed: tuple[ExtractFailure, ...] = ()


class WebSearchProvider(Protocol):
    """带元信息的 Search 入口（PRD 7.3：topic / 日期范围 / 请求 ID）。"""

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        *,
        topic: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_published_date: bool | None = None,
        search_depth: str = "advanced",
        include_answer: bool = False,
    ) -> WebSearchResponse:
        """联网搜索；失败抛 SearchError。"""
        ...


class ExtractProvider(Protocol):
    """网页原文读取（PRD 7.3：format=markdown、extract_depth=advanced）。"""

    async def extract(
        self,
        urls: Sequence[str],
        *,
        query: str | None = None,
        extract_depth: str = "advanced",
        format: str = "markdown",
    ) -> ExtractResponse:
        """读取网页原文；``query`` 省略时要求整页内容；失败抛 SearchError。"""
        ...
