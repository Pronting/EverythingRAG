"""联网搜索层接口：SearchProvider 协议 + SearchResult（对齐可插拔适配器路线）。

隐私约束：搜索结果只承载 title/url/snippet/content（用于组装上下文与来源展示），
不承载 API key；失败抛 SearchError，消息清晰且绝不泄露密钥 / 正文。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SearchError(Exception):
    """联网搜索领域错误（未配置 / 未启用 / 调用失败），消息清晰可读、不泄露密钥。"""


@dataclass(frozen=True)
class SearchResult:
    """单条联网搜索结果：标题 + 链接 + 摘要 + 可选正文。"""

    title: str
    url: str
    snippet: str
    content: str = ""


class SearchProvider(Protocol):
    """联网搜索 provider 接口（可插拔：Tavily / SearXNG / Brave...）。"""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """执行一次联网搜索，返回结果列表；失败抛 SearchError。"""
        ...
