"""联网搜索 provider（可插拔）：Tavily 等，统一经 OutboundClient 审计出网。

对齐 CLAUDE.md「可插拔适配器」路线：SearchProvider 薄协议 + 工厂注册，新实现
（SearXNG / Brave / Serper）不改问答编排层。出网只允许发生在 provider 客户端内，
且必须经唯一 OutboundClient 记录审计事件（provider="search"）。
"""

from app.search.base import SearchError, SearchProvider, SearchResult
from app.search.factory import create_search_provider_from_config

__all__ = [
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "create_search_provider_from_config",
]
