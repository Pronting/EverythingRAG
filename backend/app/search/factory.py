"""联网搜索 provider 工厂：按 SearchConfig 构造（可插拔，零联网构造）。

outbound 缺省接全局 outbound_client：真实出网自动进入网络审计（/api/audit、
/api/status.privacy.outbound_state 如实上报）。未启用/未配置 raise SearchError，
问答编排层据此把错误收敛为友好 error 帧（不崩、不 500）。
"""

from __future__ import annotations

from app.core.outbound import OutboundClient, outbound_client
from app.core.settings_store import SearchConfig
from app.search.base import SearchError, SearchProvider
from app.search.tavily import DEFAULT_TAVILY_BASE_URL, TavilySearchProvider


def create_search_provider_from_config(
    search: SearchConfig,
    outbound: OutboundClient | None = outbound_client,
) -> SearchProvider:
    """按设置构造联网搜索 provider；未启用/未配置 raise SearchError（零联网）。"""
    if not search.enabled:
        raise SearchError("联网搜索未启用，请在设置中开启")
    if search.provider_type == "tavily":
        if search.api_key is None:
            raise SearchError("联网搜索未配置：Tavily 需要 API Key")
        return TavilySearchProvider(
            api_key=search.api_key.get_secret_value(),
            base_url=search.base_url or DEFAULT_TAVILY_BASE_URL,
            outbound=outbound,
        )
    if search.provider_type == "searxng":
        if not search.base_url:
            raise SearchError("联网搜索未配置：SearXNG 需要 Base URL")
        raise SearchError("SearXNG provider 尚未实现，请使用 Tavily")
    raise SearchError(f"未知联网搜索 provider: {search.provider_type}")
