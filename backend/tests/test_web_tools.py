"""联网工具与 Tavily Search/Extract 测试（PRD 7.3 / 7.1）。Owner: T04。

**零真实外呼**：所有 HTTP 都经注入的 ``httpx.MockTransport``（**不** monkeypatch 全局
``httpx.AsyncClient``，避免并行跑测试时跨用例串扰）。断言：

- 搜索参数按 PRD 7.3（``search_depth=advanced`` / ``include_answer=false`` / topic / 日期）；
- Extract 用 ``format=markdown`` + ``extract_depth=advanced``，省略 ``query`` 时要整页；
- 逐项检查 ``results`` 与 ``failed_results``，HTTP 成功不等于内容完整；
- 出网只经 ``OutboundClient`` 审计（请求数 == 审计事件数）；
- A01：Web 路径的本地知识正文检索次数为 **0**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent.contracts import AgentKind
from app.agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolPermissionError,
    ToolStatus,
    ToolUnavailableError,
    build_default_registry,
)
from app.core.outbound import OutboundClient
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore
from app.retrieval.vector_retriever import RetrievedChunk
from app.search.base import SearchError, SearchResult
from app.search.tavily import TavilySearchProvider
from tests.fakes import FakeEmbedder

LONG_PAGE = "长" * 2500


# ---------------------------------------------------------------- MockTransport


class _TransportRecorder:
    """``httpx.MockTransport`` 处理器：记录请求，按配置返回响应或抛异常。"""

    def __init__(
        self,
        *,
        status: int = 200,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.payload: dict[str, Any] = payload if payload is not None else {}
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, json=self.payload)

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [json.loads(request.content.decode("utf-8")) for request in self.requests]

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


def _provider(
    recorder: _TransportRecorder, outbound: OutboundClient | None = None
) -> TavilySearchProvider:
    return TavilySearchProvider(
        api_key="sk-test", outbound=outbound, transport=httpx.MockTransport(recorder)
    )


def _hit(url: str = "https://example.com/1", content: str = "摘要正文") -> dict[str, Any]:
    return {
        "title": "标题",
        "url": url,
        "content": content,
        "published_date": "2026-09-19T00:00:00Z",
        "score": 0.8,
    }


# ---------------------------------------------------------------- 工具侧替身


class _SpyRetriever:
    """本地知识检索替身：只统计调用次数（A01 断言次数为 0）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(
        self, query_text: str, query_vector: list[float], where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append(query_text)
        return []


class _LegacySearchOnlyProvider:
    """只有旧 ``search()`` 的 provider（V1.1.0 形状）：工具层要能安全退化。"""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls.append(query)
        return list(self._results)


@dataclass(frozen=True)
class _WorkspaceStub:
    """无 artifacts 目录的工作区替身（验证工具如实报「未配置」）。"""

    workspace_id: str = "ws-a"
    knowledge_revision: int = 0


def _runtime_workspace(tmp_path: Path, *, workspace_id: str = "ws-a", name: str = "甲区") -> Any:
    store = WorkspaceStore(tmp_path)
    record = store.create(name, workspace_id=workspace_id)
    context = WorkspaceContextFactory(tmp_path, store).resolve(record.id)
    context.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return context


def _tool_context(
    workspace: Any,
    *,
    provider: Any = None,
    extract_provider: Any = None,
    agent_kind: AgentKind | None = AgentKind.WEB,
    retriever: Any = None,
    embedder: Any = None,
    redactor: Any = None,
    web_budget: Any = None,
) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        search_provider=provider,
        extract_provider=extract_provider,
        agent_kind=agent_kind,
        retriever=retriever,
        embedder=embedder,
        redactor=redactor,
        web_budget=web_budget,
    )


# =====================================================================
# 1. Tavily Search（PRD 7.3）
# =====================================================================


async def test_search_web_defaults_follow_prd_73() -> None:
    recorder = _TransportRecorder(payload={"results": [_hit()], "request_id": "req-1"})
    outbound = OutboundClient(max_entries=10)
    provider = _provider(recorder, outbound)

    response = await provider.search_web("DeepSeek V4", 5)

    assert recorder.urls == ["https://api.tavily.com/search"]
    payload = recorder.payloads[0]
    assert payload["query"] == "DeepSeek V4"
    assert payload["max_results"] == 5
    assert payload["search_depth"] == "advanced"
    assert payload["include_answer"] is False
    assert payload["include_raw_content"] is False
    assert "topic" not in payload
    assert "start_date" not in payload
    assert "end_date" not in payload
    assert "include_published_date" not in payload
    assert payload["api_key"] == "sk-test"

    assert response.request_id == "req-1"
    assert response.results[0].published_at == "2026-09-19T00:00:00Z"
    assert response.results[0].score == 0.8

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].provider == "search"
    assert entries[0].destination == "api.tavily.com"  # 只记 host
    assert entries[0].method == "web_search"
    assert entries[0].status == 200


async def test_search_web_forwards_topic_dates_and_overrides() -> None:
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    provider = _provider(recorder)

    await provider.search_web(
        "今日新闻",
        3,
        topic="news",
        start_date="2026-09-18",
        end_date="2026-09-19",
        include_published_date=True,
        search_depth="basic",
        include_answer=True,
    )

    payload = recorder.payloads[0]
    assert payload["topic"] == "news"
    assert payload["start_date"] == "2026-09-18"
    assert payload["end_date"] == "2026-09-19"
    assert payload["include_published_date"] is True
    assert payload["search_depth"] == "basic"
    assert payload["include_answer"] is True


async def test_search_legacy_wrapper_still_returns_result_list() -> None:
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    provider = _provider(recorder)

    results = await provider.search("查询")

    assert isinstance(results, list)
    assert results[0].title == "标题"
    assert results[0].url == "https://example.com/1"
    assert results[0].snippet == "摘要正文"


async def test_search_http_error_records_status_and_raises() -> None:
    outbound = OutboundClient(max_entries=10)
    recorder = _TransportRecorder(status=429, payload={})
    provider = _provider(recorder, outbound)

    with pytest.raises(SearchError, match="HTTP 429"):
        await provider.search_web("查询")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status == 429
    assert entries[0].method == "web_search"


async def test_search_network_error_records_none_and_is_sanitized() -> None:
    outbound = OutboundClient(max_entries=10)
    recorder = _TransportRecorder(error=httpx.ConnectError("boom-internal-detail"))
    provider = _provider(recorder, outbound)

    with pytest.raises(SearchError, match="联网搜索失败"):
        await provider.search_web("查询")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status is None
    assert "boom-internal-detail" not in str(entries[0])


async def test_search_without_outbound_records_nothing() -> None:
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    provider = _provider(recorder, outbound=None)

    response = await provider.search_web("查询")

    assert len(response.results) == 1  # 行为照常，只是不记审计


# =====================================================================
# 2. Tavily Extract（PRD 7.3）
# =====================================================================


async def test_extract_uses_markdown_advanced_and_omits_query() -> None:
    outbound = OutboundClient(max_entries=10)
    recorder = _TransportRecorder(
        payload={
            "results": [
                {
                    "url": "https://example.com/1",
                    "final_url": "https://example.com/final",
                    "title": "标题",
                    "published_date": "2026-09-19T00:00:00Z",
                    "raw_content": "# 标题\n正文",
                }
            ],
            "failed_results": [{"url": "https://example.com/2", "error": "404 not found"}],
            "request_id": "req-x",
        }
    )
    provider = _provider(recorder, outbound)

    response = await provider.extract(["https://example.com/1", "https://example.com/2"])

    assert recorder.urls == ["https://api.tavily.com/extract"]
    payload = recorder.payloads[0]
    assert payload["urls"] == ["https://example.com/1", "https://example.com/2"]
    assert payload["format"] == "markdown"
    assert payload["extract_depth"] == "advanced"
    assert "query" not in payload  # 省略 query = 要整页内容

    assert response.request_id == "req-x"
    page = response.pages[0]
    assert page.content == "# 标题\n正文"
    assert page.final_url == "https://example.com/final"
    assert page.published_at == "2026-09-19T00:00:00Z"
    assert page.content_complete is True
    assert response.failed[0].url == "https://example.com/2"
    assert response.failed[0].error.startswith("404")

    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].method == "web_extract"
    assert entries[0].status == 200


async def test_extract_forwards_query_and_flags_truncated_and_empty() -> None:
    recorder = _TransportRecorder(
        payload={
            "results": [
                {"url": "https://example.com/1", "raw_content": "部分内容", "truncated": True},
                {"url": "https://example.com/2", "raw_content": ""},
            ]
        }
    )
    provider = _provider(recorder)

    response = await provider.extract(["https://example.com/1"], query="关键指标")

    assert recorder.payloads[0]["query"] == "关键指标"
    assert response.pages[0].content_complete is False  # 提供商明确截断
    assert response.pages[1].content == ""
    assert response.pages[1].content_complete is False  # 空正文同样不算完整


async def test_extract_http_error_audits_and_raises() -> None:
    outbound = OutboundClient(max_entries=10)
    recorder = _TransportRecorder(status=500, payload={})
    provider = _provider(recorder, outbound)

    with pytest.raises(SearchError, match="网页读取失败（HTTP 500）"):
        await provider.extract(["https://example.com/1"])

    assert [entry.method for entry in outbound.entries()] == ["web_extract"]
    assert outbound.entries()[0].status == 500


async def test_extract_rejects_empty_and_oversized_url_lists_without_network() -> None:
    recorder = _TransportRecorder(payload={"results": []})
    provider = _provider(recorder)

    with pytest.raises(SearchError, match="未提供 URL"):
        await provider.extract([])
    with pytest.raises(SearchError, match="最多 5 个 URL"):
        await provider.extract([f"https://example.com/{index}" for index in range(6)])

    assert recorder.requests == []  # 参数校验先于任何出网


# =====================================================================
# 3. web_search 工具
# =====================================================================


async def test_web_search_tool_returns_hits_and_request_id(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()], "request_id": "req-1"})
    outbound = OutboundClient(max_entries=10)
    context = _tool_context(_runtime_workspace(tmp_path), provider=_provider(recorder, outbound))

    result = await registry.execute(ToolName.WEB_SEARCH, {"query": "公开资料"}, context)

    assert result.status is ToolStatus.OK
    assert result.network is True
    assert result.data["request_id"] == "req-1"
    assert result.data["count"] == 1
    assert result.data["results"][0]["url"] == "https://example.com/1"
    assert result.data["results"][0]["published_at"] == "2026-09-19T00:00:00Z"
    assert result.data["redacted"] is False
    assert len(outbound.entries()) == 1


async def test_web_search_tool_empty_and_partial_limits(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": []})
    provider = _provider(recorder)
    context = _tool_context(_runtime_workspace(tmp_path), provider=provider)

    empty = await registry.execute(ToolName.WEB_SEARCH, {"query": "没有结果"}, context)
    limited = await registry.execute(
        ToolName.WEB_SEARCH, {"query": "限定", "max_results": 2, "topic": "news"}, context
    )

    assert empty.status is ToolStatus.EMPTY
    assert empty.data["code"] == "no_results"
    assert recorder.payloads[0]["max_results"] == 5
    assert recorder.payloads[1]["max_results"] == 2
    assert recorder.payloads[1]["topic"] == "news"
    assert limited.status is ToolStatus.EMPTY


async def test_web_search_tool_without_provider_is_unavailable(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _tool_context(_runtime_workspace(tmp_path), provider=None)

    with pytest.raises(ToolUnavailableError):
        await registry.execute(ToolName.WEB_SEARCH, {"query": "查询"}, context)


async def test_web_search_tool_maps_search_error_to_service_error(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(status=503, payload={})
    outbound = OutboundClient(max_entries=10)
    context = _tool_context(_runtime_workspace(tmp_path), provider=_provider(recorder, outbound))

    with pytest.raises(ToolExecutionError) as info:
        await registry.execute(ToolName.WEB_SEARCH, {"query": "查询"}, context)

    assert info.value.code is ToolErrorCode.SERVICE_ERROR
    assert len(outbound.entries()) == 1


async def test_web_search_tool_redacts_arguments_before_wire(tmp_path: Path) -> None:
    """脱敏挂载点必须作用在**真正出网的参数**上（本任务只留挂载点，不做姓名识别）。"""

    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    provider = _provider(recorder)
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=provider,
        redactor=lambda tool, arguments: {**arguments, "query": "[人名已脱敏]"},
    )

    result = await registry.execute(ToolName.WEB_SEARCH, {"query": "张三 的报销"}, context)

    assert recorder.payloads[0]["query"] == "[人名已脱敏]"
    assert result.data["query"] == "[人名已脱敏]"
    assert result.data["redacted"] is True


async def test_web_search_tool_supports_legacy_search_only_provider(tmp_path: Path) -> None:
    registry = build_default_registry()
    provider = _LegacySearchOnlyProvider(
        [SearchResult(title="标题", url="https://example.com/1", snippet="摘要")]
    )
    context = _tool_context(_runtime_workspace(tmp_path), provider=provider)

    result = await registry.execute(ToolName.WEB_SEARCH, {"query": "查询"}, context)

    assert result.status is ToolStatus.OK
    assert result.data["request_id"] == ""
    assert result.data["results"][0]["snippet"] == "摘要"
    assert provider.calls == ["查询"]


# =====================================================================
# 4. read_web_page 工具
# =====================================================================


async def test_read_web_page_short_page_is_inline_without_artifact(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(
        payload={"results": [{"url": "https://example.com/1", "raw_content": "短正文"}]}
    )
    workspace = _runtime_workspace(tmp_path)
    context = _tool_context(workspace, provider=_provider(recorder))

    result = await registry.execute(
        ToolName.READ_WEB_PAGE, {"urls": ["https://example.com/1"]}, context
    )

    assert result.status is ToolStatus.OK
    page = result.data["pages"][0]
    assert page["content"] == "短正文"
    assert page["artifact_id"] == ""  # 短页不落 artifacts（缺失字段保存为空）
    assert page["content_complete"] is True
    assert page["read_range"] == {"offset": 0, "length": 3, "total": 3, "complete": True}
    assert not (workspace.artifacts_dir / "web").exists()


async def test_read_web_page_long_page_writes_artifact_and_supports_windows(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(
        payload={
            "results": [{"url": "https://example.com/long", "raw_content": LONG_PAGE}],
            "request_id": "req-long",
        }
    )
    workspace = _runtime_workspace(tmp_path)
    outbound = OutboundClient(max_entries=10)
    context = _tool_context(workspace, provider=_provider(recorder, outbound))

    result = await registry.execute(
        ToolName.READ_WEB_PAGE, {"urls": ["https://example.com/long"], "limit": 100}, context
    )

    assert result.status is ToolStatus.PARTIAL  # 只读了窗口，正文未读完
    page = result.data["pages"][0]
    assert page["content"] == LONG_PAGE[:100]
    assert page["content_complete"] is False
    assert page["read_range"]["complete"] is False
    assert page["read_range"]["total"] == 2500
    assert len(page["content_hash"]) == 64

    artifact = workspace.artifacts_dir / "web" / f"{page['content_hash']}.md"
    assert artifact.is_file()
    assert artifact.read_text(encoding="utf-8") == LONG_PAGE
    assert result.artifact_id == page["content_hash"]
    assert result.artifact_ref == f"web/{page['content_hash']}.md"
    assert len(outbound.entries()) == 1


async def test_read_web_page_reports_partial_and_empty_item_by_item(tmp_path: Path) -> None:
    registry = build_default_registry()
    partial = _TransportRecorder(
        payload={
            "results": [{"url": "https://example.com/1", "raw_content": "甲"}],
            "failed_results": [{"url": "https://example.com/2", "error": "404"}],
        }
    )
    nothing = _TransportRecorder(
        payload={
            "results": [],
            "failed_results": [{"url": "https://example.com/3", "error": "timeout"}],
        }
    )
    workspace = _runtime_workspace(tmp_path)

    partial_result = await registry.execute(
        ToolName.READ_WEB_PAGE,
        {"urls": ["https://example.com/1", "https://example.com/2"]},
        _tool_context(workspace, provider=_provider(partial)),
    )
    empty_result = await registry.execute(
        ToolName.READ_WEB_PAGE,
        {"urls": ["https://example.com/3"]},
        _tool_context(workspace, provider=_provider(nothing)),
    )

    assert partial_result.status is ToolStatus.PARTIAL
    assert partial_result.data["failed_count"] == 1
    assert partial_result.data["failed"][0]["url"] == "https://example.com/2"
    assert empty_result.status is ToolStatus.EMPTY
    assert empty_result.data["pages"] == []
    assert empty_result.data["code"] == "no_content"


async def test_read_web_page_requires_extract_capable_provider(tmp_path: Path) -> None:
    registry = build_default_registry()
    legacy = _LegacySearchOnlyProvider([])  # 只有 search()，没有 extract()
    context = _tool_context(_runtime_workspace(tmp_path), provider=legacy)

    with pytest.raises(ToolUnavailableError):
        await registry.execute(
            ToolName.READ_WEB_PAGE, {"urls": ["https://example.com/1"]}, context
        )


async def test_read_web_page_forwards_query_and_redacts_urls(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(
        payload={"results": [{"url": "https://example.com/1", "raw_content": "正文"}]}
    )
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder),
        redactor=lambda tool, arguments: {**arguments, "query": "已脱敏"},
    )

    await registry.execute(
        ToolName.READ_WEB_PAGE,
        {"urls": ["https://example.com/1"], "query": "张三"},
        context,
    )

    assert recorder.payloads[0]["query"] == "已脱敏"


# =====================================================================
# 5. A01 与出网收敛
# =====================================================================


async def test_a01_web_path_never_touches_local_knowledge_retrieval(tmp_path: Path) -> None:
    """A01：Web 路径下本地知识正文检索次数必须是 0。"""

    registry = build_default_registry()
    spy = _SpyRetriever()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    outbound = OutboundClient(max_entries=10)
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder, outbound),
        retriever=spy,
        embedder=FakeEmbedder(),
    )

    assert registry.get(ToolName.SEARCH_KNOWLEDGE).allows(AgentKind.WEB) is False
    knowledge_calls = (
        (ToolName.SEARCH_KNOWLEDGE, {"query": "内部报销制度"}),
        (ToolName.READ_DOCUMENT, {"document_id": "doc-a"}),
        (ToolName.LOOKUP_CATALOG, {"query": "内部报销制度"}),
    )
    for name, arguments in knowledge_calls:
        with pytest.raises(ToolPermissionError) as info:
            await registry.execute(name, arguments, context)
        assert info.value.code is ToolErrorCode.NOT_PERMITTED

    assert spy.calls == []

    web = await registry.execute(ToolName.WEB_SEARCH, {"query": "公开资料"}, context)

    assert web.status is ToolStatus.OK
    assert spy.calls == []  # Web 路径跑完，本地正文检索次数仍为 0
    assert len(outbound.entries()) == 1


async def test_read_document_for_web_agent_is_rejected_without_reading(tmp_path: Path) -> None:
    registry = build_default_registry()
    context = _tool_context(_runtime_workspace(tmp_path))

    with pytest.raises(ToolPermissionError):
        await registry.execute(ToolName.READ_DOCUMENT, {"document_id": "doc-a"}, context)


async def test_all_outbound_goes_through_outbound_client(tmp_path: Path) -> None:
    """出网只经 OutboundClient：注入传输层的请求数与审计事件数一一对应。"""

    registry = build_default_registry()
    outbound = OutboundClient(max_entries=10)
    recorder = _TransportRecorder(
        payload={
            "results": [_hit()],
            "failed_results": [],
            "request_id": "req-1",
        }
    )
    context = _tool_context(_runtime_workspace(tmp_path), provider=_provider(recorder, outbound))

    await registry.execute(ToolName.WEB_SEARCH, {"query": "公开资料"}, context)
    await registry.execute(
        ToolName.READ_WEB_PAGE, {"urls": ["https://example.com/1"]}, context
    )

    assert [entry.method for entry in outbound.entries()] == ["web_search", "web_extract"]
    assert len(recorder.requests) == 2
    assert outbound.outbound_state == "has-outbound"
    assert all(entry.destination == "api.tavily.com" for entry in outbound.entries())
    outbound.clear()
    assert outbound.outbound_state == "local-only"


async def test_web_pages_are_read_without_a_workspace_artifacts_dir(tmp_path: Path) -> None:
    """工作区没有 artifacts 目录时，短页仍可读（长页才需要落盘）。"""

    registry = build_default_registry()
    recorder = _TransportRecorder(
        payload={"results": [{"url": "https://example.com/1", "raw_content": "短正文"}]}
    )
    context = _tool_context(_WorkspaceStub(), provider=_provider(recorder))

    short = await registry.execute(
        ToolName.READ_WEB_PAGE, {"urls": ["https://example.com/1"]}, context
    )
    assert short.status is ToolStatus.OK

    long_recorder = _TransportRecorder(
        payload={"results": [{"url": "https://example.com/long", "raw_content": LONG_PAGE}]}
    )
    with pytest.raises(ToolUnavailableError):
        await registry.execute(
            ToolName.READ_WEB_PAGE,
            {"urls": ["https://example.com/long"]},
            _tool_context(_WorkspaceStub(), provider=_provider(long_recorder)),
        )
