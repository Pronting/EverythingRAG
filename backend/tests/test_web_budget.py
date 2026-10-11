"""联网预算与「思考等级 → 联网」机制测试（用户裁定）。

覆盖：各等级额度与单调性、关闭思考禁用、限定资料范围禁用、额度硬门（搜索/网页/时长）、
重复查询不重复出网、尝试次数上限、工具层在出网前拒绝、Web 分支降级为资料不足（不判死
run）、研究循环按缺口申请补充搜索、补充结果并回父任务后覆盖率重算。全程零真实外呼。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.agent.contracts import (
    AgentKind,
    EvidenceSourceType,
    IntentKind,
    ReasoningLevel,
    ResearchTask,
    RouteDecision,
    TaskStatus,
)
from app.agent.research import (
    ConclusionBasis,
    RagResearcher,
    RequirementState,
    ResearchNextAction,
    ResearchProgress,
    ResearchRequest,
    ResearchResult,
    WebResearcher,
    make_evidence,
    merge_supplementary_result,
)
from app.agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolName,
    ToolStatus,
    WebBudgetExhaustedError,
    build_default_registry,
)
from app.agent.web_budget import (
    DEFAULT_WEB_BUDGETS,
    MAX_ATTEMPTS_PER_OPERATION,
    REASON_BUDGET_EXHAUSTED,
    REASON_DISABLED,
    REASON_DUPLICATE,
    REASON_NOT_CONFIGURED,
    REASON_TIME_EXHAUSTED,
    WebBudget,
    WebBudgetLedger,
    budgets_from_env,
    local_only_question,
    resolve_web_policy,
)
from app.core.outbound import OutboundClient
from tests.fakes_agent import app_store  # noqa: F401 - 共享夹具（内存 app.db）
from tests.test_web_tools import (
    _hit,
    _provider,
    _runtime_workspace,
    _tool_context,
    _TransportRecorder,
)

# =====================================================================
# 1. 集中配置与策略
# =====================================================================


def test_default_budgets_are_zero_off_and_increase_with_level() -> None:
    off = DEFAULT_WEB_BUDGETS[ReasoningLevel.OFF]
    low = DEFAULT_WEB_BUDGETS[ReasoningLevel.LOW]
    high = DEFAULT_WEB_BUDGETS[ReasoningLevel.HIGH]
    top = DEFAULT_WEB_BUDGETS[ReasoningLevel.MAX]

    assert off.queries == 0 and off.pages == 0 and off.seconds == 0
    for smaller, larger in ((low, high), (high, top)):
        assert smaller.queries < larger.queries
        assert smaller.pages < larger.pages
        assert smaller.seconds < larger.seconds


def test_policy_disables_web_when_thinking_is_off() -> None:
    policy = resolve_web_policy(reasoning_level=ReasoningLevel.OFF)

    assert policy.allowed is False
    assert policy.reason_code == REASON_DISABLED
    assert "关闭思考" in policy.reason_text
    assert policy.budget.queries == 0


def test_policy_allows_web_with_level_budget() -> None:
    policy = resolve_web_policy(reasoning_level=ReasoningLevel.HIGH)

    assert policy.allowed is True
    assert policy.budget == DEFAULT_WEB_BUDGETS[ReasoningLevel.HIGH]


def test_policy_respects_local_only_scope_and_unconfigured_provider() -> None:
    scoped = resolve_web_policy(reasoning_level=ReasoningLevel.HIGH, knowledge_only=True)
    assert scoped.allowed is False and scoped.reason_code == REASON_DISABLED

    asked = resolve_web_policy(
        reasoning_level=ReasoningLevel.HIGH, local_only_question_flag=True
    )
    assert asked.allowed is False

    unconfigured = resolve_web_policy(
        reasoning_level=ReasoningLevel.HIGH, search_configured=False
    )
    assert unconfigured.allowed is False
    assert unconfigured.reason_code == REASON_NOT_CONFIGURED


def test_local_only_question_detection() -> None:
    assert local_only_question("只根据我的资料回答，不要用你的通用知识。")
    assert local_only_question("仅使用本地文档回答这个问题")
    assert local_only_question("这条只以知识库为准")
    assert not local_only_question("货拉拉的记忆系统有淘汰策略吗？")


def test_budgets_can_be_overridden_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVERYTHING_RAG_WEB_BUDGETS", '{"high": [1, 2, 3]}')

    budgets = budgets_from_env()

    assert budgets[ReasoningLevel.HIGH] == WebBudget(1, 2, 3.0)
    assert budgets[ReasoningLevel.MAX] == DEFAULT_WEB_BUDGETS[ReasoningLevel.MAX]


# =====================================================================
# 2. 额度账本
# =====================================================================


def _ledger(level: ReasoningLevel = ReasoningLevel.LOW, **kwargs: Any) -> WebBudgetLedger:
    return WebBudgetLedger(resolve_web_policy(reasoning_level=level, **kwargs))


def test_ledger_enforces_query_and_page_caps() -> None:
    ledger = _ledger(ReasoningLevel.LOW)  # 2 查询 / 3 页 / 20s

    assert ledger.reserve_search("第一个问题").allowed
    assert ledger.reserve_search("第二个问题").allowed
    denied = ledger.reserve_search("第三个问题")
    assert denied.allowed is False
    assert denied.reason_code == REASON_BUDGET_EXHAUSTED
    assert ledger.queries_used == 2

    assert ledger.reserve_pages(3).allowed
    assert ledger.reserve_pages(1).allowed is False
    assert ledger.pages_used == 3


def test_ledger_skips_duplicate_query_without_spending_budget() -> None:
    ledger = _ledger(ReasoningLevel.LOW)

    assert ledger.reserve_search("OpenClaw 记忆淘汰").allowed
    again = ledger.reserve_search("  openclaw   记忆淘汰 ")

    assert again.allowed is False
    assert again.reason_code == REASON_DUPLICATE
    assert ledger.queries_used == 1
    assert ledger.attempts_for("OpenClaw 记忆淘汰") == 1


def test_ledger_counts_failures_and_blocks_endless_retry() -> None:
    ledger = _ledger(ReasoningLevel.HIGH)

    ledger.reserve_search("会失败的查询")
    ledger.note_result("web_search", ok=False, detail="503", query="会失败的查询")
    ledger.reserve_search("会失败的查询")  # 重试仍要占用额度

    assert ledger.queries_used == 1  # 重复查询不重复扣减
    assert ledger.attempts_for("会失败的查询") == 1
    assert ledger.attempts_exhausted("会失败的查询") is False
    ledger._attempts["会失败的查询"] = MAX_ATTEMPTS_PER_OPERATION
    assert ledger.attempts_exhausted("会失败的查询") is True


def test_ledger_denies_when_policy_forbids() -> None:
    ledger = _ledger(ReasoningLevel.OFF)

    denied = ledger.reserve_search("任何查询")

    assert denied.allowed is False
    assert denied.reason_code == REASON_DISABLED
    assert ledger.queries_used == 0


def test_ledger_stops_after_time_budget() -> None:
    now = {"value": 0.0}
    ledger = WebBudgetLedger(
        resolve_web_policy(reasoning_level=ReasoningLevel.LOW),
        clock=lambda: now["value"],
    )
    assert ledger.reserve_search("第一").allowed
    now["value"] = 25.0  # 超过 20s 时长额度

    denied = ledger.reserve_pages(1)

    assert denied.allowed is False
    assert denied.reason_code == REASON_TIME_EXHAUSTED


def test_ledger_snapshot_and_summary_are_readable() -> None:
    ledger = _ledger(ReasoningLevel.HIGH)
    ledger.reserve_search("查询一", phase="web:t-r1-web")

    snapshot = ledger.snapshot()

    assert snapshot["queries_used"] == 1
    assert snapshot["queries_limit"] == DEFAULT_WEB_BUDGETS[ReasoningLevel.HIGH].queries
    assert snapshot["phase"] == "web:t-r1-web"
    assert "查询 1/4" in ledger.summary_text()
    assert ledger.events()[0]["tool"] == "web_search"


# =====================================================================
# 3. 工具层硬门（不是提示词要求）
# =====================================================================


async def test_web_search_tool_is_refused_before_any_outbound_call(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    outbound = OutboundClient(max_entries=10)
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder, outbound),
        web_budget=_ledger(ReasoningLevel.OFF),
    )

    with pytest.raises(WebBudgetExhaustedError) as info:
        await registry.execute(ToolName.WEB_SEARCH, {"query": "查询"}, context)

    assert info.value.code is ToolErrorCode.WEB_BUDGET_EXHAUSTED
    assert recorder.requests == []  # 没有出网
    assert outbound.entries() == []


async def test_web_search_tool_records_budget_snapshot_when_allowed(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    ledger = _ledger(ReasoningLevel.HIGH)
    context = _tool_context(
        _runtime_workspace(tmp_path), provider=_provider(recorder), web_budget=ledger
    )

    result = await registry.execute(ToolName.WEB_SEARCH, {"query": "公开资料"}, context)

    assert result.status is ToolStatus.OK
    assert result.data["web_budget"]["queries_used"] == 1
    assert ledger.queries_used == 1


async def test_duplicate_query_reports_its_own_error_code(tmp_path: Path) -> None:
    """重复查询与服务错误分开报（「防止重复执行相同查询」要能被看见）。"""

    from app.agent.tools import WebQueryDuplicateError

    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder),
        web_budget=_ledger(ReasoningLevel.HIGH),
    )

    await registry.execute(ToolName.WEB_SEARCH, {"query": "同一个查询"}, context)
    with pytest.raises(WebQueryDuplicateError) as info:
        await registry.execute(ToolName.WEB_SEARCH, {"query": " 同一个查询 "}, context)

    assert info.value.code is ToolErrorCode.WEB_QUERY_DUPLICATE
    assert len(recorder.requests) == 1


async def test_web_search_tool_stops_after_query_budget(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(payload={"results": [_hit()]})
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder),
        web_budget=_ledger(ReasoningLevel.LOW),
    )

    await registry.execute(ToolName.WEB_SEARCH, {"query": "一"}, context)
    await registry.execute(ToolName.WEB_SEARCH, {"query": "二"}, context)
    with pytest.raises(WebBudgetExhaustedError):
        await registry.execute(ToolName.WEB_SEARCH, {"query": "三"}, context)

    assert len(recorder.requests) == 2


async def test_read_web_page_narrows_to_remaining_pages(tmp_path: Path) -> None:
    registry = build_default_registry()
    recorder = _TransportRecorder(
        payload={"results": [{"url": "https://example.com/1", "raw_content": "正文"}], "failed_results": []}
    )
    context = _tool_context(
        _runtime_workspace(tmp_path),
        provider=_provider(recorder),
        extract_provider=_provider(recorder),
        web_budget=_ledger(ReasoningLevel.LOW),  # 3 页额度
    )

    result = await registry.execute(
        ToolName.READ_WEB_PAGE,
        {"urls": [f"https://example.com/{index}" for index in range(5)]},
        context,
    )

    assert result.status in {ToolStatus.OK, ToolStatus.PARTIAL}
    assert result.data["web_budget"]["pages_used"] == 3
    assert len(recorder.payloads[0]["urls"]) == 3


# =====================================================================
# 4. Web 分支：额度不可用时降级为资料不足（不判死 run）
# =====================================================================


class _DenyingRegistry:
    """执行两次：第一次由工具层抛「额度不允许」，验证研究分支的降级收口。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, name: Any, arguments: Any, context: Any, **kwargs: Any) -> Any:
        del kwargs
        self.calls.append(str(name))
        raise WebBudgetExhaustedError("当前为「关闭思考」模式，联网已禁用")


def _web_request() -> ResearchRequest:
    task = ResearchTask(
        task_id="t-r1-web",
        question="OpenClaw 的记忆淘汰是怎么做的？",
        requirement_ids=["R1"],
        agent_kind=AgentKind.WEB,
    )
    return ResearchRequest(
        task=task,
        original_question="OpenClaw 的记忆淘汰是怎么做的？",
        requirement_texts={"R1": "OpenClaw 的记忆淘汰是怎么做的？"},
        decision=RouteDecision(
            question="q", resolved_question="q", intent=IntentKind.RESEARCH, needs_web=True
        ),
    )


async def test_web_researcher_degrades_to_gap_when_budget_denied() -> None:
    registry = _DenyingRegistry()

    result = await WebResearcher(registry).research(_web_request(), _tool_context_stub())

    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.stop_reason == "web_budget_unavailable"
    assert result.evidence == []
    assert result.gaps and result.gaps[0].requirement_ids == ["R1"]
    assert any("联网不可用" in item for item in result.limitations)
    assert result.progress.next_action is ResearchNextAction.INSUFFICIENT


def _tool_context_stub() -> ToolContext:
    from tests.test_research_loop import _Workspace

    return ToolContext(
        workspace=_Workspace(),
        run_id="run-a",
        task_id="t-r1-web",
        agent_kind=AgentKind.WEB,
    )


# =====================================================================
# 5. 研究循环按缺口申请补充联网
# =====================================================================


class _EmptyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: Any, arguments: Any, context: Any, **kwargs: Any) -> Any:
        del context, kwargs
        from tests.test_research_loop import _result

        self.calls.append((str(name), dict(arguments)))
        return _result(ToolStatus.EMPTY, {"code": "no_results", "chunks": [], "count": 0})


def _rag_request(*, web_allowed: bool) -> ResearchRequest:
    task = ResearchTask(
        task_id="t-r1-knowledge",
        question="货拉拉的记忆淘汰策略是什么？",
        requirement_ids=["R1"],
        agent_kind=AgentKind.RAG,
    )
    return ResearchRequest(
        task=task,
        original_question="货拉拉的记忆淘汰策略是什么？",
        requirement_texts={"R1": "货拉拉的记忆淘汰策略是什么？"},
        decision=RouteDecision(
            question="q",
            resolved_question="q",
            intent=IntentKind.RESEARCH,
            needs_knowledge=True,
        ),
        web_allowed=web_allowed,
    )


async def test_rag_loop_requests_supplementary_web_when_allowed() -> None:
    registry = _EmptyRegistry()

    result = await RagResearcher(registry, max_query_reformulations=0).research(
        _rag_request(web_allowed=True), _tool_context_stub()
    )

    assert result.web_requests, "本地资料不足且允许联网时必须提出补充搜索"
    assert result.stop_reason == "web_subtask_required"


async def test_rag_loop_does_not_request_web_when_not_allowed() -> None:
    registry = _EmptyRegistry()

    result = await RagResearcher(registry, max_query_reformulations=0).research(
        _rag_request(web_allowed=False), _tool_context_stub()
    )

    assert result.web_requests == []
    assert result.stop_reason == "insufficient_material"


# =====================================================================
# 6. 补充结果并回父任务
# =====================================================================


def _parent_result() -> ResearchResult:
    return ResearchResult(
        task_id="t-r1-knowledge",
        agent_kind=AgentKind.RAG,
        status=TaskStatus.COMPLETED_WITH_GAPS,
        evidence=[],
        conclusions=[],
        progress=ResearchProgress(
            task_id="t-r1-knowledge",
            round=1,
            requirement_coverage={"R1": RequirementState.MISSING},
        ),
        stop_reason="web_subtask_required",
    )


def test_merge_supplementary_result_recomputes_coverage_and_conclusions() -> None:
    web_evidence = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id="t-r1-web",
        source_type=EvidenceSourceType.WEB,
        source_ref="https://example.com/a",
        content="淘汰策略：按 TTL 过期后清理，容量上限触发 LRU。",
        supported_requirements=["R1"],
    )
    extra = ResearchResult(
        task_id="t-r1-web",
        agent_kind=AgentKind.WEB,
        status=TaskStatus.COMPLETED,
        evidence=[web_evidence],
        progress=ResearchProgress(task_id="t-r1-web", round=1),
    )

    merged = merge_supplementary_result(
        _parent_result(),
        extra,
        requirement_texts={"R1": "货拉拉的记忆淘汰策略是什么？"},
        detail="补充联网检索",
    )

    assert merged.status is TaskStatus.COMPLETED
    assert merged.progress.requirement_coverage["R1"] is RequirementState.ANSWERED
    assert merged.conclusions[0].basis is ConclusionBasis.SUPPORTED
    assert merged.conclusions[0].evidence_ids == [web_evidence.evidence_id]
    assert merged.gaps == []
    assert merged.progress.steps[-1].detail == "补充联网检索"


def test_merge_supplementary_result_keeps_gap_when_still_missing() -> None:
    extra = ResearchResult(
        task_id="t-r1-web",
        agent_kind=AgentKind.WEB,
        status=TaskStatus.COMPLETED_WITH_GAPS,
        limitations=["联网不可用"],
        progress=ResearchProgress(task_id="t-r1-web", round=1),
    )

    merged = merge_supplementary_result(
        _parent_result(), extra, requirement_texts={"R1": "货拉拉的记忆淘汰策略是什么？"}
    )

    assert merged.status is TaskStatus.COMPLETED_WITH_GAPS
    assert merged.gaps and merged.gaps[0].requirement_ids == ["R1"]
    assert "联网不可用" in merged.limitations


# =====================================================================
# 7. 补充联网子任务必须先落 agent_tasks（operations 有复合外键）
# =====================================================================


async def test_operation_registration_requires_the_task_row(app_store: Any) -> None:
    """回归：联网操作登记前，子任务必须已存在于 ``agent_tasks``。

    真实运行中复现过一次 ``IntegrityError``：补充联网子任务没落库就去登记 ``operations``，
    触发复合外键 ``(run_id, task_id) → agent_tasks(run_id, id)``。这里把该契约钉住。
    """

    await app_store.create_workspace("ws-1", "sk-ws-1", name="甲区")
    await app_store.create_conversation("conv-1", "ws-1")
    run = await app_store.create_run(
        conversation_id="conv-1", workspace_id="ws-1", client_request_id=None
    )
    await app_store.create_task(
        run_id=run.id,
        task_id="t-r1-knowledge-web1",
        agent_kind=AgentKind.WEB,
        question="缺口问题",
    )

    record = await app_store.register_operation(
        operation_id="op-1",
        tool_name="web_search",
        args_hash="hash",
        run_id=run.id,
        task_id="t-r1-knowledge-web1",
    )

    assert record.operation_id == "op-1"
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        await app_store.register_operation(
            operation_id="op-2",
            tool_name="web_search",
            args_hash="hash",
            run_id=run.id,
            task_id="t-unknown-task",
        )
