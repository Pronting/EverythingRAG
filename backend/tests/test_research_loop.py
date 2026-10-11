"""研究循环测试（PRD 7）。Owner: T07。

覆盖：六步可观察、`context_top_k=8`、原问题与内部术语保留、空 / 无关 / 服务错误的准确
状态、精度线索读原文、每轮六字段更新、四条件判定、相同资料停滞、Web 子任务、证据去重与
要求归属、Web 分支（Search/Extract/日期范围/独立来源）、General 分支（不得生成引用编号）、
A10 场景。全程离线：假注册表 + 假工具结果，无 Chroma、无出网、无模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.agent.contracts import (
    AgentKind,
    EvidenceSourceType,
    Freshness,
    IntentKind,
    ResearchTask,
    RouteDecision,
    TaskStatus,
    citation_prefix,
)
from app.agent.research import (
    CONDITION_AMBIGUITY,
    CONDITION_DIFFERENCES,
    CONDITION_REQUIREMENTS,
    CONDITION_SOURCED,
    DEFAULT_TOP_K,
    GENERAL_EVIDENCE_LIMITATION,
    RESEARCH_STEPS,
    ConclusionBasis,
    GeneralResearcher,
    MaterialGap,
    RagResearcher,
    RequirementState,
    ResearchConclusion,
    ResearchNextAction,
    ResearchNotSupportedError,
    ResearchProgress,
    ResearchRequest,
    ResearchStepName,
    WebPageEvidence,
    WebPageReading,
    WebResearcher,
    attribute_requirements,
    content_hash_of,
    dedupe_evidence,
    detect_numeric_conflicts,
    evaluate_completion,
    evidence_index,
    independent_source_report,
    make_evidence,
    numeric_claims,
    research_task,
    researcher_for,
    resolve_date_range,
    text_supports,
)
from app.agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolResult,
    ToolStatus,
    ToolUnavailableError,
)

REQUIREMENT_TEXT = "报销制度的时限是多少"
REQUIREMENT_ID = "R1"
TASK_ID = "t-r1-knowledge"
CHUNK_TEXT = "报销申请的时限为 15 个工作日，需在费用发生后提交，逾期不予受理。"


# =====================================================================
# 替身与夹具
# =====================================================================


@dataclass(frozen=True)
class _Workspace:
    """工具上下文需要的最小工作区（不触碰真实数据目录）。"""

    workspace_id: str = "ws-a"
    name: str = "甲区"
    knowledge_revision: int = 3
    artifacts_dir: Path | None = None
    default_permission_mode: str = "ask"


def _result(
    status: ToolStatus,
    data: dict[str, Any] | None = None,
    *,
    message: str = "",
    artifact_id: str = "",
    network: bool = False,
) -> ToolResult:
    return ToolResult(
        tool="",
        status=status,
        data=dict(data or {}),
        message=message,
        artifact_id=artifact_id,
        network=network,
    )


@dataclass
class _ScriptedRegistry:
    """按脚本返回 ``ToolResult`` 的注册表替身（只实现研究循环消费的 ``execute``）。"""

    search: dict[str, ToolResult] = field(default_factory=dict)
    reads: dict[str, ToolResult] = field(default_factory=dict)
    history: ToolResult | None = None
    web: ToolResult | None = None
    extract: ToolResult | None = None
    failures: dict[str, Exception] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def execute(
        self, name: str | ToolName, arguments: dict[str, Any], context: ToolContext, **_kw: Any
    ) -> ToolResult:
        tool = str(name)
        args = dict(arguments)
        self.calls.append((tool, args))
        if tool in self.failures:
            raise self.failures[tool]
        if tool == str(ToolName.SEARCH_KNOWLEDGE):
            return self.search.get(
                str(args.get("query")),
                _result(ToolStatus.EMPTY, {"code": "no_results", "chunks": [], "count": 0}),
            )
        if tool == str(ToolName.READ_DOCUMENT):
            return self.reads.get(
                str(args.get("document_id")),
                _result(
                    ToolStatus.NOT_FOUND,
                    {"code": "document_not_found", "document_id": args.get("document_id")},
                ),
            )
        if tool == str(ToolName.SEARCH_HISTORY):
            return self.history or _result(ToolStatus.EMPTY, {"messages": [], "total": 0})
        if tool == str(ToolName.WEB_SEARCH):
            return self.web or _result(ToolStatus.EMPTY, {"results": [], "count": 0})
        if tool == str(ToolName.READ_WEB_PAGE):
            return self.extract or _result(ToolStatus.EMPTY, {"pages": [], "failed": []})
        raise AssertionError(f"未脚本化的工具调用：{tool}")

    def args_of(self, tool: ToolName) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == str(tool)]

    def tools(self) -> list[str]:
        return [name for name, _args in self.calls]


def _chunk(
    document_id: str = "doc-a",
    *,
    index: int = 0,
    text: str = CHUNK_TEXT,
    similarity: float | None = 0.9,
    heading: str = "第一章 报销时限",
    name: str = "报销制度.md",
) -> dict[str, Any]:
    return {
        "block_id": f"{document_id}::chunk-{index}",
        "document_id": document_id,
        "document_name": name,
        "topic": "内部资料",
        "text": text,
        "similarity": similarity,
        "match_type": "dense",
        "heading_path": heading,
        "anchor": heading,
        "chunk_type": "text",
        "platform": "local",
    }


def _search_result(chunks: list[dict[str, Any]], *, status: ToolStatus = ToolStatus.OK) -> ToolResult:
    data: dict[str, Any] = {
        "chunks": chunks,
        "count": len(chunks),
        "candidate_count": len(chunks),
        "knowledge_revision": 3,
    }
    if status is ToolStatus.IRRELEVANT:
        data["code"] = "irrelevant"
    return _result(status, data)


def _read_result(
    text: str = CHUNK_TEXT,
    *,
    revision: str = "rev-9",
    document_id: str = "doc-a",
    total: int | None = None,
    offset: int = 0,
) -> ToolResult:
    return _result(
        ToolStatus.OK,
        {
            "document_id": document_id,
            "document_name": "报销制度.md",
            "topic": "内部资料",
            "document_revision": revision,
            "block_count": 1,
            "total_chars": len(text) if total is None else total,
            "offset": offset,
            "text": text,
            "headings": ["第一章 报销时限"],
            "next_offset": None,
            "knowledge_revision": 3,
        },
    )


def _context(*, agent_kind: AgentKind = AgentKind.RAG, run_id: str = "run-a") -> ToolContext:
    return ToolContext(
        workspace=_Workspace(),
        run_id=run_id,
        task_id=TASK_ID,
        agent_kind=agent_kind,
    )


def _task(
    *,
    task_id: str = TASK_ID,
    question: str = REQUIREMENT_TEXT,
    requirement_ids: tuple[str, ...] = (REQUIREMENT_ID,),
    agent_kind: AgentKind = AgentKind.RAG,
    expected_evidence: str = "本工作区知识原文片段与文档版本（search_knowledge / read_document）",
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        question=question,
        requirement_ids=list(requirement_ids),
        agent_kind=agent_kind,
        expected_evidence=expected_evidence,
    )


def _request(
    task: ResearchTask | None = None,
    *,
    decision: RouteDecision | None = None,
    resolved_question: str = "",
    requirement_texts: dict[str, str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> ResearchRequest:
    active = task or _task()
    texts = requirement_texts or {rid: active.question for rid in active.requirement_ids}
    return ResearchRequest(
        task=active,
        original_question=active.question,
        resolved_question=resolved_question,
        requirement_texts=texts,
        decision=decision,
        date_range=date_range,
    )


def _decision(question: str = REQUIREMENT_TEXT, **overrides: Any) -> RouteDecision:
    base: dict[str, Any] = {
        "question": question,
        "resolved_question": question,
        "intent": IntentKind.RESEARCH,
        "needs_knowledge": True,
    }
    base.update(overrides)
    return RouteDecision(**base)


def _step_names(request_result: Any) -> list[str]:
    return [step.name.value for step in request_result.progress.steps]


# =====================================================================
# 1. PRD 7.2 六步循环
# =====================================================================


async def test_rag_loop_records_prd_7_2_six_steps() -> None:
    registry = _ScriptedRegistry(search={REQUIREMENT_TEXT: _search_result([_chunk()])})

    result = await RagResearcher(registry).research(_request(), _context())

    assert _step_names(result) == [step.value for step in RESEARCH_STEPS]
    assert result.status is TaskStatus.COMPLETED
    assert result.stop_reason == "completed"
    assert result.progress.next_action is ResearchNextAction.FINALIZE
    # 第 2 步确实调用了混合检索（经 ToolRegistry），第 3 步如实记录「无待读原文」
    assert registry.tools() == [str(ToolName.SEARCH_KNOWLEDGE)]
    read_step = next(step for step in result.progress.steps if step.name is ResearchStepName.READ_SOURCE)
    assert read_step.evidence_ids == []


async def test_default_context_top_k_is_eight() -> None:
    registry = _ScriptedRegistry(search={REQUIREMENT_TEXT: _search_result([_chunk()])})

    result = await RagResearcher(registry).research(_request(), _context())

    assert RagResearcher(registry).context_top_k == DEFAULT_TOP_K == 8
    assert registry.args_of(ToolName.SEARCH_KNOWLEDGE)[0]["top_k"] == 8
    assert result.evidence


async def test_queries_keep_original_question_and_internal_terms() -> None:
    registry = _ScriptedRegistry()
    decision = _decision(
        "报销时限是多少？",
        entities=["P99"],
        unresolved_terms=["风控阈值"],
    )
    request = _request(_task(question="报销时限是多少？"), decision=decision, resolved_question="报销制度的时限是多少")

    queries = RagResearcher(registry).initial_queries(request)

    assert queries[0] == "报销时限是多少？"
    assert all("报销制度的时限是多少" in query for query in queries[1:])
    assert any("P99" in query for query in queries)
    assert any("风控阈值" in query for query in queries)


async def test_empty_results_report_no_material_status() -> None:
    registry = _ScriptedRegistry()

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.stop_reason == "insufficient_material"
    assert result.evidence == []
    assert result.open_questions
    assert result.gaps and result.gaps[0].requirement_ids == [REQUIREMENT_ID]
    assert result.progress.requirement_coverage[REQUIREMENT_ID] is RequirementState.MISSING
    assert "没有命中片段" in result.progress.steps[1].detail


async def test_empty_results_reformulate_query_before_declaring_gap() -> None:
    """没有结果时，先按缺失主题改写查询再次检索（用户实测缺陷 C04），并有明确上限。"""

    registry = _ScriptedRegistry()
    researcher = RagResearcher(registry, max_query_reformulations=1)
    request = _request(
        _task(question="所以关于记忆和成本来说，你有什么比较好的思路吗"),
        resolved_question="面试要求设计一个 Agent —— 所以关于记忆和成本来说，你有什么比较好的思路吗",
        requirement_texts={REQUIREMENT_ID: "所以关于记忆和成本来说，你有什么比较好的思路吗"},
    )

    result = await researcher.research(request, _context())

    queries = [str(args.get("query")) for args in registry.args_of(ToolName.SEARCH_KNOWLEDGE)]
    assert len(queries) == len({query.casefold() for query in queries}), queries
    assert all("Agent" in query for query in queries[1:])
    assert any("记忆" in query and "成本" in query for query in queries), queries
    assert result.stop_reason == "insufficient_material"
    # 「本轮没找到」与「知识库没有」必须区分清楚
    assert "本轮共执行" in result.gaps[0].missing
    assert "不等于知识库一定没有相关内容" in result.gaps[0].impact


async def test_query_reformulation_respects_cap() -> None:
    registry = _ScriptedRegistry()
    researcher = RagResearcher(registry, max_query_reformulations=0)

    result = await researcher.research(_request(), _context())

    queries = registry.args_of(ToolName.SEARCH_KNOWLEDGE)
    assert len(queries) == 1  # 关闭改写时只跑初始查询
    assert result.stop_reason == "insufficient_material"


async def test_reformulated_query_that_hits_fills_the_gap() -> None:
    """改写后的查询命中资料时，缺口必须被填上（而不是继续报资料不足）。"""

    request = _request()
    request.requirement_texts[REQUIREMENT_ID] = "报销制度 时限 提交 逾期"
    probe = RagResearcher(_ScriptedRegistry(), max_query_reformulations=1)
    candidates = probe.reformulated_queries(request, [REQUIREMENT_ID], set())
    assert candidates, "缺失主题必须能改写出新的查询"
    keyword_query = candidates[0]

    registry = _ScriptedRegistry(search={keyword_query: _search_result([_chunk()])})
    researcher = RagResearcher(registry, max_query_reformulations=1)

    result = await researcher.research(request, _context())

    assert result.evidence, registry.tools()
    assert result.progress.requirement_coverage[REQUIREMENT_ID] is RequirementState.ANSWERED


async def test_named_term_present_in_evidence_counts_as_support() -> None:
    """用户自己说的术语在证据原文里确实出现时，这条证据可作为依据（C01）。

    与「同义扩展线索」不同：只有原文真的含该术语才算（用户明确要求）。
    """

    text = "在 AI Native 阶段，需求到上线要闭环。"
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([_chunk(text=text)])}
    )
    request = _request(
        _task(question=REQUIREMENT_TEXT),
        decision=_decision(REQUIREMENT_TEXT, entities=["AI Native"]),
        requirement_texts={REQUIREMENT_ID: "有什么 AI Native 的使用场景吗"},
    )

    result = await RagResearcher(registry).research(request, _context())

    assert result.evidence
    assert result.evidence[0].supported_requirements == [REQUIREMENT_ID]


async def test_named_term_absent_from_evidence_does_not_count_as_support() -> None:
    text = "记忆管理包括实体链接与概念标签，只改写检索方式，不改写事实。"
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([_chunk(text=text)])}
    )
    request = _request(
        _task(question=REQUIREMENT_TEXT),
        decision=_decision(REQUIREMENT_TEXT, entities=["AI Native"]),
        requirement_texts={REQUIREMENT_ID: "有什么 AI Native 的使用场景吗"},
    )

    result = await RagResearcher(registry).research(request, _context())

    assert result.evidence
    assert result.evidence[0].supported_requirements == []


async def test_irrelevant_candidates_are_reported_as_not_relevant() -> None:
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([], status=ToolStatus.IRRELEVANT)}
    )

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.evidence == []
    assert "资料无关" in result.progress.steps[1].detail


async def test_low_similarity_candidates_are_not_evidence() -> None:
    weak = _chunk(similarity=0.1)
    registry = _ScriptedRegistry(search={REQUIREMENT_TEXT: _search_result([weak])})

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.evidence == []
    assert "低相关度候选" in result.progress.steps[1].detail
    assert result.progress.requirement_coverage[REQUIREMENT_ID] is RequirementState.PARTIAL
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert any("低相关度候选未作为依据" in item for item in result.limitations)


async def test_service_error_returns_failed_with_accurate_error() -> None:
    registry = _ScriptedRegistry(
        failures={
            str(ToolName.SEARCH_KNOWLEDGE): ToolExecutionError(
                "检索服务不可用", code=ToolErrorCode.SERVICE_ERROR
            )
        }
    )

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.SERVICE_ERROR.value
    assert result.error.resumable is True
    assert "retrieval" in result.error.node
    assert result.progress.steps[-1].status == "failed"


async def test_unconfigured_retriever_returns_failed_not_configured() -> None:
    registry = _ScriptedRegistry(
        failures={
            str(ToolName.SEARCH_KNOWLEDGE): ToolUnavailableError("未配置嵌入模型")
        }
    )

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.NOT_CONFIGURED.value
    assert result.error.resumable is True


async def test_precision_question_reads_matching_section() -> None:
    question = "报销制度的定义是什么"
    read_text = "报销制度的定义是：员工因公支出后按标准申请补偿的规定，时限见第三章。"
    registry = _ScriptedRegistry(
        search={question: _search_result([_chunk()])},
        reads={"doc-a": _read_result(read_text)},
    )
    request = _request(_task(question=question), resolved_question="")

    result = await RagResearcher(registry).research(request, _context())

    reads = registry.args_of(ToolName.READ_DOCUMENT)
    assert reads and reads[0]["document_id"] == "doc-a"
    assert reads[0]["heading"] == "第一章 报销时限"
    assert result.status is TaskStatus.COMPLETED
    assert any("#read:" in item.source_ref for item in result.evidence)


async def test_read_document_revision_is_propagated_to_chunk_evidence() -> None:
    question = "报销制度的定义是什么"
    registry = _ScriptedRegistry(
        search={question: _search_result([_chunk()])},
        reads={"doc-a": _read_result(revision="rev-9")},
    )

    result = await RagResearcher(registry).research(_request(_task(question=question)), _context())

    chunk_evidence = [item for item in result.evidence if "#read:" not in item.source_ref]
    assert chunk_evidence
    assert {item.document_revision for item in chunk_evidence} == {"rev-9"}


async def test_history_branch_uses_history_source_type() -> None:
    decision = _decision(needs_history=True)
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([_chunk()])},
        history=_result(
            ToolStatus.OK,
            {
                "messages": [
                    {
                        "conversation_id": "conv-1",
                        "message_id": "msg-9",
                        "role": "assistant",
                        "created_at": "2026-09-01T00:00:00Z",
                        "excerpt": "报销时限按 15 个工作日执行",
                    }
                ],
                "total": 1,
            },
        ),
    )

    result = await RagResearcher(registry).research(_request(decision=decision), _context())

    assert str(ToolName.SEARCH_HISTORY) in registry.tools()
    history = [
        item for item in result.evidence if item.source_type is EvidenceSourceType.HISTORY
    ]
    assert history and history[0].source_ref == "conv-1#msg-9"


# =====================================================================
# 2. 每轮进展与四条件（PRD 7.4）
# =====================================================================


async def test_progress_updates_all_prd_7_4_fields_every_round() -> None:
    weak = _chunk(similarity=0.1)
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([weak])},
        reads={"doc-a": _read_result()},
    )

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.rounds >= 2
    record_steps = [
        step for step in result.progress.steps if step.name is ResearchStepName.RECORD
    ]
    assert len(record_steps) == result.rounds
    for step in record_steps:
        assert "覆盖" in step.detail
    progress = result.progress
    assert progress.requirement_coverage
    assert progress.evidence_ids == [item.evidence_id for item in result.evidence]
    assert progress.expected_gain
    assert progress.material_fingerprint
    assert isinstance(progress.conflicts, list)
    assert isinstance(progress.open_questions, list)


def test_completion_requires_all_four_conditions() -> None:
    ready = ResearchProgress(
        task_id="t",
        requirement_coverage={REQUIREMENT_ID: RequirementState.ANSWERED},
    )
    conclusion = ResearchConclusion(statement="时限是 15 个工作日", evidence_ids=["ev-1"])

    assessment = evaluate_completion(ready, conclusions=[conclusion])

    assert assessment.can_finalize is True
    assert set(assessment.conditions) == {
        CONDITION_REQUIREMENTS,
        CONDITION_SOURCED,
        CONDITION_DIFFERENCES,
        CONDITION_AMBIGUITY,
    }


def test_completion_reports_each_missing_condition() -> None:
    uncovered = ResearchProgress(
        task_id="t",
        requirement_coverage={REQUIREMENT_ID: RequirementState.MISSING},
    )
    assert CONDITION_REQUIREMENTS in evaluate_completion(
        uncovered, conclusions=[ResearchConclusion(statement="s", evidence_ids=["ev-1"])]
    ).missing

    unsourced = ResearchProgress(
        task_id="t",
        requirement_coverage={REQUIREMENT_ID: RequirementState.ANSWERED},
    )
    assert CONDITION_SOURCED in evaluate_completion(
        unsourced, conclusions=[ResearchConclusion(statement="s", evidence_ids=[])]
    ).missing

    gapped = ResearchProgress(
        task_id="t",
        requirement_coverage={REQUIREMENT_ID: RequirementState.ANSWERED},
        conflicts=["同一单位 ms 出现不同数值"],
    )
    assert CONDITION_DIFFERENCES in evaluate_completion(
        gapped, conclusions=[ResearchConclusion(statement="s", evidence_ids=["ev-1"])]
    ).missing

    ambiguous = ResearchProgress(
        task_id="t",
        requirement_coverage={REQUIREMENT_ID: RequirementState.ANSWERED},
        open_questions=["阈值指的是哪一档"],
    )
    assessment = evaluate_completion(
        ambiguous, conclusions=[ResearchConclusion(statement="s", evidence_ids=["ev-1"])]
    )
    assert CONDITION_AMBIGUITY in assessment.missing
    assert assessment.reasons


async def test_repeated_same_material_ends_with_gaps() -> None:
    weak = _chunk(similarity=0.1)
    registry = _ScriptedRegistry(search={REQUIREMENT_TEXT: _search_result([weak])})

    result = await RagResearcher(registry, max_stalled_rounds=1).research(
        _request(), _context()
    )

    assert result.stop_reason == "same_material"
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.progress.same_material_rounds == 1
    assert result.open_questions


async def test_web_subtask_is_proposed_when_local_material_missing() -> None:
    decision = _decision(needs_web=True, freshness=Freshness.CURRENT)
    registry = _ScriptedRegistry()

    result = await RagResearcher(registry).research(_request(decision=decision), _context())

    assert result.progress.next_action is ResearchNextAction.WEB_SUBTASK
    assert result.stop_reason == "web_subtask_required"
    assert result.web_requests == [REQUIREMENT_TEXT]
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS


async def test_continue_local_reads_source_then_reports_gap() -> None:
    """A10：第一次资料不足 → 调整动作（读取原文）→ 后续动作对应明确缺口。"""

    weak = _chunk(similarity=0.1)
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([weak])},
        reads={"doc-a": _read_result(text="差旅住宿标准按城市分档执行，具体条件另行通知。")},
    )

    result = await RagResearcher(registry).research(_request(), _context())

    actions = [
        step.detail.split("：", 1)[0]
        for step in result.progress.steps
        if step.name is ResearchStepName.NEXT_ACTION
    ]
    assert actions[0] == ResearchNextAction.CONTINUE_LOCAL.value
    assert actions[-1] == ResearchNextAction.INSUFFICIENT.value
    assert registry.args_of(ToolName.READ_DOCUMENT)
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.gaps and result.gaps[0].requirement_ids == [REQUIREMENT_ID]
    assert result.open_questions == [REQUIREMENT_TEXT]


async def test_no_round_limit_by_default() -> None:
    """PRD 7.4：本版不设任务总时长与业务搜索轮数上限。"""

    assert RagResearcher(_ScriptedRegistry()).max_rounds is None


# =====================================================================
# 3. 证据对象（PRD 8.1）
# =====================================================================


def test_evidence_dedupe_merges_requirements_and_limitations() -> None:
    first = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a#block-1",
        content="同一段原文",
        supported_requirements=["R1"],
        limitations=["限制一"],
    )
    second = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a#block-1",
        content="同一段原文",
        supported_requirements=["R2"],
        limitations=["限制二"],
        document_title="CreateMemory - Amazon Bedrock AgentCore Control Plane",
        heading_path="eventExpiryDuration",
    )

    merged = dedupe_evidence([first, second])

    assert len(merged) == 1
    assert merged[0].evidence_id == first.evidence_id
    assert merged[0].supported_requirements == ["R1", "R2"]
    assert merged[0].limitations == ["限制一", "限制二"]
    assert merged[0].document_title == "CreateMemory - Amazon Bedrock AgentCore Control Plane"
    assert merged[0].heading_path == "eventExpiryDuration"


def test_evidence_identity_uses_source_and_content_hash() -> None:
    common = {
        "workspace_id": "ws-a",
        "run_id": "run-a",
        "task_id": TASK_ID,
        "source_type": EvidenceSourceType.KNOWLEDGE,
        "source_ref": "doc-a#block-1",
    }

    first = make_evidence(content="原文 A", **common)
    same = make_evidence(content="原文 A", **common)
    different = make_evidence(content="原文 B", **common)

    assert first.evidence_id == same.evidence_id
    assert first.evidence_id != different.evidence_id
    assert first.content_hash == content_hash_of("原文 A")


def test_evidence_all_twelve_fields_are_populated() -> None:
    evidence = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a#block-1",
        content=CHUNK_TEXT,
        document_revision="rev-9",
        supported_requirements=["R1"],
        limitations=["片段限制"],
    )

    assert evidence.workspace_id == "ws-a"
    assert evidence.run_id == "run-a"
    assert evidence.task_id == TASK_ID
    assert evidence.source_type is EvidenceSourceType.KNOWLEDGE
    assert evidence.source_ref == "doc-a#block-1"
    assert evidence.document_revision == "rev-9"
    assert evidence.retrieved_at.endswith("Z")
    assert evidence.supported_requirements == ["R1"]
    assert evidence.limitations == ["片段限制"]


def test_attribute_requirements_uses_programmatic_support() -> None:
    evidence = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a#block-1",
        content=CHUNK_TEXT,
    )

    attributed = attribute_requirements(
        [evidence],
        {REQUIREMENT_ID: REQUIREMENT_TEXT, "R2": "差旅住宿标准怎么规定"},
    )

    assert attributed[0].supported_requirements == [REQUIREMENT_ID]
    assert text_supports(REQUIREMENT_TEXT, CHUNK_TEXT) is True
    assert text_supports("完全无关的外星语词汇", CHUNK_TEXT) is False


async def test_open_questions_do_not_claim_unsupported_requirements() -> None:
    registry = _ScriptedRegistry(
        search={REQUIREMENT_TEXT: _search_result([_chunk(text="差旅住宿标准按城市分档执行。")])}
    )

    result = await RagResearcher(registry).research(_request(), _context())

    assert result.evidence  # 片段本身是证据（原文），但不支持该要求
    assert result.progress.requirement_coverage[REQUIREMENT_ID] is RequirementState.PARTIAL
    assert all(not item.supported_requirements for item in result.evidence)
    assert result.conclusions[0].basis is ConclusionBasis.GAP


def test_numeric_conflicts_are_detected_from_evidence() -> None:
    left = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a#1",
        content="风控阈值是 50 ms",
    )
    right = make_evidence(
        workspace_id="ws-a",
        run_id="run-a",
        task_id=TASK_ID,
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-b#1",
        content="风控阈值是 80 ms",
    )

    conflicts = detect_numeric_conflicts([left, right])

    assert numeric_claims("阈值 50 ms") == [("50", "ms")]
    assert len(conflicts) == 1
    assert "50" in conflicts[0] and "80" in conflicts[0]


# =====================================================================
# 4. Web 分支（PRD 7.3）
# =====================================================================


def _page(
    url: str = "https://news.example.com/a",
    *,
    content: str = "本地时区今天发布的 AI 新闻正文，包含事件发生时间与来源说明。",
    published_at: str = "2026-09-19",
    complete: bool = True,
    artifact_id: str = "",
) -> dict[str, Any]:
    return {
        "url": url,
        "final_url": url,
        "title": "今日 AI 新闻",
        "published_at": published_at,
        "retrieved_at": "2026-09-19T08:00:00Z",
        "content_hash": content_hash_of(content),
        "artifact_id": artifact_id,
        "read_range": {"offset": 0, "length": len(content), "total": len(content), "complete": complete},
        "content_complete": complete,
        "content": content,
    }


def test_resolve_date_range_uses_user_timezone() -> None:
    now = datetime(2026, 9, 19, 20, 0, tzinfo=UTC)

    assert resolve_date_range(Freshness.CURRENT, now=now, timezone="Asia/Shanghai") == (
        "2026-09-20",
        "2026-09-21",
    )
    assert resolve_date_range(
        Freshness.DATE_RANGE, now=now, explicit_range=("2026-01-01", "2026-01-31")
    ) == ("2026-01-01", "2026-01-31")
    assert resolve_date_range(Freshness.DATE_RANGE, now=now) is None
    assert resolve_date_range(Freshness.NONE, now=now) is None
    assert resolve_date_range(Freshness.CURRENT, now=now, timezone="Mars/Olympus") is not None


async def test_web_search_passes_news_parameters_from_date_range() -> None:
    decision = _decision("今天 AI 新闻", needs_knowledge=False, needs_web=True, freshness=Freshness.CURRENT)
    task = _task(
        question="今天 AI 新闻",
        agent_kind=AgentKind.WEB,
        expected_evidence="公开可核实来源的网页原文、发布时间与内容 hash",
    )
    registry = _ScriptedRegistry(
        web=_result(ToolStatus.OK, {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1}),
        extract=_result(ToolStatus.OK, {"pages": [_page()], "failed": [], "count": 1}),
    )

    result = await WebResearcher(registry).research(
        _request(task, decision=decision, date_range=("2026-09-19", "2026-09-20")), _context()
    )

    args = registry.args_of(ToolName.WEB_SEARCH)[0]
    assert args["topic"] == "news"
    assert args["start_date"] == "2026-09-19"
    assert args["end_date"] == "2026-09-20"
    assert args["include_published_date"] is True
    assert registry.args_of(ToolName.READ_WEB_PAGE)[0]["urls"] == ["https://news.example.com/a"]
    assert result.evidence and result.evidence[0].source_type is EvidenceSourceType.WEB
    assert result.evidence[0].content_hash == content_hash_of(_page()["content"])
    assert result.status is TaskStatus.COMPLETED
    assert _step_names(result) == [step.value for step in RESEARCH_STEPS]


async def test_web_page_artifact_and_partial_content_are_recorded() -> None:
    decision = _decision("今天 AI 新闻", needs_knowledge=False, needs_web=True, freshness=Freshness.CURRENT)
    task = _task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)
    registry = _ScriptedRegistry(
        web=_result(ToolStatus.OK, {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1}),
        extract=_result(
            ToolStatus.PARTIAL,
            {
                "pages": [_page(complete=False, artifact_id="abc123")],
                "failed": [{"url": "https://news.example.com/b", "error": "timeout"}],
                "count": 1,
                "failed_count": 1,
            },
        ),
    )

    result = await WebResearcher(registry).research(_request(task, decision=decision), _context())

    evidence = result.evidence[0]
    assert any("content_complete=false" in item for item in evidence.limitations)
    assert any("artifact abc123" in item for item in evidence.limitations)
    assert any("逐项读取失败" in step.detail for step in result.progress.steps)


async def test_web_snippet_fallback_is_marked_as_partial() -> None:
    decision = _decision("今天 AI 新闻", needs_knowledge=False, needs_web=True, freshness=Freshness.CURRENT)
    task = _task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)
    registry = _ScriptedRegistry(
        web=_result(
            ToolStatus.OK,
            {
                "results": [
                    {
                        "title": "今日 AI 新闻",
                        "url": "https://news.example.com/a",
                        "snippet": "今天 AI 新闻摘要：某某公司发布新模型",
                        "published_at": "2026-09-19",
                        "score": 0.9,
                    }
                ],
                "count": 1,
            },
        ),
        failures={
            str(ToolName.READ_WEB_PAGE): ToolExecutionError(
                "extract 不可用", code=ToolErrorCode.SERVICE_ERROR
            )
        },
    )

    result = await WebResearcher(registry).research(_request(task, decision=decision), _context())

    assert result.evidence
    assert any("未读取网页原文" in item for item in result.evidence[0].limitations)
    assert result.progress.requirement_coverage["R1"] is RequirementState.PARTIAL
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert any("搜索摘要" in item for item in result.limitations)


def test_web_page_reading_reuses_shared_contract() -> None:
    """§6 D32：网页证据字段只有一个来源，研究层不得维护第二份定义。"""

    from app.agent import contracts

    assert WebPageReading is contracts.WebPageEvidence
    assert WebPageEvidence is contracts.WebPageEvidence


async def test_web_page_read_range_is_preserved_and_not_merged_with_completeness() -> None:
    """`content_complete`（提供商部分正文）与 `read_range.complete`（本次窗口）是两个轴。"""

    page = _page(complete=False)
    page["read_range"] = {"offset": 0, "length": 20000, "total": 50000, "complete": False}
    registry = _ScriptedRegistry(
        web=_result(
            ToolStatus.OK,
            {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1},
        ),
        extract=_result(ToolStatus.PARTIAL, {"pages": [page], "failed": [], "count": 1}),
    )

    result = await WebResearcher(registry).research(
        _request(_task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)), _context()
    )

    limitations = result.evidence[0].limitations
    assert any("content_complete=false" in item for item in limitations)
    assert any("read_range.complete=false" in item for item in limitations)


async def test_web_page_window_axis_alone_does_not_claim_partial_provider_content() -> None:
    page = _page(complete=True)
    page["read_range"] = {"offset": 0, "length": 40, "total": 50000, "complete": False}
    registry = _ScriptedRegistry(
        web=_result(
            ToolStatus.OK,
            {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1},
        ),
        extract=_result(ToolStatus.PARTIAL, {"pages": [page], "failed": [], "count": 1}),
    )

    result = await WebResearcher(registry).research(
        _request(_task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)), _context()
    )

    limitations = result.evidence[0].limitations
    assert any("read_range.complete=false" in item for item in limitations)
    assert not any("content_complete=false" in item for item in limitations)


async def test_web_page_payload_drift_fails_explicitly() -> None:
    """提供商字段漂移必须显式失败，而不是静默丢字段（§6 D32）。"""

    drifted = _page()
    drifted["provider_extra_field"] = "unexpected"
    registry = _ScriptedRegistry(
        web=_result(
            ToolStatus.OK,
            {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1},
        ),
        extract=_result(ToolStatus.OK, {"pages": [drifted], "failed": [], "count": 1}),
    )

    result = await WebResearcher(registry).research(
        _request(_task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)), _context()
    )

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == "invalid_web_payload"
    assert result.error.resumable is False


def test_independent_source_report_merges_syndicated_reports() -> None:
    pages = [
        WebPageReading(url="https://news.example.com/a", final_url="https://news.example.com/a"),
        WebPageReading(url="https://news.example.com/b", final_url="https://news.example.com/b"),
        WebPageReading(url="https://other.example.org/c", final_url="https://other.example.org/c"),
    ]

    report = independent_source_report(pages)

    assert report.independent_count == 2
    assert report.merged == ["https://news.example.com/b"]


async def test_web_researcher_service_error_returns_failed() -> None:
    registry = _ScriptedRegistry(
        failures={
            str(ToolName.WEB_SEARCH): ToolExecutionError(
                "搜索服务不可用", code=ToolErrorCode.SERVICE_ERROR
            )
        }
    )
    task = _task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)

    result = await WebResearcher(registry).research(_request(task), _context())

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.SERVICE_ERROR.value


# =====================================================================
# 5. General 分支（PRD 8.1）
# =====================================================================


async def test_general_researcher_requires_a_model() -> None:
    result = await GeneralResearcher(model=None).research(
        _request(_task(question="为什么这样设计", agent_kind=AgentKind.GENERAL)), _context()
    )

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == ToolErrorCode.NOT_CONFIGURED.value


async def test_general_evidence_is_general_knowledge_without_citation() -> None:
    request = _request(_task(question="为什么这样设计比较好", agent_kind=AgentKind.GENERAL))
    researcher = GeneralResearcher(model=lambda _request: "这属于常见的分层设计取舍。")

    result = await researcher.research(request, _context())

    evidence = result.evidence[0]
    assert evidence.source_type is EvidenceSourceType.GENERAL_KNOWLEDGE
    assert citation_prefix(evidence.source_type) is None
    assert evidence.limitations == [GENERAL_EVIDENCE_LIMITATION]
    assert result.conclusions[0].evidence_ids == [evidence.evidence_id]
    assert result.status is TaskStatus.COMPLETED


async def test_general_advice_is_marked_as_proposal() -> None:
    request = _request(_task(question="这种方案怎么改进", agent_kind=AgentKind.GENERAL))
    researcher = GeneralResearcher(model=lambda _request: "建议采用分级限流，并补充独立来源验证。")

    result = await researcher.research(request, _context())

    assert result.evidence[0].source_type is EvidenceSourceType.PROPOSAL
    assert citation_prefix(result.evidence[0].source_type) is None


async def test_general_cannot_cover_internal_fact_requirement() -> None:
    question = "我们内部的报销制度为什么这样设计"
    request = _request(_task(question=question, agent_kind=AgentKind.GENERAL))
    researcher = GeneralResearcher(model=lambda _request: "通用解释：这是成本与合规的取舍。")

    result = await researcher.research(request, _context())

    assert result.progress.requirement_coverage["R1"] is RequirementState.PARTIAL
    assert result.status is TaskStatus.COMPLETED_WITH_GAPS
    assert result.gaps and "内部事实" in result.gaps[0].impact


async def test_general_empty_model_output_fails() -> None:
    request = _request(_task(question="为什么这样设计", agent_kind=AgentKind.GENERAL))

    result = await GeneralResearcher(model=lambda _request: "   ").research(request, _context())

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_code == "empty_model_output"


# =====================================================================
# 6. 调度入口
# =====================================================================


def test_researcher_for_dispatches_by_agent_kind() -> None:
    registry = _ScriptedRegistry()

    assert isinstance(researcher_for(AgentKind.RAG, registry=registry), RagResearcher)
    assert isinstance(researcher_for(AgentKind.SYNTHESIS, registry=registry), RagResearcher)
    assert isinstance(researcher_for(AgentKind.WEB, registry=registry), WebResearcher)
    assert isinstance(
        researcher_for(AgentKind.GENERAL, general_model=lambda _r: "解释"), GeneralResearcher
    )
    with pytest.raises(ResearchNotSupportedError):
        researcher_for(AgentKind.INTENT, registry=registry)


async def test_research_task_dispatches_web_tasks_without_touching_local_tools() -> None:
    """A01 结构：Web 任务只走联网工具，本地知识检索次数为 0。"""

    task = _task(question="今天 AI 新闻", agent_kind=AgentKind.WEB)
    registry = _ScriptedRegistry(
        web=_result(ToolStatus.OK, {"results": [{"title": "n", "url": "https://news.example.com/a", "snippet": "s"}], "count": 1}),
        extract=_result(ToolStatus.OK, {"pages": [_page()], "failed": [], "count": 1}),
    )

    result = await research_task(_request(task), _context(agent_kind=AgentKind.WEB), registry=registry)

    assert result.evidence
    assert registry.tools() == [str(ToolName.WEB_SEARCH), str(ToolName.READ_WEB_PAGE)]
    assert str(ToolName.SEARCH_KNOWLEDGE) not in registry.tools()


async def test_research_task_uses_injected_researcher() -> None:
    registry = _ScriptedRegistry(search={REQUIREMENT_TEXT: _search_result([_chunk()])})

    result = await research_task(
        _request(), _context(), registry=registry, researcher=RagResearcher(registry)
    )

    assert result.status is TaskStatus.COMPLETED
    assert evidence_index(result.evidence)


def test_material_gap_requires_requirement_mapping() -> None:
    gap = MaterialGap(requirement_ids=["R1"], missing="缺", impact="影响")

    assert gap.requirement_ids == ["R1"]
    assert "缺" in gap.missing


# =====================================================================
# C21：标点变体复用已有结果（规划侧与执行侧共用同一去重口径）
# =====================================================================


async def test_punctuation_variant_queries_are_claimed_once() -> None:
    """任务问题与补全问题只差中间标点时，只执行一次检索（C21.4）。"""

    registry = _ScriptedRegistry()
    question = "文档中，比如记忆，成本这些维度。都是怎么样设计的？"
    resolved = "文档中  比如记忆，成本这些维度。都是怎么样设计的？"
    request = _request(
        _task(question=question),
        resolved_question=resolved,
        requirement_texts={REQUIREMENT_ID: question},
    )

    await RagResearcher(registry, max_query_reformulations=0).research(request, _context())

    queries = [str(args.get("query")) for args in registry.args_of(ToolName.SEARCH_KNOWLEDGE)]
    assert len(queries) == 1, queries
    assert "记忆" in queries[0] and "成本" in queries[0]


async def test_research_state_claims_equivalent_punctuation_variants() -> None:
    from app.agent.research import RunResearchState

    shared = RunResearchState()
    first = await shared.claim_queries(["文档中，比如记忆，成本这些维度。都是怎么样设计的？"])
    second = await shared.claim_queries(["文档中  比如记忆，成本这些维度。都是怎么样设计的？"])

    assert first, "第一次必须真的执行"
    assert second == [], "标点变体必须复用已有结果，不得重复检索"


async def test_research_state_keeps_distinct_topic_queries() -> None:
    from app.agent.research import RunResearchState

    shared = RunResearchState()
    await shared.claim_queries(["报销时限是多少？"])
    second = await shared.claim_queries(["报销制度的时限是多少"])

    assert second == ["报销制度的时限是多少"], "换措辞的研究请求仍要执行"
