"""Synthesis / Verifier 测试（PRD 8.1 / 8.2 / 8.3）。Owner: T07。

覆盖：证据汇集与去重、结论→证据映射、引用编号分配与稳定性、PRD 8.2 八项检查各自触发、
verdict 汇总（pass / revise / research / clarify / insufficient）、模型语义支持接缝、
A11/A12/A32 场景。全程离线：程序检查为主，模型核对用注入替身。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.contracts import (
    AgentKind,
    AnswerBasis,
    CitationEntry,
    CitationSourceStatus,
    Evidence,
    EvidenceSourceType,
    RunStatus,
    SourceType,
    VerificationCheck,
    VerificationSeverity,
    VerifierVerdict,
)
from app.agent.research import (
    ConclusionBasis,
    MaterialGap,
    ResearchConclusion,
    ResearchProgress,
    ResearchResult,
    evidence_index,
    make_evidence,
)
from app.agent.verification import (
    CitationError,
    ModelVerification,
    ModelVerificationIssue,
    VerificationRequest,
    answer_basis_display,
    assign_citation_ids,
    collect_evidence,
    compute_answer_basis,
    decide_verdict,
    parse_answer_basis,
    render_candidate,
    render_gap_section,
    run_program_checks,
    synthesize_answer,
    verify_answer,
)

WORKSPACE = "ws-a"
RUN_ID = "run-a"
TASK_ID = "t-r1-knowledge"
REQUIREMENT_ID = "R1"
REQUIREMENTS = {REQUIREMENT_ID: "报销制度的时限是多少"}
LIMIT_TEXT = "报销制度的时限为 15 个工作日，需在费用发生后提交，逾期不予受理。"


# =====================================================================
# 夹具
# =====================================================================


def _evidence(
    *,
    evidence_ref: str = "doc-a#block-1",
    content: str = LIMIT_TEXT,
    source_type: EvidenceSourceType = EvidenceSourceType.KNOWLEDGE,
    document_revision: str = "rev-1",
    workspace_id: str = WORKSPACE,
    requirements: tuple[str, ...] = (REQUIREMENT_ID,),
    limitations: tuple[str, ...] = (),
) -> Evidence:
    return make_evidence(
        workspace_id=workspace_id,
        run_id=RUN_ID,
        task_id=TASK_ID,
        source_type=source_type,
        source_ref=evidence_ref,
        content=content,
        document_revision=document_revision,
        supported_requirements=requirements,
        limitations=limitations,
    )


def _conclusion(
    statement: str,
    evidence_ids: list[str],
    *,
    requirements: tuple[str, ...] = (REQUIREMENT_ID,),
    basis: ConclusionBasis = ConclusionBasis.SUPPORTED,
) -> ResearchConclusion:
    return ResearchConclusion(
        statement=statement,
        evidence_ids=evidence_ids,
        requirement_ids=list(requirements),
        basis=basis,
    )


def _result(
    *,
    conclusions: list[ResearchConclusion] | None = None,
    evidence: list[Evidence] | None = None,
    gaps: list[MaterialGap] | None = None,
    open_questions: list[str] | None = None,
    conflicts: list[str] | None = None,
    status: RunStatus = RunStatus.COMPLETED,
    task_id: str = TASK_ID,
) -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        agent_kind=AgentKind.RAG,
        status=status,
        evidence=list(evidence or []),
        conclusions=list(conclusions or []),
        gaps=list(gaps or []),
        open_questions=list(open_questions or []),
        progress=ResearchProgress(task_id=task_id, conflicts=list(conflicts or [])),
    )


def _citation(
    label: str,
    evidence: Evidence,
    *,
    source_type: SourceType = SourceType.KNOWLEDGE,
    workspace_id: str = WORKSPACE,
    snippet: str = "片段",
) -> CitationEntry:
    return CitationEntry(
        label=label,
        source_type=source_type,
        evidence_id=evidence.evidence_id,
        workspace_id=workspace_id,
        source_ref=evidence.source_ref,
        snippet=snippet,
        source_status=CitationSourceStatus.ACTIVE,
    )


def _placeholder(evidence: Evidence) -> str:
    return "{{" + evidence.evidence_id + "}}"


def _candidate(**overrides: Any) -> Any:
    from app.agent.verification import CandidateAnswer

    base: dict[str, Any] = {
        "question": "报销制度的时限是多少",
        "content": "报销时限为 15 个工作日。",
        "workspace_id": WORKSPACE,
        "run_id": RUN_ID,
        "covered_requirements": [REQUIREMENT_ID],
        "answer_basis": AnswerBasis.KNOWLEDGE,
        "basis_components": [SourceType.KNOWLEDGE],
    }
    base.update(overrides)
    return CandidateAnswer(**base)


# =====================================================================
# 1. Synthesis（PRD 8.1 / 8.2）
# =====================================================================


def test_synthesize_answer_aggregates_and_dedupes_evidence() -> None:
    first = _evidence()
    duplicate = first.model_copy()
    other = _evidence(evidence_ref="doc-b#block-1", content="差旅标准按城市分档。")

    candidate = synthesize_answer(
        "报销制度的时限是多少",
        [
            _result(evidence=[first], conclusions=[_conclusion("doc-a#1：时限 15 个工作日", [first.evidence_id])]),
            _result(evidence=[duplicate, other], task_id="t-r2-knowledge"),
        ],
        requirements=REQUIREMENTS,
        workspace_id=WORKSPACE,
        run_id=RUN_ID,
    )

    assert len(candidate.evidence) == 2
    assert {item.evidence_id for item in candidate.evidence} == {
        first.evidence_id,
        other.evidence_id,
    }


def test_synthesize_answer_keeps_conclusion_evidence_mapping() -> None:
    evidence = _evidence()
    conclusion = _conclusion("doc-a#1：时限 15 个工作日", [evidence.evidence_id])

    candidate = synthesize_answer(
        "报销制度的时限是多少",
        [_result(evidence=[evidence], conclusions=[conclusion])],
        requirements=REQUIREMENTS,
    )

    assert candidate.conclusions[0].evidence_ids == [evidence.evidence_id]
    assert candidate.conclusions[0].requirement_ids == [REQUIREMENT_ID]
    assert REQUIREMENT_ID in candidate.covered_requirements
    assert candidate.uncovered_requirements == []


def test_synthesize_answer_marks_uncovered_requirements_and_gaps() -> None:
    gap = MaterialGap(
        requirement_ids=["R2"], missing="未取得依据", impact="相关结论不能发布"
    )

    candidate = synthesize_answer(
        "两个要求",
        [_result(gaps=[gap], open_questions=["差旅标准"])],
        requirements={REQUIREMENT_ID: "报销制度的时限是多少", "R2": "差旅住宿标准怎么规定"},
    )

    assert candidate.uncovered_requirements == [REQUIREMENT_ID, "R2"]
    assert candidate.gaps == [gap]
    assert candidate.answer_basis is AnswerBasis.INSUFFICIENT
    assert candidate.basis_components == []


def test_synthesize_answer_assigns_no_citation_labels_before_finalization() -> None:
    evidence = _evidence()

    candidate = synthesize_answer(
        "报销制度的时限是多少",
        [
            _result(
                evidence=[evidence],
                conclusions=[_conclusion("doc-a#1：时限 15 个工作日", [evidence.evidence_id])],
            )
        ],
        requirements=REQUIREMENTS,
    )

    assert _placeholder(evidence) in candidate.content
    assert "[S1]" not in candidate.content
    assert candidate.citations == []


def test_render_candidate_writes_gap_lines_for_missing_requirements() -> None:
    content = render_candidate(
        [_conclusion("差旅标准：本工作区未找到可用依据", [], basis=ConclusionBasis.GAP)],
        {"R1": "差旅住宿标准怎么规定"},
    )

    assert "### 差旅住宿标准怎么规定" in content
    assert "缺少依据" in content


def test_compute_answer_basis_single_and_mixed() -> None:
    knowledge = _evidence()
    web = _evidence(
        evidence_ref="https://example.com/a", source_type=EvidenceSourceType.WEB
    )

    assert compute_answer_basis([knowledge]) == (AnswerBasis.KNOWLEDGE, [SourceType.KNOWLEDGE])
    assert compute_answer_basis([knowledge, web]) == (
        AnswerBasis.MIXED,
        [SourceType.KNOWLEDGE, SourceType.WEB],
    )


def test_compute_answer_basis_general_and_insufficient() -> None:
    general = _evidence(
        evidence_ref="general:t-1", source_type=EvidenceSourceType.GENERAL_KNOWLEDGE
    )

    assert compute_answer_basis([general]) == (AnswerBasis.GENERAL, [])
    assert compute_answer_basis([]) == (AnswerBasis.INSUFFICIENT, [])


def test_collect_evidence_dedupes_across_tasks() -> None:
    shared = _evidence()

    collected = collect_evidence(
        [
            _result(evidence=[shared]),
            _result(evidence=[shared.model_copy()], task_id="t-r2-knowledge"),
        ]
    )

    assert len(collected) == 1
    assert evidence_index(collected)[shared.evidence_id].content == LIMIT_TEXT


# =====================================================================
# 2. 引用编号（PRD 8.3）
# =====================================================================


def test_assign_citation_ids_numbers_each_prefix_from_one() -> None:
    knowledge = _evidence()
    knowledge_2 = _evidence(evidence_ref="doc-b#block-1", content="第二条原文")
    web = _evidence(evidence_ref="https://example.com/a", source_type=EvidenceSourceType.WEB)
    history = _evidence(evidence_ref="conv-1#msg-1", source_type=EvidenceSourceType.HISTORY)

    content = (
        "第一段 " + _placeholder(knowledge) + "，第二段 " + _placeholder(knowledge_2)
        + "，网页 " + _placeholder(web) + "，历史 " + _placeholder(history)
    )
    assignment = assign_citation_ids(content, [knowledge, knowledge_2, web, history])

    assert "[S1]" in assignment.content and "[S2]" in assignment.content
    assert "[W1]" in assignment.content
    assert "[H1]" in assignment.content
    assert assignment.labels[knowledge.evidence_id] == "S1"
    assert [item.label for item in assignment.citations] == ["H1", "S1", "S2", "W1"]
    assert all(item.snippet for item in assignment.citations)


def test_assign_citation_ids_reuses_same_source_without_renumbering() -> None:
    evidence = _evidence()
    content = "甲 " + _placeholder(evidence) + " 乙 " + _placeholder(evidence)

    first = assign_citation_ids(content, [evidence])
    second = assign_citation_ids(first.content, [evidence], existing=first.citations)

    assert first.content.count("[S1]") == 2
    assert "[S2]" not in first.content
    assert first.labels == second.labels == {evidence.evidence_id: "S1"}
    assert second.content == first.content
    assert first.reused_labels == ["S1"]


def test_assign_citation_ids_accepts_existing_labels() -> None:
    evidence = _evidence()
    existing = [_citation("S7", evidence)]

    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence), [evidence], existing=existing
    )

    assert assignment.content.endswith("[S7]")
    assert assignment.labels[evidence.evidence_id] == "S7"


def test_assign_citation_ids_rejects_unknown_placeholder() -> None:
    with pytest.raises(CitationError, match="没有对应证据"):
        assign_citation_ids("正文 {{ev-missing}}", [_evidence()])


def test_assign_citation_ids_rejects_non_document_source() -> None:
    general = _evidence(
        evidence_ref="general:t-1", source_type=EvidenceSourceType.GENERAL_KNOWLEDGE
    )

    with pytest.raises(CitationError, match="不得生成文档引用编号"):
        assign_citation_ids("正文 " + _placeholder(general), [general])


def test_assign_citation_ids_reports_unused_evidence() -> None:
    used = _evidence()
    unused = _evidence(evidence_ref="doc-z#block-1", content="未被引用的原文")

    assignment = assign_citation_ids(
        "正文 " + _placeholder(used), [used, unused]
    )

    assert assignment.unused_evidence_ids == [unused.evidence_id]


# =====================================================================
# 3. 程序核对：PRD 8.2 八项检查
# =====================================================================


def test_program_check_requirement_coverage() -> None:
    candidate = _candidate(covered_requirements=[])

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.REQUIREMENT_COVERAGE)
    assert issue.requirement_ids == [REQUIREMENT_ID]
    assert "报销制度的时限是多少" in issue.statement
    assert issue.required_action
    assert result.ok is False


def test_program_check_requirement_coverage_accepts_explicit_gap() -> None:
    candidate = _candidate(
        covered_requirements=[],
        uncovered_requirements=[REQUIREMENT_ID],
        gaps=[MaterialGap(requirement_ids=[REQUIREMENT_ID], missing="缺", impact="影响")],
        answer_basis=AnswerBasis.INSUFFICIENT,
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    assert not [
        item for item in result.issues if item.check is VerificationCheck.REQUIREMENT_COVERAGE
    ]


def test_program_check_factual_support_without_evidence() -> None:
    candidate = _candidate(conclusions=[_conclusion("时限为 15 个工作日", [])])

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.FACTUAL_SUPPORT)
    assert issue.statement == "时限为 15 个工作日"
    assert issue.severity is VerificationSeverity.HIGH


def test_program_check_factual_support_missing_evidence_id() -> None:
    candidate = _candidate(conclusions=[_conclusion("时限为 15 个工作日", ["ev-ghost"])])

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.FACTUAL_SUPPORT)
    assert issue.evidence_ids == ["ev-ghost"]


def test_program_check_numbers_and_units_rejects_value_not_in_source() -> None:
    evidence = _evidence(content="风控阈值是 50 ms")
    candidate = _candidate(
        evidence=[evidence],
        conclusions=[_conclusion("风控阈值是 99 ms", [evidence.evidence_id])],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.NUMBERS_AND_UNITS)
    assert "99" in issue.statement
    assert issue.evidence_ids == [evidence.evidence_id]


def test_program_check_numbers_and_units_detects_unit_mismatch() -> None:
    evidence = _evidence(content="风控阈值是 50 ms")
    candidate = _candidate(
        evidence=[evidence],
        conclusions=[_conclusion("风控阈值是 50 s", [evidence.evidence_id])],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.NUMBERS_AND_UNITS)
    assert "单位" in issue.required_action


def test_program_check_numbers_and_units_accepts_matching_claim() -> None:
    evidence = _evidence(content="风控阈值是 50 ms")
    candidate = _candidate(
        evidence=[evidence],
        conclusions=[_conclusion("风控阈值是 50 ms", [evidence.evidence_id])],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    assert not [
        item for item in result.issues if item.check is VerificationCheck.NUMBERS_AND_UNITS
    ]


def test_program_check_version_and_scope_reports_multiple_revisions() -> None:
    old = _evidence(document_revision="rev-1")
    new = _evidence(
        evidence_ref="doc-b#block-1", content="新版本的时限是 10 个工作日", document_revision="rev-2"
    )
    candidate = _candidate(
        evidence=[old, new],
        conclusions=[
            _conclusion("旧版时限 15 个工作日", [old.evidence_id]),
            _conclusion("新版时限 10 个工作日", [new.evidence_id]),
        ],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(
        item for item in result.issues if item.check is VerificationCheck.VERSION_AND_SCOPE
    )
    assert "rev-1" in issue.statement and "rev-2" in issue.statement
    assert set(issue.evidence_ids) == {old.evidence_id, new.evidence_id}


def test_program_check_version_and_scope_reports_unknown_version_token() -> None:
    evidence = _evidence()
    candidate = _candidate(
        evidence=[evidence],
        conclusions=[_conclusion("1.9.9 版本的时限是 15 个工作日", [evidence.evidence_id])],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(
        item for item in result.issues if item.check is VerificationCheck.VERSION_AND_SCOPE
    )
    assert "1.9.9" in issue.required_action


def test_program_check_source_conflict_from_numeric_claims() -> None:
    left = _evidence(content="风控阈值是 50 ms")
    right = _evidence(evidence_ref="doc-b#block-1", content="风控阈值是 80 ms")
    candidate = _candidate(evidence=[left, right])

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.SOURCE_CONFLICT)
    assert "50" in issue.statement and "80" in issue.statement
    assert "继续查询" in issue.required_action


def test_program_check_source_conflict_is_cleared_when_both_values_appear() -> None:
    left = _evidence(content="风控阈值是 50 ms")
    right = _evidence(evidence_ref="doc-b#block-1", content="风控阈值是 80 ms")
    candidate = _candidate(
        evidence=[left, right],
        content="两种来源分别给出 50 与 80 ms，需按版本区分。",
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    assert not [
        item for item in result.issues if item.check is VerificationCheck.SOURCE_CONFLICT
    ]


def test_program_check_internal_fact_requires_internal_source() -> None:
    general = _evidence(
        evidence_ref="general:t-1", source_type=EvidenceSourceType.GENERAL_KNOWLEDGE
    )
    candidate = _candidate(
        evidence=[general],
        conclusions=[_conclusion("我们内部的时限是 15 个工作日", [general.evidence_id])],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.INTERNAL_FACT)
    assert "单独表述" in issue.required_action


def test_program_check_internal_fact_flags_uncited_statement() -> None:
    candidate = _candidate(content="我们内部的报销制度规定时限为 15 个工作日。")

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.INTERNAL_FACT)
    assert "第1句" in issue.statement


def test_program_check_internal_fact_ignores_headings_and_gap_notes() -> None:
    candidate = _candidate(
        content="### 我们内部的报销制度\n\n（缺少依据：我们内部未找到可用依据）",
        covered_requirements=[],
        uncovered_requirements=[REQUIREMENT_ID],
        gaps=[MaterialGap(requirement_ids=[REQUIREMENT_ID], missing="缺", impact="影响")],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    assert not [item for item in result.issues if item.check is VerificationCheck.INTERNAL_FACT]


def test_program_check_citation_validity_flags_unknown_label() -> None:
    candidate = _candidate(content="报销时限为 15 个工作日 [S9]。")

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.CITATION_VALIDITY)
    assert "S9" in issue.statement
    assert "没有对应的来源映射" in issue.statement


def test_program_check_citation_validity_flags_prefix_mismatch() -> None:
    evidence = _evidence()
    candidate = _candidate(
        content="报销时限为 15 个工作日 [W1]。",
        evidence=[evidence],
        citations=[_citation("W1", evidence)],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.CITATION_VALIDITY)
    assert "前缀与来源类型不符" in issue.statement


def test_program_check_citation_validity_flags_cross_workspace_reference() -> None:
    evidence = _evidence(workspace_id="ws-b")
    candidate = _candidate(
        content="报销时限为 15 个工作日 [S1]。",
        evidence=[evidence],
        citations=[_citation("S1", evidence, workspace_id="ws-b")],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.CITATION_VALIDITY)
    assert "不属于当前工作区" in issue.statement


def test_program_check_citation_validity_reports_unused_citation() -> None:
    evidence = _evidence()
    candidate = _candidate(
        content="报销时限为 15 个工作日。",
        evidence=[evidence],
        citations=[_citation("S1", evidence)],
    )

    result = run_program_checks(candidate, requirements=REQUIREMENTS)

    issue = next(item for item in result.issues if item.check is VerificationCheck.CITATION_VALIDITY)
    assert "没有出现在正文中" in issue.statement


def test_program_check_material_status_after_deletion() -> None:
    """A32：生成后资料被删除时，发布前必须检测到并由核对拦下。"""

    evidence = _evidence()
    candidate = _candidate(
        content="报销时限为 15 个工作日 [S1]。",
        evidence=[evidence],
        citations=[_citation("S1", evidence)],
        conclusions=[_conclusion("doc-a#1：时限 15 个工作日", [evidence.evidence_id])],
    )

    from app.agent.verification import MaterialStatusReport

    result = run_program_checks(
        candidate,
        requirements=REQUIREMENTS,
        material_status=MaterialStatusReport(
            invalid_evidence_ids=[evidence.evidence_id],
            knowledge_revision_changed=True,
            notes=["知识在生成后被清空"],
        ),
    )

    material_issues = [
        item for item in result.issues if item.check is VerificationCheck.MATERIAL_STATUS
    ]
    assert len(material_issues) == 2
    assert material_issues[0].evidence_ids == [evidence.evidence_id]
    assert "重新核对" in material_issues[0].required_action


# =====================================================================
# 4. verdict 汇总与 verify_answer
# =====================================================================


def _clean_candidate() -> Any:
    evidence = _evidence()
    return _candidate(
        content="报销时限为 15 个工作日 [S1]。",
        evidence=[evidence],
        citations=[_citation("S1", evidence)],
        conclusions=[_conclusion("doc-a#1：报销时限为 15 个工作日", [evidence.evidence_id])],
    )


async def test_verify_answer_passes_on_clean_candidate() -> None:
    """A12：所有引用 ID 有效且关键陈述有对应原文支持 → pass。"""

    verification = await verify_answer(_clean_candidate(), requirements=REQUIREMENTS)

    assert verification.verdict is VerifierVerdict.PASS
    assert verification.issues == []
    assert verification.is_publishable() is True
    assert verification.covered_requirements == [REQUIREMENT_ID]
    assert verification.checked_at.endswith("Z")


async def test_verify_answer_revises_when_citation_is_broken() -> None:
    candidate = _clean_candidate()
    candidate.citations = []

    verification = await verify_answer(candidate, requirements=REQUIREMENTS)

    assert verification.verdict is VerifierVerdict.REVISE
    issue = next(
        item
        for item in verification.issues
        if item.check is VerificationCheck.CITATION_VALIDITY
    )
    assert issue.required_action


async def test_verify_answer_researches_on_version_conflict() -> None:
    """A11：真实文档版本冲突时不能合并成无依据结论，结论为继续研究。"""

    old = _evidence(document_revision="rev-1", content="风控阈值是 50 ms")
    new = _evidence(
        evidence_ref="doc-b#block-1", document_revision="rev-2", content="风控阈值是 80 ms"
    )
    candidate = _candidate(
        evidence=[old, new],
        conclusions=[
            _conclusion("旧版阈值 50 ms", [old.evidence_id]),
            _conclusion("新版阈值 80 ms", [new.evidence_id]),
        ],
        content="### 风控阈值\n\n两版资料的取值不一致，需要继续确认适用范围。",
    )

    verification = await verify_answer(candidate, requirements=REQUIREMENTS)

    assert verification.verdict is VerifierVerdict.RESEARCH
    checks = {issue.check for issue in verification.issues}
    assert VerificationCheck.SOURCE_CONFLICT in checks
    assert VerificationCheck.VERSION_AND_SCOPE in checks
    for issue in verification.issues:
        assert issue.required_action
    conflict_issue = next(
        item for item in verification.issues if item.check is VerificationCheck.SOURCE_CONFLICT
    )
    assert "50" in conflict_issue.statement and "80" in conflict_issue.statement


async def test_verify_answer_clarifies_blocking_question() -> None:
    candidate = _clean_candidate()
    candidate.blocking_questions = ["你说的时限是指报销还是入账？"]

    verification = await verify_answer(candidate, requirements=REQUIREMENTS)

    assert verification.verdict is VerifierVerdict.CLARIFY
    assert verification.open_questions == ["你说的时限是指报销还是入账？"]
    assert verification.is_publishable() is False


async def test_verify_answer_returns_insufficient_when_material_is_missing() -> None:
    candidate = _candidate(
        content="### 报销制度的时限\n\n（缺少依据：本工作区未找到可用依据）",
        covered_requirements=[],
        uncovered_requirements=[REQUIREMENT_ID],
        gaps=[MaterialGap(requirement_ids=[REQUIREMENT_ID], missing="无原文", impact="不能发布")],
        answer_basis=AnswerBasis.INSUFFICIENT,
        basis_components=[],
    )

    verification = await verify_answer(candidate, requirements=REQUIREMENTS)

    assert verification.verdict is VerifierVerdict.INSUFFICIENT
    assert verification.issues == []
    assert verification.is_publishable() is False


async def test_verify_answer_uses_injected_model_verifier() -> None:
    class _Verifier:
        def __init__(self) -> None:
            self.requests: list[VerificationRequest] = []

        async def review(self, request: VerificationRequest) -> ModelVerification:
            self.requests.append(request)
            return ModelVerification(
                issues=[
                    ModelVerificationIssue(
                        check=VerificationCheck.FACTUAL_SUPPORT,
                        statement="报销时限为 15 个工作日",
                        evidence_ids=[],
                        requirement_ids=[REQUIREMENT_ID],
                        required_action="补充支持该陈述的原文",
                        severity=VerificationSeverity.HIGH,
                    )
                ],
                notes=["模型认为该陈述缺少直接支持"],
            )

    verifier = _Verifier()
    candidate = _candidate(
        content="报销时限为 15 个工作日 [S1]。",
        evidence=[_evidence()],
        citations=[_citation("S1", _evidence())],
    )

    verification = await verify_answer(
        candidate, requirements=REQUIREMENTS, model_verifier=verifier
    )

    assert verifier.requests and verifier.requests[0].question == candidate.question
    assert verification.verdict is VerifierVerdict.REVISE
    assert any(
        issue.required_action == "补充支持该陈述的原文" for issue in verification.issues
    )


async def test_verify_answer_awaits_async_material_check() -> None:
    evidence = _evidence()

    async def material_check(items: list[Evidence]) -> Any:
        from app.agent.verification import MaterialStatusReport

        return MaterialStatusReport(
            invalid_evidence_ids=[item.evidence_id for item in items],
        )

    candidate = _candidate(
        content="报销时限为 15 个工作日 [S1]。",
        evidence=[evidence],
        citations=[_citation("S1", evidence)],
        conclusions=[_conclusion("doc-a#1：时限 15 个工作日", [evidence.evidence_id])],
    )

    verification = await verify_answer(
        candidate, requirements=REQUIREMENTS, material_check=material_check
    )

    assert verification.verdict is VerifierVerdict.REVISE
    assert any(
        issue.check is VerificationCheck.MATERIAL_STATUS for issue in verification.issues
    )


def test_decide_verdict_priority() -> None:
    clean = _clean_candidate()
    assert decide_verdict([], candidate=clean) is VerifierVerdict.PASS

    gapped = _candidate(uncovered_requirements=[REQUIREMENT_ID], answer_basis=AnswerBasis.INSUFFICIENT)
    assert decide_verdict([], candidate=gapped) is VerifierVerdict.INSUFFICIENT

    from app.agent.contracts import VerificationIssue

    conflict = VerificationIssue(
        check=VerificationCheck.SOURCE_CONFLICT, statement="冲突", required_action="继续查询"
    )
    coverage = VerificationIssue(
        check=VerificationCheck.REQUIREMENT_COVERAGE, statement="缺口", required_action="补充研究"
    )
    numbers = VerificationIssue(
        check=VerificationCheck.NUMBERS_AND_UNITS, statement="数值", required_action="修订"
    )

    assert decide_verdict([numbers], candidate=clean) is VerifierVerdict.REVISE
    assert (
        decide_verdict([numbers], candidate=clean, blocking_questions=["澄清"])
        is VerifierVerdict.CLARIFY
    )
    assert decide_verdict([conflict, numbers], candidate=clean) is VerifierVerdict.RESEARCH
    assert decide_verdict([coverage], candidate=clean) is VerifierVerdict.RESEARCH


# =====================================================================
# 5. answer_basis 展示兼容（PRD 8.3 末句）
# =====================================================================


def test_answer_basis_parsing_keeps_legacy_values_displayable() -> None:
    assert parse_answer_basis("knowledge") is AnswerBasis.KNOWLEDGE
    assert parse_answer_basis("mixed") is AnswerBasis.MIXED
    assert parse_answer_basis("legacy_unknown") is None
    assert answer_basis_display("knowledge") == "本工作区知识"
    assert answer_basis_display("legacy_unknown") == "legacy_unknown"


def test_render_gap_section_shows_verified_content_and_impact() -> None:
    section = render_gap_section(
        [MaterialGap(requirement_ids=[REQUIREMENT_ID], missing="无原文", impact="不能发布")],
        REQUIREMENTS,
    )

    assert "缺少的依据" in section
    assert "对结论的影响" in section
    assert "无原文" in section
