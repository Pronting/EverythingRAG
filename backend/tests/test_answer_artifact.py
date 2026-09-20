"""AnswerArtifact 发布测试（PRD 8.3）。Owner: T07。

覆盖：`pass` → `completed`、资料不足 → `completed_with_gaps`（含缺少依据与影响）、
其余 verdict 拒绝发布、引用编号稳定性与 `[S1]/[W1]/[M1]/[H1]` 前缀、`answer_basis` 九个取值
与 `basis_components`、通用解释不生成文档引用编号、运行错误 → `failed` 且保留研究记录与
继续入口，以及「Planner → 研究 → Synthesis → 编号 → Verifier → 发布」的端到端顺序。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.contracts import (
    AgentError,
    AgentKind,
    AnswerArtifact,
    AnswerBasis,
    EvidenceSourceType,
    IntentKind,
    ResearchTask,
    RouteDecision,
    RunStatus,
    SourceType,
    TaskStatus,
    VerificationCheck,
    VerificationIssue,
    VerificationResult,
    VerifierVerdict,
)
from app.agent.planner import PlannerAgent, PlanSettings
from app.agent.research import (
    ConclusionBasis,
    GeneralResearcher,
    MaterialGap,
    RagResearcher,
    ResearchConclusion,
    ResearchProgress,
    ResearchRequest,
    ResearchResult,
    make_evidence,
    research_task,
)
from app.agent.tools import ToolContext, ToolName, ToolResult, ToolStatus
from app.agent.verification import (
    AnswerVersionError,
    CandidateAnswer,
    CitationError,
    PublicationError,
    answer_version,
    assign_citation_ids,
    build_failure_outcome,
    compute_answer_basis,
    publish_answer,
    synthesize_answer,
    verify_answer,
)

WORKSPACE = "ws-a"
RUN_ID = "run-a"
TASK_ID = "t-r1-knowledge"
REQUIREMENT_ID = "R1"
QUESTION = "报销制度的时限是多少"
REQUIREMENTS = {REQUIREMENT_ID: QUESTION}
CHUNK_TEXT = "报销申请的时限为 15 个工作日，需在费用发生后提交，逾期不予受理。"


# =====================================================================
# 夹具
# =====================================================================


@dataclass(frozen=True)
class _Workspace:
    workspace_id: str = WORKSPACE
    name: str = "甲区"
    knowledge_revision: int = 3
    artifacts_dir: Any = None
    default_permission_mode: str = "ask"


@dataclass
class _Registry:
    """最小注册表替身：只回答 `search_knowledge`。"""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    async def execute(
        self, name: str | ToolName, arguments: dict[str, Any], context: ToolContext, **_kw: Any
    ) -> ToolResult:
        self.calls.append(str(name))
        if str(name) != str(ToolName.SEARCH_KNOWLEDGE):
            raise AssertionError(f"未脚本化的工具调用：{name}")
        return ToolResult(
            tool=str(name),
            status=ToolStatus.OK,
            data={"chunks": list(self.chunks), "count": len(self.chunks)},
        )


def _chunk() -> dict[str, Any]:
    return {
        "block_id": "doc-a::chunk-0",
        "document_id": "doc-a",
        "document_name": "报销制度.md",
        "topic": "内部资料",
        "text": CHUNK_TEXT,
        "similarity": 0.9,
        "match_type": "dense",
        "heading_path": "第一章 报销时限",
        "anchor": "第一章 报销时限",
        "chunk_type": "text",
        "platform": "local",
    }


def _evidence(
    *,
    evidence_ref: str = "doc-a#block-1",
    content: str = CHUNK_TEXT,
    source_type: EvidenceSourceType = EvidenceSourceType.KNOWLEDGE,
    workspace_id: str = WORKSPACE,
) -> Any:
    return make_evidence(
        workspace_id=workspace_id,
        run_id=RUN_ID,
        task_id=TASK_ID,
        source_type=source_type,
        source_ref=evidence_ref,
        content=content,
        document_revision="rev-1",
        supported_requirements=[REQUIREMENT_ID],
    )


def _conclusion(evidence_ids: list[str], statement: str = "doc-a#1：报销时限为 15 个工作日") -> ResearchConclusion:
    return ResearchConclusion(
        statement=statement,
        evidence_ids=evidence_ids,
        requirement_ids=[REQUIREMENT_ID],
        basis=ConclusionBasis.SUPPORTED,
    )


def _passed() -> VerificationResult:
    return VerificationResult(verdict=VerifierVerdict.PASS, checked_at="2026-09-19T00:00:00Z")


def _candidate(**overrides: Any) -> CandidateAnswer:
    base: dict[str, Any] = {
        "question": QUESTION,
        "content": "报销时限为 15 个工作日。",
        "workspace_id": WORKSPACE,
        "run_id": RUN_ID,
        "covered_requirements": [REQUIREMENT_ID],
        "answer_basis": AnswerBasis.KNOWLEDGE,
        "basis_components": [SourceType.KNOWLEDGE],
    }
    base.update(overrides)
    return CandidateAnswer(**base)


def _placeholder(evidence: Any) -> str:
    return "{{" + evidence.evidence_id + "}}"


def _labelled_candidate(**overrides: Any) -> CandidateAnswer:
    """已分配引用编号的候选答案（S1 + 一条知识证据）。"""

    evidence = _evidence()
    assignment = assign_citation_ids("报销时限为 15 个工作日 " + _placeholder(evidence), [evidence])
    base: dict[str, Any] = {
        "content": assignment.content,
        "evidence": [evidence],
        "citations": assignment.citations,
        "conclusions": [_conclusion([evidence.evidence_id])],
    }
    base.update(overrides)
    return _candidate(**base)


# =====================================================================
# 1. 发布规则（PRD 8.3）
# =====================================================================


def test_publish_after_pass_creates_completed_artifact() -> None:
    candidate = _labelled_candidate()

    artifact = publish_answer(candidate, _passed(), requirements=REQUIREMENTS)

    assert artifact.status is RunStatus.COMPLETED
    assert artifact.content.endswith("[S1]")
    assert artifact.run_id == RUN_ID
    assert artifact.workspace_id == WORKSPACE
    assert artifact.citations and artifact.citations[0].label == "S1"
    assert artifact.verification is not None
    assert artifact.verification.verdict is VerifierVerdict.PASS
    assert artifact.answer_basis is AnswerBasis.KNOWLEDGE
    assert artifact.basis_components == [SourceType.KNOWLEDGE]
    assert artifact.content_hash
    assert artifact.created_at.endswith("Z")


def test_publish_artifact_answer_id_is_stable_and_overridable() -> None:
    candidate = _labelled_candidate()

    first = publish_answer(candidate, _passed(), requirements=REQUIREMENTS)
    second = publish_answer(candidate, _passed(), requirements=REQUIREMENTS)
    explicit = publish_answer(
        candidate, _passed(), requirements=REQUIREMENTS, answer_id="ans-custom"
    )

    assert first.answer_id == second.answer_id
    assert first.answer_id.startswith("ans-")
    assert explicit.answer_id == "ans-custom"


def test_publish_with_gaps_uses_completed_with_gaps_and_explains() -> None:
    candidate = _candidate(
        content="### 报销制度的时限\n\n已核实：时限为 15 个工作日。",
        uncovered_requirements=[REQUIREMENT_ID],
        gaps=[
            MaterialGap(
                requirement_ids=[REQUIREMENT_ID],
                missing="缺少原文依据",
                impact="该结论不能作为已核实内容发布",
            )
        ],
        answer_basis=AnswerBasis.INSUFFICIENT,
        basis_components=[],
    )
    verification = VerificationResult(verdict=VerifierVerdict.INSUFFICIENT)

    artifact = publish_answer(candidate, verification, requirements=REQUIREMENTS)

    assert artifact.status is RunStatus.COMPLETED_WITH_GAPS
    assert "已核实：时限为 15 个工作日。" in artifact.content  # 已核实内容保留
    assert "缺少的依据" in artifact.content
    assert "对结论的影响" in artifact.content
    assert "缺少原文依据" in artifact.content
    assert artifact.answer_basis is AnswerBasis.INSUFFICIENT


def test_publish_gap_section_added_only_once() -> None:
    candidate = _candidate(
        uncovered_requirements=[REQUIREMENT_ID],
        gaps=[MaterialGap(requirement_ids=[REQUIREMENT_ID], missing="缺", impact="影响")],
        answer_basis=AnswerBasis.INSUFFICIENT,
    )
    verification = VerificationResult(verdict=VerifierVerdict.INSUFFICIENT)

    artifact = publish_answer(candidate, verification, requirements=REQUIREMENTS)

    assert artifact.content.count("资料不足说明") == 1


@pytest.mark.parametrize(
    "verdict",
    [VerifierVerdict.REVISE, VerifierVerdict.RESEARCH, VerifierVerdict.CLARIFY],
)
def test_publish_rejects_non_publishable_verdicts(verdict: VerifierVerdict) -> None:
    candidate = _labelled_candidate()
    verification = VerificationResult(
        verdict=verdict,
        issues=[
            VerificationIssue(
                check=VerificationCheck.FACTUAL_SUPPORT,
                statement="陈述",
                required_action="修订",
            )
        ],
    )

    with pytest.raises(PublicationError, match=verdict.value):
        publish_answer(candidate, verification, requirements=REQUIREMENTS)


def test_publish_rejects_insufficient_when_gaps_not_allowed() -> None:
    candidate = _candidate(answer_basis=AnswerBasis.INSUFFICIENT)

    with pytest.raises(PublicationError):
        publish_answer(
            candidate,
            VerificationResult(verdict=VerifierVerdict.INSUFFICIENT),
            requirements=REQUIREMENTS,
            allow_gaps=False,
        )


def test_answer_artifact_only_accepts_published_statuses() -> None:
    with pytest.raises(ValidationError):
        AnswerArtifact(
            answer_id="ans-1",
            run_id=RUN_ID,
            content="正文",
            status=RunStatus.RUNNING,
        )


def test_answer_version_uses_content_hash_and_status() -> None:
    artifact = publish_answer(_labelled_candidate(), _passed(), requirements=REQUIREMENTS)

    version = answer_version(artifact)

    assert version.startswith(artifact.content_hash[:12])
    assert version.endswith(RunStatus.COMPLETED.value)


def test_answer_version_changes_with_content() -> None:
    first = publish_answer(_labelled_candidate(), _passed(), requirements=REQUIREMENTS)
    second = publish_answer(
        _labelled_candidate(content="报销时限为 10 个工作日 [S1]。"),
        _passed(),
        requirements=REQUIREMENTS,
    )

    assert answer_version(first) != answer_version(second)
    assert first.content_hash != second.content_hash


def test_answer_version_requires_content_hash() -> None:
    artifact = AnswerArtifact(answer_id="ans-1", run_id=RUN_ID, content="正文")

    with pytest.raises(AnswerVersionError):
        answer_version(artifact)


# =====================================================================
# 2. answer_basis 与 basis_components（PRD 8.3）
# =====================================================================


@pytest.mark.parametrize(
    ("source_type", "basis"),
    [
        (EvidenceSourceType.KNOWLEDGE, AnswerBasis.KNOWLEDGE),
        (EvidenceSourceType.WEB, AnswerBasis.WEB),
        (EvidenceSourceType.MEMORY, AnswerBasis.MEMORY),
        (EvidenceSourceType.HISTORY, AnswerBasis.HISTORY),
        (EvidenceSourceType.CATALOG, AnswerBasis.CATALOG),
        (EvidenceSourceType.PRODUCT, AnswerBasis.PRODUCT),
    ],
)
def test_compute_answer_basis_covers_all_document_sources(
    source_type: EvidenceSourceType, basis: AnswerBasis
) -> None:
    evidence = _evidence(source_type=source_type, evidence_ref=f"{source_type.value}:#1")

    computed, components = compute_answer_basis([evidence])

    assert computed is basis
    assert components == [SourceType(source_type.value)]


def test_all_nine_answer_basis_values_round_trip() -> None:
    assert len(list(AnswerBasis)) == 9
    for basis in AnswerBasis:
        artifact = publish_answer(
            _labelled_candidate(answer_basis=basis), _passed(), requirements=REQUIREMENTS
        )
        assert artifact.answer_basis is basis


def test_publish_keeps_mixed_basis_components() -> None:
    knowledge = _evidence()
    web = _evidence(evidence_ref="https://example.com/a", source_type=EvidenceSourceType.WEB)
    basis, components = compute_answer_basis([knowledge, web])
    assignment = assign_citation_ids(
        f"知识 {_placeholder(knowledge)} 与网页 {_placeholder(web)}", [knowledge, web]
    )

    artifact = publish_answer(
        _candidate(
            content=assignment.content,
            evidence=[knowledge, web],
            citations=assignment.citations,
            conclusions=[_conclusion([knowledge.evidence_id, web.evidence_id])],
            answer_basis=basis,
            basis_components=components,
        ),
        _passed(),
        requirements=REQUIREMENTS,
    )

    assert artifact.answer_basis is AnswerBasis.MIXED
    assert artifact.basis_components == [SourceType.KNOWLEDGE, SourceType.WEB]
    assert {citation.label for citation in artifact.citations} == {"S1", "W1"}


def test_citation_prefixes_match_source_types() -> None:
    prefixes = {
        EvidenceSourceType.KNOWLEDGE: "S",
        EvidenceSourceType.WEB: "W",
        EvidenceSourceType.MEMORY: "M",
        EvidenceSourceType.HISTORY: "H",
    }
    for source_type, prefix in prefixes.items():
        evidence = _evidence(source_type=source_type, evidence_ref=f"{source_type.value}:#1")
        assignment = assign_citation_ids("正文 " + _placeholder(evidence), [evidence])
        assert assignment.citations[0].label == f"{prefix}1"
        assert assignment.citations[0].source_type is SourceType(source_type.value)


@pytest.mark.parametrize(
    "source_type", [EvidenceSourceType.CATALOG, EvidenceSourceType.PRODUCT]
)
def test_catalog_and_product_sources_are_not_numbered(
    source_type: EvidenceSourceType,
) -> None:
    evidence = _evidence(source_type=source_type, evidence_ref=f"{source_type.value}:#1")

    with pytest.raises(CitationError, match="不得生成文档引用编号"):
        assign_citation_ids("正文 " + _placeholder(evidence), [evidence])


def test_general_answer_has_no_document_citation() -> None:
    general_text = "建议采用分级限流，并补充独立来源验证。"
    request = ResearchRequest(
        task=ResearchTask(
            task_id="t-r1-general",
            question="这种方案怎么改进",
            requirement_ids=[REQUIREMENT_ID],
            agent_kind=AgentKind.GENERAL,
            expected_evidence="通用解释或设计建议",
        ),
        original_question="这种方案怎么改进",
        requirement_texts={REQUIREMENT_ID: "这种方案怎么改进"},
    )
    context = ToolContext(
        workspace=_Workspace(), run_id=RUN_ID, task_id="t-r1-general", agent_kind=AgentKind.GENERAL
    )
    result = asyncio.run(
        GeneralResearcher(model=lambda _r: general_text).research(request, context)
    )
    candidate = synthesize_answer(
        "这种方案怎么改进",
        [result],
        requirements={REQUIREMENT_ID: "这种方案怎么改进"},
        workspace_id=WORKSPACE,
        run_id=RUN_ID,
    )
    artifact = publish_answer(candidate, _passed(), requirements={REQUIREMENT_ID: "这种方案怎么改进"})

    assert artifact.citations == []
    assert "[S" not in artifact.content
    assert "[W" not in artifact.content
    assert artifact.answer_basis is AnswerBasis.GENERAL
    assert artifact.basis_components == []


# =====================================================================
# 3. 引用编号稳定性（PRD 8.3）
# =====================================================================


def test_citation_numbering_is_stable_across_publish_rounds() -> None:
    candidate = _labelled_candidate()

    first = publish_answer(candidate, _passed(), requirements=REQUIREMENTS)
    second = publish_answer(
        candidate.model_copy(update={"content": candidate.content}),
        _passed(),
        requirements=REQUIREMENTS,
    )

    assert first.content == second.content
    assert first.content_hash == second.content_hash
    assert [item.label for item in first.citations] == [
        item.label for item in second.citations
    ]


def test_reused_source_is_not_renumbered() -> None:
    evidence = _evidence()
    content = (
        "第一处 " + _placeholder(evidence) + "，第二处 " + _placeholder(evidence)
    )

    assignment = assign_citation_ids(content, [evidence])

    assert assignment.content.count("[S1]") == 2
    assert len(assignment.citations) == 1
    assert assignment.labels == {evidence.evidence_id: "S1"}


def test_citations_carry_workspace_and_snippet() -> None:
    evidence = _evidence()

    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        workspace_names={WORKSPACE: "甲区"},
    )

    citation = assignment.citations[0]
    assert citation.evidence_id == evidence.evidence_id
    assert citation.workspace_id == WORKSPACE
    assert citation.workspace_name == "甲区"
    assert citation.snippet.startswith("报销申请的时限")
    assert citation.source_ref == evidence.source_ref


def test_every_label_in_artifact_resolves_to_evidence() -> None:
    """A12：正文里的每个引用 ID 都能解析到一条证据与来源映射。"""

    knowledge = _evidence()
    web = _evidence(evidence_ref="https://example.com/a", source_type=EvidenceSourceType.WEB)
    assignment = assign_citation_ids(
        f"知识 {_placeholder(knowledge)} 与网页 {_placeholder(web)}", [knowledge, web]
    )
    candidate = _candidate(
        content=assignment.content,
        evidence=[knowledge, web],
        citations=assignment.citations,
        conclusions=[_conclusion([knowledge.evidence_id, web.evidence_id])],
    )

    artifact = publish_answer(candidate, _passed(), requirements=REQUIREMENTS)

    labels = re.findall(r"\[([SWMH]\d+)\]", artifact.content)
    by_label = {citation.label: citation for citation in artifact.citations}
    index = {item.evidence_id: item for item in candidate.evidence}
    assert labels == ["S1", "W1"]
    for label in labels:
        citation = by_label[label]
        assert citation.evidence_id in index
        assert citation.snippet
    assert {citation.evidence_id for citation in artifact.citations} == set(index)


def test_citation_invalidity_blocks_publication() -> None:
    """引用有效性不通过时不能发布（PRD 8.2 / 8.3）。"""

    candidate = _candidate(content="报销时限为 15 个工作日 [S9]。")  # 没有 S9 的来源映射
    verification = VerificationResult(
        verdict=VerifierVerdict.REVISE,
        issues=[
            VerificationIssue(
                check=VerificationCheck.CITATION_VALIDITY,
                statement="正文引用 [S9] 没有对应的来源映射",
                required_action="修订引用或答案",
            )
        ],
    )

    with pytest.raises(PublicationError):
        publish_answer(candidate, verification, requirements=REQUIREMENTS)


# =====================================================================
# 4. 运行错误（PRD 8.3 failed）
# =====================================================================


def test_failure_outcome_preserves_records_and_resume_entry() -> None:
    evidence = _evidence()
    completed = ResearchResult(
        task_id="t-r1-knowledge",
        agent_kind=AgentKind.RAG,
        status=TaskStatus.COMPLETED,
        evidence=[evidence],
        conclusions=[_conclusion([evidence.evidence_id])],
        progress=ResearchProgress(task_id="t-r1-knowledge", round=1),
    )
    failed = ResearchResult(
        task_id="t-r2-web",
        agent_kind=AgentKind.WEB,
        status=TaskStatus.FAILED,
        progress=ResearchProgress(task_id="t-r2-web", round=1),
        error=AgentError(error_code="service_error", message="搜索服务不可用", resumable=True),
    )
    error = AgentError(
        error_code="service_error", message="搜索服务不可用", node="research:t-r2-web", resumable=True
    )

    outcome = build_failure_outcome([completed, failed], error, run_id=RUN_ID)

    assert outcome.status is RunStatus.FAILED
    assert outcome.error.error_code == "service_error"
    assert outcome.completed_task_ids == ["t-r1-knowledge"]
    assert outcome.failed_task_ids == ["t-r2-web"]
    assert outcome.evidence_ids == [evidence.evidence_id]
    assert outcome.has_research_records() is True
    assert len(outcome.results) == 2
    assert RUN_ID in outcome.resume_action and "resume" in outcome.resume_action


# =====================================================================
# 5. 端到端顺序（runtime 必须调用的顺序）
# =====================================================================


def _decision() -> RouteDecision:
    return RouteDecision(
        question=QUESTION,
        resolved_question=QUESTION,
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
    )


async def test_end_to_end_order_from_plan_to_artifact() -> None:
    """runtime 顺序：Planner → 研究 → Synthesis → 引用编号 → Verifier → 发布。"""

    registry = _Registry(chunks=[_chunk()])
    planner = PlannerAgent(workspace_id=WORKSPACE, knowledge_revision=3, settings=PlanSettings())
    plan = planner.plan(QUESTION, _decision())
    assert plan.tasks and plan.coverage().uncovered == []
    task = plan.tasks[0]

    assert planner.levels(plan) == [[task.task_id]]
    assert planner.ready(plan, completed=[]) == [task]

    request = ResearchRequest(
        task=task,
        original_question=QUESTION,
        requirement_texts={REQUIREMENT_ID: QUESTION},
        decision=plan.decision,
    )
    context = ToolContext(
        workspace=_Workspace(), run_id=RUN_ID, task_id=task.task_id, agent_kind=task.agent_kind
    )
    result = await research_task(request, context, registry=registry, researcher=RagResearcher(registry))
    assert result.status is TaskStatus.COMPLETED
    assert registry.calls == [str(ToolName.SEARCH_KNOWLEDGE)]

    candidate = synthesize_answer(
        QUESTION,
        [result],
        requirements={item.requirement_id: item.text for item in plan.requirements},
        workspace_id=WORKSPACE,
        run_id=RUN_ID,
    )
    assert candidate.citations == []  # 编号在最终候选答案形成时才分配
    assignment = assign_citation_ids(candidate.content, candidate.evidence)
    candidate = candidate.model_copy(
        update={"content": assignment.content, "citations": assignment.citations}
    )
    assert "[S1]" in candidate.content

    verification = await verify_answer(
        candidate, requirements={item.requirement_id: item.text for item in plan.requirements}
    )
    assert verification.verdict is VerifierVerdict.PASS, verification.issues

    artifact = publish_answer(
        candidate,
        verification,
        requirements={item.requirement_id: item.text for item in plan.requirements},
    )
    assert artifact.status is RunStatus.COMPLETED
    assert artifact.citations[0].label == "S1"
    assert artifact.content_hash
    assert answer_version(artifact).startswith(artifact.content_hash[:12])
