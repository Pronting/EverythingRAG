"""答案组织与依据核对：Synthesis / Verifier / AnswerArtifact 发布（PRD 8）。Owner: T07。

职责：

- **Synthesis**（:func:`synthesize_answer`）：汇集全部子任务结果与证据，去重后产出
  候选答案，并保留「关键结论 → 证据 ID」的映射（PRD 8.1/8.2）。
- **引用编号**（:func:`assign_citation_ids`）：候选正文用 ``{{ev-…}}`` 占位符表达来源，
  编号在**最终候选答案形成时**分配（``[S1]/[W1]/[M1]/[H1]``，PRD 8.3）；同一来源复用
  同一编号，通用解释/建议（``general_knowledge``/``proposal``）**不能**生成编号。
- **Verifier**（:func:`verify_answer`）：按 PRD 8.2 表逐项核对八项检查，程序负责引用身份、
  字段、数字格式与要求覆盖，模型负责语义支持（通过注入的 :class:`ModelVerifier`），
  返回 ``verdict`` 与每项含「受影响要求、陈述位置、证据 ID、建议动作」的 ``issues``。
- **发布**（:func:`publish_answer`）：仅 ``pass`` → ``completed``；资料不足
  （``insufficient``）→ ``completed_with_gaps``，正文包含已核实内容、缺少的依据与影响；
  其余结论一律拒绝发布。运行错误用 :func:`build_failure_outcome` 产出 ``failed`` 并保留
  已完成研究记录与继续入口（PRD 8.3）。

契约字段来自 :mod:`app.agent.contracts`（只读）：``VerificationIssue`` /
``VerificationResult`` / ``AnswerArtifact`` / ``CitationEntry`` 直接复用，不另建同义模型。
本版 ``AnswerArtifact`` 没有独立的版本字段，因此「版本」以
:func:`answer_version`（内容 hash + 状态）表达，已在交付说明中记录为字段缺口。
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, Protocol

from pydantic import Field

from app.agent.contracts import (
    AgentError,
    AnswerArtifact,
    AnswerBasis,
    CitationEntry,
    CitationSourceStatus,
    ContractModel,
    Evidence,
    RunStatus,
    SourceType,
    TaskStatus,
    VerificationCheck,
    VerificationIssue,
    VerificationResult,
    VerificationSeverity,
    VerifierVerdict,
    citation_prefix,
    utc_now_iso,
)
from app.agent.research import (
    ConclusionBasis,
    MaterialGap,
    ResearchConclusion,
    ResearchResult,
    dedupe_evidence,
    detect_numeric_conflicts,
    evidence_index,
    numeric_claims,
)

__all__ = [
    "ANSWER_BASIS_LABELS",
    "LABEL_PATTERN",
    "PLACEHOLDER_PATTERN",
    "AnswerVersionError",
    "CandidateAnswer",
    "CitationAssignment",
    "CitationError",
    "MaterialStatusCheck",
    "MaterialStatusReport",
    "ModelVerification",
    "ModelVerificationIssue",
    "ModelVerifier",
    "ProgramCheckResult",
    "PublicationError",
    "RunFailure",
    "answer_basis_display",
    "answer_version",
    "assign_citation_ids",
    "build_failure_outcome",
    "collect_evidence",
    "compute_answer_basis",
    "decide_verdict",
    "parse_answer_basis",
    "publish_answer",
    "render_candidate",
    "render_gap_section",
    "run_program_checks",
    "synthesize_answer",
    "verify_answer",
]

#: 候选正文里的来源占位符（编号在最终候选答案形成时分配，PRD 8.3）。
PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{\{(ev-[A-Za-z0-9_-]+)\}\}")

#: 正文里的引用标注（``[S1]`` / ``[W2]`` / ``[M1]`` / ``[H1]``）。
LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[([SWMH])(\d+)\]")

_VERSION_TOKEN: Final[re.Pattern[str]] = re.compile(r"v?\d+\.\d+(?:\.\d+)?")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^\n。！？!?；;]+")

#: 判定「内部事实」的线索（PRD 8.2 内部事实：公司定义、现有实现、既定方案）。
_INTERNAL_MARKERS: Final[tuple[str, ...]] = (
    "我们",
    "内部",
    "本项目",
    "本工作区",
    "既有方案",
    "现有方案",
    "我们的实现",
)

#: 内部资料来源（只有这些来源能支撑内部事实）。
_INTERNAL_SOURCES: Final[frozenset[SourceType]] = frozenset(
    {SourceType.KNOWLEDGE, SourceType.MEMORY, SourceType.HISTORY, SourceType.CATALOG}
)

#: ``answer_basis`` 的展示名；未知的旧取值原样展示（PRD 8.3 末句）。
ANSWER_BASIS_LABELS: Final[dict[AnswerBasis, str]] = {
    AnswerBasis.KNOWLEDGE: "本工作区知识",
    AnswerBasis.WEB: "网页来源",
    AnswerBasis.GENERAL: "通用知识",
    AnswerBasis.PRODUCT: "产品说明",
    AnswerBasis.CATALOG: "知识目录",
    AnswerBasis.MEMORY: "工作区记忆",
    AnswerBasis.HISTORY: "历史讨论",
    AnswerBasis.MIXED: "多来源综合",
    AnswerBasis.INSUFFICIENT: "资料不足",
}

#: 单一来源 → ``answer_basis`` 取值。
_SINGLE_BASIS: Final[dict[SourceType, AnswerBasis]] = {
    SourceType.KNOWLEDGE: AnswerBasis.KNOWLEDGE,
    SourceType.WEB: AnswerBasis.WEB,
    SourceType.MEMORY: AnswerBasis.MEMORY,
    SourceType.HISTORY: AnswerBasis.HISTORY,
    SourceType.CATALOG: AnswerBasis.CATALOG,
    SourceType.PRODUCT: AnswerBasis.PRODUCT,
}


class CitationError(ValueError):
    """引用编号分配失败（未知占位符 / 非文档来源试图生成编号）。"""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems: list[str] = [str(problem) for problem in problems]
        super().__init__("; ".join(self.problems) or "引用编号分配失败")


class PublicationError(ValueError):
    """答案不满足发布条件（只有 ``pass`` 与资料不足的 ``insufficient`` 可以发布）。"""


class AnswerVersionError(ValueError):
    """答案版本无法确定。"""


# =====================================================================
# 1. 候选答案与核对输入
# =====================================================================


class CandidateAnswer(ContractModel):
    """Synthesis 产出的候选答案（正文 + 来源映射 + 覆盖账目 + 缺口）。"""

    question: str
    content: str = ""
    workspace_id: str = ""
    run_id: str = ""
    conclusions: list[ResearchConclusion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[CitationEntry] = Field(default_factory=list)
    covered_requirements: list[str] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[MaterialGap] = Field(default_factory=list)
    answer_basis: AnswerBasis = AnswerBasis.INSUFFICIENT
    basis_components: list[SourceType] = Field(default_factory=list)
    #: 需要用户确认才能继续的阻塞性歧义（→ ``clarify``）。
    blocking_questions: list[str] = Field(default_factory=list)

    def evidence_by_id(self) -> dict[str, Evidence]:
        return evidence_index(self.evidence)

    def has_evidence(self) -> bool:
        return any(item.content.strip() for item in self.evidence)


class CitationAssignment(ContractModel):
    """一次引用编号分配的结果（PRD 8.3）。"""

    content: str
    citations: list[CitationEntry] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    reused_labels: list[str] = Field(default_factory=list)
    unused_evidence_ids: list[str] = Field(default_factory=list)


class MaterialStatusReport(ContractModel):
    """发布前的资料状态（PRD 5.3 第 6 条 / 8.2 资料状态）。

    由运行层注入：它知道知识是否被删除、记忆是否失效、工作区是否归档。
    """

    invalid_evidence_ids: list[str] = Field(default_factory=list)
    knowledge_revision_changed: bool = False
    notes: list[str] = Field(default_factory=list)


MaterialStatusCheck = Callable[
    [Sequence[Evidence]], "MaterialStatusReport | Awaitable[MaterialStatusReport]"
]
"""资料状态检查入口（同步或异步均可）。"""


class ModelVerificationIssue(ContractModel):
    """模型语义核对发现的问题（映射进 :class:`VerificationIssue`）。"""

    check: VerificationCheck = VerificationCheck.FACTUAL_SUPPORT
    statement: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    required_action: str = ""
    severity: VerificationSeverity = VerificationSeverity.MEDIUM


class ModelVerification(ContractModel):
    """模型核对输出（语义支持 / 冲突 / 需要澄清的问题）。"""

    issues: list[ModelVerificationIssue] = Field(default_factory=list)
    ambiguity_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VerificationRequest(ContractModel):
    """交给模型核对的材料（PRD 8.2：程序检查字段，模型判断语义支持）。"""

    question: str
    content: str
    conclusions: list[ResearchConclusion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    requirements: dict[str, str] = Field(default_factory=dict)


class ModelVerifier(Protocol):
    """模型核对入口（真实 Provider 或测试替身）。"""

    async def review(self, request: VerificationRequest) -> ModelVerification: ...


class ProgramCheckResult(ContractModel):
    """程序核对结果（PRD 8.2：引用身份、字段、数字格式、要求覆盖）。"""

    ok: bool = True
    issues: list[VerificationIssue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunFailure(ContractModel):
    """运行错误结果（PRD 8.3：``failed`` + 保留已完成研究记录与继续入口）。"""

    run_id: str
    status: RunStatus = RunStatus.FAILED
    error: AgentError
    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    results: list[ResearchResult] = Field(default_factory=list)
    resume_action: str = "POST /api/runs/{run_id}/resume：从失败位置继续并复用已完成研究"
    created_at: str = Field(default_factory=utc_now_iso)

    def has_research_records(self) -> bool:
        return bool(self.results)


# =====================================================================
# 2. Synthesis（PRD 8.1 / 8.2）
# =====================================================================


def collect_evidence(results: Sequence[ResearchResult]) -> list[Evidence]:
    """汇集全部子任务证据并按来源身份 + 内容 hash 去重（PRD 8.1）。"""

    return dedupe_evidence([item for result in results for item in result.evidence])


def compute_answer_basis(evidences: Sequence[Evidence]) -> tuple[AnswerBasis, list[SourceType]]:
    """由证据来源推导 ``answer_basis`` 与 ``basis_components``（PRD 8.3）。"""

    components: list[SourceType] = []
    general_only = False
    for item in evidences:
        try:
            source = SourceType(item.source_type.value)
        except ValueError:
            general_only = True
            continue
        if source not in components:
            components.append(source)
    if not components:
        return (AnswerBasis.GENERAL if general_only else AnswerBasis.INSUFFICIENT), []
    if len(components) == 1 and not general_only:
        return _SINGLE_BASIS[components[0]], components
    return AnswerBasis.MIXED, components


def render_candidate(
    conclusions: Sequence[ResearchConclusion],
    requirement_texts: Mapping[str, str],
) -> str:
    """离线候选正文渲染：每项要求一节，已核实结论带来源占位符，缺口如实标注。

    有模型时由模型产出正文（同样使用 ``{{ev-…}}`` 占位符），本函数是**无模型回退**，
    保证 Synthesis 在离线单测与模型不可用时仍产出可核对的结构。
    """

    blocks: list[str] = []
    seen: set[str] = set()
    for conclusion in conclusions:
        for requirement_id in conclusion.requirement_ids or ["?"]:
            if requirement_id in seen:
                continue
            seen.add(requirement_id)
            title = requirement_texts.get(requirement_id, requirement_id)
            blocks.append(f"### {title}")
            if conclusion.basis is ConclusionBasis.GAP:
                blocks.append(f"（缺少依据：{conclusion.statement}）")
                continue
            marker = "".join(f"{{{{{eid}}}}}" for eid in conclusion.evidence_ids)
            blocks.append(f"{conclusion.statement}{marker}")
    return "\n\n".join(blocks)


def synthesize_answer(
    question: str,
    results: Sequence[ResearchResult],
    *,
    requirements: Mapping[str, str],
    extra_evidence: Sequence[Evidence] = (),
    workspace_id: str = "",
    run_id: str = "",
    model: Callable[[VerificationRequest], str] | None = None,
) -> CandidateAnswer:
    """汇集子任务结果，产出候选答案与「结论 → 证据」映射（PRD 8.1/8.2）。

    ``model`` 给出时由模型撰写正文（仍用占位符标注来源）；否则用
    :func:`render_candidate` 的结构化回退。**编号不在这里分配**（PRD 8.3：
    编号在最终候选答案形成时由 :func:`assign_citation_ids` 分配）。
    """

    evidence = dedupe_evidence([*collect_evidence(results), *extra_evidence])
    conclusions = [item for result in results for item in result.conclusions]
    supported_requirements = {
        requirement_id
        for conclusion in conclusions
        if conclusion.basis is not ConclusionBasis.GAP and conclusion.evidence_ids
        for requirement_id in conclusion.requirement_ids
    }
    covered = [rid for rid in requirements if rid in supported_requirements]
    uncovered = [rid for rid in requirements if rid not in supported_requirements]
    gaps = [gap for result in results for gap in result.gaps]
    open_questions = list(
        dict.fromkeys(item for result in results for item in result.open_questions)
    )
    conflicts = list(
        dict.fromkeys(item for result in results for item in result.progress.conflicts)
    )
    basis, components = compute_answer_basis(evidence)
    request = VerificationRequest(
        question=question,
        content="",
        conclusions=conclusions,
        evidence=evidence,
        requirements=dict(requirements),
    )
    if model is not None:
        content = str(model(request) or "").strip()
    else:
        content = render_candidate(conclusions, requirements)
    candidate = CandidateAnswer(
        question=question,
        content=content,
        workspace_id=workspace_id,
        run_id=run_id,
        conclusions=conclusions,
        evidence=evidence,
        covered_requirements=covered,
        uncovered_requirements=uncovered,
        open_questions=open_questions,
        conflicts=conflicts,
        gaps=gaps,
        answer_basis=basis,
        basis_components=components,
        # 注意：``open_questions`` 是「要求未被覆盖」，对应资料不足（insufficient）；
        # ``blocking_questions`` 是「需要用户确认才能继续」（clarify），由调用方显式给出，
        # 两者不能互相顶替，否则会把资料缺口误判成需要澄清。
        blocking_questions=[],
    )
    return candidate


def render_gap_section(
    gaps: Sequence[MaterialGap],
    requirement_texts: Mapping[str, str],
) -> str:
    """资料不足时的说明段落：已核实内容 / 缺少的依据 / 对结论的影响（PRD 8.3）。"""

    lines = ["### 资料不足说明"]
    if not gaps:
        lines.append("缺少的依据：未获得可核对的关键资料。")
        lines.append("对结论的影响：相关结论只作为待确认信息，不作为已核实内容。")
        return "\n".join(lines)
    for gap in gaps:
        titles = "、".join(
            requirement_texts.get(rid, rid) for rid in gap.requirement_ids
        ) or "当前问题"
        lines.append(f"- 要求「{titles}」：缺少的依据——{gap.missing or '未取得'}。")
        lines.append(f"  对结论的影响——{gap.impact or '相关陈述不能作为已核实内容发布。'}")
    return "\n".join(lines)


# =====================================================================
# 3. 引用编号（PRD 8.3）
# =====================================================================


def assign_citation_ids(
    content: str,
    evidence: Sequence[Evidence],
    *,
    existing: Sequence[CitationEntry] = (),
    workspace_names: Mapping[str, str] | None = None,
    relevance_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> CitationAssignment:
    """把 ``{{ev-…}}`` 占位符替换为稳定编号并生成来源映射（PRD 8.3）。

    规则：编号按**首次出现顺序**、按前缀分别编号（``S`` / ``W`` / ``M`` / ``H``）；
    同一来源复用同一编号（含 ``existing`` 里已分配的编号，不重新编号）；
    非文档来源（``general_knowledge`` / ``proposal``）不得生成编号，直接报错。

    ``relevance_by_source``（纯追加的可选入参，集成 t5）：``{source_ref: 相关性筛选记录}``，
    值来自**本次真实筛选结果**（``app.retrieval.rerank_pipeline`` 的选择产物，由运行层从
    检索工具的入选来源收集）。命中时把 ``relevance_checked`` / ``rerank_score`` /
    ``score_type`` / ``profile_id`` / ``evidence_hash`` / ``member_refs`` 原样写到该引用的
    追加字段上；**未命中或未提供时保持默认值**（不伪造分数/概率/正确率，PRD §8.3）。
    """

    index = evidence_index(evidence)
    relevance = relevance_by_source or {}
    citations: dict[str, CitationEntry] = {item.label: item for item in existing}
    # 已有编号先登记进 evidence → label 映射，重复调用时不会重新编号（PRD 8.3）。
    labels: dict[str, str] = {
        item.evidence_id: item.label for item in existing if item.evidence_id
    }
    problems: list[str] = []
    counters: dict[str, int] = {}
    for citation in existing:
        prefix = str(citation.label[:1])
        match = re.fullmatch(r"([SWMH])(\d+)", citation.label)
        if match:
            counters[prefix] = max(counters.get(prefix, 0), int(match.group(2)))

    def label_for(evidence_id: str) -> str:
        known = labels.get(evidence_id)
        if known is not None:
            return known
        for citation in citations.values():
            if citation.evidence_id == evidence_id:
                labels[evidence_id] = citation.label
                return citation.label
        item = index.get(evidence_id)
        if item is None:
            problems.append(f"占位符 {{{{ {evidence_id} }}}} 没有对应证据")
            return evidence_id
        prefix = citation_prefix(item.source_type)
        if prefix is None:
            problems.append(
                f"来源 {item.source_type.value} 不得生成文档引用编号（PRD 8.1）"
            )
            return evidence_id
        counters[prefix] = counters.get(prefix, 0) + 1
        label = f"{prefix}{counters[prefix]}"
        labels[evidence_id] = label
        citations[label] = CitationEntry(
            label=label,
            source_type=SourceType(item.source_type.value),
            evidence_id=item.evidence_id,
            workspace_id=item.workspace_id,
            workspace_name=(workspace_names or {}).get(item.workspace_id, ""),
            source_ref=item.source_ref,
            snippet=_first_sentence(item.content, 200),
            source_status=CitationSourceStatus.ACTIVE,
            **_relevance_citation_fields(relevance.get(str(item.source_ref))),
        )
        return label

    ordered: list[str] = []
    referenced: list[str] = []
    for match in PLACEHOLDER_PATTERN.finditer(content):
        evidence_id = match.group(1)
        referenced.append(evidence_id)
        ordered.append(label_for(evidence_id))
    if problems:
        raise CitationError(problems)
    rendered = PLACEHOLDER_PATTERN.sub(
        lambda match: f"[{label_for(match.group(1))}]", content
    )
    reused = [label for label in ordered if ordered.count(label) > 1]
    unused = [item.evidence_id for item in evidence if item.evidence_id not in set(referenced)]
    return CitationAssignment(
        content=rendered,
        citations=sorted(citations.values(), key=lambda item: item.label),
        labels=labels,
        reused_labels=list(dict.fromkeys(reused)),
        unused_evidence_ids=unused,
    )


#: 引用条目上「已筛选资料」的追加字段（集成 t5，全部带默认值，纯追加）。
_RELEVANCE_CITATION_KEYS = (
    "relevance_checked",
    "rerank_score",
    "score_type",
    "profile_id",
    "evidence_hash",
    "member_refs",
)


def _relevance_citation_fields(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """筛选记录 → 引用条目的追加字段（**只透传，不补造**）。

    * 没有记录（未启用筛选 / 该来源未被筛选选中）→ 返回空 dict ⇒ 契约默认值生效；
    * 有记录 → 只取 :data:`_RELEVANCE_CITATION_KEYS` 里**确实存在**的键，其余留默认。
      这样记录里多出的诊断字段不会被误当成契约字段，也不会因为缺键而填 0。
    """

    if not record:
        return {}
    return {key: record[key] for key in _RELEVANCE_CITATION_KEYS if key in record}


# =====================================================================
# 4. 程序核对：PRD 8.2 八项检查
# =====================================================================


def _issue(
    check: VerificationCheck,
    *,
    requirement_ids: Sequence[str] = (),
    statement: str = "",
    evidence_ids: Sequence[str] = (),
    required_action: str = "",
    severity: VerificationSeverity = VerificationSeverity.MEDIUM,
) -> VerificationIssue:
    return VerificationIssue(
        check=check,
        requirement_ids=list(requirement_ids),
        statement=statement,
        evidence_ids=list(evidence_ids),
        required_action=required_action,
        severity=severity,
    )


def _statements(content: str) -> list[tuple[int, str]]:
    """按句切分正文，返回 ``(序号, 句子)``，用于定位问题陈述。"""

    result: list[tuple[int, str]] = []
    for match in _SENTENCE_SPLIT.finditer(str(content or "")):
        text = match.group(0).strip()
        if text:
            result.append((len(result) + 1, text))
    return result


def _normalized(text: str) -> str:
    return _WHITESPACE.sub("", str(text or "")).casefold()


def _first_sentence(text: str, limit: int = 200) -> str:
    cleaned = _WHITESPACE.sub(" ", str(text or "")).strip()
    if not cleaned:
        return ""
    match = re.search(r"[。！？!?；;\n]", cleaned)
    candidate = cleaned[: match.start() + 1] if match else cleaned
    return candidate if len(candidate) <= limit else candidate[:limit] + "…"


def run_program_checks(
    candidate: CandidateAnswer,
    *,
    requirements: Mapping[str, str],
    material_status: MaterialStatusReport | None = None,
) -> ProgramCheckResult:
    """PRD 8.2 的八项检查（程序侧）：引用身份、字段、数字格式与要求覆盖。"""

    index = candidate.evidence_by_id()
    issues: list[VerificationIssue] = []
    issues.extend(_check_requirement_coverage(candidate, requirements))
    issues.extend(_check_factual_support(candidate, index))
    issues.extend(_check_numbers_and_units(candidate, index))
    issues.extend(_check_version_and_scope(candidate, index))
    issues.extend(_check_source_conflict(candidate))
    issues.extend(_check_internal_facts(candidate, index))
    issues.extend(_check_citation_validity(candidate, index))
    issues.extend(_check_material_status(candidate, index, material_status))
    notes = [
        f"程序核对 {len(issues)} 项问题",
        f"证据 {len(candidate.evidence)} 条；引用 {len(candidate.citations)} 条",
    ]
    return ProgramCheckResult(ok=not issues, issues=issues, notes=notes)


def _check_requirement_coverage(
    candidate: CandidateAnswer, requirements: Mapping[str, str]
) -> list[VerificationIssue]:
    """问题覆盖：每项用户要求对应回答内容或明确资料缺口（PRD 8.2 第 1 行）。"""

    issues: list[VerificationIssue] = []
    for requirement_id, text in requirements.items():
        covered = requirement_id in candidate.covered_requirements
        gap = any(requirement_id in item.requirement_ids for item in candidate.gaps)
        if covered or gap:
            continue
        issues.append(
            _issue(
                VerificationCheck.REQUIREMENT_COVERAGE,
                requirement_ids=[requirement_id],
                statement=f"要求「{text}」在答案中没有对应内容，也没有资料缺口说明",
                required_action="补充研究或修订答案（PRD 8.2）",
                severity=VerificationSeverity.HIGH,
            )
        )
    return issues


def _check_factual_support(
    candidate: CandidateAnswer, index: Mapping[str, Evidence]
) -> list[VerificationIssue]:
    """事实支持：关键陈述必须能由引用原文支持（PRD 8.2 第 2 行）。"""

    issues: list[VerificationIssue] = []
    for conclusion in candidate.conclusions:
        if conclusion.basis is ConclusionBasis.GAP:
            continue
        if not conclusion.evidence_ids:
            issues.append(
                _issue(
                    VerificationCheck.FACTUAL_SUPPORT,
                    requirement_ids=conclusion.requirement_ids,
                    statement=conclusion.statement,
                    required_action="查询原文、删除无依据陈述或说明不确定性（PRD 8.2）",
                    severity=VerificationSeverity.HIGH,
                )
            )
            continue
        for evidence_id in conclusion.evidence_ids:
            evidence = index.get(evidence_id)
            if evidence is None:
                issues.append(
                    _issue(
                        VerificationCheck.FACTUAL_SUPPORT,
                        requirement_ids=conclusion.requirement_ids,
                        statement=conclusion.statement,
                        evidence_ids=[evidence_id],
                        required_action="修正引用或删除无法核对的陈述（PRD 8.2）",
                        severity=VerificationSeverity.HIGH,
                    )
                )
            elif not evidence.content.strip():
                issues.append(
                    _issue(
                        VerificationCheck.FACTUAL_SUPPORT,
                        requirement_ids=conclusion.requirement_ids,
                        statement=conclusion.statement,
                        evidence_ids=[evidence_id],
                        required_action="重新读取原文后核对（PRD 8.2）",
                    )
                )
    return issues


def _check_numbers_and_units(
    candidate: CandidateAnswer, index: Mapping[str, Evidence]
) -> list[VerificationIssue]:
    """数字与单位：数值、单位、分母、时间范围、测试条件一致（PRD 8.2 第 3 行）。"""

    issues: list[VerificationIssue] = []
    for conclusion in candidate.conclusions:
        if conclusion.basis is ConclusionBasis.GAP:
            continue
        support = _normalized(
            "\n".join(
                index[evidence_id].content
                for evidence_id in conclusion.evidence_ids
                if evidence_id in index
            )
        )
        if not support:
            continue
        for value, unit in numeric_claims(conclusion.statement):
            if value not in support:
                issues.append(
                    _issue(
                        VerificationCheck.NUMBERS_AND_UNITS,
                        requirement_ids=conclusion.requirement_ids,
                        statement=conclusion.statement,
                        evidence_ids=conclusion.evidence_ids,
                        required_action=(
                            f"修订数值 {value}{unit}，或引用包含该数值的原文后重新核对（PRD 8.2）"
                        ),
                        severity=VerificationSeverity.HIGH,
                    )
                )
                continue
            if _normalized(f"{value}{unit}") not in support:
                issues.append(
                    _issue(
                        VerificationCheck.NUMBERS_AND_UNITS,
                        requirement_ids=conclusion.requirement_ids,
                        statement=conclusion.statement,
                        evidence_ids=conclusion.evidence_ids,
                        required_action=(
                            f"核对单位：陈述写 {value}{unit}，引用原文的单位不同（PRD 8.2）"
                        ),
                        severity=VerificationSeverity.HIGH,
                    )
                )
    return issues


def _check_version_and_scope(
    candidate: CandidateAnswer, index: Mapping[str, Evidence]
) -> list[VerificationIssue]:
    """版本与范围：资料版本、服务对象、适用环境对应用户问题（PRD 8.2 第 4 行）。"""

    issues: list[VerificationIssue] = []
    by_revision: dict[str, list[str]] = {}
    for conclusion in candidate.conclusions:
        if conclusion.basis is ConclusionBasis.GAP:
            continue
        for evidence_id in conclusion.evidence_ids:
            evidence = index.get(evidence_id)
            if evidence is not None and evidence.document_revision:
                by_revision.setdefault(evidence.document_revision, []).append(evidence_id)
    if len(by_revision) > 1:
        revisions = "、".join(sorted(by_revision))
        issues.append(
            _issue(
                VerificationCheck.VERSION_AND_SCOPE,
                evidence_ids=[eid for ids in by_revision.values() for eid in ids],
                statement=f"引用了多个文档版本（{revisions}），未说明适用范围",
                required_action="阅读关联资料或并列说明版本差异（PRD 8.2）",
                severity=VerificationSeverity.HIGH,
            )
        )
    for conclusion in candidate.conclusions:
        if conclusion.basis is ConclusionBasis.GAP:
            continue
        support = _normalized(
            "\n".join(
                index[evidence_id].content
                for evidence_id in conclusion.evidence_ids
                if evidence_id in index
            )
        )
        for token in _VERSION_TOKEN.findall(conclusion.statement):
            if support and _normalized(token) not in support:
                issues.append(
                    _issue(
                        VerificationCheck.VERSION_AND_SCOPE,
                        requirement_ids=conclusion.requirement_ids,
                        statement=conclusion.statement,
                        evidence_ids=conclusion.evidence_ids,
                        required_action=f"确认版本 {token} 是否属于引用资料范围（PRD 8.2）",
                    )
                )
                break
    return issues


def _check_source_conflict(candidate: CandidateAnswer) -> list[VerificationIssue]:
    """来源冲突：比较原始来源、时间和适用条件（PRD 8.2 第 5 行）。"""

    issues: list[VerificationIssue] = []
    conflicts = list(dict.fromkeys([*candidate.conflicts, *detect_numeric_conflicts(candidate.evidence)]))
    content = _normalized(candidate.content)
    for conflict in conflicts:
        values = [value for value, _unit in numeric_claims(conflict)]
        if values and all(_normalized(value) in content for value in values):
            continue
        issues.append(
            _issue(
                VerificationCheck.SOURCE_CONFLICT,
                statement=conflict,
                required_action="继续查询，保留无法确定的冲突并并列说明（PRD 8.2）",
                severity=VerificationSeverity.HIGH,
            )
        )
    return issues


def _check_internal_facts(
    candidate: CandidateAnswer, index: Mapping[str, Evidence]
) -> list[VerificationIssue]:
    """内部事实：公司定义、现有实现、既定方案必须具有本区资料依据（PRD 8.2 第 6 行）。"""

    issues: list[VerificationIssue] = []
    for conclusion in candidate.conclusions:
        if not any(marker in conclusion.statement for marker in _INTERNAL_MARKERS):
            continue
        sources = {
            index[evidence_id].source_type
            for evidence_id in conclusion.evidence_ids
            if evidence_id in index
        }
        if any(item in _INTERNAL_SOURCES for item in sources):
            continue
        issues.append(
            _issue(
                VerificationCheck.INTERNAL_FACT,
                requirement_ids=conclusion.requirement_ids,
                statement=conclusion.statement,
                evidence_ids=conclusion.evidence_ids,
                required_action="补充本区资料依据，或把公共建议单独表述（PRD 8.2）",
                severity=VerificationSeverity.HIGH,
            )
        )
    for position, sentence in _statements(candidate.content):
        if not any(marker in sentence for marker in _INTERNAL_MARKERS):
            continue
        if LABEL_PATTERN.search(sentence):
            continue
        # 标题与缺口说明是「要求原文/缺口标注」的回显，不是内部事实陈述，不能算问题。
        stripped = sentence.lstrip()
        if stripped.startswith("#") or "缺少依据" in stripped:
            continue
        issues.append(
            _issue(
                VerificationCheck.INTERNAL_FACT,
                statement=f"第{position}句：{sentence[:80]}",
                required_action="为内部事实补充本区资料引用，或改为公共建议表述（PRD 8.2）",
            )
        )
    return issues


def _check_citation_validity(
    candidate: CandidateAnswer, index: Mapping[str, Evidence]
) -> list[VerificationIssue]:
    """引用有效性：来源 ID 存在且片段支持引用位置的陈述（PRD 8.2 第 7 行）。"""

    issues: list[VerificationIssue] = []
    labels_in_content: list[str] = [
        f"{match.group(1)}{match.group(2)}" for match in LABEL_PATTERN.finditer(candidate.content)
    ]
    by_label: dict[str, CitationEntry] = {}
    for citation in candidate.citations:
        previous = by_label.get(citation.label)
        if previous is not None and previous.evidence_id != citation.evidence_id:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用编号 {citation.label} 对应了多个来源",
                    evidence_ids=[previous.evidence_id, citation.evidence_id],
                    required_action="修订引用编号或答案（PRD 8.2）",
                    severity=VerificationSeverity.HIGH,
                )
            )
        by_label[citation.label] = citation
        evidence = index.get(citation.evidence_id)
        if evidence is None:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 的来源不存在",
                    evidence_ids=[citation.evidence_id],
                    required_action="修订引用来源（PRD 8.2）",
                    severity=VerificationSeverity.HIGH,
                )
            )
            continue
        if evidence.workspace_id and candidate.workspace_id and evidence.workspace_id != candidate.workspace_id:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 的证据不属于当前工作区",
                    evidence_ids=[citation.evidence_id],
                    required_action="撤销跨工作区引用（PRD 8.1）",
                    severity=VerificationSeverity.HIGH,
                )
            )
        if citation.source_type != SourceType(evidence.source_type.value):
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 的来源类型与证据不一致",
                    evidence_ids=[citation.evidence_id],
                    required_action="修正引用身份（PRD 8.2）",
                    severity=VerificationSeverity.HIGH,
                )
            )
        expected_prefix = citation_prefix(citation.source_type)
        if expected_prefix is None:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 的来源类型不得生成文档引用编号",
                    evidence_ids=[citation.evidence_id],
                    required_action="改为 general_knowledge/proposal 表述（PRD 8.1）",
                    severity=VerificationSeverity.HIGH,
                )
            )
        elif not citation.label.startswith(expected_prefix):
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 的前缀与来源类型不符",
                    evidence_ids=[citation.evidence_id],
                    required_action="按 S/W/M/H 前缀重新编号（PRD 8.3）",
                )
            )
        if not citation.snippet.strip():
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 缺少可核对片段",
                    evidence_ids=[citation.evidence_id],
                    required_action="补充引用片段（PRD 14.4）",
                    severity=VerificationSeverity.LOW,
                )
            )
        if citation.label not in labels_in_content:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"引用 {citation.label} 没有出现在正文中",
                    evidence_ids=[citation.evidence_id],
                    required_action="删除未使用的引用映射",
                    severity=VerificationSeverity.LOW,
                )
            )
    for label in labels_in_content:
        if label not in by_label:
            issues.append(
                _issue(
                    VerificationCheck.CITATION_VALIDITY,
                    statement=f"正文引用 [{label}] 没有对应的来源映射",
                    required_action="修订引用或答案（PRD 8.2）",
                    severity=VerificationSeverity.HIGH,
                )
            )
    return issues


def _check_material_status(
    candidate: CandidateAnswer,
    index: Mapping[str, Evidence],
    material_status: MaterialStatusReport | None,
) -> list[VerificationIssue]:
    """资料状态：发布前知识与记忆仍可使用（PRD 8.2 第 8 行 / A32）。"""

    if material_status is None:
        return []
    issues: list[VerificationIssue] = []
    for evidence_id in material_status.invalid_evidence_ids:
        statement = next(
            (
                conclusion.statement
                for conclusion in candidate.conclusions
                if evidence_id in conclusion.evidence_ids
            ),
            f"证据 {evidence_id} 已失效",
        )
        issues.append(
            _issue(
                VerificationCheck.MATERIAL_STATUS,
                statement=statement,
                evidence_ids=[evidence_id],
                required_action="撤销失效证据并重新核对受影响陈述（PRD 8.2 / A32）",
                severity=VerificationSeverity.HIGH,
            )
        )
    if material_status.knowledge_revision_changed:
        issues.append(
            _issue(
                VerificationCheck.MATERIAL_STATUS,
                statement="生成答案后工作区知识版本发生变化",
                evidence_ids=[item.evidence_id for item in index.values()],
                required_action="重新核对受影响陈述（PRD 8.2 / 5.3）",
                severity=VerificationSeverity.HIGH,
            )
        )
    return issues


# =====================================================================
# 5. Verifier（PRD 8.2）
# =====================================================================


def decide_verdict(
    issues: Sequence[VerificationIssue],
    *,
    candidate: CandidateAnswer,
    blocking_questions: Sequence[str] = (),
) -> VerifierVerdict:
    """按问题集合汇总 ``verdict``（``pass`` 是唯一允许无缺口发布的结论）。"""

    if not issues:
        has_gaps = bool(candidate.gaps or candidate.uncovered_requirements)
        if has_gaps or candidate.answer_basis is AnswerBasis.INSUFFICIENT:
            return VerifierVerdict.INSUFFICIENT
        if blocking_questions:
            # 没有问题但仍有需要用户确认的关键歧义：不能作为 pass 发布（PRD 8.2）。
            return VerifierVerdict.CLARIFY
        return VerifierVerdict.PASS
    checks = {issue.check for issue in issues}
    needs_research = bool(
        checks & {VerificationCheck.SOURCE_CONFLICT, VerificationCheck.REQUIREMENT_COVERAGE}
    )
    if needs_research:
        return VerifierVerdict.RESEARCH
    if blocking_questions:
        return VerifierVerdict.CLARIFY
    return VerifierVerdict.REVISE


async def verify_answer(
    candidate: CandidateAnswer,
    *,
    requirements: Mapping[str, str],
    material_check: MaterialStatusCheck | None = None,
    model_verifier: ModelVerifier | None = None,
) -> VerificationResult:
    """核对候选答案（PRD 8.2）：程序检查字段与结构，模型判断语义支持。"""

    material_status: MaterialStatusReport | None = None
    if material_check is not None:
        produced = material_check(candidate.evidence)
        if inspect.isawaitable(produced):
            produced = await produced
        material_status = produced
    program = run_program_checks(
        candidate, requirements=requirements, material_status=material_status
    )
    issues = list(program.issues)
    blocking = list(candidate.blocking_questions)
    if model_verifier is not None:
        review = await model_verifier.review(
            VerificationRequest(
                question=candidate.question,
                content=candidate.content,
                conclusions=candidate.conclusions,
                evidence=candidate.evidence,
                requirements=dict(requirements),
            )
        )
        for item in review.issues:
            issues.append(
                VerificationIssue(
                    check=item.check,
                    requirement_ids=list(item.requirement_ids),
                    statement=item.statement,
                    evidence_ids=list(item.evidence_ids),
                    required_action=item.required_action,
                    severity=item.severity,
                )
            )
        blocking.extend(review.ambiguity_questions)
    verdict = decide_verdict(issues, candidate=candidate, blocking_questions=blocking)
    return VerificationResult(
        verdict=verdict,
        issues=issues,
        covered_requirements=list(candidate.covered_requirements),
        uncovered_requirements=list(candidate.uncovered_requirements),
        # 待确认问题 = 候选答案的开放问题 + 需要用户澄清的阻塞问题（verdict=clarify 时
        # 运行层据此发起澄清卡片；契约没有独立的 blocking 字段，见模块 docstring）。
        open_questions=list(dict.fromkeys([*candidate.open_questions, *blocking])),
        conflicts=list(candidate.conflicts),
        checked_at=utc_now_iso(),
    )


# =====================================================================
# 6. 发布（PRD 8.3）
# =====================================================================


def answer_version(artifact: AnswerArtifact) -> str:
    """答案版本：契约没有独立版本字段，用「内容 hash + 状态」表达（见模块 docstring）。"""

    if not artifact.content_hash:
        raise AnswerVersionError("AnswerArtifact 缺少 content_hash，无法确定版本")
    return f"{artifact.content_hash[:12]}@{artifact.status.value}"


def publish_answer(
    candidate: CandidateAnswer,
    verification: VerificationResult,
    *,
    requirements: Mapping[str, str],
    answer_id: str | None = None,
    allow_gaps: bool = True,
) -> AnswerArtifact:
    """按 PRD 8.3 发布答案：``pass`` → ``completed``；资料不足 → ``completed_with_gaps``。"""

    if verification.verdict is VerifierVerdict.PASS:
        status = RunStatus.COMPLETED
    elif verification.verdict is VerifierVerdict.INSUFFICIENT and allow_gaps:
        status = RunStatus.COMPLETED_WITH_GAPS
    else:
        raise PublicationError(
            f"verdict={verification.verdict.value} 不允许发布；"
            f"需处理 {len(verification.issues)} 项核对问题（PRD 8.3）"
        )
    content = candidate.content
    if status is RunStatus.COMPLETED_WITH_GAPS:
        gap_section = render_gap_section(candidate.gaps, requirements)
        if gap_section not in content:
            content = f"{content}\n\n{gap_section}".strip()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    active_id = answer_id or "ans-" + hashlib.sha256(
        f"{candidate.run_id}|{digest}".encode()
    ).hexdigest()[:24]
    return AnswerArtifact(
        answer_id=active_id,
        run_id=candidate.run_id,
        workspace_id=candidate.workspace_id,
        content=content,
        citations=list(candidate.citations),
        verification=verification,
        content_hash=digest,
        answer_basis=candidate.answer_basis,
        basis_components=list(candidate.basis_components),
        status=status,
        created_at=utc_now_iso(),
    )


def build_failure_outcome(
    results: Sequence[ResearchResult],
    error: AgentError,
    *,
    run_id: str,
) -> RunFailure:
    """运行错误 → ``failed``，保留已完成研究记录与继续入口（PRD 8.3）。"""

    return RunFailure(
        run_id=run_id,
        status=RunStatus.FAILED,
        error=error,
        completed_task_ids=[
            result.task_id
            for result in results
            if result.status in {TaskStatus.COMPLETED, TaskStatus.COMPLETED_WITH_GAPS}
        ],
        failed_task_ids=[
            result.task_id for result in results if result.status is TaskStatus.FAILED
        ],
        evidence_ids=[item.evidence_id for result in results for item in result.evidence],
        results=list(results),
        resume_action=f"POST /api/runs/{run_id}/resume：从失败位置继续并复用已完成研究",
    )


# =====================================================================
# 7. answer_basis 展示兼容（PRD 8.3 末句）
# =====================================================================


def parse_answer_basis(raw: str) -> AnswerBasis | None:
    """解析 ``answer_basis``；历史消息中的旧取值不认识时返回 ``None``（原样展示）。"""

    try:
        return AnswerBasis(str(raw))
    except ValueError:
        return None


def answer_basis_display(raw: str) -> str:
    """展示用文案：新取值给中文名，未知旧取值原样返回。"""

    basis = parse_answer_basis(raw)
    if basis is None:
        return str(raw)
    return ANSWER_BASIS_LABELS.get(basis, str(raw))
