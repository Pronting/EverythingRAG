from __future__ import annotations

import ast
import hashlib
import math
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
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
    citation_prefix,
    utc_now_iso,
)
from app.agent.evidence_text import evidence_location_text, split_evidence_content
from app.agent.research import (
    ConclusionBasis,
    MaterialGap,
    ResearchConclusion,
    ResearchResult,
    dedupe_evidence,
    evidence_index,
    evidence_ready_for_answer,
)

PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{\{(ev-[A-Za-z0-9_-]+)\}\}")
LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[([SWMH])(\d+)\]")
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
_SINGLE_BASIS = {
    SourceType.KNOWLEDGE: AnswerBasis.KNOWLEDGE,
    SourceType.WEB: AnswerBasis.WEB,
    SourceType.MEMORY: AnswerBasis.MEMORY,
    SourceType.HISTORY: AnswerBasis.HISTORY,
    SourceType.CATALOG: AnswerBasis.CATALOG,
    SourceType.PRODUCT: AnswerBasis.PRODUCT,
}


class CitationError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = [str(problem) for problem in problems]
        super().__init__("; ".join(self.problems) or "引用编号分配失败")


class AnswerVersionError(ValueError):
    pass


class SynthesisUnavailable(RuntimeError):
    """候选答案无法由模型产出：run 必须明确失败（不再回落到确定性拼接）。"""

    code = "synthesis_model_unavailable"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


class CandidateAnswer(ContractModel):
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

    def evidence_by_id(self) -> dict[str, Evidence]:
        return evidence_index(self.evidence)

    def has_evidence(self) -> bool:
        return any(item.content.strip() for item in self.evidence)


class CitationAssignment(ContractModel):
    content: str
    citations: list[CitationEntry] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    reused_labels: list[str] = Field(default_factory=list)
    unused_evidence_ids: list[str] = Field(default_factory=list)


class CitationGrounding(ContractModel):
    marker: str
    claim: str
    evidence_id: str
    quote: str
    calculation: str = ""
    inputs: dict[str, float] = Field(default_factory=dict)
    result: float | None = None


class CitationSpan(ContractModel):
    evidence_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class GroundedParagraph(ContractModel):
    text: str = Field(description="一个自然段或一个列表项；各自然段分别建立正文项并就地绑定引用")
    citations: list[CitationSpan] = Field(default_factory=list)
    calculation: str = Field(default="", description="仅数值演算填写使用inputs变量的Python算式；普通事实与文字推断保持空字符串")
    inputs: dict[str, float] = Field(default_factory=dict)
    result: float | None = None


class SourceScope(ContractModel):
    subject: str
    requested_operation: str
    documented_operation: str
    matches_request: bool
    evidence_ids: list[str]
    requested_dimensions: list[str] = Field(default_factory=list, description="用户明确指定的比较维度，未指定时为空")
    provided_dimensions: list[str] = Field(default_factory=list, description="原文直接提供所问操作具体内容的比较维度；仅提及职责或相关操作不算提供该维度")


class SourceScopeReview(ContractModel):
    source_scope: list[SourceScope] = Field(min_length=1)


class SynthesisReview(ContractModel):
    source_scope: list[SourceScope] = Field(default_factory=list)
    content: str = ""
    covered_requirements: list[str]
    gaps: list[MaterialGap]
    citations: list[CitationGrounding] = Field(default_factory=list)
    paragraphs: list[GroundedParagraph] = Field(default_factory=list)


class ParagraphSynthesisReview(ContractModel):
    paragraphs: list[GroundedParagraph]
    covered_requirements: list[str]
    gaps: list[MaterialGap]


class GeneralAnswerAssessment(ContractModel):
    issues: list[str] = Field(description="候选回答中确定的事实、成立条件或示例错误；无错误时为空")
    corrected_answer: str = Field(default="", description="存在错误时给出修正后的完整回答，保持用户语言、字数、格式及角色要求；无错误时为空")


class SemanticIssue(ContractModel):
    paragraph_index: int = Field(ge=0)
    reason: str = Field(min_length=1)


class ScopeAssessment(ContractModel):
    requested_scope: str = Field(min_length=1)
    answer_scope: str = Field(min_length=1)
    narrower_than_requested: bool
    paragraph_indices: list[int]
    reason: str = Field(min_length=1)
    substitutes_operation: bool = Field(default=False, description="是否用同一对象的其他操作或阶段的规则回答用户指定操作")
    broader_than_requested: bool = Field(default=False, description="是否展开用户未要求的相邻流程或实现细节，并将其未提供的内容登记为缺口")


class SemanticAssessment(ContractModel):
    scope_check: ScopeAssessment
    issues: list[SemanticIssue]
    covered_requirements: list[str]
    gaps: list[MaterialGap]


def apply_semantic_assessment(
    review: SynthesisReview, assessment: SemanticAssessment, requirement_ids: set[str],
) -> dict[int, str]:
    covered = set(assessment.covered_requirements)
    missing = {rid for gap in assessment.gaps for rid in gap.requirement_ids}
    if covered | missing != requirement_ids or any(not gap.requirement_ids for gap in assessment.gaps):
        raise SynthesisUnavailable("语义验收未完整对应用户要求")
    issues: dict[int, str] = {}
    scope = assessment.scope_check
    scope_issue = scope.narrower_than_requested or scope.substitutes_operation or scope.broader_than_requested
    if scope_issue and not scope.paragraph_indices:
        raise SynthesisUnavailable("语义验收指出范围问题但未指定对应正文")
    for position in scope.paragraph_indices:
        if not 0 <= position < len(review.paragraphs):
            raise SynthesisUnavailable("语义验收引用了不存在的正文项")
        if scope_issue:
            issues[position] = scope.reason
    for item in assessment.issues:
        if item.paragraph_index >= len(review.paragraphs):
            raise SynthesisUnavailable("语义验收引用了不存在的正文项")
        issues[item.paragraph_index] = "; ".join(filter(None, [issues.get(item.paragraph_index), item.reason]))
    review.covered_requirements = [rid for rid in assessment.covered_requirements if rid not in missing]
    review.gaps = list(assessment.gaps)
    return issues


class ParagraphBindingError(SynthesisUnavailable):
    def __init__(self, issues: Mapping[int, str]) -> None:
        self.issues = dict(issues)
        super().__init__("；".join(f"正文项 {index}: {issue}" for index, issue in issues.items()))


class ParagraphCorrection(ContractModel):
    index: int = Field(ge=0)
    paragraph: GroundedParagraph
    gaps: list[MaterialGap] = Field(default_factory=list)


class ParagraphCorrections(ContractModel):
    corrections: list[ParagraphCorrection]


_GAP_DISCLOSURE_MARKERS: Final[tuple[str, ...]] = (
    "未取得", "未找到", "未提供", "未说明", "未覆盖", "未确认", "没有取得",
    "没有找到", "没有提供", "没有说明", "未写出", "没有写出", "未出现", "没有出现",
    "缺少", "不足", "无法确认", "不能确认", "未给出", "没有给出",
    "cannot confirm", "not found", "not provided", "missing", "insufficient",
)
_HYPOTHETICAL_MARKERS: Final[tuple[str, ...]] = (
    "假设示例", "假定示例", "举例说明", "以下是假设", "示例中假设", "hypothetical example",
)


def _cjk_bigrams(text: str) -> set[str]:
    values: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        values.update(run[index:index + 2] for index in range(len(run) - 1))
    return values


def _ascii_identity_terms(text: str) -> set[str]:
    return {
        item.casefold() for item in re.findall(r"[A-Za-z][A-Za-z0-9_.+/#-]{1,}", text)
        if item.casefold() not in {"the", "and", "for", "with", "from"}
    }


def _gap_is_explicit(paragraph: str, missing: str) -> bool:
    """缺口允许同义表达，同时要求正文明确表达资料不足。"""

    folded = paragraph.casefold()
    if not any(marker in folded for marker in _GAP_DISCLOSURE_MARKERS):
        return False
    compact_paragraph = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", folded)
    compact_missing = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", missing.casefold())
    if compact_missing and compact_missing in compact_paragraph:
        return True
    missing_ascii = _ascii_identity_terms(missing)
    paragraph_ascii = _ascii_identity_terms(paragraph)
    shared_ascii = missing_ascii & paragraph_ascii
    missing_cjk = _cjk_bigrams(missing)
    shared_cjk = missing_cjk & _cjk_bigrams(paragraph)
    if missing_ascii and len(shared_ascii) / len(missing_ascii) >= 0.5:
        return True
    return bool(missing_cjk) and len(shared_cjk) >= 2 and (
        len(shared_cjk) / len(missing_cjk) >= 0.25
    )


def _is_explicit_hypothetical(text: str) -> bool:
    folded = text.casefold()
    return any(marker in folded for marker in _HYPOTHETICAL_MARKERS)


def apply_paragraph_corrections(
    review: SynthesisReview, corrections: ParagraphCorrections,
    positions: set[int], requirement_ids: set[str],
) -> SynthesisReview:
    supplied = [item.index for item in corrections.corrections]
    if len(supplied) != len(set(supplied)) or set(supplied) != positions:
        raise SynthesisUnavailable("修正内容未准确对应需要核验的正文项")
    result = review.model_copy(deep=True)
    for item in corrections.corrections:
        previous = review.paragraphs[item.index]
        if (
            previous.citations
            and not item.paragraph.citations
            and not item.gaps
            and not _is_explicit_hypothetical(item.paragraph.text)
        ):
            raise SynthesisUnavailable("带来源的正文修正后缺少来源及明确缺口")
        for gap in item.gaps:
            if not gap.requirement_ids or not set(gap.requirement_ids) <= requirement_ids:
                raise SynthesisUnavailable(
                    f"正文项{item.index}的gaps.requirement_ids={gap.requirement_ids!r}无效；"
                    f"允许的ID为{sorted(requirement_ids)!r}。仅登记用户明确要求且未能回答的内容；"
                    "未要求的补充细节不登记gaps。"
                )
            if not gap.missing.strip() or not _gap_is_explicit(item.paragraph.text, gap.missing):
                raise SynthesisUnavailable(f"正文项{item.index}未明确说明所登记的缺口：{gap.missing}；请在该项正文直接说明这一内容未获确认")
        result.paragraphs[item.index] = item.paragraph
        result.gaps.extend(item.gaps)
    return result


def citation_source_lines(item: Evidence) -> list[str]:
    body = evidence_location_text(body=split_evidence_content(item.content)[1])
    return [
        part for line in body.splitlines() if line.strip()
        for part in textwrap.wrap(line, width=320, replace_whitespace=False,
                                  drop_whitespace=False, break_long_words=True)
    ]


def _without_supported_source_metadata(text: str, sources: Sequence[Evidence]) -> str:
    """仅排除已选来源的完整标题、URL 与版本文字，业务数字仍须检查原文。"""
    titles = {item.document_title for item in sources if item.document_title}
    headings = {item.heading_path for item in sources if item.heading_path}
    urls = {item.source_ref for item in sources
            if urlsplit(item.source_ref).scheme in {"http", "https"}}
    for match in re.finditer(r"https?://[^\s<>\"'`）)，；。]+", text):
        address = match.group(0).rstrip(").,;")
        if address not in urls:
            raise SynthesisUnavailable(f"正文URL未对应所选来源的实际地址：{address}")
        text = text.replace(address, "来源地址")
    for left, right in (("《", "》"), ("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'), ("`", "`")):
        for mention in re.findall(
            re.escape(left) + r"([^\n" + re.escape(left + right) + r"]{2,240})" + re.escape(right),
            text,
        ):
            if (len(mention) >= 8 and any(mention in title for title in titles)
                    and (len(re.findall(r"[A-Za-z]{2,}", mention)) >= 2
                         or len(re.findall(r"[\u4e00-\u9fff]", mention)) >= 4)):
                text = text.replace(left + mention + right, "来源标题")
            if re.match(r"\d+(?:\.\d+)+\s+\S", mention) and any(
                re.search(rf"(?m)^{re.escape(mention)}\s*$", split_evidence_content(item.content)[1])
                for item in sources
            ):
                text = text.replace(left + mention + right, "来源章节")
            filenames = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:mdx?|pdf|docx?|txt|html?)\b", mention, re.IGNORECASE)
            if not filenames and re.search(r"\.(?:mdx?|pdf|docx?|txt|html?)$", mention, re.IGNORECASE):
                filenames = [mention]
            for filename in filenames:
                if filename not in titles and not any(filename in item.content for item in sources):
                    raise SynthesisUnavailable(f"正文文件名未对应所选来源标题或原文：{filename}")
    for title in sorted(titles, key=len, reverse=True):
        if re.search(r"[A-Za-z\u4e00-\u9fff]", title):
            text = text.replace(title, "来源标题")
    for heading in sorted(headings, key=len, reverse=True):
        if re.search(r"[A-Za-z\u4e00-\u9fff]", heading):
            text = text.replace(heading, "来源章节")
    versions: set[str] = set()
    for address in urls:
        parsed = urlsplit(address)
        for part in [*(parsed.hostname or "").split("."), *parsed.path.split("/")]:
            if re.fullmatch(r"v\d+(?:[.-]\d+)+", part):
                versions.add(part.replace("-", "."))
    for item in sources:
        versions.update(re.findall(r"\bv\d+(?:\.\d+)+\b", f"{item.document_title} {item.document_revision}"))
    for match in re.finditer(r"\bv\d+(?:\.\d+)+\b", text):
        if match.group(0) in versions:
            text = text.replace(match.group(0), "来源版本")
    for match in list(re.finditer(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", text)):
        number = match.group(0)
        window = text[max(0, match.start() - 48):min(len(text), match.end() + 48)]
        for identity in [*titles, *headings]:
            if number not in re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", identity):
                continue
            shared_ascii = _ascii_identity_terms(window) & _ascii_identity_terms(identity)
            shared_cjk = _cjk_bigrams(window) & _cjk_bigrams(identity)
            if len(shared_cjk) >= 4 or (shared_ascii and len(shared_cjk) >= 2):
                text = text[:match.start()] + (" " * len(number)) + text[match.end():]
                break
    return text


def _check_quoted_terms(text: str, quote: str, sources: Sequence[Evidence]) -> None:
    """原文中的明确引语应由选取的行支持，反馈可见原文的位置。"""
    headings = {item.heading_path for item in sources} | {item.document_title for item in sources}
    for item in sources:
        tokens = MarkdownIt().parse(split_evidence_content(item.content)[1])
        for index, token in enumerate(tokens[1:], 1):
            if token.type == "inline" and tokens[index - 1].type == "heading_open":
                headings.add("".join(child.content for child in token.children or []
                                     if child.type in {"text", "code_inline"}))
    terms = set()
    for left, right in (("「", "」"), ("『", "』"), ("“", "”"), ('"', '"')):
        for match in re.finditer(
            re.escape(left) + r"([^\n" + re.escape(left + right) + r"]{2,120})" + re.escape(right),
            text,
        ):
            if text[match.end():].lstrip().startswith(("一节", "章节", "小节", "一章")):
                continue
            terms.add(match.group(1))
    for term in sorted(terms):
        if term in quote or any(term in heading for heading in headings):
            continue
        locations = sorted({
            f"{item.evidence_id}:L{number}"
            for item in sources
            for number, line in enumerate(citation_source_lines(item), 1)
            if term in line
        })
        if locations:
            raise SynthesisUnavailable(
                f"引语{term!r}未被所选原文行覆盖；该引语位于{', '.join(locations)}。"
                "保留这项陈述时请选择对应行；与用户问题无关的补充可以删除。"
            )


def bind_review_paragraphs(
    review: SynthesisReview,
    evidence: Sequence[Evidence],
    *,
    question: str,
    verified_context: Sequence[str] = (),
) -> str:
    index = {item.evidence_id: item for item in evidence if citation_prefix(item.source_type)}
    verified_numbers = set(re.findall(
        r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", "\n".join(verified_context)
    ))
    issues: dict[int, str] = {}
    rendered: list[str] = []
    if not review.paragraphs:
        raise SynthesisUnavailable("结构化正文没有内容")
    for position, paragraph in enumerate(review.paragraphs):
        try:
            if re.search(r"(?:\[\d+\])+\s*[。.!！]?\s*$", paragraph.text):
                raise SynthesisUnavailable(
                    "text末尾含数字引用编号；引用仅填写citations字段，请删除text中的[1]等编号。"
                    "正文中的业务数值应保留，并选择支持它们的原文行。"
                )
            if not paragraph.text.strip() or "{{" in paragraph.text or LABEL_PATTERN.search(paragraph.text):
                raise SynthesisUnavailable("正文须为无内部引用标记的自然文本")
            if paragraph.citations:
                tokens = MarkdownIt().parse(paragraph.text)
                multiple_paragraphs = sum(token.type == "paragraph_open" for token in tokens) > 1
                escaped_breaks = any(
                    "\\n\\n" in child.content
                    for token in tokens if token.type == "inline"
                    for child in token.children or [] if child.type == "text"
                )
                if multiple_paragraphs or escaped_breaks:
                    raise SynthesisUnavailable(
                        "一个有来源的正文项包含多个自然段或双重转义换行，末尾引用无法逐段对应。"
                        "请将当前项写成一个连贯自然段并绑定支持该段的引用；正文无需添加开场概述。"
                        "每个独立自然段应使用单独的paragraphs项及其citations，不要在text内放入字面的\\n\\n。"
                    )
            selected = []
            quotes = []
            for span in paragraph.citations:
                item = index.get(span.evidence_id)
                if item is None:
                    raise SynthesisUnavailable("来源ID未出现在本轮证据中")
                lines = citation_source_lines(item)
                if not 1 <= span.start_line <= span.end_line <= len(lines):
                    raise SynthesisUnavailable("原文行范围超出该来源正文")
                if span.end_line - span.start_line >= 12:
                    raise SynthesisUnavailable("原文范围过长，请按具体陈述分别选择支持行")
                quotes.append("\n".join(lines[span.start_line - 1:span.end_line]))
                selected.append(item.evidence_id)
            quote = "\n".join(quotes)
            if paragraph.citations:
                numeric_text = _without_supported_source_metadata(
                    paragraph.text, [index[key] for key in selected],
                )
                _check_quoted_terms(numeric_text, quote, [index[key] for key in selected])
                for section in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)+)\s*(?:章节|节)", paragraph.text):
                    number = section.group(1)
                    if any(re.search(
                        rf"(?<![\d.]){re.escape(number)}(?![\d.])",
                        split_evidence_content(index[key].content)[0],
                    ) or re.search(
                        rf"(?m)^{re.escape(number)}\s+[^\n]{{1,100}}$",
                        split_evidence_content(index[key].content)[1],
                    ) for key in selected):
                        numeric_text = numeric_text.replace(section.group(0), "章节")
                numbers = set(re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", numeric_text))
                quoted_numbers = set(re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", quote))
                if paragraph.calculation:
                    _validate_calculation(CitationGrounding(
                        marker=str(position), claim=paragraph.text, quote=quote,
                        evidence_id=selected[0], calculation=paragraph.calculation,
                        inputs=paragraph.inputs, result=paragraph.result,
                    ), question)
                elif not numbers <= quoted_numbers | verified_numbers:
                    missing = sorted(numbers - quoted_numbers - verified_numbers)
                    ranges = ", ".join(f"{span.evidence_id}:L{span.start_line}-L{span.end_line}" for span in paragraph.citations)
                    raise SynthesisUnavailable(
                        f"陈述数值{missing!r}未出现在所选原文行{ranges}；这些行包含数值{sorted(quoted_numbers)!r}。"
                        "请为每个陈述选择包含对应数值及含义的原文行；来源没有的额外陈述应删除，"
                        "用户要求的数值无法确认时明确说明。演算须提供可复算的输入与表达式。"
                    )
                if any(operator in paragraph.text for operator in ("=", "＝", "÷")):
                    expression = re.split(r"[。；;\n，,）]", re.split(r"[=＝]", paragraph.text, maxsplit=1)[-1], maxsplit=1)[0]
                    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression.casefold()))
                    if not identifiers <= set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", quote.casefold())):
                        raise SynthesisUnavailable("所选原文行未包含公式变量")
            markers = "".join("{{" + key + "}}" for key in dict.fromkeys(selected))
            rendered.append(paragraph.text + (" " + markers if markers else ""))
        except (ValueError, SynthesisUnavailable) as exc:
            issues[position] = str(exc)
    if issues:
        raise ParagraphBindingError(issues)
    return "\n\n".join(rendered)


def _validate_calculation(entry: CitationGrounding, question: str) -> None:
    try:
        tree = ast.parse(entry.calculation, mode="eval")
    except SyntaxError as exc:
        raise SynthesisUnavailable("calculation只能填写数值算式；文字推断应放在text，calculation保持空字符串") from exc
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
               ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
    nodes = list(ast.walk(tree))
    if len(nodes) > 64 or any(not isinstance(node, allowed) for node in nodes):
        raise SynthesisUnavailable("引用推导包含无法核验的运算")
    if any(isinstance(node, ast.Constant) and type(node.value) not in (int, float) for node in nodes):
        raise SynthesisUnavailable("引用推导只能包含数值运算")
    names = {node.id for node in nodes if isinstance(node, ast.Name)}
    if not names or names != set(entry.inputs):
        raise SynthesisUnavailable("引用推导的输入与运算变量未对应")
    source_numbers = {
        float(value) for value in re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", question + " " + entry.quote)
    }
    if any(value not in source_numbers for value in entry.inputs.values()):
        raise SynthesisUnavailable("引用推导的输入未见于用户问题或来源原文")
    try:
        result = eval(compile(tree, "<citation-calculation>", "eval"), {"__builtins__": {}}, entry.inputs)
    except ArithmeticError as exc:
        raise SynthesisUnavailable("引用推导无法完成数值运算，请检查分母和输入") from exc
    if entry.result is None or not math.isclose(result, entry.result, rel_tol=1e-9, abs_tol=1e-9):
        raise SynthesisUnavailable("引用推导结果与所给运算不一致")
    percent_scaled = any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        and any(isinstance(operand, ast.Constant) and operand.value == 100
                for operand in (node.left, node.right))
        for node in nodes
    )
    claim_numbers = {
        float(value) / (100 if percent and not percent_scaled else 1)
        for value, percent in re.findall(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(\s*[%％])?", entry.claim)
    }
    if not any(math.isclose(result, value, rel_tol=1e-9, abs_tol=1e-9) for value in claim_numbers):
        raise SynthesisUnavailable("引用推导结果未体现在对应陈述中")


def bind_review_citations(
    review: SynthesisReview, evidence: Sequence[Evidence], *, question: str = "",
) -> str:
    markers = re.findall(r"\{\{(cite-\d+)\}\}", review.content)
    entries = {entry.marker: entry for entry in review.citations}
    if len(markers) != len(set(markers)) or len(entries) != len(review.citations) or set(markers) != set(entries):
        raise SynthesisUnavailable("引用核查条目与正文引用未完整对应")
    if PLACEHOLDER_PATTERN.search(review.content):
        raise SynthesisUnavailable("正文仍包含未核验原文的引用")
    citable = [item for item in evidence if citation_prefix(item.source_type)]
    index = evidence_index(citable)
    resolved: dict[str, str] = {}
    for marker, entry in entries.items():
        claim = " ".join(entry.claim.split())
        preceding = review.content.split("{{" + marker + "}}", 1)[0]
        preceding = re.sub(r"\{\{cite-\d+\}\}", "", preceding)
        if not claim or not " ".join(preceding.split()).endswith(claim):
            raise SynthesisUnavailable(f"引用 {marker} 未与所核查陈述对应")
        quote = " ".join(evidence_location_text(body=entry.quote).split())
        if not quote:
            raise SynthesisUnavailable("引用核查没有提供对应原文")
        matches = [
            item for item in citable
            if quote in " ".join(evidence_location_text(
                body=split_evidence_content(item.content)[1],
            ).split())
        ]
        if not matches:
            raise SynthesisUnavailable("引用核查提供的摘录未出现在已取得的原文中")
        numbers = set(re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", claim))
        quoted_numbers = set(re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", quote))
        if entry.calculation:
            _validate_calculation(entry, question)
        elif not numbers <= quoted_numbers:
            raise SynthesisUnavailable(f"引用 {marker} 的原文摘录未包含所陈述的数值")
        if any(operator in claim for operator in ("=", "＝", "÷")):
            expression = re.split(r"[。；;\n，,）]", re.split(r"[=＝]", claim, maxsplit=1)[-1], maxsplit=1)[0]
            identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression.casefold()))
            quoted_identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", quote.casefold()))
            if not identifiers <= quoted_identifiers:
                raise SynthesisUnavailable(f"引用 {marker} 的原文摘录未包含公式中的变量")
        requested = index.get(entry.evidence_id)
        selected = next((item for item in matches if item.evidence_id == entry.evidence_id), None)
        if selected is None:
            same_document = [
                item for item in matches
                if requested is not None and requested.source_ref
                and item.source_type == requested.source_type
                and item.source_ref.split("#", 1)[0] == requested.source_ref.split("#", 1)[0]
            ]
            if not same_document:
                raise SynthesisUnavailable("引用摘录与所声明的文档身份不一致")
            selected = min(same_document, key=lambda item: len(item.content))
        resolved[marker] = selected.evidence_id
    return re.sub(
        r"\{\{(cite-\d+)\}\}",
        lambda match: "{{" + resolved[match.group(1)] + "}}",
        review.content,
    )


class SynthesisRequest(ContractModel):
    question: str
    content: str = ""
    conclusions: list[ResearchConclusion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    requirements: dict[str, str] = Field(default_factory=dict)
    review: SynthesisReview | None = None


class RunFailure(ContractModel):
    run_id: str
    status: RunStatus = RunStatus.FAILED
    error: AgentError
    completed_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    results: list[ResearchResult] = Field(default_factory=list)
    resume_action: str = "从失败位置继续并复用已完成研究"
    created_at: str = Field(default_factory=utc_now_iso)

    def has_research_records(self) -> bool:
        return bool(self.results)


def collect_evidence(results: Sequence[ResearchResult]) -> list[Evidence]:
    return dedupe_evidence([item for result in results for item in result.evidence])


def compute_answer_basis(evidences: Sequence[Evidence]) -> tuple[AnswerBasis, list[SourceType]]:
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


def render_candidate(conclusions: Sequence[ResearchConclusion], requirement_texts: Mapping[str, str]) -> str:
    """确定性拼接（**不再是回答的兜底路径**）。

    仅用于离线核对/取证（例如人工查看「结论 + 占位符」的原始形态）。模型不可用或未按
    引用格式作答时，答案链路不再调用它 —— 见 :class:`SynthesisUnavailable`。
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for conclusion in conclusions:
        for requirement_id in conclusion.requirement_ids or ["?"]:
            if requirement_id in seen:
                continue
            seen.add(requirement_id)
            blocks.append(f"### {requirement_texts.get(requirement_id, requirement_id)}")
            if conclusion.basis is ConclusionBasis.GAP:
                blocks.append(f"（缺少依据：{conclusion.statement}）")
                continue
            marker = "".join(f"{{{{{eid}}}}}" for eid in conclusion.evidence_ids)
            blocks.append(f"{conclusion.statement}{marker}")
    return "\n\n".join(blocks)


def synthesize_answer(
    question: str, results: Sequence[ResearchResult], *, requirements: Mapping[str, str],
    extra_evidence: Sequence[Evidence] = (), workspace_id: str = "", run_id: str = "",
    model: Callable[[SynthesisRequest], str] | None = None,
    extra_gaps: Sequence[MaterialGap] = (),
) -> CandidateAnswer:
    evidence = [
        item for item in dedupe_evidence([*collect_evidence(results), *extra_evidence])
        if evidence_ready_for_answer(item)
    ]
    aliases = {
        member_id: item.evidence_id
        for item in evidence
        for member_id in [item.evidence_id, *item.member_evidence_ids]
    }
    conclusions = [
        item.model_copy(update={
            "evidence_ids": list(dict.fromkeys(aliases[evidence_id] for evidence_id in item.evidence_ids)),
        })
        for result in results
        for item in result.conclusions
        if all(evidence_id in aliases for evidence_id in item.evidence_ids)
    ]
    citable_ids = {item.evidence_id for item in evidence if citation_prefix(item.source_type)}
    source_supported_requirements = {
        rid for item in evidence if item.evidence_id in citable_ids
        for rid in item.supported_requirements
    }
    supported_requirements = {
        requirement_id for conclusion in conclusions
        if conclusion.basis is not ConclusionBasis.GAP and conclusion.evidence_ids
        for requirement_id in conclusion.requirement_ids
    } | source_supported_requirements
    conclusions = [
        item for item in conclusions
        if item.basis is not ConclusionBasis.GAP
        or not item.requirement_ids
        or any(rid not in source_supported_requirements for rid in item.requirement_ids)
    ]
    basis, components = compute_answer_basis(evidence)
    request = SynthesisRequest(
        question=question, conclusions=conclusions, evidence=evidence, requirements=dict(requirements),
    )
    if model is None:
        # 没有可用的对话模型 -> 明确失败。旧实现回落 ``render_candidate``，产出
        # 「内部标识符＋原文首句」的拼接稿，看着像答案却答非所问（用户实测缺陷）。
        raise SynthesisUnavailable(
            "没有可用的对话模型，无法综合答案（不再用确定性拼接冒充回答）"
        )
    content = str(model(request) or "").strip()
    # 缺口只报「没有被任何来源回答」的要求：同一项要求已由通用解释或其他任务回答时，
    # 不能再列成「资料不足」（旧实现会因此反复输出资料不足说明，把正文挤满）。
    gaps = [
        gap
        for result in results
        for gap in result.gaps
        if any(rid not in source_supported_requirements for rid in gap.requirement_ids)
    ]
    # 外部核查得出的**用户交付要求**缺口（例如「要求官方来源但没有取得官方页面」）一并
    # 计入终态：否则答案已承认没交付，run 仍登记 completed、gaps=[]（缺陷 F19）。
    gaps.extend(extra_gaps)
    if request.review is not None:
        supported_requirements = set(request.review.covered_requirements)
        gaps = [*request.review.gaps, *extra_gaps]
        supported_requirements.difference_update(
            rid for gap in extra_gaps for rid in gap.requirement_ids
        )
    gaps = list({
        (tuple(sorted(gap.requirement_ids)), gap.missing, gap.impact): gap
        for gap in gaps
    }.values())
    return CandidateAnswer(
        question=question, content=content, workspace_id=workspace_id, run_id=run_id,
        conclusions=conclusions, evidence=evidence,
        covered_requirements=[rid for rid in requirements if rid in supported_requirements],
        uncovered_requirements=[rid for rid in requirements if rid not in supported_requirements],
        open_questions=list(dict.fromkeys(item for result in results for item in result.open_questions)),
        conflicts=list(dict.fromkeys(item for result in results for item in result.progress.conflicts)),
        gaps=gaps,
        answer_basis=basis, basis_components=components,
    )


def render_gap_section(gaps: Sequence[MaterialGap], requirement_texts: Mapping[str, str]) -> str:
    lines = ["### 资料不足说明"]
    if not gaps:
        lines.append("未获得回答所需的关键资料，相关结论仍需确认。")
    for gap in gaps:
        titles = "、".join(requirement_texts.get(rid, rid) for rid in gap.requirement_ids) or "当前问题"
        missing = _gap_display_text(gap.missing or "未取得")
        impact = _gap_display_text(gap.impact or "相关陈述仍需确认")
        lines.append(f"- 要求「{titles}」：缺少的依据：{missing}")
        lines.append(f"  对结论的影响：{impact}")
    return "\n".join(lines)


def _gap_display_text(text: str) -> str:
    terms = {
        "confirmed_source_scope": "已核对的原文范围",
        "provided_dimensions": "原文提供的内容",
        "requested_dimensions": "所问项目",
        "requirement_ids": "问题要求",
        "evidence_ids": "来源依据",
    }
    pattern = "|".join(terms)
    text = re.sub(rf"[（(][^（）()]*\b(?:{pattern})\b[^（）()]*[）)]", "", text)
    for identifier, label in terms.items():
        text = re.sub(rf"\b{identifier}\b", label, text)
    return text.strip().rstrip("。.!！；;，,") + "。"


def assign_citation_ids(
    content: str, evidence: Sequence[Evidence], *, existing: Sequence[CitationEntry] = (),
    workspace_names: Mapping[str, str] | None = None,
    relevance_by_source: Mapping[str, Mapping[str, Any]] | None = None,
) -> CitationAssignment:
    index = evidence_index(evidence)
    relevance = relevance_by_source or {}
    citations = {item.label: item for item in existing}
    labels = {item.evidence_id: item.label for item in existing if item.evidence_id}
    problems: list[str] = []
    counters: dict[str, int] = {}
    for citation in existing:
        match = re.fullmatch(r"([SWMH])(\d+)", citation.label)
        if match:
            prefix = match.group(1)
            counters[prefix] = max(counters.get(prefix, 0), int(match.group(2)))

    def label_for(evidence_id: str) -> str:
        if evidence_id in labels:
            return labels[evidence_id]
        item = index.get(evidence_id)
        if item is None:
            problems.append(f"占位符 {evidence_id} 没有对应证据")
            return evidence_id
        prefix = citation_prefix(item.source_type)
        if prefix is None:
            problems.append(f"来源 {item.source_type.value} 不得生成文档引用编号")
            return evidence_id
        counters[prefix] = counters.get(prefix, 0) + 1
        label = f"{prefix}{counters[prefix]}"
        labels[evidence_id] = label
        heading, body = split_evidence_content(item.content)
        citations[label] = CitationEntry(
            label=label, source_type=SourceType(item.source_type.value), evidence_id=item.evidence_id,
            workspace_id=item.workspace_id, workspace_name=(workspace_names or {}).get(item.workspace_id, ""),
            # 来源面板展示**证据正文全文**（章节标题单独放在 heading_path；内部定位标识留在
            # source_ref，只用于定位）。此前这里取「首句」，会把「标题：…\n内容：…」形态的
            # 证据截成只剩标题 —— 用户实测的「引用只有几个字」就是这个截断。
            source_ref=item.source_ref, snippet=body,
            heading_path=item.heading_path or heading,
            # 文档标题随证据一起到达引用面板（缺陷 F13）：只有章节名时，用户与模型都
            # 无法确认这段来自哪篇文章。
            document_title=item.document_title,
            source_status=CitationSourceStatus.ACTIVE,
            **_relevance_citation_fields(relevance.get(str(item.source_ref))),
        )
        return label

    referenced = [match.group(1) for match in PLACEHOLDER_PATTERN.finditer(content)]
    ordered = [label_for(evidence_id) for evidence_id in referenced]
    if problems:
        raise CitationError(problems)
    rendered = PLACEHOLDER_PATTERN.sub(lambda match: f"[{label_for(match.group(1))}]", content)
    return CitationAssignment(
        content=rendered, citations=sorted(
            citations.values(), key=lambda item: (item.label[0], int(item.label[1:]))
        ), labels=labels,
        reused_labels=list(dict.fromkeys(label for label in ordered if ordered.count(label) > 1)),
        unused_evidence_ids=[item.evidence_id for item in evidence if item.evidence_id not in set(referenced)],
    )


_RELEVANCE_CITATION_KEYS = (
    "relevance_checked", "rerank_score", "score_type", "profile_id", "evidence_hash", "member_refs", "similarity",
)


def _relevance_citation_fields(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    return {key: record[key] for key in _RELEVANCE_CITATION_KEYS if key in record}


def answer_version(artifact: AnswerArtifact) -> str:
    if not artifact.content_hash:
        raise AnswerVersionError("AnswerArtifact 缺少 content_hash")
    return f"{artifact.content_hash[:12]}@{artifact.status.value}"


def publish_answer(
    candidate: CandidateAnswer, *, requirements: Mapping[str, str], answer_id: str | None = None,
) -> AnswerArtifact:
    if not candidate.content.strip():
        raise ValueError("答案正文为空")
    index = candidate.evidence_by_id()
    citations = {item.label: item for item in candidate.citations}
    problems: list[str] = []
    for match in LABEL_PATTERN.finditer(candidate.content):
        label = "".join(match.groups())
        citation = citations.get(label)
        evidence = index.get(citation.evidence_id) if citation is not None else None
        if citation is None or evidence is None:
            problems.append(f"引用 {label} 没有对应资料")
        elif citation.workspace_id != candidate.workspace_id or evidence.workspace_id != candidate.workspace_id:
            problems.append(f"引用 {label} 不属于当前工作区")
        elif citation_prefix(evidence.source_type) != match.group(1):
            problems.append(f"引用 {label} 的来源类型不匹配")
    if problems:
        raise CitationError(problems)
    has_gaps = bool(candidate.gaps or candidate.uncovered_requirements)
    status = RunStatus.COMPLETED_WITH_GAPS if has_gaps else RunStatus.COMPLETED
    content = candidate.content
    if has_gaps:
        undisclosed = [gap for gap in candidate.gaps if not _gap_is_explicit(content, gap.missing)]
        gap_section = render_gap_section(undisclosed, requirements)
        if (undisclosed or not candidate.gaps) and gap_section not in content:
            content = f"{content}\n\n{gap_section}".strip()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    active_id = answer_id or "ans-" + hashlib.sha256(f"{candidate.run_id}|{digest}".encode()).hexdigest()[:24]
    return AnswerArtifact(
        answer_id=active_id, run_id=candidate.run_id, workspace_id=candidate.workspace_id,
        content=content, citations=list(candidate.citations), content_hash=digest,
        answer_basis=candidate.answer_basis, basis_components=list(candidate.basis_components),
        status=status, created_at=utc_now_iso(),
    )


def build_failure_outcome(results: Sequence[ResearchResult], error: AgentError, *, run_id: str) -> RunFailure:
    return RunFailure(
        run_id=run_id, error=error,
        completed_task_ids=[result.task_id for result in results if result.status in {TaskStatus.COMPLETED, TaskStatus.COMPLETED_WITH_GAPS}],
        failed_task_ids=[result.task_id for result in results if result.status is TaskStatus.FAILED],
        evidence_ids=[item.evidence_id for result in results for item in result.evidence], results=list(results),
        resume_action=f"POST /api/runs/{run_id}/resume：从失败位置继续并复用已完成研究",
    )


def parse_answer_basis(raw: str) -> AnswerBasis | None:
    try:
        return AnswerBasis(str(raw))
    except ValueError:
        return None


def answer_basis_display(raw: str) -> str:
    basis = parse_answer_basis(raw)
    return ANSWER_BASIS_LABELS.get(basis, str(raw)) if basis is not None else str(raw)
