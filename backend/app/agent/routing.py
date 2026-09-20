"""Intent Agent、范围检查与路由规则（PRD 6.1–6.3）。Owner: T06（routing）。

本模块是**意图识别层**的唯一实现，三件事：

1. **路由输入组装**（PRD 6.1）：当前问题 + 当前日期与用户时区 + 必要最近历史 +
   本会话固定记忆快照 + 当前工作区主题概览 + 目录查询得到的术语与文件线索。
   提示走 PRD 9.3 的固定顺序（复用 ``app.agent.context.build_context``）。
2. **规则引擎**（PRD 6.3 九条规则）：确定性、可离线单测；模型只做补充，
   自报置信度仅进诊断，不参与验收。硬规则（产品/概览、时效、内部引用、空知识库）
   在合并时覆盖模型输出。
3. **范围检查与调整**（PRD 6.1 / 6.3 末段）：目录命中只用于定位原文；缺明确解释的
   术语标 ``definition_unknown`` 并进入 ``unresolved_terms``；Planner 可在研究过程中
   通过 :func:`adjust_route_decision` 调整来源需求。

``RouteDecision`` 字段定义在 :mod:`app.agent.contracts`（T00b 冻结，PRD 6.2 全 12
字段），本模块不重新定义模型，也不改契约。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent.contracts import Freshness, IntentKind, RouteDecision
from app.agent.events import EventType, StoredEvent, build_event
from app.generation.knowledge_catalog import (
    CatalogMatch,
    CatalogPromptProjection,
    KnowledgeCatalog,
)
from app.generation.prompt_policy import QueryMode, classify_query

__all__ = [
    "CLARIFICATION_MAX_QUESTION_CHARS",
    "DEFAULT_CATALOG_HIT_LIMIT",
    "MAX_RECENT_HISTORY",
    "ROUTING_RULES_VERSION",
    "RULE_DESCRIPTIONS",
    "IntentAgent",
    "RouteDiagnostics",
    "RouteParseError",
    "RoutePrompt",
    "RouteResult",
    "RoutingHistoryTurn",
    "RoutingInput",
    "RuleHit",
    "RuleOutcome",
    "ScopeCheckResult",
    "adjust_route_decision",
    "build_route_decided_event",
    "build_route_prompt",
    "check_scope",
    "evaluate_route_rules",
    "resolve_reference",
    "route_decided_payload",
    "route_question",
]

#: 路由规则版本（规则层语义变化时递增；路由评测报告会记录它）。
ROUTING_RULES_VERSION: Final[str] = "routing-rules-v1"

#: 进入路由提示的目录条目数上限（完整目录留在本地，PRD 6.1）。
DEFAULT_CATALOG_HIT_LIMIT: Final[int] = 6

#: 路由输入里最多带的最近历史轮数（必要历史，非全量）。
MAX_RECENT_HISTORY: Final[int] = 8

#: 短问题 + 含义不明术语 → 主动澄清（PRD 6.3 R4 / 13 章）。
CLARIFICATION_MAX_QUESTION_CHARS: Final[int] = 24

#: 目录命中的最低相关性：只有标题 / 术语 / 别名级别的**子串级**命中才算「线索」。
#: 仅词元重叠（100–199）或词元恰好出现在标题里（300）不算，避免把「这个方案呢？」
#: 这类泛指问题误判成内部资料问题。
CATALOG_CLUE_MIN_SCORE: Final[int] = 400

#: PRD 6.3 决策规则表（九条），规则 id 与描述一一对应。
RULE_DESCRIPTIONS: Final[dict[str, str]] = {
    "R1": "用户明确引用笔记、内部文档、既有方案或当前项目 → 查询知识库",
    "R2": "对话已围绕某内部资料展开，用户继续追问 → 历史补全对象并继续知识查询",
    "R3": "内部术语目录命中且问题涉及其在本系统中的定义或实现 → 查询对应原文",
    "R4": "术语含义不明确、可能指内部对象 → 范围检查后决定路径",
    "R5": "用户明确要求今天、最新、当前版本等信息 → 联网查询并检查时间",
    "R6": "通用解释同时涉及已有内部方案 → General 与 RAG 分工",
    "R7": "仅询问产品功能或文档数量 → 使用产品说明或结构化统计",
    "R8": "用户回忆以前的讨论 → 查询工作区记忆与历史记录",
    "R9": "知识库为空 → 通用与网页问题仍可处理；内部资料问题说明当前资料情况",
}

_INTERNAL_MARKERS: Final[tuple[str, ...]] = (
    "我的笔记",
    "我笔记",
    "笔记里",
    "我的文档",
    "我的资料",
    "我们内部",
    "内部资料",
    "内部文档",
    "内部方案",
    "内部系统",
    "内部术语",
    "我们的项目",
    "我们项目",
    "本项目",
    "当前项目",
    "我们的方案",
    "我们方案",
    "既有方案",
    "现有方案",
    "我们的实现",
    "我们实现",
    "我们的代码",
    "知识库",
    "资料库",
    "导入的",
    "工作区",
    "公司内部",
)

_FRESHNESS_CURRENT_MARKERS: Final[tuple[str, ...]] = (
    "今天",
    "今日",
    "最新",
    "最新版",
    "最近",
    "现在",
    "实时",
    "刚刚",
    "刚发布",
    "当下",
    "如今",
    "目前最新",
    "这几天",
    "这两天",
    "当前版本",
    "最新版本",
)

_FRESHNESS_RANGE_MARKERS: Final[tuple[str, ...]] = (
    "本周",
    "这周",
    "上周",
    "本月",
    "这个月",
    "上个月",
    "本季度",
    "上个季度",
    "今年",
    "去年",
    "近一周",
    "近一个月",
    "近三个月",
    "近半年",
    "近一年",
    "最近一周",
    "最近一个月",
    "最近几天",
    "近期",
    "过去一周",
    "过去一个月",
    "过去一年",
)

_FRESHNESS_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:从|自)\s*\d{4}\s*[-/年]|近\s*\d+\s*(?:天|周|个月|年)|"
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*(?:到|至|-|~)"
)

_PRONOUNS: Final[tuple[str, ...]] = (
    "这个方案",
    "那个方案",
    "这个文档",
    "那个文档",
    "这种方式",
    "那种方式",
    "第二种方式",
    "第一种方式",
    "该方案",
    "该文档",
    "那套",
    "这套",
    "那种",
    "这种",
    "这篇",
    "那篇",
    "上述",
    "前者",
    "后者",
    "这些",
    "那些",
    "这个",
    "那个",
    "它",
)

_MEMORY_MARKERS: Final[tuple[str, ...]] = (
    "还记得",
    "我之前问过",
    "我以前问过",
    "我们之前聊过",
    "之前讨论过",
    "之前我们",
    "上次我们",
    "上次提到",
    "之前提到过",
    "以前说过",
    "之前说过",
    "我提过",
    "我们聊过",
    "回忆一下",
    "帮我回忆",
)

_GENERAL_MARKERS: Final[tuple[str, ...]] = (
    "是什么",
    "为什么",
    "原理",
    "概念",
    "怎么理解",
    "如何理解",
    "一般",
    "通常",
    "常见的",
    "最佳实践",
    "设计建议",
    "架构建议",
    "有什么通用方法",
    "怎么排查",
)

_DEFINITION_INTENT_MARKERS: Final[tuple[str, ...]] = (
    "定义",
    "是什么",
    "什么意思",
    "含义",
    "怎么实现",
    "如何实现",
    "实现方式",
    "原理",
    "机制",
    "怎么做的",
    "如何落地",
)

_TOPIC_CUT_MARKERS: Final[tuple[str, ...]] = (
    "？",
    "?",
    "。",
    "，",
    ",",
    "是",
    "怎么",
    "如何",
    "为什么",
    "有哪些",
    "哪些",
    "多少",
)

_ROUTE_DECISION_FIELDS: Final[frozenset[str]] = frozenset(RouteDecision.model_fields)


class _RoutingModel(BaseModel):
    """路由模型的共同基类（与共享契约一致：多传字段直接报错）。"""

    model_config = ConfigDict(extra="forbid")


class RouteParseError(ValueError):
    """模型返回的路由结论结构不合法（未知 intent、非法 JSON 等）。"""


# =====================================================================
# 1. 路由输入（PRD 6.1）
# =====================================================================


class RoutingHistoryTurn(_RoutingModel):
    """必要的最近历史轮次（路由只带必要历史，不带全量会话）。"""

    role: Literal["user", "assistant"]
    content: str
    created_at: str = ""


class RoutingInput(_RoutingModel):
    """Intent Agent 的全部输入（PRD 6.1）。

    ``catalog_hits`` 由 :meth:`KnowledgeCatalog.search` 产出；``catalog_clues``
    是更轻的线索文本（例如工作区主题概览里的词），供规则层在无目录实例时使用。
    """

    question: str
    now: str = ""
    timezone: str = ""
    recent_history: list[RoutingHistoryTurn] = Field(default_factory=list)
    memory_snapshot: str = ""
    workspace_overview: str = ""
    catalog_hits: list[CatalogMatch] = Field(default_factory=list)
    catalog_clues: list[str] = Field(default_factory=list)
    file_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    workspace_id: str = ""
    knowledge_revision: int = Field(default=0, ge=0)

    @property
    def knowledge_available(self) -> bool:
        """知识库是否非空（PRD 6.3 R9 的判定依据）。"""

        return self.file_count > 0

    def recent_user_turns(self) -> list[RoutingHistoryTurn]:
        return [turn for turn in self.recent_history if turn.role == "user"]


# =====================================================================
# 2. 规则层（PRD 6.3）
# =====================================================================


class RuleHit(_RoutingModel):
    """命中的一条决策规则及可复核的证据串。"""

    rule_id: str
    description: str
    evidence: str = ""


class RuleOutcome(_RoutingModel):
    """规则层输出：结论 + 命中规则 + 说明（不含模型调用）。"""

    decision: RouteDecision
    hits: list[RuleHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def rule_ids(self) -> list[str]:
        return [hit.rule_id for hit in self.hits]


class RouteDiagnostics(_RoutingModel):
    """诊断信息：只用于排查，不作为验收依据（PRD 6.2 末段）。"""

    model_used: bool = False
    confidence: float | None = None
    dropped_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ScopeCheckResult(_RoutingModel):
    """范围检查结果（PRD 6.1）：目录只用于定位原文，不构成依据。"""

    decision: RouteDecision
    knowledge_required: bool = False
    web_required: bool = False
    catalog_empty: bool = True
    scope_evidence_ids: list[str] = Field(default_factory=list)
    resolved_terms: list[str] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    definition_unknown_terms: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RouteResult(_RoutingModel):
    """一次路由的完整结果（图节点消费 ``decision``，评测消费全部字段）。"""

    decision: RouteDecision
    scope: ScopeCheckResult
    rules: list[RuleHit] = Field(default_factory=list)
    diagnostics: RouteDiagnostics = Field(default_factory=RouteDiagnostics)


def _contains(text: str, markers: Sequence[str]) -> str:
    """返回第一个命中的标记（便于记录可复核证据），未命中返回空串。"""

    for marker in markers:
        if marker in text:
            return marker
    return ""


def _detect_freshness(question: str) -> Freshness:
    if _FRESHNESS_RANGE_RE.search(question):
        return Freshness.DATE_RANGE
    if _contains(question, _FRESHNESS_RANGE_MARKERS):
        return Freshness.DATE_RANGE
    if _contains(question, _FRESHNESS_CURRENT_MARKERS):
        return Freshness.CURRENT
    return Freshness.NONE


def _topic_of(text: str) -> str:
    """从上一轮问题里取一个可替换指代的对象（启发式、可复核、长度受限）。"""

    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    cut = len(cleaned)
    for marker in _TOPIC_CUT_MARKERS:
        position = cleaned.find(marker)
        if 0 < position < cut:
            cut = position
    topic = cleaned[:cut].strip("，,。 的")
    return topic[:30]


def resolve_reference(
    question: str, history: Sequence[RoutingHistoryTurn]
) -> tuple[str, str]:
    """按历史补全指代（PRD 6.3 R2）：返回 ``(resolved_question, note)``。

    没有指代词或没有可用历史时原样返回，并把原因写进 note，不猜测对象。
    """

    matched = _contains(question, _PRONOUNS)
    if not matched:
        return question, ""
    previous = [turn for turn in history if turn.role == "user"]
    if not previous:
        return question, "出现指代但本轮没有可用历史，未补全对象"
    topic = _topic_of(previous[-1].content)
    if not topic:
        return question, "上一轮问题没有可提取的对象，未补全指代"
    for pronoun in sorted(_PRONOUNS, key=len, reverse=True):
        if pronoun in question:
            return question.replace(pronoun, topic, 1), f"指代补全：{matched} → {topic}"
    return question, ""


def _catalog_term_matches(input: RoutingInput) -> list[CatalogMatch]:
    return [match for match in input.catalog_hits if match.kind == "term"]


def _catalog_document_matches(input: RoutingInput) -> list[CatalogMatch]:
    return [
        match
        for match in input.catalog_hits
        if match.kind != "term" and match.score >= CATALOG_CLUE_MIN_SCORE
    ]


def _mentioned_in_question(question: str, match: CatalogMatch) -> bool:
    """术语/别名是否真的出现在问题里（弱命中不能当「本轮相关术语」）。"""

    normalized = question.casefold()
    for label in (match.matched_on, *match.aliases):
        wanted = str(label or "").casefold().strip()
        if len(wanted) >= 2 and wanted in normalized:
            return True
    return False


def _mentioned_term_matches(question: str, input: RoutingInput) -> list[CatalogMatch]:
    return [
        match for match in _catalog_term_matches(input) if _mentioned_in_question(question, match)
    ]


def evaluate_route_rules(input: RoutingInput) -> RuleOutcome:
    """纯规则路由（无模型调用）：落地 PRD 6.3 九条规则。

    规则按固定优先级求值；所有命中规则都记进 ``hits``（不合并成单一原因），
    因此评测可以逐条核对「哪条规则把问题定向到了哪个来源」。
    """

    question = " ".join(str(input.question or "").split())
    resolved_question, reference_note = resolve_reference(question, input.recent_history)
    hits: list[RuleHit] = []
    notes: list[str] = []

    def hit(rule_id: str, evidence: str) -> None:
        hits.append(
            RuleHit(
                rule_id=rule_id,
                description=RULE_DESCRIPTIONS[rule_id],
                evidence=evidence,
            )
        )

    decision = RouteDecision(
        question=question,
        resolved_question=resolved_question or question,
        intent=IntentKind.RESEARCH,
    )
    if reference_note:
        notes.append(reference_note)

    # ---- R7：仅询问产品功能或文档数量（最高优先级，避免误送检索）
    mode = classify_query(question)
    if mode is QueryMode.PRODUCT_HELP:
        hit("R7", "产品功能问题标记命中")
        decision.intent = IntentKind.PRODUCT
        decision.reason_summary = "询问产品功能，使用产品说明回答"
        return RuleOutcome(decision=decision, hits=hits, notes=notes)
    if mode is QueryMode.KNOWLEDGE_OVERVIEW:
        hit("R7", "文档数量 / 概览问题标记命中")
        decision.intent = IntentKind.CATALOG
        decision.reason_summary = "询问工作区资料概览，使用结构化统计回答"
        return RuleOutcome(decision=decision, hits=hits, notes=notes)

    internal_marker = _contains(question, _INTERNAL_MARKERS)
    term_matches = _mentioned_term_matches(question, input)
    document_matches = _catalog_document_matches(input)
    unknown_terms = [match for match in term_matches if match.definition_unknown]
    general_marker = _contains(question, _GENERAL_MARKERS)

    # ---- R8：回忆以前的讨论
    memory_marker = _contains(question, _MEMORY_MARKERS)
    if memory_marker:
        hit("R8", f"回忆标记：{memory_marker}")
        decision.intent = IntentKind.MEMORY
        decision.needs_history = True

    # ---- R1：明确引用内部资料 / 项目
    if internal_marker:
        hit("R1", f"内部引用标记：{internal_marker}")
        decision.needs_knowledge = True
    elif document_matches:
        hit("R1", f"目录文件线索：{document_matches[0].matched_on}")
        decision.needs_knowledge = True

    # ---- R2：围绕内部资料继续追问（指代 + 历史）
    if reference_note and (
        decision.needs_knowledge or reference_note.startswith("指代补全")
    ):
        hit("R2", reference_note)
        decision.needs_knowledge = True

    # ---- R3：内部术语目录命中 + 问其定义或实现
    definition_marker = _contains(question, _DEFINITION_INTENT_MARKERS)
    if term_matches and definition_marker:
        hit("R3", f"术语命中 {term_matches[0].matched_on} + 定义意图 {definition_marker}")
        decision.needs_knowledge = True
        decision.scope_evidence_ids = _merge_unique(
            decision.scope_evidence_ids,
            [match.document_id for match in term_matches if match.document_id],
        )

    # ---- R4：术语含义不明确，可能指内部对象
    if unknown_terms:
        names = [match.matched_on for match in unknown_terms]
        hit("R4", f"含义不明术语：{'、'.join(names)}")
        decision.needs_knowledge = True
        decision.unresolved_terms = _merge_unique(decision.unresolved_terms, names)
        notes.append("目录命中但缺少明确解释：先做范围检查，只用于定位原文")
        if len(question) <= CLARIFICATION_MAX_QUESTION_CHARS and not decision.needs_history:
            decision.intent = IntentKind.CLARIFICATION
            notes.append("问题很短且术语不明，先请用户确认对象")

    # ---- R5：明确要求今天 / 最新 / 当前版本
    freshness = _detect_freshness(question)
    if freshness is not Freshness.NONE:
        marker = (
            _contains(question, _FRESHNESS_RANGE_MARKERS)
            or _contains(question, _FRESHNESS_CURRENT_MARKERS)
            or "时间范围表达"
        )
        hit("R5", f"时效标记：{marker}（{freshness.value}）")
        decision.needs_web = True
        decision.freshness = freshness
        if not decision.needs_knowledge:
            decision.reason_summary = "需要公开且可核实的最新信息，走联网路径"

    # ---- R6：通用解释同时涉及已有内部方案
    if general_marker and (decision.needs_knowledge or bool(input.catalog_clues)):
        hit("R6", f"通用解释标记：{general_marker}，同时有内部线索")
        decision.needs_general = True

    # ---- R9：知识库为空
    if not input.knowledge_available:
        hit("R9", "本区已登记文档数为 0")
        if decision.needs_knowledge:
            notes.append("当前工作区没有已登记资料：内部资料问题需说明资料情况")
        elif not decision.needs_web:
            notes.append("当前工作区没有已登记资料：通用或联网问题仍可处理")

    # ---- 兜底意图：没有内部/联网要求时按通用解释
    if (
        not decision.needs_knowledge
        and not decision.needs_web
        and decision.intent is IntentKind.RESEARCH
    ):
        decision.intent = IntentKind.GENERAL
        decision.needs_general = True

    if input.catalog_clues:
        clues = [clue for clue in input.catalog_clues[:3] if clue]
        if clues:
            decision.scope_evidence_ids = _merge_unique(decision.scope_evidence_ids, clues)
            notes.append(f"工作区主题线索：{'、'.join(clues)}")

    if not decision.reason_summary:
        decision.reason_summary = _reason_summary(decision)
    return RuleOutcome(decision=decision, hits=hits, notes=notes)


def _reason_summary(decision: RouteDecision) -> str:
    parts: list[str] = []
    if decision.needs_knowledge:
        parts.append("查本工作区资料")
    if decision.needs_web:
        parts.append("联网核实最新信息")
    if decision.needs_history:
        parts.append("查历史讨论")
    if decision.needs_general and not parts:
        parts.append("用通用知识解释")
    if not parts:
        parts.append("按通用路径处理")
    return "；".join(parts)


def _merge_unique(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    seen = {str(item) for item in existing}
    result = [str(item) for item in existing]
    for item in extra:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _unique(values: Sequence[str]) -> list[str]:
    return _merge_unique([], values)


# =====================================================================
# 3. 范围检查与调整（PRD 6.1 / 6.3 末段）
# =====================================================================


def check_scope(
    decision: RouteDecision,
    *,
    catalog: KnowledgeCatalog | None = None,
    input: RoutingInput | None = None,
    clue_terms: Sequence[str] | None = None,
    text_lookup: Callable[[str], str] | None = None,
) -> ScopeCheckResult:
    """对路由结论做范围检查（PRD 6.1）：

    - 目录命中只用于定位原文，结果写进 ``scope_evidence_ids``；
    - 缺明确解释的术语进 ``unresolved_terms``（不当作已定义概念）；
    - 引用片段必须真的存在于原文：给 ``text_lookup`` 时逐条校验，校验不通过的术语
      进 ``unresolved_terms`` 并记 note。
    """

    catalog_empty = True
    scope_ids: list[str] = []
    resolved_terms: list[str] = []
    unknown_terms: list[str] = []
    notes: list[str] = []

    candidate_terms = [str(term) for term in (clue_terms or ())]
    if not candidate_terms and input is not None:
        candidate_terms = [match.matched_on for match in _catalog_term_matches(input)]

    if catalog is not None:
        catalog_empty = catalog.is_empty()
        question = decision.resolved_question or decision.question
        result = catalog.search(question, limit=8)
        for match in result.matches:
            if match.document_id and match.score >= CATALOG_CLUE_MIN_SCORE:
                scope_ids.append(match.document_id)
            if match.kind != "term" or not _mentioned_in_question(question, match):
                continue
            if match.definition_unknown:
                unknown_terms.append(match.matched_on)
                continue
            entry = _term_entry(catalog, match.matched_on, match.document_id)
            if text_lookup is not None and entry is not None:
                body = text_lookup(entry.document_id) or ""
                if entry.snippet and entry.snippet not in body:
                    unknown_terms.append(match.matched_on)
                    notes.append(f"术语 {match.matched_on} 的引用片段未在原文复现，转为待确认")
                    continue
            resolved_terms.append(match.matched_on)
    elif input is not None:
        catalog_empty = not input.knowledge_available
        scope_ids.extend(
            match.document_id
            for match in input.catalog_hits
            if match.score >= CATALOG_CLUE_MIN_SCORE
        )
        for match in _mentioned_term_matches(decision.resolved_question or decision.question, input):
            if match.definition_unknown:
                unknown_terms.append(match.matched_on)
            else:
                resolved_terms.append(match.matched_on)

    unresolved = _merge_unique(
        [*decision.unresolved_terms, *unknown_terms],
        [term for term in candidate_terms if term in unknown_terms],
    )
    updated = decision
    if scope_ids:
        updated = adjust_route_decision(
            decision,
            scope_evidence_ids=_merge_unique(decision.scope_evidence_ids, _unique(scope_ids)),
        )
        notes.append("目录命中仅用于定位原文，不作为答案依据")
    if catalog_empty:
        notes.append("本区知识目录为空")

    return ScopeCheckResult(
        decision=updated,
        knowledge_required=updated.needs_knowledge,
        web_required=updated.needs_web,
        catalog_empty=catalog_empty,
        scope_evidence_ids=list(updated.scope_evidence_ids),
        resolved_terms=_unique(resolved_terms),
        unresolved_terms=unresolved,
        definition_unknown_terms=_unique(unknown_terms),
        rules=[],
        notes=notes,
    )


def _term_entry(catalog: KnowledgeCatalog, term: str, document_id: str) -> Any:
    for entry in catalog.terms:
        if entry.term == term and (not document_id or entry.document_id == document_id):
            return entry
    return None


def adjust_route_decision(
    decision: RouteDecision,
    *,
    resolved_question: str | None = None,
    needs_knowledge: bool | None = None,
    needs_web: bool | None = None,
    needs_history: bool | None = None,
    needs_general: bool | None = None,
    freshness: Freshness | str | None = None,
    entities: Sequence[str] | None = None,
    scope_evidence_ids: Sequence[str] | None = None,
    unresolved_terms: Sequence[str] | None = None,
    reason_summary: str | None = None,
    intent: IntentKind | str | None = None,
) -> RouteDecision:
    """研究过程中调整来源需求（PRD 6.3 末段，Planner 使用）。

    只允许显式传入的字段变化；返回**新对象**（原结论保持可追溯）。
    """

    payload = decision.model_dump(mode="json")
    updates: dict[str, Any] = {
        "resolved_question": resolved_question,
        "needs_knowledge": needs_knowledge,
        "needs_web": needs_web,
        "needs_history": needs_history,
        "needs_general": needs_general,
        "freshness": freshness,
        "entities": list(entities) if entities is not None else None,
        "scope_evidence_ids": list(scope_evidence_ids) if scope_evidence_ids is not None else None,
        "unresolved_terms": list(unresolved_terms) if unresolved_terms is not None else None,
        "reason_summary": reason_summary,
        "intent": intent,
    }
    for key, value in updates.items():
        if value is None:
            continue
        payload[key] = value
    if not str(payload.get("resolved_question") or "").strip():
        payload["resolved_question"] = payload.get("question", "")
    return RouteDecision.model_validate(payload)


# =====================================================================
# 4. 提示组装（PRD 6.1 + 9.3 固定顺序）
# =====================================================================

_DUTY_PROMPT_HEADER: Final[str] = (
    "你是 Everything RAG 的意图识别 Agent（Intent Agent）。"
    "你的唯一职责是判断这个问题需要哪些来源，并输出结构化结论；不要回答问题本身。"
)

_DUTY_PROMPT_RULES: Final[str] = """判断规则（与产品决策规则一致，按优先级）：
R1 明确引用笔记、内部文档、既有方案或当前项目 → 需要本工作区资料
R2 对话已围绕某内部资料展开、用户用“这个方案/第二种方式”等继续追问 → 用历史补全对象后查资料
R3 内部术语在目录里命中，且问题问它的定义或实现 → 查对应原文
R4 术语含义不明确、可能指内部对象 → 标记待确认术语，先做范围检查
R5 明确要求今天、最新、当前版本 → 需要联网并检查时间
R6 通用解释同时涉及已有内部方案 → 通用解释与工作区资料都要
R7 仅询问产品功能或文档数量 → 用产品说明或结构化统计，不要检索正文
R8 用户回忆以前的讨论 → 查工作区记忆与历史记录
R9 知识库为空 → 通用和联网问题仍可处理；内部资料问题要说明当前资料情况

要求：
- needs_knowledge 与 needs_web 可以同时为 true。
- intent 只能取 general / research / product / catalog / memory / clarification。
- freshness 只能取 none / current / date_range。
- 目录片段只是定位线索，不能当作事实依据；不要引用其中的原文当答案。
- 只输出 JSON 对象，字段：question / resolved_question / intent / needs_knowledge /
  needs_web / needs_history / needs_general / freshness / entities /
  scope_evidence_ids / unresolved_terms / reason_summary。不要输出额外字段、不要输出置信度。"""


class RoutePrompt(_RoutingModel):
    """一次路由请求的完整提示（已按 PRD 9.3 固定顺序组装）。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    embedded_chars: int = Field(default=0, ge=0)
    catalog_terms: int = Field(default=0, ge=0)
    injected_documents: int = Field(default=0, ge=0)


def build_route_prompt(
    input: RoutingInput,
    *,
    catalog: KnowledgeCatalog | None = None,
    projection: CatalogPromptProjection | None = None,
    catalog_hit_limit: int = DEFAULT_CATALOG_HIT_LIMIT,
) -> RoutePrompt:
    """组装 Intent Agent 的模型输入（PRD 6.1 路由输入 + 9.3 固定顺序）。

    只注入与本轮相关的目录条目（``projection``），完整目录留在本地。
    """

    from app.agent.context import ContextSnapshot, build_context

    query = input.question
    if projection is None and catalog is not None:
        projection = catalog.prompt_projection(query, limit=catalog_hit_limit)

    dynamic_lines: list[str] = []
    if input.now:
        dynamic_lines.append(f"当前时间：{input.now}")
    if input.timezone:
        dynamic_lines.append(f"用户时区：{input.timezone}")
    dynamic_lines.append(f"当前工作区：{input.workspace_id or '（默认）'}")
    dynamic_lines.append(
        f"本区资料规模：{input.file_count} 个文档 / {input.chunk_count} 个片段"
        f"（知识版本 {input.knowledge_revision}）"
    )
    if input.workspace_overview:
        dynamic_lines.append(f"工作区主题概览：{input.workspace_overview}")
    if input.catalog_clues:
        dynamic_lines.append("工作区主题线索：" + "、".join(input.catalog_clues[:8]))
    if projection is not None and projection.text:
        dynamic_lines.append("本轮相关目录条目（只用于定位原文）：" + projection.text)
    else:
        dynamic_lines.append("本轮相关目录条目：无（目录未命中）")

    history_messages = [
        {"role": turn.role, "content": turn.content}
        for turn in input.recent_history[-MAX_RECENT_HISTORY:]
    ]
    messages = build_context(
        duty_prompt=f"{_DUTY_PROMPT_HEADER}\n\n{_DUTY_PROMPT_RULES}",
        snapshot=ContextSnapshot(input.memory_snapshot or ""),
        history=history_messages,
        latest=[{"role": "user", "content": query}],
        dynamic=[{"role": "user", "content": "\n".join(dynamic_lines)}],
    )
    embedded = sum(len(str(message.get("content") or "")) for message in messages)
    return RoutePrompt(
        messages=messages,
        document_ids=list(projection.document_ids) if projection is not None else [],
        embedded_chars=embedded,
        catalog_terms=projection.term_count if projection is not None else 0,
        injected_documents=len(projection.document_ids) if projection is not None else 0,
    )


# =====================================================================
# 5. 模型输出解析（严格、不过度信任）
# =====================================================================


def parse_route_decision(
    raw: str | Mapping[str, Any], *, question: str = ""
) -> tuple[RouteDecision, RouteDiagnostics]:
    """解析并校验模型输出的 ``RouteDecision``。

    未知字段只进诊断（模型的额外字段不改变契约）；``intent`` / ``freshness`` 取值
    非法时抛 :class:`RouteParseError`，由调用方回退到规则层结论。
    """

    payload: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RouteParseError(f"路由结论不是合法 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RouteParseError("路由结论必须是 JSON 对象")

    dropped = [
        str(key)
        for key in payload
        if key not in _ROUTE_DECISION_FIELDS and key != "confidence"
    ]
    confidence_raw = payload.get("confidence")
    confidence: float | None = None
    if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool):
        confidence = float(confidence_raw)

    data = {key: value for key, value in payload.items() if key in _ROUTE_DECISION_FIELDS}
    if not str(data.get("question") or "").strip():
        data["question"] = question
    if not str(data.get("resolved_question") or "").strip():
        data["resolved_question"] = data["question"]
    if "intent" not in data:
        raise RouteParseError("路由结论缺少 intent")
    try:
        decision = RouteDecision.model_validate(data)
    except Exception as exc:  # pydantic 校验失败统一转成路由协议错误
        raise RouteParseError(f"路由结论校验失败: {exc}") from exc
    return decision, RouteDiagnostics(
        model_used=True,
        confidence=confidence,
        dropped_fields=dropped,
    )


# =====================================================================
# 6. Intent Agent
# =====================================================================


class _RouteModel(Protocol):
    """只依赖非流式 ``complete``（T03 的 ChatModel 已提供）。"""

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class IntentAgent:
    """意图识别 Agent：规则层 + 可选的模型补充（PRD 4.2 / 6.1–6.3）。

    规则层始终先跑，并作为**硬约束**覆盖模型输出：产品/概览问题不查正文、时效问题
    必须联网、内部引用必须查资料、空知识库要说明资料情况。模型的 ``confidence``
    只写进诊断字段，不作为验收依据。
    """

    def __init__(
        self,
        *,
        model: _RouteModel | None = None,
        catalog: KnowledgeCatalog | None = None,
        text_lookup: Callable[[str], str] | None = None,
        rules_version: str = ROUTING_RULES_VERSION,
        catalog_hit_limit: int = DEFAULT_CATALOG_HIT_LIMIT,
    ) -> None:
        self.model = model
        self.catalog = catalog
        self.text_lookup = text_lookup
        self.rules_version = rules_version
        self.catalog_hit_limit = catalog_hit_limit

    # ------------------------------------------------------------ 规则层

    def enrich(self, input: RoutingInput) -> RoutingInput:
        """把目录线索补进路由输入（调用方没给 hits 时由本 Agent 查询目录）。"""

        if input.catalog_hits or self.catalog is None or self.catalog.is_empty():
            return input
        result = self.catalog.search(input.question, limit=self.catalog_hit_limit)
        if not result.matches:
            return input
        return input.model_copy(update={"catalog_hits": list(result.matches)})

    def decide(self, input: RoutingInput) -> RouteResult:
        """纯规则路由（离线可用，不调用模型）。"""

        enriched = self.enrich(input)
        outcome = evaluate_route_rules(enriched)
        scope = check_scope(
            outcome.decision,
            catalog=self.catalog,
            input=enriched,
            text_lookup=self.text_lookup,
        )
        scope.rules = outcome.rule_ids
        return RouteResult(
            decision=scope.decision,
            scope=scope,
            rules=outcome.hits,
            diagnostics=RouteDiagnostics(
                model_used=False,
                confidence=None,
                notes=list(outcome.notes),
            ),
        )

    def build_prompt(self, input: RoutingInput) -> RoutePrompt:
        enriched = self.enrich(input)
        return build_route_prompt(
            enriched,
            catalog=self.catalog,
            catalog_hit_limit=self.catalog_hit_limit,
        )

    # ------------------------------------------------------------ 模型层

    async def decide_with_model(
        self, input: RoutingInput, *, model: _RouteModel | None = None
    ) -> RouteResult:
        """模型补充路由：模型输出只做补充，硬规则仍然由规则层掌权。"""

        active_model = model or self.model
        base = self.decide(input)
        if active_model is None:
            base.diagnostics.notes.append("未注入模型，使用规则层结论")
            return base

        prompt = self.build_prompt(input)
        try:
            response = await active_model.complete(prompt.messages)
            model_decision, diagnostics = parse_route_decision(
                _response_content(response), question=input.question
            )
        except RouteParseError as exc:
            base.diagnostics.notes.append(f"模型结论不可用，回退规则层：{exc}")
            return base

        merged = self._merge(base, model_decision)
        scope = check_scope(
            merged,
            catalog=self.catalog,
            input=self.enrich(input),
            text_lookup=self.text_lookup,
        )
        scope.rules = base.scope.rules
        diagnostics.notes = list(base.diagnostics.notes)
        return RouteResult(
            decision=scope.decision,
            scope=scope,
            rules=base.rules,
            diagnostics=diagnostics,
        )

    def _merge(self, base: RouteResult, model_decision: RouteDecision) -> RouteDecision:
        """合并规则层与模型结论：硬规则优先，其余以模型补充。"""

        rule_decision = base.decision
        rule_ids = set(base.scope.rules)
        merged = model_decision

        # 产品 / 概览问题（R7）不接受模型的检索要求。
        if "R7" in rule_ids:
            merged = adjust_route_decision(
                merged,
                intent=rule_decision.intent,
                needs_knowledge=False,
                needs_web=False,
                needs_general=False,
                needs_history=False,
                freshness=Freshness.NONE,
            )
        # 时效要求（R5）必须联网。
        if "R5" in rule_ids:
            merged = adjust_route_decision(
                merged, needs_web=True, freshness=rule_decision.freshness
            )
        # 内部引用（R1/R2/R3）必须查资料。
        if rule_ids & {"R1", "R2", "R3"}:
            merged = adjust_route_decision(merged, needs_knowledge=True)
        # 回忆讨论（R8）必须查历史。
        if "R8" in rule_ids:
            merged = adjust_route_decision(merged, needs_history=True)
        # 空知识库（R9）保持规则层的资料情况说明。
        if "R9" in rule_ids and rule_decision.needs_knowledge:
            merged = adjust_route_decision(merged, needs_knowledge=True)
        # 规则层识别出的待确认术语必须保留（模型漏报不算通过）。
        if rule_decision.unresolved_terms:
            merged = adjust_route_decision(
                merged,
                unresolved_terms=_merge_unique(
                    merged.unresolved_terms, rule_decision.unresolved_terms
                ),
            )
        if not merged.reason_summary:
            merged = adjust_route_decision(merged, reason_summary=rule_decision.reason_summary)
        return merged


def _response_content(response: Any) -> Any:
    """从模型响应里取正文（兼容 str / dict / 带 content 属性的对象）。"""

    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        return response.get("content") or ""
    return getattr(response, "content", "") or ""


def route_question(
    input: RoutingInput,
    *,
    agent: IntentAgent | None = None,
    catalog: KnowledgeCatalog | None = None,
    text_lookup: Callable[[str], str] | None = None,
) -> RouteDecision:
    """离线规则层路由入口（图节点与离线评测使用；需要模型时用 ``IntentAgent``）。"""

    active = agent or IntentAgent(catalog=catalog, text_lookup=text_lookup)
    return active.decide(input).decision


# =====================================================================
# 7. route.decided 事件（契约由 T00b 冻结，本模块只按模型构造）
# =====================================================================


def route_decided_payload(decision: RouteDecision) -> dict[str, Any]:
    """``route.decided`` 的 payload：形状固定为 ``{"decision": <RouteDecision>}``。"""

    return {"decision": decision.model_dump(mode="json")}


def build_route_decided_event(
    *,
    run_id: str,
    event_seq: int,
    decision: RouteDecision,
    execution_id: str = "",
    task_id: str | None = None,
    created_at: str = "",
) -> StoredEvent:
    """按冻结契约构造 ``route.decided`` 事件（多一个字段就会抛协议错误）。"""

    return build_event(
        run_id=run_id,
        event_seq=event_seq,
        type=EventType.ROUTE_DECIDED,
        payload=route_decided_payload(decision),
        execution_id=execution_id,
        task_id=task_id,
        created_at=created_at,
    )
