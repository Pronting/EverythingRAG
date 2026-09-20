"""研究循环：RAG / General / Web 子 Agent 与证据收集（PRD 7）。Owner: T07。

职责：

- RAG 循环严格按 PRD 7.2 六步执行，并对每一步留下可观察记录
  （:class:`ResearchStep`，六步顺序见 :data:`RESEARCH_STEPS`）：
  生成检索查询 → 混合检索并评价相关性 → 读取对应章节或关联原文 → 记录已能回答与仍缺
  内容 → 缺内部定义时继续本地读取、需公开原理/最新版本/外部方法时向 Planner 提 Web
  子任务 → 返回证据 / 结论 / 限制条件 / 未解决问题。
- 每轮研究更新 PRD 7.4 的六个字段（``requirement_coverage`` / ``evidence_ids`` /
  ``open_questions`` / ``conflicts`` / ``next_action`` / ``expected_gain``）。
- 进入最终核对的四个条件由 :func:`evaluate_completion` 判定；持续拿到相同资料且无新线索
  时以资料不足结束（``completed_with_gaps``）并说明未解决部分。
- Web 分支按 PRD 7.3 实现（Search 线索 + Extract 原文的形状由 T04 工具层提供）。
- 工具协议由 :mod:`app.agent.tools` 拥有，本模块**只消费** ``ToolRegistry.execute``；
  通用解释通过注入的 :data:`GeneralModel` 调用，离线可测。

默认值：``context_top_k=8``（PRD 7.2 / 18.1）；并发预算由
:mod:`app.agent.planner` 的 ``ConcurrencyBudget`` 管理；本版**不设**总时长与业务搜索
轮数上限（PRD 7.4），``max_rounds`` 默认 ``None``，仅供调用方做运行保护。

证据口径（PRD 8.1）：来源身份 + 内容 hash 决定证据身份，文档版本作为可补齐的元数据
（同来源同内容视为一条证据，读取原文后补上 ``document_revision``）；``supported_requirements``
由程序按「要求文本与证据正文的词元支持度」判定，而不是「本轮查了什么就都算支持」。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final
from zoneinfo import ZoneInfo

from pydantic import Field, ValidationError

from app.agent.contracts import (
    DEFAULT_CONTEXT_TOP_K,
    AgentError,
    AgentKind,
    ContractModel,
    Evidence,
    EvidenceSourceType,
    Freshness,
    ResearchTask,
    RouteDecision,
    TaskStatus,
    WebPageEvidence,
    citation_prefix,
    utc_now_iso,
)
from app.agent.planner import PRECISION_MARKERS
from app.agent.tools import (
    ToolContext,
    ToolError,
    ToolErrorCode,
    ToolName,
    ToolPermissionError,
    ToolRegistry,
    ToolResult,
    ToolStatus,
    ToolUnavailableError,
    build_default_registry,
)

__all__ = [
    "CONDITION_AMBIGUITY",
    "CONDITION_DIFFERENCES",
    "CONDITION_REQUIREMENTS",
    "CONDITION_SOURCED",
    "DEFAULT_TOP_K",
    "DEFAULT_WEB_MAX_PAGES",
    "DEFAULT_WEB_MAX_RESULTS",
    "GENERAL_EVIDENCE_LIMITATION",
    "MAX_STALLED_ROUNDS",
    "PROPOSAL_MARKERS",
    "RAG_SIMILARITY_FLOOR",
    "RESEARCH_STEPS",
    "SUPPORT_RATIO",
    "WEB_EVIDENCE_LIMITATION",
    "BaseResearcher",
    "CitationMismatchError",
    "CompletionAssessment",
    "ConclusionBasis",
    "GeneralModel",
    "GeneralResearcher",
    "IndependentSourceReport",
    "MaterialGap",
    "RagResearcher",
    "RequirementState",
    "ResearchConclusion",
    "ResearchNextAction",
    "ResearchNotSupportedError",
    "ResearchProgress",
    "ResearchRequest",
    "ResearchResult",
    "ResearchStep",
    "ResearchStepName",
    "WebPageContractError",
    "WebPageEvidence",
    "WebPageReading",
    "WebResearcher",
    "attribute_requirements",
    "content_hash_of",
    "dedupe_evidence",
    "detect_numeric_conflicts",
    "evaluate_completion",
    "evidence_id_for",
    "evidence_index",
    "independent_source_report",
    "make_evidence",
    "numeric_claims",
    "research_task",
    "researcher_for",
    "resolve_date_range",
    "text_supports",
]

#: 每次知识检索的候选数起点（PRD 7.2 / 18.1）。
DEFAULT_TOP_K: Final[int] = DEFAULT_CONTEXT_TOP_K

#: 片段相关度门控（低于该值只作为弱候选，不作为依据；PRD 7.2 第 2 步）。
RAG_SIMILARITY_FLOOR: Final[float] = 0.35

#: 「要求文本 ↔ 证据正文」支持度阈值（用于 ``supported_requirements`` 判定）。
SUPPORT_RATIO: Final[float] = 0.2

#: 连续拿到相同资料且无新线索的轮数上限（PRD 7.4：以资料不足结束）。
MAX_STALLED_ROUNDS: Final[int] = 2

#: Web 分支默认读取的网页数与搜索结果数（PRD 7.3）。
DEFAULT_WEB_MAX_PAGES: Final[int] = 3
DEFAULT_WEB_MAX_RESULTS: Final[int] = 5

#: 通用解释与建议的证据限制说明（PRD 8.1：不得生成文档引用编号）。
GENERAL_EVIDENCE_LIMITATION: Final[str] = (
    "通用解释/设计建议，不是本工作区文档证据，不生成文档引用编号（PRD 8.1）"
)
WEB_EVIDENCE_LIMITATION: Final[str] = "网页材料，日期含义以原文为准（PRD 7.3）"

#: 建议类内容的判定词：命中则标记为 ``proposal``（PRD 8.1）。
PROPOSAL_MARKERS: Final[tuple[str, ...]] = (
    "建议",
    "推荐",
    "可以采用",
    "最佳实践",
    "应该",
    "应当",
)

_NUMBER_CLAIM: Final[re.Pattern[str]] = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|ms|毫秒|秒|s|分钟|min|小时|h|MB|GB|KB|TB|QPS|qps|万|亿|倍|次)"
)
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"[。！？!?；;\n]")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_TERM_CANDIDATE: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_\-.]{2,}|\d{2,}")

#: 进入最终核对的四条件名称（PRD 7.4），供事件与测试引用。
CONDITION_REQUIREMENTS: Final[str] = "requirements_covered"
CONDITION_SOURCED: Final[str] = "key_facts_sourced"
CONDITION_DIFFERENCES: Final[str] = "differences_handled"
CONDITION_AMBIGUITY: Final[str] = "no_blocking_ambiguity"


def _now() -> str:
    return utc_now_iso()


# =====================================================================
# 1. 枚举：六步、下一步动作、要求状态、结论依据
# =====================================================================


class ResearchStepName(StrEnum):
    """PRD 7.2 六步（顺序即执行顺序，见 :data:`RESEARCH_STEPS`）。"""

    QUERIES = "queries"
    RETRIEVAL = "retrieval"
    READ_SOURCE = "read_source"
    RECORD = "record"
    NEXT_ACTION = "next_action"
    RETURN = "return"


#: PRD 7.2 的六步顺序（测试按此断言「每步可观察」）。
RESEARCH_STEPS: Final[tuple[ResearchStepName, ...]] = (
    ResearchStepName.QUERIES,
    ResearchStepName.RETRIEVAL,
    ResearchStepName.READ_SOURCE,
    ResearchStepName.RECORD,
    ResearchStepName.NEXT_ACTION,
    ResearchStepName.RETURN,
)


class ResearchNextAction(StrEnum):
    """一轮研究结束时的下一步动作（PRD 7.2 第 5 步 / 7.4）。"""

    CONTINUE_LOCAL = "continue_local"
    WEB_SUBTASK = "web_subtask"
    FINALIZE = "finalize"
    INSUFFICIENT = "insufficient"
    CLARIFY = "clarify"


class RequirementState(StrEnum):
    """一项用户要求当前的覆盖状态（PRD 7.4 ``requirement_coverage``）。"""

    ANSWERED = "answered"
    PARTIAL = "partial"
    MISSING = "missing"
    CONFLICT = "conflict"


class ConclusionBasis(StrEnum):
    """结论的依据类型（PRD 8.1/8.2：关键事实必须有来源，缺口要显式标记）。"""

    SUPPORTED = "supported"
    GENERAL = "general"
    GAP = "gap"


class ResearchNotSupportedError(ValueError):
    """该 Agent 职责没有对应研究分支（调用方用法错误，应显式失败）。"""


class CitationMismatchError(ValueError):
    """通用解释/建议不得生成文档引用编号（PRD 8.1）。"""


class WebPageContractError(ValueError):
    """网页证据 payload 与共享契约 ``WebPageEvidence`` 不一致（字段漂移）。

    §6 D32 之后网页证据字段只有一个来源：工具层字段改名 / 新增会让
    ``extra="forbid"`` 报错，本错误把它转成研究层的显式失败，避免静默丢字段。
    """


# =====================================================================
# 2. 可观察记录与每轮进展（PRD 7.4）
# =====================================================================


class ResearchStep(ContractModel):
    """六步中的一步（可直接投影成 ``agent.*`` / ``tool.*`` 事件）。"""

    name: ResearchStepName
    round: int = Field(default=1, ge=1)
    detail: str = ""
    queries: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: str = "ok"
    at: str = Field(default_factory=_now)


class ResearchProgress(ContractModel):
    """一轮研究后的进展（PRD 7.4 六个字段 + 停滞计数与六步记录）。

    ``round=0`` 表示尚未开始第一轮（循环每次先把轮次加一），因此下界是 ``0``。
    """

    task_id: str
    round: int = Field(default=1, ge=0)
    requirement_coverage: dict[str, RequirementState] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    next_action: ResearchNextAction = ResearchNextAction.CONTINUE_LOCAL
    expected_gain: str = ""
    same_material_rounds: int = Field(default=0, ge=0)
    material_fingerprint: str = ""
    steps: list[ResearchStep] = Field(default_factory=list)


class ResearchConclusion(ContractModel):
    """子 Agent 的一条结论：**同时携带证据 ID**（PRD 8.1 末句）。"""

    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    basis: ConclusionBasis = ConclusionBasis.SUPPORTED


class MaterialGap(ContractModel):
    """资料缺口（PRD 8.3：已完成但资料不足时展示缺少的依据与影响）。"""

    requirement_ids: list[str] = Field(default_factory=list)
    missing: str = ""
    impact: str = ""


class CompletionAssessment(ContractModel):
    """PRD 7.4 的四个进入最终核对条件。"""

    can_finalize: bool = False
    conditions: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ResearchRequest(ContractModel):
    """一次研究执行的输入（runtime 组装：任务 + 上下文 + 时效范围）。"""

    task: ResearchTask
    original_question: str = ""
    resolved_question: str = ""
    requirement_texts: dict[str, str] = Field(default_factory=dict)
    decision: RouteDecision | None = None
    #: 用户时区下的日期范围（PRD 7.3「今天新闻」），由 :func:`resolve_date_range` 生成。
    date_range: tuple[str, str] | None = None

    def requirement_ids(self) -> list[str]:
        return list(self.task.requirement_ids)

    def requirement_text(self, requirement_id: str) -> str:
        return self.requirement_texts.get(requirement_id, self.task.question)


class ResearchResult(ContractModel):
    """一次研究任务的产出（PRD 7.2 第 6 步 / 7.4）。"""

    task_id: str
    agent_kind: AgentKind
    status: TaskStatus = TaskStatus.COMPLETED
    evidence: list[Evidence] = Field(default_factory=list)
    conclusions: list[ResearchConclusion] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    web_requests: list[str] = Field(default_factory=list)
    gaps: list[MaterialGap] = Field(default_factory=list)
    progress: ResearchProgress
    stop_reason: str = "completed"
    rounds: int = Field(default=1, ge=1)
    error: AgentError | None = None

    def evidence_ids(self) -> list[str]:
        return [item.evidence_id for item in self.evidence]

    @property
    def is_failure(self) -> bool:
        return self.status is TaskStatus.FAILED


# =====================================================================
# 3. 证据组装、去重与要求归属（PRD 8.1）
# =====================================================================


def content_hash_of(text: str) -> str:
    """内容 hash（UTF-8，可复算）。"""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def evidence_id_for(
    source_type: EvidenceSourceType, source_ref: str, content_hash: str
) -> str:
    """稳定证据 ID：来源身份 + 内容 hash（PRD 8.1 的去重口径）。"""

    payload = "\x1f".join((str(source_type.value), str(source_ref), str(content_hash)))
    return "ev-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def make_evidence(
    *,
    workspace_id: str,
    run_id: str,
    task_id: str | None,
    source_type: EvidenceSourceType,
    source_ref: str,
    content: str,
    document_revision: str = "",
    retrieved_at: str = "",
    supported_requirements: Sequence[str] = (),
    limitations: Sequence[str] = (),
) -> Evidence:
    """构造一条 ``Evidence``（PRD 8.1 全 12 字段）。"""

    digest = content_hash_of(content)
    return Evidence(
        evidence_id=evidence_id_for(source_type, source_ref, digest),
        workspace_id=workspace_id,
        run_id=run_id,
        task_id=task_id,
        source_type=source_type,
        source_ref=str(source_ref),
        content=str(content),
        content_hash=digest,
        document_revision=str(document_revision or ""),
        retrieved_at=retrieved_at or _now(),
        supported_requirements=list(dict.fromkeys(supported_requirements)),
        limitations=list(dict.fromkeys(limitations)),
    )


def dedupe_evidence(evidences: Sequence[Evidence]) -> list[Evidence]:
    """按来源身份 + 内容 hash 去重；合并要求与限制，补齐已知文档版本（PRD 8.1）。"""

    merged: dict[str, Evidence] = {}
    for item in evidences:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item.model_copy(deep=True)
            continue
        existing.supported_requirements = list(
            dict.fromkeys([*existing.supported_requirements, *item.supported_requirements])
        )
        existing.limitations = list(dict.fromkeys([*existing.limitations, *item.limitations]))
        if not existing.document_revision and item.document_revision:
            existing.document_revision = item.document_revision
    return list(merged.values())


def evidence_index(evidences: Sequence[Evidence]) -> dict[str, Evidence]:
    """证据 ID → 证据（Synthesis / Verifier 的查找表）。"""

    return {item.evidence_id: item for item in evidences}


def support_tokens(text: str) -> list[str]:
    """要求文本 → 支持度词元（中文二元组 + 拉丁词 + 数字）。"""

    cleaned = _WHITESPACE.sub("", str(text or "").casefold())
    tokens = [match.group(0) for match in _TERM_CANDIDATE.finditer(cleaned)]
    cjk = "".join(ch for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    tokens.extend(cjk[index : index + 2] for index in range(len(cjk) - 1))
    return list(dict.fromkeys(tokens))


def text_supports(requirement_text: str, content: str) -> bool:
    """证据正文是否支持该要求：词元命中比例达到 :data:`SUPPORT_RATIO` 或命中 >= 3 个。"""

    tokens = support_tokens(requirement_text)
    if not tokens:
        return False
    haystack = str(content or "").casefold()
    hits = sum(1 for token in tokens if token in haystack)
    if hits == 0:
        return False
    return hits / len(tokens) >= SUPPORT_RATIO or hits >= 3


def attribute_requirements(
    evidences: Sequence[Evidence], requirements: Mapping[str, str]
) -> list[Evidence]:
    """按词元支持度重写 ``supported_requirements``（PRD 8.1）。

    取舍：子 Agent 摘要与原文证据都携带证据 ID，但「一条证据支持哪几项要求」必须由
    程序判定（要求文本 vs 证据正文），否则「本轮查了什么就都算支持」会让覆盖率虚高，
    正是 PRD 8.2「问题覆盖 / 事实支持」两项检查要防的失效模式。
    """

    result: list[Evidence] = []
    for item in evidences:
        matched = [
            requirement_id
            for requirement_id, text in requirements.items()
            if text_supports(text, item.content)
        ]
        result.append(item.model_copy(update={"supported_requirements": matched}))
    return result


# =====================================================================
# 4. 数字与单位：冲突检测（PRD 7.4 / 8.2）
# =====================================================================


def numeric_claims(text: str) -> list[tuple[str, str]]:
    """抽取 ``(数值, 单位)`` 断言（用于数字/单位一致性核对，PRD 8.2）。"""

    return [
        (match.group("value"), match.group("unit").casefold())
        for match in _NUMBER_CLAIM.finditer(str(text or ""))
    ]


def detect_numeric_conflicts(evidences: Sequence[Evidence]) -> list[str]:
    """同一单位出现不同数值即视为来源冲突（返回可读的冲突说明）。"""

    by_unit: dict[str, dict[str, list[str]]] = {}
    for item in evidences:
        for value, unit in numeric_claims(item.content):
            by_unit.setdefault(unit, {}).setdefault(value, []).append(item.evidence_id)
    conflicts: list[str] = []
    for unit, values in sorted(by_unit.items()):
        if len(values) < 2:
            continue
        parts = [f"{value}{unit}（{', '.join(ids)}）" for value, ids in sorted(values.items())]
        conflicts.append(f"同一单位 {unit} 出现不同数值：" + "；".join(parts))
    return conflicts


# =====================================================================
# 5. 进入最终核对的四个条件（PRD 7.4）
# =====================================================================


def evaluate_completion(
    progress: ResearchProgress,
    *,
    conclusions: Sequence[ResearchConclusion],
) -> CompletionAssessment:
    """判定是否可以进入最终核对（PRD 7.4 的四个条件，逐条给出结论）。"""

    states = list(progress.requirement_coverage.values())
    covered = bool(states) and all(state is RequirementState.ANSWERED for state in states)
    unsourced = [
        conclusion.statement
        for conclusion in conclusions
        if conclusion.basis is ConclusionBasis.SUPPORTED and not conclusion.evidence_ids
    ]
    conditions = {
        CONDITION_REQUIREMENTS: covered,
        CONDITION_SOURCED: not unsourced,
        CONDITION_DIFFERENCES: not progress.conflicts,
        CONDITION_AMBIGUITY: not progress.open_questions,
    }
    missing = [name for name, ok in conditions.items() if not ok]
    reasons: list[str] = []
    if not covered:
        reasons.append("仍有用户要求未被覆盖")
    if unsourced:
        reasons.append(f"{len(unsourced)} 条关键陈述缺少来源")
    if progress.conflicts:
        reasons.append("存在未处理的数字或版本差异")
    if progress.open_questions:
        reasons.append("仍有需要继续确认的关键歧义")
    return CompletionAssessment(
        can_finalize=not missing, conditions=conditions, missing=missing, reasons=reasons
    )


# =====================================================================
# 6. Web 分支辅助（PRD 7.3）
# =====================================================================


#: 网页证据直接复用共享契约 :class:`app.agent.contracts.WebPageEvidence`（架构文档 §6
#: D32：PRD 7.3 的字段单一来源）。``WebPageReading`` 只是**薄再导出**，不再维护第二份
#: 字段定义；``read_range`` 因此保留 ``offset`` / ``length`` / ``total`` / ``complete``。
#: 注意两个不同的轴：``content_complete`` = 提供商只给了部分正文（PRD 7.3 末段），
#: ``read_range.complete`` = 本次读取窗口是否覆盖全文；两者不可互相顶替。
WebPageReading = WebPageEvidence


class IndependentSourceReport(ContractModel):
    """独立来源统计（PRD 7.3：报道转述同一新闻时按实际原始发布者计算）。"""

    sources: list[str] = Field(default_factory=list)
    merged: list[str] = Field(default_factory=list)
    independent_count: int = 0


def _domain_of(url: str) -> str:
    text = str(url or "").split("//")[-1]
    return text.split("/")[0].split("?")[0].casefold()


def independent_source_report(pages: Sequence[WebPageReading]) -> IndependentSourceReport:
    """按域名合并转述报道，返回独立来源数（PRD 7.3）。"""

    sources: list[str] = []
    merged: list[str] = []
    for page in pages:
        domain = _domain_of(page.final_url or page.url)
        if not domain:
            continue
        if domain in sources:
            merged.append(page.url or page.final_url)
            continue
        sources.append(domain)
    return IndependentSourceReport(
        sources=sources, merged=merged, independent_count=len(sources)
    )


def resolve_date_range(
    freshness: Freshness,
    *,
    now: datetime,
    timezone: str = "",
    explicit_range: tuple[str, str] | None = None,
) -> tuple[str, str] | None:
    """按**用户时区**计算日期范围（PRD 7.3：「今天新闻」用本地日期）。

    ``current`` → 用户时区里的当天；``date_range`` → 调用方给出的显式范围（没有就
    返回 ``None``：不猜测范围）；``none`` → ``None``。时区不可识别时退回本机时区。
    """

    if freshness is Freshness.NONE:
        return None
    if freshness is Freshness.DATE_RANGE:
        return explicit_range
    zone: Any = None
    if timezone:
        try:
            zone = ZoneInfo(timezone)
        except Exception:  # noqa: BLE001 -- 时区名不可识别时退回本机时区，不阻断研究
            zone = None
    local_now = now.astimezone(zone) if zone is not None else now.astimezone()
    today = local_now.date()
    return (today.isoformat(), (today + timedelta(days=1)).isoformat())


# =====================================================================
# 7. 研究分支基类
# =====================================================================

GeneralModel = Callable[[ResearchRequest], str]
"""通用解释模型入口：接收研究请求，返回解释/建议正文（可由真实 Provider 实现）。"""


class BaseResearcher:
    """研究分支公共骨架：统一的失败封装与结束步骤记录。"""

    agent_kind: AgentKind = AgentKind.RAG

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def build_request(self, task: ResearchTask, **kwargs: Any) -> ResearchRequest:
        return ResearchRequest(task=task, **kwargs)

    def _finish_steps(self, progress: ResearchProgress, result_kind: str) -> None:
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RETURN,
                round=progress.round,
                detail=f"返回证据/结论/限制条件/未解决问题（{result_kind}）",
                evidence_ids=list(progress.evidence_ids),
            )
        )

    def _failure(
        self,
        task: ResearchTask,
        progress: ResearchProgress,
        exc: Exception,
        *,
        node: str,
        error_code: str,
        resumable: bool,
    ) -> ResearchResult:
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RETURN,
                round=progress.round,
                detail=f"执行失败：{error_code}",
                status="failed",
            )
        )
        return ResearchResult(
            task_id=task.task_id,
            agent_kind=task.agent_kind,
            status=TaskStatus.FAILED,
            progress=progress,
            stop_reason=error_code,
            rounds=progress.round,
            error=AgentError(
                error_code=error_code,
                message=str(exc),
                node=node,
                resumable=resumable,
                required_action="继续该任务：修复失败原因后从本任务重新执行（PRD 7.4）",
                created_at=_now(),
            ),
        )


def _error_code_of(exc: ToolError) -> str:
    return str(exc.code)


def _resumable(exc: ToolError) -> bool:
    """服务错误与未配置可继续；结构性错误（权限/参数）不可继续。"""

    return exc.code in {ToolErrorCode.SERVICE_ERROR, ToolErrorCode.NOT_CONFIGURED}


def _failure_of(exc: ToolError) -> tuple[str, bool]:
    return _error_code_of(exc), _resumable(exc)


# =====================================================================
# 8. RAG 分支：PRD 7.2 六步循环
# =====================================================================


class RagResearcher(BaseResearcher):
    """RAG Agent：六步循环（知识检索 + 原文读取 + 历史检索）。"""

    agent_kind = AgentKind.RAG

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        context_top_k: int = DEFAULT_TOP_K,
        similarity_floor: float = RAG_SIMILARITY_FLOOR,
        max_rounds: int | None = None,
        max_stalled_rounds: int = MAX_STALLED_ROUNDS,
    ) -> None:
        super().__init__(registry)
        self.context_top_k = context_top_k
        self.similarity_floor = similarity_floor
        self.max_rounds = max_rounds  # None = 无业务轮数上限（PRD 7.4）
        self.max_stalled_rounds = max_stalled_rounds

    # ---- 第 1 步：查询构造 ----

    def initial_queries(self, request: ResearchRequest) -> list[str]:
        """保留原问题与内部术语的初始查询（PRD 7.2 第 1 步）。"""

        queries: list[str] = []
        values = [request.task.question, request.resolved_question, request.original_question]
        for value in values:
            text = _WHITESPACE.sub(" ", str(value or "")).strip()
            if text and text not in queries:
                queries.append(text)
        decision = request.decision
        if decision is not None:
            for term in [*decision.entities, *decision.unresolved_terms]:
                text = _WHITESPACE.sub(" ", str(term or "")).strip()
                if text and text not in queries:
                    queries.append(text)
        for match in _TERM_CANDIDATE.finditer(str(request.task.question).casefold()):
            term = match.group(0)
            if term not in queries:
                queries.append(term)
        return queries

    def precision_targets(self, request: ResearchRequest) -> list[str]:
        """需要读原文的精度线索（定义/数值/边界条件/方案结论，PRD 7.2 第 3 步）。"""

        text = f"{request.task.question} {' '.join(request.requirement_texts.values())}".casefold()
        return [marker for marker in PRECISION_MARKERS if marker in text]

    def wants_history(self, request: ResearchRequest) -> bool:
        decision = request.decision
        if decision is not None and decision.needs_history:
            return True
        return "历史" in (request.task.expected_evidence or "")

    # ---- 主循环 ----

    async def research(self, request: ResearchRequest, context: ToolContext) -> ResearchResult:
        task = request.task
        workspace_id = context.workspace_id
        run_id = context.run_id or ""
        requirement_ids = task.requirement_ids
        progress = ResearchProgress(task_id=task.task_id, round=0)
        evidences: list[Evidence] = []
        weak_chunks: dict[str, list[dict[str, Any]]] = {}
        documents_seen: dict[str, dict[str, Any]] = {}
        read_done: set[tuple[str, str]] = set()
        read_targets: list[tuple[str, str]] = []
        used_queries: set[str] = set()
        conclusions: list[ResearchConclusion] = []
        limitations: list[str] = []
        web_requests: list[str] = []
        queries = self.initial_queries(request)
        precision = self.precision_targets(request)
        requirement_texts = {
            requirement_id: request.requirement_text(requirement_id)
            for requirement_id in requirement_ids
        }
        status = TaskStatus.COMPLETED
        stop_reason = "completed"
        previous_fingerprint = ""
        revisions: dict[str, str] = {}

        if not requirement_ids:
            return self._failure(
                task,
                progress,
                ValueError("任务没有关联任何用户要求"),
                node=f"research:{task.task_id}",
                error_code="invalid_task",
                resumable=False,
            )

        while True:
            if self.max_rounds is not None and progress.round >= self.max_rounds:
                stop_reason = "round_cap"
                break
            progress.round += 1
            round_no = progress.round
            pending_queries = [q for q in queries if _query_key(q) not in used_queries]
            pending_reads = [
                target for target in read_targets if target not in read_done
            ]
            if not pending_queries and not pending_reads:
                stop_reason = (
                    "no_new_action" if stop_reason == "completed" else stop_reason
                )
                break

            progress.steps.append(
                ResearchStep(
                    name=ResearchStepName.QUERIES,
                    round=round_no,
                    detail=(
                        "生成检索查询并保留原问题与内部术语"
                        + (f"；精度线索 {'/'.join(precision)}" if precision else "")
                    ),
                    queries=list(pending_queries),
                )
            )

            round_evidence: list[Evidence] = []
            retrieval_detail: list[str] = []
            retrieval_tools: list[str] = []
            service_error: ToolError | None = None
            for query in pending_queries:
                used_queries.add(_query_key(query))
                try:
                    result = await self.registry.execute(
                        ToolName.SEARCH_KNOWLEDGE,
                        {"query": query, "top_k": self.context_top_k},
                        context,
                    )
                except ToolError as exc:
                    service_error = exc
                    break
                retrieval_tools.append(f"{ToolName.SEARCH_KNOWLEDGE}:{query}")
                self._collect_retrieval(
                    query,
                    result,
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task.task_id,
                    requirement_ids=requirement_ids,
                    evidences=round_evidence,
                    weak_chunks=weak_chunks,
                    documents_seen=documents_seen,
                    notes=retrieval_detail,
                )
            if service_error is not None:
                error_code, resumable = _failure_of(service_error)
                return self._failure(
                    task,
                    progress,
                    service_error,
                    node=f"research:{task.task_id}:retrieval",
                    error_code=error_code,
                    resumable=resumable,
                )

            if self.wants_history(request) and pending_queries:
                history_evidence, history_notes = await self._search_history(
                    pending_queries[0],
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task.task_id,
                    requirement_ids=requirement_ids,
                    context=context,
                )
                round_evidence.extend(history_evidence)
                retrieval_detail.extend(history_notes)
                if history_evidence:
                    retrieval_tools.append(str(ToolName.SEARCH_HISTORY))

            progress.steps.append(
                ResearchStep(
                    name=ResearchStepName.RETRIEVAL,
                    round=round_no,
                    detail="；".join(retrieval_detail) or "本轮没有待检索查询",
                    tool_calls=retrieval_tools,
                    evidence_ids=[item.evidence_id for item in round_evidence],
                )
            )

            read_evidence, read_notes, attempted = await self._read_sources(
                read_targets=pending_reads,
                allow_auto=bool(precision) and round_no == 1,
                run_id=run_id,
                workspace_id=workspace_id,
                task_id=task.task_id,
                requirement_ids=requirement_ids,
                context=context,
                weak_chunks=weak_chunks,
                documents_seen=documents_seen,
            )
            read_done.update(attempted)
            round_evidence.extend(read_evidence)
            for item in read_evidence:
                if item.document_revision:
                    revisions[str(item.source_ref).split("#", 1)[0]] = item.document_revision
            progress.steps.append(
                ResearchStep(
                    name=ResearchStepName.READ_SOURCE,
                    round=round_no,
                    detail="；".join(read_notes) or "本轮没有待读取原文",
                    evidence_ids=[item.evidence_id for item in read_evidence],
                )
            )

            evidences = dedupe_evidence([*evidences, *round_evidence])
            evidences = _propagate_document_revisions(evidences, revisions)
            evidences = attribute_requirements(evidences, requirement_texts)
            fingerprint = _material_fingerprint(evidences)
            if fingerprint == previous_fingerprint and not round_evidence:
                progress.same_material_rounds += 1
            previous_fingerprint = fingerprint
            coverage = self._coverage(requirement_ids, evidences, weak_chunks, documents_seen)
            conclusions = self._conclusions(request, evidences, coverage)
            limitations = _limitations(evidences, retrieval_detail, weak_chunks)
            progress.requirement_coverage = coverage
            progress.evidence_ids = [item.evidence_id for item in evidences]
            progress.open_questions = [
                request.requirement_text(rid)
                for rid, state in coverage.items()
                if state is not RequirementState.ANSWERED
            ]
            progress.conflicts = detect_numeric_conflicts(evidences)
            progress.material_fingerprint = fingerprint
            progress.steps.append(
                ResearchStep(
                    name=ResearchStepName.RECORD,
                    round=round_no,
                    detail=(
                        "记录已能回答与仍缺内容：覆盖 "
                        f"{sum(1 for s in coverage.values() if s is RequirementState.ANSWERED)}"
                        f"/{len(coverage)} 项要求；证据 {len(evidences)} 条；"
                        f"待确认 {len(progress.open_questions)} 项"
                    ),
                    evidence_ids=list(progress.evidence_ids),
                )
            )

            assessment = evaluate_completion(progress, conclusions=conclusions)
            next_action, expected_gain, new_reads, new_web = self._next_action(
                request=request,
                progress=progress,
                assessment=assessment,
                documents_seen=documents_seen,
                weak_chunks=weak_chunks,
                read_done=read_done,
            )
            progress.next_action = next_action
            progress.expected_gain = expected_gain
            if new_web:
                web_requests = new_web
            for target in new_reads:
                if target not in read_targets:
                    read_targets.append(target)
            progress.steps.append(
                ResearchStep(
                    name=ResearchStepName.NEXT_ACTION,
                    round=round_no,
                    detail=f"{next_action.value}：{expected_gain}",
                )
            )

            if next_action is ResearchNextAction.FINALIZE:
                stop_reason = "completed"
                break
            if progress.same_material_rounds >= self.max_stalled_rounds:
                stop_reason = "same_material"
                break
            if next_action is ResearchNextAction.WEB_SUBTASK:
                stop_reason = "web_subtask_required"
                break
            if next_action is ResearchNextAction.CLARIFY:
                stop_reason = "clarification_required"
                break
            if next_action is ResearchNextAction.INSUFFICIENT:
                stop_reason = "insufficient_material"
                break
            still_pending = [t for t in read_targets if t not in read_done] or [
                q for q in queries if _query_key(q) not in used_queries
            ]
            if not still_pending:
                stop_reason = "insufficient_material"
                break

        if stop_reason != "completed":
            status = TaskStatus.COMPLETED_WITH_GAPS
        final_evidences = attribute_requirements(
            _propagate_document_revisions(dedupe_evidence(evidences), revisions),
            requirement_texts,
        )
        coverage = self._coverage(requirement_ids, final_evidences, weak_chunks, documents_seen)
        conclusions = self._conclusions(request, final_evidences, coverage)
        progress.evidence_ids = [item.evidence_id for item in final_evidences]
        progress.requirement_coverage = coverage
        progress.open_questions = [
            request.requirement_text(rid)
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        gaps = [
            MaterialGap(
                requirement_ids=[rid],
                missing="本工作区未找到足以回答该要求的依据",
                impact="该要求的相关陈述不能作为已核实内容发布，只能作为资料缺口说明",
            )
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        self._finish_steps(progress, status.value)
        return ResearchResult(
            task_id=task.task_id,
            agent_kind=task.agent_kind,
            status=status,
            evidence=final_evidences,
            conclusions=conclusions,
            limitations=limitations,
            open_questions=list(progress.open_questions),
            web_requests=list(web_requests),
            gaps=gaps,
            progress=progress,
            stop_reason=stop_reason,
            rounds=progress.round,
        )

    # ---- 各步实现 ----

    def _collect_retrieval(
        self,
        query: str,
        result: ToolResult,
        *,
        workspace_id: str,
        run_id: str,
        task_id: str,
        requirement_ids: Sequence[str],
        evidences: list[Evidence],
        weak_chunks: dict[str, list[dict[str, Any]]],
        documents_seen: dict[str, dict[str, Any]],
        notes: list[str],
    ) -> None:
        chunks = [dict(item) for item in result.data.get("chunks", [])]
        for chunk in chunks:
            document_id = str(chunk.get("document_id") or "")
            if document_id:
                documents_seen.setdefault(document_id, chunk)
        if result.status is ToolStatus.EMPTY:
            notes.append(f"查询「{query}」没有命中片段（{result.data.get('code', 'no_results')}）")
            return
        if result.status is ToolStatus.IRRELEVANT:
            notes.append(f"查询「{query}」有候选但均未达到相关度门控，视为资料无关")
            return
        relevant = [chunk for chunk in chunks if _is_relevant(chunk, self.similarity_floor)]
        weak = [chunk for chunk in chunks if not _is_relevant(chunk, self.similarity_floor)]
        if weak:
            weak_chunks.setdefault(query, []).extend(weak)
            notes.append(f"查询「{query}」有 {len(weak)} 个低相关度候选，未作为依据")
        for chunk in relevant:
            evidences.append(
                make_evidence(
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task_id,
                    source_type=EvidenceSourceType.KNOWLEDGE,
                    source_ref=f"{chunk.get('document_id')}#{chunk.get('block_id')}",
                    content=str(chunk.get("text") or ""),
                    supported_requirements=requirement_ids,
                    limitations=["片段为文档局部内容，关键结论需按位置读取原文确认（PRD 8.1）"],
                )
            )
        notes.append(
            f"查询「{query}」命中 {len(chunks)} 个片段，其中 {len(relevant)} 个达到相关度门控"
        )

    async def _search_history(
        self,
        query: str,
        *,
        workspace_id: str,
        run_id: str,
        task_id: str,
        requirement_ids: Sequence[str],
        context: ToolContext,
    ) -> tuple[list[Evidence], list[str]]:
        try:
            result = await self.registry.execute(
                ToolName.SEARCH_HISTORY, {"query": query, "limit": 5}, context
            )
        except ToolUnavailableError as exc:
            return [], [f"历史检索不可用：{exc}"]
        except ToolPermissionError as exc:
            return [], [f"历史检索被拒绝：{exc}"]
        messages = [dict(item) for item in result.data.get("messages", [])]
        if not messages:
            return [], ["历史检索没有命中消息"]
        evidences = [
            make_evidence(
                workspace_id=workspace_id,
                run_id=run_id,
                task_id=task_id,
                source_type=EvidenceSourceType.HISTORY,
                source_ref=f"{message.get('conversation_id')}#{message.get('message_id')}",
                content=str(message.get("excerpt") or ""),
                supported_requirements=requirement_ids,
                limitations=["历史讨论片段，需结合当时上下文理解（PRD 8.1）"],
            )
            for message in messages
        ]
        return evidences, [f"历史检索命中 {len(evidences)} 条消息"]

    async def _read_sources(
        self,
        *,
        read_targets: Sequence[tuple[str, str]],
        allow_auto: bool,
        run_id: str,
        workspace_id: str,
        task_id: str,
        requirement_ids: Sequence[str],
        context: ToolContext,
        weak_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
        documents_seen: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[Evidence], list[str], list[tuple[str, str]]]:
        evidences: list[Evidence] = []
        notes: list[str] = []
        attempted: list[tuple[str, str]] = []
        targets = list(read_targets)
        if not targets and allow_auto:
            targets = self._auto_read_targets(weak_chunks, documents_seen)
        for document_id, heading in targets:
            arguments: dict[str, Any] = {"document_id": document_id}
            if heading:
                arguments["heading"] = heading
            attempted.append((document_id, heading))
            try:
                result = await self.registry.execute(ToolName.READ_DOCUMENT, arguments, context)
            except ToolError as exc:
                notes.append(f"读取原文失败（{document_id}）：{exc.message}")
                continue
            if result.status is not ToolStatus.OK:
                code = result.data.get("code", result.status.value)
                notes.append(f"读取原文未返回正文（{document_id}）：{code}")
                continue
            window = str(result.data.get("text") or "")
            total = int(result.data.get("total_chars") or 0)
            offset = int(result.data.get("offset") or 0)
            end = offset + len(window)
            limitations = []
            if end < total:
                limitations.append(f"仅读取原文窗口 {offset}-{end}/{total}，其余位置仍需读取")
            evidences.append(
                make_evidence(
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task_id,
                    source_type=EvidenceSourceType.KNOWLEDGE,
                    source_ref=f"{document_id}#read:{offset}",
                    content=window,
                    document_revision=str(result.data.get("document_revision") or ""),
                    supported_requirements=requirement_ids,
                    limitations=limitations,
                )
            )
            notes.append(f"读取原文 {document_id}（{len(window)} 字符，窗口 {offset}-{end}/{total}）")
        return evidences, notes, attempted

    def _auto_read_targets(
        self,
        weak_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
        documents_seen: Mapping[str, Mapping[str, Any]],
    ) -> list[tuple[str, str]]:
        """精度线索存在时读最相关文档的对应章节（PRD 7.2 第 3 步）。"""

        targets: list[tuple[str, str]] = []
        for document_id, chunk in documents_seen.items():
            targets.append((document_id, str(chunk.get("heading_path") or "").strip()))
            if len(targets) >= 2:
                break
        for chunks in weak_chunks.values():
            for chunk in chunks:
                document_id = str(chunk.get("document_id") or "")
                if document_id and all(target[0] != document_id for target in targets):
                    targets.append(
                        (document_id, str(chunk.get("heading_path") or "").strip())
                    )
        return targets[:3]

    def _coverage(
        self,
        requirement_ids: Sequence[str],
        evidences: Sequence[Evidence],
        weak_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
        documents_seen: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, RequirementState]:
        coverage: dict[str, RequirementState] = {}
        candidates = bool(weak_chunks) or bool(documents_seen)
        for requirement_id in requirement_ids:
            covered = any(
                requirement_id in item.supported_requirements and bool(item.content.strip())
                for item in evidences
            )
            if covered:
                coverage[requirement_id] = RequirementState.ANSWERED
                continue
            coverage[requirement_id] = (
                RequirementState.PARTIAL if candidates else RequirementState.MISSING
            )
        return coverage

    def _conclusions(
        self,
        request: ResearchRequest,
        evidences: Sequence[Evidence],
        coverage: Mapping[str, RequirementState],
    ) -> list[ResearchConclusion]:
        conclusions: list[ResearchConclusion] = []
        for requirement_id in request.task.requirement_ids:
            state = coverage.get(requirement_id, RequirementState.MISSING)
            supported = [
                item
                for item in evidences
                if requirement_id in item.supported_requirements and item.content.strip()
            ]
            if supported and state is RequirementState.ANSWERED:
                conclusions.append(
                    ResearchConclusion(
                        statement=_evidence_statement(supported[0]),
                        evidence_ids=[item.evidence_id for item in supported],
                        requirement_ids=[requirement_id],
                        basis=ConclusionBasis.SUPPORTED,
                    )
                )
                continue
            reason = (
                "仅有低相关度候选或读取窗口，缺少可直接引用的依据"
                if state is RequirementState.PARTIAL
                else "本工作区未找到可用依据"
            )
            conclusions.append(
                ResearchConclusion(
                    statement=f"{request.requirement_text(requirement_id)}：{reason}",
                    evidence_ids=[],
                    requirement_ids=[requirement_id],
                    basis=ConclusionBasis.GAP,
                )
            )
        return conclusions

    def _next_action(
        self,
        *,
        request: ResearchRequest,
        progress: ResearchProgress,
        assessment: CompletionAssessment,
        documents_seen: Mapping[str, Mapping[str, Any]],
        weak_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
        read_done: set[tuple[str, str]],
    ) -> tuple[ResearchNextAction, str, list[tuple[str, str]], list[str]]:
        if assessment.can_finalize:
            return (ResearchNextAction.FINALIZE, "四项完成条件均已满足，进入最终核对", [], [])
        missing = [
            rid
            for rid, state in progress.requirement_coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        unread = [
            target
            for target in self._auto_read_targets(weak_chunks, documents_seen)
            if target not in read_done
        ]
        if unread:
            return (
                ResearchNextAction.CONTINUE_LOCAL,
                f"读取 {len(unread)} 处关联原文以补足缺口：" + ", ".join(t[0] for t in unread),
                unread,
                [],
            )
        needs_web = bool(
            request.decision is not None
            and (
                request.decision.needs_web
                or request.decision.freshness is not Freshness.NONE
            )
        )
        if needs_web and missing:
            return (
                ResearchNextAction.WEB_SUBTASK,
                "缺少内部定义，且问题需要公开原理/最新版本：向 Planner 提出 Web 子任务",
                [],
                [request.requirement_text(rid) for rid in missing],
            )
        if missing:
            return (
                ResearchNextAction.INSUFFICIENT,
                "继续本地查询无法获得新线索，以资料不足结束并说明未解决部分",
                [],
                [],
            )
        return (ResearchNextAction.CLARIFY, "仍有关键歧义需要用户确认后才能继续", [], [])


# =====================================================================
# 9. Web 分支（PRD 7.3）
# =====================================================================


class WebResearcher(BaseResearcher):
    """Web Agent：Search 线索 + Extract 原文，逐项检查结果（PRD 7.3）。"""

    agent_kind = AgentKind.WEB

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        max_results: int = DEFAULT_WEB_MAX_RESULTS,
        max_pages: int = DEFAULT_WEB_MAX_PAGES,
    ) -> None:
        super().__init__(registry)
        self.max_results = max_results
        self.max_pages = max_pages

    def search_arguments(self, request: ResearchRequest) -> dict[str, Any]:
        """Search 参数（PRD 7.3：topic / 日期范围 / include_published_date）。"""

        freshness = (
            request.decision.freshness if request.decision is not None else Freshness.NONE
        )
        arguments: dict[str, Any] = {
            "query": request.task.question,
            "max_results": self.max_results,
            "include_published_date": True,
        }
        if freshness is not Freshness.NONE:
            arguments["topic"] = "news"
        if request.date_range is not None:
            arguments["start_date"], arguments["end_date"] = request.date_range
        return arguments

    async def research(self, request: ResearchRequest, context: ToolContext) -> ResearchResult:
        task = request.task
        workspace_id = context.workspace_id
        run_id = context.run_id or ""
        requirement_ids = task.requirement_ids
        requirement_texts = {
            requirement_id: request.requirement_text(requirement_id)
            for requirement_id in requirement_ids
        }
        progress = ResearchProgress(task_id=task.task_id, round=1)
        arguments = self.search_arguments(request)
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.QUERIES,
                round=1,
                detail="构造联网检索参数（topic / 日期范围 / include_published_date）",
                queries=[str(arguments["query"])],
            )
        )
        try:
            search_result = await self.registry.execute(ToolName.WEB_SEARCH, arguments, context)
        except ToolError as exc:
            error_code, resumable = _failure_of(exc)
            return self._failure(
                task,
                progress,
                exc,
                node=f"research:{task.task_id}:web_search",
                error_code=error_code,
                resumable=resumable,
            )

        hits = [dict(item) for item in search_result.data.get("results", [])]
        notes = [f"联网搜索命中 {len(hits)} 条线索"]
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RETRIEVAL,
                round=1,
                detail="；".join(notes),
                tool_calls=[str(ToolName.WEB_SEARCH)],
            )
        )

        evidences: list[Evidence] = []
        limitations: list[str] = [WEB_EVIDENCE_LIMITATION]
        pages: list[WebPageReading] = []
        urls = [str(hit.get("url") or "") for hit in hits if hit.get("url")][: self.max_pages]
        read_tool_used = False
        if urls:
            try:
                read_result = await self.registry.execute(
                    ToolName.READ_WEB_PAGE, {"urls": urls}, context
                )
            except ToolError as exc:
                notes.append(f"网页原文读取失败：{exc.message}")
                limitations.append(f"网页原文读取失败：{exc.message}")
            else:
                read_tool_used = True
                try:
                    pages, page_notes, page_evidence = self._pages(
                        read_result,
                        workspace_id=workspace_id,
                        run_id=run_id,
                        task_id=task.task_id,
                        requirement_ids=requirement_ids,
                    )
                except WebPageContractError as exc:
                    # 工具层字段与共享契约漂移：显式失败，不静默丢字段（§6 D32）。
                    return self._failure(
                        task,
                        progress,
                        exc,
                        node=f"research:{task.task_id}:read_web_page",
                        error_code="invalid_web_payload",
                        resumable=False,
                    )
                notes.extend(page_notes)
                evidences.extend(page_evidence)
                limitations.extend(_page_limitations(pages))

        if not evidences and hits:
            evidences.extend(
                _snippet_evidence(
                    hits,
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task.task_id,
                    requirement_ids=requirement_ids,
                    read_tool_used=read_tool_used,
                )
            )
            if evidences:
                limitations.append("本轮结论基于搜索摘要，覆盖率标记为部分覆盖")
                notes.append("退回使用搜索摘要（未读取网页原文）")

        evidences = attribute_requirements(dedupe_evidence(evidences), requirement_texts)
        coverage: dict[str, RequirementState] = {}
        for requirement_id in requirement_ids:
            supported = any(
                requirement_id in item.supported_requirements and item.content.strip()
                for item in evidences
            )
            if supported and read_tool_used:
                coverage[requirement_id] = RequirementState.ANSWERED
            elif supported:
                coverage[requirement_id] = RequirementState.PARTIAL
            else:
                coverage[requirement_id] = RequirementState.MISSING
        report = independent_source_report(pages)
        if report.merged:
            notes.append(
                f"合并 {len(report.merged)} 条转述来源，独立来源 {report.independent_count} 个"
            )
        conclusions = _web_conclusions(request, evidences, pages)
        progress.requirement_coverage = coverage
        progress.evidence_ids = [item.evidence_id for item in evidences]
        progress.open_questions = [
            request.requirement_text(rid)
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        progress.conflicts = detect_numeric_conflicts(evidences)
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.READ_SOURCE,
                round=1,
                detail="；".join(notes),
                tool_calls=[str(ToolName.READ_WEB_PAGE)] if read_tool_used else [],
                evidence_ids=list(progress.evidence_ids),
            )
        )
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RECORD,
                round=1,
                detail=(
                    "记录已能回答与仍缺内容：覆盖 "
                    f"{sum(1 for s in coverage.values() if s is RequirementState.ANSWERED)}"
                    f"/{len(coverage)} 项要求；证据 {len(evidences)} 条"
                ),
                evidence_ids=list(progress.evidence_ids),
            )
        )
        assessment = evaluate_completion(progress, conclusions=conclusions)
        if assessment.can_finalize:
            progress.next_action = ResearchNextAction.FINALIZE
            progress.expected_gain = "四项完成条件均已满足，进入最终核对"
            status = TaskStatus.COMPLETED
            stop_reason = "completed"
        else:
            progress.next_action = ResearchNextAction.INSUFFICIENT
            progress.expected_gain = "公开来源不足以覆盖全部要求，按缺口结束"
            status = TaskStatus.COMPLETED_WITH_GAPS
            stop_reason = "insufficient_material"
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.NEXT_ACTION,
                round=1,
                detail=f"{progress.next_action.value}：{progress.expected_gain}",
            )
        )
        self._finish_steps(progress, status.value)
        gaps = [
            MaterialGap(
                requirement_ids=[rid],
                missing="公开来源未提供可核对的内容",
                impact="该要求不能作为已核实内容发布",
            )
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        return ResearchResult(
            task_id=task.task_id,
            agent_kind=task.agent_kind,
            status=status,
            evidence=evidences,
            conclusions=conclusions,
            limitations=limitations,
            open_questions=list(progress.open_questions),
            gaps=gaps,
            progress=progress,
            stop_reason=stop_reason,
            rounds=1,
        )

    def _pages(
        self,
        result: ToolResult,
        *,
        workspace_id: str,
        run_id: str,
        task_id: str,
        requirement_ids: Sequence[str],
    ) -> tuple[list[WebPageReading], list[str], list[Evidence]]:
        pages: list[WebPageReading] = []
        notes: list[str] = []
        evidences: list[Evidence] = []
        for raw in result.data.get("pages", []):
            # 一步构造：工具 payload 就是 ``WebPageEvidence.model_dump()``，
            # ``extra="forbid"`` 会在提供商字段漂移时显式报错，而不是静默丢字段。
            try:
                page = WebPageEvidence.model_validate(raw)
            except ValidationError as exc:
                raise WebPageContractError(
                    f"网页证据字段与共享契约不一致（§6 D32）：{exc.error_count()} 处不合法"
                ) from exc
            pages.append(page)
            if not page.content.strip():
                continue
            limitations = [WEB_EVIDENCE_LIMITATION]
            if not page.content_complete:
                limitations.append("提供商只返回部分正文（content_complete=false），可支持范围受限")
            if page.read_range.total and not page.read_range.complete:
                limitations.append(
                    "本次读取窗口未覆盖全文（read_range.complete=false），其余位置仍需读取"
                )
            if page.artifact_id:
                limitations.append(f"完整正文保存在任务资料 artifact {page.artifact_id}")
            evidences.append(
                make_evidence(
                    workspace_id=workspace_id,
                    run_id=run_id,
                    task_id=task_id,
                    source_type=EvidenceSourceType.WEB,
                    source_ref=page.url or page.final_url,
                    content=page.content,
                    supported_requirements=requirement_ids,
                    limitations=limitations,
                )
            )
        for item in result.data.get("failed", []):
            notes.append(f"逐项读取失败：{item.get('url')}（{item.get('error')}）")
        if pages:
            complete = sum(1 for page in pages if page.content_complete)
            notes.append(f"读取 {len(pages)} 个网页，其中 {complete} 个正文完整")
        return pages, notes, evidences


def _snippet_evidence(
    hits: Sequence[Mapping[str, Any]],
    *,
    workspace_id: str,
    run_id: str,
    task_id: str,
    requirement_ids: Sequence[str],
    read_tool_used: bool,
) -> list[Evidence]:
    """没有读到原文时退回搜索摘要，并显式标注「不能作为依据」（PRD 7.3）。"""

    evidences: list[Evidence] = []
    for hit in hits:
        url = str(hit.get("url") or "")
        snippet = str(hit.get("snippet") or "")
        if not url or not snippet:
            continue
        limitations = ["仅搜索摘要，未读取网页原文；读取原文后才能作为依据"]
        if not read_tool_used:
            limitations.append("网页原文读取工具未执行")
        evidences.append(
            make_evidence(
                workspace_id=workspace_id,
                run_id=run_id,
                task_id=task_id,
                source_type=EvidenceSourceType.WEB,
                source_ref=url,
                content=snippet,
                supported_requirements=requirement_ids,
                limitations=limitations,
            )
        )
    return evidences


def _page_limitations(pages: Sequence[WebPageReading]) -> list[str]:
    return [
        f"来源 {page.url} 标注发布时间 {page.published_at}（以原文为准）"
        for page in pages
        if page.published_at
    ]


def _web_conclusions(
    request: ResearchRequest,
    evidences: Sequence[Evidence],
    pages: Sequence[WebPageReading],
) -> list[ResearchConclusion]:
    by_url = {page.url or page.final_url: page for page in pages}
    conclusions: list[ResearchConclusion] = []
    for requirement_id in request.task.requirement_ids:
        supported = [
            item
            for item in evidences
            if requirement_id in item.supported_requirements and item.content.strip()
        ]
        if not supported:
            conclusions.append(
                ResearchConclusion(
                    statement=f"{request.requirement_text(requirement_id)}：公开来源没有可核对内容",
                    requirement_ids=[requirement_id],
                    basis=ConclusionBasis.GAP,
                )
            )
            continue
        best = supported[0]
        page = by_url.get(best.source_ref)
        title = (page.title if page is not None else "") or best.source_ref
        published = (
            f"，发布时间 {page.published_at}"
            if page is not None and page.published_at
            else ""
        )
        conclusions.append(
            ResearchConclusion(
                statement=f"{title}（{best.source_ref}{published}）：{_first_sentence(best.content, 160)}",
                evidence_ids=[item.evidence_id for item in supported],
                requirement_ids=[requirement_id],
                basis=ConclusionBasis.SUPPORTED,
            )
        )
    return conclusions


# =====================================================================
# 10. General 分支（PRD 8.1：解释与建议标记 general_knowledge / proposal）
# =====================================================================

#: 要求文本出现这些词说明它涉及内部事实，通用解释不能算已覆盖（PRD 8.2 内部事实）。
_INTERNAL_MARKERS: Final[tuple[str, ...]] = (
    "我们",
    "本项目",
    "内部",
    "本工作区",
    "既有方案",
    "现有方案",
)


class GeneralResearcher(BaseResearcher):
    """General Agent：通用解释与设计建议，**不生成文档引用编号**。"""

    agent_kind = AgentKind.GENERAL

    def __init__(self, model: GeneralModel | None = None) -> None:
        super().__init__(registry=None)
        self.model = model

    async def research(self, request: ResearchRequest, context: ToolContext) -> ResearchResult:
        task = request.task
        progress = ResearchProgress(task_id=task.task_id, round=1)
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.QUERIES,
                round=1,
                detail="通用解释不检索本工作区知识正文（PRD 4.2 General Agent）",
            )
        )
        if self.model is None:
            return self._failure(
                task,
                progress,
                ToolUnavailableError(
                    "未配置通用解释模型", code=ToolErrorCode.NOT_CONFIGURED
                ),
                node=f"research:{task.task_id}:general",
                error_code=str(ToolErrorCode.NOT_CONFIGURED),
                resumable=True,
            )
        text = _clean_text(self.model(request))
        if not text:
            return self._failure(
                task,
                progress,
                ValueError("通用解释模型返回空内容"),
                node=f"research:{task.task_id}:general",
                error_code="empty_model_output",
                resumable=True,
            )
        is_proposal = any(marker in text for marker in PROPOSAL_MARKERS)
        source_type = (
            EvidenceSourceType.PROPOSAL if is_proposal else EvidenceSourceType.GENERAL_KNOWLEDGE
        )
        if citation_prefix(source_type) is not None:  # pragma: no cover - 结构性保证
            raise CitationMismatchError("通用解释与建议不得生成文档引用编号（PRD 8.1）")
        evidence = make_evidence(
            workspace_id=context.workspace_id,
            run_id=context.run_id or "",
            task_id=task.task_id,
            source_type=source_type,
            source_ref=f"general:{task.task_id}",
            content=text,
            supported_requirements=task.requirement_ids,
            limitations=[GENERAL_EVIDENCE_LIMITATION],
        )
        coverage: dict[str, RequirementState] = {}
        for requirement_id in task.requirement_ids:
            requirement_text = request.requirement_text(requirement_id)
            internal = any(marker in requirement_text for marker in _INTERNAL_MARKERS)
            coverage[requirement_id] = (
                RequirementState.PARTIAL if internal else RequirementState.ANSWERED
            )
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RETRIEVAL,
                round=1,
                detail="模型产出通用解释/建议，不检索本工作区知识正文",
            )
        )
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.READ_SOURCE,
                round=1,
                detail="无内部原文可读，通用解释只作为参考数据（PRD 8.1）",
                evidence_ids=[evidence.evidence_id],
            )
        )
        progress.evidence_ids = [evidence.evidence_id]
        progress.requirement_coverage = coverage
        progress.open_questions = [
            request.requirement_text(rid)
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.RECORD,
                round=1,
                detail="记录：通用结论标记 general_knowledge/proposal，不进入文档引用编号",
                evidence_ids=[evidence.evidence_id],
            )
        )
        conclusions = [
            ResearchConclusion(
                statement=f"{request.requirement_text(rid)}：{_first_sentence(text, 200)}",
                evidence_ids=[evidence.evidence_id],
                requirement_ids=[rid],
                basis=ConclusionBasis.GENERAL,
            )
            for rid in task.requirement_ids
        ]
        assessment = evaluate_completion(progress, conclusions=conclusions)
        progress.next_action = (
            ResearchNextAction.FINALIZE
            if assessment.can_finalize
            else ResearchNextAction.INSUFFICIENT
        )
        progress.expected_gain = (
            "通用解释已完成，进入最终核对"
            if assessment.can_finalize
            else "内部事实仍需本工作区资料，按缺口结束"
        )
        progress.steps.append(
            ResearchStep(
                name=ResearchStepName.NEXT_ACTION,
                round=1,
                detail=f"{progress.next_action.value}：{progress.expected_gain}",
            )
        )
        status = (
            TaskStatus.COMPLETED if assessment.can_finalize else TaskStatus.COMPLETED_WITH_GAPS
        )
        self._finish_steps(progress, status.value)
        gaps = [
            MaterialGap(
                requirement_ids=[rid],
                missing="通用解释不能替代本工作区的内部事实依据",
                impact="涉及内部实现/既有方案的陈述仍需本区资料支持（PRD 8.2 内部事实）",
            )
            for rid, state in coverage.items()
            if state is not RequirementState.ANSWERED
        ]
        return ResearchResult(
            task_id=task.task_id,
            agent_kind=task.agent_kind,
            status=status,
            evidence=[evidence],
            conclusions=conclusions,
            limitations=[GENERAL_EVIDENCE_LIMITATION],
            open_questions=list(progress.open_questions),
            gaps=gaps,
            progress=progress,
            stop_reason="completed" if assessment.can_finalize else "insufficient_material",
            rounds=1,
        )


# =====================================================================
# 11. 调度入口
# =====================================================================


def researcher_for(
    agent_kind: AgentKind,
    *,
    registry: ToolRegistry | None = None,
    general_model: GeneralModel | None = None,
    context_top_k: int = DEFAULT_TOP_K,
    max_rounds: int | None = None,
) -> BaseResearcher:
    """按任务职责选择研究分支（PRD 4.2 / 6.4）。"""

    if agent_kind in {AgentKind.RAG, AgentKind.SYNTHESIS}:
        return RagResearcher(registry, context_top_k=context_top_k, max_rounds=max_rounds)
    if agent_kind is AgentKind.WEB:
        return WebResearcher(registry)
    if agent_kind is AgentKind.GENERAL:
        return GeneralResearcher(model=general_model)
    raise ResearchNotSupportedError(f"没有研究分支支持 AgentKind={agent_kind}")


async def research_task(
    request: ResearchRequest,
    context: ToolContext,
    *,
    registry: ToolRegistry | None = None,
    researcher: BaseResearcher | None = None,
    general_model: GeneralModel | None = None,
    context_top_k: int = DEFAULT_TOP_K,
    max_rounds: int | None = None,
) -> ResearchResult:
    """执行一个研究任务（runtime 的调用入口：按职责自动选分支）。"""

    active = researcher or researcher_for(
        request.task.agent_kind,
        registry=registry,
        general_model=general_model,
        context_top_k=context_top_k,
        max_rounds=max_rounds,
    )
    return await active.research(request, context)


# =====================================================================
# 12. 小工具
# =====================================================================


def _propagate_document_revisions(
    evidences: Sequence[Evidence], revisions: Mapping[str, str]
) -> list[Evidence]:
    """把读取原文得到的文档版本回填到同一文档的片段证据（PRD 8.2 版本与范围）。

    片段证据本身不带版本；一旦 ``read_document`` 返回该文档的 ``document_revision``，
    就把版本补齐到同一文档的全部片段证据上，让版本核对有可比较的字段。
    """

    if not revisions:
        return list(evidences)
    patched: list[Evidence] = []
    for item in evidences:
        document_id = str(item.source_ref).split("#", 1)[0]
        revision = revisions.get(document_id)
        if revision and not item.document_revision:
            patched.append(item.model_copy(update={"document_revision": revision}))
            continue
        patched.append(item)
    return patched


def _query_key(query: str) -> str:
    return _WHITESPACE.sub(" ", str(query or "")).strip().casefold()


def _is_relevant(chunk: Mapping[str, Any], floor: float) -> bool:
    """片段相关性：有相似度就按门控；缺相似度（BM25-only）视为可用的词法命中。"""

    value = chunk.get("similarity")
    if value is None or value == "":
        return True
    try:
        return float(value) >= floor
    except (TypeError, ValueError):
        return True


def _material_fingerprint(evidences: Sequence[Evidence]) -> str:
    parts = sorted(f"{item.evidence_id}:{item.content_hash}" for item in evidences)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def _limitations(
    evidences: Sequence[Evidence],
    notes: Sequence[str],
    weak_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    limitations: list[str] = []
    for item in evidences:
        limitations.extend(item.limitations)
    if weak_chunks:
        total = sum(len(chunks) for chunks in weak_chunks.values())
        limitations.append(f"有 {total} 个低相关度候选未作为依据（PRD 7.2 第 2 步）")
    limitations.extend(note for note in notes if "失败" in note or "未达到" in note)
    return list(dict.fromkeys(limitations))


def _evidence_statement(evidence: Evidence) -> str:
    """证据 → 一句可核对的中文陈述（来源身份 + 内容首句）。"""

    head = " ".join(str(evidence.source_ref).split())
    return f"{head}：{_first_sentence(evidence.content, 160)}"


def _first_sentence(text: str, limit: int = 200) -> str:
    cleaned = _WHITESPACE.sub(" ", str(text or "")).strip()
    if not cleaned:
        return ""
    match = _SENTENCE_END.search(cleaned)
    candidate = cleaned[: match.start() + 1] if match else cleaned
    return candidate if len(candidate) <= limit else candidate[:limit] + "…"


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, Mapping):
        return _WHITESPACE.sub(" ", str(text.get("content") or "")).strip()
    return _WHITESPACE.sub(" ", str(text)).strip()
