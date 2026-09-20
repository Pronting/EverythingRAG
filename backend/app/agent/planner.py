"""Planner Agent：子问题拆分、依赖校验与并发预算（PRD 6.4）。Owner: T07。

职责：

- 由已理解的问题与来源需求生成 ``ResearchTask`` 列表（PRD 6.4 表，全部 9 个字段）。
- 完整性检查：较长输入必须提取**全部**用户要求、逐字保留原文，并把每项要求映射到
  至少一个任务（或一条澄清记录，见 :class:`ClarificationRecord`）。
- 依赖校验：依赖环、缺失任务 ID、无来源的完成声明由 Pydantic（``ResearchTask`` /
  ``validate_task_graph``）与程序检查（:func:`validate_plan`）共同拒绝。
- 并发预算：默认每个任务最多 3 个研究 Agent、应用同时最多 4 个模型请求，均可配置
  （PRD 6.4 / 18.1）。**不设**任务总时长与总搜索轮数上限。
- 相同工作区 + 相同知识版本 + 相同查询目标的重复任务复用已有结果
  （:class:`TaskResultCache`）。

与其他模块的边界：

- 任务模型的字段由 :mod:`app.agent.contracts` 冻结（只读），本模块不新增同义模型。
- 工具执行、检索、模型调用都不在本模块：这里只有**纯计划**，因此可离线单测。
- 模型提出的任务可经 ``proposed_tasks`` 注入；结构性问题一律拒绝，覆盖缺口由程序补齐
  （见 :func:`build_plan` 的说明）。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from pydantic import Field

from app.agent.contracts import (
    DEFAULT_MODEL_REQUEST_CONCURRENCY,
    DEFAULT_RESEARCH_AGENT_CONCURRENCY,
    AgentKind,
    ContractModel,
    ResearchTask,
    RouteDecision,
    TaskStatus,
    utc_now_iso,
    validate_task_graph,
)

__all__ = [
    "COMBINE_MARKERS",
    "LONG_INPUT_MIN_CHARS",
    "TASK_SOURCES",
    "ClarificationRecord",
    "ConcurrencyBudget",
    "Plan",
    "PlanSettings",
    "PlanValidationError",
    "PlannerAgent",
    "RequirementCoverage",
    "TaskResultCache",
    "TaskSource",
    "UserRequirement",
    "build_plan",
    "build_web_subtask",
    "dependency_order",
    "extract_requirements",
    "is_long_input",
    "ready_tasks",
    "requirement_coverage",
    "result_key",
    "validate_plan",
]

#: 输入长度到该值即按「复杂长问题」做完整性检查（PRD 6.4：输入长度用于提示检查完整性）。
LONG_INPUT_MIN_CHARS: Final[int] = 200

#: 出现这些词说明用户要求把通用解释与内部资料**结合**（PRD 6.3 R6 / A03）：
#: 该要求下的 General 任务依赖同要求的 RAG 任务（先查内部方案，再给通用解释）。
COMBINE_MARKERS: Final[tuple[str, ...]] = (
    "结合",
    "综合",
    "对比",
    "比较",
    "配合",
    "相比",
    "兼顾",
)

#: 内部资料线索（PRD 6.3 R1/R2/R3）。与 routing 的关键词表同源语义，此处只用于
#: 无 RouteDecision 时按要求文本推断来源；路由仍是唯一权威判定（T06）。
_INTERNAL_MARKERS: Final[tuple[str, ...]] = (
    "笔记",
    "文档",
    "资料",
    "内部",
    "本项目",
    "我们项目",
    "我们的",
    "既有方案",
    "现有方案",
    "实现",
    "代码",
    "知识库",
    "工作区",
    "方案",
)

#: 时效线索（PRD 6.3 R5）。
_FRESHNESS_MARKERS: Final[tuple[str, ...]] = (
    "今天",
    "今日",
    "最新",
    "最近",
    "现在",
    "当前版本",
    "本周",
    "这周",
    "本月",
    "新闻",
)

#: 通用解释 / 设计建议线索。
_GENERAL_MARKERS: Final[tuple[str, ...]] = (
    "为什么",
    "原理",
    "是什么",
    "解释",
    "建议",
    "推荐",
    "怎么设计",
    "如何设计",
    "最佳实践",
    "区别",
)

#: 需要读取对应章节或关联原文的判定词（PRD 7.2 第 3 步，Planner 侧保留线索）。
PRECISION_MARKERS: Final[tuple[str, ...]] = (
    "定义",
    "是什么",
    "含义",
    "数值",
    "数字",
    "阈值",
    "上限",
    "下限",
    "边界",
    "条件",
    "结论",
    "方案",
    "p99",
    "qps",
    "版本",
)

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^\n。；;!?！？]+")
_LIST_MARKER: Final[re.Pattern[str]] = re.compile(
    r"^(?:[-*•·]|\(?\d{1,2}\)?[.、)）]|第[一二三四五六七八九十百\d]{1,3}[步项点])[\s、.)）]*"
)
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_PUNCTUATION: Final[re.Pattern[str]] = re.compile(
    r"[\s。；;!?！？，,、：:“”\"'（）()\[\]【】\-—_]+"
)


# =====================================================================
# 1. 用户要求（PRD 6.4：提取全部要求、保留原文）
# =====================================================================


class UserRequirement(ContractModel):
    """一条用户要求：结构化文本 + **逐字原文**片段。

    ``quote`` 一定是原问题的连续子串（构造时按正则在原问题上切分），``text`` 只是
    去掉列表标记后的展示文本。完整性检查（:func:`validate_plan`）与覆盖率统计都
    认 ``requirement_id``，这样「每项要求映射到至少一个任务」可被程序核对。
    """

    requirement_id: str
    text: str
    quote: str
    order: int = Field(default=0, ge=0)


class ClarificationRecord(ContractModel):
    """问题需要先澄清时的记录（PRD 6.2 ``clarification`` / A09「任务或澄清记录」）。"""

    question: str
    requirement_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    unresolved_terms: list[str] = Field(default_factory=list)


class TaskSource(StrEnum):
    """任务的证据来源路径（PRD 6.4「根据证据来源判断是否创建多个任务」）。"""

    KNOWLEDGE = "knowledge"
    HISTORY = "history"
    WEB = "web"
    GENERAL = "general"
    PRODUCT = "product"


#: 来源 → Agent 职责（PRD 4.2）与 ``expected_evidence`` 说明。
TASK_SOURCES: Final[dict[TaskSource, tuple[AgentKind, str]]] = {
    TaskSource.KNOWLEDGE: (
        AgentKind.RAG,
        "本工作区知识原文片段与文档版本（search_knowledge / read_document）",
    ),
    TaskSource.HISTORY: (
        AgentKind.RAG,
        "本工作区历史会话消息与来源（search_history / read_history）",
    ),
    TaskSource.WEB: (
        AgentKind.WEB,
        "公开可核实来源的网页原文、发布时间与内容 hash（web_search / read_web_page）",
    ),
    TaskSource.GENERAL: (
        AgentKind.GENERAL,
        "通用解释或设计建议（标记 general_knowledge / proposal，不生成文档引用编号）",
    ),
    TaskSource.PRODUCT: (
        AgentKind.RAG,
        "产品说明与结构化统计（get_workspace_overview / lookup_catalog）",
    ),
}

#: 完成状态（声明完成的任务必须有来源，PRD 6.4 末段）。
_COMPLETION_STATUSES: Final[frozenset[TaskStatus]] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.COMPLETED_WITH_GAPS}
)


def is_long_input(question: str) -> bool:
    """输入是否达到「复杂长问题」阈值（PRD 6.4 / A09）。"""

    return len(_WHITESPACE.sub(" ", str(question or "")).strip()) >= LONG_INPUT_MIN_CHARS


def extract_requirements(question: str) -> list[UserRequirement]:
    """把原问题切成逐字保留的用户要求（句子级 + 列表标记剥离）。

    取舍：句子级切分（换行 / ``。；;!?！？``）而不是按逗号切碎——逗号两侧通常属于
    同一项要求，按逗号切会产生大量同义任务。列表项（``1.`` / ``-`` / ``第一步``）的
    标记被剥离，其余原文逐字保留在 ``quote`` 中，因此「保留原文」可被程序核对。
    """

    text = str(question or "")
    requirements: list[UserRequirement] = []
    for match in _SENTENCE_SPLIT.finditer(text):
        quote = match.group(0)
        cleaned = _LIST_MARKER.sub("", quote).strip()
        if not cleaned:
            continue
        requirements.append(
            UserRequirement(
                requirement_id=f"R{len(requirements) + 1}",
                text=cleaned,
                quote=quote,
                order=len(requirements),
            )
        )
    if not requirements:
        raise PlanValidationError(["问题为空，无法提取用户要求"])
    return requirements


# =====================================================================
# 2. 计划与校验
# =====================================================================


class PlanSettings(ContractModel):
    """并发预算的可配置入口（PRD 6.4 / 18.1）。

    刻意**只**包含并发参数：PRD 7.4 明确本版不设置任务总时长与业务搜索轮数上限，
    因此这里没有 ``max_total_seconds`` / ``max_rounds`` 之类的字段。
    """

    research_agent_concurrency: int = Field(
        default=DEFAULT_RESEARCH_AGENT_CONCURRENCY, ge=1, description="每任务同时运行的研究 Agent 上限。"
    )
    model_request_concurrency: int = Field(
        default=DEFAULT_MODEL_REQUEST_CONCURRENCY, ge=1, description="应用同时执行的模型请求上限。"
    )


class RequirementCoverage(ContractModel):
    """要求覆盖率账目（PRD 6.4 / A09）。"""

    covered: list[str] = Field(default_factory=list)
    uncovered: list[str] = Field(default_factory=list)
    clarified: list[str] = Field(default_factory=list)
    by_task: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def ratio(self) -> float:
        total = len(self.covered) + len(self.uncovered)
        return 1.0 if total == 0 else len(self.covered) / total

    @property
    def complete(self) -> bool:
        return not self.uncovered


class Plan(ContractModel):
    """一次复杂问题的完整计划（PRD 6.4）。

    ``question`` 是**原问题全文**（保留原文），``requirements`` 是逐句要求，
    ``tasks`` 是 ``ResearchTask`` 列表，``clarification`` 在需要先澄清时存在。
    """

    question: str
    workspace_id: str = ""
    knowledge_revision: int = Field(default=0, ge=0)
    long_input: bool = False
    requirements: list[UserRequirement] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
    clarification: ClarificationRecord | None = None
    settings: PlanSettings = Field(default_factory=PlanSettings)
    decision: RouteDecision | None = None
    created_at: str = ""
    notes: list[str] = Field(default_factory=list)

    def task_by_id(self, task_id: str) -> ResearchTask | None:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def task_ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]

    def coverage(self) -> RequirementCoverage:
        return requirement_coverage(self)

    def to_payload(self) -> dict[str, object]:
        """JSON 安全投影（用于 ``plan.updated`` 事件负载）。"""

        return self.model_dump(mode="json")


class PlanValidationError(ValueError):
    """计划结构不合法（PRD 6.4 末段：就地失败，绝不返回空计划）。"""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons: list[str] = [str(reason) for reason in reasons]
        super().__init__("; ".join(self.reasons) or "计划校验失败")


def requirement_coverage(plan: Plan) -> RequirementCoverage:
    """统计每项要求被哪些任务覆盖（澄清记录也算覆盖，见 A09）。"""

    known = {requirement.requirement_id for requirement in plan.requirements}
    by_task: dict[str, list[str]] = {}
    covered: list[str] = []
    for task in plan.tasks:
        mapped = [rid for rid in task.requirement_ids if rid in known]
        by_task[task.task_id] = mapped
        for rid in mapped:
            if rid not in covered:
                covered.append(rid)
    clarified: list[str] = []
    if plan.clarification is not None:
        clarified = [rid for rid in plan.clarification.requirement_ids if rid in known]
        for rid in clarified:
            if rid not in covered:
                covered.append(rid)
    uncovered = [
        requirement.requirement_id
        for requirement in plan.requirements
        if requirement.requirement_id not in covered
    ]
    return RequirementCoverage(
        covered=covered, uncovered=uncovered, clarified=clarified, by_task=by_task
    )


def _task_has_sources(
    task: ResearchTask, evidence_ids: Mapping[str, Sequence[str]] | None
) -> bool:
    """任务是否给出了来源依据：声明了 ``expected_evidence`` 或已记录证据 ID。"""

    if str(task.expected_evidence or "").strip():
        return True
    if evidence_ids is None:
        return False
    return bool(evidence_ids.get(task.task_id))


def validate_plan(
    plan: Plan,
    *,
    evidence_ids: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """计划完整性校验；不合法直接抛 :class:`PlanValidationError`。

    检查项（对应 PRD 6.4 末段与 A09）：

    1. 任务 ID 非空、不重复；
    2. 父任务与 ``depends_on`` 都存在、无依赖环（复用 ``validate_task_graph``）；
    3. 每项用户要求至少映射到一个任务或澄清记录；
    4. 任务引用的 ``requirement_ids`` 必须存在，且任务本身至少关联一项要求；
    5. 声明完成（``completed`` / ``completed_with_gaps``）的任务必须有来源。
    """

    reasons: list[str] = []
    seen: set[str] = set()
    for task in plan.tasks:
        task_id = str(task.task_id or "").strip()
        if not task_id:
            reasons.append("存在缺失任务 ID 的任务")
            continue
        if task_id != task.task_id:
            reasons.append(f"任务 ID 含首尾空白：{task.task_id!r}")
        if task_id in seen:
            reasons.append(f"任务 ID 重复：{task_id}")
        seen.add(task_id)
    if reasons:
        raise PlanValidationError(reasons)

    try:
        validate_task_graph(plan.tasks)
    except ValueError as exc:
        raise PlanValidationError([str(exc)]) from exc

    known = {requirement.requirement_id for requirement in plan.requirements}
    for task in plan.tasks:
        if not task.requirement_ids:
            reasons.append(f"任务 {task.task_id} 没有关联任何用户要求")
        unknown = [rid for rid in task.requirement_ids if rid not in known]
        if unknown:
            reasons.append(f"任务 {task.task_id} 关联了不存在的要求：{', '.join(unknown)}")
        if task.status in _COMPLETION_STATUSES and not _task_has_sources(task, evidence_ids):
            reasons.append(
                f"任务 {task.task_id} 声称 {task.status.value} 但没有任何来源依据"
            )
    if plan.clarification is not None:
        unknown = [
            rid for rid in plan.clarification.requirement_ids if rid not in known
        ]
        if unknown:
            reasons.append(f"澄清记录关联了不存在的要求：{', '.join(unknown)}")

    coverage = requirement_coverage(plan)
    if coverage.uncovered:
        reasons.append(f"以下用户要求没有任务或澄清记录：{', '.join(coverage.uncovered)}")
    if not plan.tasks and plan.clarification is None:
        reasons.append("计划既没有任务也没有澄清记录")

    if reasons:
        raise PlanValidationError(reasons)


# =====================================================================
# 3. 依赖顺序与并行调度（PRD 6.4）
# =====================================================================


def dependency_order(tasks: Sequence[ResearchTask]) -> list[list[str]]:
    """按依赖分层（Kahn）：同一层内的任务相互独立，可并行执行。

    返回形如 ``[["t1", "t2"], ["t3"]]``；有环或缺依赖时抛
    :class:`PlanValidationError`（不返回部分结果）。
    """

    try:
        validate_task_graph(tasks)
    except ValueError as exc:
        raise PlanValidationError([str(exc)]) from exc

    remaining = {task.task_id: set(task.depends_on) for task in tasks}
    levels: list[list[str]] = []
    while remaining:
        ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
        if not ready:
            raise PlanValidationError(["任务依赖存在环，无法分层"])
        levels.append(ready)
        for task_id in ready:
            remaining.pop(task_id)
        for deps in remaining.values():
            deps.difference_update(ready)
    return levels


def ready_tasks(
    tasks: Sequence[ResearchTask], completed: Iterable[str]
) -> list[ResearchTask]:
    """前置已完成、且自身未开始的任务（PRD 6.4：独立并行 / 依赖后启动）。"""

    done = set(completed)
    ready: list[ResearchTask] = []
    for task in tasks:
        if task.status is not TaskStatus.QUEUED:
            continue
        if all(dependency in done for dependency in task.depends_on):
            ready.append(task)
    return ready


# =====================================================================
# 4. 并发预算（PRD 6.4 / 18.1）
# =====================================================================


@dataclass
class BudgetStats:
    """并发观测值（用于断言「不超过上限」）。"""

    research_agents: int
    model_requests: int
    peak_research_agents: int = 0
    peak_model_requests: int = 0
    active_research_agents: int = 0
    active_model_requests: int = 0


class ConcurrencyBudget:
    """同时执行的资源上限：研究 Agent 与应用级模型请求（PRD 6.4 / 18.1）。

    只限制**并发数量**，不限制总时长或总轮数（PRD 7.4）；调用方按需在 ``async with``
    中使用 :meth:`research_slot` / :meth:`model_slot`。
    """

    def __init__(
        self,
        *,
        research_agents: int = DEFAULT_RESEARCH_AGENT_CONCURRENCY,
        model_requests: int = DEFAULT_MODEL_REQUEST_CONCURRENCY,
    ) -> None:
        if research_agents < 1 or model_requests < 1:
            raise ValueError("并发上限必须 >= 1")
        self.stats = BudgetStats(research_agents=research_agents, model_requests=model_requests)
        self._research_sem = asyncio.Semaphore(research_agents)
        self._model_sem = asyncio.Semaphore(model_requests)

    @classmethod
    def from_settings(cls, settings: PlanSettings | None = None) -> ConcurrencyBudget:
        active = settings or PlanSettings()
        return cls(
            research_agents=active.research_agent_concurrency,
            model_requests=active.model_request_concurrency,
        )

    @property
    def research_agents(self) -> int:
        return self.stats.research_agents

    @property
    def model_requests(self) -> int:
        return self.stats.model_requests

    @asynccontextmanager
    async def research_slot(self) -> AsyncIterator[None]:
        async with self._research_sem:
            self.stats.active_research_agents += 1
            self.stats.peak_research_agents = max(
                self.stats.peak_research_agents, self.stats.active_research_agents
            )
            try:
                yield
            finally:
                self.stats.active_research_agents -= 1

    @asynccontextmanager
    async def model_slot(self) -> AsyncIterator[None]:
        async with self._model_sem:
            self.stats.active_model_requests += 1
            self.stats.peak_model_requests = max(
                self.stats.peak_model_requests, self.stats.active_model_requests
            )
            try:
                yield
            finally:
                self.stats.active_model_requests -= 1


# =====================================================================
# 5. 重复任务复用（PRD 6.4：相同工作区 + 知识版本 + 查询目标）
# =====================================================================


def _canonical_question(question: str) -> str:
    return _PUNCTUATION.sub("", str(question or "").casefold())


def result_key(workspace_id: str, knowledge_revision: int, question: str) -> str:
    """复用的比较键：工作区 + 知识版本 + 规范化查询目标（PRD 6.4）。"""

    payload = "\x1f".join(
        (str(workspace_id or ""), str(int(knowledge_revision or 0)), _canonical_question(question))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class TaskResultCache:
    """已保存结果的复用表：命中即把 ``result_ref`` 直接挂到新任务上。"""

    def __init__(self, entries: Mapping[str, str] | None = None) -> None:
        self._entries: dict[str, str] = dict(entries or {})

    def key_for(self, task: ResearchTask, *, workspace_id: str, knowledge_revision: int) -> str:
        return result_key(workspace_id, knowledge_revision, task.question)

    def lookup(
        self, task: ResearchTask, *, workspace_id: str, knowledge_revision: int
    ) -> str | None:
        return self._entries.get(
            self.key_for(task, workspace_id=workspace_id, knowledge_revision=knowledge_revision)
        )

    def record(
        self,
        task: ResearchTask,
        result_ref: str,
        *,
        workspace_id: str,
        knowledge_revision: int,
    ) -> str:
        key = self.key_for(task, workspace_id=workspace_id, knowledge_revision=knowledge_revision)
        self._entries[key] = str(result_ref)
        return key

    def as_mapping(self) -> dict[str, str]:
        return dict(self._entries)


# =====================================================================
# 6. 计划生成
# =====================================================================


def build_web_subtask(
    parent: ResearchTask,
    question: str,
    *,
    requirement_ids: Sequence[str] | None = None,
    index: int = 1,
) -> ResearchTask:
    """由 Planner 为「需要公开原理/最新版本」的缺口建一个 Web 子任务（PRD 7.2 第 5 步）。

    RAG 研究循环通过 :attr:`ResearchResult.web_requests` 提出缺口，Planner 用它落成正式
    任务：``parent_task_id`` 指向提出者、``depends_on`` 保证父任务结果先完成、
    ``requirement_ids`` 默认继承父任务（缺口属于同一项用户要求）。
    """

    active_requirements = (
        list(requirement_ids)
        if requirement_ids is not None
        else list(parent.requirement_ids)
    )
    if not active_requirements:
        raise PlanValidationError([f"父任务 {parent.task_id} 没有关联用户要求，无法建 Web 子任务"])
    return ResearchTask(
        task_id=f"{parent.task_id}-web{index}",
        parent_task_id=parent.task_id,
        question=str(question or parent.question),
        requirement_ids=active_requirements,
        agent_kind=AgentKind.WEB,
        depends_on=[parent.task_id],
        expected_evidence=TASK_SOURCES[TaskSource.WEB][1],
        status=TaskStatus.QUEUED,
    )


def _sources_for(requirement: UserRequirement, decision: RouteDecision | None) -> list[TaskSource]:
    """一项要求需要哪些来源路径（PRD 6.3 决策规则 + 6.4 来源判断）。"""

    text = f"{requirement.text} {requirement.quote}".casefold()
    internal = any(marker in text for marker in _INTERNAL_MARKERS)
    freshness = any(marker in text for marker in _FRESHNESS_MARKERS)
    general = any(marker in text for marker in _GENERAL_MARKERS)

    if decision is None:
        sources: list[TaskSource] = []
        if internal:
            sources.append(TaskSource.KNOWLEDGE)
        if freshness:
            sources.append(TaskSource.WEB)
        if general or not sources:
            sources.append(TaskSource.GENERAL)
        if (
            internal
            and any(marker in text for marker in COMBINE_MARKERS)
            and TaskSource.GENERAL not in sources
        ):
            sources.append(TaskSource.GENERAL)
        return sources

    sources = []
    if decision.intent.value == "product":
        sources.append(TaskSource.PRODUCT)
    if decision.needs_knowledge:
        sources.append(TaskSource.KNOWLEDGE)
    if decision.needs_history:
        sources.append(TaskSource.HISTORY)
    if decision.needs_web:
        sources.append(TaskSource.WEB)
    if decision.needs_general:
        sources.append(TaskSource.GENERAL)
    if not sources and decision.intent.value != "clarification":
        # 路由没有给出明确来源（例如纯 product 之外的 intent）：按文本线索兜底，
        # 但没有线索时至少给通用解释路径，避免「无任务」被静默接受。
        sources = _sources_for(requirement, None)
    return sources


def _task_id(requirement_id: str, source: TaskSource) -> str:
    return f"t-{requirement_id.lower()}-{source.value}"


def _build_tasks(
    requirements: Sequence[UserRequirement],
    decision: RouteDecision | None,
) -> tuple[list[ResearchTask], list[str]]:
    """按「要求 × 来源」建任务；相同查询目标去重后合并要求（PRD 6.4）。"""

    notes: list[str] = []
    grouped: dict[tuple[str, TaskSource], ResearchTask] = {}
    requires_knowledge: dict[str, str] = {}
    for requirement in requirements:
        sources = _sources_for(requirement, decision)
        for source in sources:
            key = (_canonical_question(requirement.text), source)
            agent_kind, expected = TASK_SOURCES[source]
            existing = grouped.get(key)
            if existing is not None:
                if requirement.requirement_id not in existing.requirement_ids:
                    existing.requirement_ids = [
                        *existing.requirement_ids,
                        requirement.requirement_id,
                    ]
                notes.append(
                    f"重复查询目标复用任务 {existing.task_id}（要求 {requirement.requirement_id}）"
                )
                continue
            grouped[key] = ResearchTask(
                task_id=_task_id(requirement.requirement_id, source),
                question=requirement.text,
                requirement_ids=[requirement.requirement_id],
                agent_kind=agent_kind,
                depends_on=[],
                expected_evidence=expected,
                status=TaskStatus.QUEUED,
            )
            if source is TaskSource.KNOWLEDGE:
                requires_knowledge[requirement.requirement_id] = grouped[key].task_id

    tasks = list(grouped.values())
    # 结合类要求（PRD 6.3 R6 / A03）：通用解释在同一要求内部依据之后启动。
    for requirement in requirements:
        if not any(marker in requirement.text for marker in COMBINE_MARKERS):
            continue
        knowledge_task_id = requires_knowledge.get(requirement.requirement_id)
        general_task_id = _task_id(requirement.requirement_id, TaskSource.GENERAL)
        general_task = next((task for task in tasks if task.task_id == general_task_id), None)
        if (
            knowledge_task_id
            and general_task is not None
            and knowledge_task_id not in general_task.depends_on
        ):
            general_task.depends_on = [*general_task.depends_on, knowledge_task_id]
            notes.append(
                f"任务 {general_task.task_id} 依赖 {knowledge_task_id}（先内部资料后通用解释）"
            )
    return tasks, notes


def build_plan(
    question: str,
    decision: RouteDecision | None = None,
    *,
    requirements: Sequence[UserRequirement] | None = None,
    proposed_tasks: Sequence[ResearchTask] | None = None,
    existing_results: Mapping[str, str] | None = None,
    workspace_id: str = "",
    knowledge_revision: int = 0,
    settings: PlanSettings | None = None,
) -> Plan:
    """由已理解的问题生成 ``Plan``（PRD 6.4）。

    模型提出的任务（``proposed_tasks``）会先参与结构校验：依赖环、缺失任务 ID、无来源
    的完成声明直接抛 :class:`PlanValidationError`；模型漏掉的要求由程序按来源补任务
    （并在 ``notes`` 记录），因此「每项要求映射到至少一个任务」不依赖模型自觉。
    ``existing_results`` 是 :class:`TaskResultCache` 的映射：命中即复用结果（``result_ref``）。
    """

    active_settings = settings or PlanSettings()
    text = str(question or "")
    active_requirements = (
        list(requirements) if requirements is not None else extract_requirements(text)
    )
    validate_requirement_quotes(active_requirements, text)
    long_input = is_long_input(text)

    rule_tasks, notes = _build_tasks(active_requirements, decision)
    tasks: list[ResearchTask] = []
    model_covered: set[str] = set()
    if proposed_tasks:
        proposals = [task.model_copy(deep=True) for task in proposed_tasks]
        # 模型输出先单独校验结构（环 / 缺失 ID / 无来源完成声明），再补覆盖。
        try:
            validate_task_graph(proposals)
        except ValueError as exc:
            raise PlanValidationError([f"模型计划不合法：{exc}"]) from exc
        for task in proposals:
            if not str(task.task_id or "").strip():
                raise PlanValidationError(["模型计划存在缺失任务 ID 的任务"])
            if task.status in _COMPLETION_STATUSES and not _task_has_sources(task, None):
                raise PlanValidationError(
                    [f"模型计划的任务 {task.task_id} 声称完成但没有任何来源依据"]
                )
        for task in proposals:
            if task.status in _COMPLETION_STATUSES:
                task.status = TaskStatus.QUEUED
                task.result_ref = None
        tasks.extend(proposals)
        model_covered.update(rid for task in proposals for rid in task.requirement_ids)
        notes.append(f"采用模型提出的 {len(proposals)} 个任务")

    for task in rule_tasks:
        if model_covered and all(rid in model_covered for rid in task.requirement_ids):
            notes.append(
                f"要求 {', '.join(task.requirement_ids)} 已由模型任务覆盖，跳过 {task.task_id}"
            )
            continue
        tasks.append(task)

    clarification: ClarificationRecord | None = None
    clarify_intent = decision is not None and decision.intent.value == "clarification"
    if clarify_intent or not tasks:
        unresolved = list(decision.unresolved_terms) if decision is not None else []
        clarification = ClarificationRecord(
            question=text,
            requirement_ids=[requirement.requirement_id for requirement in active_requirements],
            reason="问题需要先确认对象或范围才能确定来源路径（PRD 6.2 clarification）",
            unresolved_terms=unresolved,
        )
        notes.append("没有可执行任务，改为澄清记录")

    # 重复任务复用：相同工作区 + 知识版本 + 查询目标的已有结果直接挂回。
    cache = TaskResultCache(existing_results)
    reused = 0
    for task in tasks:
        ref = cache.lookup(task, workspace_id=workspace_id, knowledge_revision=knowledge_revision)
        if ref is None:
            continue
        task.result_ref = ref
        task.status = TaskStatus.COMPLETED
        reused += 1
    if reused:
        notes.append(f"{reused} 个任务复用已有结果")

    plan = Plan(
        question=text,
        workspace_id=workspace_id,
        knowledge_revision=knowledge_revision,
        long_input=long_input,
        requirements=active_requirements,
        tasks=tasks,
        clarification=clarification,
        settings=active_settings,
        decision=decision,
        created_at=utc_now_iso(),
        notes=notes,
    )
    validate_plan(plan)
    return plan


# =====================================================================
# 7. Planner Agent 门面（runtime 调用入口）
# =====================================================================


@dataclass
class PlannerAgent:
    """Planner Agent 门面：计划生成 + 校验 + 调度查询，状态最小且可离线单测。"""

    workspace_id: str = ""
    knowledge_revision: int = 0
    settings: PlanSettings = field(default_factory=PlanSettings)
    results: TaskResultCache = field(default_factory=TaskResultCache)

    @property
    def budget(self) -> ConcurrencyBudget:
        return ConcurrencyBudget.from_settings(self.settings)

    def plan(
        self,
        question: str,
        decision: RouteDecision | None = None,
        *,
        requirements: Sequence[UserRequirement] | None = None,
        proposed_tasks: Sequence[ResearchTask] | None = None,
    ) -> Plan:
        """生成并校验计划（不合法直接抛 :class:`PlanValidationError`）。"""

        return build_plan(
            question,
            decision,
            requirements=requirements,
            proposed_tasks=proposed_tasks,
            existing_results=self.results.as_mapping(),
            workspace_id=self.workspace_id,
            knowledge_revision=self.knowledge_revision,
            settings=self.settings,
        )

    def ready(self, plan: Plan, completed: Iterable[str]) -> list[ResearchTask]:
        return ready_tasks(plan.tasks, completed)

    def levels(self, plan: Plan) -> list[list[str]]:
        return dependency_order(plan.tasks)

    def record_result(self, task: ResearchTask, result_ref: str) -> str:
        """把一次成功结果登记进复用表（供后续相同查询目标复用）。"""

        return self.results.record(
            task,
            result_ref,
            workspace_id=self.workspace_id,
            knowledge_revision=self.knowledge_revision,
        )

    def validate(
        self, plan: Plan, *, evidence_ids: Mapping[str, Sequence[str]] | None = None
    ) -> None:
        validate_plan(plan, evidence_ids=evidence_ids)


def validate_requirement_quotes(
    requirements: Sequence[UserRequirement], question: str
) -> None:
    """自检 ``quote`` 仍是原问题的连续子串（保证「保留原文」不被改写）。"""

    for requirement in requirements:
        if requirement.quote and requirement.quote not in question:
            raise PlanValidationError(
                [f"要求 {requirement.requirement_id} 的原文片段不在原问题中（原文被改写）"]
            )
