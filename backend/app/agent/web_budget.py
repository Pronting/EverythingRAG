"""联网预算与联网策略：**思考等级 → 联网权限 + 整轮共享额度**。

用户裁定（本轮新增，取代「联网只看用户是否点批准」的旧口径）：

- **关闭思考（off）**：禁用联网，只用模型一般知识与本地资料回答；
- **其他等级**：开放联网，等级越高允许的额度越多；
- **共享额度**：一次用户提问及其全部子任务、重试、补充搜索共享同一份额度；
- **额度是上限**：信息足够时可以直接回答，不强制联网；
- **资料范围**：用户明确要求只用本地 / 指定文档时（设置页「仅知识库」或问题里明确
  要求），联网额度为 0，不允许通过网络扩展回答依据。

设计要点：

1. 额度是**代码里的集中配置**（:data:`DEFAULT_WEB_BUDGETS`），可用环境变量
   ``EVERYTHING_RAG_WEB_BUDGETS``（JSON）覆盖；不在提示词里「请模型自觉遵守」，
   而是由 :class:`WebBudgetLedger` 在**联网工具入口**实际扣减与拒绝。
2. 额度分三类：**搜索查询条数**、**网页读取条数**、**联网阶段总时长**。查询按次计
   （一次请求带多条查询就按条数计），读取网页单独计。
3. 同一个 run 的所有子任务共用同一个账本（:meth:`RunManager.web_budget_for`），并发
   执行也不会超额：扣减发生在 ``await`` 之前，事件循环内不会交错。
4. 重复查询不重复扣减也不重复出网（:data:`REASON_DUPLICATE`）——「防止重复执行相同查询」。
5. 失败与重试：**每次真实出网尝试都扣一次**（失败也扣）；同一操作最多重试
   :data:`MAX_ATTEMPTS_PER_OPERATION` 次，之后不再尝试。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from app.agent.contracts import ReasoningLevel

__all__ = [
    "DEFAULT_WEB_BUDGETS",
    "MAX_ATTEMPTS_PER_OPERATION",
    "REASON_BUDGET_EXHAUSTED",
    "REASON_DISABLED",
    "REASON_DUPLICATE",
    "REASON_NOT_CONFIGURED",
    "REASON_TIME_EXHAUSTED",
    "WEB_BUDGETS_ENV",
    "WebBudget",
    "WebBudgetLedger",
    "WebPolicy",
    "WebReservation",
    "budgets_from_env",
    "local_only_question",
    "resolve_web_policy",
]


# =====================================================================
# 1. 集中配置：按思考等级分配额度
# =====================================================================


@dataclass(frozen=True)
class WebBudget:
    """一个等级的联网额度上限（``queries`` 搜索查询条数 / ``pages`` 网页读取条数 / ``seconds`` 秒）。"""

    queries: int = 0
    pages: int = 0
    seconds: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.queries > 0 and self.pages > 0 and self.seconds > 0

    def as_dict(self) -> dict[str, Any]:
        return {"queries": self.queries, "pages": self.pages, "seconds": self.seconds}


#: 各思考等级的默认额度。依据：
#: - ``off`` = 0：用户裁定「关闭思考时禁用联网」；
#: - ``low``：够做一次「查一条 → 读两三页」的最小核实；
#: - ``high``：够做「多子问题分别核实 + 读原文」（4 条查询 ≈ 2–3 个子问题各 1–2 条）；
#: - ``max``：够做一轮完整的「多来源交叉核实」，同时仍由时长上限兜住成本。
DEFAULT_WEB_BUDGETS: Final[dict[ReasoningLevel, WebBudget]] = {
    ReasoningLevel.OFF: WebBudget(0, 0, 0.0),
    ReasoningLevel.LOW: WebBudget(2, 3, 20.0),
    ReasoningLevel.HIGH: WebBudget(4, 8, 45.0),
    ReasoningLevel.MAX: WebBudget(8, 16, 90.0),
}

#: 覆盖默认额度的环境变量（JSON：``{"low": [2, 3, 20], "high": {"queries": 4, ...}}``）。
WEB_BUDGETS_ENV: Final[str] = "EVERYTHING_RAG_WEB_BUDGETS"

#: 同一操作（同一 run + 同参数）允许的联网尝试次数上限（失败也计入额度）。
MAX_ATTEMPTS_PER_OPERATION: Final[int] = 2

REASON_DISABLED: Final[str] = "web_disabled"
REASON_NOT_CONFIGURED: Final[str] = "web_not_configured"
REASON_BUDGET_EXHAUSTED: Final[str] = "web_budget_exhausted"
REASON_TIME_EXHAUSTED: Final[str] = "web_time_exhausted"
REASON_DUPLICATE: Final[str] = "duplicate_query"


def _coerce_budget(value: Any, *, fallback: WebBudget) -> WebBudget:
    """把配置值解析成 :class:`WebBudget`（``[queries, pages, seconds]`` 或对象）。"""

    try:
        if isinstance(value, Mapping):
            return WebBudget(
                queries=max(0, int(value.get("queries", fallback.queries))),
                pages=max(0, int(value.get("pages", fallback.pages))),
                seconds=max(0.0, float(value.get("seconds", fallback.seconds))),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
            return WebBudget(
                queries=max(0, int(items[0])),
                pages=max(0, int(items[1])),
                seconds=max(0.0, float(items[2])),
            )
        if isinstance(value, (int, float)):
            return WebBudget(int(value), int(value), fallback.seconds)
    except (TypeError, ValueError):
        return fallback
    return fallback


def budgets_from_env(env: Mapping[str, str] | None = None) -> dict[ReasoningLevel, WebBudget]:
    """读取环境变量覆盖（缺省 = 全用 :data:`DEFAULT_WEB_BUDGETS`）。"""

    source = os.environ if env is None else env
    raw = str(source.get(WEB_BUDGETS_ENV) or "").strip()
    budgets = dict(DEFAULT_WEB_BUDGETS)
    if not raw:
        return budgets
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return budgets
    if not isinstance(payload, Mapping):
        return budgets
    for level in ReasoningLevel:
        if level.value in payload:
            budgets[level] = _coerce_budget(payload[level.value], fallback=budgets[level])
    return budgets


# =====================================================================
# 2. 联网策略（是否允许 + 额度）
# =====================================================================


@dataclass(frozen=True)
class WebPolicy:
    """一次 run 的联网策略结论（权限 + 额度 + 可读原因）。"""

    allowed: bool
    budget: WebBudget
    reason_code: str = ""
    reason_text: str = ""
    reasoning_level: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasoning_level": self.reasoning_level,
            "reason_code": self.reason_code,
            "reason": self.reason_text,
            **self.budget.as_dict(),
        }


#: 「只用本地 / 指定文档」的表达（问题里明确要求时，联网额度为 0）。
_LOCAL_ONLY_RE: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?:只|仅|仅限|只用|只依据|只根据|只参考)[^。！？?]{0,12}(?:资料|文档|知识库|本地|笔记|原文)"),
    re.compile(r"(?:资料|文档|知识库|本地)[^。！？?]{0,8}(?:范围|以内|之内|为准)"),
    re.compile(r"(?:不(?:要|用|需|必|得)|禁止)[^。！？?]{0,10}(?:联网|网络|上网|搜索网页|外部)"),
)


def local_only_question(question: str) -> bool:
    """问题里是否**明确要求**只用本地 / 指定文档回答（不许联网扩展）。"""

    text = " ".join(str(question or "").split())
    if not text:
        return False
    for pattern in _LOCAL_ONLY_RE:
        for match in pattern.finditer(text):
            scope = match.group(0)
            if re.search(r"官方|官网", scope) and not re.search(r"本地|知识库|笔记|保存|下载", scope):
                continue
            return True
    return False


def resolve_web_policy(
    *,
    reasoning_level: ReasoningLevel | str | None,
    knowledge_only: bool = False,
    local_only_question_flag: bool = False,
    search_configured: bool = True,
    budgets: Mapping[ReasoningLevel, WebBudget] | None = None,
) -> WebPolicy:
    """按思考等级与资料范围解析联网策略（唯一实现，运行层与工具层共用）。"""

    try:
        level = ReasoningLevel(str(reasoning_level or ReasoningLevel.OFF))
    except ValueError:
        level = ReasoningLevel.OFF
    table = dict(budgets) if budgets is not None else budgets_from_env()
    budget = table.get(level, DEFAULT_WEB_BUDGETS[level])

    def policy(allowed: bool, code: str, text: str, *, zero: bool = False) -> WebPolicy:
        return WebPolicy(
            allowed=allowed,
            budget=WebBudget(0, 0, 0.0) if zero else budget,
            reason_code=code,
            reason_text=text,
            reasoning_level=level.value,
        )

    if not search_configured:
        return policy(
            False, REASON_NOT_CONFIGURED,
            "联网搜索未配置或未启用，本轮只能使用一般知识与本地资料",
            zero=True,
        )
    if knowledge_only or local_only_question_flag:
        return policy(
            False, REASON_DISABLED,
            "你要求只依据本地资料回答，本轮不会联网扩展依据",
            zero=True,
        )
    if level is ReasoningLevel.OFF:
        return policy(
            False, REASON_DISABLED,
            "当前为「关闭思考」模式，联网已禁用，因此无法查询或核对网络原文",
            zero=True,
        )
    if not budget.enabled:
        return policy(False, REASON_DISABLED, "当前等级的联网额度为 0，本轮不联网", zero=True)
    return policy(True, "", "")


# =====================================================================
# 3. 整轮共享额度账本
# =====================================================================


@dataclass(frozen=True)
class WebReservation:
    """一次额度申请的结论（允许 / 拒绝 + 可读原因）。"""

    allowed: bool
    reason_code: str = ""
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason_code": self.reason_code, "message": self.message}


@dataclass
class WebBudgetLedger:
    """**一次 run 共享**的联网额度账本（并发安全：扣减发生在 ``await`` 之前）。

    - ``reserve_search`` / ``reserve_pages`` 在真正出网前调用；拒绝时不扣减、不执行；
    - 重复查询（规范化后相同）返回 :data:`REASON_DUPLICATE`，不扣减也不重复出网；
    - ``note_result`` 记录每次尝试的阶段与结果（成功/失败都算已扣减），供过程记录与
      回答里的缺口说明使用；时间额度从**第一次联网申请**开始计。
    """

    policy: WebPolicy
    clock: Callable[[], float] = time.monotonic
    _queries_used: int = field(default=0, init=False)
    _pages_used: int = field(default=0, init=False)
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _seen_queries: set[str] = field(default_factory=set, init=False)
    _events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _started_at: float | None = field(default=None, init=False)
    #: 当前暂停的起点（等待用户批准 / 等待用户输入期间）。``None`` = 正在计时。
    _paused_at: float | None = field(default=None, init=False)
    #: 累计暂停时长（等待用户的时间**不计入**网络执行时间，缺陷 F16）。
    _paused_total: float = field(default=0.0, init=False)
    #: 仅累计首次联网申请之后的暂停，供执行用时扣除。
    _execution_paused_total: float = field(default=0.0, init=False)
    phase: str = field(default="", init=False)

    # ---- 只读状态 ----

    @property
    def allowed(self) -> bool:
        return self.policy.allowed

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def pages_used(self) -> int:
        return self._pages_used

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def elapsed(self) -> float:
        """已用网络执行时间，不含等待用户批准、澄清或继续执行的时长。"""

        if self._started_at is None:
            return 0.0
        now = self._paused_at if self._paused_at is not None else self.clock()
        return max(0.0, now - self._started_at - self._execution_paused_total)

    def wait_seconds(self) -> float:
        """等待用户操作的累计时长，单独供过程记录展示。"""

        total = self._paused_total
        if self._paused_at is not None:
            total += max(0.0, self.clock() - self._paused_at)
        return total

    def pause(self) -> None:
        """等待用户批准、澄清或继续执行时暂停计时，重复调用幂等。"""

        if self._paused_at is None:
            self._paused_at = self.clock()

    def resume(self) -> None:
        """恢复计时（用户已裁决 / 已提供输入），把这段等待累计进 ``wait_seconds``。"""

        if self._paused_at is None:
            return
        now = self.clock()
        self._paused_total += max(0.0, now - self._paused_at)
        if self._started_at is not None:
            self._execution_paused_total += max(0.0, now - max(self._paused_at, self._started_at))
        self._paused_at = None

    def remaining_seconds(self) -> float:
        return max(0.0, self.policy.budget.seconds - self.elapsed())

    def attempts_for(self, query: str) -> int:
        return self._attempts.get(_canonical_query(query), 0)

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._events[-limit:])

    def snapshot(self) -> dict[str, Any]:
        """可安全放进事件 / 工具结果 / 过程说明的账本快照（不含查询正文之外的内容）。"""

        budget = self.policy.budget
        return {
            "allowed": self.policy.allowed,
            "reason_code": self.policy.reason_code,
            "reason": self.policy.reason_text,
            "reasoning_level": self.policy.reasoning_level,
            "queries_used": self._queries_used,
            "queries_limit": budget.queries,
            "pages_used": self._pages_used,
            "pages_limit": budget.pages,
            "elapsed_seconds": round(self.elapsed(), 2),
            "seconds_limit": budget.seconds,
            "paused": self.paused,
            "wait_seconds": round(self.wait_seconds(), 2),
            "phase": self.phase,
        }

    def restore_usage(self, *, queries: int, pages: int) -> None:
        """进程重启后按持久工具事件恢复已经消耗的次数。"""

        if self._queries_used or self._pages_used or self._started_at is not None:
            raise ValueError("联网额度账本已开始使用，不能再次恢复次数")
        self._queries_used = max(0, int(queries))
        self._pages_used = max(0, int(pages))

    def summary_text(self) -> str:
        """一行可读的额度状态（进过程说明与缺口说明）。

        等待用户批准的时长**单独**说明：它不是网络执行时间，不能算进额度
        （用户实测缺陷 F16：批准等待 27 秒把 low 档的 20 秒额度吃光，批准后的页面读取
        立刻被判 ``web_budget_exhausted``）。
        """

        budget = self.policy.budget
        text = (
            f"联网额度：查询 {self._queries_used}/{budget.queries}、"
            f"网页 {self._pages_used}/{budget.pages}、"
            f"用时 {self.elapsed():.0f}/{budget.seconds:.0f}s"
        )
        waited = self.wait_seconds()
        if waited >= 1.0:
            text += f"（另有等待用户操作 {waited:.0f}s，不计入联网用时）"
        return text

    # ---- 扣减 ----

    def _deny(self, code: str, message: str) -> WebReservation:
        return WebReservation(False, code, message)

    def _ensure_window(self) -> WebReservation | None:
        if not self.policy.allowed:
            return self._deny(self.policy.reason_code or REASON_DISABLED, self.policy.reason_text)
        if self._started_at is not None and self.elapsed() >= self.policy.budget.seconds:
            return self._deny(
                REASON_TIME_EXHAUSTED,
                f"本轮联网时间额度已用完（{self.policy.budget.seconds:.0f}s），"
                "改用已有资料回答并说明缺口",
            )
        return None

    def reserve_search(self, query: str, *, phase: str = "") -> WebReservation:
        """申请一次联网搜索额度（同 run 内相同查询不重复扣减、不重复出网）。"""

        denied = self._ensure_window()
        if denied is not None:
            return denied
        key = _canonical_query(query)
        if key and key in self._seen_queries:
            return self._deny(
                REASON_DUPLICATE,
                f"同一查询本轮已经执行过（{_preview(query)}），不重复联网",
            )
        if self._queries_used >= self.policy.budget.queries:
            return self._deny(
                REASON_BUDGET_EXHAUSTED,
                f"本轮联网搜索额度已用完（{self._queries_used}/{self.policy.budget.queries} 条查询），"
                "改用已有资料回答并说明缺口",
            )
        self.phase = phase or self.phase
        self._started_at = self._started_at if self._started_at is not None else self.clock()
        self._queries_used += 1
        self._seen_queries.add(key)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        self._record("web_search", query=query, outcome="reserved", phase=self.phase)
        return WebReservation(True)

    def reserve_pages(self, count: int, *, phase: str = "") -> WebReservation:
        """申请网页读取额度（按页计）。"""

        denied = self._ensure_window()
        if denied is not None:
            return denied
        wanted = max(1, int(count))
        if self._pages_used >= self.policy.budget.pages:
            return self._deny(
                REASON_BUDGET_EXHAUSTED,
                f"本轮网页读取额度已用完（{self._pages_used}/{self.policy.budget.pages} 页），"
                "改用已有资料回答并说明缺口",
            )
        allowed = min(wanted, self.policy.budget.pages - self._pages_used)
        self.phase = phase or self.phase
        self._started_at = self._started_at if self._started_at is not None else self.clock()
        self._pages_used += allowed
        self._record("read_web_page", query=f"{allowed} 页", outcome="reserved", phase=self.phase)
        return WebReservation(True, "", "" if allowed >= wanted else f"仅剩 {allowed} 页额度，已按剩余额度读取")

    def attempts_exhausted(self, query: str) -> bool:
        """同一查询的尝试次数是否已达上限（防止无限重试）。"""

        key = _canonical_query(query)
        return self._attempts.get(key, 0) >= MAX_ATTEMPTS_PER_OPERATION

    def note_result(self, tool: str, *, ok: bool, detail: str = "", query: str = "") -> None:
        self._record(tool, query=query, outcome="succeeded" if ok else "failed", phase=self.phase, detail=detail)

    def _record(
        self, tool: str, *, query: str, outcome: str, phase: str = "", detail: str = ""
    ) -> None:
        self._events.append(
            {
                "seq": len(self._events) + 1,
                "tool": tool,
                "phase": phase,
                "query": _preview(query),
                "outcome": outcome,
                "queries": f"{self._queries_used}/{self.policy.budget.queries}",
                "pages": f"{self._pages_used}/{self.policy.budget.pages}",
                "elapsed": round(self.elapsed(), 2),
                "detail": detail,
            }
        )


def _canonical_query(query: str) -> str:
    from app.agent.query_plan import stable_query_identity

    identity = stable_query_identity(query, source_kind="web")
    if identity.objects:
        return "semantic:" + identity.key
    return re.sub(r"\s+", " ", str(query or "")).strip().casefold()


def _preview(query: str, limit: int = 60) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    return text if len(text) <= limit else text[:limit] + "…"
