"""V1.2.0 Agent 共享契约：枚举、状态机与稳定标识（t1 冻结区）。

本模块是全部 Agent 工作流的**公共词汇表**：只定义跨模块共享的枚举、状态迁移
规则与稳定标识格式，不含 IO、模型调用或任何业务实现。每个取值都能在
`docs/PRD/v1.2.0-PRD.md` 中逐条指认，改动必须同步 PRD：

- 任务状态      → PRD 12.3（11 个状态 + 迁移图）
- SSE 事件类型  → PRD 13.3（13 行 / 20 个 type 取值）
- 批准状态      → PRD 10.2（pending/approved/denied/superseded）
- 澄清状态      → PRD 5.4 + 13.2（pending/answered/superseded）
- 证据来源类型  → PRD 8.1（6 类权威来源；General Agent 另加 2 类）
- 发布依据      → PRD 8.3（`answer_basis` 9 个取值）
- 权限模式      → PRD 10.1（ask/auto/full，默认 ask）
- 思考强度      → PRD 9.1（off/low/high/max，默认 high）
- 意图与时效    → PRD 6.2（`RouteDecision.intent` / `freshness`）
- Agent 职责    → PRD 4.2（9 个节点角色）
- 核对结论      → PRD 8.2（`verdict=pass/revise/research/clarify/insufficient`）
- 运行身份      → PRD 12.1（`thread_id` 规则见 `main_thread_id` / `task_thread_id`）

文件分区约定（避免两名成员改同一段）：

1. ``t1 冻结区``：本文件上方的枚举、状态集合、迁移表、标识与引用编号辅助函数。
   由 contracts-db 冻结；后续工作流**只读复用**，不得修改取值或语义。
2. ``t2 追加区``：文件末尾的 Pydantic 契约模型（`RouteDecision`、`ResearchTask`、
   `Evidence`、`AnswerArtifact`、`ApprovalRequest`、`InputRequest`、
   `ModelCapabilities`、`ProviderMessage`、`ToolCall`、`OperationRecord`、
   `AgentError` 及 PRD 18.1 工程默认值常量），由 T00b 在末尾追加。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # 只用于类型标注：契约层不在运行期依赖 app_store
    from app.agent.app_store import AppStore

__all__ = [
    "CANCELLABLE_RUN_STATUSES",
    "CITATION_PREFIX",
    "DEFAULT_PERMISSION_MODE",
    "DEFAULT_REASONING_LEVEL",
    "MAIN_THREAD_TEMPLATE",
    "RESUMABLE_RUN_STATUSES",
    "RUN_STATUS_LABELS",
    "RUN_STATUS_TRANSITIONS",
    "SUBSCRIPTION_CLOSING_EVENTS",
    "TASK_STATUS_TRANSITIONS",
    "TASK_THREAD_TEMPLATE",
    "TERMINAL_RUN_STATUSES",
    "AgentKind",
    "AnswerBasis",
    "ApprovalStatus",
    "EventType",
    "EvidenceSourceType",
    "Freshness",
    "InputRequestStatus",
    "IntentKind",
    "PermissionMode",
    "ReasoningLevel",
    "RunStatus",
    "SourceType",
    "TaskStatus",
    "VerifierVerdict",
    "can_transition_run",
    "can_transition_task",
    "citation_prefix",
    "is_main_thread_id",
    "main_thread_id",
    "task_thread_id",
    "thread_of_run",
]


# =====================================================================
# t1 冻结区 · 1. 任务状态（PRD 12.3）
# =====================================================================


class RunStatus(StrEnum):
    """一次用户问题（run）的执行状态。取值与 PRD 12.3 表逐条一致。

    ``RunStatus`` 与 ``TaskStatus`` 都来自 PRD 12.3 同一张表，两者取值集合
    必须完全相同；区分类型只是为了在类型标注上表达「run 的状态」与
    「agent_task 的状态」。契约测试覆盖二者一致性。
    """

    QUEUED = "queued"  # 等待执行资源
    RUNNING = "running"  # 正在处理
    WAITING_CHILDREN = "waiting_children"  # 等待独立研究任务
    WAITING_APPROVAL = "waiting_approval"  # 等待用户批准
    WAITING_INPUT = "waiting_input"  # 等待用户澄清
    INTERRUPTED = "interrupted"  # 进程退出或上下文变更导致执行中断
    FAILED = "failed"  # 模型、工具或运行系统明确失败
    CANCELLING = "cancelling"  # 正在取消活动请求和子任务
    CANCELLED = "cancelled"  # 用户停止或会话移动终止原执行
    COMPLETED = "completed"  # 答案通过核对并保存
    COMPLETED_WITH_GAPS = "completed_with_gaps"  # 已说明无法取得的关键资料


class TaskStatus(StrEnum):
    """一个研究子任务（`agent_tasks.status`）的状态，取值与 PRD 12.3 一致。"""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_CHILDREN = "waiting_children"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"


class ApprovalStatus(StrEnum):
    """批准请求状态（PRD 10.2）。同一 run 内由事务从 ``pending`` 单向迁移。"""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    SUPERSEDED = "superseded"


class InputRequestStatus(StrEnum):
    """澄清（输入）请求状态。

    PRD 13.2 定义「澄清回答 + 问题版本」的提交入口，PRD 5.4 规定因会话移动
    终止时澄清请求转为 ``superseded``；被终止 run 的待澄清卡片同样走
    ``superseded``（PRD 12.4：重启后仅恢复待批准/待澄清卡片，不恢复已终止项）。
    """

    PENDING = "pending"
    ANSWERED = "answered"
    SUPERSEDED = "superseded"


# =====================================================================
# t1 冻结区 · 2. 来源与依据（PRD 8.1 / 8.3）
# =====================================================================


class SourceType(StrEnum):
    """权威来源类型（PRD 8.1 第一句）。只有这 6 类能进入证据与引用体系。"""

    KNOWLEDGE = "knowledge"  # 本工作区知识原文
    WEB = "web"  # 网页原文
    MEMORY = "memory"  # 工作区记忆
    HISTORY = "history"  # 本工作区历史讨论
    CATALOG = "catalog"  # 知识目录（术语 / 文件线索）
    PRODUCT = "product"  # 产品说明与结构化统计


class EvidenceSourceType(StrEnum):
    """``Evidence.source_type`` 的取值全集（PRD 8.1）。

    = ``SourceType`` 的 6 类 + General Agent 的 2 类产出。``GENERAL_KNOWLEDGE``
    与 ``PROPOSAL`` 是模型解释/建议，不是文档证据，**不得生成文档引用编号**
    （PRD 8.1 末句），因此不在 :data:`CITATION_PREFIX` 中。
    """

    KNOWLEDGE = "knowledge"
    WEB = "web"
    MEMORY = "memory"
    HISTORY = "history"
    CATALOG = "catalog"
    PRODUCT = "product"
    GENERAL_KNOWLEDGE = "general_knowledge"
    PROPOSAL = "proposal"


class AnswerBasis(StrEnum):
    """正式答案的来源依据（PRD 8.3：``answer_basis`` 扩展为这 9 个取值）。

    多种来源同时成立时取 ``MIXED``，并由 ``basis_components`` 精确表达组合
    （PRD 8.3），因此 ``basis_components`` 的成员类型是 :class:`SourceType`。
    """

    KNOWLEDGE = "knowledge"
    WEB = "web"
    GENERAL = "general"
    PRODUCT = "product"
    CATALOG = "catalog"
    MEMORY = "memory"
    HISTORY = "history"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


# =====================================================================
# t1 冻结区 · 3. 会话策略（PRD 10.1 / 9.1）
# =====================================================================


class PermissionMode(StrEnum):
    """Agent 联网工具权限模式（PRD 10.1）。所有模式都执行人名脱敏与工作区范围。"""

    ASK = "ask"  # 请求批准：联网搜索及网页读取执行前暂停
    AUTO = "auto"  # 自动批准：审核节点确认符合当前问题后批准
    FULL = "full"  # 完全访问权限：自动执行已配置问答工具


class ReasoningLevel(StrEnum):
    """思考强度档位（PRD 9.1），直接映射 DeepSeek 原生档位与请求参数。"""

    OFF = "off"  # thinking.type=disabled，省略 reasoning_effort
    LOW = "low"  # thinking.type=enabled, reasoning_effort=low
    HIGH = "high"  # thinking.type=enabled, reasoning_effort=high
    MAX = "max"  # thinking.type=enabled, reasoning_effort=max


# =====================================================================
# t1 冻结区 · 4. 路由与 Agent 角色（PRD 6.2 / 4.2）
# =====================================================================


class IntentKind(StrEnum):
    """``RouteDecision.intent``（PRD 6.2）。"""

    GENERAL = "general"
    RESEARCH = "research"
    PRODUCT = "product"
    CATALOG = "catalog"
    MEMORY = "memory"
    CLARIFICATION = "clarification"


class Freshness(StrEnum):
    """``RouteDecision.freshness``（PRD 6.2）：时效要求。"""

    NONE = "none"
    CURRENT = "current"
    DATE_RANGE = "date_range"


class AgentKind(StrEnum):
    """Agent 职责（PRD 4.2）。默认共用同一个已配置模型。"""

    INTENT = "intent"  # Intent Agent
    PLANNER = "planner"  # Planner Agent
    GENERAL = "general"  # General Agent
    RAG = "rag"  # RAG Agent
    WEB = "web"  # Web Agent
    SYNTHESIS = "synthesis"  # Synthesis Agent
    VERIFIER = "verifier"  # Verifier Agent
    APPROVAL_REVIEWER = "approval_reviewer"  # Approval Reviewer
    MEMORY_EXTRACTOR = "memory_extractor"  # Memory Extractor


class VerifierVerdict(StrEnum):
    """核对结论（PRD 8.2）。``pass`` 是唯一允许发布的结论。"""

    PASS = "pass"
    REVISE = "revise"
    RESEARCH = "research"
    CLARIFY = "clarify"
    INSUFFICIENT = "insufficient"


# =====================================================================
# t1 冻结区 · 5. SSE 事件类型（PRD 13.3）
# =====================================================================


class EventType(StrEnum):
    """``run_events.type`` / SSE ``type`` 的取值全集（PRD 13.3）。

    这是**唯一**的事件类型来源：``app/agent/events.py`` 只定义统一事件对象与
    payload 模型并复用本枚举，不得另行定义第二份类型表。
    """

    RUN_STATUS = "run.status"
    ROUTE_DECIDED = "route.decided"
    PLAN_UPDATED = "plan.updated"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"
    INPUT_REQUESTED = "input.requested"
    INPUT_RESOLVED = "input.resolved"
    EVIDENCE_ADDED = "evidence.added"
    VERIFICATION_UPDATED = "verification.updated"
    ANSWER_META = "answer.meta"
    ANSWER_DELTA = "answer.delta"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_FAILED = "memory.failed"


#: 结束一次答案订阅的事件（PRD 13.3 末段）：收到后前端关闭该次 EventSource。
SUBSCRIPTION_CLOSING_EVENTS: Final[frozenset[EventType]] = frozenset(
    {EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}
)


# =====================================================================
# t1 冻结区 · 6. 状态机（PRD 12.3 迁移图）
# =====================================================================

#: ``run`` 状态迁移表。键为当前状态，值为允许到达的目标状态集合。
#: 与 PRD 12.3 stateDiagram 逐条对应；``cancelled → running`` 仅在
#: ``resumable=true`` 时成立，由 :func:`can_transition_run` 的 ``resumable`` 参数把关。
RUN_STATUS_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLING}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_CHILDREN,
            RunStatus.WAITING_APPROVAL,
            RunStatus.WAITING_INPUT,
            RunStatus.COMPLETED,
            RunStatus.COMPLETED_WITH_GAPS,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
            RunStatus.CANCELLING,
        }
    ),
    RunStatus.WAITING_CHILDREN: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLING}
    ),
    RunStatus.WAITING_APPROVAL: frozenset({RunStatus.RUNNING, RunStatus.CANCELLING}),
    RunStatus.WAITING_INPUT: frozenset({RunStatus.RUNNING, RunStatus.CANCELLING}),
    # 用户继续失败/中断/停止的任务：从记录位置重新驱动执行
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
    RunStatus.INTERRUPTED: frozenset({RunStatus.RUNNING}),
    RunStatus.CANCELLED: frozenset({RunStatus.RUNNING}),  # 仅 resumable=true
    RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED}),
    # 终态：PRD 12.3 中 completed / completed_with_gaps 指向 [*]
    RunStatus.COMPLETED: frozenset(),
    RunStatus.COMPLETED_WITH_GAPS: frozenset(),
}

#: 子任务使用与 run 相同的状态词汇与迁移规则（PRD 12.3 同一张表）。
TASK_STATUS_TRANSITIONS: Final[dict[TaskStatus, frozenset[TaskStatus]]] = {
    TaskStatus(status): frozenset(TaskStatus(target) for target in targets)
    for status, targets in RUN_STATUS_TRANSITIONS.items()
}

#: 终态：不再接受任何迁移。
TERMINAL_RUN_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}
)

#: 可以被用户「继续」的状态（PRD 12.3 / 12.4）。``cancelled`` 还需 ``resumable=true``。
RESUMABLE_RUN_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.CANCELLED}
)

#: 可以接受停止（取消）请求的活动状态（PRD 12.3「允许的后续操作」列）。
CANCELLABLE_RUN_STATUSES: Final[frozenset[RunStatus]] = frozenset(
    {
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.WAITING_CHILDREN,
        RunStatus.WAITING_APPROVAL,
        RunStatus.WAITING_INPUT,
    }
)

#: 状态的中文含义（PRD 12.3 表「含义」列），供事件与界面复用。
RUN_STATUS_LABELS: Final[dict[RunStatus, str]] = {
    RunStatus.QUEUED: "等待执行资源",
    RunStatus.RUNNING: "正在处理",
    RunStatus.WAITING_CHILDREN: "等待独立研究任务",
    RunStatus.WAITING_APPROVAL: "等待用户批准",
    RunStatus.WAITING_INPUT: "等待用户澄清",
    RunStatus.INTERRUPTED: "进程退出或上下文变更导致执行中断",
    RunStatus.FAILED: "模型、工具或运行系统明确失败",
    RunStatus.CANCELLING: "正在取消活动请求和子任务",
    RunStatus.CANCELLED: "用户停止或会话移动终止原执行",
    RunStatus.COMPLETED: "答案通过核对并保存",
    RunStatus.COMPLETED_WITH_GAPS: "已说明无法取得的关键资料",
}


def can_transition_run(
    current: RunStatus,
    target: RunStatus,
    *,
    resumable: bool = True,
) -> bool:
    """判断 ``run`` 状态迁移是否合法（PRD 12.3）。

    ``resumable`` 只影响 ``cancelled → running``：该迁移仅当 run 记录的
    ``resumable=true`` 时允许（用户停止与「会话移动」终止的 run 不可恢复）。
    """

    if target not in RUN_STATUS_TRANSITIONS.get(current, frozenset()):
        return False
    if current is RunStatus.CANCELLED and target is RunStatus.RUNNING:
        return resumable
    return True


def can_transition_task(
    current: TaskStatus,
    target: TaskStatus,
    *,
    resumable: bool = True,
) -> bool:
    """判断研究子任务状态迁移是否合法，规则同 :func:`can_transition_run`。"""

    return can_transition_run(
        RunStatus(current.value), RunStatus(target.value), resumable=resumable
    )


# =====================================================================
# t1 冻结区 · 7. 运行身份标识（PRD 12.1）
# =====================================================================

#: 主图 thread：``run:<run_id>:main``（PRD 12.1）。
MAIN_THREAD_TEMPLATE: Final[str] = "run:{run_id}:main"

#: 研究子图 thread：``run:<run_id>:task:<task_id>``（PRD 12.1）。
TASK_THREAD_TEMPLATE: Final[str] = "run:{run_id}:task:{task_id}"

_MAIN_SUFFIX: Final[str] = ":main"
_TASK_MARKER: Final[str] = ":task:"


def main_thread_id(run_id: str) -> str:
    """主图执行上下文 ID（PRD 12.1）。"""

    return MAIN_THREAD_TEMPLATE.format(run_id=run_id)


def task_thread_id(run_id: str, task_id: str) -> str:
    """研究子图执行上下文 ID（PRD 12.1）。``task_id`` 不得包含 ``:``。"""

    if not task_id or ":" in task_id:
        raise ValueError("task_id 必须非空且不能包含 ':'")
    return TASK_THREAD_TEMPLATE.format(run_id=run_id, task_id=task_id)


def is_main_thread_id(thread_id: str) -> bool:
    """是否为 ``run:<run_id>:main`` 形式的主图 thread。

    必须与 :func:`thread_of_run` 的解析结果一致：既校验 ``run:`` 前缀，也要求
    ``run_id`` 非空，并排除子图形式 ``run:<run_id>:task:<task_id>``（子图末尾是
    ``task_id``，因此 :data:`_TASK_MARKER` 要按「出现在 ``main`` 之前」判断，
    不能按 ``endswith`` 判断）。
    """

    if not thread_id.startswith("run:"):
        return False
    if thread_of_run(thread_id) is None:
        return False
    return thread_id.endswith(_MAIN_SUFFIX) and _TASK_MARKER not in thread_id


def thread_of_run(thread_id: str) -> str | None:
    """从 thread_id 解析所属 ``run_id``；格式不合法时返回 ``None``。

    解析只做形状校验（PRD 12.1 规则），不校验 run 是否存在于数据库。
    """

    if not thread_id.startswith("run:"):
        return None
    if thread_id.endswith(_MAIN_SUFFIX):
        run_id = thread_id[len("run:") : -len(_MAIN_SUFFIX)]
    elif _TASK_MARKER in thread_id:
        run_id = thread_id[len("run:") : thread_id.index(_TASK_MARKER)]
    else:
        return None
    return run_id or None


# =====================================================================
# t1 冻结区 · 8. 引用编号（PRD 8.3）
# =====================================================================

#: 正文引用前缀：知识 ``[S1]``、网页 ``[W1]``、记忆 ``[M1]``、历史 ``[H1]``（PRD 8.3）。
#: 编号在最终候选答案形成时分配，复用同一来源不重新编号；其余来源类型
#: （catalog / product / general_knowledge / proposal）不生成编号。
CITATION_PREFIX: Final[dict[SourceType, str]] = {
    SourceType.KNOWLEDGE: "S",
    SourceType.WEB: "W",
    SourceType.MEMORY: "M",
    SourceType.HISTORY: "H",
}


def citation_prefix(source_type: SourceType | EvidenceSourceType) -> str | None:
    """返回引用编号前缀；该来源类型不参与编号时返回 ``None``。"""

    try:
        authoritative = SourceType(source_type.value)
    except ValueError:
        return None
    return CITATION_PREFIX.get(authoritative)


# =====================================================================
# t1 冻结区 · 9. 会话级默认值（PRD 18.1）
# =====================================================================

#: 新会话默认权限模式（PRD 10.1 / 18.1）：请求批准。
DEFAULT_PERMISSION_MODE: Final[PermissionMode] = PermissionMode.ASK

#: 新会话默认思考强度（PRD 9.1 / 18.1）：深度思考。
DEFAULT_REASONING_LEVEL: Final[ReasoningLevel] = ReasoningLevel.HIGH


# =====================================================================
# t2 追加区（T00b 拥有）· 1. 版本、默认值与内存 schema 常量
# =====================================================================
# 上方属 t1 冻结区：枚举取值、状态迁移表、thread_id 规则与引用前缀只读复用。
# 本节只新增常量与 Pydantic 模型，不改动 t1 语义（`__all__` 仅追加新名字）。

#: `app.db` schema 版本（写入 `PRAGMA user_version`），建表幂等靠它判定。
#: v2（T_d29 / D29）：`workspaces` 追加 `pinned` 与 `order_index` —— 侧边栏置顶与手工
#: 位次从 JSON 存档迁入 app.db 权威列（此前 app.db 无处存这两项，工作区写路径无法收口）。
#: v1 老库由 `ensure_schema` 幂等 `ALTER TABLE ... ADD COLUMN`（带默认值）就地升级，
#: 不重写数据、不改变既有行为。
#: v3（T_taskid / t30）：`agent_tasks` 的**身份改为 run 作用域**（`PRIMARY KEY (run_id, id)`）
#: 并把 `parent_task_id` 自引用与 `operations` / `approvals` / `input_requests` 对任务的引用
#: 改为**复合**外键 `(run_id, task_id) → agent_tasks(run_id, id)`。v2 老库由
#: `upgrade_schema` 幂等**重建表**（SQLite 不能 ALTER 主键/外键）后就地升级，既有行原样保留。
APP_DB_SCHEMA_VERSION: Final[int] = 3

#: 统一事件对象的 `schema_version`（PRD 13.3）。
EVENT_SCHEMA_VERSION: Final[int] = 2

#: `memory.json` 文件格式版本（PRD 11.2）。
MEMORY_FILE_SCHEMA_VERSION: Final[int] = 1

#: `memory.json` 文件级字段（PRD 11.2，9 个；顺序即落盘顺序）。
MEMORY_FILE_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "enabled",
    "revision",
    "invalidation_revision",
    "updated_at",
    "entries",
    "invalidations",
    "processed_cursors",
    "applied_operations",
)

#: `memory.json` 条目级字段（PRD 11.2，12 个）。
MEMORY_ENTRY_KEYS: Final[tuple[str, ...]] = (
    "id",
    "kind",
    "text",
    "source_conversation_id",
    "source_message_ids",
    "created_at",
    "updated_at",
    "user_pinned",
    "status",
    "superseded_by",
    "operation_id",
)
# 注：``MemoryInvalidation`` 另有 5 个字段（memory_id / source_message_ids /
# quote / kind / replacement），PRD 11.4 只规定「记忆 ID、来源消息 ID、原文位置、
# 撤销类型及替代内容」，未给字段名表，故不进 ``MEMORY_*_KEYS`` 常量。

# ---- PRD 18.1 工程默认值（跨工作流唯一来源）----

#: 每个任务同时运行的研究 Agent 上限（PRD 6.4 / 18.1）。
DEFAULT_RESEARCH_AGENT_CONCURRENCY: Final[int] = 3

#: 应用同时执行的模型请求上限（PRD 6.4 / 18.1）。
DEFAULT_MODEL_REQUEST_CONCURRENCY: Final[int] = 4

#: 每次知识检索的上下文候选数（PRD 7.2 / 18.1）。
DEFAULT_CONTEXT_TOP_K: Final[int] = 8

#: 单条记忆长度上限（PRD 11.2 / 18.1）。
DEFAULT_MEMORY_MAX_CHARS: Final[int] = 300

#: 自动构建的会话快照长度上限（PRD 11.2 / 18.1）。
DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS: Final[int] = 8000

#: 用户单条输入长度上限（PRD 13.2 / 18.1）。
DEFAULT_USER_INPUT_MAX_CHARS: Final[int] = 32000

#: 上下文整理阈值：可用输入空间占比（PRD 9.3 / 18.1）。
DEFAULT_CONTEXT_COMPACTION_RATIO: Final[float] = 0.8


def utc_now_iso() -> str:
    """统一时间戳格式：UTC ISO-8601 到秒（``2026-09-19T08:00:00Z``）。

    PRD 未规定时间格式；全链路统一用字符串而非 datetime，便于 SQLite 排序、
    JSON 落盘与前端直接展示（与 V1.1.0 既有 `created_at: str` 约定一致）。
    """

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# =====================================================================
# t2 追加区 · 2. 补充枚举（PRD 5.4 / 11.1 / 11.2 / 11.4 / 8.2 / 9.2 / 10.2 / 14.4）
# =====================================================================


class WorkspaceStatus(StrEnum):
    """工作区状态（PRD 5.4：删除工作区采用可恢复的归档处理）。"""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationStatus(StrEnum):
    """会话状态（PRD 5.4 / 15.1：归档会话保留历史与检索能力）。"""

    ACTIVE = "active"
    ARCHIVED = "archived"


class OperationStatus(StrEnum):
    """模型或工具操作的执行状态（PRD 12.1 / 12.4：操作记录与重放复用）。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ApprovalDecisionKind(StrEnum):
    """批准按钮语义（PRD 10.2）：允许本次 / 允许本轮同类操作 / 拒绝。"""

    ALLOW_ONCE = "allow_once"
    ALLOW_SCOPE = "allow_scope"
    DENY = "deny"


class CitationSourceStatus(StrEnum):
    """引用卡片来源状态（PRD 14.4：知识已删除或原工作区已归档时保留片段并标记）。"""

    ACTIVE = "active"
    DELETED = "deleted"
    ARCHIVED = "archived"


class VerificationCheck(StrEnum):
    """核对项目（PRD 8.2 表 8 行）。"""

    REQUIREMENT_COVERAGE = "requirement_coverage"  # 问题覆盖
    FACTUAL_SUPPORT = "factual_support"  # 事实支持
    NUMBERS_AND_UNITS = "numbers_and_units"  # 数字与单位
    VERSION_AND_SCOPE = "version_and_scope"  # 版本与范围
    SOURCE_CONFLICT = "source_conflict"  # 来源冲突
    INTERNAL_FACT = "internal_fact"  # 内部事实
    CITATION_VALIDITY = "citation_validity"  # 引用有效性
    MATERIAL_STATUS = "material_status"  # 资料状态


class VerificationSeverity(StrEnum):
    """核对问题严重度（PRD 8.2：每项含建议动作，未定义等级，此处仅供排序展示）。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MemoryKind(StrEnum):
    """工作区记忆类别（PRD 11.1 表 4 行）。"""

    RESPONSE_PREFERENCE = "response_preference"
    TOPIC = "topic"
    PROGRESS = "progress"
    CONFIRMED_CONCLUSION = "confirmed_conclusion"


class MemoryEntryStatus(StrEnum):
    """记忆条目状态（PRD 11.2：active / forgotten / superseded / orphaned）。"""

    ACTIVE = "active"
    FORGOTTEN = "forgotten"
    SUPERSEDED = "superseded"
    ORPHANED = "orphaned"


class MemoryInvalidationKind(StrEnum):
    """记忆失效类型（PRD 11.4：人工纠正 / 删除 / 关闭记忆）。"""

    CORRECTED = "corrected"
    FORGOTTEN = "forgotten"
    DISABLED = "disabled"


class BackgroundJobStatus(StrEnum):
    """后台任务状态（PRD 11.5：提炼任务状态单独展示）。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =====================================================================
# t2 追加区 · 3. 契约模型基类
# =====================================================================


class ContractModel(BaseModel):
    """全部共享契约模型的基类。

    取舍（写进架构文档决策记录）：``extra="forbid"`` —— PRD 12.5 要求「任何节点
    返回的结构化内容不合法时就地失败，不能用空对象或通用回答替代」，因此多传
    未声明字段视为契约违规并直接报错。需要新增字段时向 contracts-db 提需求，
    不要在别的模块另建同义模型。
    """

    model_config = ConfigDict(extra="forbid")


# =====================================================================
# t2 追加区 · 4. 路由、计划与研究（PRD 6.2 / 6.4）
# =====================================================================


class RouteDecision(ContractModel):
    """意图识别输出（PRD 6.2 表，逐字对齐 12 个字段）。"""

    question: str
    resolved_question: str
    intent: IntentKind
    needs_knowledge: bool = False
    needs_web: bool = False
    needs_history: bool = False
    needs_general: bool = False
    freshness: Freshness = Freshness.NONE
    entities: list[str] = Field(default_factory=list)
    scope_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    reason_summary: str = ""

    def needs_any_source(self) -> bool:
        """是否至少选择了一个来源路径（``clarification`` 之外必须为真）。"""

        return (
            self.needs_knowledge
            or self.needs_web
            or self.needs_history
            or self.needs_general
        )


class ResearchTask(ContractModel):
    """一个研究子任务（PRD 6.4 表，逐字对齐 9 个字段）。"""

    task_id: str
    parent_task_id: str | None = None
    question: str
    requirement_ids: list[str] = Field(default_factory=list)
    agent_kind: AgentKind = AgentKind.RAG
    depends_on: list[str] = Field(default_factory=list)
    expected_evidence: str = ""
    status: TaskStatus = TaskStatus.QUEUED
    result_ref: str | None = None

    @model_validator(mode="after")
    def _no_self_dependency(self) -> ResearchTask:
        if self.task_id in self.depends_on:
            raise ValueError(f"任务 {self.task_id} 不能依赖自身")
        if self.parent_task_id == self.task_id:
            raise ValueError(f"任务 {self.task_id} 不能以自身为父任务")
        return self


def validate_task_graph(tasks: Sequence[ResearchTask]) -> None:
    """校验任务图：ID 唯一、父任务与依赖存在、无依赖环（PRD 6.4 末段）。

    失败时抛 :class:`ValueError`，由调用方（Planner）就地失败，不返回空计划。
    """

    by_id: dict[str, ResearchTask] = {}
    for task in tasks:
        if task.task_id in by_id:
            raise ValueError(f"任务 ID 重复: {task.task_id}")
        by_id[task.task_id] = task

    for task in tasks:
        if task.parent_task_id is not None and task.parent_task_id not in by_id:
            raise ValueError(f"任务 {task.task_id} 的父任务不存在: {task.parent_task_id}")
        for dependency in task.depends_on:
            if dependency not in by_id:
                raise ValueError(f"任务 {task.task_id} 依赖的任务不存在: {dependency}")

    state: dict[str, int] = {}

    def visit(task_id: str) -> None:
        status = state.get(task_id, 0)
        if status == 1:
            raise ValueError(f"任务依赖存在环: {task_id}")
        if status == 2:
            return
        state[task_id] = 1
        for dependency in by_id[task_id].depends_on:
            visit(dependency)
        state[task_id] = 2

    for task_id in by_id:
        visit(task_id)


# =====================================================================
# t2 追加区 · 5. 证据、核对与答案（PRD 8.1 / 8.2 / 8.3）
# =====================================================================


class Evidence(ContractModel):
    """一条证据（PRD 8.1，逐字对齐 12 个字段）。"""

    evidence_id: str
    workspace_id: str
    run_id: str
    task_id: str | None = None
    source_type: EvidenceSourceType
    source_ref: str
    content: str
    content_hash: str
    document_revision: str = ""
    retrieved_at: str
    supported_requirements: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class CitationEntry(ContractModel):
    """答案正文里的一条引用（PRD 8.3 编号 + 14.4 引用卡片）。"""

    label: str  # 形如 "S1" / "W2" / "M1" / "H1"
    source_type: SourceType
    evidence_id: str
    workspace_id: str
    workspace_name: str = ""
    source_ref: str = ""
    snippet: str = ""
    source_status: CitationSourceStatus = CitationSourceStatus.ACTIVE


class WebReadRange(ContractModel):
    """一次网页读取覆盖的字符范围（PRD 7.3「读取范围」）。

    本版由 T04 在 `app/agent/tools.py` 定义，T_contract 升格为共享契约
    （架构文档 §6 D32）：T07 的 Web Agent 产出它、Synthesis / Verifier 读它、
    前端引用卡片展示它、t11 的 A01 / A11 验收断言它 —— 都是跨模块数据，
    不应让下游 import 工具层内部结构。`tools.py` 保留同名别名再导出，
    既有 `from app.agent.tools import WebReadRange` 调用方零改动。
    """

    offset: int = Field(default=0, ge=0, description="本次窗口起始字符位置。")
    length: int = Field(default=0, ge=0, description="本次窗口字符数。")
    total: int = Field(default=0, ge=0, description="全文总字符数。")
    complete: bool = Field(default=False, description="本窗口是否覆盖全文。")


class WebPageEvidence(ContractModel):
    """网页证据对象（PRD 7.3 字段表）：``url`` / ``final_url`` / ``title`` /
    ``published_at`` / ``retrieved_at`` / ``content_hash`` / ``artifact_id`` /
    读取范围；缺失的来源字段保存为空（PRD 7.3 末段），日期含义由原文确认。

    升格说明同 :class:`WebReadRange`（§6 D32）。字段名、默认值与校验语义与
    T04 原定义逐字一致，`model_dump()` 形状不变。
    """

    url: str = ""
    final_url: str = ""
    title: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    artifact_id: str = ""
    read_range: WebReadRange = Field(default_factory=WebReadRange)
    content_complete: bool = False
    #: 本次窗口正文（不是全文；全文在 artifact 里）。
    content: str = ""


class VerificationIssue(ContractModel):
    """核对发现的一项问题（PRD 8.2：受影响要求 + 陈述位置 + 证据 ID + 建议动作）。"""

    check: VerificationCheck
    requirement_ids: list[str] = Field(default_factory=list)
    statement: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    required_action: str = ""
    severity: VerificationSeverity = VerificationSeverity.MEDIUM


class VerificationResult(ContractModel):
    """Verifier 输出（PRD 8.2）。``verdict=pass`` 是唯一允许发布的结论。"""

    verdict: VerifierVerdict
    issues: list[VerificationIssue] = Field(default_factory=list)
    covered_requirements: list[str] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    checked_at: str = ""

    def is_publishable(self) -> bool:
        return self.verdict is VerifierVerdict.PASS


class AnswerArtifact(ContractModel):
    """已核对通过的答案（PRD 8.3 / 13.1 ``answer_artifacts``）。"""

    answer_id: str
    run_id: str
    workspace_id: str = ""
    content: str
    citations: list[CitationEntry] = Field(default_factory=list)
    verification: VerificationResult | None = None
    content_hash: str = ""
    answer_basis: AnswerBasis = AnswerBasis.INSUFFICIENT
    basis_components: list[SourceType] = Field(default_factory=list)
    status: RunStatus = RunStatus.COMPLETED
    created_at: str = ""

    @model_validator(mode="after")
    def _status_must_be_published_state(self) -> AnswerArtifact:
        allowed = {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}
        if self.status not in allowed:
            raise ValueError("答案状态只能是 completed 或 completed_with_gaps（PRD 8.3）")
        return self


# =====================================================================
# t2 追加区 · 6. 批准与澄清（PRD 10.2 / 13.2）
# =====================================================================


class ApprovalDisplay(ContractModel):
    """批准卡片展示内容（PRD 10.2：目的、工具、查询或 URL、日期范围、子任务、脱敏提示）。"""

    purpose: str = ""
    tool_name: str = ""
    query_or_urls: list[str] = Field(default_factory=list)
    date_range: str = ""
    task_question: str = ""
    redaction_notice: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ApprovalScope(ContractModel):
    """「允许本轮同类操作」的授权范围（PRD 10.2：run、工具类别、问题范围、规则版本）。"""

    run_id: str
    tool_category: str
    question_scope: str
    rule_version: str


class ApprovalDecision(ContractModel):
    """用户决定（PRD 10.2：允许本次 / 允许本轮同类操作 / 拒绝，可带修改后的参数）。"""

    kind: ApprovalDecisionKind
    edited_parameters: dict[str, Any] | None = None
    decided_at: str = ""
    decided_by: str = "user"
    note: str = ""


class ApprovalRequest(ContractModel):
    """批准请求（PRD 10.2 全部字段 + 13.1 ``approvals``）。"""

    approval_id: str
    run_id: str
    task_id: str | None = None
    interrupt_id: str = ""
    operation_id: str
    args_hash: str
    context_epoch: int = 1
    rule_version: str = ""
    display: ApprovalDisplay = Field(default_factory=ApprovalDisplay)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: ApprovalDecision | None = None
    scope: ApprovalScope | None = None
    #: 本版新增字段（PRD 10.2 未列）：解释 ``superseded`` 的原因。PRD 5.4 与 11.4
    #: 要求批准失效时说明来源（会话移动 / 记忆失效），仅凭状态无法表达原因。
    superseded_reason: str = ""
    created_at: str = ""
    resolved_at: str | None = None

    @model_validator(mode="after")
    def _decision_matches_status(self) -> ApprovalRequest:
        if self.status is ApprovalStatus.PENDING and self.decision is not None:
            raise ValueError("pending 批准不能带用户决定")
        if (
            self.status in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED}
            and self.decision is None
        ):
            raise ValueError("approved/denied 批准必须带用户决定")
        return self


class InputRequest(ContractModel):
    """澄清请求（PRD 13.1 ``input_requests`` / 13.2 提交回答与问题版本）。"""

    input_request_id: str
    run_id: str
    task_id: str | None = None
    question: str
    options: list[str] = Field(default_factory=list)
    reason: str = ""
    status: InputRequestStatus = InputRequestStatus.PENDING
    answer: str | None = None
    question_version: int = Field(default=1, ge=1)
    created_at: str = ""
    resolved_at: str | None = None

    @model_validator(mode="after")
    def _answer_matches_status(self) -> InputRequest:
        if self.status is InputRequestStatus.ANSWERED and self.answer is None:
            raise ValueError("answered 澄清必须带回答")
        if self.status is InputRequestStatus.PENDING and self.answer is not None:
            raise ValueError("pending 澄清不能带回答")
        return self


# =====================================================================
# t2 追加区 · 7. 模型协议与操作记录（PRD 9.2 / 12.1 / 12.5）
# =====================================================================


class ModelCapabilities(ContractModel):
    """Provider 能力声明（PRD 9.2，逐字对齐 7 个字段）。"""

    supports_tools: bool = False
    supports_thinking_toggle: bool = False
    reasoning_levels: list[ReasoningLevel] = Field(default_factory=list)
    supports_image_input: bool = False
    context_limit: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    history_protocol: str = ""

    def supports_level(self, level: ReasoningLevel) -> bool:
        """界面只显示可用选项（PRD 9.2）；不支持所选功能时由调用方返回配置错误。"""

        return level in self.reasoning_levels

    @model_validator(mode="after")
    def _thinking_levels_consistent(self) -> ModelCapabilities:
        extra = set(self.reasoning_levels) - {ReasoningLevel.OFF}
        if extra and not self.supports_thinking_toggle:
            raise ValueError("声明了思考档位就必须 supports_thinking_toggle=True")
        return self


class ToolCall(ContractModel):
    """模型返回的一次工具调用（PRD 9.2：按 ``index`` 汇集流式参数）。"""

    index: int = Field(default=0, ge=0)
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""
    complete: bool = True


class ProviderMessage(ContractModel):
    """Provider 协议消息（PRD 13.1 ``provider_messages``）。"""

    context_id: str
    context_epoch: int = Field(default=1, ge=0)
    agent_context_epoch: int = Field(default=1, ge=0)
    sequence: int = Field(default=0, ge=0)
    protocol: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    execution_id: str = ""
    created_at: str = ""


class OperationRecord(ContractModel):
    """一次模型或工具操作的记录（PRD 12.1 / 12.4：``operation_id`` 唯一，重放复用）。"""

    operation_id: str
    run_id: str | None = None
    task_id: str | None = None
    tool_name: str
    args_hash: str
    status: OperationStatus = OperationStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    result_ref: str | None = None
    created_at: str = ""
    updated_at: str = ""


class AgentError(ContractModel):
    """错误对象（PRD 12.5：错误码、用户说明、失败节点、操作 ID、可否继续、所需操作）。"""

    error_code: str
    message: str
    node: str = ""
    operation_id: str | None = None
    resumable: bool = False
    required_action: str = ""
    created_at: str = ""


# =====================================================================
# t2 追加区 · 8. 工作区上下文视图（PRD 5.1 / 5.2）
# =====================================================================


class WorkspaceContext(ContractModel):
    """不可变工作区上下文**数据视图**（PRD 5.2）。

    PRD 5.2 要求上下文覆盖「身份、目录、当前知识版本、服务实例和工作区默认设置」。
    服务实例（Chroma / retriever / embedder）不是可序列化契约，因此本模型只承载
    身份、目录、版本与默认设置；T01 的运行时上下文必须以本模型为数据形状持有
    服务句柄（字段名以此为准），工具执行器从中取 ``workspace_id``，模型参数里
    不出现 ``workspace_id``。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    storage_key: str
    name: str = ""
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    knowledge_revision: int = Field(default=0, ge=0)
    default_permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL
    memory_enabled: bool = True
    data_dir: Path
    root_dir: Path

    # ---- 目录（PRD 5.1）----

    @property
    def documents_dir(self) -> Path:
        return self.root_dir / "documents"

    @property
    def state_db(self) -> Path:
        return self.root_dir / "state.db"

    @property
    def image_state_db(self) -> Path:
        return self.root_dir / "image_state.db"

    @property
    def chroma_dir(self) -> Path:
        return self.root_dir / "chroma"

    @property
    def memory_json(self) -> Path:
        return self.root_dir / "memory.json"

    @property
    def checkpoints_db(self) -> Path:
        return self.root_dir / "checkpoints.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def jieba_cache_dir(self) -> Path:
        return self.root_dir / "cache" / "jieba"

    @classmethod
    def build(
        cls,
        *,
        workspace_id: str,
        storage_key: str,
        data_dir: Path | str,
        name: str = "",
        **kwargs: Any,
    ) -> WorkspaceContext:
        """按 PRD 5.1 由 ``storage_key`` 构建目录树（``data_dir/workspaces/<key>/``）。"""

        base = Path(data_dir)
        return cls(
            workspace_id=workspace_id,
            storage_key=storage_key,
            name=name,
            data_dir=base,
            root_dir=base / "workspaces" / storage_key,
            **kwargs,
        )


@runtime_checkable
class WorkspaceServiceHub(Protocol):
    """PRD 5.2 的「**服务实例**」在实现层的契约（不进 Pydantic 数据模型）。

    取舍：``WorkspaceContext`` 是可序列化、可冻结的数据视图；Chroma / retriever /
    embedder / ``app.db`` 访问器都是 IO 对象，放进 Pydantic 会破坏「不可变数据」
    与「不含 IO」两条约定，因此服务实例由实现层
    ``app/core/workspace_context.py``（T01）持有，并通过本 Protocol 冻结成员名，
    避免 t4 与 t9 各自决定：

    - ``view``：上述数据视图（身份、目录、知识版本、默认设置）。
    - ``app_store``：该 ``data_dir/app.db`` 的访问层实例；**生命周期由 API lifespan
      创建的单实例持有**（RunManager 与工具执行器复用同一个），上下文只引用不拥有。

    实现层可以额外持有 ``retriever`` / ``vector_store`` / ``embedder`` 等句柄，
    但上面两个成员名由契约冻结；需要改名或新增跨模块成员时向 contracts-db 提需求。
    """

    view: WorkspaceContext
    app_store: AppStore


# =====================================================================
# t2 追加区 · 9. 工作区记忆文件 schema（PRD 11.1 / 11.2 / 11.4）
# =====================================================================


class MemoryFileEntry(ContractModel):
    """``memory.json`` 的一条记忆（PRD 11.2 条目级 12 字段）。

    长度上限（单条 300 字符，PRD 11.2 / 18.1）**不在模型层强制**：用户主动保存
    的内容可以超限，超限时的提示与整理入口由 T08 负责。
    """

    id: str
    kind: MemoryKind
    text: str
    source_conversation_id: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    user_pinned: bool = False
    status: MemoryEntryStatus = MemoryEntryStatus.ACTIVE
    superseded_by: str | None = None
    operation_id: str = ""

    @model_validator(mode="after")
    def _superseded_needs_target(self) -> MemoryFileEntry:
        if self.status is MemoryEntryStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError("superseded 记忆必须给出 superseded_by")
        return self


class MemoryInvalidation(ContractModel):
    """记忆失效记录（PRD 11.4：记忆 ID、来源消息 ID、原文位置、撤销类型、替代内容）。"""

    memory_id: str
    source_message_ids: list[str] = Field(default_factory=list)
    quote: str = ""
    kind: MemoryInvalidationKind
    replacement: str = ""
    created_at: str = ""


class MemoryFile(ContractModel):
    """``memory.json`` 根对象（PRD 11.2 文件级 9 字段）。"""

    schema_version: int = MEMORY_FILE_SCHEMA_VERSION
    enabled: bool = True
    revision: int = Field(default=0, ge=0)
    invalidation_revision: int = Field(default=0, ge=0)
    updated_at: str = ""
    entries: list[MemoryFileEntry] = Field(default_factory=list)
    invalidations: list[MemoryInvalidation] = Field(default_factory=list)
    #: 会话 → 已提炼到的消息 ID（PRD 11.5 ``processed_cursors``）。
    processed_cursors: dict[str, str] = Field(default_factory=dict)
    #: 已应用的操作 ID（PRD 11.2 ``applied_operations``，去重防重复后台任务）。
    applied_operations: list[str] = Field(default_factory=list)


# =====================================================================
# t2 追加区 · 10. 校验辅助
# =====================================================================


def dump_contract(model: ContractModel, *, exclude_none: bool = False) -> dict[str, Any]:
    """把契约模型转成 JSON 安全的 dict（统一给 SQLite JSON 列与 SSE 负载用）。"""

    return model.model_dump(mode="json", exclude_none=exclude_none)


def validate_contract(model_type: type[ContractModel], data: Mapping[str, Any]) -> ContractModel:
    """校验外部数据（模型输出 / 种子 JSON）是否满足契约，失败抛 ``ValidationError``。

    PRD 12.5：结构化内容不合法时就地失败，绝不用空对象或通用回答替代。
    """

    return model_type.model_validate(data)


__all__ += [
    "APP_DB_SCHEMA_VERSION",
    "DEFAULT_CONTEXT_COMPACTION_RATIO",
    "DEFAULT_CONTEXT_TOP_K",
    "DEFAULT_MEMORY_MAX_CHARS",
    "DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS",
    "DEFAULT_MODEL_REQUEST_CONCURRENCY",
    "DEFAULT_RESEARCH_AGENT_CONCURRENCY",
    "DEFAULT_USER_INPUT_MAX_CHARS",
    "EVENT_SCHEMA_VERSION",
    "MEMORY_ENTRY_KEYS",
    "MEMORY_FILE_KEYS",
    "MEMORY_FILE_SCHEMA_VERSION",
    "AgentError",
    "AnswerArtifact",
    "ApprovalDecision",
    "ApprovalDecisionKind",
    "ApprovalDisplay",
    "ApprovalRequest",
    "ApprovalScope",
    "BackgroundJobStatus",
    "CitationEntry",
    "CitationSourceStatus",
    "ContractModel",
    "ConversationStatus",
    "Evidence",
    "InputRequest",
    "MemoryEntryStatus",
    "MemoryFile",
    "MemoryFileEntry",
    "MemoryInvalidation",
    "MemoryInvalidationKind",
    "MemoryKind",
    "ModelCapabilities",
    "OperationRecord",
    "OperationStatus",
    "ProviderMessage",
    "ResearchTask",
    "RouteDecision",
    "ToolCall",
    "VerificationCheck",
    "VerificationIssue",
    "VerificationResult",
    "VerificationSeverity",
    "WebPageEvidence",
    "WebReadRange",
    "WorkspaceContext",
    "WorkspaceServiceHub",
    "WorkspaceStatus",
    "dump_contract",
    "utc_now_iso",
    "validate_contract",
    "validate_task_graph",
]
