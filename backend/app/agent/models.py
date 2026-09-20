"""Agent 运行期内部模型：LangGraph state 与节点间 DTO（PRD 4.3、12.2）。Owner: T02。

与 :mod:`app.agent.contracts` 的分工：
- ``contracts.py`` = **跨工作流冻结契约**（枚举、状态机、Pydantic 数据模型），
  多工作流共同读写，取值改动需同步 PRD。
- ``models.py`` = **图内部状态**（``AgentState`` / ``ResearchState`` / 节点入参出参
  DTO），只服务于 graph/runtime 内部，不跨工作流复用。

状态形状的三条取舍（写进架构文档 §6 D34）：

1. state 用 ``TypedDict`` 而不是 Pydantic 模型 —— LangGraph 的 channel 语义按字段
   做 dict 合并，TypedDict 是它的原生形状；Pydantic 模型会在每次节点返回时重建。
2. state 里只放**可序列化**的值（dict/list/str/bool/int）：checkpoint 要落
   ``checkpoints.db``，放 Pydantic 对象或服务句柄会让恢复路径依赖运行期对象图。
   跨节点传递的契约对象统一用 ``model_dump(mode="json")`` 后的 dict，需要强类型时
   由节点自己 ``model_validate`` 回来。
3. 服务句柄（ToolRegistry / AppStore / Provider）**不进 state**，由
   :class:`GraphServices` 在编译期闭包注入（见 ``graph.py``）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, TypedDict

from pydantic import ConfigDict, Field

from app.agent.contracts import AgentKind, ContractModel, RunStatus, TaskStatus

__all__ = [
    "AGENT_PHASES",
    "PHASE_LABELS",
    "AgentState",
    "GraphServices",
    "NodeOutcome",
    "PendingApproval",
    "ResearchState",
    "new_agent_state",
    "new_research_state",
    "run_status_of",
]


#: 阶段取值（PRD 4.3 的节点顺序 + 两个暂停阶段）。刻意用普通字符串常量而不是
#: ``StrEnum``：``RunStatusPayload.phase`` 是自由字符串（PRD 13.3 未固定取值表），
#: 这里只给运行层一个统一词表，避免各处手写。
AGENT_PHASES: Final[tuple[str, ...]] = (
    "load_snapshot",
    "route",
    "plan",
    "research",
    "synthesis",
    "verification",
    "approval",
    "clarify",
    "publish",
    "done",
)

#: 阶段 → 对外展示名（``run.status.message`` 用中文阶段说明，PRD 13.3）。
PHASE_LABELS: Final[dict[str, str]] = {
    "load_snapshot": "读取工作区记忆与会话快照",
    "route": "识别意图与来源",
    "plan": "拆解子问题",
    "research": "执行研究与查资料",
    "synthesis": "汇总候选答案",
    "verification": "核对候选答案",
    "approval": "等待用户批准联网操作",
    "clarify": "等待用户澄清问题",
    "publish": "保存答案",
    "done": "已完成",
}


class PendingApproval(ContractModel):
    """state 里挂起的批准请求指针（卡片本体已保存在 ``approvals`` 表）。

    节点**先持久化** ``ApprovalRequest`` 再 ``interrupt()``：中断节点在恢复时会从头
    重跑（PRD 10.3），因此本对象必须能在重跑时用同一 ``approval_id`` 读回原请求，
    否则会生成第二张卡片。
    """

    approval_id: str
    tool_name: str
    operation_id: str = ""
    args_hash: str = ""
    args_summary: str = ""
    task_id: str | None = None
    category: str = ""
    question_scope: str = ""


class NodeOutcome(ContractModel):
    """节点返回给运行层的诊断摘要（不含正文；正文走事件与答案表）。"""

    node: str
    status: TaskStatus = TaskStatus.COMPLETED
    summary: str = ""
    error_code: str = ""
    resumable: bool = False

    def is_failure(self) -> bool:
        return self.status is TaskStatus.FAILED


class AgentState(TypedDict, total=False):
    """主图 state（PRD 4.3）。``total=False``：只有走过的节点才有对应键。

    ★ = 由 RunManager 在起点写入，其余由节点写：

    - ★ ``run_id`` / ★ ``workspace_id`` / ★ ``conversation_id`` / ★ ``question``
    - ``phase``：当前阶段，见 :data:`AGENT_PHASES`
    - ``history``：当前上下文段的历史轮次 ``[{"role","content"}]``
    - ``memory_context``：工作区记忆固定快照文本（PRD 9.3 输入顺序第 3 位）
    - ``knowledge_revision``：本次运行看到的知识版本（重放复用要带上它）
    - ``decision``：``RouteDecision.model_dump(mode="json")``
    - ``plan``：``Plan.model_dump(mode="json")``
    - ``requirement_texts``：``{requirement_id: 文本}``，核对与渲染共用
    - ``task_order``：子任务执行顺序（planner ``levels`` 展平）
    - ``task_results``：``{task_id: ResearchResult.model_dump(mode="json")}``
    - ``completed_task_ids`` / ``failed_task_ids``
    - ``resume_from``：用户「继续」时从哪个阶段续跑
    - ``pending_approval``：:class:`PendingApproval` 的 dump；非空即等批准
    - ``approval_grants``：本轮已授权的同类操作范围（``ApprovalScope`` 的 dump 列表）
    - ``clarification``：``InputRequest.model_dump(mode="json")``；非空即等澄清
    - ``candidate``：``CandidateAnswer.model_dump(mode="json")``
    - ``verification``：``VerificationResult.model_dump(mode="json")``
    - ``answer_id`` / ``answer_status``：发布结果
    - ``revisions``：revise / research 回边计数（防死循环，PRD 7.4）
    - ``error``：``AgentError.model_dump(mode="json")``
    - ``outcomes``：节点诊断摘要列表（``NodeOutcome`` 的 dump）
    """

    run_id: str
    workspace_id: str
    conversation_id: str
    question: str
    phase: str
    history: list[dict[str, str]]
    memory_context: str
    knowledge_revision: int
    decision: dict[str, Any]
    plan: dict[str, Any]
    requirement_texts: dict[str, str]
    task_order: list[str]
    task_results: dict[str, dict[str, Any]]
    completed_task_ids: list[str]
    failed_task_ids: list[str]
    resume_from: str
    pending_approval: dict[str, Any]
    approval_decision: dict[str, Any]
    approval_grants: list[dict[str, Any]]
    clarification: dict[str, Any]
    candidate: dict[str, Any]
    verification: dict[str, Any]
    answer_id: str
    answer_status: str
    revisions: int
    error: dict[str, Any]
    outcomes: list[dict[str, Any]]


class ResearchState(TypedDict, total=False):
    """研究子图 state（PRD 7.2 步骤序列；一个 ``ResearchTask`` 一个 thread）。

    - ``task_id`` / ``question`` / ``agent_kind``：任务身份
    - ``run_id`` / ``workspace_id``：子图必须自带身份，否则工具上下文无法组装
      （子图有独立 ``thread_id``，不能从主图 state 里"顺手"取）
    - ``task``：``ResearchTask.model_dump(mode="json")``
    - ``request``：``ResearchRequest.model_dump(mode="json")``
    - ``rounds``：已完成轮数（``max_rounds`` 是子图的运行保护）
    - ``result``：``ResearchResult.model_dump(mode="json")``
    - ``status`` / ``stop_reason`` / ``error``：收尾信息
    """

    task_id: str
    question: str
    agent_kind: str
    run_id: str
    workspace_id: str
    task: dict[str, Any]
    request: dict[str, Any]
    rounds: int
    result: dict[str, Any]
    status: str
    stop_reason: str
    error: dict[str, Any]


class GraphServices(ContractModel):
    """图编译期注入的服务句柄集合（**不进 state**，见模块 docstring 第 3 条）。

    ``graph.py`` 只依赖本对象的成员名，不 import runtime / tools / app_store 的实现
    细节；RunManager 在 ``build_main_graph`` 时把真实实现塞进来，测试塞替身。
    用 ``arbitrary_types_allowed`` 承载句柄（它们不是可序列化数据）。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    #: 发布事件：``await emit(event_type, payload_dict, task_id=None)``。
    emit: Any = None
    #: 读工作区记忆固定快照文本：``await memory_context(state) -> str``。
    memory_context: Any = None
    #: 意图识别：``await route(question, history) -> RouteDecision``。
    route: Any = None
    #: 计划：``plan(question, decision) -> Plan``（planner 全同步）。
    plan: Any = None
    #: 单个研究任务：``await research(state, task_dump, request_dump) -> ResearchResult``。
    research: Any = None
    #: 研究子图编译产物（None = 不退子图，直接调 ``research``）。
    research_graph: Any = None
    #: 汇总候选答案：``synthesize(question, results, requirements) -> CandidateAnswer``。
    synthesize: Any = None
    #: 核对：``await verify(state, candidate) -> VerificationResult``。
    verify: Any = None
    #: 发布：``publish(state, candidate, verification) -> AnswerArtifact | None``。
    publish: Any = None
    #: 写子任务状态：``await save_task(state, task_id, status, thread_id, result_ref)``。
    save_task: Any = None
    #: 读回已保存的批准请求：``await load_approval(approval_id) -> ApprovalRequest | None``。
    load_approval: Any = None
    #: 应用批准决定（恢复时把决定写进 grants）：``apply_decision(state, resume_value)``。
    apply_decision: Any = None
    #: 发起澄清：``await request_input(state, question, options, reason) -> InputRequest``。
    request_input: Any = None
    #: 当前 run 的工具上下文：``await tool_context(state, task_id) -> ToolContext``。
    tool_context: Any = None
    #: 运行保护：主图 recursion_limit（PRD 7.4），默认 1000。
    recursion_limit: int = Field(default=1000, ge=1)


def new_agent_state(
    *,
    run_id: str,
    workspace_id: str,
    conversation_id: str,
    question: str,
    history: Sequence[Mapping[str, str]] = (),
    memory_context: str = "",
    knowledge_revision: int = 0,
    resume_from: str = "",
) -> AgentState:
    """构造一份主图初始 state（RunManager 唯一入口，避免各处手搓 dict）。"""

    state: AgentState = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "conversation_id": conversation_id,
        "question": question,
        "phase": "load_snapshot",
        "history": [dict(turn) for turn in history],
        "memory_context": memory_context,
        "knowledge_revision": int(knowledge_revision),
        "task_results": {},
        "completed_task_ids": [],
        "failed_task_ids": [],
        "approval_grants": [],
        "revisions": 0,
        "outcomes": [],
    }
    if resume_from:
        state["resume_from"] = resume_from
    return state


def new_research_state(
    *,
    task_id: str,
    question: str,
    agent_kind: AgentKind | str,
    request: Mapping[str, Any] | None = None,
) -> ResearchState:
    """构造一份研究子图初始 state。"""

    return ResearchState(
        task_id=task_id,
        question=question,
        agent_kind=str(getattr(agent_kind, "value", agent_kind)),
        request=dict(request or {}),
        rounds=0,
    )


def run_status_of(phase: str) -> RunStatus:
    """阶段 → 运行状态（``run.status`` 事件的 ``status`` 字段）。"""

    if phase == "approval":
        return RunStatus.WAITING_APPROVAL
    if phase == "clarify":
        return RunStatus.WAITING_INPUT
    if phase == "done":
        return RunStatus.COMPLETED
    return RunStatus.RUNNING
