"""LangGraph 拓扑：主图与研究子图（PRD 4.3 流程图、10.3 暂停恢复、12.2 RunManager）。Owner: T09。

职责：
- ``StateGraph`` 主图：加载快照 → 意图识别 → 计划 → 研究分支 → 汇集 → 核对 → 发布。
- 研究子图：单个 ``ResearchTask`` 的检索/联网循环（一个任务一个 ``thread_id``）。
- 条件跳转：``verdict`` 驱动 revise / research / clarify 回边（PRD 8.2）。
- ``recursion_limit`` 属运行保护（PRD 7.4），触发时由运行层保存为可继续的错误状态。

节点实现分散在 routing.py / planner.py / research.py / tools.py / approval.py /
verification.py，本模块只负责**接线与编译**：节点函数只从
:class:`app.agent.models.GraphServices` 取能力（不 import runtime / app_store 的
实现细节），因此同一套节点既能被 LangGraph 驱动，也能被下面的离线驱动
:func:`run_pipeline` 驱动。

关键取舍（写进架构文档 §6 D34）：

1. **两套驱动、同一批节点**：LangGraph 路径（带 ``AsyncSqliteSaver`` 检查点与
   ``interrupt()`` 暂停）与 :func:`run_pipeline`（离线/无检查点场景）共用节点函数与
   边表 :data:`MAIN_GRAPH_EDGES`，不会出现「图里一条边、测试里另一条边」的双实现漂移。
2. **暂停只发生在两个节点**：``approval``（PRD 10.3 批准）与 ``clarify``（PRD 5.4 澄清）。
   其余节点必须一次性跑完，否则恢复语义无法解释。
3. **失败即收敛**：任一子任务失败时研究节点返回带 ``error`` 的状态，主图直接结束
   （``failed``），不进入合成与核对 —— 子任务失败绝不能让父图停在 ``waiting_children``
   （PRD 12.2 末段）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final

from app.agent.approval import ApprovalRequired, approval_node, suspend
from app.agent.contracts import (
    AgentError,
    AnswerArtifact,
    ContractModel,
    EventType,
    Evidence,
    InputRequest,
    InputRequestStatus,
    ResearchTask,
    RouteDecision,
    RunStatus,
    TaskStatus,
    VerificationResult,
    VerifierVerdict,
    utc_now_iso,
)
from app.agent.models import AgentState, GraphServices, NodeOutcome, ResearchState
from app.agent.planner import Plan

__all__ = [
    "CLARIFY_MAX_ROUNDS",
    "MAIN_GRAPH_EDGES",
    "MAIN_GRAPH_NODES",
    "RESEARCH_GRAPH_EDGES",
    "RESEARCH_GRAPH_NODES",
    "REVISION_MAX_ROUNDS",
    "after_research",
    "after_verification",
    "build_main_graph",
    "build_research_graph",
    "clarify_node",
    "main_edges",
    "run_pipeline",
]

#: 主图节点（PRD 4.3 mermaid 的节点名，一一对应）。
#: ``revise`` 是 t31 追加的**回边计数**节点（不是 PRD 的业务节点）：两条驱动都必须递增
#: ``revisions``，否则 LangGraph 路径的 revise/research 回边会绕到 ``recursion_limit``。
MAIN_GRAPH_NODES: Final[tuple[str, ...]] = (
    "load_snapshot",
    "route",
    "plan",
    "research",
    "approval",
    "synthesis",
    "verification",
    "revise",
    "clarify",
    "publish",
)

#: 研究子图节点（PRD 7.2 的检索循环）。
RESEARCH_GRAPH_NODES: Final[tuple[str, ...]] = ("prepare", "run", "assess", "collect")

#: 主图边表（``(source, target)``；``after_*`` 是条件边，分支见 :func:`main_edges`）。
MAIN_GRAPH_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("load_snapshot", "route"),
    ("route", "plan"),
    ("plan", "research"),
    ("research", "after_research"),
    ("approval", "research"),
    ("synthesis", "verification"),
    ("verification", "after_verification"),
    ("revise", "after_revision"),
    ("clarify", "research"),
    ("publish", "END"),
)

#: 研究子图边表。
RESEARCH_GRAPH_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("prepare", "run"),
    ("run", "assess"),
    ("assess", "after_assess"),
    ("collect", "END"),
)

#: 回边次数上限（PRD 7.4 运行保护）：超过即按资料不足收敛，不再无限重试。
REVISION_MAX_ROUNDS: Final[int] = 2
CLARIFY_MAX_ROUNDS: Final[int] = 1


def main_edges() -> dict[str, tuple[str, ...]]:
    """条件边分支表（供编译与测试断言「PRD 4.3 的边都在」）。"""

    return {
        "after_research": ("synthesis", "approval", "failed"),
        # 回边统一先经过 ``revise`` 计数，再由 ``after_revision`` 分派到三条回边之一。
        "after_verification": ("publish", "revise"),
        "after_revision": ("synthesis", "research", "clarify"),
        "after_assess": ("collect", "run"),
    }


# =====================================================================
# 节点实现（services 闭包注入；节点本身只碰 state）
# =====================================================================


def _outcomes(state: Mapping[str, Any], outcome: NodeOutcome) -> list[dict[str, Any]]:
    outcomes = list(state.get("outcomes") or [])
    outcomes.append(outcome.model_dump(mode="json"))
    return outcomes


def _phase_update(state: Mapping[str, Any], phase: str, summary: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "outcomes": _outcomes(state, NodeOutcome(node=phase, summary=summary)),
    }


async def _maybe_await(value: Any) -> Any:
    return await value if asyncio.iscoroutine(value) else value


async def _emit(
    services: GraphServices,
    state: Mapping[str, Any],
    event_type: EventType,
    payload: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> None:
    if services.emit is None:
        return
    await services.emit(event_type, dict(payload), task_id=task_id)


def make_load_snapshot_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """读取工作区记忆固定快照（PRD 9.3 输入顺序第 3 位）。"""

    async def node(state: AgentState) -> dict[str, Any]:
        text = ""
        if services.memory_context is not None:
            text = str(await _maybe_await(services.memory_context(state)) or "")
        await _emit(
            services,
            state,
            EventType.RUN_STATUS,
            {
                "status": RunStatus.RUNNING.value,
                "phase": "load_snapshot",
                "message": "读取工作区记忆与会话快照",
            },
        )
        return {
            "memory_context": text,
            **_phase_update(state, "load_snapshot", "已载入记忆快照与当前上下文段"),
        }

    return node


def make_route_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """意图识别（PRD 6.1）：决策写入 state 并发布 ``route.decided``。"""

    async def node(state: AgentState) -> dict[str, Any]:
        decision: RouteDecision = RouteDecision.model_validate(
            await _maybe_await(services.route(state))
        )
        await _emit(
            services,
            state,
            EventType.ROUTE_DECIDED,
            {"decision": decision.model_dump(mode="json")},
        )
        return {
            "decision": decision.model_dump(mode="json"),
            **_phase_update(state, "route", f"意图={decision.intent.value}"),
        }

    return node


def make_plan_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """计划（PRD 6.4）：产出子任务、要求账目与执行顺序，并发布 ``plan.updated``。"""

    async def node(state: AgentState) -> dict[str, Any]:
        plan: Plan = Plan.model_validate(await _maybe_await(services.plan(state)))
        requirement_texts = {item.requirement_id: item.text for item in plan.requirements}
        await _emit(
            services,
            state,
            EventType.PLAN_UPDATED,
            {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "question": task.question,
                        "agent_kind": task.agent_kind.value,
                        "status": task.status.value,
                        "depends_on": list(task.depends_on),
                        "requirement_ids": list(task.requirement_ids),
                    }
                    for task in plan.tasks
                ],
                "covered_requirements": [],
                "uncovered_requirements": list(requirement_texts),
            },
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "requirement_texts": requirement_texts,
            "task_order": list(plan.task_ids()),
            **_phase_update(state, "plan", f"{len(plan.tasks)} 个子任务"),
        }

    return node


def make_research_node(
    services: GraphServices,
    *,
    concurrency: int = 3,
) -> Callable[[AgentState], Awaitable[dict]]:
    """研究分支：按依赖分层并行执行子任务（PRD 6.4 / 12.2）。

    每个子任务：
    - 先落 ``agent_tasks``（稳定 ID + 独立线程名，重复分配复用既有记录）；
    - 已成功完成的子任务**不重复执行**（用户「继续」时的复用口径，PRD 12.4）；
    - 命中需批准的工具时抛 :class:`ApprovalRequired`：本节点把它转成
      ``pending_approval`` 并**立即返回**（父图此时不再调用模型，PRD 12.2）；
    - 任一子任务失败：记录失败 task_id、保留已完成结果，并让主图收敛到 ``failed``。
    """

    async def node(state: AgentState) -> dict[str, Any]:
        plan = Plan.model_validate(state.get("plan") or {})
        tasks = {task.task_id: task for task in plan.tasks}
        results = dict(state.get("task_results") or {})
        completed = list(state.get("completed_task_ids") or [])
        failed = list(state.get("failed_task_ids") or [])
        order = [task_id for task_id in (state.get("task_order") or tasks) if task_id in tasks]
        pending = [task_id for task_id in order if task_id not in completed]

        for batch in _ready_batches(pending, tasks, initially_done=completed):
            for task_id in batch:
                await _save_task(services, state, tasks[task_id], TaskStatus.RUNNING)
                await _emit(
                    services,
                    state,
                    EventType.AGENT_STARTED,
                    {
                        "agent_kind": tasks[task_id].agent_kind.value,
                        "thread_id": _task_thread(state, task_id),
                        "task_id": task_id,
                        "question": tasks[task_id].question,
                    },
                    task_id=task_id,
                )
            semaphore = asyncio.Semaphore(max(1, int(concurrency)))

            # 默认参数把信号量绑进闭包（否则所有闭包共享循环末尾的那个变量）。
            async def guarded(task_id: str, gate: asyncio.Semaphore = semaphore) -> Any:
                async with gate:
                    return await _run_one(services, state, tasks[task_id])

            produced = await asyncio.gather(
                *(guarded(task_id) for task_id in batch),
                return_exceptions=True,
            )
            # 先收齐本层**成功**的结果，再处理挂起 / 失败：同层的兄弟任务已经产出的
            # 有效结果必须保留（PRD 12.2「已完成的有效结果保留」），不能被一个失败吞掉。
            approval: ApprovalRequired | None = None
            failure: tuple[str, BaseException] | None = None
            for task_id, outcome in zip(batch, produced, strict=True):
                if isinstance(outcome, ApprovalRequired):
                    approval = approval or outcome
                    await _save_task(services, state, tasks[task_id], TaskStatus.QUEUED)
                    continue
                if isinstance(outcome, BaseException):
                    failed.append(task_id)
                    await _save_task(
                        services, state, tasks[task_id], TaskStatus.FAILED, error=outcome
                    )
                    if failure is None:
                        failure = (task_id, outcome)
                    continue
                result, ref = outcome
                if _is_failed_result(result):
                    # 子 Agent **如实返回**的失败结果（不是异常）：RAG/WEB 的「服务错误」与
                    # General 的「未配置模型」都走这条（research.py 的契约就是返回准确状态）。
                    # 它必须按失败收敛并**保留子任务的错误码**（PRD 12.5），否则会被当成
                    # completed 记进台账，主图既不收敛也无法给出带类型的错误。
                    agent_error = result.get("error") or RuntimeError("研究任务失败")
                    failed.append(task_id)
                    await _save_task(
                        services, state, tasks[task_id], TaskStatus.FAILED, error=agent_error
                    )
                    if failure is None:
                        failure = (task_id, agent_error)
                    continue
                results[task_id] = result
                completed.append(task_id)
                await _save_task(
                    services, state, tasks[task_id], TaskStatus.COMPLETED, result_ref=ref
                )
                await _emit(
                    services,
                    state,
                    EventType.AGENT_COMPLETED,
                    {
                        "agent_kind": tasks[task_id].agent_kind.value,
                        "status": TaskStatus.COMPLETED.value,
                        "summary": f"{len(result.get('evidence') or [])} 条证据",
                    },
                    task_id=task_id,
                )
                summaries = _evidence_summaries(result)
                if summaries:
                    await _emit(
                        services,
                        state,
                        EventType.EVIDENCE_ADDED,
                        {"items": summaries},
                        task_id=task_id,
                    )

            if approval is not None:
                return {
                    "pending_approval": approval.pending.model_dump(mode="json"),
                    "task_results": results,
                    "completed_task_ids": completed,
                    "failed_task_ids": failed,
                    **_phase_update(
                        state, "approval", f"等待批准：{approval.pending.tool_name}"
                    ),
                }
            if failure is not None:
                task_id, error = failure
                return _failed_update(state, tasks[task_id], error, completed, results)

        return {
            "task_results": results,
            "completed_task_ids": completed,
            "failed_task_ids": failed,
            **_phase_update(state, "research", f"完成 {len(completed)} 个子任务"),
        }

    return node


def make_synthesis_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """汇集子任务结果，产出候选答案（PRD 8.1）。

    ``revise`` 回边再次进入本节点时复用同一批 ``task_results``，因此合成是**幂等**的：
    候选答案只由结果与要求文本决定。
    """

    async def node(state: AgentState) -> dict[str, Any]:
        results = list((state.get("task_results") or {}).values())
        candidate: ContractModel = await _maybe_await(
            services.synthesize(state, results, dict(state.get("requirement_texts") or {}))
        )
        return {
            "candidate": candidate.model_dump(mode="json"),
            **_phase_update(state, "synthesis", "候选答案已生成"),
        }

    return node


def make_verification_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """核对候选答案（PRD 8.2），并发布 ``verification.updated``。"""

    async def node(state: AgentState) -> dict[str, Any]:
        verification = VerificationResult.model_validate(await services.verify(state))
        await _emit(
            services,
            state,
            EventType.VERIFICATION_UPDATED,
            {
                "result": verification.model_dump(mode="json"),
                "pending": list(verification.uncovered_requirements),
            },
        )
        return {
            "verification": verification.model_dump(mode="json"),
            **_phase_update(state, "verification", f"结论={verification.verdict.value}"),
        }

    return node


async def clarify_node(state: AgentState, *, services: Any) -> dict[str, Any]:
    """澄清节点：写 ``input_requests`` 后 ``interrupt()`` 等用户回答（PRD 5.4 / 13.2）。

    与批准节点同构：**先持久化再中断**，恢复时用同一 ``input_request_id`` 读回，
    因此不会重复弹卡。
    """

    del services
    dump = dict(state.get("clarification") or {})
    if not dump:
        return {"phase": "research"}
    request = InputRequest.model_validate(dump)
    if request.status is InputRequestStatus.PENDING:
        answer = suspend(_clarify_payload(request))
        return {
            "clarification": request.model_copy(
                update={"status": InputRequestStatus.ANSWERED, "answer": str(answer or "")}
            ).model_dump(mode="json"),
            **_phase_update(state, "clarify", "已收到澄清回答"),
        }
    return {
        "clarification": {},
        **_phase_update(state, "clarify", "澄清已回答"),
    }


def _clarify_payload(request: InputRequest) -> dict[str, Any]:
    return {
        "input_request_id": request.input_request_id,
        "run_id": request.run_id,
        "question": request.question,
        "options": list(request.options),
        "reason": request.reason,
        "question_version": request.question_version,
    }


def make_publish_node(services: GraphServices) -> Callable[[AgentState], Awaitable[dict]]:
    """发布：把已核对候选答案落库（单事务，PRD 13.1 末段），由运行层实现写事务。

    回边次数用尽（:data:`REVISION_MAX_ROUNDS`）而结论仍是 revise/research/clarify 时，
    按 PRD 8.3 收敛为 **``completed_with_gaps`` 并说明缺少的依据**，而不是把整次运行
    判死：那是「资料确实拿不到」的正常结局，用户仍应看到答案与缺口清单。
    """

    async def node(state: AgentState) -> dict[str, Any]:
        state = _converge_exhausted(state)
        artifact = await _maybe_await(services.publish(state))
        if artifact is None:
            return _failed_update(
                state,
                None,
                RuntimeError("发布阶段没有产出答案对象"),
                list(state.get("completed_task_ids") or []),
                dict(state.get("task_results") or {}),
            )
        artifact = AnswerArtifact.model_validate(artifact)
        return {
            "answer_id": artifact.answer_id,
            "answer_status": artifact.status.value,
            "verification": (state.get("verification") or {}),
            **_phase_update(state, "publish", "答案已保存"),
        }

    return node


def _converge_exhausted(state: AgentState) -> AgentState:
    """回边用尽时把不可发布的核对结论降级为 ``insufficient``（带说明，PRD 8.3）。"""

    verification = VerificationResult.model_validate(state.get("verification") or {})
    if verification.verdict in {VerifierVerdict.PASS, VerifierVerdict.INSUFFICIENT}:
        return state
    if int(state.get("revisions") or 0) < REVISION_MAX_ROUNDS:
        return state
    from app.agent.contracts import VerificationCheck, VerificationIssue, VerificationSeverity

    downgraded = verification.model_copy(
        update={
            "verdict": VerifierVerdict.INSUFFICIENT,
            "issues": [
                *verification.issues,
                VerificationIssue(
                    check=VerificationCheck.MATERIAL_STATUS,
                    statement="修订与补研次数已用尽，按资料不足发布",
                    required_action="稍后可用「继续」补充研究，或补充资料后重问",
                    severity=VerificationSeverity.MEDIUM,
                ),
            ],
        }
    )
    merged = dict(state)
    merged["verification"] = downgraded.model_dump(mode="json")
    return merged  # type: ignore[return-value]


# =====================================================================
# 条件边
# =====================================================================


def after_research(state: AgentState) -> str:
    """研究之后：有错误 → 结束；有批准挂起 → 批准节点；否则 → 合成。"""

    if state.get("error"):
        return "failed"
    if state.get("pending_approval"):
        return "approval"
    return "synthesis"


def after_verification(state: AgentState) -> str:
    """核对之后（PRD 8.2）：pass/insufficient → 发布；revise/research/clarify → 回边。

    回边次数达到 :data:`REVISION_MAX_ROUNDS` 时回到 ``publish``：由发布节点按
    ``verdict`` 决定能否带缺口发布，不能则收敛为**可继续的错误状态**（PRD 7.4 / 8.3）。
    """

    verification = VerificationResult.model_validate(state.get("verification") or {})
    if verification.verdict in {VerifierVerdict.PASS, VerifierVerdict.INSUFFICIENT}:
        return "publish"
    if int(state.get("revisions") or 0) >= REVISION_MAX_ROUNDS:
        return "publish"
    if verification.verdict is VerifierVerdict.CLARIFY:
        return "clarify"
    if verification.verdict is VerifierVerdict.RESEARCH:
        return "research"
    return "synthesis"


def _is_failed_result(result: Mapping[str, Any]) -> bool:
    """子 Agent 的结果是不是**如实上报的失败**（``status=failed``）。

    研究服务返回的是 **JSON 化后的 dict**（见 :func:`_run_one`），所以状态取 ``result["status"]``
    而不是属性；两种形态都兼容，避免又一处「图里看属性、状态里是字典」的漂移。
    """

    status = result.get("status")
    return str(getattr(status, "value", status)) == TaskStatus.FAILED.value


def _error_field(error: Any, name: str, default: Any = None) -> Any:
    """同时支持异常对象、``AgentError`` 契约对象与 JSON dict（三者都会流到这里）。"""

    if isinstance(error, Mapping):
        return error.get(name, default)
    return getattr(error, name, default)


def _failed_update(
    state: Mapping[str, Any],
    task: ResearchTask | None,
    error: BaseException | Any,
    completed: Sequence[str],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """子任务失败时的 state 更新：保存失败 task_id、保留已完成结果、主图收敛（PRD 12.2）。"""

    # 错误码优先级：工具层异常的 ``code`` → 契约 ``AgentError.error_code`` → 异常类名。
    # 子 Agent 如实返回的 ``AgentError``（如 ``not_configured``）必须原样带到 run 上，
    # 否则用户看到的是 "dict" / "RuntimeError" 这类无信息量的码（PRD 12.5）。
    error_code = str(
        _error_field(error, "code", "")
        or _error_field(error, "error_code", "")
        or type(error).__name__
    )
    message = str(_error_field(error, "message", "") or "")
    agent_error = AgentError(
        error_code=error_code,
        message=message or f"研究子任务失败：{type(error).__name__}",
        node=str(_error_field(error, "node", "") or "research"),
        operation_id=_error_field(error, "operation_id", None),
        resumable=bool(_error_field(error, "resumable", True)),
        required_action="POST /api/workspaces/{workspace_id}/runs/{run_id}/resume 继续",
        created_at=utc_now_iso(),
    )
    return {
        "error": agent_error.model_dump(mode="json"),
        "task_results": dict(results),
        "completed_task_ids": list(completed),
        "failed_task_ids": [task.task_id] if task is not None else [],
        **_phase_update(state, "done", "研究子任务失败，已保存可继续的位置"),
    }


# =====================================================================
# 编译（LangGraph）
# =====================================================================


def build_research_graph(services: GraphServices, *, checkpointer: Any = None) -> Any:
    """编译研究子图（PRD 7.2）。``langgraph`` 不可用时返回 ``None``。"""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:  # pragma: no cover - langgraph 是 V1.2.0 依赖
        return None

    graph = StateGraph(ResearchState)

    async def prepare(state: ResearchState) -> dict[str, Any]:
        return {"rounds": int(state.get("rounds") or 0), "status": TaskStatus.RUNNING.value}

    async def run(state: ResearchState) -> dict[str, Any]:
        payload = await _maybe_await(
            services.research(
                {
                    "run_id": state.get("run_id", ""),
                    "workspace_id": state.get("workspace_id", ""),
                    "question": state.get("question", ""),
                },
                dict(state.get("task") or {}),
                dict(state.get("request") or {}),
            )
        )
        return {
            "result": payload.model_dump(mode="json")
            if isinstance(payload, ContractModel)
            else dict(payload),
            "rounds": int(state.get("rounds") or 0) + 1,
        }

    async def assess(state: ResearchState) -> dict[str, Any]:
        return {"stop_reason": str((state.get("result") or {}).get("stop_reason") or "completed")}

    async def collect(state: ResearchState) -> dict[str, Any]:
        result = state.get("result") or {}
        return {"status": str(result.get("status") or TaskStatus.COMPLETED.value)}

    graph.add_node("prepare", prepare)
    graph.add_node("run", run)
    graph.add_node("assess", assess)
    graph.add_node("collect", collect)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "run")
    graph.add_edge("run", "assess")
    graph.add_conditional_edges(
        "assess",
        lambda state: "collect" if int(state.get("rounds") or 0) >= 1 else "run",
        {"collect": "collect", "run": "run"},
    )
    graph.add_edge("collect", END)
    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()


def build_main_graph(services: GraphServices, *, checkpointer: Any = None) -> Any:
    """编译主图（PRD 4.3）。``langgraph`` 不可用时返回 ``None``（运行层退离线驱动）。"""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:  # pragma: no cover
        return None

    graph = StateGraph(AgentState)
    graph.add_node("load_snapshot", make_load_snapshot_node(services))
    graph.add_node("route", make_route_node(services))
    graph.add_node("plan", make_plan_node(services))
    graph.add_node("research", make_research_node(services))

    async def approval(state: AgentState) -> dict[str, Any]:
        return await approval_node(state, services=services)

    async def clarify(state: AgentState) -> dict[str, Any]:
        return await clarify_node(state, services=services)

    graph.add_node("approval", approval)
    graph.add_node("synthesis", make_synthesis_node(services))
    graph.add_node("verification", make_verification_node(services))
    graph.add_node("clarify", clarify)
    graph.add_node("publish", make_publish_node(services))

    async def revise(state: AgentState) -> dict[str, Any]:
        """回边递增 ``revisions``（与离线驱动 :func:`run_pipeline` 同一语义）。

        **为什么需要这个节点**：``after_verification`` 靠 ``state["revisions"]`` 判定
        「修订/补研次数是否用尽」，而早先只有离线驱动会递增它 —— LangGraph 路径下计数器
        永远是 0，于是 ``revise``/``research`` 回边会一直绕到 ``recursion_limit``，
        用户看到的是 ``GraphRecursionError`` 而不是 PRD 8.3 的「按资料不足发布」。
        """

        return {"revisions": int(state.get("revisions") or 0) + 1}

    def after_revision(state: AgentState) -> str:
        verification = VerificationResult.model_validate(state.get("verification") or {})
        if verification.verdict is VerifierVerdict.CLARIFY:
            return "clarify"
        if verification.verdict is VerifierVerdict.RESEARCH:
            return "research"
        return "synthesis"

    graph.add_node("revise", revise)

    graph.add_edge(START, "load_snapshot")
    graph.add_edge("load_snapshot", "route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "research")
    graph.add_conditional_edges(
        "research",
        after_research,
        {"approval": "approval", "synthesis": "synthesis", "failed": END},
    )
    graph.add_edge("approval", "research")
    graph.add_edge("synthesis", "verification")
    graph.add_conditional_edges(
        "verification",
        after_verification,
        # 回边一律先经过 ``revise`` 计数，再由 ``after_revision`` 分派 —— 两条驱动
        # 因此共享同一套「次数用尽」语义。
        {"publish": "publish", "synthesis": "revise", "research": "revise", "clarify": "revise"},
    )
    graph.add_conditional_edges(
        "revise",
        after_revision,
        {"synthesis": "synthesis", "research": "research", "clarify": "clarify"},
    )
    graph.add_edge("clarify", "research")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)


# =====================================================================
# 离线驱动（无检查点；同一张边表）
# =====================================================================


async def run_pipeline(
    services: GraphServices,
    state: AgentState,
    *,
    concurrency: int | None = None,
) -> AgentState:
    """按 :data:`MAIN_GRAPH_EDGES` 走一遍主图（不含检查点与 LangGraph ``interrupt()``）。

    暂停语义：进入 ``approval`` / ``clarify`` 阶段时，若对应请求**仍未裁决**，驱动
    立即返回当前 state（运行层据此把 run 置为 ``waiting_approval`` / ``waiting_input``）；
    已裁决时调用节点函数本身（它不再中断，只把决定并入 state）。
    """

    nodes: dict[str, Callable[[AgentState], Awaitable[dict[str, Any]]]] = {
        "load_snapshot": make_load_snapshot_node(services),
        "route": make_route_node(services),
        "plan": make_plan_node(services),
        "research": make_research_node(services, concurrency=concurrency or 3),
        "synthesis": make_synthesis_node(services),
        "verification": make_verification_node(services),
        "publish": make_publish_node(services),
    }
    for name in ("load_snapshot", "route", "plan", "research"):
        state = await _step(nodes[name], state)

    guard = 2 * REVISION_MAX_ROUNDS + 4
    for _ in range(guard):
        branch = after_research(state)
        if branch == "failed":
            return state
        if branch == "approval":
            pending = dict(state.get("pending_approval") or {})
            request = await _maybe_await(
                services.load_approval(pending.get("approval_id", ""))
                if services.load_approval is not None
                else None
            )
            if request is None or str(getattr(request, "status", "pending")) == "pending":
                return state
            state = await _merge(state, await approval_node(state, services=services))
            continue
        state = await _step(nodes["synthesis"], state)
        state = await _step(nodes["verification"], state)
        follow = after_verification(state)
        if follow == "publish":
            return await _step(nodes["publish"], state)
        state = dict(state)  # type: ignore[assignment]
        state["revisions"] = int(state.get("revisions") or 0) + 1
        if follow == "clarify":
            clarification = dict(state.get("clarification") or {})
            if not clarification or clarification.get("status") == "pending":
                state["phase"] = "clarify"
                return state
            state["clarification"] = {}
            continue
        state = await _step(nodes["research"], state)
    return state


async def _step(
    node: Callable[[AgentState], Awaitable[dict[str, Any]]],
    state: AgentState,
) -> AgentState:
    return await _merge(state, await node(state))


async def _merge(state: AgentState, updates: Mapping[str, Any]) -> AgentState:
    merged = dict(state)
    merged.update({key: value for key, value in updates.items() if value is not None})
    return merged  # type: ignore[return-value]


# =====================================================================
# 研究子任务的小工具
# =====================================================================


async def _run_one(
    services: GraphServices,
    state: AgentState,
    task: ResearchTask,
) -> tuple[dict[str, Any], str]:
    if services.research_graph is not None:
        payload: Any = await _run_subgraph(services, state, task)
    else:
        payload = await _maybe_await(
            services.research(state, task.model_dump(mode="json"), {})
        )
    if isinstance(payload, ContractModel):
        result = payload.model_dump(mode="json")
    else:
        result = dict(payload)
    return result, str(result.get("task_id") or task.task_id)


async def _run_subgraph(
    services: GraphServices,
    state: AgentState,
    task: ResearchTask,
) -> Mapping[str, Any]:
    research_state: ResearchState = {
        "task_id": task.task_id,
        "question": task.question,
        "agent_kind": task.agent_kind.value,
        "run_id": str(state.get("run_id") or ""),
        "workspace_id": str(state.get("workspace_id") or ""),
        "task": task.model_dump(mode="json"),
        "request": {},
    }
    config = {"configurable": {"thread_id": _task_thread(state, task.task_id)}}
    produced = await services.research_graph.ainvoke(research_state, config=config)
    return dict(produced.get("result") or {})


def _ready_batches(
    pending: Sequence[str],
    tasks: Mapping[str, ResearchTask],
    *,
    initially_done: Sequence[str] = (),
) -> list[list[str]]:
    """按依赖分层：同层可并行（PRD 6.4）。已完成的依赖视为满足。"""

    remaining = [task_id for task_id in pending if task_id in tasks]
    done: set[str] = set(initially_done)
    batches: list[list[str]] = []
    while remaining:
        layer = [
            task_id
            for task_id in remaining
            if all(dep in done or dep not in tasks for dep in tasks[task_id].depends_on)
        ]
        if not layer:
            # 依赖成环（planner 已在计划阶段拒绝）或依赖不在计划内：退化为当前顺序，
            # 不允许调度器在这里静默丢任务。
            layer = list(remaining)
        batches.append(layer)
        done.update(layer)
        blocked = set(layer)
        remaining = [task_id for task_id in remaining if task_id not in blocked]
    return batches


async def _save_task(
    services: GraphServices,
    state: AgentState,
    task: ResearchTask,
    status: TaskStatus,
    *,
    result_ref: str | None = None,
    error: BaseException | None = None,
) -> None:
    if services.save_task is None:
        return
    await services.save_task(
        state,
        task,
        status,
        _task_thread(state, task.task_id),
        result_ref=result_ref,
        error=error,
    )


def _task_thread(state: Mapping[str, Any], task_id: str) -> str:
    from app.agent.contracts import task_thread_id

    return task_thread_id(str(state.get("run_id") or ""), task_id)


def _evidence_summaries(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in result.get("evidence") or []:
        evidence = Evidence.model_validate(raw)
        items.append(
            {
                "evidence_id": evidence.evidence_id,
                "source_type": evidence.source_type.value,
                "title": evidence.source_ref,
                "source_ref": evidence.source_ref,
                "snippet": (evidence.content or "")[:200],
            }
        )
    return items
