from __future__ import annotations

import asyncio
import logging
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
    utc_now_iso,
)
from app.agent.models import AgentState, GraphServices, NodeOutcome, ResearchState
from app.agent.planner import Plan

logger = logging.getLogger(__name__)

__all__ = [
    "CLARIFY_MAX_ROUNDS",
    "MAIN_GRAPH_EDGES",
    "MAIN_GRAPH_NODES",
    "RESEARCH_GRAPH_EDGES",
    "RESEARCH_GRAPH_NODES",
    "after_research",
    "build_main_graph",
    "build_research_graph",
    "clarify_node",
    "main_edges",
    "run_pipeline",
]

#: 主图节点（PRD 4.3 mermaid 的节点名，一一对应）。
MAIN_GRAPH_NODES: Final[tuple[str, ...]] = (
    "load_snapshot",
    "route",
    "plan",
    "research",
    "approval",
    "synthesis",
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
    ("synthesis", "publish"),
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

CLARIFY_MAX_ROUNDS: Final[int] = 1


def main_edges() -> dict[str, tuple[str, ...]]:
    """条件边分支表（供编译与测试断言「PRD 4.3 的边都在」）。"""

    return {
        "after_research": ("synthesis", "approval", "failed"),
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
        if services.memory_context is None:
            return {
                "memory_context": "",
                **_phase_update(state, "load_snapshot", "记忆功能已关闭"),
            }
        text = ""
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
        covered = sorted({
            requirement_id
            for task in plan.tasks
            for requirement_id in task.requirement_ids
            if requirement_id in requirement_texts
        })
        uncovered = [
            text for requirement_id, text in requirement_texts.items()
            if requirement_id not in covered
        ]
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
                        "query_identity": task.query_identity,
                        "query_reuse_reason": task.query_reuse_reason,
                    }
                    for task in plan.tasks
                ],
                "covered_requirements": covered,
                # 未覆盖项给要求文本：旧实现把 dict 的键（R1..Rn）当文本发出，
                # 前端因此永远显示「全部未覆盖」。
                "uncovered_requirements": uncovered,
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
            batch_state = dict(state)
            batch_state["task_results"] = dict(results)
            batch_state["completed_task_ids"] = list(completed)
            batch_state["failed_task_ids"] = list(failed)

            # 默认参数把信号量绑进闭包（否则所有闭包共享循环末尾的那个变量）。
            async def guarded(
                task_id: str,
                gate: asyncio.Semaphore = semaphore,
                current: AgentState = batch_state,  # type: ignore[assignment]
            ) -> Any:
                async with gate:
                    return await _run_one(services, current, tasks[task_id])

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
                    logger.error(
                        "research task %s failed", task_id,
                        exc_info=(type(outcome), outcome, outcome.__traceback__),
                    )
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
                        "summary": f"已完成查询：{tasks[task_id].question}",
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
    """汇集研究结果并生成答案。"""

    async def node(state: AgentState) -> dict[str, Any]:
        results = list((state.get("task_results") or {}).values())
        await _emit(services, state, EventType.RUN_STATUS, {
            "status": "running", "phase": "synthesis", "message": "正在整理回答",
        })
        candidate: ContractModel = await _maybe_await(
            services.synthesize(state, results, dict(state.get("requirement_texts") or {}))
        )
        return {
            "candidate": candidate.model_dump(mode="json"),
            **_phase_update(state, "synthesis", "候选答案已生成"),
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
    """保存答案并发送完成事件。"""

    async def node(state: AgentState) -> dict[str, Any]:
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
            **_phase_update(state, "publish", "答案已保存"),
        }

    return node


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
    message = str(
        _error_field(error, "user_message", "") or _error_field(error, "message", "") or ""
    )
    agent_error = AgentError(
        error_code=error_code,
        message=message or f"研究子任务失败：{type(error).__name__}",
        node=str(_error_field(error, "node", "") or "research"),
        operation_id=_error_field(error, "operation_id", None),
        resumable=bool(_error_field(error, "resumable", True)),
        required_action=str(_error_field(error, "required_action", "") or (
            "点击“继续执行”重试当前任务"
            if bool(_error_field(error, "resumable", True)) else "请重新发送问题"
        )),
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
                state.get("context") or {
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
    graph.add_node("clarify", clarify)
    graph.add_node("publish", make_publish_node(services))

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
    graph.add_edge("synthesis", "publish")
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
        "publish": make_publish_node(services),
    }
    for name in ("load_snapshot", "route", "plan", "research"):
        state = await _step(nodes[name], state)

    while True:
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
            state = await _step(nodes["research"], state)
            continue
        state = await _step(nodes["synthesis"], state)
        return await _step(nodes["publish"], state)


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
        "context": dict(state),
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
    from app.agent.evidence_text import split_evidence_content

    items: list[dict[str, Any]] = []
    for raw in result.get("evidence") or []:
        evidence = Evidence.model_validate(raw)
        heading, body = split_evidence_content(evidence.content)
        # ``evidence.added`` 的 item 形状是**冻结契约**（只认 evidence_id / source_type /
        # title / source_ref / snippet），因此文档标题在这里拼进 ``title``：只显示章节名时，
        # 界面会把章节名当成文章名，用户无法确认这段来自哪篇文章（用户实测缺陷 F13）。
        # 完整的文档标题另有到达路径：``Evidence.document_title`` → 引用条目的
        # ``document_title`` / ``heading_path``（见 verification.assign_citation_ids）。
        title = evidence.document_title or heading or "已读取的资料"
        if evidence.document_title and heading and heading not in evidence.document_title:
            title = f"{evidence.document_title} · {heading}"
        items.append(
            {
                "evidence_id": evidence.evidence_id,
                "source_type": evidence.source_type.value,
                "title": title,
                "source_ref": evidence.source_ref,
                "snippet": body[:200],
            }
        )
    return items
