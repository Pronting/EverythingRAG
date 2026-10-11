"""V1.2.0 run 层测试夹具：假服务、假工作区上下文、最小工具注册表（T09 用）。

为什么需要这套替身（`tests/fakes_agent.py` 只覆盖契约层数据）：

1. **假工作区上下文工厂**：``RunManager`` 只经 ``context_factory.resolve(id)`` 取
   ``checkpoints.db`` / ``knowledge_revision`` 等只读属性；真实的
   ``WorkspaceContextFactory`` 要读磁盘工作区目录，单测不需要那一步。
2. **假服务集**（:class:`FakeServices`）：确定性的 route/plan/research/verify/publish，
   让「图怎么走」「批准挂在哪一步」「失败怎么收敛」可以精确断言，且**绝不外呼**。
3. **最小工具注册表**：``web_search`` 需要批准、``search_knowledge`` 不需要，
   用来验证权限门在任何执行之前拦截（PRD 10.1 / A01）。

所有用例都只用 tmp 工作区与内存 app.db，不触碰真实 ``~/.everything-rag``。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.approval import NameRedactor
from app.agent.contracts import (
    DEFAULT_PERMISSION_MODE,
    AgentKind,
    AnswerBasis,
    CitationEntry,
    Evidence,
    EvidenceSourceType,
    InputRequest,
    IntentKind,
    ResearchTask,
    RouteDecision,
    SourceType,
    TaskStatus,
)
from app.agent.models import AgentState, GraphServices
from app.agent.planner import Plan, PlanSettings, UserRequirement
from app.agent.research import ResearchConclusion, ResearchProgress, ResearchResult
from app.agent.runtime import ApprovalGate, RunManager
from app.agent.verification import CandidateAnswer
from app.agent.verification import publish_answer as _publish_answer

__all__ = [
    "FakeServices",
    "StubContext",
    "StubContextFactory",
    "build_manager",
    "build_registry",
    "make_evidence",
    "seed_conversation",
]


@dataclass(frozen=True)
class StubContext:
    """``RunManager`` 需要的只读上下文属性（与真实 ``WorkspaceContext`` 同形）。"""

    workspace_id: str
    storage_key: str
    root: Path
    default_permission_mode: str = str(DEFAULT_PERMISSION_MODE)
    knowledge_revision: int = 0

    @property
    def checkpoints_db(self) -> Path:
        return self.root / "checkpoints.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"


@dataclass
class StubContextFactory:
    """按 ``workspace_id`` 返回 stub 上下文的工厂（接口与真实工厂一致）。"""

    root: Path
    _contexts: dict[str, StubContext] = field(default_factory=dict)

    def resolve(self, workspace_id: str | None = None) -> StubContext:
        key = workspace_id or "ws-default"
        if key not in self._contexts:
            self._contexts[key] = StubContext(
                workspace_id=key, storage_key=f"sk-{key}", root=self.root / key
            )
        return self._contexts[key]


def build_registry(*, approval_tools: Sequence[str] = ("web_search",)) -> Any:
    """最小工具注册表（ToolSpec + 纯内存 handler，**不发任何网络请求**）。"""

    from app.agent.tools import (
        SearchKnowledgeParams,
        ToolContext,
        ToolName,
        ToolRegistry,
        ToolResult,
        ToolSpec,
        ToolStatus,
        WebSearchParams,
    )

    registry = ToolRegistry()
    #: 实际执行过的工具名（断言「拒绝后没有执行」用；ToolRegistry 是普通类，可挂属性）。
    registry.call_log = []
    #: 实际发给工具的参数（断言「出网参数已脱敏」用）。
    registry.query_log = []

    async def knowledge(params: SearchKnowledgeParams, context: ToolContext) -> ToolResult:
        del context
        registry.call_log.append(str(ToolName.SEARCH_KNOWLEDGE))
        registry.query_log.append(params.model_dump(mode="json"))
        return ToolResult(
            tool=str(ToolName.SEARCH_KNOWLEDGE),
            status=ToolStatus.OK,
            data={"items": [{"title": params.query, "text": "本地命中"}]},
        )

    async def web(params: WebSearchParams, context: ToolContext) -> ToolResult:
        del context
        registry.call_log.append(str(ToolName.WEB_SEARCH))
        registry.query_log.append(params.model_dump(mode="json"))
        return ToolResult(
            tool=str(ToolName.WEB_SEARCH),
            status=ToolStatus.OK,
            data={"results": [{"title": params.query, "url": "https://example.com/a"}]},
        )

    registry.register(
        ToolSpec(
            name=str(ToolName.SEARCH_KNOWLEDGE),
            description="本地知识检索",
            params_model=SearchKnowledgeParams,
            handler=knowledge,
            returns="items[]（本地命中）",
        )
    )
    registry.register(
        ToolSpec(
            name=str(ToolName.WEB_SEARCH),
            description="联网搜索",
            params_model=WebSearchParams,
            handler=web,
            returns="results[]（搜索引擎返回）",
            networked=True,
            requires_approval=str(ToolName.WEB_SEARCH) in set(approval_tools),
        )
    )
    return registry


def make_evidence(index: int = 1, *, source_type: EvidenceSourceType | None = None) -> Evidence:
    """确定性证据（内容随 index 变化，便于断言去重与引用编号）。"""

    return Evidence(
        evidence_id=f"ev-{index:04d}",
        workspace_id="ws-1",
        run_id="run-1",
        source_type=source_type or EvidenceSourceType.KNOWLEDGE,
        source_ref=f"doc-{index}.md",
        content=f"证据 {index} 的正文",
        content_hash=f"hash-{index}",
        retrieved_at="2026-09-19T08:00:00Z",
    )


async def seed_conversation(
    store: Any,
    *,
    workspace_id: str = "ws-1",
    storage_key: str = "sk-ws-1",
    conversation_id: str = "conv-1",
) -> None:
    """建好工作区 + 会话（run 由 RunManager.create_run 建）。"""

    await store.create_workspace(workspace_id, storage_key, name="测试工作区")
    await store.create_conversation(conversation_id, workspace_id)


def build_manager(
    store: Any,
    *,
    services: FakeServices,
    root: Path,
    registry: Any = None,
    use_langgraph: bool = True,
    private_names: Sequence[str] = ("张三",),
    research_concurrency: int = 3,
    image_ingest: Any = None,
) -> RunManager:
    """装配一个 RunManager（假服务 + stub 上下文 + 内存 app.db）。

    ``image_ingest`` 留空时 RunManager 会把图片引用如实记为「未接入落库」——
    这正是生产未接线的形态，可用来断言**不静默忽略**。
    """

    return RunManager(
        store=store,
        context_factory=StubContextFactory(root=root),
        registry=registry or build_registry(),
        services_factory=services.factory,
        use_langgraph=use_langgraph,
        private_names=private_names,
        research_concurrency=research_concurrency,
        image_ingest=image_ingest,
    )


@dataclass
class FakeServices:
    """确定性的图服务集合（每个成员都记录调用次数，供断言）。"""

    task_count: int = 2
    approve_tool: str | None = None
    fail_tasks: tuple[int, ...] = ()
    calls: dict[str, int] = field(default_factory=dict)
    research_tasks_seen: list[str] = field(default_factory=list)
    emitted: list[str] = field(default_factory=list)
    approved: bool = False
    research_started: asyncio.Event | None = None
    research_release: asyncio.Event | None = None

    def _bump(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def factory(self, manager: RunManager, run: Any, state: AgentState) -> GraphServices:
        """按当前 run 现造一套服务（与 ``default_services`` 同一签名）。"""

        del state

        async def emit(
            event_type: Any, payload: Mapping[str, Any], *, task_id: str | None = None
        ) -> None:
            value = str(getattr(event_type, "value", event_type))
            self._bump(f"emit:{value}")
            self.emitted.append(value)
            await manager.publish_event(run.id, event_type, dict(payload), task_id=task_id)

        async def memory_context(current: Mapping[str, Any]) -> str:
            del current
            self._bump("memory_context")
            return "记忆快照：用户偏好简洁回答"

        async def route(current: Mapping[str, Any]) -> RouteDecision:
            del current
            self._bump("route")
            return RouteDecision(
                question="测试问题",
                resolved_question="测试问题",
                intent=IntentKind.RESEARCH,
                needs_knowledge=True,
            )

        async def plan(current: Mapping[str, Any]) -> Plan:
            del current
            self._bump("plan")
            requirements = [
                UserRequirement(
                    requirement_id=f"r{index}", text=f"要求 {index}", quote="测试问题"
                )
                for index in range(1, self.task_count + 1)
            ]
            tasks = [
                ResearchTask(
                    task_id=f"t{index}",
                    question=f"子问题 {index}",
                    requirement_ids=[f"r{index}"],
                    agent_kind=AgentKind.RAG,
                )
                for index in range(1, self.task_count + 1)
            ]
            return Plan(
                question="测试问题",
                workspace_id=run.workspace_id,
                requirements=requirements,
                tasks=tasks,
                settings=PlanSettings(),
            )

        async def research(
            current: Mapping[str, Any],
            task_dump: Mapping[str, Any],
            request_dump: Mapping[str, Any],
        ) -> ResearchResult:
            del request_dump
            task = ResearchTask.model_validate(task_dump)
            self._bump("research")
            self.research_tasks_seen.append(task.task_id)
            if self.research_started is not None:
                self.research_started.set()
            if self.research_release is not None:
                await self.research_release.wait()
            index = int(task.task_id.lstrip("t") or "1")
            if index in self.fail_tasks:
                raise RuntimeError(f"子任务 {task.task_id} 失败")
            if self.approve_tool is not None and not self.approved:
                await self._call_approval_tool(manager, run, current, task)
            evidence = make_evidence(index)
            return ResearchResult(
                task_id=task.task_id,
                agent_kind=task.agent_kind,
                status=TaskStatus.COMPLETED,
                evidence=[evidence],
                conclusions=[
                    ResearchConclusion(
                        statement=f"结论 {index}",
                        evidence_ids=[evidence.evidence_id],
                        requirement_ids=[f"r{index}"],
                    )
                ],
                progress=ResearchProgress(
                    task_id=task.task_id, round=1, evidence_ids=[evidence.evidence_id]
                ),
            )

        def synthesize(
            current: Mapping[str, Any],
            results: Sequence[Mapping[str, Any]],
            requirements: Mapping[str, str],
        ) -> CandidateAnswer:
            del requirements
            self._bump("synthesize")
            parsed = [ResearchResult.model_validate(item) for item in results]
            evidence = [item for result in parsed for item in result.evidence]
            conclusions = [item for result in parsed for item in result.conclusions]
            citations = [
                CitationEntry(
                    label=f"S{index}",
                    source_type=SourceType.KNOWLEDGE,
                    evidence_id=item.evidence_id,
                    workspace_id=run.workspace_id,
                )
                for index, item in enumerate(evidence, start=1)
            ]
            return CandidateAnswer(
                question=str(current.get("question") or "测试问题"),
                content="### 答案\n" + "".join(f"{item.statement}[S1]" for item in conclusions),
                workspace_id=run.workspace_id,
                run_id=run.id,
                conclusions=conclusions,
                evidence=evidence,
                citations=citations,
                covered_requirements=[f"r{index}" for index in range(1, self.task_count + 1)],
                answer_basis=AnswerBasis.KNOWLEDGE,
                basis_components=[SourceType.KNOWLEDGE],
            )

        async def publish(current: Mapping[str, Any]) -> Any:
            self._bump("publish")
            candidate = CandidateAnswer.model_validate(current.get("candidate") or {})
            artifact = _publish_answer(
                candidate,
                requirements=dict(current.get("requirement_texts") or {}),
            )
            await manager.store.commit_answer(artifact=artifact, deltas=[artifact.content])
            return artifact

        async def save_task(
            current: Mapping[str, Any],
            task: ResearchTask,
            status: TaskStatus,
            thread_id: str,
            *,
            result_ref: str | None = None,
            error: BaseException | None = None,
        ) -> None:
            del current, error
            self._bump(f"save_task:{status.value}")
            await manager.store.create_task(
                run_id=run.id,
                task_id=task.task_id,
                agent_kind=task.agent_kind,
                question=task.question,
                requirement_ids=task.requirement_ids,
                depends_on=task.depends_on,
                thread_id=thread_id,
            )
            await manager.store.update_task_status(
                run.id, task.task_id, status, result_ref=result_ref, thread_id=thread_id
            )

        async def load_approval(approval_id: str) -> Any:
            return await manager.store.get_approval(approval_id)

        async def request_input(
            current: Mapping[str, Any], question: str, options: Sequence[str], reason: str
        ) -> InputRequest:
            del current
            self._bump("request_input")
            request = InputRequest(
                input_request_id=f"in-{run.id}",
                run_id=run.id,
                question=question,
                options=list(options),
                reason=reason,
            )
            return await manager.store.request_input(request)

        return GraphServices(
            emit=emit,
            memory_context=memory_context,
            route=route,
            plan=plan,
            research=research,
            synthesize=synthesize,
            publish=publish,
            save_task=save_task,
            load_approval=load_approval,
            request_input=request_input,
        )

    async def _call_approval_tool(
        self, manager: RunManager, run: Any, current: Mapping[str, Any], task: ResearchTask
    ) -> None:
        """经权限门执行一次需要批准的工具（未授权时抛 :class:`ApprovalRequired`）。"""

        from app.agent.tools import ToolContext

        context = ToolContext(
            workspace=manager._context(run.workspace_id),
            app_store=manager.store,
            run_id=run.id,
            task_id=task.task_id,
            agent_kind=task.agent_kind,
            redactor=NameRedactor(("张三",)),
        )
        gate = ApprovalGate(
            manager=manager,
            registry=manager.registry,
            run=run,
            state=current,  # type: ignore[arg-type]
            question_scope=str(current.get("question") or ""),
            redactor=context.redactor,
        )
        await gate.execute(str(self.approve_tool), {"query": "张三 近况"}, context)
        self.approved = True
