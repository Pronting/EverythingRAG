"""LangGraph 拓扑与图行为测试（PRD 4.3 / 7.4 / 8.2，验收 A13–A19 的结构性部分）。

覆盖：
1. **边表**：主图节点与 ``MAIN_GRAPH_EDGES`` 覆盖 PRD 4.3 全部边（含 Verifier 的
   revise / research / clarify 三条回边），研究子图四条边；
2. **可编译**：``build_main_graph`` / ``build_research_graph`` 用真实 LangGraph 编译成功；
3. **两条驱动同语义**：LangGraph 驱动与 :func:`run_pipeline` 离线驱动得出结论一致；
4. **回边**：``verdict=revise`` 触发重新合成后才发布；
5. **失败收敛**：子任务失败 → 主图不进入合成/核对，直接以错误状态结束；
6. **运行保护**：回边次数上限（``REVISION_MAX_ROUNDS``）后收敛，不再无限重试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import graph as agent_graph
from app.agent.contracts import RunStatus, VerifierVerdict
from app.agent.models import GraphServices, new_agent_state
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, seed_conversation

# =====================================================================
# 拓扑
# =====================================================================


def test_main_graph_covers_prd_4_3_edges() -> None:
    nodes = set(agent_graph.MAIN_GRAPH_NODES)
    assert nodes == {
        "load_snapshot",
        "route",
        "plan",
        "research",
        "approval",
        "synthesis",
        "verification",
        # t31 追加：回边计数节点（LangGraph 路径此前从不递增 revisions ⇒ 回边绕到
        # recursion_limit；离线驱动在同一处已递增，这里把语义补齐）
        "revise",
        "clarify",
        "publish",
    }
    edges = dict(agent_graph.MAIN_GRAPH_EDGES)
    assert edges["load_snapshot"] == "route"
    assert edges["route"] == "plan"
    assert edges["plan"] == "research"
    assert edges["research"] == "after_research"
    assert edges["approval"] == "research"  # 批准后回到研究
    assert edges["synthesis"] == "verification"
    assert edges["verification"] == "after_verification"
    assert edges["revise"] == "after_revision"
    assert edges["clarify"] == "research"
    assert edges["publish"] == "END"
    # 条件边：Verifier 的三条回边经 revise 计数后分派 + 发布 + 研究层的三条分支
    branches = agent_graph.main_edges()
    assert set(branches["after_verification"]) == {"publish", "revise"}
    assert set(branches["after_revision"]) == {"synthesis", "research", "clarify"}
    assert set(branches["after_research"]) == {"synthesis", "approval", "failed"}


def test_research_subgraph_edges() -> None:
    edges = dict(agent_graph.RESEARCH_GRAPH_EDGES)
    assert edges == {
        "prepare": "run",
        "run": "assess",
        "assess": "after_assess",
        "collect": "END",
    }
    assert agent_graph.RESEARCH_GRAPH_NODES == ("prepare", "run", "assess", "collect")


def test_graphs_compile_with_langgraph() -> None:
    services = GraphServices()
    main = agent_graph.build_main_graph(services)
    research = agent_graph.build_research_graph(services)
    assert main is not None, "LangGraph 未安装或编译失败"
    assert research is not None
    assert set(main.get_graph().nodes) >= set(agent_graph.MAIN_GRAPH_NODES)


def test_after_verification_branches() -> None:
    base = {"revisions": 0}

    def state(verdict: VerifierVerdict) -> dict:
        return {**base, "verification": {"verdict": verdict.value}}

    assert agent_graph.after_verification(state(VerifierVerdict.PASS)) == "publish"
    assert agent_graph.after_verification(state(VerifierVerdict.INSUFFICIENT)) == "publish"
    assert agent_graph.after_verification(state(VerifierVerdict.REVISE)) == "synthesis"
    assert agent_graph.after_verification(state(VerifierVerdict.RESEARCH)) == "research"
    assert agent_graph.after_verification(state(VerifierVerdict.CLARIFY)) == "clarify"
    # 回边用尽后收敛到发布（由发布节点决定能否带缺口发布）
    exhausted = {"revisions": agent_graph.REVISION_MAX_ROUNDS, "verification": {"verdict": "revise"}}
    assert agent_graph.after_verification(exhausted) == "publish"


def test_after_research_priorities() -> None:
    assert agent_graph.after_research({"pending_approval": {"approval_id": "ap-1"}}) == "approval"
    assert agent_graph.after_research({"error": {"error_code": "x"}}) == "failed"
    assert agent_graph.after_research({}) == "synthesis"


# =====================================================================
# 图行为（经 RunManager，两条驱动都要一致）
# =====================================================================


@pytest.mark.parametrize("use_langgraph", [True, False])
async def test_pipeline_reaches_publish_on_both_drivers(app_store, tmp_path: Path, use_langgraph: bool) -> None:
    await seed_conversation(app_store)
    services = FakeServices(task_count=2)
    manager = build_manager(
        app_store, services=services, root=tmp_path, use_langgraph=use_langgraph
    )
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="测试问题"
        )
        await manager.run_to_completion(run.id)
        done = await manager.require_run(run.id)
        assert done.status is RunStatus.COMPLETED
        answers = await app_store.get_answer(done.answer_id)
        assert answers is not None and answers.content
        assert services.calls.get("route") == 1
        assert services.calls.get("plan") == 1
        assert services.calls.get("synthesize") == 1
        assert services.calls.get("verify") == 1
        assert services.calls.get("publish") == 1
        tasks = await app_store.list_tasks(run.id)
        assert {task.status.value for task in tasks} == {"completed"}
    finally:
        await manager.aclose()


async def test_revise_verdict_loops_back_to_synthesis(app_store, tmp_path: Path) -> None:
    """``revise`` 回边：先合成 → 核对不通过 → 再合成 → 通过才发布（PRD 8.2）。"""

    class Flaky(FakeServices):
        def factory(self, manager, run, state):  # type: ignore[no-untyped-def]
            services = super().factory(manager, run, state)

            async def verify(current):  # type: ignore[no-untyped-def]
                count = self.calls.get("verify", 0) + 1
                self.calls["verify"] = count
                verdict = VerifierVerdict.REVISE if count == 1 else VerifierVerdict.PASS
                from app.agent.verification import VerificationResult

                return VerificationResult(verdict=verdict, checked_at="2026-09-19T08:00:00Z")

            services.verify = verify
            return services

    await seed_conversation(app_store)
    services = Flaky(task_count=1)
    manager = build_manager(app_store, services=services, root=tmp_path)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="测试问题"
        )
        await manager.run_to_completion(run.id)
        done = await manager.require_run(run.id)
        assert done.status is RunStatus.COMPLETED
        assert services.calls["verify"] == 2  # 走了一次回边
        assert services.calls["synthesize"] == 2
        assert services.calls["publish"] == 1
    finally:
        await manager.aclose()


async def test_task_failure_stops_before_synthesis(app_store, tmp_path: Path) -> None:
    await seed_conversation(app_store)
    services = FakeServices(task_count=2, fail_tasks=(1,))
    manager = build_manager(app_store, services=services, root=tmp_path)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="测试问题"
        )
        await manager.run_to_completion(run.id)
        done = await manager.require_run(run.id)
        assert done.status is RunStatus.FAILED
        assert "synthesize" not in services.calls  # 子任务失败不进合成
        assert "publish" not in services.calls
        tasks = await app_store.list_tasks(run.id)
        failed = [task for task in tasks if task.status.value == "failed"]
        assert [task.id for task in failed] == ["t1"]
        # 已完成的有效结果保留（PRD 12.2）
        state = manager._states[run.id]
        assert state["failed_task_ids"] == ["t1"]
    finally:
        await manager.aclose()


def test_offline_pipeline_runs_without_manager() -> None:
    """:func:`run_pipeline` 在没有任何运行层依赖时也能跑（纯服务注入）。"""

    services = GraphServices()
    assert services.recursion_limit == 1000  # PRD 7.4 运行保护默认值
    state = new_agent_state(
        run_id="run-x", workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    assert state["phase"] == "load_snapshot"
    assert agent_graph.MAIN_GRAPH_NODES[0] == "load_snapshot"
