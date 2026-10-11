from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import graph as agent_graph
from app.agent.contracts import RunStatus
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
        "clarify",
        "publish",
    }
    edges = dict(agent_graph.MAIN_GRAPH_EDGES)
    assert edges["load_snapshot"] == "route"
    assert edges["route"] == "plan"
    assert edges["plan"] == "research"
    assert edges["research"] == "after_research"
    assert edges["approval"] == "research"  # 批准后回到研究
    assert edges["synthesis"] == "publish"
    assert edges["clarify"] == "research"
    assert edges["publish"] == "END"
    branches = agent_graph.main_edges()
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
        assert services.calls.get("publish") == 1
        tasks = await app_store.list_tasks(run.id)
        assert {task.status.value for task in tasks} == {"completed"}
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


async def test_load_snapshot_skips_reader_when_memory_is_disabled() -> None:
    events: list[object] = []

    async def emit(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append((args, kwargs))

    services = GraphServices(memory_context=None, emit=emit)
    state = new_agent_state(
        run_id="run-memory-off",
        workspace_id="ws-1",
        conversation_id="conv-1",
        question="问题",
    )
    result = await agent_graph.make_load_snapshot_node(services)(state)

    assert result["memory_context"] == ""
    assert result["outcomes"][-1]["summary"] == "记忆功能已关闭"
    assert events == []
