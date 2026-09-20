"""图收敛回归（t31）：主图必须收敛，而不是绕到 ``recursion_limit``。

背景（t31 实测）：``after_verification`` 靠 ``state["revisions"]`` 判定「修订/补研次数用尽」，
但**只有离线驱动** ``run_pipeline`` 会递增它 —— LangGraph 路径下计数器恒为 0，于是任何
``revise`` / ``research`` 结论的回边都会一直绕到 ``recursion_limit``，用户看到的是
``GraphRecursionError`` 而不是 PRD 8.3 的「按资料不足发布」。

本文件把「回边按上限收敛」锁死：核对结论**永远**是 ``research`` 时也必须收尾，且核对轮次
不超过 ``REVISION_MAX_ROUNDS + 1``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.agent import graph as agent_graph
from app.agent.contracts import RunStatus, VerifierVerdict
from app.agent.runtime import RunManager
from app.agent.verification import VerificationResult
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.fakes_run import FakeServices, StubContextFactory, build_registry, seed_conversation


async def test_revision_backedges_converge_instead_of_recursing(
    tmp_path: Path, app_store
) -> None:
    """核对结论始终不可发布时，回边必须按 ``REVISION_MAX_ROUNDS`` 收敛。"""

    await seed_conversation(app_store)
    base = FakeServices(task_count=2)

    def factory(manager: RunManager, run: Any, state: Any) -> Any:
        services = base.factory(manager, run, state)

        async def always_research(current: Any) -> VerificationResult:
            del current
            return VerificationResult(verdict=VerifierVerdict.RESEARCH)

        return services.model_copy(update={"verify": always_research})

    manager = RunManager(
        store=app_store,
        context_factory=StubContextFactory(root=tmp_path),
        registry=build_registry(),
        services_factory=factory,
        use_langgraph=True,
    )
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        await manager.run_to_completion(run.id)
        final = await manager.require_run(run.id)

        assert final.error_code != "GraphRecursionError", "回边不得绕到 recursion_limit"
        assert final.status in {RunStatus.COMPLETED_WITH_GAPS, RunStatus.FAILED}
        rounds = [
            event
            for event in await manager.read_events(run.id)
            if event.type.value == "verification.updated"
        ]
        assert len(rounds) <= agent_graph.REVISION_MAX_ROUNDS + 1, (
            f"核对轮次应受 REVISION_MAX_ROUNDS 约束，实际 {len(rounds)} 次"
        )
        # 次数用尽 → 降级为 insufficient 并附一条说明（PRD 8.3）
        state = manager._states.get(run.id) or {}
        verification = VerificationResult.model_validate(state.get("verification") or {})
        assert verification.verdict is VerifierVerdict.INSUFFICIENT
        assert any("用尽" in issue.statement for issue in verification.issues)
        # 降级后可发布：答案按「资料不足」发布，而不是把整次运行判死
        assert final.status is RunStatus.COMPLETED_WITH_GAPS
        assert final.answer_id
    finally:
        await manager.aclose()


def test_main_graph_carries_the_revision_counter_node() -> None:
    """结构锁：LangGraph 主图必须有 ``revise`` 计数节点，否则「次数用尽」永不成立。"""

    assert "revise" in agent_graph.MAIN_GRAPH_NODES
    assert dict(agent_graph.MAIN_GRAPH_EDGES)["revise"] == "after_revision"
    branches = agent_graph.main_edges()
    assert branches["after_verification"] == ("publish", "revise")
    assert branches["after_revision"] == ("synthesis", "research", "clarify")


@pytest.mark.parametrize("verdict", list(VerifierVerdict))
def test_every_verdict_is_routable(verdict: VerifierVerdict) -> None:
    """五种 verdict 都必须有去处（不留死路），且非发布结论经 ``revise`` 后仍能到位。"""

    state: Any = {
        "verification": VerificationResult(verdict=verdict).model_dump(mode="json"),
        "revisions": 0,
    }
    follow = agent_graph.after_verification(state)
    # ``after_verification`` 返回的是**语义目标**；图里再由条件边映射把非发布目标
    # 引到 ``revise`` 计数节点，最后 ``after_revision`` 分派回同一语义目标。
    assert follow in {"publish", "synthesis", "research", "clarify"}
    assert set(agent_graph.main_edges()["after_verification"]) == {"publish", "revise"}
    if follow != "publish":
        assert follow in agent_graph.main_edges()["after_revision"]
