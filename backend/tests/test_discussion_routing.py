import pytest

from app.agent.contracts import RouteDecision
from app.agent.planner import PlannerAgent, _sources_for, extract_requirements
from app.agent.routing import IntentAgent, RoutingHistoryTurn, RoutingInput, discussion_request

HISTORY = [
    RoutingHistoryTurn(role="user", content="对于一些云端agent，货拉拉有什么做法实践？"),
    RoutingHistoryTurn(role="assistant", content="文章提到了异步云端 Agent，并说明相关流程仍在探索。"),
]


@pytest.mark.parametrize("question", [
    "你有什么补充的说法？", "我说你怎么看货拉拉他们的做法？",
    "那你怎么看这个方案？", "说说你的看法", "你有什么建议？",
    "你觉得这个思路怎么样？", "What do you think?",
])
def test_discussion_uses_current_conversation(question):
    result = IntentAgent().decide(RoutingInput(question=question, recent_history=HISTORY))
    assert result.decision.intent.value == "general"
    assert result.decision.needs_general
    assert not result.decision.needs_source_search
    assert not result.decision.needs_knowledge
    assert not result.decision.needs_history
    assert not result.decision.needs_web
    plan = PlannerAgent().plan(question, result.decision)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].agent_kind.value == "general"


@pytest.mark.parametrize("question", [
    "你怎么看他们的做法？请引用原文依据。",
    "你有什么补充的资料？", "原文还有什么补充的说法？",
    "你怎么看这个方案？联网查证一下。",
    "你怎么看这个方案？查询最新进展。",
    "根据文档说说你的看法", "你觉得它的实际效果怎么样？",
    "你有什么补充的说法？请查询历史记录。",
])
def test_explicit_source_requests_keep_research(question):
    assert not discussion_request(question, has_history=True)


def test_discussion_without_context_requires_routing():
    assert not discussion_request("你有什么补充的说法？", has_history=False)


def test_general_analysis_is_planned_with_knowledge():
    question = "我说你怎么看货拉拉他们的做法？"
    decision = RouteDecision(
        question=question, resolved_question=question, intent="research",
        needs_knowledge=True, needs_general=True,
    )
    sources = _sources_for(extract_requirements(question)[0], decision)
    assert [source.value for source in sources] == ["knowledge", "general"]
