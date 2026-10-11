import pytest

from app.agent.contracts import AgentKind, ResearchTask
from app.agent.query_plan import canonical_query
from app.agent.research import RagResearcher, ResearchRequest, build_default_registry
from app.agent.routing import RouteDecision


def request_for(original: str, task: str, resolved: str = "") -> ResearchRequest:
    return ResearchRequest(
        task=ResearchTask(task_id="query-context", question=task,
                          requirement_ids=["R1"], agent_kind=AgentKind.RAG),
        original_question=original,
        resolved_question=resolved or original,
        requirement_texts={"R1": task},
    )


@pytest.mark.parametrize("focus", ["Deployment", "Service"])
def test_supplement_preserves_described_article(focus: str) -> None:
    question = "知识库中介绍 K8S 十四个基础概念的文章，如何说明 Deployment 和 Service 各自的职责？"
    request = request_for(question, f"知识库中介绍 K8S 十四个基础概念的文章里 {focus} 的职责是什么")
    queries = RagResearcher(build_default_registry()).initial_queries(request)
    assert len(queries) == 2
    assert queries[1] == question
    assert all("K8S 十四个基础概念的文章" in query for query in queries)
    assert not {"Deployment", "Service"}.intersection(queries)


@pytest.mark.parametrize(
    ("original", "task", "resolved", "terms"),
    [
        ("按《开发实践》介绍 AI Native 和 Spec 的协作", "Spec 怎样参与",
         "按《开发实践》介绍 AI Native 和 Spec 的协作", ["《开发实践》", "AI Native", "Spec"]),
        ("依据 Harness 文章说明 tool/call 到 tool/result 的关系", "tool/call 的职责",
         "依据 Harness 文章说明 tool/call 到 tool/result 的关系", ["Harness", "tool/call 到 tool/result"]),
        ("它的成本怎样控制", "Prompt Cache 的成本",
         "《记忆工程》中的 Agent 如何控制成本", ["《记忆工程》", "Agent", "Prompt Cache"]),
    ],
)
def test_reformulation_preserves_source_and_complete_terms(
    original: str, task: str, resolved: str, terms: list[str],
) -> None:
    request = request_for(original, task, resolved)
    researcher = RagResearcher(build_default_registry())
    queries = researcher.reformulated_queries(request, ["R1"], set())
    assert queries
    assert all(term in queries[0] for term in terms)
    assert researcher.reformulated_queries(
        request, ["R1"], {canonical_query(query) for query in queries},
    ) == []


def test_entity_expansion_keeps_phrase_spaces() -> None:
    request = request_for("根据指定文章说明设计思路", "说明设计思路")
    request.decision = RouteDecision(
        question=request.original_question, resolved_question=request.original_question,
        intent="research", needs_knowledge=True, entities=["AI Native"],
    )
    queries = RagResearcher(build_default_registry()).initial_queries(request)
    assert any("指定文章" in query and "AI Native" in query for query in queries)
    assert all("AINative" not in query for query in queries)


def test_reversed_relationships_remain_distinct() -> None:
    researcher = RagResearcher(build_default_registry())
    forward = request_for("《权限设计》中 Policy 引用 Memory", "引用关系")
    reverse = request_for("《权限设计》中 Memory 引用 Policy", "引用关系")
    a = researcher.reformulated_queries(forward, ["R1"], set())[0]
    b = researcher.reformulated_queries(reverse, ["R1"], set())[0]
    assert "Policy 引用 Memory" in a
    assert "Memory 引用 Policy" in b
    assert canonical_query(a) != canonical_query(b)
