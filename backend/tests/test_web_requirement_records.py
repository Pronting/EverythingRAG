import json
import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.agent.research import (
    ResearchResult,
    _web_parameter_names,
    attribute_requirements,
    support_tokens,
)
from app.agent.runtime import (
    _execution_notes,
    _refresh_web_results,
    _semantic_assessment_prompt,
    _web_limit_notes,
)
from app.agent.verification import SynthesisRequest, SynthesisReview


def saved_state(run_id):
    database = Path.home() / ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/checkpoints.db"
    if not database.exists():
        pytest.skip("本地真实 T20 记录未提供")
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT type,checkpoint FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id DESC LIMIT 1",
            (f"run:{run_id}:main",),
        ).fetchone()
    if row is None:
        pytest.skip("本地真实 T20 checkpoint 未提供")
    return JsonPlusSerializer().loads_typed(row)["channel_values"]


@pytest.fixture
def actual_state():
    return saved_state("run-13437faf9d1044cf82629ae80703ca80")


def test_actual_v07_resume_preserves_parameter_sources():
    path = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-repaired-r3-20260922/V07-evidence.json"
    if not path.exists():
        pytest.skip("本地真实 V07 记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    state = saved_state(record["browser"]["run_id"])
    results = [ResearchResult.model_validate(item) for item in state["task_results"].values()]
    _refresh_web_results(results, state, state["requirement_texts"])
    assert any(item.supported_requirements and str(item.source_type) == "web"
               for result in results for item in result.evidence)


def test_actual_v08_verifier_receives_off_restriction():
    state = saved_state("run-e0009aa87aa24f8099db50d835fb71d3")
    limits = _web_limit_notes(state)
    assert limits and "允许联网=False" in limits[0]
    request = SynthesisRequest(question=state["question"], requirements=state["requirement_texts"], evidence=[])
    review = SynthesisReview(paragraphs=[], covered_requirements=[], gaps=[])
    prompt = _semantic_assessment_prompt(request, review, [], [], limits=limits)
    packet = json.loads(prompt.partition("\n待验收数据：\n")[2])
    assert packet["execution_limits"] == limits


def test_real_web_parameters_reach_requirements(actual_state):
    task = ResearchResult.model_validate(actual_state["task_results"]["t-q2-web"])
    question = next(item["question"] for item in actual_state["plan"]["tasks"] if item["task_id"] == task.task_id)
    evidence = attribute_requirements(task.evidence, actual_state["requirement_texts"], task_question=question)
    supported = [item.source_ref for item in evidence if item.supported_requirements]
    assert any(url.endswith("API_UpdateMemory.html") for url in supported)
    assert any(url.endswith("API_HarnessManagedMemoryConfiguration.html") for url in supported)
    assert not any(url.endswith("API_Memory.html") for url in supported)


@pytest.mark.parametrize("question", [
    "请根据 AWS 官方资料说明 Bedrock AgentCore Memory 事件保留时长的允许范围",
    "核实 Bedrock AgentCore Memory 的 eventExpiryDuration 最小值和最大值",
])
def test_actual_sources_support_alternative_wording(actual_state, question):
    task = ResearchResult.model_validate(actual_state["task_results"]["t-q2-web"])
    attributed = attribute_requirements(task.evidence, {"R1": question},
                                        task_question="Bedrock AgentCore Memory event expiry duration range")
    assert any(item.supported_requirements and "UpdateMemory" in item.source_ref for item in attributed)


def test_actual_unrelated_parameters_remain_uncovered(actual_state):
    task = ResearchResult.model_validate(actual_state["task_results"]["t-q2-web"])
    attributed = attribute_requirements(task.evidence,
                                        {"R1": "Bedrock AgentCore Memory payloadByteLimit 的范围"},
                                        task_question="Bedrock AgentCore Memory payload byte limit range")
    assert all(not item.supported_requirements for item in attributed)


def test_actual_execution_notes_include_web_tools(actual_state):
    results = [ResearchResult.model_validate(item) for item in actual_state["task_results"].values()]
    notes = " ".join(_execution_notes(actual_state, results))
    web_count = sum(
        item.source_type.value == "web" for result in results for item in result.evidence
    )
    assert "web_search" in notes and "read_web_page" in notes
    assert f"网页证据 {web_count} 条" in notes


def test_actual_resume_reattributes_existing_pages(actual_state):
    results = [ResearchResult.model_validate(item) for item in actual_state["task_results"].values()]
    before = [(item.evidence_id, item.content) for result in results for item in result.evidence]
    _refresh_web_results(results, actual_state, actual_state["requirement_texts"])
    assert before == [(item.evidence_id, item.content) for result in results for item in result.evidence]
    web = next(result for result in results if result.task_id == "t-q2-web")
    assert any(item.supported_requirements and "UpdateMemory" in item.source_ref for item in web.evidence)
    assert any(item.evidence_ids for item in web.conclusions)


def test_parameter_query_uses_real_names_without_search_operators(actual_state):
    task = ResearchResult.model_validate(actual_state["task_results"]["t-q2-web"])
    question = "site:docs.aws.amazon.com bedrock-agentcore memory event expiry duration valid range retention"
    names = [name for item in task.evidence for name in _web_parameter_names(question, item.content)]
    assert "eventExpiryDuration" in names
    assert all("site" != name and " " not in name and ".com" not in name for name in names)


def test_parameter_name_is_recovered_from_equivalent_range_wording():
    content = """## CreateMemory

`eventExpiryDuration` controls when raw events expire.
"""
    names = _web_parameter_names(
        "AgentCore Memory event retention duration allowable range", content
    )
    assert names == ["eventExpiryDuration"]


def test_english_spaces_preserve_word_boundaries():
    tokens = support_tokens("Bedrock AgentCore Memory 和 AI Native")
    assert {"bedrock", "agentcore", "memory", "native"}.issubset(tokens)
    assert "bedrockagentcorememory" not in tokens
