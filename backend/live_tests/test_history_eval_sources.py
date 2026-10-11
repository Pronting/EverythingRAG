from __future__ import annotations

import asyncio
import json
import sqlite3
from functools import partial
from pathlib import Path

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import Evidence
from app.agent.planner import PlannerAgent
from app.agent.research import RagResearcher, ResearchRequest, RunResearchState
from app.agent.routing import IntentAgent, RoutingInput, document_intent, source_requirement
from app.agent.runtime import _chat_config_from_settings, _synthesis_prompt, plan_history_queries, section_model_for_chat, semantic_review_decision
from app.agent.verification import SynthesisRequest, SynthesisReview
from app.agent.tools import ReadHistoryParams, SearchHistoryParams, ToolContext, ToolName, _read_history, _search_history, build_default_registry
from app.api.deps import app_db_path, get_workspace_context, workspace_history_search, workspace_history_seam


RESULTS = Path(__file__).resolve().parents[2] / ".tmp_probe/agent-live-eval-20260929/results"


@pytest.mark.parametrize("case_id", [f"H{index:02}" for index in range(1, 11)])
def test_real_history_questions_keep_history_source(case_id):
    cases = json.loads((RESULTS.parent / "cases.json").read_text(encoding="utf-8"))["cases"]
    question = next(case["question"] for case in cases if case["id"] == case_id)
    decision = IntentAgent().decide(RoutingInput(question=question, file_count=241)).decision
    assert decision.needs_history
    assert not decision.needs_knowledge
    assert not decision.needs_web
    assert source_requirement(question) == ""
    plan = PlannerAgent().plan(question, decision)
    assert plan.tasks
    assert all("历史" in task.expected_evidence for task in plan.tasks)


def test_named_source_fact_and_supplied_sentence_have_distinct_routes():
    question = json.loads((RESULTS / "F16.json").read_text(encoding="utf-8"))["question"]
    decision = IntentAgent().decide(RoutingInput(question=question, file_count=241)).decision
    assert decision.needs_knowledge
    assert not decision.needs_web
    assert document_intent("把这句话翻译成英文：《书名》很好看") == ""


def test_history_recall_with_explicit_current_official_verification_keeps_web_requirement():
    question = "请查询以前的对话：我曾要求查 Python 官方文档确认 asyncio.TaskGroup 的什么信息？找回原话，并现在核实官方文档。"
    decision = IntentAgent().decide(RoutingInput(question=question, file_count=241)).decision
    assert decision.needs_history
    assert decision.needs_web
    assert source_requirement(question)


def test_current_input_is_excluded_and_earlier_same_conversation_remains_available():
    async def run():
        workspace = get_workspace_context("default")
        store = await AppStore.open(app_db_path())
        try:
            cursor = await store.connection.execute(
                "SELECT r.id AS run_id, m.id AS message_id, m.content, m.conversation_id, m.sequence "
                "FROM runs r JOIN messages current ON current.id = r.input_message_id "
                "JOIN messages m ON m.conversation_id = r.conversation_id AND m.sequence < current.sequence "
                "WHERE m.role = 'user' AND LENGTH(m.content) BETWEEN 20 AND 200 "
                "ORDER BY r.created_at DESC, m.sequence DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            await cursor.close()
            assert row is not None
            context = ToolContext(workspace=workspace, app_store=store, run_id=row["run_id"],
                                  history_search=partial(workspace_history_seam, workspace))
            active = await store.get_run(row["run_id"])
            result = await _search_history(SearchHistoryParams(query=row["content"], limit=50), context)
            ids = {item["message_id"] for item in result.data["messages"]}
            assert row["message_id"] in ids
            assert active.input_message_id not in ids
            reading = await _read_history(ReadHistoryParams(conversation_id=row["conversation_id"],
                                                            around_sequence=row["sequence"], limit=9), context)
            read_ids = {item["message_id"] for item in reading.data["messages"]}
            assert row["message_id"] in read_ids
            assert active.input_message_id not in read_ids
        finally:
            await store.close()

    asyncio.run(run())


def test_real_repeated_follow_up_messages_keep_their_conversation_locations():
    with sqlite3.connect(app_db_path()) as connection:
        rows = connection.execute(
            "SELECT content FROM messages JOIN conversations c ON c.id = messages.conversation_id "
            "WHERE c.workspace_id = 'default' AND role = 'user' AND sequence > 1 "
            "AND LENGTH(content) BETWEEN 5 AND 100 GROUP BY content "
            "HAVING COUNT(*) BETWEEN 2 AND 10 ORDER BY COUNT(*) DESC LIMIT 5"
        ).fetchall()
        assert rows
        search = workspace_history_search(get_workspace_context("default"))
        try:
            for (content,) in rows:
                expected = {row[0] for row in connection.execute(
                    "SELECT messages.id FROM messages JOIN conversations c ON c.id = messages.conversation_id "
                    "WHERE c.workspace_id = 'default' AND content = ? AND role = 'user' AND sequence > 1", (content,)
                )}
                hits = search.search(content, limit=100, distinct_content=True)
                assert expected <= {hit.message_id for hit in hits}
        finally:
            search.close()


def test_history_cache_respects_queries_read_ranges_and_run_snapshot():
    async def run():
        workspace = get_workspace_context("default")
        store = await AppStore.open(app_db_path())
        try:
            baseline = json.loads((RESULTS / "H06.json").read_text(encoding="utf-8"))
            other = json.loads((RESULTS / "H07.json").read_text(encoding="utf-8"))
            context = ToolContext(workspace=workspace, app_store=store, run_id=baseline["run_id"],
                                  history_search=partial(workspace_history_seam, workspace))
            shared = RunResearchState()
            registry = build_default_registry()
            assert not await shared.can_reuse(ToolName.SEARCH_HISTORY, {"query": "Python", "limit": 30}, context)
            first = await shared.execute(registry, ToolName.SEARCH_HISTORY, {"query": "Python", "limit": 30}, context)
            assert not await shared.can_reuse(ToolName.SEARCH_HISTORY, {"query": "Python 字典键", "limit": 30}, context)
            specific = await shared.execute(registry, ToolName.SEARCH_HISTORY, {"query": "Python 字典键", "limit": 30}, context)
            assert first.data["messages"] and specific.data["messages"]
            hit = specific.data["messages"][0]
            arguments = {"conversation_id": hit["conversation_id"], "around_sequence": hit["sequence"], "limit": 9}
            readings = await asyncio.gather(*(shared.execute(registry, ToolName.READ_HISTORY, arguments, context) for _ in range(2)))
            assert any(reading.reused and reading.reuse_reason == "工具、参数与知识版本一致" for reading in readings)
            assert readings[0].data == readings[1].data
            assert not await shared.can_reuse(ToolName.READ_HISTORY, {**arguments, "limit": 11}, context)
            wider = await shared.execute(registry, ToolName.READ_HISTORY, {**arguments, "limit": 11}, context)
            assert wider.data["messages"]
            other_context = ToolContext(workspace=workspace, app_store=store, run_id=other["run_id"],
                                        history_search=partial(workspace_history_seam, workspace))
            assert not await shared.can_reuse(ToolName.READ_HISTORY, arguments, other_context)
            another_run = await shared.execute(registry, ToolName.READ_HISTORY, arguments, other_context)
            assert another_run.data["messages"]
        finally:
            await store.close()

    asyncio.run(run())


def test_recorded_history_time_and_count_claims_require_semantic_review():
    result = json.loads((RESULTS.parent / "results-repaired-history-final/H07.json").read_text(encoding="utf-8"))
    directory = Path.home() / ".everything-rag/workspaces/default/artifacts/synthesis" / result["run_id"]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("review-*.json")]
    record = next(item for item in records if item.get("accepted") and item.get("evidence"))
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    request = SynthesisRequest(question=record["question"], requirements=record["requirements"], evidence=evidence)
    review = SynthesisReview.model_validate(record["review"])
    assert semantic_review_decision(request, review, evidence) == (True, "history_time_or_count_claim")


def test_actual_history_message_timestamps_reach_synthesis():
    result = json.loads((RESULTS / "H07.json").read_text(encoding="utf-8"))
    question = result["question"]
    decision = IntentAgent().decide(RoutingInput(question=question, file_count=241)).decision
    plan = PlannerAgent().plan(question, decision, retrieval_queries=["asyncio.TaskGroup"])

    async def run():
        workspace = get_workspace_context("default")
        store = await AppStore.open(app_db_path())
        try:
            context = ToolContext(workspace=workspace, app_store=store, run_id=result["run_id"],
                                  history_search=partial(workspace_history_seam, workspace))
            research = await RagResearcher().research(ResearchRequest(task=plan.tasks[0], original_question=question), context)
            assert research.evidence
            prompt = _synthesis_prompt(SynthesisRequest(question=question), research.evidence, structured=True)
            for item in research.evidence:
                message = await store.get_message(item.source_ref.split("#")[1])
                assert f"消息时间：{message.created_at}" in item.content
                assert message.created_at in prompt
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("case_id", [f"H{index:02}" for index in range(1, 11)])
def test_real_model_queries_read_original_history_messages(case_id):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    question = result["question"]
    query_plan = plan_history_queries(question, _chat_config_from_settings(), "off")
    decision = IntentAgent().decide(RoutingInput(question=question, file_count=241)).decision
    plan = PlannerAgent().plan(question, decision, retrieval_queries=query_plan.queries)
    expected = {source["message_id"] for source in result["source_locations"].values()}

    async def run():
        workspace = get_workspace_context("default")
        store = await AppStore.open(app_db_path())
        try:
            context = ToolContext(workspace=workspace, app_store=store, run_id=result["run_id"],
                                  history_search=partial(workspace_history_seam, workspace))
            existing_run = await store.get_run(result["run_id"])
            evidence = []
            for task in plan.tasks:
                hits = await _search_history(SearchHistoryParams(query=task.question, limit=30), context)
                assert all(item["message_id"] != existing_run.input_message_id for item in hits.data["messages"])
                research = await RagResearcher(section_model=section_model_for_chat(reasoning_level="off", history=True)).research(
                    ResearchRequest(task=task, original_question=question), context,
                )
                evidence.extend(research.evidence)
            refs = {item.source_ref.split("#")[1] for item in evidence}
            for message_id in expected:
                original = await store.get_message(message_id)
                matches = []
                for reference in refs:
                    message = await store.get_message(reference)
                    if message.role == "user" and message.content == original.content and message.created_at < existing_run.created_at:
                        matches.append(message.id)
                assert matches, {"case_id": case_id, "queries": query_plan.queries, "missing": message_id}
                print(json.dumps({"case_id": case_id, "expected_message_id": message_id,
                                  "original_message_ids": matches, "queries": query_plan.queries}, ensure_ascii=False))
        finally:
            await store.close()

    asyncio.run(run())
