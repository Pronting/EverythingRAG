from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.agent.planner import extract_requirements
from app.agent.query_plan import confirmed_document_scope, research_document_scope
from app.agent.research import text_supports
from app.agent.contracts import ResearchTask
from app.agent.research import RagResearcher, ResearchRequest
from app.agent.tools import ToolContext
from app.api.deps import get_embedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY
from app.api.deps import get_vector_store
from app.generation.knowledge_catalog import build_knowledge_catalog


RESULTS = Path(__file__).resolve().parents[2] / ".tmp_probe/agent-live-eval-20260929/results"


@pytest.mark.parametrize("case_id", ["F01", "F14"])
def test_read_original_supports_actual_fact_question(case_id):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    originals = [source["text"] for event in result["events"]
                 if event["type"] == "tool.completed" and event["payload"]["tool_name"] == "read_document"
                 for source in event["payload"]["sources"]]
    assert originals
    requirements = extract_requirements(result["question"])
    assert len(requirements) == 1
    assert any(text_supports(requirements[0].text, original) for original in originals)


@pytest.mark.parametrize("case_id", ["F10", "F11", "F12", "F13"])
def test_short_book_title_resolves_actual_index(case_id):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    catalog = build_knowledge_catalog(get_vector_store().list_blocks())
    scope = confirmed_document_scope(result["question"], catalog)
    expected = [document.document_id for document in catalog.documents
                if document.title.removesuffix(".md").strip("《》") == "影响力"]
    assert expected
    assert scope == expected


def test_same_book_unrelated_original_cannot_support_tax_question():
    result = json.loads((RESULTS / "F01.json").read_text(encoding="utf-8"))
    unrelated = [source["text"] for event in result["events"]
                 if event["type"] == "tool.completed" and event["payload"]["tool_name"] == "search_knowledge"
                 for source in event["payload"]["sources"]
                 if "产品设计的经济学" in source["title"] or "供求关系实践" in source["title"]]
    assert unrelated
    assert all(not text_supports(result["question"], body) for body in unrelated)


def test_same_book_topic_mention_cannot_support_classification_question():
    result = json.loads((RESULTS / "F14.json").read_text(encoding="utf-8"))
    unrelated = [text for _, text, metadata in get_vector_store().list_blocks()
                 if Path(str(metadata.get("source_file") or "")).name == "《认知觉醒》.md"
                 and metadata.get("heading_path") == "开启自我改变的原动力"
                 and "焦虑" in text]
    assert unrelated
    assert all(not text_supports(result["question"], body) for body in unrelated)


@pytest.mark.parametrize("case_id", ["X05", "X08"])
def test_actual_cross_document_tasks_use_their_own_source_scope(case_id):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    assert len(extract_requirements(result["question"])) == 1
    tasks = next(event["payload"]["tasks"] for event in result["events"] if event["type"] == "plan.updated")
    catalog = build_knowledge_catalog(get_vector_store().list_blocks())
    actual = [research_document_scope(result["question"], task["question"], catalog, scoped_query=True)
              for task in tasks]
    expected = [[document.document_id for document in catalog.documents
                 if Path(location["path"]).stem == document.title.removesuffix(".md")]
                for location in result["source_locations"].values()]
    assert len(expected) == 2
    assert all(len(scope) == 1 for scope in expected)
    assert actual == expected


@pytest.mark.parametrize("case_id", ["X04", "X06"])
def test_actual_combined_book_task_keeps_both_document_scopes(case_id):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    task = next(event["payload"]["tasks"][0] for event in result["events"] if event["type"] == "plan.updated")
    catalog = build_knowledge_catalog(get_vector_store().list_blocks())
    actual = research_document_scope(result["question"], task["question"], catalog, scoped_query=True)
    titles = {Path(location["path"]).stem for location in result["source_locations"].values()}
    expected = [document.document_id for document in catalog.documents
                if document.title.removesuffix(".md") in titles]
    assert len(expected) == 2
    assert set(actual) == set(expected)


def test_actual_task_without_named_document_does_not_inherit_other_object():
    result = json.loads((RESULTS / "X07.json").read_text(encoding="utf-8"))
    tasks = next(event["payload"]["tasks"] for event in result["events"] if event["type"] == "plan.updated")
    catalog = build_knowledge_catalog(get_vector_store().list_blocks())
    actual = [research_document_scope(result["question"], task["question"], catalog, scoped_query=True)
              for task in tasks]
    assert actual[0] == []
    assert actual[1] == [document.document_id for document in catalog.documents if "TencentDB Agent Memory" in document.title]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id, answer_term", [("F01", "粉红税"), ("F08", "安全边际"), ("F14", "完成焦虑")])
async def test_actual_scoped_question_retries_original_requirement_and_reads_answer(case_id, answer_term):
    result = json.loads((RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    task_data = next(event["payload"]["tasks"][0] for event in result["events"]
                     if event["type"] == "plan.updated")
    task = ResearchTask.model_validate({**task_data, "scoped_query": True})
    requirements = {item.requirement_id: item.text for item in extract_requirements(result["question"])}
    task.requirement_ids = list(requirements)
    store = get_vector_store()
    scope = confirmed_document_scope(result["question"], build_knowledge_catalog(store.list_blocks()))
    request = ResearchRequest(task=task, original_question=result["question"],
                              resolved_question=result["question"], requirement_texts=requirements,
                              document_scope=scope)
    context = ToolContext(workspace=None, vector_store=store, embedder=get_embedder(),
                          retriever=HybridRetriever(store, min_similarity=DEFAULT_MIN_SIMILARITY))
    research = await RagResearcher().research(request, context)
    record = RESULTS.parent / f"document-research-{case_id}-{uuid4().hex}.json"
    record.write_text(research.model_dump_json(indent=2), encoding="utf-8")
    assert research.evidence, research.error
    assert any(answer_term in item.content and "#read:" in item.source_ref for item in research.evidence)
    assert all(research.progress.requirement_coverage[rid].value == "answered" for rid in requirements)
