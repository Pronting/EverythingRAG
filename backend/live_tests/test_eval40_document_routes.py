from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.query_plan import confirmed_document_scope, merge_semantic_queries, stable_query_identity
from app.agent.routing import RoutingInput, evaluate_route_rules
from app.generation.knowledge_catalog import build_knowledge_catalog


DATA = Path(__file__).resolve().parents[2] / ".tmp_probe/agent-eval40-20260930"


@pytest.fixture(scope="module")
def corpus():
    blocks = json.loads((DATA / "indexed-blocks.json").read_text(encoding="utf-8"))["blocks"]
    cases = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    catalog = build_knowledge_catalog([(item["block_id"], item["text"], item) for item in blocks])
    return catalog, {item["id"]: item for item in cases}


@pytest.mark.parametrize("case_id", ["Q02", "Q03", "Q07", "Q19", "Q31", "Q33", "Q38"])
def test_named_documents_keep_local_route_and_complete_scope(corpus, case_id):
    catalog, cases = corpus
    case = cases[case_id]
    source_paths = {item["path"] for item in case["source_refs"]}
    expected = {document_id for document_id, source in catalog._source_files.items() if source in source_paths}
    assert expected
    assert set(confirmed_document_scope(case["question"], catalog)) == expected
    outcome = evaluate_route_rules(RoutingInput(
        question=case["question"], catalog_hits=catalog.search(case["question"]).matches,
        file_count=len(catalog.documents), chunk_count=4718,
    ))
    assert outcome.decision.needs_knowledge
    assert not outcome.decision.needs_web


def test_runtime_and_stm_queries_keep_distinct_subjects(corpus):
    _, cases = corpus
    record = json.loads((DATA / "results-10001/Q02.json").read_text(encoding="utf-8"))
    first = next(event["payload"]["tasks"][0]["question"] for event in record["events"] if event["type"] == "plan.updated")
    question = cases["Q02"]["question"]
    second = "航空客服 AgentCore 上篇 " + next(part for part in question.split("？") if "STM" in part)
    queries = [first, second]
    identities = [stable_query_identity(item, source_kind="knowledge", context_text=question) for item in queries]
    assert identities[0].key != identities[1].key
    merged, _ = merge_semantic_queries(queries, source_kind="knowledge", context_text=question)
    assert merged == queries
