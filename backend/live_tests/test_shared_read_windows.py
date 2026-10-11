from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.agent.contracts import AgentKind
from app.agent.research import RagResearcher, RunResearchState
from app.agent.tools import ToolContext, ToolName, ToolResult, build_default_registry
from app.api.deps import get_vector_store


RESULTS = Path(__file__).resolve().parents[2] / ".tmp_probe/agent-eval40-20260930/results-10002"


def recorded(case_id: str) -> dict:
    directory = RESULTS.parent / "results-10001" if case_id == "Q12" else RESULTS
    return json.loads((directory / f"{case_id}.json").read_text(encoding="utf-8"))


def actual_read_arguments(case_id: str) -> list[dict]:
    result = recorded(case_id)
    tasks = next(event["payload"]["tasks"] for event in result["events"] if event["type"] == "plan.updated")
    document_id = result["citations"][0]["source_ref"].split("#read:")[0]
    return [{"document_id": document_id, "query": task["question"], "limit": 12000}
            for task in tasks if task["agent_kind"] == "rag"]


def context() -> ToolContext:
    return ToolContext(workspace=None, vector_store=get_vector_store(), agent_kind=AgentKind.RAG)


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ["Q12", "Q16"])
async def test_real_parallel_queries_reuse_the_same_original_window(case_id: str):
    arguments = actual_read_arguments(case_id)
    assert len(arguments) == 2
    shared = RunResearchState()
    registry = build_default_registry()
    results = await asyncio.gather(*[
        shared.execute(registry, ToolName.READ_DOCUMENT, argument, context()) for argument in arguments
    ])
    assert results[0].data["text"] == results[1].data["text"]
    assert len(results[0].data["text"]) == 12000
    assert sum(result.reused for result in results) == 1
    assert all("read_document_content" not in result.to_payload() for result in results)
    assert all("read_document_content" not in result.data for result in results)


@pytest.mark.asyncio
async def test_real_different_query_window_and_explicit_offset_are_read():
    shared = RunResearchState()
    registry = build_default_registry()
    current = context()
    first = await shared.execute(registry, ToolName.READ_DOCUMENT, actual_read_arguments("Q16")[0], current)
    arguments = actual_read_arguments("Q17")[0]
    expected = await registry.execute(ToolName.READ_DOCUMENT, arguments, current)
    assert expected.data["offset"] != first.data["offset"]
    different = await shared.execute(registry, ToolName.READ_DOCUMENT, arguments, current)
    assert different.reused is False
    assert different.data == expected.data
    shorter = await shared.execute(registry, ToolName.READ_DOCUMENT, {**arguments, "offset": 100, "limit": 1000}, current)
    assert shorter.reused is False
    assert shorter.data["offset"] == 100
    assert len(shorter.data["text"]) == 1000


@pytest.mark.asyncio
async def test_restored_window_preserves_query_specific_reading():
    registry = build_default_registry()
    current = context()
    arguments = actual_read_arguments("Q16")[0]
    original = await registry.execute(ToolName.READ_DOCUMENT, arguments, current)
    restored = ToolResult.from_payload(str(ToolName.READ_DOCUMENT), original.to_payload())
    shared = RunResearchState()
    shared.restore(ToolName.READ_DOCUMENT, arguments, restored, current)
    different = await shared.execute(registry, ToolName.READ_DOCUMENT, actual_read_arguments("Q17")[0], current)
    assert different.reused is False
    assert different.data["offset"] != original.data["offset"]


@pytest.mark.asyncio
async def test_real_parent_section_preserves_independent_child_paths():
    result = recorded("Q26")
    source = result["case"]["source_refs"][0]
    parent = source["heading"].rsplit(" > ", 1)[0]
    document_id = result["citations"][0]["source_ref"].split("#read:")[0]
    evidence, _, _ = await RagResearcher()._read_sources(
        read_targets=[(document_id, parent)], allow_auto=False, run_id="eval40-read-parent",
        workspace_id="default", task_id="parent", requirement_ids=["R1"], context=context(),
        weak_chunks={}, documents_seen={}, focus=result["case"]["question"],
    )
    assert len(evidence) == 1
    assert evidence[0].heading_path == parent
    assert "3.2.3.1 提取范围" in evidence[0].content
    assert "3.2.3.2 候选事实的质量要求" in evidence[0].content
    assert source["quote"] in evidence[0].content
