from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pytest
import xxhash

from app.agent.contracts import ResearchTask
from app.agent.research import (
    RagResearcher,
    ResearchRequest,
    _cover_named_document_sections,
    evidence_ready_for_answer,
)
from app.agent.tools import (
    ReadDocumentParams,
    ToolContext,
    ToolStatus,
    _ordered_document_blocks,
    _read_document,
    document_id_of_block,
)
from app.api.deps import get_embedder, get_vector_store
from app.ingestion.chunker import chunk_document
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.pipeline import IngestionPipeline, make_block_id, read_text
from app.ingestion.scanner import discover_file


def _real_document_with_lexical_disorder(min_chars: int = 0):
    store = get_vector_store()
    grouped = defaultdict(list)
    for block in store.list_blocks():
        grouped[document_id_of_block(block[0])].append(block)
    for document_id, blocks in grouped.items():
        if len(blocks) < 11:
            continue
        source = Path(str(blocks[0][2].get("source_file") or ""))
        if not source.is_file() or xxhash.xxh64(source.read_bytes()).hexdigest() != blocks[0][2].get("doc_id"):
            continue
        chunks = chunk_document(parse_markdown(read_text(source)), str(source))
        order = {chunk.block_id: index for index, chunk in enumerate(chunks)}
        keys = [re.sub(r"-s\d+$", "", block_id.rsplit("::", 1)[-1]) for block_id, _, _ in blocks]
        if not all(key in order for key in keys):
            continue
        expected = sorted(
            blocks,
            key=lambda block: (
                order[re.sub(r"-s\d+$", "", block[0].rsplit("::", 1)[-1])],
                int(re.search(r"-s(\d+)$", block[0]).group(1)) if re.search(r"-s(\d+)$", block[0]) else 0,
            ),
        )
        if sum(len(block[1]) for block in expected) < min_chars:
            continue
        lexical = sorted(blocks, key=lambda block: (str(block[2].get("heading_path") or ""), block[0]))
        if [block[0] for block in expected] != [block[0] for block in lexical]:
            return store, document_id, expected
    pytest.fail("真实索引缺少可验证的章节顺序样本")


def test_existing_index_recovers_order_for_available_matching_sources():
    grouped = defaultdict(list)
    for block in get_vector_store().list_blocks():
        grouped[document_id_of_block(block[0])].append(block)
    checked = 0
    errors = []
    for document_id, blocks in grouped.items():
        source = Path(str(blocks[0][2].get("source_file") or ""))
        if not source.is_file() or xxhash.xxh64(source.read_bytes()).hexdigest() != blocks[0][2].get("doc_id"):
            continue
        try:
            ordered = _ordered_document_blocks(blocks)
        except Exception as exc:
            errors.append((document_id, type(exc).__name__))
            continue
        assert len(ordered) == len(blocks)
        checked += 1
    assert checked > 0
    assert not errors, f"有 {len(errors)} 篇版本一致的旧文档无法恢复原始顺序"


def test_real_image_descriptions_keep_markdown_positions():
    grouped = defaultdict(list)
    for block in get_vector_store().list_blocks():
        grouped[document_id_of_block(block[0])].append(block)
    for blocks in grouped.values():
        if not any(meta.get("chunk_type") == "image_description" for _, _, meta in blocks):
            continue
        if not any(meta.get("chunk_type") == "text" for _, _, meta in blocks):
            continue
        source = Path(str(blocks[0][2].get("source_file") or ""))
        if not source.is_file() or xxhash.xxh64(source.read_bytes()).hexdigest() != blocks[0][2].get("doc_id"):
            continue
        chunks = chunk_document(parse_markdown(read_text(source)), str(source))
        positions = {chunk.block_id: index for index, chunk in enumerate(chunks)}
        if not all(block_id.split("::", 2)[-1] in positions for block_id, _, _ in blocks):
            continue
        actual = _ordered_document_blocks(blocks)
        assert [block[0] for block in actual] == [
            block[0] for block in sorted(
                blocks, key=lambda item: positions[item[0].split("::", 2)[-1]]
            )
        ]
        assert any(meta.get("chunk_type") == "image_description" for _, _, meta in actual)
        return
    pytest.fail("真实索引缺少可核对来源的图文混合文档")


def test_real_image_task_survives_order_column_migration_and_reopen():
    source_db = Path.home() / ".everything-rag/image_state.db"
    assert source_db.is_file()
    copied_db = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-read-order/image-state.db"
    copied_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(copied_db) as target:
            source.backup(target)
    state = ImageStateStore(copied_db)
    try:
        with state._lock:
            candidates = state._conn.execute(
                "SELECT block_id FROM image_tasks WHERE status='failed'"
            ).fetchall()
        for row in candidates:
            task = state.get_task(str(row["block_id"]))
            assert task is not None
            source = Path(task.source_file)
            if not source.is_file() or xxhash.xxh64(source.read_bytes()).hexdigest() != task.doc_id:
                continue
            chunks = chunk_document(parse_markdown(read_text(source)), str(source))
            matching = [
                (index, chunk) for index, chunk in enumerate(chunks)
                if chunk.kind == "image" and
                make_block_id(task.doc_id, chunk.block_id, source_file=source) == task.block_id
            ]
            if not matching:
                continue
            position, chunk = matching[0]
            discovered = discover_file(source)
            assert discovered is not None
            restored = IngestionPipeline(
                get_embedder(), get_vector_store(), state_store=state
            ).make_image_task(
                discovered, chunk, document_order=position
            )
            assert restored is not None
            assert restored.block_id == task.block_id
            assert state.add_task(restored)
            assert state.get_task(task.block_id).document_order == position
            state.close()
            reopened = ImageStateStore(copied_db)
            try:
                assert reopened.get_task(task.block_id).document_order == position
                assert any(item.block_id == task.block_id and item.document_order == position
                           for item in reopened.load_pending())
            finally:
                reopened.close()
            return
        pytest.fail("真实失败图片任务缺少版本一致的 Markdown 来源")
    finally:
        state.close()


@pytest.mark.asyncio
async def test_real_index_read_document_preserves_markdown_order_and_continuation():
    store, document_id, expected = _real_document_with_lexical_disorder()
    context = ToolContext(workspace=None, vector_store=store)
    expected_text = "\n\n".join(
        f"章节：{block[2]['heading_path']}\n{block[1]}" if block[2].get("heading_path") else block[1]
        for block in expected
    )
    pieces = []
    offset = 0
    while True:
        result = await _read_document(ReadDocumentParams(document_id=document_id, limit=12000, offset=offset), context)
        pieces.append(result.data["text"])
        if result.data["next_offset"] is None:
            break
        assert result.data["next_offset"] > offset
        offset = result.data["next_offset"]
    assert "".join(pieces) == expected_text
    assert result.data["total_chars"] == len(expected_text)
    assert result.data["document_revision"] == expected[0][2]["doc_id"]


@pytest.mark.asyncio
async def test_real_index_heading_filter_and_missing_read_cases():
    store, document_id, blocks = _real_document_with_lexical_disorder()
    context = ToolContext(workspace=None, vector_store=store)
    heading = next(str(meta["heading_path"]) for _, _, meta in blocks if meta.get("heading_path"))
    filtered = await _read_document(
        ReadDocumentParams(document_id=document_id, heading=heading), context
    )
    no_heading = await _read_document(
        ReadDocumentParams(document_id=document_id, heading="不存在的章节名称"), context
    )
    unknown = await _read_document(
        ReadDocumentParams(document_id="real-index-document-missing"), context
    )
    at_end = await _read_document(
        ReadDocumentParams(document_id=document_id, offset=10000000), context
    )
    assert filtered.status is ToolStatus.OK
    assert heading in filtered.data["headings"]
    assert filtered.data["text"].startswith(f"章节：{heading}\n")
    assert no_heading.status is ToolStatus.EMPTY
    assert no_heading.data["code"] == "heading_not_found"
    assert unknown.status is ToolStatus.NOT_FOUND
    assert unknown.data["code"] == "document_not_found"
    assert at_end.status is ToolStatus.EMPTY
    assert at_end.data["code"] == "end_of_document"


@pytest.mark.asyncio
async def test_real_index_long_document_windows_support_answer_and_citation():
    store, document_id, _ = _real_document_with_lexical_disorder(24000)
    context = ToolContext(workspace=None, vector_store=store)
    researcher = RagResearcher()
    windows, _, attempted = await researcher._read_sources(
        read_targets=[(document_id, "")], allow_auto=False,
        run_id="real-index-read", workspace_id="default", task_id="read",
        requirement_ids=["R1"], context=context,
        weak_chunks={}, documents_seen={}, focus="",
    )
    assert len(windows) >= 3
    assert len(attempted) == len(windows)
    assert all(evidence_ready_for_answer(item) for item in windows)
    assert all(
        any("引用范围以该窗口为准" in limitation for limitation in item.limitations)
        for item in windows
    )
    assert windows[1].source_start == windows[0].source_end
    assert windows[1].source_end > windows[1].source_start
    coverage = researcher._coverage(["R1"], windows, {}, {})
    request = ResearchRequest(task=ResearchTask(task_id="read", question="文档内容", requirement_ids=["R1"]))
    conclusion = researcher._conclusions(request, windows, coverage)[0]
    assert set(conclusion.evidence_ids) == {item.evidence_id for item in windows}


@pytest.mark.asyncio
async def test_named_openclaw_document_is_read_after_formula_section_selection():
    store = get_vector_store()
    candidates = {}
    formula_target = None
    named_document_id = None
    for block_id, text, metadata in store.list_blocks():
        source = Path(str(metadata.get("source_file") or ""))
        heading = str(metadata.get("heading_path") or "")
        document_id = document_id_of_block(block_id)
        if "记忆系统的工程实践与演进" in source.name and "4.2 路径二" in heading:
            formula_target = (document_id, heading)
            formula_candidate = candidates.setdefault(formula_target, {
                "document_title": source.name,
                "heading": heading,
                "excerpts": [],
            })
            if len(formula_candidate["excerpts"]) < 3:
                formula_candidate["excerpts"].append(text[:2000])
        if "Harness视角破解Agent应用Token" not in source.name:
            continue
        named_document_id = document_id
        if not heading.startswith("二、Harness视角下的解决方案 > 2."):
            continue
        candidate = candidates.setdefault((document_id, heading), {
            "document_title": source.name,
            "heading": heading,
            "excerpts": [],
        })
        if len(candidate["excerpts"]) < 3:
            candidate["excerpts"].append(text[:2000])
    assert formula_target is not None
    assert named_document_id is not None
    assert len(candidates) >= 3
    question = "本地资料 OpenClaw 记忆淘汰机制 触发条件 评分公式 参数取值"
    request = ResearchRequest(task=ResearchTask(
        task_id="openclaw", question=question, requirement_ids=["R1"], scoped_query=True,
    ))
    selected = _cover_named_document_sections(
        candidates, [formula_target], question=question,
        named_terms=RagResearcher().named_terms(request),
    )
    assert selected[0] == formula_target
    assert selected[1][0] == named_document_id
    assert "2.2. 记忆管理" in selected[1][1]
    evidence, _, attempted = await RagResearcher()._read_sources(
        read_targets=selected, allow_auto=False,
        run_id="real-index-openclaw", workspace_id="default", task_id="openclaw",
        requirement_ids=["R1"], context=ToolContext(workspace=None, vector_store=store),
        weak_chunks={}, documents_seen={}, focus=question,
    )
    assert selected[1] in attempted
    assert any(
        item.source_ref.startswith(f"{named_document_id}#read:")
        and "Dreaming" in item.content
        and evidence_ready_for_answer(item)
        for item in evidence
    )


@pytest.mark.asyncio
async def test_real_mixed_revision_document_reads_unchanged_openclaw_section():
    store = get_vector_store()
    document_id = "320eb7eb6f498929a36901b3b6599c26"
    blocks = [
        block for block in store.list_blocks()
        if document_id_of_block(block[0]) == document_id
    ]
    assert len({metadata.get("doc_id") for _, _, metadata in blocks}) > 1
    heading = "4. Part 2. 记忆的写入纪律与失效机制 > 4.2 路径二：公式打分"
    selected = [block for block in blocks if block[2].get("heading_path") == heading]
    assert len(selected) >= 5
    source = Path(str(selected[0][2]["source_file"]))
    assert xxhash.xxh64(source.read_bytes()).hexdigest() != selected[0][2]["doc_id"]
    current_chunks = {
        chunk.block_id: chunk.text
        for chunk in chunk_document(parse_markdown(read_text(source)), str(source))
    }
    assert all(block_id.split("::", 2)[-1] in current_chunks
               and text == current_chunks[block_id.split("::", 2)[-1]]
               for block_id, text, _ in selected)
    context = ToolContext(workspace=None, vector_store=store)
    result = await _read_document(
        ReadDocumentParams(document_id=document_id, heading=heading), context
    )
    assert result.status is ToolStatus.OK
    assert result.data["block_count"] == len(selected)
    assert "OpenClaw 的 Dreaming 系统" in result.data["text"]
    assert "0.24" in result.data["text"]
    evidence, _, attempted = await RagResearcher()._read_sources(
        read_targets=[(document_id, heading)], allow_auto=False,
        run_id="real-index-mixed-revision", workspace_id="default", task_id="openclaw",
        requirement_ids=["R1"], context=context,
        weak_chunks={}, documents_seen={}, focus="OpenClaw Dreaming",
    )
    assert attempted == [(document_id, heading)]
    assert any(item.source_ref.startswith(f"{document_id}#read:")
               and evidence_ready_for_answer(item) for item in evidence)
