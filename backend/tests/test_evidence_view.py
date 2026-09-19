"""证据视图 EvidenceView 构造器（PRD §5.3/§5.4 / 任务卡 T14）。

验收要点：
* ``candidate_id`` 由 scope、成员 ID/范围/内容哈希、视图类型与视图版本生成，跨进程稳定；
* ``evidence_text`` 逐字来自原始 block 的核验切片，绝不改写、摘要或补写；
* ``ranking_text`` 是固定模板且可重复构造，``payload_hash`` 绑定模板文本与视图版本；
* 字符范围用原文重新切片核验；结构化 Markdown（表格/代码/列表）保持完整；
* 外呼文本与序列化里没有本机绝对路径；
* ``EvidenceView`` 结构上满足 T03 的 ``RerankScorable``，可直接用于 ``RerankRequest``；
* 既有 ``RetrievedChunk`` / trace 只增加可选字段，旧调用方不传也能构造。

全部使用本地构造的真实原文片段，零出网、零真实索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.retrieval.evidence_view import (
    MEMBERSHIP_SUPPLEMENT,
    VIEW_VERSION,
    EvidenceSlice,
    EvidenceView,
    MemberRef,
    RetrievalCandidate,
    RetrievalScope,
    build_candidate_views,
    build_chunk_views,
    display_name_for,
    index_revision_for,
    payload_hash_for,
    ranking_text_for,
    scope_id_for,
    sha256_key,
    sha256_text,
)
from app.retrieval.markdown_structure import MIN_SEGMENT_CHARS, structure_segments
from app.retrieval.reranker import (
    RerankRequest,
    RerankScorable,
    RerankValidationError,
    RouteSignals,
    ViewKind,
)
from app.retrieval.trace import RetrievalCandidateTrace, RetrievalTrace
from app.retrieval.vector_retriever import RetrievedChunk

SCOPE = RetrievalScope.from_where({"platform": "local"}, index_revision="chunks__fake__v1")


@dataclass(frozen=True)
class FakeChunk:
    """最小候选形状：与 ``RetrievedChunk`` 同字段，便于测试构造。"""

    block_id: str
    text: str
    source_file: str = "docs/note.md"
    chunk_type: str = "text"
    heading_path: str | None = "检索 > 总览"
    similarity: float | None = 0.81
    match_type: str = "hybrid"
    route_signals: RouteSignals | None = None


def _section(title: str, body_times: int = 80) -> str:
    """一个标题 + 其正文段落；总高度足以生成独立引用视图。"""

    return f"## {title}\n\n" + f"{title}正文。" * body_times + "\n\n"


def _long_table(rows: int = 12) -> str:
    lines = ["| 指标 | 数值 | 说明 |", "| --- | --- | --- |"]
    lines.extend(f"| 指标{i} | {i * 3} | 备注{i} |" for i in range(rows))
    return "\n".join(lines)


def _sections_text(count: int = 4, body_times: int = 80) -> str:
    return "".join(_section(f"第{index}节", body_times) for index in range(1, count + 1))


# ------------------------------------------------------------------ 1. 基础类型与契约


def test_sha256_key_is_unambiguous_across_boundaries() -> None:
    """长度前缀编码：不同分段方式不会产生同一哈希。"""

    assert sha256_key(["ab", "c"]) != sha256_key(["a", "bc"])
    assert sha256_key(["ab", "c"]) == sha256_key(["ab", "c"])
    assert len(sha256_text("原文")) == 64


def test_scope_id_is_stable_and_distinguishes_scopes() -> None:
    assert scope_id_for(None) == "global"
    assert scope_id_for({"platform": "local"}) == scope_id_for({"platform": "local"})
    assert scope_id_for({"platform": "local"}) != scope_id_for({"platform": "private"})
    assert scope_id_for({"platform": {"$in": ["local", "private"]}}) != scope_id_for(
        {"platform": "local"}
    )


def test_index_revision_reads_vectorstore_collection_name() -> None:
    class Store:
        collection_name = "chunks__bge-m3__v1"

    assert index_revision_for(Store()) == "chunks__bge-m3__v1"
    assert index_revision_for(Store(), "override") == "override"
    assert index_revision_for(object()) == "unknown-index"


def test_scope_rejects_empty_identifiers() -> None:
    with pytest.raises(RerankValidationError):
        RetrievalScope(scope_id="", index_revision="rev")
    with pytest.raises(RerankValidationError):
        RetrievalScope(scope_id="global", index_revision=" ")


def test_display_name_never_exposes_directories() -> None:
    assert display_name_for("C:/private/客户/检索笔记.md") == "检索笔记"
    assert display_name_for("a\\b\\note.MD") == "note"
    assert display_name_for(None) == ""


def test_member_ref_requires_valid_span_and_hash() -> None:
    with pytest.raises(RerankValidationError):
        MemberRef(block_id="b", content_hash="not-a-hash")
    with pytest.raises(RerankValidationError):
        MemberRef(block_id="b", content_hash=sha256_text("x"), start=5, end=5)
    with pytest.raises(RerankValidationError):
        MemberRef(block_id="b", content_hash=sha256_text("x"), start=5, end=4)
    with pytest.raises(RerankValidationError):
        MemberRef(block_id="b", content_hash=sha256_text("x"), relation="unknown")
    member = MemberRef(block_id="b", content_hash=sha256_text("x"))
    assert (member.start, member.end) == (0, None)  # 整段成员


# ------------------------------------------------------------------ 2. 主视图


def test_primary_view_covers_whole_block_verbatim() -> None:
    text = "第一段说明。\n\n第二段说明。"
    chunk = FakeChunk(block_id="b1", text=text)
    views = build_chunk_views(chunk, SCOPE)

    assert len(views) == 1
    view = views[0]
    assert view.evidence_text == text  # 逐字来自原文
    assert view.primary_block_id == "b1"
    assert view.primary_span == (0, len(text))
    assert view.view_kind is ViewKind.BLOCK
    assert view.scope_id == SCOPE.scope_id
    assert view.index_revision == SCOPE.index_revision
    assert view.view_version == VIEW_VERSION
    assert view.display_name == "note"
    assert view.slices[0].content_hash == sha256_text(text)
    assert view.member_refs[0].relation == "primary"


def test_candidate_id_and_payload_hash_are_reproducible() -> None:
    chunk = FakeChunk(block_id="b1", text="同一段原文。")
    first = build_chunk_views(chunk, SCOPE)[0]
    second = build_chunk_views(chunk, SCOPE)[0]

    assert first.candidate_id == second.candidate_id
    assert first.payload_hash == second.payload_hash
    assert first.candidate_id.startswith("ev1-")
    assert first.candidate_id.endswith("b1")


def test_candidate_id_changes_with_scope_content_and_version() -> None:
    chunk = FakeChunk(block_id="b1", text="同一段原文。")
    base = build_chunk_views(chunk, SCOPE)[0]
    other_scope = RetrievalScope.from_where(
        {"platform": "private"}, index_revision=SCOPE.index_revision
    )
    other_text = build_chunk_views(FakeChunk(block_id="b1", text="另一段原文。"), SCOPE)[0]

    assert base.candidate_id != build_chunk_views(chunk, other_scope)[0].candidate_id
    assert base.candidate_id != other_text.candidate_id
    # 视图版本变化必然改变身份与哈希
    assert base.payload_hash != payload_hash_for(base.ranking_text, "evidence-view-v2", base.candidate_id)


def test_ranking_text_is_fixed_template_and_rebuildable() -> None:
    chunk = FakeChunk(block_id="b1", text="原文内容。")
    view = build_chunk_views(chunk, SCOPE)[0]

    assert view.ranking_text == ranking_text_for(view.display_name, view.heading_path, "原文内容。")
    assert "来源：note" in view.ranking_text
    assert "标题：检索 > 总览" in view.ranking_text
    assert view.ranking_text.endswith("原文内容。")
    assert view.payload_hash == payload_hash_for(view.ranking_text, VIEW_VERSION, view.candidate_id)


def test_view_serialisation_contains_no_absolute_path() -> None:
    chunk = FakeChunk(block_id="b1", text="原文内容。", source_file="C:/private/客户/笔记.md")
    view = build_chunk_views(chunk, SCOPE)[0]

    payload = str(view.to_dict())
    assert "C:/private" not in payload
    assert "客户" not in payload
    assert view.display_name == "笔记"
    assert view.to_dict(include_text=False).get("evidence_text") is None


def test_view_rejects_evidence_text_that_is_not_a_verified_slice() -> None:
    """改写后的正文与其切片正文不一致时，构造必须立即失败。"""

    chunk = FakeChunk(block_id="b1", text="原始正文。")
    view = build_chunk_views(chunk, SCOPE)[0]

    with pytest.raises(RerankValidationError, match="拼接"):
        EvidenceView(
            candidate_id=view.candidate_id,
            scope_id=view.scope_id,
            index_revision=view.index_revision,
            primary_block_id=view.primary_block_id,
            display_name=view.display_name,
            heading_path=view.heading_path,
            evidence_text="被改写过的正文。",
            ranking_text=view.ranking_text,
            payload_hash=view.payload_hash,
            slices=view.slices,
            slice_texts=view.slice_texts,
        )


def test_view_requires_payload_hash_to_match_ranking_text() -> None:
    chunk = FakeChunk(block_id="b1", text="原始正文。")
    view = build_chunk_views(chunk, SCOPE)[0]

    with pytest.raises(RerankValidationError, match="payload_hash"):
        EvidenceView(
            candidate_id=view.candidate_id,
            scope_id=view.scope_id,
            index_revision=view.index_revision,
            primary_block_id=view.primary_block_id,
            display_name=view.display_name,
            heading_path=view.heading_path,
            evidence_text=view.evidence_text,
            ranking_text=view.ranking_text + "改写",
            payload_hash=view.payload_hash,
            slices=view.slices,
            slice_texts=view.slice_texts,
        )


def test_view_rejects_display_name_with_path() -> None:
    chunk = FakeChunk(block_id="b1", text="原始正文。")
    view = build_chunk_views(chunk, SCOPE)[0]

    with pytest.raises(RerankValidationError, match="basename"):
        EvidenceView(
            candidate_id=view.candidate_id,
            scope_id=view.scope_id,
            index_revision=view.index_revision,
            primary_block_id=view.primary_block_id,
            display_name="客户/笔记.md",
            heading_path=view.heading_path,
            evidence_text=view.evidence_text,
            ranking_text=view.ranking_text,
            payload_hash=view.payload_hash,
            slices=view.slices,
            slice_texts=view.slice_texts,
        )


def test_view_rejects_empty_text_and_bad_route_signals() -> None:
    with pytest.raises(RerankValidationError, match="正文为空"):
        build_chunk_views(FakeChunk(block_id="b1", text=""), SCOPE)
    with pytest.raises(RerankValidationError, match="route_signals"):
        build_chunk_views(
            FakeChunk(block_id="b1", text="正文。", route_signals={"dense_score": 1.0}), SCOPE
        )


def test_image_text_view_is_atomic() -> None:
    text = "图片描述正文。" * 200
    chunk = FakeChunk(block_id="img-1", text=text, chunk_type="image_description")
    views = build_chunk_views(chunk, SCOPE)

    assert len(views) == 1  # 图片文字不生成结构化引用视图（PRD §5.5）
    assert views[0].view_kind is ViewKind.IMAGE_TEXT
    assert views[0].evidence_text == text


# ------------------------------------------------------------------ 3. 与 T03 接口对齐


def test_view_satisfies_rerank_scorable_and_request_accepts_it() -> None:
    chunk = FakeChunk(
        block_id="b1",
        text="原文内容。",
        route_signals=RouteSignals(dense_score=0.8, dense_rank=1, rrf_score=0.03, rrf_rank=1),
    )
    view = build_chunk_views(chunk, SCOPE)[0]

    assert isinstance(view, RerankScorable)
    request = RerankRequest(
        query="检索怎么做",
        scope_id=SCOPE.scope_id,
        index_revision=SCOPE.index_revision,
        documents=(view,),
        profile_id="p1",
        instruction_id="i1",
    )
    assert request.candidate_ids == (view.candidate_id,)
    assert request.top_n_value == 1
    assert "原文内容" not in str(request.to_audit_dict())


def test_view_scope_mismatch_is_rejected_by_request() -> None:
    other = RetrievalScope.from_where({"platform": "private"}, index_revision="rev-1")
    view = build_chunk_views(FakeChunk(block_id="b1", text="原文内容。"), other)[0]

    with pytest.raises(RerankValidationError, match="scope 越界"):
        RerankRequest(
            query="问题",
            scope_id="global",
            index_revision="rev-1",
            documents=(view,),
            profile_id="p1",
            instruction_id="i1",
        )


def test_candidate_record_projects_to_view() -> None:
    candidate = RetrievalCandidate(
        candidate_id="cand-1",
        block_id="b1",
        scope=SCOPE,
        route_signals=RouteSignals(bm25_score=1.2, bm25_rank=3, lexical_only=True),
    )
    views = build_candidate_views(
        [candidate], SCOPE, block_texts={"b1": "候选原文内容。"}
    )

    assert len(views) == 1
    assert views[0].evidence_text == "候选原文内容。"
    assert views[0].route_signals is not None
    assert views[0].route_signals.lexical_only is True


def test_candidate_without_text_is_rejected() -> None:
    candidate = RetrievalCandidate(candidate_id="cand-1", block_id="missing", scope=SCOPE)
    with pytest.raises(RerankValidationError, match="缺少正文"):
        build_candidate_views([candidate], SCOPE, block_texts={})


# ------------------------------------------------------------------ 4. 结构化引用视图


def test_structure_views_are_contiguous_slices_of_the_same_block() -> None:
    text = _sections_text(count=4)
    views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)

    assert len(views) > 1
    primary, *pieces = views
    assert primary.evidence_text == text
    assert "".join(view.evidence_text for view in pieces) == text  # 无缺口、无重叠
    previous_end = 0
    for view in pieces:
        start, end = view.primary_span
        assert start == previous_end
        assert end > start
        assert view.evidence_text == text[start:end]
        assert view.primary_block_id == "b1"
        assert view.slices[0].content_hash == sha256_text(view.evidence_text)
        assert len(view.evidence_text) < len(text)
        assert view.candidate_id != primary.candidate_id
        previous_end = end
    assert previous_end == len(text)


def test_structure_views_keep_headings_with_their_body() -> None:
    text = _sections_text(count=4)
    views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)
    pieces = views[1:]

    for index, view in enumerate(pieces, 1):
        heading = f"## 第{index}节"
        assert heading in view.evidence_text
        assert view.evidence_text.startswith(heading)


def test_table_region_is_preserved_inside_one_view() -> None:
    table = _long_table(rows=12)
    text = f"## 数据表\n\n{table}\n\n结尾说明。"
    views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)

    assert "".join(view.evidence_text for view in views) == text
    for view in views:
        body = view.evidence_text
        if "| 指标 | 数值 | 说明 |" not in body:
            continue
        # 表格整体保留：表头分隔行与所有数据行都在同一个视图里
        assert "| --- | --- | --- |" in body
        assert "| 指标11 |" in body
        assert body.index("| 指标1 |") < body.index("| 指标11 |")


def test_code_and_list_regions_stay_intact() -> None:
    code = "```python\ndef answer():\n    return 42\n```"
    items = "\n".join(f"- 步骤{index}：说明内容" for index in range(1, 8))
    text = f"## 代码段\n\n{'前置说明。' * 60}\n\n{code}\n\n## 清单段\n\n{items}\n\n{'尾部说明。' * 60}"
    all_views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)
    pieces = list(all_views[1:])

    assert len(pieces) >= 2
    assert "".join(view.evidence_text for view in pieces) == text
    assert any(code in view.evidence_text for view in pieces)
    assert any(items in view.evidence_text for view in pieces)


def test_question_and_answer_are_not_separated() -> None:
    question = "Q: 为什么检索结果不稳定？\nA: " + "因为候选窗口与阈值需要校准。" * 40
    text = question + "\n\n" + "后续补充说明。" * 60
    all_views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)
    pieces = list(all_views[1:])

    assert "".join(view.evidence_text for view in pieces) == text
    paired = [view for view in pieces if "Q: 为什么检索结果不稳定？" in view.evidence_text]
    assert paired, "问答对必须出现在某个引用视图里"
    for view in paired:
        assert "A: 因为候选窗口与阈值需要校准。" in view.evidence_text


def test_no_structure_views_when_boundaries_are_unreliable() -> None:
    text = "单一段落，没有可确认的结构边界。" * 40
    views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)

    assert len(views) == 1
    assert structure_segments(text) == ((0, len(text)),)


def test_structure_views_can_be_disabled() -> None:
    text = _sections_text(count=4)
    views = build_chunk_views(
        FakeChunk(block_id="b1", text=text), SCOPE, include_structure_views=False
    )

    assert len(views) == 1


def test_min_split_chars_keeps_small_blocks_whole() -> None:
    text = "## 小节\n\n" + "正文内容。" * 200
    assert len(text) > MIN_SEGMENT_CHARS
    views = build_chunk_views(
        FakeChunk(block_id="b1", text=text), SCOPE, min_split_chars=len(text) + 1
    )

    assert len(views) == 1
    assert views[0].evidence_text == text


# ------------------------------------------------------------------ 5. 成员与切片契约


def test_member_span_matches_the_original_text() -> None:
    text = _sections_text(count=4)
    views = build_chunk_views(FakeChunk(block_id="b1", text=text), SCOPE)

    assert len(views) > 1
    covered = 0
    for view in views[1:]:
        member = view.member_refs[0]
        start, end = member.start, member.end
        assert end is not None
        assert text[start:end] == view.evidence_text
        assert member.content_hash == sha256_text(text[start:end])
        assert member.block_char_offset == start
        covered += member.length
    assert covered == len(text)


def test_supplement_member_keeps_its_own_block_and_span() -> None:
    """T05 的组合视图复用同一成员契约：成员可来自不同 block，各带自己的范围与哈希。"""

    head = "公式：折扣 = 价格 × 系数。"
    body = "紧邻解释：系数由活动类型决定。" * 4
    member = MemberRef(
        block_id="b2",
        content_hash=sha256_text(body),
        relation=MEMBERSHIP_SUPPLEMENT,
        start=0,
        end=len(body),
        block_char_offset=0,
    )
    assert member.relation == MEMBERSHIP_SUPPLEMENT
    assert member.length == len(body)

    view = EvidenceView(
        candidate_id="ev1-test-b1",
        scope_id=SCOPE.scope_id,
        index_revision=SCOPE.index_revision,
        primary_block_id="b1",
        display_name="note",
        heading_path="检索 > 公式",
        evidence_text=head + body,
        ranking_text=ranking_text_for("note", "检索 > 公式", head + body),
        payload_hash=payload_hash_for(
            ranking_text_for("note", "检索 > 公式", head + body),
            VIEW_VERSION,
            "ev1-test-b1",
        ),
        slices=(
            EvidenceSlice("b1", 0, len(head), sha256_text(head)),
            EvidenceSlice("b2", 0, len(body), sha256_text(body)),
        ),
        slice_texts=(head, body),
        member_refs=(
            MemberRef(block_id="b1", content_hash=sha256_text(head)),
            member,
        ),
    )

    assert view.evidence_text.endswith(body)
    assert [item.block_id for item in view.slices] == ["b1", "b2"]
    assert isinstance(view, RerankScorable)


# ------------------------------------------------------------------ 6. 旧调用方兼容


def test_retrieved_chunk_gains_only_optional_fields() -> None:
    kwargs: dict[str, Any] = {
        "block_id": "b1",
        "text": "正文",
        "similarity": 0.5,
        "source_file": "a.md",
        "platform": "local",
        "chunk_type": "text",
        "anchor": None,
        "heading_path": None,
        "metadata": {},
    }
    legacy = RetrievedChunk(**kwargs)
    assert legacy.evidence_view is None

    view = build_chunk_views(FakeChunk(block_id="b1", text="正文"), SCOPE)[0]
    attached = RetrievedChunk(**kwargs, evidence_view=view)
    assert attached.evidence_view is view
    assert attached.text == "正文"  # 旧字段语义不变


def test_candidate_trace_gains_only_optional_evidence_hash() -> None:
    legacy = RetrievalCandidateTrace(
        block_id="b1",
        signals=(),
        fused_score=0.03,
        dense_similarity=0.8,
        gate_reason="accepted_dense_only",
        gate_threshold=0.6,
        selected=True,
        selection_reason="anchor",
    )
    assert legacy.evidence_hash is None
    assert legacy.to_dict()["evidence_hash"] is None

    digest = sha256_text("ranking text")
    updated = RetrievalCandidateTrace(
        block_id="b1",
        signals=(),
        fused_score=0.03,
        dense_similarity=0.8,
        gate_reason="accepted_dense_only",
        gate_threshold=0.6,
        selected=True,
        selection_reason="anchor",
        evidence_hash=digest,
    )
    trace = RetrievalTrace(variant_kinds=("original",), candidates=(updated,))
    assert trace.to_dict()["candidates"][0]["evidence_hash"] == digest
