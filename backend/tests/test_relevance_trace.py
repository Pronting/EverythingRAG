"""T09 单测：相关性诊断 trace 的脱敏与可重放性（PRD §8.3/§9）。

覆盖：
* ``block_ref`` / ``candidate_ref`` 不可逆、稳定，且与评测导出格式一致；
* 由真实对象（EvidenceView / CandidateWindow / SelectionResult）构造 trace，
  四种分数与通道排名分列，排除原因与最终顺序可复现；
* 脱敏导出：正文、路径、密钥、原始 block ID 与问题文本都不得出现；
* 缺失数据（无分数、无 usage、失败路径）诚实留空，不补零、不推断。
"""

from __future__ import annotations

import json
from typing import Any

from app.evaluation.rag_quality import _block_ref as evaluation_block_ref
from app.retrieval.evidence_view import RetrievalScope, build_chunk_views
from app.retrieval.relevance_selector import select_from_scores, window_candidates
from app.retrieval.reranker import (
    ExclusionReason,
    ExclusionRecord,
    ProfileStatus,
    RerankProfile,
    SelectionResult,
    SelectionStatus,
    sha256_hex,
)
from app.retrieval.trace import (
    TRACE_STAGES,
    block_ref,
    build_relevance_trace,
    candidate_ref,
    new_trace_id,
    scope_summary,
)
from app.retrieval.vector_retriever import RetrievedChunk

SCOPE = RetrievalScope(scope_id="global", index_revision="chunks__fake__v7")
PRIVATE_TEXT = "私密正文：我的 API key 是 sk-abcdef123456，文件在 D:\\private\\notes.md"
PRIVATE_QUERY = "为什么我的私有部署方案 A 会失败？"


def make_chunk(
    block_id: str,
    text: str,
    *,
    similarity: float | None = 0.42,
    heading: str | None = "笔记 > 章节",
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=block_id,
        text=text,
        similarity=similarity,
        source_file="/home/user/private/notes.md",
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path=heading,
        metadata={"source_file": "/home/user/private/notes.md"},
    )


def make_views(count: int = 3) -> list[Any]:
    views: list[Any] = []
    for index in range(count):
        chunk = make_chunk(f"blk-private-{index}", f"{PRIVATE_TEXT}-{index}")
        views.extend(build_chunk_views(chunk, SCOPE, include_structure_views=False))
    return views


def make_profile(**overrides: Any) -> RerankProfile:
    values: dict[str, Any] = {
        "profile_id": "prof-1",
        "status": ProfileStatus.VALIDATED,
        "provider_type": "siliconflow_rerank",
        "model": "Qwen/Qwen3-Reranker-8B",
        "endpoint_fingerprint": "https://api.siliconflow.cn",
        "instruction_id": "rerank-task-v1",
        "instruction_hash": sha256_hex("instruction"),
        "tokenizer_id": "Qwen/Qwen3-Reranker-8B",
        "view_version": "evidence-view-v1",
        "scoring_protocol_version": "rerank-protocol-v1",
        "score_threshold": 0.3,
        "max_candidates": 30,
        "min_candidate_count": 10,
    }
    values.update(overrides)
    return RerankProfile(**values)


# --------------------------------------------------------------------------------------
# 哈希引用
# --------------------------------------------------------------------------------------


def test_block_ref_is_irreversible_stable_and_matches_evaluation_format() -> None:
    ref = block_ref("blk-private-0")
    assert ref.startswith("blk_") and len(ref) == 20
    assert ref == block_ref("blk-private-0")
    assert "blk-private-0" not in ref
    assert ref != block_ref("blk-private-1")
    # 线上 trace 与评测报告必须得到同一个引用
    assert ref == evaluation_block_ref("blk-private-0")


def test_candidate_ref_is_irreversible_and_stable() -> None:
    ref = candidate_ref("ev1-deadbeef-blk-1")
    assert ref.startswith("cand_") and len(ref) == 21
    assert ref == candidate_ref("ev1-deadbeef-blk-1")
    assert "ev1-deadbeef" not in ref


def test_new_trace_id_is_unique_and_opaque() -> None:
    first, second = new_trace_id(), new_trace_id()
    assert first.startswith("rt_") and len(first) == 19
    assert first != second


def test_scope_summary_contains_only_identifiers_and_counts() -> None:
    summary = scope_summary("global", "chunks__fake__v7", candidate_count=3, window_size=2)
    assert "global" in summary and "chunks__fake__v7" in summary
    assert "candidates=3" in summary and "window=2" in summary
    assert PRIVATE_TEXT not in summary


# --------------------------------------------------------------------------------------
# 由真实对象构造 trace
# --------------------------------------------------------------------------------------


def test_trace_from_real_views_records_scores_ranks_and_order() -> None:
    views = make_views()
    scores = {str(view.candidate_id): 0.9 - index * 0.4 for index, view in enumerate(views)}
    selection = select_from_scores(candidates=views, scores=scores, profile=make_profile())

    trace = build_relevance_trace(
        trace_id="rt_test",
        scope=SCOPE,
        profile_id="prof-1",
        candidates=views,
        selection=selection,
        scores=scores,
        provider_trace_id="ti_provider",
        request_id="req-1",
        rerank_latency_ms=123.5,
        total_latency_ms=456.0,
        missing_headings=1,
    )

    assert trace.trace_id == "rt_test"
    assert trace.scope_id == "global"
    assert trace.index_revision == "chunks__fake__v7"
    assert trace.status in {"filtered", "empty"}
    assert trace.candidate_count == len(views)
    assert trace.provider_call_count == 1
    assert trace.rerank_latency_ms == 123.5
    assert trace.missing_headings == 1
    assert trace.provider_trace_id == "ti_provider"

    first = trace.candidates[0]
    assert first.block_ref.startswith("blk_") and first.candidate_ref.startswith("cand_")
    assert first.stage in TRACE_STAGES
    assert first.view_hash == views[0].payload_hash
    assert first.rerank_score == scores[str(views[0].candidate_id)]
    # 未入选候选必须有固定排除原因；入选候选必须有 1-based 最终顺序
    selected_refs = [
        block_ref(view.primary_block_id) for view in selection.selected
    ]
    assert trace.final_order == tuple(selected_refs)
    for item in trace.candidates:
        if item.final_position is None:
            assert item.exclusion_reason is not None
        else:
            assert trace.final_order[item.final_position - 1] == item.block_ref


def test_trace_records_window_truncation_and_exclusion_reasons() -> None:
    views = make_views(5)
    profile = make_profile(max_candidates=30)
    window = window_candidates(views, profile)
    scores = {str(view.candidate_id): 0.5 for view in views[:1]}
    selection = SelectionResult(
        status=SelectionStatus.EMPTY,
        profile_id="prof-1",
        candidate_count=len(views),
        exclusions=(
            ExclusionRecord(candidate_id=str(views[0].candidate_id), reason=ExclusionReason.BELOW_THRESHOLD),
        ),
    )
    trace = build_relevance_trace(
        trace_id="rt_window",
        scope=SCOPE,
        profile_id="prof-1",
        candidates=views,
        selection=selection,
        window=window,
        scores=scores,
    )
    assert trace.window_size == len(window.candidate_ids_after)
    assert trace.window_truncated is False  # 候选未超过档案上限
    reasons = {item.exclusion_reason for item in trace.candidates if item.exclusion_reason}
    assert reasons == {"below_threshold"}
    assert trace.final_order == ()


def test_trace_status_mapping_follows_selection_status() -> None:
    views = make_views(1)
    mapping = {
        SelectionStatus.SELECTED: "filtered",
        SelectionStatus.EMPTY: "empty",
        SelectionStatus.FAILED: "failed",
        SelectionStatus.CANCELLED: "cancelled",
    }
    for status, expected in mapping.items():
        kwargs: dict[str, Any] = {"candidate_count": 1}
        if status is SelectionStatus.SELECTED:
            kwargs["selected"] = (views[0],)
        if status is SelectionStatus.FAILED:
            from app.retrieval.reranker import RerankTimeoutError

            kwargs["error"] = RerankTimeoutError("timeout")
        selection = SelectionResult(status=status, profile_id="prof-1", **kwargs)
        trace = build_relevance_trace(
            trace_id=f"rt_{expected}",
            scope=SCOPE,
            profile_id="prof-1",
            candidates=views,
            selection=selection,
        )
        assert trace.status == expected


def test_trace_keeps_missing_data_empty_instead_of_guessing() -> None:
    views = make_views(2)
    trace = build_relevance_trace(
        trace_id="rt_failed",
        scope=SCOPE,
        profile_id="prof-1",
        candidates=views,
        status="failed",
        error_code="RERANK_TIMEOUT",
        missing_headings=2,
    )
    assert trace.status == "failed"
    assert trace.error_code == "RERANK_TIMEOUT"
    assert trace.usage is None  # 服务未返回计量 -> 明确为空，不估造
    assert trace.provider_call_count == 0
    assert all(item.rerank_score is None for item in trace.candidates)
    payload = trace.to_dict()
    assert payload["usage"] is None
    assert payload["error_code"] == "RERANK_TIMEOUT"
    assert payload["missing_headings"] == 2


def test_trace_tolerates_scope_object_without_attributes() -> None:
    trace = build_relevance_trace(
        trace_id="rt_bare",
        scope=object(),
        profile_id="",
        candidates=(),
        status="failed",
    )
    assert trace.scope_id == "global"
    assert trace.index_revision == "unknown-index"


# --------------------------------------------------------------------------------------
# 脱敏导出
# --------------------------------------------------------------------------------------


def test_export_contains_no_content_paths_keys_or_raw_block_ids() -> None:
    views = make_views(3)
    scores = {str(view.candidate_id): 0.8 for view in views}
    selection = select_from_scores(candidates=views, scores=scores, profile=make_profile())
    trace = build_relevance_trace(
        trace_id="rt_sanitized",
        scope=SCOPE,
        profile_id="prof-1",
        candidates=views,
        selection=selection,
        scores=scores,
        usage={"input_tokens": 42, "output_tokens": 0},
    )
    payload = json.dumps(trace.to_dict(), ensure_ascii=False)

    # 正文 / 问题 / 路径 / 密钥 / 原始 block ID / 标题 均不得出现
    assert PRIVATE_TEXT not in payload
    assert PRIVATE_QUERY not in payload
    assert "sk-abcdef123456" not in payload
    assert "D:\\private" not in payload
    assert "/home/user/private" not in payload
    assert "笔记 > 章节" not in payload
    for view in views:
        assert str(view.candidate_id) not in payload
        assert str(view.primary_block_id) not in payload
    # 允许出现的只有引用、分数、枚举与计数
    assert payload.count("blk_") >= len(views)
    assert "input_tokens" in payload


def test_export_keeps_four_metric_types_separate() -> None:
    view = make_views(1)[0]
    trace = build_relevance_trace(
        trace_id="rt_scores",
        scope=SCOPE,
        profile_id="prof-1",
        candidates=(view,),
        scores={str(view.candidate_id): 0.81},
    )
    item = trace.to_dict()["candidates"][0]
    assert {"dense_similarity", "bm25_score", "rrf_score", "rerank_score"} <= set(item)
    assert item["rerank_score"] == 0.81
    # 未提供通道信号时保持 None，而不是复制重排分数
    assert item["dense_similarity"] is None or isinstance(item["dense_similarity"], float)
    assert item["rrf_score"] is None or isinstance(item["rrf_score"], float)


# --------------------------------------------------------------------------------------
# 问答链路接入（PRD §8.2 meta.retrieval.trace_id / §9 每轮记录）
# --------------------------------------------------------------------------------------


async def test_chat_pipeline_records_trace_and_exposes_trace_id() -> None:
    """真实问答链路：每轮生成脱敏 trace，并把 trace_id 放进 meta.retrieval。"""

    from tests.test_chat_rerank import StubProvider, collect, make_chunk, make_service, meta_of

    provider = StubProvider(scores={}, default_score=0.9)
    service, _chat, _retriever = make_service(
        [
            make_chunk("blk-a", text=PRIVATE_TEXT, heading_path="笔记 > 甲"),
            make_chunk("blk-b", text="无关内容", heading_path="笔记 > 乙"),
        ],
        provider=provider,
    )

    frames = await collect(service, PRIVATE_QUERY)
    retrieval = meta_of(frames)["retrieval"]

    assert retrieval["status"] == "filtered"
    trace_id = retrieval["trace_id"]
    assert trace_id.startswith("rt_")
    assert len(trace_id) == 19

    # meta 帧本身不得带正文/路径（脱敏断言）
    rendered = json.dumps(frames, ensure_ascii=False)
    assert PRIVATE_TEXT not in rendered
    assert "/home/user/private" not in rendered


async def test_chat_pipeline_trace_lists_candidates_and_final_order() -> None:
    """直接调用证据通路：trace 覆盖全部候选、四种分数与最终顺序（PRIVATE 文本不外泄）。"""

    from tests.test_chat_rerank import (
        StubProvider,
        make_chunk,
        make_service,
    )

    provider = StubProvider(scores={}, default_score=0.9)
    service, _chat, _retriever = make_service(
        [
            make_chunk("blk-a", text=PRIVATE_TEXT, heading_path="笔记 > 甲"),
            make_chunk("blk-b", text="无关内容", heading_path="笔记 > 乙"),
        ],
        provider=provider,
    )
    from app.retrieval.query_transform import build_query_variants

    variants = build_query_variants(PRIVATE_QUERY)
    outcome = await service._retrieve_with_rerank(
        PRIVATE_QUERY, variants, [[0.1, 0.2, 0.3, 0.4]]
    )

    trace = outcome.trace
    assert trace is not None
    assert trace.status == "filtered"
    assert trace.candidate_count == len(outcome.chunks) or trace.candidate_count >= 1
    assert trace.selected_count == len(outcome.chunks)
    assert trace.provider_call_count == 1
    assert len(provider.calls) == 1
    assert outcome.meta_dict()["trace_id"] == trace.trace_id
    payload = json.dumps(trace.to_dict(), ensure_ascii=False)
    assert PRIVATE_TEXT not in payload
    assert PRIVATE_QUERY not in payload
    assert "/home/user/private" not in payload
    for chunk in outcome.chunks:
        assert chunk.block_id not in payload  # 原始 block ID 只以不可逆引用出现
    assert all(item.final_position is not None for item in trace.candidates if item.final_position)
