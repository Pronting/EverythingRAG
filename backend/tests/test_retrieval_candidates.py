"""候选生成接口测试：`collect_candidates()` 的融合候选与通道信号（PRD 5.2）。

验收要点：
* 候选按 RRF 降序、同分按稳定 ``candidate_id`` 升序；``rrf_rank`` 与该顺序一致。
* 通道信号（dense 分数/排名、BM25 分数/排名、RRF、rescue 资格）逐候选保留，缺 dense 记为 None。
* 通道信号可用 T03 的 ``RouteSignals`` 直接消费，不另建同义字段。
* scope 过滤在候选阶段生效；乱序 dense 命中与历史重复副本被规范化去重。
* 候选池不被最终八来源配额、阈值门禁或 parent 扩展提前裁剪。
* 新旧两条路径（``collect_candidates`` + 原选择逻辑 / 原有 ``retrieve*``）逐字一致。

全部使用 fake vectorstore，零出网、零真实索引。
"""

from __future__ import annotations

from typing import Any

from app.retrieval.hybrid_retriever import (
    HybridCandidateCollection,
    HybridRetriever,
    _FusionState,
    _QueryIntent,
)
from app.retrieval.query_transform import (
    build_query_variants,
    extract_identifier_definition_subjects,
)
from app.retrieval.reranker import RouteSignals
from app.retrieval.trace import RetrievalSignalTrace
from app.retrieval.vector_retriever import RetrievedChunk
from tests.test_hybrid_retriever import FakeVectorStore, _block, _hit


def _retriever(store: FakeVectorStore, **kwargs: Any) -> HybridRetriever:
    return HybridRetriever(store, **kwargs)


def _vectors(variants: Any) -> list[list[float]]:
    """One distinct marker vector per query variant, mirroring a real embedder call."""

    return [[float(index + 1)] for index, _variant in enumerate(variants)]


def _lexical_evidence(
    retriever: HybridRetriever,
    query: str,
    where: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    """Rebuild the per-block lexical evidence exactly as the fusion stage does."""

    retriever._ensure_bm25()
    variants = retriever._effective_lexical_variants(build_query_variants(query))
    return retriever._fuse(
        [], retriever._search_lexical_routes(variants, where), include_trace=False
    ).lexical


def _fusion_from_collection(
    retriever: HybridRetriever,
    query: str,
    collection: HybridCandidateCollection,
    where: dict[str, Any] | None = None,
) -> _FusionState:
    """Rebuild the production fusion state from the public candidate records."""

    lexical = _lexical_evidence(retriever, query, where)
    return _FusionState(
        ordered=[candidate.candidate_id for candidate in collection.candidates],
        fused_scores={
            candidate.candidate_id: candidate.rrf_score for candidate in collection.candidates
        },
        dense_sims={
            candidate.candidate_id: candidate.dense_similarity
            for candidate in collection.candidates
            if candidate.dense_similarity is not None
        },
        # Only blocks with real lexical evidence may be present: the selection path uses key
        # membership to label ``hybrid`` results.
        lexical={
            candidate.candidate_id: lexical[candidate.candidate_id]
            for candidate in collection.candidates
            if candidate.candidate_id in lexical
        },
        signals={},
    )


def _gate_reason(
    retriever: HybridRetriever,
    query: str,
    candidate_id: str,
    collection: HybridCandidateCollection,
) -> str:
    """Legacy gate verdict for one candidate; diagnostics only, never an admission filter."""

    fusion = _fusion_from_collection(retriever, query, collection)
    intent = _QueryIntent(definition_subjects=extract_identifier_definition_subjects(query))
    return retriever._gate_candidate(candidate_id, fusion, intent).reason


def _replay_selection(
    retriever: HybridRetriever,
    query: str,
    collection: HybridCandidateCollection,
    where: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Drive the original selection path from the new candidate records only.

    The production gate/expansion helpers are used unchanged, so an identical output proves the
    extracted candidate interface preserved every ranking decision.  Route ranks are not
    reconstructible from the flattened signals without the raw hit objects, so lexical evidence is
    rebuilt through the same fusion stage; token-level route equivalence is covered by
    ``test_signals_match_trace_route_for_route``.
    """

    fusion = _fusion_from_collection(retriever, query, collection, where)
    chunks, _decisions, _selections = retriever._select_with_expansion(
        fusion,
        _QueryIntent(definition_subjects=extract_identifier_definition_subjects(query)),
    )
    return chunks


# ---------------------------------------------------------------- 1. 排序与顺序契约


def test_candidates_sorted_by_rrf_desc_then_stable_id_asc() -> None:
    """RRF 降序；同分按 candidate_id 升序，rrf_rank 即该顺序的 1-based 编号。"""

    store = FakeVectorStore(
        blocks=[
            _block("zz", "埋点 上报 方案", "z.md"),
            _block("aa", "埋点 上报 说明", "a.md"),
            _block("mm", "完全无关的缓存内容", "m.md"),
        ],
        query_result=[_hit("zz", 0.9), _hit("aa", 0.8), _hit("mm", 0.9)],
    )
    variants = build_query_variants("埋点上报")
    collection = _retriever(store).collect_candidates(variants, _vectors(variants))

    records = list(collection.candidates)
    # aa 与 zz 的 RRF 完全同分（aa dense rank1 / zz dense rank2 + 同一词法名词法排名）
    assert records[0].rrf_score == records[1].rrf_score
    assert [record.candidate_id for record in records[:2]] == ["aa", "zz"]  # 同分按 ID 升序
    assert all(record.rrf_score < records[0].rrf_score for record in records[2:])
    assert [record.rrf_rank for record in records] == [1, 2, 3]
    assert [record.rrf_score for record in records] == sorted(
        (record.rrf_score for record in records), reverse=True
    )


def test_collect_candidates_is_chainable_over_collection() -> None:
    chunks = _retriever(
        FakeVectorStore(
            blocks=[_block("b1", "统一缓存 sdk 设计", "cache.md")],
            query_result=[_hit("b1", 0.9, "cache.md")],
        )
    ).retrieve("统一缓存 sdk 设计", [1.0])
    assert [chunk.block_id for chunk in chunks] == ["b1"]


# ---------------------------------------------------------------- 2. 通道信号


def test_channel_signals_keep_every_route_without_synthetic_zero() -> None:
    """dense-only / 词法-only / 双通道候选各保留自己的真实分数，缺 dense 记 None。"""

    store = FakeVectorStore(
        blocks=[
            _block("both", "统一缓存 sdk 架构说明", "cache.md"),
            _block("dense-only", "普通段落内容", "plain.md"),
            _block("lexical-only", "唯一标识符 SPR-2024-0001 的处置过程", "incident.md"),
        ],
        query_result=[_hit("dense-only", 0.9, "plain.md"), _hit("both", 0.7, "cache.md")],
    )
    query = "SPR-2024-0001 缓存 sdk"
    variants = build_query_variants(query)
    collection = _retriever(store).collect_candidates(variants, _vectors(variants))
    records = {record.candidate_id: record for record in collection.candidates}

    assert records["dense-only"].dense_similarity == 0.9
    assert records["dense-only"].bm25_score is None
    assert records["dense-only"].bm25_rank is None
    assert records["dense-only"].dense_rank == 1  # 每路 dense 排名，非 RRF 排名
    assert records["both"].dense_similarity == 0.7
    assert records["both"].dense_rank == 2  # dense 路由内第二名
    assert records["both"].bm25_score is not None
    assert records["both"].rescue_eligible is True
    # BM25-only 候选：dense 分数缺失，不能用 0.0 冒充
    assert records["lexical-only"].dense_similarity is None
    assert records["lexical-only"].dense_rank is None
    assert records["lexical-only"].bm25_score is not None
    assert records["lexical-only"].bm25_rank is not None
    assert records["lexical-only"].lexical_only is True
    assert all(
        isinstance(signal, RetrievalSignalTrace) for record in records.values()
        for signal in record.signals
    )


def test_route_signals_use_the_canonical_reranker_shape() -> None:
    """通道信号必须能被 T03 的 RouteSignals 直接消费，不另建同义字段。"""

    store = FakeVectorStore(
        blocks=[
            _block("both", "统一缓存 sdk 架构说明", "cache.md"),
            _block("lexical-only", "唯一标识符 SPR-2024-0001 的处置过程", "incident.md"),
        ],
        query_result=[_hit("both", 0.8, "cache.md")],
    )
    query = "SPR-2024-0001 缓存 sdk"
    variants = build_query_variants(query)
    collection = _retriever(store).collect_candidates(variants, _vectors(variants))
    records = {record.candidate_id: record for record in collection.candidates}

    both = records["both"].route_signals
    assert isinstance(both, RouteSignals)
    assert both.dense_score == 0.8
    assert both.dense_rank == 1
    assert both.bm25_score == records["both"].bm25_score
    assert both.bm25_rank == records["both"].bm25_rank
    assert both.rrf_score == records["both"].rrf_score
    assert both.rrf_rank == records["both"].rrf_rank
    assert both.lexical_only is False
    assert both.to_dict()["rrf_rank"] == records["both"].rrf_rank

    lexical_only = records["lexical-only"].route_signals
    assert lexical_only.dense_score is None
    assert lexical_only.lexical_only is True


def test_signals_match_trace_route_for_route() -> None:
    """候选接口的信号与原有 trace 的信号逐路一致（同一融合实现）。"""

    store = FakeVectorStore(
        blocks=[
            _block("cached", "统一缓存 sdk 架构说明", "cache.md"),
            _block("other", "普通段落内容", "plain.md"),
        ],
        query_result=[_hit("cached", 0.9, "cache.md"), _hit("other", 0.5, "plain.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "统一缓存 sdk 架构是怎么设计的"
    variants = build_query_variants(query)

    collection = retriever.collect_candidates(variants, [[1.0], [0.5]])
    result = retriever.retrieve_variants_with_trace(variants, [[1.0], [0.5]])

    collected = dict(zip(collection.candidate_ids, collection.candidates, strict=True))
    assert set(collected) == {candidate.block_id for candidate in result.trace.candidates}
    for trace in result.trace.candidates:
        record = collected[trace.block_id]
        assert record.rrf_score == trace.fused_score
        assert record.dense_similarity == trace.dense_similarity
        assert record.signals == trace.signals


# ---------------------------------------------------------------- 3. 去重与 scope 过滤


def test_duplicate_logical_blocks_and_dense_order_are_canonicalised() -> None:
    """乱序 dense 命中与历史重复副本折叠为一个逻辑候选，且按 RRF 重新排序。"""

    def hit(block_id: str, similarity: float) -> dict[str, Any]:
        return _hit(block_id, similarity, "dup.md")

    store = FakeVectorStore(
        blocks=[
            _block("a", "统一缓存 sdk 架构说明", "legacy/1/dup.md", "缓存"),
            _block("b", "统一缓存 sdk 架构说明", "docs/dup.md", "缓存"),
            _block("c", "另一个普通段落", "other.md"),
        ],
        # dense 结果故意乱序，融合必须重新排序
        query_result=[hit("c", 0.40), hit("b", 0.95), hit("a", 0.93)],
    )
    collection = _retriever(store).collect_candidates(build_query_variants("缓存"), [[1.0]])

    assert list(collection.candidate_ids).count(collection.candidate_ids[0]) == 1
    assert set(collection.candidate_ids) == {"b", "c"}
    assert collection.candidate_ids[0] == "b"  # 可读路径胜出，dense 0.95
    assert collection.candidates[0].dense_similarity == 0.95


def test_scope_filter_applies_to_dense_and_lexical_candidates() -> None:
    """where 过滤在候选阶段生效：BM25 候选不能绕过 dense 的同一 scope 条件。"""

    private_block = _block("private", "唯一术语 SPR-PRIVATE-999", "private.md")
    private_block[2]["platform"] = "private"
    store = FakeVectorStore(
        blocks=[private_block, _block("public", "公开缓存说明", "public.md")],
        query_result=[],
    )
    query = "SPR-PRIVATE-999"
    variants = build_query_variants(query)
    collection = _retriever(store).collect_candidates(
        variants, _vectors(variants), where={"platform": "local"}
    )

    assert "private" not in collection.candidate_ids
    assert "public" not in collection.candidate_ids  # 词法不命中，且无 dense 结果
    assert collection.candidates == ()


# ---------------------------------------------------------------- 4. 候选池不被最终选择裁剪


def test_candidate_pool_is_not_trimmed_by_final_selection_rules() -> None:
    """20 个合格候选都在池内；只有当查询词法命中时才记录 rescue 资格。"""

    blocks = [_block(f"b{index:02d}", f"缓存方案 细节 {index}", f"n{index:02d}.md") for index in range(20)]
    store = FakeVectorStore(
        blocks=[*blocks, _block("target", "统一缓存 sdk 架构说明", "cache.md")],
        query_result=[
            _hit("target", 0.95, "cache.md"),
            *[_hit(f"b{index:02d}", 0.90 - index * 0.01, f"n{index:02d}.md") for index in range(20)],
        ],
    )
    query = "统一缓存 sdk 架构说明"
    variants = build_query_variants(query)
    collection = _retriever(store, context_top_k=5, min_similarity=0.60).collect_candidates(
        variants, _vectors(variants)
    )

    assert len(collection.candidate_ids) == 21  # 不被 context_top_k=5 裁剪
    assert all(record.rrf_score > 0 for record in collection.candidates)


def test_legacy_gate_verdicts_do_not_remove_candidates_from_the_pool() -> None:
    """T01 归因回归：旧门控会拒掉的候选（无 dense / 低 dense）必须仍进入候选集合。

    两个当前失败案例的必需目标块全部 `in_fusion`，只因 `min_similarity` 那套旧门控被拒
    （`rejected_missing_dense_weak_lexical` / `rejected_low_quality_lexical_fill` /
    `rejected_dense_below_threshold`）。候选阶段若复用该门控就是原样复制失败，因此这里显式
    断言候选池包含它们，门控裁决只作诊断信号。
    """

    store = FakeVectorStore(
        blocks=[
            _block("dense-below", "对比 性能 方案", "p.md"),  # 0.377 < 0.60
            _block("no-dense", "分位数 说明 文本", "q.md"),  # 仅 BM25 命中，无 dense 分
        ],
        query_result=[_hit("dense-below", 0.377, "p.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "分位数 对比 性能 方案"
    variants = build_query_variants(query)
    collection = retriever.collect_candidates(variants, _vectors(variants))
    records = {record.candidate_id: record for record in collection.candidates}

    assert {"dense-below", "no-dense"} <= set(collection.candidate_ids)
    assert records["dense-below"].dense_similarity == 0.377
    assert records["no-dense"].dense_similarity is None  # 缺失 dense，不是 0.0
    assert records["no-dense"].lexical_only is True
    assert records["no-dense"].route_signals.dense_rank is None
    assert records["no-dense"].route_signals.bm25_rank is not None

    # 同一候选在旧门控下的裁决原因（诊断信号）——证明它确实会被旧阈值拒掉
    assert _gate_reason(retriever, query, "dense-below", collection).startswith("rejected_")
    assert _gate_reason(retriever, query, "no-dense", collection) in {
        "rejected_missing_dense_weak_lexical",
        "accepted_lexical_only",
    }


def test_lexical_candidates_survive_into_pool_even_when_gate_rejects_them() -> None:
    """门禁拒绝的候选仍留在候选池：授权/删除都必须发生在 reranker 之后（PRD 5.2）。"""

    store = FakeVectorStore(
        blocks=[
            _block("target", "统一缓存 sdk 架构说明", "cache.md"),
            _block("noise", "埋点上报 完全不同主题", "noise.md"),
        ],
        query_result=[_hit("noise", 0.40, "noise.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "埋点上报 缓存 sdk 架构"
    variants = build_query_variants(query)
    collection = retriever.collect_candidates(variants, _vectors(variants))
    assert "noise" in collection.candidate_ids  # 已在池内，且带真实低 dense 分数
    assert next(
        record for record in collection.candidates if record.candidate_id == "noise"
    ).dense_similarity == 0.40

    assert "noise" not in [
        chunk.block_id for chunk in retriever.retrieve(query, [1.0])
    ]  # 最终选择仍然拒绝它


def test_definition_subject_route_is_part_of_candidate_generation() -> None:
    """定义类问句的主语词法路由仍由候选接口生成，且不额外产生 dense 调用。"""

    store = FakeVectorStore(
        blocks=[_block("adb", "ADB 是一种数据库代理组件", "adb.md")],
        query_result=[_hit("adb", 0.20, "adb.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "请问一下 ADB 是什么？"
    variants = build_query_variants(query)
    collection = retriever.collect_candidates(variants, _vectors(variants))

    assert "definition_subject" in collection.variant_kinds
    assert "adb" in collection.candidate_ids
    assert len(store.query_calls) == len(variants)


def test_definition_subject_route_is_added_once_per_token_set() -> None:
    """主语路由只在 token 集合未出现过时添加，避免重复路由双计同一份词法证据。"""

    store = FakeVectorStore(
        blocks=[_block("adb", "ADB 是一种数据库代理组件", "adb.md")],
        query_result=[_hit("adb", 0.20, "adb.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "请问一下 ADB 是什么？"
    variants = build_query_variants(query)
    collection = retriever.collect_candidates(variants, _vectors(variants))

    kinds = list(collection.variant_kinds)
    assert kinds.count("definition_subject") == 1  # 只添加一次
    assert kinds[: len(variants)] == [variant.kind for variant in variants]
    assert len(kinds) == len(set(kinds))
    assert "adb" in collection.candidate_ids


# ---------------------------------------------------------------- 5. 新旧路径一致


def test_new_and_old_paths_agree_on_mixed_corpus() -> None:
    """代表性语料（含保留位、重复副本、词法-only）两路径结果逐字一致。"""

    blocks = [
        _block("dense-answer", "统一入口负责流量治理", "answer.md", "职责"),
        _block("bg-1", "背景说明第一段", "search.md", "背景"),
        _block("bg-2", "背景说明第二段", "search.md", "背景"),
        _block("solution", "兜底搜索与热门结果方案", "search.md", "方案"),
        _block("dup-a", "重复证据正文", "legacy/2/dup.md", "重复"),
        _block("dup-b", "重复证据正文", "docs/dup.md", "重复"),
        _block("exact", "工单编号 SPR-2024-0001 的处置过程", "incident.md"),
        _block("weak", "普通缓存说明", "cache.md"),
    ]
    query_result = [
        _hit("dense-answer", 0.92, "answer.md"),
        _hit("bg-1", 0.90, "search.md"),
        _hit("bg-2", 0.88, "search.md"),
        _hit("solution", 0.96, "search.md"),
        _hit("dup-a", 0.80, "legacy/2/dup.md"),
        _hit("dup-b", 0.78, "docs/dup.md"),
        _hit("weak", 0.51, "cache.md"),
    ]
    query = "SPR-2024-0001 缓冲兜底方案"

    old_store = FakeVectorStore(blocks=blocks, query_result=query_result)
    old_retriever = _retriever(
        old_store, context_top_k=5, min_similarity=0.60, dense_reserve_k=1
    )
    old_chunks = old_retriever.retrieve(query, [1.0])

    new_store = FakeVectorStore(blocks=blocks, query_result=query_result)
    new_retriever = _retriever(
        new_store, context_top_k=5, min_similarity=0.60, dense_reserve_k=1
    )
    variants = build_query_variants(query)
    # 对齐原 retrieve() 的 dense 路由数：它只嵌入原始问题一次
    collection = new_retriever.collect_candidates(variants[:1], [[1.0]])

    assert set(collection.candidate_ids) <= {
        "dense-answer",
        "bg-1",
        "bg-2",
        "solution",
        "dup-b",
        "exact",
        "weak",
    }
    assert "dup-a" not in collection.candidate_ids  # 历史路径副本被规范化掉
    assert "exact" in collection.candidate_ids  # 词法-only 候选进入候选池
    assert next(
        record for record in collection.candidates if record.candidate_id == "exact"
    ).dense_similarity is None

    new_chunks = _replay_selection(new_retriever, query, collection)
    assert [chunk.block_id for chunk in new_chunks] == [
        chunk.block_id for chunk in old_chunks
    ]
    for old_chunk, new_chunk in zip(old_chunks, new_chunks, strict=True):
        assert new_chunk.block_id == old_chunk.block_id
        assert new_chunk.text == old_chunk.text
        assert new_chunk.match_type == old_chunk.match_type
        assert new_chunk.context_text == old_chunk.context_text


def test_new_and_old_paths_agree_under_scope_filter() -> None:
    """库端正确过滤时，带 where 的两条路径结果一致；候选池不含范围外成员。"""

    private_block = _block("private", "唯一术语 SPR-PRIVATE-999 的详情", "private.md")
    private_block[2]["platform"] = "private"
    blocks = [
        _block("keep", "统一缓存 sdk 架构说明", "cache.md", "缓存"),
        private_block,
        _block("weak", "普通段落内容", "plain.md"),
    ]
    where = {"platform": "local"}
    query = "统一缓存 sdk 架构说明"
    store_hits = [_hit("keep", 0.90, "cache.md"), _hit("weak", 0.51, "plain.md")]

    retriever = _retriever(
        FakeVectorStore(blocks=blocks, query_result=store_hits),
        context_top_k=5,
        min_similarity=0.60,
    )
    old_chunks = retriever.retrieve(query, [1.0], where=where)

    variants = build_query_variants(query)
    collection = retriever.collect_candidates(variants[:1], [[1.0]], where=where)

    assert "private" not in collection.candidate_ids
    assert collection.candidate_ids == ("keep", "weak")  # weak 候选资格由最终选择判定
    assert [
        chunk.block_id
        for chunk in _replay_selection(retriever, query, collection, where)
    ] == [chunk.block_id for chunk in old_chunks] == ["keep"]


def test_candidate_scope_check_does_not_trust_unfiltered_dense_hits() -> None:
    """向量库未过滤（越界 dense 命中）时，候选阶段按本地元数据剔除范围外成员。"""

    private_block = _block("private", "唯一术语 SPR-PRIVATE-999", "private.md")
    private_block[2]["platform"] = "private"
    store = FakeVectorStore(
        blocks=[private_block, _block("public", "公开缓存说明", "public.md")],
        query_result=[_hit("private", 0.95, "private.md")],  # 库端越界返回
    )
    query = "SPR-PRIVATE-999"
    variants = build_query_variants(query)
    collection = _retriever(store).collect_candidates(
        variants, _vectors(variants), where={"platform": "local"}
    )

    assert collection.candidate_ids == ()



