"""相关性筛选器测试：候选窗口 → 批次评分 → 阈值筛选 → 最终选择（PRD §5.1/5.2/5.5/5.6）。

覆盖：
* 候选窗口上限来自档案（30/50/100），RRF 降序 + 同分 candidate_id 升序，截断记
  ``outside_candidate_window``，并报告截取前后的候选集合；
* 批次提交 ``top_n`` 等于候选数，响应逐位校验；缺项/数量不符/分数越界立即失败；
* 阈值完全来自档案；同分按 RRF 分数 → candidate_id 排序；
* 去重、``covered_by_bundle``、来源上限与 16,000 字符预算（``budget_exceeded``，
  跳过后续而不从中间截取）；
* selected / empty / failed / cancelled 四类状态，服务失败不伪装成无资料。

Provider 是测试替身（零出网、零真实服务）；真实服务路径由 T06 的真实 HTTP Provider 覆盖。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

import pytest

from app.retrieval.evidence_view import (
    EvidenceSlice,
    EvidenceView,
    MemberRef,
    RetrievalScope,
    sha256_text,
)
from app.retrieval.relevance_selector import (
    SCORING_PROTOCOL_VERSION,
    CandidateWindow,
    score_candidates,
    select_from_scores,
    select_relevance,
    window_candidates,
)
from app.retrieval.reranker import (
    EVIDENCE_CHAR_BUDGET,
    MAX_SELECTED_SOURCES,
    ExclusionReason,
    ProfileStatus,
    RerankAuthFailedError,
    RerankInvalidResponseError,
    RerankProfile,
    RerankProvider,
    RerankProviderCapabilities,
    RerankProviderType,
    RerankRequest,
    RerankResult,
    RerankScore,
    RerankUsage,
    RerankValidationError,
    RouteSignals,
    ScoreScope,
    SelectionStatus,
    ViewKind,
)

SCOPE = RetrievalScope.from_where({"platform": "local"}, index_revision="chunks__fake__v1")


def make_capabilities(**overrides: Any) -> RerankProviderCapabilities:
    values: dict[str, Any] = {
        "provider_type": "siliconflow_rerank",
        "model": "BAAI/bge-reranker-v2-m3",
        "tokenizer_id": "stub-tokenizer",
        "score_min": 0.0,
        "score_max": 1.0,
    }
    values.update(overrides)
    return RerankProviderCapabilities(**values)


def make_profile(**overrides: Any) -> RerankProfile:
    values: dict[str, Any] = {
        "profile_id": "profile-cal-1",
        "status": ProfileStatus.VALIDATED,
        "provider_type": "siliconflow_rerank",
        "model": "BAAI/bge-reranker-v2-m3",
        "endpoint_fingerprint": "https://api.example.com",
        "instruction_id": "instr-1",
        "instruction_hash": sha256_text("instruction"),
        "tokenizer_id": "stub-tokenizer",
        "view_version": "evidence-view-v1",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "score_threshold": 0.35,
        "max_candidates": 50,
        "min_candidate_count": 20,
    }
    values.update(overrides)
    return RerankProfile(**values)


def make_view(
    candidate_id: str,
    *,
    text: str | None = None,
    rrf_score: float | None = None,
    rrf_rank: int | None = None,
    block_id: str | None = None,
    spans: tuple[tuple[int, int], ...] | None = None,
    view_kind: ViewKind = ViewKind.BLOCK,
    scope: RetrievalScope = SCOPE,
) -> EvidenceView:
    """构造一个可提交评分的真实 ``EvidenceView``（单成员，必要时多切片）。"""

    block = block_id or f"block-{candidate_id}"
    if spans is None:
        body = text if text is not None else f"原文-{candidate_id}。" * 8
        spans = ((0, len(body)),)
        slices = (
            EvidenceSlice(block, 0, len(body), sha256_text(body)),
        )
        slice_texts = (body,)
        evidence = body
    else:
        source = text if text is not None else f"原文-{candidate_id}。"
        pieces = tuple(source[start:end] for start, end in spans)
        slices = tuple(
            EvidenceSlice(block, start, end, sha256_text(piece))
            for (start, end), piece in zip(spans, pieces, strict=True)
        )
        slice_texts = pieces
        evidence = "".join(pieces)
    members = tuple(
        MemberRef(
            block_id=item.block_id,
            content_hash=item.content_hash,
            start=item.start,
            end=item.end,
            block_char_offset=item.start,
        )
        for item in slices
    )
    ranking = f"来源：note\n标题：检索 > 总览\n原文：\n{evidence}"
    from app.retrieval.evidence_view import payload_hash_for

    return EvidenceView(
        candidate_id=candidate_id,
        scope_id=scope.scope_id,
        index_revision=scope.index_revision,
        primary_block_id=block,
        display_name="note",
        heading_path="检索 > 总览",
        evidence_text=evidence,
        ranking_text=ranking,
        payload_hash=payload_hash_for(ranking, "evidence-view-v1", candidate_id),
        slices=slices,
        slice_texts=slice_texts,
        member_refs=members,
        route_signals=(
            None
            if rrf_score is None and rrf_rank is None
            else RouteSignals(rrf_score=rrf_score, rrf_rank=rrf_rank)
        ),
        view_kind=view_kind,
    )


@dataclass
class StubProvider(RerankProvider):
    """测试替身：按请求候选给出可控分数，不改动请求集合。"""

    scores: dict[str, float]
    overrides: dict[str, Any] = None  # type: ignore[assignment]
    error: Exception | None = None
    missing: tuple[str, ...] = ()
    swap_pairs: bool = False
    usage: RerankUsage | None = None
    calls: list[RerankRequest] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.overrides = dict(self.overrides or {})
        self.calls = []
        self._capabilities = make_capabilities(**self.overrides)

    @property
    def provider_type(self) -> str:
        return RerankProviderType.SILICONFLOW_RERANK

    @property
    def capabilities(self) -> RerankProviderCapabilities:
        return self._capabilities

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        scores = []
        for index, candidate_id in enumerate(request.candidate_ids):
            if candidate_id in self.missing:
                continue
            reported_id = candidate_id
            if self.swap_pairs:
                # 模拟「分数贴到了相邻位置」：index 保留但 candidate_id 错位
                reported_id = request.candidate_ids[(index + 1) % len(request.candidate_ids)]
            scores.append(
                RerankScore(
                    candidate_id=reported_id,
                    index=index,
                    score=self.scores.get(candidate_id, 0.5),
                )
            )
        return RerankResult(
            profile_id=request.profile_id,
            provider_type=self.provider_type,
            model=str(self._capabilities.model),
            document_count=len(request.candidate_ids),
            scores=tuple(scores),
            usage=self.usage,
            score_scope=ScoreScope.REQUEST_RELATIVE,
        )


async def run_selection(
    candidates: list[EvidenceView],
    *,
    scores: dict[str, float],
    profile: RerankProfile | None = None,
    provider: StubProvider | None = None,
    scope: RetrievalScope = SCOPE,
    **kwargs: Any,
) -> Any:
    active_profile = profile or make_profile()
    active_provider = provider or StubProvider(scores=scores)
    return await select_relevance(
        query="检索相关性怎么判断",
        candidates=candidates,
        profile=active_profile,
        provider=active_provider,
        scope=scope,
        **kwargs,
    )


def reasons(result: Any) -> dict[str, str]:
    return {record.candidate_id: str(record.reason) for record in result.exclusions}


# ------------------------------------------------------------------ 1. 候选窗口


def test_window_sorts_by_rrf_desc_then_candidate_id_asc() -> None:
    candidates = [
        make_view("c", rrf_score=0.01),
        make_view("b", rrf_score=0.03),
        make_view("a", rrf_score=0.03),
        make_view("d"),
    ]
    window = window_candidates(candidates, make_profile())

    assert isinstance(window, CandidateWindow)
    assert [candidate.candidate_id for candidate in window.ordered] == ["a", "b", "c", "d"]
    assert window.truncation == 0
    assert window.candidate_ids_before == window.candidate_ids_after


def test_window_limit_comes_from_profile_and_reports_both_sides() -> None:
    candidates = [make_view(f"c{index:02d}", rrf_score=1.0 - index / 100) for index in range(60)]
    window = window_candidates(candidates, make_profile(max_candidates=50, min_candidate_count=20))

    assert window.limit == 50
    assert len(window.ordered) == 50
    assert window.truncation == 10
    assert len(window.candidate_ids_before) == 60
    assert len(window.candidate_ids_after) == 50
    assert window.outside[0].candidate_id == "c50"


def test_window_can_skip_truncation_for_coverage_reporting() -> None:
    candidates = [make_view(f"c{index:02d}", rrf_score=1.0 - index / 100) for index in range(60)]
    window = window_candidates(
        candidates, make_profile(max_candidates=50, min_candidate_count=20), exclude_outside=False
    )

    assert len(window.ordered) == 60
    assert window.outside == ()
    assert window.candidate_ids_before == window.candidate_ids_after


def test_window_without_rrf_signal_still_keeps_candidates() -> None:
    candidates = [make_view("b"), make_view("a"), make_view("c")]
    window = window_candidates(candidates, make_profile())

    assert [candidate.candidate_id for candidate in window.ordered] == ["a", "b", "c"]


def test_window_shrinks_to_min_candidate_count_when_token_budget_is_exceeded() -> None:
    """PRD §6.3：候选总预算超限时按确定性顺序（RRF 降序 + candidate_id 升序）缩容并记录。"""

    candidates = [make_view(f"c{index}", rrf_score=0.1 * (index + 1)) for index in range(8)]
    profile = make_profile(max_candidates=50, min_candidate_count=3, max_request_tokens=10)
    window = window_candidates(candidates, profile, token_counter=lambda text: 4)  # 8×4 = 32 > 10

    # 缩容到 min(min_candidate_count, 候选数)，顺序仍是 RRF 降序
    assert [candidate.candidate_id for candidate in window.ordered] == ["c7", "c6", "c5"]
    assert window.shrunk_to == 3
    assert [candidate.candidate_id for candidate in window.shrunk_out] == [
        "c4",
        "c3",
        "c2",
        "c1",
        "c0",
    ]
    assert window.candidate_ids_after == ("c7", "c6", "c5")
    # 缩容到 3 条（12 token）仍 > 10：必须让调用方失败，而不是把超限请求发出去
    assert window.token_budget_exceeded is True


def test_window_shrink_is_deterministic_and_only_uses_measured_limits() -> None:
    """缩容与输入顺序无关；档案没有实测上限（``None``）时绝不缩容、绝不近似 token。"""

    base = [make_view(f"c{index}", rrf_score=0.1 * (index + 1)) for index in range(8)]
    profile = make_profile(max_candidates=50, min_candidate_count=2, max_request_tokens=10)

    first = window_candidates(base, profile, token_counter=lambda text: 4)
    shuffled = window_candidates(list(reversed(base)), profile, token_counter=lambda text: 4)
    assert [c.candidate_id for c in first.ordered] == [c.candidate_id for c in shuffled.ordered]
    assert [c.candidate_id for c in first.shrunk_out] == [c.candidate_id for c in shuffled.shrunk_out]

    unmeasured = window_candidates(
        base, make_profile(max_candidates=50, min_candidate_count=2), token_counter=lambda text: 10**6
    )
    assert len(unmeasured.ordered) == 8
    assert unmeasured.shrunk_to is None and unmeasured.shrunk_out == ()
    assert unmeasured.token_budget_exceeded is False
    # 没有 tokenizer 时也不缩容：绝不拿字符数当 token 数
    no_counter = window_candidates(base, profile)
    assert len(no_counter.ordered) == 8 and no_counter.shrunk_to is None


# ------------------------------------------------------------------ 2. 批次评分


@pytest.mark.asyncio
async def test_score_candidates_submits_one_batch_with_top_n_equal_count() -> None:
    candidates = [make_view("a"), make_view("b"), make_view("c")]
    provider = StubProvider(scores={"a": 0.9, "b": 0.4, "c": 0.1})
    outcome = await score_candidates(
        query="问题",
        candidates=candidates,
        profile=make_profile(),
        provider=provider,
        scope=SCOPE,
    )

    assert len(provider.calls) == 1  # 同一候选集合一次提交
    assert provider.calls[0].top_n_value == 3
    assert provider.calls[0].candidate_ids == ("a", "b", "c")
    assert provider.calls[0].scope_id == SCOPE.scope_id
    assert provider.calls[0].index_revision == SCOPE.index_revision
    assert outcome.score_by_id == {"a": 0.9, "b": 0.4, "c": 0.1}
    assert outcome.score_scope == str(ScoreScope.REQUEST_RELATIVE)
    assert "问题" not in str(outcome.to_dict())  # 脱敏投影不含问题正文


@pytest.mark.asyncio
async def test_score_candidates_rejects_incomplete_response() -> None:
    candidates = [make_view("a"), make_view("b")]
    provider = StubProvider(scores={"a": 0.9}, missing=("b",))
    with pytest.raises(RerankInvalidResponseError):
        await score_candidates(
            query="问题",
            candidates=candidates,
            profile=make_profile(),
            provider=provider,
            scope=SCOPE,
        )


@pytest.mark.asyncio
async def test_score_candidates_rejects_index_mismatch() -> None:
    """响应 index 必须指向提交位置上的那个候选，错位立即失败。"""

    candidates = [make_view("a"), make_view("b")]
    provider = StubProvider(scores={"a": 0.9, "b": 0.4}, swap_pairs=True)
    with pytest.raises(RerankInvalidResponseError):
        await score_candidates(
            query="问题",
            candidates=candidates,
            profile=make_profile(),
            provider=provider,
            scope=SCOPE,
        )


@pytest.mark.asyncio
async def test_score_candidates_rejects_score_out_of_declared_range() -> None:
    candidates = [make_view("a")]
    provider = StubProvider(scores={"a": 0.9}, overrides={"score_max": 0.5})
    with pytest.raises(RerankInvalidResponseError):
        await score_candidates(
            query="问题",
            candidates=candidates,
            profile=make_profile(),
            provider=provider,
            scope=SCOPE,
        )


@pytest.mark.asyncio
async def test_score_candidates_rejects_empty_input() -> None:
    provider = StubProvider(scores={})
    with pytest.raises(Exception, match="候选列表为空"):
        await score_candidates(
            query="问题",
            candidates=(),
            profile=make_profile(),
            provider=provider,
            scope=SCOPE,
        )


# ------------------------------------------------------------------ 3. 阈值与排序


@pytest.mark.asyncio
async def test_selected_keeps_threshold_qualified_views_in_score_order() -> None:
    candidates = [make_view("a"), make_view("b"), make_view("c")]
    result = await run_selection(
        candidates, scores={"a": 0.20, "b": 0.90, "c": 0.60}
    )

    assert result.status is SelectionStatus.SELECTED
    assert [view.candidate_id for view in result.selected] == ["b", "c"]
    assert result.candidate_count == 3
    assert reasons(result) == {"a": str(ExclusionReason.BELOW_THRESHOLD)}
    # 阈值来自档案：调高阈值后分数合格者也被排除
    strict = await run_selection(
        candidates, scores={"a": 0.20, "b": 0.90, "c": 0.60}, profile=make_profile(score_threshold=0.8)
    )
    assert [view.candidate_id for view in strict.selected] == ["b"]


@pytest.mark.asyncio
async def test_identical_scores_break_ties_by_rrf_then_candidate_id() -> None:
    candidates = [
        make_view("a", rrf_score=0.01),
        make_view("b", rrf_score=0.05),
        make_view("c", rrf_score=0.05),
    ]
    result = await run_selection(candidates, scores={"a": 0.9, "b": 0.9, "c": 0.9})

    assert [view.candidate_id for view in result.selected] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_candidate_window_truncation_is_recorded() -> None:
    candidates = [
        make_view(f"c{index:02d}", rrf_score=1.0 - index / 100) for index in range(5)
    ]
    profile = make_profile(max_candidates=30, min_candidate_count=3)
    provider = StubProvider(scores={f"c{index:02d}": 0.9 for index in range(5)})
    # 放大候选数到超过 30：窗口只保留前 30，其余记 outside_candidate_window
    many = candidates + [make_view(f"x{index:02d}", rrf_score=1.0 - index / 100) for index in range(40)]
    result = await run_selection(
        many, scores={candidate.candidate_id: 0.9 for candidate in many}, profile=profile, provider=provider
    )

    assert result.candidate_count == 30
    outside = [
        candidate_id for candidate_id, reason in reasons(result).items()
        if reason == str(ExclusionReason.OUTSIDE_CANDIDATE_WINDOW)
    ]
    assert len(outside) == 15
    assert len(provider.calls[0].documents) == 30


# ------------------------------------------------------------------ 4. 去重与预算


def _slice_view(
    candidate_id: str,
    source: str,
    block_id: str,
    start: int,
    end: int,
    *,
    view_kind: ViewKind = ViewKind.BLOCK,
) -> EvidenceView:
    """构造一个**自洽**的分段视图：evidence_text == 原文切片，成员范围与之逐字对应。"""

    return make_view(
        candidate_id,
        text=source,
        spans=((start, end),),
        block_id=block_id,
        view_kind=view_kind,
    )


@pytest.mark.asyncio
async def test_duplicate_payload_is_excluded_once() -> None:
    shared = "完全相同的原文内容。" * 4
    first = make_view("a", text=shared, block_id="shared-block")
    second = make_view("b", text=shared, block_id="shared-block")
    result = await run_selection([first, second], scores={"a": 0.9, "b": 0.8})

    assert [view.candidate_id for view in result.selected] == ["a"]
    assert reasons(result)["b"] == str(ExclusionReason.DUPLICATE)


@pytest.mark.asyncio
async def test_bundle_covers_member_view() -> None:
    """组合（整块范围）与其更小切片成员：只保留覆盖更完整的组合（PRD §5.4）。"""

    source = "组合覆盖测试正文。" * 20
    bundle = make_view(
        "bundle",
        text=source,
        spans=((0, len(source)),),
        block_id="shared-block",
        view_kind=ViewKind.SUPPORT_BUNDLE,
    )
    member = _slice_view("member", source, "shared-block", 0, 20)
    result = await run_selection([bundle, member], scores={"bundle": 0.9, "member": 0.7})

    assert [view.candidate_id for view in result.selected] == ["bundle"]
    assert reasons(result)["member"] == str(ExclusionReason.COVERED_BY_BUNDLE)


@pytest.mark.asyncio
async def test_failed_when_profile_argument_is_invalid() -> None:
    """非法档案以 failed 返回，不向外抛异常、也不伪装成空结果。"""

    provider = StubProvider(scores={"a": 0.9})
    result = await select_relevance(
        query="问题",
        candidates=[make_view("a")],
        profile="not-a-profile",  # type: ignore[arg-type]
        provider=provider,
        scope=SCOPE,
    )

    assert result.status is SelectionStatus.FAILED
    assert result.error is not None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_failed_when_scope_has_no_identity() -> None:
    provider = StubProvider(scores={"a": 0.9})
    result = await select_relevance(
        query="问题",
        candidates=[make_view("a")],
        profile=make_profile(),
        provider=provider,
        scope=object(),
    )

    assert result.status is SelectionStatus.FAILED
    assert provider.calls == []


@pytest.mark.asyncio
async def test_two_slices_of_one_block_both_stay_eligible() -> None:
    """同 block 的两个较小引用视图不是重复：两者覆盖不同原文，必须都能入选。"""

    body = "同一分块的两段不同正文。" * 10
    first = make_view("s1", text=body, spans=((0, 60),))
    second = make_view("s2", text=body, spans=((60, 120),))
    result = await run_selection([first, second], scores={"s1": 0.9, "s2": 0.8})

    assert [view.candidate_id for view in result.selected] == ["s1", "s2"]
    assert result.exclusions == ()


@pytest.mark.asyncio
async def test_slice_view_duplicates_whole_block_view_for_the_same_span() -> None:
    """整块视图与覆盖完全相同范围的视图是重复，只保留更完整（整块）的那个。"""

    source = "整块与切片重合的正文。" * 10
    whole = make_view("whole", text=source, block_id="shared-block")
    same_span = _slice_view("slice", source, "shared-block", 0, len(source))
    result = await run_selection([whole, same_span], scores={"whole": 0.9, "slice": 0.8})

    assert [view.candidate_id for view in result.selected] == ["whole"]
    assert reasons(result)["slice"] == str(ExclusionReason.DUPLICATE)


@pytest.mark.asyncio
async def test_budget_skips_oversized_view_and_keeps_later_one() -> None:
    big = make_view("big", text="超长正文。" * 4000)  # 20,000 字符 > 16,000 预算
    small = make_view("small", text="短正文。" * 10)
    result = await run_selection([big, small], scores={"big": 0.99, "small": 0.90})

    assert [view.candidate_id for view in result.selected] == ["small"]
    assert reasons(result)["big"] == str(ExclusionReason.BUDGET_EXCEEDED)
    assert sum(len(view.evidence_text) for view in result.selected) <= EVIDENCE_CHAR_BUDGET
    # 不得从中间截取：入选正文与视图原文逐字相同
    assert result.selected[0].evidence_text == small.evidence_text


@pytest.mark.asyncio
async def test_source_limit_never_exceeds_eight() -> None:
    candidates = [make_view(f"c{index}") for index in range(12)]
    result = await run_selection(
        candidates, scores={candidate.candidate_id: 0.9 for candidate in candidates}
    )

    assert result.status is SelectionStatus.SELECTED
    assert len(result.selected) == MAX_SELECTED_SOURCES
    over_quota = [
        candidate_id for candidate_id, reason in reasons(result).items()
        if reason == str(ExclusionReason.BUDGET_EXCEEDED)
    ]
    assert len(over_quota) == 4


@pytest.mark.asyncio
async def test_profile_budget_is_the_hard_cap() -> None:
    first = make_view("a", text="第一段正文。" * 20)
    second = make_view("b", text="第二段正文。" * 20)
    result = await run_selection(
        [first, second],
        scores={"a": 0.9, "b": 0.8},
        profile=make_profile(evidence_char_budget=len(first.evidence_text) + 5),
    )

    assert [view.candidate_id for view in result.selected] == ["a"]
    assert reasons(result)["b"] == str(ExclusionReason.BUDGET_EXCEEDED)


# ------------------------------------------------------------------ 5. 四类状态


@pytest.mark.asyncio
async def test_empty_when_everything_is_below_threshold() -> None:
    result = await run_selection(
        [make_view("a"), make_view("b")], scores={"a": 0.1, "b": 0.2}
    )

    assert result.status is SelectionStatus.EMPTY
    assert result.selected == ()
    assert result.error is None
    assert set(reasons(result).values()) == {str(ExclusionReason.BELOW_THRESHOLD)}


@pytest.mark.asyncio
async def test_empty_when_no_candidate_at_all() -> None:
    result = await run_selection([], scores={})

    assert result.status is SelectionStatus.EMPTY
    assert result.candidate_count == 0


@pytest.mark.asyncio
async def test_failed_carries_provider_error_and_does_not_look_empty() -> None:
    provider = StubProvider(scores={}, error=RerankAuthFailedError("HTTP 401"))
    result = await run_selection(
        [make_view("a")], scores={}, provider=provider
    )

    assert result.status is SelectionStatus.FAILED
    assert result.selected == ()
    assert result.error is not None
    assert result.error.code.value == "RERANK_AUTH_FAILED"
    assert result.error.to_event()["message"] == "筛选服务认证失败"


@pytest.mark.asyncio
async def test_failed_when_candidate_scope_is_out_of_range() -> None:
    other = RetrievalScope.from_where({"platform": "private"}, index_revision=SCOPE.index_revision)
    outsider = make_view("outsider", scope=other)
    result = await run_selection([outsider], scores={"outsider": 0.9})

    assert result.status is SelectionStatus.FAILED
    assert reasons(result)["outsider"] == str(ExclusionReason.SCOPE_MISMATCH)


@pytest.mark.asyncio
async def test_failed_when_candidate_version_changed() -> None:
    stale_scope = RetrievalScope(
        scope_id=SCOPE.scope_id, index_revision="chunks__other__v9"
    )
    stale = make_view("stale", scope=stale_scope)
    result = await run_selection([stale], scores={"stale": 0.9})

    assert result.status is SelectionStatus.FAILED
    assert reasons(result)["stale"] == str(ExclusionReason.VERSION_MISMATCH)


@pytest.mark.asyncio
async def test_cancelled_before_any_call() -> None:
    provider = StubProvider(scores={"a": 0.9})
    result = await run_selection(
        [make_view("a")], scores={"a": 0.9}, provider=provider, is_cancelled=lambda: True
    )

    assert result.status is SelectionStatus.CANCELLED
    assert result.selected == ()
    assert result.error is None  # 取消不是错误状态（PRD §5.6/§8.2）
    assert provider.calls == []  # 取消后不再外呼


@pytest.mark.asyncio
async def test_cancelled_after_scoring_stops_before_selection() -> None:
    calls = {"count": 0}

    def cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    provider = StubProvider(scores={"a": 0.9})
    result = await run_selection(
        [make_view("a")], scores={"a": 0.9}, provider=provider, is_cancelled=cancelled
    )

    assert len(provider.calls) == 1  # 完成评分但结果被取消
    assert result.status is SelectionStatus.CANCELLED


@pytest.mark.asyncio
async def test_asyncio_cancellation_propagates() -> None:
    class HangingProvider(StubProvider):
        async def rerank(self, request: RerankRequest) -> RerankResult:
            await asyncio.sleep(5)
            raise AssertionError("不应到达")  # pragma: no cover

    provider = HangingProvider(scores={})
    task = asyncio.create_task(
        select_relevance(
            query="问题",
            candidates=[make_view("a")],
            profile=make_profile(),
            provider=provider,
            scope=SCOPE,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ------------------------------------------------------------------ 6. 契约与预算常量


def test_scoring_protocol_version_is_readable_by_settings() -> None:
    from app.core.settings_store import _module_constant

    assert _module_constant("app.retrieval.relevance_selector", "SCORING_PROTOCOL_VERSION") == (
        SCORING_PROTOCOL_VERSION
    )


def test_selector_does_not_invent_fallback_thresholds() -> None:
    """阈值必须来自档案：源码里不出现阈值字面量兜底。"""

    import inspect

    source = inspect.getsource(__import__("app.retrieval.relevance_selector", fromlist=["x"]))
    assert "0.5" not in source
    assert "score_threshold=" not in source.replace("profile.score_threshold", "")


@pytest.mark.asyncio
async def test_select_relevance_and_select_from_scores_are_equivalent() -> None:
    """线上入口（select_relevance）与评测/来源装配入口（select_from_scores）必须同流程。

    PRD §10.3 禁止评测专用规则：两条入口对同一批候选与分数必须给出逐字段一致的
    SelectionResult（状态、入选集合与顺序、排除原因与顺序、候选数）。
    """

    shared = "完全相同的原文内容。" * 4
    body = "整块与切片重合的正文。" * 10
    candidates = [
        make_view("hi", rrf_score=0.09),
        make_view("dup-a", text=shared, block_id="shared-block", rrf_score=0.08),
        make_view("dup-b", text=shared, block_id="shared-block", rrf_score=0.07),
        make_view("bundle", text=body, spans=((0, len(body)),), block_id="b-block",
                  view_kind=ViewKind.SUPPORT_BUNDLE, rrf_score=0.06),
        make_view("member", text=body, spans=((0, 20),), block_id="b-block", rrf_score=0.05),
        make_view("mid", rrf_score=0.04),
        make_view("big", text="超长正文。" * 4000, rrf_score=0.03),
        make_view("low", rrf_score=0.02),
    ]
    scores = {
        "hi": 0.95,
        "dup-a": 0.90,
        "dup-b": 0.80,
        "bundle": 0.75,
        "member": 0.70,
        "mid": 0.50,
        "big": 0.40,
        "low": 0.01,
    }
    profile = make_profile()

    online = await run_selection(candidates, scores=scores, profile=profile)
    offline = select_from_scores(candidates=candidates, scores=scores, profile=profile)

    assert online.status is offline.status
    assert online.candidate_count == offline.candidate_count
    assert [view.candidate_id for view in online.selected] == [
        view.candidate_id for view in offline.selected
    ]
    assert [record.candidate_id for record in online.exclusions] == [
        record.candidate_id for record in offline.exclusions
    ]
    assert [str(record.reason) for record in online.exclusions] == [
        str(record.reason) for record in offline.exclusions
    ]
    # 覆盖到主要排除原因，确保等价不是「两边都空」造成的假等价
    kinds = {str(record.reason) for record in online.exclusions}
    assert str(ExclusionReason.BELOW_THRESHOLD) in kinds
    assert str(ExclusionReason.DUPLICATE) in kinds
    assert str(ExclusionReason.COVERED_BY_BUNDLE) in kinds
    assert str(ExclusionReason.BUDGET_EXCEEDED) in kinds
    assert online.selected


@pytest.mark.asyncio
async def test_two_entries_agree_on_empty_and_error_paths() -> None:
    """空结果与服务失败两条路径也不能分叉。"""

    candidates = [make_view("a"), make_view("b")]
    profile = make_profile()

    online_empty = await run_selection(candidates, scores={"a": 0.1, "b": 0.2}, profile=profile)
    offline_empty = select_from_scores(
        candidates=candidates, scores={"a": 0.1, "b": 0.2}, profile=profile
    )
    assert online_empty.status is offline_empty.status is SelectionStatus.EMPTY
    assert online_empty.selected == offline_empty.selected == ()
    assert online_empty.exclusions == offline_empty.exclusions

    failing = await run_selection(
        [make_view("a")],
        scores={},
        profile=profile,
        provider=StubProvider(scores={}, error=RerankAuthFailedError("HTTP 401")),
    )
    assert failing.status is SelectionStatus.FAILED
    assert failing.error is not None and failing.error.code.value == "RERANK_AUTH_FAILED"


@pytest.mark.asyncio
async def test_select_from_scores_records_missing_score_instead_of_dropping_it() -> None:
    """缺分候选不得入选，但**必须被记录**（不变量：全部候选 == selected ∪ exclusions）。

    固定 7 类原因里没有「缺分」这一类，缺分等价于「不能证明它达到阈值」，因此记
    ``below_threshold``；关键是它不能像以前那样静默消失。
    """

    candidates = [make_view("scored"), make_view("unscored")]
    result = select_from_scores(candidates=candidates, scores={"scored": 0.9}, profile=make_profile())

    assert [view.candidate_id for view in result.selected] == ["scored"]
    assert reasons(result) == {"unscored": str(ExclusionReason.BELOW_THRESHOLD)}
    assert {record.detail for record in result.exclusions} == {"missing score"}
    selected = {view.candidate_id for view in result.selected}
    excluded = {record.candidate_id for record in result.exclusions}
    assert selected | excluded == {"scored", "unscored"}
    assert selected & excluded == set()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.5, -0.1])
def test_select_from_scores_rejects_non_finite_or_out_of_range_scores(bad: float) -> None:
    """PRD §6.2：非有限或越界分数立即失败，绝不入选。

    ``NaN < threshold`` 恒为假：不显式拒绝就会把「非数值」当成「不低于阈值」写进上下文。
    """

    candidates = [make_view("a"), make_view("b")]
    with pytest.raises(RerankInvalidResponseError):
        select_from_scores(
            candidates=candidates,
            scores={"a": 0.9, "b": bad},
            profile=make_profile(),
        )


def test_select_from_scores_rejects_scores_of_unknown_candidates() -> None:
    """评分映射出现非候选键（混入别批分数）时直接拒绝，避免三入口出现字段级分歧。"""

    with pytest.raises(RerankValidationError):
        select_from_scores(
            candidates=[make_view("a")],
            scores={"a": 0.9, "not-a-candidate": 0.8},
            profile=make_profile(),
        )


@pytest.mark.asyncio
async def test_both_entries_reject_non_finite_scores_and_select_nothing() -> None:
    """三入口对同一类非法分数行为一致：立即失败，不选出任何证据、也不伪装成「无资料」。"""

    candidates = [make_view("a"), make_view("b")]

    with pytest.raises(RerankInvalidResponseError):
        select_from_scores(
            candidates=candidates,
            scores={"a": 0.9, "b": float("nan")},
            profile=make_profile(),
        )

    # 线上入口：Provider 的分数在构造 RerankScore 时就被有限性校验拒绝
    failing = await run_selection(
        candidates,
        scores={"a": 0.9, "b": float("nan")},
    )
    assert failing.status is SelectionStatus.FAILED
    assert failing.selected == ()
    assert failing.error is not None
    assert failing.error.code.value == "RERANK_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_score_candidates_passes_provider_usage_through_unchanged() -> None:
    """PRD §6.2/§9：服务实际返回的 usage 必须逐字段透传，缺失保持 None（不填 0）。"""

    candidates = [make_view("a"), make_view("b")]
    usage = RerankUsage(
        input_tokens=347,
        output_tokens=12,
        image_tokens=None,
        billed_units=1.5,
    )
    provider = StubProvider(scores={"a": 0.9, "b": 0.4}, usage=usage)
    outcome = await score_candidates(
        query="问题",
        candidates=candidates,
        profile=make_profile(),
        provider=provider,
        scope=SCOPE,
    )

    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 347
    assert outcome.usage.output_tokens == 12
    assert outcome.usage.image_tokens is None
    assert outcome.usage.billed_units == pytest.approx(1.5)
    assert outcome.usage.reported is True
    assert outcome.to_dict()["usage"] == {
        "input_tokens": 347,
        "output_tokens": 12,
        "image_tokens": None,
        "billed_units": 1.5,
    }

    # 服务未返回计量：保持 None，不变成 0、不变成空 dict、不估算
    silent = StubProvider(scores={"a": 0.9, "b": 0.4}, usage=None)
    outcome_silent = await score_candidates(
        query="问题",
        candidates=candidates,
        profile=make_profile(),
        provider=silent,
        scope=SCOPE,
    )
    assert outcome_silent.usage is None
    assert outcome_silent.to_dict()["usage"] is None


def test_scoring_outcome_declares_usage_field_without_default_guessing() -> None:
    """契约：usage 字段存在且默认 None（不是 0、不是空 dict）。"""

    from dataclasses import fields

    from app.retrieval.relevance_selector import ScoringOutcome

    declared = {item.name: item for item in fields(ScoringOutcome)}
    assert "usage" in declared
    assert declared["usage"].default is None


@pytest.mark.asyncio
async def test_wrapper_matches_select_relevance_field_by_field() -> None:
    """薄包装（带计量）与 select_relevance 必须逐字段一致，且 usage 来自服务。"""

    from app.retrieval.relevance_selector import select_relevance_with_scores

    candidates = [
        make_view("hi", rrf_score=0.09),
        make_view("dup-a", text="重复正文。" * 10, block_id="d1", rrf_score=0.08),
        make_view("dup-b", text="重复正文。" * 10, block_id="d1", rrf_score=0.07),
        make_view("low", rrf_score=0.02),
    ]
    scores = {"hi": 0.95, "dup-a": 0.90, "dup-b": 0.80, "low": 0.01}
    usage = RerankUsage(input_tokens=321, output_tokens=8, image_tokens=None, billed_units=0.75)
    profile = make_profile()

    direct = await select_relevance(
        query="问题",
        candidates=candidates,
        profile=profile,
        provider=StubProvider(scores=scores, usage=usage),
        scope=SCOPE,
    )
    wrapped = await select_relevance_with_scores(
        query="问题",
        candidates=candidates,
        profile=profile,
        provider=StubProvider(scores=scores, usage=usage),
        scope=SCOPE,
    )

    assert wrapped.selection.status is direct.status
    assert wrapped.selection.profile_id == direct.profile_id
    assert wrapped.selection.candidate_count == direct.candidate_count
    assert [view.candidate_id for view in wrapped.selection.selected] == [
        view.candidate_id for view in direct.selected
    ]
    assert wrapped.selection.exclusions == direct.exclusions

    assert wrapped.outcome is not None
    assert wrapped.usage is not None
    assert wrapped.usage.input_tokens == 321
    assert wrapped.usage.output_tokens == 8
    assert wrapped.usage.image_tokens is None
    assert wrapped.usage.billed_units == pytest.approx(0.75)
    assert {score.candidate_id for score in wrapped.scores} == {
        str(view.candidate_id) for view in candidates
    }


@pytest.mark.asyncio
async def test_wrapper_returns_no_outcome_when_scoring_did_not_complete() -> None:
    """空候选 / 失败 / 取消时没有评分记录，也就没有计量（None，不是 0）。"""

    from app.retrieval.relevance_selector import select_relevance_with_scores

    empty = await select_relevance_with_scores(
        query="问题",
        candidates=[],
        profile=make_profile(),
        provider=StubProvider(scores={}),
        scope=SCOPE,
    )
    assert empty.selection.status is SelectionStatus.EMPTY
    assert empty.outcome is None
    assert empty.usage is None
    assert empty.scores == ()

    failed = await select_relevance_with_scores(
        query="问题",
        candidates=[make_view("a")],
        profile=make_profile(),
        provider=StubProvider(scores={}, error=RerankAuthFailedError("HTTP 401")),
        scope=SCOPE,
    )
    assert failed.selection.status is SelectionStatus.FAILED
    assert failed.outcome is None and failed.usage is None

    cancelled = await select_relevance_with_scores(
        query="问题",
        candidates=[make_view("a")],
        profile=make_profile(),
        provider=StubProvider(scores={"a": 0.9}),
        scope=SCOPE,
        is_cancelled=lambda: True,
    )
    assert cancelled.selection.status is SelectionStatus.CANCELLED
    assert cancelled.outcome is None and cancelled.usage is None


@pytest.mark.asyncio
async def test_three_entries_share_one_pipeline() -> None:
    """三条入口（select_relevance / select_from_scores / 带计量包装）选择语义一致。"""

    from app.retrieval.relevance_selector import select_relevance_with_scores

    candidates = [make_view("a"), make_view("b")]
    scores = {"a": 0.9, "b": 0.6}
    profile = make_profile()

    direct = await select_relevance(
        query="问题",
        candidates=candidates,
        profile=profile,
        provider=StubProvider(scores=scores),
        scope=SCOPE,
    )
    wrapped = await select_relevance_with_scores(
        query="问题",
        candidates=candidates,
        profile=profile,
        provider=StubProvider(scores=scores),
        scope=SCOPE,
    )
    scored = select_from_scores(candidates=candidates, scores=scores, profile=profile)

    assert (
        [view.candidate_id for view in direct.selected]
        == [view.candidate_id for view in wrapped.selection.selected]
        == [view.candidate_id for view in scored.selected]
    )
    assert direct.exclusions == wrapped.selection.exclusions == scored.exclusions
    assert direct.candidate_count == wrapped.selection.candidate_count == scored.candidate_count


@pytest.mark.asyncio
async def test_selected_views_keep_their_own_identity_fields() -> None:
    view = make_view("a", rrf_score=0.02, rrf_rank=1)
    result = await run_selection([view], scores={"a": 0.9})

    assert len(result.selected) == 1
    chosen = result.selected[0]
    assert chosen.candidate_id == "a"
    assert chosen.payload_hash == view.payload_hash
    assert chosen.scope_id == SCOPE.scope_id
    assert chosen.index_revision == SCOPE.index_revision
    assert chosen.evidence_text == view.evidence_text


@pytest.mark.asyncio
async def test_selection_result_status_values_match_prd() -> None:
    assert {status.value for status in SelectionStatus} == {
        "selected",
        "empty",
        "failed",
        "cancelled",
    }
    assert {reason.value for reason in ExclusionReason} == {
        "below_threshold",
        "duplicate",
        "covered_by_bundle",
        "outside_candidate_window",
        "budget_exceeded",
        "scope_mismatch",
        "version_mismatch",
    }


def test_window_uses_profile_choices_only() -> None:
    for limit in (30, 50, 100):
        profile = make_profile(max_candidates=limit, min_candidate_count=1)
        window = window_candidates([make_view(f"c{index}") for index in range(120)], profile)
        assert window.limit == limit
        assert len(window.ordered) == limit


@pytest.mark.asyncio
async def test_min_candidate_count_does_not_pad_selection() -> None:
    """候选较少不构成错误：档案最低保留数只用于超限收缩，不补足结果。"""

    result = await run_selection([make_view("a")], scores={"a": 0.9})

    assert result.status is SelectionStatus.SELECTED
    assert [view.candidate_id for view in result.selected] == ["a"]


def test_replace_helper_is_not_needed_but_profile_is_frozen() -> None:
    profile = make_profile()
    updated = replace(profile, score_threshold=0.6)
    assert updated.score_threshold == 0.6
    assert profile.score_threshold == 0.35
