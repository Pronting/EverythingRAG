"""T12：按 PRD 契约补齐的新模块边界单测（纯函数级验收，PRD §10.3）。

与既有模块测试互补，只覆盖**当前未被断言**的 PRD 契约边界：

* §5.4/§5.6 组合（``support_bundle``）与其成员视图重叠时的去重；
* §5.6 来源上限与正文字符预算的**精确边界**，以及「不得从中间截取已评分文本」；
* §5.3 显示名不得暴露路径、半开区间的 code point 语义、视图版本变化导致视图哈希变化；
* §9 排除原因只能取固定七项，且逐项按字面记录；
* §6.3 长度限制的精确边界（等于上限可用，超过即拒绝）。

说明：
* 全部为纯函数/类型级检查，使用真实笔记片段与真实 Pydantic/dataclass 校验；
* 不 mock 模型分数、不伪造 HTTP 服务；Provider 解析器行为由 ``test_live_rerank_provider``
  用真实响应副本变体覆盖；
* 已知与 PRD 不符但尚未修复的行为用 ``xfail(strict=True)`` 标注原因，不把缺陷固化成断言。
"""

from __future__ import annotations

import pytest

from app.retrieval.evidence_view import (
    VIEW_VERSION,
    EvidenceSlice,
    EvidenceView,
    MemberRef,
    RetrievalScope,
    display_name_for,
    payload_hash_for,
    ranking_text_for,
    sha256_text,
)
from app.retrieval.relevance_selector import select_from_scores
from app.retrieval.rerank_provider import SiliconFlowRerankProvider, capabilities_for_model
from app.retrieval.reranker import (
    EVIDENCE_CHAR_BUDGET,
    MAX_SELECTED_SOURCES,
    ExclusionReason,
    ProfileStatus,
    RerankInputTooLongError,
    RerankProfile,
    SelectionStatus,
)
from app.retrieval.trace import build_relevance_trace

SCOPE = RetrievalScope(scope_id="global", index_revision="chunks__notes__v7")
NOTE_HEADING = "技术笔记 > 同步"


def make_profile(**overrides) -> RerankProfile:
    values = {
        "profile_id": "prof-edges",
        "status": ProfileStatus.VALIDATED,
        "provider_type": "siliconflow_rerank",
        "model": "Qwen/Qwen3-Reranker-8B",
        "endpoint_fingerprint": "https://api.siliconflow.cn",
        "instruction_id": "rerank-task-v1",
        "instruction_hash": sha256_text("instruction"),
        "tokenizer_id": "Qwen/Qwen3-Reranker-8B",
        "view_version": VIEW_VERSION,
        "scoring_protocol_version": "rerank-protocol-v1",
        "score_threshold": 0.3,
        "max_candidates": 30,
        "min_candidate_count": 5,
    }
    values.update(overrides)
    return RerankProfile(**values)


def make_view(
    candidate_id: str,
    text: str,
    *,
    block_id: str | None = None,
    start: int = 0,
    slices: tuple[EvidenceSlice, ...] | None = None,
    slice_texts: tuple[str, ...] | None = None,
    member_refs: tuple[MemberRef, ...] | None = None,
    view_version: str = VIEW_VERSION,
    display_name: str | None = None,
    view_kind=None,
) -> EvidenceView:
    """构造一份合法的证据视图（切片哈希与正文一致，payload_hash 与 ranking_text 一致）。"""

    block = block_id or f"blk-{candidate_id}"
    resolved_slices = slices or (
        EvidenceSlice(block, start, start + len(text), sha256_text(text)),
    )
    resolved_slice_texts = slice_texts or (text,)
    resolved_refs = member_refs or (
        MemberRef(
            block_id=block,
            content_hash=sha256_text(text),
            relation="primary",
            start=start,
            end=start + len(text),
        ),
    )
    name = display_name if display_name is not None else display_name_for("技术笔记.md")
    ranking = ranking_text_for(name, NOTE_HEADING, text)
    kwargs = {}
    if view_kind is not None:
        kwargs["view_kind"] = view_kind
    return EvidenceView(
        candidate_id=candidate_id,
        scope_id=SCOPE.scope_id,
        index_revision=SCOPE.index_revision,
        primary_block_id=resolved_slices[0].block_id,
        display_name=name,
        heading_path=NOTE_HEADING,
        evidence_text=text,
        ranking_text=ranking,
        payload_hash=payload_hash_for(ranking, view_version, candidate_id),
        slices=resolved_slices,
        slice_texts=resolved_slice_texts,
        member_refs=resolved_refs,
        view_version=view_version,
        **kwargs,
    )


def make_bundle_and_members():
    """一份「短公式 + 紧邻解释」的组合，以及它的两个成员独立视图（PRD §5.4）。"""

    from app.retrieval.reranker import ViewKind

    formula = "公式：loss = a + b"
    explanation = "解释：a 为误差项，b 为正则项"
    formula_slice = EvidenceSlice("blk-formula", 0, len(formula), sha256_text(formula))
    explanation_slice = EvidenceSlice(
        "blk-explanation", 0, len(explanation), sha256_text(explanation)
    )
    formula_view = make_view(
        "member-formula",
        formula,
        slices=(formula_slice,),
        member_refs=(
            MemberRef("blk-formula", sha256_text(formula), "primary", 0, len(formula)),
        ),
    )
    explanation_view = make_view(
        "member-explanation",
        explanation,
        slices=(explanation_slice,),
        member_refs=(
            MemberRef("blk-explanation", sha256_text(explanation), "primary", 0, len(explanation)),
        ),
    )
    bundle = make_view(
        "bundle-formula-explanation",
        formula + explanation,
        slices=(formula_slice, explanation_slice),
        slice_texts=(formula, explanation),
        member_refs=(
            MemberRef("blk-formula", sha256_text(formula), "primary", 0, len(formula)),
            MemberRef(
                "blk-explanation", sha256_text(explanation), "supplement", 0, len(explanation)
            ),
        ),
        view_kind=ViewKind.SUPPORT_BUNDLE,
    )
    return bundle, formula_view, explanation_view


# --------------------------------------------------------------------------------------
# PRD §5.4/§5.6：组合与其成员视图只保留一个版本
# --------------------------------------------------------------------------------------


def _artifact_multiset(views) -> list[tuple[str, int, int]]:
    """把视图的原文范围展开成列表，用于断言「每个原文范围恰好出现一次」。"""

    ranges: list[tuple[str, int, int]] = []
    for view in views:
        for item in view.slices:
            ranges.append((item.block_id, item.start, item.end))
    return sorted(ranges)


def test_bundle_kept_first_covers_its_member_views() -> None:
    """组合分数更高时：成员视图以 ``covered_by_bundle`` 排除，最终只留组合。"""

    bundle, formula_view, explanation_view = make_bundle_and_members()
    result = select_from_scores(
        candidates=[formula_view, explanation_view, bundle],
        scores={bundle.candidate_id: 0.9, formula_view.candidate_id: 0.5, explanation_view.candidate_id: 0.4},
        profile=make_profile(),
    )

    assert [view.candidate_id for view in result.selected] == [bundle.candidate_id]
    reasons = {record.candidate_id: record.reason for record in result.exclusions}
    assert reasons == {
        formula_view.candidate_id: ExclusionReason.COVERED_BY_BUNDLE,
        explanation_view.candidate_id: ExclusionReason.COVERED_BY_BUNDLE,
    }
    # 固定字面值（PRD §9）
    assert {str(record.reason) for record in result.exclusions} == {"covered_by_bundle"}
    # 每个原文范围在最终上下文中恰好出现一次
    assert _artifact_multiset(result.selected) == _artifact_multiset([bundle])


@pytest.mark.parametrize("reverse", [False, True], ids=["bundle-last", "bundle-first"])
def test_qualified_bundle_wins_its_group_regardless_of_score_order(reverse: bool) -> None:
    """组合分数合格即代表整个原文范围组：成员按 ``covered_by_bundle`` 排除（PRD §5.4）。

    captain 裁定（``.tmp_probe/rag-relevance/CAPTAIN-DECISIONS.md`` D12）：组合是「完整文本
    作为一个 document」的评分，与片段分数**不可比**（PRD §5.4 明写组合分数只表示组合整体
    相关，不能推断其中每句话都相关），因此**组合合格即胜出**，与类内分数高低、候选顺序无关。

    本用例刻意让成员分数更高（公式 0.9 / 解释 0.4 vs 组合 0.5）：组合仍代表整组。关键断言是
    **不得出现组合与成员同时入选**（同一段原文重复），也不得互相排除后全空。
    """

    bundle, formula_view, explanation_view = make_bundle_and_members()
    candidates = [formula_view, explanation_view, bundle]
    if reverse:
        candidates = list(reversed(candidates))
    result = select_from_scores(
        candidates=candidates,
        scores={
            bundle.candidate_id: 0.5,
            formula_view.candidate_id: 0.9,
            explanation_view.candidate_id: 0.4,
        },
        profile=make_profile(),
    )

    selected = [view.candidate_id for view in result.selected]
    # 不得全空；范围组只留一个版本，且是覆盖完整的那一个
    assert selected == [bundle.candidate_id]
    reasons = {record.candidate_id: str(record.reason) for record in result.exclusions}
    assert reasons == {
        formula_view.candidate_id: "covered_by_bundle",
        explanation_view.candidate_id: "covered_by_bundle",
    }
    # 每个原文范围在最终上下文中恰好出现一次（PRD §10.1 按唯一原文范围计数）
    ranges = _artifact_multiset(result.selected)
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == set(_artifact_multiset([bundle]))


def test_member_views_are_still_allowed_when_bundle_is_below_threshold() -> None:
    """组合未达阈值时不入选；成员视图各自独立评分，合格者仍可作为证据（PRD §5.4）。"""

    bundle, formula_view, explanation_view = make_bundle_and_members()
    result = select_from_scores(
        candidates=[formula_view, explanation_view, bundle],
        scores={
            bundle.candidate_id: 0.1,
            formula_view.candidate_id: 0.8,
            explanation_view.candidate_id: 0.7,
        },
        profile=make_profile(),
    )
    assert [view.candidate_id for view in result.selected] == [
        formula_view.candidate_id,
        explanation_view.candidate_id,
    ]
    reasons = {record.candidate_id: record.reason for record in result.exclusions}
    assert reasons[bundle.candidate_id] is ExclusionReason.BELOW_THRESHOLD


# --------------------------------------------------------------------------------------
# PRD §5.6：来源上限与字符预算的精确边界
# --------------------------------------------------------------------------------------


def test_source_limit_allows_exactly_eight_and_excludes_the_ninth() -> None:
    views = [make_view(f"cand-{index}", f"证据片段 {index}") for index in range(9)]
    scores = {view.candidate_id: 0.9 - index * 0.01 for index, view in enumerate(views)}
    result = select_from_scores(candidates=views, scores=scores, profile=make_profile())

    assert len(result.selected) == MAX_SELECTED_SOURCES == 8
    assert [view.candidate_id for view in result.selected] == [
        view.candidate_id for view in views[:8]
    ]
    assert len(result.exclusions) == 1
    assert result.exclusions[0].candidate_id == views[8].candidate_id
    assert result.exclusions[0].reason is ExclusionReason.BUDGET_EXCEEDED


def test_character_budget_accepts_exact_limit_and_never_truncates_text() -> None:
    half = EVIDENCE_CHAR_BUDGET // 2
    first = make_view("cand-big-1", "甲" * half)
    second = make_view("cand-big-2", "乙" * half)
    overflow = make_view("cand-overflow", "丙" * 10)
    result = select_from_scores(
        candidates=[first, second, overflow],
        scores={first.candidate_id: 0.9, second.candidate_id: 0.8, overflow.candidate_id: 0.7},
        profile=make_profile(),
    )

    assert [view.candidate_id for view in result.selected] == [
        first.candidate_id,
        second.candidate_id,
    ]
    assert sum(len(view.evidence_text) for view in result.selected) == EVIDENCE_CHAR_BUDGET
    # 入选正文与视图原文逐字相同：不得从中间截取已评分文本
    for view in result.selected:
        assert view.evidence_text == view.slice_texts[0]
        assert len(view.evidence_text) == half
    assert result.exclusions[0].candidate_id == overflow.candidate_id
    assert result.exclusions[0].reason is ExclusionReason.BUDGET_EXCEEDED


def make_oversized_bundle_and_members():
    """同一 block 上的整块组合（正文超过字符预算）与它覆盖的三个成员切片。

    组合正文 20,007 code point > 16,000 字符预算；成员一 10,000、成员二 10,007（两者
    合计同样超预算），成员三落在成员一内部。
    """

    from app.retrieval.reranker import ViewKind

    block = "blk-long"
    body = "甲乙丙丁戊己庚辛。" * 2223  # 20,007 code point
    assert len(body) > EVIDENCE_CHAR_BUDGET
    first_text = body[:10_000]
    second_text = body[10_000:]
    inner_text = first_text[:400]
    bundle = make_view(
        "bundle-whole-block",
        body,
        slices=(
            EvidenceSlice(block, 0, len(first_text), sha256_text(first_text)),
            EvidenceSlice(block, 10_000, len(body), sha256_text(second_text)),
        ),
        slice_texts=(first_text, second_text),
        member_refs=(
            MemberRef(block, sha256_text(first_text), "primary", 0, len(first_text)),
            MemberRef(block, sha256_text(second_text), "supplement", 10_000, len(body)),
        ),
        view_kind=ViewKind.SUPPORT_BUNDLE,
    )
    first_member = make_view("member-first", first_text, block_id=block, start=0)
    second_member = make_view("member-second", second_text, block_id=block, start=10_000)
    inner_member = make_view("member-inner", inner_text, block_id=block, start=0)
    return bundle, first_member, second_member, inner_member


def test_budget_skipped_bundle_does_not_exclude_its_member_views() -> None:
    """覆盖方被超预算跳过时不具备覆盖能力：成员照常作为候选，绝不被误排除（PRD §5.4/§5.6）。

    ``covered_by_bundle`` 只在覆盖方**实际入选**时生效。本用例里组合正文 20,007 字符超过
    16,000 字符预算、自己被跳过，其成员仍按排序依次尝试入选——否则该原文范围组既有分数合格
    的候选、又拿不到任何证据（全空），既丢证据也给不出正确的排除原因。
    """

    bundle, first_member, second_member, inner_member = make_oversized_bundle_and_members()
    result = select_from_scores(
        candidates=[bundle, first_member, second_member, inner_member],
        scores={
            bundle.candidate_id: 0.95,
            first_member.candidate_id: 0.9,
            second_member.candidate_id: 0.85,
            inner_member.candidate_id: 0.5,
        },
        profile=make_profile(),
    )

    assert result.status is SelectionStatus.SELECTED
    assert [view.candidate_id for view in result.selected] == [first_member.candidate_id]
    reasons = {record.candidate_id: record.reason for record in result.exclusions}
    # 覆盖方自己被超预算跳过：记 budget_exceeded，不得假称它覆盖了成员
    assert reasons[bundle.candidate_id] is ExclusionReason.BUDGET_EXCEEDED
    # 成员一不再被静默丢弃（旧实现里它既不入选也没有任何排除记录）
    assert first_member.candidate_id not in reasons
    # 成员二因剩余预算不足跳过：原因是预算，不是 covered_by_bundle
    assert reasons[second_member.candidate_id] is ExclusionReason.BUDGET_EXCEEDED
    # 成员三落在已入选的成员一范围内：仍按 covered_by_bundle 排除，不重复进入上下文
    assert reasons[inner_member.candidate_id] is ExclusionReason.COVERED_BY_BUNDLE
    # 排除原因仍在固定七项内（PRD §9）
    assert {str(reason) for reason in reasons.values()} <= {
        "below_threshold",
        "duplicate",
        "covered_by_bundle",
        "outside_candidate_window",
        "budget_exceeded",
        "scope_mismatch",
        "version_mismatch",
    }
    # 每个原文范围恰好出现一次，且正文逐字未截断、总长不超预算
    ranges = _artifact_multiset(result.selected)
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == set(_artifact_multiset([first_member]))
    assert result.selected[0].evidence_text == first_member.evidence_text
    assert sum(len(view.evidence_text) for view in result.selected) <= EVIDENCE_CHAR_BUDGET


def test_every_candidate_lands_in_selected_or_exclusions() -> None:
    """不变量：每个候选最终必须落在 ``selected`` 或 ``exclusions`` 之一，不得静默丢弃。

    根因回归（t15）：``_skipped_reason()`` 曾在「代表未入选」时返回 ``None``，被代表拦下的
    成员于是既不入选、也没有任何排除记录——范围组既有合格候选又全空，且无法诊断。本用例覆盖
    两种「代表未入选」：组合因超预算被跳过、组合低于阈值；两种情况下合格成员都必须各自有明确
    归属（入选，或按固定原因排除），且不得同时出现在两边。

    只在 ``select_from_scores`` 的阈值/去重/预算阶段断言：``failed``/``cancelled`` 是整批
    不成立，没有逐候选的结论（PRD §8.4）；分数映射里**整条缺失**的候选按「未评分不下发」
    处理（``_qualify_optional``），也不在本用例范围内。
    """

    bundle, first_member, second_member, inner_member = make_oversized_bundle_and_members()
    oversized = [bundle, first_member, second_member, inner_member]
    budget_result = select_from_scores(
        candidates=oversized,
        scores={
            bundle.candidate_id: 0.95,
            first_member.candidate_id: 0.9,
            second_member.candidate_id: 0.85,
            inner_member.candidate_id: 0.5,
        },
        profile=make_profile(),
    )

    thin_bundle, formula_view, explanation_view = make_bundle_and_members()
    below_threshold = [formula_view, explanation_view, thin_bundle]
    threshold_result = select_from_scores(
        candidates=below_threshold,
        scores={
            thin_bundle.candidate_id: 0.1,
            formula_view.candidate_id: 0.8,
            explanation_view.candidate_id: 0.7,
        },
        profile=make_profile(),
    )

    for candidates, result in ((oversized, budget_result), (below_threshold, threshold_result)):
        selected = {view.candidate_id for view in result.selected}
        excluded = {record.candidate_id for record in result.exclusions}
        assert selected | excluded == {view.candidate_id for view in candidates}
        assert selected & excluded == set()
        assert selected  # 有合格候选就不得全空
        assert result.status is SelectionStatus.SELECTED


# --------------------------------------------------------------------------------------
# PRD §5.3：显示名、半开区间与视图版本
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "D:\\private\\notes\\技术笔记.md",
        "/home/user/private/技术笔记.md",
        "C:/Users/user/private/技术笔记.md",
        "..\\..\\secret\\技术笔记.md",
    ],
)
def test_display_name_never_exposes_directories(raw: str) -> None:
    """外呼与来源展示都不得出现本机绝对路径（PRD §5.3/§9）。"""

    name = display_name_for(raw)
    # PRD §5.3：只给 basename（是否保留扩展名属实现细节，不作断言）
    assert name.startswith("技术笔记")
    for marker in ("\\", "/", "private", "secret", ".."):
        assert marker not in name


def test_member_span_is_half_open_over_unicode_code_points() -> None:
    """``[start, end)`` 按 Python Unicode code point 记录，可用原文重新切片核验（PRD §5.3）。"""

    text = "步骤①：先同步 📁 目录，再重建索引。"
    start = text.index("📁")
    end = start + len("📁 目录")
    span = EvidenceSlice("blk-unicode", start, end, sha256_text(text[start:end]))
    view = make_view(
        "cand-unicode",
        text[start:end],
        slices=(span,),
        start=start,
        member_refs=(
            MemberRef("blk-unicode", sha256_text(text[start:end]), "primary", start, end),
        ),
    )

    assert text[view.slices[0].start : view.slices[0].end] == view.evidence_text
    assert view.member_refs[0].start == start and view.member_refs[0].end == end


def test_payload_hash_changes_when_view_version_changes() -> None:
    """视图版本变化导致视图哈希变化（PRD §5.3/A07）。"""

    text = "证据正文"
    current = make_view("cand-v1", text)
    future = make_view("cand-v1", text, view_version="evidence-view-v2")

    assert current.payload_hash != future.payload_hash
    assert future.payload_hash == payload_hash_for(
        future.ranking_text, "evidence-view-v2", "cand-v1"
    )


# --------------------------------------------------------------------------------------
# PRD §9：排除原因只能取固定七项
# --------------------------------------------------------------------------------------


def test_exclusion_reasons_are_always_one_of_the_fixed_seven() -> None:
    fixed = {
        "below_threshold",
        "duplicate",
        "covered_by_bundle",
        "outside_candidate_window",
        "budget_exceeded",
        "scope_mismatch",
        "version_mismatch",
    }
    views = [make_view(f"cand-{index}", f"片段 {index}") for index in range(10)]
    # 前九个高于阈值（第八个之后由来源上限产生 budget_exceeded），最后一个低于阈值
    scores = {
        view.candidate_id: (0.9 - index * 0.01 if index < 9 else 0.05)
        for index, view in enumerate(views)
    }
    result = select_from_scores(candidates=views, scores=scores, profile=make_profile())

    produced = {str(record.reason) for record in result.exclusions}
    assert produced <= fixed
    assert {"below_threshold", "budget_exceeded"} <= produced

    trace = build_relevance_trace(
        trace_id="rt_edges",
        scope=SCOPE,
        profile_id="prof-edges",
        candidates=views,
        selection=result,
        scores=scores,
    )
    traced = {
        item.exclusion_reason for item in trace.candidates if item.exclusion_reason is not None
    }
    assert traced <= fixed


# --------------------------------------------------------------------------------------
# PRD §6.3：长度限制的精确边界
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_len", [8, 9])
async def test_token_limit_boundary_is_exact(chunk_len: int) -> None:
    """等于单条上限可提交；超过一个计数单位即拒绝，且拒绝发生在任何外呼之前（PRD §6.3）。"""

    import httpx

    from app.retrieval.reranker import RerankCandidate, RerankRequest, sha256_hex

    capabilities = capabilities_for_model("Qwen/Qwen3-Reranker-8B", max_document_tokens=8)
    document = "x" * chunk_len
    request = RerankRequest(
        query="q",
        scope_id="global",
        index_revision="rev-1",
        documents=(
            RerankCandidate(
                candidate_id="cand-1",
                scope_id="global",
                index_revision="rev-1",
                ranking_text=document,
                payload_hash=sha256_hex(document),
            ),
        ),
        profile_id="prof-edges",
        instruction_id="rerank-task-v1",
    )
    calls: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=(
                b'{"id":"req-1","results":[{"index":0,"relevance_score":0.9}],"meta":{}}'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SiliconFlowRerankProvider(
        model="Qwen/Qwen3-Reranker-8B",
        api_key="test-key",
        capabilities=capabilities,
        token_counter=len,  # 测试替身：只验证长度门控管线，不代表真实 tokenizer 计数
        client=client,
        outbound=None,
    )
    try:
        if chunk_len <= 8:
            result = await provider.rerank(request)
            assert result.document_count == 1
            assert len(calls) == 1
        else:
            with pytest.raises(RerankInputTooLongError):
                await provider.rerank(request)
            assert calls == []  # 超限时不得发起任何外呼
    finally:
        await client.aclose()
