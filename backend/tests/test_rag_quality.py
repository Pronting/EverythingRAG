"""真实评估器的指标与严格块命中规则测试。"""

from __future__ import annotations

import json

import pytest

from app.evaluation.rag_quality import (
    ExpectedTarget,
    NegativeCase,
    PositiveCase,
    chunk_matches_expectation,
    evaluate_retrieval,
    quality_gate_passes,
)
from app.retrieval.query_transform import QueryVariant
from app.retrieval.vector_retriever import RetrievedChunk


class _FakeEmbedder:
    fingerprint = "fake"
    dim = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


class _FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results

    def retrieve_variants(
        self, variants: tuple[QueryVariant, ...], vectors: tuple[list[float], ...]
    ) -> list[RetrievedChunk]:
        del vectors
        return self.results[variants[0].text]


def _chunk(
    source: str,
    heading: str,
    similarity: float | None = 0.8,
    *,
    text: str = "答案证据",
    block_id: str | None = None,
    context_text: str | None = None,
    metadata: dict[str, str] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=block_id or f"{source}:{heading}",
        text=text,
        similarity=similarity,
        source_file=source,
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path=heading,
        metadata=metadata or {},
        context_text=context_text,
    )


def test_chunk_match_requires_exact_source_basename_and_full_heading() -> None:
    case = PositiveCase(
        "p",
        "cart",
        "q",
        "购物车分表.md",
        "数据库 > 交互",
        ("答案证据",),
    )

    assert chunk_matches_expectation(_chunk(r"C:\notes\购物车分表.md", "数据库 > 交互"), case)
    assert chunk_matches_expectation(_chunk("C:/notes/购物车分表．ＭＤ", "数据库　＞　交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\旧购物车分表.md", "数据库 > 交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\购物车分表.md", "交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\购物车分表.md", "数据库"), case)
    assert not chunk_matches_expectation(
        _chunk(r"C:\notes\购物车分表.md", "数据库 > 交互 > 子标题"), case
    )
    assert not chunk_matches_expectation(
        _chunk(r"C:\notes\购物车分表.md", "数据库 > 交互", text="相同标题的错误块"),
        case,
    )


def test_chunk_match_rejects_sibling_heading_even_when_parent_contains_both_names() -> None:
    parent = "2. 安全性：工具安全性、数据安全性、数据隔离"
    case = PositiveCase(
        "p",
        "agent_security",
        "q",
        "客服 Agent 构建.md",
        f"{parent} > 工具安全性",
        ("答案证据",),
    )

    assert chunk_matches_expectation(_chunk("客服 Agent 构建.md", f"{parent} > 工具安全性"), case)
    assert not chunk_matches_expectation(_chunk("客服 Agent 构建.md", parent), case)
    assert not chunk_matches_expectation(
        _chunk("客服 Agent 构建.md", f"{parent} > 数据安全性"), case
    )


def test_chunk_match_accepts_explicit_heading_alias_from_short_section_merge() -> None:
    case = PositiveCase(
        "p",
        "ops",
        "q",
        "事故复盘.md",
        "故障 > 根因",
        ("连接池耗尽",),
    )
    chunk = _chunk(
        "事故复盘.md",
        "故障 > 现象 / 根因",
        text="【现象】\n接口超时\n\n【根因】\n连接池耗尽",
        metadata={
            "heading_aliases": json.dumps(
                ["故障 > 现象", "故障 > 根因"], ensure_ascii=False
            )
        },
    )

    assert chunk_matches_expectation(chunk, case)


def test_chunk_match_uses_same_heading_effective_context() -> None:
    case = PositiveCase("p", "tracking", "q", "埋点.md", "SPM > 规则", ("公式证据",))
    chunk = _chunk(
        "埋点.md",
        "SPM > 规则",
        text="字段解释表",
        context_text="标题：SPM > 规则\n内容：字段解释表\n\n同标题补充信息：公式证据",
    )

    assert chunk_matches_expectation(chunk, case)


def test_acceptable_targets_make_supported_versions_explicit() -> None:
    heading = "二、SPM及SCM设计 > 1、SPM > 1.1 规则"
    case = PositiveCase(
        id="p",
        category="tracking",
        query="q",
        acceptable_targets=(
            ExpectedTarget("埋点2.0设计文档V2.0.1.md", heading, ("答案证据",)),
            ExpectedTarget("埋点2.0设计文档V2.0.2.md", heading, ("答案证据",)),
        ),
    )

    assert chunk_matches_expectation(_chunk("埋点2.0设计文档V2.0.1.md", heading), case)
    assert chunk_matches_expectation(_chunk("埋点2.0设计文档V2.0.2.md", heading), case)
    assert not chunk_matches_expectation(_chunk("埋点2.0设计文档V2.0.3.md", heading), case)
    assert not chunk_matches_expectation(
        _chunk("埋点2.0设计文档V2.0.2.md", "二、SPM及SCM设计 > 2、SCM > 2.1 规则"),
        case,
    )


def test_evaluate_retrieval_reports_strict_metrics_and_batches_queries() -> None:
    positives = (
        PositiveCase(
            "p1",
            "cart",
            "我想问一下购物车交互是什么样的？",
            "购物车分表.md",
            "交互",
            ("答案证据",),
        ),
        PositiveCase(
            "p2",
            "ops",
            "CPU 过载",
            "CPU.md",
            "CPU过载排查",
            ("答案证据",),
        ),
    )
    negatives = (NegativeCase("n1", "火星生态系统"),)
    embedder = _FakeEmbedder()
    retriever = _FakeRetriever(
        {
            positives[0].query: [
                _chunk("购物车分表.md", "方案对比", similarity=None),
                _chunk("购物车分表.md", "交互", block_id="private-block-id"),
            ],
            positives[1].query: [_chunk("内存.md", "CPU过载排查")],
            negatives[0].query: [_chunk("无关.md", "其他")],
        }
    )

    report = evaluate_retrieval(
        embedder=embedder,
        retriever=retriever,  # type: ignore[arg-type]
        positives=positives,
        negatives=negatives,
        embed_batch_size=2,
    )

    assert report.positive_total == 2
    assert report.hit_at_1 == 0.0
    assert report.hit_at_5 == 0.5
    assert report.hit_at_8 == 0.5
    assert report.mrr_at_8 == 0.25
    assert report.negative_false_context_rate == 1.0
    assert report.positive_outcomes[0].matched_rank == 2
    assert report.positive_outcomes[0].candidates[0].similarity is None
    assert len(embedder.calls) >= 2
    assert all(len(call) <= 2 for call in embedder.calls)
    serialized = str(report.to_dict())
    assert "答案证据" not in serialized
    assert "private-block-id" not in serialized
    assert report.positive_outcomes[0].candidates[0].block_ref.startswith("blk_")
    assert len(report.positive_outcomes[0].candidates[0].block_ref) == 20

    assert not quality_gate_passes(report, min_hit_at_8=0.95, max_negative_false_context_rate=0.0)
    assert not quality_gate_passes(report, min_hit_at_8=0.50, max_negative_false_context_rate=1.0)

    gate_positives = tuple(
        PositiveCase(
            f"gate-p-{index}",
            "cart",
            f"正例 {index}",
            "购物车分表.md",
            "交互",
            ("答案证据",),
        )
        for index in range(50)
    )
    gate_negatives = tuple(NegativeCase(f"gate-n-{index}", f"负例 {index}") for index in range(10))
    gate_results = {
        case.query: [
            _chunk(
                "购物车分表.md",
                "交互",
                block_id=f"private-block-{case.id}",
            )
        ]
        for case in gate_positives
    }
    gate_results.update({case.query: [] for case in gate_negatives})
    passing = evaluate_retrieval(
        embedder=_FakeEmbedder(),
        retriever=_FakeRetriever(gate_results),  # type: ignore[arg-type]
        positives=gate_positives,
        negatives=gate_negatives,
    )
    assert quality_gate_passes(passing)


def test_required_targets_only_pass_after_every_target_is_retrieved() -> None:
    targets = (
        ExpectedTarget("埋点.md", "SPM和SCM拼接规则", ("拼接证据",)),
        ExpectedTarget("埋点.md", "总结 > 新旧方案优缺点", ("对比证据",)),
    )
    incomplete = PositiveCase(
        id="incomplete",
        category="tracking",
        query="只有一个目标",
        required_targets=targets,
    )
    complete = PositiveCase(
        id="complete",
        category="tracking",
        query="两个目标都出现",
        required_targets=targets,
    )
    retriever = _FakeRetriever(
        {
            incomplete.query: [_chunk("埋点.md", "SPM和SCM拼接规则", text="拼接证据")],
            complete.query: [
                _chunk("埋点.md", "总结 > 新旧方案优缺点", text="对比证据"),
                _chunk("无关.md", "其他"),
                _chunk("埋点.md", "SPM和SCM拼接规则", text="拼接证据"),
            ],
        }
    )

    report = evaluate_retrieval(
        embedder=_FakeEmbedder(),
        retriever=retriever,  # type: ignore[arg-type]
        positives=(incomplete, complete),
        negatives=(),
    )

    assert report.positive_outcomes[0].matched_rank is None
    assert report.positive_outcomes[1].matched_rank == 3
    assert report.hit_at_1 == 0.0
    assert report.hit_at_5 == 0.5
    assert chunk_matches_expectation(
        _chunk("埋点.md", targets[0].heading, text="拼接证据"), complete
    )
    assert chunk_matches_expectation(
        _chunk("埋点.md", targets[1].heading, text="对比证据"), complete
    )


def test_positive_case_rejects_mixed_or_duplicate_target_modes() -> None:
    target = ExpectedTarget("a.md", "A", ("证据",))
    with pytest.raises(ValueError, match="只能使用一种目标模式"):
        PositiveCase(
            "mixed",
            "test",
            "q",
            "a.md",
            "A",
            ("证据",),
            acceptable_targets=(target,),
        )
    with pytest.raises(ValueError, match="重复目标"):
        PositiveCase(
            id="duplicate",
            category="test",
            query="q",
            acceptable_targets=(target, ExpectedTarget("Ａ.MD", " A ", ("证据",))),
        )
    with pytest.raises(ValueError, match="basename"):
        ExpectedTarget("notes/a.md", "A", ("证据",))
    with pytest.raises(ValueError, match="required_evidence"):
        ExpectedTarget("a.md", "A")


def test_quality_gate_rejects_invalid_thresholds() -> None:
    positives = (PositiveCase("p", "test", "q", "a.md", "A", ("答案证据",)),)
    embedder = _FakeEmbedder()
    report = evaluate_retrieval(
        embedder=embedder,
        retriever=_FakeRetriever({"q": [_chunk("a.md", "A")]}),  # type: ignore[arg-type]
        positives=positives,
        negatives=(),
    )

    for invalid in (-0.1, 1.1):
        with pytest.raises(ValueError):
            quality_gate_passes(
                report,
                min_hit_at_8=invalid,
                max_negative_false_context_rate=0.0,
            )
