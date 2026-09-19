"""真实评估器的指标与严格块命中规则测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.rag_quality import (
    RELEVANCE_SCHEMA_VERSION,
    EvidenceSpan,
    ExpectedTarget,
    NegativeCase,
    PositiveCase,
    RelevanceCase,
    RelevanceTarget,
    RelevanceView,
    chunk_matches_expectation,
    evaluate_relevance,
    evaluate_retrieval,
    judge_view,
    load_relevance_cases,
    quality_gate_passes,
    relevance_gate_passes,
    validate_relevance_cases,
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
        "示例分表.md",
        "数据库 > 交互",
        ("答案证据",),
    )

    assert chunk_matches_expectation(_chunk(r"C:\notes\示例分表.md", "数据库 > 交互"), case)
    assert chunk_matches_expectation(_chunk("C:/notes/示例分表．ＭＤ", "数据库　＞　交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\旧示例分表.md", "数据库 > 交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\示例分表.md", "交互"), case)
    assert not chunk_matches_expectation(_chunk(r"C:\notes\示例分表.md", "数据库"), case)
    assert not chunk_matches_expectation(
        _chunk(r"C:\notes\示例分表.md", "数据库 > 交互 > 子标题"), case
    )
    assert not chunk_matches_expectation(
        _chunk(r"C:\notes\示例分表.md", "数据库 > 交互", text="相同标题的错误块"),
        case,
    )


def test_chunk_match_rejects_sibling_heading_even_when_parent_contains_both_names() -> None:
    parent = "2. 安全策略：工具安全、数据安全、数据隔离"
    case = PositiveCase(
        "p",
        "agent_security",
        "q",
        "客服示例.md",
        f"{parent} > 工具安全性",
        ("答案证据",),
    )

    assert chunk_matches_expectation(_chunk("客服示例.md", f"{parent} > 工具安全性"), case)
    assert not chunk_matches_expectation(_chunk("客服示例.md", parent), case)
    assert not chunk_matches_expectation(
        _chunk("客服示例.md", f"{parent} > 数据安全性"), case
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
    heading = "二、示例设计 > 1、SPM > 1.1 规则"
    case = PositiveCase(
        id="p",
        category="tracking",
        query="q",
        acceptable_targets=(
            ExpectedTarget("示例设计文档V2.0.1.md", heading, ("答案证据",)),
            ExpectedTarget("示例设计文档V2.0.2.md", heading, ("答案证据",)),
        ),
    )

    assert chunk_matches_expectation(_chunk("示例设计文档V2.0.1.md", heading), case)
    assert chunk_matches_expectation(_chunk("示例设计文档V2.0.2.md", heading), case)
    assert not chunk_matches_expectation(_chunk("示例设计文档V2.0.3.md", heading), case)
    assert not chunk_matches_expectation(
        _chunk("示例设计文档V2.0.2.md", "二、示例设计 > 2、SCM > 2.1 规则"),
        case,
    )


def test_evaluate_retrieval_reports_strict_metrics_and_batches_queries() -> None:
    positives = (
        PositiveCase(
            "p1",
            "cart",
            "我想问一下购物车交互是什么样的？",
            "示例分表.md",
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
                _chunk("示例分表.md", "方案对比", similarity=None),
                _chunk("示例分表.md", "交互", block_id="private-block-id"),
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
            "示例分表.md",
            "交互",
            ("答案证据",),
        )
        for index in range(50)
    )
    gate_negatives = tuple(NegativeCase(f"gate-n-{index}", f"负例 {index}") for index in range(10))
    gate_results = {
        case.query: [
            _chunk(
                "示例分表.md",
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


# ---------------------------------------------------------------------------
# 分级相关性数据结构（PRD 7.1/7.2）
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span(start: int, end: int, text: str) -> EvidenceSpan:
    assert len(text) == end - start
    return EvidenceSpan(start=start, end=end, sha256=_sha256(text))


def _target(
    target_id: str,
    role: str,
    grade: int | None,
    source: str,
    heading: str,
    evidence: tuple[str, ...] = (),
    *,
    group: str = "",
    irrelevant_spans: tuple[EvidenceSpan, ...] = (),
) -> RelevanceTarget:
    return RelevanceTarget(
        target_id=target_id,
        role=role,
        grade=grade,
        source=source,
        heading=heading,
        required_evidence=evidence,
        group=group,
        irrelevant_spans=irrelevant_spans,
    )


def _view(source: str, heading: str, text: str, block_ref: str | None = None) -> RelevanceView:
    return RelevanceView(
        block_ref=block_ref or f"blk_{source}",
        source=source,
        heading=heading,
        text=text,
    )


def test_evidence_span_requires_ordered_offsets_and_lowercase_sha256() -> None:
    span = _span(2, 6, "关键结论")
    assert span.length == 4

    for invalid in ((0, 0), (5, 1), (-1, 3)):
        with pytest.raises(ValueError):
            EvidenceSpan(start=invalid[0], end=invalid[1], sha256="a" * 64)
    with pytest.raises(ValueError):
        EvidenceSpan(start=0, end=3, sha256="A" * 64)
    with pytest.raises(ValueError):
        EvidenceSpan(start=0, end=3, sha256="abc")
    with pytest.raises(TypeError):
        EvidenceSpan(start=0.5, end=3, sha256="a" * 64)  # type: ignore[arg-type]


def test_relevance_target_enforces_role_and_grade_contract() -> None:
    required = _target("t", "required", 3, "a.md", "A > 1", ("关键结论",), group="g1")
    assert required.role == "required" and required.group == "g1"

    with pytest.raises(ValueError, match="required_evidence"):
        _target("t", "required", 3, "a.md", "A")
    with pytest.raises(ValueError, match="等级必须为 2 或 3"):
        _target("t", "required", 1, "a.md", "A", ("证据",))
    with pytest.raises(ValueError, match="等级必须为 0 或 1"):
        _target("t", "forbidden", 3, "a.md", "A", ("证据",))
    with pytest.raises(ValueError, match="role"):
        _target("t", "primary", 3, "a.md", "A", ("证据",))
    with pytest.raises(ValueError, match="basename"):
        _target("t", "allowed", 2, "notes/a.md", "A")
    with pytest.raises(ValueError, match="grade"):
        _target("t", "allowed", 5, "a.md", "A")


def test_relevance_case_groups_equivalent_evidence_once() -> None:
    case = RelevanceCase(
        case_id="C04-001",
        family="C04",
        group_id="C04-g-01",
        split="test",
        query="综合问题",
        targets=(
            _target("t1", "required", 3, "a.md", "A", ("关键结论",), group="g1"),
            _target("t2", "required", 2, "a2.md", "A2", ("等价证据",), group="g1"),
            _target("t3", "required", 2, "b.md", "B", ("补充证据",), group="g2"),
            _target("t4", "forbidden", 0, "x.md", "X", ("无关内容",)),
        ),
    )

    groups = case.required_groups()
    assert set(groups) == {"g1", "g2"}
    assert len(groups["g1"]) == 2
    assert len(case.required_targets) == 3
    assert len(case.forbidden_targets) == 1
    assert not case.has_unjudged_targets


def test_relevance_case_rejects_duplicate_targets_and_bad_enums() -> None:
    target = _target("t1", "required", 3, "a.md", "A", ("关键结论",), group="g1")
    with pytest.raises(ValueError, match="重复目标"):
        RelevanceCase(
            case_id="C01-001",
            family="C01",
            group_id="g",
            split="test",
            query="q",
            targets=(target, target),
        )
    with pytest.raises(ValueError, match="family"):
        RelevanceCase(case_id="x", family="C09", group_id="g", split="test", query="q")
    with pytest.raises(ValueError, match="split"):
        RelevanceCase(case_id="x", family="C01", group_id="g", split="train", query="q")
    with pytest.raises(ValueError, match="review_status"):
        RelevanceCase(
            case_id="x",
            family="C01",
            group_id="g",
            split="test",
            query="q",
            review_status="checked",
        )


def test_validate_relevance_cases_rejects_group_crossing_splits() -> None:
    calibration = RelevanceCase(
        case_id="C05-001",
        family="C05",
        group_id="C05-g-01",
        split="calibration",
        query="q1",
        needs_answer=False,
    )
    test = RelevanceCase(
        case_id="C05-002",
        family="C05",
        group_id="C05-g-01",
        split="test",
        query="q2",
        needs_answer=False,
    )
    validate_relevance_cases((calibration,))
    with pytest.raises(ValueError, match="整组只进一个集合"):
        validate_relevance_cases((calibration, test))


def _dataset_payload() -> dict:
    return {
        "schema_version": RELEVANCE_SCHEMA_VERSION,
        "family": "C01",
        "dataset_id": "synthetic-structure-fixture",
        "corpus_sha256": "b" * 64,
        "cases": [
            {
                "case_id": "C01-001",
                "family": "C01",
                "group_id": "C01-g-01",
                "split": "calibration",
                "query": "同一个词的两个含义",
                "needs_answer": True,
                "targets": [
                    {
                        "target_id": "C01-001-T1",
                        "role": "required",
                        "grade": 3,
                        "source": "a.md",
                        "heading": "A > 1",
                        "required_evidence": ["关键结论"],
                        "group": "",
                        "irrelevant_spans": [],
                    },
                    {
                        "target_id": "C01-001-T2",
                        "role": "forbidden",
                        "grade": 0,
                        "source": "x.md",
                        "heading": "X",
                        "required_evidence": ["无关内容"],
                        "group": "",
                        "irrelevant_spans": [],
                    },
                    {
                        "target_id": "C01-001-T3",
                        "role": "allowed",
                        "grade": "unjudged",
                        "source": "u.md",
                        "heading": "U",
                        "required_evidence": [],
                        "group": "",
                        "irrelevant_spans": [],
                    },
                ],
            }
        ],
    }


def test_load_relevance_cases_parses_grades_and_unjudged(tmp_path: Path) -> None:
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    cases = load_relevance_cases(path, expected_corpus_sha256="b" * 64)

    assert len(cases) == 1
    case = cases[0]
    assert case.family == "C01" and case.split == "calibration"
    assert [target.grade for target in case.targets] == [3, 0, None]
    assert case.has_unjudged_targets
    assert not case.is_redacted


def _redacted_dataset_payload() -> dict:
    """脱敏结构（PRD §7.2）：只留结构字段与私文哈希。"""

    query = _private_dataset_payload()["cases"][0]["query"]
    return {
        "schema_version": 1,
        "redaction": {"policy": "PRD RAG-REL-001 §7.2", "loader": "scripts/relevance_cases/loader.py"},
        "family": "C01",
        "corpus_sha256": "b" * 64,
        "cases": [
            {
                "case_id": "C01-001",
                "family": "C01",
                "group_id": "C01-g-01",
                "split": "calibration",
                "needs_answer": True,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "case_sha256": "c" * 64,
                "targets": [
                    {
                        "target_id": "C01-001-T1",
                        "role": "required",
                        "grade": 3,
                        "group": "",
                    }
                ],
            }
        ],
    }


def _private_dataset_payload() -> dict:
    return {
        "schema_version": 1,
        "family": "C01",
        "corpus_sha256": "b" * 64,
        "cases": [
            {
                "case_id": "C01-001",
                "query": "同一个词的两个含义",
                "note": "",
                "targets": [
                    {
                        "target_id": "C01-001-T1",
                        "source": "a.md",
                        "heading": "A > 1",
                        "required_evidence": ["关键结论"],
                        "irrelevant_spans": [],
                        "relevant_spans": [],
                        "note": "直接给出结论",
                    }
                ],
            }
        ],
    }


def test_load_relevance_cases_requires_private_dataset_for_redacted_structure(tmp_path: Path) -> None:
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_redacted_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="脱敏结构"):
        load_relevance_cases(path)

    cases = load_relevance_cases(path, allow_redacted=True)
    assert cases[0].is_redacted
    assert cases[0].query is None
    assert cases[0].targets[0].source is None
    with pytest.raises(ValueError, match="尚未组装本地私文"):
        validate_relevance_cases(cases)


def test_load_relevance_cases_merges_private_dataset_by_case_and_target(tmp_path: Path) -> None:
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_redacted_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    cases = load_relevance_cases(path, private_dataset=_private_dataset_payload())

    case = cases[0]
    assert not case.is_redacted
    assert case.query == "同一个词的两个含义"
    target = case.targets[0]
    assert target.source == "a.md" and target.heading == "A > 1"
    assert target.required_evidence == ("关键结论",)
    assert target.note == "直接给出结论"


def test_load_relevance_cases_rejects_private_dataset_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_redacted_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    private = _private_dataset_payload()
    private["cases"][0]["query"] = "换了问题原文"
    with pytest.raises(ValueError, match="query 哈希不同"):
        load_relevance_cases(path, private_dataset=private)

    private = _private_dataset_payload()
    private["cases"][0]["query"] = "同一个词的两个含义"
    private["cases"][0]["targets"][0]["target_id"] = "C01-001-T9"
    with pytest.raises(ValueError, match="缺少目标"):
        load_relevance_cases(path, private_dataset=private)


def test_target_without_heading_still_requires_evidence_once_assembled(tmp_path: Path) -> None:
    """真实无标题块必须靠证据片段定位；这条检查只在组装后（有 source）生效。"""

    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload["cases"][0]["targets"][0]
    target["heading"] = ""
    target["required_evidence"] = []
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="无标题目标必须提供证据片段"):
        load_relevance_cases(path)


def test_evaluate_relevance_rejects_unassembled_redacted_cases(tmp_path: Path) -> None:
    """未组装的脱敏结构不得进指标入口：必须报错，而不是返回「全空但有效」的报告。

    改前行为（T23-F1）：`evaluate_relevance` 返回 `precision_macro=None` 且
    `empty_result_rate=1.0`，同时 `precision_valid=True` / `annotation_complete=True`，
    门禁据此报成「A03 Precision=None」「A04 Recall=0.0」等**质量问题**——数据缺失被伪装成
    质量不达标。这里断言指标入口直接拒收。
    """

    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_redacted_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    redacted = load_relevance_cases(path, allow_redacted=True)
    assert redacted[0].is_redacted  # 前提：确实是未组装状态

    with pytest.raises(ValueError, match="数据集未组装") as excinfo:
        evaluate_relevance(redacted, scored={}, returned={})
    message = str(excinfo.value)
    assert redacted[0].case_id in message, "错误信息必须点名未组装的题，便于定位"
    assert "私文" in message and "README" in message, "错误信息必须说明如何组装"

    # 对照：同一批案例组装后可以正常评测（空检索是合法输入，不触发本守卫）
    assembled = load_relevance_cases(path, private_dataset=_private_dataset_payload())
    report = evaluate_relevance(assembled, scored={}, returned={})
    assert report.dataset_case_total == 1
    assert report.overall.empty_result_rate == 1.0


def test_evaluate_relevance_rejects_partially_assembled_cases(tmp_path: Path) -> None:
    """只要有一个目标缺少来源（未组装缺口）就必须报错，不接受半成品评测。"""

    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_redacted_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    cases = load_relevance_cases(path, allow_redacted=True)
    complete = load_relevance_cases(path, private_dataset=_private_dataset_payload())
    mixed = (complete[0], cases[0])

    with pytest.raises(ValueError, match="数据集未组装"):
        evaluate_relevance(mixed, scored={}, returned={})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(family="C09"), "family"),
        (lambda payload: payload.update(description="未知顶层字段"), "未知顶层字段"),
        (lambda payload: payload.update(corpus_sha256="short"), "64 位语料内容哈希"),
        (
            lambda payload: payload["cases"][0].update(corpus_sha256="d" * 64),
            "语料哈希与数据集不一致",
        ),
    ],
)
def test_load_relevance_cases_rejects_schema_drift(tmp_path: Path, mutation, message) -> None:
    payload = _dataset_payload()
    payload["cases"][0]["corpus_sha256"] = "b" * 64
    mutation(payload)
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises((ValueError, TypeError), match=message):
        load_relevance_cases(path)


def test_load_relevance_cases_rejects_stale_corpus_hash(tmp_path: Path) -> None:
    path = tmp_path / "cases-C01.json"
    path.write_text(json.dumps(_dataset_payload(), ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="语料哈希已过期"):
        load_relevance_cases(path, expected_corpus_sha256="c" * 64)


# ---------------------------------------------------------------------------
# 六项指标（PRD 10.1）
# ---------------------------------------------------------------------------


def _metric_cases() -> tuple[RelevanceCase, ...]:
    """三条题：综合题（两个必要证据族）、部分题、无证据题。"""

    case_a = RelevanceCase(
        case_id="C04-001",
        family="C04",
        group_id="C04-g-01",
        split="test",
        query="综合问题",
        targets=(
            _target("a1", "required", 3, "a.md", "A", ("关键结论",), group="g1"),
            _target("a2", "required", 2, "b.md", "B", ("补充解释",), group="g2"),
            _target("a3", "forbidden", 0, "x.md", "X", ("完全无关",), group="gx"),
            _target("a4", "allowed", 2, "c.md", "C", ("互补说明",), group="g3"),
        ),
    )
    case_b = RelevanceCase(
        case_id="C04-002",
        family="C04",
        group_id="C04-g-02",
        split="test",
        query="只有一个必要证据的问题",
        targets=(
            _target("b1", "required", 3, "d.md", "D", ("直接答案",), group="g1"),
            _target("b2", "forbidden", 1, "y.md", "Y", ("话题相近",), group="gy"),
        ),
    )
    case_c = RelevanceCase(
        case_id="C05-001",
        family="C05",
        group_id="C05-g-01",
        split="test",
        query="库里没有答案的问题",
        needs_answer=False,
        targets=(_target("c1", "forbidden", 0, "z.md", "Z", ("近似内容",), group="gz"),),
    )
    return (case_a, case_b, case_c)


def _metric_views() -> tuple[dict[str, list[RelevanceView]], dict[str, list[RelevanceView]]]:
    scored = {
        "C04-001": [_view("a.md", "A", "关键结论在这里"), _view("x.md", "X", "完全无关的内容")],
        "C04-002": [_view("y.md", "Y", "话题相近但没有答案")],
        "C05-001": [_view("z.md", "Z", "近似内容但不是所需信息")],
    }
    returned = {
        "C04-001": [_view("a.md", "A", "关键结论在这里"), _view("b.md", "B", "补充解释说明")],
        "C04-002": [_view("y.md", "Y", "话题相近但没有答案")],
        "C05-001": [],
    }
    return scored, returned


def test_candidate_coverage_and_required_evidence_recall_are_group_based() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()

    report = evaluate_relevance(cases, scored=scored, returned=returned)
    overall = report.overall

    # 必要证据族共 3 个（C04-001 两个、C04-002 一个）；进入评分集合的只有 a1、b1 = 1 个
    assert overall.required_group_total == 3
    assert overall.required_group_scored == 1
    assert overall.candidate_coverage == pytest.approx(1 / 3, abs=1e-6)
    # 最终结果覆盖 a1、a2 两个族，b1 未覆盖
    assert overall.required_group_covered == 2
    assert overall.required_evidence_recall == pytest.approx(2 / 3, abs=1e-6)
    assert overall.required_evidence_recall_macro == pytest.approx(0.5, abs=1e-6)
    assert report.outcomes[1].evidence_recall == 0.0
    assert report.outcomes[1].required_group_scored == 0


def test_returned_precision_is_macro_over_answered_cases_with_micro_ratio() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()

    overall = evaluate_relevance(cases, scored=scored, returned=returned).overall

    assert overall.case_total == 3
    assert overall.answered_case_total == 2
    assert overall.empty_case_total == 1
    assert overall.empty_result_rate == pytest.approx(1 / 3, abs=1e-6)
    assert overall.precision_macro == pytest.approx(0.5, abs=1e-6)
    assert overall.precision_micro == pytest.approx(2 / 3, abs=1e-6)
    assert overall.precision_defined_case_total == 2
    assert overall.returned_view_total == 3
    assert overall.relevant_view_total == 2


def test_hard_negative_admission_rate_uses_scored_denominator() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()

    overall = evaluate_relevance(cases, scored=scored, returned=returned).overall

    assert overall.hard_negative_scored == 3
    assert overall.hard_negative_admitted == 1
    assert overall.hard_negative_admission_rate == pytest.approx(1 / 3, abs=1e-6)


def test_empty_false_context_rate_covers_no_evidence_questions() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()

    overall = evaluate_relevance(cases, scored=scored, returned=returned).overall
    assert overall.no_evidence_case_total == 1
    assert overall.no_evidence_false_context == 0
    assert overall.empty_false_context_rate == 0.0

    noisy = dict(returned)
    noisy["C05-001"] = [_view("z.md", "Z", "近似内容但不是所需信息")]
    admitted = evaluate_relevance(cases, scored=scored, returned=noisy).overall
    assert admitted.no_evidence_false_context == 1
    assert admitted.empty_false_context_rate == 1.0


def test_irrelevant_character_ratio_counts_whole_irrelevant_view_and_unique_spans() -> None:
    irrelevant_text = "话题相近但没有答案"  # 9 字，等级 1 → 整段无关
    span_text = "这一段无关内容"  # 7 字
    case = RelevanceCase(
        case_id="C03-001",
        family="C03",
        group_id="C03-g-01",
        split="test",
        query="多主题分块",
        targets=(
            _target("t1", "required", 3, "a.md", "A", ("关键结论",), group="g1"),
            _target("t2", "forbidden", 1, "y.md", "Y", (irrelevant_text,), group="gy"),
            _target(
                "t3",
                "allowed",
                2,
                "m.md",
                "M",
                ("混合块主题",),
                group="g2",
                irrelevant_spans=(_span(0, len(span_text), span_text),),
            ),
        ),
    )
    mixed_text = f"{span_text}混合块主题"
    views = {
        "C03-001": [
            _view("a.md", "A", "关键结论在这里"),
            _view("y.md", "Y", irrelevant_text),
            _view("m.md", "M", mixed_text),
            _view("m.md", "M", mixed_text),  # 同一块重复出现：无关范围只计一次
        ]
    }

    report = evaluate_relevance((case,), scored=views, returned=views)
    overall = report.overall

    judged = len("关键结论在这里") + len(irrelevant_text) + len(mixed_text) * 2
    assert overall.judged_char_total == judged
    assert overall.irrelevant_char_total == len(irrelevant_text) + len(span_text)
    assert overall.irrelevant_character_ratio == pytest.approx(
        (len(irrelevant_text) + len(span_text)) / judged, abs=1e-6
    )


def test_unjudged_returned_view_invalidates_precision_and_is_not_counted_as_zero() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()
    noisy = dict(returned)
    noisy["C04-001"] = [
        _view("a.md", "A", "关键结论在这里"),
        _view("unknown.md", "未知", "没有被标注过的块"),
    ]

    report = evaluate_relevance(cases, scored=scored, returned=noisy)

    assert report.overall.precision_valid is False
    assert report.unjudged_case_ids == ("C04-001",)
    unjudged_case = report.outcomes[0]
    assert unjudged_case.unjudged_returned == 1
    assert unjudged_case.precision_defined is False
    assert unjudged_case.precision is None
    # 该题退出宏平均，但另一题仍为 0.0
    assert report.overall.precision_defined_case_total == 1
    assert report.overall.precision_macro == 0.0


def test_span_outside_view_text_marks_annotation_incomplete() -> None:
    span_text = "无关片段"
    case = RelevanceCase(
        case_id="C06-001",
        family="C06",
        group_id="C06-g-01",
        split="test",
        query="补充内容",
        targets=(
            _target(
                "t1",
                "required",
                2,
                "m.md",
                "M",
                ("主题内容",),
                group="g1",
                # 标注范围指向原文更靠后的位置；当返回视图与标注时正文不一致时，
                # 指标不能静默按 0 计入，而是把该案例标为标注不完整。
                irrelevant_spans=(_span(100, 104, span_text),),
            ),
        ),
    )
    views = {"C06-001": [_view("m.md", "M", "主题内容")]}

    report = evaluate_relevance((case,), scored=views, returned=views)

    assert report.overall.annotation_complete is False
    assert report.overall.precision_valid is False
    assert report.outcomes[0].span_mismatch == 1
    assert report.outcomes[0].irrelevant_chars == 0


def test_relevance_report_reports_per_family_and_per_split() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()

    report = evaluate_relevance(cases, scored=scored, returned=returned)

    assert set(report.families) == {"C04", "C05"}
    assert report.families["C05"].case_total == 1
    assert report.families["C05"].required_group_total == 0
    assert report.families["C05"].precision_macro is None
    assert report.families["C05"].empty_result_rate == 1.0
    assert report.families["C04"].empty_false_context_rate is None
    assert set(report.splits) == {"test"}
    assert report.splits["test"].case_total == 3
    serialized = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "关键结论" not in serialized  # 报告不携带正文


def test_relevance_gate_reports_a03_a04_a05_failures() -> None:
    cases = _metric_cases()
    scored, returned = _metric_views()
    failing = evaluate_relevance(cases, scored=scored, returned=returned)

    passed, failures = relevance_gate_passes(failing)
    assert passed is False
    assert any("A03" in item for item in failures)
    assert any("A04" in item for item in failures)
    assert any("A05" in item for item in failures)

    perfect_case = RelevanceCase(
        case_id="C04-900",
        family="C04",
        group_id="C04-g-900",
        split="test",
        query="完全命中",
        targets=(_target("t1", "required", 3, "a.md", "A", ("关键结论",), group="g1"),),
    )
    no_evidence_case = RelevanceCase(
        case_id="C05-900",
        family="C05",
        group_id="C05-g-900",
        split="test",
        query="无证据问题",
        needs_answer=False,
        targets=(_target("t1", "forbidden", 0, "z.md", "Z", ("近似内容",), group="gz"),),
    )
    views = {"C04-900": [_view("a.md", "A", "关键结论在这里")], "C05-900": []}
    perfect = evaluate_relevance((perfect_case, no_evidence_case), scored=views, returned=views)

    passed, failures = relevance_gate_passes(perfect, min_recall=1.0, min_family_recall=1.0)
    assert passed is True, failures
    assert failures == ()


def test_irrelevant_span_is_located_by_text_when_view_carries_supplement() -> None:
    """生产视图正文可附带同标题补充文字，标注偏移不再一一对应，此时按标注文本定位。"""

    quote = "这一段无关补充文字"
    case = RelevanceCase(
        case_id="C06-002",
        family="C06",
        group_id="C06-g-02",
        split="test",
        query="补充内容",
        targets=(
            _target(
                "t1",
                "required",
                2,
                "m.md",
                "M",
                ("主内容",),
                group="g1",
                irrelevant_spans=(
                    EvidenceSpan(start=0, end=len(quote), sha256=_sha256(quote), text=quote),
                ),
            ),
        ),
    )
    view_text = f"标题：M\n内容：主内容\n\n同标题补充信息：\n{quote}"
    views = {"C06-002": [_view("m.md", "M", view_text)]}

    report = evaluate_relevance((case,), scored=views, returned=views)

    assert report.outcomes[0].span_mismatch == 0
    assert report.overall.annotation_complete is True
    assert report.outcomes[0].irrelevant_chars == len(quote)
    assert report.outcomes[0].judged_chars == len(view_text)


def test_judge_view_grades_annotated_blocks_and_marks_others_unjudged() -> None:
    case = _metric_cases()[0]

    assert judge_view(case, _view("a.md", "A", "关键结论在这里")) == 3
    assert judge_view(case, _view("b.md", "B", "补充解释说明")) == 2
    assert judge_view(case, _view("x.md", "X", "完全无关的内容")) == 0
    # 标题对但正文不含标注证据：不是命中，也不是 0，而是未标注
    assert judge_view(case, _view("a.md", "A", "标题相同但正文不同")) is None
    # 来源不同、完全没被标注过的块一律 unjudged，不得当成无关
    assert judge_view(case, _view("unknown.md", "未知", "从来没有被标注过的块")) is None


def test_evidence_span_text_must_match_offsets_and_hash() -> None:
    quote = "关键结论"
    EvidenceSpan(start=0, end=len(quote), sha256=_sha256(quote), text=quote)
    with pytest.raises(ValueError, match="长度"):
        EvidenceSpan(start=0, end=99, sha256=_sha256(quote), text=quote)
    with pytest.raises(ValueError, match="sha256 不一致"):
        EvidenceSpan(start=0, end=len(quote), sha256="a" * 64, text=quote)


def test_relevance_view_from_chunk_uses_effective_context_text() -> None:
    chunk = RetrievedChunk(
        block_id="private-block-id",
        text="短内容",
        similarity=0.42,
        source_file=r"C:\notes\埋点.md",
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path="SPM > 规则",
        metadata={},
        context_text="标题：SPM > 规则\n内容：短内容\n\n同标题补充信息：公式证据",
    )

    view = RelevanceView.from_chunk(chunk)

    assert view.source == "埋点.md"
    assert view.block_ref.startswith("blk_")
    assert "公式证据" in view.text
    assert "private-block-id" not in view.block_ref
