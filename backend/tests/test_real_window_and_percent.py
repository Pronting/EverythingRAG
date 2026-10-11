import json
from pathlib import Path

import pytest

from app.agent.contracts import Evidence
from app.agent.research import MaterialGap
from app.agent.tools import focused_window_offset
from app.agent.verification import (
    CitationGrounding,
    GroundedParagraph,
    ParagraphBindingError,
    ParagraphCorrection,
    ParagraphCorrections,
    SynthesisReview,
    SynthesisUnavailable,
    _validate_calculation,
    apply_paragraph_corrections,
    bind_review_paragraphs,
)


@pytest.mark.parametrize("query", [
    "知识库中 AI Native 的具体实践有哪些？",
    "文章怎样推进 AI Native，人员与流程有什么变化？",
])
def test_actual_article_window_includes_named_section(query):
    path = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-repaired-r3-20260922/v02-real-blocks.json"
    if not path.exists():
        pytest.skip("本地真实文档记录未提供")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    blocks = [item for item in record["blocks"] if item["block_id"].startswith("d97abafc6576660b99ccafeb682a88a9")]
    blocks.sort(key=lambda item: (str(item.get("heading_path") or ""), item["block_id"]))
    content = "\n\n".join((f"章节：{item['heading_path']}\n" if item.get("heading_path") else "") + item["text"] for item in blocks)
    offset = focused_window_offset(content, query, 12000)
    window = content[offset:offset + 12000]
    assert 0 < offset <= len(content) - 12000
    assert "第三阶段 · AI Native" in window
    assert "组织形态" in window and "结果审计者" in window


@pytest.mark.parametrize("claim", ["计算结果为80%。", "计算结果为80％。"])
def test_ratio_calculation_supports_percentage(claim):
    entry = CitationGrounding(marker="c", claim=claim, quote="FPY=(总行数-改动行数)/总行数",
                              evidence_id="calculation", calculation="(total-changed)/total",
                              inputs={"total": 100, "changed": 20}, result=0.8)
    _validate_calculation(entry, "总行数100行，改动20行。")
    for incorrect in ("计算结果为90%。", "计算结果为0.8%。"):
        with pytest.raises(SynthesisUnavailable):
            _validate_calculation(entry.model_copy(update={"claim": incorrect}), "总行数100行，改动20行。")


def test_actual_v05_percentage_and_unsupported_zero():
    path = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-5f05fdc834a040aaba2e4df7ec32bfee/review-17d0b09159374f84a829e1aba4f32acb.json"
    )
    if not path.exists():
        pytest.skip("本地真实 V05 记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    paragraphs = [GroundedParagraph.model_validate(item) for item in json.loads(record["raw"])["paragraphs"]]
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    review = SynthesisReview(paragraphs=[paragraphs[1]], covered_requirements=["R1"], gaps=[])
    assert "80%" in bind_review_paragraphs(review, evidence, question=record["question"])
    review.paragraphs = [paragraphs[3]]
    with pytest.raises(ParagraphBindingError):
        bind_review_paragraphs(review, evidence, question=record["question"])


def test_document_title_number_is_metadata_only_when_identity_matches():
    source = Evidence(
        evidence_id="ev-k8s", workspace_id="ws", run_id="run", source_type="knowledge",
        source_ref="doc#section", content="Deployment 负责保持一组 Pod 处于运行状态。",
        content_hash="hash", retrieved_at="2026-09-23T00:00:00Z",
        document_title="文章精选一文带你理解14个K8S必备基础概念.md",
    )
    supported = SynthesisReview(
        paragraphs=[GroundedParagraph(
            text="《一文带你理解14个 K8S 基础概念》说明 Deployment 负责保持 Pod 运行。",
            citations=[{"evidence_id": "ev-k8s", "start_line": 1, "end_line": 1}],
        )],
        covered_requirements=["R1"], gaps=[],
    )
    assert "14个 K8S" in bind_review_paragraphs(supported, [source], question="职责是什么")

    unsupported = supported.model_copy(deep=True)
    unsupported.paragraphs[0].text = "Deployment 负责让 14 个 Pod 保持运行。"
    with pytest.raises(ParagraphBindingError):
        bind_review_paragraphs(unsupported, [source], question="职责是什么")


def test_heading_number_is_metadata_only_when_chapter_identity_matches():
    source = Evidence(
        evidence_id="ev-agent", workspace_id="ws", run_id="run", source_type="knowledge",
        source_ref="doc#section", content="Agent 负责理解意图并选择合适工具。",
        content_hash="hash", retrieved_at="2026-09-23T00:00:00Z",
        document_title="航空客服智能体实践.md",
        heading_path="五、核心技术实现 > 5.1 Agent 主体设计与实现",
    )
    review = SynthesisReview(
        paragraphs=[GroundedParagraph(
            text="在「五、核心技术实现 > 5.1 Agent 主体设计与实现」中，Agent 负责理解意图并选择合适工具。",
            citations=[{"evidence_id": "ev-agent", "start_line": 1, "end_line": 1}],
        )],
        covered_requirements=["R1"], gaps=[],
    )
    assert "5.1 Agent" in bind_review_paragraphs(review, [source], question="Agent 如何设计")


def test_gap_correction_accepts_equivalent_explicit_wording():
    review = SynthesisReview(
        paragraphs=[GroundedParagraph(text="旧正文", citations=[])],
        covered_requirements=[], gaps=[],
    )
    corrections = ParagraphCorrections(corrections=[ParagraphCorrection(
        index=0,
        paragraph=GroundedParagraph(
            text="本轮没有取得 eventExpiryDuration 单位为天的官方直接说明。",
            citations=[],
        ),
        gaps=[MaterialGap(
            requirement_ids=["R1"],
            missing="官方页面未直接说明 eventExpiryDuration 的单位是天",
            impact="单位仍需确认",
        )],
    )])

    corrected = apply_paragraph_corrections(review, corrections, {0}, {"R1"})
    assert corrected.gaps[0].requirement_ids == ["R1"]


def test_correction_can_turn_unsupported_numbers_into_hypothetical_example():
    review = SynthesisReview(
        paragraphs=[GroundedParagraph(
            text="旧正文包含来源外数字。",
            citations=[{"evidence_id": "ev-1", "start_line": 1, "end_line": 1}],
        )],
        covered_requirements=["R1"], gaps=[],
    )
    corrections = ParagraphCorrections(corrections=[ParagraphCorrection(
        index=0,
        paragraph=GroundedParagraph(
            text="假设示例：员工在第二天调岗，用于说明历史事实与当前事实需要分别保存。",
            citations=[],
        ),
    )])

    corrected = apply_paragraph_corrections(review, corrections, {0}, {"R1"})
    assert corrected.paragraphs[0].citations == []
