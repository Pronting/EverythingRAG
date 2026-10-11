import json
from pathlib import Path

import pytest

from app.agent.contracts import Evidence, EvidenceSourceType
from app.agent.research import dedupe_evidence, make_evidence
from app.agent.runtime import (
    _semantic_assessment_prompt,
    _synthesis_prompt,
    semantic_review_decision,
)
from app.agent.verification import (
    GroundedParagraph,
    ParagraphBindingError,
    SynthesisRequest,
    SynthesisReview,
    bind_review_paragraphs,
)


def test_actual_t19_preserves_web_identity_in_both_prompts() -> None:
    directory = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-887c2064ee3f46728579a4a57d7e3616"
    )
    if not directory.exists():
        pytest.skip("本地真实 T19 记录未提供")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("review-*.json")]
    record = next(item for item in records if item.get("review") and any(
        source["evidence_id"] == "ev-aaf50617c825d681dc219a99" for source in item["evidence"]
    ))
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    web = next(item for item in evidence if item.evidence_id == "ev-aaf50617c825d681dc219a99")
    assert web.source_type == "web"
    assert web.source_ref == (
        "https://kubernetes.io/zh-cn/docs/tasks/configure-pod-container/"
        "configure-liveness-readiness-startup-probes"
    )
    request = SynthesisRequest(question=record["question"],
                               requirements=record["requirements"], evidence=evidence)
    review = SynthesisReview.model_validate(record["review"])
    generation = _synthesis_prompt(request, evidence, structured=True)
    assert f"来源类型：{web.source_type}；来源位置：{web.source_ref}" in generation
    assessment = _semantic_assessment_prompt(request, review, evidence, [])
    packet = json.loads(assessment.partition("\n待验收数据：\n")[2])
    actual = next(item for item in packet["available_evidence"] if item["evidence_id"] == web.evidence_id)
    assert actual["source_type"] == web.source_type
    assert actual["source_ref"] == web.source_ref
    assert actual["document_title"] == web.document_title
    assert web.source_ref in assessment


def test_actual_s19_url_version_and_business_numbers_are_separate() -> None:
    path = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-887c2064ee3f46728579a4a57d7e3616/review-410dc82c5bbe4d4aad2e2047b24cba60.json"
    )
    if not path.exists():
        pytest.skip("本地真实 S19 记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    correction = next(item for item in json.loads(record["raw"])["corrections"] if item["index"] == 6)
    paragraph = GroundedParagraph.model_validate(correction["paragraph"])
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    review = SynthesisReview(paragraphs=[paragraph], covered_requirements=["R1"], gaps=[])
    assert "v1-32.docs.kubernetes.io" in bind_review_paragraphs(review, evidence, question=record["question"])
    for text in (
        paragraph.text.replace("v1-32.docs.kubernetes.io", "v1-99.docs.kubernetes.io"),
        paragraph.text.replace("v1.32", "v1.99"),
        paragraph.text + "业务处理时限为 987654 秒。",
    ):
        changed = review.model_copy(deep=True)
        changed.paragraphs[0].text = text
        with pytest.raises(ParagraphBindingError):
            bind_review_paragraphs(changed, evidence, question=record["question"])


def test_actual_t12_full_title_identity_and_business_numbers() -> None:
    path = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-8f0459ae4165452294582a74d6a0f1c9/review-0fb590e21aba468daa9d3daec26bed47.json"
    )
    if not path.exists():
        pytest.skip("本地真实 T12 纠正记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    correction = next(item for item in json.loads(record["raw"])["corrections"] if item["index"] == 4)
    paragraph = GroundedParagraph.model_validate(correction["paragraph"])
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    source = next(item for item in evidence if item.evidence_id == paragraph.citations[0].evidence_id)
    title = source.document_title
    original = paragraph.text
    assert title not in original
    paragraph.text = original.replace("14 个 K8S 必备基础概念", title)
    review = SynthesisReview(paragraphs=[paragraph], covered_requirements=["R1"], gaps=[])
    assert title in bind_review_paragraphs(review, evidence, question=record["question"])
    short_title = review.model_copy(deep=True)
    short_title.paragraphs[0].text = original
    assert "14 个 K8S 必备基础概念" in bind_review_paragraphs(
        short_title, evidence, question=record["question"]
    )
    for unsupported in (
        paragraph.text.replace(title, title.replace("14", "15")),
        paragraph.text.replace(title, title.replace("基础概念", "其他概念")),
        paragraph.text + "该接口的调用上限为 14 次。",
    ):
        changed = review.model_copy(deep=True)
        changed.paragraphs[0].text = unsupported
        with pytest.raises(ParagraphBindingError):
            bind_review_paragraphs(changed, evidence, question=record["question"])


def _web_evidence(
    content: str,
    *,
    url: str = "https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_CreateMemory.html",
    heading: str = "eventExpiryDuration",
    start: int = 0,
) -> Evidence:
    return make_evidence(
        workspace_id="ws-1",
        run_id="run-1",
        task_id="task-1",
        source_type=EvidenceSourceType.WEB,
        source_ref=url,
        content=content,
        document_title="CreateMemory - API Reference",
        heading_path=heading,
        source_start=start,
        source_end=start + len(content),
        supported_requirements=["R1"],
    )


def test_same_page_overlapping_fragments_merge_with_aliases() -> None:
    first = _web_evidence("eventExpiryDuration\nMinimum: 7\nMaximum: 365", start=100)
    second = _web_evidence("Minimum: 7\nMaximum: 365\nType: Integer", start=120)

    merged = dedupe_evidence([first, second])

    assert len(merged) == 1
    assert first.evidence_id in merged[0].member_evidence_ids
    assert second.evidence_id in merged[0].member_evidence_ids
    assert "Type: Integer" in merged[0].content


def test_different_locale_or_heading_remains_independent() -> None:
    english = _web_evidence("Minimum: 7\nMaximum: 365")
    japanese = _web_evidence(
        "Minimum: 7\nMaximum: 365",
        url="https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/APIReference/API_CreateMemory.html",
    )
    other_heading = _web_evidence(
        "Minimum: 1\nMaximum: 100", heading="memoryStrategies"
    )

    assert len(dedupe_evidence([english, japanese, other_heading])) == 3


def test_deterministic_review_can_skip_second_model_call() -> None:
    evidence = _web_evidence("eventExpiryDuration Minimum: 7 Maximum: 365 days")
    review = SynthesisReview(
        paragraphs=[GroundedParagraph(
            text="eventExpiryDuration 的范围为 7 到 365 days。",
            citations=[{"evidence_id": evidence.evidence_id, "start_line": 1, "end_line": 1}],
        )],
        covered_requirements=["R1"],
        gaps=[],
    )
    request = SynthesisRequest(
        question="eventExpiryDuration 的范围是什么？",
        requirements={"R1": "eventExpiryDuration 的范围是什么？"},
        evidence=[evidence],
    )

    assert semantic_review_decision(request, review, [evidence]) == (
        False,
        "deterministic_checks_complete",
    )
