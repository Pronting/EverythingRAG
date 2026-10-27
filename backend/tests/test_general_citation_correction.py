import json
from pathlib import Path

import pytest

from app.agent.contracts import Evidence, EvidenceSourceType
from app.agent.research import ResearchConclusion, make_evidence
from app.agent.runtime import _semantic_assessment_prompt, _synthesis_prompt
from app.agent.verification import (
    GroundedParagraph,
    ParagraphBindingError,
    ParagraphCorrection,
    ParagraphCorrections,
    SynthesisRequest,
    SynthesisReview,
    SynthesisUnavailable,
    apply_paragraph_corrections,
    bind_review_paragraphs,
)


@pytest.fixture
def actual_http_run():
    path = Path.home() / (
        ".everything-rag/workspaces/default/artifacts/synthesis/"
        "run-8bcfe4cbee234068a80b6c8f32d94dbe/"
        "review-1eb705cd65d841f1a70d70b7171f59e5.json"
    )
    if not path.exists():
        pytest.skip("本机 HTTP 问题执行记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    review = SynthesisReview.model_validate_json(record["raw"])
    background = make_evidence(
        workspace_id="default", run_id=evidence[0].run_id, task_id="general",
        source_type=EvidenceSourceType.GENERAL_KNOWLEDGE, source_ref="general:http",
        content=review.paragraphs[0].text,
    )
    request = SynthesisRequest(
        question=record["question"], requirements=record["requirements"],
        evidence=[*evidence, background],
        conclusions=[ResearchConclusion(
            statement=background.content, evidence_ids=[background.evidence_id], basis="general",
        )],
    )
    return request, evidence, background, review


def test_general_evidence_id_is_absent_from_synthesis_prompt(actual_http_run):
    request, evidence, background, _ = actual_http_run
    for structured in (False, True):
        prompt = _synthesis_prompt(request, evidence, [background], structured=structured)
        assert background.evidence_id not in prompt
        assert background.content in prompt
        assert evidence[0].evidence_id in prompt
    restricted = _synthesis_prompt(request, evidence, structured=True)
    assert background.content not in restricted


def test_general_correction_preserves_source_numbers_and_business_citation(actual_http_run):
    request, evidence, background, review = actual_http_run
    with pytest.raises(ParagraphBindingError) as failure:
        bind_review_paragraphs(review, evidence, question=request.question)
    assert set(failure.value.issues) == {0, 1}
    source_paragraph = review.paragraphs[1].model_copy(deep=True)
    source_paragraph.text = (
        "当 requestState 为 1 时，表示命中行为风控黑名单，网关拒绝请求并返回 "
        '`{"code":105222,"msg":"Too many requests"}`。'
    )
    corrections = ParagraphCorrections(corrections=[
        ParagraphCorrection(index=0, paragraph=GroundedParagraph(text=background.content)),
        ParagraphCorrection(index=1, paragraph=source_paragraph),
    ])
    with pytest.raises(SynthesisUnavailable, match="缺少来源"):
        apply_paragraph_corrections(review, corrections, {0, 1}, {"R1"})
    corrected = apply_paragraph_corrections(
        review, corrections, {0, 1}, {"R1"}, allow_general_explanation=True,
    )
    content = bind_review_paragraphs(corrected, evidence, question=request.question)
    assert "HTTP 429" in content
    assert "105222" in content
    assert corrected.paragraphs[0].citations == []
    assert corrected.paragraphs[1].citations == review.paragraphs[1].citations
    assert corrected.paragraphs[2] == review.paragraphs[2]
    assert review.paragraphs[0].citations
    for unsupported in ("429", "987654"):
        changed = corrected.model_copy(deep=True)
        changed.paragraphs[1].text += f"网关响应状态码为 {unsupported}。"
        with pytest.raises(ParagraphBindingError):
            bind_review_paragraphs(changed, evidence, question=request.question)


def test_general_semantic_assessment_receives_background(actual_http_run):
    request, evidence, background, review = actual_http_run
    review.paragraphs[0].citations = []
    prompt = _semantic_assessment_prompt(request, review, evidence, [], background=[background])
    packet = json.loads(prompt.partition("\n待验收数据：\n")[2])
    assert packet["general_knowledge"] == [background.content]
    assert packet["paragraphs"][0]["sources"] == []
    assert packet["paragraphs"][1]["sources"]
    assert "不能从相同提示文案推断状态码相同" in prompt
