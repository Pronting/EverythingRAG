import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent.contracts import AgentKind, Evidence, ResearchTask
from app.agent.evidence_text import evidence_location_text
from app.agent.planner import build_plan, dependency_order
from app.agent.research import MaterialGap, ResearchProgress, ResearchRequest, ResearchResult
from app.agent.routing import RoutingInput, route_question
from app.agent.runtime import (
    ModelCallTiming,
    SynthesisCorrectionBudget,
    _execution_notes,
    _general_evidence_digest,
    _previous_citation_evidence,
    _reset_for_resume,
    _resume_state_from_checkpoint,
    _semantic_assessment_prompt,
)
from app.agent.verification import (
    CitationGrounding,
    CitationSpan,
    GroundedParagraph,
    ParagraphBindingError,
    ParagraphCorrection,
    ParagraphCorrections,
    ParagraphSynthesisReview,
    SemanticAssessment,
    SynthesisRequest,
    SynthesisReview,
    SynthesisUnavailable,
    apply_paragraph_corrections,
    apply_semantic_assessment,
    bind_review_citations,
    bind_review_paragraphs,
    citation_source_lines,
)


@pytest.mark.parametrize("question", [
    "货拉拉 AI Coding 实践文章中的 FPY 如何计算？为什么 FPY 高仍然不能证明代码质量好？",
    "请解释权限最小化原则，并结合知识库中 AWS AgentCore 航空客服案例，说明 Identity、Policy 和 Gateway 如何限制工具访问。",
])
def test_general_waits_for_all_research_queries(question):
    decision = route_question(RoutingInput(question=question, file_count=241, chunk_count=4718))
    plan = build_plan(question, decision, retrieval_queries=[question, " ".join(question.split())])
    research_ids = {task.task_id for task in plan.tasks if task.agent_kind is not AgentKind.GENERAL}
    general = [task for task in plan.tasks if task.agent_kind is AgentKind.GENERAL]
    assert research_ids
    assert general
    for task in general:
        assert research_ids <= set(task.depends_on)
    levels = dependency_order(plan.tasks)
    assert all(task.task_id not in levels[0] for task in general)


def recorded_citations(case):
    path = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-verify-20260922" / f"{case}-evidence.json"
    if not path.is_file():
        pytest.skip("本机真实浏览器证据未提供")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(record["answers"][0]["citations"])


def test_previous_real_sources_preserve_identity_and_full_text():
    citations = recorded_citations("C22.2")
    evidence = _previous_citation_evidence(citations, workspace_id="default", run_id="resume-check")
    assert len(evidence) > 6
    by_source = {item.source_ref: item for item in evidence}
    for citation in citations:
        if citation["source_type"] != "knowledge":
            continue
        item = by_source[citation["source_ref"]]
        assert citation["snippet"] in item.content
        assert item.document_title == citation["document_title"]
        assert item.supported_requirements == []


def test_general_receives_all_real_sources_without_duplicates():
    evidence = _previous_citation_evidence(
        recorded_citations("C22.2"), workspace_id="default", run_id="digest-check",
    )
    result = ResearchResult(
        task_id="previous", agent_kind=AgentKind.RAG, evidence=evidence,
        progress=ResearchProgress(task_id="previous"),
    )
    request = ResearchRequest(
        task=ResearchTask(task_id="general", agent_kind=AgentKind.GENERAL, question="继续按文档讲其他设计因素"),
        dependency_results=[result, result],
    )
    digest = _general_evidence_digest(request)
    assert len(digest) == len(evidence)
    assert len(digest) > 6
    assert "复用" in _execution_notes({}, [result])[0]


def test_resume_preserves_research_and_plan():
    question = "继续按文档讲其他设计因素"
    decision = route_question(RoutingInput(question=question, file_count=241, chunk_count=4718))
    plan = build_plan(question, decision)
    state = {
        "plan": plan.model_dump(mode="json"), "phase": "done",
        "completed_task_ids": plan.task_ids(), "failed_task_ids": [],
        "task_results": {}, "error": {"code": "synthesis_model_unavailable"},
    }
    resumed = _reset_for_resume(state)
    assert resumed["plan"] == state["plan"]
    assert resumed["completed_task_ids"] == state["completed_task_ids"]
    assert resumed["error"] == {}


def test_checkpoint_resume_preserves_original_revision_and_grants():
    initial = {
        "run_id": "run-1",
        "workspace_id": "ws-1",
        "conversation_id": "conv-1",
        "question": "原问题",
        "knowledge_revision": 12,
        "approval_grants": [],
    }
    checkpoint = {
        **initial,
        "knowledge_revision": 7,
        "approval_grants": [{"tool_category": "network"}],
        "completed_task_ids": ["t1"],
        "task_results": {"t1": {"task_id": "t1"}},
        "error": {"error_code": "provider_timeout"},
    }

    resumed = _resume_state_from_checkpoint(initial, checkpoint)

    assert resumed["knowledge_revision"] == 7
    assert resumed["approval_grants"] == [{"tool_category": "network"}]
    assert resumed["completed_task_ids"] == ["t1"]
    assert resumed["error"] == {}


def test_real_formula_quote_resolves_actual_section():
    citations = recorded_citations("T13")
    evidence = _previous_citation_evidence(citations, workspace_id="default", run_id="quote-check")
    formula = next(
        line for item in evidence for line in item.content.splitlines()
        if line.startswith("FPY ＝")
    )
    introduction = next(item for item in evidence if "读完你会知道" in item.content)
    review = SynthesisReview(
        content=formula + " {{cite-1}}", covered_requirements=["R1"], gaps=[],
        citations=[CitationGrounding(
            marker="cite-1", claim=formula, quote=formula, evidence_id=introduction.evidence_id,
        )],
    )
    rendered = bind_review_citations(review, evidence)
    supporting = next(item for item in evidence if formula in item.content)
    assert "{{" + supporting.evidence_id + "}}" in rendered
    assert introduction.evidence_id not in rendered


def test_topic_preview_cannot_support_formula_variables():
    citations = recorded_citations("T13")
    evidence = _previous_citation_evidence(citations, workspace_id="default", run_id="quote-check")
    formula = next(
        line for item in evidence for line in item.content.splitlines()
        if line.startswith("FPY ＝")
    )
    introduction = next(item for item in evidence if "读完你会知道" in item.content)
    quote = next(line for line in introduction.content.splitlines() if "FPY 的公式" in line)
    review = SynthesisReview(
        content=formula + " {{cite-1}}", covered_requirements=["R1"], gaps=[],
        citations=[CitationGrounding(
            marker="cite-1", claim=formula, quote=quote, evidence_id=introduction.evidence_id,
        )],
    )
    with pytest.raises(SynthesisUnavailable, match="公式中的变量"):
        bind_review_citations(review, evidence)


def test_user_numbers_can_be_calculated_from_source_formula():
    evidence = _previous_citation_evidence(
        recorded_citations("T13"), workspace_id="default", run_id="calculation-check",
    )
    formula = next(line for item in evidence for line in item.content.splitlines() if line.startswith("FPY ＝"))
    source = next(item for item in evidence if formula in item.content)
    claim = "按题目给出的行数计算，FPY 为 80%。"
    review = SynthesisReview(
        content=claim + " {{cite-1}}", covered_requirements=["r1"], gaps=[],
        citations=[CitationGrounding(
            marker="cite-1", claim=claim, quote=formula, evidence_id=source.evidence_id,
            calculation="(total-changed)/total*100", inputs={"total": 100, "changed": 20}, result=80,
        )],
    )
    content = bind_review_citations(review, evidence, question="提交版本有100行，改了20行，FPY是多少？")
    assert source.evidence_id in content
    review.citations[0].result = 90
    with pytest.raises(SynthesisUnavailable, match="运算不一致"):
        bind_review_citations(review, evidence, question="提交版本有100行，改了20行，FPY是多少？")


def test_repeated_marker_must_not_bind_different_claims():
    citations = recorded_citations("T13")
    evidence = _previous_citation_evidence(citations, workspace_id="default", run_id="marker-check")
    quote = next(line for item in evidence for line in item.content.splitlines() if line.startswith("FPY ＝"))
    source = next(item for item in evidence if quote in item.content)
    review = SynthesisReview(
        content=quote + " {{cite-1}}\n\n另一项陈述 {{cite-1}}", covered_requirements=["r1"], gaps=[],
        citations=[CitationGrounding(marker="cite-1", claim=quote, quote=quote, evidence_id=source.evidence_id)],
    )
    with pytest.raises(SynthesisUnavailable, match="未完整对应"):
        bind_review_citations(review, evidence)


@pytest.mark.parametrize(("case", "format_marker"), [("C21.3", "**"), ("T12", "|")])
def test_quote_validation_matches_displayed_real_source(case, format_marker):
    citations = recorded_citations(case)
    evidence = _previous_citation_evidence(citations, workspace_id="default", run_id="display-check")
    source = next(item for item in evidence if format_marker in item.content)
    line = next(line for line in source.content.splitlines() if format_marker in line)
    quote = evidence_location_text(body=line)
    review = SynthesisReview(
        content=quote + " {{cite-1}}", covered_requirements=["r1"], gaps=[],
        citations=[CitationGrounding(marker="cite-1", claim=quote, quote=quote, evidence_id=source.evidence_id)],
    )
    assert source.evidence_id in bind_review_citations(review, evidence)
    review.citations[0].quote += "未包含在原文的额外事实"
    with pytest.raises(SynthesisUnavailable, match="未出现在"):
        bind_review_citations(review, evidence)


def test_structured_paragraph_uses_actual_formula_lines():
    evidence = _previous_citation_evidence(
        recorded_citations("T13"), workspace_id="default", run_id="line-check",
    )
    source = next(item for item in evidence if any(line.startswith("FPY ＝") for line in citation_source_lines(item)))
    lines = citation_source_lines(source)
    position = next(index for index, line in enumerate(lines, 1) if line.startswith("FPY ＝"))
    paragraph = GroundedParagraph(text=lines[position - 1], citations=[CitationSpan(
        evidence_id=source.evidence_id, start_line=position, end_line=position,
    )])
    review = SynthesisReview(paragraphs=[paragraph], covered_requirements=["r1"], gaps=[])
    content = bind_review_paragraphs(review, evidence, question="FPY如何计算？")
    assert source.evidence_id in content
    paragraph.citations[0].end_line = len(lines) + 1
    with pytest.raises(ParagraphBindingError, match="超出"):
        bind_review_paragraphs(review, evidence, question="FPY如何计算？")


def test_structured_paragraph_reports_only_invalid_item():
    evidence = _previous_citation_evidence(
        recorded_citations("T13"), workspace_id="default", run_id="line-check",
    )
    source = next(item for item in evidence if any(line.startswith("FPY ＝") for line in citation_source_lines(item)))
    lines = citation_source_lines(source)
    position = next(index for index, line in enumerate(lines, 1) if line.startswith("FPY ＝"))
    review = SynthesisReview(paragraphs=[
        GroundedParagraph(text="## 指标定义"),
        GroundedParagraph(text=lines[position - 1], citations=[CitationSpan(
            evidence_id=source.evidence_id, start_line=1, end_line=1,
        )]),
    ], covered_requirements=["r1"], gaps=[])
    with pytest.raises(ParagraphBindingError) as caught:
        bind_review_paragraphs(review, evidence, question="FPY如何计算？")
    assert set(caught.value.issues) == {1}


def test_structured_paragraph_accepts_verified_runtime_count():
    evidence = _previous_citation_evidence(
        recorded_citations("T13"), workspace_id="default", run_id="runtime-count",
    )
    source = next(
        item
        for item in evidence
        if any(line.startswith("FPY ＝") for line in citation_source_lines(item))
    )
    lines = citation_source_lines(source)
    position = next(index for index, line in enumerate(lines, 1) if line.startswith("FPY ＝"))
    paragraph = GroundedParagraph(
        text=lines[position - 1] + "；本轮网页证据 73 条",
        citations=[CitationSpan(
            evidence_id=source.evidence_id,
            start_line=position,
            end_line=position,
        )],
    )
    review = SynthesisReview(paragraphs=[paragraph], covered_requirements=["r1"], gaps=[])

    with pytest.raises(ParagraphBindingError):
        bind_review_paragraphs(review, evidence, question="FPY如何计算？")
    content = bind_review_paragraphs(
        review,
        evidence,
        question="FPY如何计算？",
        verified_context=["本轮网页证据 73 条"],
    )

    assert source.evidence_id in content


def test_local_correction_preserves_verified_items_and_source_requirement():
    evidence = _previous_citation_evidence(
        recorded_citations("T13"), workspace_id="default", run_id="correction-check",
    )
    source = next(item for item in evidence if any(line.startswith("FPY ＝") for line in citation_source_lines(item)))
    lines = citation_source_lines(source)
    position = next(index for index, line in enumerate(lines, 1) if line.startswith("FPY ＝"))
    original = GroundedParagraph(text=lines[position - 1], citations=[CitationSpan(
        evidence_id=source.evidence_id, start_line=1, end_line=1,
    )])
    review = SynthesisReview(paragraphs=[GroundedParagraph(text="## 指标定义"), original],
                             covered_requirements=["r1"], gaps=[])
    correction = ParagraphCorrection(index=1, paragraph=original.model_copy(deep=True))
    correction.paragraph.citations[0].start_line = position
    correction.paragraph.citations[0].end_line = position
    fixed = apply_paragraph_corrections(review, ParagraphCorrections(corrections=[correction]), {1}, {"r1"})
    assert fixed.paragraphs[0] == review.paragraphs[0]
    assert review.paragraphs[1].citations[0].start_line == 1
    assert source.evidence_id in bind_review_paragraphs(fixed, evidence, question="FPY如何计算？")
    correction.paragraph.citations = []
    with pytest.raises(SynthesisUnavailable, match="缺少来源"):
        apply_paragraph_corrections(review, ParagraphCorrections(corrections=[correction]), {1}, {"r1"})
    correction.gaps = [MaterialGap(requirement_ids=["r1"], missing="现有材料不足以确认该计算方式", impact="无法确认结果")]
    with pytest.raises(SynthesisUnavailable, match="未明确说明"):
        apply_paragraph_corrections(review, ParagraphCorrections(corrections=[correction]), {1}, {"r1"})
    correction.paragraph.text = correction.gaps[0].missing
    with_gap = apply_paragraph_corrections(review, ParagraphCorrections(corrections=[correction]), {1}, {"r1"})
    assert with_gap.gaps[0].requirement_ids == ["r1"]
    assert with_gap.paragraphs[0] == review.paragraphs[0]


def test_correction_accepts_explicit_missing_measurement_wording() -> None:
    original = GroundedParagraph(
        text="eventExpiryDuration 以 ISO 8601 duration 表示。",
        citations=[],
    )
    review = SynthesisReview(
        paragraphs=[original],
        covered_requirements=[],
        gaps=[],
    )
    gap = MaterialGap(
        requirement_ids=["r1"],
        missing="eventExpiryDuration 的最小值 3 与最大值 365 所对应的具体时间单位",
        impact="无法确认具体时间量纲",
    )
    correction = ParagraphCorrection(
        index=0,
        paragraph=GroundedParagraph(
            text=(
                "eventExpiryDuration 可确认范围为 3–365；所取得的正文行里并没有写出"
                "具体时间单位。"
            ),
            citations=[],
        ),
        gaps=[gap],
    )

    fixed = apply_paragraph_corrections(
        review,
        ParagraphCorrections(corrections=[correction]),
        {0},
        {"r1"},
    )

    assert fixed.gaps == [gap]


def test_actual_review_distinguishes_section_identity_from_factual_values():
    path = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-fix-20260922/C21.3-replay-spans.json"
    if not path.exists():
        pytest.skip("本地真实模型审核记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    review = SynthesisReview.model_validate(record["review"])
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    assert bind_review_paragraphs(review, evidence, question=record["attempts"][0]["question"])
    paragraph = next(item for item in review.paragraphs if "3.1 节" in item.text)
    paragraph.text = paragraph.text.replace("3.1 节", "999.9 节")
    with pytest.raises(ParagraphBindingError, match="未包含陈述数值"):
        bind_review_paragraphs(review, evidence, question=record["attempts"][0]["question"])


def test_actual_nested_calculation_is_rejected_at_its_exact_location():
    directory = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-dbb73103e00c4298b1c2c5bc80095e75"
    )
    if not directory.exists():
        pytest.skip("本地真实浏览器审核记录未提供")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("review-*.json")]
    record = next(item for item in records if "extra_forbidden" in item["error"])
    with pytest.raises(ValidationError) as caught:
        ParagraphSynthesisReview.model_validate_json(record["raw"])
    assert {error["loc"][-1] for error in caught.value.errors()} == {"calculation", "inputs", "result"}
    assert all(error["loc"][:4] == ("paragraphs", 10, "citations", 0) for error in caught.value.errors())
    schema = ParagraphSynthesisReview.model_json_schema()
    assert set(schema["properties"]) == {"paragraphs", "covered_requirements", "gaps"}
    assert set(schema["$defs"]["CitationSpan"]["properties"]) == {"evidence_id", "start_line", "end_line"}
    assert {"calculation", "inputs", "result"} <= set(schema["$defs"]["GroundedParagraph"]["properties"])


def test_semantic_input_keeps_user_context_and_excludes_assistant_history():
    directory = Path(__file__).resolve().parents[2] / ".tmp_probe"
    source_path = directory / "rag-fix-20260922/C21.4-replay-spans.json"
    history_path = directory / "rag-repaired-r3-20260922/C21.2-evidence.json"
    if not source_path.exists() or not history_path.exists():
        pytest.skip("本地真实对话记录未提供")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    previous = json.loads(history_path.read_text(encoding="utf-8-sig"))
    assistant_text = previous["answers"][0]["content"]
    evidence = [Evidence.model_validate(item) for item in source["evidence"]]
    review = SynthesisReview.model_validate(source["review"])
    request = SynthesisRequest(question=source["attempts"][0]["question"], evidence=evidence,
                               requirements=source["attempts"][0]["requirements"])
    user_question = previous["browser"]["question"]
    prompt = _semantic_assessment_prompt(request, review, evidence, [
        {"role": "user", "content": user_question}, {"role": "assistant", "content": assistant_text},
    ])
    assert user_question in prompt
    assert assistant_text not in prompt
    assert all(span.evidence_id in prompt for item in review.paragraphs for span in item.citations)


def test_actual_semantic_assessment_keeps_body_and_updates_coverage():
    path = Path(__file__).resolve().parents[2] / ".tmp_probe/rag-fix-20260922/R21.3-semantic-scope.json"
    if not path.exists():
        pytest.skip("本地真实语义验收记录未提供")
    record = json.loads(path.read_text(encoding="utf-8"))
    artifact = record["input_artifact"]
    review = SynthesisReview.model_validate(artifact["review"])
    previous_body = [item.model_dump() for item in review.paragraphs]
    assessment = SemanticAssessment.model_validate(record["assessment"])
    issues = apply_semantic_assessment(review, assessment, set(artifact["requirements"]))
    assert set(issues) == {11, 15}
    assert not review.gaps
    assert set(review.covered_requirements) == {"R1", "R2"}
    assert [item.model_dump() for item in review.paragraphs] == previous_body


@pytest.mark.asyncio
async def test_model_call_timing_cancels_actual_async_task():
    timing = ModelCallTiming()
    started = asyncio.Event()
    released = asyncio.Event()

    async def operation():
        timing.start()
        timing.attach()
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()
            timing.finish("cancelled")

    task = asyncio.create_task(operation())
    await started.wait()
    timing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert released.is_set()
    assert timing.finished_event.is_set()
    assert timing.snapshot()["outcome"] == "cancelled"


def test_actual_mechanical_then_semantic_failures_have_separate_corrections():
    directory = Path.home() / (
        ".everything-rag/workspaces/e767925c5a795840a3b284728dfa5873/artifacts/synthesis/"
        "run-ea4211dd473b420fab232b8d6ab84988"
    )
    if not directory.exists():
        pytest.skip("本地真实分阶段失败记录未提供")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("review-*.json")]
    failures = sorted((item for item in records if item.get("accepted") is False), key=lambda item: item["attempt"])
    assert [item["stage"] for item in failures] == ["mechanical", "semantic"]
    budget = SynthesisCorrectionBudget()
    for failure in failures:
        semantic = failure["stage"] == "semantic"
        if semantic:
            budget.begin_semantic()
        assert budget.reserve_correction(semantic=semantic)
    budget.begin_semantic()
    assert not budget.reserve_correction(semantic=False)
    assert budget.reserve_correction(semantic=True)
    budget.begin_semantic()
    assert not budget.reserve_correction(semantic=True)
    with pytest.raises(SynthesisUnavailable, match="三次上限"):
        budget.begin_semantic()
