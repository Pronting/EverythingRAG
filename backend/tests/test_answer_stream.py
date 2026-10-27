import json
from pathlib import Path

import pytest

from app.agent.answer_stream import AnswerStream
from app.agent.contracts import Evidence, EvidenceSourceType, utc_now_iso
from app.agent.research import content_hash_of, evidence_id_for
from app.agent.verification import citation_source_lines


def readme_stream():
    source = Path(__file__).resolve().parents[2] / "README.zh-CN.md"
    content = source.read_text(encoding="utf-8")
    digest = content_hash_of(content)
    evidence = Evidence(
        evidence_id=evidence_id_for(EvidenceSourceType.KNOWLEDGE, str(source), digest),
        workspace_id="workspace", run_id="run", source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref=str(source), content=content, content_hash=digest,
        retrieved_at=utc_now_iso(), supported_requirements=["R1"],
    )
    lines = citation_source_lines(evidence)
    paragraphs = [
        {"text": line, "citations": [{
            "evidence_id": evidence.evidence_id, "start_line": index, "end_line": index,
        }]}
        for index, line in enumerate(lines, start=1)
        if "本地优先的 RAG" in line or "本地知识库管理" in line
    ]
    assert len(paragraphs) == 2
    document = {"paragraphs": paragraphs, "covered_requirements": ["R1"], "gaps": []}
    return AnswerStream(evidence=[evidence], question="项目的用途"), document


def test_sourced_paragraphs_are_validated_before_append() -> None:
    stream, document = readme_stream()
    raw = json.dumps(document, ensure_ascii=False)
    deltas = []
    for end in range(1, len(raw) + 1):
        delta = stream.update("call", "synthesis_review", raw[:end])
        if delta:
            deltas.append(delta)
            assert not delta["replace"]
            assert not delta["draft"]
    assert len(deltas) == 2
    assert "".join(item["text"] for item in deltas) == stream.content
    assert stream.content == "\n\n".join(
        item["text"] + " [S1]" for item in document["paragraphs"]
    ).strip()
    assert stream.update("call", "synthesis_review", raw) is None


def test_published_content_cannot_be_replaced() -> None:
    stream = AnswerStream()
    stream.update("call", "general_research", "Python")
    with pytest.raises(ValueError, match="只能继续追加"):
        stream.update("call", "general_research", "JavaScript")
    with pytest.raises(ValueError, match="不能启动另一份正文"):
        stream.update("other-call", "general_research", "Python语言")
    assert stream.content == "Python"


def test_invalid_source_waits_for_validation_before_publication() -> None:
    stream, document = readme_stream()
    document["paragraphs"][0]["citations"][0]["end_line"] = 1000000
    assert stream.update("call", "synthesis_review", json.dumps(document)) is None
    assert not stream.content
    assert stream.paragraph_count == 0


def test_invalid_suffix_preserves_published_prefix_until_corrected() -> None:
    stream, document = readme_stream()
    citation = document["paragraphs"][1]["citations"][0]
    original_end = citation["end_line"]
    citation["end_line"] = 1000000
    first = stream.update("call", "synthesis_review", json.dumps(document))
    assert first is not None
    assert stream.paragraph_count == 1
    assert document["paragraphs"][1]["text"] not in stream.content
    citation["end_line"] = original_end
    second = stream.update("call", "synthesis_review", json.dumps(document))
    assert second is not None
    assert stream.paragraph_count == 2
    assert stream.content == first["text"] + second["text"]
    assert not second["replace"]


def test_internal_review_never_emits_answer() -> None:
    stream, document = readme_stream()
    raw = json.dumps(document)
    for stage in ("semantic_review", "synthesis_correction", "general_verification"):
        assert stream.update("call", stage, raw) is None
    stream.enabled = False
    assert stream.update("call", "synthesis_review", raw) is None
