from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt
from pydantic_core import from_json

from app.agent.contracts import Evidence
from app.agent.verification import (
    GroundedParagraph,
    ParagraphBindingError,
    SynthesisReview,
    assign_citation_ids,
    bind_review_paragraphs,
)


@dataclass
class AnswerStream:
    answer_id: str = ""
    content: str = ""
    index: int = 0
    paragraph_count: int = 0
    enabled: bool = True
    evidence: list[Evidence] = field(default_factory=list)
    question: str = ""
    verified_context: list[str] = field(default_factory=list)

    def update(self, stream_id: str, stage: str, raw: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if stage == "synthesis_review":
            source = raw.lstrip()
            if not source.startswith("{"):
                tokens = MarkdownIt().parse(source)
                if len(tokens) != 1 or tokens[0].type != "fence":
                    return None
                source = tokens[0].content.lstrip()
                if not source.startswith("{"):
                    return None
            try:
                document = from_json(source)
            except ValueError:
                document = from_json(source, allow_partial=True)
                # 下一项开始后，前面的完整正文项才可以发布。
                paragraphs = document.get("paragraphs", [])[:-1]
            else:
                paragraphs = document.get("paragraphs", [])
            if not paragraphs:
                return None
            review = SynthesisReview(
                paragraphs=[GroundedParagraph.model_validate(item) for item in paragraphs],
                covered_requirements=[], gaps=[],
            )
            try:
                content = bind_review_paragraphs(
                    review, self.evidence, question=self.question,
                    verified_context=self.verified_context,
                )
            except ParagraphBindingError as exc:
                # 尚未通过引用检查的内容留给完整响应的纠正流程。
                review.paragraphs = review.paragraphs[:min(exc.issues)]
                if not review.paragraphs:
                    return None
                content = bind_review_paragraphs(
                    review, self.evidence, question=self.question,
                    verified_context=self.verified_context,
                )
            self.paragraph_count = len(review.paragraphs)
            content = assign_citation_ids(content, self.evidence).content.strip()
        elif stage in {"general_research", "synthesis_generation"}:
            content = raw.strip()
        else:
            return None
        if not content or content == self.content:
            return None
        answer_id = f"stream-{stream_id}"
        if self.answer_id and self.answer_id != answer_id:
            raise ValueError("正文已发布，不能启动另一份正文替换当前回答")
        if not content.startswith(self.content):
            raise ValueError("正文已发布，后续内容只能继续追加")
        payload = {
            "answer_id": answer_id, "index": self.index,
            "text": content[len(self.content):], "draft": False, "replace": False,
        }
        self.answer_id = answer_id
        self.content = content
        self.index += 1
        return payload
