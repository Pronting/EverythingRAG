import json
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.contracts import Evidence


class EvidenceSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    requirement_id: str
    quote_field: Literal["content", "document_title", "source_ref"] = "content"
    quote: str = Field(min_length=8)
    explanation: str = Field(min_length=1)


class EvidenceSupportReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: list[EvidenceSupport]


def evidence_support_instructions() -> str:
    return (
        "依据原文含义确认哪些资料能回答对应要求，问题和原文可能使用不同语言。"
        "连续追问使用resolved_question和history确认对象，区分功能引入版本与文档版本。"
        "仅有对象名称或相关话题不算支持；原文必须给出对应事实、机制或要求的完整数值条件。"
        "区分参数、阶段、版本、范围及来源限定，不能用其他参数的数值补充答案。"
        "对确认的对应关系返回逐字连续摘录quote、摘录所在quote_field和简短explanation。"
        "quote_field为content、document_title或source_ref；功能事实必须选择content，"
        "标题和URL只能支持文档版本及来源地址。无法确认的关系不列入supported。"
        "只输出符合下面JSON Schema的对象："
        + json.dumps(EvidenceSupportReview.model_json_schema(), ensure_ascii=False)
    )


def apply_evidence_support(
    review: EvidenceSupportReview,
    evidence: Sequence[Evidence],
    requirements: Mapping[str, str],
) -> None:
    sources = {item.evidence_id: item for item in evidence}
    for match in review.supported:
        source = sources.get(match.evidence_id)
        if source is None:
            raise ValueError(f"资料确认返回未知evidence_id={match.evidence_id}；允许值为{list(sources)!r}")
        if match.requirement_id not in requirements:
            raise ValueError(f"资料确认返回未知requirement_id={match.requirement_id}；允许值为{list(requirements)!r}")
        if match.quote not in getattr(source, match.quote_field):
            raise ValueError(
                f"资料{match.evidence_id}的摘录{match.quote!r}未完整出现在{match.quote_field}字段。"
                "请逐字选择所声明字段内的连续文字，标题和地址只能支持来源与文档版本。"
            )
    for match in review.supported:
        for source in evidence:
            if source.evidence_id == match.evidence_id and match.requirement_id not in source.supported_requirements:
                source.supported_requirements.append(match.requirement_id)
