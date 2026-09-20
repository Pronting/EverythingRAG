"""`RouteDecision` 契约、模型输出解析与 Intent Agent 合并测试（PRD 6.2 / 4.2）。

覆盖：12 字段契约与默认值、枚举取值、模型输出严格校验（未知字段只进诊断、非法取值
抛 `RouteParseError`）、规则层对模型结论的硬约束合并、`route.decided` 事件按冻结模型
构造（多字段 / 少字段都抛协议错误）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.contracts import Freshness, IntentKind, RouteDecision
from app.agent.events import (
    EVENT_PAYLOAD_MODELS,
    EventProtocolError,
    RouteDecidedPayload,
    parse_event,
)
from app.agent.routing import (
    IntentAgent,
    RouteParseError,
    RoutingInput,
    build_route_decided_event,
    parse_route_decision,
    route_decided_payload,
    route_question,
)
from app.generation.knowledge_catalog import build_knowledge_catalog
from tests.fakes_agent import FakeChatProvider

EXPECTED_FIELDS = {
    "question",
    "resolved_question",
    "intent",
    "needs_knowledge",
    "needs_web",
    "needs_history",
    "needs_general",
    "freshness",
    "entities",
    "scope_evidence_ids",
    "unresolved_terms",
    "reason_summary",
}


def _catalog():
    blocks = [
        (
            "b1",
            "内部风控 P99 方案：P99 表示第 99 百分位延迟。",
            {
                "source_file": r"C:\notes\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "heading_path": "风控 > P99 方案",
            },
        )
    ]
    return build_knowledge_catalog(blocks, workspace_id="ws-1", knowledge_revision=1)


# ---------------------------------------------------------------------
# 契约层
# ---------------------------------------------------------------------


def test_route_decision_has_prd_6_2_fields_and_defaults() -> None:
    assert set(RouteDecision.model_fields) == EXPECTED_FIELDS
    assert len(RouteDecision.model_fields) == 12
    required = {
        name for name, field in RouteDecision.model_fields.items() if field.is_required()
    }
    assert required == {"question", "resolved_question", "intent"}

    decision = RouteDecision(question="q", resolved_question="q", intent=IntentKind.GENERAL)
    assert decision.needs_knowledge is False
    assert decision.needs_web is False
    assert decision.needs_history is False
    assert decision.needs_general is False
    assert decision.freshness is Freshness.NONE
    assert decision.entities == []
    assert decision.scope_evidence_ids == []
    assert decision.unresolved_terms == []
    assert decision.reason_summary == ""


def test_route_decision_accepts_every_enum_value() -> None:
    for intent in IntentKind:
        decision = RouteDecision(question="q", resolved_question="q", intent=intent)
        assert decision.intent is intent
    for freshness in Freshness:
        decision = RouteDecision(
            question="q", resolved_question="q", intent=IntentKind.RESEARCH, freshness=freshness
        )
        assert decision.freshness is freshness


def test_route_decision_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {"question": "q", "resolved_question": "q", "intent": "general", "confidence": 0.9}
        )


# ---------------------------------------------------------------------
# 模型输出解析
# ---------------------------------------------------------------------


def test_parse_route_decision_reads_json_and_fills_question() -> None:
    payload = json.dumps(
        {
            "intent": "research",
            "needs_knowledge": True,
            "needs_web": True,
            "freshness": "current",
            "entities": ["限流"],
            "reason_summary": "查资料并联网",
        }
    )
    decision, diagnostics = parse_route_decision(payload, question="原问题")
    assert decision.question == "原问题"
    assert decision.resolved_question == "原问题"
    assert decision.intent is IntentKind.RESEARCH
    assert decision.needs_knowledge is True and decision.needs_web is True
    assert decision.freshness is Freshness.CURRENT
    assert diagnostics.model_used is True
    assert diagnostics.dropped_fields == []
    assert diagnostics.confidence is None


def test_parse_route_decision_keeps_confidence_only_in_diagnostics() -> None:
    raw = {
        "question": "q",
        "resolved_question": "q",
        "intent": "general",
        "confidence": 0.87,
        "self_check": "high",
    }
    decision, diagnostics = parse_route_decision(raw)
    assert diagnostics.confidence == pytest.approx(0.87)
    assert diagnostics.dropped_fields == ["self_check"]
    assert "confidence" not in decision.model_dump()
    assert set(decision.model_dump()) == EXPECTED_FIELDS


def test_parse_route_decision_strips_code_fences() -> None:
    body = json.dumps({"intent": "catalog"}, ensure_ascii=False)
    decision, _ = parse_route_decision(f"```json\n{body}\n```")
    assert decision.intent is IntentKind.CATALOG


@pytest.mark.parametrize(
    "payload",
    [
        "不是 JSON",
        json.dumps({"needs_knowledge": True}),
        json.dumps({"intent": "knowledge"}),
        json.dumps({"intent": "research", "freshness": "yesterday"}),
        json.dumps(["research"]),
    ],
)
def test_parse_route_decision_rejects_invalid_payloads(payload: str) -> None:
    with pytest.raises(RouteParseError):
        parse_route_decision(payload, question="q")


# ---------------------------------------------------------------------
# Intent Agent
# ---------------------------------------------------------------------


def test_offline_decide_uses_rules_only() -> None:
    agent = IntentAgent(catalog=_catalog())
    result = agent.decide(RoutingInput(question="我的笔记里 P99 方案是怎么写的？", file_count=1))
    assert result.diagnostics.model_used is False
    assert result.diagnostics.confidence is None
    assert result.decision.needs_knowledge is True
    assert route_question(RoutingInput(question="今天有什么新闻？")).needs_web is True


class RecordingProvider:
    """记录 `messages` 的假 Provider（不联网、不调用真实模型）。

    `tests/fakes_agent.FakeChatProvider` 把请求当 dict 存（协议回复场景），这里需要
    逐条检查组装好的 messages，所以用这个更贴合的替身。
    """

    def __init__(self, payload: str | dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[list[dict[str, Any]]] = []

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append([dict(message) for message in messages])
        content = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return {
            "content": content,
            "reasoning_content": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _provider_response(decision: dict[str, Any]) -> RecordingProvider:
    return RecordingProvider(decision)


async def test_model_decision_is_merged_but_hard_rules_win() -> None:
    agent = IntentAgent(catalog=_catalog())
    provider = _provider_response(
        {
            "question": "q",
            "intent": "general",
            "needs_knowledge": False,
            "needs_web": False,
            "confidence": 0.9,
            "reason_summary": "模型说是通用问题",
        }
    )
    result = await agent.decide_with_model(
        RoutingInput(question="我的笔记里写了什么？", file_count=1), model=provider
    )
    # R1 是硬规则：模型说「不需要资料」也必须查资料。
    assert result.decision.needs_knowledge is True
    assert result.diagnostics.model_used is True
    assert result.diagnostics.confidence == pytest.approx(0.9)
    assert provider.calls and isinstance(provider.calls[0], list)
    # 进入模型的消息里包含职责提示与目录片段，且不含绝对路径。
    joined = json.dumps(provider.calls[0], ensure_ascii=False)
    assert "Intent Agent" in joined
    assert "C:\\\\notes" not in joined


async def test_model_cannot_add_retrieval_to_product_question() -> None:
    agent = IntentAgent()
    provider = _provider_response(
        {
            "intent": "research",
            "needs_knowledge": True,
            "needs_web": True,
            "reason_summary": "模型想检索",
        }
    )
    result = await agent.decide_with_model(
        RoutingInput(question="Everything RAG 是做什么的？", file_count=3), model=provider
    )
    assert result.decision.intent is IntentKind.PRODUCT
    assert result.decision.needs_knowledge is False
    assert result.decision.needs_web is False


async def test_model_cannot_drop_freshness_requirement() -> None:
    agent = IntentAgent()
    provider = _provider_response({"intent": "general", "needs_web": False, "freshness": "none"})
    result = await agent.decide_with_model(
        RoutingInput(question="今天有什么 AI 新闻？", file_count=0), model=provider
    )
    assert result.decision.needs_web is True
    assert result.decision.freshness is Freshness.CURRENT


async def test_invalid_model_output_falls_back_to_rules() -> None:
    agent = IntentAgent()
    provider = FakeChatProvider(
        default_response={"content": "抱歉，我无法给出结构化结论", "tool_calls": []}
    )
    result = await agent.decide_with_model(
        RoutingInput(question="我的笔记里写了什么？", file_count=1), model=provider
    )
    assert result.diagnostics.model_used is False
    assert result.decision.needs_knowledge is True
    assert any("回退规则层" in note for note in result.diagnostics.notes)


async def test_missing_model_keeps_rule_layer_result() -> None:
    agent = IntentAgent()
    result = await agent.decide_with_model(RoutingInput(question="通用问题", file_count=0))
    assert result.diagnostics.model_used is False
    assert any("未注入模型" in note for note in result.diagnostics.notes)


# ---------------------------------------------------------------------
# route.decided 事件（PRD 13.3，契约由 T00b 冻结）
# ---------------------------------------------------------------------


def test_route_decided_payload_matches_frozen_model() -> None:
    decision = RouteDecision(
        question="q",
        resolved_question="q",
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
    )
    payload = route_decided_payload(decision)
    assert set(payload) == {"decision"}
    assert set(payload["decision"]) == EXPECTED_FIELDS
    model = EVENT_PAYLOAD_MODELS[next(key for key in EVENT_PAYLOAD_MODELS if key.value == "route.decided")]
    assert model is RouteDecidedPayload
    assert RouteDecidedPayload.model_validate(payload).decision == decision


def test_build_route_decided_event_round_trips() -> None:
    decision = RouteDecision(question="q", resolved_question="q", intent=IntentKind.GENERAL)
    event = build_route_decided_event(
        run_id="run-1", event_seq=3, decision=decision, execution_id="exec-1"
    )
    assert event.type.value == "route.decided"
    assert event.run_id == "run-1"
    assert event.event_seq == 3
    parsed = parse_event(event.model_dump())
    assert parsed.payload["decision"]["intent"] == "general"


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": None},
        {},
        {"decision": {"question": "q", "resolved_question": "q", "intent": "general", "extra": 1}},
    ],
)
def test_route_decided_payload_rejects_wrong_shapes(payload: dict[str, Any]) -> None:
    with pytest.raises(EventProtocolError):
        parse_event(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "event_seq": 1,
                "type": "route.decided",
                "payload": payload,
            }
        )
