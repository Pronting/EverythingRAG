"""不可变 Agent 策略、查询路由、六种答案依据与输出边界测试。"""

from __future__ import annotations

import pytest

from app.generation.chat_service import _build_messages
from app.generation.knowledge_catalog import KnowledgeOverview
from app.generation.prompt_policy import (
    IMMUTABLE_CORE_POLICY,
    POLICY_VERSION,
    AnswerBasis,
    AnswerPreferences,
    OutputGuard,
    QueryMode,
    build_system_prompt,
    choose_answer_basis,
    classify_query,
)


@pytest.mark.parametrize("mode", list(QueryMode))
def test_custom_instructions_reach_model_messages_without_replacing_core(mode: QueryMode) -> None:
    custom = "以产品设计师的视角回答，先给结论。"
    prefs = AnswerPreferences(custom_instructions=custom, knowledge_only=True)
    messages = _build_messages(
        message="请分析", chunks=[], web_results=[], mode=mode,
        basis=choose_answer_basis(mode, knowledge_only=True), preferences=prefs,
        knowledge_overview=KnowledgeOverview(),
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(IMMUTABLE_CORE_POLICY)
    assert custom in messages[0]["content"]
    assert "仅使用本地知识内容" in messages[0]["content"]
    assert "以上述规则为准" in messages[0]["content"]
    assert build_system_prompt(AnswerPreferences(custom_instructions="  \n")) == build_system_prompt()


@pytest.mark.parametrize(
    ("mode", "knowledge", "web", "knowledge_only", "expected"),
    [
        (QueryMode.CONTENT_QA, True, False, False, AnswerBasis.KNOWLEDGE),
        (QueryMode.CONTENT_QA, True, True, False, AnswerBasis.WEB),
        (QueryMode.CONTENT_QA, False, False, False, AnswerBasis.GENERAL),
        (QueryMode.PRODUCT_HELP, False, False, False, AnswerBasis.PRODUCT),
        (QueryMode.KNOWLEDGE_OVERVIEW, False, False, False, AnswerBasis.CATALOG),
        (QueryMode.CONTENT_QA, False, False, True, AnswerBasis.INSUFFICIENT),
    ],
)
def test_all_six_answer_bases(
    mode: QueryMode,
    knowledge: bool,
    web: bool,
    knowledge_only: bool,
    expected: AnswerBasis,
) -> None:
    assert choose_answer_basis(
        mode,
        has_knowledge=knowledge,
        has_web=web,
        knowledge_only=knowledge_only,
    ) is expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("知识库里有什么内容？", QueryMode.KNOWLEDGE_OVERVIEW),
        ("介绍一下知识库整体情况", QueryMode.KNOWLEDGE_OVERVIEW),
        ("能给我看看文档库的内容分布吗？", QueryMode.KNOWLEDGE_OVERVIEW),
        ("帮我总结一下导入的文档", QueryMode.KNOWLEDGE_OVERVIEW),
        ("概括所有资料的整体情况", QueryMode.KNOWLEDGE_OVERVIEW),
        ("Everything RAG 是做什么的？", QueryMode.PRODUCT_HELP),
        ("everything rag 到底是干嘛的？", QueryMode.PRODUCT_HELP),
        ("介绍一下当前项目的整体情况", QueryMode.PRODUCT_HELP),
        ("购物车为什么需要分表？", QueryMode.CONTENT_QA),
        ("总结《行为风控数据分析》", QueryMode.CONTENT_QA),
    ],
)
def test_query_modes(question: str, expected: QueryMode) -> None:
    assert classify_query(question) is expected


def test_preferences_only_append_whitelisted_expression_rules() -> None:
    prompt = build_system_prompt(
        AnswerPreferences(
            language="en",
            verbosity="detailed",
            tone="professional",
            response_format="bullets",
            knowledge_only=True,
        )
    )
    assert prompt.startswith(IMMUTABLE_CORE_POLICY)
    assert "Everything RAG" in prompt
    assert "使用英文" in prompt
    assert "仅使用本地知识内容" in prompt
    assert POLICY_VERSION in prompt
    assert "版本号、阈值和单位必须原样保全" in prompt


def test_output_guard_handles_cross_token_citations_and_leading_artifact() -> None:
    guard = OutputGuard({"S1", "W1"})
    pieces = [
        guard.feed("，有效 [S"),
        guard.feed("1]，联网 [W1]，伪造 [S"),
        guard.feed("2]，普通方括号 [说明]。"),
        guard.flush(),
    ]
    assert "".join(pieces) == "有效 [S1]，联网 [W1]，伪造 ，普通方括号 [说明]。"


def test_output_guard_does_not_delete_body_phrases() -> None:
    text = "以下为通用知识，非你的资料库内容：正文仍应完整。"
    guard = OutputGuard()
    assert guard.feed(text) + guard.flush() == text
