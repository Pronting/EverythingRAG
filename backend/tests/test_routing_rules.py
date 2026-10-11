"""路由规则与范围检查测试（PRD 6.1–6.3）。Owner: T06（routing）。

逐条覆盖 PRD 6.3 的九条决策规则，外加空知识库、`needs_knowledge` 与 `needs_web`
同时为真、目录线索阈值、提示只注入相关目录条目、范围检查与结论调整。
"""

from __future__ import annotations

from typing import Any

from app.agent.contracts import Freshness, IntentKind, RouteDecision
from app.agent.routing import (
    CATALOG_CLUE_MIN_SCORE,
    CLARIFICATION_MAX_QUESTION_CHARS,
    ROUTING_RULES_VERSION,
    RULE_DESCRIPTIONS,
    IntentAgent,
    RoutingHistoryTurn,
    RoutingInput,
    adjust_route_decision,
    build_route_prompt,
    check_scope,
    evaluate_route_rules,
    resolve_reference,
    route_question,
)
from app.generation.knowledge_catalog import CatalogMatch, build_knowledge_catalog

RAW_ROOT = r"C:\Users\tester\notes"


def _catalog_blocks() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "b1",
            "内部风控 P99 方案：P99 表示第 99 百分位延迟。本方案采用分层限流。",
            {
                "source_file": rf"{RAW_ROOT}\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "doc_id": "hash-1",
                "heading_path": "风控 > P99 方案",
                "anchor": "p99",
            },
        ),
        (
            "b2",
            "坑位归因用于投放复盘，但这里的定义写得并不清楚。",
            {
                "source_file": rf"{RAW_ROOT}\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "doc_id": "hash-1",
                "heading_path": "风控 > 坑位归因",
                "anchor": "attr",
            },
        ),
    ]


def _catalog():
    return build_knowledge_catalog(
        _catalog_blocks(), workspace_id="ws-1", knowledge_revision=2
    )


def _hits(query: str, *, limit: int = 6) -> list[CatalogMatch]:
    return list(_catalog().search(query, limit=limit).matches)


def _input(question: str, **kwargs: Any) -> RoutingInput:
    base: dict[str, Any] = {"question": question, "file_count": 2, "chunk_count": 2}
    base.update(kwargs)
    return RoutingInput(**base)


def _decide(question: str, **kwargs: Any) -> Any:
    return IntentAgent(catalog=_catalog()).decide(_input(question, **kwargs))


# ---------------------------------------------------------------------
# 九条规则逐条覆盖
# ---------------------------------------------------------------------


def test_rule_r1_explicit_internal_reference() -> None:
    result = _decide("我的笔记里是怎么写限流方案的？")
    assert "R1" in result.scope.rules
    assert result.decision.needs_knowledge is True
    assert result.decision.intent is IntentKind.RESEARCH


def test_official_api_lookup_does_not_add_local_research() -> None:
    result = _decide(
        "请仅查 AWS 官方文档，确认 CreateMemory 的 eventExpiryDuration 参数范围和单位"
    )

    assert result.decision.needs_web is True
    assert result.decision.needs_knowledge is False


def test_python_official_lookup_does_not_add_local_research() -> None:
    result = _decide(
        "请联网查 Python 官方文档，说明 asyncio.TaskGroup 的用途并给出来源。"
    )

    assert result.decision.needs_web is True
    assert result.decision.needs_knowledge is False


def test_rule_r1_fires_on_catalog_document_clue() -> None:
    matches = [
        CatalogMatch(
            kind="heading",
            document_id="doc-1",
            title="下单流程.md",
            heading_path="下单 > 库存预占",
            snippet="库存预占",
            score=400,
            matched_on="库存预占",
        )
    ]
    outcome = evaluate_route_rules(_input("库存预占是怎么实现的？", catalog_hits=matches))
    assert "R1" in outcome.rule_ids
    assert outcome.decision.needs_knowledge is True


def test_weak_catalog_match_does_not_trigger_internal_route() -> None:
    weak = CatalogMatch(
        kind="heading",
        document_id="doc-1",
        heading_path="别的主题",
        snippet="别的主题",
        score=CATALOG_CLUE_MIN_SCORE - 1,
        matched_on="别的主题",
    )
    outcome = evaluate_route_rules(_input("这个东西怎么做？", catalog_hits=[weak]))
    assert "R1" not in outcome.rule_ids
    assert outcome.decision.needs_knowledge is False


def test_rule_r2_follow_up_resolves_reference_from_history() -> None:
    history = [RoutingHistoryTurn(role="user", content="内部风控 P99 方案是怎么设计的？")]
    result = _decide("这个方案在高并发下会有什么问题？", recent_history=history)
    assert "R2" in result.scope.rules
    assert result.decision.needs_knowledge is True
    assert result.decision.resolved_question.startswith("内部风控 P99 方案")
    assert result.decision.resolved_question.endswith("在高并发下会有什么问题？")


def test_rule_r2_without_history_keeps_question_and_records_reason() -> None:
    resolved, note = resolve_reference("这个方案有什么问题？", [])
    assert resolved == "这个方案有什么问题？"
    assert "没有可用历史" in note
    result = _decide("这个方案有什么问题？")
    assert "R2" not in result.scope.rules
    assert any("没有可用历史" in item for item in result.diagnostics.notes)


def test_rule_r3_term_hit_with_definition_intent() -> None:
    result = _decide("P99 方案的定义是什么？", catalog_hits=_hits("P99 方案的定义是什么？"))
    assert "R3" in result.scope.rules
    assert result.decision.needs_knowledge is True
    assert result.decision.scope_evidence_ids


def test_rule_r3_requires_definition_intent_and_term_mention() -> None:
    # 术语命中但没有「定义 / 实现」意图 → 不因术语本身强行并入 R3 之外的行为。
    plain = _decide("随便聊聊进度", catalog_hits=_hits("P99 方案"))
    assert "R3" not in plain.scope.rules
    # 目录里有别的术语，但问题没提到它 → 不算本轮相关术语。
    other = _decide("P99 方案的定义是什么？", catalog_hits=_hits("坑位归因"))
    assert other.decision.unresolved_terms == []


def test_rule_r4_unknown_term_is_marked_and_short_question_clarifies() -> None:
    question = "坑位归因是啥？"
    assert len(question) <= CLARIFICATION_MAX_QUESTION_CHARS
    result = _decide(question, catalog_hits=_hits(question))
    assert "R4" in result.scope.rules
    assert result.decision.unresolved_terms == ["坑位归因"]
    assert result.decision.intent is IntentKind.CLARIFICATION
    assert result.decision.needs_knowledge is True
    assert any("缺少明确解释" in note for note in result.diagnostics.notes)
    assert "坑位归因" in result.scope.definition_unknown_terms
    assert any("仅用于定位原文" in note for note in result.scope.notes)


def test_rule_r4_long_question_does_not_clarify_even_with_unknown_term() -> None:
    question = "坑位归因这个术语在我们整套投放复盘体系里到底是怎么定义和使用的？"
    assert len(question) > CLARIFICATION_MAX_QUESTION_CHARS
    result = _decide(question, catalog_hits=_hits("坑位归因"))
    assert result.decision.unresolved_terms == ["坑位归因"]
    assert result.decision.intent is not IntentKind.CLARIFICATION


def test_rule_r5_freshness_current_and_date_range() -> None:
    current = _decide("今天 AI 领域有什么新闻？")
    assert "R5" in current.scope.rules
    assert current.decision.needs_web is True
    assert current.decision.freshness is Freshness.CURRENT

    ranged = _decide("近三个月 AI 芯片有什么新品发布？")
    assert ranged.decision.needs_web is True
    assert ranged.decision.freshness is Freshness.DATE_RANGE

    none = _decide("我们的限流方案是怎么设计的？")
    assert none.decision.freshness is Freshness.NONE


def test_rule_r6_general_explanation_together_with_internal_material() -> None:
    result = _decide("我们内部的限流方案为什么选择分层限流，一般是怎么权衡的？")
    assert "R6" in result.scope.rules
    assert result.decision.needs_general is True
    assert result.decision.needs_knowledge is True


def test_rule_r7_product_help_and_overview_do_not_retrieve_content() -> None:
    product = _decide("Everything RAG 会把我的资料上传到云端吗？")
    assert "R7" in product.scope.rules
    assert product.decision.intent is IntentKind.PRODUCT
    assert product.decision.needs_knowledge is False
    assert product.decision.needs_web is False

    overview = _decide("我的资料库一共有多少个切片？")
    assert "R7" in overview.scope.rules
    assert overview.decision.intent is IntentKind.CATALOG
    assert overview.decision.needs_knowledge is False


def test_rule_r7_precedence_beats_internal_markers() -> None:
    # 问题里含内部标记「我的资料」，但问的是产品隐私能力 → 仍走产品说明。
    result = _decide("Everything RAG 会把我的文档上传到云端吗？")
    assert result.decision.intent is IntentKind.PRODUCT
    assert result.decision.needs_knowledge is False


def test_rule_r8_recalling_earlier_discussion() -> None:
    result = _decide("还记得我们之前聊过的限流方案吗？")
    assert "R8" in result.scope.rules
    assert result.decision.intent is IntentKind.MEMORY
    assert result.decision.needs_history is True


def test_rule_r9_empty_workspace_keeps_general_and_web_paths() -> None:
    general = _decide("数据库慢查询一般要怎么排查？", file_count=0, chunk_count=0)
    assert "R9" in general.scope.rules
    assert general.decision.intent is IntentKind.GENERAL
    assert any("通用或联网问题仍可处理" in note for note in general.diagnostics.notes)

    internal = _decide("我的笔记里写了什么限流方案？", file_count=0, chunk_count=0)
    assert internal.decision.needs_knowledge is True
    assert any("需说明资料情况" in note for note in internal.diagnostics.notes)

    web = _decide("今天有什么新闻？", file_count=0, chunk_count=0)
    assert web.decision.needs_web is True


def test_knowledge_and_web_can_be_required_together() -> None:
    result = _decide("我们内部的限流方案和现在最新的业界实践相比怎么样？")
    assert result.decision.needs_knowledge is True
    assert result.decision.needs_web is True
    assert result.decision.freshness is Freshness.CURRENT


def test_rule_r1_fires_on_document_lookup_question() -> None:
    """「某份文档在哪里 / 有没有」：先真的检索一次，再决定怎么回答。"""

    result = _decide("我们团队的火星运维手册在哪里？")

    assert "R1" in result.scope.rules
    assert result.decision.needs_knowledge is True
    # 证据文案统一为「指定资料的内容问法」：文档寻址与「某篇文章里的依据」同属一类
    # （问的是知识库里某一篇具体资料的内容，必须先真的检索再回答）。
    assert any("指定资料的内容问法" in hit.evidence for hit in result.rules)


def test_rule_r1_document_lookup_needs_both_intent_and_object() -> None:
    assert _decide("怎么写一份用户手册？").decision.needs_knowledge is False


def test_rule_descriptions_cover_nine_rules() -> None:
    assert sorted(RULE_DESCRIPTIONS) == [f"R{index}" for index in range(1, 10)]
    assert ROUTING_RULES_VERSION.startswith("routing-rules-v")


# ---------------------------------------------------------------------
# 术语理解参与检索（用户实测缺陷 C01）
# ---------------------------------------------------------------------


def test_rule_r6_term_with_concept_intent_also_searches_knowledge() -> None:
    """术语 + 概念/场景意图：既用一般知识解释，也把术语作为线索去查资料。"""

    result = _decide("有什么 AI Native 的使用场景吗？")

    assert "R6" in result.scope.rules
    assert result.decision.needs_general is True
    assert result.decision.needs_knowledge is True
    assert result.decision.entities == ["AI Native"]
    assert any("只是检索线索" in note for note in result.diagnostics.notes)


def test_rule_r6_term_branch_keeps_only_the_specific_expression() -> None:
    """「AI」只是「AI Native」的前缀 → 只保留更具体的术语，不把超泛词当查询。"""

    result = _decide("你知道 AI Native 是什么意思吗？就是 AI 原生，类似云原生的意思，再去找一下。")

    assert result.decision.entities == ["AI Native"]


def test_rule_r6_term_branch_needs_concept_intent_and_non_empty_workspace() -> None:
    # 没有概念 / 场景意图时，不因为出现英文术语就强行检索资料
    plain = _decide("K8s 集群昨天扩容了")
    assert plain.decision.needs_knowledge is False
    # 空知识库时不产生知识任务（R9 口径不变）
    empty = _decide("有什么 AI Native 的使用场景吗？", file_count=0, chunk_count=0)
    assert empty.decision.needs_knowledge is False


def test_rule_r6_keeps_plain_general_questions_general() -> None:
    result = _decide("数据库慢查询一般要怎么排查？")
    assert result.decision.needs_general is True
    assert result.decision.needs_knowledge is False


# ---------------------------------------------------------------------
# 指代补全（用户实测缺陷 C04/C05）
# ---------------------------------------------------------------------


def test_phrase_pronoun_is_replaced_in_place_with_previous_object() -> None:
    history = [RoutingHistoryTurn(role="user", content="内部风控 P99 方案是怎么设计的？")]

    resolved, note = resolve_reference("这个方案在高并发下会有什么问题？", history)

    assert resolved == "内部风控 P99 方案在高并发下会有什么问题？"
    assert "指代补全" in note


def test_bare_pronoun_prepends_history_topic_without_rewriting_question() -> None:
    history = [RoutingHistoryTurn(role="user", content="如果面试要求设计一个 Agent，需要考虑哪些因素？")]
    question = "所以关于记忆和成本来说，你有什么比较好的思路吗？如果让你回答这个面试题的话。"

    resolved, note = resolve_reference(question, history)

    # 用户本轮原话一字不改，上文主题前置；不会出现「回答⋯面试题面试题」这类重复表达
    assert question in resolved
    assert resolved.startswith("面试要求设计一个 Agent")
    assert "面试题面试题" not in resolved
    assert note.startswith("指代补全")


def test_reference_completion_ignores_assistant_turns() -> None:
    history = [RoutingHistoryTurn(role="assistant", content="好的，我来解释一下这个方案")]
    resolved, _note = resolve_reference("这个方案怎么落地？", history)
    assert resolved == "这个方案怎么落地？"


def test_route_question_returns_decision() -> None:
    decision = route_question(_input("我的笔记里写了什么？"), catalog=_catalog())
    assert isinstance(decision, RouteDecision)
    assert decision.question == "我的笔记里写了什么？"
    assert decision.resolved_question


# ---------------------------------------------------------------------
# 范围检查与结论调整
# ---------------------------------------------------------------------


def test_check_scope_reports_unresolved_and_resolved_terms() -> None:
    catalog = _catalog()
    decision = RouteDecision(
        question="P99 方案和坑位归因分别是什么？",
        resolved_question="P99 方案和坑位归因分别是什么？",
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
    )
    scope = check_scope(decision, catalog=catalog)
    assert scope.knowledge_required is True
    assert scope.catalog_empty is False
    assert "P99 方案" in scope.resolved_terms
    assert "坑位归因" in scope.unresolved_terms
    assert scope.definition_unknown_terms == ["坑位归因"]
    assert scope.scope_evidence_ids
    assert any("仅用于定位原文" in note for note in scope.notes)


def test_check_scope_demotes_terms_whose_snippet_is_not_in_text() -> None:
    catalog = _catalog()
    decision = RouteDecision(
        question="P99 方案是什么？",
        resolved_question="P99 方案是什么？",
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
    )
    scope = check_scope(decision, catalog=catalog, text_lookup=lambda _id: "完全不含该片段的正文")
    assert "P99 方案" not in scope.resolved_terms
    assert "P99 方案" in scope.unresolved_terms
    assert any("未在原文复现" in note for note in scope.notes)


def test_check_scope_without_catalog_uses_input_signals() -> None:
    decision = RouteDecision(
        question="坑位归因是什么意思？",
        resolved_question="坑位归因是什么意思？",
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
    )
    scope = check_scope(decision, input=_input("坑位归因是什么意思？", catalog_hits=_hits("坑位归因")))
    assert scope.scope_evidence_ids
    assert "坑位归因" in scope.unresolved_terms


def test_adjust_route_decision_only_changes_given_fields() -> None:
    original = RouteDecision(
        question="问题",
        resolved_question="问题",
        intent=IntentKind.RESEARCH,
        needs_knowledge=True,
        freshness=Freshness.NONE,
    )
    updated = adjust_route_decision(original, needs_web=True, freshness=Freshness.CURRENT)
    assert updated is not original
    assert updated.needs_web is True
    assert updated.freshness is Freshness.CURRENT
    assert updated.needs_knowledge is True
    assert original.needs_web is False and original.freshness is Freshness.NONE
    assert updated.question == original.question


def test_adjust_route_decision_keeps_resolved_question_non_empty() -> None:
    original = RouteDecision(question="问题", resolved_question="", intent=IntentKind.GENERAL)
    try:
        updated = adjust_route_decision(original, needs_general=True)
    except Exception:  # noqa: BLE001 - 契约本身不允许空 resolved_question
        return
    assert updated.resolved_question == "问题"


# ---------------------------------------------------------------------
# 提示组装（PRD 6.1 + 9.3）
# ---------------------------------------------------------------------


def test_build_route_prompt_follows_prd_input_order_and_injects_only_relevant() -> None:
    catalog = _catalog()
    prompt = build_route_prompt(
        RoutingInput(
            question="P99 方案是怎么设计的？",
            now="2026-09-19T10:00:00+08:00",
            timezone="Asia/Shanghai",
            recent_history=[RoutingHistoryTurn(role="user", content="先看风控")],
            memory_snapshot="用户偏好：先给结论。",
            workspace_overview='{"topics":["风控"]}',
            workspace_id="ws-1",
            file_count=2,
            chunk_count=2,
        ),
        catalog=catalog,
    )
    roles = [message["role"] for message in prompt.messages]
    assert roles[0] == "system"
    joined = "\n".join(str(message["content"]) for message in prompt.messages)
    assert "Intent Agent" in joined
    assert "用户偏好：先给结论。" in joined
    assert "先看风控" in joined
    assert "2026-09-19T10:00:00+08:00" in joined
    assert "Asia/Shanghai" in joined
    assert "知识版本 0" in joined
    assert "P99" in joined
    assert prompt.document_ids
    assert prompt.embedded_chars > 0
    assert RAW_ROOT not in joined
    assert "C:\\" not in joined


def test_build_route_prompt_without_catalog_says_no_hits() -> None:
    prompt = build_route_prompt(RoutingInput(question="今天有什么新闻？", file_count=0))
    joined = "\n".join(str(message["content"]) for message in prompt.messages)
    assert "目录未命中" in joined
    assert prompt.document_ids == []


def test_intent_agent_enriches_input_from_catalog() -> None:
    agent = IntentAgent(catalog=_catalog())
    enriched = agent.enrich(RoutingInput(question="P99 方案是怎么设计的？", file_count=2))
    assert enriched.catalog_hits
    assert enriched.catalog_hits[0].document_id


# ---------------------------------------------------------------------
# C21：泛指文档指向 -> 本地资料研究（不再 general-only）
# ---------------------------------------------------------------------


DOC_SCOPE_QUESTION = "关于文档中，他们是怎么设计agent的？应该有好几个维度吧"


def test_generic_document_question_requires_local_knowledge() -> None:
    result = _decide(DOC_SCOPE_QUESTION)

    assert "R1" in result.scope.rules
    assert result.decision.needs_knowledge is True
    assert result.decision.needs_general is False, "明确问文档内容时不能只用通用解释顶替"
    assert any("泛指文档" in hit.evidence for hit in result.rules)


def test_generic_document_question_without_article_name_searches_many_docs() -> None:
    """没有唯一文章名：不得要求用户先给出文章，也不得单独检索「文档中」。"""

    result = _decide(DOC_SCOPE_QUESTION)
    assert result.decision.needs_knowledge is True
    # 证据文案是「泛指文档」，不把用户当成已经点名了某篇文章
    r1_evidence = [hit.evidence for hit in result.rules if hit.rule_id == "R1"]
    assert r1_evidence and all("泛指文档" in item for item in r1_evidence)


def test_action_on_document_does_not_trigger_document_research() -> None:
    result = _decide("帮我把这段总结写进文档中")

    assert result.decision.needs_knowledge is False


def test_follow_up_keeps_agent_object_when_previous_turn_is_generic_scope() -> None:
    history = [
        RoutingHistoryTurn(role="user", content="rag 有需要什么值得了解的知识吗"),
        RoutingHistoryTurn(role="user", content="那agent有什么特别值得了解的知识吗"),
        RoutingHistoryTurn(role="user", content=DOC_SCOPE_QUESTION),
    ]

    resolved, note = resolve_reference("比如记忆，成本这些维度。都是怎么样设计的？", history)

    # 研究对象来自更早一轮，而不是上一轮的「文档中」
    assert "agent" in resolved.casefold()
    assert "文档中" not in resolved
    assert note.startswith("指代补全")


def test_follow_up_without_reference_marker_keeps_object_from_history() -> None:
    history = [RoutingHistoryTurn(role="user", content="关于文档中，他们是怎么设计agent的？")]

    resolved, _note = resolve_reference("比如记忆，成本这些维度。都是怎么样设计的？", history)

    assert "agent" in resolved.casefold()
