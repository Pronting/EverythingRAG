"""查询规划测试（用户实测缺陷 C01–C05 的回归线）。

覆盖：同义补充合并、独立主题保留、比较参照带上主题、上文主题带进检索、术语扩展只作为
检索线索、模型输出校验与回落。全程离线（无模型、无检索、无出网）。
"""

from __future__ import annotations

import json

import pytest

from app.agent.query_plan import (
    MAX_QUERIES,
    QueryPlannerAgent,
    _model_queries_are_facets,
    canonical_query,
    clause_is_standalone,
    compose_research_query,
    content_terms,
    generic_document_content_ask,
    generic_document_scope,
    has_research_object,
    history_topic,
    is_referential,
    merge_queries,
    merge_semantic_queries,
    model_plan_needed,
    parse_query_plan,
    stable_query_identity,
    topic_phrase,
)

#: C02：同一个问题被切成两条重复且不完整的查询。
INTERVIEW_QUESTION = (
    "有一个经典面试题：如果让你设计一个 Agent，你会考虑哪些因素？有哪些维度值得考虑？"
)
#: C03：比较参照与目标问题失去关联。
COMPARE_QUESTION = "货拉拉的记忆系统有淘汰策略吗？类似 OpenClaw 的那种方案。"
#: C04：记忆与成本的追问（主题在上一轮）。
FOLLOW_UP_QUESTION = "所以关于记忆和成本来说，你有什么比较好的思路吗？如果让你回答这个面试题的话。"
FOLLOW_UP_HISTORY = [
    {"role": "user", "content": "如果面试要求设计一个 Agent，需要考虑哪些因素？"},
    {"role": "assistant", "content": "（上一轮的回答）"},
]


# =====================================================================
# 1. 句子判据
# =====================================================================


def test_supplement_clause_without_object_is_not_standalone() -> None:
    assert clause_is_standalone("有哪些维度值得考虑") is False
    assert clause_is_standalone("我们资料里怎么做的") is False
    assert clause_is_standalone("如果让你回答这个面试题的话") is False


def test_clause_with_own_object_is_standalone() -> None:
    assert clause_is_standalone("货拉拉的记忆系统有淘汰策略吗") is True
    assert clause_is_standalone("如果让你设计一个 Agent，你会考虑哪些因素") is True
    assert clause_is_standalone("腾讯云的向量库怎么选型") is True


def test_referential_detection() -> None:
    assert is_referential("类似 OpenClaw 的那种方案") is True
    assert is_referential("也就是说，记忆过期之后怎么清理") is True
    assert is_referential("货拉拉的记忆系统有淘汰策略吗") is False


def test_content_terms_drop_question_words() -> None:
    terms = content_terms("有哪些维度值得考虑")

    assert "哪些" not in terms and "值得" not in terms and "考虑" not in terms
    assert "agent" in content_terms("设计一个 Agent 需要什么")


# =====================================================================
# 2. 合并与查询生成（C02 / C03）
# =====================================================================


def test_interview_question_is_merged_into_one_query() -> None:
    queries, notes = merge_queries(INTERVIEW_QUESTION)

    assert len(queries) == 1
    assert queries[0].startswith("有一个经典面试题")
    assert "有哪些维度值得考虑" in queries[0]
    assert any("合并" in note for note in notes)


def test_comparison_reference_query_keeps_both_systems_and_topic() -> None:
    queries, _notes = merge_queries(COMPARE_QUESTION)

    assert len(queries) == 1
    assert "货拉拉的记忆系统" in queries[0]
    assert "淘汰策略" in queries[0]
    assert "OpenClaw" in queries[0]


def test_independent_topics_still_get_separate_queries() -> None:
    queries, notes = merge_queries("货拉拉的记忆淘汰策略是什么？腾讯云的向量库怎么选型？")

    assert len(queries) == 2
    assert queries[0].startswith("货拉拉的记忆淘汰策略")
    assert queries[1].startswith("腾讯云的向量库")
    assert any("分别检索" in note for note in notes)


def test_queries_never_exceed_cap() -> None:
    question = "甲系统的方案是什么？乙系统的策略是什么？丙系统的机制是什么？丁系统的流程是什么？"

    queries, _notes = merge_queries(question)

    assert len(queries) <= MAX_QUERIES


# =====================================================================
# 3. 多轮上下文（C04 / C05）
# =====================================================================


def test_history_topic_strips_lead_and_question_tail() -> None:
    assert topic_phrase("如果面试要求设计一个 Agent，需要考虑哪些因素？") == "面试要求设计一个 Agent"
    assert topic_phrase("货拉拉的记忆系统有淘汰策略吗？") == "货拉拉的记忆系统有淘汰策略"
    assert history_topic(FOLLOW_UP_HISTORY) == "面试要求设计一个 Agent"


def test_history_topic_ignores_assistant_turns() -> None:
    assert history_topic([{"role": "assistant", "content": "好的，我来说明一下"}]) == ""


def test_follow_up_carries_history_topic_into_queries() -> None:
    plan = QueryPlannerAgent().decide(FOLLOW_UP_QUESTION, history=FOLLOW_UP_HISTORY)

    assert plan.queries[0].startswith("所以关于记忆和成本")
    assert any("面试要求设计一个 Agent" in query for query in plan.queries)
    assert any("上文主题" in note for note in plan.notes)


def test_follow_up_without_reference_marker_does_not_add_context_query() -> None:
    plan = QueryPlannerAgent().decide(
        "今天天气怎么样", history=[{"role": "user", "content": "货拉拉的记忆系统有淘汰策略吗？"}]
    )

    assert plan.queries == ["今天天气怎么样"]


# =====================================================================
# 4. 模型整理（术语扩展只是检索线索）
# =====================================================================


def test_model_plan_needed_only_for_multi_clause_or_term_or_follow_up() -> None:
    assert model_plan_needed("有什么 AI Native 的使用场景吗？") is True
    assert model_plan_needed("请解释一下什么是 AI Native。") is True
    assert model_plan_needed(INTERVIEW_QUESTION) is True
    assert model_plan_needed("货拉拉的记忆系统有淘汰策略吗？", history=FOLLOW_UP_HISTORY) is False
    assert model_plan_needed("报销制度是什么", history=FOLLOW_UP_HISTORY) is False


def test_parse_query_plan_keeps_queries_tied_to_the_question() -> None:
    plan = parse_query_plan(
        json.dumps(
            {
                "interpretation": "AI Native 指 AI 原生",
                "terms": ["AI 原生", "云原生"],
                "queries": ["AI Native 的使用场景", "完全无关的另一个话题：红烧肉怎么做"],
            },
            ensure_ascii=False,
        ),
        question="有什么 AI Native 的使用场景吗？",
        terms_hint=[],
    )

    assert plan.queries == ["AI Native 的使用场景"]
    assert plan.terms == ["AI 原生", "云原生"]
    assert plan.interpretation


def test_parse_query_plan_rejects_empty_and_overlong() -> None:
    with pytest.raises(ValueError):
        parse_query_plan({"queries": []}, question="AI Native 是什么")
    with pytest.raises(ValueError):
        parse_query_plan({"queries": ["AI " + "很长" * 80]}, question="AI Native 是什么")
    with pytest.raises(TypeError):
        parse_query_plan(123, question="AI Native 是什么")


async def test_planner_keeps_original_question_first_and_appends_model_queries() -> None:
    calls: list[str] = []

    def fake_model(duty: str, prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "interpretation": "AI 原生",
                "terms": ["AI 原生"],
                # 第一条与用户原话只差标点 -> 必须合并，不再多检索一次
                "queries": ["有什么 AI Native 的使用场景吗", "AI 原生 落地场景"],
            },
            ensure_ascii=False,
        )

    plan = await QueryPlannerAgent(model=fake_model).plan("有什么 AI Native 的使用场景吗？")

    assert calls, "含术语的概念问题必须请模型整理查询"
    assert plan.queries[0] == "有什么 AI Native 的使用场景吗"
    assert "AI 原生 落地场景" in plan.queries
    assert len(plan.queries) == 2, plan.queries
    assert plan.source == "model"


async def test_planner_falls_back_when_model_raises_or_returns_garbage() -> None:
    def broken(duty: str, prompt: str) -> str:
        raise RuntimeError("provider down")

    failed = await QueryPlannerAgent(model=broken).plan(INTERVIEW_QUESTION)
    assert failed.source == "rules"
    assert failed.queries and "有哪些维度值得考虑" in failed.queries[0]
    assert any("模型不可用" in note for note in failed.notes)

    garbage = await QueryPlannerAgent(model=lambda duty, prompt: "不是 JSON").plan(INTERVIEW_QUESTION)
    assert garbage.source == "rules"
    assert garbage.queries


async def test_planner_without_model_is_deterministic() -> None:
    plan = await QueryPlannerAgent(model=None).plan(COMPARE_QUESTION)

    assert plan.source == "rules"
    assert len(plan.queries) == 1


# =====================================================================
# 5. C21：探索式连续追问（泛指文档指向 + 研究对象承接 + 统一去重）
# =====================================================================


#: C21.3：明确提到「文档中」，但没有给出唯一文章名。
DOC_SCOPE_QUESTION = "关于文档中，他们是怎么设计agent的？应该有好几个维度吧"
#: C21.4：省略了 Agent 对象，只留关注维度。
DIMENSION_FOLLOW_UP = "比如记忆，成本这些维度。都是怎么样设计的？"
C21_HISTORY = [
    {"role": "user", "content": "rag 有需要什么值得了解的知识吗"},
    {"role": "assistant", "content": "（通用回答）"},
    {"role": "user", "content": "那agent有什么特别值得了解的知识吗"},
    {"role": "assistant", "content": "（通用回答）"},
    {"role": "user", "content": DOC_SCOPE_QUESTION},
]


def test_generic_document_scope_is_recognized_without_article_name() -> None:
    assert generic_document_scope("关于文档中，他们是怎么设计agent的？") == "文档中"
    assert generic_document_scope("资料里有没有写淘汰策略？") == "资料里"
    # 复合名词 / 动作不算泛指文档指向
    assert generic_document_scope("文档中台怎么接入？") == ""
    assert generic_document_scope("把结果写进文档里") == "文档里"


def test_generic_document_content_ask_needs_scope_and_intent(question: str = "") -> None:
    assert generic_document_content_ask(DOC_SCOPE_QUESTION) == "是怎么"
    assert generic_document_content_ask("文档里有哪些记忆设计的做法？")
    # 只有空间指向、没有内容意图：不是「问文档内容」
    assert generic_document_content_ask("把结果写进文档中") == ""
    # 只有内容意图、没有文档指向：属于外部问题
    assert generic_document_content_ask("他们是怎么设计agent的？") == ""


def test_topic_phrase_keeps_agent_object_from_document_question() -> None:
    # C21.3 的旧实现会把主题截成「文档中」，Agent 对象丢失
    assert topic_phrase(DOC_SCOPE_QUESTION) == "设计 agent"
    assert topic_phrase("文档中他们是怎么设计agent的") == "设计 agent"
    # 名词短语型主题保持既有行为
    assert topic_phrase("货拉拉的记忆系统有淘汰策略吗？") == "货拉拉的记忆系统有淘汰策略"


def test_history_topic_prefers_turn_with_research_object() -> None:
    assert history_topic(C21_HISTORY) == "设计 agent"
    assert has_research_object("设计 agent") is True
    assert has_research_object("文档中") is False
    assert has_research_object("比如记忆") is False


def test_follow_up_merges_topic_into_query_instead_of_bare_topic() -> None:
    plan = QueryPlannerAgent().decide(DIMENSION_FOLLOW_UP, history=C21_HISTORY)

    joined = " ".join(plan.queries)
    assert "记忆" in joined and "成本" in joined
    assert "agent" in joined.casefold(), plan.queries
    # 「设计 agent」不再作为独立查询（旧实现会把历史主题单列一条）
    assert "设计 agent" not in plan.queries, plan.queries
    assert len(plan.queries) == 1, plan.queries
    assert any("上文主题" in note for note in plan.notes)


def test_document_question_gets_local_document_query_with_object() -> None:
    plan = QueryPlannerAgent().decide(DOC_SCOPE_QUESTION, history=C21_HISTORY[:3])

    assert len(plan.queries) == 1, plan.queries
    assert "agent" in plan.queries[0].casefold()
    # 泛指「文档中」被当作来源范围，不再单独成查询
    assert "文档中" not in plan.queries


def test_compose_research_query_keeps_original_wording_and_topic() -> None:
    merged = compose_research_query("设计 agent", DIMENSION_FOLLOW_UP)

    assert DIMENSION_FOLLOW_UP.rstrip("？?") in merged
    assert "agent" in merged.casefold()
    # 片段式追问（对象被省略）时，上文对象前置
    fragment = compose_research_query("设计 agent", "记忆和成本呢？")
    assert fragment.startswith("设计 agent")
    assert "记忆和成本" in fragment


def test_canonical_query_is_the_single_dedupe_form() -> None:
    # C21.4 两次真实检索只差中间标点，规划与研究执行必须用同一口径判成同一条
    assert canonical_query("文档中，比如记忆，成本这些维度。都是怎么样设计的？") == canonical_query(
        "文档中  比如记忆，成本这些维度。都是怎么样设计的？"
    )
    assert canonical_query("K8S 是什么？") == canonical_query("K8S 是什么")
    # 真正不同的查询不能被合并
    assert canonical_query("报销时限是多少？") != canonical_query("报销制度的时限是多少")


def test_model_facets_reuse_one_complete_question() -> None:
    anchor = "知识库中介绍 K8S 十四个基础概念的文章，如何说明 Deployment 和 Service 各自的职责？"
    facets = [
        "知识库中介绍 K8S 十四个基础概念的文章 如何说明 Deployment 的职责作用",
        "知识库中介绍 K8S 十四个基础概念的文章 如何说明 Service 的职责作用",
    ]

    assert _model_queries_are_facets(anchor, facets) is True
    assert _model_queries_are_facets(
        anchor,
        [facets[0], "AWS AgentCore Memory eventExpiryDuration 参数范围"],
    ) is False


def test_bilingual_official_queries_share_stable_identity() -> None:
    chinese = (
        "AWS 官方文档中 CreateMemory 的 eventExpiryDuration 参数范围和单位是什么？"
    )
    english = (
        "site:docs.aws.amazon.com/bedrock-agentcore CreateMemory "
        "eventExpiryDuration valid range days API Reference"
    )

    assert stable_query_identity(chinese).key == stable_query_identity(english).key
    queries, notes = merge_semantic_queries([chinese, english])
    assert queries == [chinese]
    assert notes and "语义查询身份一致" in notes[0]


def test_official_query_identity_ignores_redundant_parent_product_name() -> None:
    scoped = (
        "site:docs.aws.amazon.com Amazon Bedrock AgentCore Memory CreateMemory "
        "eventExpiryDuration minimum maximum unit"
    )
    translated = (
        "site:docs.aws.amazon.com bedrock-agentcore CreateMemory "
        "eventExpiryDuration API Reference"
    )
    context = (
        "请联网查 AWS 官方文档，核验 Bedrock AgentCore Memory 的 CreateMemory 请求中 "
        "eventExpiryDuration 允许填写的最小值、最大值和单位，并给出页面链接。"
    )

    assert stable_query_identity(
        scoped, source_kind="web", context_text=context
    ).key == stable_query_identity(
        translated, source_kind="web", context_text=context
    ).key


def test_official_site_and_wording_share_stable_identity() -> None:
    first = stable_query_identity(
        "site:docs.python.org Python asyncio.TaskGroup 官方文档 从哪个版本开始提供 version added"
    )
    second = stable_query_identity(
        "Python official documentation asyncio.TaskGroup availability added in version "
        "site:docs.python.org"
    )

    assert first.key == second.key
    assert first.objects == ("asyncio.taskgroup",)
    assert first.source_scope == ("python-official",)


def test_namespaced_identifier_space_and_dot_forms_share_identity() -> None:
    context = "请查 Python 官方文档，确认 asyncio.TaskGroup 从哪个版本开始提供。"
    spaced = stable_query_identity(
        'site:docs.python.org asyncio TaskGroup "Added in version"',
        context_text=context,
    )
    dotted = stable_query_identity(
        "Python 官方文档 asyncio.TaskGroup 新增版本 New in version site:docs.python.org",
        context_text=context,
    )

    assert spaced.key == dotted.key
    assert spaced.objects == ("asyncio.taskgroup",)


def test_context_qualifies_identifier_and_ignores_url_file_name() -> None:
    context = "请查 Python 官方文档，确认 asyncio.TaskGroup 从哪个版本开始提供。"
    qualified = stable_query_identity(
        'site:docs.python.org asyncio.TaskGroup "Added in version"',
        source_kind="web",
        context_text=context,
    )
    reordered = stable_query_identity(
        'site:docs.python.org/3/library/asyncio-task.html TaskGroup "New in version" asyncio',
        source_kind="web",
        context_text=context,
    )

    assert qualified.key == reordered.key
    assert reordered.objects == ("asyncio.taskgroup",)
    assert reordered.source_scope == ("python-official", "web")


def test_official_product_wording_matches_docs_domain_scope() -> None:
    context = "请查 Python 官方文档，确认 asyncio.TaskGroup 从哪个版本开始提供。"
    domain = stable_query_identity(
        'site:docs.python.org asyncio.TaskGroup "Added in version"',
        source_kind="web",
        context_text=context,
    )
    wording = stable_query_identity(
        "Python docs asyncio.TaskGroup introduced in which Python version official documentation",
        source_kind="web",
        context_text=context,
    )

    assert domain.key == wording.key
    assert domain.source_scope == ("python-official", "web")


def test_official_query_identity_ignores_api_identifier_prefix() -> None:
    context = (
        "请联网查 AWS 官方文档，核验 Bedrock AgentCore Memory 的 CreateMemory 请求中 "
        "eventExpiryDuration 允许填写的最小值、最大值和单位，并给出页面链接。"
    )
    prefixed = stable_query_identity(
        "site:docs.aws.amazon.com/bedrock-agentcore/latest/APIReference "
        "API_CreateMemory eventExpiryDuration minimum maximum unit",
        source_kind="web",
        context_text=context,
    )
    plain = stable_query_identity(
        "Amazon Bedrock AgentCore Memory API Reference CreateMemory "
        "eventExpiryDuration 事件过期时长 约束 页面链接",
        source_kind="web",
        context_text=context,
    )

    assert prefixed.key == plain.key
    assert prefixed.objects == ("creatememory", "eventexpiryduration")


def test_version_queries_ignore_redundant_member_and_added_wording() -> None:
    context = "请联网查 Python 官方文档，确认 asyncio.TaskGroup 从哪个 Python 版本开始提供。"
    reference = stable_query_identity(
        'site:docs.python.org asyncio.TaskGroup "Added in version" TaskGroup 任务组 版本',
        source_kind="web",
        context_text=context,
    )
    whats_new = stable_query_identity(
        "Python 3.11 What's New asyncio.TaskGroup 新增 任务组 结构化并发",
        source_kind="web",
        context_text=context,
    )

    assert reference.key == whats_new.key
    assert reference.objects == ("asyncio.taskgroup",)
