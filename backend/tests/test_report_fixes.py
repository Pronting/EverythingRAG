"""报告 F02/F03/F04/F05/F13/F14/F17/F18/F19 的回归用例（离线、零出网）。

对应的浏览器证据见 `.tmp_probe/rag-e2e-20260921/REPORT.md` 的 T01—T20。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent import research as agent_research
from app.agent import routing as agent_routing
from app.agent.contracts import (
    AgentKind,
    EvidenceSourceType,
    ResearchTask,
)
from app.agent.evidence_text import evidence_display_text
from app.agent.research import (
    RagResearcher,
    ResearchRequest,
    RunResearchState,
    build_default_registry,
    build_web_search_query,
    rank_web_hits,
    shared_read_key,
)
from app.agent.tools import _markdown_heading_path


def _request(question: str, **overrides: Any) -> ResearchRequest:
    decision = overrides.pop(
        "decision",
        agent_routing.route_question(
            agent_routing.RoutingInput(question=question, file_count=10, chunk_count=100)
        ),
    )
    task = ResearchTask(
        task_id="t-1",
        question=question,
        requirement_ids=["R1"],
        agent_kind=AgentKind.RAG,
    )
    return ResearchRequest(
        task=task, original_question=question, resolved_question=question, decision=decision, **overrides
    )


def _researcher() -> RagResearcher:
    return RagResearcher(build_default_registry())


def test_web_window_keeps_markdown_heading_path() -> None:
    content = "# Memory API\n\n简介。\n\n## CreateMemory\n\n### eventExpiryDuration\n\nValid Range: 3-365."

    assert _markdown_heading_path(content, content.index("Valid Range")) == (
        "Memory API > CreateMemory > eventExpiryDuration"
    )


def test_web_window_uses_nearby_api_parameter_without_markdown_heading() -> None:
    content = """Required: No

eventExpiryDuration
Event retention in days. Defaults to 30.

Type: Integer
"""
    assert _markdown_heading_path(content, content.index("Event retention")) == (
        "eventExpiryDuration"
    )


# =====================================================================
# F02：完整术语不得在执行层被缩成孤立单词
# =====================================================================


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("AI Native 的具体实践有哪些？", "AI Native"),
        ("tool/call 与 tool/result 是什么关系？", "tool/call"),
        ("文章里的 Prompt Cache 是怎么处理的？", "Prompt Cache"),
        ("K8S 里的 readinessProbe 失败会怎样？", "readinessProbe"),
    ],
)
def test_initial_queries_keep_complete_terms(question: str, expected: str) -> None:
    queries = _researcher().initial_queries(_request(question))

    assert any(expected in query for query in queries), queries
    # 完整术语不得被拆成孤立的单词查询。
    for token in expected.split():
        if len(token) < 3:
            continue
        assert token.casefold() not in {query.strip().casefold() for query in queries}, queries


def test_initial_queries_drop_low_information_tokens() -> None:
    """泛化词 / 中文动词不得单独成查询（用户实测 T12：`介绍` 把无关文档拉进候选）。"""

    decision = agent_routing.route_question(
        agent_routing.RoutingInput(
            question="知识库中介绍 K8S 十四个基础概念的文章，如何说明职责？",
            file_count=10,
            chunk_count=100,
        )
    )
    decision = decision.model_copy(update={"entities": ["介绍", "AI Native"]})
    queries = _researcher().initial_queries(_request("知识库中介绍 K8S 十四个基础概念的文章", decision=decision))

    assert "介绍" not in queries
    assert any("AI Native" in query for query in queries)


def test_initial_queries_add_limited_term_expansion() -> None:
    """术语补充有上限：上游限制 3 条查询时，执行层不得把实际检索扩到十几条。"""

    question = "请解释 Prompt Cache 与 KV Cache、Prefix Cache、Context Cache 之间的关系"
    queries = _researcher().initial_queries(_request(question))

    assert len(queries) <= agent_research.MAX_INITIAL_QUERIES, queries
    # 原问题永远在第一位（改写只是追加线索）
    assert queries[0] == question


# =====================================================================
# F03：重复查询（含句末标点差异）不得重复执行
# =====================================================================


def test_query_key_normalizes_punctuation_whitespace_and_case() -> None:
    """执行侧去重与规划侧共用 ``query_plan.canonical_query``（缺陷 C21.4）。

    旧实现只统一空白 / 大小写 / 首尾标点，中间标点不同的两条查询会被各检索一次；
    C21.4 实测两条只差中间逗号与破折号的查询返回完全相同的 8 个片段。现在比较键去掉
    全部标点，但查询文本本身不被改写（术语里的 ``tool/call`` 仍原样发给检索）。
    """

    key = agent_research._query_key

    assert key("K8S 是什么？") == key("K8S 是什么")
    assert key("AI  Native") == key("ai native")
    # 中间标点与首尾标点一样不影响语义 -> 同一条查询（C21.4）
    assert key("文档中，比如记忆，成本这些维度。都是怎么样设计的？") == key(
        "文档中  比如记忆，成本这些维度。都是怎么样设计的？"
    )
    assert key("「记忆淘汰策略」。") == key("记忆淘汰策略")
    assert key("tool/call 有什么区别？") == key("tool/call 有什么区别")
    assert key("A") != key("B")
    assert key("报销时限是多少？") != key("报销制度的时限是多少")


@pytest.mark.asyncio
async def test_run_research_state_claims_each_query_once() -> None:
    shared = RunResearchState()

    first = await shared.claim_queries(["Prompt Cache 的影响？", "Prompt Cache 的影响"])
    second = await shared.claim_queries(["prompt cache 的影响。", "另一个问题"])

    assert first == ["Prompt Cache 的影响？"]
    assert second == ["另一个问题"], "等价查询不得被第二个子任务重复执行"


@pytest.mark.asyncio
async def test_run_research_state_claims_each_read_once() -> None:
    shared = RunResearchState()

    first = await shared.claim_reads([("doc-1", "3.3 记忆管理"), ("doc-2", "")])
    second = await shared.claim_reads([("doc-1", "3.3 记忆管理"), ("doc-2", "4.2 淘汰")])

    assert first == [("doc-1", "3.3 记忆管理"), ("doc-2", "")]
    assert second == [("doc-2", "4.2 淘汰")], "同一章节不得被重复读取"
    assert shared_read_key("doc-1", "3.3 记忆管理") in shared.read_keys()


# =====================================================================
# F13：文档身份随证据传递
# =====================================================================


def test_evidence_display_text_carries_document_title() -> None:
    text = evidence_display_text("标题：3.3 记忆管理\n内容：记忆按作用域分三层。", document_title="存之有序.md")

    assert text.startswith("《存之有序.md》")
    assert "3.3 记忆管理" in text


def test_evidence_display_text_without_title_is_unchanged() -> None:
    """没有文档标题时，行为与引入该参数前逐字一致（不猜标题）。"""

    assert evidence_display_text("标题：3.3 记忆管理\n内容：记忆按作用域分三层。") == (
        "3.3 记忆管理：记忆按作用域分三层。"
    )


def test_evidence_statement_keeps_document_identity() -> None:
    from app.agent.research import _evidence_statement, make_evidence

    item = make_evidence(
        workspace_id="ws-1",
        run_id="run-1",
        task_id="t-1",
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-1#block-1",
        content="标题：4.2 淘汰与生命周期\n内容：记忆按淘汰策略删除过期条目。",
        document_title="货拉拉大模型记忆系统（一）.md",
    )
    statement = _evidence_statement(item)

    assert "货拉拉大模型记忆系统（一）.md" in statement
    assert "doc-1" not in statement and "block-1" not in statement


# =====================================================================
# F05 / F17：联网意图与官方来源要求
# =====================================================================


@pytest.mark.parametrize(
    "question",
    [
        "请联网搜索 Kubernetes 官方文档，说明 readinessProbe 与 livenessProbe 的差异。",
        "请联网查找 AWS 官方资料，核实事件保留时长的范围。",
        "帮我查一下 Kubernetes 官方文档里的探针说明",
        "上网查一下这个参数的最新取值",
        "联网核实一下这个结论是否过时",
    ],
)
def test_explicit_web_request_covers_synonyms(question: str) -> None:
    assert agent_routing.explicit_web_request(question) != ""


@pytest.mark.parametrize(
    "question",
    [
        "不要联网查找，只用知识库里的文章回答",
        "不用搜索网页，直接按你的知识回答",
        "别去联网查了，先看本地资料",
    ],
)
def test_explicit_web_request_respects_negation(question: str) -> None:
    assert agent_routing.explicit_web_request(question) == ""


def test_route_uses_official_web_only_for_explicit_official_question() -> None:
    question = "请联网查找 AWS 官方资料，核实 Bedrock AgentCore Memory 的事件保留时长允许设置的范围，并给出官方链接。"
    decision = agent_routing.route_question(
        agent_routing.RoutingInput(question=question, file_count=10, chunk_count=100)
    )

    assert decision.needs_web is True, "「联网查找」必须保持明确的联网意图"
    assert decision.needs_knowledge is False


def test_web_search_query_drops_instruction_and_keeps_need() -> None:
    question = "请联网搜索 Kubernetes 官方文档，说明 readinessProbe 与 livenessProbe 失败后的行为差异，并提供官方来源。"
    query = build_web_search_query(question)

    assert "请联网搜索" not in query
    assert "并提供官方来源" not in query
    assert "readinessProbe" in query


def test_official_domains_are_explicit_and_conservative() -> None:
    assert agent_routing.official_domains("请联网搜索 Kubernetes 官方文档") == ["kubernetes.io"]
    assert "docs.aws.amazon.com" in agent_routing.official_domains("AWS 官方资料")
    assert agent_routing.official_domains("请联网搜索某个小众框架的官方文档") == []


def test_rank_web_hits_prefers_official_domain() -> None:
    hits = [
        {"url": "https://medium.com/x"},
        {"url": "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"},
        {"url": "https://blog.example.com/y"},
    ]
    ranked = rank_web_hits(hits, preferred_domains=["kubernetes.io"])

    assert ranked[0]["url"] == hits[1]["url"]
    assert len(ranked) == len(hits), "官方页面不存在时仍保留其它来源，不丢结果"


# =====================================================================
# F15：指定文章的内容问题必须走正文研究（问"库里有哪些文章"不算）
# =====================================================================


@pytest.mark.parametrize(
    "question",
    [
        "知识库中介绍 K8S 十四个基础概念的文章，如何说明 Deployment 和 Service 各自的职责？",
        "TencentDB Agent Memory 那篇文章为什么强调 Policy 可以引用 Memory？请给出文中的依据。",
        "请根据存之有序那篇 Agent 记忆系统文章，说明双时间模型如何处理事实过期。",
    ],
)
def test_document_content_questions_go_through_knowledge(question: str) -> None:
    decision = agent_routing.route_question(
        agent_routing.RoutingInput(question=question, file_count=10, chunk_count=100)
    )

    assert decision.needs_knowledge is True, question


def test_plain_overview_and_product_questions_do_not_force_retrieval() -> None:
    for question in ("知识库里有哪些文章？", "Everything RAG 是做什么的？"):
        decision = agent_routing.route_question(
            agent_routing.RoutingInput(question=question, file_count=10, chunk_count=100)
        )
        assert decision.needs_knowledge is False, question


# =====================================================================
# F12：缺少相似度时的词法命中必须有真实重叠
# =====================================================================


def test_lexical_relevance_requires_overlap() -> None:
    chunk = {"text": "本周值班安排与工位调整通知", "similarity": None}

    assert agent_research._is_relevant(chunk, 0.35, query="K8S Deployment 的副本维持职责") is False
    assert agent_research._is_relevant(
        {"text": "Deployment 负责维持副本数并滚动更新", "similarity": None},
        0.35,
        query="K8S Deployment 的副本维持职责",
    )


def test_low_similarity_still_wins_over_lexical_overlap() -> None:
    """有向量分数就以它为准：低于门控不得因为正文里恰好有同样的词而翻案。"""

    chunk = {"text": "Deployment 的副本维持职责", "similarity": 0.05}

    assert agent_research._is_relevant(chunk, 0.35, query="Deployment 的副本维持职责") is False


# =====================================================================
# F04：目录型片段触发继续读取正文
# =====================================================================


def test_outline_chunks_trigger_reading_the_document() -> None:
    documents_seen = {
        "doc-1": {
            "document_id": "doc-1",
            "heading_path": "目录",
            "text": "目录\n1. AiBox\n2. Spec 驱动\n3. 交付指标",
        }
    }
    targets = agent_research._outline_read_targets(documents_seen, {})

    assert targets == [("doc-1", "")]


def test_normal_chunks_do_not_trigger_outline_reading() -> None:
    documents_seen = {
        "doc-1": {
            "document_id": "doc-1",
            "heading_path": "3.1 AiBox",
            "text": "AiBox 把需求、规格与验证串成一条流水线，交付前必须跑完独立审查。" * 3,
        }
    }

    assert agent_research._outline_read_targets(documents_seen, {}) == []


# =====================================================================
# F16：等待用户批准的时间不计入联网额度
# =====================================================================


def test_web_budget_elapsed_excludes_approval_waiting() -> None:
    """批准等待必须与实际网络执行时间分开统计（用户实测 T19：等待 27 秒吃光 20 秒额度）。"""

    from app.agent.contracts import ReasoningLevel
    from app.agent.web_budget import WebBudget, WebBudgetLedger, WebPolicy

    now = {"value": 1000.0}
    ledger = WebBudgetLedger(
        policy=WebPolicy(
            allowed=True,
            budget=WebBudget(queries=2, pages=3, seconds=20.0),
            reasoning_level=ReasoningLevel.LOW.value,
        ),
        clock=lambda: now["value"],
    )

    assert ledger.reserve_search("Kubernetes 官方文档").allowed
    now["value"] += 3.0  # 搜索真实执行 3 秒
    assert ledger.elapsed() == 3.0

    ledger.pause()  # 等用户点「继续」：27 秒
    now["value"] += 27.0
    assert ledger.elapsed() == 3.0, "等待批准的时间不得计入联网用时"
    assert ledger.wait_seconds() == 27.0
    assert ledger.remaining_seconds() == 17.0

    ledger.resume()
    now["value"] += 5.0  # 网页读取真实执行 5 秒
    assert ledger.elapsed() == 8.0
    assert ledger.wait_seconds() == 27.0

    assert ledger.reserve_pages(1).allowed, "批准后仍应能用剩余额度读取页面"
    assert "等待用户操作" in ledger.summary_text()
