import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.agent.contracts import ResearchTask
from app.agent.query_plan import (
    QueryPlannerAgent,
    confirmed_document_scope,
    distinct_queries,
    generic_document_content_ask,
    model_plan_needed,
    parse_query_plan,
)
from app.agent.research import (
    RagResearcher,
    ResearchRequest,
    RunResearchState,
    WebResearcher,
    _looks_like_outline,
    _outline_read_targets,
    _read_focus,
    _web_content_supports,
    rank_web_hits,
    select_web_urls,
    web_page_family_key,
)
from app.agent.tools import (
    ToolExecutionError,
    ToolResult,
    document_scope_filter,
    focused_window_offset,
)
from app.generation.knowledge_catalog import CatalogDocument, KnowledgeCatalog, _document_id_for

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / ".tmp_probe/rag-verify-20260922"


def test_instruction_fragments_do_not_become_queries():
    queries = distinct_queries(["不要联网", "附上出处", "文档中", "AI Native 的具体实践"])
    assert queries == ["AI Native 的具体实践"]
    plan = parse_query_plan(
        {"queries": ["不要联网", "Deployment 与 Service 的职责"]},
        question="不要联网。说明 Deployment 与 Service 的职责",
    )
    assert plan.queries == ["Deployment 与 Service 的职责"]


def test_query_deduplication_preserves_relationship_direction():
    pairs = [
        ["Policy 可以引用 Memory", "Memory 可以引用 Policy"],
        ["Gateway 调用 Agent", "Agent 调用 Gateway"],
        ["甲方授权乙方", "乙方授权甲方"],
        ["100 转换为 200", "200 转换为 100"],
        ["Memory 可以改写 Policy", "Memory 不可以改写 Policy"],
    ]
    for queries in pairs:
        assert distinct_queries(queries) == queries
    assert distinct_queries(["Policy 引用 Memory？", "policy 引用 memory"]) == [
        "Policy 引用 Memory？"
    ]


def test_task_does_not_repeat_original_question():
    request = ResearchRequest(
        task=ResearchTask(task_id="research", question="Agent 记忆与成本设计"),
        original_question="比如记忆，成本这些维度。都是怎么样设计的？",
        resolved_question="设计 Agent —— 比如记忆，成本这些维度。都是怎么样设计的？",
    )
    assert RagResearcher().initial_queries(request) == ["Agent 记忆与成本设计"]


def test_knowledge_content_request_is_not_catalog_only():
    assert generic_document_content_ask("知识库中 AI Native 的具体实践有哪些？")
    assert generic_document_content_ask("资料里消息队列的设计取舍有哪些？")


def test_real_intro_requests_full_document():
    data = json.loads((EVIDENCE / "V02-evidence.json").read_text(encoding="utf-8"))
    # 从真实浏览器存档取得引用正文，核对完整阅读条件。
    citations = json.loads(data["answers"][0]["citations"])
    assert citations
    text = next(item["snippet"] for item in citations if "读完你会知道" in item["snippet"])
    assert _looks_like_outline({"text": text})


def test_real_official_full_text_window_reaches_parameter_body():
    source = Path.home() / ".everything-rag/workspaces/default/artifacts/web/1e78bcbcf38b1886621b893bc11e203ef2af13aa3350afd4290998522a8dc1c7.md"
    content = source.read_text(encoding="utf-8")
    query = "创建 Bedrock AgentCore Memory 时 eventExpiryDuration 最小和最大分别是多少天"
    offset = focused_window_offset(content, query, 20000)
    assert offset > 0
    assert "eventExpiryDuration" in content[offset:offset + 20000]
    assert "365" in content[offset:offset + 20000]


def test_rank_requires_actual_official_host():
    result = rank_web_hits(
        [{"url": "https://example.com/docs.aws.amazon.com"},
         {"url": "https://docs.aws.amazon.com/reference"}],
        preferred_domains=["docs.aws.amazon.com"],
    )
    assert result[0]["url"] == "https://docs.aws.amazon.com/reference"


def test_numeric_bounds_belong_to_requested_parameter():
    question = "timeoutSeconds 的最小和最大范围是什么？"
    assert not _web_content_supports(question,
        "[timeoutSeconds](#timeoutSeconds)\n\n## retryCount\n\nMinimum: 1. Maximum: 20.")
    assert not _web_content_supports(question,
        "## timeoutSeconds\n\nMaximum: 60.\n\n## retryCount\n\nMinimum: 1. Maximum: 20.")
    assert _web_content_supports(question,
        "## timeoutSeconds\n\nMinimum: 5. Maximum: 60.\n\n## retryCount\n\nMinimum: 1.")
    assert not _web_content_supports(question,
        "timeoutSeconds 配置更新时间 2026-03-03")


def test_real_web_pages_do_not_supply_full_range():
    path = ROOT / ".tmp_probe/rag-fix-20260922/T20-evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data["operations"][-1]["saved_result"]["result"]["data"]["pages"]
    question = "Bedrock AgentCore Memory 事件保留时长允许设置的范围"
    assert pages
    assert not any(_web_content_supports(question, page["content"]) for page in pages)


def test_real_intro_is_retained_after_image_chunk():
    path = ROOT / ".tmp_probe/rag-fix-20260922/V02-evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    operation = next(item for item in data["operations"] if item["tool_name"] == "search_knowledge")
    saved = operation["saved_result"]
    result = ToolResult.from_payload(saved["tool"], saved["result"])
    documents = {}
    RagResearcher()._collect_retrieval(
        result.data["query"], result, workspace_id=data["run"]["workspace_id"],
        run_id=data["run"]["id"], task_id=operation["task_id"], requirement_ids=[],
        evidences=[], weak_chunks={}, documents_seen=documents, notes=[],
    )
    introduction = next(item for item in result.data["chunks"]
                        if "读完你会知道" in item["text"])
    assert (introduction["document_id"], "") in _outline_read_targets(documents, {})


def test_recorded_tool_result_cache_copy_preserves_frozen_result():
    data = json.loads((EVIDENCE / "V02-evidence.json").read_text(encoding="utf-8"))
    saved = data["operations"][0]["saved_result"]
    original = ToolResult.from_payload(saved["tool"], saved["result"])
    original_payload = original.to_payload()
    fresh = RunResearchState._result_copy(original, cached=False)
    cached = RunResearchState._result_copy(original, cached=True)
    assert fresh.reused == original.reused
    assert cached.reused is True
    assert cached.operation_id == original.operation_id
    assert cached.data == original.data
    assert cached.data is not original.data
    cached.data["chunks"].clear()
    assert original.to_payload() == original_payload
    assert fresh.data["chunks"]


def test_recorded_tasks_share_same_reading_focus():
    path = ROOT / ".tmp_probe/rag-fix-final-20260922/T13-evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [item for item in data["tasks"] if item["agent_kind"] == "rag"]
    requirement_texts = {"R1": tasks[0]["question"], "R2": tasks[1]["question"]}
    focuses = []
    for item in tasks:
        task = ResearchTask(task_id=item["id"], question=item["question"],
                            requirement_ids=json.loads(item["requirement_ids"]))
        request = ResearchRequest(task=task, original_question=tasks[2]["question"])
        focuses.append(_read_focus(request, task.requirement_ids, requirement_texts))
    assert len(set(focuses)) == 1
    assert focuses[0].count("FPY 如何计算") == 1


def test_catalog_and_tool_document_ids_use_actual_source_metadata():
    database = Path.home() / ".everything-rag/chroma.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        records = connection.execute(
            "SELECT e.embedding_id, m.string_value FROM embeddings e "
            "JOIN embedding_metadata m ON e.id=m.id WHERE m.key='source_file'"
        ).fetchall()
    assert records
    blocks = [(block_id, "", {"source_file": source}) for block_id, source in records]
    block_id, source = records[0]
    tool_id = block_id.split("::", 1)[0]
    by_catalog = document_scope_filter(blocks, [_document_id_for(source)])
    by_tool = document_scope_filter(blocks, [tool_id])
    assert by_catalog[0] == by_tool[0]
    assert by_tool[1] <= by_catalog[1]
    assert source in by_catalog[0]["source_file"]["$in"]
    assert tool_id in by_catalog[1]
    with pytest.raises(ToolExecutionError, match="1 项"):
        document_scope_filter(blocks, [_document_id_for(source), "missing-document-identity"])


def test_recorded_document_titles_reach_search_scope():
    chunks = []
    for case in ("T12", "T15", "V10"):
        data = json.loads((EVIDENCE / f"{case}-evidence.json").read_text(encoding="utf-8"))
        for operation in data["operations"]:
            chunks.extend(operation.get("saved_result", {}).get("result", {})
                          .get("data", {}).get("chunks", []))
    documents = {chunk["document_id"]: CatalogDocument(
        document_id=chunk["document_id"], title=chunk["document_name"]
    ) for chunk in chunks}
    catalog = KnowledgeCatalog(documents=list(documents.values()))
    for document in catalog.documents:
        question = f"依据《{document.title}》说明文档里的具体设计"
        scope = confirmed_document_scope(question, catalog)
        request = ResearchRequest(
            task=ResearchTask(task_id="source-check", question=question), document_scope=scope,
        )
        assert document.document_id in RagResearcher().search_arguments(request, question)["document_scope"]
    assert confirmed_document_scope("记忆系统工程实践与 Token 成本两篇文章", catalog) == []
    assert len(confirmed_document_scope("对照《存之有序，治之有矩》和《取之有度，用之有节》", catalog)) == 2


def test_explicit_web_facts_use_existing_query_planning():
    questions = [
        "核对 PostgreSQL 的最大连接数默认值",
        "查询 Kubernetes 默认的终止宽限时间",
        "核实 Azure Cosmos DB 数据过期时间的可配置范围",
        "请联网查找 AWS 官方资料，核实 Bedrock AgentCore Memory 的事件保留时长允许设置的范围，并给出官方链接。",
    ]
    for question in questions:
        assert model_plan_needed(question, needs_web=True)
        prompt = QueryPlannerAgent().build_prompt(question, needs_web=True)
        assert "后续官方原文验证" in prompt
        assert question in prompt
    assert not model_plan_needed("核对 PostgreSQL 的最大连接数默认值")


def test_rewritten_query_retains_original_official_source_constraint():
    request = ResearchRequest(
        task=ResearchTask(task_id="official-check", question="Kubernetes terminationGracePeriodSeconds default"),
        original_question="请联网核实 Kubernetes 官方文档中的终止宽限时间默认值",
    )
    assert "site:kubernetes.io" in WebResearcher().search_arguments(request)["query"]
    assert not _web_content_supports("timeoutSeconds minimum maximum", "timeoutSeconds\n\nMaximum: 60.")


def test_recorded_python_docs_urls_read_one_canonical_page():
    hits = [
        {"url": "https://docs.python.org/3/library/asyncio-task.html#asyncio.TaskGroup"},
        {"url": "https://docs.python.org/3/library/asyncio-task.html?highlight=cancel"},
        {"url": "https://docs.python.org/3/library/asyncio-task.html?highlight=run_coroutine_threadsafe"},
    ]
    assert select_web_urls(hits, limit=3) == [
        "https://docs.python.org/3/library/asyncio-task.html"
    ]
    assert web_page_family_key("https://docs.python.org/3/library/asyncio-task.html?view=source") != (
        web_page_family_key("https://docs.python.org/3/library/asyncio-task.html")
    )
