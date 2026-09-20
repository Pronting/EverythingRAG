"""工作区知识目录测试（PRD 6.1 / 5.3.4）。Owner: T06（routing）。

覆盖：目录结构（标题/层级/主题/术语/别名/来源 ID）、精确词与别名查询、全文检索与
分页、`definition_unknown`、模型术语候选的程序校验（引用片段必须存在）、提示只注入
相关条目、对外投影不含绝对路径、按 knowledge_revision 失效重建。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore
from app.generation.knowledge_catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogMatch,
    KnowledgeCatalog,
    TermCandidate,
    apply_term_candidates,
    build_knowledge_catalog,
    build_term_extraction_prompt,
    catalog_cache_key,
    catalog_cache_size,
    catalog_for_area,
    clear_catalog_cache,
    invalidate_catalog,
    lookup_catalog_query,
    normalize_catalog_text,
    parse_term_candidates,
)

RAW_ROOT = r"C:\Users\tester\notes"


def _blocks() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "b1",
            "内部风控 P99 方案：P99 表示第 99 百分位延迟，用于刻画长尾。本方案采用分层限流。",
            {
                "source_file": rf"{RAW_ROOT}\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "platform": "local",
                "doc_id": "hash-1",
                "heading_path": "风控 > P99 方案",
                "anchor": "p99",
                "heading_aliases": json.dumps(["P99", "尾延迟方案"], ensure_ascii=False),
            },
        ),
        (
            "b2",
            "坑位归因用于投放复盘。",
            {
                "source_file": rf"{RAW_ROOT}\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "doc_id": "hash-1",
                "heading_path": "风控 > 坑位归因",
                "anchor": "attribution",
            },
        ),
        (
            "b3",
            "以图搜图功能消耗 token 过多的问题在这里复盘。",
            {
                "source_file": rf"{RAW_ROOT}\识图复盘.md",
                "chunk_type": "text",
                "source_type": "document",
                "doc_id": "hash-2",
                "heading_path": "识图 > token 消耗",
                "anchor": "token",
            },
        ),
    ]


def _catalog() -> KnowledgeCatalog:
    return build_knowledge_catalog(_blocks(), workspace_id="ws-1", knowledge_revision=4)


class FakeVectorStore:
    """按 source_file 返回块文本的最小替身（只用于引用片段校验）。"""

    def __init__(self, mapping: dict[str, dict[str, str]]) -> None:
        self.mapping = mapping

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return self.mapping.get(source_file, {})


# ---------------------------------------------------------------------
# 目录结构
# ---------------------------------------------------------------------


def test_catalog_documents_headings_terms_and_source_ids() -> None:
    catalog = _catalog()
    assert catalog.schema_version == CATALOG_SCHEMA_VERSION
    assert catalog.workspace_id == "ws-1"
    assert catalog.knowledge_revision == 4
    assert len(catalog.documents) == 2

    assert sorted(document.title for document in catalog.documents) == [
        "识图复盘.md",
        "风控方案.md",
    ]
    assert all(document.document_id.startswith("doc-") for document in catalog.documents)
    assert all(document.content_id for document in catalog.documents)

    fengkong = next(document for document in catalog.documents if document.title == "风控方案.md")
    assert [heading.path for heading in fengkong.headings] == ["风控 > P99 方案", "风控 > 坑位归因"]
    assert fengkong.headings[0].level == 2
    assert fengkong.headings[0].title == "P99 方案"
    assert fengkong.headings[0].aliases == ["P99", "尾延迟方案"]
    assert fengkong.headings[0].chunk_count == 1
    assert fengkong.topic == "notes"
    assert fengkong.chunk_count == 2

    terms = {term.term: term for term in catalog.terms}
    assert set(terms) == {"P99 方案", "坑位归因", "token 消耗"}
    # 有明确解释的术语给出片段；没有解释的标 definition_unknown（PRD 6.1）。
    assert terms["P99 方案"].definition_unknown is False
    assert terms["P99 方案"].evidence_verified is True
    assert "第 99 百分位延迟" in terms["P99 方案"].snippet
    assert terms["坑位归因"].definition_unknown is True
    assert terms["坑位归因"].heading_path == "风控 > 坑位归因"
    assert terms["坑位归因"].document_id == fengkong.document_id


def test_catalog_projection_never_leaks_absolute_paths() -> None:
    catalog = _catalog()
    dumped = catalog.model_dump_json()
    assert "C:" not in dumped
    assert "tester" not in dumped
    result = catalog.search("P99", limit=10)
    assert "C:" not in result.model_dump_json()


# ---------------------------------------------------------------------
# 查询：精确词 / 别名 / 全文
# ---------------------------------------------------------------------


def test_exact_term_and_alias_lookup() -> None:
    catalog = _catalog()
    assert [term.term for term in catalog.lookup_term("P99 方案")] == ["P99 方案"]
    assert [term.term for term in catalog.lookup_term("P99")] == ["P99 方案"]
    assert [term.term for term in catalog.lookup_term("尾延迟方案")] == ["P99 方案"]
    assert [term.term for term in catalog.lookup_term("ｐ９９")] == ["P99 方案"]  # 全角 + 大小写
    assert catalog.lookup_term("不存在的术语") == []


def test_search_supports_terms_headings_and_full_text() -> None:
    catalog = _catalog()
    term_hits = catalog.search("P99 方案")
    assert term_hits.matches[0].kind == "term"
    assert term_hits.matches[0].matched_on == "P99 方案"

    heading_hits = catalog.search("token 消耗")
    assert heading_hits.matches[0].document_id
    assert heading_hits.matches[0].kind in {"term", "heading"}

    # 全文（目录字段级）检索：用主题词命中目录条目。
    assert catalog.search("识图").total >= 1
    assert catalog.search("").total == 0


def test_search_pagination_is_stable_and_bounded() -> None:
    catalog = _catalog()
    first = catalog.search("风控", limit=1)
    assert len(first.matches) == 1
    assert first.next_cursor == 1
    second = catalog.search("风控", limit=1, cursor=1)
    assert second.next_cursor == 2
    assert first.matches[0].document_id == second.matches[0].document_id
    # 上限收敛，避免一次把整库目录塞进提示。
    assert len(catalog.search("风控", limit=10_000).matches) <= 50


def test_lookup_catalog_query_result_is_pathless_and_paginated() -> None:
    catalog = _catalog()
    payload = lookup_catalog_query(catalog, "坑位归因", cursor=0, limit=2)
    assert payload["total"] >= 1
    assert payload["items"]
    assert "C:" not in json.dumps(payload, ensure_ascii=False)
    assert payload["items"][0]["definition_unknown"] is True


def test_search_matches_by_kind_filter() -> None:
    catalog = _catalog()
    only_terms = catalog.search("P99", kinds=["term"])
    assert only_terms.matches
    assert all(match.kind == "term" for match in only_terms.matches)
    only_docs = catalog.search("风控", kinds=["document"])
    assert only_docs.matches
    assert all(match.kind == "document" for match in only_docs.matches)


def test_catalog_match_model_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CatalogMatch(kind="term", document_id="d", unexpected=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------
# 提示只注入相关条目
# ---------------------------------------------------------------------


def test_prompt_projection_only_includes_relevant_entries() -> None:
    catalog = _catalog()
    projection = catalog.prompt_projection("P99 方案是怎么设计的？", limit=4)
    assert projection.text
    assert "P99" in projection.text
    assert "识图复盘" not in projection.text
    assert "以图搜图" not in projection.text
    assert len(projection.document_ids) == len(set(projection.document_ids))
    assert "D1" in projection.text


def test_prompt_projection_marks_unknown_definition() -> None:
    catalog = _catalog()
    projection = catalog.prompt_projection("坑位归因是什么意思？", limit=4)
    assert "缺少明确解释" in projection.text
    assert projection.term_count == 1


# ---------------------------------------------------------------------
# 模型术语候选：程序校验引用片段
# ---------------------------------------------------------------------


def test_term_extraction_prompt_lists_documents_and_headings() -> None:
    catalog = _catalog()
    prompt = build_term_extraction_prompt(catalog.documents, max_terms=5)
    for document in catalog.documents:
        assert document.document_id in prompt
        assert document.title in prompt
    assert "P99 方案" in prompt


def test_parse_term_candidates_handles_json_and_fences() -> None:
    payload = json.dumps(
        [{"term": "P99", "document_id": "doc-1", "snippet": "x"}], ensure_ascii=False
    )
    assert [item.term for item in parse_term_candidates(payload)] == ["P99"]
    assert [item.term for item in parse_term_candidates(f"```json\n{payload}\n```")] == ["P99"]
    assert parse_term_candidates('{"terms": ' + payload + "}")[0].term == "P99"
    with pytest.raises(ValueError):
        parse_term_candidates("不是 JSON")
    with pytest.raises(TypeError):
        parse_term_candidates({"terms": "not-a-list"})  # type: ignore[arg-type]


def test_apply_term_candidates_accepts_only_verifiable_evidence() -> None:
    catalog = _catalog()
    document = next(document for document in catalog.documents if document.title == "风控方案.md")
    candidates = [
        TermCandidate(
            term="尾延迟",
            aliases=["tail latency"],
            document_id=document.document_id,
            heading_path="风控 > P99 方案",
            snippet="P99 表示第 99 百分位延迟",
        ),
        TermCandidate(
            term="编造的术语",
            document_id=document.document_id,
            heading_path="风控 > P99 方案",
            snippet="这段文字在原文里并不存在",
        ),
        TermCandidate(
            term="未知文档的术语",
            document_id="doc-does-not-exist",
            snippet="任意",
        ),
        TermCandidate(
            term="标题不存在的术语",
            document_id=document.document_id,
            heading_path="风控 > 不存在的标题",
            snippet="内部风控 P99 方案",
        ),
        TermCandidate(
            term="缺少解释的术语",
            document_id=document.document_id,
            heading_path="风控 > 坑位归因",
            snippet="",
        ),
    ]
    result = apply_term_candidates(catalog, candidates, text_lookup=lambda _id: "")
    accepted = {term.term: term for term in result.accepted}
    assert set(accepted) == {"尾延迟", "缺少解释的术语"}
    assert accepted["尾延迟"].evidence_verified is True
    assert accepted["尾延迟"].aliases == ["tail latency"]
    assert accepted["缺少解释的术语"].definition_unknown is True
    reasons = {rejection.term: rejection.reason for rejection in result.rejected}
    assert "引用片段不存在于原文" in reasons["编造的术语"]
    assert "文档 ID 不存在" in reasons["未知文档的术语"]
    assert "标题位置不存在" in reasons["标题不存在的术语"]
    # 接受的术语进入目录并可被查询（别名也生效）。
    assert catalog.lookup_term("tail latency")
    assert "尾延迟" in [term.term for term in catalog.terms]
    assert catalog.lookup_term("缺少解释的术语")[0].definition_unknown is True


def test_apply_term_candidates_prefers_full_text_lookup_over_truncated_sample() -> None:
    catalog = _catalog()
    document = next(document for document in catalog.documents if document.title == "风控方案.md")
    long_text = "无关前缀。" * 500 + "限流算法细节：令牌桶按固定速率补充令牌。"
    candidate = TermCandidate(
        term="限流算法细节",
        document_id=document.document_id,
        heading_path="风控 > P99 方案",
        snippet="令牌桶按固定速率补充令牌",
    )
    # 样例正文被截断，只有走全文访问器才能通过校验（避免误拒真实片段）。
    assert "令牌桶" not in catalog.text_for(document.document_id)
    result = apply_term_candidates(catalog, [candidate], text_lookup=lambda _id: long_text)
    assert [term.term for term in result.accepted] == ["限流算法细节"]


def test_internal_text_lookup_maps_source_id_to_blocks_without_exposing_path() -> None:
    catalog = _catalog()
    document = next(document for document in catalog.documents if document.title == "风控方案.md")
    vector_store = FakeVectorStore(
        {rf"{RAW_ROOT}\风控方案.md": {"b1": "第一块正文", "b2": "第二块正文"}}
    )
    lookup = catalog.internal_text_lookup(vector_store)
    assert "第一块正文" in lookup(document.document_id)
    assert "第二块正文" in lookup(document.document_id)
    assert lookup("doc-unknown") == ""


# ---------------------------------------------------------------------
# 缓存与知识版本（PRD 5.3.4）
# ---------------------------------------------------------------------


class _Area:
    """`WorkspaceKnowledgeArea` 的最小替身：只用目录缓存需要的成员。"""

    def __init__(self, context: Any, blocks: list[tuple[str, str, dict[str, Any]]]) -> None:
        self.context = context
        self.index_fingerprint = "chunks__fake__v1"
        self._blocks = blocks

    def blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._blocks


def _real_workspace(tmp_path: Path) -> tuple[WorkspaceContextFactory, Any, WorkspaceStore, Any]:
    store = WorkspaceStore(tmp_path / "appdata")
    workspace = store.create_with_uuid("测试工作区")
    factory = WorkspaceContextFactory(tmp_path / "erag-data", store)
    factory.ensure_dirs(workspace.id)
    return factory, factory.resolve(workspace.id), store, workspace


def test_catalog_cache_key_covers_workspace_revision_fingerprint(tmp_path: Path) -> None:
    _factory, context, _store, workspace = _real_workspace(tmp_path)
    key = catalog_cache_key(context, index_fingerprint="finger-a")
    assert workspace.id in key
    assert "finger-a" in key
    assert key.endswith(f"|ws={workspace.id}")
    assert catalog_cache_key(context, index_fingerprint="finger-b") != key
    assert (
        catalog_cache_key(context, index_fingerprint="finger-a", generator_version="catalog-v2")
        != key
    )


def test_catalog_for_area_caches_and_rebuilds_after_revision_bump(tmp_path: Path) -> None:
    clear_catalog_cache()
    factory, context, store, workspace = _real_workspace(tmp_path)
    first = catalog_for_area(_Area(context, _blocks()))
    assert catalog_cache_size() == 1
    assert first.workspace_id == workspace.id
    assert catalog_for_area(_Area(context, _blocks())) is first

    # 导入 / 同步 / 删除提交后递增本区 knowledge_revision → 缓存键变化 → 重建。
    store.bump_knowledge_revision(workspace.id)
    refreshed = factory.resolve(workspace.id)
    assert refreshed.knowledge_revision == 1
    second = catalog_for_area(_Area(refreshed, _blocks()))
    assert second is not first
    assert second.knowledge_revision == 1
    assert catalog_cache_size() == 2
    assert invalidate_catalog(workspace.id) >= 1
    assert catalog_cache_size() == 0
    clear_catalog_cache()


def test_normalize_catalog_text_is_stable() -> None:
    assert normalize_catalog_text("  Ｐ９９  方案 ") == "p99 方案"
    assert normalize_catalog_text(None) == ""
