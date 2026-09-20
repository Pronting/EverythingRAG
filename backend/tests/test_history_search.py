"""中文历史检索（PRD 11.5 / 13.2；验收 A22）。

A22 要求"实际历史主题和内部术语能够命中正确消息"。这条同时依赖五件事，缺一项就查不中：

1. 每工作区独立的 ``jieba.Tokenizer``（``tmp_dir`` / ``cache_file`` 在本区）；
2. **导入与查询同一个** ``cut_for_search()`` 配置（取自 ``JIEBA_TOKENIZER_MODE`` 唯一定义）；
3. 内部术语词典 —— 合成词必须整体成词，否则查询侧的子词在导入侧建立的索引里不存在；
4. FTS5 ``unicode61`` + 标准引号规则 + 参数绑定（查询表达式**绝不拼串**）；
5. 检索按 ``workspace_id`` 收口（分区分词、分区查询）。

断言刻意采用**真分叉样例**（``知识库管理系统`` / ``多模态向量检索``：``cut`` 与
``cut_for_search`` 词数不同）。用 ``固定快照`` 这类不退化样例做"同模式"断言会**永真**，
等于测试空跑。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.app_store import AppStore
from app.core.workspace_context import (
    JIEBA_DICT_VERSION,
    JIEBA_GENERATOR_VERSION,
    JIEBA_TOKENIZER_MODE,
    WorkspacePaths,
)
from app.memory.history import (
    HistoryIndexUnavailableError,
    HistoryQueryError,
    HistorySearch,
    HistorySearchError,
    WorkspaceHistoryIndex,
    build_match_expression,
    history_cache_file,
    history_index_fingerprint,
    history_tokenizer,
    load_terms_dictionary,
    normalize_date_bounds,
    quote_terms,
    terms_version,
    tokenize_for_index,
)
from tests.fakes_memory import workspace_paths_for

#: 真分叉样例：``cut`` 给 2 词、``cut_for_search`` 给 6 词。
COMPOUND_TERM = "多模态向量检索"
#: 真分叉样例二：``cut`` 给 2 词、``cut_for_search`` 给 6 词。
COMPOUND_TERM_2 = "知识库管理系统"
#: 内部术语词典（每区一份）。
TERMS = (COMPOUND_TERM, COMPOUND_TERM_2, "固定快照")


async def _build(tmp_path: Path):  # type: ignore[no-untyped-def]
    """搭一份真实的文件 ``app.db``：两个工作区、归档会话，两区各自建索引。"""

    paths = workspace_paths_for(tmp_path, "default")
    db = paths.data_dir / "app.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    store = await AppStore.open(db)
    await store.create_workspace("default", "default", name="默认")
    await store.create_workspace("other", "other-key", name="另一个区")

    main = await store.create_conversation("conv-main", "default", title="主会话")
    archived = await store.create_conversation("conv-arch", "default", title="归档会话")
    await store.update_conversation("conv-arch", status="archived")
    other_conv = await store.create_conversation("conv-other", "other", title="别的区")

    contents = [
        ("你回答一律用中文", "user"),
        (f"我们讨论一下 {COMPOUND_TERM} 的选型，注意 operation_id 幂等", "user"),
        (f"{COMPOUND_TERM_2} 该怎么分层？", "user"),
        ("固定快照如何失效？invalidation_revision 会增加", "assistant"),
        ("今天天气不错", "user"),
    ]
    records = []
    for content, role in contents:
        records.append(
            await store.append_message(conversation_id=main.id, role=role, content=content)
        )
    archived_record = await store.append_message(
        conversation_id=archived.id, role="user", content="归档会话里也提到 固定快照"
    )
    other_content = f"{COMPOUND_TERM} 你回答一律用中文"
    other_record = await store.append_message(
        conversation_id=other_conv.id, role="user", content=other_content
    )

    index = WorkspaceHistoryIndex(store, workspace_id="default", paths=paths, terms=TERMS)
    for record in records:
        await index.index_message(
            message_id=record.id, conversation_id=main.id, content=record.content
        )
    await index.index_message(
        message_id=archived_record.id,
        conversation_id=archived.id,
        content=archived_record.content,
    )

    # 另一个工作区**确实建了索引**：这样"查不到"才是 workspace_id 收口的结果，
    # 而不是"因为根本没入库"这种假隔离。
    other_index = WorkspaceHistoryIndex(
        store,
        workspace_id="other",
        paths=workspace_paths_for(tmp_path, "other-key"),
        terms=TERMS,
    )
    await other_index.index_message(
        message_id=other_record.id, conversation_id=other_conv.id, content=other_content
    )

    return SimpleNamespace(
        paths=paths,
        other_paths=workspace_paths_for(tmp_path, "other-key"),
        db=db,
        store=store,
        main=main,
        archived=archived,
        other_conv=other_conv,
        records=records,
        archived_record=archived_record,
        other_record=other_record,
        index=index,
        other_index=other_index,
    )


@pytest.fixture
async def bench(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = await _build(tmp_path)
    try:
        yield built
    finally:
        await built.store.close()


def _search(bench, *, workspace_id: str = "default", **kwargs: object) -> HistorySearch:  # type: ignore[no-untyped-def]
    return HistorySearch(
        bench.db,
        workspace_id=workspace_id,
        paths=bench.paths if workspace_id == "default" else bench.other_paths,
        terms=TERMS,
        **kwargs,
    )


# ---------------------------------------------------------------- A22


def test_chinese_queries_hit_the_right_messages(bench) -> None:  # type: ignore[no-untyped-def]
    """A22：中文主题查询命中正确消息，且返回的是**原文**。"""

    with _search(bench) as search:
        hits = search.search("中文")
        assert [hit.message_id for hit in hits] == [bench.records[0].id]
        assert hits[0].content == "你回答一律用中文"

        assert [hit.message_id for hit in search.search("天气")] == [bench.records[4].id]
        assert "天气" in search.search("天气")[0].snippet

        invalidation_hits = search.search("invalidation_revision")
        assert [hit.message_id for hit in invalidation_hits] == [bench.records[3].id]
        assert invalidation_hits[0].role == "assistant"


def test_compound_terms_are_true_divergence_and_get_hit(bench) -> None:  # type: ignore[no-untyped-def]
    """内部合成术语必须命中；前提（样例真分叉）本身也被锁死，避免永真断言。"""

    tokenizer = history_tokenizer(bench.paths, terms=TERMS)
    for term in (COMPOUND_TERM, COMPOUND_TERM_2):
        whole = list(tokenizer.cut(term))
        search_mode = list(tokenizer.cut_for_search(term))
        assert len(search_mode) > len(whole), (
            f"{term} 不是分叉样例：cut={whole} cut_for_search={search_mode}"
        )

    with _search(bench) as search:
        assert [hit.message_id for hit in search.search(COMPOUND_TERM)] == [bench.records[1].id]
        assert [hit.message_id for hit in search.search(COMPOUND_TERM_2)] == [bench.records[2].id]
        assert {hit.message_id for hit in search.search("固定快照")} == {
            bench.archived_record.id,
            bench.records[3].id,
        }
    assert load_terms_dictionary(bench.paths) == sorted(TERMS)


def test_index_stores_tokens_but_results_return_original_text(bench) -> None:  # type: ignore[no-untyped-def]
    """FTS5 只存分词字段，原始消息单独保存；查询结果返回原文（PRD 11.5）。"""

    with _search(bench) as search:
        hit = search.search("中文")[0]
        assert hit.content == "你回答一律用中文"
        assert tokenize_for_index(hit.content, search.tokenizer) != hit.content


def test_workspace_isolation_never_crosses_workspaces(bench) -> None:  # type: ignore[no-untyped-def]
    """A 区的查询不得返回 B 区内容 —— 而 B 区的索引确实存在（真隔离，非假隔离）。"""

    with _search(bench, workspace_id="other") as other:
        other_hits = other.search("中文")
        assert [hit.message_id for hit in other_hits] == [bench.other_record.id]

    with _search(bench) as search:
        hits = search.search("中文")
        assert hits != []
        assert bench.other_record.id not in {hit.message_id for hit in hits}
        assert search.index_size() == len(bench.records) + 1


def test_archived_conversations_remain_searchable(bench) -> None:  # type: ignore[no-untyped-def]
    """会话归档保留历史搜索能力（PRD 11.5）。"""

    with _search(bench) as search:
        hits = search.search("固定快照")
        assert bench.archived_record.id in {hit.message_id for hit in hits}


def test_deleted_messages_disappear_from_results(bench) -> None:  # type: ignore[no-untyped-def]
    """消息被删除后索引残留行不得泄漏到结果里（检索侧 JOIN ``messages``）。"""

    connection = sqlite3.connect(str(bench.db))
    connection.execute("DELETE FROM messages WHERE id = ?", (bench.records[4].id,))
    connection.commit()
    connection.close()

    with _search(bench) as search:
        assert search.search("天气") == []


# ---------------------------------------------------------------- 索引指纹与重建


async def test_rebuild_marks_index_current_with_frozen_version_fields(bench) -> None:  # type: ignore[no-untyped-def]
    """重建后写入当前指纹；标记里带上模式/词典/术语版本（PRD 11.5 末段）。"""

    assert bench.index.needs_rebuild() is True  # 从未写过标记
    result = await bench.index.rebuild()

    assert result["indexed"] >= 1
    assert bench.index.needs_rebuild() is False
    marker = bench.index.read_marker()
    assert marker["fingerprint"] == bench.index.fingerprint
    assert marker["tokenizer_version"] == JIEBA_GENERATOR_VERSION
    assert marker["tokenizer_mode"] == JIEBA_TOKENIZER_MODE
    assert marker["dict_version"] == JIEBA_DICT_VERSION
    assert marker["terms_version"] == terms_version(TERMS)


async def test_ensure_fresh_rebuilds_only_when_fingerprint_changes(bench) -> None:  # type: ignore[no-untyped-def]
    assert await bench.index.ensure_fresh() == {"indexed": len(bench.records) + 1, "rebuilt": 1}
    assert await bench.index.ensure_fresh() == {"indexed": 0, "rebuilt": 0}


def test_terms_change_yields_a_different_fingerprint(bench) -> None:  # type: ignore[no-untyped-def]
    """术语变化 → 指纹变化 → 触发本区文本索引重建。"""

    extended = WorkspaceHistoryIndex(
        bench.store,
        workspace_id="default",
        paths=bench.paths,
        terms=[*TERMS, "新的内部术语"],
    )
    assert extended.fingerprint != bench.index.fingerprint

    plain = WorkspaceHistoryIndex(bench.store, workspace_id="default", paths=bench.paths)
    assert plain.fingerprint != bench.index.fingerprint


def test_fingerprint_tracks_tokenizer_and_terms() -> None:
    assert history_index_fingerprint(terms_version_value="0") != history_index_fingerprint(
        terms_version_value="abc"
    )
    assert terms_version([]) == "0"
    assert terms_version(["b", "a"]) == terms_version(["a", "b"])


def test_cache_file_name_carries_mode_dict_and_terms_version(tmp_path: Path) -> None:
    """``cache_file`` 必须包含词典版本（PRD 11.5），且随术语变化换文件。"""

    paths: WorkspacePaths = workspace_paths_for(tmp_path, "default")
    version = terms_version(["某个术语"])
    name = history_cache_file(paths, terms_version_value=version).name

    assert JIEBA_TOKENIZER_MODE in name
    assert JIEBA_DICT_VERSION in name
    assert version in name
    assert history_cache_file(paths, terms_version_value="0").name != name


def test_tokenizer_without_terms_reuses_the_shared_per_workspace_factory(
    tmp_path: Path,
) -> None:
    """没有内部术语时复用 T01 的每区工厂（同一实例），不另造第二份权威分词器。"""

    paths = workspace_paths_for(tmp_path, "default")
    first = history_tokenizer(paths)
    second = history_tokenizer(paths)
    third = history_tokenizer(paths, terms=["固定快照"])

    assert first is second
    assert third is not first


# ---------------------------------------------------------------- 查询构造安全


def test_query_expression_neutralizes_fts5_operators_and_quotes(bench) -> None:  # type: ignore[no-untyped-def]
    """``"`` / ``OR`` / ``NEAR`` / ``*`` 只能当普通词，不得改变语义或造出语法错误。"""

    tokenizer = history_tokenizer(bench.paths, terms=TERMS)
    expression = build_match_expression('中文" OR "天气', tokenizer)
    assert '"OR"' in expression

    star = build_match_expression("NEAR* OR 中文", tokenizer)
    assert "*" not in star
    assert '"NEAR"' in star

    with _search(bench) as search:
        # 当成普通词：不抛异常；因为索引里没有 "OR"/"NEAR" 词，结果为「无命中」而不是报错。
        assert search.search('中文" OR "天气') == []
        assert search.search("NEAR* OR 中文") == []


def test_quote_terms_escapes_embedded_quotes() -> None:
    assert quote_terms(['a"b']) == ['"a""b"']
    assert quote_terms(["", " ,。", "ok"]) == ['"ok"']


def test_empty_or_punctuation_only_query_raises(bench) -> None:  # type: ignore[no-untyped-def]
    with _search(bench) as search:
        for query in ["", "   ", "，。！？"]:
            with pytest.raises(HistorySearchError):
                search.search(query)


def test_missing_database_raises_clear_error(tmp_path: Path) -> None:
    paths = workspace_paths_for(tmp_path, "default")
    search = HistorySearch(
        tmp_path / "erag-data" / "no-such-app.db", workspace_id="default", paths=paths
    )
    try:
        with pytest.raises(HistorySearchError):
            search.search("中文")
    finally:
        search.close()


# ---------------------------------------------------------------- 过滤与读取


def test_date_filter_and_pagination(bench) -> None:  # type: ignore[no-untyped-def]
    with _search(bench) as search:
        assert search.search("中文", date_from="2999-01-01") == []
        assert search.search("中文", date_to="1999-01-01") == []
        assert search.search("中文", date_from="1999-01-01", date_to="2999-01-01") != []

        assert len(search.search("固定快照")) == 2
        first_page = search.search("固定快照", limit=1)
        second_page = search.search("固定快照", limit=1, offset=1)
        assert len(first_page) == 1 and len(second_page) == 1
        assert first_page[0].message_id != second_page[0].message_id
        assert search.search("固定快照", limit=1, offset=9) == []


def test_read_around_returns_surrounding_messages(bench) -> None:  # type: ignore[no-untyped-def]
    with _search(bench) as search:
        window = search.read_around(bench.main.id, bench.records[2].id, before=1, after=1)
        assert [item.sequence for item in window] == [2, 3, 4]
        assert {item.message_id for item in window} == {
            bench.records[1].id,
            bench.records[2].id,
            bench.records[3].id,
        }
        assert search.read_around(bench.main.id, "no-such-message") == []


# ---------------------------------------------------------------- 日期「含当日」


def test_normalize_date_bounds_expands_date_only_upper_bound() -> None:
    """``YYYY-MM-DD`` 上界必须扩成"含当日整天"，完整时间戳保持原样。"""

    assert normalize_date_bounds("2026-09-01", None) == ("2026-09-01", None)
    assert normalize_date_bounds(None, "2026-09-01") == (None, "2026-09-01T~")
    assert normalize_date_bounds(
        "2026-09-01T00:00:00Z", "2026-09-02T23:59:59Z"
    ) == ("2026-09-01T00:00:00Z", "2026-09-02T23:59:59Z")
    assert normalize_date_bounds("  ", "") == (None, None)
    assert normalize_date_bounds(None, None) == (None, None)


def test_date_only_to_bound_still_includes_that_whole_day(bench) -> None:  # type: ignore[no-untyped-def]
    """``date_to=<今天>`` 必须含今天的消息（``created_at`` 是带时刻的时间戳）。

    修复前 ``"2026-09-19T08:00:00Z" <= "2026-09-19"`` 为假，当天的消息会被整体排除 ——
    工具契约写明日期"含当日"，那是静默少给结果。
    """

    from app.agent.contracts import utc_now_iso

    today = utc_now_iso()[:10]
    with _search(bench) as search:
        assert search.search("中文", date_to=today) != []
        assert search.search("中文", date_from=today) != []
        assert search.search("中文", date_from=today, date_to=today) != []
        assert search.search("中文", date_to="1999-01-01") == []
        assert search.search("中文", date_from="2999-01-01") == []


# ---------------------------------------------------------------- 错误码


def test_error_codes_distinguish_invalid_query_from_unavailable_index(tmp_path: Path) -> None:
    """空查询 → ``invalid_query``（用户输入问题）；索引不可用 → ``service_unavailable``。

    两者都由 ``HistorySearchError`` 兜底，但工具层应按 ``error_code`` 分流，而不是拿消息
    文本做判断。
    """

    paths = workspace_paths_for(tmp_path, "default")
    search = HistorySearch(
        tmp_path / "erag-data" / "missing.db", workspace_id="default", paths=paths
    )
    try:
        # 索引不可用（连接失败）。
        with pytest.raises(HistoryIndexUnavailableError) as unavailable:
            search.search("中文")
        assert unavailable.value.error_code == "service_unavailable"
        assert isinstance(unavailable.value, HistorySearchError)

        # 空查询：即使索引也不可用，也应优先报"输入有问题"（坏请求不该被服务状态掩盖）。
        with pytest.raises(HistoryQueryError) as invalid:
            search.search("   ")
        assert invalid.value.error_code == "invalid_query"
        assert isinstance(invalid.value, HistorySearchError)
    finally:
        search.close()


# ---------------------------------------------------------------- 冻结接缝


async def test_seam_contract_shape_counts_and_cursor(bench) -> None:  # type: ignore[no-untyped-def]
    """t5 冻结的 ``ToolContext.history_search`` 接缝：形状、``total``、偏移游标。"""

    from app.api.deps import workspace_history_seam

    context = SimpleNamespace(workspace_id="default", paths=bench.paths, catalog=None)
    base = {"start_date": None, "end_date": None}

    first = await workspace_history_seam(
        context, "固定快照", {**base, "limit": 1, "cursor": None}
    )
    assert first["total"] == 2
    assert first["conversation_count"] == 2
    assert len(first["messages"]) == 1
    assert first["next_cursor"] == "1"
    hit = first["messages"][0]
    for key in (
        "message_id",
        "conversation_id",
        "sequence",
        "role",
        "created_at",
        "content",
        "snippet",
    ):
        assert key in hit
    # 返回形状是 messages，而不是内部实现用的 results。
    assert "results" not in first

    second = await workspace_history_seam(
        context, "固定快照", {**base, "limit": 1, "cursor": first["next_cursor"]}
    )
    assert len(second["messages"]) == 1
    assert second["messages"][0]["message_id"] != hit["message_id"]
    assert second["next_cursor"] is None


async def test_seam_has_no_phantom_next_page_on_exact_boundary(bench) -> None:  # type: ignore[no-untyped-def]
    """恰好取满最后一页时 ``next_cursor`` 必须是 ``None``。

    "本页条数 == limit 就认为还有下一页"的启发式在这里会给出一个指向空页的游标；本实现
    用真实 ``total`` 判定，因此不会。
    """

    from app.api.deps import workspace_history_seam

    context = SimpleNamespace(workspace_id="default", paths=bench.paths, catalog=None)
    exact = await workspace_history_seam(
        context,
        "固定快照",
        {"limit": 2, "cursor": None, "start_date": None, "end_date": None},
    )
    assert len(exact["messages"]) == 2
    assert exact["total"] == 2
    assert exact["next_cursor"] is None


async def test_seam_honours_dates_and_limit_bounds(bench) -> None:  # type: ignore[no-untyped-def]
    """接缝把日期传给检索（含当日），并把越界 ``limit`` 夹到合法区间。"""

    from app.agent.contracts import utc_now_iso
    from app.api.deps import workspace_history_seam
    from app.memory.history import MAX_HISTORY_LIMIT

    context = SimpleNamespace(workspace_id="default", paths=bench.paths, catalog=None)
    today = utc_now_iso()[:10]

    in_range = await workspace_history_seam(
        context,
        "固定快照",
        {"limit": 10, "cursor": None, "start_date": today, "end_date": today},
    )
    assert in_range["total"] == 2

    out_of_range = await workspace_history_seam(
        context,
        "固定快照",
        {"limit": 10, "cursor": None, "start_date": "2999-01-01", "end_date": None},
    )
    assert out_of_range["messages"] == [] and out_of_range["total"] == 0

    clamped = await workspace_history_seam(
        context,
        "固定快照",
        {"limit": 10_000, "cursor": None, "start_date": None, "end_date": None},
    )
    assert len(clamped["messages"]) == 2
    assert clamped["total"] == 2
    assert MAX_HISTORY_LIMIT >= 2


async def test_prune_orphans_removes_index_rows_without_messages(bench) -> None:  # type: ignore[no-untyped-def]
    """会话删除走 CASCADE，索引行不会自动消失；清理入口要能删掉它们。"""

    before = await bench.store.history_index_size()
    connection = sqlite3.connect(str(bench.db))
    connection.execute("DELETE FROM messages WHERE id = ?", (bench.records[4].id,))
    connection.commit()
    connection.close()

    assert await bench.index.prune_orphans() == 1
    assert await bench.store.history_index_size() == before - 1
