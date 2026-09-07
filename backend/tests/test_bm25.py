"""BM25 词法检索测试：字符 bigram 切词 + 词频/逆文档频率评分。

隐私红线：纯内存计算，无网络；不依赖 jieba（中文近似用「单字+相邻字 bigram」切词）。
"""

from __future__ import annotations

import socket

import pytest

from app.retrieval.bm25 import BM25Index, tokenize

# ---------------------------------------------------------------- 1. 切词


def test_tokenize_ascii_words_and_cjk() -> None:
    """ASCII 词整体成 token；CJK 拆为相邻 bigram（单字弃，抗函数词噪音）；大小写归一。"""
    toks = tokenize("Cider 埋点平台 platform")
    assert "cider" in toks
    assert "platform" in toks
    assert "埋点" in toks  # CJK bigram
    assert "点平" in toks
    assert "平台" in toks
    assert "埋" not in toks and "点" not in toks  # 单字不产 token


def test_tokenize_handles_single_cjk_char() -> None:
    """单个 CJK 字符不产 token（无相邻 bigram）。"""
    assert tokenize("木") == []
    assert tokenize("量入为出") == ["量入", "入为", "为出"]


def test_tokenize_english_phrase() -> None:
    """英文短语按空白拆词。"""
    toks = tokenize("RocketMQ nameserver 部署")
    assert "rocketmq" in toks
    assert "nameserver" in toks


def test_tokenize_handles_punct() -> None:
    """标点不产生 token，不崩溃。"""
    toks = tokenize("实·时 风控（方案）！")
    assert toks


# ---------------------------------------------------------------- 2. 检索相关性


def test_bm25_ranks_exact_term_doc_first() -> None:
    """包含查询词最多的文档得分最高、排第一。"""
    index = BM25Index(
        [
            "这是一个关于量子物理的科普文章，讨论量子态叠加。",
            "埋点平台通过 SPM 管理埋点生命周期，支持简单可查询。",
            "MySQL 的 MVCC 用 undolog 实现，PG 用写时复制。",
        ]
    )
    top = index.search("埋点平台 管理 埋点", top_k=3)
    assert top[0][0] == 1  # 第二篇含「埋点平台」「埋点」最高


def test_bm25_top_k_bounds() -> None:
    """top_k 限制返回条数且按分数降序。"""
    index = BM25Index(["alpha beta gamma", "beta gamma", "gamma"])
    top = index.search("gamma", top_k=2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]  # 分数非增


def test_repeated_term_does_not_corrupt_document_frequency() -> None:
    """同词在标题/正文重复不能令 df>文档数、IDF 变负并吞掉真实命中。"""

    index = BM25Index(["实时风控方案 实时风控方案 限流", "缓存设计 普通说明"])
    hits = index.search("实时风控方案 限流", top_k=2)
    assert hits
    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_detailed_hit_exposes_coverage_and_rarity_for_safe_rescue() -> None:
    """详细词法证据可区分完整稀有命中和只撞上常见词的弱命中。"""

    index = BM25Index(
        [
            "common common SPR-2024-0001 shopping cart",
            "common unrelated document",
            "common another document",
        ]
    )
    hits = index.search_detailed("common SPR-2024-0001", top_k=3)
    by_index = {hit.index: hit for hit in hits}
    assert by_index[0].coverage == pytest.approx(1.0)
    assert by_index[0].matched_terms == 2
    assert by_index[0].rarity > by_index[1].rarity
    assert by_index[0].identifier_query_terms == 2
    assert by_index[0].identifier_matched_terms == 2
    assert by_index[1].identifier_query_terms == 2
    assert by_index[1].identifier_matched_terms == 1


def test_bm25_no_overlap_scores_zero() -> None:
    """查询与语料无重叠词 -> 分数 0，排序稳定。"""
    index = BM25Index(["中文 语料 内容", "another english doc"])
    top = index.search("完全不存在的词 xyz", top_k=2)
    assert all(score == 0 for _, score in top)


# ---------------------------------------------------------------- 3. 边界


def test_bm25_empty_corpus() -> None:
    """空语料：不抛错，返回空列表。"""
    index = BM25Index([])
    assert index.search("query", top_k=5) == []


def test_bm25_single_doc() -> None:
    """单文档语料：命中返回它，未命中返回空。"""
    index = BM25Index(["唯一 文档 内容"])
    assert len(index.search("文档", top_k=5)) == 1
    assert index.search("无关", top_k=5) == []


def test_bm25_query_without_token() -> None:
    """空查询字符串：返回空列表，不抛错。"""
    index = BM25Index(["some content"])
    assert index.search("", top_k=5) == []


# ---------------------------------------------------------------- 4. 零出网


def test_bm25_zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切 socket，BM25 建索引 + 检索仍成功 => 零出网。"""

    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected outbound network call")

    monkeypatch.setattr(socket.socket, "connect", deny)
    index = BM25Index(["埋点平台 生命周期 可查询", "事故报告 复盘"])
    assert len(index.search("埋点", top_k=2)) == 1
