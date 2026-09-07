"""BM25 词法检索：字符 bigram 切词 + 词频/逆文档频率评分（混合检索优化③的词法支路）。

中文没有空格分词，这里用轻量近似：CJK 段拆成「单字 + 相邻字 bigram」（无需 jieba），
ASCII 词整体保留（小写归一）。纯内存计算，零出网；语料规模为千级块时建索引毫秒级、
单查询打分微秒级。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.retrieval.percentiles import canonical_percentiles

#: 切词正则：ASCII 词（字母/数字/下划线/@/. /-）整体保留；CJK 段拆单字+bigram。
_ASCII_WORD_RE = re.compile(r"[0-9a-zA-Z_@.\-]+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")

# These terms describe question form or extremely broad relations rather than a knowledge entity.
# They still participate in BM25 scoring/ranking, but cannot by themselves authorize a lexical-
# only rescue. This blocks high-IDF accidental bigrams such as ``的应`` and ``何实`` without
# weakening exact product names, identifiers or domain phrases.
_LOW_INFORMATION_TERMS = frozenset(
    {
        "中的",
        "的应",
        "的作",
        "的对",
        "的方",
        "的内",
        "的是",
        "何实",
        "如何",
        "怎么",
        "怎样",
        "什么",
        "是什",
        "发生",
        "生了",
        "了什",
        "之前",
        "前的",
        "在知",
        "知识",
        "识库",
        "库中",
        "哪些",
        "是否",
        "需要",
        "通过",
        "进行",
        "使用",
        "可以",
        "分别",
        "相关",
        "内容",
        "详情",
        "介绍",
        "说明",
        "实现",
        "系统",
        "应用",
        "作用",
        "机制",
        "控制",
        "计算",
        "分析",
        "平均",
    }
)

_ASCII_QUERY_STOP_TERMS = frozenset(
    {"and", "are", "for", "from", "how", "in", "into", "is", "of", "the", "to", "what"}
)


@dataclass(frozen=True)
class BM25Hit:
    """One lexical hit with corpus-independent gate evidence.

    ``coverage`` is the fraction of unique query terms present in the document. ``rarity``
    compares matched-term IDF with the maximum possible IDF for this corpus.  Hybrid retrieval
    uses both values to distinguish exact lexical rescue from a one-bigram coincidence.
    """

    index: int
    score: float
    matched_terms: int
    query_terms: int
    coverage: float
    rarity: float
    informative_matched_terms: int
    informative_query_terms: int
    informative_coverage: float
    identifier_matched_terms: int
    identifier_query_terms: int
    anchor_matched_terms: int
    anchor_query_terms: int


def tokenize(text: str) -> list[str]:
    """轻量切词：ASCII 词整体 + CJK 相邻 bigram，小写归一。

    中文无空格，用「相邻字 bigram」作为词法单元（比单字更抗噪音：单字「的/了/是」
    等函数词会产生大量假命中）；单字 CJK 不产 token。示例：
    ``Cider 埋点平台`` -> ["cider", "埋点","点平","平台"]。
    """
    text = canonical_percentiles(text)
    tokens: list[str] = []
    for run in _ASCII_WORD_RE.findall(text) + _CJK_RUN_RE.findall(text):
        lowered = run.lower()
        if _ASCII_WORD_RE.fullmatch(lowered):
            tokens.append(lowered)
            continue
        chars = list(lowered)
        if len(chars) >= 2:
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
    return tokens


class BM25Index:
    """BM25 检索索引：预计算 idf 与文档长度，供查询打分。

    参数 k1=1.5 / b=0.75 为 BM25 常用缺省。search 返回 (文档下标, 分数) 降序 top_k；
    仅返回分数 > 0 的命中（无重叠词 -> 空列表）。
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._doc_tokens: list[Counter[str]] = []
        self._doc_len: list[int] = []
        df: Counter[str] = Counter()
        for text in corpus:
            toks = tokenize(text)
            counter = Counter(toks)
            self._doc_tokens.append(counter)
            self._doc_len.append(len(toks))
            # Document frequency counts a term at most once per document.  Updating with the
            # Counter itself incorrectly accumulates term frequency, which can make df > n and
            # produce negative IDF for repeated title/body terms.
            df.update(counter.keys())
        n = len(corpus)
        self._n = n
        self._avgdl = sum(self._doc_len) / n if n else 0.0
        self._k1 = k1
        self._b = b
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._max_idf = math.log(1 + (n - 0.5) / 0.5) if n else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """按 BM25 打分返回 (文档下标, 分数) 降序 top_k；空语料/空查询/无命中返回 []。"""
        return [(hit.index, hit.score) for hit in self.search_detailed(query, top_k)]

    def search_detailed(self, query: str, top_k: int) -> list[BM25Hit]:
        """Return lexical hits with score, query coverage and matched-term rarity."""

        if self._n == 0 or top_k <= 0:
            return []
        query_token_sequence = tokenize(query)
        query_tokens = set(query_token_sequence)
        if not query_tokens:
            return []
        informative_query_tokens = {
            term for term in query_tokens if term not in _LOW_INFORMATION_TERMS
        }
        identifier_query_tokens = {
            term for term in informative_query_tokens if _is_identifier_term(term)
        }
        informative_sequence = list(
            dict.fromkeys(
                term for term in query_token_sequence if term in informative_query_tokens
            )
        )
        anchor_width = min(6, max(1, math.ceil(len(informative_sequence) * 0.4)))
        anchor_query_tokens = set(informative_sequence[:anchor_width])
        results: list[BM25Hit] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            doc_len = self._doc_len[idx]
            denom = 1.0 - self._b + self._b * (doc_len / self._avgdl) if self._avgdl else 1.0
            score = 0.0
            matched: list[str] = []
            for term in query_tokens:
                tf = doc_tokens.get(term, 0)
                if tf == 0 or term not in self._idf:
                    continue
                matched.append(term)
                score += self._idf[term] * (tf * (self._k1 + 1)) / (tf + self._k1 * denom)
            if score > 0:
                coverage = len(matched) / len(query_tokens)
                informative_matched = [
                    term for term in matched if term in informative_query_tokens
                ]
                informative_coverage = (
                    len(informative_matched) / len(informative_query_tokens)
                    if informative_query_tokens
                    else 0.0
                )
                identifier_matched_terms = sum(
                    _is_identifier_term(term) for term in informative_matched
                )
                rarity = (
                    sum(self._idf[term] for term in matched) / len(matched) / self._max_idf
                    if matched and self._max_idf
                    else 0.0
                )
                results.append(
                    BM25Hit(
                        index=idx,
                        score=score,
                        matched_terms=len(matched),
                        query_terms=len(query_tokens),
                        coverage=coverage,
                        rarity=rarity,
                        informative_matched_terms=len(informative_matched),
                        informative_query_terms=len(informative_query_tokens),
                        informative_coverage=informative_coverage,
                        identifier_matched_terms=identifier_matched_terms,
                        identifier_query_terms=len(identifier_query_tokens),
                        anchor_matched_terms=len(set(matched) & anchor_query_tokens),
                        anchor_query_terms=len(anchor_query_tokens),
                    )
                )
        results.sort(key=lambda hit: (-hit.score, hit.index))
        return results[:top_k]


def _is_identifier_term(term: str) -> bool:
    """Whether an ASCII token is specific enough to act as exact lexical evidence."""

    if not _ASCII_WORD_RE.fullmatch(term):
        return False
    # A bare number such as the ``99`` in ``99 分位`` is a measurement, not an identifier.
    # Concrete identifiers must contain at least one letter (ADB, K8S, SPR-123, ...).
    return (
        len(term) >= 2
        and term not in _ASCII_QUERY_STOP_TERMS
        and re.search(r"[a-zA-Z]", term) is not None
    )
