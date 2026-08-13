"""BM25 词法检索：字符 bigram 切词 + 词频/逆文档频率评分（混合检索优化③的词法支路）。

中文没有空格分词，这里用轻量近似：CJK 段拆成「单字 + 相邻字 bigram」（无需 jieba），
ASCII 词整体保留（小写归一）。纯内存计算，零出网；语料规模为千级块时建索引毫秒级、
单查询打分微秒级。
"""

from __future__ import annotations

import math
import re
from collections import Counter

#: 切词正则：ASCII 词（字母/数字/下划线/@/. /-）整体保留；CJK 段拆单字+bigram。
_ASCII_WORD_RE = re.compile(r"[0-9a-zA-Z_@.\-]+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """轻量切词：ASCII 词整体 + CJK 相邻 bigram，小写归一。

    中文无空格，用「相邻字 bigram」作为词法单元（比单字更抗噪音：单字「的/了/是」
    等函数词会产生大量假命中）；单字 CJK 不产 token。示例：
    ``Cider 埋点平台`` -> ["cider", "埋点","点平","平台"]。
    """
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
            df.update(counter)
        n = len(corpus)
        self._n = n
        self._avgdl = sum(self._doc_len) / n if n else 0.0
        self._k1 = k1
        self._b = b
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """按 BM25 打分返回 (文档下标, 分数) 降序 top_k；空语料/空查询/无命中返回 []。"""
        if self._n == 0 or top_k <= 0:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        results: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            doc_len = self._doc_len[idx]
            denom = 1.0 - self._b + self._b * (doc_len / self._avgdl) if self._avgdl else 1.0
            score = 0.0
            for term in query_tokens:
                tf = doc_tokens.get(term, 0)
                if tf == 0 or term not in self._idf:
                    continue
                score += self._idf[term] * (tf * (self._k1 + 1)) / (tf + self._k1 * denom)
            if score > 0:
                results.append((idx, score))
        results.sort(key=lambda pair: (-pair[1], pair[0]))
        return results[:top_k]
