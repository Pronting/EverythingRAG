"""纯向量检索器：查询向量 -> vectorstore.query(candidate_k) -> top-context_k 带来源块。

对齐技术选型决策文档 T1/T6：candidate_k=20 召回候选 / context_top_k=5 入上下文；
混合检索（BM25）+ 重排为 v0.2/v1.0 演进，本模块不做。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.vectorstore.base import VectorStore


class RetrievalError(Exception):
    """检索层领域错误（空查询 / 未建索引），消息必须清晰可读。"""


@dataclass(frozen=True)
class RetrievedChunk:
    """命中块：block_id + 原文 + 相似度 + 来源，供问答层引用出处。"""

    block_id: str
    text: str
    # Dense cosine similarity is optional for BM25-only and parent-expansion results.  Keeping
    # the missing value as None prevents API consumers from presenting "0%" as a real score.
    similarity: float | None
    source_file: str
    platform: str
    chunk_type: str
    anchor: str | None
    heading_path: str | None
    metadata: dict[str, Any]  # 原始元数据（标量 dict）
    match_type: str = "dense"
    # A short atomic block remains the citation target, while this optional same-section view
    # gives the model and source panel enough surrounding meaning to interpret it correctly.
    context_text: str | None = None


# 缺失字段的安全默认值（真实索引源_file 必有；mock/历史数据可缺）
_DEFAULT_SOURCE_FILE = ""
_DEFAULT_PLATFORM = "local"
_DEFAULT_CHUNK_TYPE = "text"

#: 相关度门控缺省阈值：相似度低于该值的命中视为无关，不进上下文。
#: 修复根因 2——库外/常识问题不再被无关 chunk 污染（生产装配处注入该值，
#: 核心类缺省 None 保持向后兼容）。
#: 注：0.55 原为 bge-m3（1024 维余弦）校准；切换 Qwen3-Embedding-8B（4096 维）
#: 后无关内容基线相似度升至 0.38~0.57，0.55 已拦不住噪声，实测校准上调到 0.60。
DEFAULT_MIN_SIMILARITY = 0.60


class VectorRetriever:
    """按查询向量召回带来源块（纯向量：候选放大 + 上下文裁剪 + 相关度门控）。

    空查询 / 未建索引抛 RetrievalError；有数据但无匹配返回 []（非错误）。
    min_similarity 非 None 时：相似度低于阈值的命中被过滤，全部低于则返回 []。
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        candidate_k: int = 20,
        context_top_k: int = 5,
        min_similarity: float | None = None,
    ) -> None:
        self._vectorstore = vectorstore
        self._candidate_k = candidate_k
        self._context_top_k = context_top_k
        self._min_similarity = min_similarity

    def retrieve(
        self,
        query_text: str,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """向量检索 top-context_k；query_text 供混合检索词法支路使用（纯向量忽略）。"""
        self._reject_empty_query(query_vector)
        if self._vectorstore.count() == 0:
            raise RetrievalError("知识库尚未建立索引，请先导入并同步文档")
        hits = self._vectorstore.query(query_vector, self._candidate_k, where=where)
        return self._rank_and_map(hits)

    def _reject_empty_query(self, query_vector: list[float]) -> None:
        """空列表 / 全零向量 -> 领域错误（嵌入失败或未嵌入的查询向量）。"""
        if not query_vector or all(value == 0.0 for value in query_vector):
            raise RetrievalError("查询向量为空，请先对查询文本进行嵌入")

    def _rank_and_map(self, hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """按 similarity 降序（同分按 block_id 稳定），门控 + 裁剪到 context_top_k 后映射。"""
        if not hits:
            return []
        ranked = sorted(
            hits,
            key=lambda hit: (-float(hit.get("similarity", 0.0)), str(hit.get("block_id", ""))),
        )
        if self._min_similarity is not None:
            ranked = [
                hit for hit in ranked if float(hit.get("similarity", 0.0)) >= self._min_similarity
            ]
        return [self._to_chunk(hit) for hit in ranked[: self._context_top_k]]

    def _to_chunk(self, hit: dict[str, Any]) -> RetrievedChunk:
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        return RetrievedChunk(
            block_id=str(hit.get("block_id", "")),
            text=str(hit.get("text", "")),
            similarity=float(hit.get("similarity", 0.0)),
            source_file=_get_str(metadata, "source_file", _DEFAULT_SOURCE_FILE),
            platform=_get_str(metadata, "platform", _DEFAULT_PLATFORM),
            chunk_type=_get_str(metadata, "chunk_type", _DEFAULT_CHUNK_TYPE),
            anchor=_get_str(metadata, "anchor"),
            heading_path=_get_str(metadata, "heading_path"),
            metadata=dict(metadata),
        )


def _get_str(metadata: dict[str, Any], key: str, default: str | None = None) -> str | None:
    """取元数据标量字符串字段；缺失 / 非字符串安全回退默认。"""
    value = metadata.get(key)
    return value if isinstance(value, str) else default
