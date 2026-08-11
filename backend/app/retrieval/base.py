"""检索层接口（对齐技术选型决策文档 T1/T6）。

MVP 仅纯向量检索（candidate_k=20 / context_top_k=5）；
混合检索（BM25）+ 重排为 v0.2/v1.0 演进，接口预留。
"""
from __future__ import annotations

from typing import Any, Protocol


class Retriever(Protocol):
    """统一检索接口。MVP 实现 VectorRetriever。"""

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按查询向量检索，返回命中块（含 metadata 与相似度）。
        top_k 对齐决策：candidate_k=20（候选）/ context_top_k=5（入上下文）。
        """
        ...
