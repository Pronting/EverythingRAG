"""检索层接口（对齐技术选型决策文档 T1/T6）。

MVP 仅纯向量检索（candidate_k=20 候选 / context_top_k=5 入上下文）；
混合检索（BM25）+ 重排为 v0.2/v1.0 演进，接口预留。
"""
from __future__ import annotations

from typing import Any, Protocol

from app.retrieval.vector_retriever import RetrievedChunk


class Retriever(Protocol):
    """统一检索接口。MVP 实现 VectorRetriever（纯向量，无混合/重排）。"""

    def retrieve(
        self,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """按查询向量检索，返回带来源的命中块（候选放大 + 上下文裁剪）。

        candidate_k=20：召回候选数；context_top_k=5：入上下文的块数；
        二者由 VectorRetriever 构造参数决定，接口不暴露 top_k。
        """
        ...
