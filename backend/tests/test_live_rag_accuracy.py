"""可选的真实索引召回门禁：至少 66 个正例 + 10 个负例逐条显示为独立测试。

默认 CI 不连接用户配置的嵌入服务；发布或本地验收时设置
``EVERYTHING_RAG_RUN_LIVE_EVAL=1``。fixture 只读当前 Chroma，查询嵌入按批发送。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.api.deps import get_embedder, get_vector_store
from app.evaluation.rag_quality import (
    NegativeCase,
    PositiveCase,
    QualityReport,
    evaluate_retrieval,
    load_eval_cases,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY

ROOT = Path(__file__).resolve().parents[2]
POSITIVES, NEGATIVES = load_eval_cases(ROOT / "scripts" / "rag_eval_cases.json")
RUN_LIVE = os.environ.get("EVERYTHING_RAG_RUN_LIVE_EVAL") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_LIVE,
    reason="设置 EVERYTHING_RAG_RUN_LIVE_EVAL=1 才运行真实索引与云端查询嵌入",
)


def _case_id(case: PositiveCase | NegativeCase) -> str:
    return case.id


@pytest.fixture(scope="module")
def live_report() -> QualityReport:
    embedder = get_embedder()
    vectorstore = get_vector_store()
    retriever = HybridRetriever(
        vectorstore,
        context_top_k=8,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )
    return evaluate_retrieval(
        embedder=embedder,
        retriever=retriever,
        positives=POSITIVES,
        negatives=NEGATIVES,
        embed_batch_size=16,
    )


@pytest.mark.parametrize("case", POSITIVES, ids=_case_id)
def test_live_positive_case_hits_expected_block(
    case: PositiveCase, live_report: QualityReport
) -> None:
    outcome = next(item for item in live_report.positive_outcomes if item.id == case.id)
    assert outcome.matched_rank is not None and outcome.matched_rank <= 8, (
        f"{outcome.id} 未命中期望来源+标题+正文证据；Top-8={outcome.candidates}"
    )


@pytest.mark.parametrize("case", NEGATIVES, ids=_case_id)
def test_live_negative_case_has_no_context(case: NegativeCase, live_report: QualityReport) -> None:
    outcome = next(item for item in live_report.negative_outcomes if item.id == case.id)
    assert outcome.retrieved_count == 0, f"{outcome.id} 错误放行上下文：{outcome.candidates}"
