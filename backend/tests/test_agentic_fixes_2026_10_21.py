"""回归测试：2026-10-21 Agentic 评测的 P0 修复。

覆盖：核验数值容差、产品名不再清零知识检索、联网 reason 不再虚构、
显式联网仍保留本地对照、产品意图改用通用 Agent、网页身份兜底。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import AgentKind, Evidence, EvidenceSourceType, ReasoningLevel
from app.agent.planner import TASK_SOURCES, TaskSource
from app.agent.research import (
    GATE_FALLBACK_SIMILARITY,
    _CATALOG_LIST_PATTERN,
    _web_evidence_supports,
)
from app.agent.runtime import RUN_ACTIVE_TIMEOUT_SECONDS
from app.agent.routing import RoutingInput, evaluate_route_rules
from app.agent.verification import _numbers_equal, _unsupported_numbers


def test_numeric_rounding_is_supported() -> None:
    assert _numbers_equal("96", "96.1")
    assert _numbers_equal("96.06", "96.1")
    assert _unsupported_numbers({"96"}, {"96.1"}) == []


def test_numeric_years_are_not_fuzzy() -> None:
    assert not _numbers_equal("2024", "2023")
    assert _unsupported_numbers({"2024"}, {"2023"}) == ["2024"]


def test_percent_decimal_equivalence() -> None:
    assert _numbers_equal("0.96", "96")


def _decide(question: str):
    return evaluate_route_rules(
        RoutingInput(question=question, file_count=5, chunk_count=20)
    ).decision


def test_product_name_keeps_knowledge_search() -> None:
    decision = _decide("Kubernetes 里最小的调度单位是什么？")
    assert decision.needs_knowledge is True


def test_product_web_reason_is_not_fake_web_request() -> None:
    decision = _decide("Kubernetes 里最小的调度单位是什么？")
    assert "根据请求联网" not in (decision.reason_summary or "")


def test_explicit_web_request_keeps_both_sources() -> None:
    decision = _decide("联网搜索 Kubernetes 最小的调度单位")
    assert decision.needs_web is True
    assert decision.needs_knowledge is True


def test_product_intent_uses_general_agent() -> None:
    assert TASK_SOURCES[TaskSource.PRODUCT][0] is AgentKind.GENERAL


def test_web_identity_fallback_accepts_official_page() -> None:
    evidence = Evidence(
        evidence_id="W1",
        workspace_id="ws",
        run_id="run",
        source_type=EvidenceSourceType.WEB,
        source_ref="https://kubernetes.io/zh-cn/docs/concepts/workloads/pods/",
        content="Pod 是 Kubernetes 中最小的可部署计算单元。",
        content_hash="0" * 64,
        retrieved_at="2026-10-21T00:00:00Z",
        document_title="Pods | Kubernetes",
    )
    assert _web_evidence_supports("Kubernetes 里最小的调度单位是什么？", evidence)


def test_medium_reasoning_is_accepted_as_high() -> None:
    assert ReasoningLevel("medium") is ReasoningLevel.HIGH
    assert ReasoningLevel("MID") is ReasoningLevel.HIGH


def test_run_timeout_fits_model_roundtrips() -> None:
    # 整轮上限必须显著高于单次模型往返；关闭思考时不再有 30s 的整轮硬超时。
    assert RUN_ACTIVE_TIMEOUT_SECONDS >= 120


def test_gate_fallback_is_below_production_gate() -> None:
    assert 0 < GATE_FALLBACK_SIMILARITY < 0.60


def test_catalog_listing_questions_are_detected() -> None:
    assert _CATALOG_LIST_PATTERN.search("记忆系统有哪几篇文章")
    assert _CATALOG_LIST_PATTERN.search("列出关于检索的文档")
    assert not _CATALOG_LIST_PATTERN.search("Kubernetes 最小的调度单位是什么")


def test_status_endpoint_uses_existing_helper() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "main.py"
    ).read_text(encoding="utf-8")
    assert "api_deps.get_workspace_vector_store(" in source
    assert "api_deps.workspace_vector_store(" not in source


