"""Privacy-safe structured diagnostics for retrieval pipelines.

两类记录，都不含任何正文、绝对路径、密钥或服务原始响应体：

1. 混合检索 trace（``RetrievalTrace``）：融合、门控与选择的内部信号；
2. 相关性筛选 trace（``RelevanceTrace``，PRD §9）：每轮记录 ``trace_id``、scope 摘要、
   知识版本、档案、候选 ID 的哈希引用（不可逆 ``block_ref`` / ``candidate_ref``）、通道排名、
   四种分数的独立字段、视图哈希、排除原因、服务调用数量、耗时、usage 与最终顺序。

线上与评测使用同一份定义：``block_ref`` 与评测导出的 ``blk_<sha256[:16]>`` 完全一致，
同一 block 在两侧得到同一个引用。完整内容对照只允许存在于用户主动运行的本地诊断文件。
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.retrieval.vector_retriever import RetrievedChunk


@dataclass(frozen=True)
class RetrievalSignalTrace:
    """One ranking signal emitted by one dense or lexical query route."""

    route: str
    rank: int
    score: float
    coverage: float | None = None
    rarity: float | None = None
    informative_matched_terms: int | None = None
    informative_query_terms: int | None = None
    informative_coverage: float | None = None
    identifier_matched_terms: int | None = None
    identifier_query_terms: int | None = None
    anchor_matched_terms: int | None = None
    anchor_query_terms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "rank": self.rank,
            "score": self.score,
            "coverage": self.coverage,
            "rarity": self.rarity,
            "informative_matched_terms": self.informative_matched_terms,
            "informative_query_terms": self.informative_query_terms,
            "informative_coverage": self.informative_coverage,
            "identifier_matched_terms": self.identifier_matched_terms,
            "identifier_query_terms": self.identifier_query_terms,
            "anchor_matched_terms": self.anchor_matched_terms,
            "anchor_query_terms": self.anchor_query_terms,
        }


@dataclass(frozen=True)
class RetrievalCandidateTrace:
    """Fusion, gating and selection outcome for one opaque candidate block."""

    block_id: str
    signals: tuple[RetrievalSignalTrace, ...]
    fused_score: float
    dense_similarity: float | None
    gate_reason: str
    gate_threshold: float | None
    selected: bool
    selection_reason: str | None
    #: 可选：候选对应证据视图的 ``payload_hash``（PRD §9 的「视图哈希」）。
    #: 只保存哈希本身，正文与路径永不进入 trace。
    evidence_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "signals": [signal.to_dict() for signal in self.signals],
            "fused_score": self.fused_score,
            "dense_similarity": self.dense_similarity,
            "gate_reason": self.gate_reason,
            "gate_threshold": self.gate_threshold,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True)
class RetrievalTrace:
    """Complete trace for one call, excluding all user and document content."""

    variant_kinds: tuple[str, ...]
    candidates: tuple[RetrievalCandidateTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_kinds": list(self.variant_kinds),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved chunks paired with an explicitly requested diagnostic trace."""

    chunks: tuple[RetrievedChunk, ...]
    trace: RetrievalTrace


# --------------------------------------------------------------------------------------
# 相关性筛选诊断 trace（PRD §9）
# --------------------------------------------------------------------------------------

#: 相关性 trace 的阶段标记（候选从进入窗口到最终入选的所在环节）。
TRACE_STAGES = ("window", "score", "select")


def block_ref(block_id: str) -> str:
    """把内部 block_id 转成稳定、不可逆、且不含标题信息的日志引用。

    与评测导出（``rag_quality._block_ref``）使用同一算法：同一 block 在线上 trace 与
    评测报告中得到同一个引用，便于对照（PRD §9）。
    """

    digest = hashlib.sha256(str(block_id).encode("utf-8", errors="replace")).hexdigest()
    return f"blk_{digest[:16]}"


def candidate_ref(candidate_id: str) -> str:
    """证据视图候选 ID 的不可逆引用（``cand_<sha256[:16]>``）。"""

    digest = hashlib.sha256(str(candidate_id).encode("utf-8", errors="replace")).hexdigest()
    return f"cand_{digest[:16]}"


def new_trace_id() -> str:
    """生成本轮诊断 trace 标识（随机、不含时间戳/用户信息/问题）。"""

    return f"rt_{uuid.uuid4().hex[:16]}"


def _optional_float(value: Any) -> float | None:
    """只保留有限数值；其余（缺失、NaN、字符串）一律为 None，不猜测。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_int(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= minimum else None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _reason_text(value: Any) -> str | None:
    """排除原因统一存字符串（枚举值）；未知或空值存 None。"""

    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw) if isinstance(raw, str) and raw.strip() else None


@dataclass(frozen=True)
class RelevanceCandidateTrace:
    """单个候选在相关性筛选中的可重放记录（不含正文、路径与原始 ID）。

    * ``block_ref`` / ``candidate_ref``：不可逆引用（PRD §9）；
    * 四种分数分列：``dense_similarity``、``bm25_score``、``rrf_score``、``rerank_score``；
    * 通道排名分列：``dense_rank``、``bm25_rank``、``rrf_rank``；
    * ``view_hash``：评分文本的 ``payload_hash``（视图哈希）；
    * ``stage`` + ``exclusion_reason``：候选止步的环节与固定排除原因；
    * ``final_position``：1-based 最终顺序（未入选为 None）。
    """

    block_ref: str
    candidate_ref: str
    stage: str = "score"
    view_kind: str = "block"
    view_hash: str | None = None
    dense_similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    exclusion_reason: str | None = None
    final_position: int | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_ref": self.block_ref,
            "candidate_ref": self.candidate_ref,
            "stage": self.stage,
            "view_kind": self.view_kind,
            "view_hash": self.view_hash,
            "dense_similarity": self.dense_similarity,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf_rank": self.rrf_rank,
            "exclusion_reason": self.exclusion_reason,
            "final_position": self.final_position,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class RelevanceTrace:
    """一轮相关性筛选的完整脱敏记录（PRD §9）。

    ``scope_summary`` 只包含 scope 标识、知识版本与计数；``usage`` 为服务返回的计量
    （未返回时为 ``None``，绝不估造）；``final_order`` 是最终来源顺序的 ``block_ref`` 列表。
    """

    trace_id: str
    scope_id: str
    scope_summary: str
    index_revision: str
    profile_id: str
    status: str
    candidates: tuple[RelevanceCandidateTrace, ...] = ()
    final_order: tuple[str, ...] = ()
    candidate_count: int = 0
    window_size: int = 0
    window_truncated: bool = False
    provider_call_count: int = 0
    rerank_latency_ms: float | None = None
    total_latency_ms: float | None = None
    provider_trace_id: str | None = None
    request_id: str | None = None
    usage: Mapping[str, Any] | None = None
    error_code: str | None = None
    missing_headings: int = 0

    @property
    def selected_count(self) -> int:
        return len(self.final_order)

    def to_dict(self) -> dict[str, Any]:
        """唯一的对外投影：只含引用、枚举、计数与耗时（无正文/路径/密钥）。"""

        return {
            "trace_id": self.trace_id,
            "scope_summary": self.scope_summary,
            "scope_id": self.scope_id,
            "index_revision": self.index_revision,
            "profile_id": self.profile_id,
            "status": self.status,
            "candidate_count": self.candidate_count,
            "window_size": self.window_size,
            "window_truncated": self.window_truncated,
            "provider_call_count": self.provider_call_count,
            "rerank_latency_ms": self.rerank_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "provider_trace_id": self.provider_trace_id,
            "request_id": self.request_id,
            "usage": dict(self.usage) if self.usage is not None else None,
            "error_code": self.error_code,
            "missing_headings": self.missing_headings,
            "selected_count": self.selected_count,
            "final_order": list(self.final_order),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def scope_summary(
    scope_id: str,
    index_revision: str,
    *,
    candidate_count: int = 0,
    window_size: int = 0,
) -> str:
    """scope 摘要：只写标识与计数，不写任何正文或路径。"""

    return (
        f"scope={scope_id or 'global'}; revision={index_revision or 'unknown-index'}; "
        f"candidates={candidate_count}; window={window_size}"
    )


def _scope_value(scope: Any, attribute: str, default: str) -> str:
    value = getattr(scope, attribute, None)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _status_from_selection(selection: Any) -> str:
    raw = getattr(getattr(selection, "status", None), "value", None)
    return {
        "selected": "filtered",
        "empty": "empty",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(raw or "", "failed")


def _candidate_block_id(candidate: Any) -> str:
    return str(
        getattr(candidate, "primary_block_id", None)
        or getattr(candidate, "block_id", None)
        or getattr(candidate, "candidate_id", "")
    )


def build_relevance_trace(
    *,
    trace_id: str,
    scope: Any,
    profile_id: str,
    candidates: Sequence[Any],
    selection: Any | None = None,
    window: Any | None = None,
    scores: Mapping[str, float] | None = None,
    status: str | None = None,
    provider_call_count: int = 0,
    rerank_latency_ms: float | None = None,
    total_latency_ms: float | None = None,
    provider_trace_id: str | None = None,
    request_id: str | None = None,
    usage: Mapping[str, Any] | None = None,
    error_code: str | None = None,
    missing_headings: int = 0,
) -> RelevanceTrace:
    """由本轮真实对象构造脱敏 trace（PRD §9）。

    入参使用结构约定而不是具体类型，避免检索诊断层反向依赖证据视图/选择器模块：

    * ``scope``：有 ``scope_id`` / ``index_revision`` 的对象（``RetrievalScope``）；
    * ``candidates``：有 ``candidate_id`` / ``primary_block_id`` / ``payload_hash`` /
      ``view_kind`` / ``route_signals`` 的证据视图；
    * ``window``：``CandidateWindow``（``candidate_ids_after`` / ``truncation``）；
    * ``selection``：``SelectionResult``（``selected`` / ``exclusions`` / ``status``）。

    分数缺失一律留 ``None``；不补零、不推断正文、不记录原始 block ID。
    """

    scope_id = _scope_value(scope, "scope_id", "global")
    index_revision = _scope_value(scope, "index_revision", "unknown-index")
    selected_items = tuple(getattr(selection, "selected", ()) or ())
    window_ids: set[str] | None = None
    if window is not None:
        window_ids = {str(item) for item in getattr(window, "candidate_ids_after", ()) or ()}
    exclusion_by_id = {
        str(getattr(record, "candidate_id", "")): _reason_text(getattr(record, "reason", None))
        for record in (getattr(selection, "exclusions", ()) or ())
    }
    position_by_id = {
        str(getattr(item, "candidate_id", "")): position
        for position, item in enumerate(selected_items, start=1)
    }
    resolved_scores = scores or {}

    candidate_traces: list[RelevanceCandidateTrace] = []
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id", "") or "")
        signals = getattr(candidate, "route_signals", None)
        if candidate_id in exclusion_by_id:
            stage = "select"
        elif window_ids is not None and candidate_id not in window_ids:
            stage = "window"
        else:
            stage = "score" if candidate_id in resolved_scores else "window"
        candidate_traces.append(
            RelevanceCandidateTrace(
                block_ref=block_ref(_candidate_block_id(candidate)),
                candidate_ref=candidate_ref(candidate_id),
                stage=stage,
                view_kind=str(getattr(getattr(candidate, "view_kind", "block"), "value", "block")),
                view_hash=_optional_text(getattr(candidate, "payload_hash", None)),
                dense_similarity=_optional_float(getattr(signals, "dense_score", None)),
                bm25_score=_optional_float(getattr(signals, "bm25_score", None)),
                rrf_score=_optional_float(getattr(signals, "rrf_score", None)),
                rerank_score=_optional_float(resolved_scores.get(candidate_id)),
                dense_rank=_optional_int(getattr(signals, "dense_rank", None), minimum=1),
                bm25_rank=_optional_int(getattr(signals, "bm25_rank", None), minimum=1),
                rrf_rank=_optional_int(getattr(signals, "rrf_rank", None), minimum=1),
                exclusion_reason=exclusion_by_id.get(candidate_id),
                final_position=position_by_id.get(candidate_id),
                truncated=stage == "window",
            )
        )

    final_order = tuple(block_ref(_candidate_block_id(item)) for item in selected_items)
    resolved_status = status or (
        _status_from_selection(selection) if selection is not None else "failed"
    )
    window_size = len(window_ids) if window_ids is not None else len(candidate_traces)
    return RelevanceTrace(
        trace_id=trace_id,
        scope_id=scope_id,
        scope_summary=scope_summary(
            scope_id,
            index_revision,
            candidate_count=len(candidate_traces),
            window_size=window_size,
        ),
        index_revision=index_revision,
        profile_id=str(profile_id or ""),
        status=resolved_status,
        candidates=tuple(candidate_traces),
        final_order=final_order,
        candidate_count=len(candidate_traces),
        window_size=window_size,
        window_truncated=bool(
            window is not None and int(getattr(window, "truncation", 0) or 0) > 0
        ),
        provider_call_count=_optional_int(provider_call_count)
        or (1 if resolved_scores else 0),
        rerank_latency_ms=_optional_float(rerank_latency_ms),
        total_latency_ms=_optional_float(total_latency_ms),
        provider_trace_id=_optional_text(provider_trace_id),
        request_id=_optional_text(request_id),
        usage=dict(usage) if usage else None,
        error_code=_optional_text(error_code),
        missing_headings=_optional_int(missing_headings) or 0,
    )
