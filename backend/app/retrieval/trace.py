"""Privacy-safe structured diagnostics for the hybrid retrieval pipeline.

Trace records are opt-in.  They intentionally contain no query text, chunk text, document path,
heading or metadata.  Opaque block IDs and numeric ranking signals are sufficient to explain why
a candidate entered or left the final context.
"""

from __future__ import annotations

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
