"""Hybrid retrieval with multi-query RRF, guarded lexical rescue and opt-in traces.

The pipeline keeps the exact user query, optionally adds one deterministic normalised variant,
and unions dense and BM25 candidates. Missing dense evidence stays ``None`` internally: a strong
lexical-only candidate can be rescued, while weak one-bigram coincidences are rejected by query
coverage and term-rarity gates.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.bm25 import BM25Hit, BM25Index, tokenize
from app.retrieval.image_evidence import image_facts
from app.retrieval.percentiles import canonical_percentiles
from app.retrieval.query_transform import (
    QueryVariant,
    build_query_variants,
    extract_identifier_definition_subjects,
    normalize_query,
)
from app.retrieval.trace import (
    RetrievalCandidateTrace,
    RetrievalResult,
    RetrievalSignalTrace,
    RetrievalTrace,
)
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk, _get_str
from app.vectorstore.base import VectorStore


@dataclass(frozen=True)
class _LexicalEvidence:
    route: str
    rank: int
    hit: BM25Hit
    rescue_eligible: bool


@dataclass(frozen=True)
class _GateDecision:
    accepted: bool
    reason: str
    threshold: float | None


@dataclass(frozen=True)
class _QueryIntent:
    definition_subjects: tuple[str, ...] = ()
    percentile_subject: str | None = None


@dataclass
class _FusionState:
    ordered: list[str]
    fused_scores: dict[str, float]
    dense_sims: dict[str, float]
    lexical: dict[str, list[_LexicalEvidence]]
    signals: dict[str, list[RetrievalSignalTrace]]


class HybridRetriever:
    """Dense and BM25 candidate union followed by RRF, gates and parent expansion.

    ``retrieve`` remains backwards compatible and automatically uses the normalised variant for
    BM25. Callers that embed every query variant should use ``retrieve_variants`` to enable the
    second dense route as well. Trace collection is available only through explicit
    ``*_with_trace`` methods and never records text or paths.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        candidate_k: int = 100,
        context_top_k: int = 8,
        bm25_k: int = 50,
        min_similarity: float | None = None,
        rrf_k: int = 60,
        expand_min_similarity: float = 0.68,
        lexical_floor: float = 0.45,
        bm25_rescue_k: int = 15,
        lexical_min_coverage: float = 0.25,
        lexical_min_rarity: float = 0.20,
        lexical_min_matched_terms: int = 2,
        lexical_long_query_min_coverage: float = 0.30,
        lexical_long_query_terms: int = 8,
        lexical_override_min_similarity: float = 0.35,
        lexical_refine_min_similarity: float = 0.50,
        lexical_refine_max_terms: int = 3,
        lexical_reserve_k: int = 3,
        dense_reserve_k: int = 2,
        max_chunks_per_heading: int = 2,
        max_chunks_per_parent: int = 3,
        max_chunks_per_source: int | None = None,
        numeric_penalty: float = 0.10,
        dense_only_penalty: float = 0.05,
        dense_overfetch_factor: int = 4,
        short_chunk_chars: int = 50,
        short_context_target_chars: int = 120,
        short_context_max_chars: int = 600,
    ) -> None:
        self._vectorstore = vectorstore
        self._candidate_k = candidate_k
        self._context_top_k = context_top_k
        self._bm25_k = bm25_k
        self._min_similarity = min_similarity
        self._rrf_k = rrf_k
        self._expand_min_similarity = expand_min_similarity
        self._lexical_floor = lexical_floor
        self._bm25_rescue_k = bm25_rescue_k
        self._lexical_min_coverage = lexical_min_coverage
        self._lexical_min_rarity = lexical_min_rarity
        self._lexical_min_matched_terms = lexical_min_matched_terms
        self._lexical_long_query_min_coverage = lexical_long_query_min_coverage
        self._lexical_long_query_terms = lexical_long_query_terms
        self._lexical_override_min_similarity = lexical_override_min_similarity
        self._lexical_refine_min_similarity = lexical_refine_min_similarity
        self._lexical_refine_max_terms = lexical_refine_max_terms
        self._lexical_reserve_k = lexical_reserve_k
        self._dense_reserve_k = dense_reserve_k
        self._max_chunks_per_heading = max_chunks_per_heading
        self._max_chunks_per_parent = max_chunks_per_parent
        self._max_chunks_per_source = (
            self._context_top_k
            if max_chunks_per_source is None
            else max(1, max_chunks_per_source)
        )
        self._numeric_penalty = numeric_penalty
        self._dense_only_penalty = dense_only_penalty
        self._dense_overfetch_factor = max(1, dense_overfetch_factor)
        self._short_chunk_chars = max(1, short_chunk_chars)
        self._short_context_target_chars = max(
            self._short_chunk_chars, short_context_target_chars
        )
        self._short_context_max_chars = max(
            self._short_context_target_chars, short_context_max_chars
        )
        self._bm25: BM25Index | None = None
        self._blocks: dict[str, tuple[str, dict[str, Any]]] = {}
        self._index_to_id: dict[int, str] = {}
        self._canonical_by_id: dict[str, str] = {}

    def retrieve(
        self,
        query_text: str,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with original dense evidence and original+normalised lexical evidence."""

        variants = build_query_variants(query_text)
        original = variants[0] if variants else QueryVariant(kind="original", text=query_text)
        chunks, _trace = self._run(
            lexical_variants=variants,
            dense_routes=((original, query_vector),),
            where=where,
            include_trace=False,
        )
        return chunks

    def retrieve_with_trace(
        self,
        query_text: str,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Explicitly retrieve with a privacy-safe structured diagnostic trace."""

        variants = build_query_variants(query_text)
        original = variants[0] if variants else QueryVariant(kind="original", text=query_text)
        chunks, trace = self._run(
            lexical_variants=variants,
            dense_routes=((original, query_vector),),
            where=where,
            include_trace=True,
        )
        assert trace is not None
        return RetrievalResult(chunks=tuple(chunks), trace=trace)

    def retrieve_variants(
        self,
        query_variants: Sequence[QueryVariant],
        query_vectors: Sequence[list[float]],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with one dense and lexical route per supplied query variant."""

        variants = self._validate_variants(query_variants, query_vectors)
        chunks, _trace = self._run(
            lexical_variants=variants,
            dense_routes=tuple(zip(variants, query_vectors, strict=True)),
            where=where,
            include_trace=False,
        )
        return chunks

    def retrieve_variants_with_trace(
        self,
        query_variants: Sequence[QueryVariant],
        query_vectors: Sequence[list[float]],
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Multi-query retrieval with an explicitly requested privacy-safe trace."""

        variants = self._validate_variants(query_variants, query_vectors)
        chunks, trace = self._run(
            lexical_variants=variants,
            dense_routes=tuple(zip(variants, query_vectors, strict=True)),
            where=where,
            include_trace=True,
        )
        assert trace is not None
        return RetrievalResult(chunks=tuple(chunks), trace=trace)

    def _run(
        self,
        lexical_variants: Sequence[QueryVariant],
        dense_routes: Sequence[tuple[QueryVariant, list[float]]],
        where: dict[str, Any] | None,
        include_trace: bool,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        for _variant, vector in dense_routes:
            self._reject_empty_query(vector)
        if self._vectorstore.count() == 0:
            raise RetrievalError("知识库尚未建立索引，请先导入并同步文档")

        # Historical imports may contain the same logical block under several source paths.
        # Over-fetch before canonicalisation so duplicate ANN neighbours cannot consume the
        # complete candidate window and hide otherwise relevant documents.
        dense_k = self._candidate_k * self._dense_overfetch_factor
        dense_hits = [
            (variant.kind, self._vectorstore.query(vector, dense_k, where=where))
            for variant, vector in dense_routes
        ]
        original_query = lexical_variants[0].text if lexical_variants else ""
        metric = canonical_percentiles(normalize_query(original_query)).strip()
        intent = _QueryIntent(
            definition_subjects=extract_identifier_definition_subjects(original_query),
            percentile_subject=metric if re.fullmatch(r"p\d{1,2}(?:\.\d+)?", metric) else None,
        )
        # Definition questions need one subject-only lexical route. Query scaffolding such as
        # “什么是” can otherwise push the actual definition outside BM25's candidate window.
        # This route is lexical-only, so it fixes recall without adding another embedding call.
        effective_lexical_variants = list(lexical_variants)
        existing_token_sets = {
            frozenset(tokenize(variant.text)) for variant in effective_lexical_variants
        }
        for subject in intent.definition_subjects:
            token_set = frozenset(tokenize(subject))
            if token_set not in existing_token_sets:
                effective_lexical_variants.append(
                    QueryVariant(kind="definition_subject", text=subject)
                )
                existing_token_sets.add(token_set)
        lexical_hits = self._search_lexical_routes(effective_lexical_variants, where)
        fusion = self._fuse(dense_hits, lexical_hits, include_trace=include_trace)
        chunks, decisions, selections = self._select_with_expansion(fusion, intent)

        if not include_trace:
            return chunks, None
        trace = self._build_trace(
            effective_lexical_variants, fusion, decisions, selections
        )
        return chunks, trace

    def _search_lexical_routes(
        self,
        variants: Sequence[QueryVariant],
        where: dict[str, Any] | None,
    ) -> list[tuple[str, list[BM25Hit]]]:
        bm25 = self._ensure_bm25()
        routes: list[tuple[str, list[BM25Hit]]] = []
        for variant in variants:
            # A metadata filter may exclude the first global top-k results, so safely over-fetch
            # before applying the equivalent in-memory filter.
            lexical_k = len(self._blocks) if where else self._bm25_k
            hits = bm25.search_detailed(variant.text, lexical_k)
            if where:
                hits = [
                    hit
                    for hit in hits
                    if (block_id := self._index_to_id.get(hit.index)) is not None
                    and _metadata_matches_where(self._blocks[block_id][1], where)
                ][: self._bm25_k]
            routes.append((variant.kind, hits))
        return routes

    def _fuse(
        self,
        dense_routes: Sequence[tuple[str, list[dict[str, Any]]]],
        lexical_routes: Sequence[tuple[str, list[BM25Hit]]],
        include_trace: bool,
    ) -> _FusionState:
        """Fuse every independent route with RRF and retain missing dense evidence as absent."""

        scores: dict[str, float] = {}
        dense_sims: dict[str, float] = {}
        lexical: dict[str, list[_LexicalEvidence]] = {}
        signals: dict[str, list[RetrievalSignalTrace]] = {}

        for route_kind, hits in dense_routes:
            route = f"dense:{route_kind}"
            seen_in_route: set[str] = set()
            for rank, hit in enumerate(hits, 1):
                raw_block_id = str(hit.get("block_id", ""))
                if not raw_block_id:
                    continue
                block_id = self._canonical_by_id.get(raw_block_id, raw_block_id)
                if block_id in seen_in_route:
                    continue
                seen_in_route.add(block_id)
                similarity = float(hit.get("similarity", 0.0))
                scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (self._rrf_k + rank)
                dense_sims[block_id] = max(dense_sims.get(block_id, similarity), similarity)
                if include_trace:
                    signals.setdefault(block_id, []).append(
                        RetrievalSignalTrace(route=route, rank=rank, score=similarity)
                    )

        for route_kind, hits in lexical_routes:
            route = f"bm25:{route_kind}"
            for rank, hit in enumerate(hits, 1):
                block_id = self._index_to_id.get(hit.index)
                if block_id is None:
                    continue
                scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (self._rrf_k + rank)
                evidence = _LexicalEvidence(
                    route=route,
                    rank=rank,
                    hit=hit,
                    rescue_eligible=self._is_lexical_rescue(route_kind, rank, hit),
                )
                lexical.setdefault(block_id, []).append(evidence)
                if include_trace:
                    signals.setdefault(block_id, []).append(
                        RetrievalSignalTrace(
                            route=route,
                            rank=rank,
                            score=hit.score,
                            coverage=hit.coverage,
                            rarity=hit.rarity,
                            informative_matched_terms=hit.informative_matched_terms,
                            informative_query_terms=hit.informative_query_terms,
                            informative_coverage=hit.informative_coverage,
                            identifier_matched_terms=hit.identifier_matched_terms,
                            identifier_query_terms=hit.identifier_query_terms,
                            anchor_matched_terms=hit.anchor_matched_terms,
                            anchor_query_terms=hit.anchor_query_terms,
                        )
                    )

        ordered = [
            block_id
            for block_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        return _FusionState(
            ordered=ordered,
            fused_scores=scores,
            dense_sims=dense_sims,
            lexical=lexical,
            signals=signals,
        )

    def _is_lexical_rescue(self, route_kind: str, rank: int, hit: BM25Hit) -> bool:
        if rank > self._bm25_rescue_k:
            return False
        if hit.rarity < self._lexical_min_rarity:
            return False
        # If a query contains a concrete ASCII identifier (Warebase, ADB, ticket IDs), generic
        # Chinese bigrams are not enough. A candidate must actually contain that identifier.
        if hit.identifier_query_terms and not hit.identifier_matched_terms:
            return False
        if hit.identifier_matched_terms:
            non_identifier_query_terms = hit.informative_query_terms - hit.identifier_query_terms
            non_identifier_matched_terms = (
                hit.informative_matched_terms - hit.identifier_matched_terms
            )
            # In a compound question such as “K8S 发生了什么事故”, the identifier alone is
            # insufficient: a troubleshooting note that merely mentions K8S is not an incident
            # record. Definition-only queries (“什么是 ADB”) have no additional term and retain
            # exact-identifier rescue.
            return not non_identifier_query_terms or bool(non_identifier_matched_terms)
        # A long conversational query contains many generic bigrams ("公司/对比/详情"). Every
        # route must independently cover enough informative terms; a normalized copy does not
        # receive a weaker gate than the original.
        if (
            hit.informative_query_terms >= self._lexical_long_query_terms
            and hit.informative_coverage < self._lexical_long_query_min_coverage
        ):
            return False
        if hit.informative_query_terms == 1 and hit.informative_matched_terms == 1:
            return True
        if hit.informative_matched_terms < self._lexical_min_matched_terms:
            return False
        # Long conversational queries dilute coverage with connective bigrams. Three independent
        # informative matches (for example "分表"/"京东"/"淘宝") remain strong evidence; broad
        # relation terms and one-/two-bigram coincidences cannot authorize a rescue.
        absolute_term_floor = max(3, self._lexical_min_matched_terms)
        return (
            hit.informative_coverage >= self._lexical_min_coverage
            or hit.informative_matched_terms >= absolute_term_floor
        )

    def _select_with_expansion(
        self,
        fusion: _FusionState,
        intent: _QueryIntent,
    ) -> tuple[list[RetrievedChunk], dict[str, _GateDecision], dict[str, str]]:
        decisions: dict[str, _GateDecision] = {}
        anchors: list[str] = []
        for block_id in fusion.ordered:
            if block_id not in self._blocks:
                decisions[block_id] = _GateDecision(False, "missing_block", None)
                continue
            decision = self._gate_candidate(block_id, fusion, intent)
            decisions[block_id] = decision
            if decision.accepted:
                anchors.append(block_id)

        if not anchors:
            return [], decisions, {}

        anchors, lexical_reserved, dense_reserved = self._prioritize_anchors(
            anchors, fusion, intent
        )

        expansion_sections: set[tuple[str, str]] = set()
        for block_id in anchors:
            dense_sim = fusion.dense_sims.get(block_id)
            if dense_sim is not None and dense_sim >= self._expand_min_similarity:
                expansion_sections.add(_section_key(self._blocks[block_id][1]))

        expansion_counts: dict[tuple[str, str], int] = {}
        for block_id in fusion.ordered:
            if block_id not in self._blocks:
                continue
            if self._blocks[block_id][1].get("chunk_type") == "image_description":
                continue
            section = _section_key(self._blocks[block_id][1])
            if section in expansion_sections:
                expansion_counts[section] = expansion_counts.get(section, 0) + 1
        expansion_sections = {
            section for section, count in expansion_counts.items() if count > 1
        }
        has_expansion_candidate = self._context_top_k >= 3 and bool(expansion_sections)
        anchor_limit = (
            max(1, self._context_top_k - 1)
            if has_expansion_candidate
            else self._context_top_k
        )

        results: list[RetrievedChunk] = []
        selections: dict[str, str] = {}
        seen: set[str] = set()
        heading_counts: dict[tuple[str, str], int] = {}
        parent_counts: dict[tuple[str, str], int] = {}
        source_counts: dict[str, int] = {}
        deferred: list[str] = []

        def _append(block_id: str, reason: str) -> None:
            text, metadata = self._blocks[block_id]
            similarity = fusion.dense_sims.get(block_id)
            if reason == "parent_section_expansion":
                match_type = "expanded"
            elif block_id in fusion.dense_sims and block_id in fusion.lexical:
                match_type = "hybrid"
            elif block_id in fusion.dense_sims:
                match_type = "dense"
            else:
                match_type = "lexical"
            context_text = (
                f"标题：{metadata.get('heading_path') or ''}\n图片证据：{image_facts(text)}"
                if metadata.get("chunk_type") == "image_description"
                else self._hydrate_short_context(block_id, text, metadata, fusion)
            )
            if context_text is None:
                context_text = self._hydrate_exact_heading_context(
                    block_id, text, metadata, fusion
                )
            results.append(
                self._to_chunk(
                    block_id,
                    text,
                    metadata,
                    similarity,
                    match_type=match_type,
                    context_text=context_text,
                )
            )
            selections[block_id] = reason
            seen.add(block_id)
            source = _source_key(metadata)
            source_counts[source] = source_counts.get(source, 0) + 1

        for anchor_index, block_id in enumerate(anchors):
            if len(results) >= anchor_limit:
                deferred.extend(anchors[anchor_index:])
                break
            metadata = self._blocks[block_id][1]
            heading_key = _heading_key(metadata, block_id)
            parent_key = _section_key(metadata)
            source_key = _source_key(metadata)
            if (
                heading_counts.get(heading_key, 0) >= self._max_chunks_per_heading
                or parent_counts.get(parent_key, 0) >= self._max_chunks_per_parent
                or source_counts.get(source_key, 0) >= self._max_chunks_per_source
            ):
                deferred.append(block_id)
                continue
            if block_id in lexical_reserved:
                reason = "lexical_reserve"
            elif block_id in dense_reserved:
                reason = "dense_reserve"
            else:
                reason = "anchor"
            _append(block_id, reason)
            heading_counts[heading_key] = heading_counts.get(heading_key, 0) + 1
            parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1

        def _backfill_deferred() -> None:
            for deferred_id in deferred:
                if len(results) >= self._context_top_k:
                    break
                metadata = self._blocks[deferred_id][1]
                if (
                    deferred_id not in seen
                    and source_counts.get(_source_key(metadata), 0)
                    < self._max_chunks_per_source
                ):
                    _append(deferred_id, "diversity_backfill")

        def _finalize() -> tuple[
            list[RetrievedChunk], dict[str, _GateDecision], dict[str, str]
        ]:
            return results, decisions, selections

        for block_id in fusion.ordered:
            if len(results) >= self._context_top_k:
                break
            if block_id in seen or block_id not in self._blocks:
                continue
            decision = decisions.get(block_id)
            if decision is not None and decision.reason in {
                "rejected_lexical_dense_contradiction",
                "rejected_low_quality_lexical_fill",
                "rejected_numeric_noise_below_threshold",
            }:
                # Expansion may recover an ordinary below-threshold sibling, but it must never
                # re-introduce a candidate rejected by an explicit contradiction/noise veto.
                continue
            metadata = self._blocks[block_id][1]
            if source_counts.get(_source_key(metadata), 0) >= self._max_chunks_per_source:
                continue
            if metadata.get("chunk_type") == "image_description":
                continue
            if _section_key(metadata) not in expansion_sections:
                continue
            _append(block_id, "parent_section_expansion")
        _backfill_deferred()
        return _finalize()

    def _prioritize_anchors(
        self,
        anchors: list[str],
        fusion: _FusionState,
        intent: _QueryIntent,
    ) -> tuple[list[str], set[str], set[str]]:
        """Balance independently qualified lexical and dense anchors.

        RRF rewards candidates repeated across routes. Without explicit reserves, several chunks
        from one repeatedly matched heading can crowd out a dense rank-1 answer or a neighbouring
        answer heading. Reserved candidates are therefore diversified at one chunk per heading;
        the normal result pass may still use ``max_chunks_per_heading`` for answer continuity.
        """

        rescue_candidates = [
            block_id
            for block_id in anchors
            if any(
                evidence.rescue_eligible for evidence in fusion.lexical.get(block_id, [])
            )
        ]
        rescue_candidates.sort(
            key=lambda block_id: (
                not any(
                    evidence.rescue_eligible and evidence.hit.identifier_matched_terms
                    for evidence in fusion.lexical[block_id]
                ),
                -max(
                    evidence.hit.informative_coverage
                    for evidence in fusion.lexical[block_id]
                    if evidence.rescue_eligible
                ),
                -max(
                    evidence.hit.informative_matched_terms
                    for evidence in fusion.lexical[block_id]
                    if evidence.rescue_eligible
                ),
                min(
                    evidence.rank
                    for evidence in fusion.lexical[block_id]
                    if evidence.rescue_eligible
                ),
                block_id not in fusion.dense_sims,
                -fusion.fused_scores[block_id],
                block_id,
            )
        )
        lexical_reserved = self._diversified_reserve(
            rescue_candidates,
            min(self._lexical_reserve_k, self._context_top_k),
        )
        lexical_set = set(lexical_reserved)

        dense_candidates = sorted(
            (
                block_id
                for block_id in anchors
                if block_id in fusion.dense_sims and block_id not in lexical_set
            ),
            key=lambda block_id: (
                -fusion.dense_sims[block_id],
                -fusion.fused_scores[block_id],
                block_id,
            ),
        )
        dense_reserved = self._diversified_reserve(
            dense_candidates,
            min(self._dense_reserve_k, max(0, self._context_top_k - len(lexical_reserved))),
        )
        dense_set = set(dense_reserved)
        reserved_set = lexical_set | dense_set
        prioritized = [
            *lexical_reserved,
            *dense_reserved,
            *(block_id for block_id in anchors if block_id not in reserved_set),
        ]
        # Prefer explicit prose in the same section when it has stronger evidence, or a better
        # lexical rank and almost equal dense relevance. Image summaries can otherwise displace
        # their own explanation; the complementary picture remains available via hydration.
        for index, block_id in enumerate(prioritized):
            if self._blocks[block_id][1].get("chunk_type") != "image_description":
                continue
            heading = _exact_heading_key(self._blocks[block_id][1])
            if heading is None:
                continue
            for other_index in range(index + 1, len(prioritized)):
                other_id = prioritized[other_index]
                other_meta = self._blocks[other_id][1]
                if (
                    other_meta.get("chunk_type") in {"text", "table"}
                    and _exact_heading_key(other_meta) == heading
                    and (
                        (fusion.dense_sims.get(other_id, -1) > fusion.dense_sims.get(block_id, -1)
                         and fusion.fused_scores[other_id] > fusion.fused_scores[block_id])
                        or (
                            other_id in fusion.dense_sims
                            and fusion.dense_sims[other_id] >= fusion.dense_sims.get(block_id, 0) - 0.05
                            and min((e.rank for e in fusion.lexical.get(other_id, [])
                                     if e.rescue_eligible), default=999)
                            < min((e.rank for e in fusion.lexical.get(block_id, [])
                                   if e.rescue_eligible), default=999)
                        )
                    )
                ):
                    prioritized[index], prioritized[other_index] = other_id, block_id
                    for reserve in (lexical_set, dense_set):
                        if block_id in reserve and other_id not in reserve:
                            reserve.remove(block_id)
                            reserve.add(other_id)
                    break
        if intent.definition_subjects:
            # Missing an explicit definition is a ranking penalty, not a hard veto. This keeps
            # useful mentions available when the corpus has no textbook sentence, while putting
            # “X 是…”, “X（全称）” and definition-labelled sections first whenever present.
            prioritized.sort(
                key=lambda block_id: not _has_definition_evidence(
                    self._blocks[block_id][0],
                    self._blocks[block_id][1],
                    intent.definition_subjects,
                )
            )
        return prioritized, lexical_set, dense_set

    def _diversified_reserve(self, candidates: Sequence[str], limit: int) -> list[str]:
        """Take strong candidates without spending multiple reserve slots on one heading."""

        if limit <= 0:
            return []
        reserved: list[str] = []
        headings: set[tuple[str, str]] = set()
        parent_counts: dict[tuple[str, str], int] = {}
        source_counts: dict[str, int] = {}
        for block_id in candidates:
            metadata = self._blocks[block_id][1]
            heading_key = _heading_key(metadata, block_id)
            parent_key = _section_key(metadata)
            source_key = _source_key(metadata)
            if (
                heading_key in headings
                or parent_counts.get(parent_key, 0) >= self._max_chunks_per_parent
                or source_counts.get(source_key, 0) >= self._max_chunks_per_source
            ):
                continue
            reserved.append(block_id)
            headings.add(heading_key)
            parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            if len(reserved) >= limit:
                break
        return reserved

    def _gate_candidate(
        self,
        block_id: str,
        fusion: _FusionState,
        intent: _QueryIntent,
    ) -> _GateDecision:
        has_definition_evidence = True
        if intent.definition_subjects:
            text, metadata = self._blocks[block_id]
            has_definition_evidence = _has_definition_evidence(
                text, metadata, intent.definition_subjects
            )

        def accept(reason: str, threshold: float | None) -> _GateDecision:
            if not has_definition_evidence:
                reason = f"{reason}_definition_fallback"
            return _GateDecision(True, reason, threshold)

        if self._min_similarity is None:
            return accept("accepted_no_threshold", None)

        dense_sim = fusion.dense_sims.get(block_id)
        lexical_evidence = fusion.lexical.get(block_id, [])
        has_lexical = bool(lexical_evidence)
        has_rescue = any(evidence.rescue_eligible for evidence in lexical_evidence)

        # Missing dense evidence is unknown, not a synthetic 0.0, so qualified BM25-only hits are
        # valid. A real, extremely low dense score is different: it is contradictory evidence and
        # vetoes the lexical match, including an exact identifier mentioned in an unrelated block.
        if has_rescue:
            text, _metadata = self._blocks[block_id]
            if intent.definition_subjects and _has_parenthetical_definition_evidence(
                text, intent.definition_subjects
            ):
                # An exact identifier plus an explicit “X 是…”/“X（全称）” statement is stronger
                # evidence than a low generic embedding score for a very short acronym query.
                return accept("accepted_definition_evidence", None)
            eligible_lexical = [
                evidence for evidence in lexical_evidence if evidence.rescue_eligible
            ]
            if (
                dense_sim is None
                and eligible_lexical
                and all(
                    evidence.hit.informative_query_terms >= self._lexical_long_query_terms
                    and evidence.hit.anchor_matched_terms == 0
                    for evidence in eligible_lexical
                )
            ):
                return _GateDecision(
                    False,
                    "rejected_lexical_only_missing_anchor",
                    None,
                )
            strongest_lexical_terms = max(
                evidence.hit.informative_matched_terms for evidence in lexical_evidence
            )
            strong_conversational_rescue = any(
                evidence.rescue_eligible
                and evidence.hit.informative_matched_terms >= 3
                and evidence.hit.informative_coverage >= self._lexical_long_query_min_coverage
                and evidence.hit.rarity >= 0.60
                and evidence.hit.anchor_matched_terms >= 1
                for evidence in lexical_evidence
            )
            # A precise percentile plus explanatory prose is meaningful evidence even when
            # a long usage question dilutes dense similarity. Keep the low-dense veto below.
            exact_metric_prose = (
                intent.percentile_subject is not None
                and intent.percentile_subject in tokenize(text)
                and len(text) >= self._short_chunk_chars
                and not _is_numeric_noise(text)
            )
            if (
                dense_sim is not None
                and dense_sim < self._lexical_refine_min_similarity
                and strongest_lexical_terms <= self._lexical_refine_max_terms
                and not exact_metric_prose
                and not strong_conversational_rescue
                and not (intent.definition_subjects and has_definition_evidence)
            ):
                return _GateDecision(
                    False,
                    "rejected_low_quality_lexical_fill",
                    self._lexical_refine_min_similarity,
                )
            if dense_sim is None:
                return accept("accepted_lexical_only", None)
            if dense_sim < self._lexical_override_min_similarity:
                return _GateDecision(
                    False,
                    "rejected_lexical_dense_contradiction",
                    self._lexical_override_min_similarity,
                )
            if dense_sim >= self._lexical_floor:
                return accept("accepted_lexical_rescue", self._lexical_floor)
            return accept("accepted_lexical_override", None)

        if dense_sim is None:
            return _GateDecision(False, "rejected_missing_dense_weak_lexical", None)

        if not has_lexical:
            threshold = self._min_similarity + self._dense_only_penalty
            accepted_reason = "accepted_dense_only"
        else:
            threshold = self._min_similarity
            accepted_reason = "accepted_dense_with_weak_lexical"

        numeric_noise = _is_numeric_noise(self._blocks[block_id][0])
        if numeric_noise:
            threshold += self._numeric_penalty
        if dense_sim < threshold:
            reason = (
                "rejected_numeric_noise_below_threshold"
                if numeric_noise
                else "rejected_dense_below_threshold"
            )
            return _GateDecision(False, reason, threshold)
        return accept(accepted_reason, threshold)

    def _hydrate_short_context(
        self,
        block_id: str,
        text: str,
        metadata: dict[str, Any],
        fusion: _FusionState,
    ) -> str | None:
        """Attach bounded same-section context to atomic blocks shorter than 50 characters."""

        stripped = text.strip()
        if len(stripped) >= self._short_chunk_chars:
            return None

        heading = str(metadata.get("heading_path", "") or "").strip()
        if not heading:
            return None

        target_section = _section_key(metadata)
        target_parent = target_section[1]
        parts = [f"标题：{heading}\n内容：{stripped}"]
        content_chars = len(stripped)

        ordered_ids = [
            *fusion.ordered,
            *(candidate_id for candidate_id in sorted(self._blocks) if candidate_id not in fusion.fused_scores),
        ]
        for candidate_id in ordered_ids:
            if candidate_id == block_id:
                continue
            sibling_text, sibling_metadata = self._blocks[candidate_id]
            sibling_heading = str(sibling_metadata.get("heading_path", "") or "").strip()
            if str(sibling_metadata.get("source_file", "") or "") != target_section[0]:
                continue
            if target_parent:
                if _section_key(sibling_metadata) != target_section:
                    continue
            elif not sibling_heading or sibling_heading != heading:
                continue

            sibling = sibling_text.strip()
            if sibling_metadata.get("chunk_type") == "image_description":
                sibling = image_facts(sibling)
            if not sibling or sibling == stripped:
                continue
            remaining = self._short_context_max_chars - content_chars
            if remaining <= 0:
                break
            sibling = sibling[:remaining]
            label = "同章节上下文"
            if sibling_heading != heading:
                label += f"（{sibling_heading}）"
            parts.append(f"{label}：\n{sibling}")
            content_chars += len(sibling)
            if content_chars >= self._short_context_target_chars:
                break

        return "\n\n".join(parts)

    def _hydrate_exact_heading_context(
        self,
        block_id: str,
        text: str,
        metadata: dict[str, Any],
        fusion: _FusionState,
    ) -> str | None:
        """Attach one complementary text/table/image block from the exact source heading."""

        if metadata.get("chunk_type") not in {"text", "table"}:
            return None
        heading_key = _exact_heading_key(metadata)
        if heading_key is None:
            return None
        ordered_ids = [
            *fusion.ordered,
            *(candidate_id for candidate_id in sorted(self._blocks) if candidate_id not in fusion.fused_scores),
        ]
        for candidate_id in ordered_ids:
            if candidate_id == block_id:
                continue
            companion_text, companion_metadata = self._blocks[candidate_id]
            if _exact_heading_key(companion_metadata) != heading_key:
                continue
            if companion_metadata.get("chunk_type") == "image_description":
                companion_text = image_facts(companion_text)
            companion_context = companion_text.strip()[: self._short_context_max_chars]
            if not companion_context or companion_context == text.strip():
                continue
            heading = heading_key[1]
            label = (
                "同标题图片信息"
                if companion_metadata.get("chunk_type") == "image_description"
                else "同标题补充信息"
            )
            return (
                f"标题：{heading}\n内容：{text.strip()}\n\n"
                f"{label}：\n{companion_context}"
            )
        return None

    def _build_trace(
        self,
        variants: Sequence[QueryVariant],
        fusion: _FusionState,
        decisions: dict[str, _GateDecision],
        selections: dict[str, str],
    ) -> RetrievalTrace:
        candidates: list[RetrievalCandidateTrace] = []
        ordered_ids = [
            *fusion.ordered,
            *(block_id for block_id in selections if block_id not in fusion.fused_scores),
        ]
        for block_id in ordered_ids:
            decision = decisions.get(
                block_id,
                _GateDecision(
                    block_id in selections,
                    "accepted_via_exact_heading" if block_id in selections else "not_evaluated",
                    None,
                ),
            )
            candidates.append(
                RetrievalCandidateTrace(
                    block_id=block_id,
                    signals=tuple(fusion.signals.get(block_id, [])),
                    fused_score=fusion.fused_scores.get(block_id, 0.0),
                    dense_similarity=fusion.dense_sims.get(block_id),
                    gate_reason=decision.reason,
                    gate_threshold=decision.threshold,
                    selected=block_id in selections,
                    selection_reason=selections.get(block_id),
                )
            )
        return RetrievalTrace(
            variant_kinds=tuple(variant.kind for variant in variants),
            candidates=tuple(candidates),
        )

    def _ensure_bm25(self) -> BM25Index:
        """Build a logical-block-deduplicated lexical index.

        Old upload versions stored every task in a new UUID directory, so identical content got
        different path-derived block IDs.  Canonicalising by document/content identity here keeps
        both dense and lexical retrieval from returning the same evidence more than once.
        """

        if self._bm25 is None:
            texts: list[str] = []
            self._blocks = {}
            self._index_to_id = {}
            self._canonical_by_id = {}
            grouped: dict[tuple[str, ...], list[tuple[str, str, dict[str, Any]]]] = {}
            for block_id, text, metadata in self._vectorstore.list_blocks():
                grouped.setdefault(_logical_block_key(text, metadata, block_id), []).append(
                    (block_id, text, metadata)
                )

            canonical_blocks: list[tuple[str, str, dict[str, Any]]] = []
            for candidates in grouped.values():
                canonical = min(candidates, key=_canonical_block_priority)
                canonical_blocks.append(canonical)
                for block_id, _text, _metadata in candidates:
                    self._canonical_by_id[block_id] = canonical[0]

            canonical_blocks.sort(key=lambda item: item[0])
            for index, (block_id, text, metadata) in enumerate(canonical_blocks):
                title = _title_from_path(metadata.get("source_file", ""))
                heading = metadata.get("heading_path") or ""
                evidence = image_facts(text) if metadata.get("chunk_type") == "image_description" else text
                # Full structured evidence remains keyword-searchable (dates, values, identifiers).
                # Dense vectors independently use image_index_text, avoiding a noisy mega-vector.
                texts.append(f"{title} {heading} {evidence}".strip())
                self._blocks[block_id] = (text, metadata)
                self._index_to_id[index] = block_id
            self._bm25 = BM25Index(texts)
        return self._bm25

    @staticmethod
    def _validate_variants(
        query_variants: Sequence[QueryVariant],
        query_vectors: Sequence[list[float]],
    ) -> tuple[QueryVariant, ...]:
        variants = tuple(query_variants)
        if not variants:
            raise RetrievalError("查询文本为空")
        if len(variants) != len(query_vectors):
            raise ValueError("query_variants 与 query_vectors 数量必须一致")
        if variants[0].kind != "original":
            raise ValueError("第一条查询变体必须保留原始查询")
        return variants

    @staticmethod
    def _reject_empty_query(query_vector: list[float]) -> None:
        if not query_vector or all(value == 0.0 for value in query_vector):
            raise RetrievalError("查询向量为空，请先对查询文本进行嵌入")

    @staticmethod
    def _to_chunk(
        block_id: str,
        text: str,
        metadata: dict[str, Any],
        similarity: float | None,
        *,
        match_type: str = "dense",
        context_text: str | None = None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            block_id=block_id,
            text=text,
            similarity=similarity,
            source_file=_get_str(metadata, "source_file", ""),
            platform=_get_str(metadata, "platform", "local"),
            chunk_type=_get_str(metadata, "chunk_type", "text"),
            anchor=_get_str(metadata, "anchor"),
            heading_path=_get_str(metadata, "heading_path"),
            metadata=dict(metadata),
            match_type=match_type,
            context_text=context_text,
        )


def _title_from_path(source_file: str) -> str:
    name = Path(source_file).name
    return name[: -len(".md")] if name.lower().endswith(".md") else name


def _has_definition_evidence(
    text: str,
    metadata: dict[str, Any],
    subjects: Sequence[str],
) -> bool:
    """Require an explicit definitional statement or definition-labelled section."""

    heading = str(metadata.get("heading_path", "") or "")
    title = _title_from_path(str(metadata.get("source_file", "") or ""))
    searchable = f"{heading}\n{title}\n{text}"
    for subject in subjects:
        escaped = re.escape(subject)
        if re.search(escaped, searchable, re.IGNORECASE) is None:
            continue
        if re.search(
            rf"{escaped}\s*(?:是(?:一种|一个|用于|用来|指)?|指(?:的是)?|即|全称(?:是|为)|属于|代表)",
            text,
            re.IGNORECASE,
        ):
            return True
        if _has_parenthetical_definition_evidence(text, (subject,)):
            return True
        if re.search(
            r"定义|简介|概述|是什么|什么意思|全称",
            f"{heading}\n{title}",
            re.IGNORECASE,
        ):
            return True
    return False


def _has_parenthetical_definition_evidence(
    text: str,
    subjects: Sequence[str],
) -> bool:
    """Return whether a subject is immediately expanded in brackets or parentheses."""

    return any(
        re.search(
            rf"{re.escape(subject)}\s*[（(【\[][^）)】\]\n]{{2,80}}[）)】\]]",
            text,
            re.IGNORECASE,
        )
        is not None
        for subject in subjects
    )


_LEGACY_TASK_SEGMENT_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _logical_block_key(
    text: str,
    metadata: dict[str, Any],
    block_id: str,
) -> tuple[str, ...]:
    """Identity for one piece of evidence, independent of its historical upload path.

    The heading is part of the evidence identity.  A document can intentionally repeat the
    same formula/table under several business sections; collapsing those blocks by body text
    alone silently transfers the evidence to whichever heading happens to win canonicalisation
    and makes section-specific questions impossible to answer.  Historical upload copies still
    collapse because they retain the same full heading path.
    """

    normalized_text = text.strip()
    content_hash = (
        hashlib.sha1(normalized_text.encode("utf-8"), usedforsecurity=False).hexdigest()
        if normalized_text
        else block_id
    )
    return (
        content_hash,
        str(metadata.get("chunk_type", "text") or "text"),
        _logical_heading_identity(metadata),
    )


def _logical_heading_identity(metadata: dict[str, Any]) -> str:
    heading = str(metadata.get("heading_path", "") or "").strip()
    if heading:
        return " ".join(heading.casefold().split())
    # Unheaded document blocks cannot rely on path-derived IDs because historical imports put
    # the same file below different UUID directories.  Their exact body hash remains the safest
    # available identity and is already present in the other key component.
    return "__unheaded__"


def _canonical_block_priority(
    candidate: tuple[str, str, dict[str, Any]],
) -> tuple[int, int, int, str, str]:
    """Prefer stable, human-readable source paths over legacy task UUID directories."""

    block_id, _text, metadata = candidate
    source = str(metadata.get("source_file", "") or "")
    parts = tuple(part for part in re.split(r"[\\/]", source) if part)
    legacy_segments = sum(1 for part in parts if _LEGACY_TASK_SEGMENT_RE.fullmatch(part))
    return (legacy_segments, len(parts), len(source), source.casefold(), block_id)


_NUMERIC_PUNCT = frozenset("；：，。、()（）$.,/[]{}:;|")


def _is_numeric_noise(text: str, ratio: float = 0.45, min_len: int = 50) -> bool:
    if not text or len(text) < min_len:
        return False
    numeric_or_punctuation = sum(
        1 for character in text if character.isdigit() or character in _NUMERIC_PUNCT
    )
    return numeric_or_punctuation / len(text) > ratio


def _section_key(metadata: dict[str, Any]) -> tuple[str, str]:
    source = metadata.get("source_file", "") or ""
    heading = metadata.get("heading_path", "") or ""
    parts = heading.split(" > ")
    parent = " > ".join(parts[:-1]) if len(parts) > 1 else ""
    return (source, parent)


def _source_key(metadata: dict[str, Any]) -> str:
    return str(metadata.get("source_file", "") or "")


def _heading_key(metadata: dict[str, Any], block_id: str) -> tuple[str, str]:
    """Stable per-source heading key; unheaded blocks remain independently selectable."""

    source = str(metadata.get("source_file", "") or "")
    heading = str(metadata.get("heading_path", "") or "")
    return (source, heading or f"__block__:{block_id}")


def _exact_heading_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    """Exact source + non-empty heading identity used for conservative image attachment."""

    source = str(metadata.get("source_file", "") or "")
    heading = str(metadata.get("heading_path", "") or "").strip()
    if not source or not heading:
        return None
    return (source, heading)


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Safely mirror common Chroma equality filters for the in-memory BM25 route.

    Unknown operators fail closed so a lexical candidate can never bypass a vector-store filter.
    """

    for key, expected in where.items():
        if key == "$and" and isinstance(expected, list):
            if not all(
                isinstance(clause, dict) and _metadata_matches_where(metadata, clause)
                for clause in expected
            ):
                return False
            continue
        if key == "$or" and isinstance(expected, list):
            if not any(
                isinstance(clause, dict) and _metadata_matches_where(metadata, clause)
                for clause in expected
            ):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if set(expected) == {"$eq"} and actual == expected["$eq"]:
                continue
            if (
                set(expected) == {"$in"}
                and isinstance(expected["$in"], list)
                and actual in expected["$in"]
            ):
                continue
            return False
        if actual != expected:
            return False
    return True
