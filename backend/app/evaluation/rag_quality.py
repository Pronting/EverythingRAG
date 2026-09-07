"""块级 RAG 召回评估：真实查询嵌入 + 生产 HybridRetriever。

正例必须同时命中来源、标题路径和正文证据，避免同文档或同标题错误块造成指标虚高；
负例用于观察无依据上下文的误放行率。输出只保留脱敏块引用、文件名、标题和分数。
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import QueryVariant, build_query_variants
from app.retrieval.vector_retriever import RetrievedChunk
from app.vectorstore.embedder import Embedder

EVAL_SCHEMA_VERSION = 3
MIN_POSITIVE_CASES = 50
MIN_NEGATIVE_CASES = 10


@dataclass(frozen=True)
class ExpectedTarget:
    """一个可核验的块目标：精确来源、完整标题和正文证据必须同时命中。"""

    source: str
    heading: str
    required_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("目标 source 不能为空")
        if not isinstance(self.heading, str) or not self.heading.strip():
            raise ValueError("目标 heading 不能为空")
        if self.source != _source_name(self.source):
            raise ValueError("目标 source 必须是文件 basename，不能包含目录")
        if not isinstance(self.required_evidence, tuple) or not self.required_evidence:
            raise ValueError("目标 required_evidence 必须是非空字符串数组")
        normalised_evidence: list[str] = []
        for evidence in self.required_evidence:
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError("目标 required_evidence 不能包含空值")
            normalised_evidence.append(_normalise(evidence))
        if len(normalised_evidence) != len(set(normalised_evidence)):
            raise ValueError("目标 required_evidence 不能重复")


@dataclass(frozen=True)
class PositiveCase:
    id: str
    category: str
    query: str
    expected_source: str | None = None
    expected_heading: str | None = None
    required_evidence: tuple[str, ...] = ()
    acceptable_targets: tuple[ExpectedTarget, ...] = ()
    required_targets: tuple[ExpectedTarget, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("id", "category", "query"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"正例 {field_name} 不能为空")

        has_legacy_field = self.expected_source is not None or self.expected_heading is not None
        if has_legacy_field:
            if not isinstance(self.expected_source, str) or not self.expected_source.strip():
                raise ValueError(f"简单目标 expected_source 不能为空: {self.id!r}")
            if not isinstance(self.expected_heading, str) or not self.expected_heading.strip():
                raise ValueError(f"简单目标 expected_heading 不能为空: {self.id!r}")
            # 简单字段保留兼容性，但仍执行与新 schema 相同的 basename 契约。
            ExpectedTarget(
                self.expected_source,
                self.expected_heading,
                self.required_evidence,
            )
        elif self.required_evidence:
            raise ValueError(
                f"required_evidence 只能与简单目标 expected_source/expected_heading 同用: {self.id!r}"
            )

        modes = sum(
            (
                has_legacy_field,
                bool(self.acceptable_targets),
                bool(self.required_targets),
            )
        )
        if modes != 1:
            raise ValueError(
                f"正例必须且只能使用一种目标模式（简单/acceptable/required）: {self.id!r}"
            )
        targets = (*self.acceptable_targets, *self.required_targets)
        if any(not isinstance(target, ExpectedTarget) for target in targets):
            raise TypeError(f"目标必须是 ExpectedTarget: {self.id!r}")
        keys = [_target_key(target) for target in targets]
        if len(keys) != len(set(keys)):
            raise ValueError(f"同一正例不能包含重复目标: {self.id!r}")


@dataclass(frozen=True)
class NegativeCase:
    id: str
    query: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("负例 id 不能为空")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"负例 query 不能为空: {self.id!r}")


@dataclass(frozen=True)
class CandidateSummary:
    block_ref: str
    source: str
    heading: str
    similarity: float | None


@dataclass(frozen=True)
class CaseOutcome:
    id: str
    query: str
    matched_rank: int | None
    retrieved_count: int
    candidates: tuple[CandidateSummary, ...]


@dataclass(frozen=True)
class QualityReport:
    generated_at_unix: float
    duration_seconds: float
    positive_total: int
    negative_total: int
    hit_at_1: float
    hit_at_5: float
    hit_at_8: float
    mrr_at_8: float
    negative_false_context_rate: float
    category_metrics: dict[str, dict[str, float | int]]
    positive_outcomes: tuple[CaseOutcome, ...]
    negative_outcomes: tuple[CaseOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_gate_passes(
    report: QualityReport,
    *,
    min_hit_at_8: float = 1.0,
    max_negative_false_context_rate: float = 0.0,
) -> bool:
    """发布门禁：聚合阈值和每个正/负例都必须严格通过。"""

    if not 0.0 <= min_hit_at_8 <= 1.0:
        raise ValueError("min_hit_at_8 必须在 0 到 1 之间")
    if not 0.0 <= max_negative_false_context_rate <= 1.0:
        raise ValueError("max_negative_false_context_rate 必须在 0 到 1 之间")
    positive_cases_pass = (
        report.positive_total >= MIN_POSITIVE_CASES
        and len(report.positive_outcomes) == report.positive_total
        and all(
            outcome.matched_rank is not None and outcome.matched_rank <= 8
            for outcome in report.positive_outcomes
        )
    )
    negative_cases_pass = (
        report.negative_total >= MIN_NEGATIVE_CASES
        and len(report.negative_outcomes) == report.negative_total
        and all(outcome.retrieved_count == 0 for outcome in report.negative_outcomes)
    )
    return (
        positive_cases_pass
        and negative_cases_pass
        and report.hit_at_8 >= min_hit_at_8
        and report.negative_false_context_rate <= max_negative_false_context_rate
    )


def load_eval_cases(path: Path) -> tuple[tuple[PositiveCase, ...], tuple[NegativeCase, ...]]:
    """读取并严格校验黄金集；重复 ID 和不完整块目标立即失败。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("评估集顶层必须是 JSON object")
    required_top_level = {"schema_version", "positive_cases", "negative_cases"}
    if set(payload) != required_top_level:
        raise ValueError("评估集顶层必须且只能包含 schema_version/positive_cases/negative_cases")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != EVAL_SCHEMA_VERSION
    ):
        raise ValueError(f"评估集 schema_version 必须为 {EVAL_SCHEMA_VERSION}")
    raw_positives = payload["positive_cases"]
    raw_negatives = payload["negative_cases"]
    if not isinstance(raw_positives, list) or not isinstance(raw_negatives, list):
        raise TypeError("positive_cases 和 negative_cases 必须是 JSON array")
    positives = tuple(_parse_positive_case(item) for item in raw_positives)
    negatives = tuple(_parse_negative_case(item) for item in raw_negatives)
    ids = [case.id for case in (*positives, *negatives)]
    if len(ids) != len(set(ids)):
        raise ValueError("评估集 case id 必须唯一")
    if len(positives) < MIN_POSITIVE_CASES:
        raise ValueError(f"正例不足 {MIN_POSITIVE_CASES} 条，不能作为本项目的准确率门禁")
    if len(negatives) < MIN_NEGATIVE_CASES:
        raise ValueError(f"负例不足 {MIN_NEGATIVE_CASES} 条，不能作为本项目的准确率门禁")
    return positives, negatives


def evaluate_retrieval(
    *,
    embedder: Embedder,
    retriever: HybridRetriever,
    positives: tuple[PositiveCase, ...],
    negatives: tuple[NegativeCase, ...],
    embed_batch_size: int = 16,
) -> QualityReport:
    """对所有 case 批量嵌入后逐条走生产多路召回，返回严格块级指标。"""

    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size 必须大于 0")
    started = time.perf_counter()
    all_cases: tuple[PositiveCase | NegativeCase, ...] = (*positives, *negatives)
    variants_by_id = {case.id: build_query_variants(case.query) for case in all_cases}
    flat_variants = [variant for case in all_cases for variant in variants_by_id[case.id]]
    vectors = _embed_query_variants(embedder, flat_variants, embed_batch_size)

    vectors_by_id: dict[str, tuple[list[float], ...]] = {}
    cursor = 0
    for case in all_cases:
        variants = variants_by_id[case.id]
        vectors_by_id[case.id] = tuple(vectors[cursor : cursor + len(variants)])
        cursor += len(variants)

    positive_outcomes: list[CaseOutcome] = []
    negative_outcomes: list[CaseOutcome] = []
    for case in positives:
        chunks = retriever.retrieve_variants(variants_by_id[case.id], vectors_by_id[case.id])
        rank = _matched_rank(chunks, case)
        positive_outcomes.append(_outcome(case.id, case.query, rank, chunks))
    for case in negatives:
        chunks = retriever.retrieve_variants(variants_by_id[case.id], vectors_by_id[case.id])
        negative_outcomes.append(_outcome(case.id, case.query, None, chunks))

    duration = time.perf_counter() - started
    return _build_report(
        positives=positives,
        positive_outcomes=tuple(positive_outcomes),
        negative_outcomes=tuple(negative_outcomes),
        duration_seconds=duration,
    )


def chunk_matches_expectation(chunk: RetrievedChunk, case: PositiveCase) -> bool:
    """块是否精确命中该 case 的任一显式目标。

    ``required_targets`` 的“全部证据都已出现”是结果集级约束，由 ``_matched_rank``
    计算；本函数只用于识别一个块是否属于该 case 的目标集合。
    """

    return any(_chunk_matches_target(chunk, target) for target in _case_targets(case))


def _matched_rank(chunks: list[RetrievedChunk], case: PositiveCase) -> int | None:
    """返回完整满足 case 时的名次；复合目标取最后一个必需块首次出现的位置。"""

    if case.required_targets:
        ranks: list[int] = []
        for target in case.required_targets:
            rank = next(
                (
                    index
                    for index, chunk in enumerate(chunks, start=1)
                    if _chunk_matches_target(chunk, target)
                ),
                None,
            )
            if rank is None:
                return None
            ranks.append(rank)
        return max(ranks)
    return next(
        (
            index
            for index, chunk in enumerate(chunks, start=1)
            if chunk_matches_expectation(chunk, case)
        ),
        None,
    )


def _chunk_matches_target(chunk: RetrievedChunk, target: ExpectedTarget) -> bool:
    actual_source = _normalise_source(chunk.source_file)
    expected_source = _normalise_source(target.source)
    expected_heading = _normalise_heading(target.heading)
    actual_headings = {_normalise_heading(chunk.heading_path or "")}
    raw_aliases = chunk.metadata.get("heading_aliases")
    if isinstance(raw_aliases, str) and raw_aliases:
        try:
            aliases = json.loads(raw_aliases)
        except json.JSONDecodeError:
            aliases = []
        if isinstance(aliases, list):
            actual_headings.update(
                _normalise_heading(alias) for alias in aliases if isinstance(alias, str)
            )
    if actual_source != expected_source or expected_heading not in actual_headings:
        return False
    # ``context_text`` is provenance-safe same-heading evidence assembled by the production
    # retriever (for example a 41-character formula next to its field-definition table). The
    # model sees this effective context, so the evaluator must score the same payload.
    normalised_text = _normalise(chunk.context_text or chunk.text)
    return all(_normalise(evidence) in normalised_text for evidence in target.required_evidence)


def _case_targets(case: PositiveCase) -> tuple[ExpectedTarget, ...]:
    if case.acceptable_targets:
        return case.acceptable_targets
    if case.required_targets:
        return case.required_targets
    if case.expected_source is None or case.expected_heading is None:
        # PositiveCase.__post_init__ 已禁止此状态；保留防御式错误以免静默放宽。
        raise ValueError(f"正例缺少目标: {case.id!r}")
    return (
        ExpectedTarget(
            case.expected_source,
            case.expected_heading,
            case.required_evidence,
        ),
    )


def _parse_positive_case(raw: object) -> PositiveCase:
    if not isinstance(raw, dict):
        raise TypeError("positive_cases 中的元素必须是 JSON object")
    allowed = {
        "id",
        "category",
        "query",
        "expected_source",
        "expected_heading",
        "required_evidence",
        "acceptable_targets",
        "required_targets",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"正例包含未知字段: {sorted(unknown)!r}")
    acceptable = _parse_targets(raw.get("acceptable_targets", []), "acceptable_targets")
    required = _parse_targets(raw.get("required_targets", []), "required_targets")
    evidence = _parse_evidence(raw.get("required_evidence", []), "required_evidence")
    try:
        return PositiveCase(
            id=raw["id"],
            category=raw["category"],
            query=raw["query"],
            expected_source=raw.get("expected_source"),
            expected_heading=raw.get("expected_heading"),
            required_evidence=evidence,
            acceptable_targets=acceptable,
            required_targets=required,
        )
    except KeyError as exc:
        raise ValueError(f"正例缺少字段: {exc.args[0]}") from exc


def _parse_targets(raw: object, field_name: str) -> tuple[ExpectedTarget, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} 必须是 JSON array")
    targets: list[ExpectedTarget] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"{field_name} 中的目标必须是 JSON object")
        if set(item) != {"source", "heading", "required_evidence"}:
            raise ValueError(f"{field_name} 目标必须且只能包含 source/heading/required_evidence")
        targets.append(
            ExpectedTarget(
                source=item["source"],
                heading=item["heading"],
                required_evidence=_parse_evidence(
                    item["required_evidence"],
                    f"{field_name}.required_evidence",
                ),
            )
        )
    return tuple(targets)


def _parse_evidence(raw: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} 必须是 JSON array")
    evidence = tuple(raw)
    if any(not isinstance(item, str) for item in evidence):
        raise TypeError(f"{field_name} 必须只包含字符串")
    return evidence


def _parse_negative_case(raw: object) -> NegativeCase:
    if not isinstance(raw, dict):
        raise TypeError("negative_cases 中的元素必须是 JSON object")
    if set(raw) != {"id", "query"}:
        raise ValueError("负例必须且只能包含 id/query")
    return NegativeCase(id=raw["id"], query=raw["query"])


def _embed_query_variants(
    embedder: Embedder,
    variants: list[QueryVariant],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(variants), batch_size):
        texts = [variant.text for variant in variants[start : start + batch_size]]
        vectors.extend(embedder.embed_queries(texts))
    if len(vectors) != len(variants):
        raise RuntimeError(f"嵌入返回数量错误：期望 {len(variants)}，实际 {len(vectors)}")
    return vectors


def _outcome(
    case_id: str,
    query: str,
    rank: int | None,
    chunks: list[RetrievedChunk],
) -> CaseOutcome:
    candidates = tuple(
        CandidateSummary(
            block_ref=_block_ref(chunk.block_id),
            source=_source_name(chunk.source_file),
            heading=chunk.heading_path or "",
            similarity=(
                round(float(chunk.similarity), 6) if chunk.similarity is not None else None
            ),
        )
        for chunk in chunks
    )
    return CaseOutcome(
        id=case_id,
        query=query,
        matched_rank=rank,
        retrieved_count=len(chunks),
        candidates=candidates,
    )


def _build_report(
    *,
    positives: tuple[PositiveCase, ...],
    positive_outcomes: tuple[CaseOutcome, ...],
    negative_outcomes: tuple[CaseOutcome, ...],
    duration_seconds: float,
) -> QualityReport:
    positive_total = len(positive_outcomes)

    def hit_at(k: int, outcomes: tuple[CaseOutcome, ...] = positive_outcomes) -> float:
        if not outcomes:
            return 0.0
        return sum(
            item.matched_rank is not None and item.matched_rank <= k for item in outcomes
        ) / len(outcomes)

    mrr = (
        sum(
            1.0 / item.matched_rank
            for item in positive_outcomes
            if item.matched_rank is not None and item.matched_rank <= 8
        )
        / positive_total
        if positive_total
        else 0.0
    )
    negative_fcr = (
        sum(item.retrieved_count > 0 for item in negative_outcomes) / len(negative_outcomes)
        if negative_outcomes
        else 0.0
    )
    by_category: dict[str, list[CaseOutcome]] = defaultdict(list)
    category_by_id = {case.id: case.category for case in positives}
    for outcome in positive_outcomes:
        by_category[category_by_id[outcome.id]].append(outcome)
    category_metrics = {
        category: {
            "total": len(outcomes),
            "hit_at_1": round(hit_at(1, tuple(outcomes)), 6),
            "hit_at_5": round(hit_at(5, tuple(outcomes)), 6),
            "hit_at_8": round(hit_at(8, tuple(outcomes)), 6),
        }
        for category, outcomes in sorted(by_category.items())
    }
    return QualityReport(
        generated_at_unix=time.time(),
        duration_seconds=round(duration_seconds, 3),
        positive_total=positive_total,
        negative_total=len(negative_outcomes),
        hit_at_1=round(hit_at(1), 6),
        hit_at_5=round(hit_at(5), 6),
        hit_at_8=round(hit_at(8), 6),
        mrr_at_8=round(mrr, 6),
        negative_false_context_rate=round(negative_fcr, 6),
        category_metrics=category_metrics,
        positive_outcomes=positive_outcomes,
        negative_outcomes=negative_outcomes,
    )


def _source_name(source_file: str) -> str:
    normalised = source_file.replace("\\", "/")
    return PurePosixPath(normalised).name


def _normalise_source(value: str) -> str:
    return _normalise(_source_name(unicodedata.normalize("NFKC", value)))


def _normalise_heading(value: str) -> str:
    # ``>`` 是 parser 生成 heading_path 的层级分隔符；只规范分隔符两侧空白，
    # 不删除父级或标点，因而父/子/兄弟标题仍严格不相等。
    normalised = unicodedata.normalize("NFKC", value)
    return ">".join(_normalise(part) for part in normalised.split(">"))


def _target_key(target: ExpectedTarget) -> tuple[str, ...]:
    return (
        _normalise_source(target.source),
        _normalise_heading(target.heading),
        *(_normalise(evidence) for evidence in target.required_evidence),
    )


def _block_ref(block_id: str) -> str:
    """把内部 block_id 转成稳定、不可逆且不含路径/正文的报告引用。"""

    digest = hashlib.sha256(block_id.encode("utf-8", errors="replace")).hexdigest()
    return f"blk_{digest[:16]}"


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
