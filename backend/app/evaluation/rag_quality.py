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
from collections.abc import Collection, Mapping, Sequence
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
    # ``context_text`` is provenance-safe same-heading evidence assembled by the production
    # retriever (for example a 41-character formula next to its field-definition table). The
    # model sees this effective context, so the evaluator must score the same payload.
    return _target_matches(
        source=chunk.source_file,
        headings=actual_headings,
        text=chunk.context_text or chunk.text,
        target=target,
        normalised_headings=True,
    )


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


def _normalise_heading(value: str | None) -> str:
    # ``>`` 是 parser 生成 heading_path 的层级分隔符；只规范分隔符两侧空白，
    # 不删除父级或标点，因而父/子/兄弟标题仍严格不相等。
    normalised = unicodedata.normalize("NFKC", value or "")
    return ">".join(_normalise(part) for part in normalised.split(">"))


def _target_key(target: ExpectedTarget) -> tuple[str, ...]:
    if target.source is None:
        # 脱敏结构：来源/标题/证据正文都不在仓库，只能用 target_id 判重。
        return (target.target_id,)
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


# ---------------------------------------------------------------------------
# 分级相关性数据集与六项指标（PRD 7.1 标注单位 / 7.2 真实数据集 / 10.1 指标定义）
#
# 与旧黄金集（PositiveCase/NegativeCase，严格块命中）并存：旧口径继续为 A02 服务，
# 本节新增分级相关性（0/1/2/3 + unjudged）与六项指标，服务于 A03–A06。
# ---------------------------------------------------------------------------

RELEVANCE_SCHEMA_VERSION = 1
RELEVANCE_FAMILIES = ("C01", "C02", "C03", "C04", "C05", "C06")
RELEVANCE_SPLITS = ("calibration", "test")
TARGET_ROLES = ("required", "allowed", "forbidden")
REVIEW_STATUSES = ("reviewed", "unjudged")

#: PRD 7.1 四个等级的语义（0/1 排除，2 保留候选资格，3 优先保留）。
GRADE_IRRELEVANT = 0
GRADE_TOPIC_ONLY = 1
GRADE_SUPPORTING = 2
GRADE_DIRECT = 3


@dataclass(frozen=True)
class EvidenceSpan:
    """一段原文范围（字符偏移），用于无关字符比例与人工核对。

    ``sha256`` 是该范围原始文本的哈希，保证偏移对应的文字没有被静默改写；
    本地私有数据集可以附带 ``text``（必须与偏移和哈希一致），评测时优先按 ``text``
    在最终正文中定位——生产视图的正文可能带有同标题补充文字，偏移不再一一对应。
    """

    start: int
    end: int
    sha256: str
    text: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("EvidenceSpan.start 必须是整数")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("EvidenceSpan.end 必须是整数")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("EvidenceSpan 必须是 start>=0 且 end>start 的半开区间")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("EvidenceSpan.sha256 必须是 64 位十六进制")
        if any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("EvidenceSpan.sha256 必须是小写十六进制")
        if self.text is not None:
            if not isinstance(self.text, str):
                raise TypeError("EvidenceSpan.text 必须是字符串")
            if len(self.text) != self.end - self.start:
                raise ValueError("EvidenceSpan.text 长度必须等于 end-start")
            if hashlib.sha256(self.text.encode("utf-8", errors="replace")).hexdigest() != self.sha256:
                raise ValueError("EvidenceSpan.text 与 sha256 不一致")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RelevanceTarget:
    """一个已标注候选目标：必要证据、允许的互补证据或明确排除的难负例。

    ``group`` 表示证据族：互为等价的不同块共用同一 group，指标按组计一次。
    ``grade`` 为 PRD 7.1 的 0/1/2/3；``None`` 表示该候选尚未审阅（unjudged）。

    仓库内交付的是**脱敏结构**（PRD §7.2）：``source``/``heading``/``required_evidence``
    的正文不在仓库里，解析时留 ``None``/空值；必须由 ``scripts/relevance_cases/loader.py``
    与本地私文组装后才可用于评测（未组装就评测会抛出明确错误，绝不静默当作无关）。
    """

    target_id: str
    role: str
    grade: int | None
    source: str | None
    heading: str | None = ""
    required_evidence: tuple[str, ...] = ()
    group: str = ""
    irrelevant_spans: tuple[EvidenceSpan, ...] = ()
    relevant_spans: tuple[EvidenceSpan, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("目标 target_id 不能为空")
        if self.source is not None:
            if not isinstance(self.source, str) or not self.source.strip():
                raise ValueError(f"目标 source 不能为空：{self.target_id!r}")
            if self.source != _source_name(self.source):
                raise ValueError(f"目标 source 必须是文件 basename：{self.target_id!r}")
        if self.heading is not None and not isinstance(self.heading, str):
            raise TypeError(f"目标 heading 必须是字符串：{self.target_id!r}")
        if self.role not in TARGET_ROLES:
            raise ValueError(f"目标 role 必须是 {TARGET_ROLES}：{self.target_id!r}")
        # 真实语料里存在没有标题的块；这类目标必须靠证据片段区分，否则无法唯一定位。
        # 脱敏结构（source 为 None）不适用本检查：等价性在组装后由
        # ``validate_relevance_cases`` 与加载器负责。
        if (
            self.source is not None
            and not (self.heading or "").strip()
            and not any(str(item).strip() for item in self.required_evidence)
        ):
            raise ValueError(f"无标题目标必须提供证据片段：{self.target_id!r}")
        if self.grade is not None:
            if isinstance(self.grade, bool) or not isinstance(self.grade, int):
                raise TypeError(f"目标 grade 必须是 0-3 或 None：{self.target_id!r}")
            if self.grade not in (0, 1, 2, 3):
                raise ValueError(f"目标 grade 必须是 0-3 或 None：{self.target_id!r}")
        if self.role == "required" and self.source is not None:
            # 脱敏结构里证据原文不在仓库，只有组装本地私文后才能校验这条。
            if not self.required_evidence:
                raise ValueError(f"必要证据目标必须提供 required_evidence：{self.target_id!r}")
            if self.grade is not None and self.grade < GRADE_SUPPORTING:
                raise ValueError(
                    f"必要证据目标的等级必须为 2 或 3：{self.target_id!r}"
                )
        if self.role == "forbidden" and self.grade is not None and self.grade > GRADE_TOPIC_ONLY:
            raise ValueError(
                f"排除目标的等级必须为 0 或 1：{self.target_id!r}"
            )
        object.__setattr__(
            self, "required_evidence", tuple(_normalise(item) for item in self.required_evidence)
        )
        for span in (*self.irrelevant_spans, *self.relevant_spans):
            if not isinstance(span, EvidenceSpan):
                raise TypeError(f"span 必须是 EvidenceSpan：{self.target_id!r}")

    @property
    def is_unjudged(self) -> bool:
        return self.grade is None


@dataclass(frozen=True)
class RelevanceCase:
    """一条真实语料问题的分级标注；分组规则由 ``validate_relevance_cases`` 检查。

    ``query`` 为 ``None`` 表示这是一条**脱敏结构案例**（PRD §7.2：问题原文不入仓库），
    必须先用 ``scripts/relevance_cases/loader.py`` 组装本地私文；未组装的案例不得评测。
    """

    case_id: str
    family: str
    group_id: str
    split: str
    query: str | None
    targets: tuple[RelevanceTarget, ...] = ()
    needs_answer: bool = True
    scope_id: str = "global"
    review_status: str = "reviewed"
    corpus_sha256: str | None = None
    note: str = ""
    query_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("case_id", "group_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"案例 {field_name} 不能为空")
        if self.query is not None and (not isinstance(self.query, str) or not self.query.strip()):
            raise ValueError(f"案例 query 不能为空：{self.case_id!r}")
        if self.query_sha256 is not None:
            if not isinstance(self.query_sha256, str) or len(self.query_sha256) != 64:
                raise ValueError(f"案例 query_sha256 必须是 64 位十六进制：{self.case_id!r}")
            if any(char not in "0123456789abcdef" for char in self.query_sha256):
                raise ValueError(f"案例 query_sha256 必须是小写十六进制：{self.case_id!r}")
        if self.family not in RELEVANCE_FAMILIES:
            raise ValueError(f"案例 family 必须是 C01-C06：{self.case_id!r}")
        if self.split not in RELEVANCE_SPLITS:
            raise ValueError(f"案例 split 必须是 calibration/test：{self.case_id!r}")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"案例 review_status 必须是 reviewed/unjudged：{self.case_id!r}")
        if not isinstance(self.needs_answer, bool):
            raise TypeError(f"案例 needs_answer 必须是布尔值：{self.case_id!r}")
        keys: list[tuple[str, ...]] = []
        for target in self.targets:
            if not isinstance(target, RelevanceTarget):
                raise TypeError(f"targets 只能包含 RelevanceTarget：{self.case_id!r}")
            keys.append(_target_key(target))
        if len(keys) != len(set(keys)):
            raise ValueError(f"同一案例不能包含重复目标：{self.case_id!r}")

    @property
    def is_redacted(self) -> bool:
        """是否为未组装的脱敏结构案例（问题原文与来源/标题/证据正文不在仓库内）。"""

        return self.query is None or any(target.source is None for target in self.targets)

    @property
    def required_targets(self) -> tuple[RelevanceTarget, ...]:
        return tuple(target for target in self.targets if target.role == "required")

    @property
    def allowed_targets(self) -> tuple[RelevanceTarget, ...]:
        return tuple(target for target in self.targets if target.role == "allowed")

    @property
    def forbidden_targets(self) -> tuple[RelevanceTarget, ...]:
        return tuple(target for target in self.targets if target.role == "forbidden")

    def required_groups(self) -> dict[str, tuple[RelevanceTarget, ...]]:
        """必要证据族：同族等价目标只计一次；没写 group 的用 target_id 自成一族。"""

        groups: dict[str, list[RelevanceTarget]] = {}
        for target in self.required_targets:
            groups.setdefault(target.group or target.target_id, []).append(target)
        return {key: tuple(value) for key, value in groups.items()}

    @property
    def has_unjudged_targets(self) -> bool:
        return any(target.is_unjudged for target in self.targets)


@dataclass(frozen=True)
class RelevanceView:
    """一轮检索最终返回或进入评分集合的视图；``text`` 是实际评分/送入模型的正文。"""

    block_ref: str
    source: str
    heading: str
    text: str
    similarity: float | None = None
    rerank_score: float | None = None
    match_type: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("视图 text 必须是字符串")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("视图 source 不能为空")
        if not isinstance(self.block_ref, str) or not self.block_ref.strip():
            raise ValueError("视图 block_ref 不能为空")

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> RelevanceView:
        """从生产 ``RetrievedChunk`` 构造视图：正文取生产同口径的 ``context_text or text``。"""

        return cls(
            block_ref=_block_ref(chunk.block_id),
            source=_source_name(chunk.source_file),
            heading=chunk.heading_path or "",
            text=chunk.context_text or chunk.text,
            similarity=chunk.similarity,
            match_type=getattr(chunk, "match_type", "") or "",
        )


def _target_matches(
    *,
    source: str,
    headings: Collection[str],
    text: str,
    target: RelevanceTarget | ExpectedTarget,
    normalised_headings: bool = False,
) -> bool:
    """来源 basename + 完整标题 + 证据片段必须同时命中（旧黄金集与分级集共用口径）。"""

    if target.source is None:
        raise ValueError(
            f"脱敏结构案例尚未组装本地私文，不能评测：{target.target_id!r} 缺少 source；"
            "请用 scripts/relevance_cases/loader.py 组装后重试"
        )
    if _normalise_source(source) != _normalise_source(target.source):
        return False
    expected_heading = _normalise_heading(target.heading)
    if normalised_headings:
        actual_headings = set(headings)
    else:
        actual_headings = {_normalise_heading(heading) for heading in headings}
    if expected_heading not in actual_headings:
        return False
    normalised_text = _normalise(text)
    return all(_normalise(evidence) in normalised_text for evidence in target.required_evidence)


def judge_view(case: RelevanceCase, view: RelevanceView) -> int | None:
    """返回该视图的人工等级；没有任何标注目标命中时为 ``None``（unjudged）。

    未列出的分块不能因为“不在黄金集里”就被当成无关（PRD 7.1），因此这里显式返回
    ``None``，由报告层判定该次 Precision 无效。
    """

    for target in case.targets:
        if _target_matches(
            source=view.source,
            headings=(view.heading,),
            text=view.text,
            target=target,
        ):
            return target.grade
    return None


def load_relevance_cases(
    path: Path,
    *,
    expected_corpus_sha256: str | None = None,
    private_dataset: Mapping[str, Any] | None = None,
    allow_redacted: bool = False,
) -> tuple[RelevanceCase, ...]:
    """读取分级相关性数据集；schema、分组与语料哈希立即校验。

    PRD §7.2：仓库内的 ``cases-C*.json`` 是**脱敏结构**（问题原文/来源 basename/标题/
    证据片段正文不入仓库）。``private_dataset`` 传入同一 family 的本地私文（Git 忽略）
    后，本函数按 ``case_id``/``target_id`` 组装成完整数据集；``allow_redacted=True``
    允许在缺少私文时返回只含非私密字段的结构案例（本体不参与评测）。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("相关性数据集顶层必须是 JSON object")
    required_top_level = {"schema_version", "family", "corpus_sha256", "cases"}
    optional_top_level = {
        "dataset_id",
        "annotation",
        "annotations",
        "annotation_protocol",
        "notes",
        "redaction",
        "counts",
    }
    unknown = set(payload) - required_top_level - optional_top_level
    if unknown:
        raise ValueError(f"相关性数据集包含未知顶层字段：{sorted(unknown)!r}")
    missing = required_top_level - set(payload)
    if missing:
        raise ValueError(f"相关性数据集缺少顶层字段：{sorted(missing)!r}")
    if payload["schema_version"] != RELEVANCE_SCHEMA_VERSION:
        raise ValueError(f"相关性数据集 schema_version 必须为 {RELEVANCE_SCHEMA_VERSION}")
    family = payload["family"]
    if family not in RELEVANCE_FAMILIES:
        raise ValueError(f"相关性数据集 family 必须是 C01-C06：{family!r}")
    corpus_sha256 = payload["corpus_sha256"]
    if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
        raise ValueError("相关性数据集必须记录 64 位语料内容哈希")
    if expected_corpus_sha256 is not None and corpus_sha256 != expected_corpus_sha256:
        raise ValueError(
            f"相关性数据集语料哈希已过期：{corpus_sha256[:12]} != {expected_corpus_sha256[:12]}"
        )
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise TypeError("相关性数据集 cases 必须是非空 JSON array")
    declared_redaction = payload.get("redaction")
    if declared_redaction and private_dataset is None and not allow_redacted:
        raise ValueError(
            f"相关性数据集 {Path(path).name} 声明为脱敏结构（PRD §7.2），"
            "但未提供本地私文；请用 scripts/relevance_cases/loader.py 组装，"
            "或显式传 allow_redacted=True 只读取结构"
        )
    merged = _merge_private_cases(raw_cases, family, private_dataset) if private_dataset else None
    raw_cases = merged if merged is not None else raw_cases
    cases = tuple(
        _parse_relevance_case(item, family, allow_redacted=allow_redacted) for item in raw_cases
    )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("相关性数据集 case_id 必须唯一")
    for case in cases:
        if case.corpus_sha256 is not None and case.corpus_sha256 != corpus_sha256:
            raise ValueError(f"案例记录的语料哈希与数据集不一致：{case.case_id!r}")
    validate_relevance_cases(cases, allow_redacted=allow_redacted)
    return cases


def _merge_private_cases(
    raw_cases: Sequence[Any],
    family: str,
    private_dataset: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """把本地私文按 ``case_id``/``target_id`` 组装回脱敏结构案例。"""

    if not isinstance(private_dataset, Mapping):
        raise TypeError("本地私文数据集必须是 JSON object")
    if private_dataset.get("family") != family:
        raise ValueError(
            f"本地私文 family={private_dataset.get('family')!r} 与结构文件 family={family!r} 不一致"
        )
    private_cases = private_dataset.get("cases")
    if not isinstance(private_cases, list) or not private_cases:
        raise ValueError("本地私文数据集 cases 必须是非空数组")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in private_cases:
        if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str):
            raise TypeError("本地私文案例必须是带 case_id 的 JSON object")
        if item["case_id"] in by_id:
            raise ValueError(f"本地私文 case_id 重复：{item['case_id']!r}")
        by_id[item["case_id"]] = item

    merged: list[dict[str, Any]] = []
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise TypeError("相关性数据集的案例必须是 JSON object")
        case_id = raw.get("case_id")
        private_case = by_id.get(str(case_id))
        if private_case is None:
            raise ValueError(f"本地私文缺少案例：{case_id!r}")
        item = dict(raw)
        query = private_case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"本地私文的 query 不能为空：{case_id!r}")
        expected_query_hash = raw.get("query_sha256")
        if expected_query_hash is not None and _sha256_text(query) != expected_query_hash:
            raise ValueError(
                f"本地私文与结构文件不匹配（query 哈希不同）：{case_id!r}"
            )
        item["query"] = query
        if private_case.get("note") is not None:
            item["note"] = private_case.get("note")

        private_targets = private_case.get("targets")
        if not isinstance(private_targets, list):
            raise TypeError(f"本地私文的 targets 必须是数组：{case_id!r}")
        by_target: dict[str, Mapping[str, Any]] = {}
        for target in private_targets:
            if not isinstance(target, Mapping) or not isinstance(target.get("target_id"), str):
                raise TypeError(f"本地私文目标必须是带 target_id 的 JSON object：{case_id!r}")
            by_target[target["target_id"]] = target
        raw_targets = raw.get("targets")
        if not isinstance(raw_targets, list):
            raise TypeError(f"案例 targets 必须是 JSON array：{case_id!r}")
        merged_targets: list[dict[str, Any]] = []
        for target in raw_targets:
            if not isinstance(target, Mapping):
                raise TypeError(f"targets 中的元素必须是 JSON object：{case_id!r}")
            target_id = target.get("target_id")
            private_target = by_target.get(str(target_id))
            if private_target is None:
                raise ValueError(f"本地私文缺少目标：{case_id!r}/{target_id!r}")
            source = private_target.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"本地私文的 target source 不能为空：{target_id!r}")
            entry = dict(target)
            entry["source"] = source
            entry["heading"] = private_target.get("heading") or ""
            entry["required_evidence"] = list(private_target.get("required_evidence") or [])
            entry["irrelevant_spans"] = list(private_target.get("irrelevant_spans") or [])
            entry["relevant_spans"] = list(private_target.get("relevant_spans") or [])
            if private_target.get("note") is not None:
                entry["note"] = private_target.get("note")
            merged_targets.append(entry)
        if len(by_target) != len(merged_targets):
            extra = sorted(set(by_target) - {str(item["target_id"]) for item in merged_targets})
            raise ValueError(f"本地私文含结构文件里没有的目标：{case_id!r} {extra!r}")
        item["targets"] = merged_targets
        merged.append(item)
    extra_cases = sorted(set(by_id) - {str(raw.get("case_id")) for raw in raw_cases})
    if extra_cases:
        raise ValueError(f"本地私文含结构文件里没有的案例：{extra_cases!r}")
    return merged


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_relevance_dataset(
    paths: Sequence[Path],
    *,
    expected_corpus_sha256: str | None = None,
    private_by_family: Mapping[str, Mapping[str, Any]] | None = None,
    allow_redacted: bool = False,
):
    """合并读取多个 family 文件，并跨文件校验 case_id 与分组唯一性。"""

    merged: list[RelevanceCase] = []
    for path in paths:
        family = Path(path).stem.removeprefix("cases-")
        private = (private_by_family or {}).get(family)
        merged.extend(
            load_relevance_cases(
                path,
                expected_corpus_sha256=expected_corpus_sha256,
                private_dataset=private,
                allow_redacted=allow_redacted,
            )
        )
    ids = [case.case_id for case in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("跨数据集合并后 case_id 必须唯一")
    validate_relevance_cases(tuple(merged), allow_redacted=allow_redacted)
    return tuple(merged)


def validate_relevance_cases(
    cases: Sequence[RelevanceCase],
    *,
    allow_redacted: bool = False,
) -> None:
    """两级分组规则（PRD 7.2）。

    - **case 级 ``group_id``**：同一原始问题的改写、同一关键证据族、复用的干扰分块族，
      整组只进入一个集合（calibration/test 二选一）。
    - **target 级 ``group``**：只有真正互为等价（同一事实的不同块表述）的目标才共用，
      且必须按案例限定为 ``<case_id>:<族名>``，不得跨题、跨集合复用；
      留空表示该目标自成一族（按 ``target_id`` 计一次）。
    """

    split_by_group: dict[str, str] = {}
    family_by_group: dict[str, str] = {}
    for case in cases:
        if case.is_redacted and not allow_redacted:
            raise ValueError(
                f"脱敏结构案例尚未组装本地私文：{case.case_id!r}；"
                "评测前请用 scripts/relevance_cases/loader.py 与本地私文组装"
            )
        previous = split_by_group.setdefault(case.group_id, case.split)
        if previous != case.split:
            raise ValueError(
                f"分组 {case.group_id!r} 同时出现在 {previous} 与 {case.split}，"
                "同源改写/同证据族必须整组只进一个集合"
            )
        previous_family = family_by_group.setdefault(case.group_id, case.family)
        if previous_family != case.family:
            raise ValueError(f"分组 {case.group_id!r} 跨了类别")

        members_by_group: dict[str, list[RelevanceTarget]] = {}
        for target in case.targets:
            if not target.group:
                continue
            if not target.group.startswith(f"{case.case_id}:"):
                raise ValueError(
                    f"target 级证据族 {target.group!r} 必须按案例限定为 "
                    f"{case.case_id}:<族名>，不得跨题复用"
                )
            members_by_group.setdefault(target.group, []).append(target)
        for group, members in members_by_group.items():
            if len(members) < 2:
                raise ValueError(f"证据族 {group!r} 只有 1 个成员，等价性不成立，应留空 group")
            if any(target.source is None or target.heading is None for target in members):
                # 脱敏结构里没有来源/标题正文，等价性只能等组装后再查。
                continue
            keys = {(target.source, target.heading) for target in members}
            if len(keys) != len(members):
                raise ValueError(f"证据族 {group!r} 的成员必须来自不同的块")


@dataclass(frozen=True)
class CaseRelevanceOutcome:
    """逐题结果：直接支撑“逐题、逐类、整体”三档报告。"""

    case_id: str
    family: str
    split: str
    needs_answer: bool
    scored_count: int
    returned_count: int
    returned_grades: tuple[int | None, ...]
    relevant_returned: int
    unjudged_returned: int
    unjudged_scored: int
    forbidden_returned: int
    required_group_total: int
    required_group_scored: int
    required_group_returned: int
    hard_negative_scored: int
    hard_negative_admitted: int
    judged_chars: int
    irrelevant_chars: int
    span_mismatch: int
    precision: float | None
    precision_defined: bool
    evidence_recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelevanceMetrics:
    """一个切片（整体 / 某类 / 某集合）的六项指标。"""

    label: str
    case_total: int
    answered_case_total: int
    empty_case_total: int
    empty_result_rate: float
    returned_view_total: int
    relevant_view_total: int
    precision_macro: float | None
    precision_micro: float | None
    precision_defined_case_total: int
    required_group_total: int
    required_group_scored: int
    required_group_covered: int
    candidate_coverage: float | None
    required_evidence_recall: float | None
    required_evidence_recall_macro: float | None
    hard_negative_scored: int
    hard_negative_admitted: int
    hard_negative_admission_rate: float | None
    no_evidence_case_total: int
    no_evidence_false_context: int
    empty_false_context_rate: float | None
    irrelevant_char_total: int
    judged_char_total: int
    irrelevant_character_ratio: float | None
    unjudged_view_total: int
    unjudged_scored_total: int
    precision_valid: bool
    annotation_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelevanceReport:
    generated_at_unix: float
    dataset_case_total: int
    overall: RelevanceMetrics
    families: dict[str, RelevanceMetrics]
    splits: dict[str, RelevanceMetrics]
    outcomes: tuple[CaseRelevanceOutcome, ...]
    unjudged_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_unix": self.generated_at_unix,
            "dataset_case_total": self.dataset_case_total,
            "overall": self.overall.to_dict(),
            "families": {key: value.to_dict() for key, value in self.families.items()},
            "splits": {key: value.to_dict() for key, value in self.splits.items()},
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "unjudged_case_ids": list(self.unjudged_case_ids),
        }


def evaluate_relevance(
    cases: Sequence[RelevanceCase],
    *,
    scored: Mapping[str, Sequence[RelevanceView]],
    returned: Mapping[str, Sequence[RelevanceView]],
) -> RelevanceReport:
    """计算 PRD 10.1 六项指标，并给出逐题 / 逐类 / 整体三档结果。

    - ``scored``：进入评分集合的候选（对应 CandidateCoverage 与难负例分母）。
    - ``returned``：最终进入生成上下文/来源的视图（对应 Precision、Recall 等）。
    - 逐题 Precision 只统计“有返回结果”的问题；空结果记为未定义并单独报告空结果率。
    - 任一返回视图为 unjudged 时 ``precision_valid=False``，该次 Precision 不具备发布资格。
    - **未组装的脱敏结构一律拒收**：``case.is_redacted`` 为真说明问题原文/来源/证据正文
      不在仓库里（PRD §7.2），此时指标只会“看起来有效但全空”，必须立即报错，
      绝不让门禁把它误判成 A03/A04 质量不达标。
    """

    redacted = sorted({case.case_id for case in cases if case.is_redacted})
    if redacted:
        preview = "、".join(redacted[:5])
        suffix = "" if len(redacted) <= 5 else f" 等 {len(redacted)} 题"
        raise ValueError(
            f"数据集未组装：{preview}{suffix} 仍是脱敏结构（问题原文/来源/证据正文不在仓库里）；"
            "需要本地私文（Git 忽略的 .tmp_probe/rag-relevance/relevance-private/）或按 "
            "scripts/relevance_cases/README.md 用 scripts/relevance_cases/loader.py 组装后再评测"
        )

    outcomes = tuple(
        _judge_case(case, scored.get(case.case_id, ()), returned.get(case.case_id, ()))
        for case in cases
    )
    overall = _summarise("overall", outcomes)
    families = {
        family: _summarise(family, tuple(item for item in outcomes if item.family == family))
        for family in sorted({item.family for item in outcomes})
    }
    splits = {
        split: _summarise(split, tuple(item for item in outcomes if item.split == split))
        for split in sorted({item.split for item in outcomes})
    }
    unjudged_cases = tuple(
        sorted({item.case_id for item in outcomes if item.unjudged_returned > 0})
    )
    return RelevanceReport(
        generated_at_unix=time.time(),
        dataset_case_total=len(outcomes),
        overall=overall,
        families=families,
        splits=splits,
        outcomes=outcomes,
        unjudged_case_ids=unjudged_cases,
    )


def relevance_gate_passes(
    report: RelevanceReport,
    *,
    min_precision: float = 0.90,
    min_family_precision: float = 0.85,
    min_recall: float = 0.95,
    min_family_recall: float = 0.90,
    max_hard_negative_rate: float = 0.05,
    max_empty_false_context_rate: float = 0.0,
) -> tuple[bool, tuple[str, ...]]:
    """PRD 10.2 A03–A05 门禁；返回 (是否通过, 失败原因)。"""

    failures: list[str] = []
    if not report.overall.precision_valid:
        failures.append("整体 Precision 无效：存在未标注（unjudged）返回视图")
    if not report.overall.annotation_complete:
        failures.append("整体标注不完整：存在 span 与正文不一致的案例")
    overall_precision = report.overall.precision_macro
    if overall_precision is None or overall_precision < min_precision:
        failures.append(
            f"整体 ReturnedPrecision={overall_precision} < {min_precision}（A03）"
        )
    overall_recall = report.overall.required_evidence_recall
    if overall_recall is None or overall_recall < min_recall:
        failures.append(
            f"整体 RequiredEvidenceRecall={overall_recall} < {min_recall}（A04）"
        )
    rate = report.overall.hard_negative_admission_rate
    if rate is not None and rate > max_hard_negative_rate:
        failures.append(f"HardNegativeAdmissionRate={rate} > {max_hard_negative_rate}（A05）")
    if report.overall.empty_false_context_rate is None:
        failures.append("EmptyFalseContextRate 无样本（缺少无证据问题）（A05）")
    elif report.overall.empty_false_context_rate > max_empty_false_context_rate:
        failures.append(
            f"EmptyFalseContextRate={report.overall.empty_false_context_rate} "
            f"> {max_empty_false_context_rate}（A05）"
        )
    for family, metrics in sorted(report.families.items()):
        if (
            metrics.precision_valid
            and metrics.precision_macro is not None
            and metrics.precision_macro < min_family_precision
        ):
            failures.append(
                f"{family} ReturnedPrecision={metrics.precision_macro} "
                f"< {min_family_precision}（A03）"
            )
        if (
            metrics.required_evidence_recall is not None
            and metrics.required_evidence_recall < min_family_recall
        ):
            failures.append(
                f"{family} RequiredEvidenceRecall={metrics.required_evidence_recall} "
                f"< {min_family_recall}（A04）"
            )
        if family == "C05" and metrics.empty_false_context_rate not in (None, 0.0):
            failures.append(
                f"C05 无证据问题误放行率={metrics.empty_false_context_rate}（A05 要求 0）"
            )
    return (not failures, tuple(failures))


def _judge_case(
    case: RelevanceCase,
    scored_views: Sequence[RelevanceView],
    returned_views: Sequence[RelevanceView],
) -> CaseRelevanceOutcome:
    required_groups = case.required_groups()
    forbidden = case.forbidden_targets

    scored_grades = [judge_view(case, view) for view in scored_views]
    returned_grades = [judge_view(case, view) for view in returned_views]
    unjudged_scored = sum(1 for grade in scored_grades if grade is None)

    def _covers(
        groups: Mapping[str, Sequence[RelevanceTarget]],
        views: Sequence[RelevanceView],
    ) -> int:
        covered = 0
        for targets in groups.values():
            hit = any(
                _target_matches(
                    source=view.source,
                    headings=(view.heading,),
                    text=view.text,
                    target=target,
                )
                for view in views
                for target in targets
            )
            covered += int(hit)
        return covered

    hard_negative_scored = sum(
        1
        for view in scored_views
        if any(
            _target_matches(
                source=view.source, headings=(view.heading,), text=view.text, target=target
            )
            for target in forbidden
        )
    )
    hard_negative_admitted = sum(
        1
        for view in returned_views
        if any(
            _target_matches(
                source=view.source, headings=(view.heading,), text=view.text, target=target
            )
            for target in forbidden
        )
    )

    judged_chars = 0
    irrelevant_chars = 0
    span_mismatch = 0
    seen_spans: set[tuple[str, int, int]] = set()
    for view, grade in zip(returned_views, returned_grades, strict=True):
        if grade is None:
            continue
        judged_chars += len(view.text)
        if grade <= GRADE_TOPIC_ONLY:
            # 等级 0/1 的视图整体无关：未显式标注范围时，整段正文计入无关字符。
            irrelevant_chars += len(view.text)
            continue
        for target in case.targets:
            if not _target_matches(
                source=view.source, headings=(view.heading,), text=view.text, target=target
            ):
                continue
            for span in target.irrelevant_spans:
                key = (view.block_ref, span.start, span.end, span.text or "")
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                if span.text is not None and span.text in view.text:
                    # 生产视图正文可能附带同标题补充文字，按标注文本定位比按偏移更可靠。
                    irrelevant_chars += len(span.text)
                elif 0 <= span.start < span.end <= len(view.text):
                    irrelevant_chars += span.length
                else:
                    span_mismatch += 1

    relevant_returned = sum(1 for grade in returned_grades if grade is not None and grade >= 2)
    unjudged_returned = sum(1 for grade in returned_grades if grade is None)
    precision_defined = bool(returned_views) and unjudged_returned == 0 and span_mismatch == 0
    precision = (
        relevant_returned / len(returned_views) if precision_defined and returned_views else None
    )
    required_total = len(required_groups)
    return CaseRelevanceOutcome(
        case_id=case.case_id,
        family=case.family,
        split=case.split,
        needs_answer=case.needs_answer,
        scored_count=len(scored_views),
        returned_count=len(returned_views),
        returned_grades=tuple(returned_grades),
        relevant_returned=relevant_returned,
        unjudged_returned=unjudged_returned,
        unjudged_scored=unjudged_scored,
        forbidden_returned=hard_negative_admitted,
        required_group_total=required_total,
        required_group_scored=_covers(required_groups, scored_views),
        required_group_returned=_covers(required_groups, returned_views),
        hard_negative_scored=hard_negative_scored,
        hard_negative_admitted=hard_negative_admitted,
        judged_chars=judged_chars,
        irrelevant_chars=irrelevant_chars,
        span_mismatch=span_mismatch,
        precision=precision,
        precision_defined=precision_defined,
        evidence_recall=(_covers(required_groups, returned_views) / required_total)
        if required_total
        else None,
    )


def _summarise(label: str, outcomes: Sequence[CaseRelevanceOutcome]) -> RelevanceMetrics:
    if not outcomes:
        return RelevanceMetrics(
            label=label,
            case_total=0,
            answered_case_total=0,
            empty_case_total=0,
            empty_result_rate=0.0,
            returned_view_total=0,
            relevant_view_total=0,
            precision_macro=None,
            precision_micro=None,
            precision_defined_case_total=0,
            required_group_total=0,
            required_group_scored=0,
            required_group_covered=0,
            candidate_coverage=None,
            required_evidence_recall=None,
            required_evidence_recall_macro=None,
            hard_negative_scored=0,
            hard_negative_admitted=0,
            hard_negative_admission_rate=None,
            no_evidence_case_total=0,
            no_evidence_false_context=0,
            empty_false_context_rate=None,
            irrelevant_char_total=0,
            judged_char_total=0,
            irrelevant_character_ratio=None,
            unjudged_view_total=0,
            unjudged_scored_total=0,
            precision_valid=False,
            annotation_complete=True,
        )

    answered = [item for item in outcomes if item.returned_count > 0]
    empty = [item for item in outcomes if item.returned_count == 0]
    precision_cases = [
        item for item in answered if item.precision_defined and item.precision is not None
    ]
    macro = (
        sum(item.precision for item in precision_cases if item.precision is not None)
        / len(precision_cases)
        if precision_cases
        else None
    )
    returned_views = sum(item.returned_count for item in outcomes)
    relevant_views = sum(item.relevant_returned for item in outcomes)
    micro = relevant_views / returned_views if returned_views else None

    required_total = sum(item.required_group_total for item in outcomes)
    required_scored = sum(item.required_group_scored for item in outcomes)
    required_covered = sum(item.required_group_returned for item in outcomes)
    recall_cases = [item for item in outcomes if item.required_group_total]
    recall_macro = (
        sum(item.evidence_recall for item in recall_cases if item.evidence_recall is not None)
        / len(recall_cases)
        if recall_cases
        else None
    )

    hard_scored = sum(item.hard_negative_scored for item in outcomes)
    hard_admitted = sum(item.hard_negative_admitted for item in outcomes)

    no_evidence = [item for item in outcomes if not item.needs_answer]
    no_evidence_false = sum(1 for item in no_evidence if item.returned_count > 0)

    judged_chars = sum(item.judged_chars for item in outcomes)
    irrelevant_chars = sum(item.irrelevant_chars for item in outcomes)
    unjudged_views = sum(item.unjudged_returned for item in outcomes)
    unjudged_scored = sum(item.unjudged_scored for item in outcomes)
    span_mismatches = sum(item.span_mismatch for item in outcomes)

    return RelevanceMetrics(
        label=label,
        case_total=len(outcomes),
        answered_case_total=len(answered),
        empty_case_total=len(empty),
        empty_result_rate=round(len(empty) / len(outcomes), 6),
        returned_view_total=returned_views,
        relevant_view_total=relevant_views,
        precision_macro=round(macro, 6) if macro is not None else None,
        precision_micro=round(micro, 6) if micro is not None else None,
        precision_defined_case_total=len(precision_cases),
        required_group_total=required_total,
        required_group_scored=required_scored,
        required_group_covered=required_covered,
        candidate_coverage=round(required_scored / required_total, 6) if required_total else None,
        required_evidence_recall=round(required_covered / required_total, 6)
        if required_total
        else None,
        required_evidence_recall_macro=round(recall_macro, 6) if recall_macro is not None else None,
        hard_negative_scored=hard_scored,
        hard_negative_admitted=hard_admitted,
        hard_negative_admission_rate=round(hard_admitted / hard_scored, 6) if hard_scored else None,
        no_evidence_case_total=len(no_evidence),
        no_evidence_false_context=no_evidence_false,
        empty_false_context_rate=round(no_evidence_false / len(no_evidence), 6)
        if no_evidence
        else None,
        irrelevant_char_total=irrelevant_chars,
        judged_char_total=judged_chars,
        irrelevant_character_ratio=round(irrelevant_chars / judged_chars, 6)
        if judged_chars
        else None,
        unjudged_view_total=unjudged_views,
        unjudged_scored_total=unjudged_scored,
        precision_valid=unjudged_views == 0 and span_mismatches == 0,
        annotation_complete=span_mismatches == 0,
    )


def _parse_relevance_case(
    raw: object, dataset_family: str, *, allow_redacted: bool = False
) -> RelevanceCase:
    if not isinstance(raw, dict):
        raise TypeError("相关性数据集的案例必须是 JSON object")
    allowed = {
        "case_id",
        "family",
        "group_id",
        "split",
        "query",
        "query_sha256",
        "case_sha256",
        "needs_answer",
        "scope_id",
        "review_status",
        "corpus_sha256",
        "targets",
        "note",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"案例包含未知字段：{sorted(unknown)!r}")
    redacted = "query" not in raw or any(
        isinstance(item, dict) and "source" not in item for item in raw.get("targets") or []
    )
    if redacted and not allow_redacted:
        raise ValueError(
            f"案例是未组装的脱敏结构（缺少 query 或目标 source）："
            f"{raw.get('case_id')!r}；评测前请用 scripts/relevance_cases/loader.py 组装本地私文"
        )
    for field_name in ("case_id", "group_id", "split"):
        if field_name not in raw:
            raise ValueError(f"案例缺少字段：{field_name}")
    family = raw.get("family", dataset_family)
    if family != dataset_family:
        raise ValueError("案例 family 必须与数据集 family 一致")
    targets = raw.get("targets")
    if not isinstance(targets, list):
        raise TypeError("案例 targets 必须是 JSON array")
    return RelevanceCase(
        case_id=raw["case_id"],
        family=family,
        group_id=raw["group_id"],
        split=raw["split"],
        query=raw.get("query"),
        targets=tuple(_parse_relevance_target(item) for item in targets),
        needs_answer=raw.get("needs_answer", True),
        scope_id=raw.get("scope_id", "global"),
        review_status=raw.get("review_status", "reviewed"),
        corpus_sha256=raw.get("corpus_sha256"),
        note=raw.get("note", ""),
        query_sha256=raw.get("query_sha256"),
    )


def _parse_relevance_target(raw: object) -> RelevanceTarget:
    if not isinstance(raw, dict):
        raise TypeError("targets 中的元素必须是 JSON object")
    allowed = {
        "target_id",
        "role",
        "grade",
        "source",
        "heading",
        "required_evidence",
        "group",
        "irrelevant_spans",
        "relevant_spans",
        "note",
        "target_sha256",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"目标包含未知字段：{sorted(unknown)!r}")
    grade = raw.get("grade")
    if grade == "unjudged":
        grade = None
    return RelevanceTarget(
        target_id=raw["target_id"],
        role=raw["role"],
        grade=grade,
        source=raw.get("source"),
        heading=raw.get("heading", ""),
        required_evidence=tuple(_parse_evidence(raw.get("required_evidence", []), "required_evidence")),
        group=raw.get("group", ""),
        irrelevant_spans=_parse_spans(raw.get("irrelevant_spans", []), "irrelevant_spans"),
        relevant_spans=_parse_spans(raw.get("relevant_spans", []), "relevant_spans"),
        note=raw.get("note", ""),
    )


def _parse_spans(raw: object, field_name: str) -> tuple[EvidenceSpan, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} 必须是 JSON array")
    spans: list[EvidenceSpan] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"{field_name} 中的范围必须是 JSON object")
        if set(item) not in ({"start", "end", "sha256"}, {"start", "end", "sha256", "text"}):
            raise ValueError(f"{field_name} 范围只能包含 start/end/sha256(+text)")
        spans.append(
            EvidenceSpan(
                start=item["start"],
                end=item["end"],
                sha256=item["sha256"],
                text=item.get("text"),
            )
        )
    return tuple(spans)
