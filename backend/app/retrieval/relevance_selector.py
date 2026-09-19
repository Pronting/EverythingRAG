"""相关性筛选器：候选窗口 → 批次评分 → 阈值筛选 → 最终选择（PRD RAG-REL-001 §5.1/5.2/5.5/5.6）。

流程（PRD §5.1 第 4–7 步）：

1. ``window_candidates`` 按档案的候选上限做确定性截取，RRF 降序、同分按稳定
   ``candidate_id`` 升序；被截掉的候选记 ``outside_candidate_window``。统计包含补充视图，
   并同时报告截取前后的候选集合，便于评测计算截取前后的证据覆盖。
2. ``score_candidates`` 把**同一批**候选一次性提交给 Provider（``top_n`` 等于候选数），
   响应逐位校验后得到每个候选恰好一个有限分数；本项目从不跨请求合并相对分数。
3. ``select_relevance`` 只保留 ``rerank_score >= profile.score_threshold`` 的视图；
   分数同值依次按 RRF 分数、``candidate_id`` 排序；随后按「原文范围组」去重：组合
   （``support_bundle``）合格即代表整组、其成员按 ``covered_by_bundle`` 排除；组合自己
   未入选（超预算/超出上限）时成员作为后备继续参与，绝不互相误排除。最后按顺序加入，
   直到达到来源上限或正文预算。超过剩余预算的视图跳过并记 ``budget_exceeded``，**绝不**
   从视图中间截取已评分文本。最终 ``selected`` 里每个原文范围只出现一次。

硬约束：

* 阈值完全来自档案（校准产物），代码里没有兜底默认值；关键词数量、旧词法补充条件与
  章节补足都不参与最终入选，只作候选阶段诊断信号（PRD §5.2/§5.6）。
* 四种返回状态：``selected`` / ``empty`` / ``failed`` / ``cancelled``；服务失败绝不
  伪装成「无相关资料」（PRD §8.4）。
* 只依赖 ``reranker`` 的 Provider 接口与统一错误类型；不解析任何 HTTP 响应。
* 长度/token 限制由 Provider 层按 ``RerankProviderCapabilities`` 校验（不硬编码 4,000，
  也不用字符数代替 token 数）；本模块不重复实现 token 计数与候选收缩。
* 图片文字更新后的「再执行一次重排和筛选」（PRD §5.5）由调用方对更新后的候选列表
  再次调用本模块完成：同一请求固定候选集合的评分口径不变。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.retrieval.reranker import (
    EVIDENCE_CHAR_BUDGET,
    MAX_SELECTED_SOURCES,
    EvidenceVersionChangedError,
    ExclusionReason,
    ExclusionRecord,
    RerankError,
    RerankInvalidResponseError,
    RerankProfile,
    RerankProvider,
    RerankRequest,
    RerankScorable,
    RerankScore,
    RerankUsage,
    RerankValidationError,
    SelectionResult,
    SelectionStatus,
    ViewKind,
)

__all__ = [
    "SCORING_PROTOCOL_VERSION",
    "CandidateWindow",
    "ExclusionReason",
    "ExclusionRecord",
    "ScoredCandidate",
    "ScoredSelection",
    "ScoringOutcome",
    "SelectionResult",
    "SelectionStatus",
    "relevance_profile",
    "score_candidates",
    "select_from_scores",
    "select_relevance",
    "select_relevance_with_scores",
    "window_candidates",
]

#: 评分协议版本：候选窗口规则、批次提交方式、阈值比较与最终排序口径。
#: 视图版本或本常量变化都会使校准档案失配（RerankProfile.compatibility_key）。
SCORING_PROTOCOL_VERSION = "rerank-protocol-v1"

#: 相对分有效范围（PRD §6.2「有限数值与范围」）。
#: Provider 自声明的范围由评分路径的 ``RerankResult.assert_scores_in_range(capabilities)``
#: 强制执行；档案本身**不携带**范围字段，因此最终选择阶段用评分协议约定的 ``[0, 1]`` 相对分
#: 区间兜底。任何 NaN / ±Inf / 区间外分数都立即失败，绝不入选——``NaN < threshold`` 恒为假，
#: 不显式拒绝就会把「非数值」当成「不低于阈值」放进上下文。
_DEFAULT_SCORE_RANGE = (0.0, 1.0)

#: 可直接提交评分的候选（T03 的 ``RerankScorable``；EvidenceView 结构上满足）。
ScoredCandidate = RerankScorable


# --------------------------------------------------------------------------------------
# 候选窗口
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateWindow:
    """候选窗口截取结果（PRD §5.2）：确定性顺序 + 截取前后的候选集合。

    ``ordered`` 是提交评分的候选，顺序即档案固定评分口径；``outside`` 是被截掉的候选 ID
    （仍会记录为 ``outside_candidate_window``）；``candidate_ids_before/after`` 供评测报告
    「截取前后的正确证据覆盖」。
    """

    ordered: tuple[RerankScorable, ...]
    outside: tuple[RerankScorable, ...]
    candidate_ids_before: tuple[str, ...]
    candidate_ids_after: tuple[str, ...]
    limit: int
    exclude_outside: bool = True
    #: §6.3 缩容诊断：因候选总 token 预算超限而缩容后的候选数（未触发时 ``None``）。
    shrunk_to: int | None = None
    #: §6.3 缩容诊断：被缩容掉的候选（调用方应按 ``budget_exceeded`` 记录，不得静默丢弃）。
    shrunk_out: tuple[RerankScorable, ...] = ()
    #: §6.3：缩容到 ``min_candidate_count`` 后**仍然**超出档案声明的 token 上限时为 True；
    #: 调用方必须失败（``RERANK_INPUT_TOO_LONG``），不得把超限请求发出去。
    token_budget_exceeded: bool = False

    @property
    def truncation(self) -> int:
        """被截掉的候选数（0 表示窗口未生效）。"""

        return len(self.outside)


def _rrf_score(candidate: Any) -> float | None:
    """取候选的 RRF 分数（诊断信号，只用于确定性排序与同分比较）。"""

    signals = getattr(candidate, "route_signals", None)
    score = getattr(signals, "rrf_score", None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score)


def _input_seq(candidate: Any) -> int:
    """输入顺序（构造顺序）：仅作为最后一级确定性兜底，避免引入未定义顺序。"""

    seq = getattr(candidate, "window_seq", None)
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else 0


def _window_key(candidate: Any, seq: int) -> tuple[float, str, int]:
    score = _rrf_score(candidate)
    return (-(score if score is not None else float("-inf")), str(candidate.candidate_id), seq)


def window_candidates(
    candidates: Iterable[RerankScorable],
    profile: RerankProfile,
    *,
    exclude_outside: bool = True,
    token_counter: Callable[[str], int] | None = None,
) -> CandidateWindow:
    """按档案上限做确定性候选窗口截取（PRD §5.2），必要时按 §6.3 缩容。

    * 顺序：RRF 降序，同分按稳定 ``candidate_id`` 升序；缺少 RRF 分数的候选排在最后
      （仍按 ``candidate_id`` 升序），绝不因缺分被删。
    * 上限：``profile.max_candidates``（校准取值 30/50/100）；统计包含补充视图，因为
      补充视图以独立候选形式提交。
    * ``exclude_outside=False`` 时只做排序统计而不截取（评测报告截取前覆盖用）。
    * §6.3 缩容：``profile.max_request_tokens`` 已实测且提供 ``token_counter`` 时，若候选
      总 token 数超过该上限，按同一确定性顺序缩容到 ``min(min_candidate_count, 候选数)``；
      缩容后仍超限则置 ``token_budget_exceeded``（调用方必须失败，不得把超限请求发出去）。
      **本批该上限实测为 ``None``**，因此缩容路径在生产上不可达——绝不编造上限、绝不用
      字符数代替 token 数。
    """

    if not isinstance(profile, RerankProfile):
        raise RerankValidationError("profile 必须是 RerankProfile")
    ordered = sorted(
        ((_window_key(candidate, seq), candidate) for seq, candidate in enumerate(candidates)),
        key=lambda item: item[0],
    )
    ranked = tuple(candidate for _key, candidate in ordered)
    limit = profile.max_candidates
    if not exclude_outside or len(ranked) <= limit:
        return _window_with_shrink(
            profile,
            ranked,
            before=ranked,
            outside=(),
            limit=limit,
            exclude_outside=exclude_outside,
            token_counter=token_counter,
        )
    return _window_with_shrink(
        profile,
        ranked[:limit],
        before=ranked,
        outside=ranked[limit:],
        limit=limit,
        exclude_outside=exclude_outside,
        token_counter=token_counter,
    )


def _window_with_shrink(
    profile: RerankProfile,
    ranked: tuple[RerankScorable, ...],
    *,
    before: tuple[RerankScorable, ...],
    outside: tuple[RerankScorable, ...],
    limit: int,
    exclude_outside: bool,
    token_counter: Callable[[str], int] | None,
) -> CandidateWindow:
    """在已排序（RRF 降序 + ``candidate_id`` 升序）候选上应用 §6.3 缩容。

    ``before`` 是**截取前**的全部候选（``candidate_ids_before`` 报告截取前覆盖用），
    ``ranked`` 才是进入缩容判断的候选。
    """

    shrunk, shrunk_out, still_over = _shrink_for_token_budget(ranked, profile, token_counter)
    return CandidateWindow(
        ordered=shrunk,
        outside=outside,
        candidate_ids_before=tuple(candidate.candidate_id for candidate in before),
        candidate_ids_after=tuple(candidate.candidate_id for candidate in shrunk),
        limit=limit,
        exclude_outside=exclude_outside,
        shrunk_to=None if not shrunk_out else len(shrunk),
        shrunk_out=shrunk_out,
        token_budget_exceeded=still_over,
    )


def _request_token_limit(profile: RerankProfile) -> int | None:
    """档案声明的**候选总 token** 上限（未实测时为 ``None``，绝不用别家数值填充）。"""

    value = getattr(profile, "max_request_tokens", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _shrink_for_token_budget(
    ranked: tuple[RerankScorable, ...],
    profile: RerankProfile,
    token_counter: Callable[[str], int] | None,
) -> tuple[tuple[RerankScorable, ...], tuple[RerankScorable, ...], bool]:
    """§6.3 确定性缩容：``(缩容后的候选, 被缩掉的候选, 是否仍超限)``。

    未声明 token 上限或没有 tokenizer 计数函数时原样返回：宁可交给 Provider 的输入校验失败，
    也不编造上限或用字符数近似 token 数。
    """

    limit = _request_token_limit(profile)
    if limit is None or token_counter is None or not ranked:
        return ranked, (), False
    if _sum_tokens(ranked, token_counter) <= limit:
        return ranked, (), False
    target = min(int(profile.min_candidate_count), len(ranked))
    shrunk = ranked[:target]
    return shrunk, ranked[target:], _sum_tokens(shrunk, token_counter) > limit


def _sum_tokens(
    candidates: Sequence[RerankScorable], token_counter: Callable[[str], int]
) -> int:
    return sum(max(0, int(token_counter(_selected_text(candidate)))) for candidate in candidates)


# --------------------------------------------------------------------------------------
# 批次评分
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringOutcome:
    """一次批次评分的结果：候选、分数与响应元数据一一对应。

    分数与候选通过 ``candidate_id`` 绑定（提交顺序即 ``index``，已由
    ``RerankResult.assert_matches_request`` 逐位校验）。

    ``usage`` 是**服务实际返回的计量**，原样带回：``None`` 表示服务没有返回计量
    （显示「服务未返回」），绝不填 0，也不做换算（PRD §6.2/§9）。
    """

    candidates: tuple[RerankScorable, ...]
    scores: tuple[RerankScore, ...]
    profile_id: str
    provider_type: str
    model: str
    score_scope: str
    request_id: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    #: Provider 返回的计量原样透传（``RerankResult.usage``）；未返回时保持 ``None``。
    usage: RerankUsage | None = None

    @property
    def score_by_id(self) -> dict[str, float]:
        return {score.candidate_id: score.score for score in self.scores}

    def to_dict(self) -> dict[str, Any]:
        """脱敏投影：只含身份、分数、计量，不含任何正文/问题。"""

        return {
            "profile_id": self.profile_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "score_scope": self.score_scope,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "scores": [score.to_dict() for score in self.scores],
        }


async def score_candidates(
    *,
    query: str,
    candidates: Sequence[RerankScorable],
    profile: RerankProfile,
    provider: RerankProvider,
    scope: Any,
    index_revision: str | None = None,
    trace_id: str | None = None,
) -> ScoringOutcome:
    """把同一批候选一次性提交给 Provider 并校验响应完整性（PRD §5.2/§6.2）。

    不吞异常、不重试、不做部分排序：任何错误都以类型明确的 ``RerankError`` 抛出，
    由调用方决定用户可见状态。
    """

    if not candidates:
        raise RerankValidationError("候选列表为空：没有可评分的证据视图")
    revision = index_revision if index_revision is not None else _scope_index_revision(scope)
    request = RerankRequest(
        query=query,
        scope_id=_scope_id(scope),
        index_revision=revision,
        documents=tuple(candidates),
        profile_id=profile.profile_id,
        instruction_id=profile.instruction_id,
        instruction_text=profile.instruction_text,
        top_n=len(candidates),
    )
    result = await provider.rerank(request)
    result.assert_matches_request(request)
    result.assert_scores_in_range(provider.capabilities)
    return ScoringOutcome(
        candidates=tuple(candidates),
        scores=tuple(result.scores),
        profile_id=result.profile_id,
        provider_type=result.provider_type,
        model=result.model,
        score_scope=str(result.score_scope),
        request_id=result.request_id,
        trace_id=result.trace_id or trace_id,
        latency_ms=result.latency_ms,
        # 服务实际返回的计量原样透传：未返回时保持 None（显示「服务未返回」），
        # 绝不填 0、不估算——否则费用报告会把「丢失计量」误报成「服务无计量」。
        usage=result.usage,
    )


def _scope_id(scope: Any) -> str:
    value = getattr(scope, "scope_id", scope)
    if not isinstance(value, str) or not value.strip():
        raise RerankValidationError("scope 必须提供非空 scope_id")
    return value


def _scope_index_revision(scope: Any) -> str:
    value = getattr(scope, "index_revision", None)
    if not isinstance(value, str) or not value.strip():
        raise RerankValidationError("scope 必须提供非空 index_revision")
    return value


# --------------------------------------------------------------------------------------
# 最终选择
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ranked:
    """进入最终选择的候选：候选对象、重排分数与排序键。"""

    candidate: RerankScorable
    score: float
    rrf: float | None
    seq: int

    @property
    def candidate_id(self) -> str:
        return str(self.candidate.candidate_id)


def _selected_text(document: Any) -> str:
    evidence = getattr(document, "evidence_text", None)
    if isinstance(evidence, str) and evidence:
        return evidence
    ranking = getattr(document, "ranking_text", "")
    return ranking if isinstance(ranking, str) else ""


def _member_slices(document: Any) -> tuple[tuple[str, int | None, int | None], ...]:
    """成员切片身份（block_id, start, end），用于重复与组合覆盖判断。"""

    slices = getattr(document, "slices", ())
    result: list[tuple[str, int | None, int | None]] = []
    for item in slices:
        block_id = getattr(item, "block_id", None)
        if not isinstance(block_id, str):
            continue
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        result.append(
            (
                block_id,
                start if isinstance(start, int) and not isinstance(start, bool) else None,
                end if isinstance(end, int) and not isinstance(end, bool) else None,
            )
        )
    return tuple(result)


def _same_scope_of_origin(first: RerankScorable, later: RerankScorable) -> bool:
    """两个视图是否覆盖同一段原文：成员切片集合相同，或同一主 block 的整块视图。"""

    if _member_slices(first) and _member_slices(first) == _member_slices(later):
        return True
    first_block = getattr(first, "primary_block_id", None)
    return isinstance(first_block, str) and first_block == getattr(later, "primary_block_id", None)


def _is_exact_duplicate(first: RerankScorable, later: RerankScorable) -> bool:
    """两个视图是否是同一份逻辑内容。

    同一段原文可能被不同 ``candidate_id`` 承载（历史导入副本、同一 block 的整块视图与
    分段视图的子集）。正文逐字相同且成员指向同一段原文时保留排序靠前、分数更高的那个。
    """

    first_text = _selected_text(first)
    later_text = _selected_text(later)
    if first_text and later_text:
        return first_text == later_text and _same_scope_of_origin(first, later)
    return _member_slices(first) == _member_slices(later) != ()


def _is_covered_by(earlier: RerankScorable, later: RerankScorable) -> bool:
    """``later`` 是否被 ``earlier`` 更完整地覆盖（同一 block 的更大切片，或整块视图）。"""

    if earlier is later:
        return False
    earlier_slices = _member_slices(earlier)
    later_slices = _member_slices(later)
    if not earlier_slices or not later_slices:
        return False
    for block_id, start, end in later_slices:
        if start is None or end is None:
            return False
        if not any(
            other_block == block_id
            and other_start is not None
            and other_end is not None
            and other_start <= start
            and end <= other_end
            for other_block, other_start, other_end in earlier_slices
        ):
            return False
    # 完全相同的切片集合视为重复而不是覆盖
    return set(earlier_slices) != set(later_slices)


def _artifacts_covered_by(
    own: frozenset[tuple[str, int | None, int | None]],
    coverage: set[tuple[str, int | None, int | None]],
) -> bool:
    """原文范围集合是否被覆盖（集合包含或逐范围包含）。"""

    if own <= coverage:
        return True
    return all(
        any(
            other[0] == item[0]
            and item[1] is not None
            and item[2] is not None
            and other[1] is not None
            and other[2] is not None
            and other[1] <= item[1]
            and item[2] <= other[2]
            for other in coverage
        )
        for item in own
    )


def _single_view_contains(
    own: frozenset[tuple[str, int | None, int | None]],
    other: frozenset[tuple[str, int | None, int | None]],
) -> bool:
    """单个其他版本是否完整包含本候选的全部原文范围（集合包含或逐范围包含）。"""

    if own <= other:
        return True
    return all(
        any(
            item[0] == candidate[0]
            and item[1] is not None
            and item[2] is not None
            and candidate[1] is not None
            and candidate[2] is not None
            and candidate[1] <= item[1]
            and item[2] <= candidate[2]
            for candidate in other
        )
        for item in own
    )


def _best_per_artifact_set(
    ranked: Sequence[_Ranked],
) -> tuple[tuple[_Ranked, ...], dict[str, _Ranked]]:
    """同一原文范围集合内的取舍：先按完整度（整块 > 组合 > 片段），同完整度按分数排序。

    返回 ``(保留的条目, 落选条目 → 代表)``。落选者是同一段原文的另一种呈现
    （整块视图 vs 拼接组合），由调用方记 ``duplicate``。
    """

    groups: dict[frozenset[tuple[str, int | None, int | None]], list[_Ranked]] = {}
    for item in ranked:
        groups.setdefault(frozenset(_member_slices(item.candidate)), []).append(item)

    picked: list[_Ranked] = []
    excluded_by: dict[str, _Ranked] = {}
    for items in groups.values():
        if len(items) == 1:
            picked.extend(items)
            continue
        best = min(items, key=_representative_key)
        picked.extend([best])
        for item in items:
            if item is not best:
                excluded_by[item.candidate_id] = best
    ordered_picked = tuple(item for item in ranked if item in picked)
    return ordered_picked, excluded_by


def _properly_contains(
    other: frozenset[tuple[str, int | None, int | None]],
    own: frozenset[tuple[str, int | None, int | None]],
) -> bool:
    """``other`` 是否严格包含 ``own``（逐范围包含也算）。"""

    if own == other:
        return False
    return _single_view_contains(own, other)


def _canonical_artifacts(
    resolved: frozenset[tuple[str, int | None, int | None]],
    sets: Sequence[frozenset[tuple[str, int | None, int | None]]],
    memo: dict[
        frozenset[tuple[str, int | None, int | None]],
        frozenset[tuple[str, int | None, int | None]],
    ],
    visiting: frozenset = frozenset(),
) -> frozenset[tuple[str, int | None, int | None]]:
    """把一个原文范围集合归并到它的「整体版本」类。

    若某个候选的范围**严格包含**本集合，则本集合只是它的一部分（组合 = 公式 + 解释，
    成员各自是组合的子集）→ 归并到那个更完整的集合上。
    """

    if resolved in memo:
        return memo[resolved]
    if resolved in visiting:  # 防御：包含关系成环时保持原集合
        return resolved
    for other in sets:
        if other != resolved and _properly_contains(other, resolved):
            follow = _canonical_artifacts(other, sets, memo, visiting | {resolved})
            memo[resolved] = follow
            return follow
    memo[resolved] = resolved
    return resolved


def _representative_key(item: _Ranked) -> tuple[float, int, int]:
    """代表优先：分数更高 → 版本更完整 → 排序更靠前。

    同一「整体版本」类（例如「公式 + 解释」）里的候选提供的是同一份完整信息，因此先取分
    数最高的那个版本；分数相同才比较呈现方式的完整度（整块 > 组合 > 片段）。
    """

    return (-item.score, _artifact_kind_rank(item.candidate), item.seq)


def _survivors(
    ranked: Sequence[_Ranked],
    artifacts: dict[str, frozenset[tuple[str, int | None, int | None]]],
) -> tuple[tuple[_Ranked, ...], dict[str, _Ranked]]:
    """按「整体版本」类选代表，并给出「候选 → 代表」映射。

    先把「是某个更完整版本的严格子集」的集合合并到那个更完整的类上（组合 = 公式 + 解释），
    再在每类里按 :func:`_representative_key` 选出一个代表：分数最高者优先，分数相同再按
    呈现方式完整度（整块 > 组合 > 片段）。只要某类里有分数合格的候选，代表必然存在——
    既不会出现组合与成员互相排除后全空，也保证每个原文范围只出现一次（PRD §5.4/§5.6）。

    非代表版本是否记 ``covered_by_bundle``，取决于代表**是否实际入选**（预算阶段判定）。
    """

    sets = list({artifacts[item.candidate_id] for item in ranked})
    memo: dict[
        frozenset[tuple[str, int | None, int | None]],
        frozenset[tuple[str, int | None, int | None]],
    ] = {}
    canonical = {
        item.candidate_id: _canonical_artifacts(artifacts[item.candidate_id], sets, memo)
        for item in ranked
    }
    classes: list[frozenset[tuple[str, int | None, int | None]]] = []
    for item in ranked:
        resolved = canonical[item.candidate_id]
        if resolved and resolved not in classes:
            classes.append(resolved)

    representative_of: dict[str, _Ranked] = {}
    for resolved in classes:
        covering = [
            item
            for item in ranked
            if _single_view_contains(resolved, artifacts[item.candidate_id])
        ]
        if not covering:
            continue
        best = min(covering, key=_representative_key)
        for item in ranked:
            if canonical[item.candidate_id] == resolved:
                representative_of[item.candidate_id] = best
    kept = tuple(item for item in ranked if representative_of.get(item.candidate_id) is item)
    return kept, representative_of


def _artifact_kind_rank(document: Any) -> int:
    """版本完整度：多 block 版本 > 单 block 整块 > 单 block 片段 > 单 block 组合。

    仅在同一原文范围（或包含关系）之间比较时使用；数值越小越完整。组合排最后，
    是因为它是「片段打包成整体」的呈现方式：当一个整块/片段视图覆盖了完全相同的
    原文范围时，优先保留原始块视图，避免给同一段原文制造两个来源编号。
    """

    slices = _member_slices(document)
    if len({item[0] for item in slices}) > 1:
        return 0 if getattr(document, "view_kind", None) == ViewKind.SUPPORT_BUNDLE else 1
    if getattr(document, "view_kind", None) == ViewKind.SUPPORT_BUNDLE:
        return 3
    return 1 if any(item[1] == 0 for item in slices) else 2


def _artifact_sets(
    ranked: Sequence[_Ranked],
) -> tuple[
    dict[str, frozenset[tuple[str, int | None, int | None]]],
    dict[frozenset[tuple[str, int | None, int | None]], _Ranked],
]:
    """每个合格候选的原文范围集合，以及每个集合的代表版本。

    代表版本按「版本完整度优先、同完整度按分数排序」选取：同一段原文如果既有整块视图
    又有把它打包成组合的版本，优先保留原始块视图，避免给同一段原文制造两个来源编号。
    """

    artifacts: dict[str, frozenset[tuple[str, int | None, int | None]]] = {}
    representatives: dict[frozenset[tuple[str, int | None, int | None]], _Ranked] = {}
    for item in ranked:
        resolved = frozenset(_member_slices(item.candidate))
        artifacts[item.candidate_id] = resolved
        current = representatives.get(resolved)
        if current is None or _represents_better(item, current):
            representatives[resolved] = item
    return artifacts, representatives


def _represents_better(candidate: _Ranked, current: _Ranked) -> bool:
    """同范围版本的代表优先级：完整度更高的版本优先（其次保持原排序）。"""

    return _artifact_kind_rank(candidate.candidate) < _artifact_kind_rank(current.candidate)


def _is_duplicate_or_covered(
    item: _Ranked,
    artifacts: dict[str, frozenset[tuple[str, int | None, int | None]]],
    representatives: dict[frozenset[tuple[str, int | None, int | None]], _Ranked],
) -> str | None:
    """诊断用：该候选是否与其他版本共享原文范围。

    ``covered_by_bundle`` 的正式记录发生在预算阶段（只有代表实际入选才生效），
    本函数只用于诊断查询。
    """

    own = artifacts[item.candidate_id]
    if not own:
        return None
    if representatives.get(own) is not None and representatives[own].candidate_id != item.candidate_id:
        return str(ExclusionReason.DUPLICATE)
    for other in representatives.values():
        if other.candidate_id != item.candidate_id and _single_view_contains(
            own, artifacts[other.candidate_id]
        ):
            return str(ExclusionReason.COVERED_BY_BUNDLE)
    return None


def _exclusion(
    candidate: RerankScorable,
    reason: ExclusionReason,
    *,
    stage: str,
    detail: str | None = None,
) -> ExclusionRecord:
    return ExclusionRecord(
        candidate_id=str(candidate.candidate_id),
        reason=reason,
        stage=stage,
        detail=detail,
    )


def _version_matches(candidate: RerankScorable, revision: str) -> bool:
    value = getattr(candidate, "index_revision", None)
    return not isinstance(value, str) or not value.strip() or value == revision


def _scope_matches(candidate: RerankScorable, scope_id: str) -> bool:
    value = getattr(candidate, "scope_id", None)
    return not isinstance(value, str) or not value.strip() or value == scope_id


def _profile_id(profile: Any) -> str:
    """档案标识；档案本身非法时留空，``SelectionResult`` 仍可安全构造。"""

    value = getattr(profile, "profile_id", "")
    return value if isinstance(value, str) else ""


def _failure(
    profile: RerankProfile,
    error: RerankError,
    *,
    status: SelectionStatus,
    exclusions: Sequence[ExclusionRecord] = (),
    candidate_count: int = 0,
) -> SelectionResult:
    """失败/取消结果。

    ``cancelled`` 不是错误状态（PRD §5.6/§8.2），因此只在 ``failed`` 时携带 ``error``。
    """

    return SelectionResult(
        status=status,
        profile_id=_profile_id(profile),
        selected=(),
        exclusions=tuple(exclusions),
        candidate_count=candidate_count,
        error=error if status is SelectionStatus.FAILED else None,
    )


@dataclass(frozen=True)
class ScoredSelection:
    """``select_relevance_with_scores`` 的返回值：选择决定 + 本次评分记录。

    ``outcome`` 为 ``None`` 表示本轮没有完成评分（空候选、失败、取消），因此没有计量；
    计量只来自服务实际返回，缺失时保持 ``None``（PRD §6.2：显示「服务未返回」，不估造）。
    """

    selection: SelectionResult
    outcome: ScoringOutcome | None = None

    @property
    def usage(self) -> RerankUsage | None:
        """服务实际返回的计量；没有评分记录时为 ``None``。"""

        return self.outcome.usage if self.outcome is not None else None

    @property
    def scores(self) -> tuple[RerankScore, ...]:
        return self.outcome.scores if self.outcome is not None else ()


async def select_relevance(
    *,
    query: str,
    candidates: Sequence[RerankScorable],
    profile: RerankProfile,
    provider: RerankProvider,
    scope: Any,
    index_revision: str | None = None,
    trace_id: str | None = None,
    exclude_outside_window: bool = True,
    is_cancelled: Callable[[], bool] | None = None,
) -> SelectionResult:
    """完整筛选流程，返回四类状态之一（PRD §5.1/§5.6）。

    ``candidates`` 是候选阶段产出的全部逻辑候选（主视图与补充视图同为候选）；
    ``is_cancelled`` 由调用方注入取消判定，在提交外呼前后各检查一次。
    契约违例（非法档案、缺少 scope 标识等）以 ``failed`` 返回，不向外抛异常。
    """

    selection, _outcome = await _run_relevance_pipeline(
        query=query,
        candidates=candidates,
        profile=profile,
        provider=provider,
        scope=scope,
        index_revision=index_revision,
        trace_id=trace_id,
        exclude_outside_window=exclude_outside_window,
        is_cancelled=is_cancelled,
    )
    return selection


async def select_relevance_with_scores(
    *,
    query: str,
    candidates: Sequence[RerankScorable],
    profile: RerankProfile,
    provider: RerankProvider,
    scope: Any,
    index_revision: str | None = None,
    trace_id: str | None = None,
    exclude_outside_window: bool = True,
    is_cancelled: Callable[[], bool] | None = None,
) -> ScoredSelection:
    """给「一步调用、同时需要选择决定与计量」的调用方的薄包装。

    用途明确：评测/脚本（如 T29 线上同流程评测、A09 的完整 usage 报告、费用报告）既要
    ``SelectionResult`` 又要服务返回的 ``usage``。它**不复制任何筛选逻辑**，只是把
    ``select_relevance`` 内部同一条链路（候选窗口 → 输入校验 → ``score_candidates`` →
    ``_qualify`` → ``select_from_scores``）的中间产物一并交回，因此与线上走的是同一份实现。

    失败/取消/空结果时 ``outcome`` 为 ``None``（没有评分记录，也就没有计量），
    调用方应据 ``selection.status`` 判断，不得把 ``None`` 当作 0。
    """

    selection, outcome = await _run_relevance_pipeline(
        query=query,
        candidates=candidates,
        profile=profile,
        provider=provider,
        scope=scope,
        index_revision=index_revision,
        trace_id=trace_id,
        exclude_outside_window=exclude_outside_window,
        is_cancelled=is_cancelled,
    )
    return ScoredSelection(outcome=outcome, selection=selection)


async def _run_relevance_pipeline(
    *,
    query: str,
    candidates: Sequence[RerankScorable],
    profile: RerankProfile,
    provider: RerankProvider,
    scope: Any,
    index_revision: str | None,
    trace_id: str | None,
    exclude_outside_window: bool,
    is_cancelled: Callable[[], bool] | None,
) -> tuple[SelectionResult, ScoringOutcome | None]:
    """唯一实现：候选窗口 → 输入校验 → 批次评分 → 阈值筛选 → 最终选择。

    ``select_relevance`` 与 ``select_relevance_with_scores`` 都只是本函数的薄适配，
    保证「线上 / 评测 / 来源装配」三条入口共享同一份筛选语义（PRD §10.3）。
    第二个返回值仅在真的完成评分时有值。
    """

    if is_cancelled is not None and is_cancelled():
        return (
            _failure(
                profile,
                RerankValidationError("筛选已被取消"),
                status=SelectionStatus.CANCELLED,
            ),
            None,
        )

    try:
        window = window_candidates(candidates, profile, exclude_outside=exclude_outside_window)
        scope_id = _scope_id(scope)
        revision = index_revision if index_revision is not None else _scope_index_revision(scope)
    except asyncio.CancelledError:
        raise
    except RerankError as error:
        return _failure(profile, error, status=SelectionStatus.FAILED), None

    exclusions: list[ExclusionRecord] = [
        _exclusion(
            candidate,
            ExclusionReason.OUTSIDE_CANDIDATE_WINDOW,
            stage="candidate_window",
            detail=f"limit={window.limit}",
        )
        for candidate in window.outside
    ]
    candidate_count = len(window.ordered)
    if not window.ordered:
        return (
            SelectionResult(
                status=SelectionStatus.EMPTY,
                profile_id=_profile_id(profile),
                exclusions=tuple(exclusions),
                candidate_count=0,
            ),
            None,
        )

    # 越界 scope / 版本不一致的候选在提交前拒绝，并保留可诊断的记录（PRD §9）。
    rejected = [
        candidate
        for candidate in window.ordered
        if not _scope_matches(candidate, scope_id) or not _version_matches(candidate, revision)
    ]
    if rejected:
        exclusions.extend(
            _exclusion(
                candidate,
                ExclusionReason.SCOPE_MISMATCH
                if not _scope_matches(candidate, scope_id)
                else ExclusionReason.VERSION_MISMATCH,
                stage="input_validation",
            )
            for candidate in rejected
        )
        return (
            _failure(
                profile,
                EvidenceVersionChangedError("候选集合包含越界 scope 或版本不一致的视图"),
                status=SelectionStatus.FAILED,
                exclusions=exclusions,
                candidate_count=candidate_count,
            ),
            None,
        )
    submittable = tuple(window.ordered)

    try:
        outcome = await score_candidates(
            query=query,
            candidates=submittable,
            profile=profile,
            provider=provider,
            scope=scope,
            index_revision=revision,
            trace_id=trace_id,
        )
        ordered = _qualify(submittable, outcome.score_by_id, profile)
    except asyncio.CancelledError:
        raise
    except RerankError as error:
        # 服务失败绝不伪装成「无相关资料」：返回 failed 并携带类型明确的错误。
        return (
            _failure(
                profile,
                error,
                status=SelectionStatus.FAILED,
                exclusions=exclusions,
                candidate_count=candidate_count,
            ),
            None,
        )

    if is_cancelled is not None and is_cancelled():
        return (
            _failure(
                profile,
                RerankValidationError("筛选已被取消"),
                status=SelectionStatus.CANCELLED,
                exclusions=exclusions,
                candidate_count=candidate_count,
            ),
            None,
        )

    selection = select_from_scores(
        candidates=submittable,
        scores=outcome.score_by_id,
        profile=profile,
        ordered=ordered,
        exclusions=exclusions,
        candidate_count=candidate_count,
    )
    return selection, outcome


def select_from_scores(
    *,
    candidates: Sequence[RerankScorable],
    scores: Mapping[str, float],
    profile: RerankProfile,
    ordered: Sequence[_Ranked] | None = None,
    exclusions: Sequence[ExclusionRecord] = (),
    candidate_count: int | None = None,
) -> SelectionResult:
    """已完成评分的候选进入阈值筛选、去重、配额与预算（PRD §5.6 的后半段）。

    供已经拿到评分记录的调用方（例如同一批分数既要筛选又要写来源字段的 ChatService）
    复用，避免为了取分数把同一批候选重复提交给 Provider。``ordered`` 省略时按
    ``scores`` 现算。

    非法输入的口径（PRD §6.2，三入口一致）：

    * 分数是 NaN / ±Inf / 超出协议范围，或 ``scores`` 出现非候选键 → **立即失败**
      （``RerankInvalidResponseError`` / ``RerankValidationError``），绝不入选；
    * 候选在 ``scores`` 里**没有分数** → 不入选，并记 ``below_threshold``（固定 7 类原因里
      没有「缺分」这一类；关键是必须被记录，不变量：全部候选 == selected ∪ exclusions）。

    前置条件（调用方责任）：候选必须是**已经过** :func:`window_candidates` 的窗口结果，
    ``outside`` 候选的 ``outside_candidate_window`` 记录由调用方通过 ``exclusions`` 传入——
    本函数不重复做窗口截取，否则截取口径会与线上分叉。
    """

    submitted = tuple(candidates)
    _assert_score_input(submitted, scores, profile)
    ranked = tuple(ordered) if ordered is not None else _qualify_optional(submitted, scores, profile)
    total = len(submitted) if candidate_count is None else candidate_count
    collected = list(exclusions)
    collected.extend(_threshold_exclusions(submitted, scores, profile))

    survivors, dedupe_exclusions, excluded_by = _dedupe(ranked)
    collected.extend(dedupe_exclusions)

    selected, budget_exclusions = _fit_budget(survivors, profile, excluded_by, ranked=ranked)
    collected.extend(budget_exclusions)

    if not selected:
        return SelectionResult(
            status=SelectionStatus.EMPTY,
            profile_id=_profile_id(profile),
            exclusions=tuple(collected),
            candidate_count=total,
        )
    return SelectionResult(
        status=SelectionStatus.SELECTED,
        profile_id=_profile_id(profile),
        selected=selected,
        exclusions=tuple(collected),
        candidate_count=total,
    )


async def relevance_profile(
    *,
    query: str,
    candidates: Sequence[RerankScorable],
    profile: RerankProfile,
    provider: RerankProvider,
    scope: Any,
    index_revision: str | None = None,
    trace_id: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> SelectionResult:
    """``select_relevance`` 的别名：调用方（ChatService / V1.2）惯用名。"""

    return await select_relevance(
        query=query,
        candidates=candidates,
        profile=profile,
        provider=provider,
        scope=scope,
        index_revision=index_revision,
        trace_id=trace_id,
        is_cancelled=is_cancelled,
    )


def _score_range(profile: Any) -> tuple[float, float]:
    """档案声明的相对分范围；档案未声明时用评分协议默认区间（PRD §6.2）。"""

    low = getattr(profile, "score_min", None)
    high = getattr(profile, "score_max", None)
    if (
        isinstance(low, (int, float))
        and not isinstance(low, bool)
        and isinstance(high, (int, float))
        and not isinstance(high, bool)
        and float(high) > float(low)
    ):
        return float(low), float(high)
    return _DEFAULT_SCORE_RANGE


def _checked_score(candidate: RerankScorable, score: object, profile: RerankProfile) -> float:
    """校验单个重排分数（PRD §6.2）：必须有限且在协议范围内，否则立即失败。"""

    candidate_id = str(candidate.candidate_id)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RerankInvalidResponseError(f"候选 {candidate_id} 的重排分数必须是数值")
    value = float(score)
    if not math.isfinite(value):
        raise RerankInvalidResponseError(f"候选 {candidate_id} 的重排分数必须是有限数值")
    low, high = _score_range(profile)
    if value < low or value > high:
        raise RerankInvalidResponseError(
            f"候选 {candidate_id} 的重排分数 {value} 超出协议范围 [{low}, {high}]"
        )
    return value


def _assert_score_input(
    submitted: Sequence[RerankScorable],
    scores: Mapping[str, float],
    profile: RerankProfile,
) -> None:
    """§6.2 输入校验：分数映射只覆盖本批候选，且每个分数有限并在协议范围内。

    非有限（NaN / ±Inf）或越界分数**立即失败**，绝不入选：``NaN < threshold`` 恒为假，不显式
    拒绝就会把「非数值」当成「不低于阈值」写进最终上下文。非候选键说明调用方把别批分数混了
    进来，同样直接拒绝，避免三个入口在非法输入下出现字段级分歧。
    """

    known = {str(candidate.candidate_id) for candidate in submitted}
    unknown = sorted(str(key) for key in scores if str(key) not in known)
    if unknown:
        raise RerankValidationError(
            f"评分映射包含非候选键（{unknown[:3]}）；同一批分数必须只覆盖本批候选"
        )
    for candidate in submitted:
        raw = scores.get(str(candidate.candidate_id))
        if raw is not None:
            _checked_score(candidate, raw, profile)


def _rank_key(item: _Ranked) -> tuple[float, float, str, int]:
    """最终排序键：重排分数降序 → RRF 分数降序 → ``candidate_id`` 升序。"""

    rrf = item.rrf if item.rrf is not None else float("-inf")
    return (-item.score, -rrf, item.candidate_id, item.seq)


def _qualify(
    submitted: Sequence[RerankScorable],
    scores: dict[str, float],
    profile: RerankProfile,
) -> tuple[_Ranked, ...]:
    """阈值筛选 + 确定性排序（PRD §5.6）：阈值完全来自档案，无兜底默认值。"""

    ranked: list[_Ranked] = []
    for seq, candidate in enumerate(submitted):
        raw = scores.get(str(candidate.candidate_id))
        if raw is None:
            raise RerankInvalidResponseError("候选缺少重排分数，不能参与最终排序")
        score = _checked_score(candidate, raw, profile)
        if score < profile.score_threshold:
            continue
        ranked.append(
            _Ranked(
                candidate=candidate,
                score=score,
                rrf=_rrf_score(candidate),
                seq=seq,
            )
        )
    return tuple(sorted(ranked, key=_rank_key))


def _threshold_exclusions(
    submitted: Sequence[RerankScorable],
    scores: Mapping[str, float],
    profile: RerankProfile,
) -> list[ExclusionRecord]:
    """阈值阶段未入选的候选：低于阈值，或在给定分数映射里没有分数。

    两者都记 ``below_threshold``——固定 7 类原因里没有「缺分」这一类，而缺分等价于「不能证明
    它达到阈值」；关键是**必须被记录**：全部候选 == selected ∪ exclusions，绝不静默丢弃。
    """

    records: list[ExclusionRecord] = []
    for candidate in submitted:
        raw = scores.get(str(candidate.candidate_id))
        if raw is None:
            records.append(
                _exclusion(
                    candidate,
                    ExclusionReason.BELOW_THRESHOLD,
                    stage="threshold",
                    detail="missing score",
                )
            )
        elif float(raw) < profile.score_threshold:
            records.append(
                _exclusion(
                    candidate,
                    ExclusionReason.BELOW_THRESHOLD,
                    stage="threshold",
                    detail=f"threshold={profile.score_threshold}",
                )
            )
    return records


def _qualify_optional(
    submitted: Sequence[RerankScorable],
    scores: Mapping[str, float],
    profile: RerankProfile,
) -> tuple[_Ranked, ...]:
    """按映射里的分数做阈值筛选与排序；缺分的候选直接视为未通过（不下发未评分内容）。

    缺分候选由 :func:`_threshold_exclusions` 记 ``below_threshold``——跳过不等于丢弃。
    """

    ranked: list[_Ranked] = []
    for seq, candidate in enumerate(submitted):
        raw = scores.get(str(candidate.candidate_id))
        if raw is None:
            continue
        score = _checked_score(candidate, raw, profile)
        if score < profile.score_threshold:
            continue
        ranked.append(
            _Ranked(
                candidate=candidate,
                score=score,
                rrf=_rrf_score(candidate),
                seq=seq,
            )
        )
    return tuple(sorted(ranked, key=_rank_key))


def _touches_bundle(first: RerankScorable, second: RerankScorable) -> bool:
    """任一视图是 ``support_bundle`` 时才谈「组合覆盖」。"""

    return ViewKind.SUPPORT_BUNDLE in (
        getattr(first, "view_kind", None),
        getattr(second, "view_kind", None),
    )


def _dedupe(
    ordered: Sequence[_Ranked],
) -> tuple[tuple[_Ranked, ...], list[ExclusionRecord], dict[_Ranked, _Ranked]]:
    """分组去重（PRD §5.4/§5.6）：每个「整体版本」类只留一个代表版本。

    * 同一类里版本完整度最高、同完整度排序最靠前的条目即代表（``_survivors`` 决定）；
    * 与代表**范围完全相同**的落选版本记 ``duplicate``（同一范围的另一种呈现）；
    * 共享部分范围（组合与成员）的落选版本留给预算阶段：代表**实际入选**时记
      ``covered_by_bundle``，代表自己没入选（超预算/超出上限）时该成员作为后备候选继续参与，
      不得仅因「它所属的组合也是候选」被排除。第三个返回值就是「候选 → 代表」映射。
    """

    artifacts, _representatives = _artifact_sets(ordered)
    kept, representative_of = _survivors(ordered, artifacts)
    kept_ids = {item.candidate_id for item in kept}

    exclusions: list[ExclusionRecord] = []
    excluded_by: dict[_Ranked, _Ranked] = {}
    for item in ordered:
        representative = representative_of.get(item.candidate_id)
        if representative is None or representative.candidate_id == item.candidate_id:
            continue
        if artifacts[item.candidate_id] == artifacts[representative.candidate_id]:
            exclusions.append(
                _exclusion(
                    item.candidate,
                    ExclusionReason.DUPLICATE,
                    stage="dedupe",
                    detail=f"same artifacts as {representative.candidate_id}",
                )
            )
        elif item.candidate_id not in kept_ids:
            excluded_by[item] = representative
    return tuple(kept), exclusions, excluded_by


def _fit_budget(
    survivors: Sequence[_Ranked],
    profile: RerankProfile,
    excluded_by: Mapping[_Ranked, _Ranked] | None = None,
    ranked: Sequence[_Ranked] | None = None,
) -> tuple[tuple[RerankScorable, ...], list[ExclusionRecord]]:
    """来源上限与正文预算（PRD §5.6）：超预算跳过并继续，绝不从中间截取。

    选用分两轮决定，入选顺序始终等于排序顺序（``score`` → RRF → ``candidate_id``）：

    1. **类代表**先按排序尝试选用：同一原文范围组里覆盖最完整的合格版本（组合合格即代表整
       组，与类内分数顺序无关）；达到来源上限或超出剩余预算的记 ``budget_exceeded``。
    2. **后备**：代表**没有实际入选**时，它覆盖的成员按排序依次尝试入选——代表未入选就不
       具备覆盖能力，绝不把成员一起误排除；若后备成员本身被已入选的兄弟版本覆盖，则记
       ``covered_by_bundle``。代表实际入选时，它的成员同样记 ``covered_by_bundle``
       （PRD §5.4）。

    两轮都占用同一份来源上限与字符预算，因此代表优先、后备只吃剩余额度，且任何已入选的候选
    都不会与另一个入选者共享同一段原文范围。
    """

    limit = min(profile.max_selected, MAX_SELECTED_SOURCES)
    budget = min(profile.evidence_char_budget, EVIDENCE_CHAR_BUDGET)
    representatives = tuple(survivors)
    representative_ids = {item.candidate_id for item in representatives}
    covered_by = excluded_by or {}
    sequence = tuple(ranked) if ranked is not None else representatives + tuple(covered_by)

    admitted: list[_Ranked] = []
    admitted_ids: set[str] = set()
    over_budget: dict[str, ExclusionRecord] = {}
    covered: dict[str, str] = {}
    used = 0

    def _admit(item: _Ranked) -> ExclusionRecord | None:
        """尝试选用；不能选用时返回应记录的 ``budget_exceeded``。"""

        nonlocal used
        if len(admitted) >= limit:
            return _exclusion(
                item.candidate,
                ExclusionReason.BUDGET_EXCEEDED,
                stage="quota",
                detail=f"max_selected={limit}",
            )
        length = len(_selected_text(item.candidate))
        if used + length > budget:
            return _exclusion(
                item.candidate,
                ExclusionReason.BUDGET_EXCEEDED,
                stage="budget",
                detail=f"budget={budget}",
            )
        admitted.append(item)
        admitted_ids.add(item.candidate_id)
        used += length
        return None

    # 第一轮：类代表（类级决定，与类内分数顺序无关）。
    for item in representatives:
        record = _admit(item)
        if record is not None:
            over_budget[item.candidate_id] = record

    # 第二轮：被代表覆盖的版本——代表入选则排除，代表未入选则它自己就是候选版本。
    fallbacks: list[_Ranked] = []
    for item in sequence:
        if item.candidate_id in representative_ids:
            continue
        representative = covered_by.get(item)
        if representative is None:
            continue  # 同一原文范围的另一种呈现，已由 ``_dedupe`` 记 duplicate
        if representative.candidate_id in admitted_ids:
            covered[item.candidate_id] = representative.candidate_id
            continue
        sibling = next(
            (other for other in fallbacks if _is_covered_by(other.candidate, item.candidate)),
            None,
        )
        if sibling is not None:
            covered[item.candidate_id] = sibling.candidate_id
            continue
        record = _admit(item)
        if record is not None:
            over_budget[item.candidate_id] = record
        else:
            fallbacks.append(item)

    selected = tuple(item.candidate for item in sorted(admitted, key=_rank_key))
    exclusions: list[ExclusionRecord] = [
        over_budget[item.candidate_id] for item in sequence if item.candidate_id in over_budget
    ]
    exclusions.extend(
        _exclusion(
            item.candidate,
            ExclusionReason.COVERED_BY_BUNDLE,
            stage="dedupe",
            detail=f"covered by {covered[item.candidate_id]}",
        )
        for item in sequence
        if item.candidate_id in covered
    )
    return selected, exclusions
