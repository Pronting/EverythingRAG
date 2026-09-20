"""把「资料相关性筛选」接进 V1.2.0 的 run 管线（PRD §5.1/§5.3/§8.4，集成接线层）。

背景（集成裁定）：相关性筛选（rerank）在 ``main`` 上原本内嵌在
``app.generation.chat_service.ChatService`` 里，进度走 V1.1.0 的 ``/api/chat`` SSE 帧
（``tool=rerank`` running/done/error）。V1.2.0 按 PRD 13.2 把 ``/api/chat`` 下线为
**410 Gone**，问答统一走 ``POST /api/workspaces/{w}/runs`` + v2 事件。本模块因此把
筛选**语义**从 ChatService 搬到检索层，由 run 管线的检索工具
（``app.agent.tools._search_knowledge``）调用。

**不改筛选核心**：评分 / 窗口 / 选择仍由 ``app.retrieval.reranker`` 与
``app.retrieval.relevance_selector`` 完成（逐字复用 ``main`` 的验收基线），本模块只做：

1. 把候选块投影成证据视图（标题按块元数据 ``heading_path`` 补齐，缺标题**绝不猜**）；
2. 显式评分一次并复用同一批分数做选择（避免为取分数重复提交 Provider）；
3. 把入选视图投影回 ``RetrievedChunk``，``context_text`` 一律写成视图 ``evidence_text``，
   使「生成上下文 == 来源面板 == 被评分文本」三条口径逐字一致（PRD §5.3/C06）；
4. 把四类状态收敛成 :class:`RerankOutcome`，供工具层转成 v2 事件与工具结果。

运行态来源：调用方（``app.api.run_services.tool_services_for``）注入
``deps.build_rerank_runtime`` 式的可调用对象，返回 ``chat_service.RerankRuntime``
（``provider`` / ``profile`` / ``config_error_code`` / ``config_error_message``）。
本模块只按**属性**使用它，不 import generation 层。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Final

from app.retrieval.evidence_view import (
    HeadingLookup,
    RetrievalScope,
    build_chunk_views,
    index_revision_for,
)
from app.retrieval.relevance_selector import score_candidates, select_from_scores
from app.retrieval.reranker import RerankError, RerankProfile, RerankProvider, SelectionStatus
from app.retrieval.trace import RelevanceTrace, build_relevance_trace, new_trace_id
from app.retrieval.vector_retriever import RetrievedChunk

__all__ = [
    "RELEVANCE_METADATA_KEYS",
    "RERANK_FAILED_MESSAGE",
    "RERANK_PROGRESS_TOOL_NAME",
    "RERANK_UNCONFIGURED_MESSAGE",
    "STATUS_DISABLED",
    "STATUS_EMPTY",
    "STATUS_FAILED",
    "STATUS_FILTERED",
    "RerankOutcome",
    "filter_chunks",
    "resolve_runtime",
    "runtime_configured",
]

#: 四类筛选状态（与 ``main`` 的 ``_RETRIEVAL_STATUS_*`` 逐字一致，便于对照回归）。
STATUS_DISABLED: Final[str] = "disabled"
STATUS_FILTERED: Final[str] = "filtered"
STATUS_EMPTY: Final[str] = "empty"
STATUS_FAILED: Final[str] = "failed"

#: 筛选进度在 v2 事件里使用的工具名（``tool.started`` / ``tool.completed`` 的
#: ``tool_name``）。与旧 chat 帧的 ``tool=rerank`` 同名，前端无需换标识。
RERANK_PROGRESS_TOOL_NAME: Final[str] = "rerank"

RERANK_FAILED_MESSAGE: Final[str] = "资料相关性筛选失败，本轮未生成答案"
RERANK_UNCONFIGURED_MESSAGE: Final[str] = "资料相关性筛选已启用但未正确配置，请在设置中检查"

#: 已筛选来源携带的元数据键（PRD §8.3/§9）。工具结果与来源字段都从这里取，
#: 保证「同一批分数既用于选择也用于展示」。
RELEVANCE_METADATA_KEYS: Final[tuple[str, ...]] = (
    "relevance_checked",
    "rerank_score",
    "score_type",
    "profile_id",
    "evidence_hash",
    "member_refs",
)


@dataclass(frozen=True)
class RerankOutcome:
    """一次检索的相关性筛选结果（工具层据此产出工具结果与进度事件）。"""

    status: str = STATUS_DISABLED
    chunks: tuple[RetrievedChunk, ...] = ()
    profile_id: str = ""
    candidate_count: int = 0
    selected_count: int = 0
    missing_headings: int = 0
    heading_diagnostics: Mapping[str, int] = field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""
    trace: RelevanceTrace | None = None

    @property
    def enabled(self) -> bool:
        """本轮是否真的执行了筛选（``disabled`` 之外都算已启用）。"""

        return self.status != STATUS_DISABLED

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    def to_dict(self) -> dict[str, Any]:
        """脱敏投影（工具结果 / 诊断）：只有状态、计数与引用，不含正文与问题。"""

        payload: dict[str, Any] = {
            "status": self.status,
            "profile_id": self.profile_id,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
        }
        if self.trace is not None:
            # 与 main 的 ``meta.retrieval.trace_id`` 同义：脱敏诊断引用（PRD §9）。
            payload["trace_id"] = getattr(self.trace, "trace_id", "")
        if self.missing_headings:
            payload["missing_headings"] = self.missing_headings
        if self.heading_diagnostics:
            payload["heading_diagnostics"] = dict(self.heading_diagnostics)
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.error_message:
            payload["error_message"] = self.error_message
        return payload


def _resolve_runtime(runtime: Any) -> Any:
    """把注入的运行态（对象或可调用）解析成一次快照；解析异常收敛为配置错误。

    刻意不 import ``app.generation.chat_service``：只按属性读取，检索层因此不反向依赖
    生成层（架构 §5 的依赖方向），同时兼容任何带同名属性的运行态对象。
    """

    candidate = runtime
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:  # noqa: BLE001 -- 运行态解析异常一律收敛为配置错误，不静默回落
            return _ConfigError("RERANK_NOT_CONFIGURED", RERANK_UNCONFIGURED_MESSAGE)
    if candidate is None:
        return None
    if not all(
        hasattr(candidate, attribute)
        for attribute in ("provider", "profile", "config_error_code")
    ):
        return _ConfigError("RERANK_NOT_CONFIGURED", RERANK_UNCONFIGURED_MESSAGE)
    return candidate


@dataclass(frozen=True)
class _ConfigError:
    """内部运行态：已启用但配置不可用（转 ``failed``，绝不静默回落旧路径）。"""

    config_error_code: str
    config_error_message: str
    provider: RerankProvider | None = None
    profile: RerankProfile | None = None

    @property
    def enabled(self) -> bool:
        return False

    @property
    def configured(self) -> bool:
        return True


def resolve_runtime(runtime: Any) -> Any | None:
    """解析本轮筛选运行态快照；``None`` = 本次装配未接筛选（等价于未启用）。

    工具层用它把「一次快照」同时用于进度判定与筛选执行，避免同一轮读两遍配置。
    """

    return _resolve_runtime(runtime)


def runtime_configured(runtime: Any) -> bool:
    """用户是否**启用**了本功能（无论配置是否可用）—— 决定是否上报筛选进度。

    与 ``main`` 的 ``RerankRuntime.configured`` 同义：已启用但配置不可用也算启用，因为
    此时必须给用户一个明确的失败而不是静默走旧路径。**半配置**（只给了 provider 或只给了
    profile）同样算启用 —— ``ChatService._resolve_rerank_runtime`` 就是这么收敛的
    （只给一半 → 配置错误），这里保持一致。
    """

    snapshot = _resolve_runtime(runtime)
    if snapshot is None:
        return False
    if getattr(snapshot, "config_error_code", None):
        return True
    if getattr(snapshot, "enabled", None) is True:
        return True
    return (
        getattr(snapshot, "provider", None) is not None
        or getattr(snapshot, "profile", None) is not None
    )


def _missing_headings_of(lookup: HeadingLookup | None, chunk_count: int) -> tuple[int, dict[str, int]]:
    """缺失标题的分项计数：缺键 / 键存在但为空 / 调用方漏传查表（诊断用）。"""

    if lookup is None:
        # 没有查表能力却需要标题：隐性口径分叉，必须可诊断。
        return 0, {"heading_lookup_absent": max(0, int(chunk_count))}
    missing = lookup.missing
    if not missing:
        return 0, {}
    stats = lookup.stats()
    return missing, {
        "heading_missing_key": stats.get("missing_key", 0),
        "heading_blank_value": stats.get("blank_value", 0),
        "heading_lookup_absent": stats.get("lookup_absent", 0),
    }


def _heading_lookup_for(vectorstore: Any) -> HeadingLookup | None:
    """本轮的标题查表；取不到返回 ``None``（视图标题留空，绝不猜标题）。"""

    if vectorstore is None:
        return None
    try:
        return HeadingLookup.for_vectorstore(vectorstore)
    except Exception:  # noqa: BLE001 -- 元数据不可读时降级为空标题，不阻断检索
        return None


def _resolve_scope(scope: Any, vectorstore: Any) -> RetrievalScope:
    """本轮资料范围：调用方注入了 ``RetrievalScope`` 就用它，否则按本区索引显式标记。"""

    resolved = scope() if callable(scope) else scope
    if isinstance(resolved, RetrievalScope):
        return resolved
    return RetrievalScope(
        scope_id="global",
        index_revision=(
            index_revision_for(vectorstore) if vectorstore is not None else "unknown-index"
        ),
    )


def _source_chunk(
    original: RetrievedChunk | None,
    view: Any,
    *,
    profile_id: str,
    rerank_score: float | None,
) -> RetrievedChunk:
    """把入选证据视图投影回 ``RetrievedChunk``（携带 relevance_* 元数据）。

    ``context_text`` 一律写为视图的 ``evidence_text``：生成上下文与来源面板都读
    ``context_text or text``，因此两处必然使用同一份已经接受检查的原文——旧的同标题补充
    文字不会因 hydration 重新进入本轮输入（PRD §5.3/C06）。向量相似度保持原值，重排分数
    只放进独立字段（不得改写 ``similarity`` 语义）。
    """

    metadata = dict(getattr(original, "metadata", None) or {})
    similarity = getattr(original, "similarity", None)
    metadata.update(
        {
            "relevance_checked": True,
            "rerank_score": rerank_score,
            "score_type": "rerank",
            "profile_id": profile_id,
            "evidence_hash": getattr(view, "payload_hash", None),
            "member_refs": [member.to_dict() for member in getattr(view, "member_refs", ())],
            "scope_id": getattr(view, "scope_id", ""),
            "index_revision": getattr(view, "index_revision", ""),
            "view_kind": str(getattr(view, "view_kind", "block")),
        }
    )
    # 图片原图路径不得随筛选结果外泄（与 ChatService 同口径）。
    metadata.pop("image_path", None)
    metadata.pop("image_content_hash", None)
    evidence_text = str(getattr(view, "evidence_text", ""))
    return RetrievedChunk(
        block_id=str(getattr(view, "primary_block_id", "") or getattr(view, "block_id", "")),
        text=evidence_text,
        similarity=similarity if isinstance(similarity, (int, float)) else None,
        source_file=(
            getattr(original, "source_file", "") or str(getattr(view, "source_file", "") or "")
        ),
        platform=getattr(original, "platform", "local") or "local",
        chunk_type=getattr(original, "chunk_type", "text") or "text",
        anchor=getattr(original, "anchor", None),
        # 标题优先取视图值：候选对象常常不带标题，而视图已按块元数据补齐（PRD §5.3）。
        heading_path=(
            str(getattr(view, "heading_path", "") or "") or getattr(original, "heading_path", None)
        ),
        metadata=metadata,
        match_type="rerank",
        context_text=evidence_text,
    )


def _selected_chunks(
    selection: Any,
    source_chunks: Mapping[str, RetrievedChunk],
    *,
    profile_id: str,
    scores: Mapping[str, float] | None = None,
) -> tuple[RetrievedChunk, ...]:
    """按最终顺序把入选视图投影成 ``RetrievedChunk``（分数取自本次评分记录）。"""

    resolved = scores or {}
    return tuple(
        _source_chunk(
            source_chunks.get(str(view.candidate_id)),
            view,
            profile_id=profile_id,
            rerank_score=resolved.get(str(view.candidate_id)),
        )
        for view in selection.selected
    )


def _trace_for(
    *,
    scope: Any,
    profile_id: str,
    views: Sequence[Any],
    selection: Any | None = None,
    scores: Mapping[str, float] | None = None,
    provider_trace_id: str | None = None,
    request_id: str | None = None,
    usage: Any = None,
    rerank_latency_ms: float | None = None,
    total_latency_ms: float | None = None,
    status: str | None = None,
    error_code: str | None = None,
    missing_headings: int = 0,
) -> RelevanceTrace:
    """本轮脱敏诊断 trace（PRD §9）：只有引用、枚举、计数与耗时。"""

    return build_relevance_trace(
        trace_id=new_trace_id(),
        scope=scope,
        profile_id=profile_id,
        candidates=list(views),
        selection=selection,
        scores=scores,
        status=status,
        provider_trace_id=provider_trace_id,
        request_id=request_id,
        usage=usage,
        rerank_latency_ms=rerank_latency_ms,
        total_latency_ms=total_latency_ms,
        error_code=error_code,
        missing_headings=missing_headings,
    )


def _failed(
    *,
    profile_id: str,
    error_code: str,
    error_message: str,
    candidate_count: int = 0,
    missing_headings: int = 0,
    heading_diagnostics: Mapping[str, int] | None = None,
    trace: RelevanceTrace | None = None,
) -> RerankOutcome:
    """构造 ``failed`` 结果（调用方必须终止本轮，不回落旧候选）。"""

    return RerankOutcome(
        status=STATUS_FAILED,
        chunks=(),
        profile_id=profile_id,
        candidate_count=candidate_count,
        selected_count=0,
        missing_headings=missing_headings,
        heading_diagnostics=dict(heading_diagnostics or {}),
        error_code=error_code,
        error_message=error_message,
        trace=trace,
    )


async def filter_chunks(
    *,
    chunks: Sequence[RetrievedChunk],
    query: str,
    runtime: Any,
    vectorstore: Any = None,
    scope: Any = None,
) -> RerankOutcome:
    """按当前筛选运行态过滤候选块；返回四类状态之一（**不抛异常**）。

    状态语义（与 ``main`` 的 ChatService 路径逐条对齐）：

    * ``disabled``：用户未启用 —— 候选原样返回，调用方走旧检索路径；
    * ``filtered``：评分 + 选择成功，``chunks`` 是最终顺序的已筛选来源；
    * ``empty``：有候选但全部未达阈值 —— 调用方按「资料无关」上报，**不**回落旧候选；
    * ``failed``：已启用但配置不可用，或评分/响应校验失败 —— 调用方必须**终止本轮**
      并指向设置，绝不静默返回未筛选内容（PRD §8.4）。

    取消（``asyncio.CancelledError``）**不吞**：直接向上传播，由运行层按停止语义处理。
    """

    candidates = tuple(chunks)
    snapshot = _resolve_runtime(runtime)
    if snapshot is None:
        return RerankOutcome(status=STATUS_DISABLED, chunks=candidates)

    resolved_scope = _resolve_scope(scope, vectorstore)
    config_error_code = getattr(snapshot, "config_error_code", None)
    provider = getattr(snapshot, "provider", None)
    profile = getattr(snapshot, "profile", None)

    if config_error_code or provider is None or profile is None:
        return _failed(
            profile_id=str(getattr(profile, "profile_id", "") or ""),
            error_code=str(config_error_code or "RERANK_NOT_CONFIGURED"),
            error_message=str(
                getattr(snapshot, "config_error_message", None) or RERANK_UNCONFIGURED_MESSAGE
            ),
            candidate_count=len(candidates),
            trace=_trace_for(
                scope=resolved_scope,
                profile_id=str(getattr(profile, "profile_id", "") or ""),
                views=(),
                status=STATUS_FAILED,
                error_code=str(config_error_code or "RERANK_NOT_CONFIGURED"),
            ),
        )

    if not candidates:
        return RerankOutcome(
            status=STATUS_EMPTY,
            chunks=(),
            profile_id=str(getattr(profile, "profile_id", "") or ""),
            candidate_count=0,
            selected_count=0,
        )

    started = perf_counter()
    heading_lookup = _heading_lookup_for(vectorstore)
    views: list[Any] = []
    source_chunks: dict[str, RetrievedChunk] = {}
    for chunk in candidates:
        for view in build_chunk_views(
            chunk,
            resolved_scope,
            include_structure_views=False,
            heading_lookup=heading_lookup,
        ):
            # 同一逻辑视图只保留第一个：原始块顺序即候选阶段顺序。
            if view.candidate_id in source_chunks:
                continue
            views.append(view)
            source_chunks[view.candidate_id] = chunk
    missing_headings, heading_diagnostics = _missing_headings_of(heading_lookup, len(candidates))

    try:
        outcome = await score_candidates(
            query=query,
            candidates=views,
            profile=profile,
            provider=provider,
            scope=resolved_scope,
        )
    except RerankError as error:
        return _failed(
            profile_id=str(profile.profile_id),
            error_code=error.code.value,
            error_message=error.user_message,
            candidate_count=len(views),
            missing_headings=missing_headings,
            heading_diagnostics=heading_diagnostics,
            trace=_trace_for(
                scope=resolved_scope,
                profile_id=str(profile.profile_id),
                views=views,
                status=STATUS_FAILED,
                error_code=error.code.value,
                total_latency_ms=_elapsed_ms(started),
                missing_headings=missing_headings,
            ),
        )

    selection = select_from_scores(
        candidates=views,
        scores=outcome.score_by_id,
        profile=profile,
    )
    trace = _trace_for(
        scope=resolved_scope,
        profile_id=str(profile.profile_id),
        views=views,
        selection=selection,
        scores=outcome.score_by_id,
        provider_trace_id=outcome.trace_id,
        request_id=outcome.request_id,
        usage=outcome.usage,
        rerank_latency_ms=outcome.latency_ms,
        total_latency_ms=_elapsed_ms(started),
        missing_headings=missing_headings,
    )

    common: dict[str, Any] = {
        "profile_id": str(profile.profile_id),
        "candidate_count": int(selection.candidate_count),
        "missing_headings": missing_headings,
        "heading_diagnostics": heading_diagnostics,
        "trace": trace,
    }
    if selection.status is SelectionStatus.SELECTED:
        selected = _selected_chunks(
            selection,
            source_chunks,
            profile_id=str(profile.profile_id),
            scores=outcome.score_by_id,
        )
        return RerankOutcome(
            status=STATUS_FILTERED,
            chunks=selected,
            selected_count=len(selected),
            **common,
        )
    if selection.status is SelectionStatus.EMPTY:
        return RerankOutcome(status=STATUS_EMPTY, chunks=(), selected_count=0, **common)
    if selection.status is SelectionStatus.CANCELLED:
        return RerankOutcome(
            status=STATUS_FAILED,
            chunks=(),
            selected_count=0,
            error_code="CANCELLED",
            error_message="本轮检索已取消",
            **common,
        )
    error = selection.error
    return RerankOutcome(
        status=STATUS_FAILED,
        chunks=(),
        selected_count=0,
        error_code=error.code.value if error is not None else "RERANK_INVALID_RESPONSE",
        error_message=error.user_message if error is not None else RERANK_FAILED_MESSAGE,
        **common,
    )


def _elapsed_ms(started: float) -> float:
    """自 ``perf_counter`` 起点起经过的毫秒数（≥0）。"""

    return max(0.0, (perf_counter() - started) * 1000.0)


#: 运行态可调用对象的签名（``deps.build_rerank_runtime`` 即此形状）。
RerankRuntimeProvider = Callable[[], Any]
