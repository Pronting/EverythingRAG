"""重排（rerank）内部对象、评分档案与错误类型（PRD RAG-REL-001 §5.3/§6/§8.4/§9）。

本模块只有类型、枚举与纯校验：不发起网络调用、不读写文件、不打印日志。
真实 HTTP 调用在 ``retrieval/rerank_provider.py``，阈值筛选与最终选择在
``retrieval/relevance_selector.py``，证据视图与成员范围在
``retrieval/evidence_view.py``，配置持久化与档案状态在 ``core/settings_store.py``。

Provider 中立：本层不写死任何厂家协议。当前交付实现 ``siliconflow_rerank``
（SiliconFlow ``POST /v1/rerank``，``results[].{index, relevance_score}``，
``meta.tokens`` 计费计量），但接口只描述「提交同一候选集合、取回全部分数」，
新 provider 通过 ``RerankProvider`` 子类与 ``RerankProviderCapabilities`` 注册。

关键约束（PRD §5.1/§6.2/§6.3/§9）：
- 一次请求只提交同一候选集合，``top_n`` 等于候选数；跨请求的相对分数不得直接合并排序
  （``ScoreScope.REQUEST_RELATIVE`` 默认即表达这一点）。
- 每个候选恰好一个有限分数；缺失 / 重复 / 越界 / NaN / 结构错误立即失败，不静默补零。
- 向量相似度、词法、RRF、重排分数分别存放（``ScoreType``）；重排分数不得写回 ``similarity``。
- 排除原因固定为 7 个（``ExclusionReason``）；词法补充只作为候选阶段信号。
- 异常在出现位置抛出，只在 API/任务边界转成用户事件（``RerankError.to_event()``）。
  异常文本与 ``detail`` 必须脱敏：不含问题正文、片段正文、绝对路径、密钥、服务原始响应体；
  校验消息只用候选位置序号，不使用原始 block ID。
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = [
    "CANDIDATE_WINDOW_CHOICES",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "ERROR_USER_MESSAGES",
    "EVIDENCE_CHAR_BUDGET",
    "EVIDENCE_VERSION_CHANGED",
    "MAX_SELECTED_SOURCES",
    "ExclusionReason",
    "ExclusionRecord",
    "ProfileStatus",
    "RerankAuthFailedError",
    "RerankCandidate",
    "RerankError",
    "RerankErrorCode",
    "RerankInputTooLongError",
    "RerankInvalidResponseError",
    "RerankNotConfiguredError",
    "RerankProfile",
    "RerankProfileMismatchError",
    "RerankProvider",
    "RerankProviderCapabilities",
    "RerankProviderType",
    "RerankRateLimitedError",
    "RerankRequest",
    "RerankResult",
    "RerankScorable",
    "RerankScore",
    "RerankServiceUnavailableError",
    "RerankTimeoutError",
    "RerankUsage",
    "RerankValidationError",
    "RouteSignals",
    "ScoreScope",
    "ScoreType",
    "SelectionResult",
    "SelectionStatus",
    "ViewKind",
    "apply_corpus_identity",
    "assert_within_token_limits",
    "corpus_identity_hash",
    "error_for_http_status",
    "sha256_hex",
]

#: 校准档案允许的重排候选上限取值（PRD §5.2：搜索范围 30、50、100）。
CANDIDATE_WINDOW_CHOICES: tuple[int, ...] = (30, 50, 100)
#: 最终来源数量上限（PRD §5.6）。
MAX_SELECTED_SOURCES = 8
#: 最终正文预算（PRD §5.6：产品验收上限 16,000 字符）。
EVIDENCE_CHAR_BUDGET = 16_000
#: 单次网络请求限制（PRD §6.2）：连接 5 秒、读取 60 秒。
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 60.0

_SHA256_HEX_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")


# --------------------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------------------


class ViewKind(StrEnum):
    """证据视图类型（PRD §5.3）。"""

    BLOCK = "block"
    SUPPORT_BUNDLE = "support_bundle"
    IMAGE_TEXT = "image_text"


class ScoreType(StrEnum):
    """四种分数分别存放，任何 UI/日志都不得把其中一种说成概率或正确率（PRD §9）。"""

    DENSE_SIMILARITY = "dense_similarity"
    LEXICAL = "lexical"
    RRF = "rrf"
    RERANK = "rerank"


class ScoreScope(StrEnum):
    """分数可比性口径。

    当前 Provider 的 ``relevance_score`` 是「当前请求内的相对分数」（PRD §7.3），
    同输入重复调用也会微变，因此不同请求的分数不得直接合并排序。
    """

    REQUEST_RELATIVE = "request_relative"
    ABSOLUTE = "absolute"


class ExclusionReason(StrEnum):
    """固定的排除原因（PRD §9）；词法补充等候选阶段信号不能覆盖最终决定。"""

    BELOW_THRESHOLD = "below_threshold"
    DUPLICATE = "duplicate"
    COVERED_BY_BUNDLE = "covered_by_bundle"
    OUTSIDE_CANDIDATE_WINDOW = "outside_candidate_window"
    BUDGET_EXCEEDED = "budget_exceeded"
    SCOPE_MISMATCH = "scope_mismatch"
    VERSION_MISMATCH = "version_mismatch"


class SelectionStatus(StrEnum):
    """最终选择的四类返回状态（PRD §5.6）。"""

    SELECTED = "selected"
    EMPTY = "empty"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProfileStatus(StrEnum):
    """档案状态：candidate（T30 校准）→ quality_passed（T32 质量验收）→ validated（T36）。"""

    CANDIDATE = "candidate"
    QUALITY_PASSED = "quality_passed"
    VALIDATED = "validated"


class RerankProviderType(StrEnum):
    """本交付的 Provider 类型（PRD §6.5）；可插拔，新实现追加新取值。"""

    SILICONFLOW_RERANK = "siliconflow_rerank"


class RerankErrorCode(StrEnum):
    """错误码。前八项是 PRD §8.4 的固定用户错误；后两项是内部契约与真实服务补充。"""

    NOT_CONFIGURED = "RERANK_NOT_CONFIGURED"
    PROFILE_MISMATCH = "RERANK_PROFILE_MISMATCH"
    AUTH_FAILED = "RERANK_AUTH_FAILED"
    RATE_LIMITED = "RERANK_RATE_LIMITED"
    TIMEOUT = "RERANK_TIMEOUT"
    INPUT_TOO_LONG = "RERANK_INPUT_TOO_LONG"
    INVALID_RESPONSE = "RERANK_INVALID_RESPONSE"
    EVIDENCE_VERSION_CHANGED = "EVIDENCE_VERSION_CHANGED"
    #: 真实服务 503/504（实测返回 ``{"code":50505,"message":"Model service overloaded"}``）
    #: 需要一个独立失败态，不能伪装成「返回数据无效」或「无相关资料」。
    SERVICE_UNAVAILABLE = "RERANK_SERVICE_UNAVAILABLE"
    #: 内部契约违例（空候选、重复候选、越界 scope、非法字段等），不对用户展示细节。
    INVALID_INPUT = "RERANK_INVALID_INPUT"


#: 错误码 → 固定用户文案（PRD §8.4）。API/任务边界只投影这两者。
ERROR_USER_MESSAGES: Mapping[RerankErrorCode, str] = {
    RerankErrorCode.NOT_CONFIGURED: "筛选服务尚未配置，请先在设置中完成配置",
    RerankErrorCode.PROFILE_MISMATCH: "当前模型或设置与已验收档案不匹配：请在设置页查看档案状态与验证日期（档案未通过验收、或模型/指令/tokenizer/视图版本/评分协议与档案不一致）",
    RerankErrorCode.AUTH_FAILED: "筛选服务认证失败",
    RerankErrorCode.RATE_LIMITED: "筛选服务请求受限，请稍后重试",
    RerankErrorCode.TIMEOUT: "筛选服务响应超时",
    RerankErrorCode.INPUT_TOO_LONG: "待检查资料超过服务输入限制",
    RerankErrorCode.INVALID_RESPONSE: "筛选服务返回的数据不完整或无效",
    RerankErrorCode.EVIDENCE_VERSION_CHANGED: "资料在本轮处理期间发生变化，请重新提问",
    RerankErrorCode.SERVICE_UNAVAILABLE: "筛选服务暂时不可用，请稍后重试",
    RerankErrorCode.INVALID_INPUT: "重排输入不满足内部契约",
}


# --------------------------------------------------------------------------------------
# 错误类型
# --------------------------------------------------------------------------------------


class RerankError(Exception):
    """重排领域错误基类：类型明确、文案固定、禁止静默降级。

    ``detail`` 只允许放脱敏短描述（状态码、字段名、位置序号）；绝不放服务原始响应体、
    问题正文、片段正文、绝对路径或密钥。``http_status`` 供审计与 API 映射使用。
    """

    code: RerankErrorCode = RerankErrorCode.INVALID_INPUT

    def __init__(self, detail: str | None = None, *, http_status: int | None = None) -> None:
        self.detail = detail
        self.http_status = http_status
        self.user_message = ERROR_USER_MESSAGES[self.code]
        text = f"{self.code.value}: {self.user_message}"
        if detail:
            text = f"{text} ({detail})"
        super().__init__(text)

    def to_event(self) -> dict[str, str]:
        """API/任务边界投影：只含错误码与固定文案，不含 detail 与响应体。"""
        return {"type": "error", "code": self.code.value, "message": self.user_message}


class RerankNotConfiguredError(RerankError):
    """未配置筛选服务（PRD §8.4）：读不到 endpoint/模型/凭证/有效档案时不得自动兜底。"""

    code = RerankErrorCode.NOT_CONFIGURED


class RerankProfileMismatchError(RerankError):
    """档案失配或不可用（PRD §6.5/R14）：模型、指令、tokenizer、视图版本、评分协议变化。"""

    code = RerankErrorCode.PROFILE_MISMATCH


class RerankAuthFailedError(RerankError):
    """认证失败（401/403）。"""

    code = RerankErrorCode.AUTH_FAILED


class RerankRateLimitedError(RerankError):
    """请求受限（429）。"""

    code = RerankErrorCode.RATE_LIMITED


class RerankTimeoutError(RerankError):
    """连接/读取超时（隐式重试已关闭，直接终止本轮）。"""

    code = RerankErrorCode.TIMEOUT


class RerankInputTooLongError(RerankError):
    """单视图或整批输入超过 Provider 已实测的长度限制（禁止静默截断正文/表格/代码）。"""

    code = RerankErrorCode.INPUT_TOO_LONG


class RerankInvalidResponseError(RerankError):
    """响应结构或分数非法（缺项、重复、越界、NaN、非有限值、数量不符）。"""

    code = RerankErrorCode.INVALID_RESPONSE


class RerankServiceUnavailableError(RerankError):
    """服务端不可用（503/504，实测 ``Model service overloaded``）。"""

    code = RerankErrorCode.SERVICE_UNAVAILABLE


class RerankValidationError(RerankError):
    """内部契约违例（编程/接线错误），不是用户配置问题。"""

    code = RerankErrorCode.INVALID_INPUT


class EvidenceVersionChangedError(RerankError):
    """候选的知识版本与请求快照不一致：资料在本轮处理期间发生变化（PRD §8.4）。"""

    code = RerankErrorCode.EVIDENCE_VERSION_CHANGED


#: 名称与错误码同名的别名，便于按 PRD 错误码直接引用。
EVIDENCE_VERSION_CHANGED = RerankErrorCode.EVIDENCE_VERSION_CHANGED

_AUTH_HTTP_STATUSES = frozenset({401, 403})
_RATE_LIMIT_HTTP_STATUSES = frozenset({429})
_TOO_LONG_HTTP_STATUSES = frozenset({413})
_SERVICE_DOWN_HTTP_STATUSES = frozenset({502, 503, 504})
_ENDPOINT_CONFIG_HTTP_STATUSES = frozenset({404, 405, 415})
_REQUEST_REJECTED_HTTP_STATUSES = frozenset({400, 406, 422})


def error_for_http_status(status: int, detail: str | None = None) -> RerankError:
    """把 HTTP 状态映射为类型明确的错误。

    实测（SiliconFlow）：401 字符串体 ``Invalid token``、429 ``TPM limit reached``、
    503/504 ``Model service overloaded``、400 参数错误。响应体原文不得进入 ``detail``。
    """
    if not isinstance(status, int) or isinstance(status, bool):
        raise RerankValidationError("status 必须是整数 HTTP 状态码")
    safe_detail = detail or f"HTTP {status}"
    if status in _AUTH_HTTP_STATUSES:
        return RerankAuthFailedError(safe_detail, http_status=status)
    if status in _RATE_LIMIT_HTTP_STATUSES:
        return RerankRateLimitedError(safe_detail, http_status=status)
    if status in _TOO_LONG_HTTP_STATUSES:
        return RerankInputTooLongError(safe_detail, http_status=status)
    if status in _SERVICE_DOWN_HTTP_STATUSES:
        return RerankServiceUnavailableError(safe_detail, http_status=status)
    if status in _ENDPOINT_CONFIG_HTTP_STATUSES:
        return RerankProfileMismatchError(safe_detail, http_status=status)
    if status in _REQUEST_REJECTED_HTTP_STATUSES:
        return RerankValidationError(safe_detail, http_status=status)
    return RerankInvalidResponseError(safe_detail, http_status=status)


# --------------------------------------------------------------------------------------
# 纯校验工具
# --------------------------------------------------------------------------------------


def sha256_hex(*parts: str) -> str:
    """档案级哈希的共享实现：各部分按顺序以 ``\\n`` 连接后取 SHA-256（UTF-8）。

    用于 ``instruction_hash`` / 语料 manifest 等档案字段。证据视图的 ``payload_hash``
    由 ``evidence_view`` 自行计算，但必须是 64 位十六进制 SHA-256。
    """
    for part in parts:
        if not isinstance(part, str):
            raise RerankValidationError("哈希输入必须是字符串")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankValidationError(f"{name} 必须是非空字符串")
    return value


def _require_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, name)


def _require_sha256(value: object, name: str) -> str:
    text = _require_text(value, name)
    if not _SHA256_HEX_RE.match(text):
        raise RerankValidationError(f"{name} 必须是 64 位十六进制 SHA-256 摘要")
    return text


def _require_optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _require_finite(
    value: object, name: str, *, error_type: type[RerankError] = RerankValidationError
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{name} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f"{name} 必须是有限数值（不接受 NaN/Infinity）")
    return number


def _require_optional_finite(
    value: object, name: str, *, error_type: type[RerankError] = RerankValidationError
) -> float | None:
    if value is None:
        return None
    return _require_finite(value, name, error_type=error_type)


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    error_type: type[RerankError] = RerankValidationError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{name} 必须是整数")
    if value < minimum:
        raise error_type(f"{name} 不能小于 {minimum}")
    return value


def _require_optional_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    error_type: type[RerankError] = RerankValidationError,
) -> int | None:
    if value is None:
        return None
    return _require_int(value, name, minimum=minimum, error_type=error_type)


EnumT = TypeVar("EnumT", bound=StrEnum)


def _coerce_enum(
    value: object,
    enum_type: type[EnumT],
    name: str,
    *,
    error_type: type[RerankError] = RerankValidationError,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    allowed = "/".join(member.value for member in enum_type)
    raise error_type(f"{name} 必须是 {allowed} 之一")


def _require_datetime(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise RerankValidationError(f"{name} 必须是 datetime")
    return value


# --------------------------------------------------------------------------------------
# 候选、通道信号与评分请求
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteSignals:
    """候选的通道原始信号（PRD §5.2/§9）：只作诊断，不授权最终入选。"""

    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    #: BM25-only 候选（无向量分数）也可以参与评分（R02）。
    lexical_only: bool = False

    def __post_init__(self) -> None:
        for name in ("dense_score", "bm25_score", "rrf_score"):
            object.__setattr__(
                self, name, _require_optional_finite(getattr(self, name), name)
            )
        for name in ("dense_rank", "bm25_rank", "rrf_rank"):
            object.__setattr__(
                self,
                name,
                _require_optional_int(getattr(self, name), name, minimum=1),
            )
        if not isinstance(self.lexical_only, bool):
            raise RerankValidationError("lexical_only 必须是布尔值")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "rrf_rank": self.rrf_rank,
            "lexical_only": self.lexical_only,
        }


@runtime_checkable
class RerankScorable(Protocol):
    """评分输入的最小结构契约。

    ``evidence_view.EvidenceView`` 在结构上直接满足本协议，因此重排类型层不必反向
    依赖证据视图模块（避免循环导入），也不同步改造既有 ``RetrievedChunk``。
    """

    @property
    def candidate_id(self) -> str: ...

    @property
    def scope_id(self) -> str: ...

    @property
    def index_revision(self) -> str: ...

    @property
    def ranking_text(self) -> str: ...

    @property
    def payload_hash(self) -> str: ...


@dataclass(frozen=True)
class RerankCandidate:
    """可直接提交评分的候选（``EvidenceView`` 的轻量实现，便于测试与离线评测）。"""

    candidate_id: str
    scope_id: str
    index_revision: str
    ranking_text: str
    payload_hash: str
    view_kind: ViewKind = ViewKind.BLOCK
    primary_block_id: str | None = None
    display_name: str | None = None
    heading_path: str | None = None
    route_signals: RouteSignals | None = None

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.scope_id, "scope_id")
        _require_text(self.index_revision, "index_revision")
        _require_text(self.ranking_text, "ranking_text")
        _require_sha256(self.payload_hash, "payload_hash")
        object.__setattr__(
            self,
            "view_kind",
            _coerce_enum(self.view_kind, ViewKind, "view_kind"),
        )
        if self.route_signals is not None and not isinstance(
            self.route_signals, RouteSignals
        ):
            raise RerankValidationError("route_signals 必须是 RouteSignals")


def _document_field(document: object, name: str, position: int) -> object:
    try:
        return getattr(document, name)
    except AttributeError:
        raise RerankValidationError(
            f"候选（位置 {position}）缺少字段 {name}"
        ) from None


@dataclass(frozen=True)
class RerankRequest:
    """一次重排请求：固定原问题、scope、知识版本与候选集合（PRD §5.1/§6.2）。

    校验在构造时完成，任何非法输入都在外呼之前失败：空问题、空候选、重复候选、
    scope 越界、知识版本不一致、缺正文、缺哈希、``top_n`` 与候选数不符。
    """

    query: str
    scope_id: str
    index_revision: str
    documents: tuple[RerankScorable, ...]
    profile_id: str
    instruction_id: str
    instruction_text: str = ""
    top_n: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.query, "query")
        scope_id = _require_text(self.scope_id, "scope_id")
        index_revision = _require_text(self.index_revision, "index_revision")
        _require_text(self.profile_id, "profile_id")
        _require_text(self.instruction_id, "instruction_id")
        if not isinstance(self.instruction_text, str):
            raise RerankValidationError("instruction_text 必须是字符串")

        documents = tuple(self.documents)
        if not documents:
            raise RerankValidationError("候选列表不能为空：没有可评分的证据视图")
        seen: dict[str, int] = {}
        for position, document in enumerate(documents):
            candidate_id = _require_text(
                _document_field(document, "candidate_id", position),
                f"候选（位置 {position}）candidate_id",
            )
            if candidate_id in seen:
                raise RerankValidationError(
                    f"候选 ID 重复（位置 {seen[candidate_id]} 与 {position}）"
                )
            seen[candidate_id] = position
            document_scope = _require_text(
                _document_field(document, "scope_id", position),
                f"候选（位置 {position}）scope_id",
            )
            if document_scope != scope_id:
                raise RerankValidationError(
                    f"候选（位置 {position}）scope 越界：与请求 scope 不一致"
                )
            document_revision = _require_text(
                _document_field(document, "index_revision", position),
                f"候选（位置 {position}）index_revision",
            )
            if document_revision != index_revision:
                raise EvidenceVersionChangedError(
                    f"候选（位置 {position}）知识版本与请求快照不一致"
                )
            _require_text(
                _document_field(document, "ranking_text", position),
                f"候选（位置 {position}）ranking_text",
            )
            _require_sha256(
                _document_field(document, "payload_hash", position),
                f"候选（位置 {position}）payload_hash",
            )
            view_kind = getattr(document, "view_kind", None)
            if view_kind is not None:
                _coerce_enum(view_kind, ViewKind, f"候选（位置 {position}）view_kind")

        if self.top_n is not None:
            top_n = _require_int(self.top_n, "top_n", minimum=1)
            if top_n != len(documents):
                raise RerankValidationError(
                    f"top_n（{top_n}）必须等于候选数（{len(documents)}）："
                    "要求返回全部分数"
                )
        object.__setattr__(self, "documents", documents)

    @property
    def top_n_value(self) -> int:
        """实际提交的 ``top_n``：恒等于候选数。"""
        return len(self.documents)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """按提交顺序排列的候选 ID（响应 ``index`` 直接索引本元组）。"""
        return tuple(document.candidate_id for document in self.documents)

    def to_audit_dict(self) -> dict[str, Any]:
        """脱敏审计视图：不含问题正文、片段正文、路径与密钥（PRD §9）。"""
        return {
            "scope_id": self.scope_id,
            "index_revision": self.index_revision,
            "profile_id": self.profile_id,
            "instruction_id": self.instruction_id,
            "candidate_count": self.top_n_value,
            "top_n": self.top_n_value,
        }


# --------------------------------------------------------------------------------------
# 分数与 Provider 响应
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankUsage:
    """服务返回的计量（PRD §6.2）。

    SiliconFlow 实测在 ``meta.tokens`` 返回 ``input_tokens`` / ``output_tokens`` /
    ``image_tokens``，并有 ``meta.billed_units``。服务未返回时保持 ``None`` 并显式显示
    「服务未返回」，不得估算或填零。
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    image_tokens: int | None = None
    billed_units: float | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "image_tokens"):
            object.__setattr__(
                self,
                name,
                _require_optional_int(
                    getattr(self, name),
                    name,
                    minimum=0,
                    error_type=RerankInvalidResponseError,
                ),
            )
        object.__setattr__(
            self,
            "billed_units",
            _require_optional_finite(
                self.billed_units, "billed_units", error_type=RerankInvalidResponseError
            ),
        )

    @property
    def reported(self) -> bool:
        """服务是否真的返回了计量。"""
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.image_tokens,
                self.billed_units,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "image_tokens": self.image_tokens,
            "billed_units": self.billed_units,
        }


@dataclass(frozen=True)
class RerankScore:
    """单个候选的独立分数（PRD §5.3/§8.3：分数类型分别保存）。"""

    candidate_id: str
    index: int
    score: float
    score_type: ScoreType = ScoreType.RERANK

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        object.__setattr__(
            self,
            "index",
            _require_int(self.index, "index", error_type=RerankInvalidResponseError),
        )
        object.__setattr__(
            self,
            "score",
            _require_finite(self.score, "score", error_type=RerankInvalidResponseError),
        )
        object.__setattr__(
            self,
            "score_type",
            _coerce_enum(
                self.score_type, ScoreType, "score_type", error_type=RerankInvalidResponseError
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "index": self.index,
            "score": self.score,
            "score_type": str(self.score_type),
        }


@dataclass(frozen=True)
class RerankResult:
    """一次 Provider 调用的完整结果：每个候选恰好一个有限分数（PRD §6.2）。"""

    profile_id: str
    provider_type: str
    model: str
    document_count: int
    scores: tuple[RerankScore, ...]
    request_id: str | None = None
    trace_id: str | None = None
    usage: RerankUsage | None = None
    latency_ms: float | None = None
    score_scope: ScoreScope = ScoreScope.REQUEST_RELATIVE

    def __post_init__(self) -> None:
        _require_text(self.profile_id, "profile_id")
        _require_text(self.provider_type, "provider_type")
        _require_text(self.model, "model")
        document_count = _require_int(
            self.document_count, "document_count", minimum=1,
            error_type=RerankInvalidResponseError,
        )
        scores = tuple(self.scores)
        if len(scores) != document_count:
            raise RerankInvalidResponseError(
                f"分数数量（{len(scores)}）与候选数（{document_count}）不一致："
                "要求返回全部分数"
            )
        seen_ids: set[str] = set()
        seen_indexes: set[int] = set()
        for score in scores:
            if not isinstance(score, RerankScore):
                raise RerankInvalidResponseError("scores 必须是 RerankScore 序列")
            if score.candidate_id in seen_ids:
                raise RerankInvalidResponseError("同一候选返回了重复分数")
            seen_ids.add(score.candidate_id)
            if score.index in seen_indexes:
                raise RerankInvalidResponseError("响应 index 重复")
            if score.index >= document_count:
                raise RerankInvalidResponseError(
                    f"响应 index 越界（{score.index} >= {document_count}）"
                )
            seen_indexes.add(score.index)
        object.__setattr__(self, "document_count", document_count)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(
            self,
            "score_scope",
            _coerce_enum(
                self.score_scope, ScoreScope, "score_scope",
                error_type=RerankInvalidResponseError,
            ),
        )
        object.__setattr__(
            self,
            "latency_ms",
            _require_optional_finite(
                self.latency_ms, "latency_ms", error_type=RerankInvalidResponseError
            ),
        )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise RerankInvalidResponseError("latency_ms 不能为负")
        object.__setattr__(self, "request_id", _require_optional_text(self.request_id, "request_id"))
        object.__setattr__(self, "trace_id", _require_optional_text(self.trace_id, "trace_id"))
        if self.usage is not None and not isinstance(self.usage, RerankUsage):
            raise RerankInvalidResponseError("usage 必须是 RerankUsage 或 None")

    @property
    def usage_reported(self) -> bool:
        """服务是否返回了计费计量；未返回时费用必须记为不可计算（PRD §9）。"""
        return self.usage is not None and self.usage.reported

    def score_for(self, candidate_id: str) -> float | None:
        for score in self.scores:
            if score.candidate_id == candidate_id:
                return score.score
        return None

    def assert_matches_request(self, request: RerankRequest) -> None:
        """响应必须与提交的候选集合逐位对应（真实服务按原输入 index 返回）。"""
        if self.document_count != request.top_n_value:
            raise RerankInvalidResponseError(
                f"响应候选数（{self.document_count}）与请求候选数"
                f"（{request.top_n_value}）不一致"
            )
        if self.profile_id != request.profile_id:
            raise RerankProfileMismatchError("响应档案与请求档案不一致")
        expected = request.candidate_ids
        for score in self.scores:
            if expected[score.index] != score.candidate_id:
                raise RerankInvalidResponseError(
                    f"响应 index（{score.index}）与候选 ID 不对应"
                )

    def assert_scores_in_range(
        self, capabilities: RerankProviderCapabilities
    ) -> None:
        """每个分数必须落在该 Provider/模型声明的范围内（PRD §6.2）。"""
        for score in self.scores:
            capabilities.assert_score_in_range(score.score)

    def to_dict(self) -> dict[str, Any]:
        """脱敏结果视图：只含分数与身份，不含任何正文。"""
        return {
            "profile_id": self.profile_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "document_count": self.document_count,
            "scores": [score.to_dict() for score in self.scores],
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "latency_ms": self.latency_ms,
            "score_scope": str(self.score_scope),
        }


# --------------------------------------------------------------------------------------
# Provider 能力与抽象接口
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankProviderCapabilities:
    """Provider 能力描述（PRD §6.3）。

    长度与数量限制必须来自真实探测（T11）：``None`` 表示尚未实测，不得用其他模型或
    其他厂商的数值（例如百炼的 4,000 tokens）填充。``supports_instruction`` 与
    ``supports_chunk_options`` 按模型声明：实测 SiliconFlow ``instruction`` 只对
    Qwen3-Reranker 系列生效，``max_chunks_per_doc``/``overlap_tokens`` 只对
    bge-reranker-v2-m3 / Pro / bce 系列生效，且不支持时服务不报错——必须由本层显式拒绝。
    """

    provider_type: str
    model: str
    tokenizer_id: str
    max_documents: int | None = None
    max_query_tokens: int | None = None
    max_document_tokens: int | None = None
    max_request_tokens: int | None = None
    score_min: float = 0.0
    score_max: float = 1.0
    supports_instruction: bool = False
    supports_chunk_options: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text(self.provider_type, "provider_type")
        _require_text(self.model, "model")
        _require_text(self.tokenizer_id, "tokenizer_id")
        for name in (
            "max_documents",
            "max_query_tokens",
            "max_document_tokens",
            "max_request_tokens",
        ):
            object.__setattr__(
                self, name, _require_optional_int(getattr(self, name), name, minimum=1)
            )
        score_min = _require_finite(self.score_min, "score_min")
        score_max = _require_finite(self.score_max, "score_max")
        if score_max <= score_min:
            raise RerankValidationError("score_max 必须大于 score_min")
        object.__setattr__(self, "score_min", score_min)
        object.__setattr__(self, "score_max", score_max)
        for name in ("supports_instruction", "supports_chunk_options"):
            if not isinstance(getattr(self, name), bool):
                raise RerankValidationError(f"{name} 必须是布尔值")
        if not isinstance(self.notes, str):
            raise RerankValidationError("notes 必须是字符串")

    def assert_document_count(self, count: int) -> None:
        if count < 1:
            raise RerankValidationError("候选数必须大于 0")
        if self.max_documents is not None and count > self.max_documents:
            raise RerankInputTooLongError(
                f"候选数（{count}）超过服务上限（{self.max_documents}）"
            )

    def assert_score_in_range(self, score: float) -> None:
        value = _require_finite(score, "score", error_type=RerankInvalidResponseError)
        if value < self.score_min or value > self.score_max:
            raise RerankInvalidResponseError(
                f"分数（{value}）超出声明的分数范围"
                f"[{self.score_min}, {self.score_max}]"
            )

    def assert_instruction_supported(self) -> None:
        """模型不支持 instruction 时显式拒绝，不能依赖服务报错（服务会静默忽略）。"""
        if not self.supports_instruction:
            raise RerankProfileMismatchError(
                f"模型 {self.model} 不支持 instruction，档案配置不匹配"
            )

    def assert_chunk_options_supported(self) -> None:
        if not self.supports_chunk_options:
            raise RerankProfileMismatchError(
                f"模型 {self.model} 不支持 max_chunks_per_doc/overlap_tokens"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_type": self.provider_type,
            "model": self.model,
            "tokenizer_id": self.tokenizer_id,
            "max_documents": self.max_documents,
            "max_query_tokens": self.max_query_tokens,
            "max_document_tokens": self.max_document_tokens,
            "max_request_tokens": self.max_request_tokens,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "supports_instruction": self.supports_instruction,
            "supports_chunk_options": self.supports_chunk_options,
            "notes": self.notes,
        }


def assert_within_token_limits(
    capabilities: RerankProviderCapabilities,
    *,
    query_tokens: int,
    document_tokens: Sequence[int],
    request_tokens: int | None = None,
) -> None:
    """调用前的长度校验（PRD §6.3）。

    token 数必须由成熟 tokenizer 实际计数（不能用字符数替代）；未实测的限制为 ``None``
    时跳过该项，不做猜测。超限抛 ``RerankInputTooLongError``，禁止静默截断。
    """
    query_count = _require_int(query_tokens, "query_tokens", minimum=0)
    if (
        capabilities.max_query_tokens is not None
        and query_count > capabilities.max_query_tokens
    ):
        raise RerankInputTooLongError(
            f"问题长度（{query_count} tokens）超过单条上限"
            f"（{capabilities.max_query_tokens}）"
        )
    counts = tuple(document_tokens)
    for position, count in enumerate(counts):
        document_count = _require_int(count, f"document_tokens[{position}]", minimum=0)
        if (
            capabilities.max_document_tokens is not None
            and document_count > capabilities.max_document_tokens
        ):
            raise RerankInputTooLongError(
                f"候选（位置 {position}）长度（{document_count} tokens）超过单条上限"
                f"（{capabilities.max_document_tokens}）"
            )
    if request_tokens is not None:
        total = _require_int(request_tokens, "request_tokens", minimum=0)
        if (
            capabilities.max_request_tokens is not None
            and total > capabilities.max_request_tokens
        ):
            raise RerankInputTooLongError(
                f"候选总长度（{total} tokens）超过请求上限"
                f"（{capabilities.max_request_tokens}）"
            )
    capabilities.assert_document_count(len(counts))


class RerankProvider(ABC):
    """重排 Provider 抽象接口（PRD §6.1/§6.2）。

    实现约定：
    - 异步，不阻塞事件循环；取消信号传递到 HTTP 请求；
    - 隐式重试关闭：429 / 认证失败 / 超时 / 协议异常直接抛类型明确的错误；
    - 一次调用提交完整候选集合，``top_n`` 等于候选数，返回每个候选恰好一个分数；
    - 失败绝不返回「成功但空列表」；无可用资料用 ``SelectionResult`` 的空状态表达。
    """

    @property
    @abstractmethod
    def provider_type(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> RerankProviderCapabilities: ...

    @abstractmethod
    async def rerank(self, request: RerankRequest) -> RerankResult: ...

    async def aclose(self) -> None:
        """释放连接池；默认无操作。"""
        return


# --------------------------------------------------------------------------------------
# 排除记录与最终选择结果
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExclusionRecord:
    """一个候选的一次排除记录（PRD §9）。

    生产日志使用不可逆 ``block_ref``；``detail`` 不得包含正文、标题或路径。
    """

    candidate_id: str
    reason: ExclusionReason
    stage: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        object.__setattr__(
            self, "reason", _coerce_enum(self.reason, ExclusionReason, "reason")
        )
        object.__setattr__(self, "stage", _require_optional_text(self.stage, "stage"))
        object.__setattr__(self, "detail", _require_optional_text(self.detail, "detail"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": str(self.reason),
            "stage": self.stage,
            "detail": self.detail,
        }


def _selected_text(document: object) -> str:
    evidence_text = getattr(document, "evidence_text", None)
    if isinstance(evidence_text, str) and evidence_text:
        return evidence_text
    ranking_text = getattr(document, "ranking_text", "")
    return ranking_text if isinstance(ranking_text, str) else ""


@dataclass(frozen=True)
class SelectionResult:
    """最终选择结果（PRD §5.6）：四类状态，0–8 项，正文不超过 16,000 字符。

    ``relevance_selector`` 复用本类型；不要另建同义类型。
    """

    status: SelectionStatus
    profile_id: str = ""
    selected: tuple[RerankScorable, ...] = ()
    exclusions: tuple[ExclusionRecord, ...] = ()
    candidate_count: int = 0
    error: RerankError | None = None

    def __post_init__(self) -> None:
        status = _coerce_enum(self.status, SelectionStatus, "status")
        selected = tuple(self.selected)
        exclusions = tuple(self.exclusions)
        candidate_count = _require_int(self.candidate_count, "candidate_count", minimum=0)
        if not isinstance(self.profile_id, str):
            raise RerankValidationError("profile_id 必须是字符串")
        if len(selected) > MAX_SELECTED_SOURCES:
            raise RerankValidationError(
                f"最终来源数（{len(selected)}）超过上限 {MAX_SELECTED_SOURCES}"
            )
        seen: set[str] = set()
        for position, document in enumerate(selected):
            candidate_id = _require_text(
                _document_field(document, "candidate_id", position),
                f"最终证据（位置 {position}）candidate_id",
            )
            if candidate_id in seen:
                raise RerankValidationError(f"最终证据重复（位置 {position}）")
            seen.add(candidate_id)
        excluded: set[str] = set()
        for record in exclusions:
            if not isinstance(record, ExclusionRecord):
                raise RerankValidationError("exclusions 必须是 ExclusionRecord 序列")
            if record.candidate_id in excluded:
                raise RerankValidationError("同一候选出现多条排除记录")
            excluded.add(record.candidate_id)
        total_chars = sum(len(_selected_text(document)) for document in selected)
        if total_chars > EVIDENCE_CHAR_BUDGET:
            raise RerankValidationError(
                f"最终正文（{total_chars} 字符）超过预算 {EVIDENCE_CHAR_BUDGET}"
            )
        if self.error is not None and not isinstance(self.error, RerankError):
            raise RerankValidationError("error 必须是 RerankError 或 None")
        if status is SelectionStatus.SELECTED:
            if not selected:
                raise RerankValidationError("status=selected 时 final 证据不能为空")
            if self.error is not None:
                raise RerankValidationError("status=selected 时不能携带错误")
        elif status is SelectionStatus.EMPTY:
            if selected:
                raise RerankValidationError("status=empty 时不能包含最终证据")
            if self.error is not None:
                raise RerankValidationError("status=empty 时不能携带错误")
        elif status is SelectionStatus.FAILED:
            if self.error is None:
                raise RerankValidationError("status=failed 时必须携带 RerankError")
            if selected:
                raise RerankValidationError("status=failed 时不能包含最终证据")
        else:  # CANCELLED
            if selected:
                raise RerankValidationError("status=cancelled 时不能包含最终证据")
            if self.error is not None:
                raise RerankValidationError("status=cancelled 不是错误状态")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "candidate_count", candidate_count)

    @property
    def evidence_char_count(self) -> int:
        return sum(len(_selected_text(document)) for document in self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "profile_id": self.profile_id,
            "selected": [
                str(_document_field(document, "candidate_id", position))
                for position, document in enumerate(self.selected)
            ],
            "exclusions": [record.to_dict() for record in self.exclusions],
            "candidate_count": self.candidate_count,
            "evidence_char_count": self.evidence_char_count,
            "error_code": self.error.code.value if self.error is not None else None,
        }


# --------------------------------------------------------------------------------------
# 校准档案
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankProfile:
    """校准档案（PRD §6.5）：线上启用的唯一权威参数集，绝不包含密钥。

    状态流转 ``candidate`` → ``quality_passed`` → ``validated``；生产启用只接受
    ``validated``。模型、指令、tokenizer、视图版本或评分协议任一变化都使档案失配，
    必须重新校准。语料新增/修改由 ``needs_validation`` 标记，但仍继续使用当前档案。
    """

    profile_id: str
    status: ProfileStatus
    provider_type: str
    model: str
    endpoint_fingerprint: str
    instruction_id: str
    instruction_hash: str
    tokenizer_id: str
    view_version: str
    scoring_protocol_version: str
    score_threshold: float
    max_candidates: int
    min_candidate_count: int
    instruction_text: str = ""
    max_selected: int = MAX_SELECTED_SOURCES
    evidence_char_budget: int = EVIDENCE_CHAR_BUDGET
    max_query_tokens: int | None = None
    max_document_tokens: int | None = None
    max_request_tokens: int | None = None
    corpus_manifest_hash: str | None = None
    calibration_set_hash: str | None = None
    test_set_hash: str | None = None
    generated_at: datetime | None = None
    validated_at: datetime | None = None
    needs_validation: bool = False
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text(self.profile_id, "profile_id")
        object.__setattr__(
            self, "status", _coerce_enum(self.status, ProfileStatus, "status")
        )
        _require_text(self.provider_type, "provider_type")
        _require_text(self.model, "model")
        _require_text(self.endpoint_fingerprint, "endpoint_fingerprint")
        _require_text(self.instruction_id, "instruction_id")
        _require_sha256(self.instruction_hash, "instruction_hash")
        _require_text(self.tokenizer_id, "tokenizer_id")
        _require_text(self.view_version, "view_version")
        _require_text(self.scoring_protocol_version, "scoring_protocol_version")
        if not isinstance(self.instruction_text, str):
            raise RerankValidationError("instruction_text 必须是字符串")

        threshold = _require_finite(self.score_threshold, "score_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise RerankValidationError("score_threshold 必须落在 [0, 1] 区间")
        object.__setattr__(self, "score_threshold", threshold)

        max_candidates = _require_int(self.max_candidates, "max_candidates", minimum=1)
        if max_candidates not in CANDIDATE_WINDOW_CHOICES:
            allowed = "/".join(str(choice) for choice in CANDIDATE_WINDOW_CHOICES)
            raise RerankValidationError(f"max_candidates 必须是 {allowed} 之一")
        min_candidates = _require_int(
            self.min_candidate_count, "min_candidate_count", minimum=1
        )
        if min_candidates > max_candidates:
            raise RerankValidationError("min_candidate_count 不能大于 max_candidates")
        object.__setattr__(self, "max_candidates", max_candidates)
        object.__setattr__(self, "min_candidate_count", min_candidates)

        max_selected = _require_int(self.max_selected, "max_selected", minimum=1)
        if max_selected > MAX_SELECTED_SOURCES:
            raise RerankValidationError(
                f"max_selected 不能超过 {MAX_SELECTED_SOURCES}"
            )
        object.__setattr__(self, "max_selected", max_selected)
        object.__setattr__(
            self,
            "evidence_char_budget",
            _require_int(self.evidence_char_budget, "evidence_char_budget", minimum=1),
        )
        for name in ("max_query_tokens", "max_document_tokens", "max_request_tokens"):
            object.__setattr__(
                self, name, _require_optional_int(getattr(self, name), name, minimum=1)
            )
        for name in (
            "corpus_manifest_hash",
            "calibration_set_hash",
            "test_set_hash",
        ):
            object.__setattr__(
                self, name, _require_optional_sha256(getattr(self, name), name)
            )
        object.__setattr__(self, "generated_at", _require_datetime(self.generated_at, "generated_at"))
        object.__setattr__(self, "validated_at", _require_datetime(self.validated_at, "validated_at"))
        if not isinstance(self.needs_validation, bool):
            raise RerankValidationError("needs_validation 必须是布尔值")
        metrics = self.metrics
        if not isinstance(metrics, Mapping):
            raise RerankValidationError("metrics 必须是映射")
        for key, value in metrics.items():
            _require_text(key, "metrics 键")
            _require_finite(value, f"metrics[{key}]")
        if not isinstance(self.notes, str):
            raise RerankValidationError("notes 必须是字符串")

    @property
    def production_ready(self) -> bool:
        """生产启用只接受 ``validated``（PRD §6.5）。"""
        return self.status is ProfileStatus.VALIDATED

    def assert_production_ready(self) -> None:
        if not self.production_ready:
            raise RerankProfileMismatchError(
                f"档案 {self.profile_id} 状态为 {self.status}，未通过验收，不能在生产启用"
            )

    def compatibility_key(self) -> tuple[str, ...]:
        """档案身份；任一字段变化即失配，必须重新校准（R14）。"""
        return (
            str(self.provider_type),
            self.model,
            self.endpoint_fingerprint,
            self.instruction_hash,
            self.tokenizer_id,
            self.view_version,
            self.scoring_protocol_version,
        )

    def assert_compatible(
        self,
        *,
        provider_type: str | None = None,
        model: str | None = None,
        endpoint_fingerprint: str | None = None,
        instruction_hash: str | None = None,
        tokenizer_id: str | None = None,
        view_version: str | None = None,
        scoring_protocol_version: str | None = None,
    ) -> None:
        """只比较显式传入的字段；任何不一致立即失配（返回校验错误，不允许静默降级）。"""
        candidates = (
            ("provider_type", provider_type, str(self.provider_type)),
            ("model", model, self.model),
            ("endpoint_fingerprint", endpoint_fingerprint, self.endpoint_fingerprint),
            ("instruction_hash", instruction_hash, self.instruction_hash),
            ("tokenizer_id", tokenizer_id, self.tokenizer_id),
            ("view_version", view_version, self.view_version),
            (
                "scoring_protocol_version",
                scoring_protocol_version,
                self.scoring_protocol_version,
            ),
        )
        mismatched = [
            name for name, actual, expected in candidates if actual is not None and actual != expected
        ]
        if mismatched:
            raise RerankProfileMismatchError(
                f"档案 {self.profile_id} 与当前配置不匹配：{', '.join(mismatched)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """脱敏投影：档案本身不含密钥，可安全用于设置页与评测报告。"""
        return {
            "profile_id": self.profile_id,
            "status": str(self.status),
            "provider_type": str(self.provider_type),
            "model": self.model,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "instruction_id": self.instruction_id,
            "instruction_hash": self.instruction_hash,
            "tokenizer_id": self.tokenizer_id,
            "view_version": self.view_version,
            "scoring_protocol_version": self.scoring_protocol_version,
            "score_threshold": self.score_threshold,
            "max_candidates": self.max_candidates,
            "min_candidate_count": self.min_candidate_count,
            "max_selected": self.max_selected,
            "evidence_char_budget": self.evidence_char_budget,
            "max_query_tokens": self.max_query_tokens,
            "max_document_tokens": self.max_document_tokens,
            "max_request_tokens": self.max_request_tokens,
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "calibration_set_hash": self.calibration_set_hash,
            "test_set_hash": self.test_set_hash,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "needs_validation": self.needs_validation,
            "metrics": dict(self.metrics),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# 语料身份（PRD §6.5：「已验收」只对应档案验证过的语料版本）
# --------------------------------------------------------------------------------------


def corpus_identity_hash(
    *,
    collection: str = "",
    embed_fingerprint: str = "",
    block_ids: Sequence[str] = (),
) -> str:
    """语料身份哈希：collection + 嵌入 fingerprint + 有序块 id 清单（PRD §6.5）。

    块 id 先排序再哈希，因此与遍历顺序无关：同一语料任意时刻得到同一身份；语料新增、删除或
    重建索引（块 id 变化）都会改变身份。不含任何正文，可安全写进档案与设置页。
    """

    digest = hashlib.sha256()
    digest.update(b"corpus-identity-v1\0")
    digest.update(str(collection).strip().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(embed_fingerprint).strip().encode("utf-8"))
    ids = sorted(str(block_id) for block_id in block_ids)
    digest.update(f"\0{len(ids)}\0".encode())
    for block_id in ids:
        digest.update(block_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def apply_corpus_identity(profile: RerankProfile, corpus_identity: str | None) -> RerankProfile:
    """按当前语料身份标记 ``needs_validation``（PRD §6.5）。

    「已验收」只对应档案验证过的语料版本：当前身份与档案 ``corpus_manifest_hash`` 不一致，
    或档案根本没登记语料身份、当前身份不可得（``None``，无从证明）时标记 ``needs_validation``
    ——只**标记**，仍继续使用当前兼容档案（§6.5 明写不阻止使用）；一致时清掉标记，让设置页
    状态如实回到 ready。
    """

    registered = profile.corpus_manifest_hash
    if not isinstance(registered, str) or not registered.strip():
        return replace(profile, needs_validation=True)
    if not isinstance(corpus_identity, str) or not corpus_identity.strip():
        return replace(profile, needs_validation=True)
    return replace(profile, needs_validation=corpus_identity.strip() != registered.strip())
