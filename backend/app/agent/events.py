"""SSE 事件与 `run_events` 的统一事件对象（PRD 13.3）。Owner: T00b（契约部分）。

本模块定义统一事件对象、各类 ``type`` 的 payload 模型、以及
编解码与校验入口。事件发布器与 SSE 序列化（帧拼装、断线补读）归 approval-privacy
（见 `docs/tech/v1.2.0-architecture.md` §2.1 / §2.6），他们只在上方追加，不改本文件
的定义。

三条硬约束（PRD 12.5 / 13.3）：

1. ``type`` 的唯一来源是 :class:`app.agent.contracts.EventType`，本模块不再定义
   第二份类型表。
2. 未知 ``schema_version``、未知 ``type``、payload 校验失败一律抛
   :class:`EventProtocolError`，**不允许静默降级或返回空对象**。
3. payload 模型 ``extra="forbid"``：多传未声明字段视为契约违规并直接报错；
   需要新增字段时向 contracts-db 提需求，由本模块统一追加。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field, ValidationError

from app.agent.contracts import (
    EVENT_SCHEMA_VERSION,
    AgentError,
    AgentKind,
    AnswerBasis,
    ApprovalDecisionKind,
    ApprovalRequest,
    CitationEntry,
    ContractModel,
    EventType,
    EvidenceSourceType,
    InputRequest,
    RouteDecision,
    RunStatus,
    SourceType,
    TaskStatus,
    VerificationResult,
    dump_contract,
)

__all__ = [
    "EVENT_PAYLOAD_MODELS",
    "EVENT_SCHEMA_VERSION",
    "AgentCompletedPayload",
    "AgentStartedPayload",
    "AnswerDeltaPayload",
    "AnswerMetaPayload",
    "ApprovalRequestedPayload",
    "ApprovalResolvedPayload",
    "EventEnvelope",
    "EventPayload",
    "EventProtocolError",
    "EvidenceAddedPayload",
    "EvidenceSummary",
    "InputRequestedPayload",
    "InputResolvedPayload",
    "MemoryFailedPayload",
    "MemoryUpdatedPayload",
    "PlanTaskSummary",
    "PlanUpdatedPayload",
    "RouteDecidedPayload",
    "RunCancelledPayload",
    "RunCompletedPayload",
    "RunFailedPayload",
    "RunStatusPayload",
    "StoredEvent",
    "ToolCompletedPayload",
    "ToolStartedPayload",
    "UsagePayload",
    "VerificationUpdatedPayload",
    "build_event",
    "dumps_event",
    "loads_event",
    "parse_event",
    "payload_model_for",
]


class EventProtocolError(ValueError):
    """事件协议错误：未知版本、未知类型或 payload 校验失败（PRD 13.3 末段）。"""


class EventPayload(ContractModel):
    """全部事件 payload 的基类（``extra="forbid"``，见模块 docstring）。"""


# =====================================================================
# payload 模型（PRD 13.3 表的 20 个 type 取值，见架构文档 D1）
# =====================================================================


class RunStatusPayload(EventPayload):
    """``run.status``：任务状态和当前处理阶段。"""

    status: RunStatus
    phase: str = ""
    message: str = ""
    model_call: dict[str, Any] | None = None


class RouteDecidedPayload(EventPayload):
    """``route.decided``：来源选择和简短原因（附 PRD 6.2 的完整 RouteDecision）。"""

    decision: RouteDecision


class PlanTaskSummary(ContractModel):
    """``plan.updated`` 里的单个子任务摘要（字段与 PRD 6.4 一致）。"""

    task_id: str
    question: str
    agent_kind: AgentKind
    status: TaskStatus
    depends_on: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    query_identity: str = ""
    query_reuse_reason: str = ""


class PlanUpdatedPayload(EventPayload):
    """``plan.updated``：子问题、依赖和完成情况。"""

    tasks: list[PlanTaskSummary] = Field(default_factory=list)
    covered_requirements: list[str] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)


class AgentStartedPayload(EventPayload):
    """``agent.started``：Agent 职责。"""

    agent_kind: AgentKind
    thread_id: str = ""
    task_id: str | None = None
    question: str = ""


class AgentCompletedPayload(EventPayload):
    """``agent.completed``：Agent 职责与结果摘要。"""

    agent_kind: AgentKind
    status: TaskStatus = TaskStatus.COMPLETED
    summary: str = ""


class ToolStartedPayload(EventPayload):
    """``tool.started``：工具动作与参数摘要。"""

    tool_name: str
    args_summary: str = ""
    operation_id: str = ""
    task_id: str | None = None
    summary: str = ""


class ProcessSource(ContractModel):
    title: str
    text: str = ""


class ModelReasoningPayload(EventPayload):
    step_id: str
    stream_id: str
    text: str
    title: str


class ToolCompletedPayload(EventPayload):
    """``tool.completed``：来源及结果数量。"""

    tool_name: str
    operation_id: str = ""
    status: str = "succeeded"  # succeeded / failed（失败细节见 error_code）
    result_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    error_code: str = ""
    summary: str = ""
    details: list[str] = Field(default_factory=list)
    sources: list[ProcessSource] = Field(default_factory=list)


class ApprovalRequestedPayload(EventPayload):
    """``approval.requested``：批准卡片（PRD 10.2 完整对象）。"""

    approval: ApprovalRequest


class ApprovalResolvedPayload(EventPayload):
    """``approval.resolved``：决定结果。"""

    approval_id: str
    status: str
    decision_kind: ApprovalDecisionKind | None = None


class InputRequestedPayload(EventPayload):
    """``input.requested``：澄清问题。"""

    request: InputRequest


class InputResolvedPayload(EventPayload):
    """``input.resolved``：澄清回答。"""

    input_request_id: str
    answer: str


class EvidenceSummary(ContractModel):
    """``evidence.added`` 里可查看的资料条目（完整证据见 Evidence 契约）。"""

    evidence_id: str
    source_type: EvidenceSourceType
    title: str = ""
    source_ref: str = ""
    snippet: str = ""


class EvidenceAddedPayload(EventPayload):
    """``evidence.added``：可查看的资料及来源。"""

    items: list[EvidenceSummary] = Field(default_factory=list)


class VerificationUpdatedPayload(EventPayload):
    """``verification.updated``：核对项目、完成状态和待补充内容（PRD 8.2）。"""

    result: VerificationResult
    pending: list[str] = Field(default_factory=list)


class AnswerMetaPayload(EventPayload):
    """``answer.meta``：同一答案的来源、内容摘要与完成状态。"""

    answer_id: str
    answer_basis: AnswerBasis
    basis_components: list[SourceType] = Field(default_factory=list)
    citations: list[CitationEntry] = Field(default_factory=list)
    content_hash: str = ""
    status: RunStatus = RunStatus.COMPLETED


class AnswerDeltaPayload(EventPayload):
    """``answer.delta``：最终正文增量；draft、replace 保留用于读取已有事件。"""

    answer_id: str
    index: int = Field(default=0, ge=0)
    text: str
    draft: bool = False
    replace: bool = False


class UsagePayload(ContractModel):
    """用量（PRD 9.3：输入、输出、推理 token 与缓存命中）。"""

    prompt_tokens: int = Field(default=0, ge=0)
    recorded_calls: int = Field(default=0, ge=0)
    missing_calls: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    prompt_cache_hit_tokens: int = Field(default=0, ge=0)
    prompt_cache_miss_tokens: int = Field(default=0, ge=0)


class RunCompletedPayload(EventPayload):
    """``run.completed``：最终答案 ID、完整性和用量。"""

    answer_id: str
    status: RunStatus = RunStatus.COMPLETED
    content_hash: str = ""
    usage: UsagePayload = Field(default_factory=UsagePayload)
    gaps: list[str] = Field(default_factory=list)


class RunFailedPayload(EventPayload):
    """``run.failed``：错误状态（PRD 12.5 错误对象）。"""

    error: AgentError


class RunCancelledPayload(EventPayload):
    """``run.cancelled``：停止状态（PRD 12.4）。"""

    stop_reason: str
    resumable: bool = False


class MemoryUpdatedPayload(EventPayload):
    """``memory.updated``：后台记忆维护结果。"""

    revision: int = Field(default=0, ge=0)
    invalidation_revision: int = Field(default=0, ge=0)
    entries_added: int = Field(default=0, ge=0)
    entries_updated: int = Field(default=0, ge=0)


class MemoryFailedPayload(EventPayload):
    """``memory.failed``：记忆维护失败（答案本身仍保留，PRD 11.5）。"""

    reason: str
    retryable: bool = True


#: 每种事件均有对应的 payload 模型。
EVENT_PAYLOAD_MODELS: Final[dict[EventType, type[EventPayload]]] = {
    EventType.MODEL_REASONING: ModelReasoningPayload,
    EventType.RUN_STATUS: RunStatusPayload,
    EventType.ROUTE_DECIDED: RouteDecidedPayload,
    EventType.PLAN_UPDATED: PlanUpdatedPayload,
    EventType.AGENT_STARTED: AgentStartedPayload,
    EventType.AGENT_COMPLETED: AgentCompletedPayload,
    EventType.TOOL_STARTED: ToolStartedPayload,
    EventType.TOOL_COMPLETED: ToolCompletedPayload,
    EventType.APPROVAL_REQUESTED: ApprovalRequestedPayload,
    EventType.APPROVAL_RESOLVED: ApprovalResolvedPayload,
    EventType.INPUT_REQUESTED: InputRequestedPayload,
    EventType.INPUT_RESOLVED: InputResolvedPayload,
    EventType.EVIDENCE_ADDED: EvidenceAddedPayload,
    EventType.VERIFICATION_UPDATED: VerificationUpdatedPayload,
    EventType.ANSWER_META: AnswerMetaPayload,
    EventType.ANSWER_DELTA: AnswerDeltaPayload,
    EventType.RUN_COMPLETED: RunCompletedPayload,
    EventType.RUN_FAILED: RunFailedPayload,
    EventType.RUN_CANCELLED: RunCancelledPayload,
    EventType.MEMORY_UPDATED: MemoryUpdatedPayload,
    EventType.MEMORY_FAILED: MemoryFailedPayload,
}


def _normalize_event_type(event_type: EventType | str) -> EventType:
    """把字符串类型收敛为 :class:`EventType`；未知类型抛协议错误而不是裸 ``ValueError``。

    PRD 12.5 末段与 13.3 末段要求「未知版本或负载校验失败」抛**明确异常**，因此
    枚举查找失败必须转成 :class:`EventProtocolError`（保留原异常为 ``__cause__``
    便于诊断；其内容不得回显到界面，见 PRD 12.5）。
    """

    if isinstance(event_type, EventType):
        return event_type
    try:
        return EventType(event_type)
    except ValueError as exc:
        raise EventProtocolError(f"未知事件类型: {event_type!r}") from exc


def payload_model_for(event_type: EventType | str) -> type[EventPayload]:
    """取某个事件类型对应的 payload 模型；未知类型抛 :class:`EventProtocolError`。"""

    return EVENT_PAYLOAD_MODELS[_normalize_event_type(event_type)]


# =====================================================================
# 统一事件对象
# =====================================================================


class EventEnvelope(ContractModel):
    """统一事件对象（PRD 13.3）：SSE ``data`` 即本模型的 JSON。

    ``payload`` 在传输层保持普通 dict（便于入库与逐字段兼容）；校验后的强类型
    负载由 :func:`parse_event` 归一化写入，形状始终与契约一致。
    """

    schema_version: int = EVENT_SCHEMA_VERSION
    run_id: str
    event_seq: int = Field(ge=1)
    execution_id: str = ""
    task_id: str | None = None
    type: EventType
    created_at: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class StoredEvent(EventEnvelope):
    """``run_events`` 行 = 事件对象 + 对象 ID 与版本（PRD 13.1 表字段）。"""

    object_id: str | None = None
    version: int | None = None


def parse_event(data: EventEnvelope | StoredEvent | Mapping[str, Any]) -> StoredEvent:
    """校验并归一化一个事件（信封 + payload），失败抛 :class:`EventProtocolError`。

    - ``schema_version`` 不等于 2 → 协议错误（未知事件版本，PRD 13.3 末段）。
    - ``type`` 不在 :class:`EventType` 内 → 协议错误。
    - payload 通不过对应模型校验 → 协议错误（不降级、不补默认值）。
    """

    raw = data if isinstance(data, Mapping) else dump_contract(data)
    try:
        event = StoredEvent.model_validate(raw)
    except ValidationError as exc:
        raise EventProtocolError(f"事件对象校验失败: {exc}") from exc

    if event.schema_version != EVENT_SCHEMA_VERSION:
        raise EventProtocolError(
            f"未知事件版本 schema_version={event.schema_version}，"
            f"本版本只支持 {EVENT_SCHEMA_VERSION}"
        )

    model = EVENT_PAYLOAD_MODELS.get(event.type)
    if model is None:  # pragma: no cover - 契约测试守护 20 个类型齐全
        raise EventProtocolError(f"事件类型缺少 payload 模型: {event.type}")
    try:
        payload = model.model_validate(event.payload)
    except ValidationError as exc:
        raise EventProtocolError(f"事件 {event.type.value} 的 payload 校验失败: {exc}") from exc

    return StoredEvent(
        schema_version=event.schema_version,
        run_id=event.run_id,
        event_seq=event.event_seq,
        execution_id=event.execution_id,
        task_id=event.task_id,
        type=event.type,
        created_at=event.created_at,
        payload=payload.model_dump(mode="json"),
        object_id=event.object_id,
        version=event.version,
    )


def build_event(
    *,
    run_id: str,
    event_seq: int,
    type: EventType | str,
    payload: EventPayload | Mapping[str, Any],
    execution_id: str = "",
    task_id: str | None = None,
    created_at: str = "",
    object_id: str | None = None,
    version: int | None = None,
) -> StoredEvent:
    """构造并校验一个事件（发布器与 RunManager 复用，避免各自手搓 dict）。"""

    body: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "run_id": run_id,
        "event_seq": event_seq,
        "execution_id": execution_id,
        "task_id": task_id,
        "type": _normalize_event_type(type),
        "created_at": created_at,
        "payload": (
            payload if isinstance(payload, Mapping) else payload.model_dump(mode="json")
        ),
        "object_id": object_id,
        "version": version,
    }
    return parse_event(body)


def dumps_event(event: EventEnvelope) -> str:
    """事件 JSON 文本（入库与 ``data`` 用；SSE 帧拼装归 approval-privacy）。"""

    return json.dumps(dump_contract(event), ensure_ascii=False, separators=(",", ":"))


def loads_event(raw: str | bytes | Mapping[str, Any]) -> StoredEvent:
    """从 JSON 文本或映射解析并校验事件；JSON 不合法同样抛协议错误。"""

    if isinstance(raw, Mapping):
        return parse_event(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventProtocolError(f"事件 JSON 解析失败: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise EventProtocolError("事件 JSON 必须是对象")
    return parse_event(decoded)
