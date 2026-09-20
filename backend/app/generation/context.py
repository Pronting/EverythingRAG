"""上下文预算、输入顺序与压缩安排（PRD 9.3 / 18.1）。Owner: T03（model-provider）。

本模块提供**纯函数**判定接口给运行层调用（PRD 9.3）：

- 提交条件：``local_prompt_tokens + reserved_output_tokens + protocol_margin
  <= provider_context_limit``；
- 整理阈值：可用输入空间达到 80%（``DEFAULT_CONTEXT_COMPACTION_RATIO``）时安排
  上下文整理，且必须等本组工具结果完整返回（``tool_round_complete``）后才动手；
- 压缩只增加该 Agent 的 ``agent_context_epoch``，不改变会话级 ``context_epoch``
  （PRD 9.3），完整历史与原文留在本地供 Agent 按需读取；
- 输入顺序固定为：稳定职责提示 → 稳定工具定义 → 工作区与会话固定快照 →
  当前上下文段历史 → 最新问题或工具结果；动态时间/检索结果/批准结果追加在动态部分。

token 计数一律来自 :mod:`app.generation.tokenizer`（官方 tokenizer + 固定
SHA-256）；本模块不做任何估算，计数不可得时由 tokenizer 层抛明确错误。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol

from app.agent.contracts import (
    DEFAULT_CONTEXT_COMPACTION_RATIO,
    DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS,
    ModelCapabilities,
)
from app.generation.tokenizer import (
    PromptCounter,
    RequestTokenReport,
    TokenizerResourceError,
)

__all__ = [
    "DEFAULT_COMPACTION_RATIO",
    "DEFAULT_PROTOCOL_MARGIN_TOKENS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "AgentContext",
    "BudgetStatus",
    "ContextBudget",
    "ContextSection",
    "ContextSnapshot",
    "TokenCounter",
    "budget_for_request",
    "build_context",
    "compact_context",
    "count_prompt_tokens",
    "count_reasoning_messages",
    "default_reserved_output_tokens",
    "estimated_protocol_margin",
    "evaluate_context_budget",
    "should_compact",
]

#: 整理阈值：可用输入空间的 80%（PRD 18.1）。
DEFAULT_COMPACTION_RATIO: Final[float] = DEFAULT_CONTEXT_COMPACTION_RATIO

#: 预留给输出（含推理与最终正文）的 token 数缺省值；有 Provider 能力时取
#: ``min(max_output_tokens, ...)`` 由调用方决定，这里只给保守缺省。
DEFAULT_RESERVED_OUTPUT_TOKENS: Final[int] = 65_536

#: 协议余量缺省值（官方消息模板、特殊 token、工具结果包装等固定开销）。
DEFAULT_PROTOCOL_MARGIN_TOKENS: Final[int] = 2_048


class BudgetStatus(StrEnum):
    """预算判定结果（PRD 9.3）。"""

    OK = "ok"  # 可直接提交
    COMPACT = "compact"  # 可提交，但应立即安排上下文整理（≥80% 可用输入空间）
    OVER_LIMIT = "over_limit"  # 超出上限，不得提交


class TokenCounter(Protocol):
    """token 计数接口（实现见 :class:`app.generation.tokenizer.PromptCounter`）。"""

    def count(self, request: Mapping[str, Any]) -> int:
        ...


class ContextSection(StrEnum):
    """固定输入顺序（PRD 9.3）。"""

    DUTY_PROMPT = "duty_prompt"  # 稳定职责提示
    TOOL_DEFINITIONS = "tool_definitions"  # 稳定工具定义
    SNAPSHOT = "snapshot"  # 工作区与会话固定快照
    HISTORY = "history"  # 当前上下文段的历史
    LATEST = "latest"  # 最新问题或工具结果
    DYNAMIC = "dynamic"  # 动态时间、检索结果、批准结果


class ContextSnapshot:
    """工作区与会话固定快照（PRD 9.3 / 11.3）。

    固定快照保证对应文本稳定（服务端缓存命中条件），因此本对象构造后为不可变；
    超长快照按 ``max_chars`` 截断并在尾部标注，不静默丢弃。
    """

    __slots__ = ("_max_chars", "_text", "truncated")

    def __init__(self, text: str = "", *, max_chars: int = DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正数")
        self._max_chars = max_chars
        self._text = text or ""
        self.truncated = len(self._text) > max_chars

    @property
    def text(self) -> str:
        if not self.truncated:
            return self._text
        return self._text[: self._max_chars] + "\n…（快照截断，完整内容可用工具读取）"

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def __bool__(self) -> bool:
        return bool(self._text)


class ContextBudget:
    """一次请求的上下文预算判定（PRD 9.3 提交条件 + 80% 整理阈值）。"""

    def __init__(
        self,
        capabilities: ModelCapabilities,
        *,
        local_prompt_tokens: int,
        reserved_output_tokens: int,
        protocol_margin: int = DEFAULT_PROTOCOL_MARGIN_TOKENS,
        compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
    ) -> None:
        if local_prompt_tokens < 0:
            raise ValueError("local_prompt_tokens 不能为负")
        if reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens 不能为负")
        if protocol_margin < 0:
            raise ValueError("protocol_margin 不能为负")
        if not 0.0 < compaction_ratio <= 1.0:
            raise ValueError("compaction_ratio 必须落在 (0, 1]")
        self.capabilities = capabilities
        self.local_prompt_tokens = local_prompt_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.protocol_margin = protocol_margin
        self.compaction_ratio = compaction_ratio

    @property
    def provider_context_limit(self) -> int:
        return self.capabilities.context_limit

    @property
    def required_tokens(self) -> int:
        """提交所需总量：prompt + 预留输出 + 协议余量。"""
        return self.local_prompt_tokens + self.reserved_output_tokens + self.protocol_margin

    @property
    def remaining_tokens(self) -> int:
        """剩余空间（可为负，表示超出上限）。"""
        return self.provider_context_limit - self.required_tokens

    @property
    def available_input_tokens(self) -> int:
        """可用输入空间：上下文上限减去预留输出与协议余量。"""
        return self.provider_context_limit - self.reserved_output_tokens - self.protocol_margin

    @property
    def compaction_threshold_tokens(self) -> int:
        """整理阈值：可用输入空间的 ``compaction_ratio``（默认 80%，PRD 18.1）。"""
        return int(self.available_input_tokens * self.compaction_ratio)

    @property
    def input_usage_ratio(self) -> float:
        """已占用可用输入空间比例（>1 表示超出上限）。"""
        if self.available_input_tokens <= 0:
            return float("inf")
        return self.local_prompt_tokens / self.available_input_tokens

    @property
    def fits(self) -> bool:
        """``local_prompt_tokens + reserved_output_tokens + protocol_margin <= 上限``。"""
        return self.required_tokens <= self.provider_context_limit

    @property
    def needs_compaction(self) -> bool:
        """已达 80% 可用输入空间 -> 安排整理（PRD 9.3 / 18.1）。"""
        return self.local_prompt_tokens >= self.compaction_threshold_tokens

    @property
    def status(self) -> BudgetStatus:
        if not self.fits:
            return BudgetStatus.OVER_LIMIT
        if self.needs_compaction:
            return BudgetStatus.COMPACT
        return BudgetStatus.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_context_limit": self.provider_context_limit,
            "local_prompt_tokens": self.local_prompt_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "protocol_margin": self.protocol_margin,
            "required_tokens": self.required_tokens,
            "remaining_tokens": self.remaining_tokens,
            "available_input_tokens": self.available_input_tokens,
            "compaction_threshold_tokens": self.compaction_threshold_tokens,
            "input_usage_ratio": self.input_usage_ratio,
            "fits": self.fits,
            "needs_compaction": self.needs_compaction,
            "status": self.status.value,
        }


def evaluate_context_budget(
    capabilities: ModelCapabilities,
    *,
    local_prompt_tokens: int,
    reserved_output_tokens: int,
    protocol_margin: int = DEFAULT_PROTOCOL_MARGIN_TOKENS,
    compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
) -> ContextBudget:
    """纯函数入口：由计数结果与 Provider 能力给出预算判定（运行层调用）。"""
    return ContextBudget(
        capabilities,
        local_prompt_tokens=local_prompt_tokens,
        reserved_output_tokens=reserved_output_tokens,
        protocol_margin=protocol_margin,
        compaction_ratio=compaction_ratio,
    )


def budget_for_request(
    capabilities: ModelCapabilities,
    request: Mapping[str, Any],
    *,
    counter: TokenCounter | None = None,
    reserved_output_tokens: int | None = None,
    protocol_margin: int = DEFAULT_PROTOCOL_MARGIN_TOKENS,
    compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
) -> ContextBudget:
    """由完整请求 + 计数器得到预算判定；计数不可得时抛明确错误（不估算）。"""
    resolved = counter if counter is not None else PromptCounter()
    local_prompt_tokens = resolved.count(request)
    return evaluate_context_budget(
        capabilities,
        local_prompt_tokens=local_prompt_tokens,
        reserved_output_tokens=(
            reserved_output_tokens
            if reserved_output_tokens is not None
            else default_reserved_output_tokens(capabilities)
        ),
        protocol_margin=protocol_margin,
        compaction_ratio=compaction_ratio,
    )


def default_reserved_output_tokens(
    capabilities: ModelCapabilities,
    *,
    floor: int = 4_096,
) -> int:
    """预留输出 token 缺省值：不裁剪模型能力声明，只保证不小于 ``floor``。"""
    return max(floor, capabilities.max_output_tokens)


def estimated_protocol_margin(capabilities: ModelCapabilities) -> int:
    """协议余量缺省值（固定常量；能力声明异常大时不缩小，保持保守）。"""
    _ = capabilities
    return DEFAULT_PROTOCOL_MARGIN_TOKENS


def should_compact(
    budget: ContextBudget,
    *,
    tool_round_complete: bool = True,
) -> bool:
    """是否安排上下文整理（PRD 9.3）。

    ``tool_round_complete=False``（本组工具结果尚未完整返回）时一律返回 False：
    必须先等本组工具调用与结果成组返回，再增加 ``agent_context_epoch``。
    """
    if not budget.needs_compaction:
        return False
    return tool_round_complete


def build_context(
    *,
    duty_prompt: str = "",
    tool_definitions: Sequence[Mapping[str, Any]] = (),
    snapshot: ContextSnapshot | str | None = None,
    history: Sequence[Mapping[str, Any]] = (),
    latest: Sequence[Mapping[str, Any]] = (),
    dynamic: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """按 PRD 9.3 固定顺序组装模型输入（纯函数）。

    顺序：职责提示 → 工具定义（作为 ``system`` 参考块，真实 ``tools`` 字段由
    provider 单独携带）→ 固定快照 → 历史 → 最新问题/工具结果 → 动态部分。
    """
    messages: list[dict[str, Any]] = []
    prefix = duty_prompt.strip()
    if prefix:
        messages.append({"role": "system", "content": prefix})
    if tool_definitions:
        messages.append(
            {
                "role": "system",
                "content": _render_tool_reference(tool_definitions),
            }
        )
    snapshot_text = snapshot.text if isinstance(snapshot, ContextSnapshot) else (snapshot or "")
    if snapshot_text:
        messages.append({"role": "system", "content": snapshot_text})
    messages.extend(_copy(item) for item in history)
    messages.extend(_copy(item) for item in latest)
    messages.extend(_copy(item) for item in dynamic)
    return messages


def compact_context(
    context: AgentContext,
    summary: str,
) -> AgentContext:
    """压缩该 Agent 的上下文：只增加 ``agent_context_epoch``（PRD 9.3）。

    会话级 ``context_epoch`` 不变；其他 Agent 继续使用各自有效上下文。压缩结果
    是「用户目标 + 已核实结论 + 关键证据位置 + 未解决问题 + 当前权限」的摘要，
    作为参考数据进入新上下文段（不制造未发生的 assistant 工具调用）。
    """
    return context.compacted(summary)


@dataclass
class AgentContext:
    """单个 Agent 的模型上下文（PRD 9.3：会话段 + Agent 段 + 压缩历史）。"""

    context_id: str
    session_epoch: int = 1
    agent_epoch: int = 1
    duty_prompt: str = ""
    tool_definitions: tuple[dict[str, Any], ...] = ()
    snapshot: ContextSnapshot | str = field(default_factory=ContextSnapshot)
    history: tuple[dict[str, Any], ...] = ()
    summaries: tuple[str, ...] = ()
    latest: tuple[dict[str, Any], ...] = ()
    dynamic: tuple[dict[str, Any], ...] = ()
    reserved_output_tokens: int | None = None
    protocol_margin: int = DEFAULT_PROTOCOL_MARGIN_TOKENS
    compaction_ratio: float = DEFAULT_COMPACTION_RATIO

    def __post_init__(self) -> None:
        """允许直接传字符串快照：统一转为固定的 :class:`ContextSnapshot`。"""
        if isinstance(self.snapshot, str):
            self.snapshot = ContextSnapshot(self.snapshot)

    @property
    def context_epoch(self) -> int:
        """会话级上下文段（压缩单个 Agent 不改变它，PRD 9.3）。"""
        return self.session_epoch

    @property
    def agent_context_epoch(self) -> int:
        """该 Agent 的上下文段；压缩后 +1（PRD 9.3）。"""
        return self.agent_epoch

    def messages(self) -> list[dict[str, Any]]:
        """当前上下文段的消息列表（按固定顺序）。"""
        summaries = [
            {"role": "system", "content": f"[历史摘要·参考数据] {item}"}
            for item in self.summaries
            if item
        ]
        return build_context(
            duty_prompt=self.duty_prompt,
            tool_definitions=self.tool_definitions,
            snapshot=self.snapshot,
            history=(*summaries, *self.history),
            latest=self.latest,
            dynamic=self.dynamic,
        )

    def request(self, model: str, **extra: Any) -> dict[str, Any]:
        """组装可用于计数的完整 Chat Completions 请求。"""
        payload: dict[str, Any] = {"model": model, "messages": self.messages()}
        if self.tool_definitions:
            payload["tools"] = [dict(item) for item in self.tool_definitions]
        payload.update(extra)
        return payload

    def token_report(
        self,
        model: str,
        *,
        counter: TokenCounter | None = None,
        **extra: Any,
    ) -> RequestTokenReport:
        """本地计数报告（官方 tokenizer；资源缺失抛明确错误）。"""
        resolved = counter if counter is not None else PromptCounter()
        request = self.request(model, **extra)
        report = getattr(resolved, "report", None)
        if callable(report):
            return report(request)
        return RequestTokenReport(local_prompt_tokens=resolved.count(request), model=model)

    def budget(
        self,
        model: str,
        capabilities: ModelCapabilities,
        *,
        counter: TokenCounter | None = None,
        **extra: Any,
    ) -> ContextBudget:
        """当前上下文的预算判定（PRD 9.3 提交条件 + 80% 整理阈值）。"""
        return budget_for_request(
            capabilities,
            self.request(model, **extra),
            counter=counter,
            reserved_output_tokens=self.reserved_output_tokens,
            protocol_margin=self.protocol_margin,
            compaction_ratio=self.compaction_ratio,
        )

    def compacted(self, summary: str) -> AgentContext:
        """返回压缩后的新上下文（不可变风格；``context_epoch`` 不变）。"""
        if not isinstance(summary, str):
            raise TypeError("压缩摘要必须是字符串")
        return AgentContext(
            context_id=self.context_id,
            session_epoch=self.session_epoch,
            agent_epoch=self.agent_epoch + 1,
            duty_prompt=self.duty_prompt,
            tool_definitions=self.tool_definitions,
            snapshot=self.snapshot,
            history=(),
            summaries=(*self.summaries, summary),
            latest=self.latest,
            dynamic=self.dynamic,
            reserved_output_tokens=self.reserved_output_tokens,
            protocol_margin=self.protocol_margin,
            compaction_ratio=self.compaction_ratio,
        )


#: 计数用的模型名占位；实际请求由运行层传入真实模型名覆盖。
_MODEL_PLACEHOLDER: Final[str] = ""

def count_prompt_tokens(request: Mapping[str, Any]) -> int:
    """完整 Chat Completions 请求的本地 token 数（PRD 9.3；资源缺失抛明确错误）。

    实现在 :mod:`app.generation.tokenizer`；此处提供运行层使用的同一入口。
    """
    from app.generation.tokenizer import count_prompt_tokens as _count

    return _count(request)


def _render_tool_reference(tool_definitions: Sequence[Mapping[str, Any]]) -> str:
    """把稳定工具定义渲染为 system 参考块（工具本身仍走 ``tools`` 字段）。"""
    lines = ["[可用工具定义·稳定块]"]
    for item in tool_definitions:
        function = item.get("function") or {}
        name = function.get("name") or item.get("name") or "unknown"
        description = function.get("description") or ""
        lines.append(f"- {name}: {description}".rstrip())
    return "\n".join(lines)


def _copy(message: Mapping[str, Any]) -> dict[str, Any]:
    return dict(message)


def count_reasoning_messages(messages: Sequence[Mapping[str, Any]]) -> int:
    """需要保留推理内容的消息条数（PRD 9.3：计数需含要求保留的推理内容）。"""
    from app.generation.tokenizer import count_reasoning_messages as _count

    return _count(messages)


_ = TokenizerResourceError  # 计数资源缺失时的明确错误类型（调用方按此捕获）
