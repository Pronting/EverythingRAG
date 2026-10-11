"""Provider 能力适配层（PRD 9.1 / 9.2 / 18.1）。Owner: T03（model-provider）。

``ModelCapabilities`` 的**字段定义**在 :mod:`app.agent.contracts`（contracts-db 拥有，
本模块只读）。本模块的职责是「适配与校验」：

- 按 provider 配置（base_url + model）解析出该模型的能力声明；
- 把用户选择的思考档位翻译为 DeepSeek 原生请求参数
  （``thinking.type`` + ``reasoning_effort``，PRD 9.1，默认 ``off``）；
- Provider 不支持所选功能时抛 :class:`CapabilityConfigurationError`
  （明确配置错误），**绝不静默降级**（PRD 9.2）；
- 未知模型不给猜测值，而是抛错并告诉用户如何显式声明能力
  （``EVERYTHING_RAG_CHAT_CAPABILITIES``）。

能力取值来源（2026-09-19 实测官方文档，见模块末 ``CAPABILITY_SOURCES``）：

- 模型表：``deepseek-flash`` / ``deepseek-v4-pro``，上下文 1M，最大输出 384K，
  支持工具调用、（flash 额外支持识图）。
- 思考模式：``thinking.type=enabled|disabled``，``reasoning_effort=low|high|max``，
  官方端点默认开启且默认 ``high``；本产品的**默认档位是 ``off``**（PRD 18.1，用户裁定），
  即默认下发 ``thinking.type=disabled``；OpenAI SDK 通过 ``extra_body`` 传 ``thinking``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Final

from app.agent.contracts import (
    DEFAULT_REASONING_LEVEL,
    ModelCapabilities,
    ReasoningLevel,
)

__all__ = [
    "CAPABILITY_ENV_VAR",
    "CAPABILITY_SOURCES",
    "DEFAULT_HISTORY_PROTOCOL",
    "HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS",
    "CapabilityConfigurationError",
    "ProviderReasoningParams",
    "ReasoningResolution",
    "apply_reasoning_params",
    "declared_capabilities_from_env",
    "describe_capabilities",
    "resolve_capabilities",
    "resolve_reasoning_params",
]

#: 用户为自定义 OpenAI 兼容端点显式声明能力的 JSON 环境变量。
CAPABILITY_ENV_VAR: Final[str] = "EVERYTHING_RAG_CHAT_CAPABILITIES"

#: 带 tools 时 reasoning_content 全量回传的协议标识（PRD 9.2 末段）。
HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS: Final[str] = "deepseek-chat-completions-reasoning-replay"

#: 其他 OpenAI 兼容端点的保守协议标识：只回传 content / tool_calls / tool 结果。
DEFAULT_HISTORY_PROTOCOL: Final[str] = "openai-chat-completions"

#: 能力取值来源（写进证据，便于人工复核；不含任何密钥与业务数据）。
CAPABILITY_SOURCES: Final[dict[str, str]] = {
    "deepseek_models": "https://api-docs.deepseek.com/quick_start/pricing",
    "deepseek_thinking": "https://api-docs.deepseek.com/guides/thinking_mode/",
    "deepseek_chat_completions": "https://api-docs.deepseek.com/api/create-chat-completion/",
    "deepseek_tool_calls": "https://api-docs.deepseek.com/guides/tool_calls/",
    "deepseek_tokenizer": (
        "https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/tokenizer.md"
    ),
}

#: 官方 DeepSeek 端点（能力表只对官方模型生效，避免把官方参数套到别的兼容端点）。
_OFFICIAL_DEEPSEEK_HOSTS: Final[frozenset[str]] = frozenset({"api.deepseek.com"})

#: ``deepseek-flash`` 能力：1M 上下文、384K 最大输出、工具、思考、识图。
_DEEPSEEK_FLASH = ModelCapabilities(
    supports_tools=True,
    supports_thinking_toggle=True,
    reasoning_levels=[
        ReasoningLevel.OFF,
        ReasoningLevel.LOW,
        ReasoningLevel.HIGH,
        ReasoningLevel.MAX,
    ],
    supports_image_input=True,
    context_limit=1_048_576,
    max_output_tokens=393_216,
    history_protocol=HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS,
)

#: ``deepseek-v4-pro`` 能力：同上下文/输出上限，官方标注不支持识图。
_DEEPSEEK_V4_PRO = ModelCapabilities(
    supports_tools=True,
    supports_thinking_toggle=True,
    reasoning_levels=[
        ReasoningLevel.OFF,
        ReasoningLevel.LOW,
        ReasoningLevel.HIGH,
        ReasoningLevel.MAX,
    ],
    supports_image_input=False,
    context_limit=1_048_576,
    max_output_tokens=393_216,
    history_protocol=HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS,
)

#: 已退役但仍被接受的历史名称 -> ``deepseek-flash``（官方计价页明确说明）。
_DEEPSEEK_ALIASES: Final[dict[str, str]] = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}

_COMMANDCODE_DEEPSEEK_FLASH_MODEL: Final[str] = "deepseek/deepseek-v4.1-flash"

DEEPSEEK_MODEL_CAPABILITIES: Final[dict[str, ModelCapabilities]] = {
    "deepseek-flash": _DEEPSEEK_FLASH,
    "deepseek-v4-pro": _DEEPSEEK_V4_PRO,
}

#: 思考档位 -> DeepSeek 原生请求参数（PRD 9.1 档位表逐字对齐）。
_REASONING_PARAMS: Final[dict[ReasoningLevel, dict[str, Any] | None]] = {
    ReasoningLevel.OFF: {"thinking": {"type": "disabled"}, "reasoning_effort": None},
    ReasoningLevel.LOW: {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    ReasoningLevel.HIGH: {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    ReasoningLevel.MAX: {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
}


class CapabilityConfigurationError(ValueError):
    """Provider 能力配置错误：不支持的能力组合 / 未声明的模型（明确、不静默降级）。"""


@dataclass(frozen=True)
class ProviderReasoningParams:
    """一次请求的思考参数（已按能力校验，可直接落请求体）。"""

    level: ReasoningLevel
    thinking_type: str | None
    reasoning_effort: str | None

    @property
    def thinking_enabled(self) -> bool:
        return self.thinking_type == "enabled"

    def extra_body(self) -> dict[str, Any] | None:
        """OpenAI SDK 的 ``extra_body``（``thinking`` 必须走这里，PRD 9.1）。"""
        if self.thinking_type is None:
            return None
        return {"thinking": {"type": self.thinking_type}}

    def request_params(self) -> dict[str, Any]:
        """请求体片段：``reasoning_effort`` 顶层 + ``extra_body.thinking``。

        ``off`` 档位省略 ``reasoning_effort``（PRD 9.1 表）。
        """
        params: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        body = self.extra_body()
        if body is not None:
            params["extra_body"] = body
        return params


@dataclass(frozen=True)
class ReasoningResolution:
    """能力解析结果：能力声明 + 本次请求的思考参数。"""

    capabilities: ModelCapabilities
    params: ProviderReasoningParams

    @property
    def supports_thinking(self) -> bool:
        return self.params.thinking_enabled


def declared_capabilities_from_env(env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """读取 ``EVERYTHING_RAG_CHAT_CAPABILITIES``（JSON 对象）；未设置返回 None。

    内容为 ``ModelCapabilities`` 字段子集（``model`` 可选，仅用于消息展示）。
    格式错误抛明确配置错误，不猜测。
    """
    source = os.environ if env is None else env
    raw = source.get(CAPABILITY_ENV_VAR)
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CapabilityConfigurationError(
            f"{CAPABILITY_ENV_VAR} 不是合法 JSON: {exc.msg}（行 {exc.lineno} 列 {exc.colno}）"
        ) from exc
    if not isinstance(parsed, dict):
        raise CapabilityConfigurationError(
            f"{CAPABILITY_ENV_VAR} 必须是 JSON 对象，例如 "
            '{"context_limit": 65536, "max_output_tokens": 8192}'
        )
    return parsed


def resolve_capabilities(
    model: str,
    base_url: str | None = None,
    *,
    declared: dict[str, Any] | None = None,
) -> ModelCapabilities:
    """按 provider 配置解析能力；未知模型抛明确配置错误（不崩溃、不猜）。

    优先级：显式 ``declared`` > ``EVERYTHING_RAG_CHAT_CAPABILITIES`` > 官方模型表。
    官方模型表只对 ``api.deepseek.com`` 生效，避免把 DeepSeek 专用参数
    套到其他 OpenAI 兼容端点。
    """
    if not model or not model.strip():
        raise CapabilityConfigurationError("对话模型名为空，无法解析 Provider 能力")

    declaration = declared if declared is not None else declared_capabilities_from_env()
    if declaration is not None:
        return _capabilities_from_declaration(model, declaration)

    key = model.strip().lower()
    if key == _COMMANDCODE_DEEPSEEK_FLASH_MODEL and _host_of(base_url) == "api.commandcode.ai":
        return _DEEPSEEK_FLASH.model_copy(update={"context_limit": 1_000_000})
    alias = _DEEPSEEK_ALIASES.get(key)
    if alias is not None and _is_official_deepseek(base_url):
        key = alias
    known = DEEPSEEK_MODEL_CAPABILITIES.get(key)
    if known is not None and _is_official_deepseek(base_url):
        return known.model_copy()

    host = _host_of(base_url)
    raise CapabilityConfigurationError(
        f"未知 Provider 能力：model={model!r} base_url host={host or '-'}。"
        "本版只内置官方 DeepSeek 模型（deepseek-flash / deepseek-v4-pro）。"
        f"其他 OpenAI 兼容端点必须显式声明能力：设置环境变量 {CAPABILITY_ENV_VAR}，"
        '例如 {"supports_tools": true, "context_limit": 65536, "max_output_tokens": 8192}。'
        "不声明时不猜测能力，也不静默降级。"
    )


def resolve_reasoning_params(
    capabilities: ModelCapabilities,
    level: ReasoningLevel | str | None = None,
    *,
    model: str = "",
) -> ProviderReasoningParams:
    """把思考档位解析为 DeepSeek 参数；能力缺失时抛明确配置错误。

    - ``level=None`` 用 PRD 18.1 默认值 ``off``（关闭思考）；
    - 非 DeepSeek 适配（``supports_thinking_toggle=False``）只允许 ``off``，
      选择 ``low/high/max`` 抛错（PRD 9.2 不得静默降级）；
    - 不支持 ``off`` 之外的档位但 capability 未声明 -> 同样抛错。
    """
    resolved = _coerce_level(level)
    if not capabilities.supports_thinking_toggle:
        if resolved is ReasoningLevel.OFF:
            return ProviderReasoningParams(level=resolved, thinking_type=None, reasoning_effort=None)
        label = f"model={model!r} " if model else ""
        raise CapabilityConfigurationError(
            f"{label}Provider 不支持思考功能（supports_thinking_toggle=False），"
            f"不能选择 reasoning_level={resolved.value!r}；请改用 off 或更换已声明能力的模型。"
        )
    if not capabilities.supports_level(resolved):
        levels = ", ".join(item.value for item in capabilities.reasoning_levels) or "（空）"
        label = f"model={model!r} " if model else ""
        raise CapabilityConfigurationError(
            f"{label}Provider 未声明思考档位 {resolved.value!r}；可用档位: {levels}。"
        )
    params = _REASONING_PARAMS[resolved]
    return ProviderReasoningParams(
        level=resolved,
        thinking_type=params["thinking"]["type"],  # type: ignore[index]
        reasoning_effort=params["reasoning_effort"],
    )


def apply_reasoning_params(
    request: dict[str, Any],
    capabilities: ModelCapabilities,
    level: ReasoningLevel | str | None = None,
    *,
    model: str = "",
) -> dict[str, Any]:
    """就地合并思考参数到请求体（含 ``extra_body.thinking``），返回同一 dict。"""
    params = resolve_reasoning_params(capabilities, level, model=model)
    request.update(params.request_params())
    return request


def describe_capabilities(capabilities: ModelCapabilities) -> dict[str, Any]:
    """能力位取值表（供运行层 / 前端只显示可用选项，PRD 9.2）。"""
    return {
        "supports_tools": capabilities.supports_tools,
        "supports_thinking_toggle": capabilities.supports_thinking_toggle,
        "reasoning_levels": [item.value for item in capabilities.reasoning_levels],
        "supports_image_input": capabilities.supports_image_input,
        "context_limit": capabilities.context_limit,
        "max_output_tokens": capabilities.max_output_tokens,
        "history_protocol": capabilities.history_protocol,
    }


def _capabilities_from_declaration(model: str, declaration: dict[str, Any]) -> ModelCapabilities:
    """把用户声明（JSON 对象）构造成 ``ModelCapabilities``，字段错误给明确提示。"""
    payload = {key: value for key, value in declaration.items() if key != "model"}
    unknown = sorted(set(payload) - set(ModelCapabilities.model_fields))
    if unknown:
        raise CapabilityConfigurationError(
            f"{CAPABILITY_ENV_VAR} 含未知字段: {', '.join(unknown)}；"
            f"可用字段: {', '.join(sorted(ModelCapabilities.model_fields))}"
        )
    missing = [
        field
        for field in ("context_limit", "max_output_tokens")
        if payload.get(field) is None
    ]
    if missing:
        raise CapabilityConfigurationError(
            f"{CAPABILITY_ENV_VAR} 缺少必填能力字段: {', '.join(missing)}（model={model!r}）"
        )
    if not payload.get("history_protocol"):
        payload["history_protocol"] = DEFAULT_HISTORY_PROTOCOL
    try:
        return ModelCapabilities(**payload)
    except ValueError as exc:  # pydantic ValidationError 也是 ValueError
        raise CapabilityConfigurationError(
            f"{CAPABILITY_ENV_VAR} 声明的能力不合法（model={model!r}）: {exc}"
        ) from exc


def _coerce_level(level: ReasoningLevel | str | None) -> ReasoningLevel:
    """档位归一化；非法取值抛明确配置错误（不静默回退默认值）。"""
    if level is None:
        return DEFAULT_REASONING_LEVEL
    if isinstance(level, ReasoningLevel):
        return level
    try:
        return ReasoningLevel(str(level))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ReasoningLevel)
        raise CapabilityConfigurationError(
            f"未知思考档位 {level!r}；可用档位: {allowed}"
        ) from exc


def _is_official_deepseek(base_url: str | None) -> bool:
    """base_url 是否为官方 DeepSeek 端点（缺省 base_url 视为官方，便于旧配置）。"""
    host = _host_of(base_url)
    return host is None or host in _OFFICIAL_DEEPSEEK_HOSTS


def _host_of(base_url: str | None) -> str | None:
    if not base_url:
        return None
    from urllib.parse import urlsplit

    host = urlsplit(base_url).hostname
    return host.lower() if host else None
