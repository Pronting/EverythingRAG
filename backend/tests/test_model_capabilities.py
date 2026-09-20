"""Provider 能力适配层测试（PRD 9.1 / 9.2 / 18.1）。Owner: T03。

零外呼：只测纯函数（枚举、能力表、参数映射、能力校验），不构造任何真实客户端。
"""

from __future__ import annotations

import pytest

from app.agent.contracts import DEFAULT_REASONING_LEVEL, ReasoningLevel
from app.generation.capabilities import (
    CAPABILITY_ENV_VAR,
    DEFAULT_HISTORY_PROTOCOL,
    HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS,
    CapabilityConfigurationError,
    apply_reasoning_params,
    declared_capabilities_from_env,
    describe_capabilities,
    resolve_capabilities,
    resolve_reasoning_params,
)

OFFICIAL = "https://api.deepseek.com"
COMPAT = "https://compat.example.test/v1"

DECLARED = {
    "supports_tools": True,
    "supports_thinking_toggle": True,
    "reasoning_levels": ["off", "low", "high", "max"],
    "supports_image_input": False,
    "context_limit": 65536,
    "max_output_tokens": 8192,
    "history_protocol": DEFAULT_HISTORY_PROTOCOL,
}


# ---------------------------------------------------------------- 能力位取值表


def test_default_reasoning_level_is_high() -> None:
    """PRD 18.1 工程默认值：思考强度默认 high。"""
    assert DEFAULT_REASONING_LEVEL is ReasoningLevel.HIGH


@pytest.mark.parametrize(
    ("model", "image_supported"),
    [("deepseek-flash", True), ("deepseek-v4-pro", False)],
)
def test_official_models_resolve_full_capability_table(model: str, image_supported: bool) -> None:
    """官方模型解析出 PRD 9.2 全 7 个能力位（deepseek-flash / deepseek-v4-pro）。"""
    caps = resolve_capabilities(model, OFFICIAL)
    assert caps.supports_tools is True
    assert caps.supports_thinking_toggle is True
    assert [item.value for item in caps.reasoning_levels] == ["off", "low", "high", "max"]
    assert caps.supports_image_input is image_supported
    assert caps.context_limit == 1_048_576  # 官方模型表：上下文 1M
    assert caps.max_output_tokens == 393_216  # 官方模型表：最大输出 384K
    assert caps.history_protocol == HISTORY_PROTOCOL_DEEPSEEK_CHAT_COMPLETIONS


def test_describe_capabilities_lists_all_seven_fields() -> None:
    """能力位取值表逐字覆盖 PRD 9.2 的 7 个字段（供界面只显示可用选项）。"""
    table = describe_capabilities(resolve_capabilities("deepseek-flash", OFFICIAL))
    assert set(table) == {
        "supports_tools",
        "supports_thinking_toggle",
        "reasoning_levels",
        "supports_image_input",
        "context_limit",
        "max_output_tokens",
        "history_protocol",
    }
    assert table["reasoning_levels"] == ["off", "low", "high", "max"]


def test_alias_maps_to_flash_only_on_official_host() -> None:
    """已退役别名 deepseek-v4-flash 在官方端点映射到 deepseek-flash 能力。"""
    caps = resolve_capabilities("deepseek-v4-flash", OFFICIAL)
    assert caps.supports_image_input is True
    assert caps.context_limit == 1_048_576


def test_official_capabilities_not_applied_to_other_endpoints() -> None:
    """非官方端点即便模型名相同也必须显式声明能力（不套用 DeepSeek 专用参数）。"""
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_capabilities("deepseek-flash", COMPAT)
    assert CAPABILITY_ENV_VAR in str(exc.value)


def test_unknown_model_raises_actionable_configuration_error() -> None:
    """未知模型给明确配置错误（含环境变量与可用模型），不是崩溃、也不是猜测。"""
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_capabilities("some-local-llm", "http://localhost:11434/v1")
    message = str(exc.value)
    assert "some-local-llm" in message
    assert "localhost" in message
    assert CAPABILITY_ENV_VAR in message
    assert "deepseek-flash" in message


def test_empty_model_name_raises() -> None:
    with pytest.raises(CapabilityConfigurationError):
        resolve_capabilities("   ", OFFICIAL)


# ---------------------------------------------------------------- 显式声明


def test_declared_capabilities_used_for_custom_endpoint() -> None:
    """显式声明优先于内置表：自定义端点可声明自己的能力（含 history_protocol 缺省）。"""
    caps = resolve_capabilities("my-model", COMPAT, declared=dict(DECLARED))
    assert caps.context_limit == 65536
    assert caps.max_output_tokens == 8192
    assert caps.supports_level(ReasoningLevel.MAX)


def test_declaration_defaults_history_protocol_when_absent() -> None:
    """未声明 history_protocol 时给保守缺省，而不是留空导致校验失败。"""
    caps = resolve_capabilities(
        "my-model",
        COMPAT,
        declared={"supports_tools": True, "context_limit": 4096, "max_output_tokens": 1024},
    )
    assert caps.history_protocol == DEFAULT_HISTORY_PROTOCOL


def test_declaration_missing_required_fields_raises() -> None:
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_capabilities("my-model", COMPAT, declared={"supports_tools": True})
    assert "context_limit" in str(exc.value)
    assert "max_output_tokens" in str(exc.value)


def test_declaration_unknown_field_raises_with_field_list() -> None:
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_capabilities(
            "my-model", COMPAT, declared={**DECLARED, "supports_video": True}
        )
    assert "supports_video" in str(exc.value)
    assert "context_limit" in str(exc.value)


def test_declaration_rejects_inconsistent_thinking_levels() -> None:
    """声明了档位却没有思考开关 -> 契约校验失败并转成明确配置错误。"""
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_capabilities(
            "my-model",
            COMPAT,
            declared={
                **DECLARED,
                "supports_thinking_toggle": False,
                "reasoning_levels": ["off", "max"],
            },
        )
    assert "supports_thinking_toggle" in str(exc.value)


def test_declaration_from_env_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EVERYTHING_RAG_CHAT_CAPABILITIES 走 JSON 声明；非 JSON 给明确错误。"""
    monkeypatch.setenv(CAPABILITY_ENV_VAR, '{"context_limit": 8192, "max_output_tokens": 512}')
    assert declared_capabilities_from_env() == {
        "context_limit": 8192,
        "max_output_tokens": 512,
    }
    monkeypatch.setenv(CAPABILITY_ENV_VAR, "{not json")
    with pytest.raises(CapabilityConfigurationError) as exc:
        declared_capabilities_from_env()
    assert CAPABILITY_ENV_VAR in str(exc.value)


def test_declaration_from_env_must_be_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CAPABILITY_ENV_VAR, "[1, 2]")
    with pytest.raises(CapabilityConfigurationError):
        declared_capabilities_from_env()


def test_env_declaration_is_used_by_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境变量声明生效：自定义端点无需改代码即可声明能力。"""
    monkeypatch.setenv(
        CAPABILITY_ENV_VAR,
        '{"supports_tools": true, "context_limit": 32768, "max_output_tokens": 4096}',
    )
    caps = resolve_capabilities("my-model", COMPAT)
    assert caps.context_limit == 32768
    assert caps.history_protocol == DEFAULT_HISTORY_PROTOCOL


# ---------------------------------------------------------------- 四档映射


@pytest.mark.parametrize(
    ("level", "thinking_type", "effort"),
    [
        (ReasoningLevel.OFF, "disabled", None),
        (ReasoningLevel.LOW, "enabled", "low"),
        (ReasoningLevel.HIGH, "enabled", "high"),
        (ReasoningLevel.MAX, "enabled", "max"),
    ],
)
def test_four_reasoning_levels_map_to_deepseek_params(
    level: ReasoningLevel,
    thinking_type: str,
    effort: str | None,
) -> None:
    """PRD 9.1 档位表逐行核对（thinking.type + reasoning_effort）。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    params = resolve_reasoning_params(caps, level)
    assert params.thinking_type == thinking_type
    assert params.reasoning_effort == effort
    request = apply_reasoning_params({}, caps, level)
    assert request["extra_body"] == {"thinking": {"type": thinking_type}}
    if effort is None:
        assert "reasoning_effort" not in request  # off 档位省略 reasoning_effort
    else:
        assert request["reasoning_effort"] == effort


def test_reasoning_level_defaults_to_high() -> None:
    """未指定档位 -> 默认 high（PRD 18.1），而不是 off。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    params = resolve_reasoning_params(caps, None)
    assert params.level is ReasoningLevel.HIGH
    assert params.reasoning_effort == "high"


def test_reasoning_level_accepts_plain_string() -> None:
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    assert resolve_reasoning_params(caps, "max").reasoning_effort == "max"


def test_unknown_reasoning_level_raises_configuration_error() -> None:
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_reasoning_params(caps, "ultra")
    assert "ultra" in str(exc.value)
    assert "max" in str(exc.value)


def test_thinking_level_without_toggle_support_is_configuration_error() -> None:
    """Provider 不支持思考时选档位 -> 明确配置错误，不静默降级为 off。"""
    caps = resolve_capabilities(
        "plain-model",
        COMPAT,
        declared={
            "supports_tools": False,
            "supports_thinking_toggle": False,
            "supports_image_input": False,
            "context_limit": 8192,
            "max_output_tokens": 1024,
        },
    )
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_reasoning_params(caps, ReasoningLevel.HIGH)
    assert "supports_thinking_toggle" in str(exc.value)
    # off 仍可用（无思考即 off），保持可用性
    off = resolve_reasoning_params(caps, ReasoningLevel.OFF)
    assert off.thinking_type is None
    assert off.extra_body() is None


def test_undeclared_level_in_declaration_is_configuration_error() -> None:
    """只声明 off/low 的 Provider 选 high -> 明确配置错误并列出可用档位。"""
    caps = resolve_capabilities(
        "partial-model",
        COMPAT,
        declared={
            "supports_thinking_toggle": True,
            "reasoning_levels": ["off", "low"],
            "context_limit": 8192,
            "max_output_tokens": 1024,
        },
    )
    assert caps.supports_level(ReasoningLevel.LOW)
    with pytest.raises(CapabilityConfigurationError) as exc:
        resolve_reasoning_params(caps, ReasoningLevel.HIGH)
    assert "high" in str(exc.value)
    assert "low" in str(exc.value)
