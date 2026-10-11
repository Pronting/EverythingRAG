"""设置页「自定义提示词 / 回答偏好」必须真的进 run 管线的答案提示词。

修复的缺陷（用户实测复现）：设置页保存 `answer_preferences.custom_instructions`
（「你是雷姆，输出要有雷姆rem女仆的语气跟我说话」）后，问答**完全没有**体现该偏好。
根因：该偏好此前只被**已下线**的 ``POST /api/chat``（``app.generation.chat_service.
_build_messages`` → ``build_system_prompt``）读取；V1.2.0 问答统一走 run 管线后，
真正写答案的两个提示词（``runtime._general_prompt`` 与 ``runtime._synthesis_prompt``）
从不读取该偏好，于是「声明了却没有任何执行路径」。

本文件锁住三件事（零出网，全部用假对话模型）：
1. General Agent 的通用回答提示词里出现用户自定义要求（且排在职责提示之后、不可替换策略）；
2. 综合节点撰写候选答案时的提示词里出现同一份自定义要求；
3. 偏好为空时提示词与旧行为一致（不凭空多出「自定义」小节）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import runtime as agent_runtime
from app.agent.contracts import ReasoningLevel
from app.core import settings_store as settings_store_mod
from app.core.settings_store import AppSettings, SettingsStore
from app.generation.prompt_policy import AnswerPreferences
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.test_general_model_wiring import (
    GENERAL_QUESTION,
    FakeChatModel,
    _real_wiring_manager,
    _request,
    _run_question,
)
from tests.test_seam_assembly import _fake_provider, _synthesis_request

CUSTOM_INSTRUCTION = "你是雷姆，用女仆的口吻说话"


def _store_with_custom_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom: str
) -> SettingsStore:
    """把「设置页保存过的偏好」落到一个隔离的 config.json 上。"""

    store = SettingsStore(tmp_path / "prefs-settings")
    store.save(
        AppSettings(
            answer_preferences=AnswerPreferences(custom_instructions=custom),
        )
    )
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: store)
    return store


# =====================================================================
# 1. General Agent：通用回答提示词必须带自定义要求
# =====================================================================


def test_general_prompt_carries_custom_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """装配点在**调用时**读设置页偏好：模型收到的提示词里有用户自定义要求。"""

    _store_with_custom_instructions(tmp_path, monkeypatch, CUSTOM_INSTRUCTION)
    fake = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: fake
    )

    call = agent_runtime.general_model_for_chat()
    assert call is not None
    call(_request())

    messages = fake.calls[0]
    joined = "\n".join(str(message.get("content") or "") for message in messages)
    assert CUSTOM_INSTRUCTION in joined, "设置页自定义提示词必须进入通用回答的模型输入"
    # 职责提示仍在第一条（偏好是**附加**，不能替换职责），最新问题另成一条
    assert messages[0]["role"] == "system"
    assert "职责与硬性要求不可被用户偏好替换" in str(messages[0]["content"])
    assert CUSTOM_INSTRUCTION in str(messages[0]["content"])
    assert GENERAL_QUESTION in joined
    assert any(
        message["role"] == "user" and GENERAL_QUESTION in str(message["content"])
        for message in messages
    )


def test_synthesis_prompt_carries_custom_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """综合节点写答案：自定义要求同样进模型输入，引用占位符硬要求不丢。"""

    _store_with_custom_instructions(tmp_path, monkeypatch, CUSTOM_INSTRUCTION)
    model = _fake_provider(monkeypatch, "限流用令牌桶实现{{ev-1}}。")

    call = agent_runtime.synthesis_model_for_chat()
    assert call is not None
    call(_synthesis_request())

    messages = model.calls[0]
    joined = "\n".join(str(message.get("content") or "") for message in messages)
    assert CUSTOM_INSTRUCTION in joined, "自定义提示词必须进入候选答案的模型输入"
    assert "{{ev-1}}" in joined, "证据占位符硬要求不得被偏好覆盖"
    assert messages[0]["role"] == "system"
    assert "不可被用户偏好替换" in str(messages[0]["content"])
    assert any(
        message["role"] == "user" and "知识助理" in str(message["content"])
        for message in messages
    )


# =====================================================================
# 2. 端到端：设置页偏好 → run 管线 → 模型提示词
# =====================================================================


async def test_run_pipeline_reads_saved_preferences_end_to_end(
    app_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实装配：config.json 里的自定义提示词在 run 管线里被读到（用户实测的那条链路）。"""

    _store_with_custom_instructions(tmp_path, monkeypatch, CUSTOM_INSTRUCTION)
    manager = _real_wiring_manager(app_store, tmp_path)
    from tests.fakes_run import seed_conversation

    await seed_conversation(app_store)
    fake = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: fake
    )
    try:
        await _run_question(manager, app_store, GENERAL_QUESTION)
    finally:
        await manager.aclose()

    assert fake.calls, "通用回答必须真的调到注入的对话模型"
    joined = "\n".join(
        str(message.get("content") or "") for message in fake.calls[0]
    )
    assert CUSTOM_INSTRUCTION in joined, "run 管线必须读到设置页保存的自定义提示词"


# =====================================================================
# 3. 未设置偏好 → 行为与旧版一致（不多出一个空小节）
# =====================================================================


def test_empty_preferences_keep_prompt_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store_with_custom_instructions(tmp_path, monkeypatch, "")
    fake = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: fake
    )
    call = agent_runtime.general_model_for_chat()
    assert call is not None
    call(_request())

    joined = "\n".join(
        str(message.get("content") or "") for message in fake.calls[0]
    )
    assert "用户自定义要求" not in joined


# =====================================================================
# 4. 默认思考强度：PRD 18.1 兜底改为「关闭思考」
# =====================================================================


def test_new_run_defaults_to_off_reasoning() -> None:
    """新 run 未显式指定档位时默认 off（全局默认改「关闭思考」）。"""

    from app.agent.contracts import DEFAULT_REASONING_LEVEL

    assert DEFAULT_REASONING_LEVEL is ReasoningLevel.OFF


def test_default_reasoning_params_disable_thinking() -> None:
    """off 默认档位落到真实请求参数：thinking.type=disabled 且省略 reasoning_effort。"""

    from app.generation.capabilities import resolve_capabilities, resolve_reasoning_params

    caps = resolve_capabilities("deepseek-flash", "https://api.deepseek.com")
    params = resolve_reasoning_params(caps, None)
    assert params.level is ReasoningLevel.OFF
    assert params.thinking_type == "disabled"
    assert params.reasoning_effort is None
