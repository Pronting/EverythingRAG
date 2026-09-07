"""设置接口：GET /api/settings（读取）+ PUT /api/settings（保存，dashboard 设置页）。

- 读取返回脱敏视图：API key 只回显 api_key_set + 尾4位 hint，绝不回显明文。
- 保存为局部更新：请求中省略/为 null 的字段保持原值；api_key 传空串表示清除。
- 校验：chat base_url 需 http(s)；embed mode=cloud 时 base_url/model 必填。
- Agent 核心策略由后端管理；用户可保存表达偏好及附加自定义要求，不能替换核心 system prompt。
- 落盘：data_dir/config.json（gitignored，明文仅存本地）。
"""

from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.core.settings_store import (
    AppSettings,
    ChatModelConfig,
    EmbedModelConfig,
    SearchConfig,
    SettingsStore,
    VisionModelConfig,
    get_settings_store,
)
from app.generation.prompt_policy import POLICY_VERSION, AnswerPreferences

router = APIRouter()

_BASE_URL_RE = re.compile(r"^https?://\S+$")


class ChatSettingsUpdate(BaseModel):
    provider_type: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None  # None=不改；""=清除；值=设置
    supports_image: bool | None = None  # 是否多模态（直接收图片输入）


class EmbedSettingsUpdate(BaseModel):
    mode: Literal["cloud"] | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None


class SearchSettingsUpdate(BaseModel):
    provider_type: Literal["tavily", "searxng"] | None = None
    enabled: bool | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    max_results: int | None = Field(default=None, ge=1, le=10)


class VisionSettingsUpdate(BaseModel):
    provider_type: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None  # None=不改；""=清除；值=设置


class AnswerPreferencesUpdate(BaseModel):
    """回答偏好及附加自定义要求；省略保留，空字符串清除自定义要求。"""

    model_config = ConfigDict(extra="forbid")

    language: Literal["auto", "zh-CN", "en"] | None = None
    verbosity: Literal["concise", "balanced", "detailed"] | None = None
    tone: Literal["natural", "professional"] | None = None
    response_format: Literal["auto", "prose", "bullets"] | None = None
    knowledge_only: bool | None = None
    custom_instructions: str | None = Field(default=None, max_length=4000)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat: ChatSettingsUpdate | None = None
    embed: EmbedSettingsUpdate | None = None
    vision: VisionSettingsUpdate | None = None
    search: SearchSettingsUpdate | None = None
    answer_preferences: AnswerPreferencesUpdate | None = None
    theme: Literal["light", "dark"] | None = None

    @model_validator(mode="after")
    def _validate_base_urls(self) -> SettingsUpdate:
        if self.chat is not None and self.chat.base_url is not None and not _BASE_URL_RE.match(
            self.chat.base_url
        ):
            raise ValueError("对话模型 Base URL 需为 http(s) 地址")
        if self.embed is not None and self.embed.base_url is not None and not _BASE_URL_RE.match(
            self.embed.base_url
        ):
            raise ValueError("嵌入模型 Base URL 需为 http(s) 地址")
        if self.search is not None and self.search.base_url is not None and not _BASE_URL_RE.match(
            self.search.base_url
        ):
            raise ValueError("联网搜索 Base URL 需为 http(s) 地址")
        if self.vision is not None and self.vision.base_url is not None and not _BASE_URL_RE.match(
            self.vision.base_url
        ):
            raise ValueError("识图模型 Base URL 需为 http(s) 地址")
        if (
            self.embed is not None
            and self.embed.mode == "cloud"
            and (not self.embed.base_url or not self.embed.model)
        ):
            raise ValueError("云端嵌入需同时提供 Base URL 与模型名")
        return self


@router.get("/api/settings")
async def get_settings_view(
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """返回脱敏设置视图（key 只给 hint）。"""
    return _settings_view(store.load())


@router.put("/api/settings")
async def update_settings(
    update: SettingsUpdate,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """局部保存设置：覆盖字段更新后原子落盘，返回脱敏视图。"""
    current = store.load()
    updated = _apply(current, update)
    store.save(updated)
    return _settings_view(updated)


def _apply(current: AppSettings, update: SettingsUpdate) -> AppSettings:
    """把部分更新叠到当前设置上（api_key 空串=清除）。"""
    chat = current.chat
    if update.chat is not None:
        chat = _overlay_chat(chat, update.chat)
    embed = current.embed
    if update.embed is not None:
        embed = _overlay_embed(embed, update.embed)
    vision = current.vision
    if update.vision is not None:
        vision = _overlay_vision(vision, update.vision)
    search = current.search
    if update.search is not None:
        search = _overlay_search(search, update.search)
    preferences = current.answer_preferences
    if update.answer_preferences is not None:
        preferences = _overlay_preferences(preferences, update.answer_preferences)
    return AppSettings(
        chat=chat,
        embed=embed,
        vision=vision,
        search=search,
        answer_preferences=preferences,
        legacy_system_prompt_backup=current.legacy_system_prompt_backup,
        avatars=current.avatars,  # 保留头像（覆盖式更新不得重置）
        theme=update.theme if update.theme is not None else current.theme,
    )


def _overlay_chat(current: ChatModelConfig, update: ChatSettingsUpdate) -> ChatModelConfig:
    data = {
        "provider_type": update.provider_type if update.provider_type is not None else current.provider_type,
        "base_url": update.base_url if update.base_url is not None else current.base_url,
        "model": update.model if update.model is not None else current.model,
        "supports_image": update.supports_image if update.supports_image is not None else current.supports_image,
    }
    if update.api_key is not None:
        # 空串=清除，非空=设置
        data["api_key"] = None if update.api_key.get_secret_value() == "" else update.api_key
    else:
        data["api_key"] = current.api_key
    return ChatModelConfig(**data)


def _overlay_embed(current: EmbedModelConfig, update: EmbedSettingsUpdate) -> EmbedModelConfig:
    data = {
        "mode": update.mode if update.mode is not None else current.mode,
        "base_url": update.base_url if update.base_url is not None else current.base_url,
        "model": update.model if update.model is not None else current.model,
    }
    if update.api_key is not None:
        data["api_key"] = None if update.api_key.get_secret_value() == "" else update.api_key
    else:
        data["api_key"] = current.api_key
    return EmbedModelConfig(**data)


def _overlay_vision(current: VisionModelConfig, update: VisionSettingsUpdate) -> VisionModelConfig:
    data = {
        "provider_type": update.provider_type if update.provider_type is not None else current.provider_type,
        "base_url": update.base_url if update.base_url is not None else current.base_url,
        "model": update.model if update.model is not None else current.model,
    }
    if update.api_key is not None:
        data["api_key"] = None if update.api_key.get_secret_value() == "" else update.api_key
    else:
        data["api_key"] = current.api_key
    return VisionModelConfig(**data)


def _overlay_search(current: SearchConfig, update: SearchSettingsUpdate) -> SearchConfig:
    data = {
        "provider_type": update.provider_type if update.provider_type is not None else current.provider_type,
        "enabled": update.enabled if update.enabled is not None else current.enabled,
        "base_url": update.base_url if update.base_url is not None else current.base_url,
        "max_results": update.max_results if update.max_results is not None else current.max_results,
    }
    if update.api_key is not None:
        data["api_key"] = None if update.api_key.get_secret_value() == "" else update.api_key
    else:
        data["api_key"] = current.api_key
    return SearchConfig(**data)


def _overlay_preferences(
    current: AnswerPreferences,
    update: AnswerPreferencesUpdate,
) -> AnswerPreferences:
    return AnswerPreferences(
        language=update.language if update.language is not None else current.language,
        verbosity=update.verbosity if update.verbosity is not None else current.verbosity,
        tone=update.tone if update.tone is not None else current.tone,
        response_format=(
            update.response_format
            if update.response_format is not None
            else current.response_format
        ),
        knowledge_only=(
            update.knowledge_only
            if update.knowledge_only is not None
            else current.knowledge_only
        ),
        custom_instructions=(
            update.custom_instructions
            if update.custom_instructions is not None
            else current.custom_instructions
        ),
    )


def _settings_view(settings: AppSettings) -> dict:
    """脱敏视图：key 只回显是否已设置 + 尾4位；头像给可访问 URL 或 null。"""
    return {
        "chat": _chat_view(settings.chat),
        "embed": _embed_view(settings.embed),
        "vision": _vision_view(settings.vision),
        "search": _search_view(settings.search),
        "answer_preferences": settings.answer_preferences.model_dump(),
        "policy": {"version": POLICY_VERSION, "managed": True},
        "avatars": {
            "user": _avatar_url(settings.avatars.user),
            "agent": _avatar_url(settings.avatars.agent),
        },
        "theme": settings.theme,
    }


def _avatar_url(filename: str | None) -> str | None:
    """头像相对文件名 -> 访问 URL；未设置返回 None（前端用内置 SVG）。"""
    return f"/api/avatars/{filename}" if filename else None


def _chat_view(chat: ChatModelConfig) -> dict:
    key = chat.api_key.get_secret_value() if chat.api_key is not None else None
    return {
        "provider_type": chat.provider_type,
        "base_url": chat.base_url,
        "model": chat.model,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
        "supports_image": chat.supports_image,
    }


def _embed_view(embed: EmbedModelConfig) -> dict:
    key = embed.api_key.get_secret_value() if embed.api_key is not None else None
    return {
        "mode": embed.mode,
        "base_url": embed.base_url,
        "model": embed.model,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
    }


def _vision_view(vision: VisionModelConfig) -> dict:
    key = vision.api_key.get_secret_value() if vision.api_key is not None else None
    return {
        "provider_type": vision.provider_type,
        "base_url": vision.base_url,
        "model": vision.model,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
    }


def _search_view(search: SearchConfig) -> dict:
    key = search.api_key.get_secret_value() if search.api_key is not None else None
    return {
        "provider_type": search.provider_type,
        "enabled": search.enabled,
        "base_url": search.base_url,
        "max_results": search.max_results,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
    }


class ConnectionTestRequest(BaseModel):
    kind: Literal["chat", "embed", "vision"]
    base_url: str = Field(max_length=2048)
    model: str = Field(max_length=256)
    api_key: SecretStr | None = None


@router.post("/api/settings/test-connection")
async def test_connection(
    request: ConnectionTestRequest,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    from urllib.parse import urlsplit

    from app.generation.connection_test import probe_model

    url = request.base_url.strip().rstrip("/")
    model = request.model.strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        return {"success": False, "message": "服务地址格式不正确", "latency_ms": 0}
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not model:
        return {"success": False, "message": "请填写有效的服务地址和模型名称", "latency_ms": 0}
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return {"success": False, "message": "服务地址不能包含凭据或查询参数", "latency_ms": 0}
    config = getattr(store.load(), request.kind)
    # Omitted key means reuse the saved key, but never send it to a changed endpoint.
    key = request.api_key
    if key is None and url == (config.base_url or "").strip().rstrip("/"):
        key = config.api_key
    return await probe_model(request.kind, url, model, key.get_secret_value() if key is not None else None)
