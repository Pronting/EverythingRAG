"""设置接口：GET /api/settings（读取）+ PUT /api/settings（保存，dashboard 设置页）。

- 读取返回脱敏视图：API key 只回显 api_key_set + 尾4位 hint，绝不回显明文。
- 保存为局部更新：请求中省略/为 null 的字段保持原值；api_key 传空串表示清除。
- 校验：chat base_url 需 http(s)；embed mode=cloud 时 base_url/model 必填。
- 落盘：data_dir/config.json（gitignored，明文仅存本地）。
"""

from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, SecretStr, model_validator

from app.core.settings_store import (
    AppSettings,
    ChatModelConfig,
    EmbedModelConfig,
    SettingsStore,
    get_settings_store,
)

router = APIRouter()

_BASE_URL_RE = re.compile(r"^https?://\S+$")


class ChatSettingsUpdate(BaseModel):
    provider_type: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None  # None=不改；""=清除；值=设置


class EmbedSettingsUpdate(BaseModel):
    mode: Literal["local", "cloud"] | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None


class SettingsUpdate(BaseModel):
    chat: ChatSettingsUpdate | None = None
    embed: EmbedSettingsUpdate | None = None
    system_prompt: str | None = Field(default=None, max_length=8000)

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
    return AppSettings(
        chat=chat,
        embed=embed,
        system_prompt=update.system_prompt if update.system_prompt is not None else current.system_prompt,
    )


def _overlay_chat(current: ChatModelConfig, update: ChatSettingsUpdate) -> ChatModelConfig:
    data = {
        "provider_type": update.provider_type if update.provider_type is not None else current.provider_type,
        "base_url": update.base_url if update.base_url is not None else current.base_url,
        "model": update.model if update.model is not None else current.model,
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


def _settings_view(settings: AppSettings) -> dict:
    """脱敏视图：key 只回显是否已设置 + 尾4位。"""
    return {
        "chat": _chat_view(settings.chat),
        "embed": _embed_view(settings.embed),
        "system_prompt": settings.system_prompt,
    }


def _chat_view(chat: ChatModelConfig) -> dict:
    key = chat.api_key.get_secret_value() if chat.api_key is not None else None
    return {
        "provider_type": chat.provider_type,
        "base_url": chat.base_url,
        "model": chat.model,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
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
