"""设置存储：data_dir/config.json 持久化 + 与环境变量合并（设置页后端）。

优先级：config.json（dashboard 设置）> 环境变量/.env > 默认值。
- chat：base_url / model / api_key 缺省回退 env（EVERYTHING_RAG_CHAT_*）。
- embed：本地 bge-m3 默认；云端需 base_url + model + api_key。
- system_prompt：默认内置提示词，用户可在设置页覆盖。
API key 存 SecretStr：API 响应只回显「是否已设置 + 尾4位」，绝不回显明文。
config.json 落在 data_dir（gitignored，类似 .env），明文仅存本地。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from app.core.config import get_settings

#: 默认系统提示词（与 ChatService 原内置一致，用户可在设置页覆盖）
DEFAULT_SYSTEM_PROMPT = (
    "你是 Everything RAG 个人知识助手。请严格基于给定的参考上下文作答，"
    "不要编造上下文之外的内容；若上下文无法回答该问题，请如实说明。"
    "引用来源时只能使用上下文中标注的序号，不得虚构来源。"
)

CONFIG_FILENAME = "config.json"


class ChatModelConfig(BaseModel):
    """对话模型配置（dashboard 可编辑）。"""

    provider_type: str = "openai_compatible"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None


class EmbedModelConfig(BaseModel):
    """嵌入模型配置：mode=local（本地 bge-m3）或 mode=cloud（OpenAI 兼容嵌入）。"""

    mode: Literal["local", "cloud"] = "local"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None


class AppSettings(BaseModel):
    """运行时有效设置（config.json 覆盖 env 后的最终值）。"""

    chat: ChatModelConfig = Field(default_factory=ChatModelConfig)
    embed: EmbedModelConfig = Field(default_factory=EmbedModelConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class SettingsStore:
    """config.json 的读写 + env 合并。线程非安全（设置页低频写，可接受）。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / CONFIG_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> AppSettings:
        """加载有效设置：config.json 存在则**权威**（dashboard 是唯一管理入口，
        save 总是写全量解析后的值）；不存在才回退 env/.env 默认。

        注：env 只是首次运行的兜底——用户一旦在设置页保存过，config.json 即接管。
        """
        file = self._read_file()
        if file is not None:
            return file
        return self._env_defaults()

    def save(self, settings: AppSettings) -> None:
        """原子写入 config.json（先写 .tmp 再 os.replace）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chat": _chat_to_json(settings.chat),
            "embed": _embed_to_json(settings.embed),
            "system_prompt": settings.system_prompt,
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def _read_file(self) -> AppSettings | None:
        if not self._path.is_file():
            return None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return AppSettings.model_validate(raw)

    def _env_defaults(self) -> AppSettings:
        env = get_settings()
        return AppSettings(
            chat=ChatModelConfig(
                base_url=env.chat_base_url,
                model=env.chat_model,
                api_key=env.chat_api_key,
            )
        )


def _chat_to_json(chat: ChatModelConfig) -> dict:
    return {
        "provider_type": chat.provider_type,
        "base_url": chat.base_url,
        "model": chat.model,
        "api_key": chat.api_key.get_secret_value() if chat.api_key is not None else None,
    }


def _embed_to_json(embed: EmbedModelConfig) -> dict:
    return {
        "mode": embed.mode,
        "base_url": embed.base_url,
        "model": embed.model,
        "api_key": embed.api_key.get_secret_value() if embed.api_key is not None else None,
    }


@lru_cache
def get_settings_store() -> SettingsStore:
    """全局设置存储单例（落在 data_dir/config.json）。"""
    return SettingsStore(get_settings().data_dir)
