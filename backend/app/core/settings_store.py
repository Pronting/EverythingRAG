"""设置存储：data_dir/config.json 持久化 + 与环境变量合并（设置页后端）。

优先级：config.json（dashboard 设置）> 环境变量/.env > 默认值。
- chat：base_url / model / api_key 缺省回退 env（EVERYTHING_RAG_CHAT_*）。
- embed：本地 bge-m3 默认；云端需 base_url + model + api_key。
- answer_preferences：结构化回答偏好及附加 custom_instructions（旧配置默认空）。
- 旧 system_prompt 首次读取时迁移为只读备份，永远不再参与模型 system 消息。
API key 存 SecretStr：API 响应只回显「是否已设置 + 尾4位」，绝不回显明文。
config.json 落在 data_dir（gitignored，类似 .env），明文仅存本地。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator

from app.core.config import get_settings
from app.generation.prompt_policy import AnswerPreferences

CONFIG_FILENAME = "config.json"
CONFIG_SCHEMA_VERSION = 2


class ChatModelConfig(BaseModel):
    """对话模型配置（dashboard 可编辑）。"""

    provider_type: str = "openai_compatible"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None
    #: 是否多模态（可直接接收图片输入）。False（默认）= 纯文本模型（如 DeepSeek），
    #: 用户贴图时由识图模型先转文字描述再交给本模型（实验性「识图代理」）。
    supports_image: bool = False


class EmbedModelConfig(BaseModel):
    """嵌入模型配置：云端 OpenAI 兼容嵌入（/embeddings）。本地嵌入已移除。

    ``mode`` 保留 ``"local"`` 仅为向后兼容读取旧 config.json（读入时自动迁移为
    ``"cloud"``）；新写入恒为 ``"cloud"``。
    """

    mode: Literal["local", "cloud"] = "cloud"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _migrate_legacy_local(self) -> EmbedModelConfig:
        if self.mode == "local":
            self.mode = "cloud"
        return self


class SearchConfig(BaseModel):
    """联网搜索配置（Tavily / SearXNG）。``enabled`` 默认 False（隐私默认零外发），
    开启并补齐凭证后，问答框的「联网搜索」按钮才可用。
    """

    provider_type: Literal["tavily", "searxng"] = "tavily"
    enabled: bool = False
    base_url: str | None = None  # SearXNG 实例地址；Tavily 缺省用官方 api.tavily.com
    api_key: SecretStr | None = None
    max_results: int = 5


class VisionModelConfig(BaseModel):
    """识图模型配置（OpenAI 兼容多模态，云端/本地 Ollama 同协议）。

    识图是功能开关（PRD 4.3/4.7）：base_url 与 model 都未配置时，图片仅存
    元数据、不调用识图模型；配齐后入库时对图床图片生成文字描述。
    """

    provider_type: str = "openai_compatible"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None


class AvatarConfig(BaseModel):
    """头像配置：存相对文件名（位于 data_dir/avatars/ 下），None=未设置（用内置 SVG）。

    文件名由上传接口生成（`{user|agent}-{随机token}{ext}`），每次上传即替换并删除旧文件，
    因此同一文件名对应的内容永不变化，前端可安全按 URL 缓存。
    """

    user: str | None = None
    agent: str | None = None


class AppSettings(BaseModel):
    """运行时有效设置（config.json 覆盖 env 后的最终值）。"""

    chat: ChatModelConfig = Field(default_factory=ChatModelConfig)
    embed: EmbedModelConfig = Field(default_factory=EmbedModelConfig)
    vision: VisionModelConfig = Field(default_factory=VisionModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    answer_preferences: AnswerPreferences = Field(default_factory=AnswerPreferences)
    # 只用于保留历史配置，API 不暴露也不允许修改，ChatService 永远不读取。
    legacy_system_prompt_backup: str | None = None
    avatars: AvatarConfig = Field(default_factory=AvatarConfig)
    theme: Literal["light", "dark"] = "light"


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
            "schema_version": CONFIG_SCHEMA_VERSION,
            "chat": _chat_to_json(settings.chat),
            "embed": _embed_to_json(settings.embed),
            "vision": _vision_to_json(settings.vision),
            "search": _search_to_json(settings.search),
            "answer_preferences": settings.answer_preferences.model_dump(),
            "avatars": {
                "user": settings.avatars.user,
                "agent": settings.avatars.agent,
            },
            "theme": settings.theme,
        }
        if settings.legacy_system_prompt_backup is not None:
            payload["legacy_system_prompt_backup"] = settings.legacy_system_prompt_backup
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
        if not isinstance(raw, dict):
            return None

        # v1 -> v2：旧任意 system_prompt 只做审计备份，不再有任何执行路径。
        migrated = False
        if "system_prompt" in raw:
            legacy = raw.pop("system_prompt")
            if "legacy_system_prompt_backup" not in raw and isinstance(legacy, str):
                raw["legacy_system_prompt_backup"] = legacy
            migrated = True
        if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raw["schema_version"] = CONFIG_SCHEMA_VERSION
            migrated = True
        if "answer_preferences" not in raw:
            raw["answer_preferences"] = AnswerPreferences().model_dump()
            migrated = True

        try:
            loaded = AppSettings.model_validate(raw)
        except (TypeError, ValueError):
            return None
        if migrated:
            self.save(loaded)
        return loaded

    def _env_defaults(self) -> AppSettings:
        env = get_settings()
        return AppSettings(
            chat=ChatModelConfig(
                base_url=env.chat_base_url,
                model=env.chat_model,
                api_key=env.chat_api_key,
            ),
            embed=EmbedModelConfig(
                mode="cloud",
                base_url=env.embed_base_url,
                model=env.embed_model,
                api_key=env.embed_api_key,
            ),
            vision=VisionModelConfig(
                base_url=env.vision_base_url,
                model=env.vision_model,
                api_key=env.vision_api_key,
            ),
            search=SearchConfig(
                provider_type=env.search_provider_type,
                enabled=env.search_enabled,
                base_url=env.search_base_url,
                api_key=env.search_api_key,
                max_results=env.search_max_results,
            ),
        )


def _chat_to_json(chat: ChatModelConfig) -> dict:
    return {
        "provider_type": chat.provider_type,
        "base_url": chat.base_url,
        "model": chat.model,
        "api_key": chat.api_key.get_secret_value() if chat.api_key is not None else None,
        "supports_image": chat.supports_image,
    }


def _embed_to_json(embed: EmbedModelConfig) -> dict:
    return {
        "mode": embed.mode,
        "base_url": embed.base_url,
        "model": embed.model,
        "api_key": embed.api_key.get_secret_value() if embed.api_key is not None else None,
    }


def _vision_to_json(vision: VisionModelConfig) -> dict:
    return {
        "provider_type": vision.provider_type,
        "base_url": vision.base_url,
        "model": vision.model,
        "api_key": vision.api_key.get_secret_value() if vision.api_key is not None else None,
    }


def _search_to_json(search: SearchConfig) -> dict:
    return {
        "provider_type": search.provider_type,
        "enabled": search.enabled,
        "base_url": search.base_url,
        "api_key": search.api_key.get_secret_value() if search.api_key is not None else None,
        "max_results": search.max_results,
    }


def is_vision_configured(vision: VisionModelConfig) -> bool:
    """识图是否可用：base_url 与 model 都配好（api_key 云端时按需，本地 Ollama 可空）。"""
    return bool(vision.base_url and vision.model)


def is_search_configured(search: SearchConfig) -> bool:
    """联网搜索是否可用：显式启用 + provider 凭证齐备。

    - tavily：需 api_key；
    - searxng：需 base_url。
    供 /api/status 与问答装配判断「联网搜索」按钮是否可用。
    """
    if not search.enabled:
        return False
    if search.provider_type == "tavily":
        return search.api_key is not None
    if search.provider_type == "searxng":
        return bool(search.base_url)
    return False


@lru_cache
def get_settings_store() -> SettingsStore:
    """全局设置存储单例（落在 data_dir/config.json）。"""
    return SettingsStore(get_settings().data_dir)
