"""设置存储：data_dir/config.json 持久化 + 与环境变量合并（设置页后端）。

优先级：config.json（dashboard 设置）> 环境变量/.env > 默认值。
- chat：base_url / model / api_key 缺省回退 env（EVERYTHING_RAG_CHAT_*）。
- embed：本地 bge-m3 默认；云端需 base_url + model + api_key。
- answer_preferences：结构化回答偏好及附加 custom_instructions（旧配置默认空）。
- rerank：资料相关性筛选（PRD §6.5）。旧配置缺少该段 -> 读为未启用，下次合法保存写入；
  启用校验读取本地校准档案（data_dir/rerank-profiles.json），生产只接受 validated 档案。
- 旧 system_prompt 首次读取时迁移为只读备份，永远不再参与模型 system 消息。
- providers / model_selection（多供应商模型选择，见 ``core/model_providers.py``）：
  ``chat`` 仍是**运行期唯一权威**；保存时由选择推导 ``chat``，所以问答链路无需改动。
  旧配置只有单段 ``chat`` 时，首次读取把它迁移成一条供应商 + 选择（不丢配置）。
API key 存 SecretStr：API 响应只回显「是否已设置 + 尾4位」，绝不回显明文。
config.json 落在 data_dir（gitignored，类似 .env），明文仅存本地。
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, model_validator

from app.core.config import get_settings
from app.core.model_providers import (
    ModelProvider,
    ModelSelection,
    chat_config_for,
    provider_from_legacy_chat,
)
from app.generation.prompt_policy import AnswerPreferences
from app.retrieval.rerank_provider import DEFAULT_ENDPOINT, validate_endpoint
from app.retrieval.reranker import (
    RerankError,
    RerankNotConfiguredError,
    RerankProfile,
    RerankValidationError,
    apply_corpus_identity,
)

CONFIG_FILENAME = "config.json"
CONFIG_SCHEMA_VERSION = 2
#: 本地校准档案文件（T30 产出，gitignored），与 config.json 同目录。
RERANK_PROFILE_FILENAME = "rerank-profiles.json"
#: 隔离验收环境开关（PRD §6.5）：只能由启动环境变量开启，设置接口无法打开。
RERANK_ACCEPTANCE_ENV = "EVERYTHING_RAG_RERANK_ACCEPTANCE"
_ACCEPTANCE_TRUTHY = frozenset({"1", "true", "yes", "on"})


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


class RerankConfig(BaseModel):
    """资料相关性筛选（重排）配置（PRD §6.5）。

    - 旧 config.json 缺少本段 -> 读为未启用（``enabled=False``），下一次合法保存写入。
    - ``api_key`` 用 SecretStr：省略/null 保留原值，空串清除，响应只给是否设置 + 尾四位。
    - ``profile_id`` 指向本地校准档案；启用时必填且必须与当前模型/指令/tokenizer/视图版本/
      评分协议匹配（生产只接受 ``validated`` 档案）。
    - 本段不包含任何从 chat/embed 自动继承的地址与密钥。
    """

    enabled: bool = False
    provider_type: str = "siliconflow_rerank"
    endpoint: str = DEFAULT_ENDPOINT
    model: str | None = None
    api_key: SecretStr | None = None
    profile_id: str | None = None
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    read_timeout_seconds: float = Field(default=60.0, gt=0, le=600)


class AppSettings(BaseModel):
    """运行时有效设置（config.json 覆盖 env 后的最终值）。"""

    chat: ChatModelConfig = Field(default_factory=ChatModelConfig)
    embed: EmbedModelConfig = Field(default_factory=EmbedModelConfig)
    vision: VisionModelConfig = Field(default_factory=VisionModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    memory_enabled: bool = True
    answer_preferences: AnswerPreferences = Field(default_factory=AnswerPreferences)
    #: 对话模型供应商注册表（设置页「模型」；可配置多个，按供应商分组）。
    providers: list[ModelProvider] = Field(default_factory=list)
    #: 当前选中的「供应商 + 模型」；None = 未选择（此时 ``chat`` 保持原值不动）。
    model_selection: ModelSelection | None = None
    #: 工作区默认思考档位对账版本（``core/reasoning_defaults.py`` 的一次性迁移标记）。
    #: 0 = 尚未对账；标记与 app.db 同处一个数据目录，因此只对该数据目录生效。
    workspace_reasoning_preferences_version: int = 0
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

    @property
    def data_dir(self) -> Path:
        """配置所在数据目录（校准档案等本地产物同目录）。"""
        return self._path.parent

    @property
    def rerank_profiles_path(self) -> Path:
        return self.data_dir / RERANK_PROFILE_FILENAME

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
        """原子写入 config.json（先写 .tmp 再 os.replace）。

        写盘前先把「当前选择」推导成 ``chat``：这样 ``chat`` 永远与选择一致，
        运行期（问答管线 / ``/api/status`` / 标题生成）继续只读 ``chat`` 即可。
        """
        settings = with_resolved_chat(settings)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "chat": _chat_to_json(settings.chat),
            "embed": _embed_to_json(settings.embed),
            "vision": _vision_to_json(settings.vision),
            "search": _search_to_json(settings.search),
            "rerank": _rerank_to_json(settings.rerank),
            "providers": _providers_to_json(settings.providers),
            "model_selection": (
                None
                if settings.model_selection is None
                else settings.model_selection.model_dump()
            ),
            "memory_enabled": settings.memory_enabled,
            "answer_preferences": settings.answer_preferences.model_dump(),
            "avatars": {
                "user": settings.avatars.user,
                "agent": settings.avatars.agent,
            },
            "theme": settings.theme,
        }
        if settings.legacy_system_prompt_backup is not None:
            payload["legacy_system_prompt_backup"] = settings.legacy_system_prompt_backup
        if settings.workspace_reasoning_preferences_version:
            # 只在非 0（真的对过账）时落盘：旧配置保持字节稳定，不在读取时被改写。
            payload["workspace_reasoning_preferences_version"] = (
                settings.workspace_reasoning_preferences_version
            )
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def mark_workspace_reasoning_preferences_reconciled(self, version: int) -> None:
        """记录「工作区默认档位已对账到第 N 版」（一次性迁移的完成标记）。

        写入走 :meth:`save` 全量落盘（与设置页保存同一路径），因此不会与其它字段冲突。
        """

        current = self.load()
        if current.workspace_reasoning_preferences_version >= version:
            return
        current.workspace_reasoning_preferences_version = version
        self.save(current)

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
        legacy_providers = False
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
        if "providers" not in raw:
            # 单段 chat 时代的配置 -> 内存里投影成一条供应商 + 选择（推导出的 chat 与原来一致）。
            # **刻意不在读取时改写文件**：读取保持无副作用，等下一次合法保存再落盘新形态
            # （与 rerank 缺段时的既有口径一致，见 test_settings_store）。
            legacy_providers = True

        try:
            loaded = AppSettings.model_validate(raw)
        except (TypeError, ValueError):
            return None
        if legacy_providers:
            providers, selection = _legacy_providers_and_selection(loaded.chat)
            loaded = loaded.model_copy(
                update={"providers": providers, "model_selection": selection}
            )
        loaded = with_resolved_chat(loaded)
        if migrated:
            self.save(loaded)
        return loaded

    def _env_defaults(self) -> AppSettings:
        env = get_settings()
        chat = ChatModelConfig(
            base_url=env.chat_base_url,
            model=env.chat_model,
            api_key=env.chat_api_key,
        )
        # env 里配好的模型同样投影成一条供应商 + 选择：设置页不该出现「空白但问答能用」。
        providers, selection = _legacy_providers_and_selection(chat)
        return AppSettings(
            chat=chat,
            providers=providers,
            model_selection=selection,
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


def with_resolved_chat(settings: AppSettings) -> AppSettings:
    """把当前选择推导成 ``chat``；选择无效/未选择时原样返回。

    这是「选择 -> 生效模型」的**唯一**收口：``chat`` 因此永远等于用户选中的那个模型，
    问答管线、``/api/status``、标题生成继续只读 ``chat``，不需要知道选择的存在。
    """

    config = chat_config_for(settings.providers, settings.model_selection)
    if config is None:
        return settings
    return settings.model_copy(update={"chat": config})


def _legacy_providers_and_selection(chat: ChatModelConfig) -> tuple[list[ModelProvider], ModelSelection | None]:
    """旧单段 ``chat``（或 env 兜底）→ 供应商列表 + 选择。"""

    provider, selection = provider_from_legacy_chat(chat)
    return ([provider] if provider is not None else []), selection


def _providers_to_json(providers: list[ModelProvider]) -> list[dict]:
    """供应商落盘形态（与 config.json 其它段一致：明文密钥仅存本地）。"""

    return [
        {
            "id": provider.id,
            "name": provider.name,
            "kind": provider.kind,
            "protocol": provider.protocol,
            "base_url": provider.base_url,
            "api_key": (
                provider.api_key.get_secret_value() if provider.api_key is not None else None
            ),
            "models": [
                {"id": entry.id, "name": entry.name, "vision": entry.vision}
                for entry in provider.models
            ],
        }
        for provider in providers
    ]


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


def _rerank_to_json(rerank: RerankConfig) -> dict:
    return {
        "enabled": rerank.enabled,
        "provider_type": rerank.provider_type,
        "endpoint": rerank.endpoint,
        "model": rerank.model,
        "api_key": (
            rerank.api_key.get_secret_value() if rerank.api_key is not None else None
        ),
        "profile_id": rerank.profile_id,
        "connect_timeout_seconds": rerank.connect_timeout_seconds,
        "read_timeout_seconds": rerank.read_timeout_seconds,
    }


# --------------------------------------------------------------------------------------
# 重排档案（PRD §6.5）：本地校准产物 + 启用校验
# --------------------------------------------------------------------------------------


def acceptance_mode_enabled() -> bool:
    """隔离验收环境是否开启（PRD §6.5）。

    只由启动环境变量 ``EVERYTHING_RAG_RERANK_ACCEPTANCE`` 决定；线上设置接口没有任何
    写入路径可以打开它。验收环境使用独立数据目录（EVERYTHING_RAG_DATA_DIR）运行。
    """
    return os.environ.get(RERANK_ACCEPTANCE_ENV, "").strip().lower() in _ACCEPTANCE_TRUTHY


def endpoint_fingerprint(endpoint: str) -> str:
    """档案里的 endpoint 身份：``scheme://host``（不含路径、查询、凭据）。"""
    if not isinstance(endpoint, str):
        return ""
    parts = urlsplit(endpoint.strip())
    return f"{parts.scheme}://{parts.hostname}" if parts.hostname else ""


@dataclass(frozen=True)
class RerankRuntimeIdentity:
    """运行时身份：当前设置 + 代码侧版本常量（PRD §6.5 / R14）。"""

    provider_type: str
    model: str | None
    endpoint_fingerprint: str
    #: 证据视图版本（evidence_view 落地后提供；未落地为 None，不做猜测）
    view_version: str | None
    #: 评分协议版本（relevance_selector 落地后提供）
    scoring_protocol_version: str | None


def _module_constant(module_name: str, attribute: str) -> str | None:
    """读取可选模块里的版本常量；模块尚未落地或暂时无法导入时返回 None。

    目的：设置页与启用校验不能因为一个可选模块的导入问题（尚未实现、正在施工、依赖缺失）
    而崩溃；无法取得运行时常量时该项不做比较，而不是放行或猜测。
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 -- 可选模块缺失/施工中，降级为「不比较该项」
        return None
    value = getattr(module, attribute, None)
    return value if isinstance(value, str) and value else None


def rerank_runtime_identity(rerank: RerankConfig) -> RerankRuntimeIdentity:
    """由当前重排设置与代码版本常量组成运行时身份。"""
    return RerankRuntimeIdentity(
        provider_type=rerank.provider_type,
        model=rerank.model,
        endpoint_fingerprint=endpoint_fingerprint(rerank.endpoint),
        view_version=_module_constant("app.retrieval.evidence_view", "VIEW_VERSION"),
        scoring_protocol_version=_module_constant(
            "app.retrieval.relevance_selector", "SCORING_PROTOCOL_VERSION"
        ),
    )


def _profile_from_mapping(raw: Mapping[str, Any]) -> RerankProfile:
    """把档案文件里的一条记录还原为 ``RerankProfile``（未知字段与非法值立即失败）。"""
    allowed = {item.name for item in fields(RerankProfile)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RerankValidationError(f"档案包含未知字段: {', '.join(unknown)}")
    data: dict[str, Any] = dict(raw)
    for key in ("generated_at", "validated_at"):
        value = data.get(key)
        if isinstance(value, str) and value:
            data[key] = datetime.fromisoformat(value)
    return RerankProfile(**data)


class RerankProfileStore:
    """本地校准档案读取器（``data_dir/rerank-profiles.json``，gitignored）。

    文件可以是 ``{"profiles": [...]}`` 或直接一个数组；文件缺失/损坏返回空表，
    单条记录非法只跳过该条，不影响其它档案，也不让设置页报错。
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> RerankProfileStore:
        return cls(Path(data_dir) / RERANK_PROFILE_FILENAME)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, RerankProfile]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        entries = raw.get("profiles") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return {}
        profiles: dict[str, RerankProfile] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                profile = _profile_from_mapping(entry)
            except (RerankError, TypeError, ValueError):
                continue
            profiles[profile.profile_id] = profile
        return profiles

    def get(self, profile_id: str | None) -> RerankProfile | None:
        if not profile_id:
            return None
        return self.load().get(profile_id)


def validate_rerank_enable(
    rerank: RerankConfig,
    profiles: Mapping[str, RerankProfile],
    *,
    acceptance: bool | None = None,
    corpus_identity: str | None = None,
) -> RerankProfile:
    """启用校验（PRD §6.5/§8.1）：返回选中的档案；任何缺失/失配抛 ``RerankError``。

    - 缺少 endpoint / model / 密钥 / 档案 -> ``RERANK_NOT_CONFIGURED``；
    - 生产环境档案状态不是 ``validated`` -> ``RERANK_PROFILE_MISMATCH``；
    - 模型、指令、tokenizer、视图版本、评分协议与档案不一致 -> ``RERANK_PROFILE_MISMATCH``；
    - 隔离验收环境（显式启动参数）允许运行 candidate / quality_passed 档案；
    - 传入 ``corpus_identity``（当前语料身份哈希）时与档案 ``corpus_manifest_hash`` 比较：
      不一致或无从证明时标记 ``needs_validation``，**只标记不阻止**继续使用当前兼容档案
      （PRD §6.5）。不传时行为不变（调用方无法取得语料身份时不得猜）。
    """
    if not endpoint_fingerprint(rerank.endpoint):
        raise RerankNotConfiguredError("重排服务地址无效或缺少主机名")
    validate_endpoint(rerank.endpoint)
    if not (rerank.model or "").strip():
        raise RerankNotConfiguredError("请先填写重排模型名")
    key = rerank.api_key.get_secret_value().strip() if rerank.api_key is not None else ""
    if not key:
        raise RerankNotConfiguredError("请先填写重排服务 API Key")
    profile_id = (rerank.profile_id or "").strip()
    if not profile_id:
        raise RerankNotConfiguredError("请先选择已校准的重排档案")
    profile = profiles.get(profile_id)
    if profile is None:
        raise RerankNotConfiguredError(f"档案 {profile_id} 不存在或未通过校验")
    acceptance_env = acceptance_mode_enabled() if acceptance is None else acceptance
    if not acceptance_env:
        profile.assert_production_ready()
    identity = rerank_runtime_identity(rerank)
    profile.assert_compatible(
        provider_type=identity.provider_type,
        model=identity.model,
        endpoint_fingerprint=identity.endpoint_fingerprint or None,
        view_version=identity.view_version,
        scoring_protocol_version=identity.scoring_protocol_version,
    )
    if corpus_identity is not None:
        # PRD §6.5：语料与档案验证过的版本不一致时标 needs_validation（只标记，不阻止）。
        profile = apply_corpus_identity(profile, corpus_identity)
    return profile


@lru_cache
def get_settings_store() -> SettingsStore:
    """全局设置存储单例（落在 data_dir/config.json）。"""
    return SettingsStore(get_settings().data_dir)
