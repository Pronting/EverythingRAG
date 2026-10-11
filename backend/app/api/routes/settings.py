"""设置接口：GET /api/settings（读取）+ PUT /api/settings（保存，dashboard 设置页）。

- 读取返回脱敏视图：API key 只回显 api_key_set + 尾4位 hint，绝不回显明文。
- 保存为局部更新：请求中省略/为 null 的字段保持原值；api_key 传空串表示清除。
- 校验：chat base_url 需 http(s)；embed mode=cloud 时 base_url/model 必填。
- 多供应商模型（providers / model_selection）：``providers`` 为**整表替换**（省略的供应商
  即被删除），同名 id 的 api_key 省略即保留原值、空串清除；``model_selection`` 决定
  ``chat``（保存时推导，见 ``SettingsStore.save``）。删掉当前选中的供应商会一并清空
  ``chat``，不留悬空引用。
- POST /api/settings/providers/models：按端点拉取可用模型列表（真实出网、经审计、不落盘密钥）。
- rerank（PRD §6.5/§8.1）：保存与启用分别校验；启用但缺凭证/endpoint/有效档案 -> 422 且
  旧设置不变；生产只接受 validated 档案，验收环境由启动参数开启。
- POST /api/settings/rerank/test：固定公开文本做真实连接测试，不读知识库、不持久化测试密钥。
- Agent 核心策略由后端管理；用户可保存表达偏好及附加自定义要求，不能替换核心 system prompt。
- 落盘：data_dir/config.json（gitignored，明文仅存本地）。
"""

from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.core.model_providers import (
    DEFAULT_PROTOCOL,
    SUPPORTED_PROTOCOLS,
    ModelProvider,
    ModelSelection,
    ProviderModelEntry,
    builtin_catalog,
    find_model,
    providers_view,
    selection_view,
)
from app.core.settings_store import (
    AppSettings,
    ChatModelConfig,
    EmbedModelConfig,
    RerankConfig,
    RerankProfileStore,
    SearchConfig,
    SettingsStore,
    VisionModelConfig,
    acceptance_mode_enabled,
    get_settings_store,
    validate_rerank_enable,
    with_resolved_chat,
)
from app.generation.prompt_policy import POLICY_VERSION, AnswerPreferences
from app.retrieval.rerank_provider import (
    DEFAULT_ENDPOINT,
    SiliconFlowRerankProvider,
    validate_endpoint,
)
from app.retrieval.reranker import RerankError, RerankErrorCode, RerankProfile

router = APIRouter()

_BASE_URL_RE = re.compile(r"^https?://\S+$")

#: 档案状态 -> 前端可读的设置状态（PRD §8.1：档案验证日期与状态）。
_RERANK_STATUS_BY_CODE = {
    RerankErrorCode.NOT_CONFIGURED: "not_configured",
    RerankErrorCode.PROFILE_MISMATCH: "profile_mismatch",
    RerankErrorCode.INVALID_INPUT: "invalid",
}


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


class RerankSettingsUpdate(BaseModel):
    """重排设置局部更新（PRD §6.5）：api_key 省略/null 保留，空串清除。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    provider_type: str | None = None
    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    api_key: SecretStr | None = None
    profile_id: str | None = Field(default=None, max_length=128)
    connect_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    read_timeout_seconds: float | None = Field(default=None, gt=0, le=600)


class AnswerPreferencesUpdate(BaseModel):
    """回答偏好及附加自定义要求；省略保留，空字符串清除自定义要求。"""

    model_config = ConfigDict(extra="forbid")

    language: Literal["auto", "zh-CN", "en"] | None = None
    verbosity: Literal["concise", "balanced", "detailed"] | None = None
    tone: Literal["natural", "professional"] | None = None
    response_format: Literal["auto", "prose", "bullets"] | None = None
    knowledge_only: bool | None = None
    custom_instructions: str | None = Field(default=None, max_length=4000)


class ProviderModelUpdate(BaseModel):
    """供应商下的一个模型（真实调用名 + 显示名 + 是否多模态）。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=120)
    vision: bool = False


class ProviderUpdate(BaseModel):
    """供应商整条状态（``providers`` 为整表替换，因此每条都必须给全）。

    ``api_key``：省略/None = 保留同名 id 的原密钥；空串 = 清除；非空 = 设置。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=60)
    kind: Literal["builtin", "custom"] = "custom"
    protocol: str | None = Field(default=None, max_length=64)
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    models: list[ProviderModelUpdate] = Field(default_factory=list, max_length=200)


class ModelSelectionUpdate(BaseModel):
    """当前选中的供应商 + 模型；两个字段都为空 = 清除选择。"""

    model_config = ConfigDict(extra="forbid")

    provider_id: str | None = Field(default=None, max_length=64)
    model_id: str | None = Field(default=None, max_length=200)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat: ChatSettingsUpdate | None = None
    embed: EmbedSettingsUpdate | None = None
    vision: VisionSettingsUpdate | None = None
    search: SearchSettingsUpdate | None = None
    rerank: RerankSettingsUpdate | None = None
    providers: list[ProviderUpdate] | None = Field(default=None, max_length=50)
    model_selection: ModelSelectionUpdate | None = None
    memory_enabled: bool | None = None
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
        if self.embed is not None and self.embed.mode == "cloud" and (
            not self.embed.base_url or not self.embed.model
        ):
            raise ValueError("云端嵌入需同时提供 Base URL 与模型名")
        if self.providers is not None:
            self._validate_providers()
        if self.rerank is not None and self.rerank.endpoint is not None:
            # 保存即校验：完整 HTTPS，或显式配置的本机回环 HTTP（PRD §6.5）
            try:
                validate_endpoint(self.rerank.endpoint)
            except RerankError as exc:
                raise ValueError(exc.user_message) from exc
        return self

    def _validate_providers(self) -> None:
        """供应商列表校验：id 唯一、地址为 http(s)、协议在支持表内。"""

        assert self.providers is not None
        seen: set[str] = set()
        for provider in self.providers:
            if provider.id in seen:
                raise ValueError(f"供应商 id 重复：{provider.id}")
            seen.add(provider.id)
            if provider.base_url is not None and provider.base_url.strip() and not _BASE_URL_RE.match(
                provider.base_url.strip()
            ):
                raise ValueError(f"供应商「{provider.name}」的 API 地址需为 http(s) 地址")
            if provider.protocol is not None and provider.protocol not in SUPPORTED_PROTOCOLS:
                raise ValueError(f"不支持的 API 协议：{provider.protocol}")
            model_ids: set[str] = set()
            for model in provider.models:
                if model.id.strip() in model_ids:
                    raise ValueError(f"供应商「{provider.name}」里模型 id 重复：{model.id}")
                model_ids.add(model.id.strip())


@router.get("/api/settings")
async def get_settings_view(
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """返回脱敏设置视图（key 只给 hint）。"""
    settings = store.load()
    profiles = RerankProfileStore.for_data_dir(store.data_dir).load()
    return _settings_view(settings, profiles=profiles)


@router.put("/api/settings")
async def update_settings(
    update: SettingsUpdate,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """局部保存设置：覆盖字段更新后原子落盘，返回脱敏视图。

    启用重排时先校验档案与凭证（PRD §8.1）：校验失败返回 422 且**不写盘**，旧设置不变。
    模型选择指向不存在的供应商/模型时同样 422（不猜、不静默回落）。
    """
    current = store.load()
    try:
        updated = _apply(current, update)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_model_selection", "message": str(exc)}
        ) from exc
    # 选择 -> chat 在保存前完成，保证「返回值」与「落盘值」是同一份（save 内部再推导一次）。
    updated = with_resolved_chat(updated)
    profiles = RerankProfileStore.for_data_dir(store.data_dir).load()
    if updated.rerank.enabled:
        try:
            validate_rerank_enable(updated.rerank, profiles)
        except RerankError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code.value, "message": exc.user_message},
            ) from exc
    store.save(updated)
    return _settings_view(updated, profiles=profiles)


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
    rerank = current.rerank
    if update.rerank is not None:
        rerank = _overlay_rerank(rerank, update.rerank)
    preferences = current.answer_preferences
    if update.answer_preferences is not None:
        preferences = _overlay_preferences(preferences, update.answer_preferences)
    providers, selection, selection_lost = _overlay_model_providers(current, update)
    if update.chat is not None and update.model_selection is None:
        # 显式直接配置 chat（旧客户端 / 脚本）优先于供应商选择：清掉可能过期、与 chat
        # 不一致的选择，避免「界面上选的是 A、实际发的是 B」。
        selection = None
        selection_lost = False
    return AppSettings(
        chat=ChatModelConfig() if selection_lost else chat,
        embed=embed,
        vision=vision,
        search=search,
        rerank=rerank,
        providers=providers,
        model_selection=selection,
        memory_enabled=(
            update.memory_enabled if update.memory_enabled is not None else current.memory_enabled
        ),
        answer_preferences=preferences,
        legacy_system_prompt_backup=current.legacy_system_prompt_backup,
        avatars=current.avatars,  # 保留头像（覆盖式更新不得重置）
        theme=update.theme if update.theme is not None else current.theme,
    )


def _overlay_model_providers(
    current: AppSettings, update: SettingsUpdate
) -> tuple[list[ModelProvider], ModelSelection | None, bool]:
    """供应商整表替换 + 选择解析。

    返回 ``(providers, selection, selection_lost)``：``selection_lost=True`` 表示原本选中的
    供应商/模型在这次更新里被删掉了 —— 调用方据此把 ``chat`` 一起清空，不留悬空引用。
    """

    providers = current.providers
    selection = current.model_selection
    selection_lost = False
    if update.providers is not None:
        providers = _replace_providers(current.providers, update.providers)
        if selection is not None and find_model(providers, selection.provider_id, selection.model_id) is None:
            selection = None
            selection_lost = True
    if update.model_selection is not None:
        provider_id = (update.model_selection.provider_id or "").strip()
        model_id = (update.model_selection.model_id or "").strip()
        if not provider_id and not model_id:
            selection = None
        else:
            if find_model(providers, provider_id, model_id) is None:
                raise ValueError("所选模型不存在，请先在设置页添加该供应商与模型")
            selection = ModelSelection(provider_id=provider_id, model_id=model_id)
            selection_lost = False
    return providers, selection, selection_lost


def _replace_providers(
    current: list[ModelProvider], updates: list[ProviderUpdate]
) -> list[ModelProvider]:
    """整表替换：省略的供应商被删除；同名 id 的 api_key 省略即保留原值。"""

    existing = {provider.id: provider for provider in current}
    result: list[ModelProvider] = []
    for item in updates:
        previous = existing.get(item.id.strip())
        api_key = previous.api_key if previous is not None else None
        if item.api_key is not None:
            api_key = None if item.api_key.get_secret_value() == "" else item.api_key
        protocol = (item.protocol or "").strip() or (
            previous.protocol if previous is not None else DEFAULT_PROTOCOL
        )
        result.append(
            ModelProvider(
                id=item.id,
                name=item.name,
                kind=item.kind,
                protocol=protocol,
                base_url=item.base_url,
                api_key=api_key,
                models=[
                    ProviderModelEntry(id=model.id, name=model.name, vision=model.vision)
                    for model in item.models
                ],
            )
        )
    return result


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


def _overlay_rerank(current: RerankConfig, update: RerankSettingsUpdate) -> RerankConfig:
    """重排设置局部更新；endpoint 去空白，api_key 空串=清除。"""
    data: dict = {
        "enabled": update.enabled if update.enabled is not None else current.enabled,
        "provider_type": (
            update.provider_type if update.provider_type is not None else current.provider_type
        ),
        "endpoint": (
            update.endpoint.strip() if update.endpoint is not None else current.endpoint
        ),
        "model": (
            update.model.strip() if update.model is not None else current.model
        ),
        "profile_id": (
            update.profile_id.strip() if update.profile_id is not None else current.profile_id
        ),
        "connect_timeout_seconds": (
            update.connect_timeout_seconds
            if update.connect_timeout_seconds is not None
            else current.connect_timeout_seconds
        ),
        "read_timeout_seconds": (
            update.read_timeout_seconds
            if update.read_timeout_seconds is not None
            else current.read_timeout_seconds
        ),
    }
    if update.api_key is not None:
        data["api_key"] = None if update.api_key.get_secret_value() == "" else update.api_key
    else:
        data["api_key"] = current.api_key
    return RerankConfig(**data)


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


def _settings_view(
    settings: AppSettings, *, profiles: dict[str, RerankProfile] | None = None
) -> dict:
    """脱敏视图：key 只回显是否已设置 + 尾4位；头像给可访问 URL 或 null。"""
    return {
        "chat": _chat_view(settings.chat),
        "embed": _embed_view(settings.embed),
        "vision": _vision_view(settings.vision),
        "search": _search_view(settings.search),
        "rerank": _rerank_view(settings.rerank, profiles or {}),
        "providers": providers_view(settings.providers, settings.model_selection),
        "model_selection": selection_view(settings.providers, settings.model_selection),
        "model_catalog": builtin_catalog(),
        "model_protocols": list(SUPPORTED_PROTOCOLS),
        "memory_enabled": settings.memory_enabled,
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


def _rerank_profile_view(profile: RerankProfile | None) -> dict | None:
    """档案投影：状态与验证日期（PRD §8.1）；不展示 tokenizer 等实现参数。"""
    if profile is None:
        return None
    return {
        "profile_id": profile.profile_id,
        "status": str(profile.status),
        "model": profile.model,
        "generated_at": profile.generated_at.isoformat() if profile.generated_at else None,
        "validated_at": profile.validated_at.isoformat() if profile.validated_at else None,
        "needs_validation": profile.needs_validation,
    }


def _rerank_status(
    rerank: RerankConfig, profiles: dict[str, RerankProfile]
) -> tuple[str, RerankProfile | None]:
    """设置状态：disabled / not_configured / profile_mismatch / needs_validation / ready。"""
    if not rerank.enabled:
        return "disabled", None
    try:
        # PRD §6.5：每次快照都拿当前语料身份与档案登记值比较，不一致即标 needs_validation
        # （只标记，不阻止继续使用当前兼容档案）；语料身份取不到时由标记函数保守处理。
        from app.api.deps import current_corpus_identity

        profile = validate_rerank_enable(
            rerank, profiles, corpus_identity=current_corpus_identity()
        )
    except RerankError as exc:
        return _RERANK_STATUS_BY_CODE.get(exc.code, "error"), None
    return ("needs_validation" if profile.needs_validation else "ready"), profile


def _rerank_view(rerank: RerankConfig, profiles: dict[str, RerankProfile]) -> dict:
    """重排设置脱敏视图（PRD §8.1）：key 只给是否设置 + 末四位。"""
    key = rerank.api_key.get_secret_value() if rerank.api_key is not None else None
    status, selected = _rerank_status(rerank, profiles)
    if selected is None and rerank.profile_id:
        selected = profiles.get(rerank.profile_id)
    return {
        "enabled": rerank.enabled,
        "provider_type": rerank.provider_type,
        "endpoint": rerank.endpoint,
        "model": rerank.model,
        "api_key_set": bool(key),
        "api_key_hint": f"...{key[-4:]}" if key else None,
        "profile_id": rerank.profile_id,
        "connect_timeout_seconds": rerank.connect_timeout_seconds,
        "read_timeout_seconds": rerank.read_timeout_seconds,
        "configured": bool(rerank.model and key and rerank.endpoint),
        "status": status,
        "acceptance_env": acceptance_mode_enabled(),
        "profile": _rerank_profile_view(selected),
        "profiles": [
            _rerank_profile_view(profile) for profile in sorted(profiles.values(), key=lambda p: p.profile_id)
        ],
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


class ProviderModelsRequest(BaseModel):
    """「获取可用模型」请求：可提交未保存的地址与密钥（服务端只用于本次请求，不落盘）。"""

    model_config = ConfigDict(extra="forbid")

    provider_id: str | None = Field(default=None, max_length=64)
    base_url: str = Field(max_length=2048)
    api_key: SecretStr | None = None


@router.post("/api/settings/providers/models")
async def list_provider_models(
    request: ProviderModelsRequest,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """按 OpenAI 兼容 ``GET {base_url}/models`` 拉取可用模型（真实出网，经审计）。

    - 密钥省略时**只在地址与已保存供应商一致时**复用（换地址不复用，与既有连接测试一致）；
    - 失败只回稳定文案（不泄露响应体与密钥），``success=false`` + 空列表；
    - 不落盘任何东西：结果由用户确认后再随 ``providers`` 一起保存。
    """

    from urllib.parse import urlsplit

    from app.generation.connection_test import list_available_models

    url = request.base_url.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
    except ValueError:
        return {"success": False, "message": "服务地址格式不正确", "models": [], "count": 0}
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"success": False, "message": "请填写有效的服务地址", "models": [], "count": 0}
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return {
            "success": False,
            "message": "服务地址不能包含凭据或查询参数",
            "models": [],
            "count": 0,
        }
    key = request.api_key
    if key is None and request.provider_id:
        saved = next(
            (item for item in store.load().providers if item.id == request.provider_id), None
        )
        if saved is not None and (saved.base_url or "").strip().rstrip("/") == url:
            key = saved.api_key
    return await list_available_models(
        url, key.get_secret_value() if key is not None else None
    )


class RerankTestRequest(BaseModel):
    """重排连接测试请求（PRD §8.1）：可提交未保存字段，服务端不持久化测试密钥。"""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    api_key: SecretStr | None = None
    profile_id: str | None = Field(default=None, max_length=128)


def _build_rerank_provider(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    instruction_text: str,
    config: RerankConfig,
) -> SiliconFlowRerankProvider:
    """构造真实 Provider（连接测试专用；测试可通过 monkeypatch 注入替身客户端）。"""
    return SiliconFlowRerankProvider(
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        instruction_text=instruction_text,
        connect_timeout_seconds=config.connect_timeout_seconds,
        read_timeout_seconds=config.read_timeout_seconds,
    )


@router.post("/api/settings/rerank/test")
async def test_rerank_connection(
    request: RerankTestRequest,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """用固定公开文本做真实连接测试：不读知识库、不落盘测试密钥、不代表质量验收。

    连通成功只说明服务可达且协议匹配（``quality_checked=false``）；质量结论只能由
    T30–T32 的真实评测给出（PRD §8.1）。
    """
    current = store.load().rerank
    endpoint = (request.endpoint or current.endpoint or DEFAULT_ENDPOINT).strip()
    model = (request.model or current.model or "").strip()
    # 未保存的密钥只用于本次测试；换地址时不复用已保存密钥（与既有连接测试一致）
    key = request.api_key
    if key is None and endpoint == (current.endpoint or "").strip():
        key = current.api_key
    key_value = key.get_secret_value().strip() if key is not None else ""
    if not model or not key_value:
        return {
            "success": False,
            "message": "请先填写重排模型名与 API Key",
            "endpoint": endpoint,
            "quality_checked": False,
        }
    try:
        validate_endpoint(endpoint)
    except RerankError as exc:
        return {
            "success": False,
            "code": exc.code.value,
            "message": exc.user_message,
            "endpoint": endpoint,
            "quality_checked": False,
        }
    profiles = RerankProfileStore.for_data_dir(store.data_dir).load()
    profile_id = (request.profile_id or current.profile_id or "").strip()
    profile = profiles.get(profile_id)
    provider = _build_rerank_provider(
        endpoint=endpoint,
        model=model,
        api_key=key_value,
        instruction_text=profile.instruction_text if profile is not None else "",
        config=current,
    )
    try:
        result = await provider.check_connection()
    finally:
        await provider.aclose()
    result["endpoint"] = endpoint
    result["profile_id"] = profile.profile_id if profile is not None else None
    result["quality_checked"] = False
    return result
