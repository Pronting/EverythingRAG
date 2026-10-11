"""对话模型供应商注册表（设置页「模型」+ 输入框模型选择器）。

本模块只负责**注册表语义**：供应商（provider）→ 模型（model）两级结构、预置目录、
脱敏投影、以及「当前选择 → 生效的 ``ChatModelConfig``」这一条纯函数映射。它**不做 IO**，
也不 import 设置存储（避免循环依赖）：

- 落盘字段与读写由 :mod:`app.core.settings_store` 负责（``AppSettings.providers`` /
  ``AppSettings.model_selection``）；
- HTTP 出网（「获取可用模型」）在 :mod:`app.generation.connection_test` 里，经唯一
  ``outbound_client`` 审计（CLAUDE.md §11 隐私红线：其余模块不得发外呼）。

三个不许违背的约定：

1. **``chat`` 仍是运行期唯一权威**。选择（``model_selection``）只用来**推导** ``chat``
   （base_url / model / api_key / supports_image），推导在保存时完成，因此问答管线、
   ``/api/status``、标题生成等既有链路一行都不用改。
2. **密钥只以 ``SecretStr`` 存放，视图只给「是否已设置 + 尾4位」**（与 config.json 其它段一致）。
3. **不猜**：选择指向不存在的供应商/模型时返回 ``None``（调用方据此清空或报错），
   绝不用「第一个模型」之类的兜底冒充用户选择。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

__all__ = [
    "BUILTIN_PROVIDERS",
    "DEFAULT_PROTOCOL",
    "LEGACY_PROVIDER_ID",
    "SUPPORTED_PROTOCOLS",
    "ModelProvider",
    "ModelSelection",
    "ProviderModelEntry",
    "builtin_catalog",
    "chat_config_for",
    "find_model",
    "model_display_name",
    "next_custom_provider_id",
    "provider_from_legacy_chat",
    "provider_status",
    "provider_view",
    "providers_view",
    "selection_view",
]

#: 本版唯一支持的 API 协议（OpenAI Chat Completions 兼容）。设置页的可选项就这一项；
#: 值照样落盘，将来新增协议时旧配置不需要迁移。
DEFAULT_PROTOCOL = "openai-completions"
SUPPORTED_PROTOCOLS: tuple[str, ...] = (DEFAULT_PROTOCOL,)

#: 旧配置（只有单段 ``chat``）迁移出的供应商 id。固定值让迁移可重复且可断言。
LEGACY_PROVIDER_ID = "existing"

#: 供应商 id 允许的字符（避免任意字符串进入选择器与 URL 关联字段）。
PROVIDER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


class ProviderModelEntry(BaseModel):
    """供应商下的一个模型：``id`` 是**真实调用名**，``name`` 只用于显示。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=120)
    #: 是否多模态（可直接接收图片输入）。True 时该模型被选为当前模型即开启「图片输入」。
    vision: bool = False

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("模型 id 不能为空")
        return stripped

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ModelProvider(BaseModel):
    """一个模型供应商（内置预置或用户自定义）。

    ``kind`` 只影响界面语义（内置项可编辑、不可删除），不影响调用行为。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=60)
    kind: Literal["builtin", "custom"] = "custom"
    protocol: str = DEFAULT_PROTOCOL
    base_url: str | None = None
    api_key: SecretStr | None = None
    models: list[ProviderModelEntry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        stripped = value.strip()
        if not re.match(PROVIDER_ID_PATTERN, stripped):
            raise ValueError(
                "供应商 id 只能包含字母、数字、点、下划线与短横线（1-64 位，首字符为字母或数字）"
            )
        return stripped

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("供应商名称不能为空")
        return stripped

    @field_validator("protocol")
    @classmethod
    def _check_protocol(cls, value: str) -> str:
        stripped = value.strip() or DEFAULT_PROTOCOL
        if stripped not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"不支持的 API 协议：{stripped}")
        return stripped

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        if not stripped:
            return None
        parts = urlsplit(stripped)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("API 地址需为 http(s) 地址")
        if parts.username or parts.password:
            raise ValueError("API 地址不能包含凭据")
        return stripped


class ModelSelection(BaseModel):
    """当前选中的「供应商 + 模型」（保存进 config.json，用于推导生效的 ``chat``）。"""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class BuiltinProvider:
    """预置供应商模板（「添加提供方」列表里的一项；不含任何密钥）。"""

    id: str
    name: str
    base_url: str
    summary: str
    requires_api_key: bool
    models: tuple[ProviderModelEntry, ...]
    docs_url: str | None = None

    def as_provider(self) -> ModelProvider:
        """把模板实例化为一个可保存的供应商（不预设密钥）。"""

        return ModelProvider(
            id=self.id,
            name=self.name,
            kind="builtin",
            base_url=self.base_url,
            models=[entry.model_copy() for entry in self.models],
        )

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "summary": self.summary,
            "requires_api_key": self.requires_api_key,
            "docs_url": self.docs_url,
            "models": [model_view(entry) for entry in self.models],
        }


#: 预置目录。**只预置官方 DeepSeek 模型**（本版内置能力表覆盖它们，开箱可用）；
#: 其它端点的模型名一律由用户自己填或「获取可用模型」拉取 —— 不替用户猜模型名。
#: 非官方端点的模型还需要在启动环境声明能力（``EVERYTHING_RAG_CHAT_CAPABILITIES``），
#: 否则问答会明确报「能力未声明」而不是静默降级（见 ``generation/capabilities.py``）。
BUILTIN_PROVIDERS: tuple[BuiltinProvider, ...] = (
    BuiltinProvider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        summary="官方端点；预置模型已内置能力声明，开箱可用",
        requires_api_key=True,
        docs_url="https://platform.deepseek.com/api_keys",
        models=(
            ProviderModelEntry(id="deepseek-flash", name="DeepSeek Flash", vision=True),
            ProviderModelEntry(id="deepseek-v4-pro", name="DeepSeek V4 Pro", vision=False),
        ),
    ),
    BuiltinProvider(
        id="siliconflow",
        name="硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        summary="OpenAI 兼容聚合端点；模型名需自行填写或拉取",
        requires_api_key=True,
        models=(),
    ),
    BuiltinProvider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        summary="OpenAI 官方端点；模型名需自行填写或拉取",
        requires_api_key=True,
        models=(),
    ),
    BuiltinProvider(
        id="ollama",
        name="Ollama（本地）",
        base_url="http://127.0.0.1:11434/v1",
        summary="本机 Ollama，无需 API 密钥；用「获取可用模型」列出已拉取的模型",
        requires_api_key=False,
        models=(),
    ),
)


def builtin_catalog() -> list[dict[str, Any]]:
    """预置目录视图（「添加提供方」列表用；零密钥、零 IO）。"""

    return [item.view() for item in BUILTIN_PROVIDERS]


def builtin_provider(builtin_id: str) -> ModelProvider | None:
    """按 id 取预置模板的实例；未知 id 返回 None（不猜）。"""

    for item in BUILTIN_PROVIDERS:
        if item.id == builtin_id:
            return item.as_provider()
    return None


def builtin_default_models(provider_id: str) -> list[ProviderModelEntry] | None:
    """预置模型目录（「恢复默认模型」用）；非预置供应商返回 None。"""

    for item in BUILTIN_PROVIDERS:
        if item.id == provider_id:
            return [entry.model_copy() for entry in item.models]
    return None


def model_display_name(entry: ProviderModelEntry) -> str:
    """模型显示名：填了显示名称用它，否则用真实模型 id（绝不显示空白）。"""

    return entry.name or entry.id


def _builtin_requires_key(provider_id: str) -> bool:
    """预置供应商是否声明需要密钥；**自定义供应商返回 False**（无从判断，不做假设）。

    自定义端点可能是本机无鉴权服务（llama.cpp / vLLM），因此「未填密钥」不当作缺配置；
    是否需要密钥由用户在编辑器里用「测试连接」得到真实答案。
    """

    for item in BUILTIN_PROVIDERS:
        if item.id == provider_id:
            return item.requires_api_key
    return False


def provider_status(provider: ModelProvider) -> Literal["ready", "needs_key", "unconfigured"]:
    """供应商配置完整度：``ready`` / ``needs_key`` / ``unconfigured``。

    判定只看「能否真的发起一次对话调用」的前置条件，不看用户是否填过：
    地址与模型目录齐备、且（不需要密钥 或 已设置密钥）时才是 ``ready``。
    """

    if not provider.base_url or not provider.models:
        return "unconfigured"
    if _builtin_requires_key(provider.id) and provider.api_key is None:
        return "needs_key"
    return "ready"


def find_model(
    providers: list[ModelProvider], provider_id: str | None, model_id: str | None
) -> tuple[ModelProvider, ProviderModelEntry] | None:
    """按选择查找（供应商 + 模型）；任一层缺失都返回 None。"""

    if not provider_id or not model_id:
        return None
    for provider in providers:
        if provider.id != provider_id:
            continue
        for entry in provider.models:
            if entry.id == model_id:
                return provider, entry
        return None
    return None


def chat_config_for(
    providers: list[ModelProvider], selection: ModelSelection | None
) -> Any | None:
    """把当前选择推导成运行期 ``ChatModelConfig``；选择无效返回 None（不兜底）。

    延迟 import ``ChatModelConfig``：它定义在 ``settings_store`` 里，而 ``settings_store``
    反过来要 import 本模块，模块级 import 会成环。
    """

    if selection is None:
        return None
    found = find_model(providers, selection.provider_id, selection.model_id)
    if found is None:
        return None
    provider, entry = found
    from app.core.settings_store import ChatModelConfig

    return ChatModelConfig(
        provider_type="openai_compatible",
        base_url=provider.base_url,
        model=entry.id,
        api_key=provider.api_key,
        supports_image=entry.vision,
    )


def model_view(entry: ProviderModelEntry) -> dict[str, Any]:
    """模型投影（无密钥字段，可直接进 API 响应）。"""

    return {"id": entry.id, "name": model_display_name(entry), "vision": entry.vision}


def provider_view(provider: ModelProvider, *, selected: ModelSelection | None = None) -> dict:
    """供应商脱敏投影：密钥只给「是否已设置 + 尾4位」。"""

    key = provider.api_key.get_secret_value() if provider.api_key is not None else None
    return {
        "id": provider.id,
        "name": provider.name,
        "kind": provider.kind,
        "protocol": provider.protocol,
        "base_url": provider.base_url,
        "api_key_set": key is not None,
        "api_key_hint": f"...{key[-4:]}" if key else None,
        "models": [model_view(entry) for entry in provider.models],
        "status": provider_status(provider),
        "selected": (
            selected is not None
            and selected.provider_id == provider.id
            and any(entry.id == selected.model_id for entry in provider.models)
        ),
    }


def providers_view(
    providers: list[ModelProvider], selection: ModelSelection | None = None
) -> list[dict]:
    return [provider_view(provider, selected=selection) for provider in providers]


def selection_view(
    providers: list[ModelProvider], selection: ModelSelection | None
) -> dict[str, Any] | None:
    """当前选择的显示投影（含供应商名与显示模型名）；无效选择返回 None。"""

    found = find_model(providers, selection.provider_id, selection.model_id) if selection else None
    if found is None or selection is None:
        return None
    provider, entry = found
    return {
        "provider_id": provider.id,
        "provider_name": provider.name,
        "model_id": entry.id,
        "model_name": model_display_name(entry),
        "vision": entry.vision,
    }


def next_custom_provider_id(providers: list[ModelProvider]) -> str:
    """生成一个未被占用的自定义供应商 id（``custom-1``、``custom-2``…）。"""

    used = {provider.id for provider in providers}
    index = 1
    while f"custom-{index}" in used:
        index += 1
    return f"custom-{index}"


def provider_from_legacy_chat(chat: Any) -> tuple[ModelProvider | None, ModelSelection | None]:
    """把旧的单段 ``chat`` 配置迁移成一个供应商 + 选择。

    只在配置里还**没有** ``providers`` 时使用（env 兜底同样走这里，让 .env 里配好的
    模型也出现在设置页，而不是「设置页空白但问答能用」）。什么都没配时返回
    ``(None, None)`` —— 不造一个空壳供应商。

    旧配置里的地址/模型名不合法时**不丢弃**：仍然迁移成一条记录（地址留空或模型目录
    留空），让用户在设置页里看见并修正，而不是静默消失。
    """

    base_url = (getattr(chat, "base_url", None) or "").strip() or None
    model = (getattr(chat, "model", None) or "").strip() or None
    if base_url is None and model is None:
        return None, None
    api_key = getattr(chat, "api_key", None)
    if api_key is not None and not isinstance(api_key, SecretStr):
        api_key = SecretStr(str(api_key))
    entry: ProviderModelEntry | None = None
    if model is not None:
        try:
            entry = ProviderModelEntry(
                id=model,
                name=None,
                vision=bool(getattr(chat, "supports_image", False)),
            )
        except ValidationError:
            entry = None
    try:
        provider = ModelProvider(
            id=LEGACY_PROVIDER_ID,
            name="现有配置",
            kind="custom",
            base_url=base_url,
            api_key=api_key,
            models=[entry] if entry is not None else [],
        )
    except ValidationError:
        # 地址格式在旧配置里就非法：保留记录但留空地址，等用户在设置页修正。
        provider = ModelProvider(
            id=LEGACY_PROVIDER_ID,
            name="现有配置",
            kind="custom",
            base_url=None,
            api_key=api_key,
            models=[entry] if entry is not None else [],
        )
    selection = (
        ModelSelection(provider_id=provider.id, model_id=entry.id) if entry is not None else None
    )
    return provider, selection
