"""多供应商模型配置测试（设置页「模型」+ 输入框模型选择器）。

覆盖三件事，每一件都对应一条不许违背的约定：

1. **注册表语义**（``core/model_providers.py``）：预置目录无密钥、选择 → ``chat`` 的映射
   只在选择有效时成立（无效即 ``None``，不猜第一个模型）；
2. **落盘与迁移**（``core/settings_store.py``）：旧单段 ``chat`` 首次读取迁移成一条供应商 +
   选择且 ``chat`` 逐字段不变；``save`` 把选择推导成 ``chat``；
3. **API 语义**（``api/routes/settings.py``）：``providers`` 整表替换 + 同名 id 密钥「省略即
   保留 / 空串清除」；删掉当前选中的供应商必须一并清空 ``chat``（不留悬空引用）；
   选择指向不存在的模型返回 422 且**不写盘**；「获取可用模型」经审计出网、错误不泄露密钥。

出网一律用替身：``list_available_models`` 被 monkeypatch，测试进程零真实外呼。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core import model_providers as mp
from app.core.settings_store import (
    AppSettings,
    ChatModelConfig,
    SettingsStore,
    get_settings_store,
)
from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def _store() -> SettingsStore:
    return app.dependency_overrides[get_settings_store]()


def _config_path() -> Path:
    return _store().path


# --------------------------------------------------------------------------------------
# 1. 注册表语义
# --------------------------------------------------------------------------------------


def test_builtin_catalog_has_no_secrets_and_unique_ids() -> None:
    """预置目录：id 唯一、都有地址、都不含密钥字段。"""

    catalog = mp.builtin_catalog()
    assert catalog, "预置目录不能为空"
    ids = [item["id"] for item in catalog]
    assert len(ids) == len(set(ids))
    for item in catalog:
        assert item["base_url"].startswith("http")
        assert "api_key" not in item
        assert isinstance(item["requires_api_key"], bool)
    assert mp.builtin_provider("deepseek") is not None
    assert mp.builtin_provider("no-such-provider") is None


def test_chat_config_maps_selection_to_effective_chat() -> None:
    """选择 → ChatModelConfig：地址/模型/密钥/多模态逐字段来自选中的那条记录。"""

    provider = mp.ModelProvider(
        id="custom-1",
        name="自定义",
        base_url="https://api.example.com/v1",
        api_key=SecretStr("sk-secret"),
        models=[mp.ProviderModelEntry(id="m-vision", name="带视觉", vision=True)],
    )
    config = mp.chat_config_for([provider], mp.ModelSelection(provider_id="custom-1", model_id="m-vision"))
    assert config is not None
    assert config.base_url == "https://api.example.com/v1"
    assert config.model == "m-vision"
    assert config.api_key is not None and config.api_key.get_secret_value() == "sk-secret"
    assert config.supports_image is True


def test_chat_config_is_none_for_unknown_or_empty_selection() -> None:
    """无效选择返回 None（绝不回落到「第一个模型」冒充用户选择）。"""

    provider = mp.ModelProvider(id="p", name="P", base_url="https://a.test/v1")
    assert mp.chat_config_for([provider], None) is None
    assert mp.chat_config_for([provider], mp.ModelSelection(provider_id="p", model_id="x")) is None
    assert mp.chat_config_for([provider], mp.ModelSelection(provider_id="q", model_id="x")) is None


def test_provider_status_reflects_configuration_completeness() -> None:
    """状态判定：缺地址/缺模型目录 -> unconfigured；预置需密钥但没填 -> needs_key。"""

    empty = mp.ModelProvider(id="deepseek", name="DeepSeek", base_url=None)
    assert mp.provider_status(empty) == "unconfigured"
    no_models = mp.ModelProvider(id="deepseek", name="DeepSeek", base_url="https://api.deepseek.com/v1")
    assert mp.provider_status(no_models) == "unconfigured"
    needs_key = mp.ModelProvider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        models=[mp.ProviderModelEntry(id="deepseek-flash")],
    )
    assert mp.provider_status(needs_key) == "needs_key"
    ready = needs_key.model_copy(update={"api_key": SecretStr("sk-1")})
    assert mp.provider_status(ready) == "ready"
    # 自定义端点不假设需要密钥（本机无鉴权服务同样可用）
    custom = mp.ModelProvider(
        id="custom-1",
        name="自定义",
        base_url="http://127.0.0.1:8080/v1",
        models=[mp.ProviderModelEntry(id="local-model")],
    )
    assert mp.provider_status(custom) == "ready"


def test_provider_view_masks_api_key() -> None:
    """视图只给 api_key_set + 尾4位，绝不含明文。"""

    provider = mp.ModelProvider(
        id="custom-1",
        name="自定义",
        base_url="https://a.test/v1",
        api_key=SecretStr("sk-do-not-leak-1234"),
        models=[mp.ProviderModelEntry(id="m", name="模型")],
    )
    view = mp.provider_view(provider, selected=mp.ModelSelection(provider_id="custom-1", model_id="m"))
    assert view["api_key_set"] is True
    assert view["api_key_hint"] == "...1234"
    assert view["selected"] is True
    assert "sk-do-not-leak-1234" not in json.dumps(view, ensure_ascii=False)


def test_selection_view_names_provider_and_model() -> None:
    """选择投影给出供应商名与显示模型名（界面据此渲染触发器）。"""

    provider = mp.ModelProvider(
        id="custom-1",
        name="commande code",
        base_url="https://api.commandcode.ai/provider/v1",
        models=[mp.ProviderModelEntry(id="deepseek/deepseek-v4.1-flash", name="DeepSeek V4.1 Flash")],
    )
    view = mp.selection_view([provider], mp.ModelSelection(provider_id="custom-1", model_id="deepseek/deepseek-v4.1-flash"))
    assert view == {
        "provider_id": "custom-1",
        "provider_name": "commande code",
        "model_id": "deepseek/deepseek-v4.1-flash",
        "model_name": "DeepSeek V4.1 Flash",
        "vision": False,
    }
    assert mp.selection_view([provider], None) is None


def test_next_custom_provider_id_avoids_collisions() -> None:
    existing = [mp.ModelProvider(id="custom-1", name="a"), mp.ModelProvider(id="custom-2", name="b")]
    assert mp.next_custom_provider_id(existing) == "custom-3"
    assert mp.next_custom_provider_id([]) == "custom-1"


def test_provider_model_rejects_invalid_values() -> None:
    """非法 id / 协议 / 地址必须在构造期就被拒绝（fail-closed）。"""

    with pytest.raises(ValueError):
        mp.ModelProvider(id="有中文", name="x")
    with pytest.raises(ValueError):
        mp.ModelProvider(id="ok", name="x", protocol="anthropic-messages")
    with pytest.raises(ValueError):
        mp.ModelProvider(id="ok", name="x", base_url="ftp://a.test")
    with pytest.raises(ValueError):
        mp.ModelProvider(id="ok", name="x", base_url="https://user:pw@a.test/v1")


def test_provider_from_legacy_chat_keeps_everything() -> None:
    """旧 chat → 供应商 + 选择：地址/模型/多模态逐字段保留。"""

    provider, selection = mp.provider_from_legacy_chat(
        ChatModelConfig(
            base_url="https://api.example.com/v1",
            model="legacy-model",
            api_key=SecretStr("sk-legacy"),
            supports_image=True,
        )
    )
    assert provider is not None and selection is not None
    assert provider.id == mp.LEGACY_PROVIDER_ID
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.models[0].id == "legacy-model"
    assert provider.models[0].vision is True
    assert selection.model_id == "legacy-model"
    # 什么都没配 -> 不造空壳
    assert mp.provider_from_legacy_chat(ChatModelConfig()) == (None, None)
    # 地址非法 -> 仍然迁移出一条记录（地址留空），用户可在设置页修正
    broken, broken_selection = mp.provider_from_legacy_chat(
        ChatModelConfig(base_url="not-a-url", model="m")
    )
    assert broken is not None and broken.base_url is None
    assert broken_selection is not None and broken_selection.model_id == "m"


# --------------------------------------------------------------------------------------
# 2. 落盘与迁移
# --------------------------------------------------------------------------------------


def test_legacy_config_migrates_to_provider_and_selection() -> None:
    """旧 config.json（只有 chat）首次读取：chat 逐字段不变 + 多出一条供应商与选择。"""

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "chat": {
                    "provider_type": "openai_compatible",
                    "base_url": "https://api.legacy.test/v1",
                    "model": "legacy-model",
                    "api_key": "sk-legacy",
                    "supports_image": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = _store().load()
    assert loaded.chat.base_url == "https://api.legacy.test/v1"
    assert loaded.chat.model == "legacy-model"
    assert loaded.chat.supports_image is True
    assert [item.id for item in loaded.providers] == [mp.LEGACY_PROVIDER_ID]
    assert loaded.model_selection is not None
    assert loaded.model_selection.model_id == "legacy-model"
    # 第二次读取走落盘后的新形态，结果必须一致（迁移可重复、不漂移）
    again = _store().load()
    assert again.chat.model == "legacy-model"
    assert [item.id for item in again.providers] == [mp.LEGACY_PROVIDER_ID]


def test_save_derives_chat_from_selection(tmp_path: Path) -> None:
    """save 把选择推导成 chat（落盘值 = 生效值）。"""

    store = SettingsStore(tmp_path)
    settings = AppSettings(
        chat=ChatModelConfig(base_url="https://old.test/v1", model="old-model"),
        providers=[
            mp.ModelProvider(
                id="custom-1",
                name="自定义",
                base_url="https://new.test/v1",
                api_key=SecretStr("sk-new"),
                models=[mp.ProviderModelEntry(id="new-model", vision=True)],
            )
        ],
        model_selection=mp.ModelSelection(provider_id="custom-1", model_id="new-model"),
    )
    store.save(settings)
    loaded = store.load()
    assert loaded.chat.base_url == "https://new.test/v1"
    assert loaded.chat.model == "new-model"
    assert loaded.chat.supports_image is True
    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert payload["chat"]["model"] == "new-model"
    assert payload["providers"][0]["api_key"] == "sk-new"
    assert payload["model_selection"] == {"provider_id": "custom-1", "model_id": "new-model"}


def test_env_defaults_expose_env_model_as_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """env 里配好的模型同样出现在设置页（不是「空白但问答能用」）。"""

    from app.core import settings_store as store_mod
    from app.core.config import settings as env_settings

    monkeypatch.setattr(env_settings, "chat_base_url", "https://env.test/v1")
    monkeypatch.setattr(env_settings, "chat_model", "env-model")
    monkeypatch.setattr(env_settings, "chat_api_key", SecretStr("sk-env"))
    loaded = store_mod.SettingsStore(tmp_path / "empty").load()
    assert loaded.chat.model == "env-model"
    assert len(loaded.providers) == 1
    assert loaded.providers[0].base_url == "https://env.test/v1"
    assert loaded.model_selection is not None and loaded.model_selection.model_id == "env-model"


# --------------------------------------------------------------------------------------
# 3. API 语义
# --------------------------------------------------------------------------------------


def _provider_payload(**overrides: Any) -> dict:
    payload: dict[str, Any] = {
        "id": "custom-1",
        "name": "commande code",
        "kind": "custom",
        "protocol": "openai-completions",
        "base_url": "https://api.commandcode.ai/provider/v1",
        "models": [
            {"id": "deepseek/deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash", "vision": True}
        ],
    }
    payload.update(overrides)
    return payload


def test_selected_model_reaches_the_real_chat_provider_factory() -> None:
    """选中的模型必须能被真实 provider 工厂构造（只有地址/模型/密钥齐备才构造成功）。

    这是「界面选的模型真的会被问答链路使用」的最短真实证据：工厂是 run 管线
    （General / 综合节点）唯一使用的构造入口，构造成功即代表配置可用；本用例零出网
    （`create_chat_model_from_config` 只做构造期校验，不发请求）。
    """

    from app.generation.providers import create_chat_model_from_config

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [
                _provider_payload(
                    base_url="https://api.deepseek.com/v1",
                    api_key="sk-deepseek-0001",
                    models=[{"id": "deepseek-flash", "name": "DeepSeek Flash", "vision": True}],
                )
            ],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek-flash"},
        },
    )
    config = _store().load().chat
    model = create_chat_model_from_config(config)
    assert model.model == "deepseek-flash"
    # /api/status 的 chat_configured 也是同一份 chat 推出的（界面与状态不各说一套）
    assert client.get("/api/status").json()["config"]["chat_configured"] is True
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.supports_image is True


def test_get_settings_exposes_providers_catalog_and_selection() -> None:
    client = _client()
    data = client.get("/api/settings").json()
    assert isinstance(data["providers"], list)
    assert data["model_catalog"], "预置目录必须随视图返回"
    assert data["model_protocols"] == list(mp.SUPPORTED_PROTOCOLS)
    # 测试环境没有 config.json，走 env 兜底：env 里若配了对话模型，就必须**已经**投影成
    # 一条供应商 + 选择（否则设置页会「空白但问答能用」）；没配则两者都为空。
    if data["model_selection"] is None:
        assert data["providers"] == []
    else:
        assert data["providers"], "有选择就必须有对应供应商"
        assert data["model_selection"]["model_id"] == data["chat"]["model"]
        assert data["chat"]["base_url"]
        assert "\"api_key\":" not in json.dumps(data["providers"], ensure_ascii=False)


def test_put_providers_and_selection_updates_effective_chat() -> None:
    """保存供应商 + 选择后：chat 立即等于选中的模型（问答链路读的就是它）。"""

    client = _client()
    resp = client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-command-1234")],
            "model_selection": {
                "provider_id": "custom-1",
                "model_id": "deepseek/deepseek-v4.1-flash",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chat"]["base_url"] == "https://api.commandcode.ai/provider/v1"
    assert data["chat"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert data["chat"]["supports_image"] is True
    assert data["model_selection"]["model_name"] == "DeepSeek V4.1 Flash"
    assert data["providers"][0]["selected"] is True
    assert "sk-command-1234" not in resp.text

    # 重新 GET 也必须一致（不是只在响应里改了一下）
    got = client.get("/api/settings").json()
    assert got["chat"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert got["providers"][0]["api_key_hint"] == "...1234"


def test_put_providers_keeps_saved_key_when_omitted_and_clears_on_empty_string() -> None:
    """整表替换的密钥语义：省略 = 保留；空串 = 清除。"""

    client = _client()
    client.put("/api/settings", json={"providers": [_provider_payload(api_key="sk-keep-9999")]})
    # 省略 api_key（界面回填时不回传密钥）-> 保留
    resp = client.put("/api/settings", json={"providers": [_provider_payload(name="改名后")]})
    assert resp.status_code == 200
    assert resp.json()["providers"][0]["api_key_set"] is True
    assert resp.json()["providers"][0]["api_key_hint"] == "...9999"
    # 空串 -> 清除
    resp = client.put("/api/settings", json={"providers": [_provider_payload(api_key="")]})
    assert resp.status_code == 200
    assert resp.json()["providers"][0]["api_key_set"] is False


def test_put_providers_replaces_the_whole_list() -> None:
    """省略的供应商即被删除（整表替换语义）。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(), _provider_payload(id="custom-2", name="另一个")],
        },
    )
    resp = client.put("/api/settings", json={"providers": [_provider_payload()]})
    assert [item["id"] for item in resp.json()["providers"]] == ["custom-1"]


def test_deleting_selected_provider_clears_selection_and_chat() -> None:
    """删掉当前选中的供应商：选择清空 + chat 清空（不留悬空引用）。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-1")],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek/deepseek-v4.1-flash"},
        },
    )
    resp = client.put("/api/settings", json={"providers": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["providers"] == []
    assert data["model_selection"] is None
    assert data["chat"]["base_url"] is None
    assert data["chat"]["model"] is None
    assert client.get("/api/settings").json()["chat"]["model"] is None


def test_deleting_selected_model_clears_selection_and_chat() -> None:
    """删掉被选中的那个模型（供应商还在）同样清空选择与 chat。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-1")],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek/deepseek-v4.1-flash"},
        },
    )
    resp = client.put(
        "/api/settings",
        json={"providers": [_provider_payload(models=[{"id": "另一个模型", "vision": False}])]},
    )
    assert resp.status_code == 200
    assert resp.json()["model_selection"] is None
    assert resp.json()["chat"]["model"] is None


def test_put_selection_unknown_model_returns_422_without_writing() -> None:
    """选择指向不存在的模型：422 且旧设置不变（不静默回落）。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-1")],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek/deepseek-v4.1-flash"},
        },
    )
    resp = client.put(
        "/api/settings",
        json={"model_selection": {"provider_id": "custom-1", "model_id": "不存在"}},
    )
    assert resp.status_code == 422
    got = client.get("/api/settings").json()
    assert got["model_selection"]["model_id"] == "deepseek/deepseek-v4.1-flash"
    assert got["chat"]["model"] == "deepseek/deepseek-v4.1-flash"


def test_put_clearing_selection_keeps_chat() -> None:
    """显式清除选择（两个字段都空）只解除绑定，不动已经生效的 chat。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-1")],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek/deepseek-v4.1-flash"},
        },
    )
    resp = client.put("/api/settings", json={"model_selection": {"provider_id": None, "model_id": None}})
    assert resp.status_code == 200
    assert resp.json()["model_selection"] is None
    assert resp.json()["chat"]["model"] == "deepseek/deepseek-v4.1-flash"


def test_put_providers_rejects_invalid_payloads() -> None:
    """整表替换的校验：id 重复 / 地址非法 / 协议不支持 / 模型 id 重复都 422。"""

    client = _client()
    duplicates = [_provider_payload(), _provider_payload()]
    assert client.put("/api/settings", json={"providers": duplicates}).status_code == 422
    assert (
        client.put(
            "/api/settings", json={"providers": [_provider_payload(base_url="ftp://a.test")]}
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/settings", json={"providers": [_provider_payload(protocol="anthropic-messages")]}
        ).status_code
        == 422
    )
    same_models = [
        _provider_payload(models=[{"id": "m"}, {"id": "m"}]),
    ]
    assert client.put("/api/settings", json={"providers": same_models}).status_code == 422


def test_put_chat_directly_clears_stale_selection() -> None:
    """旧的直接配置 chat 的写法仍然可用：生效的是它，且不会留下与 chat 不一致的选择。"""

    client = _client()
    client.put(
        "/api/settings",
        json={
            "providers": [_provider_payload(api_key="sk-1")],
            "model_selection": {"provider_id": "custom-1", "model_id": "deepseek/deepseek-v4.1-flash"},
        },
    )
    resp = client.put(
        "/api/settings",
        json={"chat": {"base_url": "https://other.test/v1", "model": "other-model", "api_key": "sk-2"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chat"]["model"] == "other-model"
    assert data["model_selection"] is None


def test_list_provider_models_returns_ids_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """「获取可用模型」：走注入的替身出网，响应只含模型 id。"""

    from app.api.routes import settings as settings_routes

    calls: list[tuple[str, str | None]] = []

    async def fake_list(base_url: str, api_key: str | None) -> dict:
        calls.append((base_url, api_key))
        return {
            "success": True,
            "message": "获取到 2 个模型",
            "models": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}],
            "count": 2,
            "latency_ms": 12,
        }

    monkeypatch.setattr("app.generation.connection_test.list_available_models", fake_list)
    client = _client()
    resp = client.post(
        "/api/settings/providers/models",
        json={
            "provider_id": "custom-1",
            "base_url": "https://api.commandcode.ai/provider/v1",
            "api_key": "sk-test-4321",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [item["id"] for item in data["models"]] == ["deepseek-chat", "deepseek-reasoner"]
    assert data["count"] == 2
    assert "sk-test-4321" not in resp.text
    assert calls == [("https://api.commandcode.ai/provider/v1", "sk-test-4321")]
    assert settings_routes  # 路由模块已在本次请求中真实参与


def test_list_provider_models_reuses_saved_key_only_for_same_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """省略密钥时只在地址与已保存供应商一致时复用（换地址不复用）。"""

    calls: list[str | None] = []

    async def fake_list(base_url: str, api_key: str | None) -> dict:
        calls.append(api_key)
        return {"success": True, "models": [], "count": 0, "message": "ok"}

    monkeypatch.setattr("app.generation.connection_test.list_available_models", fake_list)
    client = _client()
    client.put("/api/settings", json={"providers": [_provider_payload(api_key="sk-saved-7777")]})
    client.post(
        "/api/settings/providers/models",
        json={"provider_id": "custom-1", "base_url": "https://api.commandcode.ai/provider/v1"},
    )
    client.post(
        "/api/settings/providers/models",
        json={"provider_id": "custom-1", "base_url": "https://attacker.test/v1"},
    )
    assert calls == ["sk-saved-7777", None]


def test_list_provider_models_rejects_invalid_url_without_outbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法地址直接拒绝，且**不发起任何出网**（隐私红线）。"""

    called = False

    async def fake_list(base_url: str, api_key: str | None) -> dict:  # pragma: no cover - 不该被调用
        nonlocal called
        called = True
        return {"success": True, "models": [], "count": 0, "message": "ok"}

    monkeypatch.setattr("app.generation.connection_test.list_available_models", fake_list)
    client = _client()
    for payload in (
        {"base_url": "ftp://a.test/v1"},
        {"base_url": "https://user:pw@a.test/v1"},
        {"base_url": "https://a.test/v1?token=1"},
    ):
        resp = client.post("/api/settings/providers/models", json=payload)
        assert resp.status_code == 200
        assert resp.json()["success"] is False
    assert called is False
