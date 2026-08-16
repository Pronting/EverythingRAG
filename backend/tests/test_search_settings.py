"""联网搜索设置测试：SearchConfig 落盘/env 兜底 + is_search_configured + /api/settings 视图。

monkeypatch settings_store.get_settings 控制 env 值；API 测试走 conftest 的 tmp 存储。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core import settings_store as mod
from app.core.config import Settings
from app.core.settings_store import AppSettings, SearchConfig, SettingsStore, is_search_configured
from app.main import app


def _env_settings(**overrides) -> Settings:
    base = {
        "search_provider_type": "tavily",
        "search_enabled": True,
        "search_base_url": None,
        "search_api_key": SecretStr("sk-search-env"),
        "search_max_results": 7,
    }
    base.update(overrides)
    return Settings(**base)


def _store_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SettingsStore:
    monkeypatch.setattr(mod, "get_settings", lambda: _env_settings())
    return SettingsStore(tmp_path)


# ---------------------------------------------------------------- 1. 存储层


def test_search_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """无 config.json：search 回退 env；默认 provider=tavily。"""
    store = _store_with_env(monkeypatch, tmp_path)
    s = store.load()
    assert s.search.provider_type == "tavily"
    assert s.search.enabled is True
    assert s.search.api_key is not None
    assert s.search.api_key.get_secret_value() == "sk-search-env"
    assert s.search.max_results == 7


def test_search_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """search 保存 -> 落盘（明文 key 本地 gitignored）-> 重载一致。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.save(
        AppSettings(
            search=SearchConfig(
                provider_type="tavily",
                enabled=True,
                api_key=SecretStr("sk-search-custom"),
                max_results=5,
            )
        )
    )
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["search"]["enabled"] is True
    assert raw["search"]["api_key"] == "sk-search-custom"
    assert raw["search"]["max_results"] == 5
    assert store.load().search.api_key.get_secret_value() == "sk-search-custom"


def test_search_default_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """旧 config.json 无 search 字段 -> 默认 SearchConfig（disabled），不崩。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "chat": {"base_url": "http://x.test/v1", "model": "m", "api_key": None},
                "embed": {"mode": "cloud", "base_url": None, "model": None, "api_key": None},
                "system_prompt": "x",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    s = store.load()
    assert s.search.enabled is False
    assert s.search.provider_type == "tavily"


def test_is_search_configured() -> None:
    """可用性判定：需显式启用 + provider 凭证齐备。"""
    assert is_search_configured(SearchConfig(enabled=False, api_key=SecretStr("sk"))) is False
    assert is_search_configured(SearchConfig(enabled=True, api_key=None)) is False
    assert is_search_configured(SearchConfig(enabled=True, api_key=SecretStr("sk"))) is True
    assert (
        is_search_configured(SearchConfig(enabled=True, provider_type="searxng", base_url="http://x"))
        is True
    )
    assert is_search_configured(SearchConfig(enabled=True, provider_type="searxng")) is False


# ---------------------------------------------------------------- 2. API 层


def test_get_settings_includes_masked_search() -> None:
    """GET /api/settings 含 search 段（默认 disabled，key 只给 hint）。"""
    client = TestClient(app)
    data = client.get("/api/settings").json()
    assert "search" in data
    assert data["search"]["enabled"] is False
    assert data["search"]["provider_type"] == "tavily"
    assert data["search"]["api_key_set"] is False


def test_put_search_saves_and_masks() -> None:
    """PUT search -> 保存；回显 key 打码，绝不含明文。"""
    client = TestClient(app)
    resp = client.put(
        "/api/settings",
        json={
            "search": {
                "enabled": True,
                "provider_type": "tavily",
                "api_key": "sk-tavily-1234",
                "max_results": 5,
            }
        },
    )
    assert resp.status_code == 200
    data = resp.json()["search"]
    assert data["enabled"] is True
    assert data["api_key_set"] is True
    assert data["api_key_hint"] == "...1234"
    assert "sk-tavily-1234" not in resp.text

    got = client.get("/api/settings").json()["search"]
    assert got["enabled"] is True
    assert got["max_results"] == 5


def test_put_search_api_key_empty_clears() -> None:
    """search api_key 空串 -> 清除。"""
    client = TestClient(app)
    client.put("/api/settings", json={"search": {"api_key": "sk-secret"}})
    assert client.get("/api/settings").json()["search"]["api_key_set"] is True
    client.put("/api/settings", json={"search": {"api_key": ""}})
    assert client.get("/api/settings").json()["search"]["api_key_set"] is False


def test_put_search_invalid_base_url_422() -> None:
    """search base_url 非 http(s) -> 422。"""
    client = TestClient(app)
    resp = client.put("/api/settings", json={"search": {"base_url": "not-a-url"}})
    assert resp.status_code == 422
