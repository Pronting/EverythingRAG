"""设置 API 测试：GET 脱敏视图 / PUT 局部保存 / 校验 422 / api_key 清除。

conftest 注入 tmp 设置存储（不触碰真实 data_dir/config.json）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_get_settings_masks_api_key() -> None:
    """GET /api/settings：key 只回显 api_key_set + hint，绝不回显明文。"""
    client = _client()
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "chat" in data and "embed" in data and "system_prompt" in data
    body = resp.text
    assert "api_key_set" in data["chat"]
    assert data["chat"]["api_key_hint"] is None or data["chat"]["api_key_hint"].startswith("...")
    assert "sk-" not in body or "sk-env" not in body  # 无明文 key 回显


def test_put_saves_chat_and_returns_masked() -> None:
    """PUT chat 配置 -> 保存；GET 反映新值（key 打码）。"""
    client = _client()
    resp = client.put(
        "/api/settings",
        json={"chat": {"base_url": "http://custom.test/v1", "model": "custom-model", "api_key": "sk-custom"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chat"]["base_url"] == "http://custom.test/v1"
    assert data["chat"]["model"] == "custom-model"
    assert data["chat"]["api_key_set"] is True
    assert data["chat"]["api_key_hint"] == "...stom"
    assert "sk-custom" not in resp.text

    got = client.get("/api/settings").json()
    assert got["chat"]["base_url"] == "http://custom.test/v1"
    assert got["chat"]["api_key_set"] is True


def test_put_saves_system_prompt() -> None:
    """PUT system_prompt -> 保存并回读。"""
    client = _client()
    prompt = "你是我的专属助手，只用中文回答。"
    resp = client.put("/api/settings", json={"system_prompt": prompt})
    assert resp.status_code == 200
    assert client.get("/api/settings").json()["system_prompt"] == prompt


def test_put_api_key_empty_clears() -> None:
    """api_key 传空串 -> 清除（api_key_set=False）。"""
    client = _client()
    client.put("/api/settings", json={"chat": {"api_key": "sk-secret"}})
    assert client.get("/api/settings").json()["chat"]["api_key_set"] is True

    resp = client.put("/api/settings", json={"chat": {"api_key": ""}})
    assert resp.status_code == 200
    assert client.get("/api/settings").json()["chat"]["api_key_set"] is False


def test_put_embed_cloud_requires_fields() -> None:
    """mode=cloud 但缺 base_url/model -> 422。"""
    client = _client()
    resp = client.put("/api/settings", json={"embed": {"mode": "cloud"}})
    assert resp.status_code == 422


def test_put_invalid_base_url_422() -> None:
    """base_url 非 http(s) -> 422。"""
    client = _client()
    resp = client.put("/api/settings", json={"chat": {"base_url": "not-a-url"}})
    assert resp.status_code == 422


def test_put_partial_preserves_other_fields() -> None:
    """只改 embed.mode 不影响 chat。"""
    client = _client()
    client.put("/api/settings", json={"chat": {"model": "keep-me"}})
    client.put("/api/settings", json={"embed": {"mode": "cloud", "base_url": "http://e.test/v1", "model": "m"}})
    data = client.get("/api/settings").json()
    assert data["chat"]["model"] == "keep-me"
    assert data["embed"]["mode"] == "cloud"


def test_get_settings_default_theme_light() -> None:
    """默认主题为浅色。"""
    client = _client()
    assert client.get("/api/settings").json()["theme"] == "light"


def test_put_theme_saves_and_returns() -> None:
    """PUT theme=dark -> 保存并回显 dark。"""
    client = _client()
    resp = client.put("/api/settings", json={"theme": "dark"})
    assert resp.status_code == 200
    assert resp.json()["theme"] == "dark"
    assert client.get("/api/settings").json()["theme"] == "dark"
