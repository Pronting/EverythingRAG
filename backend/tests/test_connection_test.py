from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIStatusError

from app.generation import connection_test as module
from app.main import app


def test_probe_uses_draft_and_saved_key_without_saving(monkeypatch):
    client = TestClient(app)
    client.put("/api/settings", json={"chat": {
        "base_url": "https://saved.test/v1", "model": "saved", "api_key": "secret",
    }})
    calls = []
    async def probe(*args):
        calls.append(args)
        return {"success": True, "message": "连接成功", "latency_ms": 10}
    monkeypatch.setattr(module, "probe_model", probe)
    for url, key, expected in [
        ("https://saved.test/v1", None, "secret"),
        ("https://changed.test/v1", None, None),
        ("https://saved.test/v1", "", ""),
        ("https://saved.test/v1", "new-secret", "new-secret"),
    ]:
        response = client.post("/api/settings/test-connection", json={
            "kind": "chat", "base_url": url, "model": "draft", "api_key": key,
        })
        assert response.json()["success"]
        assert calls[-1] == ("chat", url, "draft", expected)
        assert "secret" not in response.text
    assert client.get("/api/settings").json()["chat"]["model"] == "saved"


@pytest.mark.parametrize("url", ["", "ftp://bad", "http://[", "https://user:password@host"])
def test_invalid_probe_url_does_not_call_provider(monkeypatch, url):
    async def forbidden(*args):
        raise AssertionError("must not call provider")
    monkeypatch.setattr(module, "probe_model", forbidden)
    result = TestClient(app).post("/api/settings/test-connection", json={
        "kind": "chat", "base_url": url, "model": "test",
    })
    assert result.status_code == 200
    assert not result.json()["success"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "embed", "vision"])
async def test_probe_calls_actual_capability_and_closes_client(monkeypatch, kind):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
            self.embeddings = self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            calls.append("closed")
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])], choices=[1])
    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    result = await module.probe_model(kind, "https://test/v1", "chosen-model", "secret")
    assert result["success"]
    assert calls[0]["model"] == "chosen-model"
    assert calls[-1] == "closed"
    if kind == "vision":
        assert calls[0]["messages"][0]["content"][1]["type"] == "image_url"
    if kind == "embed":
        assert calls[0]["input"] == ["Connection test"]


@pytest.mark.asyncio
async def test_probe_masks_provider_error(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            self.embeddings = self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def create(self, **kwargs):
            response = httpx.Response(401, request=httpx.Request("POST", "https://test/v1"))
            raise APIStatusError("secret credential", response=response, body=None)
    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    result = await module.probe_model("embed", "https://test/v1", "model", "secret")
    assert not result["success"]
    assert "secret" not in str(result)
    assert "API Key" in result["message"]
