"""会话 API 测试：列表 / 新建 / 读取 / 追加 / 标题（AI 生成用 fake 模型，零出网）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import conversations as conversations_mod
from app.main import app


def _client() -> TestClient:
    return TestClient(app)


class FakeTitleModel:
    """对话模型替身：stream 产出「测试标题」token。"""

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list, **kwargs: object):  # type: ignore[no-untyped-def]
        for token in ("测试", "标题"):
            yield token


def test_list_empty_and_create() -> None:
    client = _client()
    assert client.get("/api/conversations").json() == []

    resp = client.post("/api/conversations")
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"]
    assert data["title"] == "新对话"
    assert len(client.get("/api/conversations").json()) == 1


def test_get_unknown_404() -> None:
    assert _client().get("/api/conversations/nope").status_code == 404


def test_append_messages_and_read_back() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]

    resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["message_count"] == 2

    conv = client.get(f"/api/conversations/{conv_id}").json()
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["content"] == "你好"


def test_set_title_manual() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    resp = client.put(f"/api/conversations/{conv_id}/title", json={"title": "手动标题"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "手动标题"


def test_generate_title_uses_chat_model() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "如何配置对话模型"}]},
    )

    # 用 fake 对话模型替换（避免真实出网）
    conversations_mod.create_chat_model_from_config = lambda _chat: FakeTitleModel()
    try:
        resp = client.post(f"/api/conversations/{conv_id}/generate-title")
        assert resp.status_code == 200
        assert resp.json()["title"] == "测试标题"
        assert client.get(f"/api/conversations/{conv_id}").json()["title"] == "测试标题"
    finally:
        # 还原（其他用例需要真实实现）
        from app.generation import providers

        conversations_mod.create_chat_model_from_config = providers.create_chat_model_from_config


def test_generate_title_empty_conv_fallback() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    resp = client.post(f"/api/conversations/{conv_id}/generate-title")
    assert resp.status_code == 200
    assert resp.json()["title"] == "新对话"


def test_delete_conversation() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    assert client.delete(f"/api/conversations/{conv_id}").status_code == 204
    assert client.get(f"/api/conversations/{conv_id}").status_code == 404
    assert client.delete(f"/api/conversations/{conv_id}").status_code == 404
