"""POST /api/chat SSE 端点测试（TestClient + dependency_overrides 注入 fake，零出网）。

隐私红线：端点在测试中全部替换为 fake 依赖，不触发真实嵌入 / 检索 / 云端模型；
另有阻断一切 socket 连接的用例实证全链路零出网。
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.chat import get_chat_service
from app.generation.chat_service import ChatService
from app.generation.providers import ChatProviderError, UnconfiguredChatModel
from app.main import app
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]]


class FakeRetriever:
    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else []
        self._error = error

    def retrieve(
        self,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._error is not None:
            raise self._error
        return self._chunks


class FakeChatModel:
    def __init__(
        self,
        tokens: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._tokens = tokens if tokens is not None else ["你", "好"]
        self._error = error

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        if self._error is not None:
            raise self._error
        for token in self._tokens:
            yield token


def _chunk(
    block_id: str = "b1",
    text: str = "第一块内容",
    similarity: float = 0.92,
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=block_id,
        text=text,
        similarity=similarity,
        source_file="notes.md",
        platform="kimi",
        chunk_type="text",
        anchor="a1",
        heading_path="s1",
        metadata={"source_file": "notes.md"},
    )


def _override(service: ChatService) -> None:
    """用 fake service 覆盖 get_chat_service 依赖（fastapi 规则：覆盖精确依赖）。"""
    app.dependency_overrides[get_chat_service] = lambda: service


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """解析 SSE body：每行 'data: {json}' -> 帧 dict 列表。"""
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _stream_body(client: TestClient, message: str) -> tuple[int, str]:
    """POST /api/chat 并以流方式读回完整 body，返回 (status_code, body)。"""
    with client.stream("POST", "/api/chat", json={"message": message}) as resp:
        return resp.status_code, resp.read().decode("utf-8")


# ---------------------------------------------------------------- 1. 帧顺序


def test_chat_sse_frame_order() -> None:
    """meta -> token -> token -> done；meta.sources 含检索来源字段（similarity 保留 float）。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk()]),
        FakeChatModel(tokens=["你", "好"]),
    )
    _override(service)
    try:
        client = TestClient(app)
        status, body = _stream_body(client, "你好")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["meta", "token", "token", "done"]

    source = frames[0]["sources"][0]
    assert source["block_id"] == "b1"
    assert source["text"] == "第一块内容"
    assert source["source_file"] == "notes.md"
    assert source["platform"] == "kimi"
    assert source["chunk_type"] == "text"
    assert source["heading_path"] == "s1"
    assert source["anchor"] == "a1"
    assert source["similarity"] == pytest.approx(0.92)
    assert isinstance(source["similarity"], float)

    assert frames[1] == {"type": "token", "text": "你"}
    assert frames[2] == {"type": "token", "text": "好"}
    assert frames[3] == {"type": "done"}


# ---------------------------------------------------------------- 2. 错误路径不崩


def test_retrieval_error_returns_single_error_frame() -> None:
    """检索抛 RetrievalError -> 仅 error 帧，message 含「未建索引」，HTTP 200（SSE 流式错误）。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(error=RetrievalError("知识库尚未建立索引，请先导入并同步文档")),
        FakeChatModel(),
    )
    _override(service)
    try:
        client = TestClient(app)
        status, body = _stream_body(client, "你好")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["error"]
    assert "尚未建立索引" in frames[0]["message"]


def test_chat_model_error_returns_single_error_frame() -> None:
    """对话模型抛异常 -> 仅 error 帧，消息脱敏不泄漏异常正文。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk()]),
        FakeChatModel(error=RuntimeError("boom-internal")),
    )
    _override(service)
    try:
        client = TestClient(app)
        status, body = _stream_body(client, "你好")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["error"]
    assert "boom-internal" not in frames[0]["message"]


def test_unconfigured_provider_returns_error_frame() -> None:
    """provider 未配置（占位模型）-> 仅 error 帧，消息列出缺失的环境变量名。"""
    error = ChatProviderError(
        "对话 provider 未配置，缺少环境变量: EVERYTHING_RAG_CHAT_MODEL"
    )
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk()]),
        UnconfiguredChatModel(error),
    )
    _override(service)
    try:
        client = TestClient(app)
        status, body = _stream_body(client, "你好")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["error"]
    assert "EVERYTHING_RAG_CHAT_MODEL" in frames[0]["message"]


def test_empty_message_returns_422() -> None:
    """空消息 -> FastAPI 请求校验 422（非 500）。"""
    _override(ChatService(FakeEmbedder(), FakeRetriever(), FakeChatModel()))
    try:
        client = TestClient(app)
        resp = client.post("/api/chat", json={"message": ""})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 422


# ---------------------------------------------------------------- 3. 零出网


def test_full_chain_zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环网络连接，SSE 全链路（fake 依赖）仍成功 => 零出网。

    TestClient 的 anyio 事件循环在 Windows 依赖回环自环 socket（socketpair），
    故仅放行回环；任何真实外呼（非回环目的地）即判定失败。
    """
    _block_non_loopback(monkeypatch)
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk()]),
        FakeChatModel(tokens=["好"]),
    )
    _override(service)
    try:
        client = TestClient(app)
        status, body = _stream_body(client, "你好")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["meta", "token", "done"]


def _block_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接（真实外呼）；回环目的地放行（事件循环自环所需）。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    original_connect = socket.socket.connect

    def guarded_connect(
        self: socket.socket,
        address: tuple | str | bytes | int,
        *args: object,
        **kwargs: object,
    ) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Unexpected outbound network call to {host!r}")
        return original_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
