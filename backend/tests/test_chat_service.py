"""ChatService 单测：meta/token/done 帧序列 + 各类异常转 error 帧（fake 依赖，零出网）。

隐私红线：全部 fake 依赖（嵌入/检索/对话），无 socket/requests 调用；
错误消息不得泄漏异常正文 / 文件路径 / 密钥。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from app.generation.chat_service import ChatService
from app.generation.providers import ChatProviderError
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self._error is not None:
            raise self._error
        return [[0.1, 0.2, 0.3, 0.4]]


class FakeRetriever:
    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else []
        self._error = error
        self.calls: list[list[float]] = []

    def retrieve(
        self,
        query_vector: list[float],
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(query_vector)
        if self._error is not None:
            raise self._error
        return self._chunks


class FakeChatModel:
    def __init__(
        self,
        tokens: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._tokens = tokens if tokens is not None else ["tok1", "tok2"]
        self._error = error
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        for token in self._tokens:
            yield token


def _chunk(
    block_id: str,
    text: str = "块内容",
    similarity: float = 0.9,
    source_file: str = "notes.md",
    platform: str = "kimi",
    chunk_type: str = "text",
    anchor: str | None = "a1",
    heading_path: str | None = "s1",
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=block_id,
        text=text,
        similarity=similarity,
        source_file=source_file,
        platform=platform,
        chunk_type=chunk_type,
        anchor=anchor,
        heading_path=heading_path,
        metadata={"source_file": source_file},
    )


async def _collect(service: ChatService, message: str) -> list[dict[str, Any]]:
    return [frame async for frame in service.stream_answer(message)]


# ---------------------------------------------------------------- 1. meta -> token(s) -> done


async def test_meta_token_done_frame_sequence() -> None:
    """正常链路：meta（含来源）-> 逐 token -> done；消息带来源序号上下文。"""
    embedder = FakeEmbedder()
    retriever = FakeRetriever(chunks=[_chunk("b1", similarity=0.92), _chunk("b2", similarity=0.81)])
    chat = FakeChatModel(tokens=["你", "好"])
    service = ChatService(embedder=embedder, retriever=retriever, chat_model=chat)

    frames = await _collect(service, "你好")

    assert [frame["type"] for frame in frames] == ["meta", "token", "token", "done"]

    sources = frames[0]["sources"]
    assert len(sources) == 2
    first = sources[0]
    assert first["block_id"] == "b1"
    assert first["text"] == "块内容"
    assert first["source_file"] == "notes.md"
    assert first["similarity"] == pytest.approx(0.92)
    assert first["heading_path"] == "s1"
    assert first["anchor"] == "a1"
    assert first["chunk_type"] == "text"
    assert first["platform"] == "kimi"

    assert frames[1] == {"type": "token", "text": "你"}
    assert frames[2] == {"type": "token", "text": "好"}
    assert frames[3] == {"type": "done"}

    # 链路调用：嵌入 -> 检索
    assert embedder.calls == [["你好"]]
    assert retriever.calls == [[0.1, 0.2, 0.3, 0.4]]
    # 上下文保留来源序号，system 提示词约束
    system_msg, user_msg = chat.calls[0]
    assert system_msg["role"] == "system"
    assert "不得虚构" in system_msg["content"] or "不得编造" in system_msg["content"]
    assert "[1]" in user_msg["content"]
    assert "[2]" in user_msg["content"]


async def test_no_sources_streams_with_empty_meta() -> None:
    """检索无命中 -> meta.sources 为空数组，仍正常流式 + done。"""
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[]), FakeChatModel(tokens=["不知道"]))
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["meta", "token", "done"]
    assert frames[0]["sources"] == []


# ---------------------------------------------------------------- 2. 错误转 error 帧（终帧，不崩）


async def test_retrieval_error_becomes_single_error_frame() -> None:
    """RetrievalError -> 单个 error 帧，消息取自领域异常（含「未建索引」）。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(error=RetrievalError("知识库尚未建立索引，请先导入并同步文档")),
        FakeChatModel(),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["error"]
    assert "尚未建立索引" in frames[0]["message"]


async def test_chat_model_error_becomes_single_error_frame() -> None:
    """对话模型抛异常 -> 单个 error 帧（无 meta / done），消息脱敏不泄漏异常正文。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        FakeChatModel(error=RuntimeError("boom-internal-detail")),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["error"]
    assert "boom-internal-detail" not in frames[0]["message"]
    assert frames[0]["message"]


async def test_sync_raise_from_chat_model_becomes_error_frame() -> None:
    """MEDIUM-2：stream_chat 调用阶段同步抛异常（非迭代阶段）-> 单个 error 帧，不 500。"""

    class SyncRaisingChatModel:
        @property
        def supports_image_input(self) -> bool:
            return False

        def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            raise RuntimeError("sync-boom-internal")

    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        SyncRaisingChatModel(),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["error"]
    assert "sync-boom-internal" not in frames[0]["message"]
    assert frames[0]["message"]


async def test_provider_error_becomes_single_error_frame() -> None:
    """ChatProviderError -> 单个 error 帧，消息列出缺失的环境变量名。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        FakeChatModel(
            error=ChatProviderError(
                "对话 provider 未配置，缺少环境变量: EVERYTHING_RAG_CHAT_MODEL"
            )
        ),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["error"]
    assert "EVERYTHING_RAG_CHAT_MODEL" in frames[0]["message"]


async def test_embedder_error_becomes_single_error_frame() -> None:
    """嵌入器抛异常 -> 单个 error 帧，消息脱敏不泄漏异常正文。"""
    service = ChatService(
        FakeEmbedder(error=RuntimeError("embed-down")),
        FakeRetriever(),
        FakeChatModel(),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["error"]
    assert "embed-down" not in frames[0]["message"]
    assert frames[0]["message"]


# ---------------------------------------------------------------- 3. 零出网


async def test_zero_outbound_full_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切 socket 连接，ChatService 全链路仍成功 => 零出网。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        FakeChatModel(tokens=["ok"]),
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["meta", "token", "done"]


async def test_custom_system_prompt_used() -> None:
    """ChatService(system_prompt=...) -> 传给对话模型的 system 消息用自定义提示词（覆盖内置）。"""
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        chat,
        system_prompt="你是测试助手，只用中文回答，且不引用任何来源。",
    )
    frames = await _collect(service, "你好")
    assert [frame["type"] for frame in frames] == ["meta", "token", "done"]
    system_msg = chat.calls[0][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == "你是测试助手，只用中文回答，且不引用任何来源。"
    assert "不得虚构" not in system_msg["content"]  # 内置提示词被覆盖


async def test_default_system_prompt_when_not_given() -> None:
    """未传 system_prompt -> 用内置默认提示词。"""
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[_chunk("b1")]), chat)
    await _collect(service, "你好")
    system_msg = chat.calls[0][0]
    assert system_msg["role"] == "system"
    assert "不要编造" in system_msg["content"]


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
