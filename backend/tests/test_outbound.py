"""任务 7 隐私审计测试：出网收敛点 + 纯文本链路零出网 + provider 审计记录。

隐私红线：全部 fake/mock，无真实网络；纯文本链路断言 OutboundClient.record
从未被调用（审计为空）；provider 审计事件只含 host / 状态 / 时间，
绝不含 key / 路径 / 正文。测试路径本身零出网（阻断 socket 实证）。
"""

from __future__ import annotations

import socket
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.outbound import OutboundClient, OutboundEvent, outbound_client
from app.generation import providers
from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.generation.providers import ChatProviderError, OpenAICompatChatModel, create_chat_model
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.scanner import scan_directory
from app.models.schemas import BlockMetadata, SourceType
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import create_vector_store

# ---------------------------------------------------------------- fake 依赖（零出网）


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _UsageOnlyChunk:
    """上游收尾的 usage-only chunk：choices 为空（MEDIUM-1 回归）。"""

    def __init__(self) -> None:
        self.choices = []


class _ErrorPayload:
    """上游 error 对象（含受控 message 字段）。"""

    def __init__(self, message: str) -> None:
        self.message = message


class _ErrorChunk:
    """choices 为空但带 error 的 chunk（流式中途报错，MEDIUM-3 回归）。"""

    def __init__(self, message: str) -> None:
        self.choices = []
        self.error = _ErrorPayload(message)


class _AsyncChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCompletions:
    def __init__(self, chunks: list[Any], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.calls: list[tuple[str, list[dict[str, Any]], bool]] = []

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        **kwargs: Any,
    ) -> Any:
        self.calls.append((model, messages, stream))
        if self._error is not None:
            raise self._error
        return _AsyncChunks(self._chunks)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


class FakeEmbedder:
    """链路测试用假嵌入器：指纹 fake / dim 4，统一返回 [1,0,0,0]。"""

    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)


class FakeChatModel:
    """链路测试用本地假对话模型（不触发任何出网 / 审计）。"""

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        for token in ["好", "的"]:
            yield ChatChunk("content", token)


def _event(destination: str, status: int | None = 200) -> OutboundEvent:
    return OutboundEvent(
        provider="chat",
        destination=destination,
        method="chat.completions",
        status=status,
        started_at=datetime.now(UTC),
        duration_ms=1.0,
    )


def _metadata(chunk: Chunk, source_file: str) -> BlockMetadata:
    """Chunk -> Chroma 元数据（来源/平台/标题路径/锚点）。"""
    return BlockMetadata(
        block_id=chunk.block_id,
        doc_id=f"doc-{source_file}",
        source_type=SourceType.DOCUMENT,
        source_file=source_file,
        platform="local",
        heading_path=chunk.heading_path or None,
        anchor=chunk.anchor or None,
        chunk_type="text",
    )


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")


# ---------------------------------------------------------------- 1. OutboundEvent / OutboundClient 单元


def test_event_fields_present_and_frozen() -> None:
    """OutboundEvent 字段齐全（provider/destination/method/status/started_at/duration_ms）且不可变。"""
    started = datetime.now(UTC)
    event = OutboundEvent(
        provider="chat",
        destination="api.deepseek.com",
        method="chat.completions",
        status=200,
        started_at=started,
        duration_ms=12.5,
    )
    assert event.provider == "chat"
    assert event.destination == "api.deepseek.com"
    assert event.method == "chat.completions"
    assert event.status == 200
    assert event.started_at == started
    assert event.duration_ms == 12.5
    with pytest.raises((AttributeError, TypeError)):
        event.status = 500  # frozen dataclass 不可变


def test_bounded_ring_drops_oldest() -> None:
    """有界环形：超限丢弃最旧，仅保留最近 max_entries 条。"""
    client = OutboundClient(max_entries=3)
    for index in range(5):
        client.record(_event(f"d{index}"))
    entries = client.entries()
    assert len(entries) == 3
    assert [entry.destination for entry in entries] == ["d2", "d3", "d4"]


def test_entries_returns_immutable_copy() -> None:
    """entries() 返回副本：外部追加/替换元素不影响内部状态。"""
    client = OutboundClient(max_entries=10)
    client.record(_event("a"))
    snapshot = client.entries()
    snapshot.append(_event("tampered"))
    snapshot[0] = _event("tampered")
    assert [entry.destination for entry in client.entries()] == ["a"]


def test_state_clear_empty() -> None:
    """is_empty / outbound_state / clear 语义：无记录 local-only，有记录 has-outbound。"""
    client = OutboundClient(max_entries=10)
    assert client.is_empty()
    assert client.outbound_state == "local-only"
    client.record(_event("a"))
    assert not client.is_empty()
    assert client.outbound_state == "has-outbound"
    client.clear()
    assert client.is_empty()
    assert client.outbound_state == "local-only"


def test_record_order_preserved() -> None:
    """多事件按追加顺序返回（FIFO）。"""
    client = OutboundClient(max_entries=10)
    client.record(_event("a"))
    client.record(_event("b"))
    client.record(_event("c"))
    assert [entry.destination for entry in client.entries()] == ["a", "b", "c"]


def test_thread_safety_basic() -> None:
    """并发 record 不丢事件（threading.Lock 保证基本线程安全）。"""
    client = OutboundClient(max_entries=1000)
    threads = [
        threading.Thread(target=lambda i=index: client.record(_event(f"t{i}")))
        for index in range(50)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    entries = client.entries()
    assert len(entries) == 50
    assert len({entry.destination for entry in entries}) == 50


# ---------------------------------------------------------------- 2. provider 经 OutboundClient 记录


async def test_provider_records_one_event_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """带 outbound 调 stream_chat -> 恰一条事件：provider=chat / destination=host / status=200。"""
    outbound = OutboundClient(max_entries=10)
    completions = _FakeCompletions([_FakeChunk("你"), _FakeChunk("好")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        outbound=outbound,
    )
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "hi"}])]
    assert [c.text for c in tokens] == ["你", "好"]

    entries = outbound.entries()
    assert len(entries) == 1
    event = entries[0]
    assert event.provider == "chat"
    assert event.destination == "api.deepseek.com"  # 只记 host，无路径/query
    assert event.method == "chat.completions"
    assert event.status == 200
    assert event.started_at.tzinfo is not None
    assert event.duration_ms is not None
    assert event.duration_ms >= 0.0


async def test_provider_records_error_event_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create 抛错 -> 记录 status=None 事件，且异常照常向上传播。"""
    outbound = OutboundClient(max_entries=10)
    completions = _FakeCompletions([], error=ConnectionError("boom"))
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1",
        model="m",
        outbound=outbound,
    )
    with pytest.raises(ConnectionError):
        async for _token in model.stream_chat([{"role": "user", "content": "q"}]):
            pass
    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status is None
    assert entries[0].destination == "api.deepseek.com"
    assert entries[0].provider == "chat"
    assert entries[0].duration_ms is not None


async def test_provider_records_error_on_stream_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流式迭代中途抛错（首个 token 后网络中断）-> status=None 事件，异常传播。"""

    class _AbortStream:
        def __init__(self) -> None:
            self._sent = False

        def __aiter__(self) -> _AbortStream:
            return self

        async def __anext__(self) -> Any:
            if not self._sent:
                self._sent = True
                return _FakeChunk("x")
            raise ConnectionError("stream-aborted")

    class _AbortCompletions:
        async def create(self, *args: Any, **kwargs: Any) -> _AbortStream:
            return _AbortStream()

    class _AbortClient:
        def __init__(self) -> None:
            self.chat = _FakeChat(_AbortCompletions())

    outbound = OutboundClient(max_entries=10)
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _AbortClient())
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1",
        model="m",
        outbound=outbound,
    )
    with pytest.raises(ConnectionError):
        async for _token in model.stream_chat([{"role": "user", "content": "q"}]):
            pass
    assert len(outbound.entries()) == 1
    assert outbound.entries()[0].status is None


async def test_destination_never_leaks_path_or_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base_url 含路径/query -> destination 只记 host，不泄漏路径、query、key。"""
    outbound = OutboundClient(max_entries=10)
    completions = _FakeCompletions([_FakeChunk("a")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1/chat/completions?key=secret&x=1",
        model="m",
        outbound=outbound,
    )
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "q"}])]
    assert [c.text for c in tokens] == ["a"]
    event = outbound.entries()[0]
    assert event.destination == "api.deepseek.com"
    assert "secret" not in event.destination
    assert "/v1" not in event.destination


async def test_stream_skips_usage_only_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEDIUM-1：choices 为空的 usage-only 收尾 chunk 被跳过，不 IndexError。"""
    completions = _FakeCompletions([_UsageOnlyChunk(), _FakeChunk("x"), _UsageOnlyChunk()])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(base_url="http://localhost:9999/v1", model="m")
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "q"}])]
    assert [c.text for c in tokens] == ["x"]


async def test_provider_without_outbound_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """outbound 缺省 None -> 行为与任务 6 一致（兼容，不记录）。"""
    completions = _FakeCompletions([_FakeChunk("a")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(base_url="http://localhost:9999/v1", model="m")
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "q"}])]
    assert [c.text for c in tokens] == ["a"]


async def test_create_chat_model_wires_global_outbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRITICAL 回归：create_chat_model 缺省接全局 outbound_client（生产审计不静默失效）。

    生产链路 get_chat_service -> create_chat_model 不传 outbound，须默认接全局
    单例；真实出网后 /api/audit 与 /api/status 才能如实上报 has-outbound。
    """
    outbound_client.clear()
    completions = _FakeCompletions([_FakeChunk("你")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    settings = Settings(chat_base_url="https://api.deepseek.com/v1", chat_model="deepseek-chat")
    model = create_chat_model(settings)
    assert model._outbound is outbound_client  # 缺省已接全局单例
    tokens = [token async for token in model.stream_chat([{"role": "user", "content": "q"}])]
    assert [c.text for c in tokens] == ["你"]
    entries = outbound_client.entries()
    assert len(entries) == 1  # 生产接线后真实出网确实进审计
    assert entries[0].status == 200
    assert entries[0].destination == "api.deepseek.com"
    outbound_client.clear()


async def test_provider_records_event_on_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEDIUM-2：流式中途 aclose()（SSE 客户端断开）-> 仍记恰 1 条 status=None 事件。"""
    outbound = OutboundClient(max_entries=10)
    completions = _FakeCompletions([_FakeChunk("a"), _FakeChunk("b")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1",
        model="m",
        outbound=outbound,
    )
    stream = model.stream_chat([{"role": "user", "content": "q"}])
    first = await anext(stream)
    assert first.text == "a"
    await stream.aclose()
    entries = outbound.entries()
    assert len(entries) == 1  # create() 已发生的出网不因 aclose 漏记
    assert entries[0].status is None


async def test_provider_records_error_on_chunk_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEDIUM-3：空 choices + error chunk -> ChatProviderError 传播 + 审计 status=None。"""
    outbound = OutboundClient(max_entries=10)
    completions = _FakeCompletions([_FakeChunk("a"), _ErrorChunk("上游过载")])
    monkeypatch.setattr(providers, "AsyncOpenAI", lambda **kwargs: _FakeClient(completions))
    model = OpenAICompatChatModel(
        base_url="https://api.deepseek.com/v1",
        model="m",
        outbound=outbound,
    )
    with pytest.raises(ChatProviderError, match="上游过载"):
        async for _token in model.stream_chat([{"role": "user", "content": "q"}]):
            pass
    entries = outbound.entries()
    assert len(entries) == 1
    assert entries[0].status is None


# ---------------------------------------------------------------- 3. 纯文本链路零出网（核心）


async def test_pure_text_chain_zero_outbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实组件全链路（scan->parse->chunk->embed->upsert->retrieve->chat）零出网。

    断言：monkeypatch 的 OutboundClient.record 从未被调用、审计为空；
    同时阻断一切 socket 连接，实证纯文本路径无任何真实外呼。
    """
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)
    record_calls: list[Any] = []
    monkeypatch.setattr(outbound_client, "record", record_calls.append)

    notes = tmp_path / "notes.md"
    notes.write_text("# 标题\n\n内容段落。\n\n第二段。", encoding="utf-8")

    # 1. 扫描（真实 scanner）
    discovered = scan_directory(tmp_path)
    assert len(discovered) == 1
    source_file = discovered[0].path.name

    # 2. 解析（真实 markdown-it-py）
    parsed = parse_markdown(notes.read_text(encoding="utf-8"))

    # 3. 切块（真实 chunker）
    chunks = chunk_document(parsed, source_file=source_file)
    assert chunks

    # 4. 嵌入（fake embedder，零下载零网络）
    embedder = FakeEmbedder()
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    # 5. 向量库入库（真实 Chroma，临时目录，零网络）
    store = create_vector_store(persist_dir=tmp_path / "db", embedder=embedder)
    store.upsert(
        [
            (chunk.block_id, chunk.text, vector, _metadata(chunk, source_file))
            for chunk, vector in zip(chunks, vectors)
        ]
    )
    assert store.count() == len(chunks)

    # 6. 检索（真实 VectorRetriever）
    retriever = VectorRetriever(store)
    retrieved = retriever.retrieve("内容", embedder.embed_texts(["内容"])[0])
    assert retrieved

    # 7. 问答（本地假对话模型，零出网）
    service = ChatService(embedder=embedder, retriever=retriever, chat_model=FakeChatModel())
    frames = [frame async for frame in service.stream_answer("内容")]

    assert [frame["type"] for frame in frames] == ["meta", "token", "token", "done"]
    assert record_calls == []  # OutboundClient.record 从未被调用
    assert outbound_client.is_empty()  # 审计为空
