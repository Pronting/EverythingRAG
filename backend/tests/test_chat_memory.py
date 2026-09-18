"""ChatService × 记忆通路契约测试。

这是 v1.1.0 最关键的回归面：记忆作为**来源层**（而非静默注入）接入问答后，
必须同时满足四条硬约束——

1. ``[M1]`` 必须能通过 ``OutputGuard`` 的引用白名单（否则模型写的记忆引用被静默删除）。
2. ``meta.sources`` 必须出现 ``source_type: "memory"``，前端据此渲染「提炼」标识。
3. 记忆通路任何失败都必须静默降级，**不影响基于文档的问答**（D6）。
4. 记忆是推断而非原文：素材必须显式标注，且不让模型把它讲成文档原文。
"""

from __future__ import annotations

from typing import Any

from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.memory.models import MemoryCategory
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore
from app.retrieval.vector_retriever import RetrievedChunk

from tests.test_memory_index import RecordingMemoryVectorStore, _entry, _hit


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._error is not None:
            raise self._error
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class EmptyRetriever:
    """文档通路返回空：把断言焦点集中在记忆通路上。"""

    def retrieve(
        self, query_text: str, query_vector: list[float], where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return []


class FakeChatModel:
    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens if tokens is not None else ["答案文本"]
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append(messages)
        for token in self._tokens:
            yield ChatChunk("content", token)

    def user_message(self) -> str:
        """最近一轮 user 消息（含素材拼装结果）。"""

        for message in reversed(self.calls[-1]):
            if message["role"] == "user":
                return str(message["content"])
        raise AssertionError("未发送 user 消息")


def _document_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        block_id="doc-1",
        text="文档正文",
        similarity=0.9,
        source_file="a.md",
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path="标题",
        metadata={"source_file": "a.md"},
    )


def _memory_service(tmp_path, statements: list[str], tokens: list[str] | None = None):  # type: ignore[no-untyped-def]
    """装配一个「文档通路为空 + 记忆通路有数据」的 ChatService。"""

    store = MemoryStore(tmp_path / "memory.db")
    hits = []
    for index, statement in enumerate(statements):
        entry = _entry(store, statement, category=MemoryCategory.PREFERENCE)
        hits.append(_hit(entry.id, 0.9 - index * 0.05))
    vectorstore = RecordingMemoryVectorStore(hits=hits)
    model = FakeChatModel(tokens)
    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=EmptyRetriever(),
        chat_model=model,
        memory_retriever=MemoryRetriever(vectorstore, store),
    )
    return service, model, store


async def _collect(service: ChatService, message: str = "我之前说过什么偏好？") -> list[dict[str, Any]]:
    return [frame async for frame in service.stream_answer(message)]


# ---------------------------------------------------------------- 帧契约


async def test_meta_frame_marks_memory_sources(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """记忆命中必须出现在 meta.sources，且带 source_type/段位/回溯信息。"""

    service, _model, _store = _memory_service(tmp_path, ["用户偏好 SQLite"])
    frames = await _collect(service)

    meta = next(frame for frame in frames if frame["type"] == "meta")
    memory_sources = [
        source for source in meta["sources"] if source["source_type"] == "memory"
    ]

    assert len(memory_sources) == 1
    source = memory_sources[0]
    assert source["source_id"] == "M1"
    assert source["chunk_type"] == "memory"
    assert source["match_type"] == "memory"
    assert source["text"] == "用户偏好 SQLite"
    assert source["category"] == "preference"
    assert source["confidence"] == 0.9
    assert source["source_ref"] == "conv-1#4"
    assert source["quote"] == "原话摘录"


async def test_memory_sources_do_not_displace_document_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """记忆使用独立 M 段位：不得占用或重排文档的 S 段位编号。"""

    store = MemoryStore(tmp_path / "memory.db")
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.9)])

    class OneDocRetriever:
        def retrieve(self, query_text, query_vector, where=None):  # type: ignore[no-untyped-def]
            return [_document_chunk()]

    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=OneDocRetriever(),
        chat_model=FakeChatModel(),
        memory_retriever=MemoryRetriever(vectorstore, store),
    )
    frames = await _collect(service)
    meta = next(frame for frame in frames if frame["type"] == "meta")

    assert [source["source_id"] for source in meta["sources"]] == ["S1", "M1"]


async def test_memory_citation_survives_output_guard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``[M1]`` 必须通过引用白名单 —— 否则模型写的记忆引用会被静默删掉。

    这是本版最容易漏的一处：``OutputGuard`` 只放行白名单内的 source_id，
    而白名单源自本轮 meta.sources，段位与正则两处都必须认识 M。
    """

    service, model, _store = _memory_service(
        tmp_path, ["用户偏好 SQLite"], tokens=["你之前说过偏好 ", "[M1]", "。"]
    )
    frames = await _collect(service)

    text = "".join(frame["text"] for frame in frames if frame["type"] == "token")
    assert text == "你之前说过偏好 [M1]。"
    assert "[M1]" in text


async def test_fabricated_memory_citation_is_removed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """本轮没召回到记忆时，模型编造的 ``[M1]`` 必须被删除（与 S/W 同一规则）。"""

    service, _model, _store = _memory_service(
        tmp_path, [], tokens=["依据 ", "[M1]", " 可知"]
    )
    frames = await _collect(service)

    text = "".join(frame["text"] for frame in frames if frame["type"] == "token")
    assert "[M1]" not in text
    assert "依据" in text and "可知" in text


# ---------------------------------------------------------------- 素材与提示词


async def test_memory_material_is_labelled_as_inferred(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """记忆素材必须显式标注为"提炼推断"，并说明冲突时以本地知识为准。"""

    service, model, _store = _memory_service(tmp_path, ["用户偏好 SQLite"])
    await _collect(service)

    user_message = model.user_message()
    assert 'kind="memory"' in user_message
    assert "提炼" in user_message
    assert "不是文档原文" in user_message
    # 记忆必须出现在允许引用列表里，否则模型无法合法引用它。
    assert "M1" in user_message
    # 冲突优先级必须写死，避免模型用一条旧记忆覆盖当前文档证据。
    assert "以本地知识为准" in user_message


async def test_no_memory_leaves_prompt_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """未命中记忆时不得注入任何记忆相关措辞（保持既有提示词行为）。"""

    service, model, _store = _memory_service(tmp_path, [])
    await _collect(service)

    user_message = model.user_message()
    assert 'kind="memory"' not in user_message
    assert "记忆" not in user_message


# ---------------------------------------------------------------- 失败降级


async def test_memory_failure_does_not_break_answer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """记忆通路抛异常时必须静默降级：问答照常完成，且不产生 error 帧。"""

    store = MemoryStore(tmp_path / "memory.db")
    _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore()
    vectorstore.fail_query = True

    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=EmptyRetriever(),
        chat_model=FakeChatModel(tokens=["正常回答"]),
        memory_retriever=MemoryRetriever(vectorstore, store),
    )
    frames = await _collect(service)

    assert not [frame for frame in frames if frame["type"] == "error"]
    text = "".join(frame["text"] for frame in frames if frame["type"] == "token")
    assert text == "正常回答"
    meta = next(frame for frame in frames if frame["type"] == "meta")
    assert [source for source in meta["sources"] if source["source_type"] == "memory"] == []


async def test_service_without_memory_retriever_still_works(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """未配置记忆检索（memory_retriever=None）时行为与 v0.1.1 完全一致。"""

    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=EmptyRetriever(),
        chat_model=FakeChatModel(tokens=["回答"]),
    )
    frames = await _collect(service)

    meta = next(frame for frame in frames if frame["type"] == "meta")
    assert meta["sources"] == []
    assert not [frame for frame in frames if frame["type"] == "error"]


async def test_memory_not_retrieved_for_product_help(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """产品说明/知识概览路由不查记忆：它们由本地可信结构回答，不该拌入用户记忆。"""

    store = MemoryStore(tmp_path / "memory.db")
    entry = _entry(store, "用户偏好 SQLite")
    vectorstore = RecordingMemoryVectorStore(hits=[_hit(entry.id, 0.9)])
    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=EmptyRetriever(),
        chat_model=FakeChatModel(),
        memory_retriever=MemoryRetriever(vectorstore, store),
    )

    frames = [frame async for frame in service.stream_answer("你是谁")]
    meta = next(frame for frame in frames if frame["type"] == "meta")
    assert meta["sources"] == []


async def test_memory_is_separate_from_answer_basis(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """记忆是"来源"而非"模式"：answer_basis 枚举不得因记忆而新增取值。

    记忆本身不足以支撑"知识库里有依据"的答案，因此命中记忆但无文档/联网时，
    依据仍是 general/insufficient —— 这防止把一条提炼记忆讲成库内事实。
    """

    service, _model, _store = _memory_service(tmp_path, ["用户偏好 SQLite"])
    frames = await _collect(service)
    meta = next(frame for frame in frames if frame["type"] == "meta")

    assert meta["answer_basis"] in {"general", "insufficient"}
