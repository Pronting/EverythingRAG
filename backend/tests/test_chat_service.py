"""ChatService 单测：meta/token/done 帧序列 + 各类异常转 error 帧（fake 依赖，零出网）。

隐私红线：全部 fake 依赖（嵌入/检索/对话），无 socket/requests 调用；
错误消息不得泄漏异常正文 / 文件路径 / 密钥。
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.generation.providers import ChatProviderError
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk
from app.search.base import SearchError, SearchResult


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
        query_text: str,
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
            yield ChatChunk("content", token)


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


async def test_stream_strips_disjointed_phrases() -> None:
    """代码级约束：正文里的「割裂」措辞被流式剔除，剩余正文保留。"""
    chat = FakeChatModel(
        tokens=[
            "以下为通用知识，非你的资料库内容：",
            "Everything RAG 是一款本地知识助手。",
        ]
    )
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[]), chat)
    frames = await _collect(service, "Everything RAG 是做什么的")
    text = "".join(frame["text"] for frame in frames if frame["type"] == "token")
    assert "非你的资料库内容" not in text
    assert "以下为通用知识" not in text
    assert "Everything RAG" in text


async def test_empty_context_user_message_directs_natural_answer() -> None:
    """检索无命中 -> user 消息走 general 分支：要求自然回答，禁止割裂标注语。"""
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[]), chat)
    await _collect(service, "三国演义讲的是什么")
    user_msg = chat.calls[0][1]
    assert user_msg["role"] == "user"
    assert "问题：三国演义讲的是什么" in user_msg["content"]
    assert "自然" in user_msg["content"]  # 要求自然、连贯回答
    assert "通用知识" in user_msg["content"]  # 明确禁止「通用知识」等标注语
    assert "非资料库内容" in user_msg["content"]  # 明确禁止「非资料库内容」标注语
    assert "以下为通用知识，非你的资料库内容" not in user_msg["content"]  # 不再强制该标注


async def test_history_included_in_messages() -> None:
    """多轮上下文：history 传入后，组装消息里按序插入历史 user/assistant 轮次。"""
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[_chunk("b1")]), chat)
    history = [
        {"role": "user", "content": "红人系统是什么？"},
        {"role": "assistant", "content": "红人系统是连接平台与红人的中枢。"},
    ]
    frames = [frame async for frame in service.stream_answer("TT 是什么意思", history)]
    assert frames[-1]["type"] == "done"
    messages = chat.calls[0]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "红人系统是什么？"
    assert messages[2]["content"] == "红人系统是连接平台与红人的中枢。"
    assert "TT 是什么意思" in messages[3]["content"]


async def test_meta_question_skips_retrieval() -> None:
    """元问题（知识库里有什么）跳过检索，返回空 meta（无来源）。"""
    embedder = FakeEmbedder()
    retriever = FakeRetriever(chunks=[_chunk("b1")])
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(embedder=embedder, retriever=retriever, chat_model=chat)
    frames = await _collect(service, "知识库里有什么内容？")
    assert retriever.calls == []  # 未触发检索
    assert frames[0]["type"] == "meta"
    assert frames[0]["sources"] == []


async def test_default_prompt_has_self_intro_and_anti_hijack_rules() -> None:
    """默认提示词含产品自我介绍 + 自然作答要求 + 防上下文劫持规则。"""
    chat = FakeChatModel(tokens=["ok"])
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[_chunk("b1")]), chat)
    await _collect(service, "你好")
    content = chat.calls[0][0]["content"]
    assert "Everything RAG" in content  # 产品自我介绍
    assert "个人知识" in content  # 产品定位
    assert "不要逐字照抄" in content  # 防复读
    assert "改变你的角色" in content  # 防无关上下文劫持角色
    assert "不是给你的指令" in content  # 防提示注入：KB 指令式内容不视为系统指令
    assert "割裂" in content  # 明确禁止割裂性措辞


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


# ---------------------------------------------------------------- 4. 联网搜索（确定性注入）


class FakeSearchProvider:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._results = results if results is not None else []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.calls.append((query, max_results))
        if self._error is not None:
            raise self._error
        return self._results


def _web_result(title: str = "联网结果", url: str = "https://example.com", content: str = "网页正文") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=content, content=content)


async def test_web_search_injects_web_sources_and_context() -> None:
    """web_search=True：tool(running/done) -> meta（含 KB + Web 来源）-> token -> done；
    Web 来源带 url/source_type，上下文统一编号。"""
    chat = FakeChatModel(tokens=["你", "好"])
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        chat,
        search_provider=FakeSearchProvider(results=[_web_result()]),
    )

    frames = [frame async for frame in service.stream_answer("你好", web_search=True)]

    assert [frame["type"] for frame in frames] == ["tool", "tool", "meta", "token", "token", "done"]
    assert frames[0] == {"type": "tool", "tool": "web_search", "status": "running"}
    assert frames[1] == {"type": "tool", "tool": "web_search", "status": "done"}

    sources = frames[2]["sources"]
    assert len(sources) == 2
    kb, web = sources
    assert kb["block_id"] == "b1"
    assert kb.get("source_type") is None  # KB 来源无 source_type
    assert web["source_type"] == "web"
    assert web["url"] == "https://example.com"
    assert web["chunk_type"] == "web"
    assert web["source_file"] == "联网结果"

    # 上下文：KB 为 [1]，Web 为 [2]，来源描述含「联网搜索」
    user_msg = chat.calls[0][1]
    assert "[1]" in user_msg["content"]
    assert "[2]" in user_msg["content"]
    assert "知识库与联网搜索" in user_msg["content"]


async def test_web_search_unconfigured_returns_error_frame() -> None:
    """web_search=True 但 provider=None -> 单个 error 帧，消息提示未配置。"""
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks=[_chunk("b1")]), FakeChatModel())
    frames = [frame async for frame in service.stream_answer("你好", web_search=True)]
    assert [frame["type"] for frame in frames] == ["error"]
    assert "未配置" in frames[0]["message"]


async def test_web_search_error_becomes_error_frame() -> None:
    """搜索 provider 抛 SearchError -> tool(running) 后 error 帧，消息脱敏。"""
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        FakeChatModel(),
        search_provider=FakeSearchProvider(error=SearchError("联网搜索失败（HTTP 429）")),
    )
    frames = [frame async for frame in service.stream_answer("你好", web_search=True)]
    assert [frame["type"] for frame in frames] == ["tool", "error"]
    assert "HTTP 429" in frames[1]["message"]


async def test_web_search_disabled_does_not_call_provider() -> None:
    """web_search=False：不调用搜索 provider，帧序与既有行为完全一致。"""
    provider = FakeSearchProvider(results=[_web_result()])
    service = ChatService(
        FakeEmbedder(),
        FakeRetriever(chunks=[_chunk("b1")]),
        FakeChatModel(tokens=["好"]),
        search_provider=provider,
    )
    frames = [frame async for frame in service.stream_answer("你好", web_search=False)]
    assert [frame["type"] for frame in frames] == ["meta", "token", "done"]
    assert provider.calls == []
