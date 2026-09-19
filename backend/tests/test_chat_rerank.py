"""ChatService 接入相关性筛选：进度帧、meta.retrieval、证据一致性与三态（T08）。

覆盖：
* filtered / empty / disabled 三态与取消、失败路径；
* SSE 复用 tool 帧（tool=rerank，running/done/error），评分期间不产出 token；
* meta.retrieval 的字段与计数；failed/cancelled 不发送带旧候选的成功 meta；
* 生成上下文与来源面板使用同一份 evidence_text（旧的 context_text 补充文字不再进入）；
* 标题从块元数据补齐（真实索引的 heading_path）并参与 ranking_text。

全部 fake 依赖，零出网。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.generation.prompt_policy import AnswerPreferences
from app.retrieval.evidence_view import HeadingLookup, RetrievalScope, sha256_text
from app.retrieval.relevance_selector import SCORING_PROTOCOL_VERSION
from app.retrieval.reranker import (
    ProfileStatus,
    RerankAuthFailedError,
    RerankProfile,
    RerankProvider,
    RerankProviderCapabilities,
    RerankRequest,
    RerankResult,
    RerankScore,
    ScoreScope,
    SelectionStatus,
)
from app.retrieval.vector_retriever import RetrievedChunk

SCOPE = RetrievalScope.from_where(None, index_revision="chunks__fake__v1")
KEY = "SK-live-key-1234567890"


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _text in texts]


class FakeVectorStore:
    """只提供 list_blocks 的向量库替身（标题查表用）。"""

    def __init__(self, blocks: list[tuple[str, str, dict[str, Any]]]) -> None:
        self._blocks = blocks

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._blocks)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk], vectorstore: Any = None) -> None:
        self._chunks = chunks
        self._vectorstore = vectorstore
        self.calls = 0

    def retrieve_variants(
        self,
        variants: Any,
        vectors: Any,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return list(self._chunks)


class FakeChatModel:
    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens if tokens is not None else ["答案"]
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append(messages)
        for token in self._tokens:
            yield ChatChunk("content", token)


class StubProvider(RerankProvider):
    """测试替身：按候选给出固定分数（缺省 0.9，可显式注入低分），可注入错误。"""

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        error: Exception | None = None,
        *,
        default_score: float = 0.9,
    ) -> None:
        self._scores = scores or {}
        self._default = default_score
        self._error = error
        self.calls: list[RerankRequest] = []
        self._capabilities = RerankProviderCapabilities(
            provider_type="siliconflow_rerank",
            model="Qwen/Qwen3-Reranker-8B",
            tokenizer_id="stub",
        )

    @property
    def provider_type(self) -> str:
        return "siliconflow_rerank"

    @property
    def capabilities(self) -> RerankProviderCapabilities:
        return self._capabilities

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return RerankResult(
            profile_id=request.profile_id,
            provider_type=self.provider_type,
            model="Qwen/Qwen3-Reranker-8B",
            document_count=len(request.candidate_ids),
            scores=tuple(
                RerankScore(
                    candidate_id=candidate_id,
                    index=index,
                    score=self._scores.get(candidate_id, self._default),
                )
                for index, candidate_id in enumerate(request.candidate_ids)
            ),
            score_scope=ScoreScope.REQUEST_RELATIVE,
        )


def make_profile(**overrides: Any) -> RerankProfile:
    values: dict[str, Any] = {
        "profile_id": "profile-cal-1",
        "status": ProfileStatus.VALIDATED,
        "provider_type": "siliconflow_rerank",
        "model": "Qwen/Qwen3-Reranker-8B",
        "endpoint_fingerprint": "https://api.siliconflow.cn",
        "instruction_id": "instr-1",
        "instruction_hash": sha256_text("instruction"),
        "tokenizer_id": "stub",
        "view_version": "evidence-view-v1",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "score_threshold": 0.35,
        "max_candidates": 50,
        "min_candidate_count": 5,
    }
    values.update(overrides)
    return RerankProfile(**values)


def make_chunk(
    block_id: str,
    *,
    text: str = "块内容",
    heading_path: str | None = "s1",
    context_text: str | None = None,
    similarity: float | None = 0.9,
    source_file: str = "notes.md",
    chunk_type: str = "text",
) -> RetrievedChunk:
    return RetrievedChunk(
        block_id=block_id,
        text=text,
        similarity=similarity,
        source_file=source_file,
        platform="local",
        chunk_type=chunk_type,
        anchor="a1",
        heading_path=heading_path,
        metadata={"source_file": source_file},
        match_type="dense",
        context_text=context_text,
    )


def make_service(
    chunks: list[RetrievedChunk],
    *,
    provider: RerankProvider | None = None,
    profile: RerankProfile | None = None,
    blocks: list[tuple[str, str, dict[str, Any]]] | None = None,
    **kwargs: Any,
) -> tuple[ChatService, FakeChatModel, FakeRetriever]:
    vectorstore = FakeVectorStore(blocks or [])
    retriever = FakeRetriever(chunks, vectorstore)
    chat = FakeChatModel()
    service = ChatService(
        embedder=FakeEmbedder(),
        retriever=retriever,
        chat_model=chat,
        rerank_provider=provider,
        rerank_profile=profile if profile is not None else (make_profile() if provider else None),
        rerank_scope=SCOPE,
        **kwargs,
    )
    return service, chat, retriever


async def collect(service: ChatService, message: str) -> list[dict[str, Any]]:
    return [frame async for frame in service.stream_answer(message)]


def meta_of(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """meta 帧的位置随进度帧变化：按类型取，不写死下标。"""

    return next(frame for frame in frames if frame.get("type") == "meta")


# ------------------------------------------------------------------ 1. filtered 三态之一


async def test_filtered_reports_progress_and_selection() -> None:
    chunks = [make_chunk("b1"), make_chunk("b2")]
    provider = StubProvider()  # 默认 0.9，高于档案阈值 0.35
    service, chat, _retriever = make_service(chunks, provider=provider)

    frames = await collect(service, "检索相关性怎么判断")
    rerank_frames = [frame for frame in frames if frame.get("tool") == "rerank"]
    assert [frame["status"] for frame in rerank_frames] == ["running", "done"]

    meta = meta_of(frames)
    assert meta["retrieval"]["status"] == "filtered"
    assert meta["retrieval"]["profile_id"] == "profile-cal-1"
    assert meta["retrieval"]["selected_count"] == 2
    assert meta["retrieval"]["candidate_count"] == 2
    assert len(provider.calls) == 1
    # 帧顺序：进度帧在 meta 之前；评分期间不产出 token；最后必然 done
    types = [frame["type"] for frame in frames]
    assert types[:2] == ["tool", "tool"]
    assert types[2] == "meta"
    assert types[-1] == "done"
    assert chat.calls  # 评分完成后才生成答案


async def test_filtered_sources_and_prompt_share_the_same_evidence_text() -> None:
    """旧行为会用 context_text 把同标题补充文字塞进来源与提示词；筛选路径只用证据正文。"""

    evidence = "证据正文第一段。证据正文第二段。"
    hydrated = evidence + "\n\n同标题补充信息：\n这段文字未经评分检查，不得进入本轮输入。"
    chunk = make_chunk("b1", text=evidence, context_text=hydrated)
    provider = StubProvider()
    service, chat, _retriever = make_service([chunk], provider=provider)

    frames = await collect(service, "问题")
    source = meta_of(frames)["sources"][0]

    assert source["text"] == evidence
    prompt = chat.calls[0][-1]["content"]
    assert evidence in prompt
    assert "未经评分检查" not in prompt
    assert source["relevance_checked"] is True
    assert source["score_type"] == "rerank"
    assert source["profile_id"] == "profile-cal-1"
    assert source["evidence_hash"]
    assert source["similarity"] == pytest.approx(0.9)  # 向量相似度不被重排分数覆盖
    assert source["rerank_score"] is not None
    assert source["match_type"] == "rerank"


async def test_selected_heading_comes_from_block_metadata() -> None:
    """真实索引的标题存在块元数据 heading_path 里：必须进入 ranking_text。"""

    chunk = make_chunk("b1", text="正文", heading_path=None)
    provider = StubProvider()
    service, _chat, _retriever = make_service(
        [chunk],
        provider=provider,
        blocks=[("b1", "正文", {"heading_path": "示例风控方案 > 支持的策略", "source_file": "x.md"})],
    )

    frames = await collect(service, "问题")

    assert meta_of(frames)["sources"][0]["heading_path"] == "示例风控方案 > 支持的策略"
    request = provider.calls[0]
    ranking_text = request.documents[0].ranking_text
    assert "支持的策略" in ranking_text


async def test_missing_heading_stays_empty_and_is_counted() -> None:
    """块本身没有标题时留空并计数，绝不猜标题（诊断在 meta.retrieval 可见）。"""

    chunk = make_chunk("b1", text="正文", heading_path=None)
    provider = StubProvider()
    service, _chat, _retriever = make_service(
        [chunk], provider=provider, blocks=[("b1", "正文", {"source_file": "x.md"})]
    )

    frames = await collect(service, "问题")

    meta = meta_of(frames)
    assert meta["sources"][0]["heading_path"] is None
    assert meta["retrieval"]["missing_headings"] == 1
    # 绝不从正文首行猜标题：模板里的标题行保持为空
    ranking_text = provider.calls[0].documents[0].ranking_text
    assert "\n标题：\n" in ranking_text


# ------------------------------------------------------------------ 2. empty / disabled


async def test_empty_when_all_candidates_below_threshold() -> None:
    provider = StubProvider(default_score=0.01)  # 所有候选都低于阈值 0.35
    service, chat, _retriever = make_service(
        [make_chunk("b1"), make_chunk("b2")], provider=provider
    )

    frames = await collect(service, "问题")

    meta = meta_of(frames)
    assert meta["retrieval"]["status"] == "empty"
    assert meta["retrieval"]["selected_count"] == 0
    assert meta["retrieval"]["candidate_count"] == 2
    assert meta["sources"] == []
    assert meta["answer_basis"] == "general"  # 允许通用回答；仅知识模式才是 insufficient
    assert [frame["type"] for frame in frames] == ["tool", "tool", "meta", "token", "done"]
    # 沿用既有 choose_answer_basis：不得暗示事实来自用户内容
    assert "不要暗示这些事实来自用户内容" in chat.calls[0][-1]["content"]


async def test_empty_in_knowledge_only_mode_reports_insufficient() -> None:
    provider = StubProvider(default_score=0.01)
    service, chat, _retriever = make_service(
        [make_chunk("b1")],
        provider=provider,
        preferences=AnswerPreferences(knowledge_only=True),
    )

    frames = await collect(service, "问题")

    meta = meta_of(frames)
    assert meta["retrieval"]["status"] == "empty"
    assert meta["answer_basis"] == "insufficient"
    assert "没有足够的本地内容" in chat.calls[0][-1]["content"]


async def test_disabled_keeps_legacy_path_and_never_calls_provider() -> None:
    chunk = make_chunk("b1", context_text="旧补充文字")
    service, chat, _retriever = make_service([chunk], provider=None, profile=None)

    frames = await collect(service, "问题")

    meta = meta_of(frames)
    assert "retrieval" in meta
    assert meta["retrieval"]["status"] == "disabled"
    assert meta["retrieval"]["selected_count"] == 1
    assert all(frame.get("tool") != "rerank" for frame in frames)
    # 旧路径行为不变：来源仍使用 context_text
    assert meta["sources"][0]["text"] == "旧补充文字"
    assert "旧补充文字" in chat.calls[0][-1]["content"]
    assert "relevance_checked" not in meta["sources"][0]


async def test_knowledge_overview_and_product_help_skip_rerank() -> None:
    provider = StubProvider()
    service, _chat, _retriever = make_service([make_chunk("b1")], provider=provider)

    frames = await collect(service, "知识库里有什么")

    assert meta_of(frames)["query_mode"] == "KNOWLEDGE_OVERVIEW"
    assert "retrieval" not in meta_of(frames)
    assert provider.calls == []

    product_frames = await collect(service, "你是谁")
    assert meta_of(product_frames)["query_mode"] == "PRODUCT_HELP"
    assert "retrieval" not in meta_of(product_frames)
    assert provider.calls == []


# ------------------------------------------------------------------ 3. failed / cancelled


async def test_provider_failure_terminates_with_error_and_no_stale_meta() -> None:
    provider = StubProvider(error=RerankAuthFailedError("HTTP 401"))
    service, chat, _retriever = make_service([make_chunk("b1")], provider=provider)

    frames = await collect(service, "问题")

    assert [frame["type"] for frame in frames] == ["tool", "tool", "error"]
    assert frames[0] == {"type": "tool", "tool": "rerank", "status": "running"}
    assert frames[1] == {"type": "tool", "tool": "rerank", "status": "error"}
    assert frames[2]["message"] == "筛选服务认证失败"
    assert chat.calls == []  # 评分期间不生成正式答案
    # 失败不发送带旧候选的成功 meta
    assert not any(frame["type"] == "meta" for frame in frames)


async def test_profile_unavailable_is_a_configuration_failure_not_fallback() -> None:
    provider = StubProvider()
    service, chat, _retriever = make_service([make_chunk("b1")], provider=provider)
    service._rerank_profile_snapshot = lambda: None

    frames = await collect(service, "问题")

    assert [frame["type"] for frame in frames] == ["tool", "tool", "error"]
    assert "设置" in frames[-1]["message"]
    assert provider.calls == []
    assert chat.calls == []


async def test_cancelled_selection_ends_turn_without_generation() -> None:
    class CancelledProvider(StubProvider):
        async def rerank(self, request: RerankRequest) -> RerankResult:
            raise asyncio.CancelledError

    provider = CancelledProvider()
    service, chat, _retriever = make_service([make_chunk("b1")], provider=provider)

    with pytest.raises(asyncio.CancelledError):
        await collect(service, "问题")
    assert chat.calls == []


# ------------------------------------------------------------------ 4. 标题查表工具


def test_heading_lookup_reads_block_metadata_and_counts_missing() -> None:
    """缺失三类分开计数：缺键 / 键存在但为空 / 调用方漏传查表。"""

    lookup = HeadingLookup.for_vectorstore(
        FakeVectorStore(
            [
                ("b1", "正文", {"heading_path": "甲>乙"}),
                ("b2", "正文", {}),  # 缺键
                ("b3", "正文", {"heading_path": "   "}),  # 键存在但为空
                ("b4", "正文", {"heading_path": "丙"}),  # 有标题
            ]
        )
    )

    assert len(lookup) == 2  # 只有真正有标题的块计入长度
    assert lookup.lookup("b1") == "甲>乙"
    assert lookup.lookup("b2") == ""
    assert lookup.lookup("b3") == ""
    assert lookup.lookup("missing") == ""  # 未知 block 也算缺键
    assert lookup.lookup("b4") == "丙"
    assert lookup.stats() == {
        "queried": 5,
        "resolved": 2,
        "missing": 3,
        "missing_key": 1,
        "blank_value": 2,
        "lookup_absent": 0,
    }
    assert lookup.heading_for("b1") == "甲>乙"
    assert lookup.heading_for("missing") == ""
    assert lookup.stats()["queried"] == 5  # heading_for 不改变统计


def test_heading_lookup_reports_absent_lookup_for_candidates_with_block_id() -> None:
    """候选有 block_id 却没人传查表：必须可诊断（隐性口径分叉）。"""

    from app.retrieval.evidence_view import build_candidate_views_with_report

    candidate = make_chunk("b1", text="正文", heading_path=None)
    candidate = _CandidateRecord(candidate)
    views, report = build_candidate_views_with_report([candidate], SCOPE)

    assert len(views) == 1
    assert views[0].heading_path == ""
    assert report.heading_lookup_absent == 1
    assert report.lookup_provided is False
    assert report.to_dict()["heading_lookup_absent"] == 1


def test_heading_lookup_report_counts_filled_and_missing() -> None:
    from app.retrieval.evidence_view import build_candidate_views_with_report

    lookup = HeadingLookup.from_blocks(
        [
            ("b1", "正文", {"heading_path": "甲"}),
            ("b2", "正文", {}),
        ]
    )
    first = _CandidateRecord(make_chunk("b1", text="正文", heading_path=None))
    second = _CandidateRecord(make_chunk("b2", text="正文", heading_path=None))
    views, report = build_candidate_views_with_report([first, second], SCOPE, heading_lookup=lookup)

    assert [view.heading_path for view in views] == ["甲", ""]
    assert report.lookup_provided is True
    assert report.heading_lookup_absent == 0
    assert report.heading_missing == 1
    assert report.to_dict() == {
        "chunks": 2,
        "views": 2,
        "from_candidate": 0,
        "from_lookup": 1,
        "heading_missing": 1,
        "heading_lookup_absent": 0,
        "lookup_provided": True,
    }


@dataclass
class _CandidateRecord:
    """最小候选记录：有 block_id 与正文，但**不带** heading_path。"""

    block_id: str
    text: str

    def __init__(self, chunk: Any) -> None:
        self.block_id = chunk.block_id
        self.text = chunk.text


def test_heading_lookup_tolerates_store_without_list_blocks() -> None:
    lookup = HeadingLookup.for_vectorstore(object())
    assert len(lookup) == 0
    assert lookup.lookup("b1") == ""


def test_heading_lookup_never_invents_headings() -> None:
    """没有 heading_path 时只能是空串：不从正文首行截取、不推断层级。"""

    lookup = HeadingLookup.from_blocks([("b1", "## 看起来像标题的第一行", {"source_file": "x.md"})])
    assert lookup.lookup("b1") == ""


async def test_rerank_state_is_disabled_without_provider() -> None:
    service, _chat, _retriever = make_service([make_chunk("b1")])
    assert service.rerank_enabled is False


async def test_selection_status_values_used_by_service_are_known() -> None:
    assert {status.value for status in SelectionStatus} >= {"selected", "empty", "failed", "cancelled"}
    assert AnswerPreferences.__name__ == "AnswerPreferences"


def test_service_does_not_import_provider_http_layer() -> None:
    """T08 边界：ChatService 只依赖 Provider 接口，不解析 HTTP、不 import provider 实现。"""

    import inspect

    import app.generation.chat_service as module

    source = inspect.getsource(module)
    assert "import httpx" not in source
    assert "from app.retrieval.rerank_provider" not in source
    assert "import app.retrieval.rerank_provider" not in source


# ------------------------------------------------------------------ 5. API 层运行态装配


def _write_rerank_settings(store: Any, *, enabled: bool, profile_id: str = "profile-cal-1") -> None:
    from app.core.settings_store import RerankConfig

    app_settings = store.load()
    store.save(
        app_settings.model_copy(
            update={
                "rerank": RerankConfig(
                    enabled=enabled,
                    model="Qwen/Qwen3-Reranker-8B",
                    api_key=KEY,
                    profile_id=profile_id,
                    endpoint="https://api.siliconflow.cn/v1/rerank",
                )
            }
        )
    )


@pytest.fixture(autouse=True)
def _clear_deps_caches() -> Any:
    """deps 的模块级缓存（lru_cache / 全局单例）在测试之间必须隔离。"""

    from app.api import deps

    def clear() -> None:
        for name in ("get_settings_store", "get_state_store", "get_image_state_store"):
            clear_cache = getattr(getattr(deps, name, None), "cache_clear", None)
            if callable(clear_cache):
                clear_cache()
        clear_cache = getattr(deps._rerank_token_counter, "cache_clear", None)
        if callable(clear_cache):
            clear_cache()

    clear()
    yield
    clear()


def _write_profile(data_dir: Any) -> None:
    import json

    profile = {
        "profile_id": "profile-cal-1",
        "status": "validated",
        "provider_type": "siliconflow_rerank",
        "model": "Qwen/Qwen3-Reranker-8B",
        "endpoint_fingerprint": "https://api.siliconflow.cn",
        "instruction_id": "instr-1",
        "instruction_hash": sha256_text("instruction"),
        "tokenizer_id": "stub",
        "view_version": "evidence-view-v1",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "score_threshold": 0.35,
        "max_candidates": 50,
        "min_candidate_count": 5,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "rerank-profiles.json").write_text(
        json.dumps({"profiles": [profile]}, ensure_ascii=False), encoding="utf-8"
    )


def test_build_rerank_runtime_is_disabled_by_default(tmp_path: Any, monkeypatch: Any) -> None:
    from app.api import deps
    from app.core.config import settings as app_config
    from app.core.settings_store import SettingsStore

    store = SettingsStore(tmp_path)
    monkeypatch.setattr(deps, "get_settings_store", lambda: store)
    monkeypatch.setattr(app_config, "data_dir", tmp_path)

    runtime = deps.build_rerank_runtime()

    assert runtime.configured is False
    assert runtime.enabled is False
    assert runtime.config_error_code is None


def test_build_rerank_runtime_builds_provider_and_profile(tmp_path: Any, monkeypatch: Any) -> None:
    from app.api import deps
    from app.core.config import settings as app_config
    from app.core.settings_store import SettingsStore

    store = SettingsStore(tmp_path)
    _write_rerank_settings(store, enabled=True)
    _write_profile(tmp_path)
    monkeypatch.setattr(deps, "get_settings_store", lambda: store)
    monkeypatch.setattr(app_config, "data_dir", tmp_path)

    runtime = deps.build_rerank_runtime()

    assert runtime.config_error_code is None
    assert runtime.enabled is True
    assert runtime.profile is not None and runtime.profile.profile_id == "profile-cal-1"
    assert runtime.provider is not None
    assert runtime.provider.capabilities.model == "Qwen/Qwen3-Reranker-8B"


def test_build_rerank_runtime_reports_profile_mismatch(tmp_path: Any, monkeypatch: Any) -> None:
    from app.api import deps
    from app.core.config import settings as app_config
    from app.core.settings_store import SettingsStore

    store = SettingsStore(tmp_path)
    _write_rerank_settings(store, enabled=True, profile_id="missing-profile")
    _write_profile(tmp_path)
    monkeypatch.setattr(deps, "get_settings_store", lambda: store)
    monkeypatch.setattr(app_config, "data_dir", tmp_path)

    runtime = deps.build_rerank_runtime()

    assert runtime.enabled is False
    assert runtime.config_error_code == "RERANK_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_enabled_but_unconfigured_terminates_turn_instead_of_falling_back() -> None:
    """已启用但配置不可用：SSE 以 error 结束并指向设置，不发送旧候选 meta。"""

    from app.generation.chat_service import RerankRuntime

    service, chat, _retriever = make_service([make_chunk("b1")], provider=None, profile=None)
    service._rerank_runtime = lambda: RerankRuntime(
        config_error_code="RERANK_NOT_CONFIGURED",
        config_error_message="筛选服务尚未配置，请先在设置中完成配置",
    )

    frames = await collect(service, "问题")

    assert [frame["type"] for frame in frames] == ["tool", "tool", "error"]
    assert frames[-1]["message"] == "筛选服务尚未配置，请先在设置中完成配置"
    assert chat.calls == []
    assert not any(frame.get("type") == "meta" for frame in frames)


# ------------------------------------------------- 6. A07：评分文本 == 发送文本

class FakeImageReader:
    """图片核对替身：只记录调用，零出网。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, chunk: RetrievedChunk, question: str) -> str:
        self.calls.append(chunk.block_id)
        return "按本轮问题读取的原图证据：GMV=4826万"


@pytest.mark.asyncio
async def test_scored_path_sends_exactly_the_scored_text_and_never_silently_claims_verification() -> None:
    """A07：筛选路径下发送文本逐字等于被评分文本；未核对的原图必须显式可见（PRD §5.5 过渡期）。

    完整 §5.5（核对结果替换视图并重算 ``payload_hash`` → 移除同图旧文字补充 → 对更新后候选
    再评一次 → 未核对者按 §5.5 排除）归后续批次 T25；过渡期内**禁止在评分后改写正文**，
    否则生成上下文与来源面板用的就不是被评分的文本。本轮宁可不核对，但必须把「未核对」
    显式记入 ``meta.retrieval``，不得静默当成已核对。
    """

    image = make_chunk(
        "img1",
        text="旧识别：GMV=错误数字",
        context_text="旧识别：GMV=错误数字",
        chunk_type="image_description",
    )
    plain = make_chunk("b1", text="文字证据")
    reader = FakeImageReader()
    provider = StubProvider()  # 默认 0.9 > 阈值 0.35：两条视图都入选
    service, _chat, _retriever = make_service([image, plain], provider=provider, image_reader=reader)

    frames = await collect(service, "图片里的 GMV 是多少")
    meta = meta_of(frames)

    assert meta["retrieval"]["status"] == "filtered"
    sources = {source["block_id"]: source["text"] for source in meta["sources"]}
    scored_texts = [document.ranking_text for document in provider.calls[0].documents]
    # 每一条发送给模型/来源面板的文本都能在被评分文档里逐字找到：发送文本 == 已评分文本
    for block_id, text in sources.items():
        assert any(text and text in scored for scored in scored_texts), block_id
    # 评分后不得再改写：旧识别文字原样保留，新读出的数字绝不混入本轮输入
    assert "旧识别：GMV=错误数字" in sources["img1"]
    assert all("4826万" not in text for text in sources.values())
    assert reader.calls == []
    # 未核对不得静默：显式计数，且不伪造 image_verify 进度帧
    assert meta["retrieval"]["image_not_verified"] == 1
    assert not any(frame.get("tool") == "image_verify" for frame in frames)


@pytest.mark.asyncio
async def test_legacy_path_still_verifies_images_when_rerank_is_disabled() -> None:
    """未启用筛选的旧路径行为不变：图片核对照旧发生在生成之前。"""

    image = make_chunk(
        "img1",
        text="旧识别：GMV=错误数字",
        context_text="旧识别：GMV=错误数字",
        chunk_type="image_description",
    )
    reader = FakeImageReader()
    service, _chat, _retriever = make_service([image], provider=None, profile=None, image_reader=reader)

    frames = await collect(service, "图片里的 GMV 是多少")
    meta = meta_of(frames)

    assert reader.calls == ["img1"]
    assert "4826万" in meta["sources"][0]["text"]
    assert "image_not_verified" not in meta["retrieval"]
