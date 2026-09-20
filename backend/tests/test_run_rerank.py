"""相关性筛选（rerank）在 V1.2.0 **run 管线**上的接线与不变量（集成迁移）。

迁移来源：``main`` 的 ``tests/test_chat_rerank.py``。那一版把筛选接线内嵌在
``ChatService`` 里，进度走 V1.1.0 ``/api/chat`` 的 SSE 帧（``tool=rerank``
running/done/error）。V1.2.0 按 PRD 13.2 把 ``/api/chat`` 下线为 **410 Gone**，问答统一走
``POST /api/workspaces/{w}/runs`` + v2 事件，因此同一批不变量必须搬到 run 架构上验证：

* 四态（``filtered`` / ``empty`` / ``disabled`` / ``failed``）与**终止语义**；
* 进度改走 v2 事件 ``tool.started`` / ``tool.completed``（``tool_name="rerank"``）；
* 发送文本 == 被评分文本（A07：生成上下文与来源不得在评分后被改写）；
* 标题从块元数据 ``heading_path`` 补齐并分项计数，**绝不猜标题**；
* 已启用但配置不可用 → 本轮**明确失败**并指向设置，不静默回落到未筛选候选（PRD §8.4）。

全部替身：零出网、零真实 provider、零真实数据目录。
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.agent import runtime as agent_runtime
from app.agent.app_store import AppStore
from app.agent.contracts import (
    AgentKind,
    EventType,
    IntentKind,
    ResearchTask,
    RouteDecision,
    RunStatus,
)
from app.agent.planner import Plan, PlanSettings, UserRequirement
from app.agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolName,
    ToolStatus,
    ToolUnavailableError,
    build_default_registry,
)
from app.retrieval import rerank_pipeline
from app.retrieval.evidence_view import RetrievalScope, sha256_text
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
)
from app.retrieval.vector_retriever import RetrievedChunk
from tests.fakes_run import StubContextFactory, seed_conversation

SCOPE = RetrievalScope.from_where(None, index_revision="chunks__fake__v1")
PROFILE_ID = "profile-cal-1"

_VALID_KEY = "SK-live-key-1234567890"


# =====================================================================
# 确定性替身（零出网）
# =====================================================================


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _text in texts]


class FakeVectorStore:
    """本区向量库替身：``count``（工具前置检查）+ ``list_blocks``（标题查表）。"""

    def __init__(self, blocks: list[tuple[str, str, dict[str, Any]]]) -> None:
        self._blocks = blocks

    def count(self) -> int:
        return len(self._blocks)

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._blocks)


class FakeRetriever:
    """混合检索替身：忽略查询，返回固定候选（顺序即候选阶段顺序）。"""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls = 0

    def retrieve_variants(
        self,
        variants: Any,
        vectors: Any,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return list(self._chunks)


class StubProvider(RerankProvider):
    """筛选服务替身：按候选给固定分（缺省 0.9，可显式给低分/注入错误）。

    低分可以按 ``ranking_text`` 里的标记给（``drop_marker``），这样测试不必知道
    ``candidate_id`` 的派生规则（那是视图层的内部口径）。
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        error: Exception | None = None,
        *,
        default_score: float = 0.9,
        drop_marker: str = "",
        drop_score: float = 0.01,
    ) -> None:
        self._scores = scores or {}
        self._default = default_score
        self._error = error
        self._drop_marker = drop_marker
        self._drop_score = drop_score
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

    def _score_of(self, request: RerankRequest, candidate_id: str, index: int) -> float:
        if candidate_id in self._scores:
            return self._scores[candidate_id]
        if self._drop_marker:
            text = str(getattr(request.documents[index], "ranking_text", ""))
            if self._drop_marker in text:
                return self._drop_score
        return self._default

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
                    score=self._score_of(request, candidate_id, index),
                )
                for index, candidate_id in enumerate(request.candidate_ids)
            ),
            score_scope=ScoreScope.REQUEST_RELATIVE,
        )


@dataclass(frozen=True)
class Runtime:
    """运行态替身（与 ``app.api.deps.build_rerank_runtime`` 的返回值**同形**）。

    刻意不用 ``ChatService.RerankRuntime``：run 管线的接线层只按属性读，本替身同时是
    「接线层不反向依赖 generation 层」的证据。
    """

    provider: RerankProvider | None = None
    profile: RerankProfile | None = None
    config_error_code: str | None = None
    config_error_message: str | None = None

    @property
    def enabled(self) -> bool:
        return self.provider is not None and self.profile is not None

    @property
    def configured(self) -> bool:
        return self.enabled or self.config_error_code is not None


@dataclass
class ProgressRecorder:
    """记录工具层上报的筛选进度（``(state, error_code)``）。"""

    events: list[tuple[str, str]] = field(default_factory=list)

    async def __call__(self, state: str, error_code: str = "") -> None:
        self.events.append((state, error_code))


@dataclass(frozen=True)
class StubWorkspace:
    """工具绑定的工作区（只含工具层读取的只读属性）。"""

    workspace_id: str = "ws-1"
    knowledge_revision: int = 3


def make_profile(**overrides: Any) -> RerankProfile:
    values: dict[str, Any] = {
        "profile_id": PROFILE_ID,
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


def _blocks_of(chunks: list[RetrievedChunk]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (chunk.block_id, chunk.text, {"source_file": chunk.source_file, "heading_path": chunk.heading_path})
        for chunk in chunks
    ]


def make_context(
    chunks: list[RetrievedChunk],
    *,
    provider: RerankProvider | None = None,
    profile: RerankProfile | None = None,
    runtime: Any = None,
    blocks: list[tuple[str, str, dict[str, Any]]] | None = None,
    progress: ProgressRecorder | None = None,
    agent_kind: AgentKind = AgentKind.RAG,
) -> ToolContext:
    """组装一次检索调用的上下文（rerank 运行态按需注入）。"""

    resolved = runtime
    if resolved is None and provider is not None:
        resolved = Runtime(provider=provider, profile=profile or make_profile())
    return ToolContext(
        workspace=StubWorkspace(),
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(chunks),
        vector_store=FakeVectorStore(blocks if blocks is not None else _blocks_of(chunks)),
        rerank=resolved,
        rerank_scope=SCOPE,
        rerank_progress=progress,
        agent_kind=agent_kind,
    )


async def search(context: ToolContext, query: str = "问题", **extra: Any) -> Any:
    registry = build_default_registry()
    return await registry.execute(
        ToolName.SEARCH_KNOWLEDGE, {"query": query, **extra}, context
    )


# =====================================================================
# ① filtered：只返回入选来源，并携带 relevance_* 字段
# =====================================================================


async def test_filtered_returns_selected_chunks_with_relevance_fields() -> None:
    chunks = [make_chunk("b1"), make_chunk("b2")]
    provider = StubProvider()  # 0.9 > 档案阈值 0.35
    progress = ProgressRecorder()

    result = await search(make_context(chunks, provider=provider, progress=progress))

    assert result.status is ToolStatus.OK
    assert result.data["count"] == 2
    assert result.data["candidate_count"] == 2
    assert result.data["retrieval"]["status"] == rerank_pipeline.STATUS_FILTERED
    assert result.data["retrieval"]["profile_id"] == PROFILE_ID
    assert result.data["retrieval"]["selected_count"] == 2
    assert len(provider.calls) == 1  # 评分只提交一次（同一批分数既选择又展示）

    record = result.data["chunks"][0]
    assert record["match_type"] == "rerank"
    assert record["similarity"] == pytest.approx(0.9)  # 向量相似度不被重排分数覆盖
    assert record["relevance_checked"] is True
    assert record["score_type"] == "rerank"
    assert record["profile_id"] == PROFILE_ID
    assert record["rerank_score"] is not None
    assert record["evidence_hash"]
    # 进度：running → done（v2 事件由运行层据此发布）
    assert progress.events == [("running", ""), ("done", "")]


async def test_sources_use_the_scored_text_not_the_hydrated_context() -> None:
    """旧路径会用 ``context_text`` 把同标题补充文字塞进来源；筛选路径只用证据正文。"""

    evidence = "证据正文第一段。证据正文第二段。"
    hydrated = evidence + "\n\n同标题补充信息：\n这段文字未经评分检查，不得进入本轮输入。"
    chunk = make_chunk("b1", text=evidence, context_text=hydrated)
    provider = StubProvider()

    result = await search(make_context([chunk], provider=provider))

    record = result.data["chunks"][0]
    assert record["text"] == evidence
    assert "未经评分检查" not in record["text"]
    # 被评分的 ranking_text 必须**包含**最终发送的正文（发送文本 == 被评分文本）
    ranking_text = provider.calls[0].documents[0].ranking_text
    assert evidence in ranking_text


async def test_top_k_still_bounds_the_returned_chunks() -> None:
    chunks = [make_chunk(f"b{index}") for index in range(1, 5)]
    provider = StubProvider()

    result = await search(make_context(chunks, provider=provider), top_k=2)

    assert result.data["count"] == 2
    assert result.data["retrieval"]["selected_count"] == 4


# =====================================================================
# ② 标题：从块元数据补齐，缺失留空并分项计数（绝不猜）
# =====================================================================


async def test_selected_heading_comes_from_block_metadata() -> None:
    chunk = make_chunk("b1", text="正文", heading_path=None)
    provider = StubProvider()
    blocks = [("b1", "正文", {"heading_path": "示例风控方案 > 支持的策略", "source_file": "x.md"})]

    result = await search(make_context([chunk], provider=provider, blocks=blocks))

    assert result.data["chunks"][0]["heading_path"] == "示例风控方案 > 支持的策略"
    ranking_text = provider.calls[0].documents[0].ranking_text
    assert "支持的策略" in ranking_text


async def test_missing_heading_stays_empty_and_is_counted() -> None:
    """两类缺失必须分开计数：块清单里取不到标题（空值）/ 块根本不在清单里（缺键）。"""

    chunks = [
        make_chunk("b1", text="正文", heading_path=None),
        make_chunk("b2", text="第二段", heading_path=None),
    ]
    provider = StubProvider()
    # b1 在块清单里但元数据没有 heading_path；b2 完全不在块清单里。
    blocks = [("b1", "正文", {"source_file": "x.md"})]

    result = await search(make_context(chunks, provider=provider, blocks=blocks))

    assert [record["heading_path"] for record in result.data["chunks"]] == ["", ""]
    assert result.data["retrieval"]["missing_headings"] == 2
    assert result.data["retrieval"]["heading_diagnostics"] == {
        "heading_missing_key": 1,
        "heading_blank_value": 1,
        "heading_lookup_absent": 0,
    }
    # 绝不从正文首行猜标题：模板里的标题行保持为空
    ranking_text = provider.calls[0].documents[0].ranking_text
    assert "\n标题：\n" in ranking_text


# =====================================================================
# ③ empty / disabled
# =====================================================================


async def test_empty_when_all_candidates_below_threshold() -> None:
    provider = StubProvider(default_score=0.01)  # 全部低于阈值 0.35
    progress = ProgressRecorder()

    result = await search(
        make_context([make_chunk("b1"), make_chunk("b2")], provider=provider, progress=progress)
    )

    assert result.status is ToolStatus.IRRELEVANT
    assert result.data["code"] == "relevance_empty"
    assert result.data["chunks"] == []
    assert result.data["candidate_count"] == 2
    assert result.data["retrieval"]["status"] == rerank_pipeline.STATUS_EMPTY
    assert result.data["retrieval"]["selected_count"] == 0
    assert progress.events == [("running", ""), ("done", "")]


async def test_disabled_keeps_the_legacy_path_and_reports_no_progress() -> None:
    chunk = make_chunk("b1", context_text="旧补充文字")
    progress = ProgressRecorder()

    result = await search(make_context([chunk], runtime=None, progress=progress))

    assert result.status is ToolStatus.OK
    assert result.data["chunks"][0]["text"] == "旧补充文字"  # 旧路径行为不变
    assert "relevance_checked" not in result.data["chunks"][0]
    assert "retrieval" not in result.data
    assert progress.events == []  # 未启用不得假装筛选过


async def test_disabled_without_recall_still_reports_empty() -> None:
    """索引非空但召回为空：旧路径如实报 empty（不是 index_empty）。"""

    blocks = [("b1", "已入库的正文", {"heading_path": "s1"})]
    result = await search(make_context([], runtime=None, blocks=blocks))

    assert result.status is ToolStatus.EMPTY
    assert result.data["code"] == "no_results"


async def test_index_empty_short_circuits_before_rerank() -> None:
    """索引为空时先返回 index_empty：不读配置、不上报筛选进度。"""

    progress = ProgressRecorder()
    context = make_context(
        [make_chunk("b1")],
        runtime=Runtime(
            config_error_code="RERANK_NOT_CONFIGURED", config_error_message="未配置"
        ),
        blocks=[],
        progress=progress,
    )

    result = await search(context)

    assert result.status is ToolStatus.EMPTY
    assert result.data["code"] == "index_empty"
    assert progress.events == []


# =====================================================================
# ④ failed：明确失败并指向设置，**不静默回落**（PRD §8.4）
# =====================================================================


async def test_configuration_failure_terminates_instead_of_falling_back() -> None:
    progress = ProgressRecorder()
    runtime = Runtime(
        config_error_code="RERANK_NOT_CONFIGURED",
        config_error_message="筛选服务尚未配置，请先在设置中完成配置",
    )
    context = make_context(
        [make_chunk("b1")], runtime=runtime, progress=progress
    )

    with pytest.raises(ToolUnavailableError) as info:
        await search(context)

    assert info.value.code is ToolErrorCode.NOT_CONFIGURED
    assert "设置" in info.value.message
    # 具体筛选错误码经进度事件透出，运行层据此发布 tool.completed.error_code
    assert progress.events == [("running", ""), ("error", "RERANK_NOT_CONFIGURED")]


async def test_provider_failure_is_a_service_error_with_no_partial_results() -> None:
    provider = StubProvider(error=RerankAuthFailedError("HTTP 401"))
    progress = ProgressRecorder()

    with pytest.raises(ToolExecutionError) as info:
        await search(make_context([make_chunk("b1")], provider=provider, progress=progress))

    assert info.value.code is ToolErrorCode.SERVICE_ERROR
    assert progress.events == [("running", ""), ("error", "RERANK_AUTH_FAILED")]


async def test_cancelled_scoring_propagates_and_yields_no_chunks() -> None:
    class CancelledProvider(StubProvider):
        async def rerank(self, request: RerankRequest) -> RerankResult:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await search(make_context([make_chunk("b1")], provider=CancelledProvider()))


# =====================================================================
# ⑤ 接线层自身的契约（runtime 解析 / 边界）
# =====================================================================


def test_runtime_configured_distinguishes_disabled_from_misconfigured() -> None:
    assert rerank_pipeline.runtime_configured(None) is False
    assert rerank_pipeline.runtime_configured(Runtime()) is False
    assert rerank_pipeline.runtime_configured(Runtime(provider=StubProvider())) is True
    assert rerank_pipeline.runtime_configured(Runtime(profile=make_profile())) is True
    assert (
        rerank_pipeline.runtime_configured(Runtime(config_error_code="RERANK_NOT_CONFIGURED"))
        is True
    )


def test_production_runtime_shape_is_accepted() -> None:
    """``deps.build_rerank_runtime()`` 的返回值必须被接线层直接接受（同一运行态契约）。"""

    from app.api import deps
    from app.core.settings_store import SettingsStore

    runtime = deps.build_rerank_runtime(SettingsStore(Path(".") / "does-not-exist"))

    assert hasattr(runtime, "provider") and hasattr(runtime, "profile")
    assert rerank_pipeline.runtime_configured(runtime) is False  # 未启用 ⇒ 走旧检索路径
    assert rerank_pipeline.resolve_runtime(runtime) is runtime


def test_pipeline_does_not_import_the_provider_http_layer() -> None:
    """边界：接线层只依赖 Provider 接口，不解析 HTTP、不 import provider 实现。"""

    import inspect

    source = inspect.getsource(rerank_pipeline)
    assert "import httpx" not in source
    assert "from app.retrieval.rerank_provider" not in source
    assert "import app.retrieval.rerank_provider" not in source


def test_selection_status_values_used_by_the_pipeline_are_known() -> None:
    from app.retrieval.reranker import SelectionStatus

    assert {status.value for status in SelectionStatus} >= {
        "selected",
        "empty",
        "failed",
        "cancelled",
    }
    assert _VALID_KEY  # 测试替身只用假密钥，避免误用真实凭证


# =====================================================================
# ⑥ run API：真实 RunManager + v2 事件（迁移的落点）
# =====================================================================

QUESTION = "笔记里说橘子属于什么类水果？"
ORANGE_TEXT = "橘子属于柑橘类水果，果皮可入药，果肉多汁。"
POLAR_TEXT = "北极熊生活在北极，是最大的陆地食肉动物。"


@pytest.fixture(autouse=True)
def _zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """PRD 隐私红线：本文件全部用例阻断一切**非回环**连接（回环放行）。"""

    original_create = socket.create_connection

    def guarded_create(address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"测试期间发生非回环外呼：{host!r}")
        return original_create(address, *args, **kwargs)

    original_connect = socket.socket.connect

    def guarded_connect(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"测试期间发生非回环外呼：{host!r}")
        return original_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", guarded_create)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离设置读取 **与** 三个模型装配点（PRD 隐私红线：测试绝不真出网）。

    只隔离 ``get_settings_store()`` **不够**：config.json 缺失时 ``load()`` 会回退到
    本机 ``.env``（后端仓库里确实存在真实 provider 凭证），于是 run 链路的
    synthesis / verify 会真的构造 provider 并发起外呼（本工作流实测到
    ``AuthenticationError``）。因此这里把三个装配点直接置为「未配置」，走确定性渲染与
    纯程序核对——与 ``test_seam_assembly`` / ``test_lifespan_assembly`` 同一做法。
    """

    from app.core import settings_store as settings_store_mod

    empty = settings_store_mod.SettingsStore(tmp_path / "isolated-settings")
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: empty)
    monkeypatch.setattr(agent_runtime, "general_model_for_chat", lambda *a, **k: None)
    monkeypatch.setattr(agent_runtime, "synthesis_model_for_chat", lambda *a, **k: None)
    monkeypatch.setattr(agent_runtime, "model_verifier_for_chat", lambda *a, **k: None)


def _rag_services_factory(runtime: Any) -> Any:
    """真实 graph 服务 + 确定性 RAG 路由：只覆盖 route/plan，其余全走真实链路。"""

    def factory(manager: Any, run: Any, state: Any) -> Any:
        real = agent_runtime.default_services(manager, run, state)
        question = str((state or {}).get("question") or QUESTION)

        async def route(current: Any) -> RouteDecision:
            del current
            return RouteDecision(
                question=question,
                resolved_question=question,
                intent=IntentKind.RESEARCH,
                needs_knowledge=True,
            )

        async def plan(current: Any) -> Plan:
            del current
            requirement = UserRequirement(
                requirement_id="r1", text=question, quote=question
            )
            return Plan(
                question=question,
                workspace_id=str(run.workspace_id),
                requirements=[requirement],
                tasks=[
                    ResearchTask(
                        task_id="t1",
                        question=question,
                        requirement_ids=["r1"],
                        agent_kind=AgentKind.RAG,
                    )
                ],
                settings=PlanSettings(),
            )

        return real.model_copy(update={"route": route, "plan": plan})

    return factory


@dataclass
class RunEnv:
    store: AppStore
    manager: Any

    async def close(self) -> None:
        await self.manager.aclose()
        await self.store.close()


async def _build_run_env(tmp_path: Path, *, runtime: Any) -> RunEnv:
    """真实 RunManager（真实工具注册表 + 真实研究/合成/核对）+ 筛选接线。"""

    chunks = [
        make_chunk("b-orange", text=ORANGE_TEXT, heading_path="水果 > 柑橘类"),
        make_chunk("b-polar", text=POLAR_TEXT, heading_path="动物 > 北极熊"),
    ]
    blocks = _blocks_of(chunks)
    provider = runtime.provider if runtime is not None else None
    policy = None

    def tool_services(context: Any) -> dict[str, Any]:
        del context
        return {
            "vector_store": FakeVectorStore(blocks),
            "retriever": FakeRetriever(chunks),
            "embedder": FakeEmbedder(),
            "search_provider": None,
            "extract_provider": None,
            "_catalog": None,
            "_memory": None,
            "rerank": (lambda: runtime) if runtime is not None else None,
            "rerank_scope": SCOPE,
        }

    store = await AppStore.open(tmp_path / "app.db")
    await seed_conversation(store)
    manager = agent_runtime.RunManager(
        store=store,
        context_factory=StubContextFactory(root=tmp_path / "ws-root"),
        registry=None,  # 用真实工具注册表（含 search_knowledge 的真实实现）
        tool_services=tool_services,
        services_factory=_rag_services_factory(runtime),
        private_names=(),
    )
    del policy, provider
    return RunEnv(store=store, manager=manager)


async def _run(env: RunEnv) -> Any:
    run = await env.manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question=QUESTION
    )
    await env.manager.run_to_completion(run.id, timeout=180)
    return run.id, await env.manager.read_events(run.id)


def _events_of(events: Any, event_type: EventType) -> list[Any]:
    return [event for event in events if event.type is event_type]


def _evidence_texts(events: Any) -> str:
    return " ".join(
        str(item.get("snippet") or "")
        for event in _events_of(events, EventType.EVIDENCE_ADDED)
        for item in event.payload.get("items", [])
    )


async def test_run_api_reports_rerank_progress_and_filters_evidence(tmp_path: Path) -> None:
    """启用筛选：进度走 v2 ``tool.*`` 事件，且只有入选候选进入证据。"""

    provider = StubProvider(drop_marker="北极熊")  # 只留橘子那条（阈值 0.35）
    env = await _build_run_env(
        tmp_path, runtime=Runtime(provider=provider, profile=make_profile())
    )
    try:
        run_id, events = await _run(env)
        run = await env.store.get_run(run_id)
        assert run.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}, [
            (event.type.value, event.payload) for event in events
        ]

        started = _events_of(events, EventType.TOOL_STARTED)
        completed = _events_of(events, EventType.TOOL_COMPLETED)
        assert started, [event.type.value for event in events]
        assert completed
        assert {item.payload.get("tool_name") for item in started} == {"rerank"}
        assert {item.payload.get("tool_name") for item in completed} == {"rerank"}
        assert {item.payload.get("status") for item in completed} == {"succeeded"}
        # 事件顺序：started → completed → answer.meta（进度先于答案）
        assert started[0].event_seq < completed[0].event_seq
        meta = _events_of(events, EventType.ANSWER_META)
        assert meta and completed[-1].event_seq < meta[0].event_seq

        # 证据只来自入选候选：被筛掉的极地熊不得进入本轮材料
        texts = _evidence_texts(events)
        assert "橘子" in texts
        assert "北极熊" not in texts
        # 评分文本确实包含最终进入证据的正文（发送文本 == 被评分文本）
        assert any(
            ORANGE_TEXT in document.ranking_text
            for call in provider.calls
            for document in call.documents
        )
    finally:
        await env.close()


async def test_run_api_without_rerank_emits_no_rerank_progress(tmp_path: Path) -> None:
    """未启用：run 链路上不出现任何 rerank 进度事件（不假装筛选过）。"""

    env = await _build_run_env(tmp_path, runtime=None)
    try:
        run_id, events = await _run(env)
        run = await env.store.get_run(run_id)
        assert run.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}
        assert not _events_of(events, EventType.TOOL_STARTED)
        assert not _events_of(events, EventType.TOOL_COMPLETED)
        # 旧路径：两条候选都在
        texts = _evidence_texts(events)
        assert "橘子" in texts
    finally:
        await env.close()


async def test_run_api_fails_the_run_when_rerank_is_misconfigured(tmp_path: Path) -> None:
    """已启用但配置不可用：run 明确失败并指向设置，**不**静默返回未筛选候选。"""

    runtime = Runtime(
        config_error_code="RERANK_NOT_CONFIGURED",
        config_error_message="筛选服务尚未配置，请先在设置中完成配置",
    )
    env = await _build_run_env(tmp_path, runtime=runtime)
    try:
        run_id, events = await _run(env)
        run = await env.store.get_run(run_id)

        assert str(run.status) == RunStatus.FAILED.value
        failures = _events_of(events, EventType.RUN_FAILED)
        assert failures, [event.type.value for event in events]
        error = failures[-1].payload["error"]
        assert error["error_code"] == ToolErrorCode.NOT_CONFIGURED.value
        assert "设置" in error["message"]
        # 具体筛选错误码经进度事件透出（供前端指向设置页）
        failed_tool = [
            item
            for item in _events_of(events, EventType.TOOL_COMPLETED)
            if item.payload.get("status") == "failed"
        ]
        assert failed_tool and failed_tool[-1].payload["error_code"] == "RERANK_NOT_CONFIGURED"
        # 失败不得先发一份「成功答案」
        assert not _events_of(events, EventType.ANSWER_META)
    finally:
        await env.close()
