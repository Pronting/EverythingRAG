"""装配缝守卫与四处接线的真实装配证据（t33）。

本项目反复出现「**seam 声明了、装配处没填**」（`memory_enabled` 恒 True、默认工作区不在
app.db、`general_model` 未接、以及本轮的四处）。本文件做两件事：

1. 为四处接线给出**真实装配路径**下的证据（不是测试自建装配）：
   - `research_graph`：主图 `_invoke` 现编译研究子图 → 每任务 thread 的检查点确实存在；
   - `history_search`：`app/api/run_services.tool_services_for` → `ToolContext` → `search_history`
     走**委托路径**（FTS5），且 `total` / `conversation_count` 为 **null 时不猜数字**；
   - `synthesize(model=)`：候选答案正文来自模型输出（带 `{{ev-…}}` 占位符），模型不守规矩或
     未配置时**回落**确定性渲染。
2. **装配守卫**：把「应当被填充的 seam」列表化，断言真实装配后没有漏填；并证明守卫**非永真**
   （故意抽掉一个 seam 时必须报红）。条件性 seam（搜索/视觉/历史索引/provider）区分
   「因未配置而为 None（合法）」与「已配置却没人填（缺陷）」。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.agent import runtime as agent_runtime
from app.agent.contracts import (
    AnswerBasis,
    EvidenceSourceType,
    RunStatus,
)
from app.agent.research import RequirementState
from app.agent.verification import (
    SynthesisUnavailable,
    synthesize_answer,
)
from app.generation.base import ChatChunk
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.fakes_memory import workspace_paths_for
from tests.fakes_run import build_manager, seed_conversation

ANSWER_TEXT = "限流可以用令牌桶实现并发整形。"


class FakeChatModel:
    """确定性假对话模型（零出网）：返回脚本化文本，并记录收到的提示词。"""

    def __init__(self, text: str = ANSWER_TEXT) -> None:
        self.text = text
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk("content", self.text)


@pytest.fixture(autouse=True)
def _isolate_chat_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 `get_settings_store()` 的直接调用（conftest 只覆盖 FastAPI 依赖）。

    不隔离会读到本机 `.env` 里的真实 provider —— 既不确定、也会**真的出网**（t31 实测）。
    """

    from app.core import settings_store as settings_store_mod

    empty = settings_store_mod.SettingsStore(tmp_path / "empty-settings")
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: empty)


def _fake_provider(monkeypatch: pytest.MonkeyPatch, text: str = ANSWER_TEXT) -> FakeChatModel:
    """把「已配置的对话 provider」注入唯一装配点（见 runtime.general_model_for_chat）。"""

    model = FakeChatModel(text)
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: model
    )
    return model


def _unconfigured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.generation.providers import ChatProviderError

    def _raise(config: Any) -> Any:
        raise ChatProviderError("对话模型未配置，缺少: 对话模型 Base URL")

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", _raise
    )


# =====================================================================
# ① research_graph：研究子图真的被用，且每任务 thread 有检查点
# =====================================================================


async def test_research_subgraph_runs_with_per_task_checkpoints(
    app_store, tmp_path: Path
) -> None:
    """真实 `_invoke` 路径：研究子图生效 ⇒ 检查点里有**每任务** thread（PRD 12.2/12.4）。"""

    from tests.fakes_run import FakeServices

    await seed_conversation(app_store)
    manager = build_manager(
        app_store, services=FakeServices(task_count=2), root=tmp_path, use_langgraph=True
    )
    manager.enable_checkpoints()
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        await manager.run_to_completion(run.id)
        assert (await manager.require_run(run.id)).status in {
            RunStatus.COMPLETED,
            RunStatus.COMPLETED_WITH_GAPS,
        }

        checkpoint = tmp_path / "ws-1" / "checkpoints.db"
        assert checkpoint.is_file(), (
            "每工作区检查点文件必须存在（PRD 12.2）—— `AsyncSqliteSaver.from_conn_string` "
            "是异步上下文管理器，用同步 enter_context 会静默失败，检查点就永远不会建立"
        )
        connection = sqlite3.connect(checkpoint)
        try:
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            threads: set[str] = set()
            for table in tables:
                columns = {
                    str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
                }
                if "thread_id" not in columns:
                    continue
                threads |= {
                    str(row[0])
                    for row in connection.execute(f"SELECT DISTINCT thread_id FROM {table}")
                }
        finally:
            connection.close()
        assert threads, "主图必须写入检查点"
        assert f"run:{run.id}:main" in threads
        # **每任务独立 thread** 正是「研究子图生效」的判据（扁平路径不会写这些 thread）
        task_threads = {item for item in threads if ":task:" in item}
        assert task_threads, f"研究子图必须为每任务写检查点，实际 thread={threads}"
        assert any(item.startswith(f"run:{run.id}:task:") for item in task_threads)
    finally:
        await manager.aclose()


# =====================================================================
# ② history_search：委托 FTS5 接缝（不猜 total/conversation_count）
# =====================================================================


def _history_wired_env(app_store_file: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """让 `tool_services_for` 产出真实的 `history_search` 接缝（重 provider 换替身）。"""

    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.workspace_context import WorkspaceContextFactory
    from app.core.workspace_store import WorkspaceStore

    workspace_store = WorkspaceStore(tmp_path / "conv-data")
    workspace_store.create("甲区", "ws-1")
    factory = WorkspaceContextFactory(tmp_path / "erag-data", workspace_store)
    context = factory.resolve("ws-1")

    class _FakeVectorStore:
        def count(self) -> int:
            return 0

        def count_files(self) -> int:
            return 0

    monkeypatch.setattr(api_deps, "get_embedder", lambda: object())
    monkeypatch.setattr(rs, "HybridRetriever", lambda store, **kwargs: object())
    monkeypatch.setattr(rs, "catalog_for_area", lambda ctx: object())
    monkeypatch.setattr(api_deps, "get_workspace_knowledge_area", lambda ctx: object())
    return context, _FakeVectorStore()


async def test_search_history_delegates_to_fts_seam_without_guessing_counts(
    tmp_path: Path, app_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """装配路径：`tool_services_for` → `ToolContext` → `search_history` 走 FTS5 委托路径。"""

    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.config import settings
    from app.memory import history as history_mod

    settings.data_dir = tmp_path / "erag-data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.data_dir / "app.db"
    # 真实 app.db 文件（含 history_fts 表）+ 一条可命中消息
    store = await type(app_store).open(db_path)
    api_deps.register_app_store(store)
    try:
        await store.create_workspace("ws-1", "ws-1", name="甲区")
        await store.create_conversation("conv-1", "ws-1", title="限流讨论")
        message = await store.append_message(
            conversation_id="conv-1", role="user", content="限流用令牌桶，记得压测。"
        )
        index = history_mod.WorkspaceHistoryIndex(
            store,
            workspace_id="ws-1",
            paths=workspace_paths_for(tmp_path, "ws-1"),
            terms=(),
        )
        indexed = await index.rebuild()
        assert indexed["indexed"] >= 1, "FTS 索引必须真的写入（否则本用例会假通过）"

        context, fake_vector_store = _history_wired_env(store, tmp_path, monkeypatch)
        monkeypatch.setattr(
            api_deps, "get_workspace_vector_store", lambda ctx: fake_vector_store
        )
        monkeypatch.setattr(api_deps, "workspace_state_store", lambda ctx: object())
        monkeypatch.setattr(api_deps, "workspace_image_state_store", lambda ctx: object())

        services = rs.tool_services_for(context)
        assert services["history_search"] is not None, "FTS 索引可用时接缝必须被填充"

        # 直接调接缝：命中 + total/conversation_count 为 null（不猜数字）
        payload = await services["history_search"]("限流", {"limit": 5})
        assert any(message.id == item["message_id"] for item in payload["messages"])
        assert payload["total"] is None and payload["conversation_count"] is None
        assert payload["next_cursor"] is None
    finally:
        api_deps.register_app_store(None)
        await store.close()


async def test_history_seam_is_none_when_index_unavailable_no_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """条件性 seam：app.db 文件不存在时接缝为 None（合法）→ 工具走有界回退扫描。"""

    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.config import settings

    settings.data_dir = tmp_path / "erag-data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    api_deps.register_app_store(None)
    context, _ = _history_wired_env(None, tmp_path, monkeypatch)
    assert rs.history_search_seam(context) is None


# =====================================================================
# ④ synthesize(model=)：正文必须来自模型；不守引用格式时先纠正、仍不符即明确失败
# =====================================================================


def _synthesis_request(*, supported: bool = False) -> Any:
    """候选答案请求。

    ``supported=True`` 时证据带 ``supported_requirements`` —— 只有这类证据才会被
    ``synthesis_model_for_chat`` 当作**可引用**材料，从而触发「必须带占位符」的
    硬性检查（不带该字段的证据只作为背景材料，见函数内 ``citable`` 的说明）。
    """

    from app.agent.contracts import Evidence
    from app.agent.research import ResearchConclusion
    from app.agent.verification import SynthesisRequest

    evidence = Evidence(
        evidence_id="ev-1",
        workspace_id="ws-1",
        run_id="run-1",
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref="doc-a",
        content="限流用令牌桶实现。",
        content_hash="h",
        retrieved_at="2026-09-20T00:00:00Z",
        supported_requirements=["R1"] if supported else [],
    )
    return SynthesisRequest(
        question="限流怎么做",
        content="",
        conclusions=[
            ResearchConclusion(
                statement="限流用令牌桶实现", evidence_ids=["ev-1"], requirement_ids=["R1"]
            )
        ],
        evidence=[evidence],
        requirements={"R1": "限流怎么做"},
    )


def test_synthesis_uses_model_text_when_placeholders_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_provider(monkeypatch, "限流用令牌桶实现{{ev-1}}。")
    call = agent_runtime.synthesis_model_for_chat()
    assert call is not None
    text = call(_synthesis_request())
    assert text == "限流用令牌桶实现{{ev-1}}。"
    assert model.calls, "模型必须真的被调用"
    # 职责/偏好走 system 提示，结论与证据占位符仍在用户消息里（PRD 9.3 固定顺序）
    assert any(
        item.get("role") == "user" and "{{ev-1}}" in str(item.get("content") or "")
        for item in model.calls[0]
    )


def test_synthesis_corrects_once_then_fails_when_model_drops_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型不守占位符要求 ⇒ 走一次明确的纠正调用；仍不符合即失败（不发布无来源的答案）。

    旧实现会回落到确定性拼接稿（``render_candidate``），产出的正文看着像答案、实际是
    「内部标识符 + 原文首句」的拼接（用户实测缺陷，见 ``synthesis_model_for_chat`` 的
    说明）。这条兜底路径已按要求删除，因此这里断言新的正确行为：两次都不给占位符时
    明确抛 ``SynthesisUnavailable``，绝不用拼接稿冒充回答。
    """

    _fake_provider(monkeypatch, "限流用令牌桶实现，具体见文档。")  # 两次都缺 {{ev-…}}
    call = agent_runtime.synthesis_model_for_chat()
    assert call is not None
    with pytest.raises(SynthesisUnavailable) as failure:
        call(_synthesis_request(supported=True))
    assert "两次" in str(failure.value), "失败原因必须点明「两次都没有按引用格式输出」"

    _unconfigured_provider(monkeypatch)
    assert agent_runtime.synthesis_model_for_chat() is None, "未配置 ⇒ 没有可用模型，明确失败"


def test_synthesis_correction_gives_up_after_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两次都拿不到占位符 ⇒ ``SynthesisUnavailable``（不发布无来源的答案）。"""

    calls: list[int] = []

    class _Model:
        async def stream_chat(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
            calls.append(len(messages))
            yield ChatChunk(kind="content", text="限流用令牌桶实现，具体见文档。")

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda _config: _Model()
    )
    call = agent_runtime.synthesis_model_for_chat()
    assert call is not None
    with pytest.raises(SynthesisUnavailable):
        call(_synthesis_request(supported=True))
    assert len(calls) == 2, "只允许一次纠正调用，之后明确失败"


def test_synthesis_without_model_fails_instead_of_rendering_deterministically() -> None:
    """无模型时 `synthesize_answer` 明确失败，不再产出确定性拼接稿。"""

    with pytest.raises(SynthesisUnavailable):
        synthesize_answer(
            "限流怎么做",
            [],
            requirements={"R1": "限流怎么做"},
            run_id="run-1",
            workspace_id="ws-1",
        )


# =====================================================================
# ⑤ 装配守卫（非永真 + 条件性区分）
# =====================================================================

#: `tool_services_for` 里**无条件必填**的键（缺任一即为装配缺陷）。
REQUIRED_TOOL_SEAMS: tuple[str, ...] = (
    "vector_store",
    "embedder",
    "state_store",
    "image_store",
    "retriever",
    "knowledge_area",
    "_catalog",
    "_memory",
)

#: **条件性** seam：键存在但值可为 None（因该项未配置）；键本身**必须存在**。
CONDITIONAL_TOOL_SEAMS: tuple[str, ...] = (
    "search_provider",
    "extract_provider",
    "image_reader",
    "history_search",
)


def unfilled_seams(
    services: Mapping[str, Any], *, configured: Mapping[str, bool]
) -> list[str]:
    """返回漏填的 seam：必填项 None，或缺键；条件项在「已配置」时不得为 None。"""

    missing: list[str] = []
    for key in REQUIRED_TOOL_SEAMS:
        if services.get(key) is None:
            missing.append(key)
    for key in CONDITIONAL_TOOL_SEAMS:
        if key not in services:
            missing.append(f"{key}(键缺失)")
        elif configured.get(key) and services.get(key) is None:
            missing.append(f"{key}(已配置却为 None)")
    for key in ("general_model", "synthesis_model"):
        if configured.get(key) and services.get(key) is None:
            missing.append(f"{key}(已配置却为 None)")
    return missing


def test_assembly_guard_detects_unfilled_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """守卫：真实装配后没有漏填；**抽掉一个 seam 必须报红**（非永真）。"""

    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.config import settings

    settings.data_dir = tmp_path / "erag-data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "app.db").write_bytes(b"")  # 只为让 history_search 走"可用"分支
    context, fake_vector_store = _history_wired_env(None, tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_deps, "get_workspace_vector_store", lambda ctx: fake_vector_store
    )
    monkeypatch.setattr(api_deps, "workspace_state_store", lambda ctx: object())
    monkeypatch.setattr(api_deps, "workspace_image_state_store", lambda ctx: object())
    monkeypatch.setattr(rs, "is_search_configured", lambda config: True)
    monkeypatch.setattr(rs, "is_vision_configured", lambda config: True)
    monkeypatch.setattr(rs, "create_search_provider_from_config", lambda config: object())
    monkeypatch.setattr(rs, "create_vision_model_from_config", lambda config: object())

    services = dict(rs.tool_services_for(context))
    _fake_provider(monkeypatch)
    services.update(
        {
            "general_model": agent_runtime.general_model_for_chat(),
            "synthesis_model": agent_runtime.synthesis_model_for_chat(),
        }
    )
    configured = dict.fromkeys(
        [*CONDITIONAL_TOOL_SEAMS, "general_model", "synthesis_model"],
        True,
    )
    assert unfilled_seams(services, configured=configured) == []

    # 非永真证明：抽掉一个条件 seam 的**键** ⇒ 守卫必须报出来（与是否配置无关：键都不在了）
    mutated = {key: value for key, value in services.items() if key != "history_search"}
    assert "history_search(键缺失)" in unfilled_seams(mutated, configured=configured)
    # 抽掉一个必填 seam ⇒ 必须报出来
    mutated_required = dict(services) | {"retriever": None}
    assert "retriever" in unfilled_seams(mutated_required, configured=configured)
    # 条件区分：条件项为 None 且**未配置** ⇒ 合法，不得误报（键仍在）
    lenient = dict(services) | {
        "search_provider": None,
        "extract_provider": None,
        "image_reader": None,
        "history_search": None,
    }
    assert unfilled_seams(lenient, configured=dict.fromkeys(configured, False)) == []


def test_lifespan_entry_seams_are_consumable_by_real_tool_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """守卫**扩展**（t34）：lifespan 入口（`run_services.tool_services_for`）的产物必须能被
    真实 `ToolContext` 消费。

    只检查「键存在 / 非 None」是不够的：`_catalog` 早先被塞进了 **WorkspaceContext**
    （而 `catalog_for_area(area)` 要的是 knowledge area）—— 值非 None，却**一消费就
    AttributeError**，于是真实 lifespan 下的每个 run 都在 research 阶段失败，而逻辑型验收
    （替身装配）永远看不到。本用例把「可被真实消费」也纳入守卫。
    """

    from types import SimpleNamespace

    from app.agent.tools import ToolContext
    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.config import settings

    settings.data_dir = tmp_path / "erag-data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    context, fake_vector_store = _wired_context_without_catalog_stub(tmp_path, monkeypatch)
    monkeypatch.setattr(api_deps, "get_workspace_vector_store", lambda ctx: fake_vector_store)
    monkeypatch.setattr(api_deps, "workspace_state_store", lambda ctx: object())
    monkeypatch.setattr(api_deps, "workspace_image_state_store", lambda ctx: object())
    # knowledge_area 的替身必须**像真的**（有 `.context` / `blocks()`），否则又会自造假红
    area = SimpleNamespace(context=context, index_fingerprint="fp", blocks=list)
    monkeypatch.setattr(api_deps, "get_workspace_knowledge_area", lambda ctx: area)

    services = rs.tool_services_for(context)  # 真实入口（catalog_for_area 未被替换）
    assert services["knowledge_area"] is area
    kwargs = agent_runtime._tool_context_kwargs(dict(services))  # 真实字段过滤
    # 真实消费：t34 的真因就是这一步抛 AttributeError（当时 `_catalog` 由 context 构造）
    ToolContext(workspace=context, **kwargs)

    # 非永真证明：把 context 当 area 传（当初的写法）⇒ 真实的 catalog_for_area 必须抛
    from app.generation.knowledge_catalog import catalog_for_area as real_catalog_for_area

    with pytest.raises(AttributeError):
        real_catalog_for_area(context)


def _wired_context_without_catalog_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any]:
    """与 `_history_wired_env` 同构，但**不**替换 `catalog_for_area`（要验的就是它）。"""

    from app.api import deps as api_deps
    from app.api import run_services as rs
    from app.core.workspace_context import WorkspaceContextFactory
    from app.core.workspace_store import WorkspaceStore

    workspace_store = WorkspaceStore(tmp_path / "conv-data")
    workspace_store.create("甲区", "ws-1")
    factory = WorkspaceContextFactory(tmp_path / "erag-data", workspace_store)

    class _FakeVectorStore:
        def count(self) -> int:
            return 0

        def count_files(self) -> int:
            return 0

    monkeypatch.setattr(api_deps, "get_embedder", lambda: object())
    monkeypatch.setattr(rs, "HybridRetriever", lambda store, **kwargs: object())
    return factory.resolve("ws-1"), _FakeVectorStore()


def test_conditionally_unconfigured_seams_are_legitimately_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置搜索/视觉/provider 时对应 seam 为 None 属**合法**，不是缺陷。"""

    from app.core.config import settings

    settings.data_dir = tmp_path / "erag-data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _unconfigured_provider(monkeypatch)
    assert agent_runtime.general_model_for_chat() is None
    assert agent_runtime.synthesis_model_for_chat() is None
    # 未配置时守卫不得报「已配置却为 None」；但**缺键**仍要报（那是装配缺陷）
    missing = unfilled_seams({}, configured=dict.fromkeys(CONDITIONAL_TOOL_SEAMS, False))
    assert set(REQUIRED_TOOL_SEAMS) <= set(missing)
    assert all("已配置却为 None" not in item for item in missing)


async def test_default_services_synthesize_uses_the_chat_model(
    app_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """检查正文生成与引用编号的连接。"""

    from app.agent.contracts import AgentKind, TaskStatus
    from app.agent.research import (
        ResearchConclusion,
        ResearchNextAction,
        ResearchProgress,
        ResearchResult,
    )
    from app.agent.runtime import RunManager
    from tests.fakes_run import StubContextFactory
    from tests.fakes_run import seed_conversation as _seed

    await _seed(app_store)
    manager = RunManager(
        store=app_store,
        context_factory=StubContextFactory(root=tmp_path),
        registry=None,
        tool_services=lambda context: {"_catalog": None, "_memory": None},
        services_factory=None,  # → 真实 default_services
    )
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="限流怎么做"
        )
        model = _fake_provider(monkeypatch, "限流用令牌桶实现{{ev-1}}。")
        services = agent_runtime.default_services(
            manager, run, {"question": "限流怎么做", "requirement_texts": {"R1": "限流怎么做"}}
        )

        evidence = _synthesis_request().evidence[0]
        result = ResearchResult(
            task_id="t-r1-knowledge",
            agent_kind=AgentKind.RAG,
            status=TaskStatus.COMPLETED,
            evidence=[evidence],
            conclusions=[
                ResearchConclusion(
                    statement="限流用令牌桶实现",
                    evidence_ids=["ev-1"],
                    requirement_ids=["R1"],
                )
            ],
            progress=ResearchProgress(
                task_id="t-r1-knowledge",
                requirement_coverage={"R1": RequirementState.ANSWERED},
                next_action=ResearchNextAction.FINALIZE,
            ),
        )
        candidate = await services.synthesize(
            {"question": "限流怎么做", "requirement_texts": {"R1": "限流怎么做"}},
            [result.model_dump(mode="json")],
            {"R1": "限流怎么做"},
        )
        assert "令牌桶" in candidate.content
        assert "[S1]" in candidate.content, "模型正文的占位符必须已被编号分配器接管"
        assert "{{ev-1}}" not in candidate.content
        assert model.calls, "真实装配路径必须调用注入的对话模型"

    finally:
        await manager.aclose()
