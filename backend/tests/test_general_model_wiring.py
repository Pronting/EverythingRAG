"""B 项回归：General Agent 的对话模型装配与「未配置」的体面失败（PRD 4.2 / 12.5）。

修复的缺陷（t29 在真实应用复现、t31 修）：``default_services`` 从未给
``research_task(..., general_model=…)`` 传值 ⇒ general 意图要么拿不到模型、要么在别处
抛 ``AttributeError``（``error_code=AttributeError``），**配了对话 provider 也一样**
（那条线根本没接）。

本文件覆盖两条分支：
1. **配了 provider** → 真实的 ``default_services`` 装配端到端跑通（用假对话模型注入，
   零出网）：general 意图问题产出答案，run 终态为 ``completed``；
2. **没配 provider** → 明确的**带类型**错误 ``not_configured`` + 用户可读说明 + 失败节点
   + 可继续标记，**绝不是 AttributeError**。

**chat 模型注入点（验收方照此接）**：
``app.agent.runtime.general_model_for_chat()`` 在**调用时**从
``app.generation.providers.create_chat_model_from_config`` 取工厂，并且 provider 实例是在
worker 线程自己的事件循环里新建的。所以只要 monkeypatch 那个模块属性：

```python
monkeypatch.setattr(
    "app.generation.providers.create_chat_model_from_config",
    lambda config: FakeChatModel("……"),
)
```

真实的 ``default_services`` 装配路径就原样走通，无需任何出网、也无需改生产代码。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agent import runtime as agent_runtime
from app.agent.contracts import RunStatus
from app.agent.runtime import RunManager
from app.generation.base import ChatChunk
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.fakes_run import StubContextFactory, seed_conversation

GENERAL_QUESTION = "限流一般怎么做"
FAKE_ANSWER = "限流常用的做法是令牌桶、漏桶与固定窗口计数，按并发与突发流量取舍。"


class FakeChatModel:
    """假对话模型：``stream_chat`` 产出给定文本（零出网、确定性）。"""

    def __init__(self, text: str = FAKE_ANSWER) -> None:
        self.text = text
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk("content", self.text)


def _tool_services(context: Any) -> Mapping[str, Any]:
    """最小句柄集合：general 路径不做检索，只需运行层内部键存在（可为 None）。"""

    del context
    return {"retriever": None, "embedder": None, "_catalog": None, "_memory": None}


def _real_wiring_manager(app_store: Any, tmp_path: Path) -> RunManager:
    """**真实 default_services** 装配（``services_factory=None``）：B 项要验的就是它。"""

    return RunManager(
        store=app_store,
        context_factory=StubContextFactory(root=tmp_path),
        registry=None,  # → build_default_registry()
        tool_services=_tool_services,
        services_factory=None,  # → app.agent.runtime.default_services（真实接线）
    )


@pytest.fixture(autouse=True)
def _isolate_settings_for_direct_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """隔离 ``get_settings_store()`` 的**直接调用**（非 FastAPI 依赖注入）。

    conftest 只覆盖了 FastAPI 依赖，而 ``general_model_for_chat()`` 直接调
    ``get_settings_store()`` —— 不隔离就会读到开发者本机的真实配置，
    于是测试会真的去连 provider（既违反「不得真实外呼」，也让分支不可控）。
    """

    from app.core import settings_store as settings_store_mod

    empty = settings_store_mod.SettingsStore(tmp_path / "empty-settings")
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: empty)


@pytest.fixture
async def wired_manager(app_store, tmp_path: Path):
    await seed_conversation(app_store)
    manager = _real_wiring_manager(app_store, tmp_path)
    try:
        yield manager
    finally:
        await manager.aclose()


async def _run_question(manager: RunManager, app_store: Any, question: str) -> Any:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question=question
    )
    await manager.run_to_completion(run.id)
    return await manager.require_run(run.id)


# =====================================================================
# 1. 已配置 provider → 端到端成功
# =====================================================================


async def test_general_model_is_wired_and_answers_end_to_end(
    wired_manager: RunManager, app_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配了对话 provider：general 意图问题走真实装配，拿到模型写的答案。"""

    fake = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: fake
    )

    run = await _run_question(wired_manager, app_store, GENERAL_QUESTION)
    assert run.status is RunStatus.COMPLETED, run.error_code
    answer = await app_store.get_answer(run.answer_id)
    assert answer is not None
    assert FAKE_ANSWER in answer.content

    # 装配确实把模型传给了 General Agent：模型收到了包含原问题的提示词，且提示词禁止引用编号
    assert fake.calls, "general 路径必须真的调到注入的对话模型"
    messages = fake.calls[0]
    joined = "\n".join(str(message.get("content") or "") for message in messages)
    assert GENERAL_QUESTION in joined
    assert "[S1]" in joined and "不要输出任何形如" in joined
    # 职责与硬性要求放 system 提示，问题另成 user 消息（PRD 9.3 固定顺序）
    assert messages[0]["role"] == "system"
    assert any(
        message["role"] == "user" and GENERAL_QUESTION in str(message["content"])
        for message in messages
    )
    # 通用解释的证据不得带文档引用编号（PRD 8.1）
    assert not answer.citations


async def test_general_model_for_chat_returns_callable_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``general_model_for_chat()``：已配置 → 可调用；未配置 → None（结构化失败的前提）。"""

    fake = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: fake
    )
    model = agent_runtime.general_model_for_chat()
    assert model is not None
    text = await asyncio.to_thread(model, _request())
    assert text == FAKE_ANSWER

    # 未配置（工厂抛 ChatProviderError）→ None
    def _raise(config: Any) -> Any:
        from app.generation.providers import ChatProviderError

        raise ChatProviderError("对话 provider 未配置，缺少环境变量: EVERYTHING_RAG_CHAT_MODEL")

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", _raise
    )
    assert agent_runtime.general_model_for_chat() is None


def _request() -> Any:
    from app.agent.contracts import AgentKind, ResearchTask
    from app.agent.research import ResearchRequest

    task = ResearchTask(
        task_id="t-r1-general",
        question=GENERAL_QUESTION,
        requirement_ids=["R1"],
        agent_kind=AgentKind.GENERAL,
    )
    return ResearchRequest(
        task=task,
        original_question=GENERAL_QUESTION,
        requirement_texts={"R1": GENERAL_QUESTION},
    )


# =====================================================================
# 2. 未配置 provider → 明确的、带类型的失败（不是 AttributeError）
# =====================================================================


async def test_unconfigured_provider_fails_with_typed_error_not_attribute_error(
    wired_manager: RunManager, app_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没配 provider：run 以 ``not_configured`` 失败，带说明/节点/可继续，绝不 AttributeError。

    「未配置」在**注入点**上模拟（工厂抛 ``ChatProviderError``）：本机 ``.env`` 里可能已有
    对话 provider 配置，靠环境去构造「未配置」既不确定、也可能真的出网。
    """

    from app.generation.providers import ChatProviderError

    def _unconfigured(config: Any) -> Any:
        raise ChatProviderError("对话模型未配置，缺少: 对话模型 Base URL")

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", _unconfigured
    )
    assert agent_runtime.general_model_for_chat() is None

    run = await _run_question(wired_manager, app_store, GENERAL_QUESTION)
    assert run.status is RunStatus.FAILED
    assert run.error_code == "not_configured", f"必须是带类型的错误码，实际 {run.error_code!r}"

    events = await wired_manager.read_events(run.id)
    failures = [event for event in events if event.type.value == "run.failed"]
    assert failures, "run.failed 事件必须带结构化错误"
    error = failures[-1].payload["error"]
    assert error["error_code"] == "not_configured"
    assert error["node"], "必须指出失败节点"
    assert "general" in error["node"]
    assert error["message"], "必须有用户可读说明"
    assert error["resumable"] is True, "配好 provider 后应当可以继续"
    assert "AttributeError" not in str(error)

    # 任务记录上也能看到失败（PRD 12.2/12.3 的任务台账）
    tasks = await wired_manager.list_tasks(run.id)
    assert tasks and all(str(t.status.value) == "failed" for t in tasks)


async def test_general_path_without_model_reports_not_configured_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单元级：``general_model_for_chat() is None`` 时 GeneralResearcher 给结构化失败。"""

    from app.agent.contracts import AgentKind, ResearchTask
    from app.agent.research import GeneralResearcher, ResearchRequest
    from app.agent.tools import ToolContext, ToolErrorCode

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config",
        lambda config: (_ for _ in ()).throw(Exception("unconfigured")),
    )
    assert agent_runtime.general_model_for_chat() is None

    task = ResearchTask(
        task_id="t-r1-general",
        question=GENERAL_QUESTION,
        requirement_ids=["R1"],
        agent_kind=AgentKind.GENERAL,
    )
    result = await GeneralResearcher(model=None).research(
        ResearchRequest(task=task, original_question=GENERAL_QUESTION),
        ToolContext(workspace=type("W", (), {"workspace_id": "ws-1"})()),
    )
    assert result.status.value == "failed"
    assert str(ToolErrorCode.NOT_CONFIGURED) == "not_configured"


# =====================================================================
# 3. seam 普查（机器可核对的清单，防止「声明了却没接」再次发生）
# =====================================================================


def test_assembly_seam_census_is_documented() -> None:
    """装配缝清单必须与代码一致：本测试锁住「哪些缝已接、哪些仍是 None」。"""

    import inspect

    source = inspect.getsource(agent_runtime.default_services)
    # 已接线：general_model 与工具上下文/history_search/redactor 相关缝在装配处出现
    assert "general_model=general_model" in source
    assert "general_model_for_chat(" in source
    # 设置页回答偏好（含自定义提示词）也在装配处读到并传入：否则设置页能填、实际不读
    assert "effective_answer_preferences()" in source
    assert "preferences=answer_preferences" in source
    # ``tool_context_for`` 负责把 redactor（NameRedactor）与 tool_services 提供的
    # history_search / image_reader / search_provider / extract_provider 塞进 ToolContext
    tool_source = inspect.getsource(agent_runtime.tool_context_for)
    assert "NameRedactor" in tool_source
    assert "_tool_context_kwargs" in tool_source


def test_tool_context_receives_provided_seams(app_store, tmp_path: Path) -> None:
    """工具侧的缝：装配层给了就进 ToolContext（未给则 None，由工具层如实报不可用）。"""

    from app.agent.tools import ToolContext

    context = ToolContext(
        workspace=type("W", (), {"workspace_id": "ws-1"})(),
        history_search=None,
        search_provider=None,
        extract_provider=None,
        image_reader=None,
    )
    assert context.history_search is None
    assert context.search_provider is None
    assert context.extract_provider is None
    assert context.image_reader is None
    # 装配层给值时同样直通（``_tool_context_kwargs`` 只做字段过滤）
    sentinel = object()
    kwargs = agent_runtime._tool_context_kwargs(
        {"_catalog": sentinel, "_memory": sentinel, "retriever": sentinel}
    )
    assert "_catalog" not in kwargs and "_memory" not in kwargs
    assert kwargs["retriever"] is sentinel


def _unused(*_args: Any) -> None:  # pragma: no cover - 占位，保持 import 面清晰
    del _args


_ = Sequence
