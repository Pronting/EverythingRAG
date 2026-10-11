"""lifespan 装配路径复核（t34）：真实 `TestClient(app)` + 隔离后的 provider 注入。

背景：`tests/test_acceptance_t27_followups.py` 的装配用例在**未隔离**的情况下跑真实 lifespan，
general 意图以 `AttributeError` 收场。本文件按三步配方**干净复现**并给出判定：

1. 隔离 `app.core.settings_store.get_settings_store`（conftest 只覆盖 FastAPI 依赖，
   不隔离会读到本机 `~/.everything-rag/config.json` 里的真实 key ⇒ **真的出网**）；
2. monkeypatch `app.generation.providers.create_chat_model_from_config` → 假对话模型；
3. 走真实 `TestClient(app)` lifespan（`app/api/run_services.build_run_manager` → `default_services`）。

lifespan 这条入口与 t31 单测用的 `RunManager(..., services_factory=None)` **共用同一个
`default_services`**（`build_run_manager` 不传 `services_factory`），因此 `general_model` 的
接线位置相同；本文件用真实 HTTP + 真实 lifespan 证明这一点，并把任何 `AttributeError`
的真实来源（traceback）暴露出来。
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent import runtime as agent_runtime
from app.generation.base import ChatChunk
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）

GENERAL_QUESTION = "你好"
ANSWER_TEXT = "你好！我可以帮你检索本地资料，也可以做通用解释。"


class FakeChatModel:
    """确定性假对话模型（零出网）。"""

    def __init__(self, text: str = ANSWER_TEXT) -> None:
        self.text = text
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk("content", self.text)


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """① 隔离设置存储的直接读取（不隔离 ⇒ 读本机真实 key 并真的出网）。"""

    from app.core import settings_store as settings_store_mod

    empty = settings_store_mod.SettingsStore(tmp_path / "isolated-settings")
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: empty)


@pytest.fixture
def failure_tracebacks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """探针：记录研究入口与 `_fail` 的**完整 traceback**（定位 AttributeError 真因）。"""

    captured: list[str] = []
    original_fail = agent_runtime.RunManager._fail

    async def spy_fail(self: Any, run_id: str, exc: BaseException, *, node: str) -> None:
        captured.append(
            f"[_fail] node={node} type={type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        await original_fail(self, run_id, exc, node=node)

    monkeypatch.setattr(agent_runtime.RunManager, "_fail", spy_fail)

    from app.agent import research as agent_research

    original_research = agent_research.research_task

    async def spy_research(*args: Any, **kwargs: Any) -> Any:
        try:
            return await original_research(*args, **kwargs)
        except BaseException as exc:
            captured.append(
                f"[research_task] type={type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            raise

    monkeypatch.setattr(agent_research, "research_task", spy_research)

    original_tool_context = agent_runtime.tool_context_for

    async def spy_tool_context(*args: Any, **kwargs: Any) -> Any:
        try:
            return await original_tool_context(*args, **kwargs)
        except BaseException as exc:
            captured.append(
                f"[tool_context_for] type={type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            raise

    monkeypatch.setattr(agent_runtime, "tool_context_for", spy_tool_context)
    return captured


def _configured_provider(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    model = FakeChatModel()
    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config", lambda config: model
    )
    return model


def _unconfigured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.generation.providers import ChatProviderError

    def _raise(config: Any) -> Any:
        raise ChatProviderError("对话模型未配置，缺少: 对话模型 Base URL")

    monkeypatch.setattr("app.generation.providers.create_chat_model_from_config", _raise)


def _run_general_via_lifespan(client: TestClient) -> dict[str, Any]:
    """真实 HTTP：建工作区 → 建会话 → 建 run（general 意图）→ 轮询终态。"""

    workspace = client.post("/api/workspaces", json={"name": "复核区"})
    assert workspace.status_code == 201, workspace.text
    workspace_id = workspace.json()["id"]

    conversation = client.post("/api/conversations", params={"workspace": workspace_id})
    assert conversation.status_code == 201, conversation.text
    conversation_id = conversation.json()["id"]

    created = client.post(
        f"/api/workspaces/{workspace_id}/runs",
        json={"conversation_id": conversation_id, "message": GENERAL_QUESTION},
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]

    detail: dict[str, Any] = {}
    for _ in range(200):
        response = client.get(f"/api/workspaces/{workspace_id}/runs/{run_id}")
        assert response.status_code == 200, response.text
        detail = response.json()
        if detail["status"] in {"completed", "completed_with_gaps", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert detail.get("status"), "必须能取到 run 终态"
    return detail


def test_lifespan_general_intent_with_provider_completes(
    monkeypatch: pytest.MonkeyPatch, failure_tracebacks: list[str]
) -> None:
    """③ 真实 lifespan + 已配置 provider：general 意图必须端到端成功。"""

    model = _configured_provider(monkeypatch)
    with TestClient(app) as client:
        detail = _run_general_via_lifespan(client)

    assert not failure_tracebacks, (
        "装配路径不得以异常收场（t34 真因：tool_services_for 把 WorkspaceContext 当 "
        "knowledge area 传给 catalog_for_area ⇒ research 节点 AttributeError）：\n"
        + "\n".join(failure_tracebacks)
    )
    assert detail["status"] in {"completed", "completed_with_gaps"}, detail
    assert model.calls, "general 路径必须真的调用注入的对话模型"


def test_lifespan_general_intent_without_provider_is_typed_error(
    monkeypatch: pytest.MonkeyPatch, failure_tracebacks: list[str]
) -> None:
    """③ 真实 lifespan + 未配置 provider：必须是 `not_configured`，不得 AttributeError。"""

    _unconfigured_provider(monkeypatch)
    with TestClient(app) as client:
        detail = _run_general_via_lifespan(client)

    assert detail["status"] == "failed"
    assert detail["error_code"] == "not_configured", detail
    assert "AttributeError" not in json.dumps(detail)
    assert not [
        item for item in failure_tracebacks if "AttributeError" in item
    ], "装配路径不得出现 AttributeError：\n" + "\n".join(failure_tracebacks)


def test_lifespan_assembly_uses_default_services_factory() -> None:
    """结构锁：lifespan 的 `build_run_manager` 必须用真实 `default_services`（同一装配）。"""

    import inspect

    from app.api import run_services as rs

    source = inspect.getsource(rs.build_run_manager)
    assert "services_factory" not in source, (
        "build_run_manager 不得覆盖 services_factory：lifespan 与运行层必须共用 "
        "default_services，否则两条装配路径会分叉（t34 复核点）"
    )
    assert "context_factory=deps.get_workspace_context_factory()" in source
    assert "tool_services=tool_services_for" in source


# ---------------------------------------------------------------------
# 目录线索真的交到了路由层手里（真实缺陷回归：`_catalog_for` 读错内部键）
#
# 现象：用户在知识库里**确实有**的资料上提问，真实应用恒为
# `intent=general / needs_knowledge=false`（记录在案的 run 事件可复核），而把目录显式
# 传进 `IntentAgent` 离线跑同一句话却是 `needs_knowledge=true`。
# 真因：装配层写内部键 `"_catalog"`，运行层却读 `"catalog"` ⇒ 意图层恒为 None ⇒
# `IntentAgent.enrich` 直接返回，R1/R3 的目录线索（文档 / 术语命中）全线失效，
# 只剩显式内部标记或带历史的指代（R2）能走进知识路径。
# ---------------------------------------------------------------------

KNOWLEDGE_DOC_TITLE = "货拉拉大模型记忆系统（一）：从提取到召回的工程实践"
KNOWLEDGE_QUESTION = "货拉拉的记忆系统有淘汰策略吗。类似openclaw的那种方案"


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> Any:
    """把装配层的嵌入器换成 4 维假嵌入器。

    ``deps`` 在模块顶层绑定了 ``get_settings_store`` / 用了 ``_get_embedder_and_cache_key``，
    只 monkeypatch ``settings_store.get_settings_store`` 换不掉它们 —— 真实
    ``CloudEmbedder`` 一旦被要求算 dim 就会出网（PRD 11 隐私红线）。这里直接替换
    装配层的嵌入器来源，向量库 / 工具 / 检索因此共用同一个零出网实例。
    """

    from app.api import deps
    from tests.fakes import FakeEmbedder

    embedder = FakeEmbedder()
    monkeypatch.setattr(
        deps, "_get_embedder_and_cache_key", lambda: (embedder, "fake-runtime-digest")
    )
    return embedder


def _seed_shared_knowledge(workspace_id: str) -> str:
    """向**共享知识库**写一份真实块（与提问的工作区无关，D-KB）。"""

    from app.api import deps
    from app.models.schemas import BlockMetadata, SourceType

    context = deps.get_workspace_context_factory().resolve(workspace_id)
    store = deps.get_workspace_vector_store(context)
    store.upsert(
        [
            (
                "block-mem-1",
                "记忆提取：把会话消息变成可独立理解的事实；离线让记忆可维护。",
                [1.0, 0.0, 0.0, 0.0],
                BlockMetadata(
                    block_id="block-mem-1",
                    doc_id="doc-mem-1",
                    source_type=SourceType.DOCUMENT,
                    source_file=str(context.documents_dir / f"{KNOWLEDGE_DOC_TITLE}.md"),
                    heading_path=f"{KNOWLEDGE_DOC_TITLE} > 二、总体设计",
                    chunk_type="text",
                ),
            )
        ]
    )
    return str(context.chroma_dir)


def test_route_seam_hands_the_catalog_to_the_intent_agent(fake_embedder: Any) -> None:
    """装配层产出的目录必须被运行层按**同一个键**取到（`_catalog`）。"""

    with TestClient(app) as client:
        workspace = client.post("/api/workspaces", json={"name": "目录线索区"})
        assert workspace.status_code == 201, workspace.text
        workspace_id = workspace.json()["id"]

        from app.api import deps

        manager = agent_runtime.get_run_manager()
        assert manager is not None, "lifespan 必须注册 RunManager"
        context = deps.get_workspace_context_factory().resolve(workspace_id)
        extras = dict(manager.tool_services(context))  # type: ignore[arg-type]
        assert "_catalog" in extras, "装配层必须提供内部键 _catalog（约定见 run_services）"
        assert agent_runtime._catalog_for(manager, context) is extras["_catalog"]


def test_knowledge_question_routes_to_the_knowledge_path_through_the_real_seam(
    monkeypatch: pytest.MonkeyPatch,
    failure_tracebacks: list[str],
    fake_embedder: Any,
) -> None:
    """真实 lifespan：知识库里有这份资料时，这个问题必须走知识检索（端到端）。"""

    _configured_provider(monkeypatch)
    with TestClient(app) as client:
        workspace = client.post("/api/workspaces", json={"name": "线索区"})
        assert workspace.status_code == 201, workspace.text
        workspace_id = workspace.json()["id"]
        _seed_shared_knowledge(workspace_id)

        conversation = client.post("/api/conversations", params={"workspace": workspace_id})
        assert conversation.status_code == 201, conversation.text
        conversation_id = conversation.json()["id"]

        created = client.post(
            f"/api/workspaces/{workspace_id}/runs",
            json={"conversation_id": conversation_id, "message": KNOWLEDGE_QUESTION},
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]

        detail: dict[str, Any] = {}
        for _ in range(200):
            detail = client.get(f"/api/workspaces/{workspace_id}/runs/{run_id}").json()
            if detail["status"] in {"completed", "completed_with_gaps", "failed", "cancelled"}:
                break
            time.sleep(0.05)

        events = client.get(f"/api/workspaces/{workspace_id}/runs/{run_id}/event-log")
        assert events.status_code == 200, events.text
        payload = events.json()
        rows = payload.get("events", payload)
        decided = next(row for row in rows if row["type"] == "route.decided")
        decision = decided["payload"]["decision"]
        planned = next(row for row in rows if row["type"] == "plan.updated")
        kinds = {task["agent_kind"] for task in planned["payload"]["tasks"]}

    assert decision["needs_knowledge"] is True, decision
    assert decision["intent"] != "general", decision
    assert kinds == {"rag"}, kinds
    assert not failure_tracebacks, "\n".join(failure_tracebacks)
