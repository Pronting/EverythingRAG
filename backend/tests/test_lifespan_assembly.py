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
