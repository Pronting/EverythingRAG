"""T_verify 后续验收用例：A27 端点级独立验收 + t29 三层优先级 + t25 契约面 + 装配级检查点。

为什么单独一个文件：这些用例**只能在依赖任务全部 terminal 且工作树静止后首次运行**
（方法要求：跑验收前先确认「没有 in_progress/claimed 的写码任务」且「关键文件 mtime
近期无变化」），因此先以骨架落盘，门开启后直接跑。

**A27 的验证强度**：本文件的 A27 用例由 T_verify **独立编写**，与实现方
``tests/test_conversation_move.py`` 的断言方式不同（不 import 实现方的 fixture），
覆盖六项：``confirm`` / ``expected_context_epoch`` 乐观并发、停止 → 等待退出 →
才改归属（在改归属那一刻留探针）、拒绝路径、批准/澄清 ``SUPERSEDED``、目标区新 run 的
``origin_run_id`` 且不携带原区未发布结果、目标区快照重绑定。

**夹具修正（t32）**：``api`` 夹具此前自建 app.db 行（``ws-1`` / ``ws-2`` / ``conv-1``），
而移动端点同时要求**存档侧**会话行与目标区存档行 ⇒ 一律 404。现改为走**真实用户路径**
（``POST /api/workspaces`` + ``POST /api/conversations``）拿到真实 id，用例一律使用夹具
返回的 id；**断言一条未放宽**。

零出网：本文件所有用例都不发起外部请求（装配用例把 chat provider 工厂整体替换为假模型，
并把设置存储隔离到 tmp —— 不隔离会读开发者本机 ``~/.everything-rag/config.json`` 里的
真实明文 key 并**真的出网**）。所有用例只用 tmp 工作区。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.contracts import (
    ApprovalStatus,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
    main_thread_id,
)
from app.agent.runtime import RunManager
from app.api import deps as api_deps
from app.generation.base import ChatChunk
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, build_registry

#: 1x1 PNG（真实 data URI；请求体里就是「图片引用」本身，见 t25 契约）。
ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AL+2wAAAABJRU5ErkJggg=="
)

TERMINAL_RUN_STATUSES = {"completed", "completed_with_gaps", "failed", "cancelled"}

GENERAL_QUESTION = "你好"
FAKE_ANSWER = "你好！我可以帮你检索本地资料，也可以做通用解释。"


# =====================================================================
# 夹具：真实 API + 真实 RunManager + 内存 app.db（走真实用户路径建区/建会话）
# =====================================================================


@dataclass
class ApiHarness:
    """端点级验收夹具：真实 HTTP 客户端 + 真实 RunManager + 内存 app.db。"""

    client: Any
    manager: RunManager
    store: Any
    services: FakeServices
    registry: Any
    root: Path
    first_workspace: str
    second_workspace: str
    first_workspace_name: str
    second_workspace_name: str
    conversation: str

    def runs(self, workspace_id: str) -> str:
        return f"/api/workspaces/{workspace_id}/runs"

    def defaults(self, workspace_id: str) -> str:
        return f"/api/workspaces/{workspace_id}/defaults"

    def move(self, conversation_id: str) -> str:
        return f"/api/conversations/{conversation_id}/workspace"


@pytest.fixture
async def api(app_store, tmp_path: Path):
    registry = build_registry()
    services = FakeServices(task_count=1, approve_tool="web_search")
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/workspaces", json={"name": "甲区"})
        assert first.status_code == 201, first.text
        second = await client.post("/api/workspaces", json={"name": "乙区"})
        assert second.status_code == 201, second.text
        first_id = first.json()["id"]
        second_id = second.json()["id"]

        created = await client.post("/api/conversations", params={"workspace": first_id})
        assert created.status_code == 201, created.text

        yield ApiHarness(
            client=client,
            manager=manager,
            store=app_store,
            services=services,
            registry=registry,
            root=tmp_path,
            first_workspace=first_id,
            second_workspace=second_id,
            first_workspace_name=first.json()["name"],
            second_workspace_name=second.json()["name"],
            conversation=created.json()["id"],
        )
    app.state.run_manager = None
    api_deps.register_app_store(None)
    await manager.aclose()


async def _suspend_with_approval(harness: ApiHarness) -> Any:
    """建 run 并停在待批准状态（含一张 PENDING 卡片）。"""

    run = await harness.manager.create_run(
        workspace_id=harness.first_workspace,
        conversation_id=harness.conversation,
        question="张三最近怎么样",
    )
    await harness.manager.run_to_completion(run.id)
    assert (await harness.manager.require_run(run.id)).status is RunStatus.WAITING_APPROVAL
    return run


# =====================================================================
# A27 端点级独立验收（PRD 5.4 / R15，A27+A28）
# =====================================================================


async def test_a27_endpoint_ordering_probe_and_rejections(api: ApiHarness) -> None:
    """**改归属那一刻**原 run 必须已终止；拒绝路径必须显式拒绝（端点级，独立于实现方用例）。"""

    run = await _suspend_with_approval(api)

    # 在「改归属那一刻」留探针（服务应已：停止 → 等退出 → 才改归属）
    observed: dict[str, Any] = {}
    original_move = type(api.store).move_conversation

    async def spy(self: Any, conversation_id: str, **kwargs: Any) -> Any:
        current = await api.manager.require_run(run.id)
        observed["status"] = current.status
        observed["resumable"] = getattr(current, "resumable", None)
        # 注意入参是**会话 id**（AppStore.list_active_runs(conversation_id)）；
        # 传工作区 id 会让这个断言恒真，属于伪断言，这里按真实签名传会话 id。
        observed["active_runs"] = await api.store.list_active_runs(api.conversation)
        return await original_move(self, conversation_id, **kwargs)

    type(api.store).move_conversation = spy  # type: ignore[method-assign]
    try:
        response = await api.client.post(
            api.move(api.conversation),
            json={"workspace_id": api.second_workspace, "confirm": True},
        )
    finally:
        type(api.store).move_conversation = original_move  # type: ignore[method-assign]

    assert response.status_code in {200, 201}, response.text
    assert observed, "探针未被触发：服务没有走 move_conversation"
    assert observed["status"] is RunStatus.CANCELLED, observed
    assert observed["resumable"] is False, observed
    assert observed["active_runs"] == [], observed

    # 拒绝路径：resume / 批准 / 澄清 对被移动终止的 run 必须被拒（409）
    resumed = await api.client.post(f"{api.runs(api.first_workspace)}/{run.id}/resume")
    assert resumed.status_code == 409, resumed.text
    cards = await api.store.list_approvals(run.id)
    if cards:
        rejected = await api.client.post(
            f"/api/workspaces/{api.first_workspace}/approvals/{cards[0].approval_id}",
            json={"decision": "allow_once"},
        )
        assert rejected.status_code == 409, rejected.text
    inputs = await api.store.list_input_requests(run.id)
    if inputs:
        rejected = await api.client.post(
            f"/api/workspaces/{api.first_workspace}/inputs/{inputs[0].input_request_id}",
            json={"answer": "随便"},
        )
        assert rejected.status_code == 409, rejected.text


async def test_a27_endpoint_supersede_origin_run_and_snapshot_rebind(api: ApiHarness) -> None:
    """批准转 SUPERSEDED（含原因）；目标区新 run 带 origin_run_id 且不携带旧结果。"""

    run = await _suspend_with_approval(api)
    moved = await api.client.post(
        api.move(api.conversation),
        json={"workspace_id": api.second_workspace, "confirm": True},
    )
    assert moved.status_code in {200, 201}, moved.text

    cards = await api.store.list_approvals(run.id)
    assert cards and all(card.status is ApprovalStatus.SUPERSEDED for card in cards), cards
    assert all("conversation_moved" in str(getattr(card, "superseded_reason", "")) for card in cards)

    conversation = await api.store.get_conversation(api.conversation)
    assert conversation.workspace_id == api.second_workspace
    assert conversation.moved_from_workspace_id == api.first_workspace
    snapshot_text = getattr(conversation, "memory_snapshot_text", "") or ""
    assert snapshot_text != "" or api.second_workspace_name in snapshot_text, (
        "目标区快照必须重绑定（PRD 5.4：读取目标工作区的记忆快照）"
    )
    # 端点证据：服务自己报告快照已重绑定（比字符串匹配更硬的判据）
    payload = moved.json()
    assert payload["from_workspace_id"] == api.first_workspace, payload
    assert payload["workspace_id"] == api.second_workspace, payload
    assert payload["memory_snapshot_rebound"] is True, payload
    assert "snapshot_rebound" in payload["steps"], payload

    created = await api.client.post(
        api.runs(api.second_workspace),
        json={
            "conversation_id": api.conversation,
            "message": "移动后的问题",
            # PRD 5.4「目标工作区继续当前问题时创建新的 run…保存 origin_run_id」：
            # `origin_run_id` 是**请求字段**（PRD 13.2 的请求清单未列它，服务不会自行推断，
            # 见 t32 验收报告「未达标/风险」一节：前端目前不发送该字段）。
            "origin_run_id": run.id,
        },
    )
    assert created.status_code in {200, 201, 202}, created.text
    body = created.json()
    # `origin_run_id` 是 **runs 表的列**（PRD §13.1 表定义），POST 的返回契约只有
    # 「202 与 run ID」——因此这里核**持久化事实**（端点写出来的行），而不是
    # 要求 DTO 多一个字段。
    persisted = await api.store.get_run(body["run_id"])
    assert persisted is not None, body
    assert persisted.origin_run_id == run.id, persisted.model_dump()
    assert persisted.workspace_id == api.second_workspace, persisted.model_dump()
    detail = (
        await api.client.get(f"{api.runs(api.second_workspace)}/{body['run_id']}")
    ).json()
    assert all(
        str(task.get("result_ref") or "") in {"", "None"} for task in detail.get("tasks", [])
    ), "目标区新 run 不得携带原区未发布的研究结果"


async def test_a27_endpoint_confirm_required_and_stale_context_epoch(api: ApiHarness) -> None:
    """``confirm`` 缺失 / ``expected_context_epoch`` 过期都必须是显式拒绝（409）。"""

    await _suspend_with_approval(api)

    missing_confirm = await api.client.post(
        api.move(api.conversation), json={"workspace_id": api.second_workspace}
    )
    assert missing_confirm.status_code == 409, missing_confirm.text

    stale = await api.client.post(
        api.move(api.conversation),
        json={
            "workspace_id": api.second_workspace,
            "confirm": True,
            "expected_context_epoch": 999,
        },
    )
    assert stale.status_code == 409, stale.text

    # 被拒绝的两次请求都不得真的移动（拒绝 ≠ 静默部分生效）
    conversation = await api.store.get_conversation(api.conversation)
    assert conversation.workspace_id == api.first_workspace, conversation


# =====================================================================
# t29 三层优先级（PRD 1.3 / 14.1）
# =====================================================================


async def test_t29_three_tier_priority_request_then_conversation_then_workspace(
    api: ApiHarness,
) -> None:
    """``请求显式 → 会话保存 → 工作区默认 → PRD 18.1 兜底``；改默认值不回溯既有会话。"""

    api.services.approve_tool = None  # 本用例要连开多个 run，不需要批准门

    # ① 工作区默认值生效边界：**改默认值之后的**新会话与新 run 采用它（PRD 14.1）
    updated = await api.client.put(
        api.defaults(api.first_workspace),
        json={"permission_mode": "full", "reasoning_level": "max"},
    )
    assert updated.status_code == 200, updated.text
    # 权威事实在 app.db（工作区响应投影只含身份/排序字段，不含默认值）
    stored_defaults = await api.store.get_workspace(api.first_workspace)
    assert stored_defaults is not None
    assert stored_defaults.default_permission_mode is PermissionMode.FULL, stored_defaults
    assert stored_defaults.default_reasoning_level is ReasoningLevel.MAX, stored_defaults

    fresh = await api.client.post(
        "/api/conversations", params={"workspace": api.first_workspace}
    )
    assert fresh.status_code == 201, fresh.text
    fresh_id = fresh.json()["id"]

    created = await api.client.post(
        api.runs(api.first_workspace),
        json={"conversation_id": fresh_id, "message": "默认值问题"},
    )
    assert created.status_code == 202, created.text
    body = created.json()
    assert body["permission_mode"] == "full", body
    assert body["reasoning_level"] == "max", body
    await api.manager.wait(body["run_id"])

    # ② 会话保存值优先于工作区默认值（PRD 1.3）
    await api.store.update_conversation(
        fresh_id, permission_mode=PermissionMode.ASK, reasoning_level=ReasoningLevel.LOW
    )
    second = await api.client.post(
        api.runs(api.first_workspace),
        json={"conversation_id": fresh_id, "message": "会话值问题"},
    )
    assert second.status_code == 202, second.text
    second_body = second.json()
    assert second_body["permission_mode"] == "ask", second_body
    assert second_body["reasoning_level"] == "low", second_body
    await api.manager.wait(second_body["run_id"])

    # ③ 请求显式值优先于会话保存值，且两轴**独立**解析（只给一轴时另一轴仍按会话值）
    third = await api.client.post(
        api.runs(api.first_workspace),
        json={"conversation_id": fresh_id, "message": "请求显式值", "permission_mode": "auto"},
    )
    assert third.status_code == 202, third.text
    third_body = third.json()
    assert third_body["permission_mode"] == "auto", third_body
    assert third_body["reasoning_level"] == "low", third_body
    await api.manager.wait(third_body["run_id"])

    # ④ 改工作区默认值**不得回溯改写**既有会话的已保存值
    rebound = await api.client.put(
        api.defaults(api.first_workspace), json={"permission_mode": "auto"}
    )
    assert rebound.status_code == 200, rebound.text
    stored = await api.store.get_conversation(fresh_id)
    assert stored.permission_mode is PermissionMode.ASK, stored
    assert stored.reasoning_level is ReasoningLevel.LOW, stored


async def test_t25_workspace_defaults_endpoint_contract(api: ApiHarness) -> None:
    """``PUT /api/workspaces/{id}/defaults``：部分更新不重置另一项；400/422/404 各自正确。

    响应投影（``_workspace_payload``）只含身份/排序字段，因此「部分更新」的判据是
    **权威行 + 可观察行为**两条：app.db 行同时保留两项；随后新建的会话/run 同时拿到
    两项（若第二次 PUT 把第一项重置回兜底，新 run 的 ``permission_mode`` 就不是 auto）。
    """

    client = api.client
    api.services.approve_tool = None  # 本用例要开一次 run 观察默认值是否真的生效

    first = await client.put(api.defaults(api.first_workspace), json={"permission_mode": "auto"})
    assert first.status_code == 200, first.text
    second = await client.put(api.defaults(api.first_workspace), json={"reasoning_level": "off"})
    assert second.status_code == 200, second.text

    stored = await api.store.get_workspace(api.first_workspace)
    assert stored is not None
    assert stored.default_permission_mode is PermissionMode.AUTO, stored
    assert stored.default_reasoning_level is ReasoningLevel.OFF, stored

    conversation = await client.post(
        "/api/conversations", params={"workspace": api.first_workspace}
    )
    assert conversation.status_code == 201, conversation.text
    run = await client.post(
        api.runs(api.first_workspace),
        json={"conversation_id": conversation.json()["id"], "message": "默认值生效验证"},
    )
    assert run.status_code == 202, run.text
    assert run.json()["permission_mode"] == "auto", run.text
    assert run.json()["reasoning_level"] == "off", run.text
    await api.manager.wait(run.json()["run_id"])

    assert (await client.put(api.defaults(api.first_workspace), json={})).status_code == 400
    assert (
        await client.put(api.defaults(api.first_workspace), json={"permission_mode": "bogus"})
    ).status_code == 422
    assert (
        await client.put(
            "/api/workspaces/ws-nope/defaults", json={"permission_mode": "auto"}
        )
    ).status_code == 404


# =====================================================================
# t25 `images`：有条件生效（cached ≠ 生效）
# =====================================================================


async def test_t25_images_conditional_effectiveness(api: ApiHarness) -> None:
    """`images` 最多 4 张（5 张 → 422）；**未配置视觉时 `cached` 不等于生效**。"""

    ok = await api.client.post(
        api.runs(api.first_workspace),
        json={
            "conversation_id": api.conversation,
            "message": "带图问题",
            "images": [ONE_PIXEL_PNG],
        },
    )
    assert ok.status_code in {200, 201, 202}, ok.text
    refs = ok.json().get("image_refs") or []
    assert refs, "请求确实带了图 → 必须逐条回结果，不得静默忽略"
    statuses = {ref.get("status") for ref in refs}
    assert "described" not in statuses or statuses == {"described"}, statuses
    # 未配置视觉时必须如实标注未生效，而不是静默当作已生效
    assert all(ref.get("reason") for ref in refs if ref.get("status") != "described"), refs
    await api.manager.wait(ok.json()["run_id"])

    too_many = await api.client.post(
        api.runs(api.first_workspace),
        json={
            "conversation_id": api.conversation,
            "message": "五张图",
            "images": [ONE_PIXEL_PNG] * 5,
        },
    )
    assert too_many.status_code == 422


# =====================================================================
# 装配级验收（t31/t33/t34）：**真实 lifespan 装配**下的 general 意图 + 检查点
# =====================================================================
#
# 归因（必须准确）：本文件与 `test_acceptance_v120.py` 的既有用例大多**由测试自己装配
# 服务**（传入假 general 模型 / 假 Services），因此「真实装配里少传一个参数」这类
# **装配型缺陷**（seam 声明了但没填）永远不会被撞到。`runtime.py::default_services` 的
# `general_model` 未接线正是这一类：由 **`t29` 在真实应用探针中发现**（`POST /runs`「你好」
# → `error_code=AttributeError, node=research`），**不是验收套件发现的** —— 验收套件当时的
# 装配方式掩盖了该缺口。下列用例把「线接没接」纳入验收面。
#
# 隔离三步（**必须**，否则测试会读开发者本机 config.json 的真实明文 key 并真的出网）：
#   ① 隔离 `app.core.settings_store.get_settings_store`（conftest 只覆盖 FastAPI 依赖）；
#   ② 替换 `app.generation.providers.create_chat_model_from_config`（chat 模型的唯一注入点）；
#   ③ 走真实 `TestClient(app)` lifespan（`app/api/run_services.build_run_manager` →
#      `default_services`），不自建 Services。


class FakeChatModel:
    """确定性假对话模型（零出网）。"""

    def __init__(self, text: str = FAKE_ANSWER) -> None:
        self.text = text
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        self.calls.append([dict(message) for message in messages])
        yield ChatChunk("content", self.text)


@pytest.fixture(autouse=True)
def _isolate_settings_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """① 隔离设置存储的直接读取（不隔离 ⇒ 读本机真实 key 并真的出网）。"""

    from app.core import settings_store as settings_store_mod

    empty = settings_store_mod.SettingsStore(tmp_path / "isolated-settings")
    monkeypatch.setattr(settings_store_mod, "get_settings_store", lambda: empty)


def _wire_chat_provider(monkeypatch: pytest.MonkeyPatch, *, configured: bool) -> FakeChatModel | None:
    """② 替换 chat 模型工厂：``configured=False`` 模拟「未配置」（工厂抛带类型错误）。"""

    model: FakeChatModel | None = None
    if configured:
        model = FakeChatModel()
        monkeypatch.setattr(
            "app.generation.providers.create_chat_model_from_config", lambda config: model
        )
        return model

    from app.generation.providers import ChatProviderError

    def _raise(config: Any) -> Any:
        del config
        raise ChatProviderError("对话模型未配置，缺少: 对话模型 Base URL")

    monkeypatch.setattr("app.generation.providers.create_chat_model_from_config", _raise)
    return None


def _run_general_via_real_lifespan(
    monkeypatch: pytest.MonkeyPatch, *, configured: bool
) -> tuple[str, str, dict[str, Any], FakeChatModel | None]:
    """③ 真实装配：lifespan → 建区 → 建会话 → 建 general run → 轮询终态。"""

    model = _wire_chat_provider(monkeypatch, configured=configured)
    workspace_id = ""
    run_id = ""
    detail: dict[str, Any] = {}
    try:
        with TestClient(app) as client:
            workspace = client.post("/api/workspaces", json={"name": "装配复核区"})
            assert workspace.status_code == 201, workspace.text
            workspace_id = workspace.json()["id"]

            conversation = client.post("/api/conversations", params={"workspace": workspace_id})
            assert conversation.status_code == 201, conversation.text

            created = client.post(
                f"/api/workspaces/{workspace_id}/runs",
                json={
                    "conversation_id": conversation.json()["id"],
                    "message": GENERAL_QUESTION,
                },
            )
            assert created.status_code == 202, created.text
            run_id = created.json()["run_id"]

            for _ in range(400):
                response = client.get(f"/api/workspaces/{workspace_id}/runs/{run_id}")
                assert response.status_code == 200, response.text
                detail = response.json()
                if detail["status"] in TERMINAL_RUN_STATUSES:
                    break
                time.sleep(0.05)
    finally:
        app.state.run_manager = None
    assert detail, "必须能取到 run 详情"
    return workspace_id, run_id, detail, model


def test_assembly_general_intent_uses_real_default_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**真实装配**路径：未配置 provider 时 general 意图必须是带类型的 ``not_configured``。"""

    _workspace, _run, detail, _model = _run_general_via_real_lifespan(
        monkeypatch, configured=False
    )

    assert detail["status"] == "failed", detail
    error_code = str(detail.get("error_code") or "")
    assert error_code == "not_configured", (
        "未配 provider 必须是带类型的 not_configured（PRD 12.5），"
        f"实际 error_code={error_code!r}"
    )
    assert "AttributeError" not in json.dumps(detail), detail
    assert detail.get("resumable") is True, "配好 provider 后应当可以继续"


def test_assembly_general_intent_succeeds_when_provider_is_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已配 provider 时，general 意图必须端到端成功（同一真实装配路径）。"""

    _workspace, _run, detail, model = _run_general_via_real_lifespan(
        monkeypatch, configured=True
    )

    assert detail["status"] in {"completed", "completed_with_gaps"}, detail
    assert model is not None and model.calls, "general 路径必须真的调用注入的对话模型"


def test_assembly_run_writes_main_and_per_task_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实装配下 PRD 12.2 / 12.4 的检查点必须真的建立：主图 + **每任务** thread。

    背景（t33 修掉的真实缺陷）：``_checkpointer_for`` 曾用同步 ``enter_context`` 包装
    异步 saver ⇒ ``TypeError`` 只记 warning ⇒ 检查点**从未建立**。这里不查验自建
    saver，只查验真实 lifespan 产物：``<workspace>/checkpoints.db`` 的实际行。
    """

    workspace_id, run_id, detail, _model = _run_general_via_real_lifespan(
        monkeypatch, configured=True
    )
    assert detail["status"] in {"completed", "completed_with_gaps"}, detail

    context = api_deps.get_workspace_context_factory().resolve(workspace_id)
    path = Path(context.checkpoints_db)
    assert path.is_file(), f"检查点库必须真的建立（PRD 12.2 / 12.4）：{path}"

    threads = _checkpoint_threads(path)
    assert main_thread_id(run_id) in threads, f"主图 thread 缺失：{threads}"
    task_threads = {thread for thread in threads if thread.startswith(f"run:{run_id}:task:")}
    assert task_threads, f"每任务 thread 必须存在（PRD 12.2）：{threads}"


def _checkpoint_threads(path: Path) -> set[str]:
    """读 ``checkpoints.db`` 的 thread 集合（stdlib sqlite3，只读）。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows}
