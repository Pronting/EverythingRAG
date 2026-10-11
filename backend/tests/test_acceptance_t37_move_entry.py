"""t37 独立复核：会话移动「两入口收敛」与「分叉状态」回归探针。

**为什么另开一个文件**（独立于实现方 `tests/test_move_entry_convergence.py`）：
本文件**不复用**实现方的 `env` 夹具，而是自建夹具（真实用户路径
``POST /api/workspaces`` + ``POST /api/conversations`` + 真实 ``RunManager`` + 内存
app.db + conftest 隔离出来的 tmp 存档目录），断言口径也自行设计。这样即使实现方
用例与实现一起错（例如两边用了同一个错误口径判「同区」），本文件仍有独立判别力。

针对 `t35` 修掉的**第二个缺陷**（第一版用**存档**判「同区」，而状态机源归属取
**app.db** ⇒ 已分叉状态下跨区拖拽被误判为同区排序、整条跳过状态机）设计探针：

1. **分叉状态 → 跨区移动**：必须真走状态机（`steps` 非空且顺序含
   `stop_requested`/`runs_stopped`/`activity_exited`/`app_db_moved`），且
   **app.db 行 == 磁盘存档 == 响应 `workspace_id`**；
2. **未确认 / epoch 不符 → 零改动**（存档、app.db、`context_epoch`、run 全原样，
   run 仍 `RUNNING`、`stop_reason is None`）；
3. **同区排序 → 不递增 epoch、不写权威行（`updated_at` 不变）、响应无 `steps`**；
   额外一条**实现方未覆盖**的角度：分叉状态下拖到「app.db 当前所属区」必须按同区排序
   处理，并把**存档拉回权威区**（分叉自愈），而不是又一次改动权威行。
4. **两入口动态等价**（同输入同结果、同 409 文案）+ 请求模型 schema 面（OpenAPI）。

零出网：只用 tmp 工作区与内存 app.db；不使用任何真实 provider。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent.contracts import (
    ApprovalDisplay,
    ApprovalRequest,
    ApprovalStatus,
    RunStatus,
)
from app.api import deps as api_deps
from app.api.routes import conversations as conversations_routes
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, build_registry

RUNNING_RUN_TIMEOUT = 5.0


@dataclass
class MoveHarness:
    client: Any
    manager: Any
    store: Any
    services: FakeServices
    first_workspace: str
    second_workspace: str
    conversation: str

    @property
    def drag_url(self) -> str:
        """拖拽入口（t35 收敛前是「只改存档」的旁路）。"""

        return f"/api/conversations/{self.conversation}/move"

    @property
    def workspace_url(self) -> str:
        """右键入口（原本就是状态机路径）。"""

        return f"/api/conversations/{self.conversation}/workspace"


def _archive_store() -> Any:
    """**从磁盘重读**的存档实例（``ConversationStore`` 有内存态，跨实例断言必须重读）。

    取 conftest 注入的那个 override（同一 tmp 目录），因此与路由看到的是同一份文件。
    """

    override = app.dependency_overrides[conversations_routes.get_conversation_store]
    return override()


def _archive_workspace(harness: MoveHarness) -> str:
    conversation = _archive_store().get(harness.conversation)
    assert conversation is not None, "存档里必须有该会话（否则检测器本身失效）"
    return str(conversation.workspace_id)


async def _diverged(harness: MoveHarness) -> bool:
    """权威分叉检测：**存档归属 != app.db 归属**（旧旁路的特征信号）。"""

    record = await harness.store.get_conversation(harness.conversation)
    return str(record.workspace_id) != _archive_workspace(harness)


@pytest.fixture
async def harness(app_store, tmp_path: Path):
    registry = build_registry()
    services = FakeServices(task_count=1)
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/workspaces", json={"name": "甲区"})
        assert first.status_code == 201, first.text
        second = await client.post("/api/workspaces", json={"name": "乙区"})
        assert second.status_code == 201, second.text
        created = await client.post(
            "/api/conversations", params={"workspace": first.json()["id"]}
        )
        assert created.status_code == 201, created.text
        yield MoveHarness(
            client=client,
            manager=manager,
            store=app_store,
            services=services,
            first_workspace=first.json()["id"],
            second_workspace=second.json()["id"],
            conversation=created.json()["id"],
        )
    app.state.run_manager = None
    api_deps.register_app_store(None)
    await manager.aclose()


async def _start_running_run(harness: MoveHarness) -> str:
    """起一个**真在跑**的 run（研究阶段闸住），用于断言「移动是否真的停掉了它」。"""

    harness.services.research_started = asyncio.Event()
    harness.services.research_release = asyncio.Event()
    created = await harness.client.post(
        f"/api/workspaces/{harness.first_workspace}/runs",
        json={"conversation_id": harness.conversation, "message": "张三最近怎么样"},
    )
    assert created.status_code == 202, created.text
    run_id = str(created.json()["run_id"])
    await asyncio.wait_for(harness.services.research_started.wait(), timeout=RUNNING_RUN_TIMEOUT)
    assert harness.manager.is_running(run_id)
    return run_id


def _force_divergence(harness: MoveHarness) -> None:
    """只改**存档**归属（复现 t35 之前的旧旁路行为），app.db 一动不动。"""

    _archive_store().move_in_sidebar(
        harness.conversation, workspace_id=harness.second_workspace
    )


# =====================================================================
# 1. 分叉状态 → 跨区移动必须真走状态机（t35 第二个缺陷的独立探针）
# =====================================================================


async def test_diverged_state_cross_area_drag_still_runs_the_state_machine(
    harness: MoveHarness,
) -> None:
    run_id = await _start_running_run(harness)
    await harness.store.request_approval(
        ApprovalRequest(
            approval_id="ap-t37",
            run_id=run_id,
            operation_id="op-t37",
            args_hash="hash-t37",
            context_epoch=1,
            display=ApprovalDisplay(purpose="联网搜索", tool_name="web_search"),
        )
    )

    _force_divergence(harness)
    assert _archive_workspace(harness) == harness.second_workspace, "探针前置：存档已被改到目标区"
    record_before = await harness.store.get_conversation(harness.conversation)
    assert record_before.workspace_id == harness.first_workspace, "探针前置：app.db 仍在原区"
    assert await _diverged(harness) is True, "探针前置：必须真的处于分叉状态"

    response = await harness.client.post(
        harness.drag_url,
        json={"workspace_id": harness.second_workspace, "confirm": True},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    steps = list(payload.get("steps") or [])
    assert steps, "分叉状态下「跨区移动」不得被误判成同区排序（steps 为空即为缺陷复现）"
    order = [
        "confirmed",
        "lock_acquired",
        "stop_requested",
        "runs_stopped",
        "activity_exited",
        "app_db_moved",
        "sidebar_moved",
        "done",
    ]
    positions = [steps.index(name) for name in order]
    assert positions == sorted(positions), f"状态机步骤顺序不符：{steps}"

    # 权威单一：app.db == 磁盘存档 == 响应
    record_after = await harness.store.get_conversation(harness.conversation)
    assert record_after.workspace_id == harness.second_workspace, record_after
    assert record_after.context_epoch == record_before.context_epoch + 1
    assert _archive_workspace(harness) == harness.second_workspace, "存档必须镜像权威行"
    assert payload["workspace_id"] == harness.second_workspace
    assert payload["from_workspace_id"] == harness.first_workspace
    assert await _diverged(harness) is False, "移动后不得再分叉"
    assert payload["stopped_run_ids"] == [run_id]
    assert payload["superseded_approvals"] == 1

    # 原 run：已停止、不可恢复、归属仍在原区（历史只读边界）
    run = await harness.store.get_run(run_id)
    assert run.status is RunStatus.CANCELLED, run
    assert run.stop_reason == "conversation_moved", run
    assert run.resumable is False
    assert run.workspace_id == harness.first_workspace
    assert (await harness.store.get_approval("ap-t37")).status is ApprovalStatus.SUPERSEDED
    harness.services.research_release.set()


# =====================================================================
# 2. 未确认 / epoch 不符 → 零改动（旧行为会 200 + 改存档）
# =====================================================================


async def test_unconfirmed_cross_area_request_changes_nothing(harness: MoveHarness) -> None:
    run_id = await _start_running_run(harness)
    before = await harness.store.get_conversation(harness.conversation)

    response = await harness.client.post(
        harness.drag_url, json={"workspace_id": harness.second_workspace}
    )
    assert response.status_code == 409, response.text
    assert "确认" in response.json()["detail"], response.text

    after = await harness.store.get_conversation(harness.conversation)
    assert after.workspace_id == before.workspace_id == harness.first_workspace
    assert after.context_epoch == before.context_epoch
    assert after.updated_at == before.updated_at, "权威行不得被写"
    assert _archive_workspace(harness) == harness.first_workspace, "存档不得被改（旧缺陷正是在此）"
    assert await _diverged(harness) is False

    run = await harness.store.get_run(run_id)
    assert run.status is RunStatus.RUNNING, run
    assert run.stop_reason is None
    assert harness.manager.is_running(run_id)
    harness.services.research_release.set()


async def test_epoch_mismatch_changes_nothing(harness: MoveHarness) -> None:
    run_id = await _start_running_run(harness)
    before = await harness.store.get_conversation(harness.conversation)

    response = await harness.client.post(
        harness.drag_url,
        json={
            "workspace_id": harness.second_workspace,
            "confirm": True,
            "expected_context_epoch": 999,
        },
    )
    assert response.status_code == 409, response.text
    assert "上下文版本" in response.json()["detail"], response.text

    after = await harness.store.get_conversation(harness.conversation)
    assert after.workspace_id == harness.first_workspace
    assert after.context_epoch == before.context_epoch
    assert _archive_workspace(harness) == harness.first_workspace
    assert await _diverged(harness) is False
    run = await harness.store.get_run(run_id)
    assert run.status is RunStatus.RUNNING
    assert run.stop_reason is None
    harness.services.research_release.set()


# =====================================================================
# 3. 同区排序：不触发状态机、不写权威行；分叉时按权威区判定并自愈
# =====================================================================


async def test_same_area_ordering_touches_nothing_authoritative(harness: MoveHarness) -> None:
    before = await harness.store.get_conversation(harness.conversation)

    response = await harness.client.post(
        harness.drag_url, json={"workspace_id": harness.first_workspace}
    )
    assert response.status_code == 200, response.text
    assert "steps" not in response.json(), "同区拖拽不是移动，不得出现状态机步骤"

    # 只带落点工作区（缺 target_index —— 该字段已于 2026-09-22 移除）也按同区处理
    reordered = await harness.client.post(harness.drag_url, json={})
    assert reordered.status_code == 200, reordered.text
    assert "steps" not in reordered.json()

    after = await harness.store.get_conversation(harness.conversation)
    assert after.workspace_id == harness.first_workspace
    assert after.context_epoch == before.context_epoch, "同区拖拽不得递增 context_epoch"
    assert after.updated_at == before.updated_at, "同区拖拽不得写权威行"
    assert _archive_workspace(harness) == harness.first_workspace
    assert await _diverged(harness) is False


async def test_diverged_state_same_area_drag_heals_the_archive(harness: MoveHarness) -> None:
    """**实现方未覆盖的角度**：分叉时拖到「app.db 当前所属区」按同区处理，并把存档拉回权威区。"""

    _force_divergence(harness)
    assert await _diverged(harness) is True
    before = await harness.store.get_conversation(harness.conversation)

    response = await harness.client.post(
        harness.drag_url, json={"workspace_id": harness.first_workspace}
    )
    assert response.status_code == 200, response.text
    assert "steps" not in response.json()

    after = await harness.store.get_conversation(harness.conversation)
    assert after.context_epoch == before.context_epoch, "同区拖拽不得递增 epoch"
    assert after.updated_at == before.updated_at, "同区拖拽不得写权威行"
    assert _archive_workspace(harness) == harness.first_workspace, "存档应被拉回权威区"
    assert await _diverged(harness) is False


# =====================================================================
# 5. 非永真证明：把判据/实现换回旧行为，上面的探针必须能判死它
# =====================================================================


async def test_probe_is_non_vacuous_when_same_area_criteria_is_archive_only(
    harness: MoveHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """反事实：把「同区」判据换回**存档口径**（t35 第一版缺陷）⇒ 分叉状态下跨区拖拽必须被跳过。

    若本用例仍绿（旧判据下也走了状态机），说明第 1 条探针的断言可能恒真 —— 本用例就是
    它的判别力证明。反过来说：本用例绿 ⇒ 第 1 条探针在旧代码上必然红。
    """

    run_id = await _start_running_run(harness)
    _force_divergence(harness)
    assert await _diverged(harness) is True

    async def archive_only(sidebar: Any, conversation_id: str) -> str | None:
        del conversation_id
        return None if sidebar is None else str(sidebar.workspace_id)

    monkeypatch.setattr(
        conversations_routes, "_authoritative_workspace_id", archive_only
    )
    response = await harness.client.post(
        harness.drag_url,
        json={"workspace_id": harness.second_workspace, "confirm": True},
    )
    assert response.status_code == 200, response.text
    assert "steps" not in response.json(), (
        "旧判据（存档）下跨区拖拽会被当成同区排序 ⇒ 整条跳过状态机；"
        "本用例必须证明这一判别力"
    )
    record = await harness.store.get_conversation(harness.conversation)
    assert record.workspace_id == harness.first_workspace, "app.db 未被写（旧缺陷复现）"
    assert await _diverged(harness) is True, "分叉仍在（旧缺陷复现）"
    run = await harness.store.get_run(run_id)
    assert run.status is RunStatus.RUNNING, "旧判据下 run 根本不会被停"
    harness.services.research_release.set()


async def test_probe_is_non_vacuous_when_cross_area_entry_is_the_legacy_bypass(
    harness: MoveHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """反事实：把跨区分支换回**旧旁路**（只改存档）⇒ 未确认请求会 200 且造成分叉。"""

    async def legacy_bypass(**kwargs: Any) -> dict[str, Any]:
        store = kwargs["store"]
        target = str(kwargs["target_workspace_id"])
        moved = store.move_in_sidebar(kwargs["conversation_id"], workspace_id=target)
        assert moved is not None
        return {"id": moved.id, "workspace_id": target}

    monkeypatch.setattr(
        conversations_routes, "_move_via_state_machine", legacy_bypass
    )
    response = await harness.client.post(
        harness.drag_url, json={"workspace_id": harness.second_workspace}  # 未确认
    )
    assert response.status_code == 200, (
        "旧旁路下「未确认」也会 200 —— 我的 409 + 零改动探针正是判它"
    )
    assert await _diverged(harness) is True, "旧旁路必然造成存档与 app.db 分叉"


# =====================================================================
# 6. 两入口动态等价（同输入同结果）+ 契约面
# =====================================================================


@pytest.mark.parametrize("entry", ["move", "workspace"])
async def test_both_entries_reject_and_succeed_identically(
    app_store, tmp_path: Path, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两个入口：未确认同为 409 且文案同源；确认后都走状态机、响应同形。"""

    del monkeypatch
    registry = build_registry()
    services = FakeServices(task_count=1)
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = (await client.post("/api/workspaces", json={"name": "甲区"})).json()["id"]
            second = (await client.post("/api/workspaces", json={"name": "乙区"})).json()["id"]
            conversation = (
                await client.post("/api/conversations", params={"workspace": first})
            ).json()["id"]
            url = f"/api/conversations/{conversation}/{entry}"
            body_base = {"workspace_id": second}

            services.research_started = asyncio.Event()
            services.research_release = asyncio.Event()
            created = await client.post(
                f"/api/workspaces/{first}/runs",
                json={"conversation_id": conversation, "message": "问题"},
            )
            assert created.status_code == 202, created.text
            run_id = str(created.json()["run_id"])
            await asyncio.wait_for(
                services.research_started.wait(), timeout=RUNNING_RUN_TIMEOUT
            )

            rejected = await client.post(url, json=body_base)
            assert rejected.status_code == 409, rejected.text
            assert rejected.json()["detail"] == (
                "移动会话会携带历史内容，请确认后再移动（confirm=true，PRD 5.4）"
            ), rejected.text
            assert (await app_store.get_conversation(conversation)).context_epoch == 1

            accepted = await client.post(url, json={**body_base, "confirm": True})
            assert accepted.status_code == 200, accepted.text
            payload = accepted.json()
            assert payload["from_workspace_id"] == first
            assert payload["workspace_id"] == second
            assert payload["context_epoch"] == 2
            assert payload["stopped_run_ids"] == [run_id]
            assert "app_db_moved" in payload["steps"]
            assert (await app_store.get_conversation(conversation)).workspace_id == second
            run = await app_store.get_run(run_id)
            assert run.status is RunStatus.CANCELLED
            assert run.stop_reason == "conversation_moved"
            services.research_release.set()
    finally:
        app.state.run_manager = None
        api_deps.register_app_store(None)
        await manager.aclose()


def test_move_request_models_accept_the_prd_13_2_fields() -> None:
    """契约面：两个入口的请求模型都必须接受 `confirm` 与 `expected_context_epoch`（PRD 13.2 末段）。"""

    schema = app.openapi()
    move_schema = schema["paths"]["/api/conversations/{conversation_id}/move"]["post"]
    workspace_schema = schema["paths"]["/api/conversations/{conversation_id}/workspace"]["post"]

    def _request_model(operation: dict[str, Any]) -> dict[str, Any]:
        reference = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        return schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]

    for operation in (move_schema, workspace_schema):
        properties = _request_model(operation)["properties"]
        assert "confirm" in properties, properties
        assert "expected_context_epoch" in properties, properties
    # 2026-09-22（需求 2）：侧边栏恒为「最近活动优先」，拖拽入口不再有落点位次
    assert "target_index" not in _request_model(move_schema)["properties"]


def test_single_state_machine_call_site_and_authoritative_read() -> None:
    """结构面：路由模块里只有一处状态机调用；「同区」判断读的是权威行（app.db 优先）。"""

    module_source = inspect.getsource(conversations_routes)
    assert module_source.count("await move_conversation(") == 1, "不得再有第二条旁路"

    helper = inspect.getsource(conversations_routes._authoritative_workspace_id)
    app_db_at = helper.index("app_store.get_conversation(")
    sidebar_at = helper.index("sidebar.workspace_id")
    assert app_db_at < sidebar_at, "权威源必须优先于存档（分叉时的判定口径）"
