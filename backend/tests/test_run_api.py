"""run API 端点测试（PRD 13.2，验收 A13–A16 / A23）。

覆盖：建 run（201 + 幂等 + 409 并发 + 422 超长）、详情、跨区 404、列举、SSE 补读与
关闭、事件快照、停止/继续端点、``/api/chat`` 410。

只用内存 app.db + 假服务 + tmp 上下文；**不发任何网络请求**。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.contracts import RunStatus
from app.api import deps as api_deps
from app.main import app
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）
from tests.fakes_run import FakeServices, build_manager, seed_conversation


@pytest.fixture
async def manager(app_store, tmp_path: Path) -> AsyncIterator[object]:
    """装配 RunManager 并注册到 app.state / deps（端点与工具层都能看到它）。"""

    await seed_conversation(app_store)
    services = FakeServices()
    mgr = build_manager(app_store, services=services, root=tmp_path)
    api_deps.register_app_store(app_store)
    app.state.run_manager = mgr
    try:
        yield mgr
    finally:
        app.state.run_manager = None
        api_deps.register_app_store(None)
        await mgr.aclose()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as active:
        yield active


RUNS = "/api/workspaces/ws-1/runs"


async def _create(client: httpx.AsyncClient, **overrides) -> dict:
    body = {"conversation_id": "conv-1", "message": "测试问题", **overrides}
    response = await client.post(RUNS, json=body)
    assert response.status_code == 202, response.text  # PRD 13.2：202 + run ID
    return response.json()


# =====================================================================
# 建 run
# =====================================================================


async def test_create_run_persists_message_and_task(manager, client) -> None:
    body = await _create(client)
    assert body["status"] == RunStatus.QUEUED.value
    assert body["workspace_id"] == "ws-1"
    assert body["conversation_id"] == "conv-1"
    assert body["events_url"].endswith(f"/runs/{body['run_id']}/events")
    # 用户消息与 run 同一次请求落库（PRD 13.2）
    messages = await manager.store.list_messages("conv-1")
    assert [item.role for item in messages] == ["user"]
    assert messages[0].content == "测试问题"
    run = await manager.store.get_run(body["run_id"])
    assert run.input_message_id == messages[0].id
    await manager.wait(body["run_id"])


async def test_create_run_defaults_to_ask_permission(manager, client) -> None:
    """默认权限模式为 ask、默认思考强度为 off（PRD 10.1 / 18.1）。"""

    body = await _create(client)
    assert body["permission_mode"] == "ask"
    assert body["reasoning_level"] == "off"
    await manager.wait(body["run_id"])


async def test_client_request_id_is_idempotent(manager, client) -> None:
    first = await _create(client, client_request_id="req-1")
    second = await _create(client, client_request_id="req-1")
    assert first["run_id"] == second["run_id"]
    await manager.wait(first["run_id"])


async def test_concurrent_run_in_same_conversation_conflicts(manager, client) -> None:
    first = await _create(client)
    response = await client.post(RUNS, json={"conversation_id": "conv-1", "message": "再来一次"})
    assert response.status_code == 409
    assert "已有活动任务" in response.json()["detail"]
    await manager.wait(first["run_id"])


async def test_too_long_input_is_422(manager, client) -> None:
    response = await client.post(
        RUNS, json={"conversation_id": "conv-1", "message": "长" * 32001}
    )
    assert response.status_code == 422


async def test_unknown_workspace_is_404(manager, client) -> None:
    response = await client.post(
        "/api/workspaces/ws-missing/runs",
        json={"conversation_id": "conv-1", "message": "x"},
    )
    assert response.status_code == 404


async def test_missing_run_manager_is_503(client) -> None:
    """RunManager 尚未就绪时明确 503（而不是 500 或静默失败）。"""

    app.state.run_manager = None
    response = await client.post(
        "/api/workspaces/ws-1/runs", json={"conversation_id": "conv-1", "message": "x"}
    )
    assert response.status_code == 503


# =====================================================================
# 详情与列举
# =====================================================================


async def test_run_detail_lists_tasks_and_pending_cards(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    detail = (await client.get(f"{RUNS}/{created['run_id']}")).json()
    assert detail["status"] == RunStatus.COMPLETED.value
    assert detail["answer_id"]
    assert {task["task_id"] for task in detail["tasks"]} == {"t1", "t2"}
    assert detail["approvals"] == []
    assert detail["last_event_seq"] > 0


async def test_cross_workspace_run_is_404(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    response = await client.get(f"/api/workspaces/ws-other/runs/{created['run_id']}")
    assert response.status_code == 404


async def test_list_runs_filters_by_conversation(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    listed = (await client.get(RUNS, params={"conversation_id": "conv-1"})).json()
    assert [item["run_id"] for item in listed["runs"]] == [created["run_id"]]
    missing = await client.get(RUNS, params={"conversation_id": "conv-other"})
    assert missing.status_code == 404


# =====================================================================
# 事件流
# =====================================================================


async def test_sse_stream_replays_then_closes(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    frames: list[str] = []
    async with client.stream("GET", f"{RUNS}/{created['run_id']}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line:
                frames.append(line)
    ids = [line for line in frames if line.startswith("id: ")]
    datas = [line for line in frames if line.startswith("data: ")]
    assert len(ids) == len(datas) > 0
    assert ids == sorted(ids, key=lambda item: int(item[4:]))
    # 结束事件在里面，流自动关闭（PRD 13.3）
    assert any('"run.completed"' in line for line in datas)


async def test_sse_after_skips_replayed_events(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    detail = (await client.get(f"{RUNS}/{created['run_id']}")).json()
    last = detail["last_event_seq"]
    async with client.stream(
        "GET", f"{RUNS}/{created['run_id']}/events", params={"after": last}
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])
    assert body == ""


async def test_event_log_snapshot(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    body = (await client.get(f"{RUNS}/{created['run_id']}/event-log")).json()
    seqs = [event["event_seq"] for event in body["events"]]
    assert seqs == sorted(seqs)
    assert body["events"][0]["type"] == "run.status"
    assert body["events"][-1]["type"] == "run.completed"


# =====================================================================
# PRD 13.2 契约：202 / 图片引用 / DTO 收紧
# =====================================================================


async def test_create_run_returns_202(manager, client) -> None:
    """PRD 13.2：``返回 202 与 run ID``（异步建任务的正确语义）。"""

    response = await client.post(
        RUNS, json={"conversation_id": "conv-1", "message": "测试问题"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"]
    assert body["status"] == "queued"
    await manager.wait(body["run_id"])


async def test_images_are_persisted_and_surfaced(app_store, tmp_path: Path, client) -> None:
    """图片引用**必须**可核对：落进用户消息元数据 + run 详情返回逐条状态。"""

    await seed_conversation(app_store)
    seen: list[list[str]] = []

    async def fake_ingest(context: Any, images: list[str]) -> list[dict[str, Any]]:
        del context
        seen.append(list(images))
        return [
            {
                "image_id": f"block-{index}",
                "block_id": f"block-{index}",
                "content_hash": f"hash-{index}",
                "kind": "data_uri",
                "status": "cached",
                "reason": "原图已缓存、任务已登记；视觉模型未配置",
                "src_ref": f"data:image/png,image-{index}",
                "recorded_at": "2026-09-19T08:00:00Z",
            }
            for index, _ in enumerate(images, start=1)
        ]

    manager = build_manager(
        app_store, services=FakeServices(), root=tmp_path, image_ingest=fake_ingest
    )
    api_deps.register_app_store(app_store)
    app.state.run_manager = manager
    try:
        created = await _create(
            client, images=["data:image/png;base64,AAA", "data:image/png;base64,BBB"]
        )
        assert seen == [["data:image/png;base64,AAA", "data:image/png;base64,BBB"]]
        assert [ref["status"] for ref in created["image_refs"]] == ["cached", "cached"]
        assert created["image_refs"][0]["reason"]

        # 持久化在**用户消息**上，直接读库也能拿到
        messages = await app_store.list_messages("conv-1")
        snapshot = messages[0].source_snapshot
        assert [ref["image_id"] for ref in snapshot["images"]] == ["block-1", "block-2"]
        # 详情与消息元数据一致
        detail = (await client.get(f"{RUNS}/{created['run_id']}")).json()
        assert [ref["image_id"] for ref in detail["image_refs"]] == ["block-1", "block-2"]
        await manager.wait(created["run_id"])
    finally:
        app.state.run_manager = None
        api_deps.register_app_store(None)
        await manager.aclose()


async def test_images_are_not_silently_dropped_without_ingest(manager, client) -> None:
    """未接入图片落库时**明说未生效**（accept-and-drop 比不收更糟）。"""

    created = await _create(client, images=["data:image/png;base64,AAA"])
    assert len(created["image_refs"]) == 1
    ref = created["image_refs"][0]
    assert ref["status"] == "unavailable"
    assert "未接入图片落库" in ref["reason"]
    await manager.wait(created["run_id"])


async def test_too_many_images_is_422(manager, client) -> None:
    response = await client.post(
        RUNS,
        json={
            "conversation_id": "conv-1",
            "message": "测试问题",
            "images": ["data:image/png;base64,AAA"] * 5,
        },
    )
    assert response.status_code == 422


async def test_run_detail_exposes_parent_task_and_enum_status(manager, client, app_store) -> None:
    """PRD 14.2 父子关系显式暴露；任务状态是枚举（非法值在校验层就被拒）。"""

    created = await _create(client)
    run_id = created["run_id"]
    # 造一条带父任务的子任务行（真实运行期由 planner 的层级/依赖产生）
    await app_store.create_task(
        run_id=run_id, task_id="t-child", agent_kind="rag", question="子问题"
    )
    await app_store.create_task(
        run_id=run_id, task_id="t-parent", agent_kind="rag", question="父问题"
    )
    await app_store.update_task_status(run_id, "t-child", "completed")
    await app_store.update_task_status(run_id, "t-parent", "running")
    # 直接改父子关系列（create_task 已支持 parent_task_id，这里显式钉住读路径）
    await app_store._execute(
        "UPDATE agent_tasks SET parent_task_id = ? WHERE run_id = ? AND id = ?",
        ("t-parent", run_id, "t-child"),
    )
    detail = (await client.get(f"{RUNS}/{run_id}")).json()
    child = next(task for task in detail["tasks"] if task["task_id"] == "t-child")
    assert child["parent_task_id"] == "t-parent"
    assert child["status"] == "completed"

    # DTO 收紧：非法状态在模型层就被拒（PRD 12.5「不合法就地失败」）
    from pydantic import ValidationError

    from app.api.routes.runs import TaskSummary

    with pytest.raises(ValidationError):
        TaskSummary(task_id="t-x", agent_kind="rag", status="bogus")
    await manager.wait(run_id)


# =====================================================================
# 停止 / 继续
# =====================================================================


async def test_cancel_then_resume_endpoint(manager, client) -> None:
    created = await _create(client)
    await manager.wait(created["run_id"])
    # 已完成不可取消为活动态：直接调用仍是幂等的终态
    cancelled = await client.post(f"{RUNS}/{created['run_id']}/cancel", json={"reason": "user_stopped"})
    assert cancelled.status_code == 200
    resumed = await client.post(f"{RUNS}/{created['run_id']}/resume")
    assert resumed.status_code == 409  # 已完成的 run 不能继续（PRD 12.3）


async def test_chat_endpoint_is_gone() -> None:
    """旧入口下线（PRD 13.2）：410 而不是 404，并给出替代路径。"""

    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "x"})
    assert response.status_code == 410
    body = response.json()
    assert body["error_code"] == "chat_endpoint_removed"
    assert body["replacement"] == "/api/workspaces/{workspace_id}/runs"
    assert response.headers["link"].endswith('rel="successor-version"')
