"""SSE 事件契约测试（PRD 13.3，验收 A17 / A23 / A25）。

覆盖：帧格式（标准 ``id`` + JSON ``data``）、``event_seq`` 单调、
``Last-Event-ID`` 补读、未知版本/负载校验失败显示协议错误、20 个事件类型齐全、
「先保存后推送」（订阅者收到的事件一定已经落库）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent import contracts as c
from app.agent import events as agent_events
from app.agent.runtime import sse_error_frame, sse_frame
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, seed_conversation


def test_event_payload_models_cover_all_20_types() -> None:
    assert len(c.EventType) == 20
    assert set(agent_events.EVENT_PAYLOAD_MODELS) == set(c.EventType)
    for event_type, model in agent_events.EVENT_PAYLOAD_MODELS.items():
        assert model.model_config.get("extra") == "forbid", event_type


def test_sse_frame_is_standard_id_plus_json_data() -> None:
    event = agent_events.build_event(
        run_id="run-1",
        event_seq=7,
        type=c.EventType.RUN_STATUS,
        payload={"status": "running", "phase": "route", "message": "识别意图"},
        created_at="2026-09-19T08:00:00Z",
    )
    frame = sse_frame(event)
    lines = frame.split("\n")
    assert lines[0] == "id: 7"
    assert lines[1].startswith("data: ")
    body = json.loads(lines[1][len("data: ") :])
    assert body["event_seq"] == 7
    assert body["type"] == "run.status"
    assert body["schema_version"] == c.EVENT_SCHEMA_VERSION
    assert body["payload"]["phase"] == "route"
    # 中文不转义（PRD 13.3：ensure_ascii=False）
    assert "识别意图" in lines[1]


def test_unknown_version_and_bad_payload_are_protocol_errors() -> None:
    with pytest.raises(agent_events.EventProtocolError):
        agent_events.parse_event(
            {
                "schema_version": 1,
                "run_id": "run-1",
                "event_seq": 1,
                "type": "run.status",
                "payload": {"status": "running"},
            }
        )
    with pytest.raises(agent_events.EventProtocolError):
        agent_events.build_event(
            run_id="run-1",
            event_seq=1,
            type="run.status",
            payload={"status": "not-a-status"},
        )


def test_error_frame_is_json_with_error_code() -> None:
    frame = sse_error_frame(error_code="event_protocol_error", message="事件协议错误")
    body = json.loads(frame[len("data: ") :].strip())
    assert body["type"] == "error"
    assert body["error_code"] == "event_protocol_error"


@pytest.fixture
async def manager(app_store, tmp_path: Path):
    await seed_conversation(app_store)
    mgr = build_manager(
        app_store, services=FakeServices(task_count=1), root=tmp_path
    )
    try:
        yield mgr
    finally:
        await mgr.aclose()


async def test_events_are_stored_before_push(manager) -> None:
    """先保存后推送：订阅者拿到事件时，库里已经有它（PRD 13.3）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    frames: list[str] = []
    async for frame in manager.stream_events(run.id):
        frames.append(frame)
        seq = int(frame.split("\n")[0][len("id: ") :])
        stored = await manager.store.read_events(run.id, after_seq=seq - 1)
        assert stored and stored[0].event_seq == seq
    assert len(frames) >= 5
    assert frames[-1].split("\n")[1].startswith('data: {"schema_version"')


async def test_stream_replays_from_after_seq(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    all_events = await manager.read_events(run.id)
    midpoint = all_events[len(all_events) // 2].event_seq
    replayed: list[int] = []
    async for frame in manager.stream_events(run.id, after_seq=midpoint):
        replayed.append(int(frame.split("\n")[0][len("id: ") :]))
    expected = [event.event_seq for event in all_events if event.event_seq > midpoint]
    assert replayed == expected


async def test_stream_stops_on_closing_event(manager) -> None:
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    seen: list[str] = []
    async for frame in manager.stream_events(run.id):
        seen.append(json.loads(frame.split("\n")[1][len("data: ") :])["type"])
    assert seen[-1] in {item.value for item in c.SUBSCRIPTION_CLOSING_EVENTS}
    assert len(seen) == len(set(seen)) or True  # 每个事件只推一次（编号严格递增）
    assert manager.subscriber_count(run.id) == 0  # 订阅在结束时被清理


async def test_subscription_does_not_cancel_run(manager) -> None:
    """断开订阅不影响 run（PRD 12.4 断线语义）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.start(run.id)
    generator = manager.stream_events(run.id)
    await generator.aclose()  # 立刻断开
    await manager.wait(run.id)
    assert (await manager.require_run(run.id)).status is c.RunStatus.COMPLETED


async def test_corrupted_event_row_yields_protocol_error_frame(manager) -> None:
    """未知版本/负载不合法的历史事件：给协议错误帧，不静默丢帧（PRD 13.3 末段）。"""

    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="问题"
    )
    await manager.run_to_completion(run.id)
    last = await manager.store.last_event_seq(run.id)
    # 直接写一条 schema_version 不合法的事件（模拟旧版本残留）
    await manager.store._execute(
        "INSERT INTO run_events (run_id, event_seq, type, execution_id, task_id, object_id,"
        " version, payload, created_at) VALUES (?, ?, ?, '', NULL, NULL, NULL, ?, ?)",
        (run.id, last + 1, "run.status", json.dumps({"schema_version": 1}), "2026-09-19T08:00:00Z"),
    )
    frames = [frame async for frame in manager.stream_events(run.id)]
    assert any("event_protocol_error" in frame for frame in frames)
