"""GET /api/audit 审计面板端点测试（TestClient，零出网）。

隐私红线：端点只读全局 outbound_client；事件脱敏（host/状态/时间，
无 key/路径/正文）。autouse fixture 每用例前后清空全局审计，保证
/api/status 的 local-only 基线不被跨用例污染。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.outbound import OutboundEvent, outbound_client
from app.main import app


@pytest.fixture(autouse=True)
def _isolate_audit() -> Iterator[None]:
    outbound_client.clear()
    yield
    outbound_client.clear()


def _event(status: int | None = 200) -> OutboundEvent:
    return OutboundEvent(
        provider="chat",
        destination="api.deepseek.com",
        method="chat.completions",
        status=status,
        started_at=datetime.now(UTC),
        duration_ms=12.5,
    )


def _audit() -> dict[str, Any]:
    return TestClient(app).get("/api/audit").json()


def test_initial_local_only() -> None:
    """初始：outbound_state=local-only、events 空、count=0。"""
    data = _audit()
    assert data["outbound_state"] == "local-only"
    assert data["events"] == []
    assert data["count"] == 0


def test_injected_event_flips_state_and_payload() -> None:
    """注入事件后：has-outbound、events 含脱敏事件（无 key/路径/正文）。"""
    outbound_client.record(_event(status=200))
    data = _audit()
    assert data["outbound_state"] == "has-outbound"
    assert data["count"] == 1
    event = data["events"][0]
    assert event["provider"] == "chat"
    assert event["destination"] == "api.deepseek.com"
    assert event["method"] == "chat.completions"
    assert event["status"] == 200
    assert event["duration_ms"] == 12.5
    assert isinstance(event["started_at"], str)  # ISO 字符串
    # 脱敏：事件不含 key / 路径 / 正文
    raw = json.dumps(data, ensure_ascii=False)
    assert "sk-" not in raw
    assert "/v1" not in raw


def test_events_append_order() -> None:
    """多事件按追加顺序返回（FIFO）。"""
    outbound_client.record(_event(status=200))
    outbound_client.record(_event(status=None))
    data = _audit()
    assert data["count"] == 2
    assert [event["status"] for event in data["events"]] == [200, None]


def test_status_privacy_outbound_state_dynamic() -> None:
    """/api/status 的 privacy.outbound_state 动态跟随审计（默认 local-only）。"""
    client = TestClient(app)
    assert client.get("/api/status").json()["privacy"]["outbound_state"] == "local-only"
    outbound_client.record(_event(status=200))
    assert client.get("/api/status").json()["privacy"]["outbound_state"] == "has-outbound"
    outbound_client.clear()
    assert client.get("/api/status").json()["privacy"]["outbound_state"] == "local-only"
