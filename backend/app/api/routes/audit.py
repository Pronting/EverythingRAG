"""审计面板端点：GET /api/audit —— 网络审计（唯一出网收敛点的读取视图）。

隐私红线：事件只含 provider / 目的地 host / 方法 / 状态 / 时间，绝不含
API key、路径、正文、文件名。纯文本查询链路默认零出网，outbound_state
恒为 local-only（无事件）；仅当用户授权云端 provider 并发生真实出网后，
才有事件并转为 has-outbound。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.outbound import OutboundEvent, outbound_client

router = APIRouter()


@router.get("/api/audit")
async def audit() -> dict[str, Any]:
    """返回网络审计快照：outbound_state + 脱敏事件列表 + 事件计数。"""
    events = outbound_client.entries()
    return {
        "outbound_state": outbound_client.outbound_state,
        "events": [_event_payload(event) for event in events],
        "count": len(events),
    }


def _event_payload(event: OutboundEvent) -> dict[str, Any]:
    """OutboundEvent -> 脱敏 dict（started_at 序列化为 ISO 字符串）。"""
    return {
        "provider": event.provider,
        "destination": event.destination,
        "method": event.method,
        "status": event.status,
        "started_at": event.started_at.isoformat(),
        "duration_ms": event.duration_ms,
    }
