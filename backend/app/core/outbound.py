"""唯一出网收敛点 + 网络审计记录器（隐私红线核心）。

CLAUDE.md 隐私红线：所有出网只允许出现在模型 provider 的客户端内，且必须经
唯一 OutboundClient 记录审计；其余模块禁止发外呼。本模块是项目唯一的出网
审计收敛点。

审计事件只记脱敏后的元信息：provider / 目的地 host / 方法 / 状态 / 时间，
绝不含 API key、路径、正文、文件名。纯文本查询链路默认零出网——只要
outbound_client 无事件，outbound_state 即 local-only。

- record 线程安全（threading.Lock），有界环形（超限丢弃最旧）。
- entries() 返回不可变副本，防外部突变。
- 模块级单例 outbound_client 供 app 各处引用（测试可 monkeypatch / clear）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OutboundEvent:
    """一次出网调用的审计记录（不可变；仅含脱敏元信息）。"""

    provider: str  # 如 "chat"
    destination: str  # 只记 host（如 "api.deepseek.com"），不含路径/query/key
    method: str  # 如 "chat.completions"
    status: int | None  # HTTP 状态；异常时为 None
    started_at: datetime
    duration_ms: float | None


class OutboundClient:
    """有界、线程安全的出网审计记录器（全项目唯一出网收敛点）。"""

    def __init__(self, max_entries: int = 200) -> None:
        self._max_entries = max(1, int(max_entries))
        self._events: list[OutboundEvent] = []
        self._lock = threading.Lock()

    def record(self, event: OutboundEvent) -> None:
        """追加一条审计事件；超出容量丢弃最旧（有界环形）。"""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_entries:
                del self._events[: len(self._events) - self._max_entries]

    def entries(self) -> list[OutboundEvent]:
        """返回当前审计快照（不可变副本，防外部突变）。"""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """清空全部审计记录（测试用）。"""
        with self._lock:
            self._events.clear()

    def is_empty(self) -> bool:
        """审计为空（无出网记录）。"""
        with self._lock:
            return not self._events

    @property
    def outbound_state(self) -> str:
        """隐私状态：无出网记录 -> local-only；有记录 -> has-outbound。"""
        return "local-only" if self.is_empty() else "has-outbound"


#: 全局单例：所有出网记录统一收口于此（测试可 monkeypatch / clear）。
outbound_client = OutboundClient()
