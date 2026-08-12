"""会话存储：data_dir/conversations.json 持久化（对话历史 + AI 标题）。

单 JSON 文件足够 MVP（低频写、单用户本地）；量大可迁移 SQLite（aiosqlite 已在依赖）。
conversation 含消息列表；title 由 AI 生成或回退首个用户消息。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """单条对话消息（持久化用；sources 为检索来源快照）。"""

    role: str  # "user" | "assistant"
    content: str
    sources: list[dict[str, Any]] | None = None
    created_at: str = ""


class Conversation(BaseModel):
    """一次会话：标题 + 消息列表。"""

    id: str
    title: str = "新对话"
    created_at: str = ""
    updated_at: str = ""
    messages: list[ConversationMessage] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationStore:
    """conversations.json 的读写（thread-safe，低频写）。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "conversations.json"
        self._lock = threading.Lock()
        self._conversations: dict[str, Conversation] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw:
                conv = Conversation.model_validate(item)
                self._conversations[conv.id] = conv
        except (OSError, json.JSONDecodeError):
            self._conversations = {}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [conv.model_dump() for conv in self._conversations.values()]
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def list_summaries(self) -> list[dict[str, Any]]:
        """按最近更新倒序返回摘要列表（供侧边栏）。"""
        with self._lock:
            ordered = sorted(
                self._conversations.values(),
                key=lambda c: c.updated_at,
                reverse=True,
            )
            return [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "message_count": len(conv.messages),
                }
                for conv in ordered
            ]

    def create(self) -> Conversation:
        """创建空会话并落盘。"""
        now = _now()
        conv = Conversation(id=uuid4().hex, title="新对话", created_at=now, updated_at=now)
        with self._lock:
            self._conversations[conv.id] = conv
            self._persist()
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def append_messages(
        self,
        conversation_id: str,
        messages: list[ConversationMessage],
    ) -> Conversation | None:
        """追加消息并更新时间戳；不存在返回 None。"""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.messages.extend(messages)
            conv.updated_at = _now()
            self._persist()
            return conv

    def set_title(self, conversation_id: str, title: str) -> Conversation | None:
        """设置标题（截断到 60 字）。"""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.title = title[:60]
            self._persist()
            return conv

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            removed = self._conversations.pop(conversation_id, None)
            if removed is not None:
                self._persist()
            return removed is not None
