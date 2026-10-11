"""T08（工作区记忆与中文历史检索）专用夹具。

放在 ``tests/fakes_memory.py`` 而不是共享的 ``fakes_agent.py``：后者由 contracts-db
拥有且只允许追加，工作流专属夹具按 v1.2.0-architecture.md §4 的约定各自成文件。

这里的替身只服务于**本项目自身逻辑**的断言（快照固定、失效传播、分词与索引）。
真实模型语义不在这里验证 —— 那属于 PRD 16.1 的 live 验收，归 T10（t11）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.workspace_context import WorkspacePaths, _shared_paths
from app.generation.base import ChatChunk
from app.memory.workspace_memory import WorkspaceMemory

__all__ = [
    "RecordingChatModel",
    "ScanningChatModel",
    "extraction_reply",
    "memory_for",
    "seed_conversation",
    "workspace_paths_for",
]


def workspace_paths_for(tmp_path: Path, storage_key: str = "default") -> WorkspacePaths:
    """构造某个 ``storage_key`` 的真实路径形状（``tmp_path`` 下的隔离数据目录）。

    D-KB：知识路径共享、私有件按区 —— 本替身直接复用生产构造函数，避免两处形状漂移。
    """

    paths = _shared_paths(tmp_path / "erag-data", storage_key)
    paths.jieba_cache_dir.mkdir(parents=True, exist_ok=True)
    paths.memory_json.parent.mkdir(parents=True, exist_ok=True)
    return paths


def memory_for(tmp_path: Path, storage_key: str = "default") -> WorkspaceMemory:
    """``tmp_path`` 下某个工作区的 ``memory.json`` 访问层。"""

    paths = workspace_paths_for(tmp_path, storage_key)
    return WorkspaceMemory(
        paths.memory_json,
        storage_key=storage_key,
        data_dir=paths.data_dir,
    )


def extraction_reply(items: list[dict[str, Any]]) -> str:
    """构造提炼器的 JSON 回复（与真实提示词要求的形状一致）。"""

    return json.dumps({"memories": items}, ensure_ascii=False)


class RecordingChatModel:
    """记录**完整消息列表**的假对话模型（A21：断言下一次模型请求的上下文）。

    刻意记录整份 messages 而不是最后一条 user 消息：记忆快照是 ``system`` 段，
    "完整上下文体现变更"必须能在 system 段里被看到。
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens if tokens is not None else ["答案"]
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append([dict(message) for message in messages])
        for token in self._tokens:
            yield ChatChunk("content", token)

    def last_request_text(self) -> str:
        """最近一次请求的完整正文拼接（含 system 段）。"""

        return "\n".join(str(message.get("content", "")) for message in self.calls[-1])

    def request_count(self) -> int:
        return len(self.calls)


class ScanningChatModel:
    """按关键词回放不同 JSON 的假提炼模型（同一测试里跑多轮提炼）。"""

    def __init__(self, rule_items: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self._rules = rule_items
        self.calls: list[list[dict[str, Any]]] = []

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append([dict(message) for message in messages])
        transcript = "\n".join(str(message.get("content", "")) for message in messages)
        items: list[dict[str, Any]] = []
        for keyword, rule_items in self._rules:
            if keyword in transcript:
                items = rule_items
                break
        for token in [extraction_reply(items)]:
            yield ChatChunk("content", token)


async def seed_conversation(
    app_store: Any,
    *,
    workspace_id: str = "default",
    conversation_id: str = "conv-1",
    messages: list[tuple[str, str]] = (),
) -> Any:
    """建立一个可用会话并追加给定消息，返回 ``ConversationRecord``。"""

    conversation = await app_store.create_conversation(conversation_id, workspace_id)
    for role, content in messages:
        await app_store.append_message(
            conversation_id=conversation_id, role=role, content=content
        )
    return conversation
