"""会话存储：data_dir/conversations.json 持久化（对话历史 + AI 标题 + 归档/分支）。

单 JSON 文件足够 MVP（低频写、单用户本地）；量大可迁移 SQLite（aiosqlite 已在依赖）。
conversation 含消息列表；title 由 AI 生成或回退首个用户消息。
归档（archived）的会话不进侧边栏列表，改由设置页「归档对话」恢复；
分支（parent_id）记录 fork 来源，侧边栏据此把子对话折叠在父对话下方（对齐 Codex 客户端）。
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.workspace_store import DEFAULT_WORKSPACE_ID


class ConversationMessage(BaseModel):
    """单条对话消息；持久化正文、依据标签与来源。

    reasoning 为推理模型思维链（本地展示用，前端默认折叠）；不参与任何后续
    请求构造，只随历史一起回放。
    """

    role: str  # "user" | "assistant"
    message_id: str | None = None
    content: str
    sources: list[dict[str, Any]] | None = None
    answer_basis: str | None = None
    policy_version: str | None = None
    reasoning: str | None = None
    created_at: str = ""

    run_id: str | None = None
    citations: list[dict[str, Any]] | None = None


class Conversation(BaseModel):
    """一次会话：标题 + 消息列表 + 归档/分支/工作区元数据。"""

    id: str
    title: str = "新对话"
    created_at: str = ""
    updated_at: str = ""
    messages: list[ConversationMessage] = Field(default_factory=list)
    #: 已归档：从侧边栏隐藏，只能在设置页「归档对话」查看与恢复。
    archived: bool = False
    archived_at: str | None = None
    #: fork 来源会话 id（None = 原始对话）；侧边栏据此把分支折叠在父对话下方。
    parent_id: str | None = None
    #: fork 时父对话的标题，用于「原标题 · 分支」命名与归档页展示来源。
    forked_from_title: str | None = None
    #: 所属工作区（对话分区）；旧数据缺省落到默认工作区。
    workspace_id: str = DEFAULT_WORKSPACE_ID
    #: 创建/更新序号（仅用于同毫秒时间戳下的稳定排序，不进 UI）。
    sort_seq: int = 0


#: 进程内单调递增序号：解决同一毫秒创建/更新的会话在排序时不稳定（时间戳打平）。
_CREATED_SEQ = itertools.count(1)


def _next_seq() -> int:
    return next(_CREATED_SEQ)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sidebar_sort_key(conv: Conversation) -> tuple[float, float, str]:
    """侧边栏排序键（按**升序**排序即可得到显示顺序）。

    **唯一口径：最近活动优先**（用户 2026-09-22 需求 2，对齐 ChatGPT / Codex）。
    无论是用户消息还是 agent 消息，只要 ``updated_at`` 被刷新，该会话就排到最前。

    历史沿革：旧实现有一条「手工拖拽过的排在前并按 ``sort_index``」的高优先级档位，
    结果是一旦拖过一次，整个工作区的会话都被 ``_ensure_sort_indexes`` 写死位次、永远
    压在新会话之上。该档位与 ``sort_index`` 字段已一并移除（旧存档里的 ``sort_index``
    键在读入时被忽略，不影响任何行为）。

    末位用创建序号**倒序**兜底：同一时刻（Windows 时钟粒度约 15.6ms，时间戳极易打平）
    活动过的会话必须仍然「越新越靠前」；若用升序会退化成「创建顺序」，与 recency 语义相反。
    """

    try:
        parsed = datetime.fromisoformat(conv.updated_at)
    except ValueError:
        return (0.0, 0.0, conv.updated_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (-parsed.timestamp(), -float(conv.sort_seq), conv.updated_at)


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

    def list_summaries(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """活跃会话摘要列表（侧边栏顺序）。

        排序：**一律按最近活动倒序**（谁最后收到消息谁在最前，用户 2026-09-22 需求 2）；
        - 已归档的不返回（归档页用 list_archived_summaries）；
        - 空会话（从未交换过消息）不返回，避免侧边栏堆满「新对话」；
        - 给定 workspace_id 时只返回该工作区。
        """

        return self._summaries(
            lambda conv: (
                not conv.archived
                and len(conv.messages) > 0
                and (workspace_id is None or conv.workspace_id == workspace_id)
            ),
            key=_sidebar_sort_key,
        )

    def list_archived_summaries(self) -> list[dict[str, Any]]:
        """已归档会话摘要列表（按归档时间倒序，退化到更新时间）。

        末位用活动序号倒序兜底：归档时间打平时仍按「最近活动的排在前面」，
        不依赖字典迭代顺序（同一批归档的会话时间戳可能完全相同）。
        """

        return self._summaries(
            lambda conv: conv.archived,
            key=lambda conv: (conv.archived_at or conv.updated_at, conv.sort_seq),
            reverse=True,
        )

    def count_by_workspace(self) -> dict[str, int]:
        """各工作区的活跃会话数（供侧边栏分区切换器显示计数）。"""

        with self._lock:
            counts: dict[str, int] = {}
            for conv in self._conversations.values():
                if conv.archived or not conv.messages:
                    continue
                counts[conv.workspace_id] = counts.get(conv.workspace_id, 0) + 1
            return counts

    def set_workspace(self, conversation_id: str, workspace_id: str) -> Conversation | None:
        """把会话移动到指定工作区（不改动消息与归档状态）。"""

        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.workspace_id = workspace_id
            conv.updated_at = _now()
            conv.sort_seq = _next_seq()
            self._persist()
            return conv

    def move_in_sidebar(
        self,
        conversation_id: str,
        workspace_id: str | None = None,
    ) -> Conversation | None:
        """侧边栏拖拽落点 / 同区分支：把会话归到（可选的）目标工作区。

        自 2026-09-22（需求 2）起侧边栏**一律最近活动优先**，工作区内手工排序已取消：
        这里不再接受 ``target_index``，也不再写任何排序位——刷新 ``updated_at`` 后会话
        自然出现在目标区最前。

        仍然必须保留这个入口：同区拖拽是「把存档归属拉回 app.db 权威值」的自愈路径
        （见 ``tests/test_acceptance_t37_move_entry.py`` 的分叉用例），只是它不再排序。
        """

        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            if workspace_id is not None:
                conv.workspace_id = workspace_id

            conv.updated_at = _now()
            conv.sort_seq = _next_seq()
            self._persist()
            return conv

    def archive_workspace(self, workspace_id: str) -> int:
        """归档某工作区下的全部活跃会话，返回归档条数（删除工作区时调用）。

        已归档的会话保持原样（不刷新 archived_at），避免重排归档页顺序。
        """

        with self._lock:
            now = _now()
            moved = 0
            for conv in self._conversations.values():
                if conv.workspace_id != workspace_id or conv.archived:
                    continue
                conv.archived = True
                conv.archived_at = now
                moved += 1
            if moved:
                self._persist()
            return moved

    def search(self, query: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """Search titles and user/assistant text locally, returning compact match excerpts.

        只搜活跃且非空的会话：归档会话不进侧边栏搜索结果（可在归档页浏览）；
        给定 workspace_id 时限定在该工作区内搜索。
        """

        terms = query.casefold().split()
        if not terms:
            return self.list_summaries(workspace_id)
        with self._lock:
            results = []
            candidates = [
                conv
                for conv in self._conversations.values()
                if not conv.archived
                and conv.messages
                and (workspace_id is None or conv.workspace_id == workspace_id)
            ]
            for conv in sorted(candidates, key=lambda c: (c.updated_at, c.sort_seq), reverse=True):
                messages = [m.content for m in conv.messages if m.role in ("user", "assistant")]
                haystack = "\n".join([conv.title, *messages]).casefold()
                if not all(term in haystack for term in terms):
                    continue
                snippet = next((text for text in messages if any(
                    term in text.casefold() for term in terms
                )), "")
                offset = min((snippet.casefold().find(term) for term in terms
                              if term in snippet.casefold()), default=0)
                start = max(0, offset - 35)
                excerpt = snippet[start:start + 140].replace("\n", " ")
                results.append(self._summary(conv) | {
                    "match_excerpt": ("…" if start else "") + excerpt + (
                        "…" if start + 140 < len(snippet) else ""
                    ),
                })
            return results

    def create(
        self, workspace_id: str = DEFAULT_WORKSPACE_ID, conversation_id: str | None = None
    ) -> Conversation:
        """在指定工作区创建空会话并落盘。

        ``conversation_id`` 可选（默认自动生成）：**迁移后的会话在 ``app.db`` 里已有固定 id**，
        侧边栏存档需要与之对齐时由调用方显式给出（T_a27 的移动编排与其测试）。
        本参数纯追加，排序与时间戳语义一字未改（t12）。
        """

        now = _now()
        conv = Conversation(
            id=conversation_id or uuid4().hex,
            title="新对话",
            created_at=now,
            updated_at=now,
            sort_seq=_next_seq(),
            workspace_id=workspace_id or DEFAULT_WORKSPACE_ID,
        )
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
            conv.sort_seq = _next_seq()
            self._persist()
            return conv

    def replace_messages(
        self, conversation_id: str, messages: list[ConversationMessage]
    ) -> None:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None or conv.messages == messages:
                return
            conv.messages = messages
            if messages:
                conv.updated_at = messages[-1].created_at
            conv.sort_seq = _next_seq()
            self._persist()

    def set_title(self, conversation_id: str, title: str) -> Conversation | None:
        """设置标题（截断到 60 字）。"""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.title = title[:60]
            self._persist()
            return conv

    def set_archived(self, conversation_id: str, archived: bool) -> Conversation | None:
        """归档 / 恢复会话（归档记录时间戳，恢复清空）。"""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            conv.archived = archived
            conv.archived_at = _now() if archived else None
            self._persist()
            return conv

    def fork(self, conversation_id: str) -> Conversation | None:
        """从已有会话分支出新会话：深拷贝消息，标题追加「· 分支」。

        分支对话是独立副本：父对话后续变化（继续问答 / 归档 / 删除）都不影响它，
        删除父对话时分支会被提升为顶层，不留悬空 parent_id。
        """
        with self._lock:
            parent = self._conversations.get(conversation_id)
            if parent is None:
                return None
            now = _now()
            branch = Conversation(
                id=uuid4().hex,
                title=f"{parent.title} · 分支"[:60],
                created_at=now,
                updated_at=now,
                sort_seq=_next_seq(),
                messages=[message.model_copy(deep=True) for message in parent.messages],
                parent_id=parent.id,
                forked_from_title=parent.title,
                # 分支与父对话落在同一工作区（分区不会因 fork 而漂移）
                workspace_id=parent.workspace_id,
            )
            self._conversations[branch.id] = branch
            self._persist()
            return branch

    def truncate_from(self, conversation_id: str, index: int) -> Conversation | None:
        """从第 index 条消息起截断（编辑重发：该条及其之后的对话作废）。

        「重新编辑并重发」的语义与 ChatGPT 一致——用户改写某条提问后，它后面的
        问答全部失效，因此这里真截断历史，避免重发后出现重复或错位的旧轮次。
        index <= 0 时清空全部消息；index 不小于当前长度时原样返回。
        """

        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            if index <= 0:
                conv.messages = []
            elif index < len(conv.messages):
                conv.messages = conv.messages[:index]
            conv.updated_at = _now()
            conv.sort_seq = _next_seq()
            self._persist()
            return conv

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            removed = self._conversations.pop(conversation_id, None)
            if removed is None:
                return False
            # 分支的父会话被删除后，分支提升为顶层（不能留下悬空的 parent_id）
            for conv in self._conversations.values():
                if conv.parent_id == conversation_id:
                    conv.parent_id = None
            self._persist()
            return True

    def _summaries(
        self,
        predicate: Any,
        key: Any = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        """按条件筛选并按排序键（缺省 updated_at 升序）投影摘要。"""

        with self._lock:
            selected = [conv for conv in self._conversations.values() if predicate(conv)]
            sort_key = key or (lambda conv: conv.updated_at)
            ordered = sorted(selected, key=sort_key, reverse=reverse)
            return [self._summary(conv) for conv in ordered]

    @staticmethod
    def _summary(conv: Conversation) -> dict[str, Any]:
        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "message_count": len(conv.messages),
            "archived": conv.archived,
            "archived_at": conv.archived_at,
            "parent_id": conv.parent_id,
            "forked_from_title": conv.forked_from_title,
            "workspace_id": conv.workspace_id,
        }
