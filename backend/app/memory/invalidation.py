"""人工失效规则与"两套历史视图"（PRD 11.4，Owner: T08）。

PRD 11.4 的核心是一句容易被实现成"一刀切"的话：

    界面读取原始历史，模型读取经过失效规则处理的历史视图：**撤销的要求**从自动
    携带的有效指令中移除，**其他消息内容保留**。

也就是说失效不是"把消息删掉"，而是**两个读取面**：

- **用户视图**：原文照旧（`保留原始聊天供用户查看`），只额外标注"已撤销 / 已纠正"。
  没有它，用户会以为系统把他说过的话弄丢了。
- **模型视图**：只有**被撤销的要求**（用户明确的回答要求类指令）从自动携带的上下文里
  移除；讨论过程、结论依据等其它消息照常保留 —— 否则一次 `/忘掉` 会把整段讨论从
  模型眼前抹掉，用户再问"我们刚讨论到哪"就得不到答案。

公开人物/人名脱敏不在这里（那是 T05 的 `agent/approval.py`）；本模块只处理记忆失效。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.agent.contracts import (
    MemoryEntryStatus,
    MemoryFile,
    MemoryInvalidation,
    MemoryInvalidationKind,
    MemoryKind,
)

__all__ = [
    "REVOKED_INSTRUCTION_NOTE",
    "HistoryMessageView",
    "filter_auto_carried",
    "invalidated_memory_ids",
    "model_history_view",
    "render_invalidation_note",
    "revoked_requirement_message_ids",
    "revoked_response_requirements",
    "user_history_view",
]

#: 模型视图里被撤销要求的位置说明（**不进正文**，只作为返回结构的 reason）。
REVOKED_INSTRUCTION_NOTE = "该消息所表达的回答要求已被用户撤销/纠正，不再自动携带"

#: 只有「回答要求」类记忆失效才把来源消息从自动携带的指令里移除（PRD 11.4）。
_INSTRUCTION_KINDS: frozenset[MemoryKind] = frozenset({MemoryKind.RESPONSE_PREFERENCE})

#: 会触发"撤销要求"判断的失效类型（`disabled` 是整体关闭，不指向单条要求）。
_REVOKING_KINDS: frozenset[MemoryInvalidationKind] = frozenset(
    {
        MemoryInvalidationKind.CORRECTED,
        MemoryInvalidationKind.FORGOTTEN,
    }
)


@dataclass(frozen=True, slots=True)
class HistoryMessageView:
    """一条历史消息在某个读取面上的呈现。"""

    message_id: str
    conversation_id: str
    role: str
    content: str
    created_at: str = ""
    sequence: int = 0
    #: 模型视图专有：False = 这条消息不进入自动携带的有效指令。
    auto_carried: bool = True
    #: 该消息是否被人工失效关联。
    invalidated: bool = False
    #: 面向用户的失效标注（"（已撤销）" / "（已纠正为：…）"）；无失效为 ""。
    note: str = ""
    #: 不进入自动携带的原因（仅模型视图可能非空）。
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "auto_carried": self.auto_carried,
            "invalidated": self.invalidated,
            "note": self.note,
            "reason": self.reason,
        }


def _attr(message: Any, name: str, default: Any = "") -> Any:
    """从 dict 或对象读取字段（AppStore 的 ``MessageRecord`` 与测试字典都能直接用）。"""

    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _entry_index(memory: MemoryFile) -> dict[str, Any]:
    return {entry.id: entry for entry in memory.entries}


def revoked_requirement_message_ids(memory: MemoryFile) -> dict[str, MemoryInvalidation]:
    """``message_id -> 失效记录``：这些消息承载的是**已被撤销的回答要求**。

    判定依据（两条都要成立）：
    1. 失效类型是 ``corrected`` / ``forgotten``（``disabled`` 不指向单条要求）。
    2. 被失效的记忆本身是 ``response_preference``（"用户明确表达的语言、长度、举例
       等要求"，PRD 11.1）。

    记忆 ID 找不到对应条目时按**要求**处理（保守）：宁可少带一条可能是撤销过的指令，
    也不要让一条已被用户撤回的要求继续影响模型行为。
    """

    entries = _entry_index(memory)
    revoked: dict[str, MemoryInvalidation] = {}
    for invalidation in memory.invalidations:
        if invalidation.kind not in _REVOKING_KINDS:
            continue
        entry = entries.get(invalidation.memory_id)
        is_instruction = entry is None or (
            entry.kind in _INSTRUCTION_KINDS
            and entry.status
            in {MemoryEntryStatus.ACTIVE, MemoryEntryStatus.FORGOTTEN}
        )
        if not is_instruction:
            continue
        for message_id in invalidation.source_message_ids:
            if message_id:
                revoked.setdefault(message_id, invalidation)
    return revoked


def invalidated_memory_ids(memory: MemoryFile) -> frozenset[str]:
    """已被人工失效过的记忆 ID 集合。"""

    return frozenset(
        invalidation.memory_id for invalidation in memory.invalidations if invalidation.memory_id
    )


def revoked_response_requirements(memory: MemoryFile) -> list[str]:
    """返回已撤销的回答要求，供模型区分历史表达与当前有效要求。"""

    entries = _entry_index(memory)
    requirements: list[str] = []
    for invalidation in memory.invalidations:
        if invalidation.kind not in _REVOKING_KINDS:
            continue
        entry = entries.get(invalidation.memory_id)
        if entry is not None and entry.kind not in _INSTRUCTION_KINDS:
            continue
        requirement = invalidation.quote.strip()
        if requirement and requirement not in requirements:
            requirements.append(requirement)
    return requirements


def render_invalidation_note(invalidation: MemoryInvalidation) -> str:
    """面向用户的失效标注（PRD 11.4：展示原话同时注明已撤销或纠正）。"""

    if invalidation.kind is MemoryInvalidationKind.CORRECTED:
        replacement = (invalidation.replacement or "").strip()
        return f"（已纠正为：{replacement}）" if replacement else "（已纠正）"
    if invalidation.kind is MemoryInvalidationKind.FORGOTTEN:
        return "（已撤销）"
    return "（记忆已关闭）"


def model_history_view(
    messages: Iterable[Any], memory: MemoryFile
) -> list[HistoryMessageView]:
    """模型读取面：撤销的要求不自动携带，其他消息内容保留（PRD 11.4）。"""

    revoked = revoked_requirement_message_ids(memory)
    views: list[HistoryMessageView] = []
    for message in messages:
        message_id = str(_attr(message, "id", "") or "")
        invalidation = revoked.get(message_id)
        views.append(
            HistoryMessageView(
                message_id=message_id,
                conversation_id=str(_attr(message, "conversation_id", "") or ""),
                role=str(_attr(message, "role", "") or ""),
                content=str(_attr(message, "content", "") or ""),
                created_at=str(_attr(message, "created_at", "") or ""),
                sequence=int(_attr(message, "sequence", 0) or 0),
                auto_carried=invalidation is None,
                invalidated=invalidation is not None,
                reason=REVOKED_INSTRUCTION_NOTE if invalidation is not None else "",
            )
        )
    return views


def user_history_view(
    messages: Iterable[Any], memory: MemoryFile
) -> list[HistoryMessageView]:
    """用户读取面：原话一条不少，只标注失效（PRD 11.4）。"""

    notes: dict[str, MemoryInvalidation] = {}
    entries = _entry_index(memory)
    for invalidation in memory.invalidations:
        if not invalidation.memory_id and invalidation.kind is MemoryInvalidationKind.DISABLED:
            continue
        entry = entries.get(invalidation.memory_id)
        # 用户视图覆盖全部失效类型（含结论纠正），因为它们都改变了"这条消息还成立吗"。
        if entry is not None or invalidation.memory_id == "":
            for message_id in invalidation.source_message_ids:
                if message_id:
                    notes.setdefault(message_id, invalidation)

    views: list[HistoryMessageView] = []
    for message in messages:
        message_id = str(_attr(message, "id", "") or "")
        invalidation = notes.get(message_id)
        views.append(
            HistoryMessageView(
                message_id=message_id,
                conversation_id=str(_attr(message, "conversation_id", "") or ""),
                role=str(_attr(message, "role", "") or ""),
                content=str(_attr(message, "content", "") or ""),
                created_at=str(_attr(message, "created_at", "") or ""),
                sequence=int(_attr(message, "sequence", 0) or 0),
                auto_carried=True,
                invalidated=invalidation is not None,
                note="" if invalidation is None else render_invalidation_note(invalidation),
            )
        )
    return views


def filter_auto_carried(views: Sequence[HistoryMessageView]) -> list[HistoryMessageView]:
    """只保留进入自动携带指令集的消息（模型视图的便捷过滤）。"""

    return [view for view in views if view.auto_carried]
