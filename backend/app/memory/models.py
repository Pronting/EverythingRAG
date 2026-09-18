"""记忆条目领域模型：原子陈述 + 源指针 + 置信度。

设计要点（v1.1.0 决策 D1/D2/D7）：
- **原子条目**：一条记忆只表达一个可独立撤回的陈述，便于按条审阅/删除，
  也才能回答「这条记忆凭什么」。
- **源指针**：``source_conversation_id`` + ``source_message_index`` + ``quote``
  三元组，保证答案里的 ``[M1]`` 能点回原会话那一轮。``quote`` 是提炼时的
  原文摘录，因此即使源会话被删除，仍能展示「依据不可核验」的原始片段。
- **置信度**：提炼器给出的自我评估，低于阈值的候选不参与检索（见 store 层）。

本模块是纯数据定义，不含 IO 与模型调用。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryCategory(StrEnum):
    """记忆类别。刻意收敛到 5 类：类别越多，提炼器越容易在两可之间摇摆。"""

    PREFERENCE = "preference"  # 用户偏好（技术选型、风格、习惯）
    IDENTITY = "identity"  # 用户身份/背景事实（职业、长期项目）
    DECISION = "decision"  # 已做出的决定及其理由
    CONCLUSION = "conclusion"  # 某轮问答得到的结论/方案
    CONSTRAINT = "constraint"  # 约束条件（必须/禁止，如合规、环境限制）


class MemoryStatus(StrEnum):
    """记忆生命周期状态。

    - ``active``：参与检索。
    - ``pending``：置信度不足，待用户在记忆面板确认，**不参与检索**。
    - ``superseded``：被更新的同类记忆取代，保留可回溯，不参与检索。
    - ``forgotten``：用户主动 ``/忘掉``；软删除，不参与检索但保留记录（D7）。
    - ``orphaned``：源会话已被删除，依据不可核验，不参与检索。
    """

    ACTIVE = "active"
    PENDING = "pending"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    ORPHANED = "orphaned"


#: 默认参与检索的状态集合（其余状态需要用户显式处理）。
RETRIEVABLE_STATUSES: frozenset[MemoryStatus] = frozenset({MemoryStatus.ACTIVE})


class MemoryEntry(BaseModel):
    """一条记忆。

    ``confidence`` 与 ``status`` 是两个正交维度：前者是提炼器对陈述准确性的
    自评，后者是生命周期状态。二者都参与「是否进入检索」的判定。
    """

    id: str
    category: MemoryCategory
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)

    # ---- 源指针（可回溯是硬要求，不得为空字符串占位视为有效）----
    source_conversation_id: str = ""
    source_message_index: int = -1
    quote: str = ""

    status: MemoryStatus = MemoryStatus.ACTIVE
    #: 被哪条记忆取代（仅 ``superseded`` 时有值），用于前端展示替换链。
    superseded_by: str | None = None
    #: 由用户显式 ``/记住`` 创建（优先级最高，提炼器不得自动覆盖）。
    user_pinned: bool = False

    created_at: str = ""
    updated_at: str = ""

    def is_retrievable(self) -> bool:
        """是否允许进入检索通路。"""

        return self.status in RETRIEVABLE_STATUSES

    def source_ref(self) -> str:
        """源指针的稳定字符串形式，供前端构造回溯链接。"""

        if not self.source_conversation_id or self.source_message_index < 0:
            return ""
        return f"{self.source_conversation_id}#{self.source_message_index}"
