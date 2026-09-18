"""记忆子系统：原子记忆条目的提炼、存储、向量索引与检索通路。

与文档知识库（``ingestion``/``retrieval``）刻意分离：
- 记忆是**短的第一人称陈述**，dense 分布与文档块不同，因此拥有独立的
  Chroma collection、独立的相似度阈值与独立的召回配额（见 ``retriever.py``）。
- 记忆必须可**回溯到源会话**，所以存储层保留源指针与原文摘录。
"""

from app.memory.models import (
    RETRIEVABLE_STATUSES,
    MemoryCategory,
    MemoryEntry,
    MemoryStatus,
)
from app.memory.store import MemoryStore

__all__ = [
    "RETRIEVABLE_STATUSES",
    "MemoryCategory",
    "MemoryEntry",
    "MemoryStatus",
    "MemoryStore",
]
