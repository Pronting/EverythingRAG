"""记忆向量索引：把 active 记忆条目同步进独立 Chroma collection。

**权威源是 SQL（``MemoryStore``）**，Chroma 只是索引。因此：
- 向量里的文本永远可以在 SQL 侧被修正（编辑陈述后重新索引即可），
  不会出现"向量库和事实各说一套"。
- 检索结果回表 SQL 二次过滤状态，所以"忘记一条记忆"必须立刻生效，
  不受索引更新时机影响（见 ``retriever.py``）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.memory.index_schema import memory_block_id
from app.memory.models import MemoryCategory, MemoryEntry, MemoryStatus
from app.memory.store import MemoryStore
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)

#: 单次嵌入批大小：记忆量级小，但保持与导入管线一致的保守上限。
_EMBED_BATCH_SIZE = 64


def memory_index_text(entry: MemoryEntry) -> str:
    """构造用于嵌入的文本。

    刻意**只嵌入陈述本身**，不带类别前缀：类别是给用户看的组织维度，不是语义
    内容；拼进去会让"偏好/决定"这类词在所有记忆上重复出现，稀释真正的语义差异。
    """

    return entry.statement


def memory_metadata(entry: MemoryEntry) -> dict[str, Any]:
    """记忆块在 Chroma 中的元数据（仅标量，Chroma 约束）。

    刻意**不含 ``source_type``**：记忆不在 ``chunks`` collection 里，也没有自己的
    ``SourceType`` 枚举值——它靠所在 collection 与 chunk_type 识别，避免为了一个
    新来源去扩展文档元数据 schema。
    """

    return {
        "memory_id": entry.id,
        "category": entry.category.value,
        "confidence": float(entry.confidence),
        "user_pinned": bool(entry.user_pinned),
        "source_file": entry.category.value,
        "source_ref": entry.source_ref(),
        "quote": entry.quote,
        "created_at": entry.created_at,
    }


class MemoryIndexer:
    """把记忆条目写入/更新/移出记忆向量库。"""

    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore

    def upsert_entries(self, entries: list[MemoryEntry], store: MemoryStore | None = None) -> int:
        """写入或更新若干条记忆向量，返回成功索引的条数。

        只索引状态可检索的条目；非可检索状态（forgotten/pending/…）会被显式
        移出索引，避免"已忘记的记忆仍可被向量召回"。

        ``store`` 给定时，**先按 ID 回表取权威状态**再判定是否索引：``MemoryEntry``
        是不可变值对象，调用方手上的快照可能已过期（例如用户刚点了"忘掉"），
        若直接信任传入对象，会把已失效的记忆重新写回向量库。传入 ``store`` 可
        彻底排除这类过期快照问题。
        """

        if not entries:
            return 0

        resolved: list[MemoryEntry] = []
        for entry in entries:
            current = store.get(entry.id) if store is not None else None
            resolved.append(current if current is not None else entry)

        indexable = [entry for entry in resolved if entry.is_retrievable()]
        skipped = [entry for entry in resolved if not entry.is_retrievable()]
        if skipped:
            self.delete_entries([entry.id for entry in skipped])

        indexed = 0
        for start in range(0, len(indexable), _EMBED_BATCH_SIZE):
            batch = indexable[start : start + _EMBED_BATCH_SIZE]
            vectors = self._embedder.embed_documents([memory_index_text(item) for item in batch])
            blocks = [
                (
                    memory_block_id(entry.id),
                    memory_index_text(entry),
                    vector,
                    memory_metadata(entry),
                )
                for entry, vector in zip(batch, vectors, strict=True)
            ]
            self._vectorstore.upsert(blocks)
            indexed += len(blocks)
        return indexed

    def delete_entries(self, entry_ids: list[str]) -> None:
        """按记忆 ID 移出向量索引（软删除/失效时调用）。"""

        if not entry_ids:
            return
        self._vectorstore.delete_by_ids([memory_block_id(entry_id) for entry_id in entry_ids])

    def reindex_entry(self, entry: MemoryEntry, store: MemoryStore | None = None) -> int:
        """单条重建索引（编辑陈述后调用）。"""

        return self.upsert_entries([entry], store=store)

    def sync_from_store(self, store: MemoryStore) -> dict[str, int]:
        """把 SQL 与向量索引对齐：全量重建（手动重建入口）。

        返回 ``{"indexed": n, "removed": m}``。刻意做成全量：记忆量级在 10^3 以内，
        全量的正确性远比重建的增量优化更重要（增量需要一套失效追踪，收益不匹配）。
        """

        all_entries = store.list_entries(limit=10_000)
        stale = [entry.id for entry in all_entries if not entry.is_retrievable()]
        self.delete_entries(stale)
        indexed = self.upsert_entries(all_entries, store=store)
        return {"indexed": indexed, "removed": len(stale)}

    def indexed_count(self) -> int:
        """当前记忆 collection 的向量条数（/api/status 与测试用）。"""

        return self._vectorstore.count()


def memory_status_is_active(status: str) -> bool:
    """字符串状态是否可检索（供元数据侧的安全判定）。"""

    try:
        return MemoryStatus(status) is MemoryStatus.ACTIVE
    except ValueError:
        return False


def memory_category_label(category: MemoryCategory) -> str:
    """类别的中文标签（后端只做稳定映射，前端仍以值为准）。"""

    return {
        MemoryCategory.PREFERENCE: "偏好",
        MemoryCategory.IDENTITY: "身份",
        MemoryCategory.DECISION: "决定",
        MemoryCategory.CONCLUSION: "结论",
        MemoryCategory.CONSTRAINT: "约束",
    }[category]
