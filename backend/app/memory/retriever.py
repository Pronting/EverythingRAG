"""记忆检索通路：独立阈值 + 回表 SQL 二次过滤。

**为什么必须与文档通路分开**（v1.1.0 决策 D6）：
记忆是短的第一人称陈述（"用户偏好 SQLite 而非 Postgres"），文档块是长正文。
把两者放进同一个候选池，实际结果是二选一——要么记忆几乎永不入选，要么为了
让记忆入选而放宽阈值，导致文档召回精度下降（打坏实测的 Hit@5 指标）。
因此本模块是**独立的一条通路**，拥有自己的相似度阈值与召回配额，由上层做
配额融合，而不是共享文档通路的门禁。

**为什么检索完要回表 SQL**：向量索引可能滞后于 SQL 状态（用户刚点"忘掉"）。
``MemoryStore`` 是唯一权威源，回表过滤保证"忘记即不再被召回"，不依赖索引
更新时机。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.memory.index_schema import MEMORY_BLOCK_PREFIX
from app.memory.store import MemoryStore
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk
from app.vectorstore.base import VectorStore

#: 记忆通路的缺省相关度阈值。
#:
#: **与文档通路的 ``DEFAULT_MIN_SIMILARITY = 0.60`` 无关，必须单独校准**：
#: 记忆条目是短陈述，与用户问题的语义距离分布与长文档块不同。此值为初始假设，
#: 须用真实语料实测后回填（见 PRD 第 11 章"所有数值为假设"约定）。
DEFAULT_MEMORY_MIN_SIMILARITY = 0.45

#: 进入上下文的最大记忆条数。记忆是"补充背景"，不应挤占文档证据的位置，
#: 因此配额远小于文档 Top-K（缺省 5）。
DEFAULT_MEMORY_TOP_K = 3


@dataclass(frozen=True)
class MemoryMatch:
    """一条命中的记忆：SQL 权威内容 + 向量相似度。"""

    entry_id: str
    statement: str
    similarity: float
    category: str
    confidence: float
    source_ref: str
    quote: str


class MemoryRetriever:
    """在记忆向量库中召回与问题相关的记忆条目。"""

    def __init__(
        self,
        vectorstore: VectorStore,
        store: MemoryStore,
        candidate_k: int = 20,
        top_k: int = DEFAULT_MEMORY_TOP_K,
        min_similarity: float | None = DEFAULT_MEMORY_MIN_SIMILARITY,
    ) -> None:
        self._vectorstore = vectorstore
        self._store = store
        self._candidate_k = max(1, int(candidate_k))
        self._top_k = max(1, int(top_k))
        self._min_similarity = min_similarity

    def retrieve(self, query_vector: list[float]) -> list[MemoryMatch]:
        """召回相关记忆；**任何失败都返回空列表**，不向上抛错。

        记忆是增强而非依赖：未建索引、向量库为空、query 向量缺失都不应影响
        一轮正常问答（文档通路才是主依据）。这是与文档通路刻意不同的失败策略，
        文档通路未建索引会抛 ``RetrievalError`` 让用户知道"还没导入"。
        """

        if not query_vector or all(value == 0.0 for value in query_vector):
            return []
        try:
            hits = self._vectorstore.query(query_vector, self._candidate_k)
        except Exception:  # noqa: BLE001 -- 记忆通路失败必须静默降级，不影响主问答
            return []

        matches: list[MemoryMatch] = []
        for hit in hits:
            similarity = float(hit.get("similarity", 0.0))
            if self._min_similarity is not None and similarity < self._min_similarity:
                continue
            entry_id = _entry_id_from_block(hit.get("block_id"), hit.get("metadata"))
            if entry_id is None:
                continue
            entry = self._store.get(entry_id)
            # 回表二次过滤：SQL 是权威源，向量索引滞后不得让失效记忆泄漏。
            if entry is None or not entry.is_retrievable():
                continue
            matches.append(
                MemoryMatch(
                    entry_id=entry.id,
                    statement=entry.statement,
                    similarity=similarity,
                    category=entry.category.value,
                    confidence=float(entry.confidence),
                    source_ref=entry.source_ref(),
                    quote=entry.quote,
                )
            )

        matches.sort(key=lambda item: (-item.similarity, item.entry_id))
        return matches[: self._top_k]

    def to_chunk(self, match: MemoryMatch) -> RetrievedChunk:
        """转成 ``RetrievedChunk``，让问答层与前端复用既有来源渲染链路。

        ``chunk_type`` 用 ``"memory"`` 作为来源标记——这是本项与文档块区分的
        唯一开关，前端据此渲染「提炼」徽章与回溯入口。
        """

        return RetrievedChunk(
            block_id=f"{MEMORY_BLOCK_PREFIX}{match.entry_id}",
            text=match.statement,
            similarity=match.similarity,
            source_file=match.category,
            platform="memory",
            chunk_type="memory",
            anchor=match.source_ref or None,
            heading_path=None,
            metadata={
                "memory_id": match.entry_id,
                "category": match.category,
                "confidence": match.confidence,
                "source_ref": match.source_ref,
                "quote": match.quote,
                "source_file": match.category,
            },
            match_type="memory",
        )


def retrieve_memory_chunks(
    retriever: MemoryRetriever | None, query_vector: list[float]
) -> list[RetrievedChunk]:
    """便捷入口：检索并映射为 chunks；未配置记忆检索时返回 []。"""

    if retriever is None:
        return []
    try:
        return [retriever.to_chunk(match) for match in retriever.retrieve(query_vector)]
    except RetrievalError:
        return []
    except Exception:  # noqa: BLE001 -- 记忆通路绝不阻断问答
        return []


def _entry_id_from_block(block_id: Any, metadata: Any) -> str | None:
    """从向量命中解析记忆 ID：优先元数据，回退块 ID 前缀。"""

    if isinstance(metadata, dict):
        raw = metadata.get("memory_id")
        if isinstance(raw, str) and raw:
            return raw
    if isinstance(block_id, str) and block_id.startswith(MEMORY_BLOCK_PREFIX):
        return block_id[len(MEMORY_BLOCK_PREFIX) :] or None
    return None
