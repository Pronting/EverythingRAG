"""按工作区构建的检索面（PRD 5.3，Owner: T01）。

PRD 5.3 的前四条只有收敛到一处才守得住：

1. 每工作区独立 Chroma ``PersistentClient`` 路径（5.3.1）；
2. BM25 与章节索引**只能**从本区 collection 构建（5.3.2）；
3. 向量句柄缓存键 = ``workspace_id`` + Provider 配置摘要 + collection 身份（5.3.3）；
4. BM25 / 目录 / 术语缓存键 = ``workspace_id`` + ``knowledge_revision`` + 索引指纹 +
   生成器版本（5.3.4）。

实现：每区一个 :class:`WorkspaceKnowledgeArea`，内含本区向量库句柄、
``HybridRetriever``（dense + BM25 + 章节补充都在它内部从**本区** collection 建索引）
与知识概览（浏览 / 统计）。整区按 ``index_cache_key`` 缓存 —— ``knowledge_revision``
递增后旧键自然失效，重建时 ``HybridRetriever`` 会重新拉本区 blocks。

T04（工具层）与 T06（意图/目录）只调用 :func:`workspace_knowledge_area`，不自行
构造向量库或 BM25，否则跨区串读会重新出现。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.workspace_context import (
    BM25_GENERATOR_VERSION,
    CATALOG_GENERATOR_VERSION,
    WorkspaceContext,
)
from app.generation.knowledge_catalog import KnowledgeOverview, build_knowledge_overview
from app.retrieval.hybrid_retriever import HybridRetriever
from app.vectorstore.base import VectorStore

__all__ = [
    "WorkspaceKnowledgeArea",
    "clear_workspace_area_cache",
    "workspace_area_cache_size",
    "workspace_knowledge_area",
]

#: 进程内每区检索面缓存：键 = ``index_cache_key(...)``（PRD 5.3.4）。
_AREAS: dict[str, WorkspaceKnowledgeArea] = {}
_AREAS_LOCK = threading.RLock()


@dataclass(slots=True)
class WorkspaceKnowledgeArea:
    """一个工作区的检索面（本区向量库 + dense/BM25/章节 + 浏览统计）。

    所有成员句柄都绑定创建时的 :class:`WorkspaceContext`：后台任务持有它之后，
    用户切换界面工作区不会让这个任务改读别的知识库（PRD 5.2 末段）。
    """

    context: WorkspaceContext
    vector_store: VectorStore
    cache_key: str
    index_fingerprint: str
    #: 本区检索器（dense + BM25 + 章节补充），在 ``__post_init__`` 内绑定本区向量库。
    _retriever: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._retriever = HybridRetriever(self.vector_store)

    @property
    def workspace_id(self) -> str:
        return self.context.workspace_id

    @property
    def knowledge_revision(self) -> int:
        return self.context.knowledge_revision

    @property
    def chroma_dir(self) -> Path:
        return self.context.chroma_dir

    @property
    def retriever(self) -> HybridRetriever:
        """本区 dense + BM25 + 章节补充检索器（PRD 5.3.2）。"""

        return self._retriever

    def overview(self) -> KnowledgeOverview:
        """本区知识概览（浏览 / 统计：文件、块、图片、主题）；PRD 5.4。"""

        return build_knowledge_overview(self.vector_store)

    def blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        """本区全部块（浏览用）；只来自本区 collection。"""

        return self.vector_store.list_blocks()

    def stats(self) -> dict[str, int]:
        """本区统计（文件数 / 块数 / 图片块数）；不返回路径。"""

        return {
            "files": self.vector_store.count_files(),
            "chunks": self.vector_store.count(),
            "images": self.vector_store.count_images(),
        }


def workspace_knowledge_area(
    context: WorkspaceContext,
    *,
    vector_store: VectorStore,
    index_fingerprint: str | None = None,
    generator_version: str = BM25_GENERATOR_VERSION,
    catalog_generator_version: str = CATALOG_GENERATOR_VERSION,
    retriever_factory: Callable[[VectorStore], HybridRetriever] | None = None,
) -> WorkspaceKnowledgeArea:
    """返回本区检索面（按 ``index_cache_key`` 缓存）。

    ``index_fingerprint`` 缺省用 ``context.storage_key``；调用方（``deps``）通常传
    collection 身份，使入库版本或嵌入指纹变化时自动换键。
    """

    fingerprint = index_fingerprint or context.storage_key
    key = context.index_cache_key(
        index_fingerprint=fingerprint,
        generator_version=f"{generator_version}+{catalog_generator_version}",
    )
    # 进程内字典的键比 PRD 要求的四项**更严**：再带上级据根目录，避免不同数据目录
    # （测试 tmp、迁移前后、多实例）下同一个 workspace_id 互相命中对方的句柄。
    registry_key = f"{context.paths.root_dir}|{key}"
    with _AREAS_LOCK:
        cached = _AREAS.get(registry_key)
        if cached is not None and cached.workspace_id == context.workspace_id:
            return cached
        area = WorkspaceKnowledgeArea(
            context=context,
            vector_store=vector_store,
            cache_key=key,
            index_fingerprint=fingerprint,
        )
        if retriever_factory is not None:
            area._retriever = retriever_factory(vector_store)
        _AREAS[registry_key] = area
        return area


def clear_workspace_area_cache() -> None:
    """清空进程内检索面缓存（测试 / 迁移提交后切换布局时调用）。"""

    with _AREAS_LOCK:
        _AREAS.clear()


def workspace_area_cache_size() -> int:
    """当前缓存的检索面数量（测试断言知识版本递增会换键）。"""

    with _AREAS_LOCK:
        return len(_AREAS)
