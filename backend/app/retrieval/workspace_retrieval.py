"""共享知识库的检索面（D-KB，Owner: T01）。

D-KB（2026-09-20 用户决策）取消「按工作区划分知识库」：所有工作区读写**同一份**语料，
因此本模块只做一件事 —— 把**共享语料**的 dense + BM25 + 章节补充 + 浏览统计收敛到
一处，并按 ``index_cache_key`` 缓存：

1. Chroma persist 根 = ``data_dir``（共享，见 ``core.workspace_context``）；
2. BM25 与章节索引只从**共享 collection** 构建；
3. 向量句柄缓存键 = Provider 配置摘要 + collection 身份（**不含** ``workspace_id``）；
4. BM25 / 目录 / 术语缓存键 = ``workspace_id`` + ``knowledge_revision`` + 索引指纹 +
   生成器版本（``knowledge_revision`` 在导入后由 ``deps.knowledge_revision_bumped``
   **波及所有工作区**递增，因此各区缓存会一起失效）。

实现：每个 ``workspace_id`` 一个 :class:`WorkspaceKnowledgeArea`，内含共享向量库句柄、
``HybridRetriever``（dense + BM25 + 章节补充都在它内部从共享 collection 建索引）
与知识概览（浏览 / 统计）。

T04（工具层）与 T06（意图/目录）只调用 :func:`workspace_knowledge_area`，不自行
构造向量库或 BM25。
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

#: 检索面缓存：键 = 共享数据根 + ``index_cache_key(...)``（D-KB）。
_AREAS: dict[str, WorkspaceKnowledgeArea] = {}
_AREAS_LOCK = threading.RLock()


@dataclass(slots=True)
class WorkspaceKnowledgeArea:
    """共享语料的检索面（共享向量库 + dense/BM25/章节 + 浏览统计）。

    所有成员句柄都绑定创建时的 :class:`WorkspaceContext`：后台任务持有它之后仍然读
    同一份共享语料；``workspace_id`` 只用于缓存键与审计归属。
    """

    context: WorkspaceContext
    vector_store: VectorStore
    cache_key: str
    index_fingerprint: str
    #: 检索器（dense + BM25 + 章节补充），在 ``__post_init__`` 内绑定共享向量库。
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
        """共享语料的 dense + BM25 + 章节补充检索器。"""

        return self._retriever

    def overview(self) -> KnowledgeOverview:
        """共享知识概览（浏览 / 统计：文件、块、图片、主题）；PRD 5.4。"""

        return build_knowledge_overview(self.vector_store)

    def blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        """共享语料全部块（浏览用）。"""

        return self.vector_store.list_blocks()

    def stats(self) -> dict[str, int]:
        """共享语料统计（文件数 / 块数 / 图片块数）；不返回路径。"""

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
    """返回共享知识检索面（按 ``index_cache_key`` 缓存）。

    ``index_fingerprint`` 缺省用 ``context.storage_key``；调用方（``deps``）通常传
    collection 身份，使入库版本或嵌入指纹变化时自动换键。
    """

    fingerprint = index_fingerprint or context.storage_key
    key = context.index_cache_key(
        index_fingerprint=fingerprint,
        generator_version=f"{generator_version}+{catalog_generator_version}",
    )
    # 进程内字典的键比缓存键**更严**：再带上**共享数据根**，避免不同数据目录
    # （测试 tmp、多实例）下同一个 workspace_id 互相命中对方的句柄。D-KB 后语料共享，
    # 因此用 ``data_dir``（而不是本区私有根）作为目录身份。
    registry_key = f"{context.paths.data_dir}|{key}"
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
    """清空进程内检索面缓存（测试 / 数据目录切换时调用）。"""

    with _AREAS_LOCK:
        _AREAS.clear()


def workspace_area_cache_size() -> int:
    """当前缓存的检索面数量（测试断言知识版本递增会换键）。"""

    with _AREAS_LOCK:
        return len(_AREAS)
