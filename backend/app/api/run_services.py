"""运行期服务装配：把每区句柄接进 RunManager（app 层唯一的"线束"）。

职责边界：
- :mod:`app.agent.runtime` **不 import** ``app.api.*``（agent 层不反向依赖 API 层），
  因此「用哪些 provider 组装工具上下文」这件事放在这里，由 lifespan 注入。
- 这里只做**接线**：句柄全部来自 :mod:`app.api.deps` 的按区 provider 与
  :mod:`app.core.settings_store` 的配置，不重复实现检索 / 出网 / 识图逻辑。
- 测试用 ``RunManager(tool_services=...)`` 注入替身，本模块因此不参与单测默认路径。

约定：``tool_services(context)`` 返回的 dict 里，``_catalog`` / ``_memory``
是给运行层读记忆快照与目录用的**内部键**（ToolContext 不接受它们，构造前会被剔除，
见 :func:`app.agent.runtime.tool_context_for`）；其余键都是 ``ToolContext`` 字段。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from app.agent.app_store import AppStore
from app.agent.memory import compose_agent_memory_context, ensure_conversation_snapshot
from app.agent.runtime import RunManager
from app.agent.tools import build_default_registry
from app.api import deps
from app.api.knowledge_services import knowledge_service_for
from app.api.run_images import LocalImageFetcher, persist_run_images
from app.core.outbound import outbound_client
from app.core.settings_store import (
    get_settings_store,
    is_search_configured,
    is_vision_configured,
)
from app.generation.image_reader import KnowledgeImageReader
from app.generation.knowledge_catalog import catalog_for_area
from app.generation.vision import VisionProviderError, create_vision_model_from_config
from app.ingestion.image_fetch import ImageFetcher
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.evidence_view import RetrievalScope
from app.retrieval.hybrid_retriever import shared_retriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY

logger = logging.getLogger(__name__)

_WARM_LOCK = threading.Lock()
_WARM_STARTED = False
from app.search.base import SearchError
from app.search.factory import create_search_provider_from_config

__all__ = [
    "build_run_manager",
    "history_search_seam",
    "image_ingest_for",
    "tool_services_for",
]


def history_search_seam(
    context: Any,
) -> Callable[[str, Mapping[str, Any]], Awaitable[Mapping[str, Any]]] | None:
    """把 t8 的 **FTS5** 历史检索接进 ``ToolContext.history_search``（PRD 11.5）。

    返回 ``None`` = **FTS 索引当前不可用**（``app.db`` 文件还不存在，例如仅内存库的
    测试进程）：此时工具走**有界回退扫描**，而不是把一个必然失败的接缝装上去让
    ``search_history`` 变成服务错误。已接线时**完全委托**（t5 裁定）：

    - ``total`` / ``conversation_count`` 一律返回 ``None``：实现方没有算它们，
      **不猜数字**（工具层已允许 null）；
    - 无 bm25 分数透出是有意的（工具 schema 只有 excerpt/matched_terms）；
    - 游标就是 ``offset`` 的十进制串；满页才给 ``next_cursor``。

    ``read_history`` **不接缝化**（保持 AppStore 窗口语义，t5 裁定）。
    """

    path = deps.app_db_path()
    if not path.is_file():
        return None
    search = deps.workspace_history_search(context)
    limit_default = 20

    def _offset(cursor: Any) -> int:
        if cursor in (None, ""):
            return 0
        try:
            return max(0, int(str(cursor)))
        except (TypeError, ValueError):
            return 0

    async def seam(query: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        limit = max(1, int(params.get("limit") or limit_default))
        offset = _offset(params.get("cursor"))
        hits = await asyncio.to_thread(
            search.search,
            query,
            limit=limit,
            offset=offset,
            date_from=params.get("start_date"),
            date_to=params.get("end_date"),
        )
        return {
            "messages": [hit.as_dict() for hit in hits],
            "next_cursor": str(offset + limit) if len(hits) >= limit else None,
            "total": None,
            "conversation_count": None,
        }

    return seam


def tool_services_for(context: Any) -> Mapping[str, Any]:
    """组装该工作区的 ToolContext 句柄（PRD 5.2 / 7.1）。

    - 全部句柄按 ``context`` 解析：A 区的向量库 / 状态库 / 图片库不会串到 B 区；
    - 联网 provider 只在「设置里启用且凭证齐备」时构建，否则 ``None``
      （工具会返回友好错误，不会静默出网）；
    - 识图 provider 同理，未配置时不构建 ``image_reader``。
    """

    settings_store = get_settings_store()
    app_settings = settings_store.load()
    vector_store = deps.get_workspace_vector_store(context)
    search_provider = None
    if is_search_configured(app_settings.search):
        try:
            search_provider = create_search_provider_from_config(app_settings.search)
        except SearchError:
            search_provider = None

    vision = None
    if is_vision_configured(app_settings.vision):
        try:
            vision = create_vision_model_from_config(app_settings.vision)
        except VisionProviderError:
            vision = None

    image_store = deps.workspace_image_state_store(context)
    # ⚠️ `catalog_for_area(area)` 的入参是 **WorkspaceKnowledgeArea**（鸭子类型：
    # `context` / `index_fingerprint` / `blocks()`），**不是** WorkspaceContext。
    # 早先这里直接传 `context` ⇒ `area.context` 抛 `AttributeError`（真实 lifespan 下
    # 的研究节点必经 `tool_context_for → tool_services_for`，于是 **每个真 run 都在
    # research 阶段失败**；逻辑型验收用替身装配，所以一直没抓到 —— t34 干净复现）。
    knowledge_area = deps.get_workspace_knowledge_area(context)
    return {
        "vector_store": vector_store,
        # 嵌入模型是 search_knowledge 的前置（工具层没拿到它就直接报不可用），
        # 必须与向量库同一实例来源，否则检索会永远返回空。
        "embedder": deps.get_embedder(),
        "state_store": deps.workspace_state_store(context),
        "image_store": image_store,
        # 检索器按「向量库实例 + 参数」在进程内共享：BM25 索引构建是本进程最贵的
        # 一次性开销（4.7k 块约 2s），而本函数每次工具调用都会被重新装配。
        # 集合规模变化时 retriever 自行失效重建，语义与「每次新建」一致，只是不再重复付费。
        "retriever": shared_retriever(vector_store, min_similarity=DEFAULT_MIN_SIMILARITY),
        "knowledge_area": knowledge_area,
        "knowledge_service": knowledge_service_for(context),
        "search_provider": search_provider,
        "extract_provider": search_provider,
        "image_reader": KnowledgeImageReader(vision, image_store) if vision else None,
        # 内部键（不进 ToolContext）：目录用于路由线索，记忆用于固定快照。
        "_catalog": catalog_for_area(knowledge_area),
        "_memory": deps.workspace_memory(context),
        # FTS5 历史检索（t8）：FTS 索引可用时走委托路径，否则 None → 工具有界回退扫描。
        "history_search": history_search_seam(context),
        # ---- 相关性筛选（集成接线：main 的 rerank 从 /api/chat 迁到 run 管线）----
        # 运行态**惰性**解析：只有真正调用 search_knowledge 时才读当前设置（切换配置
        # 只影响之后的检索，且未启用时不产生任何额外 I/O 或出网）。传入本函数已加载的
        # 同一个设置存储，保证「同一次装配读同一份配置」，也让测试的设置隔离生效。
        # 未启用 → 空运行态 → 工具走旧检索路径；已启用但配置不可用 → 工具明确失败
        # 并指向设置，**不静默回落**（PRD §8.4）。
        "rerank": lambda: deps.build_rerank_runtime(settings_store),
        # 本轮资料范围：与本区向量库同一句柄，scope 标识/索引版本因此不会串区。
        "rerank_scope": lambda: RetrievalScope.for_vectorstore(vector_store),
    }


def warm_retrieval_indexes() -> None:
    """启动期后台构建一次词法索引（幂等，只跑一次）。

    BM25 索引是进程内最贵的一次性开销（4.7k 块实测 2–3.4s），原先是「第一问」替
    整个进程付这笔钱。这里在 lifespan 里起一个 daemon 线程提前建好，第一问直接命中
    缓存。预热失败只记类名（不打印异常正文），第一次检索会再试一次。
    """

    global _WARM_STARTED
    with _WARM_LOCK:
        if _WARM_STARTED:
            return
        _WARM_STARTED = True

    def build() -> None:
        try:
            context = deps.get_workspace_context()
            vector_store = deps.get_workspace_vector_store(context)
            retriever = shared_retriever(vector_store, min_similarity=DEFAULT_MIN_SIMILARITY)
            ready = retriever.ensure_index()
            logger.info("检索索引预热完成：ready=%s", ready)
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响服务可用性
            logger.warning("检索索引预热跳过：error=%s", type(exc).__name__)

    threading.Thread(target=build, name="retrieval-warmup", daemon=True).start()


def build_run_manager(store: AppStore) -> RunManager:
    """按当前设置构建 RunManager（lifespan 调用一次）。"""

    manager = RunManager(
        store=store,
        context_factory=deps.get_workspace_context_factory(),
        registry=build_default_registry(),
        tool_services=tool_services_for,
        memory_enabled=lambda: get_settings_store().load().memory_enabled,
        memory_context=lambda context, conversation_id: memory_context_for(
            store, context, conversation_id
        ),
        memory_composer=lambda context, conversation_id, history: compose_agent_memory_context(
            app_store=store,
            memory=deps.workspace_memory(context),
            conversation_id=conversation_id,
            history=history,
        ),
        memory_for_context=deps.workspace_memory,
        image_ingest=image_ingest_for,
    )
    manager.enable_checkpoints()
    return manager


async def memory_context_for(
    store: AppStore, context: Any, conversation_id: str
) -> str:
    """读取当前工作区的会话记忆，不初始化检索与模型服务。"""

    settings = get_settings_store().load()
    if not settings.memory_enabled:
        return ""
    memory = deps.workspace_memory(context)
    _conversation, snapshot = await ensure_conversation_snapshot(
        app_store=store,
        memory=memory,
        conversation_id=conversation_id,
    )
    return snapshot.text


async def image_ingest_for(context: Any, images: Sequence[str]) -> list[dict[str, Any]]:
    """把 run 附带的图片引用落进**本工作区**图片库（PRD 13.2 / 1.2 R07）。

    - 本区句柄：``workspace_image_state_store(context)`` + 本区向量库 + 本区文档状态库；
    - 下载通道只在「视觉 provider 已配置」时构建（与既有导入路径同一判据）：未配置时
      ``ImageFetcher`` 为 None，``persist_run_images`` 会**明确标记**链接未抓取，
      而不是自建 fetcher 偷偷出网；
    - 视觉已配置时传入既有 ``IngestionPipeline``，描述块 upsert 进本区向量库，
      ``verify_image(image_id=<block_id 或 content_hash>)`` 因此可以直接核对原图。
    """

    app_settings = get_settings_store().load()
    vision = None
    fetcher = None
    if is_vision_configured(app_settings.vision):
        try:
            vision = create_vision_model_from_config(app_settings.vision)
            fetcher = ImageFetcher(outbound=outbound_client)
        except VisionProviderError:
            vision = None
            fetcher = None
    image_store = deps.workspace_image_state_store(context)
    pipeline = None
    if vision is not None:
        pipeline = IngestionPipeline(
            embedder=deps.get_embedder(),
            vectorstore=deps.get_workspace_vector_store(context),
            vision=vision,
            # 既有 ImageFetcher 只走 httpx：包一层让 data URI 本地解码（前端贴图），
            # http(s) 仍走同一个审计过的 fetcher。
            fetcher=LocalImageFetcher(fetcher),
            state_store=image_store,
        )
    return await persist_run_images(
        list(images), image_store=image_store, fetcher=fetcher, pipeline=pipeline
    )
