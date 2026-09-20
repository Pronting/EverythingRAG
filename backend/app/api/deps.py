"""共享依赖：向量库注册表 + 导入任务存储单例 + 增量同步状态库。

- get_vector_store：按完整运行时 Provider 配置的不可逆摘要复用本地向量库包装器；
  collection 仍由嵌入模型 fingerprint 隔离。更换地址/密钥会换新客户端，但同名模型
  不会仅因凭证轮换而强制重建向量空间。
- get_import_task_store：全局唯一导入任务存储（重启丢失，单用户本地可接受）。
- get_state_store：增量同步的 SQLite 状态库单例（指纹表/来源根持久化）。
- conversation_store / workspace_store：会话与对话分区（JSON 持久化）单例。
以上均可被 app.dependency_overrides 替换（测试用 fake）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from functools import lru_cache
from threading import RLock
from typing import Annotated, Any

from fastapi import Depends, HTTPException

from app.api.run_images import LocalImageFetcher
from app.core.config import settings
from app.core.conversation_store import ConversationStore
from app.core.settings_store import EmbedModelConfig, get_settings_store
from app.core.workspace_context import (
    WorkspaceContext,
    WorkspaceContextFactory,
    WorkspaceNotFoundError,
    committed_storage_version,
)
from app.core.workspace_store import WORKSPACE_LAYOUT_VERSION, WorkspaceStore
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION
from app.ingestion.state_store import DocumentStateStore
from app.retrieval.workspace_retrieval import (
    WorkspaceKnowledgeArea,
    clear_workspace_area_cache,
    workspace_knowledge_area,
)
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import Embedder, create_embedder_from_config

logger = logging.getLogger(__name__)

#: 全局唯一导入任务存储（进程内，重启丢失）
import_task_store = ImportTaskStore()

#: 全局唯一会话存储（持久化到 data_dir/conversations.json）
conversation_store = ConversationStore(settings.data_dir)

#: 全局唯一工作区存储（持久化到 data_dir/workspaces.json，保证「默认」工作区存在）
workspace_store = WorkspaceStore(settings.data_dir)

#: 按运行时 Provider 配置摘要分 key 的向量库注册表。
#: collection 名仍由 ``Embedder.fingerprint`` 决定；这里仅避免包装器继续持有旧客户端。
_vector_store_registry: dict[str, VectorStore] = {}

#: 按运行时 Provider 配置摘要分 key 的嵌入器注册表（进程内单例）。
#: 关键：CloudEmbedder 的 dim 是惰性解析的（首次 embed 才知道）；向量库与导入/检索
#: 管线必须共享同一实例，否则向量库侧 dim 未解析会在 upsert 校验时抛 RuntimeError。
_embedder_registry: dict[str, Embedder] = {}
_registry_lock = RLock()


def _embedder_runtime_cache_key(config: EmbedModelConfig) -> str:
    """返回完整 Provider 配置的不可逆进程内缓存键，绝不保存或暴露明文密钥。"""

    api_key = config.api_key.get_secret_value() if config.api_key is not None else None
    provider_config = config.model_dump(mode="json")
    provider_config["api_key"] = api_key
    payload = json.dumps(
        provider_config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"embed-runtime-v1:{hashlib.sha256(payload).hexdigest()}"


def _get_embedder_and_cache_key() -> tuple[Embedder, str]:
    """原子读取当前配置并返回共享嵌入器及其运行时缓存键。"""

    config = get_settings_store().load().embed
    cache_key = _embedder_runtime_cache_key(config)
    with _registry_lock:
        cached = _embedder_registry.get(cache_key)
        if cached is not None:
            return cached, cache_key

        # This is a single-user local service.  Keeping inactive clients has no latency benefit
        # and would retain rotated API secrets in memory, so only the current runtime survives.
        _embedder_registry.clear()
        _vector_store_registry.clear()
        embedder = create_embedder_from_config(config)
        _embedder_registry[cache_key] = embedder
        return embedder, cache_key


def get_embedder() -> Embedder:
    """返回与当前完整 Provider 设置匹配的进程内共享嵌入器。"""

    embedder, _cache_key = _get_embedder_and_cache_key()
    return embedder


def get_vector_store() -> VectorStore:
    """返回与当前完整 Provider 设置匹配的进程内共享向量库包装器。"""

    with _registry_lock:
        embedder, cache_key = _get_embedder_and_cache_key()
        store = _vector_store_registry.get(cache_key)
        if store is None:
            store = create_vector_store(persist_dir=settings.data_dir, embedder=embedder)
            _vector_store_registry[cache_key] = store
        return store


def get_import_task_store() -> ImportTaskStore:
    """返回全局导入任务存储单例。"""
    return import_task_store


@lru_cache
def get_state_store() -> DocumentStateStore:
    """返回增量同步状态库单例（SQLite，落在 data_dir/state.db，重启持久）。"""
    return DocumentStateStore(settings.data_dir / "state.db")


@lru_cache
def get_image_state_store() -> ImageStateStore:
    """返回识图任务 + 内容去重的状态库单例（SQLite，落在 data_dir/image_state.db）。

    持久化：图片任务表（重启续跑）+ 图片描述缓存（跨 URL / 跨导入内容去重）。
    """
    return ImageStateStore(settings.data_dir / "image_state.db")


def resume_pending_image_tasks() -> int:
    """应用启动续跑：识图已配置时，按**工作区**加载未完成图片任务；返回投递数。

    延迟 import（vision/pipeline/worker）避免模块级循环依赖；单区失败静默降级
    （续跑是尽力而为，不影响应用启动），但**一个工作区失败不影响其它工作区** ——
    A 区的坏数据不能让 B 区的图片任务永不续跑（A07/A08）。

    每区用**本区**的图片库 / 状态库 / 向量库句柄：续跑写入必须回到任务创建时的区，
    否则 A 区启动时补做的图片会落进 B 区的索引。
    """

    from app.core.outbound import outbound_client
    from app.core.settings_store import get_settings_store, is_vision_configured
    from app.generation.vision import create_vision_model_from_config
    from app.ingestion.image_fetch import ImageFetcher
    from app.ingestion.image_worker import ImageWorker
    from app.ingestion.pipeline import IngestionPipeline

    app_settings = get_settings_store().load()
    if not is_vision_configured(app_settings.vision):
        return 0
    try:
        vision = create_vision_model_from_config(app_settings.vision, outbound=outbound_client)
    except Exception:  # noqa: BLE001 -- 识图未配置等：续跑降级，不阻断启动
        return 0

    resumed = 0
    for context in _workspace_contexts_for_resume():
        try:
            image_store = workspace_image_state_store(context)
            pipeline = IngestionPipeline(
                embedder=get_embedder(),
                vectorstore=get_workspace_vector_store(context),
                vision=vision,
                # data URI（前端贴图）本地解码；http(s) 仍走审计过的 ImageFetcher。
                fetcher=LocalImageFetcher(ImageFetcher(outbound=outbound_client)),
                state_store=image_store,
            )
            worker = ImageWorker(
                pipeline.process_image_task_and_upsert,
                image_store,
                settings.vision_concurrency,
            )
            resumed += int(worker.resume_pending() or 0)
        except Exception as exc:  # noqa: BLE001 -- 单区失败不影响其它区
            logger.warning(
                "工作区 %s 的识图续跑失败：error=%s",
                context.workspace_id,
                type(exc).__name__,
            )
    return resumed


def _workspace_contexts_for_resume() -> list[Any]:
    """枚举需要续跑的工作区上下文（按 ``storage_key`` 稳定排序，便于诊断复现）。

    app.db 未注册（例如只挂路由的裸测试）时退化为空列表：续跑是启动期尽力而为的
    清理动作，不应在数据源未就绪时猜一个「默认工作区」去写文件。
    """

    store = get_app_store()
    if store is None:
        return []
    try:
        factory = get_workspace_context_factory()
        records = _list_workspace_ids(store)
    except Exception as exc:  # noqa: BLE001 - 列表不可得即跳过续跑
        logger.warning("工作区列表不可得，跳过识别续跑：error=%s", type(exc).__name__)
        return []
    contexts: list[Any] = []
    for workspace_id in records:
        try:
            contexts.append(factory.resolve(workspace_id))
        except Exception as exc:  # noqa: BLE001 - 单区上下文解析失败即跳过该区
            logger.warning(
                "工作区 %s 上下文解析失败：error=%s", workspace_id, type(exc).__name__
            )
    return sorted(contexts, key=lambda item: str(item.storage_key))


def _list_workspace_ids(store: Any) -> list[str]:
    """列出启用中的工作区 ID。

    续跑发生在 lifespan 的**同步**阶段（还没有事件循环可 await），而 app.db 访问层是
    aiosqlite：这里改用一条**只读**的 stdlib ``sqlite3`` 连接读 ``workspaces`` 表。
    只读 + 独立连接，不会与写路径争锁，也不构成第二个写连接池（D20 的口径）。
    """

    del store  # app.db 访问层的 async 接口在同步阶段不可用，见 docstring
    path = app_db_path()
    if not path.exists():
        return []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - 打不开即跳过续跑
        logger.warning("读取工作区列表失败：error=%s", type(exc).__name__)
        return []
    try:
        rows = connection.execute(
            "SELECT id FROM workspaces WHERE status = 'active' ORDER BY storage_key"
        ).fetchall()
        return [str(row[0]) for row in rows]
    except sqlite3.Error as exc:
        logger.warning("读取工作区列表失败：error=%s", type(exc).__name__)
        return []
    finally:
        connection.close()


def read_app_db_workspaces() -> list[dict[str, Any]]:
    """同步**只读**读取 app.db 的 ``workspaces`` 权威行（存档再水合用，D25/D29）。

    与 :func:`_list_workspace_ids` 同一口径：只读 + 独立 stdlib ``sqlite3`` 连接，不与写
    路径争锁，也不构成第二个写连接池（D20）。给 ``WorkspaceContextFactory`` 用来把
    「app.db 权威 → JSON 存档」单向刷新（存档是可再生的物化视图，被删/损坏/过期都会被
    下一次解析前重建）。任何失败都返回空列表：**水合是尽力而为**，失败时读入口照旧
    （迁移前 JSON 本来就是活数据源；迁移后由写路径的镜像维持一致）。
    """

    path = app_db_path()
    if not path.exists():
        return []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("读取工作区权威行失败：error=%s", type(exc).__name__)
        return []
    try:
        connection.row_factory = sqlite3.Row
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(workspaces)").fetchall()
        }
        if "id" not in columns:
            return []
        wanted = [
            "id",
            "storage_key",
            "name",
            "status",
            "default_permission_mode",
            "default_reasoning_level",
            "knowledge_revision",
            "created_at",
            "updated_at",
        ]
        # v1 老库没有这两列（升级前）：读到的行照样可用，缺的字段在转换时取默认值。
        wanted += [column for column in ("pinned", "order_index") if column in columns]
        rows = connection.execute(f"SELECT {', '.join(wanted)} FROM workspaces").fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.warning("读取工作区权威行失败：error=%s", type(exc).__name__)
        return []
    finally:
        connection.close()


# =====================================================================
# V1.2.0 工作区知识隔离（PRD 5.1–5.3，T01）
# =====================================================================
#
# 本节只**追加**新的 provider，既有 ``get_vector_store`` / ``get_state_store`` /
# ``get_image_state_store`` 保持原签名与语义（§4：共享入口 append-only），供尚未迁移
# 到工作区的调用方与既有测试继续使用。新调用方一律走下面的按区 provider：
#
#   get_workspace_context  → 按 ``workspace_id``（查询参数）解析并校验归属
#   get_workspace_vector_store / *_state_store / *_image_state_store
#   get_workspace_knowledge_area → 本区 dense + BM25 + 章节 + 浏览统计
#
# 归属校验与目录解析只在 ``WorkspaceContextFactory`` 一处（PRD 5.1），本模块不做
# 第二份路径拼接。

#: app.db 访问层单例：由 main.py 的 lifespan 创建并注册（T02/approval-privacy）。
#: 这里只持有引用，不自行创建 —— 第二个连接池会造成 WAL 下的意外并发写。
_app_store: Any = None

#: 每区状态库 / 图片库句柄：按实际路径复用，避免每请求重开 SQLite。
_state_store_registry: dict[str, DocumentStateStore] = {}
_image_store_registry: dict[str, ImageStateStore] = {}


def register_app_store(store: Any) -> None:
    """注册 app.db 访问层单例（lifespan 启动时调用；传 None 表示关闭）。"""

    global _app_store
    _app_store = store


def get_app_store() -> Any:
    """返回 app.db 访问层单例；尚未注册时为 None（上下文仍可构造）。"""

    return _app_store


def get_workspace_store() -> WorkspaceStore:
    """工作区存储 provider（与 conversations 路由的同名单例是**同一个对象**）。

    单独提供是为了让本节新增的按区 provider 可被 ``dependency_overrides`` 隔离，
    又不必 import 路由模块（会形成 import 环）。
    """

    return workspace_store


def get_workspace_context_factory(
    store: Annotated[WorkspaceStore | None, Depends(get_workspace_store)] = None,
) -> WorkspaceContextFactory:
    """按**当前** ``settings.data_dir`` 构造上下文工厂。

    刻意不加 ``lru_cache``：测试会 monkeypatch ``settings.data_dir``，缓存会把上一个
    tmp 目录（甚至真实数据目录）的工厂带到下一个用例。缺省参数为 ``None`` 时也能
    直接调用（迁移与后台任务不是 FastAPI 依赖注入场景）。
    """

    return WorkspaceContextFactory(
        settings.data_dir,
        store or get_workspace_store(),
        app_store=get_app_store(),
        # D29：迁移后读入口以 app.db 为权威 —— 存档只是可再生的物化视图，
        # 解析前按需用权威行刷新（被删/损坏/过期都能自愈），见架构 D25/D29。
        app_db_reader=read_app_db_workspaces,
    )


def get_workspace_context(
    workspace_id: str | None = None,
    factory: Annotated[
        WorkspaceContextFactory | None, Depends(get_workspace_context_factory)
    ] = None,
) -> WorkspaceContext:
    """解析并校验归属后的每工作区不可变上下文（PRD 5.2）。

    ``workspace_id`` 是查询参数；省略时落到默认工作区（PRD 15.1 的迁移目标区）。
    工作区不存在抛 404，而不是静默回落到默认区 —— 静默回落会把 A 区的知识答给 B 区。
    """

    resolved = factory or get_workspace_context_factory()
    try:
        return resolved.resolve(workspace_id)
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="工作区不存在") from exc


def get_workspace_vector_store(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
) -> VectorStore:
    """本区向量库句柄（PRD 5.3.1 / 5.3.3）。

    缓存键含 ``workspace_id`` + Provider 配置摘要 + collection 身份：换工作区、换凭证
    端点或换入库版本都会拿到新句柄，不会继续读旧 ANN 段。
    """

    resolved = context if context is not None else get_workspace_context()
    embedder, provider_digest = _get_embedder_and_cache_key()
    collection_identity = f"chunks__{embedder.fingerprint}__v{INGESTION_SCHEMA_VERSION}"
    cache_key = resolved.vector_cache_key(
        provider_digest=provider_digest,
        collection_identity=collection_identity,
    )
    with _registry_lock:
        store = _vector_store_registry.get(cache_key)
        if store is None:
            store = create_vector_store(
                persist_dir=resolved.chroma_dir,
                collection_name=collection_identity,
                embedder=embedder,
            )
            _vector_store_registry[cache_key] = store
        return store


def workspace_state_store(context: WorkspaceContext) -> DocumentStateStore:
    """本区增量同步状态库（``<区根>/state.db``；PRD 5.1）。"""

    key = str(context.state_db)
    with _registry_lock:
        store = _state_store_registry.get(key)
        if store is None:
            store = DocumentStateStore(context.state_db)
            _state_store_registry[key] = store
        return store


def workspace_image_state_store(context: WorkspaceContext) -> ImageStateStore:
    """本区图片任务与内容缓存库（``<区根>/image_state.db``；PRD 5.1 / 5.3.5）。

    每区独立文件即天然隔离：A 区的图片 hash 在 B 区查不到（A07 的 404 语义）。
    """

    key = str(context.image_state_db)
    with _registry_lock:
        store = _image_store_registry.get(key)
        if store is None:
            store = ImageStateStore(context.image_state_db)
            _image_store_registry[key] = store
        return store


def get_workspace_state_store(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
) -> DocumentStateStore:
    return workspace_state_store(context if context is not None else get_workspace_context())


def get_workspace_image_state_store(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
) -> ImageStateStore:
    return workspace_image_state_store(
        context if context is not None else get_workspace_context()
    )


def get_workspace_knowledge_area(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
    vector_store: Annotated[VectorStore | None, Depends(get_workspace_vector_store)] = None,
) -> WorkspaceKnowledgeArea:
    """本区检索面：dense + BM25 + 章节补充 + 浏览统计（PRD 5.3.1–5.3.4）。"""

    resolved = context if context is not None else get_workspace_context()
    store = vector_store if vector_store is not None else get_workspace_vector_store(resolved)
    return workspace_knowledge_area(
        resolved,
        vector_store=store,
        index_fingerprint=resolved.storage_key,
    )


async def knowledge_revision_bumped(workspace_id: str, *, delta: int = 1) -> int | None:
    """提交成功后递增本区知识版本（PRD 5.3.6 的唯一调用点）。

    导入 / 同步 / 删除三处提交成功后调用；递增会换掉本区检索面缓存键，使下一次工具
    调用读到更新后的索引。返回新版本；工作区不存在返回 None。

    D29（迁移后）：落 **app.db** 权威列并把权威行镜像回存档 —— ``knowledge_revision``
    既是 PRD 5.3.4 的缓存键，也是发布前「资料状态」失效检查（PRD 8.2）的依据，
    只写存档而让 app.db 滞后会导致索引缓存不失效、证据失效检查失灵。
    迁移前（``storage_schema_version<2``）保持既有同步 JSON 语义。
    """

    if committed_storage_version(settings.data_dir) >= WORKSPACE_LAYOUT_VERSION:
        store = get_app_store()
        if store is not None:
            if await store.get_workspace(workspace_id) is None:
                return None
            revision = await store.bump_knowledge_revision(workspace_id, delta=delta)
            workspace_store.adopt_record(await store.get_workspace(workspace_id))
            return revision
    return workspace_store.bump_knowledge_revision(workspace_id, delta=delta)


def reset_workspace_scoped_caches() -> None:
    """清空按区句柄注册表与检索面缓存（测试隔离；迁移切换布局后同样应调用）。

    不清 ``_vector_store_registry``：它由 ``_get_embedder_and_cache_key`` 在 Provider
    配置变化时自行清空，且键里已含 workspace_id，跨区不会串读。
    """

    with _registry_lock:
        _state_store_registry.clear()
        _image_store_registry.clear()
    clear_workspace_area_cache()


# =====================================================================
# T08（memory-workspace）追加区：工作区记忆与历史检索 provider
# =====================================================================
#
# 追加规则（v1.2.0-architecture.md §4）：**只追加新的 provider 函数**，不改既有函数
# 签名与语义。因此下面一律自带 import，既不重排文件头的 import 段，也不动
# ``reset_workspace_scoped_caches``：本节的注册表都以**完整路径**为键，tmp 目录切换
# 后旧键不可能被命中，无需清理。

_memory_registry: dict[str, Any] = {}

from pathlib import Path


def app_db_path() -> Path:
    """当前 ``data_dir`` 下的 ``app.db`` 路径（历史检索读侧用；不外发）。"""

    return Path(settings.data_dir) / "app.db"


def workspace_terms(context: WorkspaceContext) -> tuple[str, ...]:
    """本区内部术语（PRD 11.5：分词加入当前工作区内部术语）。

    优先取已经绑定的目录句柄（T06 的 ``catalog``：有 ``iter_terms()`` 或 ``terms``
    就用它）；否则读本区 UTF-8 术语词典文件。两条路都不存在时返回空元组 ——
    此时检索仍然工作，只是不认识内部术语。
    """

    from app.memory.history import load_terms_dictionary

    catalog = context.catalog
    if catalog is not None:
        for attribute in ("iter_terms", "terms"):
            provider = getattr(catalog, attribute, None)
            if provider is None:
                continue
            try:
                values = provider() if callable(provider) else provider
            except Exception:  # noqa: BLE001 -- 目录句柄异常不得让检索整体失败
                values = ()
            cleaned = tuple(str(value).strip() for value in values if str(value).strip())
            if cleaned:
                return cleaned
    return tuple(load_terms_dictionary(context.paths))


def workspace_memory(context: WorkspaceContext) -> Any:
    """本区 ``memory.json`` 访问层（按实际布局解析路径；PRD 5.1 / 11.2）。"""

    from app.memory.workspace_memory import WorkspaceMemory

    key = str(context.paths.memory_json)
    with _registry_lock:
        handle = _memory_registry.get(key)
        if handle is None:
            handle = WorkspaceMemory(
                context.paths.memory_json,
                storage_key=context.storage_key,
                data_dir=context.paths.data_dir,
            )
            _memory_registry[key] = handle
        return handle


def get_workspace_memory(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
) -> Any:
    return workspace_memory(context if context is not None else get_workspace_context())


def workspace_history_search(context: WorkspaceContext) -> Any:
    """本区历史检索器（PRD 11.5 / 13.2 ``GET .../history``）。

    读侧是**同步 sqlite3**（工具调用是同步语义），与 ``app.db`` 的 aiosqlite 写连接
    互不干扰；``app.db`` 不存在时构造成功、首次查询抛 ``HistorySearchError``。
    """

    from app.memory.history import HistorySearch

    return HistorySearch(
        app_db_path(),
        workspace_id=context.workspace_id,
        paths=context.paths,
        terms=workspace_terms(context),
    )


def get_workspace_history_search(
    context: Annotated[WorkspaceContext | None, Depends(get_workspace_context)] = None,
) -> Any:
    return workspace_history_search(
        context if context is not None else get_workspace_context()
    )


def workspace_history_index(context: WorkspaceContext, app_store: Any = None) -> Any:
    """本区历史索引写入侧（PRD 11.5）：经 ``AppStore`` 与业务写入共用连接。"""

    from app.memory.history import WorkspaceHistoryIndex

    store = app_store if app_store is not None else get_app_store()
    if store is None:
        raise RuntimeError("app.db 访问层尚未注册（lifespan 未启动）")
    return WorkspaceHistoryIndex(
        store,
        workspace_id=context.workspace_id,
        paths=context.paths,
        terms=workspace_terms(context),
    )


async def workspace_history_seam(
    context: WorkspaceContext, query: str, request: Any
) -> dict[str, Any]:
    """``search_history`` 工具的冻结接缝实现（t5 定义于 ``agent/tools.py``）。

    签名对齐 ``ToolContext.history_search``：
    ``Callable[[str, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]``；本函数额外接收
    ``context``，装配处用 ``functools.partial(workspace_history_seam, context)`` 绑定即可。

    ``request``：``{"workspace_id", "limit", "cursor", "start_date", "end_date"}``，
    日期为 ``YYYY-MM-DD``（**含当日**，边界处理见 ``normalize_date_bounds``）。

    与参考适配相比有三处**刻意不同**，都是为了不静默出错：

    1. **不在临界区里 ``await``**：``HistorySearch`` 是同步 sqlite3，整体丢进
       ``asyncio.to_thread`` 执行，``close()`` 放 ``finally``（临时连接必须关）。
    2. **``next_cursor`` 用真实 ``total`` 判定**，而不是"本页取满就猜还有下一页" ——
       后者在"恰好剩一页"时会多给一个空页游标。为此多一次**聚合**查询（不回表取正文），
       换来诚实的 ``total`` / ``conversation_count``，不必给 ``null``。
    3. **``limit`` 夹到 ``[1, MAX_HISTORY_LIMIT]``**，避免调用方传越界值后游标推进量与
       实际返回条数不一致。

    ``cursor`` 语义由本实现定义：**十进制偏移量字符串**（``"0"`` / ``"20"``…），
    ``None`` 表示没有下一页。
    """

    import asyncio

    from app.memory.history import DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT

    requested = request.get("limit")
    limit = DEFAULT_HISTORY_LIMIT if requested in (None, "") else int(requested)
    limit = max(1, min(limit, MAX_HISTORY_LIMIT))
    raw_cursor = request.get("cursor")
    offset = 0 if raw_cursor in (None, "") else max(0, int(raw_cursor))
    start_date = request.get("start_date")
    end_date = request.get("end_date")

    def _query() -> tuple[list[Any], int, int]:
        search = workspace_history_search(context)
        try:
            hits = search.search(
                query, limit=limit, offset=offset, date_from=start_date, date_to=end_date
            )
            total, conversations = search.count(query, date_from=start_date, date_to=end_date)
            return hits, total, conversations
        finally:
            search.close()

    hits, total, conversations = await asyncio.to_thread(_query)
    consumed = offset + len(hits)
    return {
        "messages": [hit.as_dict() for hit in hits],
        "next_cursor": str(consumed) if consumed < total else None,
        "total": total,
        "conversation_count": conversations,
    }
