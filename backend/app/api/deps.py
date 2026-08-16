"""共享依赖：向量库注册表 + 导入任务存储单例 + 增量同步状态库。

- get_vector_store：按「当前嵌入设置」的 fingerprint 分 key 复用本地向量库。
  切换嵌入模型 -> 不同 fingerprint -> 新 collection（需重新导入）；旧 collection
  保留但不再被检索（向量空间不兼容的正确隔离）。
- get_import_task_store：全局唯一导入任务存储（重启丢失，单用户本地可接受）。
- get_state_store：增量同步的 SQLite 状态库单例（指纹表/来源根持久化）。
三者均可被 app.dependency_overrides 替换（测试用 fake）。
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.core.conversation_store import ConversationStore
from app.core.settings_store import get_settings_store
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.state_store import DocumentStateStore
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import Embedder, create_embedder_from_config

#: 全局唯一导入任务存储（进程内，重启丢失）
import_task_store = ImportTaskStore()

#: 全局唯一会话存储（持久化到 data_dir/conversations.json）
conversation_store = ConversationStore(settings.data_dir)

#: 按嵌入 fingerprint 分 key 的向量库注册表（进程内缓存，避免每次新建 Chroma client）
_vector_store_registry: dict[str, VectorStore] = {}

#: 按嵌入 fingerprint 分 key 的嵌入器注册表（进程内单例）。
#: 关键：CloudEmbedder 的 dim 是惰性解析的（首次 embed 才知道）；向量库与导入/检索
#: 管线必须共享同一实例，否则向量库侧 dim 未解析会在 upsert 校验时抛 RuntimeError。
_embedder_registry: dict[str, Embedder] = {}


def get_embedder() -> Embedder:
    """返回与当前嵌入设置匹配的进程内共享嵌入器（单例，构造零出网）。"""
    app_settings = get_settings_store().load()
    embedder = create_embedder_from_config(app_settings.embed)
    fingerprint = embedder.fingerprint
    cached = _embedder_registry.get(fingerprint)
    if cached is None:
        _embedder_registry[fingerprint] = embedder
        return embedder
    return cached


def get_vector_store() -> VectorStore:
    """返回与当前嵌入设置匹配的进程内共享向量库（构造零下载零出网）。"""
    embedder = get_embedder()
    fingerprint = embedder.fingerprint
    store = _vector_store_registry.get(fingerprint)
    if store is None:
        store = create_vector_store(persist_dir=settings.data_dir, embedder=embedder)
        _vector_store_registry[fingerprint] = store
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
    """应用启动续跑：识图已配置时，加载未完成图片任务后台补做；返回投递数。

    延迟 import（vision/pipeline/worker）避免模块级循环依赖；失败静默降级（续跑是
    尽力而为，不影响应用启动）。仅当有「pending」任务时才起工作线程。
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
        pipeline = IngestionPipeline(
            embedder=get_embedder(),
            vectorstore=get_vector_store(),
            vision=vision,
            fetcher=ImageFetcher(outbound=outbound_client),
            state_store=get_image_state_store(),
        )
    except Exception:  # noqa: BLE001 -- 嵌入/识图未配置等：续跑降级，不阻断启动
        return 0
    worker = ImageWorker(
        pipeline.process_image_task_and_upsert,
        get_image_state_store(),
        settings.vision_concurrency,
    )
    return worker.resume_pending()
