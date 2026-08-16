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
