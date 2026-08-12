"""共享依赖：向量库注册表 + 导入任务存储单例。

- get_vector_store：按「当前嵌入设置」的 fingerprint 分 key 复用本地向量库。
  切换嵌入模型 -> 不同 fingerprint -> 新 collection（需重新导入）；旧 collection
  保留但不再被检索（向量空间不兼容的正确隔离）。
- get_import_task_store：全局唯一导入任务存储（重启丢失，单用户本地可接受）。
两者均可被 app.dependency_overrides 替换（测试用 fake）。
"""

from __future__ import annotations

from app.core.config import settings
from app.core.conversation_store import ConversationStore
from app.core.settings_store import get_settings_store
from app.ingestion.import_task import ImportTaskStore
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

#: 全局唯一导入任务存储（进程内，重启丢失）
import_task_store = ImportTaskStore()

#: 全局唯一会话存储（持久化到 data_dir/conversations.json）
conversation_store = ConversationStore(settings.data_dir)

#: 按嵌入 fingerprint 分 key 的向量库注册表（进程内缓存，避免每次新建 Chroma client）
_vector_store_registry: dict[str, VectorStore] = {}


def get_vector_store() -> VectorStore:
    """返回与当前嵌入设置匹配的进程内共享向量库（构造零下载零出网）。"""
    app_settings = get_settings_store().load()
    embedder = create_embedder_from_config(app_settings.embed)
    fingerprint = embedder.fingerprint
    store = _vector_store_registry.get(fingerprint)
    if store is None:
        store = create_vector_store(persist_dir=settings.data_dir, embedder=embedder)
        _vector_store_registry[fingerprint] = store
    return store


def get_import_task_store() -> ImportTaskStore:
    """返回全局导入任务存储单例。"""
    return import_task_store
