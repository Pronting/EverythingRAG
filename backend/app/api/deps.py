"""共享依赖：向量库单例 + 导入任务存储单例。

- get_vector_store：进程内缓存的本地向量库（status 计数与 import 管线共用同一
  Chroma 客户端，避免同一 persist_dir 上多客户端并发写）。
- get_import_task_store：全局唯一导入任务存储（重启丢失，单用户本地可接受）。
两者均可被 app.dependency_overrides 替换（测试用 fake）。
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.ingestion.import_task import ImportTaskStore
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import create_vector_store

#: 全局唯一导入任务存储（进程内，重启丢失）
import_task_store = ImportTaskStore()


@lru_cache
def get_vector_store() -> VectorStore:
    """返回进程内共享的本地向量库（惰性 bge-m3，构造零下载零出网）。"""
    return create_vector_store(persist_dir=settings.data_dir)


def get_import_task_store() -> ImportTaskStore:
    """返回全局导入任务存储单例。"""
    return import_task_store
