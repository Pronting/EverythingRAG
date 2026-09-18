"""记忆向量索引的结构版本与指纹。

与 ``ingestion/index_schema.py`` 同构但**刻意独立**：记忆的表示方式（陈述文本、
是否拼接类别前缀）与文档块无关，两者必须能各自升级而不互相触发重建。

collection 名形如 ``memory__{embedder.fingerprint}__v1``；换嵌入模型时自动写入
新 collection，旧集合保留可回滚（与文档索引同一约定）。
"""

from __future__ import annotations

#: 记忆入库结构版本。陈述归一化、索引文本构造或元数据语义发生不兼容变化时递增。
MEMORY_SCHEMA_VERSION = 1

#: 记忆 collection 的块 ID 前缀，便于在 Chroma 中与文档块区分（同一 persist_dir）。
MEMORY_BLOCK_PREFIX = "mem-"


def make_memory_fingerprint(
    embedding_fingerprint: str,
    schema_version: int = MEMORY_SCHEMA_VERSION,
) -> str:
    """返回一份记忆索引表示的完整身份。

    与文档索引一致：只有嵌入空间与记忆表示同时匹配才算兼容，避免换嵌入模型后
    继续读旧的 ANN 段。
    """

    payload = f"{embedding_fingerprint}::memory-v{schema_version}"
    return payload


def memory_collection_name(
    embedding_fingerprint: str,
    schema_version: int = MEMORY_SCHEMA_VERSION,
) -> str:
    """记忆 collection 名（与文档 ``chunks__`` 前缀区分，共用同一 persist_dir）。"""

    return f"memory__{embedding_fingerprint}__v{schema_version}"


def memory_block_id(entry_id: str) -> str:
    """记忆条目的向量块 ID（同一记忆重复索引时保持稳定 -> upsert 幂等）。"""

    return f"{MEMORY_BLOCK_PREFIX}{entry_id}"
