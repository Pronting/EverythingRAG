"""入库结构版本。

只要解析、清洗、切块边界或关键元数据语义发生不兼容变化，就递增此版本。
增量同步会对旧版本文档执行一次完整重处理，随后重新回到 mtime/hash 快路径。
"""

INGESTION_SCHEMA_VERSION = 7


def make_index_fingerprint(
    embedding_fingerprint: str,
    schema_version: int = INGESTION_SCHEMA_VERSION,
) -> str:
    """Return the complete identity of one document-index representation.

    A collection is compatible only when both the embedding space and the ingestion
    representation match.  Persisting this composite value per document prevents an
    unchanged-file fast path from accidentally skipping a newly selected collection.
    """

    return f"{embedding_fingerprint}::ingestion-v{schema_version}"
