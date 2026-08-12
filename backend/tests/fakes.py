"""测试共享 fake：最小 VectorStore 替身（status 计数可控）。

conftest 与导入/状态测试共用；实现完整 VectorStore 协议（no-op），
便于任何依赖注入点替换而不触发真实 Chroma / bge-m3。
"""

from __future__ import annotations

from typing import Any


class FakeVectorStore:
    """可控计数的向量库替身：count / count_files 可配置，其余为安全 no-op。"""

    def __init__(self, file_count: int = 0, chunk_count: int = 0) -> None:
        self._files = file_count
        self._chunks = chunk_count

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        pass

    def delete_by_ids(self, block_ids: list[str]) -> None:
        pass

    def delete_by_where(self, where: dict[str, Any]) -> None:
        pass

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def count(self) -> int:
        return self._chunks

    def count_files(self) -> int:
        return self._files
