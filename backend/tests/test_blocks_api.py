"""GET /api/blocks 语义块浏览接口测试：返回块 + 元数据（来源/标题/类型）。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import deps
from app.main import app


class _BlockStore:
    """最小 VectorStore 替身：只实现 list_blocks，其余 no-op。"""

    def __init__(self, blocks: list[tuple[str, str, dict[str, Any]]]) -> None:
        self._blocks = blocks

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._blocks

    def count(self) -> int:
        return len(self._blocks)

    def count_files(self) -> int:
        return 0

    def count_images(self) -> int:
        return 0

    def upsert(self, blocks: list) -> None:
        pass

    def delete_by_ids(self, block_ids: list[str]) -> None:
        pass

    def delete_by_where(self, where: dict[str, Any]) -> None:
        pass

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {}

    def query(self, vector: list[float], top_k: int, where: dict[str, Any] | None = None) -> list:
        return []


def test_list_blocks_returns_blocks_with_metadata() -> None:
    """GET /api/blocks 返回全部块 + 元数据（chunk_type/source_file/heading_path）。"""
    store = _BlockStore(
        [
            (
                "b1",
                "架构图：系统分三层",
                {
                    "chunk_type": "image_description",
                    "source_type": "image_description",
                    "source_file": "C:/kb/notes/a.md",
                    "heading_path": "架构",
                },
            ),
            (
                "b2",
                "正文说明内容",
                {
                    "chunk_type": "text",
                    "source_type": "document",
                    "source_file": "C:/kb/notes/a.md",
                    "heading_path": None,
                },
            ),
        ]
    )
    app.dependency_overrides[deps.get_vector_store] = lambda: store
    try:
        client = TestClient(app)
        resp = client.get("/api/blocks")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    first, second = data["blocks"]
    assert first["block_id"] == "b1"
    assert first["chunk_type"] == "image_description"
    assert first["source_file"] == "C:/kb/notes/a.md"
    assert first["heading_path"] == "架构"
    assert second["chunk_type"] == "text"
    assert second["heading_path"] is None
