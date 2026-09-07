"""DocumentStateStore 单测：指纹写入/读取/覆盖/删除 + 来源根登记（纯本地 SQLite）。

验证增量同步的「记忆」持久化——指纹不丢，重启后仍可跳过未变文件。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION, make_index_fingerprint
from app.ingestion.state_store import DocumentStateStore


def _store(tmp_path: Path) -> DocumentStateStore:
    return DocumentStateStore(tmp_path / "state.db")


# ---------------------------------------------------------------- 指纹读写


def test_upsert_and_load_roundtrip(tmp_path: Path) -> None:
    """写入指纹 -> load_all 读回，字段完整一致。"""
    store = _store(tmp_path)
    index_fingerprint = make_index_fingerprint("fake")
    store.upsert_fingerprint(
        "/abs/a.md", "hash1", 100.0, 42, 3, index_fingerprint=index_fingerprint
    )

    all_docs = store.load_all()
    assert list(all_docs) == ["/abs/a.md"]
    record = all_docs["/abs/a.md"]
    assert record.content_hash == "hash1"
    assert record.mtime == 100.0
    assert record.size == 42
    assert record.chunk_count == 3
    assert record.schema_version == INGESTION_SCHEMA_VERSION
    assert record.index_fingerprint == index_fingerprint
    assert record.imported_at  # 非空


def test_legacy_database_marks_existing_rows_for_reindex(tmp_path: Path) -> None:
    """旧 state.db 自动补 schema_version=1/空索引指纹，触发一次重建。"""

    db_path = tmp_path / "legacy-state.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE documents (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            imported_at TEXT NOT NULL
        );
        CREATE TABLE sources (root_path TEXT PRIMARY KEY, last_sync_at TEXT NOT NULL);
        INSERT INTO documents VALUES ('/abs/a.md', 'old', 1.0, 10, 1, '2026-01-01');
        """
    )
    connection.commit()
    connection.close()

    store = DocumentStateStore(db_path)
    legacy = store.load_all()["/abs/a.md"]
    assert legacy.schema_version == 1
    assert legacy.index_fingerprint == ""
    assert store.needs_rebuild(
        index_fingerprint=make_index_fingerprint("fake"), current_chunk_count=1
    )

    current = make_index_fingerprint("fake")
    store.upsert_fingerprint(
        "/abs/a.md", "old", 1.0, 10, 1, index_fingerprint=current
    )
    migrated = store.load_all()["/abs/a.md"]
    assert migrated.schema_version == INGESTION_SCHEMA_VERSION
    assert migrated.index_fingerprint == current
    assert not store.needs_rebuild(index_fingerprint=current, current_chunk_count=1)
    store.close()


def test_needs_rebuild_when_current_collection_is_empty(tmp_path: Path) -> None:
    """状态已有当前文档但当前 collection 为空时，必须提示重建。"""

    store = _store(tmp_path)
    current = make_index_fingerprint("fake")
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 1, 1, index_fingerprint=current)

    assert store.needs_rebuild(index_fingerprint=current, current_chunk_count=0)
    assert not store.needs_rebuild(index_fingerprint=current, current_chunk_count=1)


def test_upsert_same_path_is_replace(tmp_path: Path) -> None:
    """同一 path 重复写入 -> 覆盖（新内容指纹），行数不增。"""
    store = _store(tmp_path)
    store.upsert_fingerprint("/abs/a.md", "old", 1.0, 10, 1)
    store.upsert_fingerprint("/abs/a.md", "new", 2.0, 20, 2)

    docs = store.load_all()
    assert len(docs) == 1
    assert docs["/abs/a.md"].content_hash == "new"
    assert docs["/abs/a.md"].size == 20


def test_delete_removes_fingerprint(tmp_path: Path) -> None:
    """删除指纹后 load_all 不再包含该 path。"""
    store = _store(tmp_path)
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 1, 1)
    store.delete("/abs/a.md")
    assert store.load_all() == {}


def test_touch_mtime_keeps_content(tmp_path: Path) -> None:
    """touch_mtime 只更新时间戳，内容指纹/size/chunk_count 不变。"""
    store = _store(tmp_path)
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 42, 3)
    store.touch_mtime("/abs/a.md", 999.0)

    record = store.load_all()["/abs/a.md"]
    assert record.mtime == 999.0
    assert record.content_hash == "h"
    assert record.size == 42
    assert record.chunk_count == 3


def test_load_all_empty_db(tmp_path: Path) -> None:
    """空库 load_all 返回空 dict，不抛错。"""
    assert _store(tmp_path).load_all() == {}


# ---------------------------------------------------------------- 来源根


def test_register_and_list_sources(tmp_path: Path) -> None:
    """登记来源根 -> list_sources 读回（有序 Path）。"""
    store = _store(tmp_path)
    store.register_source(Path("/vault"), synced_at="2026-08-12T00:00:00+00:00")
    store.register_source(Path("/notes"), synced_at="2026-08-12T00:00:00+00:00")

    assert store.list_sources() == [Path("/notes"), Path("/vault")]  # 按 root_path 排序


def test_register_source_is_idempotent(tmp_path: Path) -> None:
    """同一根重复登记 -> 不产生重复行（INSERT OR REPLACE）。"""
    store = _store(tmp_path)
    store.register_source(Path("/vault"))
    store.register_source(Path("/vault"))
    assert store.list_sources() == [Path("/vault")]


def test_list_sources_empty(tmp_path: Path) -> None:
    """未登记任何根 -> 空列表。"""
    assert _store(tmp_path).list_sources() == []


# ---------------------------------------------------------------- 持久化与隔离


def test_persists_across_reopen(tmp_path: Path) -> None:
    """数据落盘：关库重开（模拟重启）后指纹仍在 -> 增量可跳过未变文件。"""
    db_path = tmp_path / "state.db"
    store = DocumentStateStore(db_path)
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 42, 3)
    store.register_source(Path("/vault"))
    store.close()

    reopened = DocumentStateStore(db_path)
    assert reopened.load_all()["/abs/a.md"].content_hash == "h"
    assert reopened.list_sources() == [Path("/vault")]
    reopened.close()


def test_clear_resets_all(tmp_path: Path) -> None:
    """clear 清空 documents + sources（测试隔离用）。"""
    store = _store(tmp_path)
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 1, 1)
    store.register_source(Path("/vault"))
    store.clear()
    assert store.load_all() == {}
    assert store.list_sources() == []


def test_imported_at_fresh_each_upsert(tmp_path: Path) -> None:
    """imported_at 随每次 upsert 刷新（反映最近一次入库时间）。"""
    store = _store(tmp_path)
    store.upsert_fingerprint("/abs/a.md", "h1", 1.0, 1, 1)
    first = store.load_all()["/abs/a.md"].imported_at
    store.upsert_fingerprint("/abs/a.md", "h2", 2.0, 2, 1)
    second = store.load_all()["/abs/a.md"].imported_at
    assert second >= first


def test_missing_db_file_auto_created(tmp_path: Path) -> None:
    """state.db 不存在时自动创建目录与文件（父目录也补建）。"""
    db_path = tmp_path / "nested" / "deeper" / "state.db"
    store = DocumentStateStore(db_path)
    store.upsert_fingerprint("/abs/a.md", "h", 1.0, 1, 1)
    assert db_path.is_file()
    assert store.load_all() != {}
    store.close()


@pytest.fixture(autouse=True)
def _close_store(tmp_path: Path) -> None:
    """确保测试结束后关闭连接（避免 sqlite 锁残留）。"""
    yield
