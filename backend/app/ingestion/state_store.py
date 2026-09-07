"""增量同步状态库：SQLite 持久化文档指纹表 + 已注册来源根（v0.1 增量同步）。

对齐技术选型决策文档 T1/T4 的 state.db 双写设计：
- ``documents``：已知文档的指纹（path -> content_hash/mtime/size），增量同步
  据此判断 新增 / 更新 / 删除 / 未变。指纹在，重启不丢——否则每次启动都会
  把整个知识库当新文件全量重入库。
- ``sources``：用户导入/同步过的目录根（path -> last_sync_at），前端「同步」
  按钮据此免重选目录，直接对这些根做增量同步。

纯本地文件，零出网。用标准库 sqlite3（同步访问），与导入后台线程同模型；
线程安全（check_same_thread=False + RLock），单用户本地低频访问足够。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION


@dataclass(frozen=True)
class DocFingerprint:
    """已知文档的指纹记录（path 为绝对路径，identity 与 vectorstore 元数据一致）。"""

    path: str
    content_hash: str
    mtime: float
    size: int
    chunk_count: int
    schema_version: int
    index_fingerprint: str
    imported_at: str


def _now_iso() -> str:
    """UTC ISO 时间戳（无微秒，稳定可读）。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class DocumentStateStore:
    """文档指纹 + 来源根的 SQLite 持久化存储。

    所有写操作原子且加锁；同一 store 可被后台线程与请求线程共享
    （check_same_thread=False）。数据库文件不存在时自动创建目录与表。
    """

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    schema_version INTEGER NOT NULL DEFAULT {INGESTION_SCHEMA_VERSION},
                    index_fingerprint TEXT NOT NULL DEFAULT '',
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sources (
                    root_path TEXT PRIMARY KEY,
                    last_sync_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "schema_version" not in columns:
                # 历史行先标为 v1；下一次同步会识别版本落后并完整重处理一次。
                self._conn.execute(
                    "ALTER TABLE documents ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1"
                )
            if "index_fingerprint" not in columns:
                # An empty value is intentionally incompatible with every real index.
                # The next sync rebuilds each legacy row once, then writes the current value.
                self._conn.execute(
                    "ALTER TABLE documents ADD COLUMN index_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            self._conn.commit()

    # ---------------------------------------------------------------- documents

    def load_all(self) -> dict[str, DocFingerprint]:
        """返回全部已知文档指纹：{path: DocFingerprint}。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, content_hash, mtime, size, chunk_count, schema_version, "
                "index_fingerprint, imported_at FROM documents"
            ).fetchall()
        return {
            row["path"]: DocFingerprint(
                path=row["path"],
                content_hash=row["content_hash"],
                mtime=row["mtime"],
                size=row["size"],
                chunk_count=row["chunk_count"],
                schema_version=row["schema_version"],
                index_fingerprint=row["index_fingerprint"],
                imported_at=row["imported_at"],
            )
            for row in rows
        }

    def upsert_fingerprint(
        self,
        path: str,
        content_hash: str,
        mtime: float,
        size: int,
        chunk_count: int,
        schema_version: int = INGESTION_SCHEMA_VERSION,
        index_fingerprint: str = "",
    ) -> None:
        """写入/更新单文档指纹（幂等：INSERT OR REPLACE）。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (path, content_hash, mtime, size, chunk_count, schema_version,
                     index_fingerprint, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    content_hash,
                    mtime,
                    size,
                    chunk_count,
                    schema_version,
                    index_fingerprint,
                    _now_iso(),
                ),
            )
            self._conn.commit()

    def needs_rebuild(
        self,
        *,
        index_fingerprint: str,
        current_chunk_count: int,
        schema_version: int = INGESTION_SCHEMA_VERSION,
    ) -> bool:
        """Return whether persisted documents are incompatible with the current index.

        A non-empty manifest with an empty current collection is also incomplete even if
        every stored fingerprint matches (for example after a collection was recreated).
        """

        documents = self.load_all().values()
        records = list(documents)
        if not records:
            return False
        if current_chunk_count == 0:
            return True
        return any(
            record.schema_version != schema_version
            or record.index_fingerprint != index_fingerprint
            for record in records
        )

    def touch_mtime(self, path: str, mtime: float) -> None:
        """内容未变仅 mtime 变化时，只更新时间戳（保留内容指纹）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE documents SET mtime = ? WHERE path = ?",
                (mtime, path),
            )
            self._conn.commit()

    def delete(self, path: str) -> None:
        """删除单文档指纹（源文件被删除 / 移出目录时调用）。"""
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))
            self._conn.commit()

    def delete_many(self, paths: list[str]) -> int:
        """批量删除文档指纹；历史去重避免逐行提交数百次事务。"""

        if not paths:
            return 0
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany("DELETE FROM documents WHERE path = ?", ((p,) for p in paths))
            self._conn.commit()
            return self._conn.total_changes - before

    # ---------------------------------------------------------------- sources

    def register_source(self, root: Path, synced_at: str | None = None) -> None:
        """登记一个同步根目录（导入/同步成功时调用；重复登记即刷新时间）。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources (root_path, last_sync_at) VALUES (?, ?)",
                (str(root), synced_at or _now_iso()),
            )
            self._conn.commit()

    def list_sources(self) -> list[Path]:
        """返回全部已登记同步根（Path），供 /api/sync 一键同步。"""
        with self._lock:
            rows = self._conn.execute("SELECT root_path FROM sources ORDER BY root_path").fetchall()
        return [Path(row["root_path"]) for row in rows]

    def delete_source(self, root: Path) -> None:
        """移除一个来源根登记（来源目录已失效/被移动删除时调用，避免反复同步报错）。"""
        with self._lock:
            self._conn.execute("DELETE FROM sources WHERE root_path = ?", (str(root),))
            self._conn.commit()

    def clear(self) -> None:
        """清空全部表（测试用）。"""
        with self._lock:
            self._conn.execute("DELETE FROM documents")
            self._conn.execute("DELETE FROM sources")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
