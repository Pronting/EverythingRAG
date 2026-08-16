"""ingestion：图片任务 + 内容去重的持久化存储（SQLite）。

对齐 T5 §5.3/5.4：
- ``image_tasks``：每个图片引用的处理状态（pending/done/failed）。应用重启后，
  启动续跑据此加载「未完成」任务自动补做识图（block_id 幂等，重跑不重复）。
- ``image_descriptions``：图片内容哈希 -> 描述文本缓存，跨 URL / 跨导入去重——
  不同图床链接指向同一张图时，只识图一次。

纯本地，零出网；标准库 sqlite3（check_same_thread=False + RLock），
与导入后台线程 / 识图工作线程共享。
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ImageTask:
    """单个图片引用的处理任务（block_id 为确定性主键，幂等）。"""

    block_id: str
    src: str  # 图床 URL
    doc_id: str  # 文件内容指纹
    source_file: str
    heading_path: str
    anchor: str


class ImageStateStore:
    """图片任务 + 内容去重缓存的 SQLite 存储。"""

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
                """
                CREATE TABLE IF NOT EXISTS image_tasks (
                    block_id TEXT PRIMARY KEY,
                    src TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    heading_path TEXT NOT NULL DEFAULT '',
                    anchor TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    retries INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_descriptions (
                    content_hash TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------- tasks

    def add_task(self, task: ImageTask) -> bool:
        """登记任务并返回是否需处理：已 done -> False；否则置 pending -> True。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM image_tasks WHERE block_id = ?", (task.block_id,)
            ).fetchone()
            if row is not None and row["status"] == "done":
                return False
            now = _now_iso()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO image_tasks
                    (block_id, src, doc_id, source_file, heading_path, anchor,
                     status, retries, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
                """,
                (
                    task.block_id,
                    task.src,
                    task.doc_id,
                    task.source_file,
                    task.heading_path,
                    task.anchor,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return True

    def load_pending(self) -> list[ImageTask]:
        """返回全部待处理任务（status=pending），供启动续跑。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT block_id, src, doc_id, source_file, heading_path, anchor "
                "FROM image_tasks WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [
            ImageTask(
                block_id=r["block_id"],
                src=r["src"],
                doc_id=r["doc_id"],
                source_file=r["source_file"],
                heading_path=r["heading_path"],
                anchor=r["anchor"],
            )
            for r in rows
        ]

    def mark_done(self, block_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE image_tasks SET status='done', error=NULL, updated_at=? WHERE block_id=?",
                (_now_iso(), block_id),
            )
            self._conn.commit()

    def mark_failed(self, block_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE image_tasks SET status='failed', error=?, retries=retries+1, "
                "updated_at=? WHERE block_id=?",
                (error[:200], _now_iso(), block_id),
            )
            self._conn.commit()

    def count_status(self) -> dict[str, int]:
        """{pending, done, failed, total}，供 /api/status 进度展示。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM image_tasks GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "done": 0, "failed": 0}
        for row in rows:
            counts[row["status"]] = row["n"]
        counts["total"] = sum(counts.values())
        return counts

    # ------------------------------------------------------------- descriptions

    def get_description(self, content_hash: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT description FROM image_descriptions WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return row["description"] if row is not None else None

    def save_description(self, content_hash: str, description: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO image_descriptions (content_hash, description, created_at) "
                "VALUES (?, ?, ?)",
                (content_hash, description, _now_iso()),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM image_tasks")
            self._conn.execute("DELETE FROM image_descriptions")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
