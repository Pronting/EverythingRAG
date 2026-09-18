"""记忆存储：data_dir/memory.db（SQLite）持久化原子记忆条目。

沿用 ``ingestion/state_store.py`` 的既定风格（标准库 sqlite3 + check_same_thread=False
+ RLock），因为记忆与指纹库有完全相同的访问模型：单用户本地、低频写、后台 worker
线程写 / 请求线程读。

与指纹库的关键差异：
- 记忆需要**软删除与状态流转**（D7），所以主键是自增语义的 ``id``，``content_hash``
  只做去重，不承担身份。
- 记忆需要**按状态/类别过滤**，因此状态与类别作为独立列建索引，而不是塞进 JSON。
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from app.memory.models import MemoryCategory, MemoryEntry, MemoryStatus

#: 列表查询默认返回上限，避免前端一次性拉全库（记忆量级预期 10^2~10^3）。
DEFAULT_LIST_LIMIT = 500


def _now_iso() -> str:
    """UTC ISO 时间戳（无微秒），与 state_store 保持同一格式。"""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def content_hash(category: MemoryCategory, statement: str) -> str:
    """记忆的内容指纹：同一类别下同一陈述视为同一条记忆。

    归一化只做空白折叠与大小写统一——不做语义近似，因为"两条记忆是否等价"
    属于提炼器与用户的判断，不该由存储层静默合并。
    """

    normalized = " ".join(statement.casefold().split())
    payload = f"{category.value}\x00{normalized}".encode()
    return sha256(payload).hexdigest()


class MemoryStore:
    """记忆条目的 SQLite 持久化存储（线程安全）。"""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_conversation_id TEXT NOT NULL DEFAULT '',
                    source_message_index INTEGER NOT NULL DEFAULT -1,
                    quote TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    superseded_by TEXT,
                    user_pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status);
                CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash
                    ON memories (content_hash);
                CREATE INDEX IF NOT EXISTS idx_memories_source
                    ON memories (source_conversation_id);
                """
            )
            self._conn.commit()

    # ---------------------------------------------------------------- 写入

    def add(
        self,
        *,
        statement: str,
        category: MemoryCategory,
        confidence: float,
        source_conversation_id: str = "",
        source_message_index: int = -1,
        quote: str = "",
        status: MemoryStatus | None = None,
        user_pinned: bool = False,
        min_confidence: float = 0.0,
    ) -> MemoryEntry:
        """写入一条记忆；同内容已存在时幂等更新而非新增。

        ``status`` 为 None 时按置信度决定：低于 ``min_confidence`` 进 ``pending``
        （待用户确认，不参与检索），否则 ``active``。

        幂等语义：``content_hash`` 唯一索引保证重复提炼同一轮不会产生重复条目；
        已存在的条目只刷新 ``confidence``/``updated_at`` 并回到 ``active``——
        即"又被提取到一次"视为该记忆更可靠。
        """

        resolved_status = status
        if resolved_status is None:
            resolved_status = (
                MemoryStatus.ACTIVE
                if confidence >= min_confidence
                else MemoryStatus.PENDING
            )

        digest = content_hash(category, statement)
        now = _now_iso()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM memories WHERE content_hash = ?", (digest,)
            ).fetchone()
            if existing is not None:
                # 用户显式 /忘掉 的记忆不得被后台提炼"复活"（D7：遗忘是用户意图）。
                if existing["status"] == MemoryStatus.FORGOTTEN.value:
                    return self._row_to_entry(existing)
                self._conn.execute(
                    "UPDATE memories SET confidence = ?, updated_at = ?, status = ? "
                    "WHERE content_hash = ?",
                    (confidence, now, resolved_status.value, digest),
                )
                self._conn.commit()
                refreshed = self._conn.execute(
                    "SELECT * FROM memories WHERE content_hash = ?", (digest,)
                ).fetchone()
                return self._row_to_entry(refreshed)

            entry_id = uuid4().hex
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, content_hash, category, statement, confidence,
                     source_conversation_id, source_message_index, quote,
                     status, superseded_by, user_pinned, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    entry_id,
                    digest,
                    category.value,
                    statement,
                    confidence,
                    source_conversation_id,
                    source_message_index,
                    quote,
                    resolved_status.value,
                    int(user_pinned),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(row)

    def update(
        self,
        entry_id: str,
        *,
        statement: str | None = None,
        category: MemoryCategory | None = None,
        confidence: float | None = None,
        status: MemoryStatus | None = None,
    ) -> MemoryEntry | None:
        """编辑一条记忆（用户在记忆面板操作）。返回更新后的条目，不存在返回 None。

        修改 ``statement`` 或 ``category`` 会改变内容指纹，因此必须同步重算，
        否则去重索引会与真实内容脱节。
        """

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                return None

            new_statement = statement if statement is not None else row["statement"]
            new_category = (
                category if category is not None else MemoryCategory(row["category"])
            )
            fields: dict[str, object] = {
                "statement": new_statement,
                "category": new_category.value,
                "content_hash": content_hash(new_category, new_statement),
                "updated_at": _now_iso(),
            }
            if confidence is not None:
                fields["confidence"] = confidence
            if status is not None:
                fields["status"] = status.value

            assignments = ", ".join(f"{key} = ?" for key in fields)
            self._conn.execute(
                f"UPDATE memories SET {assignments} WHERE id = ?",
                (*fields.values(), entry_id),
            )
            self._conn.commit()
            updated = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(updated)

    def forget(self, entry_id: str) -> MemoryEntry | None:
        """软删除（``/忘掉``）：状态转 ``forgotten``，保留记录以便撤销与回溯。"""

        return self.update(entry_id, status=MemoryStatus.FORGOTTEN)

    def supersede(self, entry_id: str, by_id: str) -> MemoryEntry | None:
        """把一条记忆标为被 ``by_id`` 取代（新旧结论冲突时的处理）。"""

        with self._lock:
            self._conn.execute(
                "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? "
                "WHERE id = ?",
                (MemoryStatus.SUPERSEDED.value, by_id, _now_iso(), entry_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(row) if row is not None else None

    def mark_source_orphaned(self, conversation_id: str) -> int:
        """源会话被删除时，把其派生记忆标为 ``orphaned``（依据不可核验）。

        返回受影响的条数。已被用户遗忘（``forgotten``）的条目不覆盖——用户的
        遗忘意图优先于系统的失效标记。
        """

        if not conversation_id:
            return 0
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? "
                "WHERE source_conversation_id = ? AND status != ?",
                (
                    MemoryStatus.ORPHANED.value,
                    _now_iso(),
                    conversation_id,
                    MemoryStatus.FORGOTTEN.value,
                ),
            )
            self._conn.commit()
            return int(cursor.rowcount)

    # ---------------------------------------------------------------- 读取

    def get(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def list_entries(
        self,
        *,
        statuses: list[MemoryStatus] | None = None,
        categories: list[MemoryCategory] | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> list[MemoryEntry]:
        """按状态/类别过滤列出记忆（新的在前）。

        默认返回**全部状态**（含 forgotten/orphaned/pending），因为记忆面板的
        职责就是让用户看见系统记得什么、以及什么已经失效；只有检索通路才应
        过滤到 ``RETRIEVABLE_STATUSES``。
        """

        clauses: list[str] = []
        params: list[object] = []
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(status.value for status in statuses)
        if categories:
            clauses.append(f"category IN ({','.join('?' for _ in categories)})")
            params.extend(category.value for category in categories)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count(self, *, statuses: list[MemoryStatus] | None = None) -> int:
        """统计条数（状态接口与测试用）。"""

        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            params: list[object] = [status.value for status in statuses]
            sql = f"SELECT COUNT(*) AS n FROM memories WHERE status IN ({placeholders})"
        else:
            sql = "SELECT COUNT(*) AS n FROM memories"
            params = []
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def clear(self) -> None:
        """清空全部记忆（测试隔离用）。"""

        with self._lock:
            self._conn.execute("DELETE FROM memories")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            category=MemoryCategory(row["category"]),
            statement=row["statement"],
            confidence=float(row["confidence"]),
            source_conversation_id=row["source_conversation_id"],
            source_message_index=int(row["source_message_index"]),
            quote=row["quote"],
            status=MemoryStatus(row["status"]),
            superseded_by=row["superseded_by"],
            user_pinned=bool(row["user_pinned"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
