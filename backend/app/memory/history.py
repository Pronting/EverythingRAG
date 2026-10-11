"""中文历史检索：FTS5 + 每区 ``jieba`` 分词（PRD 11.5，Owner: T08）。

分工是 PRD 明确切好的：``app.db`` 的 ``history_fts`` 只存**分词后的索引字段**
（``message_id / conversation_id / workspace_id / tokens``，``tokenize=unicode61``），
**原始消息留在 ``messages`` 表**，查询结果回表取原文。所以本模块做三件事：

1. **分词**：每工作区一个 ``jieba.Tokenizer()``，``tmp_dir`` 指向本区 ``cache/jieba/``，
   ``cache_file`` 带上词典版本；按实例加载本区 UTF-8 术语词典；导入与查询**同一套**
   ``cut_for_search()`` 配置。
2. **索引**：把分词文本写进 ``history_fts``（经 ``AppStore``，与业务写入同一连接），
   并在分词器版本或术语变化时**只重建本区文本索引**。
3. **检索**：查询词项用标准 FTS5 引号规则（内部 ``"`` 转义为 ``""``）构造表达式，
   经 SQL 参数提交（**绝不拼串**），``bm25`` 排序，返回原文而不是分词结果。

为什么查询表达式必须整体走参数绑定：用户输入会直接进入 ``MATCH``。一旦拼串，
``"`` / ``*`` / ``NEAR`` / ``OR`` 都能改变查询语义，甚至构造出语法错误让检索 500。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from app.core.workspace_context import (
    JIEBA_DICT_VERSION,
    JIEBA_GENERATOR_VERSION,
    JIEBA_TOKENIZER_MODE,
    jieba_tokenizer,
)

__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "HISTORY_INDEX_VERSION",
    "HISTORY_TERMS_FILENAME",
    "MAX_HISTORY_LIMIT",
    "HistoryHit",
    "HistoryIndexUnavailableError",
    "HistoryMessage",
    "HistoryQueryError",
    "HistorySearch",
    "HistorySearchError",
    "WorkspaceHistoryIndex",
    "build_match_expression",
    "history_cache_file",
    "history_index_fingerprint",
    "history_tokenizer",
    "load_terms_dictionary",
    "normalize_date_bounds",
    "quote_terms",
    "terms_version",
    "tokenize_for_index",
    "write_terms_dictionary",
]

#: 历史索引的生成器版本：分词口径或索引字段语义变化时必须递增，否则旧索引会被继续命中。
HISTORY_INDEX_VERSION = "history-v1"

HISTORY_TERMS_FILENAME = "terms.txt"
HISTORY_INDEX_MARKER = "history_index.json"

DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 100
#: 摘要窗口（以命中词为中心的字符数）。
SNIPPET_WINDOW = 80


class HistorySearchError(RuntimeError):
    """历史检索的输入/状态错误基类（空查询、索引不可用等）。

    调用方（``search_history`` 工具）需要把"用户输入有问题"和"服务暂时不可用"映射成
    **不同的**错误码，否则一个手滑的空查询会以"服务错误"的形式呈现给用户。因此这里
    给出两个子类，并各自带 :attr:`error_code`；``except HistorySearchError`` 依旧能
    一并捕获（向后兼容）。
    """

    #: 给工具层的稳定错误码（不要改用消息文本判断）。
    error_code = "history_search_error"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class HistoryQueryError(HistorySearchError):
    """查询本身不可检索（空串、纯标点）—— 属**用户输入**问题，应映射成参数错误。"""

    error_code = "invalid_query"


class HistoryIndexUnavailableError(HistorySearchError):
    """历史索引当前不可用（``app.db`` 不存在 / FTS 表缺失）—— 属**服务状态**，可重试。"""

    error_code = "service_unavailable"


# ---------------------------------------------------------------------------
# 术语词典（PRD 11.5：加入当前工作区内部术语）
# ---------------------------------------------------------------------------


def terms_dictionary_path(paths: Any) -> Path:
    """本区术语词典路径（落在本区 ``cache/jieba/``）。"""

    return Path(paths.jieba_cache_dir) / HISTORY_TERMS_FILENAME


def write_terms_dictionary(paths: Any, terms: Iterable[str]) -> Path:
    """把工作区内部术语写成 UTF-8 词典文件（jieba ``load_userdict`` 可直接读）。

    每行一个词（不带词频/词性）：内部术语的出现频率由用户资料决定，硬编码一个词频
    反而会让分词在"该不该切开"上做出没有依据的决定。
    """

    path = terms_dictionary_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = sorted({str(term).strip() for term in terms if str(term).strip()})
    path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")
    return path


def load_terms_dictionary(paths: Any) -> list[str]:
    """读取本区术语词典；不存在返回空列表。"""

    path = terms_dictionary_path(paths)
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def terms_version(terms: Iterable[str]) -> str:
    """术语词典版本（内容指纹前 12 位；无词为 ``"0"``）。

    版本进缓存文件名与索引指纹：术语变化后"只重建相应文本索引"，而不是全库重建。
    """

    cleaned = sorted({str(term).strip() for term in terms if str(term).strip()})
    if not cleaned:
        return "0"
    payload = "\n".join(cleaned).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def history_index_fingerprint(*, terms_version_value: str) -> str:
    """本区历史索引的身份：分词器版本 + 索引结构版本 + 术语版本。

    ``JIEBA_GENERATOR_VERSION``（分词器 + 分词模式 + 词典版本）由
    ``app/core/workspace_context.py`` **唯一**定义（t4 冻结，§6.7 D26）：迁移侧与查询侧
    必须取同一个值，否则"分词器版本变化只重建相应文本索引"会两侧各判一次、各建一次。
    """

    payload = f"{JIEBA_GENERATOR_VERSION}::{HISTORY_INDEX_VERSION}::{terms_version_value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

_tokenizers: dict[str, Any] = {}
_tokenizers_lock = threading.RLock()


def history_cache_file(paths: Any, *, terms_version_value: str) -> Path:
    """本区历史检索的 jieba 缓存文件。

    PRD 11.5 要求 ``cache_file`` **包含词典版本**。这里比 T01 的基础版本多带一层
    **本区术语词典版本**：同一份 jieba 词典配上不同的内部术语会切出不同的词，缓存
    必须跟着术语走，否则换术语后仍复用按旧术语切好的缓存。
    """

    return Path(paths.jieba_cache_dir) / (
        f"jieba.history.{JIEBA_TOKENIZER_MODE}.{JIEBA_DICT_VERSION}."
        f"{terms_version_value}.cache"
    )


def history_tokenizer(paths: Any, *, terms: Sequence[str] = ()) -> Any:
    """返回本区历史检索用的 ``jieba.Tokenizer``。

    两种情形分开处理，避免同一工作区在同一进程里出现两个"权威"分词器：

    - **没有内部术语**：直接复用 T01 的每区工厂 ``jieba_tokenizer(paths)``
      （同一实例、同一版本化 ``cache_file``、同一 ``cut_for_search`` 模式）。
    - **有内部术语**：在 T01 的目录之上另建实例，``cache_file`` 再加术语版本，
      并按实例加载本区 UTF-8 词典。

    顺序很关键：``initialize()`` 会重建 ``FREQ/total``，所以用户词典必须在它**之后**
    加载，否则词会被基础词典覆盖。
    """

    version = terms_version(terms)
    cleaned = [str(term).strip() for term in terms if str(term).strip()]
    if not cleaned:
        return jieba_tokenizer(paths)

    import jieba

    cache_dir = Path(paths.jieba_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{cache_dir.resolve()}|{version}"
    with _tokenizers_lock:
        cached = _tokenizers.get(key)
        if cached is not None:
            return cached
        tokenizer = jieba.Tokenizer()
        tokenizer.tmp_dir = str(cache_dir)
        tokenizer.cache_file = str(history_cache_file(paths, terms_version_value=version))
        tokenizer.initialize()
        dictionary = write_terms_dictionary(paths, cleaned)
        tokenizer.load_userdict(str(dictionary))
        _tokenizers[key] = tokenizer
        return tokenizer


def _is_meaningful(term: str) -> bool:
    """词项是否含实际字符（丢掉 ``？`` ``，`` 这类纯标点，它们只会污染 MATCH 表达式）。"""

    return any(character.isalnum() for character in term)


def cut_for_search(text: str, tokenizer: Any) -> list[str]:
    """``cut_for_search`` 的词项流（物化为 list：它返回的是 generator）。"""

    if not text:
        return []
    return [str(token).strip() for token in tokenizer.cut_for_search(text)]


def tokenize_for_index(text: str, tokenizer: Any) -> str:
    """索引侧分词文本：保留词频（bm25 依赖 TF），空格分隔交给 FTS5 ``unicode61``。

    零依赖降级：分词器异常时退化为按空白切分，迁移/导入绝不能因为分词失败而中断。
    """

    if not text:
        return ""
    try:
        tokens = tokenize_terms(text, tokenizer)
    except Exception:  # noqa: BLE001 -- 分词失败不能阻断索引写入
        return " ".join(text.split())
    return " ".join(token for token in tokens if token)


def tokenize_terms(text: str, tokenizer: Any) -> list[str]:
    """导入与查询共用的词项序列（去空白、保序、**不去重**：TF 参与 bm25）。"""

    return [token for token in cut_for_search(text, tokenizer) if token.strip()]


def quote_terms(terms: Iterable[str]) -> list[str]:
    """标准 FTS5 引号规则：整体加引号，内部 ``"`` 双写转义。"""

    quoted: list[str] = []
    for term in terms:
        text = str(term).strip()
        if not text or not _is_meaningful(text):
            continue
        quoted.append('"' + text.replace('"', '""') + '"')
    return quoted


def build_match_expression(query: str, tokenizer: Any) -> str:
    """构造 FTS5 ``MATCH`` 表达式：词项去重保序，``AND`` 连接，逐项加引号。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for term in tokenize_terms(query, tokenizer):
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    quoted = quote_terms(ordered)
    if not quoted:
        raise HistoryQueryError("查询为空或全部为标点，未产生可检索词项")
    return " AND ".join(quoted)


# ---------------------------------------------------------------------------
# 日期边界
# ---------------------------------------------------------------------------


def _date_only(text: str) -> bool:
    return len(text) == 10 and text[4] == "-" and text[7] == "-"


def normalize_date_bounds(
    date_from: str | None, date_to: str | None
) -> tuple[str | None, str | None]:
    """把日期/时间边界规整成能与 ``created_at`` 直接做**字符串**比较的上下界。

    ``messages.created_at`` 形如 ``2026-09-19T08:00:00Z``。调用方只给 ``YYYY-MM-DD`` 时：

    - **下界**直接用日期串即可（``"2026-09-19T08:00" >= "2026-09-19"``）→ 含当日。
    - **上界不能**直接用日期串：``"2026-09-19T08:00" <= "2026-09-19"`` 为假，会把当天的
      消息**全部排除**，而工具契约写明日期是"含当日"的 —— 这属于静默少给结果。改为追加
      ``"T~"`` 哨兵：当天任何时间戳都小于它，次日任何都大于它，且没有任何真实时间戳会
      等于它，因此配 ``<=`` 恰好表达"含当日整天"。
    """

    low = None if date_from is None else str(date_from).strip() or None
    if date_to is None:
        return low, None
    high = str(date_to).strip()
    if not high:
        return low, None
    if _date_only(high):
        high = f"{high}T~"
    return low, high


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HistoryHit:
    """一条历史命中：索引侧只给位置与分数，正文来自 ``messages``（原文）。"""

    message_id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    sequence: int
    score: float
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "score": self.score,
            "snippet": self.snippet,
        }


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """``read_history`` 用的一条消息。"""

    message_id: str
    conversation_id: str
    sequence: int
    role: str
    content: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "sequence": self.sequence,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


def _snippet(content: str, terms: Sequence[str]) -> str:
    """以命中词为中心截取原文摘要（在**原文**上截，不是在分词结果上）。"""

    if not content:
        return ""
    folded = content.casefold()
    position = -1
    for term in terms:
        found = folded.find(term.casefold())
        if found >= 0 and (position < 0 or found < position):
            position = found
    if position < 0:
        return content[:SNIPPET_WINDOW]
    start = max(0, position - SNIPPET_WINDOW // 2)
    end = min(len(content), position + SNIPPET_WINDOW)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


# ---------------------------------------------------------------------------
# 索引（写侧，经 AppStore 以便与业务写入共用连接/事务）
# ---------------------------------------------------------------------------


class WorkspaceHistoryIndex:
    """本区历史全文索引的写入侧（PRD 11.5）。"""

    def __init__(
        self,
        app_store: Any,
        *,
        workspace_id: str,
        paths: Any,
        terms: Sequence[str] = (),
    ) -> None:
        self._store = app_store
        self._workspace_id = workspace_id
        self._paths = paths
        self._terms = tuple(terms)

    @property
    def tokenizer(self) -> Any:
        return history_tokenizer(self._paths, terms=self._terms)

    @property
    def fingerprint(self) -> str:
        return history_index_fingerprint(terms_version_value=terms_version(self._terms))

    @property
    def marker_path(self) -> Path:
        return Path(self._paths.jieba_cache_dir) / HISTORY_INDEX_MARKER

    def read_marker(self) -> dict[str, Any]:
        path = self.marker_path
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def write_marker(self, *, indexed: int) -> None:
        path = self.marker_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "tokenizer_version": JIEBA_GENERATOR_VERSION,
                    "tokenizer_mode": JIEBA_TOKENIZER_MODE,
                    "dict_version": JIEBA_DICT_VERSION,
                    "index_version": HISTORY_INDEX_VERSION,
                    "terms_version": terms_version(self._terms),
                    "indexed": indexed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def needs_rebuild(self) -> bool:
        """分词器版本或术语变化 → 只重建相应文本索引（PRD 11.5）。"""

        return self.read_marker().get("fingerprint") != self.fingerprint

    async def index_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        content: str,
    ) -> None:
        await self._store.index_history_message(
            message_id=message_id,
            conversation_id=conversation_id,
            workspace_id=self._workspace_id,
            tokens=tokenize_for_index(content, self.tokenizer),
        )

    async def delete_message(self, message_id: str) -> None:
        await self._store.delete_history_message(message_id)

    async def prune_orphans(self) -> int:
        """清掉 ``messages`` 已不存在的索引行（会话删除走 CASCADE，索引行不会自动消失）。

        检索侧本来就用 ``JOIN messages`` 过滤，所以残留行不会泄漏到结果里；这里只是
        不让索引无限膨胀。
        """

        async with self._store.connection_scope() as conn:
            async with conn.execute(
                "DELETE FROM history_fts WHERE message_id NOT IN (SELECT id FROM messages)"
            ) as cursor:
                return int(cursor.rowcount or 0)

    async def rebuild(self) -> dict[str, int]:
        """按 ``messages`` 重建**本区**文本索引（原文是唯一权威源）。"""

        indexed = 0
        async with self._store.transaction() as conn:
            async with conn.execute(
                "DELETE FROM history_fts WHERE workspace_id = ?", (self._workspace_id,)
            ):
                pass
            conversations = await self._store.list_conversations(
                self._workspace_id, include_archived=True
            )
            for conversation in conversations:
                for message in await self._store.list_messages(conversation.id):
                    await self.index_message(
                        message_id=message.id,
                        conversation_id=conversation.id,
                        content=message.content,
                    )
                    indexed += 1
        self.write_marker(indexed=indexed)
        return {"indexed": indexed}

    async def ensure_fresh(self) -> dict[str, int]:
        """指纹变化（或从未建过）时重建本区索引；否则不做事。"""

        if not self.needs_rebuild() and not await self._has_index_difference():
            return {"indexed": 0, "rebuilt": 0}
        result = await self.rebuild()
        return {"indexed": result["indexed"], "rebuilt": 1}

    async def _has_index_difference(self) -> bool:
        async with self._store.connection_scope() as conn:
            return await self._has_index_difference_locked(conn)

    async def _has_index_difference_locked(self, conn: Any) -> bool:
        async with conn.execute(
            "SELECT 1 FROM messages m JOIN conversations c ON c.id = m.conversation_id"
            " WHERE c.workspace_id = ? AND NOT EXISTS"
            " (SELECT 1 FROM history_fts h WHERE h.message_id = m.id"
            " AND h.conversation_id = m.conversation_id AND h.workspace_id = ?) LIMIT 1",
            (self._workspace_id, self._workspace_id),
        ) as missing:
            missing_row = await missing.fetchone()
        if missing_row is not None:
            return True
        async with conn.execute(
            "SELECT 1 FROM history_fts h LEFT JOIN messages m ON m.id = h.message_id"
            " LEFT JOIN conversations c ON c.id = m.conversation_id"
            " WHERE h.workspace_id = ? AND"
            " (m.id IS NULL OR c.workspace_id != ? OR h.conversation_id != m.conversation_id)"
            " LIMIT 1",
            (self._workspace_id, self._workspace_id),
        ) as extra:
            extra_row = await extra.fetchone()
        if extra_row is not None:
            return True
        async with conn.execute(
            "SELECT"
            " (SELECT COUNT(*) FROM history_fts WHERE workspace_id = ?),"
            " (SELECT COUNT(*) FROM messages m JOIN conversations c"
            " ON c.id = m.conversation_id WHERE c.workspace_id = ?)",
            (self._workspace_id, self._workspace_id),
        ) as counts:
            indexed_count, message_count = await counts.fetchone()
        return indexed_count != message_count


# ---------------------------------------------------------------------------
# 检索（读侧，同步 sqlite3：工具调用是同步语义，且不与事件循环争同一连接）
# ---------------------------------------------------------------------------


class HistorySearch:
    """本区历史检索（PRD 11.5 / 13.2 ``GET /api/workspaces/{w}/history``）。"""

    def __init__(
        self,
        db_path: Path | str,
        *,
        workspace_id: str,
        paths: Any,
        terms: Sequence[str] = (),
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._workspace_id = workspace_id
        self._paths = paths
        self._terms = tuple(terms)
        self._conn = connection
        self._owns = connection is None
        self._lock = threading.RLock()

    # ---- 连接 ----

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                if not self._db_path.is_file():
                    raise HistoryIndexUnavailableError(f"app.db 不存在：{self._db_path}")
                self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._owns and self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- 分词 ----

    @property
    def tokenizer(self) -> Any:
        return history_tokenizer(self._paths, terms=self._terms)

    # ---- 查询 ----

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_HISTORY_LIMIT,
        offset: int = 0,
        date_from: str | None = None,
        date_to: str | None = None,
        exclude_message_id: str | None = None,
        distinct_content: bool = False,
    ) -> list[HistoryHit]:
        """按关键词检索本区历史消息，返回**原文**（PRD 11.5：查询结果返回原文）。

        ``date_from`` / ``date_to`` 既可以给 ``YYYY-MM-DD``（**含当日整天**）也可以给完整
        时间戳；日期串的边界处理见 :func:`normalize_date_bounds`。
        """

        expression = build_match_expression(query, self.tokenizer)
        capped = max(1, min(int(limit), MAX_HISTORY_LIMIT))
        low, high = normalize_date_bounds(date_from, date_to)
        sql = (
            "SELECT history_fts.message_id AS message_id,"
            " m.conversation_id AS conversation_id, m.role AS role,"
            " m.content AS content, m.created_at AS created_at,"
            " m.sequence AS sequence, bm25(history_fts) AS score"
            " FROM history_fts JOIN messages m ON m.id = history_fts.message_id"
            " WHERE history_fts MATCH ?"
            " AND history_fts.workspace_id = ?"
            " AND (? IS NULL OR m.created_at >= ?)"
            " AND (? IS NULL OR m.created_at <= ?)"
            " AND (? IS NULL OR m.id != ?)"
        )
        if distinct_content:
            sql = (
                "WITH matches AS MATERIALIZED (" + sql + "), ranked AS ("
                "SELECT *, ROW_NUMBER() OVER (PARTITION BY role, content, "
                "CASE WHEN role = 'user' AND sequence = 1 AND LENGTH(content) >= 20 THEN '' ELSE message_id END "
                "ORDER BY created_at, message_id) AS content_rank "
                "FROM matches) SELECT * FROM ranked WHERE content_rank = 1 "
                "ORDER BY score, created_at DESC, sequence LIMIT ? OFFSET ?"
            )
        else:
            sql += " ORDER BY score, m.created_at DESC, m.sequence LIMIT ? OFFSET ?"
        params = (
            expression,
            self._workspace_id,
            low,
            low,
            high,
            high,
            exclude_message_id,
            exclude_message_id,
            capped,
            max(0, int(offset)),
        )
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        terms = tokenize_terms(query, self.tokenizer)
        return [
            HistoryHit(
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                sequence=int(row["sequence"]),
                score=float(row["score"]),
                snippet=_snippet(str(row["content"]), terms),
            )
            for row in rows
        ]

    def count(
        self,
        query: str,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        exclude_message_id: str | None = None,
        distinct_content: bool = False,
    ) -> tuple[int, int]:
        """命中的**真实**条数与会话数 ``(total, conversation_count)``。

        比"结果已满一页就猜还有下一页"诚实得多，也让工具层可以如实回报 ``total`` 而不
        是给 ``null``。只做聚合，不回表取正文。
        """

        expression = build_match_expression(query, self.tokenizer)
        low, high = normalize_date_bounds(date_from, date_to)
        sql = (
            "SELECT COUNT(*) AS total,"
            " COUNT(DISTINCT m.conversation_id) AS conversations"
            " FROM history_fts JOIN messages m ON m.id = history_fts.message_id"
            " WHERE history_fts MATCH ?"
            " AND history_fts.workspace_id = ?"
            " AND (? IS NULL OR m.created_at >= ?)"
            " AND (? IS NULL OR m.created_at <= ?)"
            " AND (? IS NULL OR m.id != ?)"
        )
        params = (expression, self._workspace_id, low, low, high, high, exclude_message_id, exclude_message_id)
        if distinct_content:
            sql = (
                "WITH matches AS (SELECT m.conversation_id, m.role, m.content, "
                "ROW_NUMBER() OVER (PARTITION BY m.role, m.content, "
                "CASE WHEN m.role = 'user' AND m.sequence = 1 AND LENGTH(m.content) >= 20 THEN '' ELSE m.id END "
                "ORDER BY m.created_at, m.id) AS content_rank "
                "FROM history_fts JOIN messages m ON m.id = history_fts.message_id "
                "WHERE history_fts MATCH ? AND history_fts.workspace_id = ? "
                "AND (? IS NULL OR m.created_at >= ?) AND (? IS NULL OR m.created_at <= ?) "
                "AND (? IS NULL OR m.id != ?)) "
                "SELECT COUNT(*) AS total, COUNT(DISTINCT conversation_id) AS conversations "
                "FROM matches WHERE content_rank = 1"
            )
        with self._lock:
            row = self.connection.execute(sql, params).fetchone()
        return int(row["total"]), int(row["conversations"])

    def read_around(
        self,
        conversation_id: str,
        message_id: str,
        *,
        before: int = 3,
        after: int = 3,
    ) -> list[HistoryMessage]:
        """读取命中位置前后的历史消息（PRD 7.1 ``read_history``）。"""

        with self._lock:
            conn = self.connection
            anchor = conn.execute(
                "SELECT sequence FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchone()
            if anchor is None:
                return []
            center = int(anchor["sequence"])
            rows = conn.execute(
                "SELECT id, conversation_id, sequence, role, content, created_at FROM messages"
                " WHERE conversation_id = ? AND sequence BETWEEN ? AND ? ORDER BY sequence",
                (conversation_id, max(0, center - max(0, int(before))), center + max(0, int(after))),
            ).fetchall()
        return [
            HistoryMessage(
                message_id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                sequence=int(row["sequence"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def index_size(self) -> int:
        """本区在 ``history_fts`` 中的索引行数（诊断/报告用）。"""

        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM history_fts WHERE workspace_id = ?",
                (self._workspace_id,),
            ).fetchone()
        return int(row["n"])


def history_search_for(
    db_path: Path | str,
    *,
    workspace_id: str,
    paths: Any,
    terms: Sequence[str] = (),
) -> HistorySearch:
    """便捷构造：按区建立历史检索器（术语缺省即只用基础词典）。"""

    return HistorySearch(db_path, workspace_id=workspace_id, paths=paths, terms=tuple(terms))
