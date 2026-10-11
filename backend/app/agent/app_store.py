"""`app.db` 业务数据库：14 张表的建表与 await 访问层（PRD 13.1）。Owner: T00b。

职责与本层边界：

- 启用 ``PRAGMA foreign_keys=ON`` 与 WAL，启动时**幂等**建表
  （``CREATE TABLE IF NOT EXISTS`` + ``PRAGMA user_version``）。
- 14 张表逐条对齐 PRD 13.1：``workspaces`` / ``conversations`` / ``messages`` /
  ``runs`` / ``agent_tasks`` / ``provider_messages`` / ``operations`` /
  ``approvals`` / ``input_requests`` / ``run_events`` / ``answer_artifacts`` /
  ``background_jobs`` / ``history_fts``（FTS5，``unicode61``）/ ``migration_jobs``；
  四个唯一约束：``(run_id, event_seq)``、会话内 ``sequence``、``operation_id``、
  会话内 ``client_request_id``。
- 提供 await 接口：工作区/会话/消息/run/task/provider 消息/操作/批准/澄清/
  事件/答案/后台任务/迁移任务的读写。
- **单事务提交**（PRD 13.1 末段）：见 :meth:`AppStore.commit_answer`。
- **延迟外键**：``runs.input_message_id`` / ``messages.run_id`` /
  ``agent_tasks.parent_task_id`` 使用 ``DEFERRABLE INITIALLY DEFERRED``，
  使同一事务内的互指写入可行（PRD 13.2 建 run 路径），引用完整性仍在 ``COMMIT``
  时校验（见 :data:`_DEFERRED_FOREIGN_KEYS`）。
- 状态**迁移合法性不在此层强制**：由 RunManager 用
  :func:`app.agent.contracts.can_transition_run` 校验后调用本层；本层只保证
  数据与约束正确。
- ``history_fts`` 只提供分词文本的写入/删除与规模统计；查询词构造（jieba 分词、
  FTS5 引号规则）归 T08，避免两处实现分词语义。

T00b 交付 schema 与访问层；运行期使用归 approval-privacy（见架构文档 §2.1/§2.6），
需要新接口时向 contracts-db 提需求，不得就地改本文件的数据形状。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import aiosqlite
from pydantic import Field

from app.agent import events as agent_events
from app.agent.contracts import (
    APP_DB_SCHEMA_VERSION,
    DEFAULT_PERMISSION_MODE,
    DEFAULT_REASONING_LEVEL,
    AgentKind,
    AnswerArtifact,
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
    BackgroundJobStatus,
    ContractModel,
    ConversationStatus,
    EventType,
    InputRequest,
    InputRequestStatus,
    OperationRecord,
    OperationStatus,
    PermissionMode,
    ProviderMessage,
    ReasoningLevel,
    RunStatus,
    TaskStatus,
    WorkspaceStatus,
    utc_now_iso,
)

__all__ = [
    "JSON_COLUMNS",
    "SCHEMA_STATEMENTS",
    "TABLE_NAMES",
    "AgentTaskRecord",
    "AnswerCommitResult",
    "AppStore",
    "AppStoreError",
    "ApprovalConflictError",
    "BackgroundJobRecord",
    "BackgroundJobSpec",
    "ConflictError",
    "ConversationRecord",
    "InputRequestConflictError",
    "MessageRecord",
    "MigrationJobRecord",
    "RecordNotFoundError",
    "RunRecord",
    "WorkspaceRecord",
    "ensure_schema",
    "new_id",
]


# =====================================================================
# 表结构（PRD 13.1）
# =====================================================================

TABLE_NAMES: Final[tuple[str, ...]] = (
    "workspaces",
    "conversations",
    "messages",
    "runs",
    "agent_tasks",
    "provider_messages",
    "operations",
    "approvals",
    "input_requests",
    "run_events",
    "answer_artifacts",
    "background_jobs",
    "history_fts",
    "migration_jobs",
)

#: 延迟外键（``DEFERRABLE INITIALLY DEFERRED``）：同一事务内的插入顺序不受限，
#: 引用完整性在 ``COMMIT`` 时统一检查（悬空引用**仍然报错**，不会静默放过）。
#:
#: 为什么必须延迟（在 SQLite 3.53.1 上实测）：``runs.input_message_id`` 与
#: ``messages.run_id`` 构成互指对，而 PRD 13.2 要求「保存用户消息并创建任务」在
#: 同一次请求内完成、PRD 13.1 要求最终答案那一批写入单事务提交；立即外键会让
#: 两种插入顺序都被拒（``FOREIGN KEY constraint failed``）。
#: ``agent_tasks.parent_task_id`` 同理：子任务可能先于父任务写入同一事务。
#:
#: 改用例引用顺序不用改：延迟约束对「先父后子」的普通写法同样有效。
_DEFERRED_FOREIGN_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("runs", "input_message_id"),
    ("messages", "run_id"),
    ("agent_tasks", "parent_task_id"),
)

def _create_table(name: str, body: str, *, if_not_exists: bool = True) -> str:
    """由**单一**表体生成建表语句（v3 重建表时要复用同一份表体，避免两处 DDL 漂移）。"""

    guard = "IF NOT EXISTS " if if_not_exists else ""
    return f"CREATE TABLE {guard}{name} (\n{body}\n)"


_CREATE_WORKSPACES = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    default_permission_mode TEXT NOT NULL DEFAULT 'ask',
    default_reasoning_level TEXT NOT NULL DEFAULT 'off',
    knowledge_revision INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    order_index REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    parent_conversation_id TEXT REFERENCES conversations(id),
    branch_from_message_id TEXT,
    permission_mode TEXT NOT NULL DEFAULT 'ask',
    reasoning_level TEXT NOT NULL DEFAULT 'off',
    context_epoch INTEGER NOT NULL DEFAULT 1,
    memory_revision INTEGER NOT NULL DEFAULT 0,
    memory_invalidation_revision INTEGER NOT NULL DEFAULT 0,
    memory_snapshot_text TEXT NOT NULL DEFAULT '',
    memory_snapshot_hash TEXT NOT NULL DEFAULT '',
    memory_snapshot_at TEXT NOT NULL DEFAULT '',
    moved_from_workspace_id TEXT REFERENCES workspaces(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source_snapshot TEXT NOT NULL DEFAULT '{}',
    run_id TEXT REFERENCES runs(id) DEFERRABLE INITIALLY DEFERRED,
    created_at TEXT NOT NULL,
    UNIQUE (conversation_id, sequence)
)
"""

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id),
    origin_run_id TEXT REFERENCES runs(id),
    client_request_id TEXT,
    input_message_id TEXT REFERENCES messages(id) DEFERRABLE INITIALLY DEFERRED,
    status TEXT NOT NULL DEFAULT 'queued',
    permission_mode TEXT NOT NULL DEFAULT 'ask',
    reasoning_level TEXT NOT NULL DEFAULT 'off',
    context_epoch INTEGER NOT NULL DEFAULT 1,
    stop_reason TEXT,
    resumable INTEGER NOT NULL DEFAULT 0,
    answer_id TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (conversation_id, client_request_id)
)
"""

#: ``agent_tasks`` 表体（v3：**身份是 run 作用域** —— ``PRIMARY KEY (run_id, id)``）。
#:
#: 为什么不是 ``id TEXT PRIMARY KEY``（v2）：planner 在每个 run 里生成的 ``task_id`` 都是
#: ``t1``/``t2``/…（逐次相同），全局主键 ⇒ 同一会话的**第二个 run** 的 ``create_task``
#: 命中既有行被静默丢弃，且带 ``thread_id`` 时还会改写前一个 run 的行（PRD 12.2 / 5.4）。
#: ``parent_task_id`` 的自引用因此改为**复合**外键：父任务必须与子任务在同一 run 内，
#: 并保留 ``DEFERRABLE INITIALLY DEFERRED``（子任务可能先于父任务写入同一事务）。
_AGENT_TASKS_BODY = """    id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_task_id TEXT,
    agent_kind TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    requirement_ids TEXT NOT NULL DEFAULT '[]',
    depends_on TEXT NOT NULL DEFAULT '[]',
    expected_evidence TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    thread_id TEXT,
    agent_context_epoch INTEGER NOT NULL DEFAULT 1,
    result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id, parent_task_id)
        REFERENCES agent_tasks(run_id, id) DEFERRABLE INITIALLY DEFERRED
"""

_CREATE_AGENT_TASKS = _create_table("agent_tasks", _AGENT_TASKS_BODY)

_CREATE_PROVIDER_MESSAGES = """
CREATE TABLE IF NOT EXISTS provider_messages (
    id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    context_epoch INTEGER NOT NULL DEFAULT 1,
    agent_context_epoch INTEGER NOT NULL DEFAULT 1,
    sequence INTEGER NOT NULL,
    protocol TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    params TEXT NOT NULL DEFAULT '{}',
    execution_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (context_id, sequence)
)
"""

#: v3：``operations`` / ``approvals`` / ``input_requests`` 对任务的引用也改为**复合**外键
#: ``(run_id, task_id) → agent_tasks(run_id, id)``。单列 ``REFERENCES agent_tasks(id)`` 在
#: v3 下会变成 ``foreign key mismatch``（父表不再有 ``id`` 单列唯一约束），写入直接报错，
#: 因此这三张子表必须与父表一起重建。``operations.run_id`` 可为 NULL —— 复合外键在任一列
#: 为 NULL 时按 SQL 语义视为满足（不会误伤「尚不属于任何 run 的操作」）。
_OPERATIONS_BODY = """    operation_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id, task_id) REFERENCES agent_tasks(run_id, id)
"""

_CREATE_OPERATIONS = _create_table("operations", _OPERATIONS_BODY)

_APPROVALS_BODY = """    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT,
    interrupt_id TEXT NOT NULL DEFAULT '',
    operation_id TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    context_epoch INTEGER NOT NULL DEFAULT 1,
    rule_version TEXT NOT NULL DEFAULT '',
    display TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    decision TEXT,
    scope TEXT,
    superseded_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (run_id, task_id) REFERENCES agent_tasks(run_id, id)
"""

_CREATE_APPROVALS = _create_table("approvals", _APPROVALS_BODY)

_INPUT_REQUESTS_BODY = """    input_request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT,
    question TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '[]',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    answer TEXT,
    question_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (run_id, task_id) REFERENCES agent_tasks(run_id, id)
"""

_CREATE_INPUT_REQUESTS = _create_table("input_requests", _INPUT_REQUESTS_BODY)

_CREATE_RUN_EVENTS = """
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    event_seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    execution_id TEXT NOT NULL DEFAULT '',
    task_id TEXT,
    object_id TEXT,
    version INTEGER,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, event_seq)
)
"""

_CREATE_ANSWER_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS answer_artifacts (
    answer_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]',
    verification TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    answer_basis TEXT NOT NULL DEFAULT 'insufficient',
    basis_components TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TEXT NOT NULL
)
"""

_CREATE_BACKGROUND_JOBS = """
CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    workspace_id TEXT REFERENCES workspaces(id),
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    operation_id TEXT,
    source_message_range TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

#: FTS5 虚拟表（PRD 11.5）：分词内容与原文分离，原始消息留在 ``messages``。
_CREATE_HISTORY_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
    message_id UNINDEXED,
    conversation_id UNINDEXED,
    workspace_id UNINDEXED,
    tokens,
    tokenize = 'unicode61'
)
"""

_CREATE_MIGRATION_JOBS = """
CREATE TABLE IF NOT EXISTS migration_jobs (
    id TEXT PRIMARY KEY,
    migration_version TEXT NOT NULL,
    stage TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    validation TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_INDEX_STATEMENTS: Final[tuple[str, ...]] = (
    (
        "CREATE INDEX IF NOT EXISTS idx_conversations_workspace "
        "ON conversations(workspace_id, status)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_agent_tasks_run ON agent_tasks(run_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_agent_tasks_thread ON agent_tasks(thread_id)",
    (
        "CREATE INDEX IF NOT EXISTS idx_provider_messages_context "
        "ON provider_messages(context_id, sequence)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_operations_task ON operations(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_input_requests_run ON input_requests(run_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_events(run_id, type)",
    "CREATE INDEX IF NOT EXISTS idx_answer_artifacts_run ON answer_artifacts(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs(status, kind)",
    "CREATE INDEX IF NOT EXISTS idx_migration_jobs_version ON migration_jobs(migration_version)",
)

#: 建表语句全集（顺序即依赖顺序：被引用表在前）。
SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    _CREATE_WORKSPACES,
    _CREATE_CONVERSATIONS,
    _CREATE_RUNS,
    _CREATE_MESSAGES,
    _CREATE_AGENT_TASKS,
    _CREATE_PROVIDER_MESSAGES,
    _CREATE_OPERATIONS,
    _CREATE_APPROVALS,
    _CREATE_INPUT_REQUESTS,
    _CREATE_RUN_EVENTS,
    _CREATE_ANSWER_ARTIFACTS,
    _CREATE_BACKGROUND_JOBS,
    _CREATE_HISTORY_FTS,
    _CREATE_MIGRATION_JOBS,
    *_INDEX_STATEMENTS,
)


def new_id(prefix: str = "") -> str:
    """生成稳定格式的 ID（``<prefix>-<32 位十六进制>``）。"""

    token = uuid.uuid4().hex
    return f"{prefix}-{token}" if prefix else token


#: v2 增量列（T_d29）：老库（``user_version=1``）按列缺失判定补列，幂等且带默认值。
_WORKSPACES_COLUMN_UPGRADES: tuple[tuple[str, str], ...] = (
    ("pinned", "ALTER TABLE workspaces ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"),
    ("order_index", "ALTER TABLE workspaces ADD COLUMN order_index REAL"),
)


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {str(row[1]) for row in rows}


async def upgrade_schema(conn: aiosqlite.Connection) -> None:
    """把既有库就地升级到当前 schema 版本（幂等：连续调用无副作用）。

    只做**带默认值的加列**，不重写数据：老库升级后既有行的行为不变
    （``pinned=0`` / ``order_index=NULL`` 等价于「未置顶、无手工位次」）。
    """

    existing = await _table_columns(conn, "workspaces")
    for column, statement in _WORKSPACES_COLUMN_UPGRADES:
        if column not in existing:
            await conn.execute(statement)
    await _upgrade_task_identity(conn)
    await conn.execute(f"PRAGMA user_version = {APP_DB_SCHEMA_VERSION}")


#: v3 需要重建表的清单（表名 → 表体）：任务身份与三条引用任务的复合外键都要改，
#: 而 SQLite 不能 ``ALTER`` 主键/外键 ⇒ 只能建新表 → 拷贝 → 删旧 → 改名。
_RESTRUCTURED_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("agent_tasks", _AGENT_TASKS_BODY),
    ("operations", _OPERATIONS_BODY),
    ("approvals", _APPROVALS_BODY),
    ("input_requests", _INPUT_REQUESTS_BODY),
)

#: 判断某张表是否已经是 v3 形状的**唯一标记**：复合引用 ``REFERENCES agent_tasks(run_id, id)``
#: （四张表的 v3 表体里都含这一句，v2 表体里都不含）。
_TASK_IDENTITY_MARKER: Final[str] = "REFERENCES agent_tasks(run_id, id)"


async def _table_sql(conn: aiosqlite.Connection, name: str) -> str | None:
    row = await _fetch_one_static(
        conn, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    )
    if row is None:
        return None
    return str(row[0] or "")


async def _fetch_one_static(
    conn: aiosqlite.Connection, sql: str, params: Sequence[Any]
) -> Any:
    cursor = await conn.execute(sql, tuple(params))
    return await cursor.fetchone()


async def _rebuild_task_identity_table(
    conn: aiosqlite.Connection, name: str, body: str
) -> None:
    """单表就地重建（保留全部行；列顺序与列集合逐字不变）。"""

    temporary = f"{name}__v3"
    await conn.execute(f"DROP TABLE IF EXISTS {temporary}")
    await conn.execute(_create_table(temporary, body, if_not_exists=False))
    await conn.execute(f"INSERT INTO {temporary} SELECT * FROM {name}")
    await conn.execute(f"DROP TABLE {name}")
    await conn.execute(f"ALTER TABLE {temporary} RENAME TO {name}")


async def _upgrade_task_identity(conn: aiosqlite.Connection) -> int:
    """v2→v3：把任务身份改成 run 作用域（重建表 + 复合外键）。返回重建的表数。

    幂等性靠**结构探测**而不是只看 ``user_version``：只要表体里没有
    :data:`_TASK_IDENTITY_MARKER` 才重建，因此「连续两次 ``ensure_schema``」
    「升级中断后续跑」「全新库直接建 v3」三种路径都收敛到同一结果、且无副作用。
    既有行按原样拷贝（不重写数据）；外键在拷贝期间关闭，结束后重新打开并跑
    ``foreign_key_check`` 记录历史遗留问题（不因历史脏数据回滚升级）。
    """

    pending: list[tuple[str, str]] = []
    for name, body in _RESTRUCTURED_TABLES:
        defined = await _table_sql(conn, name)
        if defined is None or _TASK_IDENTITY_MARKER in defined:
            continue
        pending.append((name, body))
    if not pending:
        return 0

    await conn.execute("PRAGMA foreign_keys = OFF")
    try:
        await conn.execute("BEGIN")
        for name, body in pending:
            await _rebuild_task_identity_table(conn, name, body)
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        await conn.execute("PRAGMA foreign_keys = ON")
    # 表被删过，索引随之消失：这里按既有清单重建（幂等）。
    for statement in _INDEX_STATEMENTS:
        await conn.execute(statement)
    await conn.commit()
    return len(pending)


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """幂等建表 + 就地升级 + 启用外键 + 记录 schema 版本（可对任意连接重复调用）。"""

    await conn.execute("PRAGMA foreign_keys = ON")
    for statement in SCHEMA_STATEMENTS:
        await conn.execute(statement)
    await upgrade_schema(conn)
    await conn.commit()


# =====================================================================
# 异常（409 语义给 API 层直接映射）
# =====================================================================


class AppStoreError(Exception):
    """`app.db` 访问层基类异常。"""


class RecordNotFoundError(AppStoreError, LookupError):
    """请求的记录不存在（API 层映射 404）。"""


class ConflictError(AppStoreError):
    """状态冲突（API 层映射 **409**，PRD 10.2 / 13.2）。"""


class ApprovalConflictError(ConflictError):
    """批准决定冲突：参数/上下文已变化，或与既有决定不一致（PRD 10.2）。"""


class InputRequestConflictError(ConflictError):
    """澄清回答冲突：问题版本不匹配，或与既有回答不一致（PRD 13.2）。"""


# =====================================================================
# 记录模型（表行 → 强类型；跨工作流读到的是字段名而不是原始 dict）
# =====================================================================


class WorkspaceRecord(ContractModel):
    """``workspaces`` 行（PRD 13.1）。"""

    id: str
    storage_key: str
    name: str = ""
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    default_permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL
    knowledge_revision: int = 0
    #: 侧边栏置顶（PRD 5.2 交互；v2 起为 app.db 权威列，D29 收口）。
    pinned: bool = False
    #: 手工拖拽位次（PRD 5.2）；``None`` = 从未排过，按 ``created_at``。
    order_index: float | None = None
    created_at: str = ""
    updated_at: str = ""


class ConversationRecord(ContractModel):
    """``conversations`` 行（PRD 13.1：标题、归档与分支、会话权限与强度、快照、epoch）。"""

    id: str
    workspace_id: str
    title: str = ""
    status: ConversationStatus = ConversationStatus.ACTIVE
    parent_conversation_id: str | None = None
    branch_from_message_id: str | None = None
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL
    context_epoch: int = 1
    memory_revision: int = 0
    memory_invalidation_revision: int = 0
    memory_snapshot_text: str = ""
    memory_snapshot_hash: str = ""
    memory_snapshot_at: str = ""
    moved_from_workspace_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


class MessageRecord(ContractModel):
    """``messages`` 行（PRD 13.1）。"""

    id: str
    conversation_id: str
    sequence: int = 0
    role: str
    content: str
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    created_at: str = ""


class RunRecord(ContractModel):
    """``runs`` 行（PRD 13.1 / 12.1）。"""

    id: str
    conversation_id: str
    workspace_id: str
    origin_run_id: str | None = None
    client_request_id: str | None = None
    input_message_id: str | None = None
    status: RunStatus = RunStatus.QUEUED
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL
    context_epoch: int = 1
    stop_reason: str | None = None
    resumable: bool = False
    answer_id: str | None = None
    error_code: str | None = None
    created_at: str = ""
    updated_at: str = ""


class AgentTaskRecord(ContractModel):
    """``agent_tasks`` 行（PRD 13.1：父任务、职责、子问题、依赖、thread、结果）。"""

    id: str
    run_id: str
    parent_task_id: str | None = None
    agent_kind: AgentKind
    question: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    expected_evidence: str = ""
    status: TaskStatus = TaskStatus.QUEUED
    thread_id: str | None = None
    agent_context_epoch: int = 1
    result_ref: str | None = None
    created_at: str = ""
    updated_at: str = ""


class BackgroundJobRecord(ContractModel):
    """``background_jobs`` 行（PRD 13.1 / 11.5：记忆提炼等任务）。"""

    id: str
    kind: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None
    source_message_range: dict[str, Any] = Field(default_factory=dict)
    status: BackgroundJobStatus = BackgroundJobStatus.PENDING
    result: dict[str, Any] | None = None
    attempts: int = 0
    created_at: str = ""
    updated_at: str = ""


class MigrationJobRecord(ContractModel):
    """``migration_jobs`` 行（PRD 13.1 / 15.2）。"""

    id: str
    migration_version: str
    stage: str
    source_fingerprint: str = ""
    target: str = ""
    status: str = "pending"
    validation: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class BackgroundJobSpec(ContractModel):
    """创建后台任务的入参（``job_id`` 留空时由本层生成）。"""

    kind: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None
    source_message_range: dict[str, Any] = Field(default_factory=dict)
    job_id: str | None = None


class AnswerCommitResult(ContractModel):
    """:meth:`AppStore.commit_answer` 的返回值（答案 + 消息 + 事件 + 后台任务）。"""

    artifact: AnswerArtifact
    message: MessageRecord
    events: list[agent_events.StoredEvent] = Field(default_factory=list)
    job: BackgroundJobRecord | None = None


# =====================================================================
# 行 → 记录
# =====================================================================


def _row_to_dict(row: aiosqlite.Row | Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover - 数据库损坏时才可能发生
        return default


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class AppStore:
    """`app.db` 的异步访问层（一个连接一个实例；测试用 ``:memory:``）。"""

    def __init__(self, conn: aiosqlite.Connection, *, data_dir: Path | None = None) -> None:
        self._conn = conn
        self._in_transaction = False
        self._connection_lock = asyncio.Lock()
        self._connection_owner: asyncio.Task[Any] | None = None
        self._history_data_dir = data_dir
        self._history_index_factory: Any = None

    def configure_history_index(self, factory: Any) -> None:
        self._history_index_factory = factory

    async def _history_index_for(self, workspace_id: str) -> Any:
        if self._history_index_factory is not None:
            return self._history_index_factory(workspace_id)
        if self._history_data_dir is None:
            return None
        from app.core.workspace_context import _shared_paths
        from app.memory.history import WorkspaceHistoryIndex, load_terms_dictionary

        workspace = await self._require_workspace(workspace_id)
        paths = _shared_paths(self._history_data_dir, workspace.storage_key)
        return WorkspaceHistoryIndex(
            self, workspace_id=workspace_id, paths=paths,
            terms=load_terms_dictionary(paths),
        )

    # ---- 生命周期 ----

    @property
    def connection(self) -> aiosqlite.Connection:
        """底层连接（仅供需要自定义 SQL 的拥有者使用；数据形状不要绕过本层）。"""

        return self._conn

    @classmethod
    async def open(
        cls,
        db_path: str | Path | None = None,
        *,
        wal: bool | None = None,
    ) -> AppStore:
        """打开数据库并建表；``db_path=None`` 即内存模式（测试用）。

        ``wal`` 缺省：文件库启用 WAL，内存库不启用（``:memory:`` 不支持 WAL）。
        连接以 autocommit 打开，事务由 :meth:`transaction` 显式控制。
        """

        target = ":memory:" if db_path is None else str(db_path)
        conn = await aiosqlite.connect(target, isolation_level=None)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        if wal is None:
            wal = db_path is not None
        if wal:
            await conn.execute("PRAGMA journal_mode = WAL")
        store = cls(conn, data_dir=None if db_path is None else Path(db_path).parent)
        await store.ensure_schema()
        return store

    @classmethod
    async def open_memory(cls) -> AppStore:
        """内存模式（PRD 验收要求测试可注入，不触碰真实数据目录）。"""

        return await cls.open(None)

    async def ensure_schema(self) -> None:
        await ensure_schema(self._conn)

    async def close(self) -> None:
        async with self.connection_scope():
            if self._in_transaction:
                self._in_transaction = False
                await self._rollback_quietly()
            await self._conn.close()

    @asynccontextmanager
    async def connection_scope(self) -> AsyncIterator[aiosqlite.Connection]:
        """同一任务可重入；其他任务等待连接的当前操作或事务完成。"""
        task = asyncio.current_task()
        if task is not None and self._connection_owner is task:
            yield self._conn
            return
        async with self._connection_lock:
            self._connection_owner = task
            try:
                yield self._conn
            finally:
                self._connection_owner = None

    async def _rollback_quietly(self) -> None:
        """回滚；事务已经结束时 SQLite 会报错，这里静默忽略（调用方在抛原异常）。"""

        try:
            await self._conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - 仅在事务已自动结束时发生
            pass

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """显式事务：``BEGIN IMMEDIATE`` + 提交/回滚（嵌套调用复用外层事务）。

        延迟外键（:data:`_DEFERRED_FOREIGN_KEYS`）的引用完整性在 ``COMMIT`` 时检查，
        因此**提交失败时同样回滚**，不会留下半开事务。
        """

        async with self.connection_scope():
            if self._in_transaction:
                yield self._conn
                return
            try:
                await self._execute("BEGIN IMMEDIATE")
                self._in_transaction = True
                yield self._conn
                await self._execute("COMMIT")
            except BaseException:
                await self._rollback_quietly()
                raise
            finally:
                self._in_transaction = False

    # ---- 低层查询 ----

    async def _fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        async with self.connection_scope():
            async with self._conn.execute(sql, tuple(params)) as cursor:
                row = await cursor.fetchone()
        return None if row is None else _row_to_dict(row)

    async def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self.connection_scope():
            async with self._conn.execute(sql, tuple(params)) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        async with self.connection_scope():
            try:
                async with self._conn.execute(sql, tuple(params)):
                    pass
            except sqlite3.Error as exc:
                exc.add_note(
                    f"SQLite error_code={getattr(exc, 'sqlite_errorcode', None)} "
                    f"error_name={getattr(exc, 'sqlite_errorname', None)} "
                    f"in_transaction={self._conn.in_transaction}"
                )
                raise

    async def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self._fetch_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()))

    async def _require(self, sql: str, params: Sequence[Any], what: str) -> dict[str, Any]:
        row = await self._fetch_one(sql, params)
        if row is None:
            raise RecordNotFoundError(f"{what} 不存在: {params}")
        return row

    # =================================================================
    # workspaces（PRD 13.1 / 5.1）
    # =================================================================

    async def create_workspace(
        self,
        workspace_id: str,
        storage_key: str,
        *,
        name: str = "",
        status: WorkspaceStatus = WorkspaceStatus.ACTIVE,
        default_permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE,
        default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL,
        knowledge_revision: int = 0,
        pinned: bool = False,
        order_index: float | None = None,
    ) -> WorkspaceRecord:
        """创建工作区；``storage_key`` 违反唯一约束时抛 ``IntegrityError``。"""

        now = utc_now_iso()
        await self._execute(
            "INSERT INTO workspaces (id, storage_key, name, status, default_permission_mode,"
            " default_reasoning_level, knowledge_revision, pinned, order_index,"
            " created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                workspace_id,
                storage_key,
                name,
                str(status),
                str(default_permission_mode),
                str(default_reasoning_level),
                knowledge_revision,
                int(pinned),
                order_index,
                now,
                now,
            ),
        )
        return await self.get_workspace(workspace_id)

    async def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        row = await self._fetch_one("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
        return None if row is None else WorkspaceRecord.model_validate(row)

    async def get_workspace_by_storage_key(self, storage_key: str) -> WorkspaceRecord | None:
        row = await self._fetch_one(
            "SELECT * FROM workspaces WHERE storage_key = ?", (storage_key,)
        )
        return None if row is None else WorkspaceRecord.model_validate(row)

    async def list_workspaces(
        self, *, status: WorkspaceStatus | None = None
    ) -> list[WorkspaceRecord]:
        if status is None:
            rows = await self._fetch_all("SELECT * FROM workspaces ORDER BY created_at, id")
        else:
            rows = await self._fetch_all(
                "SELECT * FROM workspaces WHERE status = ? ORDER BY created_at, id", (str(status),)
            )
        return [WorkspaceRecord.model_validate(row) for row in rows]

    async def update_workspace(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        status: WorkspaceStatus | None = None,
        default_permission_mode: PermissionMode | None = None,
        default_reasoning_level: ReasoningLevel | None = None,
        pinned: bool | None = None,
        order_index: float | None = None,
    ) -> WorkspaceRecord:
        assignments: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("name", name),
            ("status", None if status is None else str(status)),
            (
                "default_permission_mode",
                None if default_permission_mode is None else str(default_permission_mode),
            ),
            (
                "default_reasoning_level",
                None if default_reasoning_level is None else str(default_reasoning_level),
            ),
            ("pinned", None if pinned is None else int(pinned)),
            ("order_index", order_index),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if assignments:
            assignments.append("updated_at = ?")
            params.extend([utc_now_iso(), workspace_id])
            await self._execute(
                f"UPDATE workspaces SET {', '.join(assignments)} WHERE id = ?", params
            )
        return await self._require_workspace(workspace_id)

    async def reconcile_default_reasoning_level(
        self, replacements: Mapping[str, ReasoningLevel]
    ) -> tuple[int, list[WorkspaceRecord]]:
        """把仍是**旧程序默认档位**的工作区批量改写为新默认档位（一次性对账用）。

        语义边界（见 :mod:`app.core.reasoning_defaults`）：只改写**精确等于**给定旧值的工作区，
        其它取值（``off`` / ``low`` / ``max``）一字不动。返回（改写条数, 改写后的权威行），
        调用方据此把存档（``workspaces.json``）镜像成权威值。

        这是 workspaces 表的**正规写路径**：D19/A3 要求任何模块都不得绕过本类直写该表，
        因此对账的 SQL 落在这里，而不是落在 ``core`` 的迁移代码里。
        """

        pending: list[tuple[str, ReasoningLevel]] = []
        for record in await self.list_workspaces():
            target = replacements.get(str(record.default_reasoning_level))
            if target is not None and str(target) != str(record.default_reasoning_level):
                pending.append((record.id, target))
        if not pending:
            return 0, []

        updated: list[WorkspaceRecord] = []
        for workspace_id, target in pending:
            updated.append(
                await self.update_workspace(
                    workspace_id, default_reasoning_level=target
                )
            )
        return len(updated), updated

    async def reorder_workspaces(self, ordered_ids: Sequence[str]) -> list[WorkspaceRecord]:
        """按给定顺序重写手工位次（只认已存在的 id，未列出的保持原位次）。

        语义与 ``core/workspace_store.WorkspaceStore.reorder`` **逐字一致**：只写
        ``order_index``，**不动** ``updated_at``（旧实现也不动），置顶与否由 ``pinned``
        单独表达。返回全部工作区行（排序交给 ``workspace_order_key`` 单一排序规则）。
        """

        for position, workspace_id in enumerate(ordered_ids):
            await self._execute(
                "UPDATE workspaces SET order_index = ? WHERE id = ?",
                (float(position), workspace_id),
            )
        return await self.list_workspaces()

    async def bump_knowledge_revision(self, workspace_id: str, *, delta: int = 1) -> int:
        """提交成功的导入/同步/删除后增加本区知识版本（PRD 5.3 第 6 条）。"""

        await self._execute(
            "UPDATE workspaces SET knowledge_revision = knowledge_revision + ?, updated_at = ?"
            " WHERE id = ?",
            (delta, utc_now_iso(), workspace_id),
        )
        record = await self._require_workspace(workspace_id)
        return record.knowledge_revision

    async def _require_workspace(self, workspace_id: str) -> WorkspaceRecord:
        row = await self._require(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,), "工作区"
        )
        return WorkspaceRecord.model_validate(row)

    # =================================================================
    # conversations（PRD 13.1 / 5.4 / 11.3）
    # =================================================================

    async def create_conversation(
        self,
        conversation_id: str,
        workspace_id: str,
        *,
        title: str = "",
        permission_mode: PermissionMode | None = None,
        reasoning_level: ReasoningLevel | None = None,
        parent_conversation_id: str | None = None,
        branch_from_message_id: str | None = None,
    ) -> ConversationRecord:
        """创建会话；权限与强度缺省时继承工作区默认值（PRD 18.1）。"""

        workspace = await self._require_workspace(workspace_id)
        now = utc_now_iso()
        await self._execute(
            "INSERT INTO conversations (id, workspace_id, title, status, parent_conversation_id,"
            " branch_from_message_id, permission_mode, reasoning_level, context_epoch,"
            " memory_revision, memory_invalidation_revision, memory_snapshot_text,"
            " memory_snapshot_hash, memory_snapshot_at, created_at, updated_at)"
            " VALUES (?, ?, ?, 'active', ?, ?, ?, ?, 1, 0, 0, '', '', '', ?, ?)",
            (
                conversation_id,
                workspace_id,
                title,
                parent_conversation_id,
                branch_from_message_id,
                str(permission_mode or workspace.default_permission_mode),
                str(reasoning_level or workspace.default_reasoning_level),
                now,
                now,
            ),
        )
        return await self._require_conversation(conversation_id)

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        row = await self._fetch_one(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        return None if row is None else ConversationRecord.model_validate(row)

    async def list_conversations(
        self,
        workspace_id: str,
        *,
        include_archived: bool = True,
        limit: int | None = None,
    ) -> list[ConversationRecord]:
        sql = "SELECT * FROM conversations WHERE workspace_id = ?"
        params: list[Any] = [workspace_id]
        if not include_archived:
            sql += " AND status = 'active'"
        sql += " ORDER BY created_at, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [ConversationRecord.model_validate(row) for row in rows]

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: ConversationStatus | None = None,
        permission_mode: PermissionMode | None = None,
        reasoning_level: ReasoningLevel | None = None,
        context_epoch: int | None = None,
    ) -> ConversationRecord:
        assignments: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("title", title),
            ("status", None if status is None else str(status)),
            ("permission_mode", None if permission_mode is None else str(permission_mode)),
            ("reasoning_level", None if reasoning_level is None else str(reasoning_level)),
            ("context_epoch", context_epoch),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if assignments:
            assignments.append("updated_at = ?")
            params.extend([utc_now_iso(), conversation_id])
            await self._execute(
                f"UPDATE conversations SET {', '.join(assignments)} WHERE id = ?", params
            )
        return await self._require_conversation(conversation_id)

    async def bump_context_epoch(self, conversation_id: str, *, delta: int = 1) -> int:
        """人工记忆失效或会话移动后增加会话上下文版本（PRD 11.4 / 5.4）。"""

        await self._execute(
            "UPDATE conversations SET context_epoch = context_epoch + ?, updated_at = ?"
            " WHERE id = ?",
            (delta, utc_now_iso(), conversation_id),
        )
        record = await self._require_conversation(conversation_id)
        return record.context_epoch

    async def save_memory_snapshot(
        self,
        conversation_id: str,
        *,
        memory_revision: int,
        memory_invalidation_revision: int,
        memory_snapshot_text: str,
        memory_snapshot_hash: str,
        memory_snapshot_at: str | None = None,
    ) -> ConversationRecord:
        """固定会话记忆快照（PRD 11.3）：快照只写一次，恢复会话不自动替换。"""

        await self._execute(
            "UPDATE conversations SET memory_revision = ?, memory_invalidation_revision = ?,"
            " memory_snapshot_text = ?, memory_snapshot_hash = ?, memory_snapshot_at = ?,"
            " updated_at = ? WHERE id = ?",
            (
                memory_revision,
                memory_invalidation_revision,
                memory_snapshot_text,
                memory_snapshot_hash,
                memory_snapshot_at or utc_now_iso(),
                utc_now_iso(),
                conversation_id,
            ),
        )
        return await self._require_conversation(conversation_id)

    async def clear_memory_snapshot(
        self, conversation_id: str, *, invalidation_revision: int
    ) -> ConversationRecord:
        """关闭记忆时快照置空并与失效版本同时提交（PRD 11.4）。"""

        await self._execute(
            "UPDATE conversations SET memory_snapshot_text = '', memory_snapshot_hash = '',"
            " memory_snapshot_at = '', memory_invalidation_revision = ?, updated_at = ?"
            " WHERE id = ?",
            (invalidation_revision, utc_now_iso(), conversation_id),
        )
        return await self._require_conversation(conversation_id)

    async def move_conversation(
        self,
        conversation_id: str,
        *,
        target_workspace_id: str,
        expected_context_epoch: int | None = None,
    ) -> ConversationRecord:
        """会话移动到目标工作区（PRD 5.4）：保留历史、增加 ``context_epoch``。

        ``expected_context_epoch`` 与当前值不一致时抛 :class:`ConflictError`
        （PRD 13.2 的乐观并发校验）。
        """

        current = await self._require_conversation(conversation_id)
        if (
            expected_context_epoch is not None
            and current.context_epoch != expected_context_epoch
        ):
            raise ConflictError(
                f"会话上下文版本已变化：期望 {expected_context_epoch}，实际 {current.context_epoch}"
            )
        target = await self._require_workspace(target_workspace_id)
        async with self.transaction():
            await self._execute(
                "UPDATE conversations SET workspace_id = ?, moved_from_workspace_id = ?,"
                " context_epoch = context_epoch + 1, memory_revision = 0,"
                " memory_invalidation_revision = 0, memory_snapshot_text = '',"
                " memory_snapshot_hash = '', memory_snapshot_at = '', updated_at = ? WHERE id = ?",
                (
                    target.id,
                    current.workspace_id,
                    utc_now_iso(),
                    conversation_id,
                ),
            )
            history_index = await self._history_index_for(target.id)
            for message in await self.list_messages(conversation_id):
                await self.delete_history_message(message.id)
                if history_index is not None:
                    await history_index.index_message(
                        message_id=message.id,
                        conversation_id=conversation_id,
                        content=message.content,
                    )
        return await self._require_conversation(conversation_id)

    async def _require_conversation(self, conversation_id: str) -> ConversationRecord:
        row = await self._require(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,), "会话"
        )
        return ConversationRecord.model_validate(row)

    # =================================================================
    # messages（PRD 13.1：会话内 sequence 唯一）
    # =================================================================

    async def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        source_snapshot: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> MessageRecord:
        """追加一条消息，``sequence`` 在事务内分配（会话内单调且唯一）。"""

        async with self.transaction():
            return await self._append_message_locked(
                conversation_id=conversation_id,
                role=role,
                content=content,
                source_snapshot=source_snapshot,
                run_id=run_id,
                message_id=message_id,
                created_at=created_at,
            )

    async def _append_message_locked(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        source_snapshot: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> MessageRecord:
        """写入消息（调用方必须已持有事务）。"""

        sequence = int(
            await self._scalar(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
        )
        record = MessageRecord(
            id=message_id or new_id("msg"),
            conversation_id=conversation_id,
            sequence=sequence,
            role=role,
            content=content,
            source_snapshot=dict(source_snapshot or {}),
            run_id=run_id,
            created_at=created_at or utc_now_iso(),
        )
        await self._execute(
            "INSERT INTO messages (id, conversation_id, sequence, role, content,"
            " source_snapshot, run_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.conversation_id,
                record.sequence,
                record.role,
                record.content,
                _dump(record.source_snapshot),
                record.run_id,
                record.created_at,
            ),
        )
        conversation = await self._require_conversation(conversation_id)
        history_index = await self._history_index_for(conversation.workspace_id)
        if history_index is not None:
            await history_index.index_message(
                message_id=record.id,
                conversation_id=conversation_id,
                content=content,
            )
        await self._execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (utc_now_iso(), conversation_id),
        )
        return record

    async def get_message(self, message_id: str) -> MessageRecord | None:
        row = await self._fetch_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return None if row is None else _message_from_row(row)

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> list[MessageRecord]:
        """按 ``sequence`` 升序读取；``after_sequence`` 用于断点续读。"""

        sql = "SELECT * FROM messages WHERE conversation_id = ? AND sequence > ? ORDER BY sequence"
        params: list[Any] = [conversation_id, after_sequence]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [_message_from_row(row) for row in rows]

    async def message_count(self, conversation_id: str) -> int:
        return int(
            await self._scalar(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
        )

    async def update_run_settings(
        self, run_id: str, permission_mode: PermissionMode, reasoning_level: ReasoningLevel
    ) -> None:
        await self._execute(
            "UPDATE runs SET permission_mode = ?, reasoning_level = ?, updated_at = ? WHERE id = ?",
            (str(permission_mode), str(reasoning_level), utc_now_iso(), run_id),
        )

    async def truncate_messages(self, conversation_id: str, index: int) -> None:
        if index < 0:
            raise ValueError("消息位置必须大于或等于零")
        if await self.list_active_runs(conversation_id):
            raise ConflictError("请停止当前任务后编辑消息")
        messages = await self.list_messages(conversation_id)
        if index >= len(messages):
            return
        sequence = messages[index].sequence
        async with self.transaction():
            await self._execute(
                "UPDATE runs SET input_message_id = NULL, resumable = 0"
                " WHERE input_message_id IN (SELECT id FROM messages"
                " WHERE conversation_id = ? AND sequence >= ?)",
                (conversation_id, sequence),
            )
            await self._execute(
                "DELETE FROM history_fts WHERE conversation_id = ? AND message_id IN"
                " (SELECT id FROM messages WHERE conversation_id = ? AND sequence >= ?)",
                (conversation_id, conversation_id, sequence),
            )
            await self._execute(
                "DELETE FROM messages WHERE conversation_id = ? AND sequence >= ?",
                (conversation_id, sequence),
            )
            await self.bump_context_epoch(conversation_id)

    # =================================================================
    # runs（PRD 13.1 / 12.1 / 13.2）
    # =================================================================

    async def create_run(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        client_request_id: str | None = None,
        input_message_id: str | None = None,
        permission_mode: PermissionMode | None = None,
        reasoning_level: ReasoningLevel | None = None,
        context_epoch: int | None = None,
        origin_run_id: str | None = None,
        run_id: str | None = None,
        status: RunStatus = RunStatus.QUEUED,
    ) -> RunRecord:
        """创建 run；``client_request_id`` 重复时返回既有 run（PRD 13.2 幂等）。"""

        if client_request_id is not None:
            existing = await self.find_run_by_client_request(conversation_id, client_request_id)
            if existing is not None:
                return existing
        conversation = await self._require_conversation(conversation_id)
        if conversation.workspace_id != workspace_id:
            raise ConflictError(
                f"run 的工作区必须与会话一致：会话={conversation.workspace_id}，请求={workspace_id}"
            )
        now = utc_now_iso()
        record = RunRecord(
            id=run_id or new_id("run"),
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            origin_run_id=origin_run_id,
            client_request_id=client_request_id,
            input_message_id=input_message_id,
            status=status,
            permission_mode=permission_mode or conversation.permission_mode,
            reasoning_level=reasoning_level or conversation.reasoning_level,
            context_epoch=(
                conversation.context_epoch if context_epoch is None else context_epoch
            ),
            created_at=now,
            updated_at=now,
        )
        try:
            await self._execute(
                "INSERT INTO runs (id, conversation_id, workspace_id, origin_run_id,"
                " client_request_id, input_message_id, status, permission_mode,"
                " reasoning_level, context_epoch, resumable, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    record.id,
                    record.conversation_id,
                    record.workspace_id,
                    record.origin_run_id,
                    record.client_request_id,
                    record.input_message_id,
                    str(record.status),
                    str(record.permission_mode),
                    str(record.reasoning_level),
                    record.context_epoch,
                    record.created_at,
                    record.updated_at,
                ),
            )
        except Exception:
            # 并发同 client_request_id：唯一约束落败方回读既有 run（仍然幂等）
            if client_request_id is not None:
                existing = await self.find_run_by_client_request(
                    conversation_id, client_request_id
                )
                if existing is not None:
                    return existing
            raise
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        row = await self._fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        return None if row is None else RunRecord.model_validate(row)

    async def find_run_by_client_request(
        self, conversation_id: str, client_request_id: str
    ) -> RunRecord | None:
        row = await self._fetch_one(
            "SELECT * FROM runs WHERE conversation_id = ? AND client_request_id = ?",
            (conversation_id, client_request_id),
        )
        return None if row is None else RunRecord.model_validate(row)

    async def list_runs(
        self,
        conversation_id: str,
        *,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        sql = "SELECT * FROM runs WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY created_at, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [RunRecord.model_validate(row) for row in rows]

    async def list_active_runs(self, conversation_id: str) -> list[RunRecord]:
        """活动 run（同一会话同时只允许一个活动主任务，PRD 12.2）。

        调用方拿到长度 > 1 时应报告数据异常；本层只负责查询。
        """

        rows = await self._fetch_all(
            "SELECT * FROM runs WHERE conversation_id = ? AND status IN"
            " ('queued','running','waiting_children','waiting_approval','waiting_input',"
            "'cancelling','interrupted') ORDER BY created_at",
            (conversation_id,),
        )
        return [RunRecord.model_validate(row) for row in rows]

    async def list_restartable_runs(self) -> list[RunRecord]:
        """读取服务启动时需要继续执行的 run。"""

        rows = await self._fetch_all(
            "SELECT * FROM runs WHERE status IN"
            " ('queued','running','waiting_children') ORDER BY created_at, id"
        )
        return [RunRecord.model_validate(row) for row in rows]

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        stop_reason: str | None = None,
        resumable: bool | None = None,
        answer_id: str | None = None,
        error_code: str | None = None,
    ) -> RunRecord:
        """写入 run 状态；迁移合法性由 RunManager 校验（见模块 docstring）。"""

        async with self.transaction():
            return await self._update_run_locked(
                run_id,
                status,
                stop_reason=stop_reason,
                resumable=resumable,
                answer_id=answer_id,
                error_code=error_code,
            )

    async def _update_run_locked(
        self,
        run_id: str,
        status: RunStatus,
        *,
        stop_reason: str | None = None,
        resumable: bool | None = None,
        answer_id: str | None = None,
        error_code: str | None = None,
    ) -> RunRecord:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [str(status), utc_now_iso()]
        if status in {RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}:
            if stop_reason is None:
                assignments.append("stop_reason = NULL")
            if error_code is None:
                assignments.append("error_code = NULL")
        for column, value in (
            ("stop_reason", stop_reason),
            ("resumable", None if resumable is None else int(resumable)),
            ("answer_id", answer_id),
            ("error_code", error_code),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.append(run_id)
        await self._execute(f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", params)
        row = await self._require("SELECT * FROM runs WHERE id = ?", (run_id,), "run")
        return RunRecord.model_validate(row)

    # =================================================================
    # agent_tasks（PRD 12.2 / 6.4）
    # =================================================================

    async def create_task(
        self,
        *,
        run_id: str,
        task_id: str,
        agent_kind: AgentKind | str,
        question: str = "",
        parent_task_id: str | None = None,
        requirement_ids: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        expected_evidence: str = "",
        status: TaskStatus = TaskStatus.QUEUED,
        thread_id: str | None = None,
        agent_context_epoch: int = 1,
    ) -> AgentTaskRecord:
        """创建子任务；**同一 run 内**重复 ``task_id`` 时复用既有行（PRD 12.2）。

        身份是 run 作用域（v3）：不同 run 的 ``t1`` 是两条独立记录，互不覆盖也互不丢失。
        """

        existing = await self.get_task(run_id, task_id)
        if existing is not None:
            return existing
        now = utc_now_iso()
        record = AgentTaskRecord(
            id=task_id,
            run_id=run_id,
            parent_task_id=parent_task_id,
            agent_kind=AgentKind(str(agent_kind)),
            question=question,
            requirement_ids=list(requirement_ids),
            depends_on=list(depends_on),
            expected_evidence=expected_evidence,
            status=status,
            thread_id=thread_id,
            agent_context_epoch=agent_context_epoch,
            created_at=now,
            updated_at=now,
        )
        await self._execute(
            "INSERT INTO agent_tasks (id, run_id, parent_task_id, agent_kind, question,"
            " requirement_ids, depends_on, expected_evidence, status, thread_id,"
            " agent_context_epoch, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.run_id,
                record.parent_task_id,
                str(record.agent_kind),
                record.question,
                _dump(record.requirement_ids),
                _dump(record.depends_on),
                record.expected_evidence,
                str(record.status),
                record.thread_id,
                record.agent_context_epoch,
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    async def get_task(self, run_id: str, task_id: str) -> AgentTaskRecord | None:
        """按 **(`run_id`, `task_id`) 复合身份**取子任务（v3）。

        刻意**不提供**「只按 ``task_id`` 全局查」的入口：那正是跨 run 撞 id 的成因；
        任何查任务的路径都必须带 run 上下文（``run_id`` 是必填位置参数，漏传即 TypeError）。
        """

        row = await self._fetch_one(
            "SELECT * FROM agent_tasks WHERE run_id = ? AND id = ?", (run_id, task_id)
        )
        return None if row is None else _task_from_row(row)

    async def list_tasks(
        self, run_id: str, *, status: TaskStatus | None = None
    ) -> list[AgentTaskRecord]:
        sql = "SELECT * FROM agent_tasks WHERE run_id = ?"
        params: list[Any] = [run_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY created_at, id"
        rows = await self._fetch_all(sql, params)
        return [_task_from_row(row) for row in rows]

    async def update_task_status(
        self,
        run_id: str,
        task_id: str,
        status: TaskStatus,
        *,
        result_ref: str | None = None,
        thread_id: str | None = None,
        agent_context_epoch: int | None = None,
    ) -> AgentTaskRecord:
        """更新**本 run 内**该子任务的状态（v3：run 上下文必填）。"""

        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [str(status), utc_now_iso()]
        for column, value in (
            ("result_ref", result_ref),
            ("thread_id", thread_id),
            ("agent_context_epoch", agent_context_epoch),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.extend([run_id, task_id])
        await self._execute(
            f"UPDATE agent_tasks SET {', '.join(assignments)}"
            " WHERE run_id = ? AND id = ?",
            params,
        )
        row = await self._require(
            "SELECT * FROM agent_tasks WHERE run_id = ? AND id = ?",
            (run_id, task_id),
            "子任务",
        )
        return _task_from_row(row)

    # =================================================================
    # provider_messages（PRD 13.1 / 9.2）
    # =================================================================

    async def append_provider_message(self, record: ProviderMessage) -> ProviderMessage:
        """写入一条协议消息；``(context_id, sequence)`` 冲突时抛 ``IntegrityError``。"""

        created_at = record.created_at or utc_now_iso()
        await self._execute(
            "INSERT INTO provider_messages (id, context_id, context_epoch,"
            " agent_context_epoch, sequence, protocol, model, params, execution_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("pmsg"),
                record.context_id,
                record.context_epoch,
                record.agent_context_epoch,
                record.sequence,
                _dump(record.protocol),
                record.model,
                _dump(record.params),
                record.execution_id,
                created_at,
            ),
        )
        return record.model_copy(update={"created_at": created_at})

    async def list_provider_messages(
        self, context_id: str, *, after_sequence: int = 0, limit: int | None = None
    ) -> list[ProviderMessage]:
        sql = (
            "SELECT * FROM provider_messages WHERE context_id = ? AND sequence > ?"
            " ORDER BY sequence"
        )
        params: list[Any] = [context_id, after_sequence]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [
            ProviderMessage(
                context_id=row["context_id"],
                context_epoch=row["context_epoch"],
                agent_context_epoch=row["agent_context_epoch"],
                sequence=row["sequence"],
                protocol=_loads(row["protocol"], {}),
                model=row["model"],
                params=_loads(row["params"], {}),
                execution_id=row["execution_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def next_provider_sequence(self, context_id: str) -> int:
        return int(
            await self._scalar(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM provider_messages"
                " WHERE context_id = ?",
                (context_id,),
            )
        )

    # =================================================================
    # operations（PRD 12.1 / 12.4：operation_id 唯一，重放复用）
    # =================================================================

    async def register_operation(
        self,
        *,
        operation_id: str,
        tool_name: str,
        args_hash: str,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> OperationRecord:
        """登记操作；``operation_id`` 已存在时**原样返回**（不增加尝试次数）。"""

        existing = await self.get_operation(operation_id)
        if existing is not None:
            return existing
        now = utc_now_iso()
        record = OperationRecord(
            operation_id=operation_id,
            run_id=run_id,
            task_id=task_id,
            tool_name=tool_name,
            args_hash=args_hash,
            status=OperationStatus.PENDING,
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        await self._execute(
            "INSERT INTO operations (operation_id, run_id, task_id, tool_name, args_hash,"
            " status, attempts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                record.operation_id,
                record.run_id,
                record.task_id,
                record.tool_name,
                record.args_hash,
                str(record.status),
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    async def get_operation(self, operation_id: str) -> OperationRecord | None:
        row = await self._fetch_one(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        )
        return None if row is None else _operation_from_row(row)

    async def list_operations(self, run_id: str) -> list[OperationRecord]:
        """按创建顺序读取一次 run 的工具操作。"""

        rows = await self._fetch_all(
            "SELECT * FROM operations WHERE run_id = ? ORDER BY created_at, operation_id",
            (run_id,),
        )
        return [_operation_from_row(row) for row in rows]

    async def start_operation_attempt(self, operation_id: str) -> OperationRecord:
        """开始一次尝试：``attempts += 1``，状态置 ``running``（PRD 12.4）。"""

        await self._execute(
            "UPDATE operations SET status = 'running', attempts = attempts + 1,"
            " updated_at = ? WHERE operation_id = ?",
            (utc_now_iso(), operation_id),
        )
        row = await self._require(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,), "操作"
        )
        return _operation_from_row(row)

    async def finish_operation(
        self,
        operation_id: str,
        status: OperationStatus,
        *,
        result_ref: str | None = None,
    ) -> OperationRecord:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [str(status), utc_now_iso()]
        if result_ref is not None:
            assignments.append("result_ref = ?")
            params.append(result_ref)
        params.append(operation_id)
        await self._execute(
            f"UPDATE operations SET {', '.join(assignments)} WHERE operation_id = ?", params
        )
        row = await self._require(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,), "操作"
        )
        return _operation_from_row(row)

    # =================================================================
    # approvals（PRD 10.2 / 10.3）
    # =================================================================

    async def request_approval(self, request: ApprovalRequest) -> ApprovalRequest:
        """保存批准请求（在前置节点调用，PRD 10.3）。

        同一个 ``approval_id`` 已存在时返回既有对象（幂等）；同一 ``operation_id``
        已有未裁决批准时复用，避免恢复后重复弹卡（PRD 10.3）。
        """

        existing = await self.get_approval(request.approval_id)
        if existing is not None:
            return existing
        pending = await self._fetch_one(
            "SELECT * FROM approvals WHERE operation_id = ? AND status = 'pending'",
            (request.operation_id,),
        )
        if pending is not None:
            return _approval_from_row(pending)
        record = request
        if not record.created_at:
            record = record.model_copy(update={"created_at": utc_now_iso()})
        await self._execute(
            "INSERT INTO approvals (approval_id, run_id, task_id, interrupt_id, operation_id,"
            " args_hash, context_epoch, rule_version, display, status, decision, scope,"
            " superseded_reason, created_at, resolved_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.approval_id,
                record.run_id,
                record.task_id,
                record.interrupt_id,
                record.operation_id,
                record.args_hash,
                record.context_epoch,
                record.rule_version,
                _dump(record.display.model_dump(mode="json")),
                str(record.status),
                None
                if record.decision is None
                else _dump(record.decision.model_dump(mode="json")),
                None if record.scope is None else _dump(record.scope.model_dump(mode="json")),
                record.superseded_reason,
                record.created_at,
                record.resolved_at,
            ),
        )
        return record

    async def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        row = await self._fetch_one(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
        )
        return None if row is None else _approval_from_row(row)

    async def list_approvals(
        self, run_id: str, *, status: ApprovalStatus | None = None
    ) -> list[ApprovalRequest]:
        sql = "SELECT * FROM approvals WHERE run_id = ?"
        params: list[Any] = [run_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY created_at"
        rows = await self._fetch_all(sql, params)
        return [_approval_from_row(row) for row in rows]

    async def resolve_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        args_hash: str | None = None,
        context_epoch: int | None = None,
        rule_version: str | None = None,
        scope: ApprovalScope | None = None,
    ) -> ApprovalRequest:
        """提交用户决定（PRD 10.2）。

        - 已经是相同决定的终态 → **幂等返回现有结果**；
        - 与既有决定冲突 → :class:`ApprovalConflictError`（API 层 409）；
        - ``args_hash`` / ``context_epoch`` / ``rule_version`` 与保存值不一致
          → :class:`ApprovalConflictError`（过期决定，验收 A15）。
        """

        async with self.transaction():
            row = await self._require(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,), "批准"
            )
            current = _approval_from_row(row)

            if current.status is not ApprovalStatus.PENDING:
                if _same_decision(current, decision):
                    return current
                raise ApprovalConflictError(
                    f"批准 {approval_id} 已裁决为 {current.status.value}，不能改为"
                    f" {decision.kind.value}"
                )
            if args_hash is not None and args_hash != current.args_hash:
                raise ApprovalConflictError(
                    f"批准 {approval_id} 的参数已变化（args_hash 不一致），决定过期"
                )
            if context_epoch is not None and context_epoch != current.context_epoch:
                raise ApprovalConflictError(
                    f"批准 {approval_id} 的上下文版本已变化（{current.context_epoch} →"
                    f" {context_epoch}），决定过期"
                )
            if rule_version is not None and rule_version != current.rule_version:
                raise ApprovalConflictError(
                    f"批准 {approval_id} 的规则版本已变化，决定过期"
                )
            if decision.kind is ApprovalDecisionKind.ALLOW_SCOPE and scope is None:
                raise ApprovalConflictError(
                    "「允许本轮同类操作」必须给出授权范围（PRD 10.2）"
                )

            status = (
                ApprovalStatus.DENIED
                if decision.kind is ApprovalDecisionKind.DENY
                else ApprovalStatus.APPROVED
            )
            resolved = decision.model_copy(
                update={"decided_at": decision.decided_at or utc_now_iso()}
            )
            await self._execute(
                "UPDATE approvals SET status = ?, decision = ?, scope = ?, resolved_at = ?"
                " WHERE approval_id = ? AND status = 'pending'",
                (
                    str(status),
                    _dump(resolved.model_dump(mode="json")),
                    None if scope is None else _dump(scope.model_dump(mode="json")),
                    resolved.decided_at,
                    approval_id,
                ),
            )
            return await self._lock_require_approval(approval_id)

    async def supersede_pending_approvals(self, run_id: str, *, reason: str) -> int:
        """把 run 的未裁决批准转为 ``superseded``（PRD 5.4 会话移动 / 11.4 记忆失效）。"""

        rows = await self._fetch_all(
            "SELECT approval_id FROM approvals WHERE run_id = ? AND status = 'pending'",
            (run_id,),
        )
        for row in rows:
            await self._execute(
                "UPDATE approvals SET status = 'superseded', superseded_reason = ?,"
                " resolved_at = ? WHERE approval_id = ? AND status = 'pending'",
                (reason, utc_now_iso(), row["approval_id"]),
            )
        return len(rows)

    async def _lock_require_approval(self, approval_id: str) -> ApprovalRequest:
        row = await self._require(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,), "批准"
        )
        return _approval_from_row(row)

    # =================================================================
    # input_requests（PRD 13.2 / 12.3 waiting_input）
    # =================================================================

    async def request_input(self, request: InputRequest) -> InputRequest:
        """保存澄清请求；同一 ``input_request_id`` 幂等返回既有对象。"""

        existing = await self.get_input_request(request.input_request_id)
        if existing is not None:
            return existing
        record = request
        if not record.created_at:
            record = record.model_copy(update={"created_at": utc_now_iso()})
        await self._execute(
            "INSERT INTO input_requests (input_request_id, run_id, task_id, question, options,"
            " reason, status, answer, question_version, created_at, resolved_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.input_request_id,
                record.run_id,
                record.task_id,
                record.question,
                _dump(record.options),
                record.reason,
                str(record.status),
                record.answer,
                record.question_version,
                record.created_at,
                record.resolved_at,
            ),
        )
        return record

    async def get_input_request(self, input_request_id: str) -> InputRequest | None:
        row = await self._fetch_one(
            "SELECT * FROM input_requests WHERE input_request_id = ?", (input_request_id,)
        )
        return None if row is None else _input_from_row(row)

    async def list_input_requests(
        self, run_id: str, *, status: InputRequestStatus | None = None
    ) -> list[InputRequest]:
        sql = "SELECT * FROM input_requests WHERE run_id = ?"
        params: list[Any] = [run_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(str(status))
        sql += " ORDER BY created_at"
        rows = await self._fetch_all(sql, params)
        return [_input_from_row(row) for row in rows]

    async def resolve_input(
        self,
        input_request_id: str,
        *,
        answer: str,
        question_version: int | None = None,
    ) -> InputRequest:
        """提交澄清回答（PRD 13.2）。

        重复提交**相同回答**幂等返回；不同回答、版本不符或已失效 → 409 语义异常。
        """

        async with self.transaction():
            row = await self._require(
                "SELECT * FROM input_requests WHERE input_request_id = ?",
                (input_request_id,),
                "澄清请求",
            )
            current = _input_from_row(row)

            if question_version is not None and question_version != current.question_version:
                raise InputRequestConflictError(
                    f"澄清问题版本已变化：期望 {question_version}，实际"
                    f" {current.question_version}"
                )
            if current.status is InputRequestStatus.ANSWERED:
                if current.answer == answer:
                    return current
                raise InputRequestConflictError(
                    f"澄清 {input_request_id} 已回答为其他内容，不能覆盖"
                )
            if current.status is not InputRequestStatus.PENDING:
                raise InputRequestConflictError(
                    f"澄清 {input_request_id} 已处于 {current.status.value}，不能回答"
                )
            await self._execute(
                "UPDATE input_requests SET status = 'answered', answer = ?, resolved_at = ?"
                " WHERE input_request_id = ? AND status = 'pending'",
                (answer, utc_now_iso(), input_request_id),
            )
            updated = await self._fetch_one(
                "SELECT * FROM input_requests WHERE input_request_id = ?", (input_request_id,)
            )
            assert updated is not None
            return _input_from_row(updated)

    async def supersede_pending_inputs(self, run_id: str) -> int:
        """把 run 的待澄清请求转为 ``superseded``（PRD 5.4）。"""

        rows = await self._fetch_all(
            "SELECT input_request_id FROM input_requests WHERE run_id = ? AND status = 'pending'",
            (run_id,),
        )
        for row in rows:
            await self._execute(
                "UPDATE input_requests SET status = 'superseded', resolved_at = ?"
                " WHERE input_request_id = ? AND status = 'pending'",
                (utc_now_iso(), row["input_request_id"]),
            )
        return len(rows)

    # =================================================================
    # run_events（PRD 13.1 / 13.3：单调编号 + after=seq 补读）
    # =================================================================

    async def append_event(
        self,
        *,
        run_id: str,
        type: EventType | str,
        payload: Mapping[str, Any],
        execution_id: str = "",
        task_id: str | None = None,
        object_id: str | None = None,
        version: int | None = None,
        created_at: str | None = None,
    ) -> agent_events.StoredEvent:
        """追加事件：先校验 payload，再分配单调 ``event_seq`` 并写入。

        payload 校验失败抛 :class:`app.agent.events.EventProtocolError`，事件
        **不会**入库（PRD 12.5：不合法就地失败，不留半成品）。
        """

        async with self.transaction():
            return await self._append_event_locked(
                run_id=run_id,
                type=type,
                payload=payload,
                execution_id=execution_id,
                task_id=task_id,
                object_id=object_id,
                version=version,
                created_at=created_at,
            )

    async def _append_event_locked(
        self,
        *,
        run_id: str,
        type: EventType | str,
        payload: Mapping[str, Any],
        execution_id: str = "",
        task_id: str | None = None,
        object_id: str | None = None,
        version: int | None = None,
        created_at: str | None = None,
    ) -> agent_events.StoredEvent:
        """写入事件（调用方必须已持有事务；编号在同一事务内分配）。"""

        seq = await self._next_event_seq_locked(run_id)
        event = agent_events.build_event(
            run_id=run_id,
            event_seq=seq,
            type=type,
            payload=payload,
            execution_id=execution_id,
            task_id=task_id,
            created_at=created_at or utc_now_iso(),
            object_id=object_id,
            version=version,
        )
        await self._execute(
            "INSERT INTO run_events (run_id, event_seq, type, execution_id, task_id, object_id,"
            " version, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.run_id,
                event.event_seq,
                event.type.value,
                event.execution_id,
                event.task_id,
                event.object_id,
                event.version,
                _dump(event.payload),
                event.created_at,
            ),
        )
        return event

    async def _next_event_seq_locked(self, run_id: str) -> int:
        return int(
            await self._scalar(
                "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            )
        )

    async def last_event_seq(self, run_id: str) -> int:
        """该 run 当前最大事件编号（0 表示还没有事件）；刷新界面时取它（PRD 13.2）。"""

        return int(
            await self._scalar(
                "SELECT COALESCE(MAX(event_seq), 0) FROM run_events WHERE run_id = ?",
                (run_id,),
            )
        )

    async def read_events(
        self, run_id: str, *, after_seq: int = 0, limit: int | None = None
    ) -> list[agent_events.StoredEvent]:
        """按编号升序读取**严格大于** ``after_seq`` 的事件（``Last-Event-ID`` 补读）。"""

        sql = "SELECT * FROM run_events WHERE run_id = ? AND event_seq > ? ORDER BY event_seq"
        params: list[Any] = [run_id, after_seq]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [_event_from_row(row) for row in rows]

    # =================================================================
    # answer_artifacts + 单事务提交（PRD 13.1 末段）
    # =================================================================

    async def write_answer_artifact(self, artifact: AnswerArtifact) -> AnswerArtifact:
        async with self.transaction():
            return await self._insert_answer_artifact_locked(artifact)

    async def _insert_answer_artifact_locked(self, artifact: AnswerArtifact) -> AnswerArtifact:
        record = artifact
        if not record.created_at:
            record = record.model_copy(update={"created_at": utc_now_iso()})
        await self._execute(
            "INSERT INTO answer_artifacts (answer_id, run_id, workspace_id, content, citations,"
            " verification, content_hash, answer_basis, basis_components, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.answer_id,
                record.run_id,
                record.workspace_id,
                record.content,
                _dump([c.model_dump(mode="json") for c in record.citations]),
                None
                if record.verification is None
                else _dump(record.verification.model_dump(mode="json")),
                record.content_hash,
                str(record.answer_basis),
                _dump([str(component) for component in record.basis_components]),
                str(record.status),
                record.created_at,
            ),
        )
        return record

    async def get_answer(self, answer_id: str) -> AnswerArtifact | None:
        row = await self._fetch_one(
            "SELECT * FROM answer_artifacts WHERE answer_id = ?", (answer_id,)
        )
        return None if row is None else _answer_from_row(row)

    async def commit_answer(
        self,
        *,
        artifact: AnswerArtifact,
        assistant_message_content: str | None = None,
        deltas: Sequence[str] = (),
        usage: Mapping[str, Any] | None = None,
        gaps: Sequence[str] = (),
        memory_job: BackgroundJobSpec | None = None,
        execution_id: str = "",
        completed_at: str | None = None,
    ) -> AnswerCommitResult:
        """**单事务**提交正式答案（PRD 13.1 末段）。

        一个事务内按顺序完成：

        1. 写入 ``answer_artifacts``；
        2. 写入助手消息（正文 + 来源快照）；
        3. 更新 run 状态为 ``completed`` / ``completed_with_gaps`` 并记录答案 ID；
        4. 追加事件 ``answer.meta`` → ``answer.delta*`` → ``run.completed``，
           编号在事务内连续分配，最后一条带明确状态；
        5. 写入记忆提炼后台任务。

        任一步骤失败整体回滚：不留半成品答案、消息、事件或后台任务。
        """

        run_row = await self._require(
            "SELECT * FROM runs WHERE id = ?", (artifact.run_id,), "run"
        )
        run = RunRecord.model_validate(run_row)
        if artifact.status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}:
            raise ValueError("答案状态只能是 completed 或 completed_with_gaps（PRD 8.3）")

        contents = list(deltas) or [artifact.content]
        async with self.transaction():
            stored = await self._insert_answer_artifact_locked(artifact)
            message = await self._append_message_locked(
                conversation_id=run.conversation_id,
                role="assistant",
                content=assistant_message_content or artifact.content,
                source_snapshot={
                    "answer_id": stored.answer_id,
                    "answer_basis": str(stored.answer_basis),
                    "basis_components": [str(c) for c in stored.basis_components],
                    "citations": [c.model_dump(mode="json") for c in stored.citations],
                    "content_hash": stored.content_hash,
                },
                run_id=run.id,
                created_at=artifact.created_at or completed_at,
            )
            updated_run = await self._update_run_locked(
                run.id,
                stored.status,
                answer_id=stored.answer_id,
                resumable=False,
            )
            events: list[agent_events.StoredEvent] = []
            meta_payload = {
                "answer_id": stored.answer_id,
                "answer_basis": str(stored.answer_basis),
                "basis_components": [str(c) for c in stored.basis_components],
                "citations": [c.model_dump(mode="json") for c in stored.citations],
                "content_hash": stored.content_hash,
                "status": str(stored.status),
            }
            events.append(
                await self._append_event_locked(
                    run_id=run.id,
                    type=EventType.ANSWER_META,
                    payload=meta_payload,
                    execution_id=execution_id,
                    object_id=stored.answer_id,
                    version=1,
                    created_at=completed_at,
                )
            )
            for index, text in enumerate(contents):
                events.append(
                    await self._append_event_locked(
                        run_id=run.id,
                        type=EventType.ANSWER_DELTA,
                        payload={
                            "answer_id": stored.answer_id,
                            "index": index,
                            "text": text,
                        },
                        execution_id=execution_id,
                        object_id=stored.answer_id,
                        version=index + 1,
                        created_at=completed_at,
                    )
                )
            events.append(
                await self._append_event_locked(
                    run_id=run.id,
                    type=EventType.RUN_COMPLETED,
                    payload={
                        "answer_id": stored.answer_id,
                        "status": str(stored.status),
                        "content_hash": stored.content_hash,
                        "usage": dict(usage or {}),
                        "gaps": list(gaps),
                    },
                    execution_id=execution_id,
                    object_id=stored.answer_id,
                    created_at=completed_at,
                )
            )
            job = None
            if memory_job is not None:
                job = await self._insert_background_job_locked(memory_job)

        if updated_run.status is not stored.status:  # pragma: no cover - 防御性检查
            raise AppStoreError("run 状态与答案状态不一致")
        return AnswerCommitResult(
            artifact=stored, message=message, events=events, job=job
        )

    # =================================================================
    # background_jobs（PRD 13.1 / 11.5）
    # =================================================================

    async def create_background_job(self, spec: BackgroundJobSpec) -> BackgroundJobRecord:
        async with self.transaction():
            return await self._insert_background_job_locked(spec)

    async def _insert_background_job_locked(
        self, spec: BackgroundJobSpec
    ) -> BackgroundJobRecord:
        now = utc_now_iso()
        record = BackgroundJobRecord(
            id=spec.job_id or new_id("job"),
            kind=spec.kind,
            workspace_id=spec.workspace_id,
            conversation_id=spec.conversation_id,
            run_id=spec.run_id,
            operation_id=spec.operation_id,
            source_message_range=dict(spec.source_message_range),
            status=BackgroundJobStatus.PENDING,
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        await self._execute(
            "INSERT INTO background_jobs (id, kind, workspace_id, conversation_id, run_id,"
            " operation_id, source_message_range, status, attempts, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                record.id,
                record.kind,
                record.workspace_id,
                record.conversation_id,
                record.run_id,
                record.operation_id,
                _dump(record.source_message_range),
                str(record.status),
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    async def get_background_job(self, job_id: str) -> BackgroundJobRecord | None:
        row = await self._fetch_one("SELECT * FROM background_jobs WHERE id = ?", (job_id,))
        return None if row is None else _job_from_row(row)

    async def list_background_jobs(
        self,
        *,
        conversation_id: str | None = None,
        status: BackgroundJobStatus | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[BackgroundJobRecord]:
        sql = "SELECT * FROM background_jobs WHERE 1 = 1"
        params: list[Any] = []
        for column, value in (
            ("conversation_id", conversation_id),
            ("status", None if status is None else str(status)),
            ("kind", kind),
        ):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY created_at, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return [_job_from_row(row) for row in rows]

    async def update_background_job(
        self,
        job_id: str,
        status: BackgroundJobStatus,
        *,
        result: Mapping[str, Any] | None = None,
        increment_attempts: bool = False,
    ) -> BackgroundJobRecord:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [str(status), utc_now_iso()]
        if result is not None:
            assignments.append("result = ?")
            params.append(_dump(dict(result)))
        if increment_attempts:
            assignments.append("attempts = attempts + 1")
        params.append(job_id)
        await self._execute(
            f"UPDATE background_jobs SET {', '.join(assignments)} WHERE id = ?", params
        )
        row = await self._require(
            "SELECT * FROM background_jobs WHERE id = ?", (job_id,), "后台任务"
        )
        return _job_from_row(row)

    # =================================================================
    # history_fts（PRD 11.5：分词内容与原文分离）
    # =================================================================

    async def index_history_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        workspace_id: str,
        tokens: str,
    ) -> None:
        """写入/替换一条消息的分词索引（查询词构造归 T08）。"""

        async with self.transaction():
            await self._execute("DELETE FROM history_fts WHERE message_id = ?", (message_id,))
            await self._execute(
                "INSERT INTO history_fts (message_id, conversation_id, workspace_id, tokens)"
                " VALUES (?, ?, ?, ?)",
                (message_id, conversation_id, workspace_id, tokens),
            )

    async def delete_history_message(self, message_id: str) -> None:
        await self._execute("DELETE FROM history_fts WHERE message_id = ?", (message_id,))

    async def history_index_size(self) -> int:
        return int(await self._scalar("SELECT COUNT(*) FROM history_fts"))

    # =================================================================
    # migration_jobs（PRD 13.1 / 15.2）
    # =================================================================

    async def record_migration_job(
        self,
        *,
        migration_version: str,
        stage: str,
        source_fingerprint: str = "",
        target: str = "",
        status: str = "pending",
        validation: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> MigrationJobRecord:
        now = utc_now_iso()
        record = MigrationJobRecord(
            id=job_id or new_id("mig"),
            migration_version=migration_version,
            stage=stage,
            source_fingerprint=source_fingerprint,
            target=target,
            status=status,
            validation=dict(validation or {}),
            created_at=now,
            updated_at=now,
        )
        await self._execute(
            "INSERT INTO migration_jobs (id, migration_version, stage, source_fingerprint,"
            " target, status, validation, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.migration_version,
                record.stage,
                record.source_fingerprint,
                record.target,
                record.status,
                _dump(record.validation),
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    async def update_migration_job(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        validation: Mapping[str, Any] | None = None,
    ) -> MigrationJobRecord:
        assignments = ["updated_at = ?"]
        params: list[Any] = [utc_now_iso()]
        for column, value in (
            ("stage", stage),
            ("status", status),
            ("validation", None if validation is None else _dump(dict(validation))),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.append(job_id)
        await self._execute(
            f"UPDATE migration_jobs SET {', '.join(assignments)} WHERE id = ?", params
        )
        row = await self._require(
            "SELECT * FROM migration_jobs WHERE id = ?", (job_id,), "迁移任务"
        )
        return _migration_from_row(row)

    async def get_migration_job(self, job_id: str) -> MigrationJobRecord | None:
        row = await self._fetch_one("SELECT * FROM migration_jobs WHERE id = ?", (job_id,))
        return None if row is None else _migration_from_row(row)

    async def list_migration_jobs(
        self,
        *,
        status: str | None = None,
        migration_version: str | None = None,
    ) -> list[MigrationJobRecord]:
        """枚举迁移记录（PRD 15.2 的幂等与「中断后继续当前阶段」需要枚举）。

        这是迁移记录的**唯一公开枚举入口**：业务代码与测试都不得用私有
        ``_fetch_all`` 绕过（t4 的迁移用例、t11 的 A26 验收都用它）。
        顺序为 ``created_at, id``；同一秒内创建的多条记录靠 ``id`` 稳定排序。
        """

        sql = "SELECT * FROM migration_jobs WHERE 1 = 1"
        params: list[Any] = []
        for column, value in (("status", status), ("migration_version", migration_version)):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY created_at, id"
        rows = await self._fetch_all(sql, params)
        return [_migration_from_row(row) for row in rows]

    # =================================================================
    # 诊断
    # =================================================================

    async def table_names(self) -> list[str]:
        """业务表名（已剔除 FTS5 的影子表 ``history_fts_*``，便于直接与契约比对）。"""

        rows = await self._fetch_all(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        )
        return [
            row["name"]
            for row in rows
            if not row["name"].startswith(("sqlite_", "history_fts_"))
        ]

    async def user_version(self) -> int:
        return int(await self._scalar("PRAGMA user_version"))


# =====================================================================
# 行 → 记录（JSON 列在此解码，读到的字段名与契约一致）
# =====================================================================


def _message_from_row(row: Mapping[str, Any]) -> MessageRecord:
    data = _row_to_dict(row)
    data["source_snapshot"] = _loads(data.get("source_snapshot"), {})
    return MessageRecord.model_validate(data)


def _task_from_row(row: Mapping[str, Any]) -> AgentTaskRecord:
    data = _row_to_dict(row)
    data["requirement_ids"] = _loads(data.get("requirement_ids"), [])
    data["depends_on"] = _loads(data.get("depends_on"), [])
    return AgentTaskRecord.model_validate(data)


def _operation_from_row(row: Mapping[str, Any]) -> OperationRecord:
    return OperationRecord.model_validate(_row_to_dict(row))


def _approval_from_row(row: Mapping[str, Any]) -> ApprovalRequest:
    data = _row_to_dict(row)
    data["display"] = _loads(data.get("display"), {})
    data["decision"] = _loads(data.get("decision"), None)
    data["scope"] = _loads(data.get("scope"), None)
    return ApprovalRequest.model_validate(data)


def _input_from_row(row: Mapping[str, Any]) -> InputRequest:
    data = _row_to_dict(row)
    data["options"] = _loads(data.get("options"), [])
    return InputRequest.model_validate(data)


def _event_from_row(row: Mapping[str, Any]) -> agent_events.StoredEvent:
    data = _row_to_dict(row)
    payload = _loads(data.get("payload"), {})
    return agent_events.parse_event(
        {
            "schema_version": agent_events.EVENT_SCHEMA_VERSION,
            "run_id": data["run_id"],
            "event_seq": data["event_seq"],
            "execution_id": data["execution_id"],
            "task_id": data["task_id"],
            "type": data["type"],
            "created_at": data["created_at"],
            "payload": payload,
            "object_id": data["object_id"],
            "version": data["version"],
        }
    )


def _answer_from_row(row: Mapping[str, Any]) -> AnswerArtifact:
    data = _row_to_dict(row)
    data["citations"] = _loads(data.get("citations"), [])
    data["verification"] = _loads(data.get("verification"), None)
    data["basis_components"] = _loads(data.get("basis_components"), [])
    return AnswerArtifact.model_validate(data)


def _job_from_row(row: Mapping[str, Any]) -> BackgroundJobRecord:
    data = _row_to_dict(row)
    data["source_message_range"] = _loads(data.get("source_message_range"), {})
    data["result"] = _loads(data.get("result"), None)
    return BackgroundJobRecord.model_validate(data)


def _migration_from_row(row: Mapping[str, Any]) -> MigrationJobRecord:
    data = _row_to_dict(row)
    data["validation"] = _loads(data.get("validation"), {})
    return MigrationJobRecord.model_validate(data)


#: 每个表的 JSON 列 → 期望的 Python 容器类型。读路径必须全部经 :func:`_loads`
#: 反序列化后再交给 Pydantic（T00b 曾漏掉 ``migration_jobs.validation``，
#: 由往返测试抓出）；新增 JSON 列时同步登记本表，并由契约测试逐列覆盖。
JSON_COLUMNS: Final[dict[str, dict[str, type]]] = {
    "messages": {"source_snapshot": dict},
    "agent_tasks": {"requirement_ids": list, "depends_on": list},
    "provider_messages": {"protocol": dict, "params": dict},
    "approvals": {"display": dict, "decision": dict, "scope": dict},
    "input_requests": {"options": list},
    "run_events": {"payload": dict},
    "answer_artifacts": {"citations": list, "verification": dict, "basis_components": list},
    "background_jobs": {"source_message_range": dict, "result": dict},
    "migration_jobs": {"validation": dict},
}


def _same_decision(current: ApprovalRequest, incoming: ApprovalDecision) -> bool:
    """幂等判定：决定类型与修改后的参数一致即视为重复提交（PRD 10.2）。"""

    if current.decision is None:
        return False
    if current.decision.kind is not incoming.kind:
        return False
    return current.decision.edited_parameters == incoming.edited_parameters
