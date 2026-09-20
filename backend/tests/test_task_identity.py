"""任务身份 run 作用域（v3）回归：同一会话多 run 各自拥有任务记录。

修复的缺陷（t30/t31）：``agent_tasks`` 曾是 ``id TEXT PRIMARY KEY``（全局），而 planner 在
每个 run 里生成的 ``task_id`` 都是 ``t1``/``t2``/… ⇒ 同一会话的**第二个 run** 的
``create_task`` 命中既有行被静默丢弃，带 ``thread_id`` 时还会改写前一个 run 的行。
后果：PRD 12.2「为每个研究任务保存独立状态…并在进程重启后继续」与 PRD 5.4「历史 run 的
只读查询可以读取其已有记录」都不成立。

本文件覆盖：
1. store 层身份隔离（同名任务分属两个 run，互不覆盖、互不丢失）；
2. 复合外键：跨 run 的父任务引用被拒（且仍在 COMMIT 时检查 —— 悬空引用不被放过）；
3. **DEFERRABLE INITIALLY DEFERRED 语义保住**：同一事务内子任务可以先于父任务写入
   （PRD 13.2 要求「保存用户消息并创建任务」同事务完成）；
4. ``get_task`` 不再有「只按 task_id 全局查」的入口（漏传 run 上下文即 TypeError）；
5. 老库 **v2 → v3 幂等重建表升级**：结构正确、既有行**一行不丢**、升级后不再丢新 run 的任务；
6. 老库 **v1 → v3 连续升级**（项目真实历史基线是 v1）同样成立。

测试只用 tmp 数据库，全离线。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import APP_DB_SCHEMA_VERSION, AgentKind, TaskStatus
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具（pytest 从模块导入）

#: v2 老库的四张表（旧身份：``id`` 全局主键 + 单列 ``REFERENCES agent_tasks(id)``）。
_V2_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE agent_tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parent_task_id TEXT REFERENCES agent_tasks(id) DEFERRABLE INITIALLY DEFERRED,
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
    updated_at TEXT NOT NULL
);
CREATE TABLE operations (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES agent_tasks(id),
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES agent_tasks(id),
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
    resolved_at TEXT
);
CREATE TABLE input_requests (
    input_request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES agent_tasks(id),
    question TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '[]',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    answer TEXT,
    question_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
"""


def _legacy_database(path: Path, *, version: int) -> dict[str, int]:
    """造一个老库（v2 或 v1 形状），写入跨 run 的数据，返回各表行数。"""

    connection = sqlite3.connect(path)
    try:
        if version == 1:
            # v1 还没有 workspaces.pinned / order_index（t24 的升级路径）
            connection.execute(
                "CREATE TABLE workspaces (id TEXT PRIMARY KEY, storage_key TEXT NOT NULL"
                " UNIQUE, name TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active',"
                " default_permission_mode TEXT NOT NULL DEFAULT 'ask',"
                " default_reasoning_level TEXT NOT NULL DEFAULT 'high',"
                " knowledge_revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,"
                " updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO workspaces (id, storage_key, name, created_at, updated_at)"
                " VALUES ('ws-1', 'sk-1', '甲区', 'now', 'now')"
            )
        connection.executescript(_V2_SCHEMA)
        connection.execute(
            "INSERT INTO runs (id, conversation_id, workspace_id, created_at, updated_at)"
            " VALUES ('r1', 'c1', 'ws-1', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO runs (id, conversation_id, workspace_id, created_at, updated_at)"
            " VALUES ('r2', 'c1', 'ws-1', 'now', 'now')"
        )
        # r1 的父子任务对 + r2 自己的任务；旧主键下「同一会话第二个 run 的 t1」已被丢弃，
        # 那份数据**无法回溯**（升级只能保证之后不再发生）。
        connection.execute(
            "INSERT INTO agent_tasks (id, run_id, parent_task_id, agent_kind, question,"
            " requirement_ids, depends_on, status, thread_id, agent_context_epoch,"
            " created_at, updated_at) VALUES ('t1', 'r1', 't0', 'rag', '子问题', '[\"R1\"]',"
            " '[\"t0\"]', 'completed', 'run:r1:task:t1', 2, 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO agent_tasks (id, run_id, agent_kind, question, requirement_ids,"
            " depends_on, status, created_at, updated_at)"
            " VALUES ('t0', 'r1', 'planner', '父问题', '[]', '[]', 'completed', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO agent_tasks (id, run_id, agent_kind, question, requirement_ids,"
            " depends_on, status, created_at, updated_at)"
            " VALUES ('legacy-r2', 'r2', 'rag', '第二个 run 的问题', '[\"R2\"]', '[]',"
            " 'completed', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO operations (operation_id, run_id, task_id, tool_name, args_hash,"
            " status, created_at, updated_at) VALUES ('op-1', 'r1', 't1', 'search_knowledge',"
            " 'h1', 'succeeded', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO approvals (approval_id, run_id, task_id, operation_id, args_hash,"
            " created_at) VALUES ('ap-1', 'r1', 't1', 'op-1', 'h1', 'now')"
        )
        connection.execute(
            "INSERT INTO input_requests (input_request_id, run_id, task_id, question,"
            " created_at) VALUES ('in-1', 'r1', 't1', '范围？', 'now')"
        )
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
        counts = {}
        for table in ("agent_tasks", "operations", "approvals", "input_requests"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts
    finally:
        connection.close()


def _pk_columns(path: Path, table: str = "agent_tasks") -> list[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5]]
    finally:
        connection.close()


def _table_sql(path: Path, table: str) -> str:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()[0]
    finally:
        connection.close()


def _foreign_key_check(path: Path) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()


# =====================================================================
# 1. store 层：身份是 run 作用域
# =====================================================================


async def test_same_task_id_in_two_runs_is_independent(app_store: AppStore) -> None:
    """两个 run 的同名任务互不覆盖、互不丢失。"""

    await app_store.create_workspace("ws-1", "sk-1", name="甲区")
    await app_store.create_conversation("conv-1", "ws-1")
    first = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")
    second = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")

    for run in (first, second):
        await app_store.create_task(
            run_id=run.id, task_id="t1", agent_kind=AgentKind.RAG, question=f"问题 {run.id}"
        )

    assert (await app_store.get_task(first.id, "t1")).question == f"问题 {first.id}"
    assert (await app_store.get_task(second.id, "t1")).question == f"问题 {second.id}"

    await app_store.update_task_status(
        second.id, "t1", TaskStatus.COMPLETED, thread_id=f"run:{second.id}:task:t1"
    )
    # 更新第二个 run 的任务不得碰到第一个 run 的同名任务
    kept = await app_store.get_task(first.id, "t1")
    assert kept.status is TaskStatus.QUEUED
    assert kept.thread_id is None
    assert [task.id for task in await app_store.list_tasks(first.id)] == ["t1"]
    assert [task.id for task in await app_store.list_tasks(second.id)] == ["t1"]


async def test_get_task_has_no_global_lookup_path(app_store: AppStore) -> None:
    """没有「只按 task_id」的入口：漏传 run 上下文直接 TypeError，不会静默拿错对象。"""

    await app_store.create_workspace("ws-1", "sk-1")
    await app_store.create_conversation("conv-1", "ws-1")
    run = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")
    await app_store.create_task(run_id=run.id, task_id="t1", agent_kind=AgentKind.RAG)

    with pytest.raises(TypeError):
        await app_store.get_task("t1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await app_store.update_task_status("t1", TaskStatus.RUNNING)  # type: ignore[call-arg]


async def test_composite_foreign_key_rejects_cross_run_parent(app_store: AppStore) -> None:
    """父任务必须与子任务同属一个 run（复合外键），且仍按 COMMIT 时检查。"""

    await app_store.create_workspace("ws-1", "sk-1")
    await app_store.create_conversation("conv-1", "ws-1")
    first = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")
    second = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")
    await app_store.create_task(run_id=first.id, task_id="t0", agent_kind=AgentKind.PLANNER)

    with pytest.raises(sqlite3.IntegrityError):
        async with app_store.transaction():
            await app_store.connection.execute(
                "INSERT INTO agent_tasks (id, run_id, parent_task_id, agent_kind,"
                " created_at, updated_at) VALUES ('t1', ?, 't0', 'rag', 'now', 'now')",
                (second.id,),
            )
    assert await app_store.list_tasks(second.id) == []


async def test_child_task_may_still_precede_parent_in_one_transaction(
    app_store: AppStore,
) -> None:
    """复合外键仍保留 ``DEFERRABLE INITIALLY DEFERRED``：同事务内子先父后也可提交。"""

    await app_store.create_workspace("ws-1", "sk-1")
    await app_store.create_conversation("conv-1", "ws-1")
    run = await app_store.create_run(conversation_id="conv-1", workspace_id="ws-1")

    async with app_store.transaction():
        await app_store.connection.execute(
            "INSERT INTO agent_tasks (id, run_id, parent_task_id, agent_kind, created_at,"
            " updated_at) VALUES ('child', ?, 'parent', 'rag', 'now', 'now')",
            (run.id,),
        )
        await app_store.connection.execute(
            "INSERT INTO agent_tasks (id, run_id, agent_kind, created_at, updated_at)"
            " VALUES ('parent', ?, 'planner', 'now', 'now')",
            (run.id,),
        )
    child = await app_store.get_task(run.id, "child")
    assert child is not None and child.parent_task_id == "parent"


# =====================================================================
# 2. 老库升级：v2 → v3（含 v1 → v3 连续升级）
# =====================================================================


@pytest.mark.parametrize("legacy_version", [1, 2])
async def test_legacy_database_upgrades_in_place_without_losing_rows(
    tmp_path: Path, legacy_version: int
) -> None:
    """老库就地升级到 v3：结构正确、既有行一行不丢、幂等、升级后不再丢新 run 的任务。"""

    path = tmp_path / f"legacy-v{legacy_version}.db"
    before = _legacy_database(path, version=legacy_version)
    assert _pk_columns(path) == ["id"]  # 升级前是全局主键

    store = await AppStore.open(path)
    try:
        assert await store.user_version() == APP_DB_SCHEMA_VERSION == 3
        assert _pk_columns(path) == ["run_id", "id"]
        assert "REFERENCES agent_tasks(run_id, id)" in _table_sql(path, "agent_tasks")
        for table in ("operations", "approvals", "input_requests"):
            assert "REFERENCES agent_tasks(run_id, id)" in _table_sql(path, table)
        assert _foreign_key_check(path) == []

        # 既有行原样保留（老库丢失的数据无法回溯：行数与升级前逐表相等，不凭空补行）
        for table, expected in before.items():
            assert len(await store._fetch_all(f"SELECT * FROM {table}")) == expected
        legacy = await store.get_task("r1", "t1")
        assert legacy is not None
        assert (legacy.parent_task_id, legacy.agent_context_epoch) == ("t0", 2)
        assert legacy.requirement_ids == ["R1"] and legacy.depends_on == ["t0"]
        kept_child = await store.get_task("r2", "legacy-r2")
        assert kept_child is not None and kept_child.requirement_ids == ["R2"]
        # 子表的外键引用仍然有效
        assert (await store.get_approval("ap-1")).task_id == "t1"
        assert (await store.get_input_request("in-1")).task_id == "t1"

        # 升级后：同一会话两个 run 各有自己的同名任务（不再丢）
        for run_id in ("r1", "r2"):
            await store.create_task(run_id=run_id, task_id="t1", agent_kind=AgentKind.RAG)
        assert [task.run_id for task in await store.list_tasks("r1")] == ["r1", "r1"]
        assert [task.run_id for task in await store.list_tasks("r2")] == ["r2", "r2"]

        # 幂等：连续两次 ensure_schema（+ 显式 upgrade_schema）无副作用
        counts = {
            table: len(await store._fetch_all(f"SELECT * FROM {table}"))
            for table in ("agent_tasks", "operations", "approvals", "input_requests")
        }
        await store.ensure_schema()
        await store.ensure_schema()
        assert await store.user_version() == 3
        assert _pk_columns(path) == ["run_id", "id"]
        for table, expected in counts.items():
            assert len(await store._fetch_all(f"SELECT * FROM {table}")) == expected
        assert _foreign_key_check(path) == []
    finally:
        await store.close()


async def test_fresh_database_is_created_at_v3(tmp_path: Path) -> None:
    """全新库直接建 v3 形状（不经过重建），且 ``agent_tasks`` 复合外键在列。"""

    path = tmp_path / "fresh.db"
    store = await AppStore.open(path)
    try:
        assert await store.user_version() == 3
        assert _pk_columns(path) == ["run_id", "id"]
        assert "REFERENCES agent_tasks(run_id, id)" in _table_sql(path, "agent_tasks")
    finally:
        await store.close()
