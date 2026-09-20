"""`app.db` 契约测试（T00b）：建表、约束、事件补读、批准/澄清流转与答案事务。

覆盖 captain 预置的验收清单 `docs/tech/v1.2.0-architecture.md` §6.3 第 1–12 条：
14 张表、四个唯一约束、外键与 WAL 的真实生效、建表幂等、答案事务原子性、
事件补读、批准/澄清幂等与 409 语义。

夹具来自 `tests/fakes_agent.py`（显式导入才生效）。
"""

from __future__ import annotations

import sqlite3

import pytest

from app.agent import events as agent_events
from app.agent.app_store import (
    TABLE_NAMES,
    AnswerCommitResult,
    ApprovalConflictError,
    AppStore,
    BackgroundJobSpec,
    ConflictError,
    InputRequestConflictError,
    RecordNotFoundError,
)
from app.agent.contracts import (
    APP_DB_SCHEMA_VERSION,
    AgentError,
    AgentKind,
    AnswerArtifact,
    AnswerBasis,
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
    BackgroundJobStatus,
    CitationEntry,
    CitationSourceStatus,
    EventType,
    EvidenceSourceType,
    InputRequest,
    InputRequestStatus,
    OperationStatus,
    ProviderMessage,
    RunStatus,
    SourceType,
    TaskStatus,
    VerificationCheck,
    VerificationIssue,
    VerificationResult,
    VerifierVerdict,
)
from tests.fakes_agent import (  # noqa: F401 - 夹具必须显式导入到本模块命名空间
    app_store,
    app_store_file,
    seed_run,
    seeded_run,
)

TABLES_IN_APP_DB = len(TABLE_NAMES)


# =====================================================================
# 1. 建表、PRAGMA 与约束（清单 1–4、8–12）
# =====================================================================


async def test_schema_has_exactly_the_14_prd_tables(app_store: AppStore) -> None:
    names = await app_store.table_names()
    assert sorted(names) == sorted(TABLE_NAMES)
    assert len(names) == 14


async def test_schema_creation_is_idempotent_and_keeps_data(app_store: AppStore) -> None:
    await app_store.create_workspace("w1", "sk1", name="保留我")
    await app_store.ensure_schema()
    await app_store.ensure_schema()

    workspace = await app_store.get_workspace("w1")
    assert workspace is not None
    assert workspace.name == "保留我"
    assert len(await app_store.table_names()) == TABLES_IN_APP_DB


async def test_user_version_records_app_db_schema_version(app_store: AppStore) -> None:
    """schema 版本随 ``contracts.APP_DB_SCHEMA_VERSION`` 走（T_d29 起为 2）。"""

    assert await app_store.user_version() == APP_DB_SCHEMA_VERSION == 3


async def test_foreign_keys_are_on_and_enforced(app_store: AppStore) -> None:
    async with app_store.connection.execute("PRAGMA foreign_keys") as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1

    # 未建会话就写消息 → 外键必须报错（PRD 13.1 外键约束）
    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO messages (id, conversation_id, sequence, role, content, created_at)"
            " VALUES ('m-x', 'no-such-conversation', 1, 'user', 'x', 'now')"
        )


async def test_wal_enabled_for_file_db_and_off_for_memory(app_store_file: AppStore) -> None:
    async with app_store_file.connection.execute("PRAGMA journal_mode") as cursor:
        row = await cursor.fetchone()
    assert str(row[0]).lower() == "wal"

    memory_store = await AppStore.open_memory()
    try:
        async with memory_store.connection.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        assert str(row[0]).lower() != "wal"
    finally:
        await memory_store.close()


async def test_storage_key_is_unique(app_store: AppStore) -> None:
    await app_store.create_workspace("w1", "same-key")
    with pytest.raises(sqlite3.IntegrityError):
        await app_store.create_workspace("w2", "same-key")


async def test_workspace_fields_follow_prd_13_1(app_store: AppStore) -> None:
    workspace = await app_store.create_workspace("w1", "sk1", name="风控组")
    assert workspace.status == "active"
    assert workspace.default_permission_mode == "ask"
    assert workspace.default_reasoning_level == "high"
    assert workspace.knowledge_revision == 0

    assert await app_store.bump_knowledge_revision("w1") == 1
    updated = await app_store.update_workspace(
        "w1",
        name="风控组 2",
        default_permission_mode="auto",
        default_reasoning_level="max",
    )
    assert (updated.name, updated.default_permission_mode) == ("风控组 2", "auto")
    assert updated.default_reasoning_level == "max"


async def test_message_sequence_unique_within_conversation(app_store: AppStore) -> None:
    await app_store.create_workspace("w1", "sk1")
    await app_store.create_conversation("c1", "w1")

    first = await app_store.append_message(conversation_id="c1", role="user", content="一")
    second = await app_store.append_message(conversation_id="c1", role="assistant", content="二")
    assert (first.sequence, second.sequence) == (1, 2)

    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO messages (id, conversation_id, sequence, role, content, created_at)"
            " VALUES ('dup', 'c1', 2, 'user', 'x', 'now')"
        )


async def test_client_request_id_unique_per_conversation_is_idempotent(
    app_store: AppStore,
) -> None:
    await app_store.create_workspace("w1", "sk1")
    await app_store.create_conversation("c1", "w1")
    await app_store.create_conversation("c2", "w1")

    first = await app_store.create_run(
        conversation_id="c1", workspace_id="w1", client_request_id="req-1"
    )
    again = await app_store.create_run(
        conversation_id="c1", workspace_id="w1", client_request_id="req-1"
    )
    assert again.id == first.id  # PRD 13.2：同一请求重复提交返回同一个 run

    # 别的会话可以用同一个 client_request_id（唯一约束是会话内的）
    other = await app_store.create_run(
        conversation_id="c2", workspace_id="w1", client_request_id="req-1"
    )
    assert other.id != first.id

    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO runs (id, conversation_id, workspace_id, client_request_id,"
            " created_at, updated_at) VALUES ('dup-run', 'c1', 'w1', 'req-1', 'n', 'n')"
        )


async def test_run_and_task_records_carry_identity_fields(
    app_store: AppStore, seeded_run
) -> None:
    run = await app_store.get_run(seeded_run.run_id)
    assert run is not None
    assert run.workspace_id == seeded_run.workspace_id
    assert run.context_epoch == 1
    assert run.resumable is False
    assert run.origin_run_id is None
    assert run.stop_reason is None

    # PRD 13.1 runs 字段：origin_run_id / stop_reason / resumable / answer_id
    # （会话移动或记忆失效后继续的任务带来源 run；停止原因与可恢复标记一并保存）
    resumed = await app_store.create_run(
        conversation_id=seeded_run.conversation_id,
        workspace_id=seeded_run.workspace_id,
        origin_run_id=seeded_run.run_id,
    )
    assert resumed.origin_run_id == seeded_run.run_id

    stopped = await app_store.update_run_status(
        run.id,
        RunStatus.CANCELLED,
        stop_reason="conversation_moved",
        resumable=False,
        answer_id="ans-none",
        error_code=None,
    )
    assert stopped.stop_reason == "conversation_moved"
    assert stopped.resumable is False
    assert stopped.answer_id == "ans-none"

    task = await app_store.create_task(
        run_id=run.id,
        task_id="t-1",
        agent_kind=AgentKind.RAG,
        question="P99 怎么优化",
        requirement_ids=["R1"],
        depends_on=[],
    )
    assert task.thread_id is None
    assert task.agent_context_epoch == 1

    # 重复创建同一 task_id 复用既有行（PRD 12.2：重新执行分配节点时复用任务）
    reused = await app_store.create_task(run_id=run.id, task_id="t-1", agent_kind=AgentKind.RAG)
    assert reused.created_at == task.created_at

    updated = await app_store.update_task_status(
        run.id, "t-1", TaskStatus.RUNNING, thread_id="run:r:task:t-1"
    )
    assert updated.status == TaskStatus.RUNNING
    assert updated.thread_id == "run:r:task:t-1"
    assert [t.id for t in await app_store.list_tasks(run.id)] == ["t-1"]

    with pytest.raises(RecordNotFoundError):
        await app_store.update_task_status(run.id, "no-such-task", TaskStatus.RUNNING)
    # 另一 run 的同名任务**不存在**（身份是 run 作用域，不会串到别的 run 的行）
    other = await app_store.create_run(
        conversation_id=run.conversation_id, workspace_id=run.workspace_id
    )
    assert await app_store.get_task(other.id, "t-1") is None


async def test_conversation_snapshot_and_move_bump_context_epoch(app_store: AppStore) -> None:
    await app_store.create_workspace("w1", "sk1")
    await app_store.create_workspace("w2", "sk2")
    await app_store.create_conversation("c1", "w1")

    snapshot = await app_store.save_memory_snapshot(
        "c1",
        memory_revision=3,
        memory_invalidation_revision=1,
        memory_snapshot_text="讨论主题：登录流程",
        memory_snapshot_hash="hash-1",
    )
    assert snapshot.memory_revision == 3
    assert snapshot.memory_snapshot_hash == "hash-1"
    assert snapshot.context_epoch == 1

    with pytest.raises(ConflictError):
        await app_store.move_conversation(
            "c1", target_workspace_id="w2", expected_context_epoch=99
        )

    moved = await app_store.move_conversation(
        "c1", target_workspace_id="w2", expected_context_epoch=1
    )
    assert moved.workspace_id == "w2"
    assert moved.moved_from_workspace_id == "w1"
    assert moved.context_epoch == 2  # PRD 5.4：移动后增加 context_epoch
    assert moved.memory_snapshot_text == ""  # 目标区快照重新读取

    cleared = await app_store.clear_memory_snapshot("c1", invalidation_revision=2)
    assert cleared.memory_snapshot_hash == ""
    assert cleared.memory_invalidation_revision == 2


async def test_history_fts_is_fts5_and_separate_from_messages(app_store: AppStore) -> None:
    await app_store.create_workspace("w1", "sk1")
    await app_store.create_conversation("c1", "w1")
    message = await app_store.append_message(
        conversation_id="c1", role="user", content="风控 P99 时延优化"
    )
    await app_store.index_history_message(
        message_id=message.id,
        conversation_id="c1",
        workspace_id="w1",
        tokens="风控 p99 时延 优化",
    )
    assert await app_store.history_index_size() == 1

    # FTS5 虚拟表存在，且分词内容与原文分离（原文仍在 messages）
    async with app_store.connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'history_fts'"
    ) as cursor:
        ddl = (await cursor.fetchone())[0]
    assert "fts5" in ddl.lower()
    assert "unicode61" in ddl
    assert "风控 P99 时延优化" == (await app_store.get_message(message.id)).content

    await app_store.delete_history_message(message.id)
    assert await app_store.history_index_size() == 0


async def test_provider_messages_sequence_unique_per_context(app_store: AppStore) -> None:
    message = ProviderMessage(
        context_id="ctx-1",
        context_epoch=1,
        agent_context_epoch=1,
        sequence=1,
        protocol={"role": "assistant", "content": "hi"},
        model="deepseek-flash",
    )
    stored = await app_store.append_provider_message(message)
    assert stored.created_at

    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO provider_messages (id, context_id, sequence, protocol, created_at)"
            " VALUES ('dup', 'ctx-1', 1, '{}', 'now')"
        )
    assert await app_store.next_provider_sequence("ctx-1") == 2
    rows = await app_store.list_provider_messages("ctx-1")
    assert rows[0].protocol["content"] == "hi"


async def test_migration_job_roundtrip(app_store: AppStore) -> None:
    job = await app_store.record_migration_job(
        migration_version="v1.1.0->v1.2.0", stage="import_workspaces", job_id="mig-1"
    )
    assert job.status == "pending"

    updated = await app_store.update_migration_job(
        "mig-1", stage="chroma_copy", status="running", validation={"files": 12}
    )
    assert updated.stage == "chroma_copy"
    assert updated.validation == {"files": 12}


async def test_list_migration_jobs_is_the_public_enumeration_entry(app_store: AppStore) -> None:
    """t4 的迁移用例与 t11 的 A26 验收需要枚举迁移记录，不得用私有 ``_fetch_all``。"""

    await app_store.record_migration_job(
        migration_version="v1.1.0->v1.2.0", stage="import_workspaces", job_id="mig-a"
    )
    await app_store.record_migration_job(
        migration_version="v1.1.0->v1.2.0", stage="move_vectors", job_id="mig-b"
    )
    await app_store.update_migration_job("mig-b", status="running", validation={"vectors": 3})

    everything = await app_store.list_migration_jobs()
    assert {job.id for job in everything} == {"mig-a", "mig-b"}
    assert all(isinstance(job.validation, dict) for job in everything)

    assert [job.id for job in await app_store.list_migration_jobs(status="running")] == ["mig-b"]
    assert len(await app_store.list_migration_jobs(migration_version="v1.1.0->v1.2.0")) == 2
    assert await app_store.list_migration_jobs(migration_version="other") == []


async def test_message_and_run_may_reference_each_other_in_one_transaction(
    app_store: AppStore,
) -> None:
    """PRD 13.2 建 run 路径：保存用户消息与创建任务必须在同一事务内完成。

    ``runs.input_message_id`` 与 ``messages.run_id`` 是互指对，因此两者都是
    ``DEFERRABLE INITIALLY DEFERRED``；**两种插入顺序都必须提交成功**。
    """

    for index, message_first in enumerate((True, False)):
        workspace_id, conversation_id = f"w-{index}", f"c-{index}"
        run_id, message_id = f"r-{index}", f"m-{index}"
        await app_store.create_workspace(workspace_id, f"sk-{index}")
        await app_store.create_conversation(conversation_id, workspace_id)

        insert_message = (
            (
                "INSERT INTO messages (id, conversation_id, sequence, role, content, run_id,"
                " created_at) VALUES (?, ?, 1, 'user', '问题', ?, 'now')"
            ),
            (message_id, conversation_id, run_id),
        )
        insert_run = (
            (
                "INSERT INTO runs (id, conversation_id, workspace_id, input_message_id,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, 'now', 'now')"
            ),
            (run_id, conversation_id, workspace_id, message_id),
        )
        statements = (
            (insert_message, insert_run) if message_first else (insert_run, insert_message)
        )
        async with app_store.transaction():
            for sql, params in statements:
                await app_store.connection.execute(sql, params)

        stored_run = await app_store.get_run(run_id)
        assert stored_run is not None
        assert stored_run.input_message_id == message_id
        stored_message = await app_store.get_message(message_id)
        assert stored_message is not None
        assert stored_message.run_id == run_id


async def test_deferred_foreign_keys_still_reject_dangling_reference(
    app_store: AppStore,
) -> None:
    """延迟外键只在 COMMIT 时检查，而不是取消检查：悬空引用必须被拒。"""

    await app_store.create_workspace("w1", "sk1")
    await app_store.create_conversation("c1", "w1")

    with pytest.raises(sqlite3.IntegrityError):
        async with app_store.transaction():
            await app_store.connection.execute(
                "INSERT INTO messages (id, conversation_id, sequence, role, content, run_id,"
                " created_at) VALUES ('m2', 'c1', 1, 'user', '问题', 'no-such-run', 'now')"
            )

    # 提交失败后连接仍可用（事务已回滚，没有半开状态）
    assert await app_store.message_count("c1") == 0
    await app_store.create_workspace("w9", "sk9")
    assert await app_store.get_workspace("w9") is not None


async def test_child_task_may_be_written_before_parent_in_one_transaction(
    app_store: AppStore, seeded_run
) -> None:
    """``agent_tasks.parent_task_id`` 同为延迟外键：同一事务内子任务先写也可提交。"""

    async with app_store.transaction():
        await app_store.connection.execute(
            "INSERT INTO agent_tasks (id, run_id, parent_task_id, agent_kind, created_at,"
            " updated_at) VALUES ('child', ?, 'parent-task', 'rag', 'now', 'now')",
            (seeded_run.run_id,),
        )
        await app_store.connection.execute(
            "INSERT INTO agent_tasks (id, run_id, agent_kind, created_at, updated_at)"
            " VALUES ('parent-task', ?, 'planner', 'now', 'now')",
            (seeded_run.run_id,),
        )
    child = await app_store.get_task(seeded_run.run_id, "child")
    assert child is not None
    assert child.parent_task_id == "parent-task"


# =====================================================================
# 2. 操作登记与幂等（PRD 12.1 / 12.4）
# =====================================================================


async def test_operation_registration_is_idempotent(app_store: AppStore, seeded_run) -> None:
    first = await app_store.register_operation(
        operation_id="op-1", tool_name="search_knowledge", args_hash="h1", run_id=seeded_run.run_id
    )
    assert first.attempts == 0
    assert first.status == OperationStatus.PENDING

    again = await app_store.register_operation(
        operation_id="op-1", tool_name="search_knowledge", args_hash="h1", run_id=seeded_run.run_id
    )
    assert again.created_at == first.created_at
    assert again.attempts == 0  # 重复登记不增加尝试次数

    running = await app_store.start_operation_attempt("op-1")
    assert running.attempts == 1
    assert running.status == OperationStatus.RUNNING

    finished = await app_store.finish_operation(
        "op-1", OperationStatus.SUCCEEDED, result_ref="evidence:1"
    )
    assert finished.status == OperationStatus.SUCCEEDED
    assert finished.result_ref == "evidence:1"

    # operation_id 是主键：约束在数据库层真实生效（不能只靠应用层幂等）
    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO operations (operation_id, tool_name, args_hash, created_at, updated_at)"
            " VALUES ('op-1', 'other_tool', 'h2', 'now', 'now')"
        )


# =====================================================================
# 3. 批准与澄清流转（清单 7，PRD 10.2 / 13.2）
# =====================================================================


def _approval(approval_id: str, run_id: str, operation_id: str = "op-1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id=run_id,
        operation_id=operation_id,
        args_hash="args-hash",
        context_epoch=1,
        rule_version="rules-v1",
    )


async def test_approval_pending_superseded_and_decision_flow(app_store: AppStore, seeded_run) -> None:
    request = await app_store.request_approval(_approval("ap-1", seeded_run.run_id))
    assert request.status == ApprovalStatus.PENDING

    # 同一 operation 的重复请求复用既有 pending（恢复后不重复弹卡）
    reused = await app_store.request_approval(_approval("ap-dup", seeded_run.run_id))
    assert reused.approval_id == "ap-1"

    allowed = await app_store.resolve_approval(
        "ap-1",
        ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE),
        args_hash="args-hash",
        context_epoch=1,
        rule_version="rules-v1",
    )
    assert allowed.status == ApprovalStatus.APPROVED
    assert allowed.decision is not None
    assert allowed.resolved_at


async def test_approval_same_decision_is_idempotent(app_store: AppStore, seeded_run) -> None:
    await app_store.request_approval(_approval("ap-2", seeded_run.run_id))
    decision = ApprovalDecision(kind=ApprovalDecisionKind.DENY)

    first = await app_store.resolve_approval("ap-2", decision)
    second = await app_store.resolve_approval("ap-2", ApprovalDecision(kind=ApprovalDecisionKind.DENY))
    assert first.status == ApprovalStatus.DENIED
    assert second.resolved_at == first.resolved_at  # 幂等返回现有结果


async def test_approval_conflicting_decision_raises_conflict(app_store: AppStore, seeded_run) -> None:
    await app_store.request_approval(_approval("ap-3", seeded_run.run_id))
    await app_store.resolve_approval("ap-3", ApprovalDecision(kind=ApprovalDecisionKind.DENY))

    with pytest.raises(ApprovalConflictError) as excinfo:
        await app_store.resolve_approval(
            "ap-3", ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE)
        )
    assert isinstance(excinfo.value, ConflictError)  # API 层据此返回 409


async def test_approval_stale_args_or_context_is_rejected(app_store: AppStore, seeded_run) -> None:
    await app_store.request_approval(_approval("ap-4", seeded_run.run_id))

    with pytest.raises(ApprovalConflictError):
        await app_store.resolve_approval(
            "ap-4",
            ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE),
            args_hash="changed-hash",
        )
    with pytest.raises(ApprovalConflictError):
        await app_store.resolve_approval(
            "ap-4",
            ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE),
            context_epoch=7,
        )
    with pytest.raises(ApprovalConflictError):
        await app_store.resolve_approval(
            "ap-4",
            ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE),
            rule_version="rules-v2",
        )
    with pytest.raises(ApprovalConflictError):
        # 「允许本轮同类操作」必须给出授权范围（PRD 10.2）
        await app_store.resolve_approval(
            "ap-4", ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_SCOPE)
        )

    scoped = await app_store.resolve_approval(
        "ap-4",
        ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_SCOPE),
        scope=ApprovalScope(
            run_id=seeded_run.run_id,
            tool_category="web_search",
            question_scope="今天 AI 新闻",
            rule_version="rules-v1",
        ),
    )
    assert scoped.status == ApprovalStatus.APPROVED
    assert scoped.scope is not None
    assert scoped.scope.tool_category == "web_search"


async def test_supersede_pending_approvals(app_store: AppStore, seeded_run) -> None:
    await app_store.request_approval(_approval("ap-5", seeded_run.run_id, "op-5"))
    assert await app_store.supersede_pending_approvals(
        seeded_run.run_id, reason="conversation_moved"
    ) == 1
    superseded = await app_store.get_approval("ap-5")
    assert superseded is not None
    assert superseded.status == ApprovalStatus.SUPERSEDED
    assert superseded.superseded_reason == "conversation_moved"
    # 已失效的批准不能再被批准
    with pytest.raises(ApprovalConflictError):
        await app_store.resolve_approval(
            "ap-5", ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE)
        )


async def test_input_request_flow_and_version_conflicts(app_store: AppStore, seeded_run) -> None:
    request = InputRequest(
        input_request_id="in-1",
        run_id=seeded_run.run_id,
        question="你说的「这个方案」是指哪一个？",
        options=["登录流程", "风控方案"],
        question_version=1,
    )
    stored = await app_store.request_input(request)
    assert stored.status == InputRequestStatus.PENDING

    answered = await app_store.resolve_input("in-1", answer="风控方案", question_version=1)
    assert answered.status == InputRequestStatus.ANSWERED
    assert answered.answer == "风控方案"

    # 相同回答幂等
    again = await app_store.resolve_input("in-1", answer="风控方案")
    assert again.resolved_at == answered.resolved_at

    with pytest.raises(InputRequestConflictError):
        await app_store.resolve_input("in-1", answer="登录流程")
    with pytest.raises(InputRequestConflictError):
        await app_store.resolve_input("in-1", answer="风控方案", question_version=2)

    other = await app_store.request_input(
        InputRequest(input_request_id="in-2", run_id=seeded_run.run_id, question="范围？")
    )
    assert other.status == InputRequestStatus.PENDING
    assert await app_store.supersede_pending_inputs(seeded_run.run_id) == 1
    superseded = await app_store.get_input_request("in-2")
    assert superseded is not None
    assert superseded.status == InputRequestStatus.SUPERSEDED
    with pytest.raises(InputRequestConflictError):
        await app_store.resolve_input("in-2", answer="晚了")


# =====================================================================
# 4. 事件：单调编号、补读、校验（清单 2、6、21）
# =====================================================================


async def test_run_events_are_monotonic_and_catch_up_with_after(app_store: AppStore, seeded_run) -> None:
    run_id = seeded_run.run_id
    for index in range(3):
        await app_store.append_event(
            run_id=run_id,
            type=EventType.RUN_STATUS,
            payload={"status": "running", "message": f"第 {index} 步"},
        )

    assert await app_store.last_event_seq(run_id) == 3
    assert [e.event_seq for e in await app_store.read_events(run_id)] == [1, 2, 3]
    # after=seq 必须严格大于：补读语义（PRD 13.3）
    assert [e.event_seq for e in await app_store.read_events(run_id, after_seq=1)] == [2, 3]
    assert [e.event_seq for e in await app_store.read_events(run_id, after_seq=3)] == []

    with pytest.raises(sqlite3.IntegrityError):
        await app_store.connection.execute(
            "INSERT INTO run_events (run_id, event_seq, type, payload, created_at)"
            " VALUES (?, 2, 'run.status', '{}', 'now')",
            (run_id,),
        )


async def test_event_payload_and_version_are_validated_before_write(
    app_store: AppStore, seeded_run
) -> None:
    run_id = seeded_run.run_id

    with pytest.raises(agent_events.EventProtocolError):
        await app_store.append_event(
            run_id=run_id, type=EventType.RUN_STATUS, payload={"status": "不存在的状态"}
        )
    with pytest.raises(agent_events.EventProtocolError):
        await app_store.append_event(
            run_id=run_id, type="no.such.event", payload={}
        )
    with pytest.raises(agent_events.EventProtocolError):
        await app_store.append_event(
            run_id=run_id, type=EventType.RUN_STATUS, payload={"status": "running", "x": 1}
        )
    # 非法事件不入库，也不占用编号
    assert await app_store.read_events(run_id) == []
    assert await app_store.last_event_seq(run_id) == 0

    with pytest.raises(agent_events.EventProtocolError):
        agent_events.parse_event(
            {
                "schema_version": 99,
                "run_id": run_id,
                "event_seq": 1,
                "execution_id": "e",
                "type": "run.status",
                "created_at": "t",
                "payload": {"status": "running"},
            }
        )


async def test_route_decided_event_carries_contract_payload(app_store: AppStore, seeded_run) -> None:
    from app.agent.contracts import Freshness, IntentKind, RouteDecision

    decision = RouteDecision(
        question="今天 AI 圈有哪些重大新闻",
        resolved_question="今天 AI 圈有哪些重大新闻",
        intent=IntentKind.RESEARCH,
        needs_web=True,
        freshness=Freshness.CURRENT,
        reason_summary="需要最新公开信息，走联网路径",
    )
    event = await app_store.append_event(
        run_id=seeded_run.run_id,
        type=EventType.ROUTE_DECIDED,
        payload={"decision": decision.model_dump(mode="json")},
    )
    assert event.payload["decision"]["intent"] == "research"
    assert event.schema_version == 2


# =====================================================================
# 5. 答案事务（清单 5：原子性 + 事件顺序 + 记忆任务）
# =====================================================================


def _artifact(run_id: str, *, status: RunStatus = RunStatus.COMPLETED) -> AnswerArtifact:
    return AnswerArtifact(
        answer_id="ans-1",
        run_id=run_id,
        workspace_id="ws-0001",
        content="完整答案正文",
        citations=[
            CitationEntry(
                label="S1",
                source_type=SourceType.KNOWLEDGE,
                evidence_id="ev-1",
                workspace_id="ws-0001",
                source_status=CitationSourceStatus.ACTIVE,
            )
        ],
        verification=VerificationResult(
            verdict=VerifierVerdict.PASS,
            issues=[
                VerificationIssue(
                    check=VerificationCheck.CITATION_VALIDITY,
                    evidence_ids=["ev-1"],
                )
            ],
        ),
        content_hash="hash-answer",
        answer_basis=AnswerBasis.KNOWLEDGE,
        basis_components=[SourceType.KNOWLEDGE],
        status=status,
    )


async def test_commit_answer_is_atomic_and_ordered(app_store: AppStore, seeded_run) -> None:
    job_spec = BackgroundJobSpec(
        kind="memory_extract",
        workspace_id=seeded_run.workspace_id,
        conversation_id=seeded_run.conversation_id,
        run_id=seeded_run.run_id,
        source_message_range={"from": 1, "to": 2},
    )
    result = await app_store.commit_answer(
        artifact=_artifact(seeded_run.run_id),
        deltas=["完整", "答案正文"],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        gaps=[],
        memory_job=job_spec,
        execution_id="exec-1",
    )

    assert isinstance(result, AnswerCommitResult)
    assert [event.type for event in result.events] == [
        EventType.ANSWER_META,
        EventType.ANSWER_DELTA,
        EventType.ANSWER_DELTA,
        EventType.RUN_COMPLETED,
    ]
    assert [event.event_seq for event in result.events] == [1, 2, 3, 4]
    assert result.events[-1].payload["status"] == "completed"
    assert result.message.role == "assistant"
    assert result.message.source_snapshot["answer_id"] == "ans-1"
    assert result.job is not None
    assert result.job.status == BackgroundJobStatus.PENDING

    run = await app_store.get_run(seeded_run.run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.answer_id == "ans-1"
    assert await app_store.get_answer("ans-1") is not None
    assert await app_store.message_count(seeded_run.conversation_id) == 2

    stored_events = await app_store.read_events(seeded_run.run_id)
    assert [event.type.value for event in stored_events] == [
        "answer.meta",
        "answer.delta",
        "answer.delta",
        "run.completed",
    ]


async def test_commit_answer_supports_completed_with_gaps(app_store: AppStore, seeded_run) -> None:
    result = await app_store.commit_answer(
        artifact=_artifact(seeded_run.run_id, status=RunStatus.COMPLETED_WITH_GAPS),
        gaps=["缺少 2026-09 的发布说明"],
    )
    run = await app_store.get_run(seeded_run.run_id)
    assert run is not None and run.status == RunStatus.COMPLETED_WITH_GAPS
    assert result.events[-1].payload["status"] == "completed_with_gaps"
    assert result.events[-1].payload["gaps"] == ["缺少 2026-09 的发布说明"]
    # 未给 deltas 时也必须有正文片段可展示
    assert result.events[1].payload["text"] == "完整答案正文"


async def test_commit_answer_rolls_back_everything_on_failure(
    app_store: AppStore, seeded_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_spec = BackgroundJobSpec(
        kind="memory_extract", conversation_id=seeded_run.conversation_id, run_id=seeded_run.run_id
    )

    async def boom(spec: BackgroundJobSpec):
        raise RuntimeError("注入失败：模拟记忆任务写入崩溃")

    monkeypatch.setattr(app_store, "_insert_background_job_locked", boom)

    with pytest.raises(RuntimeError):
        await app_store.commit_answer(
            artifact=_artifact(seeded_run.run_id),
            deltas=["完整", "答案正文"],
            memory_job=job_spec,
        )

    # 答案、助手消息、事件、后台任务都必须被回滚（不留半成品）
    assert await app_store.get_answer("ans-1") is None
    assert await app_store.message_count(seeded_run.conversation_id) == 1
    assert await app_store.read_events(seeded_run.run_id) == []
    assert await app_store.last_event_seq(seeded_run.run_id) == 0
    assert await app_store.list_background_jobs() == []
    run = await app_store.get_run(seeded_run.run_id)
    assert run is not None
    assert run.status == RunStatus.QUEUED
    assert run.answer_id is None

    # 回滚后事务仍可用：正常提交一次仍成功
    monkeypatch.undo()
    result = await app_store.commit_answer(artifact=_artifact(seeded_run.run_id))
    assert [event.event_seq for event in result.events] == [1, 2, 3]


async def test_commit_answer_rejects_non_published_status(app_store: AppStore, seeded_run) -> None:
    artifact = _artifact(seeded_run.run_id)
    object.__setattr__(artifact, "status", RunStatus.RUNNING)
    with pytest.raises((ValueError, Exception)):
        await app_store.commit_answer(artifact=artifact)


async def test_background_job_lifecycle(app_store: AppStore, seeded_run) -> None:
    job = await app_store.create_background_job(
        BackgroundJobSpec(
            kind="memory_extract",
            conversation_id=seeded_run.conversation_id,
            workspace_id=seeded_run.workspace_id,
        )
    )
    assert job.status == BackgroundJobStatus.PENDING
    running = await app_store.update_background_job(
        job.id, BackgroundJobStatus.RUNNING, increment_attempts=True
    )
    assert (running.status, running.attempts) == (BackgroundJobStatus.RUNNING, 1)
    done = await app_store.update_background_job(
        job.id, BackgroundJobStatus.COMPLETED, result={"entries_added": 2}
    )
    assert done.result == {"entries_added": 2}
    assert [j.id for j in await app_store.list_background_jobs(status=BackgroundJobStatus.COMPLETED)] == [
        job.id
    ]


async def test_error_record_can_be_reported_as_event(app_store: AppStore, seeded_run) -> None:
    error = AgentError(
        error_code="provider_timeout",
        message="模型请求超时",
        node="research_web",
        resumable=True,
        required_action="继续任务",
    )
    event = await app_store.append_event(
        run_id=seeded_run.run_id,
        type=EventType.RUN_FAILED,
        payload={"error": error.model_dump(mode="json")},
    )
    assert event.payload["error"]["error_code"] == "provider_timeout"
    assert event.payload["error"]["resumable"] is True
    assert EvidenceSourceType.WEB.value == "web"  # 契约导入守卫


async def test_seed_run_helper_builds_usable_graph(app_store: AppStore) -> None:
    seeded = await seed_run(app_store, conversation_id="conv-x", run_id="run-x")
    run = await app_store.get_run(seeded.run_id)
    assert run is not None
    messages = await app_store.list_messages(seeded.conversation_id)
    assert [m.id for m in messages] == [seeded.user_message_id]


# =====================================================================
# 6. JSON 列往返（每条读路径都必须反序列化，不能把字符串交给 Pydantic）
# =====================================================================


async def test_json_columns_round_trip_through_every_reader(
    app_store: AppStore, seeded_run
) -> None:
    """逐列锁死往返类型：``source_snapshot`` / ``requirement_ids`` / ``depends_on`` /
    ``protocol`` / ``params`` / ``display`` / ``decision`` / ``scope`` / ``options`` /
    ``payload`` / ``citations`` / ``verification`` / ``basis_components`` /
    ``source_message_range`` / ``result`` / ``validation``。

    读取路径若漏了 ``json.loads``，Pydantic 会因「字符串不是 dict/list」报错
    （T00b 的 ``migration_jobs.validation`` 正是被这条抓出来的）。
    """

    run_id = seeded_run.run_id

    message = await app_store.append_message(
        conversation_id=seeded_run.conversation_id,
        role="assistant",
        content="带来源快照的消息",
        source_snapshot={"answer_id": "ans-rt", "citations": [{"label": "S1"}]},
    )
    assert isinstance(message.source_snapshot, dict)

    task = await app_store.create_task(
        run_id=run_id,
        task_id="task-rt",
        agent_kind=AgentKind.RAG,
        requirement_ids=["R1", "R2"],
        depends_on=[],
    )
    stored_task = await app_store.get_task(task.run_id, task.id)
    assert stored_task is not None
    assert stored_task.requirement_ids == ["R1", "R2"]
    assert stored_task.depends_on == []

    await app_store.append_provider_message(
        ProviderMessage(
            context_id="ctx-rt",
            sequence=1,
            protocol={"role": "assistant", "tool_calls": [{"id": "c1"}]},
            params={"temperature": 0},
        )
    )
    provider_message = (await app_store.list_provider_messages("ctx-rt"))[0]
    assert provider_message.protocol["tool_calls"] == [{"id": "c1"}]
    assert provider_message.params == {"temperature": 0}

    approval = await app_store.request_approval(
        ApprovalRequest(
            approval_id="ap-rt",
            run_id=run_id,
            operation_id="op-rt",
            args_hash="h",
            rule_version="v1",
            display={
                "purpose": "查询最新新闻",
                "tool_name": "web_search",
                "query_or_urls": ["AI 新闻"],
                "parameters": {"topic": "news"},
            },
        )
    )
    assert approval.display.parameters == {"topic": "news"}
    resolved = await app_store.resolve_approval(
        "ap-rt",
        ApprovalDecision(
            kind=ApprovalDecisionKind.ALLOW_SCOPE, edited_parameters={"q": "x"}
        ),
        scope=ApprovalScope(
            run_id=run_id,
            tool_category="web_search",
            question_scope="AI 新闻",
            rule_version="v1",
        ),
    )
    assert resolved.decision is not None
    assert resolved.decision.edited_parameters == {"q": "x"}
    assert resolved.scope is not None
    assert resolved.scope.tool_category == "web_search"

    await app_store.request_input(
        InputRequest(
            input_request_id="in-rt",
            run_id=run_id,
            question="指哪一个？",
            options=["A", "B"],
        )
    )
    stored_input = await app_store.get_input_request("in-rt")
    assert stored_input is not None
    assert stored_input.options == ["A", "B"]

    event = await app_store.append_event(
        run_id=run_id,
        type=EventType.RUN_STATUS,
        payload={"status": "running", "phase": "research_web"},
    )
    assert event.payload["phase"] == "research_web"
    assert (await app_store.read_events(run_id))[0].payload["status"] == "running"

    result = await app_store.commit_answer(
        artifact=_artifact(run_id),
        memory_job=BackgroundJobSpec(
            kind="memory_extract",
            workspace_id=seeded_run.workspace_id,
            conversation_id=seeded_run.conversation_id,
            run_id=run_id,
            source_message_range={"from": 1, "to": 2},
        ),
    )
    assert result.job is not None
    assert result.job.source_message_range == {"from": 1, "to": 2}
    stored_answer = await app_store.get_answer("ans-1")
    assert stored_answer is not None
    assert isinstance(stored_answer.citations, list)
    assert stored_answer.citations[0].label == "S1"
    assert stored_answer.verification is not None
    assert stored_answer.verification.verdict == VerifierVerdict.PASS
    assert stored_answer.basis_components == [SourceType.KNOWLEDGE]

    finished_job = await app_store.update_background_job(
        result.job.id, BackgroundJobStatus.COMPLETED, result={"entries_added": 1}
    )
    assert finished_job.result == {"entries_added": 1}
    reread_job = await app_store.get_background_job(result.job.id)
    assert reread_job is not None
    assert reread_job.result == {"entries_added": 1}

    await app_store.record_migration_job(
        migration_version="v1.1.0->v1.2.0",
        stage="import_workspaces",
        validation={"files": 12},
        job_id="mig-rt",
    )
    updated = await app_store.update_migration_job(
        "mig-rt", status="running", validation={"files": 12, "vectors": 340}
    )
    assert updated.validation == {"files": 12, "vectors": 340}
    reread = await app_store.get_migration_job("mig-rt")
    assert reread is not None
    assert reread.validation == {"files": 12, "vectors": 340}


async def test_json_column_registry_matches_real_schema(app_store: AppStore) -> None:
    """``JSON_COLUMNS`` 必须与真实表结构一致：不允许出现未登记的 JSON 列。"""

    from app.agent.app_store import JSON_COLUMNS

    for table, columns in JSON_COLUMNS.items():
        async with app_store.connection.execute(f"PRAGMA table_info({table})") as cursor:
            rows = await cursor.fetchall()
        actual = {row["name"] for row in rows}
        missing = set(columns) - actual
        assert not missing, f"{table} 缺少 JSON 列: {missing}"
