"""t1 冻结区契约测试：枚举取值、状态机与运行身份标识（PRD 12.1 / 12.3 / 13.3 等）。

本文件由 T00a 建立，覆盖**枚举与状态机**部分；T00b 在文件末尾追加
`app.db` schema 与 Pydantic 模型用例（`WorkspaceContext` / `RouteDecision` /
`Evidence` ...），不修改本文件上半部分。

覆盖来源：
- PRD 12.3 11 个任务状态与迁移图
- PRD 12.1 `thread_id` 规则
- PRD 13.3 SSE 事件类型（13 行展开为 20 个 `type` 取值）
- PRD 10.2 / 8.3 / 9.1 / 10.1 / 6.2 / 4.2 / 8.2
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agent import contracts as c

#: `app/agent/tools.py` 源码路径：用于「无本地重定义」守卫（读源码文本，不依赖导入副作用）。
_TOOLS_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "agent" / "tools.py"
)

# ---------------------------------------------------------------------
# 运行身份标识（PRD 12.1）—— captain 修正的边界用例
# ---------------------------------------------------------------------


def test_main_thread_id_shape() -> None:
    assert c.main_thread_id("r1") == "run:r1:main"
    assert c.task_thread_id("r1", "tA") == "run:r1:task:tA"


def test_task_thread_id_rejects_colon_and_empty() -> None:
    with pytest.raises(ValueError):
        c.task_thread_id("r1", "")
    with pytest.raises(ValueError):
        c.task_thread_id("r1", "t:A")


@pytest.mark.parametrize(
    "thread_id",
    [
        "run:r1:task:tA",
        "garbage:main",
        "run::main",
        "run:r1",
        "",
    ],
)
def test_is_main_thread_id_rejects_non_main(thread_id: str) -> None:
    """子图 / 缺少 ``run:`` 前缀 / 空 run_id / 无 ``:main`` / 空串都不是主图。"""

    assert c.is_main_thread_id(thread_id) is False


def test_is_main_thread_id_accepts_main() -> None:
    assert c.is_main_thread_id(c.main_thread_id("r1")) is True
    assert c.is_main_thread_id("run:r1:main") is True


def test_is_main_thread_id_rejects_subgraph() -> None:
    assert c.is_main_thread_id(c.task_thread_id("r1", "tA")) is False


def test_is_main_thread_id_and_thread_of_run_agree() -> None:
    """``is_main_thread_id`` 隐含 ``thread_of_run`` 可解析（口径一致）。"""

    for thread_id in (
        "run:r1:main",
        "run:r1:task:tA",
        "garbage:main",
        "run::main",
        "run:r1",
        "",
    ):
        if c.is_main_thread_id(thread_id):
            assert c.thread_of_run(thread_id) == "r1"


@pytest.mark.parametrize(
    ("thread_id", "expected"),
    [
        ("run:r1:main", "r1"),
        ("run:r1:task:tA", "r1"),
        ("run::main", None),
        ("run:r1", None),
        ("garbage:main", None),
        ("", None),
    ],
)
def test_thread_of_run(thread_id: str, expected: str | None) -> None:
    assert c.thread_of_run(thread_id) == expected


# ---------------------------------------------------------------------
# 状态枚举与状态机（PRD 12.3）
# ---------------------------------------------------------------------

_PRD_12_3_STATUSES = {
    "queued",
    "running",
    "waiting_children",
    "waiting_approval",
    "waiting_input",
    "interrupted",
    "failed",
    "cancelling",
    "cancelled",
    "completed",
    "completed_with_gaps",
}


def test_run_and_task_status_cover_prd_12_3() -> None:
    assert {s.value for s in c.RunStatus} == _PRD_12_3_STATUSES
    assert {s.value for s in c.TaskStatus} == _PRD_12_3_STATUSES
    assert len(list(c.RunStatus)) == 11
    assert len(c.RUN_STATUS_LABELS) == 11


def test_run_and_task_vocabulary_stay_aligned() -> None:
    assert {s.value for s in c.RunStatus} == {s.value for s in c.TaskStatus}


def test_terminal_and_resumable_sets() -> None:
    assert c.TERMINAL_RUN_STATUSES == {c.RunStatus.COMPLETED, c.RunStatus.COMPLETED_WITH_GAPS}
    assert c.RESUMABLE_RUN_STATUSES == {
        c.RunStatus.FAILED,
        c.RunStatus.INTERRUPTED,
        c.RunStatus.CANCELLED,
    }
    for terminal in c.TERMINAL_RUN_STATUSES:
        assert c.RUN_STATUS_TRANSITIONS[terminal] == frozenset()


def test_transitions_reject_terminal_exit_and_allow_continue() -> None:
    assert c.can_transition_run(c.RunStatus.QUEUED, c.RunStatus.RUNNING) is True
    assert c.can_transition_run(c.RunStatus.COMPLETED, c.RunStatus.RUNNING) is False
    assert c.can_transition_run(c.RunStatus.RUNNING, c.RunStatus.COMPLETED_WITH_GAPS) is True
    # 子任务失败必须能让父图离开 waiting_children（PRD 12.2）
    assert c.can_transition_run(c.RunStatus.WAITING_CHILDREN, c.RunStatus.FAILED) is True


def test_cancelled_resume_requires_resumable_flag() -> None:
    assert c.can_transition_run(c.RunStatus.CANCELLED, c.RunStatus.RUNNING, resumable=False) is False
    assert c.can_transition_run(c.RunStatus.CANCELLED, c.RunStatus.RUNNING, resumable=True) is True


def test_cancellable_statuses_match_prd_12_3_actions() -> None:
    assert c.CANCELLABLE_RUN_STATUSES == {
        c.RunStatus.QUEUED,
        c.RunStatus.RUNNING,
        c.RunStatus.WAITING_CHILDREN,
        c.RunStatus.WAITING_APPROVAL,
        c.RunStatus.WAITING_INPUT,
    }


def test_can_transition_task_mirrors_run_rules() -> None:
    assert c.can_transition_task(c.TaskStatus.QUEUED, c.TaskStatus.RUNNING) is True
    assert c.can_transition_task(c.TaskStatus.COMPLETED, c.TaskStatus.RUNNING) is False


# ---------------------------------------------------------------------
# 其余契约枚举（PRD 10.2 / 8.1 / 8.3 / 9.1 / 10.1 / 6.2 / 4.2 / 8.2 / 13.3）
# ---------------------------------------------------------------------


def test_approval_status_matches_prd_10_2() -> None:
    assert [s.value for s in c.ApprovalStatus] == [
        "pending",
        "approved",
        "denied",
        "superseded",
    ]


def test_input_request_status() -> None:
    assert {s.value for s in c.InputRequestStatus} == {"pending", "answered", "superseded"}


def test_source_types_follow_prd_8_1() -> None:
    assert {s.value for s in c.SourceType} == {
        "knowledge",
        "web",
        "memory",
        "history",
        "catalog",
        "product",
    }
    assert {s.value for s in c.EvidenceSourceType} == {
        "knowledge",
        "web",
        "memory",
        "history",
        "catalog",
        "product",
        "general_knowledge",
        "proposal",
    }


def test_answer_basis_matches_prd_8_3() -> None:
    assert {s.value for s in c.AnswerBasis} == {
        "knowledge",
        "web",
        "general",
        "product",
        "catalog",
        "memory",
        "history",
        "mixed",
        "insufficient",
    }


def test_permission_and_reasoning_levels() -> None:
    assert {s.value for s in c.PermissionMode} == {"ask", "auto", "full"}
    assert {s.value for s in c.ReasoningLevel} == {"off", "low", "high", "max"}
    assert c.DEFAULT_PERMISSION_MODE is c.PermissionMode.ASK
    assert c.DEFAULT_REASONING_LEVEL is c.ReasoningLevel.OFF


def test_route_and_agent_enums() -> None:
    assert {s.value for s in c.IntentKind} == {
        "general",
        "research",
        "product",
        "catalog",
        "memory",
        "clarification",
    }
    assert {s.value for s in c.Freshness} == {"none", "current", "date_range"}
    assert {s.value for s in c.AgentKind} == {
        "intent",
        "planner",
        "general",
        "rag",
        "web",
        "synthesis",
        "verifier",
        "approval_reviewer",
        "memory_extractor",
    }
    assert {s.value for s in c.VerifierVerdict} == {
        "pass",
        "revise",
        "research",
        "clarify",
        "insufficient",
    }


def test_event_types_match_prd_13_3() -> None:
    """事件类型包括处理过程中的模型思考增量。"""

    assert len(list(c.EventType)) == 21
    assert {e.value for e in c.EventType} == {
        "run.status",
        "route.decided",
        "plan.updated",
        "agent.started",
        "agent.completed",
        "model.reasoning",
        "tool.started",
        "tool.completed",
        "approval.requested",
        "approval.resolved",
        "input.requested",
        "input.resolved",
        "evidence.added",
        "verification.updated",
        "answer.meta",
        "answer.delta",
        "run.completed",
        "run.failed",
        "run.cancelled",
        "memory.updated",
        "memory.failed",
    }
    assert c.SUBSCRIPTION_CLOSING_EVENTS == {
        c.EventType.RUN_COMPLETED,
        c.EventType.RUN_FAILED,
        c.EventType.RUN_CANCELLED,
    }


def test_citation_prefix_matches_prd_8_3() -> None:
    assert c.citation_prefix(c.SourceType.KNOWLEDGE) == "S"
    assert c.citation_prefix(c.SourceType.WEB) == "W"
    assert c.citation_prefix(c.SourceType.MEMORY) == "M"
    assert c.citation_prefix(c.SourceType.HISTORY) == "H"
    # 目录 / 产品 / 通用解释 / 建议不生成文档引用编号（PRD 8.1 末句、8.3）
    assert c.citation_prefix(c.SourceType.CATALOG) is None
    assert c.citation_prefix(c.SourceType.PRODUCT) is None
    assert c.citation_prefix(c.EvidenceSourceType.GENERAL_KNOWLEDGE) is None
    assert c.citation_prefix(c.EvidenceSourceType.PROPOSAL) is None


def test_public_surface_is_declared() -> None:
    """``__all__`` 必须与冻结区实际定义一致，避免导出清单漂移。"""

    missing = [name for name in c.__all__ if not hasattr(c, name)]
    assert missing == []


# =====================================================================
# t2 追加区（T00b）：共享 Pydantic 模型、memory.json schema、工程默认值
# —— 上方 t1 冻结区用例只增不改（见架构文档 §6.3 第 16 条）。
# =====================================================================


from app.agent import events as agent_events
from app.agent.contracts import (
    MEMORY_ENTRY_KEYS,
    MEMORY_FILE_KEYS,
    MEMORY_FILE_SCHEMA_VERSION,
    AgentError,
    AnswerArtifact,
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalDisplay,
    ApprovalRequest,
    ApprovalScope,
    CitationEntry,
    ContractModel,
    Evidence,
    InputRequest,
    MemoryFile,
    MemoryFileEntry,
    MemoryInvalidation,
    MemoryInvalidationKind,
    ModelCapabilities,
    OperationRecord,
    ProviderMessage,
    ResearchTask,
    RouteDecision,
    ToolCall,
    VerificationResult,
    WorkspaceContext,
    WorkspaceServiceHub,
    dump_contract,
    utc_now_iso,
    validate_task_graph,
)


def _fields(model: type[ContractModel]) -> set[str]:
    return set(model.model_fields)


# ---------------------------------------------------------------------
# PRD 6.2 / 6.4 / 8.1 / 9.2 / 10.2 字段逐条对齐（§6.3 第 13–15 条）
# ---------------------------------------------------------------------


def test_route_decision_covers_prd_6_2() -> None:
    """PRD 6.2 表实际 12 行（任务描述写 11，见架构文档 §6 D16）。"""

    assert _fields(RouteDecision) == {
        "question",
        "resolved_question",
        "intent",
        "needs_knowledge",
        "needs_web",
        "needs_history",
        "needs_general",
        "freshness",
        "entities",
        "scope_evidence_ids",
        "unresolved_terms",
        "reason_summary",
    }


def test_research_task_covers_prd_6_4() -> None:
    """PRD 6.4 实际 9 个字段（任务描述写 8，见 D16）。"""

    assert _fields(ResearchTask) == {
        "task_id",
        "parent_task_id",
        "question",
        "requirement_ids",
        "agent_kind",
        "depends_on",
        "expected_evidence",
        "status",
            "result_ref",
            "query_identity",
            "query_reuse_reason",
        }


def test_evidence_covers_prd_8_1() -> None:
    """PRD 8.1 实际 12 个字段（任务描述写 11，见 D16）+ 纯追加的 ``document_title``。

    ``document_title`` 是为修复「文档身份在证据传递中丢失」（报告 F13：引用面板与模型
    只看到章节名，答案把 A 文章的章节归给 B 文章）而追加的字段；带默认值 ⇒ 既有构造与
    序列化不变，因此这里显式列出，避免再有人误删。
    """

    assert _fields(Evidence) == {
        "evidence_id",
        "workspace_id",
        "run_id",
        "task_id",
        "source_type",
        "source_ref",
        "content",
        "content_hash",
        "document_revision",
        "retrieved_at",
        "supported_requirements",
        "limitations",
            "heading_path",
            "document_title",
            "source_start",
            "source_end",
            "member_evidence_ids",
        }


def test_model_capabilities_covers_prd_9_2() -> None:
    assert _fields(ModelCapabilities) == {
        "supports_tools",
        "supports_thinking_toggle",
        "reasoning_levels",
        "supports_image_input",
        "context_limit",
        "max_output_tokens",
        "history_protocol",
    }


def test_approval_request_covers_prd_10_2() -> None:
    """PRD 10.2 全部字段 + 一处已记录的新增（superseded_reason，见 D17）。"""

    required = {
        "approval_id",
        "run_id",
        "task_id",
        "interrupt_id",
        "operation_id",
        "args_hash",
        "context_epoch",
        "rule_version",
        "display",
        "status",
        "decision",
    }
    assert required <= _fields(ApprovalRequest)
    assert _fields(ApprovalRequest) - required == {"superseded_reason", "scope", "created_at", "resolved_at"}


def test_other_contract_models_are_present_with_expected_fields() -> None:
    assert _fields(AnswerArtifact) >= {
        "answer_id",
        "run_id",
        "content",
        "citations",
        "verification",
        "content_hash",
        "answer_basis",
        "basis_components",
        "status",
    }
    assert _fields(InputRequest) >= {
        "input_request_id",
        "run_id",
        "question",
        "options",
        "status",
        "answer",
        "question_version",
    }
    assert _fields(ProviderMessage) == {
        "context_id",
        "context_epoch",
        "agent_context_epoch",
        "sequence",
        "protocol",
        "model",
        "params",
        "execution_id",
        "created_at",
    }
    assert _fields(OperationRecord) >= {
        "operation_id",
        "tool_name",
        "args_hash",
        "status",
        "attempts",
        "result_ref",
    }
    assert _fields(ToolCall) == {
        "index",
        "call_id",
        "name",
        "arguments",
        "raw_arguments",
        "complete",
    }
    assert _fields(AgentError) == {
        "error_code",
        "message",
        "node",
        "operation_id",
        "resumable",
        "required_action",
        "created_at",
    }


def test_answer_artifact_status_is_restricted_to_published_states() -> None:
    base = {
        "answer_id": "a1",
        "run_id": "r1",
        "content": "正文",
        "status": "running",
    }
    with pytest.raises(ValueError):
        AnswerArtifact.model_validate(base)
    assert AnswerArtifact.model_validate({**base, "status": "completed"}).status == "completed"
    assert (
        AnswerArtifact.model_validate({**base, "status": "completed_with_gaps"}).status
        == "completed_with_gaps"
    )


def test_contract_models_forbid_undeclared_fields() -> None:
    """PRD 12.5：结构化内容不合法就地失败，不能把多余字段当无事发生。"""

    with pytest.raises(ValueError):
        RouteDecision(
            question="q",
            resolved_question="q",
            intent="general",
            unexpected="x",  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError):
        ApprovalDisplay(purpose="p", tool_name="t", unknown=1)  # type: ignore[call-arg]


def test_approval_and_input_state_consistency_is_enforced() -> None:
    with pytest.raises(ValueError):
        ApprovalRequest(
            approval_id="a",
            run_id="r",
            operation_id="o",
            args_hash="h",
            status="approved",  # 已裁决却没有决定
        )
    with pytest.raises(ValueError):
        InputRequest(
            input_request_id="i",
            run_id="r",
            question="q",
            status="answered",  # 已回答却没有答案
        )
    decided = ApprovalRequest(
        approval_id="a",
        run_id="r",
        operation_id="o",
        args_hash="h",
        status="approved",
        decision=ApprovalDecision(kind=ApprovalDecisionKind.ALLOW_ONCE),
    )
    assert decided.decision is not None
    assert decided.decision.kind is ApprovalDecisionKind.ALLOW_ONCE


def test_verification_result_pass_is_the_only_publishable_verdict() -> None:
    assert VerificationResult(verdict="pass").is_publishable() is True
    assert VerificationResult(verdict="revise").is_publishable() is False
    assert VerificationResult(verdict="insufficient").is_publishable() is False


# ---------------------------------------------------------------------
# 任务图校验（PRD 6.4 末段：依赖环、缺失任务 ID 必须被拒）
# ---------------------------------------------------------------------


def _task(task_id: str, depends_on: list[str] | None = None, parent: str | None = None) -> ResearchTask:
    return ResearchTask(
        task_id=task_id, question=f"子问题 {task_id}", depends_on=depends_on or [], parent_task_id=parent
    )


def test_validate_task_graph_accepts_dag() -> None:
    validate_task_graph([_task("t1"), _task("t2", ["t1"]), _task("t3", ["t1", "t2"], parent="t1")])


def test_validate_task_graph_rejects_duplicate_missing_and_cycles() -> None:
    with pytest.raises(ValueError, match="重复"):
        validate_task_graph([_task("t1"), _task("t1")])
    with pytest.raises(ValueError, match="不存在"):
        validate_task_graph([_task("t1", ["nope"])])
    with pytest.raises(ValueError, match="不存在"):
        validate_task_graph([_task("t1", parent="nope")])
    with pytest.raises(ValueError, match="环"):
        validate_task_graph([_task("t1", ["t2"]), _task("t2", ["t1"])])
    with pytest.raises(ValueError, match="自身"):
        ResearchTask(task_id="t1", question="q", depends_on=["t1"])


# ---------------------------------------------------------------------
# memory.json schema（PRD 11.2，§6.3 第 18 条）
# ---------------------------------------------------------------------


def test_memory_schema_keys_match_prd_11_2() -> None:
    """PRD 11.2：文件级 9 字段、条目级 **11** 字段（§6.3 清单写 12，见 D16）。"""

    assert len(MEMORY_FILE_KEYS) == 9
    assert len(MEMORY_ENTRY_KEYS) == 11

    memory = MemoryFile()
    assert memory.schema_version == MEMORY_FILE_SCHEMA_VERSION
    assert set(dump_contract(memory)) == set(MEMORY_FILE_KEYS)

    entry = MemoryFileEntry(id="m1", kind="topic", text="讨论登录流程")
    assert set(dump_contract(entry)) == set(MEMORY_ENTRY_KEYS)


def test_memory_file_holds_entries_invalidations_cursors_and_operations() -> None:
    memory = MemoryFile(
        enabled=True,
        revision=3,
        invalidation_revision=1,
        entries=[
            MemoryFileEntry(id="m1", kind="response_preference", text="用中文回答", user_pinned=True)
        ],
        invalidations=[
            MemoryInvalidation(
                memory_id="m0",
                kind=MemoryInvalidationKind.CORRECTED,
                replacement="用中文回答，不要用英文术语",
                quote="用中文回答",
            )
        ],
        processed_cursors={"c1": "msg-9"},
        applied_operations=["op-1"],
    )
    payload = dump_contract(memory)
    assert payload["entries"][0]["user_pinned"] is True
    assert payload["invalidations"][0]["kind"] == "corrected"
    assert payload["processed_cursors"] == {"c1": "msg-9"}
    assert payload["applied_operations"] == ["op-1"]

    with pytest.raises(ValueError, match="superseded_by"):
        MemoryFileEntry(id="m2", kind="topic", text="x", status="superseded")


# ---------------------------------------------------------------------
# PRD 18.1 工程默认值（§6.3 第 17 条）
# ---------------------------------------------------------------------


def test_prd_18_1_defaults_are_centralized() -> None:
    assert c.DEFAULT_RESEARCH_AGENT_CONCURRENCY == 3
    assert c.DEFAULT_MODEL_REQUEST_CONCURRENCY == 4
    assert c.DEFAULT_CONTEXT_TOP_K == 8
    assert c.DEFAULT_MEMORY_MAX_CHARS == 300
    assert c.DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS == 8000
    assert c.DEFAULT_USER_INPUT_MAX_CHARS == 32000
    assert c.DEFAULT_CONTEXT_COMPACTION_RATIO == 0.8
    # v2（T_d29/D29）：workspaces 追加 pinned / order_index，老库按带默认值的加列升级。
    # v3（T_taskid/t31）：agent_tasks 身份改 run 作用域（PRIMARY KEY (run_id, id)），
    # 复合外键 + 幂等重建表升级。
    assert c.APP_DB_SCHEMA_VERSION == 3
    assert c.EVENT_SCHEMA_VERSION == agent_events.EVENT_SCHEMA_VERSION == 2


# ---------------------------------------------------------------------
# WorkspaceContext 视图与实现层服务句柄（PRD 5.1 / 5.2）
# ---------------------------------------------------------------------


def test_workspace_context_paths_split_shared_knowledge_and_private(tmp_path) -> None:
    """D-KB：视图的路径派生与 ``WorkspacePaths`` 一致 —— 知识共享、私有件按区。"""

    context = WorkspaceContext.build(
        workspace_id="w1", storage_key="sk1", data_dir=tmp_path, name="风控组"
    )
    # 知识：共享（data_dir 根），任何工作区相同
    assert context.documents_dir == tmp_path / "documents"
    assert context.state_db == tmp_path / "state.db"
    assert context.image_state_db == tmp_path / "image_state.db"
    assert context.chroma_dir == tmp_path
    # 私有：本区根
    assert context.root_dir == tmp_path / "workspaces" / "sk1"
    assert context.memory_json == context.root_dir / "memory.json"
    assert context.checkpoints_db == context.root_dir / "checkpoints.db"
    assert context.artifacts_dir == context.root_dir / "artifacts"
    assert context.jieba_cache_dir == context.root_dir / "cache" / "jieba"
    assert context.default_permission_mode is c.PermissionMode.ASK
    assert context.default_reasoning_level is c.ReasoningLevel.OFF

    other = WorkspaceContext.build(
        workspace_id="w2", storage_key="sk2", data_dir=tmp_path, name="营销组"
    )
    for attribute in ("documents_dir", "state_db", "image_state_db", "chroma_dir"):
        assert getattr(other, attribute) == getattr(context, attribute), attribute
    for attribute in ("memory_json", "checkpoints_db", "artifacts_dir", "jieba_cache_dir"):
        assert getattr(other, attribute) != getattr(context, attribute), attribute


def test_workspace_context_is_frozen_and_serializable(tmp_path) -> None:
    context = WorkspaceContext.build(workspace_id="w1", storage_key="sk1", data_dir=tmp_path)
    with pytest.raises(ValueError):
        context.knowledge_revision = 9  # type: ignore[misc]

    payload = dump_contract(context)
    # 服务实例不进契约：可序列化字段里没有 app_store / retriever / embedder
    assert "app_store" not in payload
    assert set(payload) == {
        "workspace_id",
        "storage_key",
        "name",
        "status",
        "knowledge_revision",
        "default_permission_mode",
        "default_reasoning_level",
        "memory_enabled",
        "data_dir",
        "root_dir",
    }


def test_workspace_service_hub_declares_implementation_members() -> None:
    """PRD 5.2 的服务实例由实现层承担，成员名由本 Protocol 冻结（见 D19）。"""

    assert set(WorkspaceServiceHub.__annotations__) == {"view", "app_store"}


def test_utc_now_iso_is_second_precision_utc() -> None:
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert len(stamp) == 20
    import datetime as _datetime

    parsed = _datetime.datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0


# ---------------------------------------------------------------------
# SSE 事件契约（PRD 13.3，§6.3 第 19–20 条）
# ---------------------------------------------------------------------


def test_every_event_type_has_a_payload_model() -> None:
    assert len(agent_events.EVENT_PAYLOAD_MODELS) == len(list(c.EventType)) == 21
    for event_type in c.EventType:
        assert agent_events.payload_model_for(event_type) is not None
        assert agent_events.payload_model_for(event_type.value) is not None


def test_event_envelope_fields_match_prd_13_3() -> None:
    assert set(agent_events.EventEnvelope.model_fields) == {
        "schema_version",
        "run_id",
        "event_seq",
        "execution_id",
        "task_id",
        "type",
        "created_at",
        "payload",
    }
    envelope = agent_events.EventEnvelope(
        run_id="r1", event_seq=1, execution_id="e1", type="run.status", payload={"status": "running"}
    )
    assert envelope.schema_version == 2
    assert agent_events.EventEnvelope.model_fields["schema_version"].default == 2


def test_unknown_event_version_and_payload_raise_protocol_error() -> None:
    from app.agent.contracts import EventType

    # 合法事件不抛：先确认这条断言不是「什么都抛」的假阳性
    ok = agent_events.build_event(
        run_id="r1", event_seq=1, type=EventType.RUN_STATUS, payload={"status": "running"}
    )
    assert ok.payload["status"] == "running"

    with pytest.raises(agent_events.EventProtocolError):
        agent_events.build_event(run_id="r1", event_seq=1, type="unknown.type", payload={})
    with pytest.raises(agent_events.EventProtocolError):
        agent_events.parse_event(
            {
                "schema_version": 1,
                "run_id": "r1",
                "event_seq": 1,
                "execution_id": "e1",
                "type": "run.status",
                "created_at": "t",
                "payload": {"status": "running"},
            }
        )
    with pytest.raises(agent_events.EventProtocolError) as excinfo:
        agent_events.build_event(run_id="r1", event_seq=1, type="unknown.type", payload={})
    # 保留原始异常，便于诊断；但类型必须是协议错误（PRD 12.5 末段、13.3 末段）
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_citation_entry_and_scope_models() -> None:
    entry = CitationEntry(
        label="W1",
        source_type=c.SourceType.WEB,
        evidence_id="ev-1",
        workspace_id="w1",
        source_status="archived",
    )
    assert entry.source_status is c.CitationSourceStatus.ARCHIVED
    scope = ApprovalScope(
        run_id="r1", tool_category="web_search", question_scope="今天新闻", rule_version="v1"
    )
    assert scope.tool_category == "web_search"


# ---------------------------------------------------------------------
# 网页证据升格为共享契约（PRD 7.3，架构文档 §6 D32）
# + t19（T_guard）永久守卫：单一来源 = contracts.py，禁止 tools.py 本地重定义
#   （收口说明见架构文档 §6 D33；字段清单/往返断言由 t15 落地，本段不重复）
# ---------------------------------------------------------------------

_WEB_READ_RANGE_FIELDS = {"offset", "length", "total", "complete"}
_WEB_PAGE_EVIDENCE_FIELDS = {
    "url",
    "final_url",
    "title",
    "heading_path",
    "published_at",
    "retrieved_at",
    "content_hash",
    "artifact_id",
    "read_range",
    "content_complete",
    "content",
}


def test_web_read_range_covers_prd_7_3() -> None:
    assert _fields(c.WebReadRange) == _WEB_READ_RANGE_FIELDS
    assert len(_fields(c.WebReadRange)) == 4


def test_web_page_evidence_covers_prd_7_3() -> None:
    assert _fields(c.WebPageEvidence) == _WEB_PAGE_EVIDENCE_FIELDS
    assert len(_fields(c.WebPageEvidence)) == 11


def test_web_page_evidence_round_trips_and_keeps_defaults() -> None:
    """往返一致 + 缺失来源字段保存为空 + 读取范围的 ge=0 校验未被升格改掉。"""

    payload = {
        "url": "https://example.com/news/1",
        "final_url": "https://example.com/news/1?utm=x",
        "title": "AI 新闻",
        "heading_path": "产品 > AI 新闻",
        "published_at": "2026-09-18",
        "retrieved_at": "2026-09-19T08:00:00Z",
        "content_hash": "hash-1",
        "artifact_id": "art-1",
        "read_range": {"offset": 0, "length": 3, "total": 10, "complete": False},
        "content_complete": False,
        "content": "abc",
    }
    evidence = c.WebPageEvidence.model_validate(payload)
    dumped = evidence.model_dump()
    assert dumped == payload
    assert c.WebPageEvidence.model_validate(dumped) == evidence
    assert isinstance(evidence.read_range, c.WebReadRange)

    # PRD 7.3 末段：缺失的来源字段保存为空（不是 None）
    empty = c.WebPageEvidence()
    assert empty.model_dump() == {
        "url": "",
        "final_url": "",
        "title": "",
        "heading_path": "",
        "published_at": "",
        "retrieved_at": "",
        "content_hash": "",
        "artifact_id": "",
        "read_range": {"offset": 0, "length": 0, "total": 0, "complete": False},
        "content_complete": False,
        "content": "",
    }

    with pytest.raises(ValueError):
        c.WebReadRange(offset=-1)
    with pytest.raises(ValueError):
        c.WebPageEvidence.model_validate({**payload, "unknown_field": 1})


def test_tools_web_evidence_is_the_contract_class_not_a_local_copy() -> None:
    """守卫（t15/t19）：`tools.py` 必须**从 contracts 导入并再导出**，不得回退成本地类。

    这一条防的是「同构漂移」——本项目已真实发生过一次
    （`research.py::WebPageReading` 当时是 `WebPageEvidence` 的本地投影；
    现已在 T06 收敛为薄再导出，见架构文档 §6 D33）。
    断言分三层：

    1. **身份**：``tools.X is contracts.X``（同一对象，不是同构副本）；
    2. **源码**：``app/agent/tools.py`` 里不得出现 ``class WebPageEvidence`` /
       ``class WebReadRange``（读源码文本，避免依赖导入副作用）；
    3. **再导出面**：两个名字仍在 ``tools.__all__`` 里（``import *`` 不丢公开面），
       且默认 dump 与同一 payload 的校验结果与契约层逐字一致。
    """

    from app.agent import tools as agent_tools

    # 1. 身份
    assert agent_tools.WebPageEvidence is c.WebPageEvidence
    assert agent_tools.WebReadRange is c.WebReadRange
    assert agent_tools.WebPageEvidence.__module__ == "app.agent.contracts"
    assert agent_tools.WebReadRange.__module__ == "app.agent.contracts"

    # 2. 源码守卫：不允许重新出现本地类定义
    source = _TOOLS_SOURCE_PATH.read_text(encoding="utf-8")
    local_definitions = re.findall(
        r"^\s*class\s+(WebPageEvidence|WebReadRange)\b", source, re.MULTILINE
    )
    assert local_definitions == [], (
        f"app/agent/tools.py 重新定义了本地类 {local_definitions}；"
        "网页证据必须只有 contracts.py 一个来源（架构文档 §6 D32/D33）"
    )

    # 3. 再导出面与形状
    assert "WebPageEvidence" in agent_tools.__all__
    assert "WebReadRange" in agent_tools.__all__
    assert agent_tools.WebPageEvidence().model_dump() == c.WebPageEvidence().model_dump()
    assert agent_tools.WebReadRange().model_dump() == c.WebReadRange().model_dump()

    payload = {"url": "https://example.com/a", "content": "x", "read_range": {"total": 1}}
    from_tools = agent_tools.WebPageEvidence.model_validate(payload)
    from_contracts = c.WebPageEvidence.model_validate(payload)
    assert from_tools.model_dump() == from_contracts.model_dump()
    assert from_tools == from_contracts


def test_web_evidence_guard_is_not_vacuous() -> None:
    """守卫非永真证明：同一套检查确实能识别「本地重定义」的形态。

    不做这一步，守卫可能写成恒真断言（例如误比对两个都指向契约层的名字），
    出问题时也不会报警。这里用合成源码片段 + 一个同名副本类验证检查有效。
    """

    from app.agent import tools as agent_tools

    snippet = (
        "class WebReadRange(ContractModel):\n"
        "    offset: int = 0\n"
        "\n"
        "class WebPageEvidence(ContractModel):\n"
        "    url: str = ''\n"
    )
    detected = re.findall(
        r"^\s*class\s+(WebPageEvidence|WebReadRange)\b", snippet, re.MULTILINE
    )
    assert detected == ["WebReadRange", "WebPageEvidence"]

    class WebPageEvidence:  # 故意模拟「本地重定义」：与契约层同名但不同对象
        pass

    assert WebPageEvidence is not c.WebPageEvidence  # 身份断言能区分副本
    assert agent_tools.WebPageEvidence is c.WebPageEvidence  # 真实 tools 仍是契约层那一个

