"""权限模式测试（PRD 10.1 / 10.4，验收 A13 / A16 / A29）。

覆盖：三种模式的判定矩阵、默认 ``ask``、10 个工具到授权类别的映射、
「运行期间改权限在**下一次工具执行前**生效」（读会话当前值而不是启动时快照）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import contracts as c
from app.agent.approval import (
    PERMISSION_ALLOW_NOW,
    PERMISSION_DENY,
    PERMISSION_NEED_APPROVAL,
    TOOL_CATEGORIES,
    decide_permission,
    granted_categories,
    scope_matches,
    tool_category,
)
from app.agent.contracts import ApprovalScope, PermissionMode, RunStatus
from app.agent.tools import ToolName
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, build_registry, seed_conversation

# =====================================================================
# 纯判定
# =====================================================================


def test_default_mode_is_ask() -> None:
    assert c.DEFAULT_PERMISSION_MODE is PermissionMode.ASK


def test_decision_matrix() -> None:
    # 不需要批准的工具：任何模式都直接执行
    for mode in ("ask", "auto", "full"):
        assert (
            decide_permission(mode=mode, requires_approval=False) == PERMISSION_ALLOW_NOW
        )
    # 需要批准 + ask：停顿
    assert (
        decide_permission(mode="ask", requires_approval=True, category="web")
        == PERMISSION_NEED_APPROVAL
    )
    # auto / full：自动放行
    for mode in ("auto", "full"):
        assert (
            decide_permission(mode=mode, requires_approval=True, category="web")
            == PERMISSION_ALLOW_NOW
        )
    # ask + 已授权类别：放行
    assert (
        decide_permission(
            mode="ask", requires_approval=True, category="web", granted_categories=["web"]
        )
        == PERMISSION_ALLOW_NOW
    )
    # 被拒绝过：任何模式都不执行
    for mode in ("ask", "auto", "full"):
        assert (
            decide_permission(mode=mode, requires_approval=True, category="web", denied=True)
            == PERMISSION_DENY
        )


def test_every_tool_has_a_category() -> None:
    assert len(ToolName) == 10
    for name in ToolName:
        assert tool_category(str(name)) in set(TOOL_CATEGORIES.values())


def test_scope_matching_requires_all_four_fields() -> None:
    scope = ApprovalScope(
        run_id="run-1", tool_category="web", question_scope="今天新闻", rule_version="v1"
    )
    base = {
        "run_id": "run-1",
        "category": "web",
        "question_scope": "今天新闻",
        "rule_version": "v1",
    }
    assert scope_matches(scope, **base)
    for field, value in (
        ("run_id", "run-2"),
        ("category", "knowledge"),
        ("question_scope", "另一个问题"),
        ("rule_version", "v2"),
    ):
        changed = {**base, field: value}
        assert not scope_matches(scope, **changed), field


def test_granted_categories_reads_state_dumps() -> None:
    grants = [
        ApprovalScope(
            run_id="r", tool_category="web", question_scope="q", rule_version="v"
        ).model_dump(mode="json")
    ]
    assert granted_categories(grants) == ["web"]


# =====================================================================
# 运行层：默认 ask + 运行期改权限即时生效
# =====================================================================


@pytest.fixture
def registry():
    return build_registry()


async def test_run_inherits_conversation_permission_mode(app_store, tmp_path: Path) -> None:
    await seed_conversation(app_store)
    await app_store.update_conversation("conv-1", permission_mode=PermissionMode.AUTO)
    manager = build_manager(app_store, services=FakeServices(task_count=1), root=tmp_path)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="问题"
        )
        assert run.permission_mode is PermissionMode.AUTO
    finally:
        await manager.aclose()


async def test_permission_change_takes_effect_before_next_tool_call(
    app_store, tmp_path: Path, registry
) -> None:
    """会话权限改成 full 后，联网工具不再需要批准（PRD 10.1 运行期重检）。"""

    await seed_conversation(app_store)
    services = FakeServices(task_count=1, approve_tool="web_search")
    manager = build_manager(
        app_store, services=services, root=tmp_path, registry=registry
    )
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="张三最近怎么样"
        )
        # 启动前把会话权限改为 full：本次运行的工具执行应当免批准
        await app_store.update_conversation("conv-1", permission_mode=PermissionMode.FULL)
        await manager.run_to_completion(run.id)
        current = await manager.require_run(run.id)
        assert current.status is RunStatus.COMPLETED
        assert await app_store.list_approvals(run.id) == []
        assert registry.call_log == ["web_search"]
    finally:
        await manager.aclose()


async def test_ask_mode_blocks_until_approved(app_store, tmp_path: Path, registry) -> None:
    """默认 ask：需要批准的工具在批准前一次都不执行（A01 的结构性保证）。"""

    await seed_conversation(app_store)
    services = FakeServices(task_count=1, approve_tool="web_search")
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    try:
        run = await manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question="张三最近怎么样"
        )
        await manager.run_to_completion(run.id)
        assert (await manager.require_run(run.id)).status is RunStatus.WAITING_APPROVAL
        assert registry.call_log == []
    finally:
        await manager.aclose()
