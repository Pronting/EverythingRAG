"""权限 / 思考强度的**三层优先级解析**（PRD 1.3 + 14.1 + 18.1）。T_defaults。

PRD 1.3「权限与思考强度具有**工作区默认值**和**会话保存值**」，14.1「按钮展示当前会话
实际设置；**工作区设置页提供默认值**」，18.1 给出工程兜底（会话权限 ``ask``、思考强度
``high``）。合起来就是本模块实现的优先级：

**新会话**：``工作区默认值 → 18.1 兜底``（工作区默认值就是「工作区设置页」写下的值，见
``PUT /api/workspaces/{id}/defaults``）。

**新 run**：``请求显式值 → 会话保存值 → 工作区默认值 → 18.1 兜底``。

要点（每条都在测试里钉住）：

1. **两个轴各自独立解析**：只显式给了 ``reasoning_level`` 时，``permission_mode`` 仍按
   后面几层解析，不会因为「请求里出现了 setting」就整体跳过。
2. **会话保存值优先于工作区默认值**（PRD 1.3）：已保存的会话值不会被工作区默认值改写 ——
   因此改工作区默认值**只影响之后新建的会话/run**（PRD 14.1 的生效边界）。
3. **兜底不是硬编码在调用点**：唯一来源是 ``contracts.DEFAULT_PERMISSION_MODE`` /
   ``DEFAULT_REASONING_LEVEL``（PRD 18.1），本模块不重复字面量。
4. 解析是**纯函数**、不碰 IO：调用方只负责把「工作区视图 / 会话行 / 请求体」读出来，
   便于单测与离线复核（同一份规则不会被三个端点各写一遍）。
"""

from __future__ import annotations

from typing import Any, NamedTuple

from app.agent.contracts import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_REASONING_LEVEL,
    PermissionMode,
    ReasoningLevel,
)

__all__ = [
    "ResolvedScope",
    "resolve_conversation_settings",
    "resolve_run_settings",
]


def _as_permission(value: Any) -> PermissionMode | None:
    """把任意来源收敛成 ``PermissionMode``；None / 空串 / 非法值都返回 None。"""

    if value is None or value == "":
        return None
    if isinstance(value, PermissionMode):
        return value
    try:
        return PermissionMode(str(value))
    except ValueError:
        return None


def _as_reasoning(value: Any) -> ReasoningLevel | None:
    if value is None or value == "":
        return None
    if isinstance(value, ReasoningLevel):
        return value
    try:
        return ReasoningLevel(str(value))
    except ValueError:
        return None


def _view_field(view: Any, name: str) -> Any:
    """读工作区视图 / 记录上的默认值字段（视图与 WorkspaceRecord 都接受）。"""

    if view is None:
        return None
    if isinstance(view, dict):
        return view.get(name)
    return getattr(view, name, None)


class ResolvedScope(NamedTuple):
    """解析结果 + 每个轴的来源（便于诊断与断言，不参与业务逻辑）。"""

    permission_mode: PermissionMode
    reasoning_level: ReasoningLevel
    permission_source: str
    reasoning_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "permission_mode": str(self.permission_mode),
            "reasoning_level": str(self.reasoning_level),
            "permission_source": self.permission_source,
            "reasoning_source": self.reasoning_source,
        }


#: 来源名（写进诊断/测试断言；不含用户数据）
SOURCE_REQUEST = "request"
SOURCE_CONVERSATION = "conversation"
SOURCE_WORKSPACE = "workspace_default"
SOURCE_FALLBACK = "prd_18_1_fallback"


def resolve_conversation_settings(view: Any = None) -> ResolvedScope:
    """新会话的设置：``工作区默认值 → PRD 18.1 兜底``。"""

    workspace_permission = _as_permission(_view_field(view, "default_permission_mode"))
    workspace_reasoning = _as_reasoning(_view_field(view, "default_reasoning_level"))
    return ResolvedScope(
        permission_mode=workspace_permission or DEFAULT_PERMISSION_MODE,
        reasoning_level=workspace_reasoning or DEFAULT_REASONING_LEVEL,
        permission_source=SOURCE_WORKSPACE if workspace_permission else SOURCE_FALLBACK,
        reasoning_source=SOURCE_WORKSPACE if workspace_reasoning else SOURCE_FALLBACK,
    )


def resolve_run_settings(
    *,
    requested_permission_mode: Any = None,
    requested_reasoning_level: Any = None,
    conversation_permission_mode: Any = None,
    conversation_reasoning_level: Any = None,
    view: Any = None,
) -> ResolvedScope:
    """新 run 的设置：``请求显式 → 会话保存 → 工作区默认 → 18.1 兜底``（逐轴）。"""

    requested_permission = _as_permission(requested_permission_mode)
    requested_reasoning = _as_reasoning(requested_reasoning_level)
    saved_permission = _as_permission(conversation_permission_mode)
    saved_reasoning = _as_reasoning(conversation_reasoning_level)
    workspace_permission = _as_permission(_view_field(view, "default_permission_mode"))
    workspace_reasoning = _as_reasoning(_view_field(view, "default_reasoning_level"))

    permission, permission_source = _pick(
        (requested_permission, SOURCE_REQUEST),
        (saved_permission, SOURCE_CONVERSATION),
        (workspace_permission, SOURCE_WORKSPACE),
        fallback=DEFAULT_PERMISSION_MODE,
    )
    reasoning, reasoning_source = _pick(
        (requested_reasoning, SOURCE_REQUEST),
        (saved_reasoning, SOURCE_CONVERSATION),
        (workspace_reasoning, SOURCE_WORKSPACE),
        fallback=DEFAULT_REASONING_LEVEL,
    )
    return ResolvedScope(
        permission_mode=permission,
        reasoning_level=reasoning,
        permission_source=permission_source,
        reasoning_source=reasoning_source,
    )


def _pick(
    *candidates: tuple[Any, str], fallback: Any
) -> tuple[Any, str]:
    """按顺序取第一个非 None 的候选；都没有就用 PRD 18.1 兜底。"""

    for value, source in candidates:
        if value is not None:
            return value, source
    return fallback, SOURCE_FALLBACK
