"""app.db ↔ 存档的**工作区行自愈**（D25 过渡期与迁移后共用一处实现）。

背景：`app.db` 的 `workspaces` 行是 run API / 会话行的外键前提（`runs.workspace_id
REFERENCES workspaces(id)`）。而 D25 规定过渡期 ``workspaces.json`` 是**唯一合法来源**、
``app.db`` 是待迁目标。两者叠加出的现实是：**只要 app.db 里缺工作区行，run API 就会
404**（实测：全新数据目录里 `POST /api/conversations` → 201，随后 `POST /runs` → 404）。

因此这里提供唯一实现 :func:`ensure_workspace_row`：app.db 缺行时，**按存档这条合法来源**
补齐（含默认权限/强度/知识版本/置顶与位次），再把权威行单向镜像回存档。它同时服务：

- ``app/api/routes/conversations.py``：工作区写端点（t24 的 `_ensure_default_workspace`
  现在转发到这里）与新会话创建；
- ``app/api/routes/runs.py``：run API 的工作区校验（缺行即自愈，而不是把 404 抛给用户）。

边界：本模块**不做**排序/投影，也不碰 conversations 表；只负责「workspaces 行存在」。
"""

from __future__ import annotations

from typing import Any

from app.agent.contracts import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_REASONING_LEVEL,
    PermissionMode,
    ReasoningLevel,
    WorkspaceStatus,
)

__all__ = ["WorkspaceRowMissingError", "ensure_workspace_row"]


class WorkspaceRowMissingError(LookupError):
    """app.db 与存档都没有该工作区（调用方据此返回 404）。"""

    def __init__(self, workspace_id: str) -> None:
        super().__init__(f"工作区不存在：{workspace_id}")
        self.workspace_id = workspace_id


async def ensure_workspace_row(
    app_store: Any,
    store: Any,
    workspace_id: str,
    *,
    allow_archive_missing: bool = False,
    storage_key_fallback: str | None = None,
    name_fallback: str = "",
) -> Any:
    """确保 app.db 有 ``workspace_id`` 行；缺行时按存档补齐并镜像回来。

    ``allow_archive_missing``（默认区专用）：连存档都没有该区时也建行（t24 的
    ``_ensure_default_workspace`` 语义），``storage_key_fallback`` 必须给出。
    """

    record = await app_store.get_workspace(workspace_id)
    if record is not None:
        return record

    archived = store.get(workspace_id)
    if archived is None and not allow_archive_missing:
        raise WorkspaceRowMissingError(workspace_id)

    storage_key = (getattr(archived, "storage_key", None) if archived else None) or storage_key_fallback
    if not storage_key:
        raise WorkspaceRowMissingError(workspace_id)

    record = await app_store.create_workspace(
        workspace_id,
        storage_key,
        name=(getattr(archived, "name", None) if archived else None) or name_fallback,
        status=WorkspaceStatus(
            str(getattr(archived, "status", WorkspaceStatus.ACTIVE)) if archived else "active"
        ),
        # 存档里的默认值必须带过来：否则自愈会把用户在设置页写下的工作区默认值
        # 悄悄重置回 PRD 18.1 兜底（改默认值却在本区不生效）。
        default_permission_mode=PermissionMode(
            str(getattr(archived, "default_permission_mode", DEFAULT_PERMISSION_MODE))
            if archived
            else str(DEFAULT_PERMISSION_MODE)
        ),
        default_reasoning_level=ReasoningLevel(
            str(getattr(archived, "default_reasoning_level", DEFAULT_REASONING_LEVEL))
            if archived
            else str(DEFAULT_REASONING_LEVEL)
        ),
        knowledge_revision=int(getattr(archived, "knowledge_revision", 0) or 0) if archived else 0,
        pinned=bool(getattr(archived, "pinned", False)) if archived else False,
        order_index=getattr(archived, "order_index", None) if archived else None,
    )
    store.adopt_record(record)
    return record
