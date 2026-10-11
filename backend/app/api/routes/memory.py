"""工作区记忆与历史检索 API（PRD 13.2，Owner: T08）。

六个端点与 PRD 13.2 表格逐行对应：

| 端点 | 语义 | 副作用 |
|---|---|---|
| ``GET    /api/workspaces/{w}/memory`` | 当前条目、revision、开关 | 无 |
| ``POST   /api/workspaces/{w}/memory`` | 用户明确要求保存的内容 | 新增（``user_pinned``），**不**失效快照 |
| ``PATCH  /api/workspaces/{w}/memory`` | 开关及预期 revision | 关闭时快照置空 |
| ``PATCH  /api/workspaces/{w}/memory/{id}`` | 修改内容及预期 revision | 人工纠正 → 快照失效并重建 |
| ``DELETE /api/workspaces/{w}/memory/{id}`` | 预期 revision | 遗忘 → 快照失效并重建 |
| ``GET    /api/workspaces/{w}/history`` | 查询、日期、分页 | 无 |

三条贯穿全部端点的规则：

1. **归属只由路径参数决定**。这里**不**用 ``Depends(get_workspace_context)``：
   那个 provider 的 ``workspace_id`` 在依赖注入下会被 FastAPI 当成**查询参数**，
   于是``/workspaces/A/memory`` 会静默解析成默认工作区 —— 跨区读到别人的记忆。
   每个处理器都显式 ``_context(workspace_id)``，工作区不存在直接 404。
2. **乐观并发**：写端点都接受可选 ``expected_revision``，不一致返回 409 而不是静默覆盖。
   记忆是用户的东西，"最后一次写入获胜"会吞掉用户在前一个界面上的编辑。
3. **不回绝对路径**：响应里不出现 ``memory.json`` / ``app.db`` 的磁盘路径
   （与既有 ``test_overview_api_never_exposes_absolute_paths`` 同一约定）。

开关放在 ``PATCH /api/workspaces/{w}/memory`` 而不是工作区改名接口：PRD 11.2 把
``enabled`` 定义在 ``memory.json`` 里，``Workspace``（``workspaces.json``，T01 拥有）
没有该字段 —— 不在这里另造第二份真相。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query

from app.agent.contracts import MEMORY_FILE_SCHEMA_VERSION, MemoryKind, utc_now_iso
from app.agent.memory import (
    InvalidationRuntimeOutcome,
    apply_invalidation_to_runtime,
    blocked_source_conversations,
)
from app.api.deps import (
    get_app_store,
    get_workspace_context,
    workspace_history_search,
    workspace_memory,
)
from app.core.workspace_context import WorkspaceNotFoundError
from app.memory.extractor import resolve_kind
from app.memory.history import HistorySearchError
from app.memory.locking import workspace_async_lock
from app.memory.workspace_memory import (
    MemoryDisabledError,
    MemoryFileError,
    MemoryMutationResult,
    MemoryRevisionConflict,
    WorkspaceMemory,
    build_snapshot,
)

router = APIRouter()

#: ``POST /memory`` 未显式给 kind 时，用回答要求线索判断是"要求"还是"主题"。
_ANSWER_REQUIREMENT_HINTS = (
    "回答", "请用", "语气", "格式", "中文", "英文", "简洁", "详细", "举例",
    "字数", "长度", "口吻", "排版", "要点",
)


def _memory_for(workspace_id: str) -> WorkspaceMemory:
    """按路径参数解析工作区并取记忆句柄（工作区不存在 → 404）。"""

    try:
        context = get_workspace_context(workspace_id)
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="工作区不存在") from exc
    return workspace_memory(context)


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=detail)


def _entry_error(exc: MemoryFileError) -> HTTPException:
    """``MemoryFileError`` 的 HTTP 映射：条目不存在 404，其它按服务端错误。"""

    message = str(exc)
    if "不存在" in message:
        return HTTPException(status_code=404, detail=message)
    return HTTPException(status_code=500, detail=message)


def _runtime_payload(outcome: InvalidationRuntimeOutcome) -> dict[str, Any]:
    return {
        "affected_conversations": list(outcome.affected_conversations),
        "context_epochs": dict(outcome.context_epochs),
        "cleared_snapshots": list(outcome.cleared_snapshots),
        "rebuilt_snapshots": list(outcome.rebuilt_snapshots),
        "superseded_approvals": outcome.superseded_approvals,
        "bumped_task_ids": list(outcome.bumped_task_ids),
        "disabled": outcome.disabled,
        "applied_at": utc_now_iso(),
    }


async def _apply_runtime(
    *,
    memory: WorkspaceMemory,
    mutation: MemoryMutationResult,
    reason: str,
) -> InvalidationRuntimeOutcome:
    """把文件侧的失效同步到 ``app.db``。"""

    app_store = get_app_store()
    if app_store is None:
        raise RuntimeError("工作区记忆需要已初始化的 AppStore")
    return await apply_invalidation_to_runtime(
        app_store=app_store, memory=memory, mutation=mutation, reason=reason
    )


async def _commit_invalidation(
    *,
    memory: WorkspaceMemory,
    mutate: Callable[[], MemoryMutationResult],
    reason: str,
) -> tuple[MemoryMutationResult, InvalidationRuntimeOutcome]:
    if get_app_store() is None:
        raise RuntimeError("工作区记忆需要已初始化的 AppStore")
    async with workspace_async_lock(memory.storage_key):
        result = mutate()
        outcome = await _apply_runtime(memory=memory, mutation=result, reason=reason)
        return result, outcome


# ===========================================================================
# GET /api/workspaces/{workspace_id}/memory
# ===========================================================================


@router.get("/api/workspaces/{workspace_id}/memory")
async def read_memory(workspace_id: str) -> dict[str, Any]:
    """当前条目、revision、开关（PRD 13.2 第 1 行）。"""

    memory = _memory_for(workspace_id)
    state = memory.load()
    app_store = get_app_store()
    blocked = (
        frozenset()
        if app_store is None
        else await blocked_source_conversations(app_store, workspace_id)
    )
    preview = build_snapshot(state)
    return {
        "workspace_id": workspace_id,
        "storage_key": memory.storage_key,
        "schema_version": state.schema_version or MEMORY_FILE_SCHEMA_VERSION,
        "enabled": state.enabled,
        "revision": state.revision,
        "invalidation_revision": state.invalidation_revision,
        "updated_at": state.updated_at,
        "entries": [entry.model_dump(mode="json") for entry in state.entries],
        "invalidations": [
            invalidation.model_dump(mode="json") for invalidation in state.invalidations
        ],
        "processed_cursors": dict(state.processed_cursors),
        "applied_operations": len(state.applied_operations),
        # 只回"当下工作区记忆"的度量，不回正文：本会话真正使用的是会话上**固定**的快照。
        "snapshot_preview": {
            "revision": preview.revision,
            "invalidation_revision": preview.invalidation_revision,
            "hash": preview.hash,
            "chars": len(preview.text),
            "truncated": preview.truncated,
            "overflow_count": preview.overflow_count,
            "needs_tidy": preview.needs_tidy,
        },
        "blocked_source_conversations": sorted(blocked),
    }


# ===========================================================================
# POST /api/workspaces/{workspace_id}/memory —— 主动记忆
# ===========================================================================


@router.post("/api/workspaces/{workspace_id}/memory", status_code=201)
async def create_memory(
    workspace_id: str,
    payload: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """用户明确要求保存的内容（PRD 13.2 第 6 行）。

    用户主动保存的条目 **不截断**（PRD 11.2：超限时由界面给出整理入口），
    也**不递增 invalidation_revision**：新增记忆不是失效事件，已有会话的快照保持
    不变（PRD 11.3）。
    """

    memory = _memory_for(workspace_id)
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text 不能为空")
    requested_kind = payload.get("kind")
    if requested_kind:
        try:
            kind = MemoryKind(str(requested_kind))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"未知 kind：{requested_kind}") from exc
    else:
        folded = text.casefold()
        guessed = (
            MemoryKind.RESPONSE_PREFERENCE
            if any(hint in folded for hint in _ANSWER_REQUIREMENT_HINTS)
            else MemoryKind.TOPIC
        )
        # 用户主动保存的"要求"也过一遍同一闸门，避免把普通主题写成要求。
        resolved_kind, _ = resolve_kind(str(guessed), text, text)
        kind = MemoryKind(resolved_kind)

    try:
        result = memory.add_entry(
            text,
            kind,
            source_conversation_id=str(payload.get("conversation_id") or ""),
            user_pinned=True,
            conversation_id=payload.get("conversation_id"),
            expected_revision=payload.get("expected_revision"),
        )
    except MemoryRevisionConflict as exc:
        raise _conflict(str(exc)) from exc
    except MemoryDisabledError as exc:
        raise _conflict(str(exc)) from exc
    except MemoryFileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "applied": result.applied,
        "reason": result.reason,
        "revision": result.memory.revision,
        "invalidation_revision": result.memory.invalidation_revision,
        "enabled": result.memory.enabled,
        "entry": None if result.entry is None else result.entry.model_dump(mode="json"),
    }


# ===========================================================================
# PATCH /api/workspaces/{workspace_id}/memory —— 开关
# ===========================================================================


@router.patch("/api/workspaces/{workspace_id}/memory")
async def update_memory_switch(
    workspace_id: str,
    payload: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """开启或关闭工作区记忆（PRD 13.2 第 2 行 / 11.4）。"""

    memory = _memory_for(workspace_id)
    if "enabled" not in payload or not isinstance(payload["enabled"], bool):
        raise HTTPException(status_code=422, detail="enabled 必须是布尔值")
    enabled = bool(payload["enabled"])
    try:
        result, outcome = await _commit_invalidation(
            memory=memory,
            mutate=lambda: memory.set_enabled(
                enabled, expected_revision=payload.get("expected_revision")
            ),
            reason="用户关闭工作区记忆" if not enabled else "用户开启工作区记忆",
        )
    except MemoryRevisionConflict as exc:
        raise _conflict(str(exc)) from exc
    except MemoryFileError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "applied": result.applied,
        "reason": result.reason,
        "enabled": result.memory.enabled,
        "revision": result.memory.revision,
        "invalidation_revision": result.memory.invalidation_revision,
        "runtime": _runtime_payload(outcome),
    }


# ===========================================================================
# PATCH /api/workspaces/{workspace_id}/memory/{memory_id} —— 人工纠正
# ===========================================================================


@router.patch("/api/workspaces/{workspace_id}/memory/{memory_id}")
async def correct_memory(
    workspace_id: str,
    memory_id: str,
    payload: Annotated[dict[str, Any], Body(...)],
) -> dict[str, Any]:
    """修改内容及预期 revision（PRD 13.2 第 3 行：人工纠正并使快照失效）。"""

    memory = _memory_for(workspace_id)
    kind = payload.get("kind")
    if kind is not None:
        try:
            MemoryKind(str(kind))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"未知 kind：{kind}") from exc
    try:
        result, outcome = await _commit_invalidation(
            memory=memory,
            mutate=lambda: memory.update_entry(
                memory_id,
                text=payload.get("text"),
                kind=kind,
                expected_revision=payload.get("expected_revision"),
                replacement=str(payload.get("replacement") or ""),
            ),
            reason=f"用户纠正记忆 {memory_id}",
        )
    except MemoryRevisionConflict as exc:
        raise _conflict(str(exc)) from exc
    except MemoryDisabledError as exc:
        raise _conflict(str(exc)) from exc
    except MemoryFileError as exc:
        raise _entry_error(exc) from exc

    return {
        "applied": result.applied,
        "reason": result.reason,
        "revision": result.memory.revision,
        "invalidation_revision": result.memory.invalidation_revision,
        "entry": None if result.entry is None else result.entry.model_dump(mode="json"),
        "runtime": _runtime_payload(outcome),
    }


# ===========================================================================
# DELETE /api/workspaces/{workspace_id}/memory/{memory_id} —— 遗忘
# ===========================================================================


@router.delete("/api/workspaces/{workspace_id}/memory/{memory_id}")
async def forget_memory(
    workspace_id: str,
    memory_id: str,
    expected_revision: Annotated[int | None, Query()] = None,
) -> dict[str, Any]:
    """遗忘并使快照失效（PRD 13.2 第 4 行）。软删除：记录保留，可回溯。"""

    memory = _memory_for(workspace_id)
    try:
        result, outcome = await _commit_invalidation(
            memory=memory,
            mutate=lambda: memory.forget_entry(
                memory_id, expected_revision=expected_revision
            ),
            reason=f"用户遗忘记忆 {memory_id}",
        )
    except MemoryRevisionConflict as exc:
        raise _conflict(str(exc)) from exc
    except MemoryFileError as exc:
        raise _entry_error(exc) from exc

    return {
        "applied": result.applied,
        "reason": result.reason,
        "revision": result.memory.revision,
        "invalidation_revision": result.memory.invalidation_revision,
        "entry": None if result.entry is None else result.entry.model_dump(mode="json"),
        "runtime": _runtime_payload(outcome),
    }


# ===========================================================================
# GET /api/workspaces/{workspace_id}/history —— 中文历史检索
# ===========================================================================


@router.get("/api/workspaces/{workspace_id}/history")
async def search_history(
    workspace_id: str,
    q: Annotated[str, Query()] = "",
    date_from: Annotated[str | None, Query()] = None,
    date_to: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """本区历史搜索：查询、日期、分页（PRD 13.2 第 5 行 / 11.5）。"""

    try:
        context = get_workspace_context(workspace_id)
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="工作区不存在") from exc

    search = workspace_history_search(context)
    try:
        hits = search.search(q, limit=limit, offset=offset, date_from=date_from, date_to=date_to)
    except HistorySearchError as exc:
        message = str(exc)
        status = 400 if "为空" in message else 503
        raise HTTPException(status_code=status, detail=message) from exc
    finally:
        search.close()
    return {
        "workspace_id": workspace_id,
        "query": q,
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
        "offset": offset,
        "results": [hit.as_dict() for hit in hits],
    }
