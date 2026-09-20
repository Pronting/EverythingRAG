"""run API（PRD 13.2 核心端点）与 SSE 事件订阅（PRD 13.3）。Owner: T09。

端点（全部按工作区限定，跨区一律 404）：

- ``POST   /api/workspaces/{workspace_id}/runs``            创建 run（保存用户消息 + 建任务）
- ``GET    /api/workspaces/{workspace_id}/runs``            列举某会话的 run
- ``GET    /api/workspaces/{workspace_id}/runs/{run_id}``    run 详情（含卡片与子任务）
- ``GET    /api/workspaces/{workspace_id}/runs/{run_id}/events``  SSE（``after=`` / ``Last-Event-ID`` 补读）
- ``POST   /api/workspaces/{workspace_id}/runs/{run_id}/cancel``  停止
- ``POST   /api/workspaces/{workspace_id}/runs/{run_id}/resume``  继续

约定：
1. 工作区一致性：run 的 ``workspace_id`` 必须等于路径上的工作区，否则 404（不是 403）——
   不泄露「别的区存在这个 run」。
2. 长度上限走 PRD 13.2 的 32000，超限 422；输入在执行前还要过一次上下文预算校验
   （预算不足直接 422，不进入执行）。
3. SSE 只取消**订阅**，不影响 run 执行（PRD 12.4 断线语义）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.contracts import (
    DEFAULT_USER_INPUT_MAX_CHARS,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
    TaskStatus,
)
from app.agent.runtime import (
    ConversationBusyError,
    InputTooLongError,
    RunManager,
    RunNotFoundError,
    RunNotRunnableError,
)
from app.api import deps
from app.api.routes.conversations import get_conversation_store
from app.api.run_images import MAX_RUN_IMAGES
from app.api.settings_scope import resolve_run_settings
from app.api.workspace_bridge import WorkspaceRowMissingError, ensure_workspace_row
from app.core.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

router = APIRouter()


# =====================================================================
# 依赖
# =====================================================================


def get_run_manager_dep(request: Request) -> RunManager:
    """取 lifespan 注册的 RunManager；未注册（例如只挂路由的裸测试）返回 503。"""

    manager = getattr(request.app.state, "run_manager", None)
    if manager is None:
        from app.agent.runtime import get_run_manager

        manager = get_run_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="RunManager 尚未就绪")
    return manager


# =====================================================================
# 请求 / 响应 schema
# =====================================================================


class RunCreateRequest(BaseModel):
    """创建 run（PRD 13.2）。

    PRD 13.2 首行的参数：问题、会话 ID、客户端请求 ID、**图片引用**、可选权限与强度。
    ``client_request_id`` 会话内唯一，重复请求幂等返回既有 run。

    ``images`` 是本次 run 附带的本工作区图片引用（``data:image/...`` 内联或 http(s)
    链接，最多 ``MAX_RUN_IMAGES`` 张）。引用会被解析、缓存原图并登记到**本区**图片库，
    逐条结果（``status`` / ``reason``）随 run 详情返回 —— 拿不到字节或未配置视觉时
    **明确标记未生效**，不会静默丢弃。
    """

    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=DEFAULT_USER_INPUT_MAX_CHARS)
    client_request_id: str | None = Field(default=None, max_length=200)
    images: list[str] = Field(default_factory=list, max_length=MAX_RUN_IMAGES)
    permission_mode: PermissionMode | None = None
    reasoning_level: ReasoningLevel | None = None
    #: 会话移动后在目标区继续当前问题（PRD 5.4）：填**被移动终止的那个 run**，
    #: 新 run 从零开始（新 thread / checkpoint），只保留这条可追溯来源链。
    origin_run_id: str | None = Field(default=None, max_length=64)


class RunSummary(BaseModel):
    run_id: str
    conversation_id: str
    workspace_id: str
    status: RunStatus
    permission_mode: PermissionMode
    reasoning_level: ReasoningLevel
    answer_id: str | None = None
    error_code: str | None = None
    stop_reason: str | None = None
    resumable: bool = False
    context_epoch: int = 1
    created_at: str = ""
    updated_at: str = ""


class ImageRef(BaseModel):
    """一条图片引用的处理结果（PRD 13.2「图片引用」，状态如实上报）。"""

    image_id: str = ""
    block_id: str = ""
    content_hash: str = ""
    kind: str = ""
    #: described / cached / pending_vision / unavailable / rejected（见 app/api/run_images.py）
    status: str
    reason: str = ""
    src_ref: str = ""
    recorded_at: str = ""


class TaskSummary(BaseModel):
    task_id: str
    agent_kind: str
    question: str = ""
    #: 收紧为枚举（PRD 12.5：结构化内容不合法就地失败，而不是把脏值透给前端）
    status: TaskStatus
    #: PRD 14.2「研究任务可以展开查看子问题…父任务展示整体完成情况」：显式父子关系，
    #: 前端不再靠 depends_on 猜。
    parent_task_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    result_ref: str | None = None


class RunDetail(RunSummary):
    events_url: str
    tasks: list[TaskSummary] = Field(default_factory=list)
    image_refs: list[ImageRef] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    last_event_seq: int = 0


class RunListResponse(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)


# =====================================================================
# 工具
# =====================================================================


def _summary(run: Any) -> RunSummary:
    return RunSummary(
        run_id=run.id,
        conversation_id=run.conversation_id,
        workspace_id=run.workspace_id,
        status=run.status,
        permission_mode=run.permission_mode,
        reasoning_level=run.reasoning_level,
        answer_id=run.answer_id,
        error_code=run.error_code,
        stop_reason=run.stop_reason,
        resumable=bool(run.resumable),
        context_epoch=int(run.context_epoch or 1),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


async def _require_run(manager: RunManager, workspace_id: str, run_id: str) -> Any:
    """取 run 并校验工作区归属（跨区 404，PRD 13.2）。"""

    try:
        run = await manager.require_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    if run.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    return run


# =====================================================================
# 创建与查询
# =====================================================================


@router.post("/api/workspaces/{workspace_id}/runs", status_code=202)
async def create_run(
    workspace_id: str,
    body: RunCreateRequest,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
    legacy_conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> RunDetail:
    """创建并启动一次执行（用户消息与 run 在同一次请求内落库，PRD 13.2）。"""

    await _require_workspace(workspace_id)
    await _ensure_conversation_row(
        manager, workspace_id, body.conversation_id, legacy_conversations
    )
    settings = await _resolve_settings(
        manager,
        workspace_id,
        conversation_id=body.conversation_id,
        requested_permission_mode=body.permission_mode,
        requested_reasoning_level=body.reasoning_level,
    )
    try:
        run = await manager.create_run(
            workspace_id=workspace_id,
            conversation_id=body.conversation_id,
            question=body.message,
            client_request_id=body.client_request_id,
            # PRD 1.3 三层优先级：请求显式 → 会话保存 → 工作区默认 → 18.1 兜底。
            # 解析唯一实现见 app/api/settings_scope.py；RunManager 只收结果。
            permission_mode=settings.permission_mode,
            reasoning_level=settings.reasoning_level,
            images=body.images,
            origin_run_id=body.origin_run_id,
        )
    except InputTooLongError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        if type(exc).__name__ == "ConflictError":
            raise HTTPException(status_code=409, detail="会话内 client_request_id 冲突") from exc
        raise HTTPException(status_code=400, detail="创建任务失败") from exc
    await manager.start(run.id)
    return await _detail(manager, run)


@router.get("/api/workspaces/{workspace_id}/runs")
async def list_runs(
    workspace_id: str,
    conversation_id: str = Query(min_length=1),
    limit: int | None = Query(default=None, ge=1, le=200),
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> RunListResponse:
    """列举某会话的 run（按创建时间）。"""

    await _require_workspace(workspace_id)
    conversation = await manager.get_conversation(conversation_id)
    if conversation is None or conversation.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    runs = await manager.list_runs(conversation_id, limit=limit)
    return RunListResponse(runs=[_summary(run) for run in runs])


@router.get("/api/workspaces/{workspace_id}/runs/{run_id}")
async def get_run(
    workspace_id: str,
    run_id: str,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> RunDetail:
    """run 详情：状态 + 子任务 + 待批准/待澄清卡片（PRD 12.4 重启后恢复卡片）。"""

    run = await _require_run(manager, workspace_id, run_id)
    return await _detail(manager, run)


async def _detail(manager: RunManager, run: Any) -> RunDetail:
    tasks = await manager.list_tasks(run.id)
    approvals = await manager.list_pending_approvals(run.id)
    inputs = await manager.list_pending_inputs(run.id)
    return RunDetail(
        **_summary(run).model_dump(),
        events_url=f"/api/workspaces/{run.workspace_id}/runs/{run.id}/events",
        tasks=[
            TaskSummary(
                task_id=task.id,
                agent_kind=str(task.agent_kind),
                question=task.question,
                status=task.status,
                parent_task_id=task.parent_task_id,
                depends_on=list(task.depends_on),
                requirement_ids=list(task.requirement_ids),
                thread_id=task.thread_id,
                result_ref=task.result_ref,
            )
            for task in tasks
        ],
        image_refs=[ImageRef.model_validate(item) for item in await manager.image_refs(run.id)],
        approvals=[item.model_dump(mode="json") for item in approvals],
        inputs=[item.model_dump(mode="json") for item in inputs],
        last_event_seq=await manager.last_event_seq(run.id),
    )


# =====================================================================
# SSE
# =====================================================================


@router.get("/api/workspaces/{workspace_id}/runs/{run_id}/events")
async def stream_run_events(
    workspace_id: str,
    run_id: str,
    request: Request,
    after: int | None = Query(default=None, ge=0),
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> StreamingResponse:
    """事件流（PRD 13.3）：标准 ``id`` + JSON ``data``；``after`` 或 ``Last-Event-ID`` 补读。

    断开只结束订阅，不取消 run（PRD 12.4）：运行层不感知订阅者。
    """

    await _require_run(manager, workspace_id, run_id)
    last_event_id = request.headers.get("last-event-id")
    start = after
    if start is None and last_event_id:
        try:
            start = int(last_event_id)
        except ValueError:
            start = 0
    return StreamingResponse(
        _sse(manager, run_id, start or 0),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(manager: RunManager, run_id: str, after_seq: int) -> AsyncIterator[str]:
    async for frame in manager.stream_events(run_id, after_seq=after_seq):
        yield frame


# =====================================================================
# 停止 / 继续
# =====================================================================


class CancelRequest(BaseModel):
    reason: str = Field(default="user_stopped", max_length=200)


@router.post("/api/workspaces/{workspace_id}/runs/{run_id}/cancel")
async def cancel_run(
    workspace_id: str,
    run_id: str,
    body: CancelRequest | None = None,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> RunDetail:
    """停止：终止主任务与活动子任务，等模型请求退出后记完成（PRD 12.4）。"""

    run = await _require_run(manager, workspace_id, run_id)
    try:
        await manager.cancel(run.id, reason=(body.reason if body else "user_stopped"))
    except RunNotRunnableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _detail(manager, await _require_run(manager, workspace_id, run_id))


@router.post("/api/workspaces/{workspace_id}/runs/{run_id}/resume")
async def resume_run(
    workspace_id: str,
    run_id: str,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> RunDetail:
    """继续：从记录位置重新驱动，已完成子任务不重复执行（PRD 12.3 / 12.4）。"""

    run = await _require_run(manager, workspace_id, run_id)
    try:
        await manager.resume(run.id)
    except RunNotRunnableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _detail(manager, await _require_run(manager, workspace_id, run_id))


# =====================================================================
# 工作区校验
# =====================================================================


async def _require_workspace(workspace_id: str) -> Any:
    """校验路径里的工作区存在（不存在 404，而不是静默回落到默认区）。

    app.db 缺该工作区行时**按存档自愈**（:mod:`app.api.workspace_bridge`）：D25 过渡期
    存档是唯一合法来源，而 app.db 的 ``workspaces`` 行是 run / 会话行的外键前提 —— 不自愈
    的话全新数据目录上「建会话 → 建 run」会直接 404（实测到过）。
    """

    store = deps.get_app_store()
    if store is not None:
        try:
            return await ensure_workspace_row(store, deps.get_workspace_store(), workspace_id)
        except WorkspaceRowMissingError as exc:
            raise HTTPException(status_code=404, detail="工作区不存在") from exc
    try:
        return deps.get_workspace_context_factory().resolve(workspace_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="工作区不存在") from exc


async def _workspace_defaults(workspace_id: str) -> Any:
    """读该工作区的默认权限/强度：**视图优先，app.db 权威行兜底**。

    D25 的唯一读取入口是 ``WorkspaceContextFactory.resolve(id).view``，这里它就是首选。
    但视图的水合源是 ``app_db_path()`` 这个**文件**：测试用内存 app.db、或存档/水合一时
    不可用时，视图会拿不到该区 —— 此时**不能 404**（工作区已由 :func:`_require_workspace`
    校验过，且迁移后 app.db 行本身就是同一权威），改为直接读 app.db 权威行；两者都拿不到
    才返回 None 交调用方按 PRD 18.1 兜底（绝不冒用别的工作区的默认值）。
    """

    try:
        return deps.get_workspace_context_factory().view_for(workspace_id)
    except Exception as exc:  # noqa: BLE001 - 视图不可得 → 退到 app.db 权威行
        logger.debug("工作区视图不可得，改用 app.db 权威行：error=%s", type(exc).__name__)
    store = deps.get_app_store()
    if store is None:
        return None
    try:
        return await store.get_workspace(workspace_id)
    except Exception:  # noqa: BLE001 - 权威行也读不到 → 调用方兜底
        return None


async def _ensure_conversation_row(
    manager: RunManager,
    workspace_id: str,
    conversation_id: str,
    legacy_store: ConversationStore,
) -> None:
    """会话在 app.db 缺行时按存档自愈（同一 id、继承工作区默认值）。

    过渡期只有存档里有会话（``POST /api/conversations`` 当时只写存档），而 app.db 的会话行
    是 run 的外键前提。自愈把它提升成 app.db 会话行，**id 保持一致**；存档里也没有该 id
    说明它本就不合法，交给后续创建路径报错（不在这里静默造假）。
    """

    store = deps.get_app_store()
    if store is None:
        return
    if await manager.get_conversation(conversation_id) is not None:
        return
    legacy = legacy_store.get(conversation_id)
    if legacy is None or str(getattr(legacy, "workspace_id", "")) != workspace_id:
        return
    settings = resolve_run_settings(view=await _workspace_defaults(workspace_id))
    await store.create_conversation(
        conversation_id,
        workspace_id,
        permission_mode=settings.permission_mode,
        reasoning_level=settings.reasoning_level,
    )


async def _resolve_settings(
    manager: RunManager,
    workspace_id: str,
    *,
    conversation_id: str,
    requested_permission_mode: Any,
    requested_reasoning_level: Any,
) -> Any:
    """按 PRD 1.3 的三层优先级解析本次 run 的权限与思考强度。

    唯一实现在 :mod:`app.api.settings_scope`（纯函数）：``请求显式 → 会话保存 →
    工作区默认 → PRD 18.1 兜底``，逐轴独立解析。
    """

    conversation = await manager.get_conversation(conversation_id)
    return resolve_run_settings(
        requested_permission_mode=requested_permission_mode,
        requested_reasoning_level=requested_reasoning_level,
        conversation_permission_mode=getattr(conversation, "permission_mode", None),
        conversation_reasoning_level=getattr(conversation, "reasoning_level", None),
        view=await _workspace_defaults(workspace_id),
    )


#: 兼容入口：``GET /api/workspaces/{workspace_id}/runs/{run_id}/events`` 的 JSON 快照
#: （前端首屏不进流时用；与 SSE 同一数据源，避免两套读法）。
@router.get("/api/workspaces/{workspace_id}/runs/{run_id}/event-log")
async def run_event_log(
    workspace_id: str,
    run_id: str,
    after: int = Query(default=0, ge=0),
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> JSONResponse:
    """按 ``event_seq`` 补读事件（``Last-Event-ID`` 的服务端等价物）。"""

    await _require_run(manager, workspace_id, run_id)
    try:
        events = await manager.read_events(run_id, after_seq=after)
    except Exception as exc:  # noqa: BLE001 - 协议错误 → 502 语义的明确错误体
        return JSONResponse(
            status_code=502,
            content={"error_code": "event_protocol_error", "message": type(exc).__name__},
        )
    return JSONResponse({"events": [event.model_dump(mode="json") for event in events]})
