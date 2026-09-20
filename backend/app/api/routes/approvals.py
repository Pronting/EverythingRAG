"""批准与澄清提交端点（PRD 10.2 / 13.2）。Owner: T09。

- ``POST /api/workspaces/{workspace_id}/approvals/{approval_id}``  提交批准决定
- ``POST /api/workspaces/{workspace_id}/inputs/{input_request_id}`` 提交澄清回答

同时提供 ``/runs/{run_id}/approvals/{id}`` / ``/runs/{run_id}/inputs/{id}`` 的嵌套别名：
前端在 run 上下文里直接拼相对路径即可，两条路径的处理完全一致（同一函数）。

语义（全部由 :mod:`app.agent.app_store` 的单事务保证）：
- 重复提交**相同**决定 → 幂等 200；不同决定 → 409；
- ``args_hash`` / ``context_epoch`` / ``rule_version`` 与保存值不一致 → 409（过期决定，A15）；
- run 因会话移动终止 → 拒绝提交（PRD 5.4 / 12.4）：移动会让未决卡片转 ``superseded``，
  此时提交必然 409，这里提前给出明确错误而不是拖到事务里。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.app_store import InputRequestConflictError, RecordNotFoundError
from app.agent.approval import ApprovalConflictError
from app.agent.contracts import ApprovalDecisionKind, RunStatus
from app.agent.runtime import RunManager, RunNotFoundError
from app.api.routes.runs import get_run_manager_dep

router = APIRouter()


class ApprovalSubmitRequest(BaseModel):
    """批准决定（PRD 10.2）：允许本次 / 允许本轮同类操作 / 拒绝，可带修改后的参数。"""

    decision: ApprovalDecisionKind
    edited_parameters: dict[str, Any] | None = None
    note: str = Field(default="", max_length=500)


class InputSubmitRequest(BaseModel):
    """澄清回答（PRD 13.2）：带问题版本，版本不符即 409。"""

    answer: str = Field(min_length=1, max_length=8000)
    question_version: int | None = Field(default=None, ge=1)


async def _require_workspace(workspace_id: str) -> Any:
    from app.api.routes.runs import _require_workspace as require

    return await require(workspace_id)


async def _submit_approval(
    manager: RunManager,
    workspace_id: str,
    approval_id: str,
    body: ApprovalSubmitRequest,
    run_id: str | None,
) -> dict[str, Any]:
    request = await manager.get_approval(approval_id)
    if request is None:
        raise HTTPException(status_code=404, detail="批准请求不存在")
    if run_id is not None and request.run_id != run_id:
        raise HTTPException(status_code=404, detail="批准请求不存在")
    run = await manager.require_run(request.run_id)
    if run is None or run.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="批准请求不存在")
    if run.status is RunStatus.CANCELLED and run.stop_reason == "conversation_moved":
        raise HTTPException(status_code=409, detail="该任务因会话移动终止，不能提交批准")
    try:
        resolved = await manager.submit_approval(
            approval_id,
            decision_kind=str(body.decision),
            edited_parameters=body.edited_parameters,
            note=body.note,
        )
    except ApprovalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="批准请求不存在") from exc
    return resolved.model_dump(mode="json")


async def _submit_input(
    manager: RunManager,
    workspace_id: str,
    input_request_id: str,
    body: InputSubmitRequest,
    run_id: str | None,
) -> dict[str, Any]:
    request = await manager.get_input_request(input_request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="澄清请求不存在")
    if run_id is not None and request.run_id != run_id:
        raise HTTPException(status_code=404, detail="澄清请求不存在")
    run = await manager.require_run(request.run_id)
    if run is None or run.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="澄清请求不存在")
    if run.status is RunStatus.CANCELLED and run.stop_reason == "conversation_moved":
        raise HTTPException(status_code=409, detail="该任务因会话移动终止，不能提交澄清")
    try:
        resolved = await manager.submit_input(
            input_request_id, answer=body.answer, question_version=body.question_version
        )
    except InputRequestConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="澄清请求不存在") from exc
    return resolved.model_dump(mode="json")


@router.post("/api/workspaces/{workspace_id}/approvals/{approval_id}")
async def submit_approval(
    workspace_id: str,
    approval_id: str,
    body: ApprovalSubmitRequest,
    request: Request,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> dict[str, Any]:
    """提交批准决定（PRD 10.2）。"""

    del request
    await _require_workspace(workspace_id)
    return await _submit_approval(manager, workspace_id, approval_id, body, None)


@router.post("/api/workspaces/{workspace_id}/runs/{run_id}/approvals/{approval_id}")
async def submit_approval_for_run(
    workspace_id: str,
    run_id: str,
    approval_id: str,
    body: ApprovalSubmitRequest,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> dict[str, Any]:
    """提交批准决定（run 上下文嵌套别名，语义同上）。"""

    await _require_workspace(workspace_id)
    return await _submit_approval(manager, workspace_id, approval_id, body, run_id)


@router.post("/api/workspaces/{workspace_id}/inputs/{input_request_id}")
async def submit_input(
    workspace_id: str,
    input_request_id: str,
    body: InputSubmitRequest,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> dict[str, Any]:
    """提交澄清回答（PRD 13.2）。"""

    await _require_workspace(workspace_id)
    return await _submit_input(manager, workspace_id, input_request_id, body, None)


@router.post("/api/workspaces/{workspace_id}/runs/{run_id}/inputs/{input_request_id}")
async def submit_input_for_run(
    workspace_id: str,
    run_id: str,
    input_request_id: str,
    body: InputSubmitRequest,
    manager: RunManager = Depends(get_run_manager_dep),  # noqa: B008
) -> dict[str, Any]:
    """提交澄清回答（run 上下文嵌套别名，语义同上）。"""

    await _require_workspace(workspace_id)
    return await _submit_input(manager, workspace_id, input_request_id, body, run_id)
