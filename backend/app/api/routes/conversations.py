"""会话接口：对话历史 + AI 标题 + 归档 + 分支 + 工作区（对齐 Codex 客户端体验）。

- GET/POST /api/conversations：活跃会话列表（可按 workspace 过滤，不含归档/空会话）/ 新建
- POST /api/conversations/{id}/workspace：把会话移动到指定工作区
- GET /api/conversations/archived：已归档会话列表（设置页「归档对话」）
- GET /api/conversations/{id}：完整会话（含消息）
- POST /api/conversations/{id}/messages：追加消息
- POST /api/conversations/{id}/generate-title：用对话模型总结标题
- PUT /api/conversations/{id}/title：手动设标题
- POST /api/conversations/{id}/archive：归档（从侧边栏隐藏）
- POST /api/conversations/{id}/restore：从归档恢复
- POST /api/conversations/{id}/fork：分支出新会话（深拷贝消息，立即打开继续追问）
- DELETE /api/conversations/{id}：删除
- GET/POST /api/workspaces：工作区列表（含会话数）/ 新建
- PUT/DELETE /api/workspaces/{id}：改名 / 删除（删除时其中对话全部归档）
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.conversation_store import ConversationMessage, ConversationStore
from app.core.settings_store import get_settings_store
from app.core.workspace_store import (
    DEFAULT_WORKSPACE_ID,
    MAX_WORKSPACE_NAME_CHARS,
    WorkspaceStore,
)
from app.generation.providers import create_chat_model_from_config

router = APIRouter()


def get_conversation_store() -> ConversationStore:
    from app.api.deps import conversation_store

    return conversation_store


def get_workspace_store() -> WorkspaceStore:
    from app.api.deps import workspace_store

    return workspace_store


class MessageIn(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    answer_basis: str | None = None
    policy_version: str | None = None
    #: 推理模型思维链（本地展示，前端默认折叠）。
    reasoning: str | None = None


class AppendMessagesRequest(BaseModel):
    messages: list[MessageIn]


class SetTitleRequest(BaseModel):
    title: str


class MoveWorkspaceRequest(BaseModel):
    workspace_id: str = Field(min_length=1)


class WorkspacePinRequest(BaseModel):
    pinned: bool


class WorkspaceReorderRequest(BaseModel):
    """按给定顺序重写工作区位次（侧边栏拖拽排序落库）。"""

    ids: list[str] = Field(min_length=1, max_length=200)


class TruncateRequest(BaseModel):
    """从第 index 条消息起截断（用户点铅笔编辑重发时调用）。"""

    index: int = Field(ge=0)


class MoveRequest(BaseModel):
    """侧边栏拖拽落点：目标工作区（可选）+ 目标位置（该区第几条）。"""

    workspace_id: str | None = Field(default=None, max_length=40)
    target_index: int | None = Field(default=None, ge=0)


class WorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_WORKSPACE_NAME_CHARS)
    #: 可选自定义 id（缺省由名称推导）；前端一般只传 name。
    id: str | None = Field(default=None, max_length=40)


@router.get("/api/workspaces")
async def list_workspaces(
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回全部工作区（置顶在前 → 手工位次 → 创建时间），带各分区活跃会话数。"""
    counts = conversations.count_by_workspace()
    return [_workspace_payload(workspace, counts.get(workspace.id, 0)) for workspace in store.list()]


@router.put("/api/workspaces/order")
async def reorder_workspaces(
    request: WorkspaceReorderRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """按给定顺序重写工作区位次（侧边栏拖拽排序），返回新的完整顺序。

    必须注册在 ``/api/workspaces/{workspace_id}`` 之前，否则会被当成工作区 id。
    """
    counts = conversations.count_by_workspace()
    ordered = store.reorder(request.ids)
    return [_workspace_payload(workspace, counts.get(workspace.id, 0)) for workspace in ordered]


@router.post("/api/workspaces", status_code=201)
async def create_workspace(
    request: WorkspaceRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """新建工作区（名称为空/重复由存储层兜底，id 冲突自动追加序号）。"""
    workspace = store.create(request.name, request.id)
    return _workspace_payload(workspace, 0)


@router.put("/api/workspaces/{workspace_id}/pin")
async def pin_workspace(
    workspace_id: str,
    request: WorkspacePinRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """置顶 / 取消置顶工作区（别针图标）。"""
    workspace = store.set_pinned(workspace_id, request.pinned)
    if workspace is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    return _workspace_payload(workspace, 0)


@router.put("/api/workspaces/{workspace_id}")
async def rename_workspace(
    workspace_id: str,
    request: WorkspaceRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """重命名工作区（默认工作区也可改名，但仍不可删除）。"""
    workspace = store.rename(workspace_id, request.name)
    if workspace is None:
        raise HTTPException(status_code=404, detail="工作区不存在或名称为空")
    return _workspace_payload(workspace, 0)


@router.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """删除工作区：其中对话全部归档（可在设置页「归档对话」找回）；默认工作区不可删。"""
    if store.get(workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    if not store.delete(workspace_id):
        raise HTTPException(status_code=400, detail="默认工作区不可删除")
    archived = conversations.archive_workspace(workspace_id)
    return {"archived": archived}


@router.get("/api/conversations")
async def list_conversations(
    q: str = Query(default="", max_length=200),
    workspace: str | None = Query(default=None, max_length=40),
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回活跃会话摘要列表（按最近更新倒序，不含归档与空会话）。

    给定 ``workspace`` 时只返回该工作区的会话（侧边栏分区切换）。
    """
    if q.strip():
        return store.search(q, workspace)
    return store.list_summaries(workspace)


@router.get("/api/conversations/archived")
async def list_archived_conversations(
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回已归档会话摘要列表（按归档时间倒序）。

    必须注册在 `/api/conversations/{conversation_id}` 之前，否则会被当成会话 id。
    """
    return store.list_archived_summaries()


@router.post("/api/conversations", status_code=201)
async def create_conversation(
    workspace: str = Query(default=DEFAULT_WORKSPACE_ID, max_length=40),
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """在指定工作区（缺省默认工作区）新建空会话，返回其摘要。"""
    conv = store.create(workspace)
    return _summary(conv)


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """返回完整会话（含消息列表）。"""
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv.model_dump()


@router.post("/api/conversations/{conversation_id}/messages")
async def append_messages(
    conversation_id: str,
    request: AppendMessagesRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """追加一轮消息（user + assistant），返回更新后的会话摘要。"""
    messages = [
        ConversationMessage(
            role=m.role,
            content=m.content,
            sources=m.sources,
            answer_basis=m.answer_basis,
            policy_version=m.policy_version,
            reasoning=m.reasoning,
        )
        for m in request.messages
    ]
    conv = store.append_messages(conversation_id, messages)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.put("/api/conversations/{conversation_id}/title")
async def set_title(
    conversation_id: str,
    request: SetTitleRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """手动设置标题。"""
    conv = store.set_title(conversation_id, request.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/truncate")
async def truncate_conversation(
    conversation_id: str,
    request: TruncateRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """从第 index 条消息起截断会话（前端「编辑重发」：该条及其之后的轮次作废）。"""
    conv = store.truncate_from(conversation_id, request.index)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message_count": len(conv.messages)}


@router.post("/api/conversations/{conversation_id}/move")
async def move_conversation_in_sidebar(
    conversation_id: str,
    request: MoveRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
    workspaces: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """侧边栏拖拽：把会话移到目标工作区并放到指定位置（同区即为手动排序）。"""
    if request.workspace_id is not None and workspaces.get(request.workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    conv = store.move_in_sidebar(
        conversation_id,
        workspace_id=request.workspace_id,
        target_index=request.target_index,
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/workspace")
async def move_conversation_workspace(
    conversation_id: str,
    request: MoveWorkspaceRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
    workspaces: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """把会话移动到另一个工作区（分区归类）。"""
    if workspaces.get(request.workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    conv = store.set_workspace(conversation_id, request.workspace_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """归档会话：从侧边栏隐藏，可在设置页「归档对话」恢复。"""
    conv = store.set_archived(conversation_id, True)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """恢复已归档会话，重新回到侧边栏。"""
    conv = store.set_archived(conversation_id, False)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/fork", status_code=201)
async def fork_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """分支会话：深拷贝源会话消息为独立新会话（侧边栏折叠在源会话下方）。"""
    branch = store.fork(conversation_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(branch)


@router.post("/api/conversations/{conversation_id}/generate-title")
async def generate_title(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, str]:
    """用对话模型从前 2 条用户消息总结标题；模型不可用时回退首个用户消息截断。"""
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    title = await _ai_title(conv)
    store.set_title(conversation_id, title)
    return {"title": title}


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> None:
    """删除会话（其分支提升为顶层，不级联删除）。"""
    if not store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")


async def _ai_title(conv: Any) -> str:
    """生成标题：优先对话模型总结，回退首个用户消息前 30 字。"""
    user_texts = [m.content for m in conv.messages if m.role == "user"][:2]
    if not user_texts:
        return "新对话"
    try:
        chat = create_chat_model_from_config(get_settings_store().load().chat)
        prompt = (
            "请用不超过 15 字的短语概括这段对话的主题，只返回标题本身，"
            "不要加引号、标点或解释。\n用户：" + "\n用户：".join(user_texts)
        )
        tokens: list[str] = []
        async for chunk in chat.stream_chat([{"role": "user", "content": prompt}]):
            if chunk.kind == "content":
                tokens.append(chunk.text)
        title = "".join(tokens).strip()
        if title:
            return title[:60]
    except Exception:  # noqa: BLE001, S110 -- 标题生成失败回退，不影响主流程
        pass
    return user_texts[0][:30]


def _summary(conv: Any) -> dict[str, Any]:
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": len(conv.messages),
        "archived": conv.archived,
        "archived_at": conv.archived_at,
        "parent_id": conv.parent_id,
        "forked_from_title": conv.forked_from_title,
        "workspace_id": conv.workspace_id,
        "sort_index": conv.sort_index,
    }


def _workspace_payload(workspace: Any, conversation_count: int) -> dict[str, Any]:
    return {
        "id": workspace.id,
        "name": workspace.name,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "is_default": workspace.is_default,
        "pinned": workspace.pinned,
        "order_index": workspace.order_index,
        "conversation_count": conversation_count,
    }
