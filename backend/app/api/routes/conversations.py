"""会话接口：对话历史 + AI 标题（侧边栏会话列表，对齐 ChatGPT 体验）。

- GET/POST /api/conversations：列表 / 新建
- GET /api/conversations/{id}：完整会话（含消息）
- POST /api/conversations/{id}/messages：追加消息
- POST /api/conversations/{id}/generate-title：用对话模型总结标题
- PUT /api/conversations/{id}/title：手动设标题；DELETE 删除
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.conversation_store import ConversationMessage, ConversationStore
from app.core.settings_store import get_settings_store
from app.generation.providers import create_chat_model_from_config

router = APIRouter()


def get_conversation_store() -> ConversationStore:
    from app.api.deps import conversation_store

    return conversation_store


class MessageIn(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    answer_basis: str | None = None
    policy_version: str | None = None


class AppendMessagesRequest(BaseModel):
    messages: list[MessageIn]


class SetTitleRequest(BaseModel):
    title: str


@router.get("/api/conversations")
async def list_conversations(
    q: str = Query(default="", max_length=200),
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回会话摘要列表（按最近更新倒序）。"""
    return store.search(q) if q.strip() else store.list_summaries()


@router.post("/api/conversations", status_code=201)
async def create_conversation(
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """新建空会话，返回其摘要。"""
    conv = store.create()
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
    """删除会话。"""
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
    }
