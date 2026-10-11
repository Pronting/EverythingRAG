"""语义块浏览接口：GET /api/blocks（列出全部块 + 元数据，供前端浏览面板）。

前端据此做分类（文本/图片）、搜索（标题/目录/关键词）与分页——本地知识库数据量小
（千级块），全量返回一次、客户端过滤更流畅（避免每次搜索都往返后端）。

**共享知识库**（D-KB）：块列表取自 ``?workspace_id=`` 解析出的工作区上下文，但知识
只有一份 —— 任何工作区看到的是**同一批**块（工作区参数只用于归属校验；未知工作区 404）。
省略参数时落到默认工作区（与 ``get_workspace_context`` 一致）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_workspace_vector_store
from app.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/api/blocks")
async def list_blocks(
    vectorstore: VectorStore = Depends(get_workspace_vector_store),  # noqa: B008
) -> dict:
    """返回**本工作区**全部语义块：block_id + 文本 + 元数据（来源文件/标题路径/类型）。"""
    blocks = vectorstore.list_blocks()
    return {
        "total": len(blocks),
        "blocks": [
            {
                "block_id": block_id,
                "text": text,
                "chunk_type": metadata.get("chunk_type", "text"),
                "source_type": metadata.get("source_type", "document"),
                "source_file": metadata.get("source_file", ""),
                "heading_path": metadata.get("heading_path"),
                "anchor": metadata.get("anchor"),
                "platform": metadata.get("platform", "local"),
            }
            for block_id, text, metadata in blocks
        ],
    }
