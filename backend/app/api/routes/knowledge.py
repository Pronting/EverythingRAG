"""知识目录接口：返回从当前向量库元数据确定性聚合的安全概览。"""

from __future__ import annotations

import io
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image

from app.api.deps import (
    get_image_state_store,
    get_import_task_store,
    get_state_store,
    get_vector_store,
)
from app.generation.knowledge_catalog import KnowledgeOverview, build_knowledge_overview
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.state_store import DocumentStateStore
from app.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/api/knowledge/images/{content_hash}")
def original_image(content_hash: str, images: ImageStateStore = Depends(get_image_state_store)) -> Response:  # noqa: B008
    if not re.fullmatch(r"[a-f0-9]{16,64}", content_hash):
        raise HTTPException(status_code=404, detail="图片不存在")
    data = images.get_image_bytes(content_hash)
    if data is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    try:
        with Image.open(io.BytesIO(data)) as image:
            mime = Image.MIME.get(image.format, "application/octet-stream")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="图片无法读取") from exc
    return Response(data, media_type=mime, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.get("/api/knowledge/overview", response_model=KnowledgeOverview)
async def knowledge_overview(
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> KnowledgeOverview:
    """返回文件/块/文本图片数、主题与代表文档；不返回原始或绝对路径。"""

    return build_knowledge_overview(vectorstore)


@router.delete("/api/knowledge")
async def delete_knowledge(
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
    state: DocumentStateStore = Depends(get_state_store),  # noqa: B008
    images: ImageStateStore = Depends(get_image_state_store),  # noqa: B008
    tasks: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
) -> dict[str, int]:
    """Clear the current index and source registrations; retain source files and chats."""
    if tasks.has_running() or images.count_status().get("pending", 0) > 0:
        raise HTTPException(status_code=409, detail="导入、同步或图片识别仍在进行，请完成后再删除")
    blocks = vectorstore.list_blocks()
    for offset in range(0, len(blocks), 1000):
        vectorstore.delete_by_ids([block[0] for block in blocks[offset:offset + 1000]])
    state.clear()
    images.clear()
    tasks.clear()
    return {"deleted_chunks": len(blocks)}
