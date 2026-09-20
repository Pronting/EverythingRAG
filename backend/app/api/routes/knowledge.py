"""知识目录接口：返回从当前向量库元数据确定性聚合的安全概览。

V1.2.0 工作区隔离（T01）：本模块的全部端点都按 ``workspace_id``（查询参数）解析
**本区**上下文（PRD 5.1 / 5.3.5）。

- 概览与清空只作用于本区向量库、本区状态库、本区图片库；
- 原图与图片任务 ID 同时校验「工作区 + 图片/任务身份」：拿别的区的图片 hash 或任务
  ID 来查一律 404（A07），而不是把别的区的资料答出去。
"""

from __future__ import annotations

import io
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image

from app.api.deps import (
    get_import_task_store,
    get_workspace_context,
    get_workspace_image_state_store,
    get_workspace_state_store,
    get_workspace_vector_store,
    knowledge_revision_bumped,
)
from app.core.workspace_context import WorkspaceContext
from app.generation.knowledge_catalog import KnowledgeOverview, build_knowledge_overview
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.state_store import DocumentStateStore
from app.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/api/knowledge/images/{content_hash}")
def original_image(
    content_hash: str,
    context: WorkspaceContext = Depends(get_workspace_context),  # noqa: B008
    images: ImageStateStore = Depends(get_workspace_image_state_store),  # noqa: B008
) -> Response:
    """返回**本区**原图；跨区 hash 与不存在的 hash 一样得到 404（A07）。"""

    del context  # 归属校验由依赖完成；这里只需确认用的是本区图片库
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


@router.get("/api/knowledge/image-tasks/{block_id}")
def image_task(
    block_id: str,
    context: WorkspaceContext = Depends(get_workspace_context),  # noqa: B008
    images: ImageStateStore = Depends(get_workspace_image_state_store),  # noqa: B008
) -> dict[str, str]:
    """返回**本区**图片任务状态；跨区任务 ID 得到 404（A07）。

    只回状态与来源归属，不回文件路径与正文（隐私红线：日志与响应不含路径）。
    """

    status = images.task_status(block_id)
    if status is None:
        raise HTTPException(status_code=404, detail="图片任务不存在")
    return {"block_id": block_id, "status": status, "workspace_id": context.workspace_id}


@router.get("/api/knowledge/overview", response_model=KnowledgeOverview)
async def knowledge_overview(
    vectorstore: VectorStore = Depends(get_workspace_vector_store),  # noqa: B008
) -> KnowledgeOverview:
    """返回**本区**的文件/块/文本图片数、主题与代表文档；不返回原始或绝对路径。"""

    return build_knowledge_overview(vectorstore)


@router.delete("/api/knowledge")
async def delete_knowledge(
    context: WorkspaceContext = Depends(get_workspace_context),  # noqa: B008
    vectorstore: VectorStore = Depends(get_workspace_vector_store),  # noqa: B008
    state: DocumentStateStore = Depends(get_workspace_state_store),  # noqa: B008
    images: ImageStateStore = Depends(get_workspace_image_state_store),  # noqa: B008
    tasks: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
) -> dict[str, int]:
    """清空**本区**索引与来源登记；保留源文件与对话，其它工作区不受影响（PRD 5.4）。"""

    if tasks.has_running() or images.count_status().get("pending", 0) > 0:
        raise HTTPException(status_code=409, detail="导入、同步或图片识别仍在进行，请完成后再删除")
    blocks = vectorstore.list_blocks()
    for offset in range(0, len(blocks), 1000):
        vectorstore.delete_by_ids([block[0] for block in blocks[offset:offset + 1000]])
    state.clear()
    images.clear()
    tasks.clear()
    # PRD 5.3.6：删除**提交成功后**递增本区知识版本，使下一次工具调用读到更新后的索引
    # （缓存键含 knowledge_revision）。响应体保持原样，不因内部版本变化改契约。
    await knowledge_revision_bumped(context.workspace_id)
    return {"deleted_chunks": len(blocks)}
