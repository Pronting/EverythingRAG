"""知识目录接口：返回从当前向量库元数据确定性聚合的安全概览。

D-KB（全局共享知识库）：知识只有一份，本模块的端点仍按 ``workspace_id``（查询参数）
解析并**校验归属**（未知工作区一律 404），但解析出的向量库 / 状态库 / 图片库是
**所有工作区共享**的那一份 —— 因此「概览 / 清空 / 原图 / 任务查询」在各区看到的结果
相同；工作区参数的作用收敛为「归属校验 + 审计归属」。

- 概览与清空作用于共享向量库、共享状态库、共享图片库（清空后**所有**工作区的
  知识版本一起递增，见 ``deps.knowledge_revision_bumped``）；
- 原图与图片任务 ID 仍校验格式与存在性：不存在的 hash / 任务 ID 一律 404。
"""

from __future__ import annotations

import asyncio
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
from app.knowledge.maintenance import maintenance_guard
from app.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/api/knowledge/images/{content_hash}")
def original_image(
    content_hash: str,
    context: WorkspaceContext = Depends(get_workspace_context),  # noqa: B008
    images: ImageStateStore = Depends(get_workspace_image_state_store),  # noqa: B008
) -> Response:
    """返回**共享知识库**里的原图；不存在的 hash 得到 404（D-KB）。

    工作区参数仍用于归属校验（未知工作区 404），但图片库是共享的：同一个 hash
    在每个工作区都取得到 —— 因为原图属于那份唯一的知识。
    """

    del context  # 归属校验由依赖完成；图片库是所有工作区共用的那一份
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
    """返回**共享知识库**里的图片任务状态；不存在的任务 ID 得到 404（D-KB）。

    只回状态与来源归属，不回文件路径与正文（隐私红线：日志与响应不含路径）。
    响应里的 ``workspace_id`` 是**本次请求**解析出的工作区（审计归属），不代表任务归属。
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
    """清空**共享知识库**的索引与来源登记；保留源文件与对话（D-KB）。

    知识库只有一份，因此本操作对**所有**工作区生效（清空后各区的知识版本一起递增）；
    工作区参数只用于归属校验。
    """

    if tasks.has_running() or images.count_status().get("pending", 0) > 0:
        raise HTTPException(status_code=409, detail="导入、同步或图片识别仍在进行，请完成后再删除")
    def clear_index():
        with maintenance_guard(vectorstore):
            blocks = vectorstore.list_blocks()
            for offset in range(0, len(blocks), 1000):
                vectorstore.delete_by_ids([block[0] for block in blocks[offset:offset + 1000]])
            state.clear()
            images.clear()
            tasks.clear()
            return len(blocks)

    try:
        deleted = await asyncio.to_thread(clear_index)
    except TimeoutError:
        raise HTTPException(status_code=409, detail="知识索引正在维护") from None
    # PRD 5.3.6（D-KB 口径）：删除**提交成功后**递增知识版本，使下一次工具调用读到
    # 更新后的索引（缓存键含 knowledge_revision）；共享知识 ⇒ **所有**工作区的版本
    # 一起递增（见 deps.knowledge_revision_bumped）。响应体保持原样，不改契约。
    await knowledge_revision_bumped(context.workspace_id)
    return {"deleted_chunks": deleted}
