"""知识库导入接口：POST /api/import（异步启动）+ GET /api/import/status/{id}（轮询进度）。

- 异步模型：POST 校验后立即返回 task_id，后台线程执行 IngestionPipeline（同步
  阻塞的 chroma/embed 不能占事件循环），线程内逐文件写进度快照。
- 进度：on_progress 回调把运行计数写入任务存储，前端轮询 GET status 展示。
- mode：现仅支持 "full"；"incremental"（增量同步）预留位，返回清晰 400。
- 隐私：目录校验 / 错误信息不泄露敏感路径细节；单文件失败由管线聚合脱敏。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_import_task_store, get_vector_store
from app.ingestion.import_task import ImportTask, ImportTaskStore
from app.ingestion.pipeline import IngestionPipeline, IngestReport
from app.vectorstore.base import VectorStore

router = APIRouter()


class ImportRequest(BaseModel):
    """导入请求：dir 为服务器本地目录路径；mode 现仅支持 full。"""

    dir: str = Field(min_length=1, max_length=1024)
    mode: Literal["full", "incremental"] = "full"


def get_import_pipeline(
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> IngestionPipeline:
    """构造导入管线（惰性 bge-m3 + 共享向量库；构造零出网）。测试可 override 为 fake。"""
    from app.vectorstore.embedder import FastEmbedEmbedder

    return IngestionPipeline(embedder=FastEmbedEmbedder(), vectorstore=vectorstore)


@router.post("/api/import", status_code=202)
async def start_import(
    request: ImportRequest,
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    pipeline: IngestionPipeline = Depends(get_import_pipeline),  # noqa: B008
) -> dict[str, str]:
    """启动一次异步导入：校验 -> 建任务 -> 后台线程跑管线 -> 立即返回 task_id。"""
    if request.mode == "incremental":
        raise HTTPException(
            status_code=400,
            detail="增量同步尚未实现，当前仅支持 mode=full（全量导入）",
        )
    root = Path(request.dir)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {request.dir}")

    task = store.create()
    thread = threading.Thread(
        target=_run_import,
        args=(task.task_id, root, pipeline, store),
        daemon=True,
        name=f"import-{task.task_id[:8]}",
    )
    thread.start()
    return {"task_id": task.task_id}


@router.get("/api/import/status/{task_id}")
async def import_status(
    task_id: str,
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
) -> dict:
    """查询任务状态：running（含实时进度）/ done（含报告）/ error（含信息）。"""
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _task_payload(task)


def _run_import(
    task_id: str,
    root: Path,
    pipeline: IngestionPipeline,
    store: ImportTaskStore,
) -> None:
    """后台线程执行体：逐文件进度回调写入任务存储；终态 done / error。"""
    try:
        report = pipeline.ingest(
            root,
            on_progress=lambda snapshot: store.update_progress(task_id, snapshot),
        )
        store.complete(task_id, report)
    except Exception as exc:  # noqa: BLE001 -- 线程根异常收敛为任务 error，不泄露路径/正文
        store.fail(task_id, f"导入失败: {type(exc).__name__}")


def _task_payload(task: ImportTask) -> dict:
    """ImportTask -> API 响应 dict（进度恒为对象，终态含 report / error）。"""
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": _progress_payload(task),
        "report": _report_payload(task.report) if task.report is not None else None,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


def _progress_payload(task: ImportTask) -> dict[str, int]:
    if task.progress is None:
        return {"files_scanned": 0, "files_parsed": 0, "files_skipped": 0, "chunks": 0}
    return {
        "files_scanned": task.progress.files_scanned,
        "files_parsed": task.progress.files_parsed,
        "files_skipped": task.progress.files_skipped,
        "chunks": task.progress.chunks,
    }


def _report_payload(report: IngestReport) -> dict:
    return {
        "files_scanned": report.files_scanned,
        "files_parsed": report.files_parsed,
        "files_skipped": report.files_skipped,
        "chunks": report.chunks,
        "blocks_upserted": report.blocks_upserted,
        "errors": list(report.errors),
    }
