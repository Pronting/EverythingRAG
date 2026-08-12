"""知识库导入接口：POST /api/import（路径异步导入）+ POST /api/import/upload
（上传式）+ GET /api/import/status/{id}（轮询进度）。

- 异步模型：POST 校验后立即返回 task_id，后台线程执行 IngestionPipeline（同步
  阻塞的 chroma/embed 不能占事件循环），线程内逐文件写进度快照。
- 上传式：浏览器选文件夹后 multipart 上传文件，落 data_dir/documents/<task_id>/
  （**持久保留**——知识库自持数据，来源可回溯），再走同一异步管线 + 进度轮询。
- 进度：on_progress 回调把运行计数写入任务存储，前端轮询 GET status 展示。
- mode：路径导入预留 "incremental"（增量同步）位，返回清晰 400。
- 隐私：目录校验 / 错误信息不泄露敏感路径细节；单文件失败由管线聚合脱敏。
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_import_task_store, get_vector_store
from app.core.config import settings
from app.core.settings_store import SettingsStore, get_settings_store
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
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> IngestionPipeline:
    """构造导入管线（按设置选本地/云端嵌入器 + 共享向量库；构造零出网）。

    嵌入器与 get_vector_store 同源（同一设置），保证 collection 与向量维度一致。
    """
    from app.vectorstore.embedder import create_embedder_from_config

    app_settings = store.load()
    embedder = create_embedder_from_config(app_settings.embed)
    return IngestionPipeline(embedder=embedder, vectorstore=vectorstore)


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


@router.post("/api/import/upload", status_code=202)
async def upload_import(
    files: list[UploadFile] = File(...),  # noqa: B008
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    pipeline: IngestionPipeline = Depends(get_import_pipeline),  # noqa: B008
) -> dict[str, str]:
    """上传式导入：浏览器选文件夹后 multipart 上传文件，落数据目录再异步导入。

    每个文件的 filename 携带相对路径（webkitRelativePath，如 subdir/a.md），
    后端严格净化（拒绝绝对路径 / .. / 盘符穿越）后按原结构写入
    data_dir/documents/<task_id>/（**持久保留**——知识库自持数据、来源可回溯），
    再走与路径导入相同的异步管线 + 进度轮询。
    """
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件")
    task = store.create()
    documents_root = _documents_root(task.task_id)
    documents_root.mkdir(parents=True, exist_ok=True)
    try:
        await _save_uploads(files, documents_root)
    except Exception as exc:
        store.fail(task.task_id, _safe_message(exc))
        shutil.rmtree(documents_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail=_safe_message(exc)) from exc

    thread = threading.Thread(
        target=_run_import,
        args=(task.task_id, documents_root, pipeline, store),
        daemon=True,
        name=f"import-upload-{task.task_id[:8]}",
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


def _documents_root(task_id: str) -> Path:
    """上传文档持久目录：data_dir/documents/<task_id>（知识库自持数据，来源可回溯）。"""
    return settings.data_dir / "documents" / task_id


async def _save_uploads(files: list[UploadFile], staging_root: Path) -> int:
    """把上传文件按净化后的相对路径写入暂存目录，返回写入文件数。

    严格防穿越：拒绝绝对路径、包含 ``..`` 或空段、含盘符的路径。
    """
    written = 0
    for upload in files:
        relative = _safe_relative_path(upload.filename or "")
        if not relative.name.lower().endswith(".md"):
            continue  # MVP 只认 Markdown，其余类型丢弃（不写入暂存）
        target = staging_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await upload.read())
        written += 1
    if written == 0:
        raise ValueError("所选内容中没有 Markdown 文件")
    return written


def _safe_relative_path(raw_name: str) -> Path:
    """净化上传文件名：统一分隔符 -> 去首尾 / -> 拒绝穿越/绝对/盘符，返回相对 Path。"""
    name = raw_name.replace("\\", "/").strip("/")
    if not name:
        raise ValueError("文件名为空")
    parts = name.split("/")
    if any(part in ("..", "") or ":" in part for part in parts):
        raise ValueError("非法文件路径")
    relative = Path(*parts)
    if relative.is_absolute():
        raise ValueError("非法文件路径")
    return relative


def _safe_message(exc: Exception) -> str:
    """把上传净化/写入异常收敛为可读消息（不泄露原始路径细节）。"""
    message = str(exc).strip()
    if message:
        return message[:120]
    return f"上传处理失败: {type(exc).__name__}"


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
