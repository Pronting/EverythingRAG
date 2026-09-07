"""知识库导入接口：POST /api/import（路径异步导入）+ POST /api/import/upload
（上传式，增量感知）+ POST /api/sync（一键同步已登记根）+ GET status（轮询）。

- 异步模型：POST 校验后立即返回 task_id，后台线程执行管线（同步阻塞的
  chroma/embed 不能占事件循环），线程内逐文件写进度快照。
- 上传式：浏览器多选文件夹/文件后 multipart 上传，按相对路径写入**规范根**
  ``data_dir/documents/``（稳定身份 = 相对路径，来源可回溯）。重新选择同一批
  文件夹上传即增量对账（mode=incremental）：只处理新增/更新/删除，未变文件
  零重嵌入；删除只作用于本次重传的顶层目录（scope），散文件只增/改不删。
- 路径式 mode=incremental：对本地目录做增量同步（指纹快路径 + 块级复用）。
- 进度：on_progress 回调把运行计数写入任务存储，前端轮询 GET status 展示。
- 隐私：目录校验 / 错误信息不泄露敏感路径细节；单文件失败由管线聚合脱敏。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import (
    get_embedder,
    get_image_state_store,
    get_import_task_store,
    get_state_store,
    get_vector_store,
)
from app.core.config import settings
from app.core.outbound import outbound_client
from app.core.settings_store import get_settings_store, is_vision_configured
from app.generation.base import VisionModel
from app.generation.vision import create_vision_model_from_config
from app.ingestion.image_fetch import ImageFetcher
from app.ingestion.image_worker import ImageWorker
from app.ingestion.import_task import ImportTask, ImportTaskStore
from app.ingestion.pipeline import IngestionPipeline, IngestReport
from app.ingestion.scanner import discover_file
from app.ingestion.state_store import DocumentStateStore
from app.ingestion.sync import SyncProgress, SyncReport, SyncService
from app.vectorstore.base import VectorStore

router = APIRouter()


class ImportRequest(BaseModel):
    """导入请求：dir 为服务器本地目录路径；mode 现仅支持 full。"""

    dir: str = Field(min_length=1, max_length=1024)
    mode: Literal["full", "incremental"] = "full"


def _build_vision_pair() -> tuple[VisionModel | None, ImageFetcher | None]:
    """按当前设置构造 (识图模型, 图片下载器)；识图未配置时返回 (None, None)。

    出网走全局审计单例；仅当 config.json 里 vision base_url+model 都配好才启用，
    否则图片仅存元数据、不调模型（PRD 4.3 验收 2）。
    """
    vision_cfg = get_settings_store().load().vision
    if not is_vision_configured(vision_cfg):
        return None, None
    vision = create_vision_model_from_config(vision_cfg, outbound=outbound_client)
    fetcher = ImageFetcher(outbound=outbound_client)
    return vision, fetcher


def get_import_pipeline(
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
    document_state_store: DocumentStateStore = Depends(get_state_store),  # noqa: B008
) -> IngestionPipeline:
    """构造导入管线（共享云端嵌入器 + 共享向量库；构造零出网）。

    嵌入器用共享单例（get_embedder），与向量库同源，保证 dim 惰性解析结果一致。
    """
    vision, fetcher = _build_vision_pair()
    image_store = get_image_state_store()
    pipeline = IngestionPipeline(
        embedder=get_embedder(),
        vectorstore=vectorstore,
        vision=vision,
        fetcher=fetcher,
        state_store=image_store,
        document_state_store=document_state_store,
    )
    if vision is not None:
        # 识图已配置：挂后台工作线程池，文本先入库、图片后台并发补全（持久化 + 重启续跑）。
        worker = ImageWorker(
            pipeline.process_image_task_and_upsert, image_store, settings.vision_concurrency
        )
        worker.start()
        pipeline.set_worker(worker)
    return pipeline


def get_sync_service(
    vectorstore: VectorStore = Depends(get_vector_store),  # noqa: B008
    state_store: DocumentStateStore = Depends(get_state_store),  # noqa: B008
) -> SyncService:
    """构造增量同步服务（共享云端嵌入器 + 共享向量库 + 状态库；构造零出网）。"""
    vision, fetcher = _build_vision_pair()
    return SyncService(
        embedder=get_embedder(),
        vectorstore=vectorstore,
        state_store=state_store,
        vision=vision,
        fetcher=fetcher,
        image_state_store=get_image_state_store(),
    )


@router.post("/api/import", status_code=202)
async def start_import(
    request: ImportRequest,
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    pipeline: IngestionPipeline = Depends(get_import_pipeline),  # noqa: B008
    sync: SyncService = Depends(get_sync_service),  # noqa: B008
) -> dict[str, str]:
    """启动一次导入：full=全量异步导入 / incremental=增量同步，立即返回 task_id。

    incremental：指纹快路径 + 块级复用，只处理新增/更新/删除，未变文件毫秒级跳过。
    两者共用同一异步任务 + 进度轮询管线（runner 需实现 ingest(root, on_progress)）。
    """
    root = Path(request.dir)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {request.dir}")
    runner: object = sync if request.mode == "incremental" else pipeline

    task = store.create()
    thread = threading.Thread(
        target=_run_import,
        args=(task.task_id, root, runner, store),
        daemon=True,
        name=f"import-{request.mode}-{task.task_id[:8]}",
    )
    thread.start()
    return {"task_id": task.task_id}


@router.post("/api/import/upload", status_code=202)
async def upload_import(
    files: list[UploadFile] = File(...),  # noqa: B008
    mode: Literal["full", "incremental"] = Form("incremental"),
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    sync: SyncService = Depends(get_sync_service),  # noqa: B008
    state_store: DocumentStateStore = Depends(get_state_store),  # noqa: B008
) -> dict[str, str]:
    """上传式导入：多选文件夹/文件 multipart 上传，落规范根后异步导入。

    mode=incremental（默认）：重新选择同一批文件夹上传 = 增量对账——指纹比对只
    处理新增/更新/删除，未变文件零重嵌入；删除只作用于本次重传的顶层目录。
    mode=full：仅入库/更新，不删除（旧语义，保守）。
    """
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件")
    documents_root = _documents_root().resolve()
    documents_root.mkdir(parents=True, exist_ok=True)
    try:
        saved_paths = await _save_uploads(files, documents_root)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_safe_message(exc)) from exc

    task = store.create()
    thread = threading.Thread(
        target=_run_upload,
        args=(task.task_id, saved_paths, documents_root, mode, sync, store, state_store),
        daemon=True,
        name=f"import-upload-{mode}-{task.task_id[:8]}",
    )
    thread.start()
    return {"task_id": task.task_id}


@router.post("/api/sync", status_code=202)
async def start_sync(
    store: ImportTaskStore = Depends(get_import_task_store),  # noqa: B008
    sync: SyncService = Depends(get_sync_service),  # noqa: B008
    state_store: DocumentStateStore = Depends(get_state_store),  # noqa: B008
) -> dict[str, str]:
    """一键增量同步：对全部已登记的知识库目录做增量同步，立即返回 task_id。

    前端「同步」按钮免重选目录——目录在导入/同步时已登记到 state_store.sources。
    未登记任何目录 -> 400 提示先导入。
    """
    roots = state_store.list_sources()
    if not roots:
        raise HTTPException(status_code=400, detail="尚未导入过任何知识库目录，请先选择文件夹导入")

    task = store.create()
    thread = threading.Thread(
        target=_run_sync_all,
        args=(task.task_id, roots, sync, store),
        daemon=True,
        name=f"sync-{task.task_id[:8]}",
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


def _run_sync_all(
    task_id: str,
    roots: list[Path],
    sync: SyncService,
    store: ImportTaskStore,
) -> None:
    """后台线程执行体（一键同步）：对多个已登记根增量同步，聚合报告。"""
    try:
        report = sync.ingest_many(
            roots,
            on_progress=lambda snapshot: store.update_progress(task_id, snapshot),
        )
        store.complete(task_id, report)
    except Exception as exc:  # noqa: BLE001
        store.fail(task_id, f"同步失败: {type(exc).__name__}")


def _run_upload(
    task_id: str,
    saved_paths: list[Path],
    documents_root: Path,
    mode: str,
    sync: SyncService,
    store: ImportTaskStore,
    state_store: DocumentStateStore,
) -> None:
    """后台线程执行体（上传式）：落盘文件 -> 发现指纹 -> 增量对账 -> 终态。"""
    try:
        # 已落盘文件 -> DiscoveredFile（读内容算指纹，身份 = 规范根相对路径）
        discovered = [d for path in saved_paths if (d := discover_file(path)) is not None]
        scopes = _upload_scopes(saved_paths, documents_root)
        if mode == "incremental":
            # 清理规范根下失效源文件（已知指纹但本次未重传、且在重传 scope 内）
            _cleanup_stale_files(state_store, documents_root, saved_paths, scopes)
        report = sync.sync_upload(
            discovered,
            deletion_scopes=scopes if mode == "incremental" else [],
            on_progress=lambda snapshot: store.update_progress(task_id, snapshot),
        )
        # 登记来源根：上传内容统一落在 documents_root 下，登记后 /api/sync 一键同步
        # 可对全部已上传内容增量对账（对齐目录式 ingest 末尾的 register_source）。
        state_store.register_source(documents_root)
        store.complete(task_id, report)
    except Exception as exc:  # noqa: BLE001 -- 线程根异常收敛为任务 error，不泄露路径/正文
        store.fail(task_id, f"导入失败: {type(exc).__name__}")


def _upload_scopes(saved_paths: list[Path], documents_root: Path) -> list[Path]:
    """本次上传涉及的顶层目录（作为删除作用域）。

    相对路径多段 -> 首段是目录 -> scope = documents_root/首段（重传该目录时，
    其下已知但本次未上传的文件视为删除）；单段（散文件）不属于任何目录 scope，
    只增/改不删——避免「不重传的散文件被误删」。
    """
    scopes: set[Path] = set()
    for path in saved_paths:
        try:
            relative = path.relative_to(documents_root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) > 1:
            scopes.add(documents_root / parts[0])
    return sorted(scopes)


def _cleanup_stale_files(
    state_store: DocumentStateStore,
    documents_root: Path,
    saved_paths: list[Path],
    scopes: list[Path],
) -> None:
    """删除规范根下已失效的源文件：已知指纹但本批未重传、且在重传 scope 内。

    保持规范根与「用户实际选择」一致（来源可回溯不残留死文件）；其块与指纹
    由 sync_upload 的删除分支清除。
    """
    saved_keys = {str(path) for path in saved_paths}
    for key in state_store.load_all():
        path = Path(key)
        if key in saved_keys:
            continue
        if not path.is_relative_to(documents_root):
            continue  # 只清理规范根内的文件
        if any(path.is_relative_to(scope) for scope in scopes):
            path.unlink(missing_ok=True)


def _documents_root() -> Path:
    """上传文档规范根：data_dir/documents（稳定相对路径 = 文档身份，增量同步用）。

    不再按 task_id 分目录——否则同一文件两次上传路径不同，指纹身份对不上，
    无法增量。知识库自持数据、来源可回溯均保留。
    """
    return settings.data_dir / "documents"


async def _save_uploads(files: list[UploadFile], root: Path) -> list[Path]:
    """把上传文件按净化后的相对路径写入规范根，返回写入文件的绝对路径列表。

    严格防穿越：拒绝绝对路径、包含 ``..`` 或空段、含盘符的路径。
    多选上传可能同名（不同文件夹都有 readme.md）-> 后者加序号去重，不互相覆盖。
    """
    written: list[Path] = []
    used: set[Path] = set()
    for upload in files:
        relative = _safe_relative_path(upload.filename or "")
        if not relative.name.lower().endswith(".md"):
            continue  # MVP 只认 Markdown，其余类型丢弃（不写入规范根）
        relative = _dedupe_path(relative, used)
        used.add(relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await upload.read())
        written.append(target)
    if not written:
        raise ValueError("所选内容中没有 Markdown 文件")
    return written


def _dedupe_path(relative: Path, used: set[Path]) -> Path:
    """同名冲突时给 stem 加序号（a.md -> a (1).md），返回唯一路径。"""
    if relative not in used:
        return relative
    stem, suffix, parent = relative.stem, relative.suffix, relative.parent
    index = 1
    candidate = parent / f"{stem} ({index}){suffix}"
    while candidate in used:
        index += 1
        candidate = parent / f"{stem} ({index}){suffix}"
    return candidate


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
    """进度快照 -> API dict：全量（parsed/chunks）与增量（processed）两种形态。"""
    progress = task.progress
    if progress is None:
        return {"files_scanned": 0, "files_parsed": 0, "files_skipped": 0, "chunks": 0}
    if hasattr(progress, "files_processed"):  # SyncProgress（增量同步）
        assert isinstance(progress, SyncProgress)
        return {
            "files_scanned": progress.files_scanned,
            "files_processed": progress.files_processed,
            "files_skipped": progress.files_skipped,
        }
    return {
        "files_scanned": progress.files_scanned,
        "files_parsed": progress.files_parsed,
        "files_skipped": progress.files_skipped,
        "chunks": progress.chunks,
    }


def _report_payload(report: IngestReport | SyncReport | None) -> dict | None:
    """导入/同步报告 -> API dict（按报告类型输出对应字段）。"""
    if report is None:
        return None
    if hasattr(report, "files_added"):  # SyncReport（增量同步）
        assert isinstance(report, SyncReport)
        return {
            "files_scanned": report.files_scanned,
            "files_added": report.files_added,
            "files_updated": report.files_updated,
            "files_deleted": report.files_deleted,
            "files_unchanged": report.files_unchanged,
            "files_skipped": report.files_skipped,
            "blocks_upserted": report.blocks_upserted,
            "blocks_deleted": report.blocks_deleted,
            "errors": list(report.errors),
        }
    return {
        "files_scanned": report.files_scanned,
        "files_parsed": report.files_parsed,
        "files_skipped": report.files_skipped,
        "chunks": report.chunks,
        "blocks_upserted": report.blocks_upserted,
        "errors": list(report.errors),
    }
