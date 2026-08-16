"""ingestion：增量同步核心（v0.1）—— 指纹快路径 + 块级复用。

把「上次入库的指纹」（state.db）与「当前磁盘快照」（scan_files 轻量扫描）
比对，分类 新增 / 更新 / 删除 / 未变：

- 未变（mtime+size 或内容哈希均同）-> 毫秒级跳过，零重嵌入；
- 新增 -> 走与全量导入一致的单文件管线（IngestionPipeline.ingest_file）；
- 更新 -> 块级复用：按文本哈希匹配 Chroma 中未变的块（复用旧向量，
  不重嵌入），只嵌入内容变化的块、删除陈旧块（对齐 PRD「改一处只更新相关块」）；
- 删除 -> 级联清掉该文件全部块与指纹。

对齐 PRD 决策 6 / §4.5：指纹判断变更（文件级），去重键/块身份判断同源（块级）。
零出网：全部本地文件系统 + 进程内向量库 + SQLite。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import xxhash

from app.generation.base import VisionModel
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.image_fetch import ImageFetcher
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.image_worker import ImageWorker
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.pipeline import IngestionPipeline, make_block_id, make_chunk_metadata, read_text
from app.ingestion.scanner import DiscoveredFile, discover_file, scan_files
from app.ingestion.state_store import DocumentStateStore
from app.models.schemas import BlockMetadata
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)

#: 单文件失败原因的聚合上限（与全量导入一致，避免海量失败时错误列表无界增长）。
_MAX_ERRORS = 50


@dataclass(frozen=True)
class SyncProgress:
    """同步进行中的运行计数快照（每处理一个文件回调一次）。"""

    files_scanned: int
    files_processed: int  # 已处理文件数（新增 + 更新 + 删除）
    files_skipped: int  # 处理失败数


@dataclass(frozen=True)
class SyncReport:
    """一次同步的汇总报告（计数 + 脱敏错误原因）。"""

    files_scanned: int
    files_added: int
    files_updated: int
    files_deleted: int
    files_unchanged: int
    files_skipped: int  # 处理失败 / 不可读
    blocks_upserted: int
    blocks_deleted: int
    errors: tuple[str, ...] = ()


class SyncService:
    """增量同步编排：指纹比对 + 三种动作 + 块级复用。

    ingest 单根；ingest_many 聚合多根（/api/sync 一键同步已注册根）。
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        state_store: DocumentStateStore,
        vision: VisionModel | None = None,
        fetcher: ImageFetcher | None = None,
        worker: ImageWorker | None = None,
        image_state_store: ImageStateStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._state_store = state_store
        if image_state_store is None:
            image_state_store = ImageStateStore(Path(":memory:"))
        self._image_state_store = image_state_store
        self._pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=vectorstore,
            vision=vision,
            fetcher=fetcher,
            state_store=image_state_store,
        )
        # 识图已配置时自动挂后台工作线程（文本先入库、图片后台补全 + 持久化续跑）。
        if worker is None and vision is not None and fetcher is not None:
            worker = ImageWorker(
                self._pipeline.process_image_task_and_upsert,
                image_state_store,
            )
            worker.start()
        self._worker = worker
        if worker is not None:
            self._pipeline.set_worker(worker)

    def ingest(
        self,
        root: Path,
        on_progress: object | None = None,
    ) -> SyncReport:
        """对 ``root`` 目录做一次增量同步，返回汇总报告。

        on_progress 可选：每处理一个文件回调 SyncProgress（含跳过），供进度展示。

        Raises:
            ValueError: ``root`` 不存在或不是目录。
        """
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"同步根目录不存在或非目录: {root}")

        files = scan_files(root)
        manifest = self._state_store.load_all()
        files_by_key = {str(file.path): file for file in files}

        # ---- 分类（快路径 mtime+size -> 需确认再读内容哈希）----
        added: list[DiscoveredFile] = []
        updated: list[DiscoveredFile] = []
        unchanged = 0
        unreadable = 0
        for stat in files:
            key = str(stat.path)
            old = manifest.get(key)
            if old is None:
                disc = discover_file(stat.path)
                if disc is None:
                    unreadable += 1
                    continue
                added.append(disc)
            elif old.mtime == stat.mtime and old.size == stat.size:
                unchanged += 1
            else:
                disc = discover_file(stat.path)
                if disc is None:
                    unreadable += 1
                    continue
                if disc.content_hash == old.content_hash:
                    # 内容未变仅 mtime 被碰 -> 视为未变，只刷新时间戳
                    unchanged += 1
                    self._state_store.touch_mtime(key, disc.mtime)
                else:
                    updated.append(disc)
        deleted = [key for key in manifest if key not in files_by_key]

        report = self._process(
            added,
            updated,
            deleted,
            scanned=len(files),
            unchanged=unchanged,
            skipped=unreadable,
            on_progress=on_progress,
        )
        # 登记来源根（供前端一键同步）
        self._state_store.register_source(root)
        return report

    def sync_upload(
        self,
        files: list[DiscoveredFile],
        deletion_scopes: list[Path] | None = None,
        on_progress: object | None = None,
    ) -> SyncReport:
        """上传式增量同步：对已落盘规范根的显式文件清单做指纹 diff。

        files: 本次上传的文件（DiscoverableFile，路径为规范根下的绝对路径）。
        deletion_scopes: 本次重传的顶层目录；其下已知但不在本清单的指纹视为删除。
            散文件不属于任何 scope -> 永不删除，只新增/更新。缺省空 = 不删除任何文件。

        与 ingest 的区别：快照由调用方给出（上传已落盘），而非目录扫描；
        删除作用域由显式 scopes 限定（避免误删未重传的其它范围）。
        """
        scopes = [Path(scope) for scope in (deletion_scopes or [])]
        manifest = self._state_store.load_all()
        current_keys = {str(file.path) for file in files}

        added: list[DiscoveredFile] = []
        updated: list[DiscoveredFile] = []
        unchanged = 0
        for file in files:
            old = manifest.get(str(file.path))
            if old is None:
                added.append(file)
            elif old.content_hash == file.content_hash:
                unchanged += 1
            else:
                updated.append(file)

        deleted = [
            key
            for key in manifest
            if key not in current_keys and any(Path(key).is_relative_to(scope) for scope in scopes)
        ]

        return self._process(
            added,
            updated,
            deleted,
            scanned=len(files),
            unchanged=unchanged,
            skipped=0,
            on_progress=on_progress,
        )

    def _process(
        self,
        added: list[DiscoveredFile],
        updated: list[DiscoveredFile],
        deleted: list[str],
        scanned: int,
        unchanged: int,
        skipped: int,
        on_progress: object | None,
    ) -> SyncReport:
        """共享执行循环：新增入库 / 更新块级复用 / 删除级联清块，汇总报告。

        ingest 与 sync_upload 共用；单文件失败聚合脱敏、不中断整体。
        """
        errors: list[str] = []
        added_ok = updated_ok = deleted_ok = 0
        blocks_upserted = 0
        blocks_deleted = 0
        processed = 0

        def emit() -> None:
            if on_progress is not None:
                on_progress(
                    SyncProgress(
                        files_scanned=scanned,
                        files_processed=processed,
                        files_skipped=len(errors),
                    )
                )

        for disc in added:
            try:
                chunks, upserted = self._pipeline.ingest_file(disc)
                self._state_store.upsert_fingerprint(
                    str(disc.path), disc.content_hash, disc.mtime, disc.size, len(chunks)
                )
                added_ok += 1
                blocks_upserted += upserted
            except Exception as exc:  # noqa: BLE001 -- 单文件失败不中断整体同步
                self._record_error(exc, errors)
            processed += 1
            emit()

        for disc in updated:
            try:
                upserted, deleted_blocks = self._reprocess_modified(disc)
                updated_ok += 1
                blocks_upserted += upserted
                blocks_deleted += deleted_blocks
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc, errors)
            processed += 1
            emit()

        for key in deleted:
            try:
                old_blocks = self._vectorstore.get_blocks_by_source(key)
                if old_blocks:
                    self._vectorstore.delete_by_ids(list(old_blocks))
                    blocks_deleted += len(old_blocks)
                self._state_store.delete(key)
                deleted_ok += 1
            except Exception as exc:  # noqa: BLE001
                self._record_error(exc, errors)
            processed += 1
            emit()

        return SyncReport(
            files_scanned=scanned,
            files_added=added_ok,
            files_updated=updated_ok,
            files_deleted=deleted_ok,
            files_unchanged=unchanged,
            files_skipped=len(errors) + skipped,
            blocks_upserted=blocks_upserted,
            blocks_deleted=blocks_deleted,
            errors=tuple(errors),
        )

    def ingest_many(
        self,
        roots: list[Path],
        on_progress: object | None = None,
    ) -> SyncReport:
        """对多个已注册根依次同步，汇总为单个报告（/api/sync 一键同步用）。

        进度按根聚合回调（每完成一个根推一次累计快照）。

        失效来源根（目录已被移动/删除）会跳过并自动移除登记，不中断其余根——
        否则残留的失效登记会让每次一键同步都失败（见 state_store.sources 治理）。
        """
        cumulative = SyncReport(0, 0, 0, 0, 0, 0, 0, 0, ())
        for root in roots:
            if not Path(root).is_dir():
                # 目录已不存在：移除失效登记，跳过（单根失败不中断整体）
                self._state_store.delete_source(root)
                cumulative = _add_reports(
                    cumulative,
                    SyncReport(
                        files_scanned=0,
                        files_added=0,
                        files_updated=0,
                        files_deleted=0,
                        files_unchanged=0,
                        files_skipped=1,
                        blocks_upserted=0,
                        blocks_deleted=0,
                        errors=("DirectoryNotFound",),
                    ),
                )
            else:
                sub = self.ingest(root)
                cumulative = _add_reports(cumulative, sub)
            if on_progress is not None:
                on_progress(
                    SyncProgress(
                        files_scanned=cumulative.files_scanned,
                        files_processed=(
                            cumulative.files_added
                            + cumulative.files_updated
                            + cumulative.files_deleted
                        ),
                        files_skipped=cumulative.files_skipped,
                    )
                )
        return cumulative

    def _reprocess_modified(self, disc: DiscoveredFile) -> tuple[int, int]:
        """块级复用：对已变更文件只嵌入变化块，复用未变块旧向量。

        返回 (blocks_upserted, blocks_deleted)。
        """
        text = read_text(disc.path)
        parsed = parse_markdown(text)
        chunks = chunk_document(parsed, source_file=str(disc.path))
        text_chunks = [chunk for chunk in chunks if chunk.kind != "image"]
        image_chunks = [chunk for chunk in chunks if chunk.kind == "image"]

        # 旧块文本哈希 -> block_id（读取便宜，只取 documents）
        old_blocks = self._vectorstore.get_blocks_by_source(str(disc.path))
        old_by_hash: dict[str, str] = {}
        for block_id, block_text in old_blocks.items():
            old_by_hash.setdefault(_text_hash(block_text), block_id)

        matched: set[str] = set()
        to_embed: list[Chunk] = []
        for chunk in text_chunks:
            old_id = old_by_hash.get(_text_hash(chunk.text))
            if old_id is not None and old_id not in matched:
                matched.add(old_id)  # 文本未变 -> 复用旧向量，不重嵌入
            else:
                to_embed.append(chunk)

        # 文本块先入库（快路径）；图片块后补——图片失败不拖累文本（T5 §5.3 文本先行）。
        upserted = 0
        if to_embed:
            vectors = self._embedder.embed_texts([chunk.text for chunk in to_embed])
            text_blocks: list[tuple[str, str, list[float], BlockMetadata]] = []
            for chunk, vector in zip(to_embed, vectors, strict=True):
                block_id = make_block_id(disc.content_hash, chunk.block_id)
                text_blocks.append(
                    (block_id, chunk.text, vector, make_chunk_metadata(block_id, disc, chunk))
                )
            self._vectorstore.upsert(text_blocks)
            upserted = len(text_blocks)
        # 图片块：重新下载 + 识图 + 嵌入（同图去重走持久化），不进文本块级复用。
        if self._worker is not None:
            for chunk in image_chunks:
                task = self._pipeline.make_image_task(disc, chunk)
                if task is not None:
                    self._worker.enqueue(task)
        else:
            blocks = self._pipeline.sync_image_blocks(disc, image_chunks)
            if blocks:
                self._vectorstore.upsert(blocks)
                upserted += len(blocks)

        stale = [block_id for block_id in old_blocks if block_id not in matched]
        if stale:
            self._vectorstore.delete_by_ids(stale)

        self._state_store.upsert_fingerprint(
            str(disc.path), disc.content_hash, disc.mtime, disc.size, len(chunks)
        )
        return upserted, len(stale)

    @staticmethod
    def _record_error(exc: Exception, errors: list[str]) -> None:
        """聚合脱敏失败原因：仅异常类型名（不含路径/正文/密钥），去重 + 封顶。"""
        reason = exc.__class__.__name__ or "UnknownError"
        logger.debug("单文件同步失败（脱敏）: %s", reason)
        if reason not in errors and len(errors) < _MAX_ERRORS:
            errors.append(reason)


def _text_hash(text: str) -> str:
    """块级去重键：块文本的 xxh64（跨版本稳定，识别「内容未变的块」）。"""
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()


def _add_reports(a: SyncReport, b: SyncReport) -> SyncReport:
    """聚合两个同步报告（计数求和；错误去重 + 封顶）。"""
    errors = list(a.errors)
    for reason in b.errors:
        if reason not in errors and len(errors) < _MAX_ERRORS:
            errors.append(reason)
    return SyncReport(
        files_scanned=a.files_scanned + b.files_scanned,
        files_added=a.files_added + b.files_added,
        files_updated=a.files_updated + b.files_updated,
        files_deleted=a.files_deleted + b.files_deleted,
        files_unchanged=a.files_unchanged + b.files_unchanged,
        files_skipped=a.files_skipped + b.files_skipped,
        blocks_upserted=a.blocks_upserted + b.blocks_upserted,
        blocks_deleted=a.blocks_deleted + b.blocks_deleted,
        errors=tuple(errors),
    )
