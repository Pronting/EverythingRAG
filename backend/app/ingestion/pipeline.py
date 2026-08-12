"""ingestion：编排层 —— 目录 → 扫描 → 解析 → 切块 → 嵌入 → 向量入库（MVP 任务 9）。

对齐 PRD 决策 6（增量同步，doc 内容指纹前缀防重复）/ 决策 8（MVP 只认 Markdown）：
把任务 1-8 各模块（scanner / parser / chunker / embedder / vectorstore）串成一条
真实管线，导入一个目录的全部 MD。纯本地文件系统 + 内存计算 + 进程内向量库，
零出网。

隐私红线：单文件失败聚合为脱敏原因（仅异常类型名，不含文件路径/正文/密钥），
日志同样脱敏；同内容再导入幂等不重复（block_id 稳定，upsert 覆盖）。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.scanner import DiscoveredFile, scan_directory
from app.models.schemas import BlockMetadata, SourceType
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)

#: 单文件失败原因的聚合上限（避免海量失败时错误列表无界增长）。
_MAX_ERRORS = 50

#: 纯文本导入的块类型（代码块在 Chunk 层不携带类型，文档块统一记 text）。
_CHUNK_TYPE = "text"


@dataclass(frozen=True)
class IngestReport:
    """一次导入的汇总报告（字段全为计数；errors 为脱敏聚合原因）。"""

    files_scanned: int
    files_parsed: int
    files_skipped: int
    chunks: int
    blocks_upserted: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgressSnapshot:
    """导入进行中的运行计数快照（每处理一个文件回调一次）。"""

    files_scanned: int
    files_parsed: int
    files_skipped: int
    chunks: int


class IngestionPipeline:
    """编排层：把扫描 → 解析 → 切块 → 嵌入 → 入库串成一条管线。

    每个 DiscoveredFile：读 → parse_markdown → chunk_document → 批次嵌入 →
    按 block_id 幂等 upsert。单文件失败不中断整体，聚合脱敏原因。
    """

    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore

    def ingest(
        self,
        root_dir: Path,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
    ) -> IngestReport:
        """导入 ``root_dir`` 下全部 MD，返回汇总报告（空目录返回全 0，不抛错）。

        on_progress 可选：每处理一个文件后回调一次运行计数快照（含成功与跳过），
        供调用方展示实时进度；缺省 None 时零回调、行为不变。
        """
        discovered = scan_directory(root_dir)
        errors: list[str] = []
        files_parsed = 0
        files_skipped = 0
        chunks_total = 0
        upserted_total = 0
        for file in discovered:
            try:
                chunks, upserted = self._ingest_file(file)
            except Exception as exc:  # noqa: BLE001 -- 单文件失败不中断整体导入
                files_skipped += 1
                self._record_error(exc, errors)
            else:
                files_parsed += 1
                chunks_total += len(chunks)
                upserted_total += upserted
            if on_progress is not None:
                on_progress(
                    ProgressSnapshot(
                        files_scanned=len(discovered),
                        files_parsed=files_parsed,
                        files_skipped=files_skipped,
                        chunks=chunks_total,
                    )
                )
        return IngestReport(
            files_scanned=len(discovered),
            files_parsed=files_parsed,
            files_skipped=files_skipped,
            chunks=chunks_total,
            blocks_upserted=upserted_total,
            errors=tuple(errors),
        )

    def _ingest_file(self, file: DiscoveredFile) -> tuple[list[Chunk], int]:
        """单文件：读 → 解析 → 切块 → 批次嵌入 → upsert，返回 (chunks, upserted)。"""
        text = _read_text(file.path)
        parsed = parse_markdown(text)
        chunks = chunk_document(parsed, source_file=str(file.path))
        if not chunks:
            return chunks, 0
        vectors = self._embedder.embed_texts([chunk.text for chunk in chunks])
        blocks: list[tuple[str, str, list[float], BlockMetadata]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            block_id = _block_id(file.content_hash, chunk.block_id)
            blocks.append((block_id, chunk.text, vector, _to_metadata(block_id, file, chunk)))
        self._vectorstore.upsert(blocks)
        return chunks, len(blocks)

    @staticmethod
    def _record_error(exc: Exception, errors: list[str]) -> None:
        """聚合脱敏失败原因：仅异常类型名（不含路径/正文/密钥），去重 + 封顶。"""
        reason = exc.__class__.__name__ or "UnknownError"
        logger.debug("单文件导入失败（脱敏）: %s", reason)
        if reason not in errors and len(errors) < _MAX_ERRORS:
            errors.append(reason)


def _read_text(path: Path) -> str:
    """读取文本文件：UTF-8 优先；中文常见编码 gb18030（GBK 超集）次之；
    再尝试 charset-normalizer 检测；最终 errors=replace 兜底（不抛错）。"""
    raw = path.read_bytes()
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    best = from_bytes(raw).best()
    if best is not None:
        return str(best)
    return raw.decode("utf-8", errors="replace")


def _block_id(content_hash: str, chunk_block_id: str) -> str:
    """全局唯一 block_id：doc 内容指纹前缀 + 锚点路径链（对齐 PRD hash(doc)::锚点）。"""
    return f"{content_hash}::{chunk_block_id}"


def _to_metadata(block_id: str, file: DiscoveredFile, chunk: Chunk) -> BlockMetadata:
    """组装块元数据：doc_id 用内容指纹（幂等去重键）；来源与锚点取自 Chunk。"""
    return BlockMetadata(
        block_id=block_id,
        doc_id=file.content_hash,
        source_type=SourceType.DOCUMENT,
        source_file=str(file.path),
        platform="local",
        heading_path=chunk.heading_path or None,
        anchor=chunk.anchor or None,
        chunk_type=_CHUNK_TYPE,
    )
