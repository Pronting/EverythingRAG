"""ingestion：编排层 —— 目录 → 扫描 → 解析 → 切块 → 嵌入 → 向量入库（MVP 任务 9）。

对齐 PRD 决策 6（增量同步，来源 + doc 内容指纹防覆盖）/ 决策 8（MVP 只认 Markdown）：
把任务 1-8 各模块（scanner / parser / chunker / embedder / vectorstore）串成一条
真实管线，导入一个目录的全部 MD。纯本地文件系统 + 内存计算 + 进程内向量库，
零出网。

隐私红线：单文件失败聚合为脱敏原因（仅异常类型名，不含文件路径/正文/密钥），
日志同样脱敏；同一来源同内容再导入幂等不重复（block_id 稳定，upsert 覆盖）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import xxhash
from charset_normalizer import from_bytes

from app.core.config import settings
from app.generation.base import VisionModel
from app.generation.vision import VisionProviderError
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.image_fetch import ImageFetcher, ImageFetchError
from app.ingestion.image_preprocess import ImageDecodeError, preprocess_image
from app.ingestion.image_state_store import ImageStateStore, ImageTask
from app.ingestion.image_worker import ImageWorker
from app.ingestion.index_schema import make_index_fingerprint
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.parallel import ordered_parallel_map
from app.ingestion.scanner import DiscoveredFile, scan_directory
from app.ingestion.state_store import DocumentStateStore
from app.models.schemas import BlockMetadata, SourceType
from app.vectorstore.base import VectorStore
from app.vectorstore.embedder import Embedder

logger = logging.getLogger(__name__)

#: 单文件失败原因的聚合上限（避免海量失败时错误列表无界增长）。
_MAX_ERRORS = 50


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


#: 语义切块：仅对长度超过该值、且含多个段落的文本块做语义切分（短块/单段不动）。
_SEMANTIC_MIN_LEN = 800
#: 相邻段落 embedding 余弦相似度低于该阈值视为话题边界，在此切分。
#: 取较低值，只切「清晰话题边界」，避免把同节内略不同的段落切碎。
_SEMANTIC_SPLIT_THRESHOLD = 0.30
#: 语义切分出的子块若低于该长度，并入相邻子块，避免产生短碎片。
_SEMANTIC_MIN_SUB_LEN = 120
_DOCUMENT_EMBED_BATCH_SIZE = 64
_INGEST_FILE_BATCH_SIZE = 32


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量已归一化时即点积，这里按通用余弦计算以兼容任意向量）。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9) if na and nb else 0.0


def _make_sub_chunk(chunk: Chunk, text: str, idx: int) -> Chunk:
    """语义切分出的子块：继承原块的来源/标题身份，block_id 加 ``-s{idx}`` 后缀保持唯一。"""
    return Chunk(
        block_id=f"{chunk.block_id}-s{idx}",
        text=text,
        source_file=chunk.source_file,
        heading_path=chunk.heading_path,
        anchor=chunk.anchor,
        seq=chunk.seq,
        kind=chunk.kind,
        heading_aliases=chunk.heading_aliases,
    )


#: 图片描述只携带同一精确标题下的去重摘要，避免顶级标题被映射成空父节后复制整篇。
_IMAGE_CONTEXT_MAX_LEN = 400


def chunk_embedding_text(chunk: Chunk) -> str:
    """构造只用于向量化的检索表示；Chroma 展示正文仍保存 ``chunk.text``。

    表格、公式和短列表往往不重复章节主题。把文档名与完整标题路径加入向量表示，
    可以让「AB 实验的基础指标」「首页运营位 SPM」等问题通过 dense 路由命中正确块，
    同时引用时不会把这些辅助前缀展示给用户。
    """

    parts: list[str] = []
    source_name = Path(chunk.source_file).stem.strip()
    if source_name:
        parts.append(f"文档：{source_name}")
    if chunk.heading_path.strip():
        parts.append(f"标题：{chunk.heading_path.strip()}")
    parts.append(f"内容：{chunk.text}")
    return "\n".join(parts)


def embed_document_texts(
    embedder: Embedder,
    texts: list[str],
    batch_size: int = _DOCUMENT_EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """有界批量走文档编码接口，兼容只实现历史接口的测试/第三方实现。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    embed_documents = getattr(embedder, "embed_documents", None)
    encode = embed_documents if callable(embed_documents) else embedder.embed_texts
    def encode_batch(batch: list[str]) -> list[list[float]]:
        vectors = encode(batch)
        if len(vectors) != len(batch):
            raise RuntimeError("文档嵌入返回数量错误")
        return vectors

    batches = (texts[start : start + batch_size] for start in range(0, len(texts), batch_size))
    return [vector for batch in ordered_parallel_map(
        encode_batch, batches, settings.ingest_embed_workers
    ) for vector in batch]


class IngestionPipeline:
    """编排层：把扫描 → 解析 → 切块 → 嵌入 → 入库串成一条管线。

    每个 DiscoveredFile：读 → parse_markdown → chunk_document → 批次嵌入 →
    按 block_id 幂等 upsert。单文件失败不中断整体，聚合脱敏原因。
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        vision: VisionModel | None = None,
        fetcher: ImageFetcher | None = None,
        state_store: ImageStateStore | None = None,
        document_state_store: DocumentStateStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._vision = vision
        self._fetcher = fetcher
        self._worker: ImageWorker | None = None
        #: 内容去重 + 任务持久化存储；None 时用内存态（单测 / 无持久化场景）。
        self._state_store = (
            state_store if state_store is not None else ImageStateStore(Path(":memory:"))
        )
        self._document_state_store = document_state_store
        self._index_fingerprint = make_index_fingerprint(embedder.fingerprint)
        #: 图文融合块正文缓存：source_file -> {精确 heading_path -> 去重摘要}。
        self._context_cache: dict[str, dict[str, str]] = {}

    def set_worker(self, worker: ImageWorker) -> None:
        """挂接识图后台工作线程：挂接后图片走后台补全，文本仍同步先入库。"""
        self._worker = worker

    def invalidate_image_context(self, source_file: str) -> None:
        """文本入库更新后清除该文件的图片上下文缓存。"""
        self._context_cache.pop(source_file, None)

    def semantic_split(self, chunks: list[Chunk]) -> list[Chunk]:
        """语义切块：仅对无标题长文按相邻段落 embedding 相似度骤降处切分。

        Markdown 标题已经是稳定且可解释的语义边界，对有标题块再次调用云端模型既会
        产生大量网络往返，也会让局部编辑改变邻段相似度并造成边界漂移。这里只补偿
        无标题长文：长度 ≥ ``_SEMANTIC_MIN_LEN`` 且含 ≥2 个非空段落时，低于阈值的
        邻段作为话题边界。代码、图片及所有带标题块原样保留。
        """
        plans: dict[int, tuple[list[str], int, int]] = {}
        paragraph_inputs: list[str] = []
        for index, chunk in enumerate(chunks):
            if (
                chunk.kind != "text"
                or chunk.heading_path.strip()
                or len(chunk.text) < _SEMANTIC_MIN_LEN
            ):
                continue
            paragraphs = [p for p in chunk.text.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                continue
            start = len(paragraph_inputs)
            paragraph_inputs.extend(paragraphs)
            plans[index] = (paragraphs, start, len(paragraphs))

        if not plans:
            return chunks
        all_vectors = embed_document_texts(self._embedder, paragraph_inputs)
        if len(all_vectors) != len(paragraph_inputs):
            raise RuntimeError(
                "语义切块向量数量错误："
                f"期望 {len(paragraph_inputs)}，实际 {len(all_vectors)}"
            )

        result: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            plan = plans.get(index)
            if plan is None:
                result.append(chunk)
                continue
            paragraphs, start, count = plan
            vectors = all_vectors[start : start + count]
            splits: list[int] = []
            for i in range(len(vectors) - 1):
                if _cosine(vectors[i], vectors[i + 1]) < _SEMANTIC_SPLIT_THRESHOLD:
                    splits.append(i + 1)
            if not splits:
                result.append(chunk)
                continue
            # 按切分点拆成子块，再合并过短的子块（避免产生短碎片）
            sub_texts: list[str] = []
            prev = 0
            for sp in splits:
                sub_texts.append("\n\n".join(paragraphs[prev:sp]))
                prev = sp
            sub_texts.append("\n\n".join(paragraphs[prev:]))
            merged: list[str] = []
            for text in sub_texts:
                if merged and len(text) < _SEMANTIC_MIN_SUB_LEN:
                    merged[-1] += "\n\n" + text
                else:
                    merged.append(text)
            for idx, text in enumerate(merged):
                result.append(_make_sub_chunk(chunk, text, idx))
        return result

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
        def complete_file(file: DiscoveredFile, chunks: list[Chunk], upserted: int) -> None:
            nonlocal files_parsed, files_skipped, chunks_total, upserted_total
            try:
                if self._document_state_store is not None:
                    self._document_state_store.upsert_fingerprint(
                        str(file.path),
                        file.content_hash,
                        file.mtime,
                        file.size,
                        len(chunks),
                        index_fingerprint=self._index_fingerprint,
                    )
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

        for start in range(0, len(discovered), _INGEST_FILE_BATCH_SIZE):
            prepared: list[tuple[DiscoveredFile, list[Chunk]]] = []
            for file, chunks, error in self.prepare_files(
                discovered[start : start + _INGEST_FILE_BATCH_SIZE]
            ):
                try:
                    if error is not None:
                        raise error
                    prepared.append((file, chunks))
                except Exception as exc:  # noqa: BLE001 -- 单文件解析失败继续批次
                    files_skipped += 1
                    self._record_error(exc, errors)
                    if on_progress is not None:
                        on_progress(
                            ProgressSnapshot(
                                files_scanned=len(discovered),
                                files_parsed=files_parsed,
                                files_skipped=files_skipped,
                                chunks=chunks_total,
                            )
                        )
            if not prepared:
                continue
            try:
                upserted_counts = self.upsert_prepared_files(prepared)
            except Exception:  # noqa: BLE001 -- 批次失败时逐文件隔离异常
                for file, chunks in prepared:
                    try:
                        (upserted,) = self.upsert_prepared_files([(file, chunks)])
                    except Exception as exc:  # noqa: BLE001 -- 坏文件不阻断其余文件
                        files_skipped += 1
                        self._record_error(exc, errors)
                        if on_progress is not None:
                            on_progress(
                                ProgressSnapshot(
                                    files_scanned=len(discovered),
                                    files_parsed=files_parsed,
                                    files_skipped=files_skipped,
                                    chunks=chunks_total,
                                )
                            )
                    else:
                        complete_file(file, chunks, upserted)
            else:
                for (file, chunks), upserted in zip(
                    prepared, upserted_counts, strict=True
                ):
                    complete_file(file, chunks, upserted)
        if self._document_state_store is not None:
            self._document_state_store.register_source(Path(root_dir).expanduser().resolve())
        repair_ann = getattr(self._vectorstore, "repair_ann_index", None)
        if upserted_total and callable(repair_ann):
            # 单批写入自检无法发现后续批次使早期 HNSW 节点脱离 ANN 的情况；
            # 全量导入结束后做一次全库修复，确保刚入库的块在生产 Top-8 可达。
            repair_ann()
        return IngestReport(
            files_scanned=len(discovered),
            files_parsed=files_parsed,
            files_skipped=files_skipped,
            chunks=chunks_total,
            blocks_upserted=upserted_total,
            errors=tuple(errors),
        )

    def ingest_file(self, file: DiscoveredFile) -> tuple[list[Chunk], int]:
        """单文件：读 → 解析 → 切块 → 批次嵌入 → upsert，返回 (chunks, upserted)。

        公共入口（增量同步复用）：与全量 `ingest` 内单文件处理路径完全一致。
        """
        chunks = self.prepare_file(file)
        (upserted,) = self.upsert_prepared_files([(file, chunks)])
        return chunks, upserted

    def prepare_file(self, file: DiscoveredFile) -> list[Chunk]:
        """读取、解析并切块，但不写向量库；供跨文件批量嵌入复用。"""

        text = read_text(file.path)
        parsed = parse_markdown(text)
        chunks = chunk_document(parsed, source_file=str(file.path))
        return self.semantic_split(chunks) if chunks else chunks

    def prepare_files(
        self, files: list[DiscoveredFile]
    ) -> Iterator[tuple[DiscoveredFile, list[Chunk], Exception | None]]:
        """Read/parse independently; return errors per file and preserve source order."""
        def prepare(file: DiscoveredFile) -> tuple[DiscoveredFile, list[Chunk], Exception | None]:
            try:
                return file, self.prepare_file(file), None
            except Exception as exc:  # noqa: BLE001 -- isolate a malformed document
                return file, [], exc

        return ordered_parallel_map(prepare, files, settings.ingest_parse_workers)

    def upsert_prepared_files(
        self,
        prepared: list[tuple[DiscoveredFile, list[Chunk]]],
    ) -> list[int]:
        """一次嵌入多份已切文档，再按文件幂等写库；返回逐文件写入块数。"""

        flattened = [
            (file, chunk)
            for file, chunks in prepared
            for chunk in chunks
            if chunk.kind != "image"
        ]
        vectors = embed_document_texts(
            self._embedder,
            [chunk_embedding_text(chunk) for _file, chunk in flattened],
        )
        if len(vectors) != len(flattened):
            raise RuntimeError(
                f"文档嵌入返回数量错误：期望 {len(flattened)}，实际 {len(vectors)}"
            )
        # Commit text across files in bounded writes, then enqueue images and persist fingerprints.
        text_blocks = []
        for (file, chunk), vector in zip(flattened, vectors, strict=True):
            block_id = make_block_id(file.content_hash, chunk.block_id, source_file=file.path)
            text_blocks.append(
                (block_id, chunk.text, vector, make_chunk_metadata(block_id, file, chunk))
            )
        for start in range(0, len(text_blocks), 512):
            self._vectorstore.upsert(text_blocks[start : start + 512])
        counts: list[int] = []
        for file, chunks in prepared:
            upserted = sum(chunk.kind != "image" for chunk in chunks)
            if upserted:
                self.invalidate_image_context(str(file.path))

            image_chunks = [chunk for chunk in chunks if chunk.kind == "image"]
            if self._worker is not None:
                for chunk in image_chunks:
                    task = self.make_image_task(file, chunk)
                    if task is not None:
                        self._worker.enqueue(task)
            else:
                blocks = self.sync_image_blocks(file, image_chunks)
                if blocks:
                    self._vectorstore.upsert(blocks)
                    upserted += len(blocks)
            counts.append(upserted)
        return counts

    def make_image_task(self, file: DiscoveredFile, chunk: Chunk) -> ImageTask | None:
        """把图片引用块转为持久化任务（无 src 时返回 None）。"""
        src = chunk.src
        if not src:
            return None
        return ImageTask(
            block_id=make_block_id(
                file.content_hash,
                chunk.block_id,
                source_file=file.path,
            ),
            src=src,
            doc_id=file.content_hash,
            source_file=str(file.path),
            heading_path=chunk.heading_path,
            anchor=chunk.anchor,
        )

    def sync_image_blocks(
        self,
        file: DiscoveredFile,
        image_chunks: list[Chunk],
    ) -> list[tuple[str, str, list[float], BlockMetadata]]:
        """同步处理图片引用（无 worker 兜底）：逐引用 process_image_task，聚合成功块。"""
        blocks: list[tuple[str, str, list[float], BlockMetadata]] = []
        for chunk in image_chunks:
            task = self.make_image_task(file, chunk)
            if task is None:
                continue
            try:
                block = self.process_image_task(task)
            except Exception as exc:  # noqa: BLE001 -- 图片路径任何异常不拖累文本
                logger.debug("图片处理异常（脱敏）: %s", type(exc).__name__)
                continue
            if block is not None:
                blocks.append(block)
        return blocks

    def _surrounding_text(self, task: ImageTask, max_len: int = _IMAGE_CONTEXT_MAX_LEN) -> str:
        """图片所属精确标题区间的去重正文摘要，按 source_file 缓存。

        旧逻辑按「父标题」聚合：顶级标题的父标题为空，导致所有顶级章节甚至整篇正文
        被复制进每张图；H2 图片也会混入兄弟章节。这里改为精确 heading_path，并对重复
        段落去重、严格限长，图片的核心检索文本仍是标题路径与识图描述。
        """
        source = task.source_file
        if source not in self._context_cache:
            by_heading: dict[str, list[str]] = {}
            try:
                blocks = self._vectorstore.list_blocks()
            except Exception:  # noqa: BLE001 -- 测试假向量库/最小实现可能无 list_blocks，融合降级为空
                blocks = []
            for _, text, meta in blocks:
                if meta.get("source_file") != source or meta.get("chunk_type") not in {
                    "text",
                    "table",
                }:
                    continue
                heading = meta.get("heading_path") or ""
                by_heading.setdefault(heading, []).append(text)
            self._context_cache[source] = {
                heading: _compact_context(texts, max_len) for heading, texts in by_heading.items()
            }
        heading = task.heading_path or ""
        return self._context_cache[source].get(heading, "")

    def process_image_task(
        self,
        task: ImageTask,
    ) -> tuple[str, str, list[float], BlockMetadata] | None:
        """处理单个图片引用：历史描述恢复 / 下载 → 去重 → 识图 → 嵌入。

        返回描述块或 None（失败：死链/非图/识图失败）。同图（同内容哈希）只识一次、
        描述持久化复用（跨 URL / 跨导入去重）；URL 失效时优先恢复旧 collection 中
        已生成的描述。同图不同标题仍分别嵌入（语境不同）。
        """
        cached = self._state_store.get_source_description(task.src)
        if cached is None:
            recover = getattr(self._vectorstore, "find_cached_image_description_by_source", None)
            if callable(recover):
                try:
                    cached = recover(task.src)
                except Exception as exc:  # noqa: BLE001 -- 旧 collection 损坏不阻断正常下载
                    logger.debug("历史图片描述恢复失败（脱敏）: %s", type(exc).__name__)
                if cached is not None:
                    self._state_store.save_description(cached[0], cached[1])
                    self._state_store.save_source_description(task.src, cached[0], cached[1])

        if cached is not None:
            content_hash, description = cached
        else:
            if self._vision is None or self._fetcher is None:
                return None
            try:
                raw = self._fetcher.fetch(task.src)
                jpeg = preprocess_image(raw)
            except (ImageFetchError, ImageDecodeError) as exc:  # 死链/非图/超限：失败
                logger.debug("图片跳过（脱敏）: %s", type(exc).__name__)
                return None
            content_hash = xxhash.xxh64(jpeg).hexdigest()
            description = self._state_store.get_description(content_hash)
            if description is None:
                try:
                    describe = getattr(self._vision, "describe_document", self._vision.describe)
                    description = describe(preprocess_image(raw, max_edge=2560, max_pixels=4_000_000))
                except VisionProviderError as exc:
                    logger.debug("识图失败（脱敏）: %s", type(exc).__name__)
                    return None
                self._state_store.save_description(content_hash, description)
            self._state_store.save_source_description(task.src, content_hash, description)
            self._state_store.save_image_bytes(content_hash, raw)
        from app.ingestion.image_document import IMAGE_REPRESENTATION_VERSION, build_image_document
        document = build_image_document(description, task.heading_path, Path(task.source_file).stem)
        vector = self._embedder.embed_texts([document.index_text])[0]
        metadata = BlockMetadata(
            block_id=task.block_id,
            doc_id=task.doc_id,
            source_type=SourceType.IMAGE_DESCRIPTION,
            source_file=task.source_file,
            platform="local",
            heading_path=task.heading_path or None,
            anchor=task.anchor or None,
            chunk_type="image_description",
            image_path=task.src,
            image_content_hash=content_hash,
            image_index_text=document.index_text,
            image_representation_version=IMAGE_REPRESENTATION_VERSION,
        )
        return (task.block_id, document.evidence_text, vector, metadata)

    def process_image_task_and_upsert(self, task: ImageTask) -> bool:
        """处理单个图片引用并 upsert（后台 worker 调用）；返回是否成功。"""
        block = self.process_image_task(task)
        if block is None:
            return False
        self._vectorstore.upsert([block])
        return True

    @staticmethod
    def _record_error(exc: Exception, errors: list[str]) -> None:
        """聚合脱敏失败原因：仅异常类型名（不含路径/正文/密钥），去重 + 封顶。"""
        reason = exc.__class__.__name__ or "UnknownError"
        logger.debug("单文件导入失败（脱敏）: %s", reason)
        if reason not in errors and len(errors) < _MAX_ERRORS:
            errors.append(reason)


def read_text(path: Path) -> str:
    """读取文本文件：UTF-8 优先；中文常见编码 gb18030（GBK 超集）次之；
    再尝试 charset-normalizer 检测；最终 errors=replace 兜底（不抛错）。

    公共助手，全量导入与增量同步（sync.py）共用同一编码容错读取。
    """
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


def make_block_id(
    content_hash: str,
    chunk_block_id: str,
    source_file: str | Path,
) -> str:
    """构造来源感知、确定且不暴露绝对路径的全局块 ID。

    只用文档内容哈希与块锚点时，两个不同文件只要内容相同就会产生相同 ID，
    Chroma 的 upsert 随后会覆盖其中一个来源。这里把规范化 ``source_file`` 的
    128-bit SHA-256 摘要纳入身份；ID 中仅保留摘要，不写入绝对路径明文。

    公共助手，全量导入、增量同步与图片任务共用同一块身份规则。
    """
    normalized_source = _normalize_source_file(source_file)
    source_hash = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()[:32]
    return f"{source_hash}::{content_hash}::{chunk_block_id}"


def _normalize_source_file(source_file: str | Path) -> str:
    """把来源路径规范化为稳定的内部哈希输入，不把结果写入 ID 或 trace。"""
    raw = os.fspath(source_file)
    if not raw.strip():
        raise ValueError("source_file must not be empty")
    path = Path(raw).expanduser()
    try:
        path = path.resolve(strict=False)
    except (OSError, RuntimeError):
        # 极端路径/链接异常时仍给出确定性绝对形式；不把该值暴露给调用方。
        path = path.absolute()
    normalized = os.path.normcase(os.path.normpath(str(path))).replace("\\", "/")
    return unicodedata.normalize("NFC", normalized)


def make_chunk_metadata(block_id: str, file: DiscoveredFile, chunk: Chunk) -> BlockMetadata:
    """组装块元数据：doc_id 用内容指纹（幂等去重键）；来源与锚点取自 Chunk。

    公共助手，全量导入与增量同步（sync.py）共用同一元数据规则。
    """
    return BlockMetadata(
        block_id=block_id,
        doc_id=file.content_hash,
        source_type=SourceType.DOCUMENT,
        source_file=str(file.path),
        platform="local",
        heading_path=chunk.heading_path or None,
        heading_aliases=(
            json.dumps(chunk.heading_aliases, ensure_ascii=False)
            if chunk.heading_aliases
            else None
        ),
        anchor=chunk.anchor or None,
        chunk_type=chunk.kind if chunk.kind in {"table", "code"} else "text",
    )


def _image_embed_text(description: str, heading_path: str, context: str = "") -> str:
    """图片描述块的嵌入/存储文本：注入标题路径 + 同父节正文作为主题上下文。

    图片本身常不含主题词（如图在「AB 测试」标题下但图里没写「AB 测试」），
    把标题路径与正文上下文拼进文本，使图片块能被主题词检索命中、信息密度更高。
    无标题/无上下文时相应省略，不产生空段落。
    """
    parts: list[str] = []
    if heading_path:
        parts.append(f"【所属主题】{heading_path}")
    if context:
        parts.append(f"【正文上下文】{context}")
    parts.append(f"【图片描述】{description}")
    return "\n\n".join(parts)


def _compact_context(texts: list[str], max_len: int) -> str:
    """按段落稳定去重并限长；不记录正文，不做外部调用。"""
    if max_len <= 0:
        return ""
    seen: set[str] = set()
    selected: list[str] = []
    used = 0
    for text in texts:
        for paragraph in text.split("\n\n"):
            paragraph = " ".join(paragraph.split()).strip()
            if not paragraph:
                continue
            key = paragraph.casefold()
            if key in seen:
                continue
            seen.add(key)
            separator_len = 2 if selected else 0
            remaining = max_len - used - separator_len
            if remaining <= 0:
                return "\n\n".join(selected)
            selected.append(paragraph[:remaining])
            used += separator_len + min(len(paragraph), remaining)
            if len(paragraph) > remaining:
                return "\n\n".join(selected)
    return "\n\n".join(selected)
