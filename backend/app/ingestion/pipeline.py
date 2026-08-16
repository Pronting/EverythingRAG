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

import xxhash
from charset_normalizer import from_bytes

from app.generation.base import VisionModel
from app.generation.vision import VisionProviderError
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.image_fetch import ImageFetcher, ImageFetchError
from app.ingestion.image_preprocess import ImageDecodeError, preprocess_image
from app.ingestion.image_worker import ImageWorker
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


#: 语义切块：仅对长度超过该值、且含多个段落的文本块做语义切分（短块/单段不动）。
_SEMANTIC_MIN_LEN = 800
#: 相邻段落 embedding 余弦相似度低于该阈值视为话题边界，在此切分。
#: 取较低值，只切「清晰话题边界」，避免把同节内略不同的段落切碎。
_SEMANTIC_SPLIT_THRESHOLD = 0.30
#: 语义切分出的子块若低于该长度，并入相邻子块，避免产生短碎片。
_SEMANTIC_MIN_SUB_LEN = 120


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
    )


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
    ) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._vision = vision
        self._fetcher = fetcher
        self._worker: ImageWorker | None = None
        #: 同图多处引用只识一次：content_hash -> 描述文本，跨文件复用（嵌入另按标题上下文）。
        self._image_cache: dict[str, str] = {}

    def set_worker(self, worker: ImageWorker) -> None:
        """挂接识图后台工作线程：挂接后图片走后台补全，文本仍同步先入库。"""
        self._worker = worker

    def semantic_split(self, chunks: list[Chunk]) -> list[Chunk]:
        """语义切块：对多段落长文本块，按相邻段落 embedding 相似度骤降处切分。

        只处理长度 ≥ ``_SEMANTIC_MIN_LEN`` 且含 ≥2 个非空段落的文本块；相邻段相似度
        低于 ``_SEMANTIC_SPLIT_THRESHOLD`` 视为话题边界，在边界处拆成更连贯的子块，
        子块继承原块标题身份（block_id 加 ``-s{idx}`` 后缀）。代码块/图片块原样保留。
        """
        result: list[Chunk] = []
        for chunk in chunks:
            if chunk.kind != "text" or len(chunk.text) < _SEMANTIC_MIN_LEN:
                result.append(chunk)
                continue
            paragraphs = [p for p in chunk.text.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                result.append(chunk)
                continue
            vectors = self._embedder.embed_texts(paragraphs)
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
        for file in discovered:
            try:
                chunks, upserted = self.ingest_file(file)
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

    def ingest_file(self, file: DiscoveredFile) -> tuple[list[Chunk], int]:
        """单文件：读 → 解析 → 切块 → 批次嵌入 → upsert，返回 (chunks, upserted)。

        公共入口（增量同步复用）：与全量 `ingest` 内单文件处理路径完全一致。
        """
        text = read_text(file.path)
        parsed = parse_markdown(text)
        chunks = chunk_document(parsed, source_file=str(file.path))
        if not chunks:
            return chunks, 0
        # 语义切块：多段落长块按话题边界细分，产出更连贯的块
        chunks = self.semantic_split(chunks)
        # 文本块先入库（快路径），图片描述块后补——图片失败不拖累文本（T5 §5.3 文本先行）。
        upserted = 0
        text_chunks = [chunk for chunk in chunks if chunk.kind != "image"]
        if text_chunks:
            vectors = self._embedder.embed_texts([chunk.text for chunk in text_chunks])
            text_blocks: list[tuple[str, str, list[float], BlockMetadata]] = []
            for chunk, vector in zip(text_chunks, vectors, strict=True):
                block_id = make_block_id(file.content_hash, chunk.block_id)
                text_blocks.append((block_id, chunk.text, vector, make_chunk_metadata(block_id, file, chunk)))
            self._vectorstore.upsert(text_blocks)
            upserted = len(text_blocks)
        image_chunks = [chunk for chunk in chunks if chunk.kind == "image"]
        if self._worker is not None:
            # 异步：图片任务投递后台线程，文本已先入库，导入不被识图拖垮。
            self._worker.enqueue(file, image_chunks)
        else:
            # 同步：图片描述直接入库（无 worker 时的兜底，行为与单测一致）。
            try:
                image_blocks = self.image_blocks(file, image_chunks)
            except Exception as exc:  # noqa: BLE001 -- 图片路径任何异常都不拖累文本（文本已入库）
                logger.debug("图片处理异常（脱敏）: %s", type(exc).__name__)
                image_blocks = []
            if image_blocks:
                self._vectorstore.upsert(image_blocks)
                upserted += len(image_blocks)
        return chunks, upserted

    def image_blocks(
        self,
        file: DiscoveredFile,
        image_chunks: list[Chunk],
    ) -> list[tuple[str, str, list[float], BlockMetadata]]:
        """把图片引用转为描述块：下载 → 预处理 → 识图 → 嵌入（同图去重只算一次）。

        未配置识图（vision/fetcher 任一缺）或单图失败时静默跳过，不阻塞文本入库；
        描述与文本共用同一嵌入模型（方案 B 核心），block_id 稳定可幂等 upsert。
        """
        if self._vision is None or self._fetcher is None or not image_chunks:
            return []
        refs: list[tuple[Chunk, str]] = []  # (chunk, content_hash)
        to_describe: dict[str, bytes] = {}  # content_hash -> 预处理后 JPEG
        for chunk in image_chunks:
            src = chunk.src
            if not src:
                continue
            try:
                raw = self._fetcher.fetch(src)
                jpeg = preprocess_image(raw)
            except (ImageFetchError, ImageDecodeError) as exc:  # 死链/非图/超限：跳过
                logger.debug("图片跳过（脱敏）: %s", type(exc).__name__)
                continue
            content_hash = xxhash.xxh64(jpeg).hexdigest()
            refs.append((chunk, content_hash))
            if content_hash not in self._image_cache and content_hash not in to_describe:
                to_describe[content_hash] = jpeg
        # 识图（同图只算一次，描述缓存到 self._image_cache）
        for content_hash, jpeg in to_describe.items():
            try:
                self._image_cache[content_hash] = self._vision.describe(jpeg)
            except VisionProviderError as exc:
                logger.debug("识图失败（脱敏）: %s", type(exc).__name__)
        # 嵌入：图片描述注入标题路径（主题上下文），按 (content_hash, heading_path) 去重。
        # 同图在不同标题下语境不同 -> 分别嵌入；同标题下同图 -> 只嵌一次。
        embed_keys: list[tuple[str, str]] = []
        embed_texts: list[str] = []
        for chunk, content_hash in refs:
            description = self._image_cache.get(content_hash)
            if description is None:
                continue
            key = (content_hash, chunk.heading_path)
            if key not in embed_keys:
                embed_keys.append(key)
                embed_texts.append(_image_embed_text(description, chunk.heading_path))
        embed_map: dict[tuple[str, str], list[float]] = {}
        if embed_texts:
            vectors = self._embedder.embed_texts(embed_texts)
            for key, vector in zip(embed_keys, vectors, strict=True):
                embed_map[key] = vector
        blocks: list[tuple[str, str, list[float], BlockMetadata]] = []
        for chunk, content_hash in refs:
            description = self._image_cache.get(content_hash)
            if description is None:
                continue  # 识图失败，该引用不产块
            embed_text = _image_embed_text(description, chunk.heading_path)
            vector = embed_map.get((content_hash, chunk.heading_path))
            if vector is None:
                continue  # 嵌入失败，该引用不产块
            block_id = make_block_id(file.content_hash, chunk.block_id)
            blocks.append(
                (
                    block_id,
                    embed_text,
                    vector,
                    make_image_metadata(block_id, file, chunk, content_hash),
                )
            )
        return blocks

    def process_image_chunks(self, file: DiscoveredFile, image_chunks: list[Chunk]) -> int:
        """图片描述块入库（后台 worker 调用）：image_blocks + upsert，返回 upserted 数。"""
        blocks = self.image_blocks(file, image_chunks)
        if blocks:
            self._vectorstore.upsert(blocks)
        return len(blocks)

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


def make_block_id(content_hash: str, chunk_block_id: str) -> str:
    """全局唯一 block_id：doc 内容指纹前缀 + 锚点路径链（对齐 PRD hash(doc)::锚点）。

    公共助手，全量导入与增量同步（sync.py）共用同一块身份规则。
    """
    return f"{content_hash}::{chunk_block_id}"


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
        anchor=chunk.anchor or None,
        chunk_type=_CHUNK_TYPE,
    )


def make_image_metadata(
    block_id: str,
    file: DiscoveredFile,
    chunk: Chunk,
    content_hash: str,
) -> BlockMetadata:
    """组装图片描述块元数据：source_type=image_description，原图 URL + 内容哈希入元数据。

    公共助手，全量导入与增量同步（sync.py）共用同一元数据规则。
    """
    return BlockMetadata(
        block_id=block_id,
        doc_id=file.content_hash,
        source_type=SourceType.IMAGE_DESCRIPTION,
        source_file=str(file.path),
        platform="local",
        heading_path=chunk.heading_path or None,
        anchor=chunk.anchor or None,
        chunk_type="image_description",
        image_path=chunk.src,
        image_content_hash=content_hash,
    )


def _image_embed_text(description: str, heading_path: str) -> str:
    """图片描述块的嵌入/存储文本：注入标题路径作为主题上下文。

    图片本身常不含主题词（如图在「AB 测试」标题下但图里没写「AB 测试」），
    把标题路径拼进文本使图片块能被主题词检索命中；标题路径很短，几乎无稀释、
    不重复（文本块与图片块职责分离）。无标题路径时原样返回纯描述。
    """
    if heading_path:
        return f"【所属主题】{heading_path}\n\n{description}"
    return description
