"""识图后台工作线程单测：队列消费 / 进度快照 / 失败计数 + 异步管线集成。

验证 T5 §5.3 异步化：文本先入库、图片后台补全、导入不被识图拖垮。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

from app.ingestion.chunker import Chunk
from app.ingestion.image_worker import ImageWorker
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import DiscoveredFile
from app.models.schemas import SourceType


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _file() -> DiscoveredFile:
    return DiscoveredFile(path=Path("/tmp/x.md"), filename="x.md", mtime=0.0, size=0, content_hash="abc")


def _chunk(i: int) -> Chunk:
    return Chunk(
        block_id=f"#{i}",
        text="",
        source_file="/tmp/x.md",
        heading_path="",
        anchor="",
        seq=i,
        kind="image",
        src=f"https://e/{i}.png",
    )


# ---------------------------------------------------------------- 1. 工作线程本身


def test_worker_processes_and_reports_snapshot() -> None:
    """入队 -> 后台消费 -> snapshot 反映 done；stop 后队列耗尽。"""
    processed: list[str] = []

    def process(file: DiscoveredFile, chunks: list[Chunk]) -> int:
        processed.extend(c.src or "" for c in chunks)
        return len(chunks)

    worker = ImageWorker(process)
    worker.enqueue(_file(), [_chunk(1), _chunk(2)])
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 2, "failed": 0}
    assert sorted(processed) == ["https://e/1.png", "https://e/2.png"]


def test_worker_records_failure_not_crash() -> None:
    """process 抛异常 -> 计入 failed，不中断后台线程。"""

    def process(file: DiscoveredFile, chunks: list[Chunk]) -> int:
        raise RuntimeError("boom")

    worker = ImageWorker(process)
    worker.enqueue(_file(), [_chunk(1)])
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 0, "failed": 1}


def test_enqueue_empty_chunks_noop() -> None:
    """空图片块入队 -> 不启动线程、pending 不变。"""
    worker = ImageWorker(lambda f, c: len(c))
    worker.enqueue(_file(), [])
    assert worker.snapshot()["pending"] == 0


# ---------------------------------------------------------------- 2. 异步管线集成


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 1.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, list[float], object]] = {}

    def upsert(self, blocks: list) -> None:
        for block_id, text, vector, meta in blocks:
            self.blocks[block_id] = (text, vector, meta)

    def count(self) -> int:
        return len(self.blocks)

    def count_images(self) -> int:
        return sum(1 for _, _, m in self.blocks.values() if m.source_type == SourceType.IMAGE_DESCRIPTION)


class FakeVision:
    def describe(self, image_bytes: bytes, prompt: str = "", mime: str = "image/jpeg") -> str:
        return "【画面主体】架构图：系统分三层。"


class FakeFetcher:
    def fetch(self, url: str) -> bytes:
        return _png_bytes()


def test_pipeline_worker_backfills_images_after_text(tmp_path: Path) -> None:
    """文本立即入库、图片后台补全：导入返回时文本已就绪，图片随后补上。"""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text(
        "# 正文\n\n这是正文说明内容段落。\n\n![](https://cdn.example.com/a.png)",
        encoding="utf-8",
    )

    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        embedder=FakeEmbedder(),
        vectorstore=store,
        vision=FakeVision(),
        fetcher=FakeFetcher(),
    )
    worker = ImageWorker(pipeline.process_image_chunks)
    worker.start()
    pipeline.set_worker(worker)

    report = pipeline.ingest(root)
    # 导入返回时：文本已入库（报告含文本块），图片还在后台
    assert report.files_parsed == 1
    assert store.count() >= 1  # 文本块已就绪

    # 等待后台补全（worker 单线程顺序消费）
    deadline = time.monotonic() + 5
    while worker.snapshot()["pending"] > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 1, "failed": 0}
    assert store.count_images() == 1
