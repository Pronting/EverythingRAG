"""识图后台工作线程单测：持久化任务 + 队列消费 + 快照 + 失败计数 + 启动续跑 + 异步管线。

验证 T5 §5.3/5.4：任务落库、应用重启后 resume_pending 续跑、内容哈希去重。
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from PIL import Image

from app.ingestion.image_state_store import ImageStateStore, ImageTask
from app.ingestion.image_worker import ImageWorker
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import SourceType


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _task(i: int = 1) -> ImageTask:
    return ImageTask(
        block_id=f"filehash::#{i}",
        src=f"https://e/{i}.png",
        doc_id="filehash",
        source_file="/tmp/x.md",
        heading_path="",
        anchor="",
    )


def _store(tmp_path: Path) -> ImageStateStore:
    return ImageStateStore(tmp_path / "image_state.db")


# ---------------------------------------------------------------- 1. 队列消费 + 快照


def test_worker_processes_and_reports_snapshot(tmp_path: Path) -> None:
    """入队 -> 后台消费 -> 快照反映 done；任务落库标记 done。"""
    processed: list[str] = []

    def process(task: ImageTask) -> bool:
        processed.append(task.src)
        return True

    store = _store(tmp_path)
    worker = ImageWorker(process, store)
    worker.enqueue(_task(1))
    worker.enqueue(_task(2))
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 2, "failed": 0}
    assert sorted(processed) == ["https://e/1.png", "https://e/2.png"]
    assert store.count_status() == {"pending": 0, "done": 2, "failed": 0, "total": 2}


def test_worker_records_failure_not_crash(tmp_path: Path) -> None:
    """process 抛异常 -> 计入 failed + 落库 failed，不中断后台线程。"""

    def process(task: ImageTask) -> bool:
        raise RuntimeError("boom")

    store = _store(tmp_path)
    worker = ImageWorker(process, store)
    worker.enqueue(_task(1))
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 0, "failed": 1}
    assert store.count_status()["failed"] == 1


def test_enqueue_done_task_skipped(tmp_path: Path) -> None:
    """已完成任务（store 已 done）再入队 -> 跳过，不重复处理。"""
    processed: list[str] = []

    def process(task: ImageTask) -> bool:
        processed.append(task.src)
        return True

    store = _store(tmp_path)
    first = ImageWorker(process, store)
    first.enqueue(_task(1))
    first.stop(timeout=5)
    assert store.count_status()["done"] == 1

    # 再次入队同 block_id（已完成）-> 跳过
    second = ImageWorker(process, store)
    second.enqueue(_task(1))
    second.stop(timeout=5)
    assert len(processed) == 1  # 只处理了一次


# ---------------------------------------------------------------- 2. 启动续跑


def test_resume_pending_after_restart(tmp_path: Path) -> None:
    """持久化 pending 任务在「重启」后 resume_pending 自动续跑。"""
    store = _store(tmp_path)
    # 模拟上次运行：登记 2 个 pending 任务（尚未处理）
    store.add_task(_task(1))
    store.add_task(_task(2))
    assert store.count_status()["pending"] == 2

    # 模拟重启：新 worker resume_pending
    processed: list[str] = []

    def process(task: ImageTask) -> bool:
        processed.append(task.src)
        return True

    worker = ImageWorker(process, store)
    n = worker.resume_pending()
    worker.stop(timeout=5)

    assert n == 2
    assert sorted(processed) == ["https://e/1.png", "https://e/2.png"]
    assert store.count_status()["pending"] == 0
    assert store.count_status()["done"] == 2


# ---------------------------------------------------------------- 3. 异步管线集成


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


def test_worker_processes_concurrently(tmp_path: Path) -> None:
    """多线程并发消费：峰值并发 > 1（并发确实生效）。"""
    active = 0
    peak = 0
    lock = threading.Lock()

    def process(task: ImageTask) -> bool:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return True

    store = _store(tmp_path)
    worker = ImageWorker(process, store, concurrency=3)
    for i in range(6):
        worker.enqueue(_task(i))
    worker.stop(timeout=10)

    assert peak >= 2  # 至少 2 个任务同时处理，证明并发生效
    assert worker.snapshot() == {"pending": 0, "done": 6, "failed": 0}


def test_pipeline_worker_backfills_images_after_text(tmp_path: Path) -> None:
    """文本立即入库、图片后台补全：导入返回时文本已就绪，图片随后补上。"""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text(
        "# 正文\n\n这是正文说明内容段落。\n\n![](https://cdn.example.com/a.png)",
        encoding="utf-8",
    )

    store = FakeVectorStore()
    state_store = _store(tmp_path)
    pipeline = IngestionPipeline(
        embedder=FakeEmbedder(),
        vectorstore=store,
        vision=FakeVision(),
        fetcher=FakeFetcher(),
        state_store=state_store,
    )
    worker = ImageWorker(pipeline.process_image_task_and_upsert, state_store)
    worker.start()
    pipeline.set_worker(worker)

    report = pipeline.ingest(root)
    assert report.files_parsed == 1
    assert store.count() >= 1  # 文本块已就绪

    deadline = time.monotonic() + 5
    while worker.snapshot()["pending"] > 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop(timeout=5)

    assert worker.snapshot() == {"pending": 0, "done": 1, "failed": 0}
    assert store.count_images() == 1
    assert state_store.count_status()["done"] == 1  # 任务持久化标记完成
