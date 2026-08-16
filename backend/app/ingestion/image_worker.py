"""ingestion：识图后台工作线程 —— 文本先入库，图片描述后台补全（T5 §5.3 异步化）。

导入主线程只做文本（快路径，立即可检索），图片任务投递到本工作线程队列，
由它逐张 fetch → 预处理 → 识图 → 嵌入 → upsert。单线程顺序消费，与导入线程
并发写 Chroma 由 Chroma 内部 SQLite 锁序列化（已实测并发 upsert 安全）。

内存队列 + 进程内去重缓存：同一进程内所有导入共享去重（同图只识一次）；
跨进程重启续跑（image_tasks 持久化）留作后续增强。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from app.ingestion.chunker import Chunk
from app.ingestion.scanner import DiscoveredFile

logger = logging.getLogger(__name__)

#: 识图后台线程名（便于排查）。
_WORKER_THREAD_NAME = "image-worker"


class ImageWorker:
    """识图后台工作线程：单线程顺序消费图片任务，提供进度快照。"""

    def __init__(self, process: Callable[[DiscoveredFile, list[Chunk]], int]) -> None:
        self._process = process
        self._queue: queue.Queue[tuple[DiscoveredFile, tuple[Chunk, ...]] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending = 0
        self._done = 0
        self._failed = 0
        self._lock = threading.Lock()

    def enqueue(self, file: DiscoveredFile, chunks: list[Chunk]) -> None:
        """投递一个文件的图片任务（幂等启动线程）。"""
        if not chunks:
            return
        with self._lock:
            self._pending += len(chunks)
        self.start()
        self._queue.put((file, tuple(chunks)))

    def start(self) -> None:
        """启动后台线程（幂等，已存活则不重复启动）。"""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=_WORKER_THREAD_NAME, daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """优雅停止：投递哨兵并等待队列耗尽（不丢在途任务）。"""
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict[str, int]:
        """当前进度快照：{pending, done, failed}。"""
        with self._lock:
            return {"pending": self._pending, "done": self._done, "failed": self._failed}

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # 停止哨兵
                break
            file, chunks = item
            done = 0
            try:
                done = self._process(file, list(chunks))
            except Exception as exc:  # noqa: BLE001 -- 单批失败不中断后台线程
                logger.debug("识图后台任务失败（脱敏）: %s", type(exc).__name__)
            with self._lock:
                self._pending -= len(chunks)
                self._done += done
                self._failed += len(chunks) - done
            self._queue.task_done()
