"""ingestion：识图后台工作线程池 —— 文本先入库，图片描述后台并发补全（T5 §5.3 异步化）。

图片任务先持久化到 image_state_store.image_tasks（pending），再由 N 个工作线程并发
逐张 fetch → 预处理 → 识图 → 嵌入 → upsert，完成后标记 done / 失败标记 failed。
应用重启后调用 ``resume_pending()`` 加载未完成任务续跑（block_id 幂等，重跑不重复）。

云端 VLM 是网络 I/O 密集，多线程并发请求 ≈ 线性提速（瓶颈在服务商并发/限流）；
describe 自带 429/5xx 退避重试。Chroma 并发 upsert 实测安全，ImageStateStore 全方法
加锁，故线程池并发安全。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from app.ingestion.image_state_store import ImageStateStore, ImageTask

logger = logging.getLogger(__name__)

#: 识图后台线程名前缀（便于排查）。
_WORKER_THREAD_NAME = "image-worker"

#: 默认并发度（云端 VLM 2-4 合理，见 T5 §5.3）。
DEFAULT_CONCURRENCY = 3


class ImageWorker:
    """识图后台工作线程池：持久化任务 + 多线程并发消费 + 进度快照 + 启动续跑。"""

    def __init__(
        self,
        process: Callable[[ImageTask], bool],
        state_store: ImageStateStore,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._process = process
        self._state_store = state_store
        self._concurrency = max(1, int(concurrency))
        self._queue: queue.Queue[ImageTask | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._pending = 0
        self._done = 0
        self._failed = 0
        self._lock = threading.Lock()

    def enqueue(self, task: ImageTask) -> None:
        """登记并投递一个图片任务（已完成则跳过，幂等）。"""
        if not self._state_store.add_task(task):
            return
        with self._lock:
            self._pending += 1
        self.start()
        self._queue.put(task)

    def resume_pending(self) -> int:
        """加载持久化待处理任务并投递（应用启动续跑）；返回投递数。"""
        tasks = self._state_store.load_pending()
        for task in tasks:
            with self._lock:
                self._pending += 1
            self._queue.put(task)
        if tasks:
            self.start()
        return len(tasks)

    def start(self) -> None:
        """启动工作线程池（幂等：补齐到 concurrency 个存活线程）。"""
        alive = [t for t in self._threads if t.is_alive()]
        self._threads = alive
        for _ in range(self._concurrency - len(alive)):
            thread = threading.Thread(
                target=self._run, name=f"{_WORKER_THREAD_NAME}-{len(self._threads)}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float | None = None) -> None:
        """优雅停止：投递 N 个哨兵并等待全部线程退出（不丢在途任务）。"""
        for _ in range(self._concurrency):
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    def snapshot(self) -> dict[str, int]:
        """当前工作线程进度快照：{pending, done, failed}（持久化总数见 state_store）。"""
        with self._lock:
            return {"pending": self._pending, "done": self._done, "failed": self._failed}

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:  # 停止哨兵
                break
            ok = False
            try:
                ok = self._process(task)
            except Exception as exc:  # noqa: BLE001 -- 单任务失败不中断后台线程
                logger.debug("识图后台任务失败（脱敏）: %s", type(exc).__name__)
            if ok:
                self._state_store.mark_done(task.block_id)
                with self._lock:
                    self._done += 1
            else:
                self._state_store.mark_failed(task.block_id, "处理失败")
                with self._lock:
                    self._failed += 1
            with self._lock:
                self._pending -= 1
            self._queue.task_done()
