"""导入任务存储：进程内线程安全 dict（单用户本地场景够用；重启丢失）。

任务状态机：running → done / error。
- create(): 生成唯一 task_id，初始 running。
- update_progress(): 后台线程逐文件写运行快照。
- complete() / fail(): 终态；complete 同时记录 last_success_at（供 /api/status 展示）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from app.ingestion.pipeline import IngestReport, ProgressSnapshot

#: 任务状态常量
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


@dataclass
class ImportTask:
    """单个导入任务的可变状态（后台线程与轮询读端共享，经 store 锁保护）。"""

    task_id: str
    status: str = STATUS_RUNNING
    progress: ProgressSnapshot | None = None
    report: IngestReport | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ImportTaskStore:
    """线程安全的导入任务注册表（dict + Lock）。"""

    def __init__(self) -> None:
        self._tasks: dict[str, ImportTask] = {}
        self._lock = threading.Lock()
        self._last_success_at: datetime | None = None

    def create(self) -> ImportTask:
        """创建任务并登记，返回其引用（初始 status=running）。"""
        with self._lock:
            task = ImportTask(task_id=uuid4().hex)
            self._tasks[task.task_id] = task
            return task

    def get(self, task_id: str) -> ImportTask | None:
        """按 id 取任务；不存在返回 None。"""
        with self._lock:
            return self._tasks.get(task_id)

    def update_progress(self, task_id: str, snapshot: ProgressSnapshot) -> None:
        """更新运行进度快照（后台线程每处理一个文件调用一次）。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.progress = snapshot
            task.updated_at = datetime.now(UTC)

    def complete(self, task_id: str, report: IngestReport) -> None:
        """任务成功完成：status=done、report 落位、进度对齐报告、记录 last_success_at。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = STATUS_DONE
            task.report = report
            task.progress = ProgressSnapshot(
                files_scanned=report.files_scanned,
                files_parsed=report.files_parsed,
                files_skipped=report.files_skipped,
                chunks=report.chunks,
            )
            task.updated_at = datetime.now(UTC)
            self._last_success_at = task.updated_at

    def fail(self, task_id: str, error: str) -> None:
        """任务失败：status=error + 可读错误信息。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = STATUS_ERROR
            task.error = error
            task.updated_at = datetime.now(UTC)

    def clear(self) -> None:
        """清空所有任务与 last_success_at（测试用）。"""
        with self._lock:
            self._tasks.clear()
            self._last_success_at = None

    @property
    def last_success_at(self) -> datetime | None:
        """最近一次成功导入的时间（供 /api/status 展示；无成功记录为 None）。"""
        with self._lock:
            return self._last_success_at
