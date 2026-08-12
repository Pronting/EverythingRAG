"""ImportTaskStore 单测：状态机 running→done/error、进度更新、线程安全、未知 id。

不依赖任何外部服务；纯内存 dict + lock 的并发语义验证。
"""

from __future__ import annotations

import threading

from app.ingestion.import_task import ImportTaskStore
from app.ingestion.pipeline import IngestReport, ProgressSnapshot


def test_create_returns_running_task() -> None:
    """create() 生成唯一 task_id，初始 status=running。"""
    store = ImportTaskStore()
    task = store.create()
    assert task.status == "running"
    assert store.get(task.task_id) is task
    assert store.get(task.task_id).progress is None


def test_update_progress_and_complete() -> None:
    """进度更新可查；complete 后 status=done、report 落位、进度对齐报告、记录 last_success_at。"""
    store = ImportTaskStore()
    task = store.create()

    snap = ProgressSnapshot(files_scanned=3, files_parsed=1, files_skipped=0, chunks=2)
    store.update_progress(task.task_id, snap)
    assert store.get(task.task_id).progress == snap

    report = IngestReport(files_scanned=3, files_parsed=3, files_skipped=0, chunks=5, blocks_upserted=5)
    store.complete(task.task_id, report)
    done = store.get(task.task_id)
    assert done.status == "done"
    assert done.report == report
    assert done.progress is not None
    assert done.progress.chunks == 5  # 终态进度 = 报告
    assert store.last_success_at is not None


def test_fail_sets_error_and_keeps_last_success() -> None:
    """fail 后 status=error + error 信息；不影响既有 last_success_at。"""
    store = ImportTaskStore()
    ok = store.create()
    store.complete(ok.task_id, IngestReport(1, 1, 0, 1, 1))
    before = store.last_success_at

    bad = store.create()
    store.fail(bad.task_id, "目录不存在")
    failed = store.get(bad.task_id)
    assert failed.status == "error"
    assert failed.error == "目录不存在"
    assert store.last_success_at == before  # 失败不更新 last_success


def test_unknown_task_returns_none() -> None:
    """未知 task_id -> None。"""
    store = ImportTaskStore()
    assert store.get("no-such-id") is None


def test_concurrent_progress_updates_no_crash() -> None:
    """并发写进度：不崩溃、最终状态可达且为合法终态。"""
    store = ImportTaskStore()
    task = store.create()

    def worker(index: int) -> None:
        for _round in range(50):
            store.update_progress(
                task.task_id,
                ProgressSnapshot(files_scanned=10, files_parsed=index, files_skipped=0, chunks=index),
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final = store.get(task.task_id)
    assert final is not None
    assert final.status == "running"
    assert final.progress is not None
    assert final.progress.files_scanned == 10
