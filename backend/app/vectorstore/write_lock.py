"""跨进程 Chroma 写锁：同一数据目录任一时刻只允许一个写操作。"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def exclusive_write_lock(path: Path, timeout_seconds: float = 60.0) -> Iterator[None]:
    """获取线程内 + 跨进程独占锁，超时明确失败而不是并发写坏索引。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _thread_lock(path)
    with lock, path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            try:
                _try_lock(handle)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("向量索引正被另一个导入/同步进程写入，请稍后重试") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(handle)


def _try_lock(handle: object) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: object) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
