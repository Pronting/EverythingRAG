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
_HELD = threading.local()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def local_read_lock(path: Path, timeout_seconds: float = 60) -> Iterator[None]:
    lock = _thread_lock(path)
    if not lock.acquire(timeout=timeout_seconds):
        raise TimeoutError("知识读取正在等待")
    try:
        yield
    finally:
        lock.release()


@contextmanager
def exclusive_write_lock(path: Path, timeout_seconds: float = 60.0) -> Iterator[None]:
    """获取线程内 + 跨进程独占锁，超时明确失败而不是并发写坏索引。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _thread_lock(path)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    if not lock.acquire(timeout=max(0.0, timeout_seconds)):
        raise TimeoutError("知识索引正在维护")
    key = str(path.resolve())
    held = getattr(_HELD, "paths", None)
    if held is None:
        held = _HELD.paths = set()
    try:
        if key in held:
            yield
            return
        with path.open("a+b") as handle:
            _acquire_file(handle, deadline)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                _unlock(handle)
    finally:
        lock.release()


def _acquire_file(handle: object, deadline: float) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    while True:
        try:
            _try_lock(handle)
            break
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError("知识索引正在维护") from None
            time.sleep(0.05)


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
