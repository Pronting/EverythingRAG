"""工作区记忆的并发原语（PRD 11.2 末段、11.4、11.5）。

PRD 对锁的规定不是"建议"，而是三条必须同时成立的不变量：

1. **先会话锁，再工作区写入锁**。顺序写错就会有死锁：会话 A 在区 1、会话 B 在区 2，
   两个线程各自"先拿区锁再拿会话锁"时会互相等待；固定「会话 → 工作区」这一顺序后，
   任何线程获取锁的偏序都一致。
2. **跨两个工作区时按 ``storage_key`` 顺序取区锁**（会话移动）。同一偏序规则，
   把"两个并发的双向移动"变成不可能死锁。
3. **应用级排他文件锁**：同一 ``data_dir`` 只允许一个写入进程，重复启动的写入进程
   拿到**明确错误**而不是静默地在两份内存状态上各自写 ``memory.json``（后写覆盖先写，
   用户记忆直接丢失）。用操作系统级建议锁而非"锁文件存在与否"，因为前者在进程崩溃时
   由内核自动释放，不会留下让用户永远启动不了的陈旧锁。

分层职责：本模块只提供锁原语与错误类型，不认识记忆内容；``workspace_memory`` 负责
在锁内做「读 → 改 → 原子替换」。
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Self

__all__ = [
    "APP_WRITE_LOCK_FILENAME",
    "AppWriteLock",
    "MemoryLockError",
    "MemoryWriteLockError",
    "conversation_async_lock",
    "memory_maintenance_locks",
    "ordered_storage_keys",
    "session_lock",
    "workspace_lock",
    "workspace_write_locks",
]

#: 应用级排他锁文件名（PRD 11.2「应用排他文件锁」）。
APP_WRITE_LOCK_FILENAME = ".memory-write.lock"


class MemoryLockError(RuntimeError):
    """记忆锁相关的基类错误。"""


class MemoryWriteLockError(MemoryLockError):
    """本 ``data_dir`` 已有写入进程持锁（PRD 11.2：返回明确错误）。"""


# ---------------------------------------------------------------------------
# 进程内锁注册表
# ---------------------------------------------------------------------------

_session_locks: dict[str, threading.RLock] = {}
_workspace_locks: dict[str, threading.RLock] = {}
_registry_guard = threading.Lock()


def _lock_for(registry: dict[str, threading.RLock], key: str) -> threading.RLock:
    """取出（或创建）``key`` 对应的可重入锁。

    ``RLock`` 而非 ``Lock``：同一线程在「会话锁内再调用一次记忆维护」是合法嵌套
    （例如提炼节点里读快照又写游标），用普通锁会自锁。
    """

    with _registry_guard:
        lock = registry.get(key)
        if lock is None:
            lock = threading.RLock()
            registry[key] = lock
        return lock


def session_lock(conversation_id: str) -> threading.RLock:
    """会话锁：同一会话的记忆维护串行（PRD 11.2）。空 id 退化为进程级共享锁。"""

    return _lock_for(_session_locks, str(conversation_id or "__global__"))


def workspace_lock(storage_key: str) -> threading.RLock:
    """工作区写锁（按 ``storage_key``）。"""

    return _lock_for(_workspace_locks, str(storage_key or "__global__"))


def ordered_storage_keys(storage_keys: Iterable[str]) -> tuple[str, ...]:
    """去重 + 按 ``storage_key`` 字典序排序（PRD 11.2 的取锁顺序）。"""

    return tuple(sorted({str(key) for key in storage_keys if key}))


class workspace_write_locks:
    """按 ``storage_key`` 字典序获取若干个工作区写锁（PRD 11.2 跨区场景）。

    用法::

        with workspace_write_locks("aaa", "bbb"):
            ...   # 两个区同时被锁住

    排序在**获取之前**完成，因此无论调用方以什么顺序传参，实际的获取顺序都一样。
    """

    def __init__(self, *storage_keys: str) -> None:
        self._keys = ordered_storage_keys(storage_keys)
        self._stack: ExitStack | None = None

    @property
    def storage_keys(self) -> tuple[str, ...]:
        return self._keys

    def __enter__(self) -> Self:
        stack = ExitStack()
        for key in self._keys:
            stack.enter_context(workspace_lock(key))
        self._stack = stack
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None


@contextmanager
def memory_maintenance_locks(
    conversation_id: str | None = None,
    *,
    storage_keys: Iterable[str] = (),
) -> Iterator[None]:
    """记忆维护的标准取锁顺序：**先会话锁，再工作区写入锁**（PRD 11.2）。"""

    with ExitStack() as stack:
        if conversation_id:
            stack.enter_context(session_lock(conversation_id))
        if tuple(storage_keys):
            stack.enter_context(workspace_write_locks(*tuple(storage_keys)))
        yield


# ---------------------------------------------------------------------------
# 会话级 asyncio 锁（跨 await 的临界区，PRD 11.2 / 5.4 / A28）
# ---------------------------------------------------------------------------

_conversation_async_locks: dict[str, asyncio.Lock] = {}


def conversation_async_lock(conversation_id: str) -> asyncio.Lock:
    """会话级 ``asyncio.Lock``：把**含 await 的临界区**也串行化（A28）。

    为什么另有一把：``memory_maintenance_locks`` 用的是 ``threading.RLock``，而 PRD 5.4 的
    「提交前再次核对会话归属与 ``context_epoch``」本身要 ``await`` 一次 app.db 读取；
    在 threading 锁里 await 会让同一事件循环上的其它任务重入临界区，互斥就失效了
    （见 ``ensure_conversation_snapshot`` 的说明）。因此：

    - **文件级**互斥仍用 ``memory_maintenance_locks``（线程/跨进程安全，临界区内不 await）；
    - **归属与版本核对 + 提交** 这一段用本锁，会话移动与记忆提炼**共用同一把**，
      于是「移动」不可能插在「核对通过」与「写文件」之间 —— A28 的确定性来源。

    空 ``conversation_id`` 退化为进程级共享锁（与 ``session_lock`` 同惯例）。
    """

    key = str(conversation_id or "__global__")
    with _registry_guard:
        lock = _conversation_async_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _conversation_async_locks[key] = lock
        return lock


# ---------------------------------------------------------------------------
# 应用级排他文件锁
# ---------------------------------------------------------------------------


def _try_lock_exclusive(fileobj: object, fd: int) -> bool:
    """尝试对 ``fd`` 取非阻塞排他锁；已被占用返回 False。

    平台差异封装在这里：Windows 用 ``msvcrt.locking``，POSIX 用 ``fcntl.flock``。
    两者都是**随进程退出自动释放**的建议锁，因此不会留下陈旧锁文件让应用再也起不来。
    """

    if os.name == "nt":  # pragma: no cover - 平台分支，CI 只跑其一
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_exclusive(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - 平台分支
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


class AppWriteLock:
    """应用级排他文件锁：同一 ``data_dir`` 只允许一个写入进程（PRD 11.2 末段）。

    实例内可重入（同一实例重复 ``acquire`` 不报错），不同实例争抢时后者拿到
    :class:`MemoryWriteLockError` —— 这正是"重复启动写入进程返回明确错误"。
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        filename: str = APP_WRITE_LOCK_FILENAME,
    ) -> None:
        self._path = Path(data_dir) / filename
        self._fd: int | None = None
        self._depth = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            self._depth += 1
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Windows 的字节区间锁要求目标区间真实存在，先把文件撑到 1 字节。
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"1")
            os.lseek(fd, 0, os.SEEK_SET)
            if not _try_lock_exclusive(None, fd):
                raise MemoryWriteLockError(
                    f"已有进程持有记忆写入锁：{self._path}（同一数据目录只允许一个写入进程）"
                )
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._fd is None:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        fd = self._fd
        self._fd = None
        _unlock_exclusive(fd)
        os.close(fd)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
