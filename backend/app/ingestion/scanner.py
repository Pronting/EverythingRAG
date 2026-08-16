"""ingestion：目录扫描 + Markdown 发现（MVP 任务 1）。

提供统一入口 ``scan_directory``（含内容指纹）与增量同步用的轻量
``scan_files``（只读 mtime/size，不读内容，快路径毫秒级）。
只做发现，不做内容解析（解析在任务 2）。纯本地文件系统操作，零出网。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)

#: 点前缀目录/文件视为隐藏，整棵跳过或忽略（如 .git / .obsidian / .hidden.md）
HIDDEN_PREFIX = "."
_MD_SUFFIX = ".md"


@dataclass(frozen=True)
class DiscoveredFile:
    """单个被发现的 Markdown 文档的结构化元数据。"""

    path: Path  # 绝对路径（resolve 后）
    filename: str
    mtime: float  # 修改时间戳（stat().st_mtime）
    size: int  # 字节数
    content_hash: str  # xxh64 十六进制内容指纹


@dataclass(frozen=True)
class FileStat:
    """轻量文件元数据（增量同步快路径用，不读内容）。

    指纹比对第一步先比 mtime + size：两者都未变即可判定未变，
    无需读取文件内容计算哈希（10 万文件毫秒级跳过）。
    """

    path: Path  # 绝对路径（resolve 后）
    mtime: float  # 修改时间戳（stat().st_mtime）
    size: int  # 字节数


def scan_directory(root: Path) -> list[DiscoveredFile]:
    """递归扫描 ``root`` 下所有 Markdown 文档，按 path 排序返回。

    隐藏目录（点前缀）整棵跳过；隐藏文件忽略；单个文件/目录读取失败时
    记录日志后跳过，不中断整体扫描。返回含内容指纹的完整元数据
    （全量导入用）。

    Raises:
        ValueError: ``root`` 不存在或不是目录。
    """
    root = _resolve_root(root)
    return [entry for entry in (_discover(path) for path in _walk_md(root)) if entry is not None]


def scan_files(root: Path) -> list[FileStat]:
    """轻量扫描：返回全部 Markdown 的 mtime/size，**不读取文件内容**。

    增量同步第一步用它做快路径：与指纹表比对 mtime+size，都未变即跳过；
    只有疑似变化的文件才调用 ``discover_file`` 读内容算哈希确认。

    Raises:
        ValueError: ``root`` 不存在或不是目录。
    """
    root = _resolve_root(root)
    discovered: list[FileStat] = []
    for path in _walk_md(root):
        try:
            stat = path.stat()
        except OSError as exc:
            logger.debug("跳过文件 %s: %s", path, exc)
            continue
        discovered.append(FileStat(path=path, mtime=stat.st_mtime, size=stat.st_size))
    return discovered


def discover_file(path: Path) -> DiscoveredFile | None:
    """对单个文件读取内容并返回完整元数据（含指纹）；失败返回 None。

    增量同步在 ``scan_files`` 快路径判出疑似变化后，调用本函数读内容
    算哈希做权威确认（内容未变仅 mtime 变 -> 视为未变）。
    """
    return _discover(path)


def _resolve_root(root: Path) -> Path:
    """校验并规范化扫描根目录；非法根快速失败。"""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"扫描根目录不存在或非目录: {root}")
    return root


def _walk_md(root: Path) -> list[Path]:
    """递归收集全部 Markdown 的绝对路径（隐藏目录/文件过滤，按 path 排序）。

    只做路径发现，不读取内容——``scan_directory`` 与 ``scan_files`` 共用。
    """
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_on_walk_error):
        dirnames[:] = [d for d in dirnames if not d.startswith(HIDDEN_PREFIX)]
        parent = Path(dirpath).resolve()
        for name in filenames:
            if name.startswith(HIDDEN_PREFIX) or not name.endswith(_MD_SUFFIX):
                continue
            paths.append(parent / name)
    paths.sort(key=str)
    return paths


def _discover(path: Path) -> DiscoveredFile | None:
    """对单个文件构造元数据；读取失败返回 None（由调用方跳过）。"""
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        content = resolved.read_bytes()
    except OSError as exc:
        logger.debug("跳过文件 %s: %s", path, exc)
        return None
    return DiscoveredFile(
        path=resolved,
        filename=resolved.name,
        mtime=stat.st_mtime,
        size=stat.st_size,
        content_hash=xxhash.xxh64(content).hexdigest(),
    )


def _on_walk_error(exc: OSError) -> None:
    """os.walk 目录遍历失败（如权限不可读）时记录并继续。"""
    logger.debug("跳过不可读目录: %s", exc)
