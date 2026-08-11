"""ingestion：目录扫描 + Markdown 发现（MVP 任务 1）。

提供统一入口 ``scan_directory``，递归扫描本地目录、发现全部 Markdown
文档并返回结构化元数据（绝对路径 / 指纹 / 时间戳等）。只做发现，不做
内容解析（解析在任务 2）。纯本地文件系统操作，零出网。
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


def scan_directory(root: Path) -> list[DiscoveredFile]:
    """递归扫描 ``root`` 下所有 Markdown 文档，按 path 排序返回。

    隐藏目录（点前缀）整棵跳过；隐藏文件忽略；单个文件/目录读取失败时
    记录日志后跳过，不中断整体扫描。

    Raises:
        ValueError: ``root`` 不存在或不是目录。
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"扫描根目录不存在或非目录: {root}")

    discovered: list[DiscoveredFile] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_on_walk_error):
        dirnames[:] = [d for d in dirnames if not d.startswith(HIDDEN_PREFIX)]
        parent = Path(dirpath).resolve()
        for name in filenames:
            if name.startswith(HIDDEN_PREFIX) or not name.endswith(_MD_SUFFIX):
                continue
            entry = _discover(parent / name)
            if entry is not None:
                discovered.append(entry)

    discovered.sort(key=lambda f: str(f.path))
    return discovered


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
