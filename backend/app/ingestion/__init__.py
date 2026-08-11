"""ingestion 数据接入包：统一入口为目录扫描 → 文档发现 → Markdown 解析。

MVP 任务 1 提供 ``scan_directory`` / ``DiscoveredFile``；任务 2 提供
``parse_markdown`` / ``ParsedMarkdown`` / ``HeadingNode``。后续任务
（切块 / 同步）在此包内扩展，保持薄接口 + 可插拔。
"""
from __future__ import annotations

from app.ingestion.markdown_parser import HeadingNode, ParsedMarkdown, parse_markdown
from app.ingestion.scanner import DiscoveredFile, scan_directory

__all__ = [
    "DiscoveredFile",
    "HeadingNode",
    "ParsedMarkdown",
    "parse_markdown",
    "scan_directory",
]
