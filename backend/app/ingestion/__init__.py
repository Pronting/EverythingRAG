"""ingestion 数据接入包：统一入口为目录扫描 → 文档发现 → Markdown 解析 → 语义切块。

MVP 任务 1 提供 ``scan_directory`` / ``DiscoveredFile``；任务 2 提供
``parse_markdown`` / ``ParsedMarkdown`` / ``HeadingNode``；任务 3 提供
``MDBlock``（解析块级结构）与 ``chunk_document`` / ``Chunk``（语义切块）。
保持薄接口 + 可插拔。
"""
from __future__ import annotations

from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.markdown_parser import HeadingNode, MDBlock, ParsedMarkdown, parse_markdown
from app.ingestion.scanner import DiscoveredFile, scan_directory

__all__ = [
    "Chunk",
    "DiscoveredFile",
    "HeadingNode",
    "MDBlock",
    "ParsedMarkdown",
    "chunk_document",
    "parse_markdown",
    "scan_directory",
]
