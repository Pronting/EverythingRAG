"""ingestion 数据接入包：统一入口为目录扫描 → 文档发现 → Markdown 解析 → 语义切块 → 编排导入。

MVP 任务 1 提供 ``scan_directory`` / ``DiscoveredFile``；任务 2 提供
``parse_markdown`` / ``ParsedMarkdown`` / ``HeadingNode``；任务 3 提供
``MDBlock``（解析块级结构）与 ``chunk_document`` / ``Chunk``（语义切块）；
任务 9 提供 ``IngestionPipeline`` / ``IngestReport``（端到端编排）。
保持薄接口 + 可插拔。
"""
from __future__ import annotations

from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.markdown_parser import HeadingNode, MDBlock, ParsedMarkdown, parse_markdown
from app.ingestion.pipeline import IngestionPipeline, IngestReport
from app.ingestion.scanner import (
    DiscoveredFile,
    FileStat,
    discover_file,
    scan_directory,
    scan_files,
)

__all__ = [
    "Chunk",
    "DiscoveredFile",
    "FileStat",
    "HeadingNode",
    "IngestReport",
    "IngestionPipeline",
    "MDBlock",
    "ParsedMarkdown",
    "chunk_document",
    "discover_file",
    "parse_markdown",
    "scan_directory",
    "scan_files",
]
