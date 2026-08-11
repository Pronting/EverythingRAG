"""ingestion 数据接入包：统一入口为目录扫描 → 文档发现。

MVP 任务 1 提供 ``scan_directory`` / ``DiscoveredFile``；后续任务
（解析 / 切块 / 同步）在此包内扩展，保持薄接口 + 可插拔。
"""
from __future__ import annotations

from app.ingestion.scanner import DiscoveredFile, scan_directory

__all__ = ["DiscoveredFile", "scan_directory"]
