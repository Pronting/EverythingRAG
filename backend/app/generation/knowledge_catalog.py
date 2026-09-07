"""从向量库元数据生成不含绝对路径的确定性知识概览。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from app.vectorstore.base import VectorStore


class KnowledgeTopic(BaseModel):
    name: str
    file_count: int
    chunk_count: int


class RepresentativeDocument(BaseModel):
    name: str
    topic: str
    chunk_count: int


class KnowledgeOverview(BaseModel):
    file_count: int = 0
    chunk_count: int = 0
    text_chunk_count: int = 0
    image_chunk_count: int = 0
    chunk_types: dict[str, int] = Field(default_factory=dict)
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    representative_documents: list[RepresentativeDocument] = Field(default_factory=list)

    def to_prompt_context(self) -> str:
        """模型只接收公开投影；原始 source_file 永远不会进入概览上下文。"""

        return (
            json.dumps(self.model_dump(), ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )


def build_knowledge_overview(vectorstore: VectorStore) -> KnowledgeOverview:
    """单次枚举块并聚合文件、块类型、主题和代表文档，排序结果完全确定。"""

    blocks = vectorstore.list_blocks()
    type_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    file_names: dict[str, str] = {}
    file_topics: dict[str, str] = {}
    topic_chunks: Counter[str] = Counter()
    topic_files: dict[str, set[str]] = defaultdict(set)
    image_chunks = 0

    for _block_id, _text, metadata in blocks:
        meta = metadata if isinstance(metadata, dict) else {}
        chunk_type = _clean_label(meta.get("chunk_type"), fallback="text")
        source_type = str(meta.get("source_type") or "")
        type_counts[chunk_type] += 1
        if source_type == "image_description" or chunk_type == "image_description":
            image_chunks += 1

        raw_source = str(meta.get("source_file") or "")
        if not raw_source:
            continue
        document_name, topic = _public_source_parts(raw_source, meta)
        file_counts[raw_source] += 1
        file_names[raw_source] = document_name
        file_topics[raw_source] = topic
        topic_chunks[topic] += 1
        topic_files[topic].add(raw_source)

    topics = [
        KnowledgeTopic(
            name=name,
            file_count=len(topic_files[name]),
            chunk_count=topic_chunks[name],
        )
        for name in sorted(topic_chunks, key=lambda item: (-topic_chunks[item], item.casefold()))[:12]
    ]
    representatives = [
        RepresentativeDocument(
            name=file_names[source],
            topic=file_topics[source],
            chunk_count=count,
        )
        for source, count in sorted(
            file_counts.items(),
            key=lambda item: (-item[1], file_names[item[0]].casefold(), file_topics[item[0]].casefold()),
        )[:12]
    ]

    return KnowledgeOverview(
        file_count=len(file_counts),
        chunk_count=len(blocks),
        text_chunk_count=max(0, len(blocks) - image_chunks),
        image_chunk_count=image_chunks,
        chunk_types=dict(sorted(type_counts.items(), key=lambda item: item[0].casefold())),
        topics=topics,
        representative_documents=representatives,
    )


def _public_source_parts(raw_source: str, metadata: dict[str, Any]) -> tuple[str, str]:
    """只返回 basename 与直接父主题；盘符、用户目录和完整路径不会向外暴露。"""

    normalized = raw_source.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("/", ".", "..")]
    name = _clean_label(parts[-1] if parts else raw_source, fallback="未命名文档")
    if len(parts) >= 2:
        topic = _clean_label(parts[-2], fallback="未分类")
    else:
        platform = str(metadata.get("platform") or "")
        topic = _clean_label(platform, fallback="未分类") if platform != "local" else "未分类"
    return name, topic


def _clean_label(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", "", str(value or "")).strip()
    return text[:120] or fallback
