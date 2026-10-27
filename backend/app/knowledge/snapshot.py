import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION
from app.knowledge.models import Document, KnowledgeError

Block = tuple[str, str, dict[str, Any]]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def document_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def source_labels(source: str) -> tuple[str, str]:
    path = PurePosixPath(source.replace("\\", "/"))
    return path.name[:120] or "未命名文档", path.parent.name[:120] or "未分类"


def ordered_blocks(blocks: list[Block]) -> list[Block]:
    positions = [metadata.get("document_order") for _, _, metadata in blocks]
    if all(type(position) is int and position >= 0 for position in positions):
        if len(set(positions)) != len(positions):
            raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
        return sorted(blocks, key=lambda item: item[2]["document_order"])

    from app.ingestion.chunker import chunk_document
    from app.ingestion.markdown_parser import parse_markdown
    from app.ingestion.pipeline import read_text
    from app.ingestion.scanner import discover_file

    sources = {str(meta.get("source_file") or "") for _, _, meta in blocks}
    if len(sources) != 1 or not next(iter(sources)):
        raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
    source = Path(next(iter(sources)))
    discovered = discover_file(source)
    if discovered is None:
        raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
    chunks = chunk_document(parse_markdown(read_text(source)), str(source))
    locations = {chunk.block_id: (index, chunk) for index, chunk in enumerate(chunks)}

    def position_of(block):
        block_id, text, meta = block
        parts = block_id.split("::", 2)
        if len(parts) != 3:
            raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
        match = re.fullmatch(r"(.+)-s(\d+)", parts[2])
        chunk_id = match.group(1) if match else parts[2]
        if chunk_id not in locations:
            raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
        index, chunk = locations[chunk_id]
        if meta.get("doc_id") != discovered.content_hash and (
            meta.get("chunk_type") == "image_description" or not (
                text in chunk.text if match else text == chunk.text
            )
        ):
            raise KnowledgeError("DOCUMENT_ORDER_UNAVAILABLE")
        return index, int(match.group(2)) if match else 0

    return sorted(blocks, key=position_of)


@dataclass(frozen=True)
class KnowledgeSnapshot:
    blocks: list[Block]
    documents: dict[str, Document]
    index_revision: str
    needs_rebuild: bool

    def list_blocks(self):
        return self.blocks

    @classmethod
    def build(cls, blocks: list[Block], identity: str, needs_rebuild: bool):
        grouped: dict[str, list[Block]] = {}
        for block in blocks:
            grouped.setdefault(document_id(block[0]), []).append(block)
        documents = {}
        for key, members in grouped.items():
            metadata = members[0][2]
            name, topic = source_labels(str(metadata.get("source_file", "")))
            # 所有文本、顺序、图片证据与元数据都参与版本计算。
            revision = digest(sorted(members, key=lambda item: item[0]))
            headings = sorted({str(m[2].get("heading_path", "")) for m in members} - {""})
            documents[key] = Document(
                document_id=key, document_name=name, topic=topic,
                document_revision=revision, chunk_count=len(members),
                headings=headings[:20], truncated=len(headings) > 20,
            )
        revision = digest([
            identity, INGESTION_SCHEMA_VERSION,
            sorted((key, doc.document_revision) for key, doc in documents.items()),
        ])
        return cls(blocks, documents, revision, needs_rebuild)
