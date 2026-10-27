from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(PublicModel):
    document_id: str
    document_name: str
    topic: str
    document_revision: str
    chunk_count: int
    headings: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    truncated: bool = False


class Topic(PublicModel):
    name: str
    file_count: int


class Overview(PublicModel):
    knowledge_scope: Literal["shared"] = "shared"
    index_revision: str
    index_status: Literal["ready", "empty", "needs_rebuild"]
    file_count: int
    chunk_count: int
    image_count: int
    topics: list[Topic]
    documents: list[Document]
    truncated: bool = False


class Catalog(PublicModel):
    index_revision: str
    documents: list[Document]
    total: int
    next_cursor: str | None = None
    truncated: bool = False


class Evidence(PublicModel):
    block_id: str
    document_id: str
    document_name: str
    document_revision: str
    topic: str
    heading_path: str
    anchor: str = ""
    chunk_type: str = ""
    text: str
    similarity: float | None = None
    match_type: str = ""
    rerank_score: float | None = None
    relevance_checked: bool = False
    truncated: bool = False


class SearchResult(PublicModel):
    index_revision: str
    status: Literal["ok", "empty", "index_empty"]
    chunks: list[Evidence]
    candidate_count: int
    rerank_status: str
    truncated: bool = False


class BlockSpan(PublicModel):
    block_id: str
    heading_path: str
    start: int
    end: int


class DocumentContent(PublicModel):
    content_origin: Literal["indexed_chunks"] = "indexed_chunks"
    index_revision: str
    document_id: str
    document_name: str
    document_revision: str
    topic: str
    heading: str | None
    text: str
    offset: int
    total_chars: int
    next_offset: int | None
    blocks: list[BlockSpan]
    truncated: bool = False


class CatalogCursor(PublicModel):
    index_revision: str
    query_hash: str
    offset: int = Field(ge=0)
