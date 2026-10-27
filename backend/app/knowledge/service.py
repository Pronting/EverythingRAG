import asyncio
from collections import Counter
from dataclasses import dataclass

from openai import APIError
from pydantic import ValidationError

from app.ingestion.index_schema import make_index_fingerprint
from app.knowledge.context import KnowledgeContext
from app.knowledge.models import (
    BlockSpan,
    Catalog,
    CatalogCursor,
    DocumentContent,
    Evidence,
    KnowledgeError,
    Overview,
    SearchResult,
    Topic,
)
from app.knowledge.snapshot import KnowledgeSnapshot, digest, document_id, ordered_blocks
from app.retrieval import rerank_pipeline
from app.retrieval.evidence_view import RetrievalScope
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import QueryVariant, build_query_variants
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY


@dataclass
class KnowledgeService:
    context: KnowledgeContext
    _retriever: HybridRetriever | None = None
    _retriever_revision: str = ""

    def _assert_current(self):
        if self.context.current_fingerprint is not None and (
            self.context.current_fingerprint() != self.context.embedder.fingerprint
        ):
            raise KnowledgeError("INDEX_CHANGED")

    def _snapshot_locked(self):
        store = self.context.vector_store
        blocks = store.list_blocks()
        rebuild = self.context.state_store.needs_rebuild(
            index_fingerprint=make_index_fingerprint(self.context.embedder.fingerprint),
            current_chunk_count=len(blocks),
        )
        return KnowledgeSnapshot.build(blocks, store._collection_name, rebuild)

    def snapshot(self):
        # Provider 注册表在文件锁之外读取，保持各入口的锁顺序一致。
        self._assert_current()
        with self.context.vector_store.read_guard():
            result = self._snapshot_locked()
        self._assert_current()
        return result

    async def overview(self, limit=10) -> Overview:
        snapshot = await asyncio.to_thread(self.snapshot)
        counts = Counter(doc.topic for doc in snapshot.documents.values())
        documents = sorted(snapshot.documents.values(), key=lambda d: (-d.chunk_count, d.document_id))
        return Overview(
            index_revision=snapshot.index_revision,
            index_status="needs_rebuild" if snapshot.needs_rebuild else (
                "ready" if snapshot.blocks else "empty"
            ), file_count=len(documents), chunk_count=len(snapshot.blocks),
            image_count=sum(m.get("chunk_type") == "image_description" for _, _, m in snapshot.blocks),
            topics=[Topic(name=name, file_count=count) for name, count in counts.most_common(limit)],
            documents=[d.model_copy(update={"headings": []}) for d in documents[:limit]],
            truncated=len(documents) > limit or len(counts) > limit,
        )

    async def catalog(self, query: str, limit=10, cursor=None) -> Catalog:
        snapshot = await asyncio.to_thread(self.snapshot)
        offset = 0
        query_hash = digest(query)
        if cursor:
            try:
                parsed = CatalogCursor.model_validate_json(cursor)
            except ValidationError:
                raise KnowledgeError("INVALID_CURSOR") from None
            if parsed.index_revision != snapshot.index_revision:
                raise KnowledgeError("STALE_CURSOR")
            if parsed.query_hash != query_hash:
                raise KnowledgeError("INVALID_CURSOR")
            offset = parsed.offset
        tokens = query.casefold().split()
        all_headings: dict[str, set[str]] = {}
        for block_id, _, metadata in snapshot.blocks:
            all_headings.setdefault(document_id(block_id), set()).add(
                str(metadata.get("heading_path", ""))
            )
        scored = []
        for doc in snapshot.documents.values():
            haystack = " ".join([
                doc.document_name, doc.topic, *all_headings[doc.document_id]
            ]).casefold()
            matched = [token for token in tokens if token in haystack]
            if matched:
                headings = [h[:500] for h in doc.headings]
                entry = doc.model_copy(update={"matched_terms": matched, "headings": headings})
                scored.append((len(matched), entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].document_name, pair[1].document_id))
        if offset > len(scored):
            raise KnowledgeError("INVALID_CURSOR")
        page = []
        used = 0
        for _, doc in scored[offset:offset + limit]:
            doc = doc.model_copy(deep=True)
            while len(doc.model_dump_json()) > self.context.max_result_chars:
                if doc.headings:
                    doc.headings.pop()
                elif doc.matched_terms:
                    doc.matched_terms.pop()
                else:
                    raise KnowledgeError("RESPONSE_TOO_LARGE")
                doc.truncated = True
            size = len(doc.model_dump_json())
            if page and used + size > self.context.max_result_chars:
                break
            page.append(doc)
            used += size
        next_offset = offset + len(page)
        next_cursor = CatalogCursor(
            index_revision=snapshot.index_revision, query_hash=query_hash, offset=next_offset,
        ).model_dump_json() if next_offset < len(scored) else None
        return Catalog(index_revision=snapshot.index_revision, documents=page, total=len(scored),
                       next_cursor=next_cursor, truncated=bool(next_cursor))

    def _read(self, document: str, heading, query, offset, limit, expected_revision):
        self._assert_current()
        result = self._read_view(document, heading, query, offset, limit, expected_revision)
        self._assert_current()
        return result

    def _read_view(self, document: str, heading, query, offset, limit, expected_revision):
        with self.context.vector_store.read_guard():
            snapshot = self._snapshot_locked()
            doc = snapshot.documents.get(document)
            if doc is None:
                raise KnowledgeError("DOCUMENT_NOT_FOUND")
            if offset and not expected_revision:
                raise KnowledgeError("DOCUMENT_REVISION_REQUIRED")
            if expected_revision and expected_revision != doc.document_revision:
                raise KnowledgeError("STALE_DOCUMENT")
            members = [b for b in snapshot.blocks if document_id(b[0]) == document]
            members = ordered_blocks(members)
            if heading:
                members = [b for b in members if heading.casefold() in
                           str(b[2].get("heading_path", "")).casefold()]
                if not members:
                    raise KnowledgeError("HEADING_NOT_FOUND")
            texts, spans = [], []
            position = 0
            for block_id, text, metadata in members:
                title = str(metadata.get("heading_path", ""))
                body = f"章节：{title}\n{text}" if title else text
                texts.append(body)
                spans.append(BlockSpan(block_id=block_id, heading_path=title[:500],
                                       start=position, end=position + len(body)))
                position += len(body) + 2
            full = "\n\n".join(texts)
            if offset > len(full):
                raise KnowledgeError("INVALID_OFFSET")
            if query and offset == 0:
                hits = [full.casefold().find(token) for token in query.casefold().split()]
                hits = [value for value in hits if value >= 0]
                offset = max(0, min(hits) - limit // 4) if hits else 0
            text = full[offset:offset + min(limit, self.context.max_result_chars)]
            end = offset + len(text)
            result = DocumentContent(
                index_revision=snapshot.index_revision, document_id=document,
                document_name=doc.document_name, document_revision=doc.document_revision,
                topic=doc.topic, heading=heading, text=text, offset=offset, total_chars=len(full),
                next_offset=end if end < len(full) else None,
                blocks=[span for span in spans if span.start < end and span.end > offset],
                truncated=end < len(full),
            )
            while len(result.model_dump_json().encode()) > 110000:
                text = text[:max(1, len(text) // 2)]
                end = offset + len(text)
                result = result.model_copy(update={
                    "text": text, "next_offset": end if end < len(full) else None,
                    "blocks": [span for span in spans if span.start < end and span.end > offset],
                    "truncated": True,
                })
                if len(text) == 1 and len(result.model_dump_json().encode()) > 110000:
                    raise KnowledgeError("RESPONSE_TOO_LARGE")
            return result, full

    async def read(self, document_id, heading=None, query=None, offset=0, limit=4000,
                   expected_document_revision=None):
        result, _ = await self.read_with_content(document_id, heading, query, offset, limit,
                                                  expected_document_revision)
        return result

    async def read_with_content(self, document_id, heading=None, query=None, offset=0, limit=4000,
                                expected_document_revision=None):
        return await asyncio.to_thread(self._read, document_id, heading, query, offset, limit,
                                       expected_document_revision)

    def _retrieve(self, snapshot, variants, vectors, document_scope, top_k):
        self._assert_current()
        with self.context.vector_store.read_guard():
            current = self._snapshot_locked()
            if current.index_revision != snapshot.index_revision:
                raise KnowledgeError("INDEX_CHANGED")
            where = None
            if document_scope:
                if set(document_scope) - snapshot.documents.keys():
                    raise KnowledgeError("DOCUMENT_NOT_FOUND")
                sources = sorted({str(m["source_file"]) for bid, _, m in snapshot.blocks
                                  if document_id(bid) in document_scope})
                where = {"source_file": {"$in": sources}}
            # 同数量内容变化也重建 BM25；保护区保证读写顺序一致。
            revision = digest([snapshot.index_revision, top_k])
            if self._retriever_revision != revision:
                self._retriever = HybridRetriever(
                    self.context.vector_store, min_similarity=DEFAULT_MIN_SIMILARITY,
                    context_top_k=max(8, top_k),
                )
                self._retriever_revision = revision
            return self._retriever.retrieve_variants(variants, vectors, where=where)

    async def search_raw(self, query, top_k=8, document_scope=None, min_similarity=None,
                         progress=None):
        snapshot = await asyncio.to_thread(self.snapshot)
        if snapshot.needs_rebuild:
            raise KnowledgeError("INDEX_NEEDS_REBUILD")
        if not snapshot.blocks:
            return snapshot, [], None, 0
        if document_scope and set(document_scope) - snapshot.documents.keys():
            raise KnowledgeError("DOCUMENT_NOT_FOUND")
        runtime = rerank_pipeline.resolve_runtime(self.context.rerank)
        filtering = rerank_pipeline.runtime_configured(runtime)
        if filtering and progress:
            await progress("running", "")
        variants = build_query_variants(query) or (QueryVariant(kind="original", text=query),)
        try:
            vectors = await asyncio.to_thread(self.context.embedder.embed_queries,
                                               [variant.text for variant in variants])
        except (APIError, TimeoutError):
            raise KnowledgeError("EMBEDDING_FAILED") from None
        chunks = await asyncio.to_thread(self._retrieve, snapshot, variants, vectors,
                                         document_scope, top_k)
        raw_count = len(chunks)
        if document_scope:
            chunks = [c for c in chunks if document_id(c.block_id) in document_scope]
        if min_similarity is not None:
            chunks = [c for c in chunks if c.similarity is not None
                      and c.similarity >= min_similarity]
        outcome = None
        if filtering:
            outcome = await rerank_pipeline.filter_chunks(
                chunks=chunks, query=query, runtime=runtime, vectorstore=snapshot,
                scope=RetrievalScope(scope_id="shared", index_revision=snapshot.index_revision),
            )
            if outcome.failed:
                if progress:
                    await progress("error", outcome.error_code)
                raise KnowledgeError("RERANK_FAILED")
            chunks = list(outcome.chunks)
        current = await asyncio.to_thread(self.snapshot)
        if current.index_revision != snapshot.index_revision:
            raise KnowledgeError("INDEX_CHANGED")
        return snapshot, chunks[:top_k], outcome, raw_count

    async def search(self, query, top_k=8, document_scope=None) -> SearchResult:
        snapshot, chunks, outcome, raw_count = await self.search_raw(query, top_k, document_scope)
        evidence = []
        allowance = self.context.max_result_chars // max(1, len(chunks))
        for chunk in chunks:
            doc = snapshot.documents[document_id(chunk.block_id)]
            metadata = chunk.metadata
            text = chunk.context_text or chunk.text
            evidence.append(Evidence(
                block_id=chunk.block_id, document_id=doc.document_id,
                document_name=doc.document_name, document_revision=doc.document_revision,
                topic=doc.topic, heading_path=str(metadata.get("heading_path", ""))[:500],
                anchor=str(metadata.get("anchor", ""))[:200],
                chunk_type=str(metadata.get("chunk_type", ""))[:80], text=text[:allowance],
                similarity=chunk.similarity, match_type=str(chunk.match_type),
                rerank_score=metadata.get("rerank_score"),
                relevance_checked=bool(metadata.get("relevance_checked")),
                truncated=len(text) > allowance,
            ))
        result = SearchResult(
            index_revision=snapshot.index_revision,
            status="ok" if evidence else ("empty" if snapshot.blocks else "index_empty"),
            chunks=evidence, candidate_count=outcome.candidate_count if outcome else raw_count,
            rerank_status=outcome.status if outcome else "disabled",
            truncated=any(e.truncated for e in evidence),
        )
        while len(result.model_dump_json().encode()) > 110000:
            longest = max(result.chunks, key=lambda item: len(item.text))
            if len(longest.text) <= 1:
                raise KnowledgeError("RESPONSE_TOO_LARGE")
            longest.text = longest.text[:len(longest.text) // 2]
            longest.truncated = True
            result.truncated = True
        return result
