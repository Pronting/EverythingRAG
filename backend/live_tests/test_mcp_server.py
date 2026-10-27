import asyncio
import json
import multiprocessing
import shutil
import socket
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
import httpx2
import pytest
import uvicorn
from chromadb.errors import NotFoundError
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
QUERY = "HybridRetriever 如何使用 dense、BM25 和 RRF 进行检索？"
sys.path.insert(0, str(ROOT / "backend"))

from app.api.deps import build_rerank_runtime, get_embedder
from app.core.config import Settings
from app.core.settings_store import SettingsStore, get_settings_store
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.state_store import DocumentStateStore
from app.knowledge.context import KnowledgeContext
from app.knowledge.models import KnowledgeError
from app.knowledge.service import KnowledgeService
from app.mcp_server.server import MCPRuntime
from app.models.schemas import BlockMetadata
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.write_lock import exclusive_write_lock


def hold_maintenance_lock(path, acquired, release):
    with exclusive_write_lock(Path(path)):
        acquired.set()
        release.wait(20)


@pytest.fixture(scope="module")
def service():
    directory = ROOT / ".tmp_probe" / "mcp-acceptance" / uuid4().hex
    sources = directory / "sources"
    sources.mkdir(parents=True)
    for source in (ROOT / "Agent.md", ROOT / "docs/tech/rag-evidence-and-recovery.md"):
        shutil.copy2(source, sources / source.name)
    embedder = get_embedder()
    vectorstore = create_vector_store(directory / "index", embedder=embedder)
    state = DocumentStateStore(directory / "state.db")
    images = ImageStateStore(directory / "images.db")
    pipeline = IngestionPipeline(embedder, vectorstore, state_store=images,
                                 document_state_store=state)
    report = pipeline.ingest(sources)
    assert not report.errors and report.blocks_upserted > 0
    context = KnowledgeContext(vectorstore, embedder, state, build_rerank_runtime)
    yield KnowledgeService(context)
    state.close()
    images.close()


@asynccontextmanager
async def listening(service, timeout=45, concurrency=4):
    from starlette.applications import Starlette
    from starlette.routing import Mount

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    token = uuid4().hex + uuid4().hex
    settings = Settings(mcp_enabled=True, mcp_token=token, port=port,
                        mcp_timeout_seconds=timeout, mcp_max_concurrency=concurrency)
    runtime = MCPRuntime(settings, lambda: service)

    @asynccontextmanager
    async def lifespan(app):
        async with runtime.server.session_manager.run():
            yield
            await runtime.aclose()

    app = Starlette(routes=[Mount("/mcp", app=runtime.app)], lifespan=lifespan)
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("MCP_START_FAILED")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp/", token, runtime
    finally:
        server.should_exit = True
        await task
        listener.close()


@pytest.mark.asyncio
async def test_real_import_http_and_internal_tools(service):
    from app.agent.tools import (
        LookupCatalogParams,
        ReadDocumentParams,
        SearchKnowledgeParams,
        ToolContext,
        _lookup_catalog,
        _read_document,
        _search_knowledge,
    )
    from app.generation.knowledge_catalog import _document_id_for

    overview = await service.overview()
    assert overview.file_count == 2 and overview.index_status == "ready"
    catalog = await service.catalog(".md", limit=1)
    assert catalog.next_cursor and len(catalog.documents) == 1
    next_page = await service.catalog(".md", limit=1, cursor=catalog.next_cursor)
    assert next_page.documents[0].document_id != catalog.documents[0].document_id
    doc = (await service.catalog("Agent.md")).documents[0]
    first = await service.read(doc.document_id, limit=100)
    second = await service.read(doc.document_id, offset=100, limit=100,
                                expected_document_revision=first.document_revision)
    full = await service.read(doc.document_id, limit=200)
    assert first.text + second.text == full.text
    result = await service.search(QUERY, document_scope=[doc.document_id])
    assert result.chunks and all(c.document_id == doc.document_id for c in result.chunks)
    assert any("BM25" in c.text for c in result.chunks)
    context = ToolContext(workspace=None, knowledge_service=service)
    internal = await _lookup_catalog(LookupCatalogParams(query="Agent.md"), context)
    assert internal.data["documents"][0]["document_id"] == doc.document_id
    internal_read = await _read_document(ReadDocumentParams(document_id=doc.document_id), context)
    assert internal_read.read_document_content.startswith(first.text)
    internal_search = await _search_knowledge(SearchKnowledgeParams(query=QUERY), context)
    assert internal_search.data["chunks"]
    source = next(m["source_file"] for bid, _, m in service.snapshot().blocks
                  if bid.split("::", 1)[0] == doc.document_id)
    scoped = await _search_knowledge(
        SearchKnowledgeParams(query=QUERY, document_scope=[_document_id_for(source)]), context,
    )
    assert scoped.data["chunks"]
    assert all(c["document_id"] == doc.document_id for c in scoped.data["chunks"])
    # 每次建立全新的 SDK 生命周期，验证重复启动和连接关闭。
    for mode in ("auto", "legacy"):
        async with listening(service) as (url, token, runtime), httpx.AsyncClient() as http:
            assert (await http.post(url)).status_code == 401
            assert (await http.delete(url)).status_code == 401
            assert (await http.post(url, headers={"Authorization": f"Bearer {token}",
                                                  "Origin": "https://external.invalid"})).status_code == 403
            async with (
                httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as sdk_http,
                Client(streamable_http_client(url, http_client=sdk_http), mode=mode) as client,
            ):
                tools = await client.list_tools()
                assert len(tools.tools) == 4
                response = await client.call_tool("get_knowledge_overview", {})
                assert not response.is_error
                assert json.loads(response.content[0].text) == response.structured_content
                assert response.structured_content["file_count"] == 2
                for args, code in [
                    ({"document_id": doc.document_id, "offset": 1}, "DOCUMENT_REVISION_REQUIRED"),
                    ({"document_id": "missing"}, "DOCUMENT_NOT_FOUND"),
                    ({"document_id": doc.document_id, "limit": 16001}, "INVALID_ARGUMENTS"),
                    ({"document_id": doc.document_id, "unexpected": "private-input"},
                     "INVALID_ARGUMENTS"),
                ]:
                    response = await client.call_tool("read_document", args)
                    assert response.is_error and response.content[0].text == code
            assert not runtime.tasks


@pytest.mark.asyncio
async def test_real_equal_count_revision_and_cache(service):
    catalog = await service.catalog(".md", limit=1)
    doc = (await service.catalog("Agent.md")).documents[0]
    before = await service.read(doc.document_id, limit=100)
    await service.search(QUERY, document_scope=[doc.document_id])
    old_retriever = service._retriever
    store = service.context.vector_store
    with store.read_guard():
        original = next(block for block in store.list_blocks()
                        if block[0].split("::", 1)[0] == doc.document_id)
        other = next(block for block in store.list_blocks() if block[1] != original[1])
        records = store._collection.get(ids=[original[0]], include=["embeddings"])
        original_vector = records["embeddings"][0].tolist()
        count = store.count()
    vector = await asyncio.to_thread(service.context.embedder.embed_texts, [other[1]])
    try:
        with exclusive_write_lock(store.maintenance_lock_path):
            store.upsert([(original[0], other[1], vector[0], BlockMetadata(**original[2]))])
        assert store.count() == count
        with pytest.raises(KnowledgeError, match="STALE_DOCUMENT"):
            await service.read(doc.document_id, offset=1,
                               expected_document_revision=before.document_revision)
        with pytest.raises(KnowledgeError, match="STALE_CURSOR"):
            await service.catalog(".md", cursor=catalog.next_cursor)
        await service.search(QUERY, document_scope=[doc.document_id])
        assert service._retriever is not old_retriever
    finally:
        with exclusive_write_lock(store.maintenance_lock_path):
            store.upsert([(original[0], original[1], original_vector, BlockMetadata(**original[2]))])


@pytest.mark.asyncio
async def test_real_lock_timeout_and_capacity(service):
    acquired, release = threading.Event(), threading.Event()
    path = service.context.vector_store.maintenance_lock_path

    def maintain():
        with exclusive_write_lock(path):
            acquired.set()
            release.wait(10)

    thread = threading.Thread(target=maintain)
    thread.start()
    await asyncio.to_thread(acquired.wait, 5)
    try:
        async with listening(service) as (_, _, runtime):
            with pytest.raises(KnowledgeError, match="INDEX_UPDATING"):
                await runtime.execute("overview")
    finally:
        release.set()
        await asyncio.to_thread(thread.join)
    with exclusive_write_lock(path), exclusive_write_lock(path, timeout_seconds=0):
        pass
    async with listening(service, timeout=0.000001, concurrency=1) as (_, _, runtime):
        with pytest.raises(KnowledgeError, match="TOOL_TIMEOUT"):
            await runtime.execute("overview")
        assert runtime.semaphore.locked()
        with pytest.raises(KnowledgeError, match="SERVER_BUSY"):
            await runtime.execute("overview")
        await runtime.aclose()
        assert not runtime.semaphore.locked()


@pytest.mark.asyncio
async def test_real_rerank_configuration_failure(service):
    directory = ROOT / ".tmp_probe" / "mcp-acceptance" / uuid4().hex
    store = SettingsStore(directory)
    configuration = get_settings_store().load().model_copy(deep=True)
    configuration.rerank.enabled = True
    configuration.rerank.profile_id = None
    store.save(configuration)
    context = replace(service.context, rerank=lambda: build_rerank_runtime(store))
    with pytest.raises(KnowledgeError, match="RERANK_FAILED"):
        await KnowledgeService(context).search(QUERY)


@pytest.mark.asyncio
async def test_empty_index_and_output_budget(service):
    directory = ROOT / ".tmp_probe" / "mcp-acceptance" / uuid4().hex
    store = create_vector_store(directory, embedder=service.context.embedder)
    state = DocumentStateStore(directory / "state.db")
    try:
        empty = KnowledgeService(replace(service.context, vector_store=store, state_store=state))
        assert (await empty.overview()).index_status == "empty"
        assert (await empty.search(QUERY)).status == "index_empty"
        assert (await empty.catalog(".md")).total == 0
    finally:
        state.close()
    limited = KnowledgeService(replace(service.context, max_result_chars=1024))
    doc = (await limited.catalog("rag-evidence-and-recovery.md")).documents[0]
    page = await limited.read(doc.document_id, limit=16000)
    assert len(page.text) <= 1024 and page.next_offset and page.truncated
    search = await limited.search(QUERY)
    assert sum(len(chunk.text) for chunk in search.chunks) <= 1024


@pytest.mark.asyncio
async def test_real_transport_measurements(service):
    direct, http_times = [], []
    async with listening(service) as (url, token, _), httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
    ) as http, Client(streamable_http_client(url, http_client=http)) as client:
        for _ in range(6):
            started = perf_counter()
            result = await service.search(QUERY, top_k=3)
            direct.append((perf_counter() - started) * 1000)
            assert result.chunks
            started = perf_counter()
            response = await client.call_tool("search_knowledge", {"query": QUERY, "top_k": 3})
            http_times.append((perf_counter() - started) * 1000)
            assert not response.is_error
            assert response.structured_content == result.model_dump()

    def metrics(values):
        ordered = sorted(values)
        return {"samples": len(values), "p50_ms": round((ordered[2] + ordered[3]) / 2, 2),
                "p95_ms": round(ordered[-1], 2)}

    report = {"direct": metrics(direct), "mcp_http": metrics(http_times),
              "failures": 0, "documents": 2, "embedding_cache": "warm",
              "rerank_status": result.rerank_status}
    (ROOT / ".tmp_probe" / "mcp-performance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def test_real_cross_process_maintenance(service):
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    process = context.Process(target=hold_maintenance_lock,
                              args=(service.context.vector_store.maintenance_lock_path,
                                    acquired, release))
    process.start()
    try:
        assert acquired.wait(15)
        with pytest.raises(TimeoutError):
            service.snapshot()
    finally:
        release.set()
        process.join(timeout=15)
    assert process.exitcode == 0
    assert service.snapshot().blocks


def test_read_guard_does_not_create_collection(service):
    directory = ROOT / ".tmp_probe" / "mcp-acceptance" / uuid4().hex
    store = create_vector_store(directory, embedder=service.context.embedder)
    store._client.delete_collection(store._collection_name)
    with pytest.raises(NotFoundError), store.read_guard():
        store.list_blocks()
    assert store._collection_name not in [c.name for c in store._client.list_collections()]


@pytest.mark.asyncio
async def test_parallel_real_http_reads(service):
    doc = (await service.catalog("Agent.md")).documents[0]
    async with listening(service) as (url, token, _), httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
    ) as http, Client(streamable_http_client(url, http_client=http)) as client:
        results = await asyncio.gather(
            client.call_tool("get_knowledge_overview", {}),
            client.call_tool("lookup_catalog", {"query": ".md"}),
            client.call_tool("search_knowledge", {"query": QUERY}),
            client.call_tool("read_document", {"document_id": doc.document_id}),
        )
        assert all(not result.is_error for result in results)
        assert len({result.structured_content["index_revision"] for result in results}) == 1
