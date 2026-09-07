"""Deterministic concurrency, ordering, failure isolation and write batching checks."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.ingestion.parallel import ordered_parallel_map
from app.ingestion.pipeline import IngestionPipeline, embed_document_texts
from app.vectorstore.embedder import CloudEmbedder
from tests.test_pipeline import FakeEmbedder, FakeVectorStore


def test_parallel_embedding_preserves_order_and_bound(monkeypatch):
    monkeypatch.setattr(settings, "ingest_embed_workers", 4)
    barrier = Barrier(4)
    lock = Lock()
    active = maximum = 0

    class Encoder:
        def embed_documents(self, texts):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1
            return [[float(text)] for text in texts]

    vectors = embed_document_texts(Encoder(), [str(i) for i in range(32)], batch_size=4)
    assert vectors == [[float(i)] for i in range(32)]
    assert maximum == 4


def test_parallel_batch_mismatch_fails_before_vector_writes():
    class Encoder:
        def embed_documents(self, texts):
            return []
    with pytest.raises(RuntimeError, match="数量错误"):
        embed_document_texts(Encoder(), ["test"])


def test_parallel_map_failure_propagates_and_none_is_valid():
    assert list(ordered_parallel_map(lambda x: x, [None, 1], 2)) == [None, 1]
    def fail(value):
        if value == 2:
            raise ValueError("bad input")
        return value
    with pytest.raises(ValueError, match="bad input"):
        list(ordered_parallel_map(fail, range(6), 2))


def test_cloud_client_constructed_once_and_reorders_provider_results(monkeypatch):
    import app.vectorstore.embedder as module
    made = []
    def make(**kwargs):
        made.append(True)
        return SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(kw["input"][i])])
                  for i in reversed(range(len(kw["input"])))],
        )))
    monkeypatch.setattr(module, "OpenAI", make)
    encoder = CloudEmbedder("http://test/v1", "test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: encoder.embed_documents(["1", "2"]), range(12)))
    assert len(made) == 1
    assert all(result == [[1.0], [2.0]] for result in results)


def test_files_parse_in_parallel_and_text_commits_are_batched(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ingest_parse_workers", 4)
    for i in range(8):
        (tmp_path / f"{i}.md").write_text(f"# Title {i}\n\nDocument {i}", encoding="utf-8")
    class Store(FakeVectorStore):
        writes = 0
        def upsert(self, blocks):
            self.writes += 1
            super().upsert(blocks)
    store = Store()
    pipeline = IngestionPipeline(FakeEmbedder(), store)
    original = pipeline.prepare_file
    barrier = Barrier(4)
    def prepare(file):
        barrier.wait(timeout=5)
        if file.filename == "3.md":
            raise ValueError("private filename")
        return original(file)
    monkeypatch.setattr(pipeline, "prepare_file", prepare)
    report = pipeline.ingest(tmp_path)
    assert report.files_parsed == 7
    assert report.files_skipped == 1
    assert report.errors == ("ValueError",)
    assert store.writes == 1
    assert len(store.blocks) == 7
    assert store.repair_calls == 1
