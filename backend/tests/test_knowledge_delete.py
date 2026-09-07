from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.state_store import DocumentStateStore
from app.main import app
from tests.fakes import FakeVectorStore


class DeletableStore(FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.blocks = [(str(i), "text", {}) for i in range(1002)]

    def list_blocks(self):
        return self.blocks

    def delete_by_ids(self, block_ids):
        assert len(block_ids) <= 1000
        self.blocks = [block for block in self.blocks if block[0] not in block_ids]


@pytest.fixture
def stores(tmp_path: Path):
    vector = DeletableStore()
    state = DocumentStateStore(tmp_path / "docs.db")
    images = ImageStateStore(tmp_path / "images.db")
    source = tmp_path / "notes.md"
    source.write_text("Keep original", encoding="utf-8")
    state.register_source(tmp_path)
    images.save_description("hash", "cached")
    app.dependency_overrides[deps.get_vector_store] = lambda: vector
    app.dependency_overrides[deps.get_state_store] = lambda: state
    app.dependency_overrides[deps.get_image_state_store] = lambda: images
    yield vector, state, images, source
    state.close()
    images.close()


def test_delete_clears_index_and_registrations_but_keeps_files(stores):
    vector, state, images, source = stores
    client = TestClient(app)
    response = client.delete("/api/knowledge")
    assert response.status_code == 200
    assert response.json() == {"deleted_chunks": 1002}
    assert vector.list_blocks() == []
    assert state.list_sources() == []
    assert images.get_description("hash") is None
    assert source.read_text(encoding="utf-8") == "Keep original"
    assert client.delete("/api/knowledge").json() == {"deleted_chunks": 0}


@pytest.mark.parametrize("running_images", [False, True])
def test_delete_rejects_background_work(stores, monkeypatch, running_images):
    vector, state, images, _ = stores
    if running_images:
        monkeypatch.setattr(images, "count_status", lambda: {"pending": 1})
    else:
        deps.import_task_store.create()
    assert TestClient(app).delete("/api/knowledge").status_code == 409
    assert len(vector.list_blocks()) == 1002
    assert state.list_sources()
    assert images.get_description("hash") == "cached"


def test_delete_real_chroma_index_through_api(stores, tmp_path):
    from app.ingestion.pipeline import IngestionPipeline
    from app.vectorstore.chroma_store import create_vector_store
    from tests.fakes import FakeEmbedder

    _, state, _images, source = stores
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "test.md").write_text("# 文档\n\n用于验证删除接口的知识内容。", encoding="utf-8")
    embedder = FakeEmbedder()
    vector = create_vector_store(persist_dir=tmp_path / "chroma", embedder=embedder)
    report = IngestionPipeline(embedder, vector, document_state_store=state).ingest(corpus)
    assert report.blocks_upserted > 0
    app.dependency_overrides[deps.get_vector_store] = lambda: vector
    assert TestClient(app).delete("/api/knowledge").status_code == 200
    assert vector.count() == 0
    assert state.load_all() == {}
    assert state.list_sources() == []
    assert source.exists()
    assert (corpus / "test.md").exists()
