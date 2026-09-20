"""工作区知识隔离（PRD 5.1–5.3、5.4；A06 / A07 / A08，T01）。

A06：两个工作区各自含**同名**真实文档（都叫 ``notes.md``）时，dense、BM25、章节补充、
浏览（目录）与统计各自只读取本区。

A07：用其它工作区的图片 hash 或任务 ID 取原图/任务状态必须 404。

A08：导入过程中切换工作区，任务仍写入**创建时**的工作区。

这些用例刻意使用**真实 Chroma**（``create_vector_store`` + 两个独立 persist 目录），
而不是计数替身 —— 串读问题只有在真实 collection 上才会暴露。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.core.workspace_context import (
    BM25_GENERATOR_VERSION,
    BackgroundTaskScope,
    WorkspaceContextFactory,
    committed_storage_version,
)
from app.core.workspace_store import WORKSPACE_LAYOUT_VERSION, WorkspaceStore
from app.ingestion.image_state_store import ImageStateStore, ImageTask
from app.main import app
from app.models.schemas import BlockMetadata, SourceType
from app.retrieval.workspace_retrieval import (
    clear_workspace_area_cache,
    workspace_area_cache_size,
    workspace_knowledge_area,
)
from app.vectorstore.chroma_store import create_vector_store
from tests.fakes import FakeEmbedder

#: 两区共用的文件名 —— A06 的关键前提是「同名不同内容」。
SHARED_FILE_NAME = "notes.md"
#: 只在 A 区出现的标记，用来断言「B 区任何路径都不会吐回 A 区的正文」。
MARKER_A = "ALPHA-ONLY-内部风控方案"
MARKER_B = "BETA-ONLY-营销投放方案"


def _contexts(tmp_path: Path, count: int = 2) -> tuple[Any, list[Any], list[Any]]:
    store = WorkspaceStore(tmp_path)
    factory = WorkspaceContextFactory(tmp_path, store)
    workspaces = [store.create(name) for name in ("甲区", "乙区", "丙区")[:count]]
    return factory, [store], [factory.resolve(workspace.id) for workspace in workspaces]


def _seed_area(
    context: Any,
    *,
    block_id: str,
    text: str,
    vector: list[float],
    heading: str,
) -> Any:
    """在**本区** chroma 目录写一个真实 collection，返回本区检索面。"""

    embedder = FakeEmbedder()
    context.chroma_dir.mkdir(parents=True, exist_ok=True)
    store = create_vector_store(
        persist_dir=context.chroma_dir,
        collection_name=f"chunks__{embedder.fingerprint}__v7",
        embedder=embedder,
    )
    store.upsert(
        [
            (
                block_id,
                text,
                vector,
                BlockMetadata(
                    block_id=block_id,
                    doc_id=f"doc-{block_id}",
                    source_type=SourceType.DOCUMENT,
                    source_file=str(context.documents_dir / SHARED_FILE_NAME),
                    heading_path=heading,
                    chunk_type="text",
                ),
            )
        ]
    )
    return workspace_knowledge_area(
        context,
        vector_store=store,
        index_fingerprint=f"chunks__{embedder.fingerprint}__v7",
    )


@pytest.fixture(autouse=True)
def _clean_area_cache() -> None:
    clear_workspace_area_cache()
    deps.reset_workspace_scoped_caches()


def test_a06_two_workspaces_same_file_name_stay_isolated(tmp_path: Path) -> None:
    """A06：dense / BM25 / 章节补充 / 浏览 / 统计分别只读本区。"""

    _factory, _stores, (context_a, context_b) = _contexts(tmp_path)
    area_a = _seed_area(
        context_a,
        block_id="block-a",
        text=f"# 风控\n\n{MARKER_A} 的 P99 延迟指标与限流阈值。",
        vector=[1.0, 0.0, 0.0, 0.0],
        heading="风控/延迟",
    )
    area_b = _seed_area(
        context_b,
        block_id="block-b",
        text=f"# 营销\n\n{MARKER_B} 的投放预算与转化率。",
        vector=[0.0, 1.0, 0.0, 0.0],
        heading="营销/预算",
    )

    # --- 统计（PRD 5.3.2 / 5.4：知识概览按区统计）---
    assert area_a.stats() == {"files": 1, "chunks": 1, "images": 0}
    assert area_b.stats() == {"files": 1, "chunks": 1, "images": 0}

    # --- 浏览：本区块与目录 ---
    assert [block[0] for block in area_a.blocks()] == ["block-a"]
    assert [block[0] for block in area_b.blocks()] == ["block-b"]
    assert area_a.overview().file_count == 1
    assert area_a.overview().representative_documents[0].name == SHARED_FILE_NAME

    # --- dense + BM25 + 章节补充：同一批查询在两个区互不可见 ---
    for area, own_id, own_vector, own_kind in (
        (area_a, "block-a", [1.0, 0.0, 0.0, 0.0], "风控"),
        (area_b, "block-b", [0.0, 1.0, 0.0, 0.0], "营销"),
    ):
        chunks = area.retriever.retrieve(own_kind, own_vector)
        assert chunks, "本区自有文档必须能被检索到"
        ids = {chunk.block_id for chunk in chunks}
        assert own_id in ids
        # 关键断言：本区检索结果里**永远**不出现另一个区的块
        assert ids <= {own_id}

        for chunk in chunks:
            haystack = f"{chunk.text}{chunk.context_text or ''}"
            other_marker = MARKER_B if own_id == "block-a" else MARKER_A
            assert other_marker not in haystack

    # --- 缓存键含 workspace_id（PRD 5.3.3）---
    assert area_a.cache_key != area_b.cache_key
    assert area_a.context.workspace_id != area_b.context.workspace_id


def test_a06_index_cache_key_changes_with_knowledge_revision(tmp_path: Path) -> None:
    """PRD 5.3.4 / 5.3.6：本区知识版本递增后必须换键（否则继续读旧索引）。"""

    store = WorkspaceStore(tmp_path)
    factory = WorkspaceContextFactory(tmp_path, store)
    workspace = store.create("甲区")
    context = factory.resolve(workspace.id)
    embedder = FakeEmbedder()
    context.chroma_dir.mkdir(parents=True, exist_ok=True)
    vector_store = create_vector_store(
        persist_dir=context.chroma_dir,
        collection_name=f"chunks__{embedder.fingerprint}__v7",
        embedder=embedder,
    )
    fingerprint = f"chunks__{embedder.fingerprint}__v7"

    first = workspace_knowledge_area(
        context, vector_store=vector_store, index_fingerprint=fingerprint
    )
    assert (
        workspace_knowledge_area(
            context, vector_store=vector_store, index_fingerprint=fingerprint
        )
        is first
    )

    store.bump_knowledge_revision(workspace.id)
    after = factory.resolve(workspace.id)
    second = workspace_knowledge_area(
        after, vector_store=vector_store, index_fingerprint=fingerprint
    )
    assert second.cache_key != first.cache_key
    assert second.knowledge_revision == 1
    assert workspace_area_cache_size() == 2


def test_a07_cross_workspace_image_hash_and_task_id_return_404(tmp_path: Path) -> None:
    """A07：原图与任务身份同时校验工作区；跨区 hash / 任务 ID 一律 404。"""

    store = deps.workspace_store
    factory = WorkspaceContextFactory(committed_version_dir(tmp_path), store)
    context_a = factory.resolve(store.create_with_uuid("甲区").id)
    context_b = factory.resolve(store.create_with_uuid("乙区").id)

    image_hash = "a" * 32
    store_a = deps.workspace_image_state_store(context_a)
    store_a.save_image_bytes(image_hash, _png_bytes())
    store_a.add_task(
        ImageTask(
            block_id="task-a",
            src="https://example.invalid/a.png",
            doc_id="doc-a",
            source_file="notes.md",
            heading_path="",
            anchor="",
        )
    )

    client = TestClient(app)
    ok = client.get(f"/api/knowledge/images/{image_hash}", params={"workspace_id": context_a.workspace_id})
    assert ok.status_code == 200, ok.text

    leaked = client.get(f"/api/knowledge/images/{image_hash}", params={"workspace_id": context_b.workspace_id})
    assert leaked.status_code == 404
    assert image_hash not in leaked.text

    task_ok = client.get("/api/knowledge/image-tasks/task-a", params={"workspace_id": context_a.workspace_id})
    assert task_ok.status_code == 200, task_ok.text
    assert task_ok.json()["workspace_id"] == context_a.workspace_id

    task_leaked = client.get("/api/knowledge/image-tasks/task-a", params={"workspace_id": context_b.workspace_id})
    assert task_leaked.status_code == 404

    # 不存在的工作区不能静默回落到默认区
    assert client.get("/api/knowledge/images/" + image_hash, params={"workspace_id": "nope"}).status_code == 404

    # 服务层同样隔离：B 区的库里没有 A 区的原图与任务
    store_b = deps.workspace_image_state_store(context_b)
    assert store_b.get_image_bytes(image_hash) is None
    assert store_b.get_task("task-a") is None
    assert store_b.task_status("task-a") is None
    # 本区独立文件：两个库的路径不同
    assert store_a is not store_b
    assert context_a.image_state_db != context_b.image_state_db


def test_a08_background_task_keeps_creation_time_workspace(tmp_path: Path) -> None:
    """A08：导入时切换工作区，任务仍写创建时的工作区。"""

    store = deps.workspace_store
    data_dir = committed_version_dir(tmp_path)
    factory = WorkspaceContextFactory(data_dir, store)
    workspace_a = store.create_with_uuid("甲区")
    workspace_b = store.create_with_uuid("乙区")
    context_a = factory.resolve(workspace_a.id)

    # 任务创建时绑定上下文（PRD 5.2 末段）
    scope = BackgroundTaskScope.bind(context_a)
    assert scope.workspace_id == workspace_a.id

    # 用户随后切到别的区并发起新的解析
    context_b = factory.resolve(workspace_b.id)
    assert context_b.workspace_id == workspace_b.id
    assert context_b.state_db != scope.state_db

    # 任务继续写**创建时**的区：指纹落进甲区 state.db，乙区一无所获
    from app.ingestion.state_store import DocumentStateStore

    task_store = DocumentStateStore(scope.state_db)
    task_store.upsert_fingerprint(str(scope.documents_dir / "notes.md"), "hash", 1.0, 3, 1)
    assert scope.state_db.is_file()
    assert DocumentStateStore(context_b.state_db).load_all() == {}
    task_store.close()


def _png_bytes() -> bytes:
    """真实可解码的 PNG（原图端点会用 PIL 校验格式）。"""

    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def committed_version_dir(tmp_path: Path) -> Path:
    """返回一个已提交 ``storage_schema_version=2`` 的数据目录（每工作区布局）。

    ``deps`` 的按区 provider 读 ``settings.data_dir``（conftest 已隔离到 tmp 并提交
    标记）；这里保持一致，避免用例自己造出旧布局而翻转解析。
    """

    from app.core.config import settings
    from app.core.workspace_context import commit_storage_version

    data_dir = Path(settings.data_dir)
    if committed_storage_version(data_dir) < WORKSPACE_LAYOUT_VERSION:
        commit_storage_version(data_dir)
    return data_dir


def test_vector_cache_key_includes_collection_identity(tmp_path: Path) -> None:
    """PRD 5.3.3：collection 身份变化必须换向量句柄（入库版本升级场景）。"""

    factory = WorkspaceContextFactory(tmp_path, WorkspaceStore(tmp_path))
    context = factory.resolve()
    first = context.vector_cache_key(provider_digest="p", collection_identity="chunks__m__v7")
    second = context.vector_cache_key(provider_digest="p", collection_identity="chunks__m__v6")
    assert first != second
    assert BM25_GENERATOR_VERSION in context.index_cache_key(
        index_fingerprint="f", generator_version=BM25_GENERATOR_VERSION
    )


def test_delete_knowledge_bumps_only_the_target_workspace_revision(tmp_path: Path) -> None:
    """PRD 5.3.6：清空知识在本区提交成功后递增**本区**版本，别区不受影响。"""

    from app.ingestion.state_store import DocumentStateStore
    from tests.fakes import FakeVectorStore

    store = deps.workspace_store
    factory = WorkspaceContextFactory(committed_version_dir(tmp_path), store)
    workspace = store.create_with_uuid("甲区")
    other = store.create_with_uuid("乙区")
    before_target = factory.resolve(workspace.id).knowledge_revision
    before_other = factory.resolve(other.id).knowledge_revision

    state = DocumentStateStore(tmp_path / "state.db")
    images = ImageStateStore(tmp_path / "images.db")
    app.dependency_overrides[deps.get_workspace_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[deps.get_workspace_state_store] = lambda: state
    app.dependency_overrides[deps.get_workspace_image_state_store] = lambda: images
    try:
        response = TestClient(app).delete(
            "/api/knowledge", params={"workspace_id": workspace.id}
        )
        assert response.status_code == 200, response.text
    finally:
        app.dependency_overrides.pop(deps.get_workspace_vector_store, None)
        app.dependency_overrides.pop(deps.get_workspace_state_store, None)
        app.dependency_overrides.pop(deps.get_workspace_image_state_store, None)
        state.close()
        images.close()

    assert factory.resolve(workspace.id).knowledge_revision == before_target + 1
    assert factory.resolve(other.id).knowledge_revision == before_other
