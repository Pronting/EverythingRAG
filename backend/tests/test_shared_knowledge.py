"""全局共享知识库（D-KB，2026-09-20 用户决策：取消「按工作区划分知识库」）。

取代原 ``test_workspace_isolation.py`` 的 A06/A07/A08：那三条验收的前提是「知识按区
隔离」，本文件按新决策把同一批不变量**反过来**钉死：

- 所有工作区解析出的知识路径相同（只有一份语料）；
- 一区写入的块、图片、指纹，其它区**都能看到**（dense / BM25 / 章节 / 概览 / 统计）；
- 知识版本递增**波及所有工作区**（否则邻区的索引缓存不会失效）；
- 工作区私有件（记忆 / 检查点 / 工件 / jieba 缓存）仍按区隔离，不得串。

这些用例刻意使用**真实 Chroma**（``create_vector_store`` + 共享 persist 目录），
而不是计数替身 —— 句柄复用与缓存键问题只有在真实 collection 上才会暴露。
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

#: 两区共用的文件名。
SHARED_FILE_NAME = "notes.md"
#: 只在 A 区导入的标记 —— 共享后 B 区**必须**也能检索到它。
MARKER_A = "ALPHA-ONLY-内部风控方案"


def _contexts(tmp_path: Path, count: int = 2) -> list[Any]:
    """在 ``tmp_path`` 下造 ``count`` 个真实工作区上下文。"""

    store = WorkspaceStore(tmp_path)
    factory = WorkspaceContextFactory(tmp_path, store)
    workspaces = [store.create(name) for name in ("甲区", "乙区", "丙区")[:count]]
    return [factory.resolve(workspace.id) for workspace in workspaces]


def _seed_area(
    context: Any,
    *,
    block_id: str,
    text: str,
    vector: list[float],
    heading: str,
) -> Any:
    """往**共享** collection 写一个真实块，返回检索面（共享向量库）。"""

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


def test_a06_knowledge_written_in_one_workspace_is_visible_in_another(
    tmp_path: Path,
) -> None:
    """A06（D-KB）：甲区导入的块，乙区的 dense / BM25 / 统计 / 浏览都能看到。"""

    context_a, context_b = _contexts(tmp_path)
    area_a = _seed_area(
        context_a,
        block_id="block-a",
        text=f"# 风控\n\n{MARKER_A} 的 P99 延迟指标与限流阈值。",
        vector=[1.0, 0.0, 0.0, 0.0],
        heading="风控/延迟",
    )
    area_b = workspace_knowledge_area(
        context_b,
        vector_store=area_a.vector_store,  # 同一份语料 ⇒ 同一个句柄
        index_fingerprint=area_a.index_fingerprint,
    )

    # 知识路径相同、块可见
    assert context_a.chroma_dir == context_b.chroma_dir
    assert context_a.documents_dir == context_b.documents_dir
    assert [block[0] for block in area_b.blocks()] == ["block-a"]
    assert area_b.stats() == {"files": 1, "chunks": 1, "images": 0}
    assert area_b.overview().representative_documents[0].name == SHARED_FILE_NAME

    # 乙区检索到的是甲区写进去的正文（共享的唯一语料）
    chunks = area_b.retriever.retrieve("风控 延迟", [1.0, 0.0, 0.0, 0.0])
    assert chunks, "共享语料必须能被任何工作区检索到"
    assert {chunk.block_id for chunk in chunks} == {"block-a"}
    assert MARKER_A in f"{chunks[0].text}{chunks[0].context_text or ''}"

    # 向量句柄键不含工作区 ⇒ 两个区拿到同一个键（同一个句柄）
    assert context_a.vector_cache_key(
        provider_digest="p", collection_identity="chunks__m__v7"
    ) == context_b.vector_cache_key(
        provider_digest="p", collection_identity="chunks__m__v7"
    )


def test_a06_index_cache_key_changes_with_knowledge_revision(tmp_path: Path) -> None:
    """PRD 5.3.4 / 5.3.6：知识版本递增后必须换键（否则继续读旧索引）。"""

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


def test_a07_image_and_task_are_visible_from_every_workspace(tmp_path: Path) -> None:
    """A07（D-KB）：图片与任务在共享库中，任何工作区都能取到；不存在的仍是 404。"""

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
    ok = client.get(
        f"/api/knowledge/images/{image_hash}", params={"workspace_id": context_a.workspace_id}
    )
    assert ok.status_code == 200, ok.text
    # 共享：乙区取同一张原图同样 200（语料只有一份）
    shared = client.get(
        f"/api/knowledge/images/{image_hash}", params={"workspace_id": context_b.workspace_id}
    )
    assert shared.status_code == 200, shared.text

    task_ok = client.get(
        "/api/knowledge/image-tasks/task-a", params={"workspace_id": context_a.workspace_id}
    )
    assert task_ok.status_code == 200, task_ok.text
    assert task_ok.json()["workspace_id"] == context_a.workspace_id
    task_shared = client.get(
        "/api/knowledge/image-tasks/task-a", params={"workspace_id": context_b.workspace_id}
    )
    assert task_shared.status_code == 200, task_shared.text
    assert task_shared.json()["workspace_id"] == context_b.workspace_id

    # 不存在的工作区不能静默回落到默认区；不存在的 hash 仍是 404
    assert (
        client.get(
            f"/api/knowledge/images/{image_hash}", params={"workspace_id": "nope"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/knowledge/images/" + "b" * 32,
            params={"workspace_id": context_a.workspace_id},
        ).status_code
        == 404
    )

    # 共享库：两个上下文解析到同一个图片库文件与同一个句柄
    store_b = deps.workspace_image_state_store(context_b)
    assert store_b is store_a
    assert context_a.image_state_db == context_b.image_state_db
    assert store_b.get_image_bytes(image_hash) is not None
    assert store_b.get_task("task-a") is not None


def test_a08_background_task_writes_the_shared_knowledge_store(tmp_path: Path) -> None:
    """A08（D-KB）：后台任务仍绑定创建时的**工作区身份**，但写的是同一份共享知识。"""

    store = deps.workspace_store
    data_dir = committed_version_dir(tmp_path)
    factory = WorkspaceContextFactory(data_dir, store)
    workspace_a = store.create_with_uuid("甲区")
    workspace_b = store.create_with_uuid("乙区")
    context_a = factory.resolve(workspace_a.id)
    context_b = factory.resolve(workspace_b.id)

    scope = BackgroundTaskScope.bind(context_a)
    assert scope.workspace_id == workspace_a.id
    assert context_b.workspace_id == workspace_b.id
    # 知识与同步状态共享 ⇒ 任务写进哪个「区」其实都写进同一个库
    assert context_b.state_db == scope.state_db
    assert context_b.documents_dir == scope.documents_dir

    from app.ingestion.state_store import DocumentStateStore

    task_store = DocumentStateStore(scope.state_db)
    task_store.upsert_fingerprint(str(scope.documents_dir / "notes.md"), "hash", 1.0, 3, 1)
    assert scope.state_db.is_file()
    # 乙区读同一个库 → 看得到甲区导入的指纹（共享知识库应有的语义）
    assert DocumentStateStore(context_b.state_db).load_all() != {}
    assert deps.workspace_state_store(context_a) is deps.workspace_state_store(context_b)
    task_store.close()


def test_workspace_private_files_stay_isolated(tmp_path: Path) -> None:
    """共享的是**知识**：记忆 / 检查点 / 工件 / 分词缓存仍按区隔离。"""

    store = WorkspaceStore(tmp_path)
    factory = WorkspaceContextFactory(tmp_path, store)
    context_a = factory.resolve(store.create("甲区").id)
    context_b = factory.resolve(store.create("乙区").id)

    for attribute in ("memory_json", "checkpoints_db", "artifacts_dir"):
        assert getattr(context_a, attribute) != getattr(context_b, attribute), attribute
    assert context_a.paths.jieba_cache_dir != context_b.paths.jieba_cache_dir

    memory_a = deps.workspace_memory(context_a)
    memory_b = deps.workspace_memory(context_b)
    assert memory_a is not memory_b
    assert memory_a.path != memory_b.path


def _png_bytes() -> bytes:
    """真实可解码的 PNG（原图端点会用 PIL 校验格式）。"""

    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def committed_version_dir(tmp_path: Path) -> Path:
    """返回一个已提交 ``storage_schema_version`` 的数据目录（工作区记录 authority=D29）。

    ``deps`` 的 provider 读 ``settings.data_dir``（conftest 已隔离到 tmp）；这里保持
    一致。D-KB 后该标记**不影响知识库布局**，只影响工作区记录写 JSON 还是 app.db。
    """

    from app.core.config import settings
    from app.core.workspace_context import commit_storage_version

    data_dir = Path(settings.data_dir)
    if committed_storage_version(data_dir) < WORKSPACE_LAYOUT_VERSION:
        commit_storage_version(data_dir)
    return data_dir


def test_vector_cache_key_includes_collection_identity(tmp_path: Path) -> None:
    """collection 身份变化必须换向量句柄（入库版本升级场景）。"""

    factory = WorkspaceContextFactory(tmp_path, WorkspaceStore(tmp_path))
    context = factory.resolve()
    first = context.vector_cache_key(provider_digest="p", collection_identity="chunks__m__v7")
    second = context.vector_cache_key(provider_digest="p", collection_identity="chunks__m__v6")
    assert first != second
    assert BM25_GENERATOR_VERSION in context.index_cache_key(
        index_fingerprint="f", generator_version=BM25_GENERATOR_VERSION
    )


def test_delete_knowledge_bumps_every_workspace_revision(tmp_path: Path) -> None:
    """D-KB：共享知识库被清空后，**所有**工作区的版本都要递增（否则邻区缓存不失效）。"""

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
    assert factory.resolve(other.id).knowledge_revision == before_other + 1
