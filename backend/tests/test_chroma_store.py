"""ChromaVectorStore 适配层测试（fake embedder + tmp_path，零网络）。

隐私红线：全部用 FakeEmbedder，底层真实 bge-m3 从不加载/下载；
Chroma 客户端显式禁用匿名遥测（零出网），并由阻断 socket 的用例实证。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION
from app.models.schemas import BlockMetadata, SourceType
from app.vectorstore import chroma_store as chroma_store_mod
from app.vectorstore.chroma_store import ChromaVectorStore, _serialize_metadata, create_vector_store


class FakeEmbedder:
    """最小嵌入器替身：指纹 fake / dim 4，满足 Embedder 协议。"""

    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)


def _block(
    block_id: str,
    source_file: str,
    platform: str = "local",
    source_type: SourceType = SourceType.DOCUMENT,
) -> BlockMetadata:
    return BlockMetadata(
        block_id=block_id,
        doc_id=f"doc-{source_file}",
        source_type=source_type,
        source_file=source_file,
        platform=platform,
        created_time="2026-08-11T10:00:00Z",
        heading_path="s1/s2",
        anchor=f"anchor-{block_id}",
        chunk_type="text",
        message_role=None,
        message_seq=None,
        dedup_key=None,
    )


VECTORS = {
    "block-a": [1.0, 0.0, 0.0, 0.0],
    "block-b": [0.0, 1.0, 0.0, 0.0],
    "block-c": [0.0, 0.0, 1.0, 0.0],
}
QUERY = [1.0, 0.5, 0.25, 0.0]


def _store(persist_dir: Path) -> ChromaVectorStore:
    return create_vector_store(persist_dir=persist_dir, embedder=FakeEmbedder())


def _seed(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            ("block-a", "text a", VECTORS["block-a"], _block("block-a", "a.md")),
            (
                "block-b",
                "text b",
                VECTORS["block-b"],
                _block("block-b", "b.md", platform="kimi", source_type=SourceType.CONVERSATION),
            ),
            ("block-c", "text c", VECTORS["block-c"], _block("block-c", "a.md")),
        ]
    )


# ---------------------------------------------------------------- 1. upsert + query 往返


def test_upsert_and_query_roundtrip(tmp_path: Path) -> None:
    """upsert 3 块 -> query 返回 top-k：block_id/text/metadata/similarity 正确且按相似度降序。"""
    store = _store(tmp_path)
    _seed(store)

    hits = store.query(QUERY, top_k=3)
    assert [hit["block_id"] for hit in hits] == ["block-a", "block-b", "block-c"]
    assert hits[0]["text"] == "text a"
    assert hits[0]["metadata"]["source_file"] == "a.md"
    # 余弦空间：similarity = 1 - distance
    assert hits[0]["similarity"] + hits[0]["distance"] == pytest.approx(1.0)
    # 与 QUERY=[1,0.5,0.25,0] 的余弦：a≈0.873 / b≈0.436 / c≈0.218，严格降序
    assert hits[0]["similarity"] == pytest.approx(0.8729, abs=0.01)
    assert hits[0]["similarity"] > hits[1]["similarity"] > hits[2]["similarity"]
    assert all(0.0 <= hit["similarity"] <= 1.0 for hit in hits)


def test_bulk_upsert_repairs_logical_record_missing_from_ann(tmp_path: Path) -> None:
    """Chroma 批量写入若只落逻辑记录、漏 ANN 节点，应逐条补写后再成功返回。"""

    class FlakyCollection:
        def __init__(self) -> None:
            self.all_ids: set[str] = set()
            self.searchable: set[str] = set()
            self.vector_to_id: dict[tuple[float, ...], str] = {}
            self.single_upserts: list[str] = []

        def count(self) -> int:
            return len(self.all_ids)

        def upsert(
            self,
            *,
            ids: list[str],
            documents: list[str],
            embeddings: list[list[float]],
            metadatas: list[dict[str, object]],
        ) -> None:
            del documents, metadatas
            self.all_ids.update(ids)
            self.vector_to_id.update(
                {tuple(vector): block_id for block_id, vector in zip(ids, embeddings, strict=True)}
            )
            if len(ids) == 1:
                self.single_upserts.extend(ids)
                self.searchable.update(ids)
            else:
                self.searchable.update(block_id for block_id in ids if block_id != "block-b")

        def query(
            self,
            *,
            query_embeddings: list[list[float]],
            n_results: int,
            include: list[str],
        ) -> dict[str, list[list[str]]]:
            del n_results, include
            rows: list[list[str]] = []
            for vector in query_embeddings:
                expected = self.vector_to_id[tuple(vector)]
                rows.append([expected] if expected in self.searchable else [])
            return {"ids": rows}

    store = _store(tmp_path)
    flaky = FlakyCollection()
    store._collection = flaky  # type: ignore[assignment]

    _seed(store)

    assert flaky.single_upserts == ["block-b"]
    assert flaky.searchable == {"block-a", "block-b", "block-c"}


def test_full_ann_repair_fixes_record_evicted_by_a_later_batch(tmp_path: Path) -> None:
    """后续批次使早期节点不可达时，导入末尾的全库修复必须把它补回。"""

    class LateEvictionCollection:
        def __init__(self) -> None:
            self.ids = ["block-a", "block-b", "block-c"]
            self.documents = ["text a", "text b", "text c"]
            self.embeddings = [VECTORS[block_id] for block_id in self.ids]
            self.metadatas = [{"source_file": f"{block_id}.md"} for block_id in self.ids]
            self.vector_to_id = {
                tuple(vector): block_id
                for block_id, vector in zip(self.ids, self.embeddings, strict=True)
            }
            self.searchable = {"block-a", "block-c"}
            self.single_upserts: list[str] = []

        def get(self, *, include: list[str]) -> dict[str, object]:
            del include
            return {
                "ids": self.ids,
                "documents": self.documents,
                "embeddings": self.embeddings,
                "metadatas": self.metadatas,
            }

        def query(
            self,
            *,
            query_embeddings: list[list[float]],
            n_results: int,
            include: list[str],
        ) -> dict[str, list[list[str]]]:
            del n_results, include
            rows = []
            for vector in query_embeddings:
                expected = self.vector_to_id[tuple(vector)]
                rows.append([expected] if expected in self.searchable else [])
            return {"ids": rows}

        def upsert(
            self,
            *,
            ids: list[str],
            documents: list[str],
            embeddings: list[list[float]],
            metadatas: list[dict[str, object]],
        ) -> None:
            del documents, embeddings, metadatas
            self.single_upserts.extend(ids)
            self.searchable.update(ids)

    store = _store(tmp_path)
    flaky = LateEvictionCollection()
    store._collection = flaky  # type: ignore[assignment]

    assert store.repair_ann_index() == 1
    assert flaky.single_upserts == ["block-b"]
    assert flaky.searchable == {"block-a", "block-b", "block-c"}
    assert store.repair_ann_index() == 0


# ---------------------------------------------------------------- 2. collection 命名


def test_create_vector_store_requires_embedder(tmp_path: Path) -> None:
    """工厂无 embedder 时抛错（本地嵌入已移除，云端嵌入需显式配置）。"""
    with pytest.raises(ValueError):
        create_vector_store(persist_dir=tmp_path)


def test_collection_naming_custom_fingerprint(tmp_path: Path) -> None:
    """默认 collection 同时按 embedder 和入库结构版本隔离。"""
    store = _store(tmp_path)
    assert store._collection_name == f"chunks__fake__v{INGESTION_SCHEMA_VERSION}"


def test_collection_isolated_by_embed_model(tmp_path: Path) -> None:
    """不同嵌入模型指向不同 collection（切换嵌入模型 = 需重新导入）。"""
    from app.vectorstore.embedder import CloudEmbedder

    local = create_vector_store(persist_dir=tmp_path, embedder=FakeEmbedder())
    cloud = create_vector_store(
        persist_dir=tmp_path, embedder=CloudEmbedder(base_url="http://e.test/v1", model="bge-m3")
    )
    assert local._collection_name == f"chunks__fake__v{INGESTION_SCHEMA_VERSION}"
    assert cloud._collection_name == (
        f"chunks__{cloud._embedder.fingerprint}__v{INGESTION_SCHEMA_VERSION}"
    )
    assert local._collection_name != cloud._collection_name


# ---------------------------------------------------------------- 3. where 过滤


def test_query_where_filter(tmp_path: Path) -> None:
    """按 source_file / platform / source_type 过滤，仅返回匹配块。"""
    store = _store(tmp_path)
    _seed(store)

    by_platform = store.query(QUERY, top_k=3, where={"platform": "kimi"})
    assert [hit["block_id"] for hit in by_platform] == ["block-b"]

    by_type = store.query(QUERY, top_k=3, where={"source_type": "document"})
    assert {hit["block_id"] for hit in by_type} == {"block-a", "block-c"}

    by_file = store.query(QUERY, top_k=3, where={"source_file": "a.md"})
    assert {hit["block_id"] for hit in by_file} == {"block-a", "block-c"}


# ---------------------------------------------------------------- 4. delete


def test_delete_by_ids_and_where(tmp_path: Path) -> None:
    """delete_by_ids 删除单块；delete_by_where 按 source_file 级联删。"""
    store = _store(tmp_path)
    _seed(store)

    store.delete_by_ids(["block-b"])
    assert {hit["block_id"] for hit in store.query(QUERY, top_k=10)} == {"block-a", "block-c"}

    store.delete_by_where({"source_file": "a.md"})
    assert store.query(QUERY, top_k=10) == []
    assert store._collection.count() == 0


# ---------------------------------------------------------------- 4b. get_blocks_by_source


def test_get_blocks_by_source_returns_block_texts(tmp_path: Path) -> None:
    """get_blocks_by_source 按来源文件返回 {block_id: text}（增量块级比对用）。"""
    store = _store(tmp_path)
    _seed(store)  # a.md -> block-a/block-c；b.md -> block-b

    a_blocks = store.get_blocks_by_source("a.md")
    assert set(a_blocks) == {"block-a", "block-c"}
    assert a_blocks["block-a"] == "text a"
    assert a_blocks["block-c"] == "text c"

    b_blocks = store.get_blocks_by_source("b.md")
    assert set(b_blocks) == {"block-b"}

    # 未知来源 / 无块 -> 空 dict
    assert store.get_blocks_by_source("ghost.md") == {}


# ---------------------------------------------------------------- 5. 幂等 upsert


def test_upsert_idempotent(tmp_path: Path) -> None:
    """同 block_id 重复 upsert 新向量/文本 -> 更新生效且无重复条目。"""
    store = _store(tmp_path)
    _seed(store)
    total_before = store._collection.count()

    store.upsert([("block-a", "text a v2", [0.0, 0.0, 0.0, 1.0], _block("block-a", "a.md"))])
    assert store._collection.count() == total_before  # 总数不变，无重复

    hits = store.query([0.0, 0.0, 0.0, 1.0], top_k=1)
    assert hits[0]["block_id"] == "block-a"
    assert hits[0]["text"] == "text a v2"  # 更新生效


def test_upsert_rejects_wrong_dim(tmp_path: Path) -> None:
    """向量维度与 embedder.dim 不一致时快速失败。"""
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.upsert([("bad", "text", [1.0, 0.0], _block("bad", "x.md"))])


# ---------------------------------------------------------------- 6. 元数据序列化


def test_serialize_metadata() -> None:
    """BlockMetadata -> Chroma 元数据 dict：source_type 为 str、None 丢弃、标量齐全。"""
    meta = BlockMetadata(
        block_id="b1",
        doc_id="d1",
        source_type=SourceType.DOCUMENT,
        source_file="a.md",
        created_time=None,
        heading_path="s1",
        anchor=None,
        chunk_type="text",
        image_path=None,
        message_role=None,
        message_seq=None,
        dedup_key=None,
    )
    serialized = _serialize_metadata(meta)
    assert serialized["source_type"] == "document"  # StrEnum -> 普通 str
    assert isinstance(serialized["source_type"], str)
    assert serialized["block_id"] == "b1"
    assert serialized["heading_path"] == "s1"
    assert serialized["platform"] == "local"
    assert "created_time" not in serialized  # None 丢弃
    assert "anchor" not in serialized
    assert all(isinstance(value, (str, int, float, bool)) for value in serialized.values())


# ---------------------------------------------------------------- 8. 边界行为


def test_boundary_empty_query_and_top_k(tmp_path: Path) -> None:
    """空 collection 查询 -> []; top_k<=0 或空向量 -> []。"""
    store = _store(tmp_path)
    assert store.query(QUERY, top_k=5) == []  # 空 collection

    _seed(store)
    assert store.query(QUERY, top_k=0) == []
    assert store.query(QUERY, top_k=-1) == []
    assert store.query([], top_k=5) == []  # 空向量


# ---------------------------------------------------------------- 9. 零出网


def test_zero_outbound_full_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切 socket 连接，全链路（create/upsert/query/delete）仍成功 => 零出网。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    monkeypatch.setattr(socket.socket, "connect", _raise_outbound)

    store = _store(tmp_path)
    _seed(store)
    hits = store.query(QUERY, top_k=3)
    assert [hit["block_id"] for hit in hits] == ["block-a", "block-b", "block-c"]
    store.delete_by_ids(["block-a"])
    assert store._collection.count() == 2


def test_two_store_instances_serialize_parallel_writes(tmp_path: Path) -> None:
    """同数据目录的两个 client 并发写时由文件锁串行化，块不丢失。"""

    left = _store(tmp_path)
    right = _store(tmp_path)

    def write(store: ChromaVectorStore, prefix: str) -> None:
        for index in range(10):
            block_id = f"{prefix}-{index}"
            store.upsert(
                [
                    (
                        block_id,
                        f"text {block_id}",
                        [1.0, float(index), 0.0, 0.0],
                        _block(block_id, f"{prefix}.md"),
                    )
                ]
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write, left, "left"),
            executor.submit(write, right, "right"),
        ]
        for future in futures:
            future.result()

    assert left.count() == 20
    assert right.count() == 20


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")


# ---------------------------------------------------------------- 10. 文件级计数


def test_count_files_distinct_source(tmp_path: Path) -> None:
    """count_files 返回去重后的 source_file 数（同一文件多块只算 1）。"""
    store = _store(tmp_path)
    _seed(store)  # a.md x2 块、b.md x1 块
    assert store.count() == 3
    assert store.count_files() == 2


def test_count_files_empty(tmp_path: Path) -> None:
    """空库 count_files = 0。"""
    store = _store(tmp_path)
    assert store.count_files() == 0


# ---------------------------------------------------------------- 4c. list_blocks（BM25 标题注入用）


def test_list_blocks_enumerates_all_with_text_and_metadata(tmp_path: Path) -> None:
    """list_blocks 返回全部块 (block_id, text, metadata)，供混合检索建 BM25 索引。"""
    store = _store(tmp_path)
    _seed(store)  # block-a/block-c（a.md）+ block-b（b.md）

    blocks = store.list_blocks()
    assert {block[0] for block in blocks} == {"block-a", "block-b", "block-c"}
    assert {block[1] for block in blocks} == {"text a", "text b", "text c"}
    meta = {block[0]: block[2] for block in blocks}
    assert meta["block-a"]["source_file"] == "a.md"
    assert meta["block-a"]["heading_path"] == "s1/s2"
    assert meta["block-b"]["platform"] == "kimi"


def test_list_blocks_empty_store(tmp_path: Path) -> None:
    """空库 list_blocks 返回空列表。"""
    store = _store(tmp_path)
    assert store.list_blocks() == []


def test_recovers_raw_image_description_from_retained_collection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    legacy = store._client.get_or_create_collection(
        name="chunks__legacy__v1",
        metadata={"hnsw:space": "cosine"},
    )
    url = "https://cdn.example.com/expired.png"
    legacy.upsert(
        ids=["legacy-image"],
        documents=[
            (
                "【所属主题】事故影响\n【正文上下文】20 万次请求\n"
                "【图片描述】\n监控截图显示 503 请求集中爆发。"
            )
        ],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        metadatas=[
            {
                "block_id": "legacy-image",
                "chunk_type": "image_description",
                "image_path": url,
                "image_content_hash": "old-hash",
                "source_file": "事故.md",
            }
        ],
    )

    assert store.find_cached_image_description_by_source(url) == (
        "old-hash",
        "监控截图显示 503 请求集中爆发。",
    )


def test_integrity_report_validates_embeddings_and_ann_without_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)

    report = store.inspect_integrity()

    assert report.healthy is True
    assert report.logical_count == report.unique_id_count == report.embedding_count == 3
    assert report.ann_result_count == report.ann_unique_id_count == 3
    assert report.invalid_dimension_count == 0
    assert report.non_finite_embedding_count == 0
    assert "text a" not in str(report.to_dict())


def test_empty_index_is_not_a_healthy_release_artifact(tmp_path: Path) -> None:
    report = _store(tmp_path).inspect_integrity()

    assert report.healthy is False
    assert report.logical_count == 0
    assert report.errors == ("EmptyIndex",)


# ---------------------------------------------------------------- 11. collection 失效自愈（崩溃根因回归）


def test_read_ops_tolerate_deleted_collection(tmp_path: Path) -> None:
    """collection 被外部删除/重建后，读操作不抛 NotFoundError、自愈为空库。

    崩溃根因回归：持久库被并发重建（切换嵌入模型 / 其他进程访问同一 persist_dir）
    时，旧 collection 句柄失效。count/count_files/query/get_blocks_by_source 必须
    自愈为「空库」，否则 /api/status 等入口直接 500 甚至拖垮服务进程。
    """
    store = _store(tmp_path)
    _seed(store)
    assert store.count_files() == 2

    store._client.delete_collection(store._collection_name)

    assert store.count() == 0
    assert store.count_files() == 0
    assert store.query(QUERY, top_k=3) == []
    assert store.get_blocks_by_source("a.md") == {}


def test_upsert_self_heals_after_collection_deleted(tmp_path: Path) -> None:
    """collection 被删后 upsert 自愈重建，随后可正常写入与查询。"""
    store = _store(tmp_path)
    _seed(store)
    store._client.delete_collection(store._collection_name)

    store.upsert([("block-a", "text a", VECTORS["block-a"], _block("block-a", "a.md"))])

    assert store.count() == 1
    assert store.count_files() == 1
    assert [hit["block_id"] for hit in store.query(QUERY, top_k=3)] == ["block-a"]


def test_read_self_heal_recreates_collection_under_cross_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读路径 NotFound 的 get_or_create 必须在跨进程写锁范围内执行。"""

    store = _store(tmp_path)
    store._client.delete_collection(store._collection_name)
    active_locks = 0
    lock_entries = 0
    original_get_or_create = store._client.get_or_create_collection

    @contextmanager
    def tracking_lock(_path: Path) -> Iterator[None]:
        nonlocal active_locks, lock_entries
        if active_locks:
            raise AssertionError("exclusive_write_lock must not be nested")
        active_locks += 1
        lock_entries += 1
        try:
            yield
        finally:
            active_locks -= 1

    def guarded_get_or_create(*args: Any, **kwargs: Any) -> Any:
        assert active_locks == 1
        return original_get_or_create(*args, **kwargs)

    monkeypatch.setattr(chroma_store_mod, "exclusive_write_lock", tracking_lock)
    monkeypatch.setattr(store._client, "get_or_create_collection", guarded_get_or_create)

    assert store.count() == 0
    assert lock_entries == 1


def test_write_self_heal_reuses_existing_lock_without_nested_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写路径已持锁时，自愈复用该锁而不是再次获取文件锁。"""

    store = _store(tmp_path)
    store._client.delete_collection(store._collection_name)
    active_locks = 0
    lock_entries = 0
    original_get_or_create = store._client.get_or_create_collection

    @contextmanager
    def non_reentrant_lock(_path: Path) -> Iterator[None]:
        nonlocal active_locks, lock_entries
        if active_locks:
            raise AssertionError("exclusive_write_lock must not be nested")
        active_locks += 1
        lock_entries += 1
        try:
            yield
        finally:
            active_locks -= 1

    def guarded_get_or_create(*args: Any, **kwargs: Any) -> Any:
        assert active_locks == 1
        return original_get_or_create(*args, **kwargs)

    monkeypatch.setattr(chroma_store_mod, "exclusive_write_lock", non_reentrant_lock)
    monkeypatch.setattr(store._client, "get_or_create_collection", guarded_get_or_create)

    store.upsert([("block-a", "text a", VECTORS["block-a"], _block("block-a", "a.md"))])

    assert store.count() == 1
    assert lock_entries == 1
