"""ChromaVectorStore 适配层测试（fake embedder + tmp_path，零网络）。

隐私红线：全部用 FakeEmbedder，底层真实 bge-m3 从不加载/下载；
Chroma 客户端显式禁用匿名遥测（零出网），并由阻断 socket 的用例实证。
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app.models.schemas import BlockMetadata, SourceType
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


# ---------------------------------------------------------------- 2. collection 命名


def test_collection_naming_default(tmp_path: Path) -> None:
    """工厂默认 collection 名精确等于 chunks__bge-m3__v1（真模型指纹）。"""
    store = create_vector_store(persist_dir=tmp_path)  # 默认 embedder = FastEmbedEmbedder（惰性，不下载）
    assert store._collection_name == "chunks__bge-m3__v1"
    assert store._collection.name == "chunks__bge-m3__v1"


def test_collection_naming_custom_fingerprint(tmp_path: Path) -> None:
    """自定义 embedder 时 collection 名 = chunks__{fingerprint}__v1。"""
    store = _store(tmp_path)
    assert store._collection_name == "chunks__fake__v1"


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
