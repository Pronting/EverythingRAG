"""IngestionPipeline 单测：计数 / block_id 全局唯一 / 幂等 / 空目录 / 单文件失败脱敏（任务 9）。

全部用轻量 fake（FakeEmbedder + FakeVectorStore 实现 VectorStore 协议），零出网、
不加载真实 Chroma / bge-m3。E2E 闭环（真实 Chroma + HTTP/SSE）见 test_e2e_closed_loop.py。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.chunker import Chunk
from app.ingestion.index_schema import make_index_fingerprint
from app.ingestion.pipeline import (
    IngestionPipeline,
    IngestReport,
    chunk_embedding_text,
    make_block_id,
)
from app.ingestion.state_store import DocumentStateStore
from app.models.schemas import BlockMetadata, SourceType

#: block_id = source_file 摘要(32 位 hex) + content_hash(16 位 hex) + chunk.block_id
_BLOCK_ID_RE = re.compile(r"^[0-9a-f]{32}::[0-9a-f]{16}::.+$")


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)) % 7.0, 0.0, 0.0, 0.0] for text in texts]


class FakeVectorStore:
    """记录型假向量库：只收 upsert，可查询 block_id / 计数。"""

    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, list[float], BlockMetadata]] = {}
        self.repair_calls = 0

    def upsert(self, blocks: list[tuple[str, str, list[float], BlockMetadata]]) -> None:
        for block_id, text, vector, meta in blocks:
            self.blocks[block_id] = (text, vector, meta)

    def delete_by_ids(self, block_ids: list[str]) -> None:
        for block_id in block_ids:
            self.blocks.pop(block_id, None)

    def delete_by_where(self, where: dict[str, str]) -> None:
        for block_id, (_, _, meta) in list(self.blocks.items()):
            if where == {"source_file": meta.source_file}:
                self.blocks.pop(block_id, None)

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {
            block_id: text
            for block_id, (text, _, meta) in self.blocks.items()
            if meta.source_file == source_file
        }

    def query(
        self, vector: list[float], top_k: int, where: dict[str, str] | None = None
    ) -> list[dict]:
        return []

    def count(self) -> int:
        return len(self.blocks)

    def repair_ann_index(self) -> int:
        self.repair_calls += 1
        return 0


class FailOnSentinelEmbedder:
    """遇特定正文触发异常，用于验证单文件失败聚合脱敏。"""

    fingerprint = "fake"
    dim = 4

    def __init__(self, sentinel: str) -> None:
        self._sentinel = sentinel

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if any(self._sentinel in text for text in texts):
            raise RuntimeError(f"boom {self._sentinel}")
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _pipeline() -> tuple[IngestionPipeline, FakeEmbedder, FakeVectorStore]:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    return IngestionPipeline(embedder=embedder, vectorstore=store), embedder, store


def test_chunk_embedding_text_adds_retrieval_context_without_changing_chunk() -> None:
    """向量输入携带文档/标题语义，展示正文对象保持原样。"""

    chunk = Chunk(
        block_id="/H2/基础指标#1",
        text="UV；转化率；GMV",
        source_file=r"C:\notes\AB实验.md",
        heading_path="指标 > 基础指标",
        anchor="基础指标",
        seq=1,
        kind="table",
    )

    assert chunk_embedding_text(chunk) == (
        "文档：AB实验\n标题：指标 > 基础指标\n内容：UV；转化率；GMV"
    )
    assert chunk.text == "UV；转化率；GMV"


# ---------------------------------------------------------------- 1. 空目录 / 无 MD


def test_empty_dir_returns_zero_report(tmp_path: Path) -> None:
    """空目录 -> 全 0 报告，不抛错。"""
    root = tmp_path / "empty"
    root.mkdir()
    pipeline, _, _ = _pipeline()
    report = pipeline.ingest(root)
    assert report == IngestReport(0, 0, 0, 0, 0, ())


def test_dir_without_md_returns_zero_report(tmp_path: Path) -> None:
    """目录内无 .md -> 全 0 报告。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "note.txt", "不是 markdown")
    pipeline, _, _ = _pipeline()
    report = pipeline.ingest(root)
    assert report.files_scanned == 0
    assert report.blocks_upserted == 0


# ---------------------------------------------------------------- 2. 正常导入 + block_id 唯一


def test_ingest_counts_and_block_ids(tmp_path: Path) -> None:
    """单文件导入：计数正确，block_id 全局唯一（source::content::锚点路径链）。"""
    root = tmp_path / "docs"
    root.mkdir()
    file = _write(
        root, "a.md", "# 标题一\n\n" + "正文内容" * 40 + "\n\n## 子标题\n\n" + "子内容" * 40 + "\n"
    )

    pipeline, _, store = _pipeline()
    report = pipeline.ingest(root)

    assert report.files_scanned == 1
    assert report.files_parsed == 1
    assert report.files_skipped == 0
    assert report.chunks == 2  # 两个标题区间各一块
    assert report.blocks_upserted == 2
    assert report.errors == ()
    assert store.count() == 2
    assert store.repair_calls == 1

    # block_id 全局唯一：source_file 摘要 + content_hash + chunk.block_id
    assert len(store.blocks) == 2
    assert all(_BLOCK_ID_RE.fullmatch(block_id) for block_id in store.blocks)
    # 元数据组装正确（两块的 doc_id/source_file 相同，取任一块即可）
    first_meta = next(iter(store.blocks.values()))[2]
    assert first_meta.doc_id == _file_hash(file)
    assert first_meta.source_type == SourceType.DOCUMENT
    assert first_meta.source_file == str(file.resolve())
    assert first_meta.platform == "local"
    assert first_meta.chunk_type == "text"


def test_full_ingest_batches_embeddings_across_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# A\n\n内容 A\n")
    _write(root, "b.md", "# B\n\n内容 B\n")
    pipeline, embedder, store = _pipeline()

    report = pipeline.ingest(root)

    assert report.files_parsed == 2
    assert store.count() == 2
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 2


def test_block_id_is_source_aware_stable_and_path_private(tmp_path: Path) -> None:
    """相同内容/锚点的不同来源不碰撞；同一规范化来源重复计算稳定且不泄露路径。"""

    first_source = tmp_path / "team-a" / "same.md"
    second_source = tmp_path / "team-b" / "same.md"
    content_hash = "0123456789abcdef"
    anchor = "/H1/same#1"

    first = make_block_id(content_hash, anchor, source_file=first_source)
    first_again = make_block_id(
        content_hash,
        anchor,
        source_file=first_source.parent / "." / first_source.name,
    )
    second = make_block_id(content_hash, anchor, source_file=second_source)

    assert first == first_again
    assert first != second
    assert _BLOCK_ID_RE.fullmatch(first)
    assert str(first_source.resolve()) not in first
    assert first_source.name not in first


def test_identical_markdown_files_are_both_retained(tmp_path: Path) -> None:
    """两个路径下完全相同的 Markdown 均保留，不能因 content_hash/锚点相同而覆盖。"""

    root = tmp_path / "docs"
    (root / "team-a").mkdir(parents=True)
    (root / "team-b").mkdir(parents=True)
    content = "# 同一标题\n\n完全相同的正文内容。\n"
    first_path = _write(root / "team-a", "same.md", content)
    second_path = _write(root / "team-b", "same.md", content)

    pipeline, _, store = _pipeline()
    report = pipeline.ingest(root)

    assert report.files_parsed == 2
    assert report.blocks_upserted == 2
    assert store.count() == 2
    assert {meta.source_file for _, _, meta in store.blocks.values()} == {
        str(first_path.resolve()),
        str(second_path.resolve()),
    }
    assert len(set(store.blocks)) == 2


def _file_hash(path: Path) -> str:
    import xxhash

    return xxhash.xxh64(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 3. 幂等重导


def test_reingest_idempotent(tmp_path: Path) -> None:
    """同目录重复导入：block_id 相同 -> upsert 幂等，总数不变。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# 标题\n\n正文内容\n")

    pipeline, _, store = _pipeline()
    first = pipeline.ingest(root)
    second = pipeline.ingest(root)

    assert first.blocks_upserted == second.blocks_upserted == 1
    assert store.count() == 1  # 不重复
    assert second.files_skipped == 0


def test_full_import_persists_document_state_for_future_migrations(tmp_path: Path) -> None:
    """路径式 full 导入也双写 state，后续模型/切片升级才能可靠判断重建。"""

    root = tmp_path / "docs"
    root.mkdir()
    path = _write(root, "a.md", "# 标题\n\n正文内容\n")
    embedder = FakeEmbedder()
    vectorstore = FakeVectorStore()
    state = DocumentStateStore(tmp_path / "state.db")
    pipeline = IngestionPipeline(
        embedder=embedder,
        vectorstore=vectorstore,
        document_state_store=state,
    )

    report = pipeline.ingest(root)

    assert report.files_parsed == 1
    record = state.load_all()[str(path.resolve())]
    assert record.content_hash == _file_hash(path)
    assert record.index_fingerprint == make_index_fingerprint(embedder.fingerprint)
    assert state.list_sources() == [root.resolve()]
    state.close()


# ---------------------------------------------------------------- 5. 编码容错


def test_non_utf8_file_decodes_via_charset_normalizer(tmp_path: Path) -> None:
    """GBK 编码文件：UTF-8 严格解码失败 -> charset-normalizer 兜底，正常入库。"""
    root = tmp_path / "docs"
    root.mkdir()
    gbk_bytes = "# 标题\n\n中文正文内容。\n".encode("gbk")
    file = root / "gbk.md"
    file.write_bytes(gbk_bytes)

    pipeline, _, store = _pipeline()
    report = pipeline.ingest(root)

    assert report.files_parsed == 1
    assert report.files_skipped == 0
    assert report.chunks == 1
    assert store.count() == 1
    (text,) = [text for text, _, _ in store.blocks.values()]
    assert "中文" in text


# ---------------------------------------------------------------- 6. 单文件失败聚合脱敏


def test_single_file_failure_skips_and_sanitizes(tmp_path: Path) -> None:
    """单文件失败 -> files_skipped 计数 + 脱敏 errors（不含路径/正文），不中断其余文件。"""
    root = tmp_path / "docs"
    root.mkdir()
    sentinel = "无法嵌入的正文片段"
    _write(root, "bad.md", f"# 坏文件\n\n{sentinel}\n")
    good = _write(root, "good.md", "# 好文件\n\n正常正文\n")

    embedder = FailOnSentinelEmbedder(sentinel)
    store = FakeVectorStore()
    pipeline = IngestionPipeline(embedder=embedder, vectorstore=store)
    report = pipeline.ingest(root)

    assert report.files_scanned == 2
    assert report.files_parsed == 1  # 只有 good.md 成功
    assert report.files_skipped == 1
    assert report.chunks == 1
    assert report.blocks_upserted == 1
    assert store.count() == 1

    # 脱敏：errors 只含异常类型名，不含路径 / 文件名 / 正文
    assert len(report.errors) == 1
    assert report.errors == ("RuntimeError",)
    for reason in report.errors:
        assert str(root) not in reason
        assert "bad.md" not in reason
        assert sentinel not in reason

    # good.md 的块正常入库
    (block_id,) = store.blocks
    assert store.blocks[block_id][2].source_file == str(good.resolve())


# ---------------------------------------------------------------- 7. 进度回调


def test_ingest_progress_callback_increments(tmp_path: Path) -> None:
    """ingest(on_progress=...) 逐文件回调：快照递增，终值等于报告计数。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# A\n\n内容 A\n")
    _write(root, "b.md", "# B\n\n内容 B\n")

    snapshots: list[object] = []
    pipeline, _, _ = _pipeline()
    report = pipeline.ingest(root, on_progress=snapshots.append)

    # 每成功/失败处理一个文件回调一次
    assert len(snapshots) == 2
    first = snapshots[0]
    last = snapshots[-1]
    # files_scanned 在扫描阶段已确定
    assert first.files_scanned == 2
    # 首个文件处理后 parsed=1
    assert first.files_parsed == 1
    # 终值等于报告
    assert last.files_parsed == report.files_parsed
    assert last.files_skipped == report.files_skipped
    assert last.chunks == report.chunks


def test_ingest_progress_default_none_unchanged(tmp_path: Path) -> None:
    """on_progress 缺省 None：行为不变、零回调（向后兼容）。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# A\n\n内容 A\n")

    pipeline, _, _ = _pipeline()
    report = pipeline.ingest(root)  # 不传 on_progress
    assert report.files_parsed == 1
    assert report.chunks == 1


def test_semantic_split_splits_at_topic_boundary() -> None:
    """语义切块：多段落长块在话题边界（相邻段相似度骤降）处切分，短块/单段不动。"""
    from app.ingestion.chunker import Chunk

    class TopicEmbedder:
        fingerprint = "fake"
        dim = 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            # 主题 A（含"苹果"）→ [1,0,0,0]；主题 B（含"汽车"）→ [0,1,0,0]
            return [[1.0, 0.0, 0.0, 0.0] if "苹果" in t else [0.0, 1.0, 0.0, 0.0] for t in texts]

    from app.vectorstore.base import VectorStore  # noqa: F401 -- 类型占位，实际用 FakeVectorStore

    pipeline = IngestionPipeline(TopicEmbedder(), FakeVectorStore())
    apple = "苹果是一种富含维生素的水果。" * 20
    car = "汽车是一种需要加油的交通工具。" * 20
    chunk = Chunk(
        block_id="#1",
        text=f"{apple}\n\n{apple}\n\n{car}",
        source_file="f.md",
        heading_path="",
        anchor="",
        seq=1,
    )
    result = pipeline.semantic_split([chunk])
    assert len(result) == 2  # 苹果×2 与 汽车×1 之间是话题边界
    assert "苹果" in result[0].text and "汽车" not in result[0].text
    assert "汽车" in result[1].text and "苹果" not in result[1].text
    assert result[0].block_id == "#1-s0"
    assert result[1].block_id == "#1-s1"

    # 单段/短块不切
    short = Chunk(
        block_id="#2", text="短内容", source_file="f.md", heading_path="B", anchor="b", seq=2
    )
    assert pipeline.semantic_split([short]) == [short]


def test_semantic_split_batches_paragraphs_from_multiple_long_chunks() -> None:
    """同一文档的多个长块应合并请求，避免按块产生大量网络往返。"""

    class BatchTopicEmbedder:
        fingerprint = "fake"
        dim = 2

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [
                [1.0, 0.0] if "主题甲" in text else [0.0, 1.0]
                for text in texts
            ]

    embedder = BatchTopicEmbedder()
    pipeline = IngestionPipeline(embedder, FakeVectorStore())
    chunks = [
        Chunk(
            block_id=f"#{index}",
            text=("主题甲内容。" * 70) + "\n\n" + ("主题乙内容。" * 70),
            source_file="f.md",
            heading_path="",
            anchor="",
            seq=index,
        )
        for index in (1, 2)
    ]

    result = pipeline.semantic_split(chunks)

    assert len(result) == 4
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 4


def test_semantic_split_does_not_reembed_titled_sections() -> None:
    """标题已提供稳定语义边界时不做二次云端段落比较。"""

    class NoCallEmbedder:
        fingerprint = "fake"
        dim = 2

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError(f"titled chunk unexpectedly embedded: {len(texts)}")

    chunk = Chunk(
        block_id="#1",
        text=("主题甲内容。" * 70) + "\n\n" + ("主题乙内容。" * 70),
        source_file="f.md",
        heading_path="已有标题",
        anchor="已有标题",
        seq=1,
    )

    assert IngestionPipeline(NoCallEmbedder(), FakeVectorStore()).semantic_split([chunk]) == [
        chunk
    ]
