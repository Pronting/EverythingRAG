"""IngestionPipeline 单测：计数 / block_id 全局唯一 / 幂等 / 空目录 / 单文件失败脱敏（任务 9）。

全部用轻量 fake（FakeEmbedder + FakeVectorStore 实现 VectorStore 协议），零出网、
不加载真实 Chroma / bge-m3。E2E 闭环（真实 Chroma + HTTP/SSE）见 test_e2e_closed_loop.py。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline, IngestReport
from app.models.schemas import BlockMetadata, SourceType

#: block_id = content_hash(16 位 hex) + "::" + chunk.block_id
_BLOCK_ID_RE = re.compile(r"^[0-9a-f]{16}::.+$")


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)) % 7.0, 0.0, 0.0, 0.0] for text in texts]


class FakeVectorStore:
    """记录型假向量库：只收 upsert，可查询 block_id / 计数。"""

    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, list[float], BlockMetadata]] = {}

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

    def query(self, vector: list[float], top_k: int, where: dict[str, str] | None = None) -> list[dict]:
        return []

    def count(self) -> int:
        return len(self.blocks)


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
    """单文件导入：计数正确，block_id 全局唯一（content_hash::锚点路径链）。"""
    root = tmp_path / "docs"
    root.mkdir()
    file = _write(root, "a.md", "# 标题一\n\n正文内容\n\n## 子标题\n\n子内容\n")

    pipeline, _, store = _pipeline()
    report = pipeline.ingest(root)

    assert report.files_scanned == 1
    assert report.files_parsed == 1
    assert report.files_skipped == 0
    assert report.chunks == 2  # 两个标题区间各一块
    assert report.blocks_upserted == 2
    assert report.errors == ()
    assert store.count() == 2

    # block_id 全局唯一：content_hash 前缀 + chunk.block_id
    assert len(store.blocks) == 2
    assert all(_BLOCK_ID_RE.fullmatch(block_id) for block_id in store.blocks)
    # 元数据组装正确（两块的 doc_id/source_file 相同，取任一块即可）
    first_meta = next(iter(store.blocks.values()))[2]
    assert first_meta.doc_id == _file_hash(file)
    assert first_meta.source_type == SourceType.DOCUMENT
    assert first_meta.source_file == str(file.resolve())
    assert first_meta.platform == "local"
    assert first_meta.chunk_type == "text"


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
