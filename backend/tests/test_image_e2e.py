"""E2E 图片闭环：真实 Chroma + 管线 + 识图（fake fetcher/vision）→ 描述块可被文本检索。

验证方案 B 核心：图片 → 文字描述 → 与文本同一嵌入模型 → 同一向量空间，
纯文本问题可命中图片描述块。零出网（fetcher 用内存 PNG，vision 用 fake）。
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore


def _png_bytes(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeEmbedder:
    """确定性嵌入：字符哈希到定长向量 + L2 归一化（余弦即字符重叠度量）。"""

    fingerprint = "fake-img-e2e"
    dim = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for ch in text:
                if not ch.strip():
                    continue
                vec[ord(ch) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeVision:
    def __init__(self, description: str) -> None:
        self._description = description
        self.calls = 0

    def describe(self, image_bytes: bytes, prompt: str = "", mime: str = "image/jpeg") -> str:
        self.calls += 1
        return self._description


class FakeFetcher:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def fetch(self, url: str) -> bytes:
        return self._data


def test_image_description_retrievable_by_text(tmp_path: Path) -> None:
    """含图文档导入 -> 文本块 + 图片描述块入 Chroma，描述可被纯文本查询命中。"""
    root = tmp_path / "docs"
    root.mkdir()
    url = "https://cdn.example.com/arch.png"
    (root / "arch.md").write_text(
        f"# 架构\n\n这是系统架构的正文说明。\n\n![]({url})",
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-img__v1",
        embedder=embedder,
    )
    vision = FakeVision("【画面主体】架构图：系统分三层，网关、服务与存储。")
    pipeline = IngestionPipeline(embedder=embedder, vectorstore=store, vision=vision, fetcher=FakeFetcher(_png_bytes()))
    report = pipeline.ingest(root)

    # 文本块 + 图片描述块都入库
    assert report.files_parsed == 1
    assert report.blocks_upserted == 2
    assert store.count() == 2

    # 图片描述块元数据完整（type / 原图 URL / 内容哈希）
    image_blocks = [b for b in store.list_blocks() if b[2].get("source_type") == "image_description"]
    assert len(image_blocks) == 1
    _, text, meta = image_blocks[0]
    assert meta["chunk_type"] == "image_description"
    assert meta["image_path"] == url
    assert meta["image_content_hash"]
    assert "架构图" in text

    # 检索：纯文本问题「架构图」能命中图片描述块（方案 B 的核心验证）
    retriever = VectorRetriever(store)
    hits = retriever.retrieve("架构图", embedder.embed_texts(["架构图"])[0])
    assert hits
    assert any("架构图" in hit.text for hit in hits)


def test_vision_unconfigured_indexes_text_only(tmp_path: Path) -> None:
    """未配识图 -> 图片不产块，文本照常入库（不因缺识图而失败）。"""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text(
        "# 正文\n\n这是正文内容段落。\n\n![](https://cdn.example.com/a.png)",
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-img__v1",
        embedder=embedder,
    )
    pipeline = IngestionPipeline(embedder=embedder, vectorstore=store)  # 无 vision/fetcher
    report = pipeline.ingest(root)

    assert report.blocks_upserted == 1
    assert store.count() == 1
    assert store.count_images() == 0
