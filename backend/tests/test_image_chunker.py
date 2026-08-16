"""切块器图片块单测：图片独立成块、带 src、继承标题上下文、不被短块合并吞掉。"""

from __future__ import annotations

from app.ingestion.chunker import chunk_document
from app.ingestion.markdown_parser import parse_markdown


def _chunks(md: str):
    return chunk_document(parse_markdown(md), source_file="doc.md")


def test_image_emits_separate_chunk_with_src() -> None:
    """纯图段落 -> 独立 image Chunk，带 src 且继承标题上下文。"""
    chunks = _chunks("## 架构\n\n![](https://cdn.example.com/a.png)")
    images = [c for c in chunks if c.kind == "image"]
    assert [(c.src, c.heading_path) for c in images] == [
        ("https://cdn.example.com/a.png", "架构")
    ]


def test_image_alt_not_merged_into_text() -> None:
    """图片 alt 不并入正文文本块（alt 噪声不进检索文本）。"""
    chunks = _chunks("## 图\n\n正文段落内容\n\n![架构图](https://cdn.example.com/a.png)")
    texts = " ".join(c.text for c in chunks if c.kind == "text")
    assert "架构图" not in texts


def test_bare_image_not_dropped_by_short_coalesce() -> None:
    """空 alt 的裸图不被短块合并吞掉，仍独立成块并保留 src。"""
    chunks = _chunks("![](https://cdn.example.com/a.png)")
    images = [c for c in chunks if c.kind == "image"]
    assert len(images) == 1
    assert images[0].src == "https://cdn.example.com/a.png"


def test_multiple_images_keep_order_and_unique_ids() -> None:
    """一行多图 -> 多个 image 块，src 顺序正确、block_id 稳定且唯一。"""
    chunks = _chunks("![](https://cdn.example.com/a.png)![](https://cdn.example.com/b.png)")
    images = [c for c in chunks if c.kind == "image"]
    assert [c.src for c in images] == [
        "https://cdn.example.com/a.png",
        "https://cdn.example.com/b.png",
    ]
    ids = [c.block_id for c in images]
    assert len(ids) == len(set(ids))


def test_image_in_headingless_doc_gets_seq_id() -> None:
    """无标题文档里的图片 -> block_id 用文档级序号，稳定唯一。"""
    chunks = _chunks("前置段落\n\n![](https://cdn.example.com/a.png)")
    images = [c for c in chunks if c.kind == "image"]
    assert images[0].block_id.startswith("#")
    assert images[0].src == "https://cdn.example.com/a.png"
