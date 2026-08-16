"""解析器图片引用提取单测：src 捕获 / 一行多图 / 混合图文拆分 / 语雀宽度语法。

纯内存字符串操作，零出网。聚焦方案 B 入库前的「图床链接」识别。
"""

from __future__ import annotations

from app.ingestion.markdown_parser import parse_markdown


def _image_blocks(md: str):
    return [b for b in parse_markdown(md).blocks if b.kind == "image"]


def test_bare_image_carries_src() -> None:
    """无 alt 的裸图床图 -> image 块 text 空、src 捕获。"""
    blocks = _image_blocks("![](https://cdn.nlark.com/yuque/0/2025/png/1.png)")
    assert [(b.text, b.src) for b in blocks] == [("", "https://cdn.nlark.com/yuque/0/2025/png/1.png")]


def test_image_with_alt_keeps_alt_and_src() -> None:
    """带 alt -> alt 存 text、src 存 src。"""
    blocks = _image_blocks("![架构图](https://cdn.example.com/a.png)")
    assert [(b.text, b.src) for b in blocks] == [("架构图", "https://cdn.example.com/a.png")]


def test_multiple_images_one_line_split() -> None:
    """一行多图逐个成块，各带各自 src（对应语料里的 ![](a)![](b)）。"""
    md = "![](https://cdn.example.com/a.png)![](https://cdn.example.com/b.png)"
    blocks = _image_blocks(md)
    assert [b.src for b in blocks] == [
        "https://cdn.example.com/a.png",
        "https://cdn.example.com/b.png",
    ]


def test_mixed_text_image_split() -> None:
    """混合图文：文字成 paragraph，图片成 image（alt 不再混入正文文本块）。"""
    parsed = parse_markdown("第一张：![架构图](https://cdn.example.com/a.png)")
    images = [b for b in parsed.blocks if b.kind == "image"]
    paragraphs = [b.text for b in parsed.blocks if b.kind == "paragraph"]
    assert [(b.text, b.src) for b in images] == [("架构图", "https://cdn.example.com/a.png")]
    assert paragraphs == ["第一张："]


def test_yuque_width_spec_alt_is_noise_but_src_kept() -> None:
    """语雀宽度语法 ![|375](url) -> alt 是 |375 噪声，src 仍被捕获。"""
    blocks = _image_blocks("![|375](https://cdn.nlark.com/yuque/0/2025/jpeg/1.jpeg)")
    assert blocks[0].src == "https://cdn.nlark.com/yuque/0/2025/jpeg/1.jpeg"
    assert blocks[0].text == "|375"


def test_relative_path_src_kept() -> None:
    """本地相对路径 src 原样保留（本地图片场景，不发起网络）。"""
    blocks = _image_blocks("![note](./assets/arch.png)")
    assert blocks[0].src == "./assets/arch.png"
