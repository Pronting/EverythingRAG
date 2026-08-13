"""ingestion MDBlock 测试：块级结构 + 锚点一致性 + 派生关系（MVP 任务 3）。

验收断言对应任务规格「交付 A」：MDBlock 各 kind 分支正确、heading blocks
与 heading_tree 锚点完全一致、normalized_text 由 blocks 派生保持一致。
全部为纯内存字符串操作，不产生文件读写与出网。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest

from app.ingestion import HeadingNode, MDBlock, parse_markdown


def _flatten(nodes: tuple[HeadingNode, ...]) -> Iterator[HeadingNode]:
    """按文档顺序展平标题树。"""
    for node in nodes:
        yield node
        yield from _flatten(node.children)


# ---------------------------------------------------------------- 1. kind 分支


def test_blocks_heading() -> None:
    """标题块：kind/level/anchor 正确，text 为标题归一化文本。"""
    parsed = parse_markdown("# 标题 A\n\nbody")
    (heading,) = [b for b in parsed.blocks if b.kind == "heading"]
    assert heading.text == "标题 A"
    assert heading.level == 1
    assert heading.anchor == "标题-a"


def test_blocks_paragraph() -> None:
    """段落块：顶层 inline 文本。"""
    parsed = parse_markdown("body 内容")
    assert [b.kind for b in parsed.blocks] == ["paragraph"]
    assert parsed.blocks[0].text == "body 内容"


def test_blocks_code_fence_and_indented() -> None:
    """代码块：围栏与缩进代码均独立成块，text 为代码内容。"""
    parsed = parse_markdown('```python\nprint("hi")\n```\n\n缩进块：\n\n    indent line\n')
    code = [b for b in parsed.blocks if b.kind == "code"]
    assert [b.text for b in code] == ['print("hi")', "indent line"]


def test_blocks_table() -> None:
    """表格块：每行一个块，行内单元格以 | 连接（行级粒度，避免单元格碎片化）。"""
    parsed = parse_markdown("| a | b |\n|---|---|\n| 1 | 2 |")
    tables = [b for b in parsed.blocks if b.kind == "table"]
    assert [b.text for b in tables] == ["a | b", "1 | 2"]


def test_blocks_list() -> None:
    """列表块：每个列表项 inline 为独立块，kind 为 list。"""
    parsed = parse_markdown("- 第一项\n- 第二项")
    lists = [b for b in parsed.blocks if b.kind == "list"]
    assert [b.text for b in lists] == ["第一项", "第二项"]


def test_blocks_quote() -> None:
    """引用块：blockquote 内 inline 文本，kind 为 quote。"""
    parsed = parse_markdown("> 引用内容")
    quotes = [b for b in parsed.blocks if b.kind == "quote"]
    assert [b.text for b in quotes] == ["引用内容"]


def test_blocks_hr() -> None:
    """分隔线块：无内容文本。"""
    parsed = parse_markdown("---")
    hrs = [b for b in parsed.blocks if b.kind == "hr"]
    assert len(hrs) == 1
    assert hrs[0].text == ""


def test_blocks_image() -> None:
    """纯图片段落归类为 image，text 为 alt 文本。"""
    parsed = parse_markdown("![alt](./a.png)")
    images = [b for b in parsed.blocks if b.kind == "image"]
    assert [b.text for b in images] == ["alt"]


def test_blocks_frontmatter() -> None:
    """frontmatter 块置于文档最前，不进入正文。"""
    parsed = parse_markdown("---\ntitle: X\n---\n# T\n\nbody")
    assert parsed.blocks[0].kind == "frontmatter"
    assert parsed.blocks[0].text == ""


def test_blocks_document_order() -> None:
    """块按文档顺序产出（前置段落 / 标题 / 代码 / 列表）。"""
    md = "前置段落\n\n# H\n\n```\ncode\n```\n\n- item"
    kinds = [b.kind for b in parse_markdown(md).blocks]
    assert kinds == ["paragraph", "heading", "code", "list"]


# ---------------------------------------------------------------- 2. 锚点一致性


def test_heading_blocks_anchors_match_heading_tree() -> None:
    """heading blocks 锚点与 heading_tree 锚点完全一致（同一 seen 序列）。"""
    md = "# A\n\n# A\n\n## B\n\n# !!!\n\nbody"
    parsed = parse_markdown(md)
    flat = list(_flatten(parsed.heading_tree))
    block_anchors = [b.anchor for b in parsed.blocks if b.kind == "heading"]
    assert block_anchors == [n.anchor for n in flat]
    assert block_anchors == ["a", "a-1", "b", "section"]


def test_normalized_text_derived_from_blocks() -> None:
    """normalized_text 与由 blocks 按 \\n\\n 连接得到的文本一致。"""
    md = "# H\n\npara\n\n```\ncode\n```\n\n- a\n- b"
    parsed = parse_markdown(md)
    joined = "\n\n".join(b.text for b in parsed.blocks if b.text)
    assert parsed.normalized_text == joined


def test_mdblock_is_frozen() -> None:
    """MDBlock 为 frozen dataclass，不可原地修改。"""
    block = MDBlock(kind="paragraph", text="x")
    with pytest.raises(FrozenInstanceError):
        block.text = "y"  # type: ignore[misc]
