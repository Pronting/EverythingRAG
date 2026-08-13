"""ingestion Markdown 解析器测试：标题树 + 归一化正文（MVP 任务 2）。

验收断言对应任务规格 7 条；全部为纯内存字符串操作，不产生文件读写与出网。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest

from app.ingestion import HeadingNode, ParsedMarkdown, parse_markdown


def _flatten(nodes: tuple[HeadingNode, ...]) -> Iterator[HeadingNode]:
    """按文档顺序展平标题树，便于一次性断言全部节点。"""
    for node in nodes:
        yield node
        yield from _flatten(node.children)


# ---------------------------------------------------------------- 1. 标题树


def test_heading_tree_nesting_and_closing() -> None:
    """H1→H2→H3 嵌套正确；同级/更高级标题关闭当前子树。"""
    md = """# 文档标题

## 第一节

### 小节甲
### 小节乙

## 第二节

# 第二部分

## 二之节
"""
    parsed = parse_markdown(md)
    roots = parsed.heading_tree
    assert len(roots) == 2

    h1 = roots[0]
    assert h1.level == 1
    assert h1.text == "文档标题"
    assert len(h1.children) == 2  # 第一节 / 第二节

    s1 = h1.children[0]
    assert s1.level == 2
    assert s1.text == "第一节"
    assert [c.text for c in s1.children] == ["小节甲", "小节乙"]
    assert all(c.level == 3 and c.children == () for c in s1.children)

    assert h1.children[1].text == "第二节"
    assert h1.children[1].children == ()

    h1b = roots[1]
    assert h1b.text == "第二部分"
    assert [c.text for c in h1b.children] == ["二之节"]

    # 全树节点数与顺序正确
    flat = list(_flatten(roots))
    assert [n.text for n in flat] == [
        "文档标题",
        "第一节",
        "小节甲",
        "小节乙",
        "第二节",
        "第二部分",
        "二之节",
    ]
    # 锚点全局唯一
    anchors = [n.anchor for n in flat]
    assert len(anchors) == len(set(anchors))
    assert anchors == ["文档标题", "第一节", "小节甲", "小节乙", "第二节", "第二部分", "二之节"]


def test_heading_with_inline_markup_is_normalized() -> None:
    """标题内联标记被归一化（强调、行内代码等保留可读文本）。"""
    parsed = parse_markdown("### Sub & *x* 与 `code`\n\nbody")
    (node,) = parsed.heading_tree
    assert node.level == 3
    assert node.text == "Sub & x 与 code"
    assert node.anchor == "sub-x-与-code"


def test_duplicate_headings_get_unique_anchors() -> None:
    """同级重复标题锚点加 -1/-2 后缀；中文/标点按 slug 规则处理。"""
    md = "# Repeat\n# Repeat\n## Repeat\n# 重复 标题！\n# !!!"
    parsed = parse_markdown(md)
    anchors = [n.anchor for n in _flatten(parsed.heading_tree)]
    assert anchors == ["repeat", "repeat-1", "repeat-2", "重复-标题", "section"]
    assert len(set(anchors)) == len(anchors)


def test_anchor_slug_collision_gets_unique_suffixes() -> None:
    """slug 基址与自然 slug 碰撞时，后缀逐个递增直到全局唯一。"""
    md = "# a-b\n\n# a b\n\n# a b 1"
    anchors = [n.anchor for n in _flatten(parse_markdown(md).heading_tree)]
    assert len(anchors) == len(set(anchors)) == 3
    assert anchors == ["a-b", "a-b-1", "a-b-1-1"]

    md2 = "# !!!\n\n# Section\n\n# Section 1"
    anchors2 = [n.anchor for n in _flatten(parse_markdown(md2).heading_tree)]
    assert len(anchors2) == len(set(anchors2)) == 3
    assert anchors2 == ["section", "section-1", "section-1-1"]


def test_heading_node_is_frozen() -> None:
    """标题节点为 frozen dataclass，不可原地修改。"""
    node = HeadingNode(level=1, text="t", anchor="t")
    with pytest.raises(FrozenInstanceError):
        node.level = 2  # type: ignore[misc]


# ---------------------------------------------------------------- 2. 归一化正文


def test_inline_markup_stripped_text_kept() -> None:
    """行内标记语法去除、可读文本保留（加粗/斜体/链接/行内代码）。"""
    md = "**加粗** *斜体* [链接](https://example.com) `code`"
    parsed = parse_markdown(md)
    body = parsed.normalized_text
    for token in ("加粗", "斜体", "链接", "code"):
        assert token in body
    for raw in (
        "**加粗**",
        "*斜体*",
        "[链接](https://example.com)",
        "https://example.com",
        "`code`",
    ):
        assert raw not in body


def test_image_alt_kept_bare_image_dropped() -> None:
    """图片保留 alt 文本；无 alt 的裸图片不产生内容。"""
    md = "第一张：![图片 alt](./a.png)\n\n第二张：![](./bare.png)"
    body = parse_markdown(md).normalized_text
    assert "图片 alt" in body
    assert "![" not in body
    assert "](./bare.png)" not in body
    assert "第二张：" in body


def test_code_blocks_preserved_as_text() -> None:
    """围栏代码块与缩进代码块内容保留为正文文本。"""
    md = '```python\nprint("hi")\n```\n\n缩进块：\n\n    indent line\n'
    body = parse_markdown(md).normalized_text
    assert 'print("hi")' in body
    assert "indent line" in body
    assert "```" not in body


def test_strikethrough_and_table_normalized() -> None:
    """删除线标记去除；GFM 表格按行保留，行内单元格以 | 分隔。"""
    md = "~~删除~~ 保留\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    body = parse_markdown(md).normalized_text
    assert "删除" in body
    assert "保留" in body
    assert "~~" not in body
    assert "a | b" in body  # 表头行整体保留（行级粒度）
    assert "1 | 2" in body  # 数据行整体保留


def test_lists_and_blockquote_content_kept() -> None:
    """无序/有序列表与引用块内容均保留，列表标记去除。"""
    md = "- 第一项\n- 第二项\n\n> 引用内容\n\n1. 有序一\n2. 有序二"
    body = parse_markdown(md).normalized_text
    for text in ("第一项", "第二项", "引用内容", "有序一", "有序二"):
        assert text in body
    assert "1." not in body


def test_line_endings_and_blank_lines_normalized() -> None:
    """CRLF→LF、连续空行折叠、行尾空白去除；相邻块间保留一个空行边界。"""
    md = "# T\r\n\r\npara one\r\nsecond line\r\n\r\n\r\n\r\n\r\nafter\r\n\r\n```\r\ncode\r\n```\r\n"
    body = parse_markdown(md).normalized_text
    assert "\r" not in body
    assert "para one second line" in body  # softbreak → 单个空格
    assert "after" in body
    assert "\n\n\n" not in body  # 块内多余空行折叠：最多保留一个空行
    # 段落边界可分辨：相邻顶层块之间恰好一个空行
    assert "para one second line\n\nafter" in body
    assert "after\n\ncode" in body


def test_paragraph_boundaries_preserved() -> None:
    """相邻段落保留一个空行分隔（不合并），供任务 3 按段落切块。"""
    assert parse_markdown("para one\n\npara two").normalized_text == "para one\n\npara two"
    assert parse_markdown("a\n\n\n\nb").normalized_text == "a\n\nb"


# ---------------------------------------------------------------- 3. frontmatter


def test_frontmatter_stripped_and_title_mapped() -> None:
    """YAML frontmatter 不进入正文；title 字段映射到结果 title。"""
    md = """---
title: 我的文档
author: me
draft: true
tags: [a, b]
---
# 正文标题

正文内容
"""
    parsed = parse_markdown(md)
    assert parsed.title == "我的文档"
    body = parsed.normalized_text
    for leaked in ("title:", "author", "draft", "tags:", "我的文档", "---"):
        assert leaked not in body
    assert "正文标题" in body
    assert "正文内容" in body
    assert [n.text for n in parsed.heading_tree] == ["正文标题"]


def test_title_fallback_to_first_h1_or_none() -> None:
    """无 frontmatter 时取首个 H1；无 H1 时为 None。"""
    assert parse_markdown("# 只有 H1\n").title == "只有 H1"
    assert parse_markdown("## 没有 H1\n").title is None
    assert parse_markdown("普通段落\n").title is None
    assert parse_markdown("---\n---\n").title is None


# ---------------------------------------------------------------- 4. 标题可检索


def test_heading_text_present_in_normalized_body() -> None:
    """标题文本同时存在于归一化正文，保证标题可被检索。"""
    md = "# Alpha\n\n## Beta\n\n### Gamma\n\nbody 段落"
    body = parse_markdown(md).normalized_text
    for heading in ("Alpha", "Beta", "Gamma"):
        assert heading in body


# ---------------------------------------------------------------- 5. 空/畸形文档


@pytest.mark.parametrize(
    "md",
    [
        "",
        "   \n\n  \t  ",
        "```\n未闭合围栏",
        "\x00\x01\x02 binary \xff\xfe 乱码",
        "plain paragraph without heading",
        "﻿# BOM 标题",
        "###   ",
    ],
)
def test_empty_and_malformed_do_not_raise(md: str) -> None:
    """空/畸形文档不抛异常，返回合法结构。"""
    parsed = parse_markdown(md)
    assert isinstance(parsed, ParsedMarkdown)
    assert isinstance(parsed.heading_tree, tuple)
    assert isinstance(parsed.normalized_text, str)


def test_malformed_specific_cases() -> None:
    """畸形文档的具体结构断言：空正文/未闭合围栏/仅段落。"""
    assert parse_markdown("").normalized_text == ""
    assert parse_markdown("   \n\n  ").normalized_text == ""

    unclosed = parse_markdown("```\n未闭合围栏\n")
    assert "未闭合围栏" in unclosed.normalized_text  # 围栏内容保留
    assert unclosed.heading_tree == ()
    assert unclosed.title is None

    plain = parse_markdown("just some text")
    assert plain.heading_tree == ()
    assert plain.normalized_text == "just some text"
    assert plain.title is None

    bom = parse_markdown("﻿# BOM 标题\n")
    assert bom.title == "BOM 标题"
    assert bom.heading_tree[0].text == "BOM 标题"


# ---------------------------------------------------------------- 6. 纯本地零出网


def test_parse_is_pure_local_no_network(monkeypatch) -> None:
    """解析过程零出网：若发起任何网络连接立即失败。"""

    def deny_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("解析过程中发起了网络连接")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)
    parsed = parse_markdown("# t\n\nbody with **加粗**\n")
    assert parsed.heading_tree[0].text == "t"
    assert "加粗" in parsed.normalized_text


def test_parse_is_deterministic() -> None:
    """同一输入两次解析结果完全一致（锚点稳定）。"""
    md = "# A 标题\n\n## B\n\n# A 标题\n\nbody"
    assert parse_markdown(md) == parse_markdown(md)


# ---------------------------------------------------------------- 7. 噪声行过滤（优化①）


def _content_blocks(md: str) -> list[str]:
    """提取非 frontmatter 的内容块文本（排除空块）。"""
    return [b.text for b in parse_markdown(md).blocks if b.text]


def test_pure_tag_line_paragraph_dropped() -> None:
    """Obsidian 纯标签行（#c端 #埋点）不产生段落块，也不进归一化正文。"""
    parsed = parse_markdown("#c端 #埋点\n\n正文内容")
    assert _content_blocks("#c端 #埋点\n\n正文内容") == ["正文内容"]
    assert "埋点" not in parsed.normalized_text
    assert "c端" not in parsed.normalized_text


def test_single_tag_line_dropped() -> None:
    """单标签行同样被丢弃（#业务 #问题吐槽 / #erp / #问题排查）。"""
    for tag in ("#业务 #问题吐槽", "#erp", "#问题排查", "#cider #架构"):
        assert _content_blocks(f"{tag}\n\n正文") == ["正文"]


def test_pure_url_line_dropped() -> None:
    """纯 URL 行无检索价值，被丢弃。"""
    assert _content_blocks("https://example.com/long/page?x=1\n\n正文") == ["正文"]


def test_pure_punct_or_emoji_line_dropped() -> None:
    """纯标点/表情行（……、===、😀）被丢弃。"""
    for noise in ("......", "！！！", "=====", "😀😀", "---"):
        assert _content_blocks(f"{noise}\n\n正文") == ["正文"]


def test_tag_line_with_real_content_kept() -> None:
    """标签与正文混排的行不是噪声，必须保留（不过滤）。"""
    md = "#c端 #埋点 这是关于埋点体系的正文说明"
    assert _content_blocks(md) == [md]


def test_url_with_trailing_cjk_text_kept() -> None:
    """URL 后无空格粘连中文说明（https://x.com，参见文档）不是噪声，整行保留。"""
    md = "https://example.com，参见文档"
    assert _content_blocks(md) == [md]


def test_tag_with_trailing_cjk_text_kept() -> None:
    """标签后跟中文正文（#标签：这是正文，全角冒号无空格）不是噪声，整行保留。"""
    md = "#c端：这是关于埋点体系的正文说明"
    assert _content_blocks(md) == [md]


def test_url_link_list_items_kept() -> None:
    """列表里的纯 URL/标签项保留：链接收藏节不因噪声过滤而整节消失。"""
    parsed = parse_markdown("# 参考资料\n\n- https://react.dev\n- https://vuejs.org")
    lists = [b.text for b in parsed.blocks if b.kind == "list"]
    assert lists == ["https://react.dev", "https://vuejs.org"]
