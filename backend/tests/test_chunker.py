"""ingestion 语义切块测试：标题锚点分层 + 防边界漂移（MVP 任务 3）。

验收断言对应任务规格 9 条：标题层级切块 / 块长上限 / 代码块整体成块 /
无标题文档 / 边界不漂移 / 确定性 / 空容错 / 元数据 / 零出网。
全部为纯内存计算，不产生文件读写与出网。
"""

from __future__ import annotations

import socket

import pytest

from app.ingestion import chunk_document, parse_markdown


def test_short_text_merges_across_images() -> None:
    """步骤式文档：截图夹在短句之间不打断正文累积，短句跨图合并为一段。"""
    a = "第一步：确定想要查询的事件、属性和时间范围。" * 3
    b = "第二步：在下拉框中找到想要查看的事件，单击即可选中。" * 3
    c = "第三步：点击红框部分选择想要查看的指标。" * 3
    md = f"# 使用方法\n\n{a}\n\n![图](x.png)\n\n{b}\n\n![图](y.png)\n\n{c}"
    chunks = chunk_document(parse_markdown(md), "f.md")
    text_chunks = [ch for ch in chunks if ch.kind == "text"]
    assert len(text_chunks) == 1  # 三步短句跨图片合并为一个文本块
    assert "第一步" in text_chunks[0].text and "第三步" in text_chunks[0].text
    assert len(text_chunks[0].text) >= 120


# ---------------------------------------------------------------- 1. 标题层级切块


def test_heading_level_chunking() -> None:
    """每个标题区间产块；过短区间前向合并，合并块保留首个标题锚点身份。"""
    md = "# H1\n\ntext-a\n\n## H2\n\ntext-b\n\n## H2b\n\ntext-c"
    chunks = chunk_document(parse_markdown(md), source_file="notes.md")
    assert len(chunks) == 1  # 三个 6 字符短区间合并为一块
    ids = [c.block_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == ["/H1/h1#1"]
    assert chunks[0].text == "text-a\n\ntext-b\n\ntext-c"
    assert chunks[0].heading_path == "H1"
    assert chunks[0].anchor == "h1"
    assert chunks[0].seq == 1


def test_heading_regions_stay_separate_when_substantial() -> None:
    """每个标题区间内容达到最小块长时不合并，保持各自独立块。"""
    md = "# A\n\n" + "内容A" * 80 + "\n\n# B\n\n" + "内容B" * 80
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.block_id for c in chunks] == ["/H1/a#1", "/H1/b#1"]
    assert [c.heading_path for c in chunks] == ["A", "B"]


def test_parent_without_direct_content_produces_no_chunk() -> None:
    """父标题无直属内容时不产块；短子标题内容前向合并。"""
    md = "# A\n\n## B\n\ntext-b\n\n## C\n\ntext-c"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert len(chunks) == 1  # 两个短子区间合并
    assert chunks[0].heading_path == "A > B"
    assert chunks[0].anchor == "b"


# ---------------------------------------------------------------- 2. 块长 ≤ 上限


def test_oversize_content_splits_into_subchunks() -> None:
    """超长标题区间拆成多个子块，每块 ≤ 上限，继承同路径，seq 递增。"""
    md = "# H\n\n" + "\n\n".join(f"段落{i}" * 10 for i in range(5))
    chunks = chunk_document(parse_markdown(md), "f.md", max_chunk_chars=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 20
        assert c.heading_path == "H"
        assert c.anchor == "h"
    assert [c.seq for c in chunks] == list(range(1, len(chunks) + 1))


# ---------------------------------------------------------------- 3. 代码块整体成块


def test_short_code_block_stays_whole() -> None:
    """≤上限的代码块整体成块，不拆分。"""
    code = "print('hi')"
    md = f"# H\n\npara\n\n```\n{code}\n```"
    chunks = chunk_document(parse_markdown(md), "f.md", max_chunk_chars=100)
    code_chunks = [c for c in chunks if c.text == code]
    assert len(code_chunks) == 1
    assert len(chunks) == 2  # para + code


def test_oversized_code_block_splits_into_pieces() -> None:
    """超长代码块按字符边界拆分：内存/上下文安全，拼接可还原原文。

    巨型代码块（如 JVM 配置转储 7.6 万字符）若整块保留，会撑爆单 batch 内存，
    且无法作为上下文喂给 LLM，故按字符边界拆分。
    """
    code = "".join(f"line_{i}" * 20 for i in range(8))  # ~900 字符
    md = f"```\n{code}\n```"
    chunks = chunk_document(parse_markdown(md), "f.md", max_chunk_chars=100)
    pieces = [c.text for c in chunks]
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces) == code  # 拆分不丢内容
    assert all(c.block_id.startswith("#") for c in chunks)  # 无标题文档逐块编号


# ---------------------------------------------------------------- 4. 无标题文档


def test_no_heading_document() -> None:
    """无标题文档相邻段落合并为连贯块（短碎片不再逐段独立成块）。"""
    md = "para1\n\npara2\n\npara3"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.block_id for c in chunks] == ["#1"]
    assert [c.text for c in chunks] == ["para1\n\npara2\n\npara3"]
    assert all(c.heading_path == "" and c.anchor == "" for c in chunks)


# ---------------------------------------------------------------- 5. 边界不漂移（核心）


def test_local_edit_does_not_drift() -> None:
    """只改局部段落：未受影响的区间 block_id 与 text 完全不变。"""
    md = "# A\n\n" + "内容A" * 60 + "\n\n# B\n\n" + "内容B" * 60 + "\n\n# C\n\n" + "内容C" * 60
    base = chunk_document(parse_markdown(md), "f.md")
    base_by_id = {c.block_id: c.text for c in base}
    edited = md.replace("内容B" * 60, "内容B-改" * 60)
    new_by_id = {c.block_id: c.text for c in chunk_document(parse_markdown(edited), "f.md")}
    for cid in ("/H1/a#1", "/H1/c#1"):
        assert cid in base_by_id
        assert base_by_id[cid] == new_by_id[cid]
    assert "/H1/b#1" in new_by_id
    assert "内容B-改" in new_by_id["/H1/b#1"]


# ---------------------------------------------------------------- 6. 确定性


def test_chunk_deterministic() -> None:
    """同一文档两次切块结果完全一致。"""
    md = "# A\n\n## B\n\nbody\n\n```\ncode\n```\n\npara"
    parsed = parse_markdown(md)
    assert chunk_document(parsed, "f.md") == chunk_document(parsed, "f.md")


# ---------------------------------------------------------------- 7. 空/畸形容错


@pytest.mark.parametrize(
    "md",
    [
        "",
        "---\n---\n",
        "# Only Heading\n",
        "   \n\n  ",
    ],
)
def test_empty_and_no_content_return_empty(md: str) -> None:
    """空文档/仅 frontmatter/无内容块 → 空列表，不抛异常。"""
    assert chunk_document(parse_markdown(md), "f.md") == []


# ---------------------------------------------------------------- 8. 元数据完整


def test_metadata_complete() -> None:
    """每个块带 source_file / anchor / heading_path / seq。"""
    md = "# T\n\n内容\n\n```\ncode\n```"
    chunks = chunk_document(parse_markdown(md), "notes/note.md")
    assert all(c.source_file == "notes/note.md" for c in chunks)
    assert all(c.heading_path == "T" and c.anchor == "t" for c in chunks)
    assert [c.seq for c in chunks] == [1, 2]


# ---------------------------------------------------------------- 9. 零出网


def test_chunk_pure_local_no_network(monkeypatch) -> None:
    """切块过程零出网：若发起任何网络连接立即失败。"""

    def deny_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("切块过程中发起了网络连接")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)
    chunks = chunk_document(parse_markdown("# H\n\npara\n\n```\ncode\n```"), "f.md")
    assert len(chunks) == 2


# ---------------------------------------------------------------- 10. 噪声行过滤（优化①）


def test_pure_tag_line_produces_no_chunk() -> None:
    """标签行不产独立块；其后的标题区间正常成块且不含标签。"""
    md = "#c端 #埋点\n# 核心宗旨\n1. 通过埋点平台管理生命周期\n2. 通过SPM囊括坑位信息"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert len(chunks) == 1
    assert chunks[0].block_id == "/H1/核心宗旨#1"
    assert "#c端" not in chunks[0].text


def test_only_tag_line_document_yields_no_chunks() -> None:
    """整篇只有标签行的文档不产生任何块（无信息可切）。"""
    assert chunk_document(parse_markdown("#c端 #埋点"), "f.md") == []
    assert chunk_document(parse_markdown("https://example.com"), "f.md") == []


def test_headingless_table_rows_merge_into_coherent_chunks() -> None:
    """无标题表格：表头语义化，相邻行合并为连贯块（行不再碎片化独立成块）。"""
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.block_id for c in chunks] == ["#1"]
    assert chunks[0].text == "a：1；b：2\n\na：3；b：4"


# ---------------------------------------------------------------- 11. 短碎片合并（优化②：检索质量根因）


def test_short_fragments_merge_into_one_chunk() -> None:
    """多个过短段落合并为一块，块长达到最小长度。"""
    md = "\n\n".join(f"第{i}行碎片内容" for i in range(10))
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert len(chunks) == 1
    assert "第0行" in chunks[0].text and "第9行" in chunks[0].text


def test_full_chunks_stay_separate() -> None:
    """达到最小块长的相邻块不合并，保持各自独立。"""
    md = "# A\n\n" + "甲" * 150 + "\n\n# B\n\n" + "乙" * 150
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.block_id for c in chunks] == ["/H1/a#1", "/H1/b#1"]
    assert chunks[0].text == "甲" * 150
    assert chunks[1].text == "乙" * 150


def test_code_chunk_never_merged() -> None:
    """代码块恒独立：正文短碎片不并入代码块，代码块也不并入正文。"""
    md = "# T\n\n短句\n\n```python\nprint('x')\n```\n\n尾句"
    chunks = chunk_document(parse_markdown(md), "f.md")
    texts = [c.text for c in chunks]
    # 短句/尾句为同区正文，可合并为一块；代码块独立
    assert any("短句" in t and "尾句" in t for t in texts)
    # 代码块内容保持完整、不与其他文本粘连
    code_chunk = next(c for c in chunks if "print('x')" in c.text)
    assert code_chunk.text == "print('x')"


def test_html_tags_stripped_from_content_but_not_code() -> None:
    """正文 HTML 标签剥离（<font> 等内联标签不污染嵌入）；代码块保留原样。"""
    md = (
        "# T\n\n人物：<font style=\"color:rgb(1,2,3);\">刚破天</font>\n\n"
        "```html\n<div>keep me</div>\n```"
    )
    chunks = chunk_document(parse_markdown(md), "f.md")
    content_chunk = next(c for c in chunks if "人物" in c.text)
    assert "人物：刚破天" in content_chunk.text
    assert "<font" not in content_chunk.text
    code_chunk = next(c for c in chunks if "keep me" in c.text)
    assert code_chunk.text == "<div>keep me</div>"
