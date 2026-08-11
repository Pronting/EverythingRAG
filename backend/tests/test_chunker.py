"""ingestion 语义切块测试：标题锚点分层 + 防边界漂移（MVP 任务 3）。

验收断言对应任务规格 9 条：标题层级切块 / 块长上限 / 代码块整体成块 /
无标题文档 / 边界不漂移 / 确定性 / 空容错 / 元数据 / 零出网。
全部为纯内存计算，不产生文件读写与出网。
"""

from __future__ import annotations

import socket

import pytest

from app.ingestion import chunk_document, parse_markdown

# ---------------------------------------------------------------- 1. 标题层级切块


def test_heading_level_chunking() -> None:
    """每个标题区间产一块：block_id 含锚点路径链且唯一，路径/锚点正确。"""
    md = "# H1\n\ntext-a\n\n## H2\n\ntext-b\n\n## H2b\n\ntext-c"
    chunks = chunk_document(parse_markdown(md), source_file="notes.md")
    assert len(chunks) == 3
    ids = [c.block_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == ["/H1/h1#1", "/H1/h1/H2/h2#1", "/H1/h1/H2/h2b#1"]
    assert [c.text for c in chunks] == ["text-a", "text-b", "text-c"]
    assert [c.heading_path for c in chunks] == ["H1", "H1 > H2", "H1 > H2b"]
    assert [c.anchor for c in chunks] == ["h1", "h2", "h2b"]
    assert [c.seq for c in chunks] == [1, 1, 1]


def test_parent_without_direct_content_produces_no_chunk() -> None:
    """父标题无直属内容时不产块；子标题内容归属父链。"""
    md = "# A\n\n## B\n\ntext-b\n\n## C\n\ntext-c"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.heading_path for c in chunks] == ["A > B", "A > C"]
    assert len(chunks) == 2


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


def test_code_block_stays_whole() -> None:
    """围栏代码超上限也不拆散；与同标题下段落分属不同块。"""
    code = "\n".join(f"line_{i}" * 20 for i in range(5))
    md = f"# H\n\npara\n\n```\n{code}\n```"
    chunks = chunk_document(parse_markdown(md), "f.md", max_chunk_chars=30)
    code_chunks = [c for c in chunks if c.text == code]
    assert len(code_chunks) == 1
    assert code_chunks[0].text == code
    assert len(chunks) == 2
    assert {c.block_id for c in chunks} == {"/H1/h#1", "/H1/h#2"}


# ---------------------------------------------------------------- 4. 无标题文档


def test_no_heading_document() -> None:
    """无标题文档按段落为最小单元切块（#1/#2/#3），无路径/锚点。"""
    md = "para1\n\npara2\n\npara3"
    chunks = chunk_document(parse_markdown(md), "f.md")
    assert [c.block_id for c in chunks] == ["#1", "#2", "#3"]
    assert [c.text for c in chunks] == ["para1", "para2", "para3"]
    assert all(c.heading_path == "" and c.anchor == "" for c in chunks)


# ---------------------------------------------------------------- 5. 边界不漂移（核心）


def test_local_edit_does_not_drift() -> None:
    """只改局部段落：未受影响的区间 block_id 与 text 完全不变。"""
    md = "# A\n\npara-A\n\n# B\n\npara-B\n\n# C\n\npara-C"
    base = chunk_document(parse_markdown(md), "f.md")
    base_by_id = {c.block_id: c.text for c in base}
    edited = md.replace("para-B", "para-B-CHANGED")
    new_by_id = {c.block_id: c.text for c in chunk_document(parse_markdown(edited), "f.md")}
    for cid in ("/H1/a#1", "/H1/c#1"):
        assert cid in base_by_id
        assert base_by_id[cid] == new_by_id[cid]
    assert "/H1/b#1" in new_by_id
    assert new_by_id["/H1/b#1"] == "para-B-CHANGED"


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
