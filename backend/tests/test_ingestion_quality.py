"""脏 Markdown/HTML 的解析、切片与图片上下文质量回归。

这些 case 只验证纯内存 ingestion 行为，不访问网络、不读用户文章、不修改 live 索引。
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion import Chunk, chunk_document, parse_markdown
from app.ingestion.image_state_store import ImageTask
from app.ingestion.pipeline import (
    IngestionPipeline,
    _compact_context,
    make_chunk_metadata,
)
from app.ingestion.scanner import DiscoveredFile


def _texts(md: str) -> list[str]:
    return [block.text for block in parse_markdown(md).blocks if block.text]


def _chunks(md: str, max_chars: int = 2000, min_chars: int = 120) -> list[Chunk]:
    return chunk_document(
        parse_markdown(md),
        "quality.md",
        max_chunk_chars=max_chars,
        min_chunk_chars=min_chars,
    )


def test_inline_html_tags_removed_visible_text_kept() -> None:
    assert _texts('<font color="red">可见正文</font>') == ["可见正文"]


def test_html_entities_are_decoded() -> None:
    assert _texts("A &amp; B &lt; C &#x4E2D;") == ["A & B < C 中"]


def test_script_style_and_comments_are_removed_with_contents() -> None:
    md = "前文<script>secret()</script><style>.bad{}</style><!-- hidden -->后文"
    text = " ".join(_texts(md))
    assert "前文" in text and "后文" in text
    assert "secret" not in text and ".bad" not in text and "hidden" not in text


def test_br_keeps_semantic_separator_without_gluing_words() -> None:
    assert _texts("京东<br/>淘宝<br>公司") == ["京东；淘宝；公司"]


def test_html_div_and_paragraphs_remain_separate_blocks() -> None:
    assert _texts("<div><p>第一段</p><p>第二段</p></div>") == ["第一段", "第二段"]


def test_html_list_becomes_readable_markdown_list() -> None:
    parsed = parse_markdown("<ul><li>京东方案</li><li>淘宝方案</li></ul>")
    assert [block.text for block in parsed.blocks if block.kind == "list"] == [
        "京东方案",
        "淘宝方案",
    ]


def test_html_heading_becomes_real_heading_boundary() -> None:
    parsed = parse_markdown("<h2>交互</h2><p>厂商对比</p>")
    heading = next(block for block in parsed.blocks if block.kind == "heading")
    assert (heading.level, heading.text) == (2, "交互")
    assert _chunks("<h2>交互</h2><p>厂商对比</p>")[0].heading_path == "交互"


def test_html_table_becomes_header_aware_table_rows() -> None:
    md = """
    <table>
      <tr><th>厂商</th><th>分表方式</th></tr>
      <tr><td>京东</td><td>用户维度</td></tr>
      <tr><td>淘宝</td><td>商品维度</td></tr>
    </table>
    """
    rows = [block.text for block in parse_markdown(md).blocks if block.kind == "table"]
    assert rows == ["厂商：京东；分表方式：用户维度", "厂商：淘宝；分表方式：商品维度"]


def test_html_table_br_is_a_cell_separator_not_a_broken_row() -> None:
    md = "<table><tr><th>规则</th></tr><tr><td>一<br>二</td></tr></table>"
    rows = [block.text for block in parse_markdown(md).blocks if block.kind == "table"]
    assert rows == ["规则：一；二"]


def test_html_img_is_preserved_as_image_reference() -> None:
    parsed = parse_markdown('<img src="https://cdn.example.com/a.png" alt="架构图">')
    image = next(block for block in parsed.blocks if block.kind == "image")
    assert (image.text, image.src) == ("架构图", "https://cdn.example.com/a.png")


def test_fenced_html_and_entities_are_preserved_verbatim() -> None:
    parsed = parse_markdown("```html\n<div>&amp;</div>\n```")
    code = next(block for block in parsed.blocks if block.kind == "code")
    assert code.text == "<div>&amp;</div>"


def test_fence_info_like_line_does_not_close_an_open_fence() -> None:
    parsed = parse_markdown(
        "```python\n正文代码\n```plain\n<div>代码里的 HTML</div>\n```\n"
        '<font color="red">围栏后的正文</font>'
    )
    code = next(block for block in parsed.blocks if block.kind == "code")
    assert "```plain" in code.text
    assert "<div>代码里的 HTML</div>" in code.text
    assert _texts('<font color="red">围栏后的正文</font>') == ["围栏后的正文"]
    assert "<font" not in " ".join(
        block.text for block in parsed.blocks if block.kind != "code"
    )


def test_inline_code_html_is_not_cleaned_as_document_html() -> None:
    assert _texts("正文 `<b>code</b>` 尾部") == ["正文 <b>code</b> 尾部"]


def test_inline_code_discards_yuque_presentation_wrappers_only() -> None:
    assert _texts('调用 `<font style="color: red">consume</font>` 方法') == [
        "调用 consume 方法"
    ]


def test_markdown_autolink_url_is_not_damaged_by_html_cleaning() -> None:
    assert _texts("入口 <https://example.com/a?x=1&y=2> &amp; 文档") == [
        "入口 https://example.com/a?x=1&y=2 & 文档"
    ]


def test_residual_emphasis_markers_are_removed() -> None:
    assert _texts("**未闭合加粗 和 ~~未闭合删除") == ["未闭合加粗 和 未闭合删除"]


def test_identifiers_single_stars_and_urls_survive_cleanup() -> None:
    text = _texts("snake_case 与 2 * 3，地址 https://example.com/a_b")[0]
    assert "snake_case" in text and "2 * 3" in text and "https://example.com/a_b" in text


def test_inline_code_markdown_markers_survive_cleanup() -> None:
    assert _texts('表达式 `value = "**"` 保留') == ['表达式 value = "**" 保留']


def test_numeric_statistics_table_is_never_dropped() -> None:
    md = "| 日期 | UV | GMV |\n|---|---:|---:|\n| 2025-11-11 | 123456 | 9876543.21 |"
    chunks = _chunks(md)
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert "9876543.21" in chunks[0].text


def test_contiguous_table_rows_pack_into_one_table_chunk() -> None:
    md = "| 厂商 | 方案 |\n|---|---|\n| 京东 | A |\n| 淘宝 | B |"
    chunks = _chunks(md)
    assert [(chunk.kind, chunk.text) for chunk in chunks] == [
        ("table", "厂商：京东；方案：A\n\n厂商：淘宝；方案：B")
    ]


def test_table_chunks_split_only_at_row_boundaries_when_possible() -> None:
    md = "| 厂商 | 方案 |\n|---|---|\n" + "\n".join(
        f"| 厂商{i} | {'方案内容' * 4}{i} |" for i in range(6)
    )
    chunks = _chunks(md, max_chars=60, min_chars=0)
    assert len(chunks) > 1
    assert all(chunk.kind == "table" and len(chunk.text) <= 60 for chunk in chunks)
    assert sum(chunk.text.count("厂商：") for chunk in chunks) == 6


def test_oversized_table_row_splits_at_semantic_cells() -> None:
    md = f"| A | B | C |\n|---|---|---|\n| {'甲' * 25} | {'乙' * 25} | {'丙' * 25} |"
    chunks = _chunks(md, max_chars=32, min_chars=0)
    assert len(chunks) >= 3
    assert all(chunk.kind == "table" and len(chunk.text) <= 32 for chunk in chunks)
    assert any("A：" in chunk.text for chunk in chunks)
    assert any("B：" in chunk.text for chunk in chunks)
    assert any("C：" in chunk.text for chunk in chunks)


def test_table_and_paragraphs_do_not_blend_chunk_types() -> None:
    md = "# 对比\n\n前置说明\n\n| 厂商 | 方案 |\n|---|---|\n| 京东 | A |\n\n后置结论"
    chunks = _chunks(md)
    assert [chunk.kind for chunk in chunks] == ["text", "table", "text"]
    assert all(chunk.heading_path == "对比" for chunk in chunks)


def test_short_chunks_never_merge_across_top_level_headings() -> None:
    chunks = _chunks("# 京东\n\n短方案\n\n# 淘宝\n\n短方案")
    assert [chunk.heading_path for chunk in chunks] == ["京东", "淘宝"]
    assert [chunk.text for chunk in chunks] == ["短方案", "短方案"]


def test_short_sibling_headings_merge_with_explicit_section_labels() -> None:
    chunks = _chunks("# 分表\n\n## 京东\n\nA\n\n## 淘宝\n\nB")
    assert len(chunks) == 1
    assert chunks[0].heading_path == "分表 > 京东 / 淘宝"
    assert chunks[0].text == "【京东】\nA\n\n【淘宝】\nB"
    assert chunks[0].heading_aliases == ("分表 > 京东", "分表 > 淘宝")


def test_short_sibling_tables_merge_without_losing_each_heading() -> None:
    chunks = _chunks(
        "# 指标\n\n## 京东\n\n| 名称 | 值 |\n|---|---|\n| UV | 10 |\n\n"
        "## 淘宝\n\n| 名称 | 值 |\n|---|---|\n| UV | 20 |"
    )
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert chunks[0].heading_path == "指标 > 京东 / 淘宝"
    assert chunks[0].text == "【京东】\n名称：UV；值：10\n\n【淘宝】\n名称：UV；值：20"
    metadata = make_chunk_metadata(
        "table-id",
        DiscoveredFile(Path("quality.md"), "quality.md", 0.0, 1, "hash"),
        chunks[0],
    )
    assert metadata.heading_aliases == '["指标 > 京东", "指标 > 淘宝"]'


def test_short_paragraphs_inside_one_heading_still_merge() -> None:
    chunks = _chunks("# 交互\n\n第一段\n\n第二段\n\n第三段")
    assert len(chunks) == 1
    assert chunks[0].text == "第一段\n\n第二段\n\n第三段"


def test_short_heading_after_code_does_not_get_absorbed_into_code() -> None:
    chunks = _chunks("# 代码\n\n```python\nprint('x')\n```\n\n# 说明\n\n短正文")
    assert [chunk.kind for chunk in chunks] == ["code", "text"]
    assert chunks[0].text == "print('x')"
    assert chunks[1].heading_path == "说明"


def test_table_and_code_metadata_keep_their_real_chunk_type() -> None:
    file = DiscoveredFile(Path("quality.md"), "quality.md", 0.0, 1, "hash")
    table = _chunks("| A |\n|---|\n| 1 |")[0]
    code = _chunks("```python\nprint(1)\n```")[0]
    assert make_chunk_metadata("table-id", file, table).chunk_type == "table"
    assert make_chunk_metadata("code-id", file, code).chunk_type == "code"


def test_image_context_is_deduplicated_and_strictly_capped() -> None:
    context = _compact_context(["重复段落\n\n唯一段落", "重复段落\n\n" + "长" * 80], 30)
    assert context.count("重复段落") == 1
    assert "唯一段落" in context
    assert len(context) <= 30


class _NoopEmbedder:
    fingerprint = "noop"
    dim = 1

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _ContextStore:
    def list_blocks(self) -> list[tuple[str, str, dict[str, str]]]:
        return [
            ("a", "京东章节正文", {"source_file": "doc.md", "chunk_type": "text", "heading_path": "京东"}),
            ("b", "淘宝章节正文", {"source_file": "doc.md", "chunk_type": "text", "heading_path": "淘宝"}),
            ("c", "子节正文", {"source_file": "doc.md", "chunk_type": "text", "heading_path": "总览 > 子节"}),
            ("d", "统计口径：成交额", {"source_file": "doc.md", "chunk_type": "table", "heading_path": "统计"}),
        ]


def test_image_context_uses_exact_heading_not_parent_or_entire_document() -> None:
    pipeline = IngestionPipeline(_NoopEmbedder(), _ContextStore())  # type: ignore[arg-type]
    task = ImageTask("id", "src", "doc", "doc.md", "京东", "京东")
    context = pipeline._surrounding_text(task)
    assert context == "京东章节正文"
    assert "淘宝" not in context and "子节" not in context


def test_nested_image_context_does_not_include_sibling_or_parent_text() -> None:
    pipeline = IngestionPipeline(_NoopEmbedder(), _ContextStore())  # type: ignore[arg-type]
    task = ImageTask("id", "src", "doc", "doc.md", "总览 > 子节", "子节")
    assert pipeline._surrounding_text(task) == "子节正文"


def test_image_context_can_use_table_text_from_the_exact_heading() -> None:
    pipeline = IngestionPipeline(_NoopEmbedder(), _ContextStore())  # type: ignore[arg-type]
    task = ImageTask("id", "src", "doc", "doc.md", "统计", "统计")
    assert pipeline._surrounding_text(task) == "统计口径：成交额"
