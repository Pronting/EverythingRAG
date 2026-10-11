from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.pipeline import read_text


SOURCE = Path.home() / ".everything-rag" / "documents" / "个人提效，攒不成组织提效：货拉拉 AI Coding 落地实践.md"


def test_real_aibox_header_only_table_survives_parsing_and_chunking():
    parsed = parse_markdown(read_text(SOURCE))
    expected = (
        "Rules　团队编码约束",
        "Skills　可复用的多步工作流",
        "Commands　高频动作的快捷指令",
        "MCP　外部服务与数据源的接入",
    )
    tables = [block.text for block in parsed.blocks if block.kind == "table"]
    for value in expected:
        assert any(value in table for table in tables), value
    chunks = chunk_document(parsed, str(SOURCE))
    for value in expected:
        assert any(value in chunk.text for chunk in chunks), value


def test_real_body_rows_keep_header_associations():
    parsed = parse_markdown(read_text(SOURCE))
    tables = [block.text for block in parsed.blocks if block.kind == "table"]
    assert any("形态：补全式" in row and "运行位置：IDE 内" in row for row in tables)
    assert any("维度：核心机制" in row and "OpenSpec（Fission-AI）" in row for row in tables)
    assert any("阶段：PRD" in row and "功能目标 · 业务规则 · 验收标准" in row for row in tables)
