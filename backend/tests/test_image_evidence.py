from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.image_evidence import image_facts
from tests.test_hybrid_retriever import FakeVectorStore, _block, _hit


def test_projection_preserves_visual_numbers_and_removes_copied_context():
    raw = (
        "【所属主题】风控\n【正文上下文】" + "重复的请求频次分析。" * 80
        + "【图片描述】【画面主体】请求曲线\n"
        "【图中文字(OCR)】P99.9=123.45ms; P99=80ms\n"
        "【图表与数据】14:00至15:00下降。\n"
        "【与上下文的关系】未提供上下文\n【可检索标签】风控，频次，P99"
    )
    result = image_facts(raw)
    assert "P99.9=123.45ms; P99=80ms" in result
    assert "14:00至15:00下降" in result
    assert "重复" not in result
    assert "未提供上下文" not in result
    assert "可检索标签" not in result


def test_unknown_legacy_description_stays_available():
    assert image_facts("旧描述：支付→风控：超时500ms") == "旧描述：支付→风控：超时500ms"
    assert image_facts("【图片描述】OCR: P99=80ms") == "OCR: P99=80ms"


def test_image_companion_budget_is_spent_on_visual_evidence():
    text = "风控指标的监控说明与业务配置，用于验证不同时间窗口的请求频次以及策略效果。" * 2
    raw_image = "【正文上下文】" + text * 10 + "【图片描述】【证据】P99=80ms，曲线下降。"
    blocks = [
        _block("text", text, "风控.md", "监控"),
        _block("image", raw_image, "风控.md", "监控"),
    ]
    blocks[1][2]["chunk_type"] = "image_description"
    result = HybridRetriever(FakeVectorStore(blocks, [_hit("text", .9)]),
                             min_similarity=.6).retrieve("监控指标", [1.0])
    chunk = next(c for c in result if c.block_id == "text")
    assert "P99=80ms，曲线下降" in chunk.context_text
    assert chunk.context_text.count(text) == 1


def test_image_copied_context_cannot_win_lexical_search():
    blocks = [
        _block("text", "罕见业务主题的真实依据", "记录.md"),
        _block("image", "【正文上下文】罕见业务主题\n【图片描述】一只猫", "照片.md"),
    ]
    blocks[1][2]["chunk_type"] = "image_description"
    retriever = HybridRetriever(FakeVectorStore(blocks, []), min_similarity=.6)
    result = retriever.retrieve("罕见业务主题", [1.0])
    assert [c.block_id for c in result] == ["text"]
