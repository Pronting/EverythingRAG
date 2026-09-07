import pytest

from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import build_query_variants, normalize_query
from tests.test_hybrid_retriever import FakeVectorStore, _block, _hit

QUERY = "关于99分位是如何在公司的业务项目中作为指标去进行调优的，是在什么样的场景下，他对99分位又对这个场景有什么样的作用呢"


def test_percentile_aliases_preserve_numeric_precision():
    for text in ["P99", "99%分位", "99百分位", "九九分位", "九十九分位", "９９分位"]:
        assert tokenize(text) == ["p99"]
    assert tokenize("99.9分位") == ["p99.9"]
    assert tokenize("99.99分位") == ["p99.99"]
    assert "p99" not in tokenize("99.9分位")
    assert "p99" not in tokenize("199分位")


def test_usage_query_keeps_original_and_specific_qualifiers():
    variants = build_query_variants(QUERY)
    assert variants[0].text == QUERY
    assert variants[1].text == "99分位"
    assert "火星" in normalize_query("关于99分位在火星项目中的作用是什么")
    assert "P95" in normalize_query("关于P99和P95在业务项目中的作用")


@pytest.mark.parametrize("score,expected", [(0.42, True), (0.1, False)])
def test_exact_metric_prose_rescued_but_dense_contradictions_rejected(score, expected):
    text = "userId、deviceId、ip 三个维度；时间窗口为一分钟、一小时、一天。频次根据用户访问统计数据的99%分位来制定，重点接口包含商详、collection、评论。"
    blocks = [_block("target", text, "实时风控方案.md")]
    blocks += [_block(str(i), "无关的商品推荐文档内容", f"{i}.md") for i in range(30)]
    retriever = HybridRetriever(FakeVectorStore(blocks, [_hit("target", score)]), min_similarity=.6)
    result = retriever.retrieve(QUERY, [1.0])
    assert ("target" in [c.block_id for c in result]) is expected
