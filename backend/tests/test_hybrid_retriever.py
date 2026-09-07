"""HybridRetriever 混合检索测试：dense + BM25（标题注入）+ RRF 融合。

核心验收：标题关键词（实时风控方案 / ApiSix 网关）只在文件名、不在块文本时，
BM25 标题注入能把它救回 top-5；近似文档对由 RRF 融合排序。
全部 fake 依赖，零出网。
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import build_query_variants
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk


class FakeVectorStore:
    """实现 VectorStore Protocol：可控 list_blocks / query / count。"""

    def __init__(
        self,
        blocks: list[tuple[str, str, dict[str, Any]]] | None = None,
        query_result: list[dict[str, Any]] | None = None,
        count: int | None = None,
    ) -> None:
        self._blocks = blocks if blocks is not None else []
        self._query_result = query_result if query_result is not None else []
        self._count = count
        self.query_calls: list[tuple[list[float], int, dict[str, Any] | None]] = []

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        raise AssertionError("retrieval 不应触发 upsert")

    def delete_by_ids(self, block_ids: list[str]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_ids")

    def delete_by_where(self, where: dict[str, Any]) -> None:
        raise AssertionError("retrieval 不应触发 delete_by_where")

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        raise AssertionError("retrieval 不应触发 get_blocks_by_source")

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._blocks

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.query_calls.append((vector, top_k, where))
        return self._query_result

    def count(self) -> int:
        return self._count if self._count is not None else len(self._blocks)


def _block(block_id: str, text: str, source_file: str, heading: str | None = None) -> dict[str, Any]:
    """构造 list_blocks 的 (block_id, text, metadata)。"""
    metadata: dict[str, Any] = {
        "block_id": block_id,
        "doc_id": f"doc-{block_id}",
        "source_type": "document",
        "source_file": source_file,
        "platform": "local",
        "chunk_type": "text",
    }
    if heading is not None:
        metadata["heading_path"] = heading
    return (block_id, text, metadata)


def _hit(block_id: str, similarity: float, source_file: str = "a.md") -> dict[str, Any]:
    """构造 vectorstore.query 的命中 dict。"""
    return {
        "block_id": block_id,
        "text": f"text-{block_id}",
        "metadata": {"block_id": block_id, "source_file": source_file, "platform": "local"},
        "similarity": similarity,
        "distance": 1.0 - similarity,
    }


# 标题注入救回场景：目标文档「实时风控方案」的关键词只在文件名，块文本不含
_TITLE_BLOCKS = [
    _block("b1", "关于订单履约全链路的设计与流转", "订单履约全链路.md", "履约"),
    _block("b2", "统一缓存 sdk 的设计说明", "cider-基建2023.md", "基建"),
    _block("b3", "风控事件和拦截操作页面化，支持 userId 维度", "实时风控方案.md", "支持的策略"),
    _block("b4", "reCAPTCHA 功能调研", "Google调研.md"),
]
_TITLE_DENSE = [
    _hit("b1", 0.70, "订单履约全链路.md"),
    _hit("b4", 0.68, "Google调研.md"),
    _hit("b2", 0.65, "cider-基建2023.md"),
    _hit("b3", 0.55, "实时风控方案.md"),
]


def _retriever(
    store: FakeVectorStore,
    candidate_k: int = 20,
    context_top_k: int = 5,
    **kwargs: Any,
) -> HybridRetriever:
    return HybridRetriever(store, candidate_k=candidate_k, context_top_k=context_top_k, **kwargs)


# ---------------------------------------------------------------- 1. 标题注入救回


def test_title_injection_rescues_keyword_in_filename() -> None:
    """关键词只在文件名 -> BM25 标题注入把它带进 top-5（纯向量会排后面）。"""
    store = FakeVectorStore(blocks=_TITLE_BLOCKS, query_result=_TITLE_DENSE)
    retriever = _retriever(store)
    chunks = retriever.retrieve("实时风控方案支持哪些维度的限流策略", [1.0, 0.0, 0.0])

    ids = [c.block_id for c in chunks]
    assert "b3" in ids  # 实时风控方案.md 被救回
    assert ids[0] == "b3"  # RRF 融合后置顶


def test_bm25_lexical_rescue_exact_term() -> None:
    """查询含唯一术语，dense 未命中 -> BM25 词法命中带入。"""
    store = FakeVectorStore(
        blocks=[
            _block("x1", "普通 段落 内容", "a.md"),
            _block("x2", "唯一标识符 SPR-2024-0001 出现在此文档", "b.md"),
        ],
        query_result=[_hit("x1", 0.9, "a.md")],  # dense 只命中 x1
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("SPR-2024-0001", [1.0, 0.0, 0.0])
    assert any(c.block_id == "x2" for c in chunks)


# ---------------------------------------------------------------- 2. RRF 融合

def test_rrf_ranks_present_in_both_higher() -> None:
    """同时被 dense + BM25 命中的块 RRF 分数更高，排前面。"""
    store = FakeVectorStore(
        blocks=[
            _block("a", "埋点平台 生命周期 管理", "a.md"),
            _block("b", "缓存 设计", "b.md"),
            _block("c", "埋点 上报", "c.md"),
        ],
        query_result=[_hit("b", 0.95), _hit("a", 0.6), _hit("c", 0.5)],
    )
    retriever = _retriever(store)
    chunks = retriever.retrieve("埋点平台 埋点", [1.0, 0.0, 0.0])
    # a 同时 dense(rank2) + bm25 -> RRF 高于仅 dense rank1 的 b
    assert chunks[0].block_id == "a"


# ---------------------------------------------------------------- 3. 门禁协调


def test_strong_lexical_evidence_lowers_dense_rescue_floor() -> None:
    """强词法证据可把 dense 门槛放宽到 lexical_floor。"""
    store = FakeVectorStore(
        blocks=[_block("keep", "实时风控方案 限流维度", "实时风控方案.md")],
        query_result=[_hit("keep", 0.52, "实时风控方案.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    chunks = retriever.retrieve("实时风控方案 限流", [1.0, 0.0, 0.0])
    assert [chunk.block_id for chunk in chunks] == ["keep"]


def test_bm25_rescue_kept_when_dense_sim_above_threshold() -> None:
    """BM25 把标题关键词块带进 top-5，且其 dense sim ≥ 阈值 -> 通过门禁保留。"""
    store = FakeVectorStore(
        blocks=[
            _block("rel", "实时风控方案 限流维度", "实时风控方案.md"),
            _block("noise", "缓存 设计 说明", "cider-基建.md"),
        ],
        query_result=[_hit("noise", 0.90, "cider-基建.md"), _hit("rel", 0.57, "实时风控方案.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    chunks = retriever.retrieve("实时风控方案 限流", [1.0, 0.0, 0.0])
    assert "rel" in [chunk.block_id for chunk in chunks]


def test_production_gate_accepts_strong_bm25_only_candidate() -> None:
    """核心回归：dense top-k 缺席时不能用 0.0 冒充相似度并错杀 BM25 rescue。"""

    store = FakeVectorStore(
        blocks=[
            _block(
                "target",
                "交互方案对比：京东购物车按用户分表，淘宝按店铺与用户组合分表",
                "购物车分表.md",
                "交互",
            ),
            _block("noise", "缓存设计普通说明", "cache.md"),
        ],
        query_result=[_hit("noise", 0.51, "cache.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)

    result = retriever.retrieve_with_trace("京东 淘宝 购物车 分表 交互 对比", [1.0, 0.0])
    assert "target" in [chunk.block_id for chunk in result.chunks]
    target = next(candidate for candidate in result.trace.candidates if candidate.block_id == "target")
    assert target.dense_similarity is None
    assert target.gate_reason == "accepted_lexical_only"
    assert target.selected is True
    selected = next(chunk for chunk in result.chunks if chunk.block_id == "target")
    assert selected.similarity is None
    assert selected.match_type == "lexical"


def test_duplicate_logical_blocks_from_legacy_paths_are_returned_once() -> None:
    """相同 doc/chunk 即使来自历史任务目录，也只能占一个 Top-K 席位。"""

    stable = _block("stable", "京东与淘宝的购物车分表方案", "documents/doc/购物车分表.md", "交互")
    legacy = _block(
        "legacy",
        "京东与淘宝的购物车分表方案",
        "documents/208c0ca74fd44a12af218fb34b57e91f/doc/购物车分表.md",
        "交互",
    )
    # 历史索引的 doc_id 曾按路径生成；内容哈希去重不能依赖两个 doc_id 恰好相同。
    assert legacy[2]["doc_id"] != stable[2]["doc_id"]
    store = FakeVectorStore(
        blocks=[stable, legacy, _block("other", "拼多多店铺分表", "documents/doc/其他.md")],
        query_result=[
            _hit("legacy", 0.94, legacy[2]["source_file"]),
            _hit("stable", 0.93, stable[2]["source_file"]),
            _hit("other", 0.80, "documents/doc/其他.md"),
        ],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve("京东 淘宝 购物车 分表", [1.0])

    assert [chunk.block_id for chunk in chunks].count("stable") == 1
    assert all(chunk.block_id != "legacy" for chunk in chunks)
    assert any(chunk.block_id == "other" for chunk in chunks)
    assert store.query_calls[0][1] == 80  # candidate_k=20，先放大再去重


def test_identical_text_under_distinct_headings_keeps_both_provenances() -> None:
    """相同表格会复用在多个业务章节，不能把目标章节折叠到另一标题名下。"""

    first = _block("homepage", "规则：site.page.area.dot", "埋点.md", "首页 > SPM")
    target = _block("cart", "规则：site.page.area.dot", "埋点.md", "购物车 > SPM")
    store = FakeVectorStore(
        blocks=[first, target],
        query_result=[
            _hit("cart", 0.90, "埋点.md"),
            _hit("homepage", 0.89, "埋点.md"),
        ],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve(
        "购物车 SPM 规则", [1.0]
    )

    assert {chunk.block_id for chunk in chunks} == {"homepage", "cart"}
    assert {chunk.heading_path for chunk in chunks} == {"首页 > SPM", "购物车 > SPM"}


def test_real_cart_comparison_wording_rescues_three_rare_terms() -> None:
    """真实长口语 case：低 coverage 但命中分表/京东/淘宝三项，BM25-only 应召回。"""

    store = FakeVectorStore(
        blocks=[
            _block(
                "cart-interaction",
                (
                    "未登录可以加购，数据存在服务端。app：京东、cider；"
                    "未登录不让加购：淘宝、拼多多、eBay。优势与劣势说明。"
                ),
                "购物车分表.md",
                "交互",
            ),
            _block("noise", "普通缓存设计与用户体验说明", "cache.md"),
        ],
        query_result=[_hit("noise", 0.50, "cache.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = (
        "好像有用一些厂商的分表对比，比如说京东、淘宝等对比，和公司的对比，"
        "我想问一下这些对比的详情是什么样的"
    )

    result = retriever.retrieve_with_trace(query, [1.0, 0.0])
    assert "cart-interaction" in [chunk.block_id for chunk in result.chunks]
    candidate = next(
        item for item in result.trace.candidates if item.block_id == "cart-interaction"
    )
    assert candidate.dense_similarity is None
    assert candidate.gate_reason == "accepted_lexical_only"
    assert any(
        signal.route == "bm25:normalized" and signal.coverage is not None
        for signal in candidate.signals
    )


def test_three_rare_terms_can_override_low_but_noncontradictory_dense_score() -> None:
    """真实长口语查询的三个独立稀有词足以纠正低质量 dense 排名。"""

    target = _block(
        "cart-interaction",
        "未登录服务端加购：京东；禁止加购：淘宝；购物车方案对比",
        "购物车分表.md",
        "交互",
    )
    store = FakeVectorStore(
        blocks=[target, _block("noise", "公司对比的一般详情", "其他.md")],
        query_result=[
            _hit("noise", 0.55, "其他.md"),
            _hit("cart-interaction", 0.41, "购物车分表.md"),
        ],
    )
    query = (
        "好像有用一些厂商的分表对比，比如说京东、淘宝等对比，和公司的对比，"
        "我想问一下这些对比的详情是什么样的"
    )

    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(query, [1.0])

    assert "cart-interaction" in [chunk.block_id for chunk in result.chunks]
    trace = next(
        item for item in result.trace.candidates if item.block_id == "cart-interaction"
    )
    assert trace.gate_reason == "accepted_lexical_override"


def test_long_conversational_original_does_not_rescue_generic_overlap() -> None:
    """长口语原句只命中“公司/对比/详情”等泛词时，不允许 BM25-only 噪声入选。"""

    store = FakeVectorStore(
        blocks=[
            _block(
                "target",
                "京东购物车按用户分表，淘宝按店铺和用户组合分表",
                "购物车分表.md",
            ),
            _block(
                "noise",
                "公司产品详情与行业方案对比，介绍业务背景和用户体验",
                "公司介绍.md",
            ),
        ],
        query_result=[],
    )
    query = (
        "好像有用一些厂商的分表对比，比如说京东、淘宝等对比，和公司的对比，"
        "我想问一下这些对比的详情是什么样的"
    )

    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(query, [1.0])

    assert "target" in [chunk.block_id for chunk in result.chunks]
    assert "noise" not in [chunk.block_id for chunk in result.chunks]


def test_moderately_low_dense_signal_keeps_qualified_lexical_rescue() -> None:
    """购物车实测区间的弱 dense 信号仍保留合格词法候选。"""

    target = _block(
        "target", "京东 淘宝 厂商 分表 方案 对比", "购物车分表.md", "交互"
    )
    lexical_only_store = FakeVectorStore(blocks=[target], query_result=[])
    low_dense_store = FakeVectorStore(
        blocks=[target], query_result=[_hit("target", 0.42, "购物车分表.md")]
    )
    query = "京东 淘宝 分表 对比"

    lexical_result = _retriever(
        lexical_only_store, min_similarity=0.60
    ).retrieve_with_trace(query, [1.0])
    dense_result = _retriever(low_dense_store, min_similarity=0.60).retrieve_with_trace(
        query, [1.0]
    )

    assert [chunk.block_id for chunk in lexical_result.chunks] == ["target"]
    assert [chunk.block_id for chunk in dense_result.chunks] == ["target"]
    dense_trace = dense_result.trace.candidates[0]
    assert dense_trace.dense_similarity == pytest.approx(0.42)
    assert dense_trace.gate_reason == "accepted_lexical_override"


def test_extremely_low_dense_signal_vetoes_broad_lexical_overlap() -> None:
    """真实负例回归：极低 dense 是反证，可拦截“古埃及税制”误配现代分税制。"""

    store = FakeVectorStore(
        blocks=[_block("tax", "中国政府 王国时期 分税制 改革", "财政.md", "分税制改革")],
        query_result=[_hit("tax", 0.25, "财政.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    result = retriever.retrieve_with_trace("古埃及中王国时期的灌溉税制度", [1.0])

    assert result.chunks == ()
    candidate = result.trace.candidates[0]
    assert candidate.gate_reason == "rejected_lexical_dense_contradiction"
    assert candidate.gate_threshold == pytest.approx(0.35)


def test_weak_one_bigram_bm25_only_candidate_is_rejected() -> None:
    """负例：长问题只偶然共享一个 bigram，不能绕过 production 门禁。"""

    store = FakeVectorStore(
        blocks=[
            _block("weak", "购物车缓存说明", "cart.md"),
            _block("dense-noise", "火星生态新闻", "mars.md"),
        ],
        query_result=[_hit("dense-noise", 0.40, "mars.md")],
    )
    retriever = _retriever(store, min_similarity=0.60)
    query = "购物车之外的火星殖民地生态循环能源农业气候系统如何长期运行"

    result = retriever.retrieve_with_trace(query, [1.0, 0.0])
    weak = next(candidate for candidate in result.trace.candidates if candidate.block_id == "weak")
    assert weak.dense_similarity is None
    assert weak.gate_reason == "rejected_missing_dense_weak_lexical"
    assert weak.selected is False


@pytest.mark.parametrize(
    ("query", "false_context"),
    [
        ("量子纠缠在量子计算退相干控制中的应用", "Data Warehouse 中的应用与计算"),
        ("火星殖民地生态系统如何实现自维持", "殖民地系统"),
        ("巴赫平均律中的对位法与和声分析", "业务中的平均数据分析"),
        ("线粒体 DNA 在细胞衰老中的作用机制", "系统中的作用机制"),
    ],
)
def test_low_information_bigrams_cannot_authorize_lexical_only_rescue(
    query: str,
    false_context: str,
) -> None:
    """问题句式/宽泛关系词即使 IDF 高，也不属于可独立放行的信息实体。"""

    store = FakeVectorStore(blocks=[_block("false", false_context, "unrelated.md")])
    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(query, [1.0])
    assert result.chunks == ()
    assert result.trace.candidates[0].gate_reason == "rejected_missing_dense_weak_lexical"


def test_conversational_variant_adds_second_dense_route_and_keeps_original() -> None:
    """原查询与去口语化查询都参与 dense+BM25 union，实体不因 rewrite 丢失。"""

    class PerVectorStore(FakeVectorStore):
        def query(
            self,
            vector: list[float],
            top_k: int,
            where: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            self.query_calls.append((vector, top_k, where))
            if vector == [1.0, 0.0]:
                return [_hit("noise", 0.70, "noise.md")]
            return [_hit("target", 0.72, "购物车分表.md")]

    store = PerVectorStore(
        blocks=[
            _block("target", "京东 淘宝 厂商 分表 方案 对比 详情", "购物车分表.md", "交互"),
            _block("noise", "普通业务说明", "noise.md"),
        ]
    )
    query = "好像有厂商分表对比，比如说京东、淘宝，我想问一下详情是什么样的"
    variants = build_query_variants(query)
    retriever = _retriever(store, min_similarity=0.60)

    result = retriever.retrieve_variants_with_trace(variants, [[1.0, 0.0], [0.0, 1.0]])
    assert len(store.query_calls) == 2
    assert "target" in [chunk.block_id for chunk in result.chunks]
    target = next(candidate for candidate in result.trace.candidates if candidate.block_id == "target")
    assert {signal.route for signal in target.signals} >= {
        "dense:normalized",
        "bm25:original",
        "bm25:normalized",
    }


def test_lexical_reserve_prevents_dual_dense_routes_from_crowding_out_exact_hit() -> None:
    """双 dense 路由不能把一个明确 identifier 的 BM25-only 块挤出最终上下文。"""

    dense_hits = [_hit(f"dense-{index}", 0.90 - index * 0.01) for index in range(10)]
    blocks = [
        *[_block(f"dense-{index}", f"普通候选 {index}", "dense.md") for index in range(10)],
        _block("exact", "工单编号 SPR-2024-0001 的处置过程", "incident.md"),
    ]
    store = FakeVectorStore(blocks=blocks, query_result=dense_hits)
    query = "请问一下 SPR-2024-0001 的详情是什么样的"
    variants = build_query_variants(query)
    result = _retriever(
        store, context_top_k=5, min_similarity=0.60
    ).retrieve_variants_with_trace(variants, [[1.0], [2.0]])

    assert result.chunks[0].block_id == "exact"
    trace = next(candidate for candidate in result.trace.candidates if candidate.block_id == "exact")
    assert trace.dense_similarity is None
    assert trace.selection_reason == "lexical_reserve"


def test_dense_reserve_prevents_multi_route_lexical_hits_from_crowding_out_dense_top1() -> None:
    """多路词法候选占满上下文时，已过门禁的 dense top-1 仍必须保留。"""

    distractors = [
        _block(f"lexical-{index}", f"缓存方案细节 对比项 {index}", f"noise-{index}.md")
        for index in range(8)
    ]
    target = _block("dense-answer", "统一入口负责流量治理", "answer.md", "职责")
    store = FakeVectorStore(
        blocks=[*distractors, target],
        query_result=[
            _hit("dense-answer", 0.92, "answer.md"),
            *[
                _hit(f"lexical-{index}", 0.80 - index * 0.01, f"noise-{index}.md")
                for index in range(8)
            ],
        ],
    )

    variants = build_query_variants("我想问一下缓存方案细节有哪些对比项")
    assert len(variants) == 2
    result = _retriever(
        store, context_top_k=5, min_similarity=0.60
    ).retrieve_variants_with_trace(
        variants,
        [[float(index + 1)] for index, _variant in enumerate(variants)],
    )

    assert "dense-answer" in [chunk.block_id for chunk in result.chunks]
    trace = next(
        candidate for candidate in result.trace.candidates if candidate.block_id == "dense-answer"
    )
    assert trace.selection_reason == "dense_reserve"


def test_dense_reserve_uses_distinct_headings_before_duplicate_chunks() -> None:
    """同标题的相邻块不能同时耗尽 dense 保留位，其他答案标题要有一个席位。"""

    store = FakeVectorStore(
        blocks=[
            _block("background-1", "背景说明第一段", "search.md", "背景"),
            _block("background-2", "背景说明第二段", "search.md", "背景"),
            _block("solution", "兜底搜索与热门结果方案", "search.md", "方案"),
        ],
        query_result=[
            _hit("background-1", 0.90, "search.md"),
            _hit("background-2", 0.88, "search.md"),
            _hit("solution", 0.86, "search.md"),
        ],
    )

    result = _retriever(
        store,
        context_top_k=2,
        min_similarity=0.60,
        lexical_reserve_k=0,
        dense_reserve_k=2,
    ).retrieve_with_trace("无结果场景如何处理", [1.0])

    assert [chunk.block_id for chunk in result.chunks] == ["background-1", "solution"]
    solution = next(
        candidate for candidate in result.trace.candidates if candidate.block_id == "solution"
    )
    assert solution.selection_reason == "dense_reserve"


def test_trace_is_opt_in_and_never_contains_text_or_path() -> None:
    """Trace 只暴露 opaque ID 与数值信号，不泄露正文、查询或路径。"""

    secret_text = "CONFIDENTIAL-CONTENT-987"
    secret_path = "C:/private/client/acquisition.md"
    store = FakeVectorStore(
        blocks=[_block("opaque-1", secret_text, secret_path)],
        query_result=[_hit("opaque-1", 0.90, secret_path)],
    )
    retriever = _retriever(store, min_similarity=0.60)
    result = retriever.retrieve_with_trace("private query phrase", [1.0, 0.0])

    payload = json.dumps(result.trace.to_dict(), ensure_ascii=False)
    assert "opaque-1" in payload
    assert secret_text not in payload
    assert secret_path not in payload
    assert "private query phrase" not in payload


def test_bm25_candidate_cannot_bypass_where_filter() -> None:
    """BM25 在内存中应用与 dense 相同的元数据过滤，避免跨平台内容泄漏。"""

    private_block = _block("private", "唯一术语 SPR-PRIVATE-999", "private.md")
    private_block[2]["platform"] = "private"
    public_block = _block("public", "公开缓存说明", "public.md")
    store = FakeVectorStore(blocks=[private_block, public_block], query_result=[])
    retriever = _retriever(store, min_similarity=0.60)

    chunks = retriever.retrieve(
        "SPR-PRIVATE-999", [1.0, 0.0], where={"platform": "local"}
    )
    assert chunks == []


def test_same_document_expansion_pulls_in_siblings_below_threshold() -> None:
    """文档级扩展：某文档一块过门禁后，同文档相似度低于阈值的兄弟块也一并召回；
    无关文档（无锚点）不进上下文。修复「问整篇文档只召回 1 块」的问题。"""
    store = FakeVectorStore(
        blocks=[
            _block("c1", "本书第一章 核心观点", "book.md", "第一章"),
            _block("c2", "本书第二章 方法论", "book.md", "第二章"),
            _block("c3", "本书第三章 案例", "book.md", "第三章"),
            _block("other", "无关 文档 内容", "other.md"),
        ],
        query_result=[
            _hit("c1", 0.80, "book.md"),
            _hit("c2", 0.50, "book.md"),  # 低于阈值，但同文档
            _hit("c3", 0.45, "book.md"),  # 低于阈值，但同文档
            _hit("other", 0.40, "other.md"),
        ],
    )
    retriever = _retriever(store, min_similarity=0.55, context_top_k=8)
    chunks = retriever.retrieve("这本书讲了什么", [1.0, 0.0, 0.0])
    ids = [chunk.block_id for chunk in chunks]
    assert "c1" in ids
    assert "c2" in ids  # 同文档兄弟块被扩展召回（即使低于阈值）
    assert "c3" in ids
    assert "other" not in ids  # 无关文档不进上下文


def test_one_source_cannot_fill_more_than_three_context_slots() -> None:
    blocks = [
        _block(f"same-{index}", f"缓存证据 {index}", "oversized.md", f"章节 {index}")
        for index in range(5)
    ]
    store = FakeVectorStore(
        blocks=blocks,
        query_result=[
            _hit(f"same-{index}", 0.90 - index * 0.02, "oversized.md")
            for index in range(5)
        ],
    )

    chunks = _retriever(
        store,
        min_similarity=0.60,
        context_top_k=8,
        max_chunks_per_source=3,
    ).retrieve("缓存证据", [1.0])

    assert len(chunks) == 3
    assert {chunk.source_file for chunk in chunks} == {"oversized.md"}


def test_anchors_prioritized_before_siblings() -> None:
    """锚点块优先于兄弟块：跨文档时，各文档的锚点先进上下文，兄弟块只补剩余名额。"""
    store = FakeVectorStore(
        blocks=[
            _block("a1", "文档A 的锚点 块", "a.md"),
            _block("a2", "文档A 的其它 内容", "a.md"),
            _block("b1", "文档B 的锚点 块", "b.md"),
            _block("b2", "文档B 的其它 内容", "b.md"),
        ],
        query_result=[
            _hit("a1", 0.90, "a.md"),
            _hit("a2", 0.40, "a.md"),
            _hit("b1", 0.80, "b.md"),
            _hit("b2", 0.35, "b.md"),
        ],
    )
    retriever = _retriever(store, min_similarity=0.55, context_top_k=2)
    chunks = retriever.retrieve("A B", [1.0, 0.0, 0.0])
    ids = [chunk.block_id for chunk in chunks]
    # 名额只有 2：两个文档的锚点 a1、b1 优先，兄弟块 a2/b2 不挤出锚点
    assert ids == ["a1", "b1"]


def test_full_anchor_pool_reserves_one_parent_expansion_slot() -> None:
    """8 个门禁锚点也不能让强父节的补充块永远没有扩展席位。"""

    anchor_blocks = [
        _block(f"anchor-{index}", f"独立锚点 {index}", f"doc-{index}.md", "独立章节")
        for index in range(1, 8)
    ]
    store = FakeVectorStore(
        blocks=[
            _block("parent", "父节核心证据", "parent.md", "父节 > 核心"),
            *anchor_blocks,
            _block("sibling", "父节补充证据", "parent.md", "父节 > 补充"),
        ],
        query_result=[
            _hit("parent", 0.95, "parent.md"),
            *[
                _hit(f"anchor-{index}", 0.90 - index * 0.01, f"doc-{index}.md")
                for index in range(1, 8)
            ],
            _hit("sibling", 0.40, "parent.md"),
        ],
    )

    result = _retriever(store, min_similarity=0.60, context_top_k=8).retrieve_with_trace(
        "完全不同的查询词", [1.0]
    )
    ids = [chunk.block_id for chunk in result.chunks]

    assert len(ids) == 8
    assert "parent" in ids
    assert "sibling" in ids
    sibling_trace = next(item for item in result.trace.candidates if item.block_id == "sibling")
    assert sibling_trace.selection_reason == "parent_section_expansion"


def test_weak_anchor_does_not_trigger_expansion() -> None:
    """弱相关命中（0.60~0.68）只返回自身，不把同文档的无关兄弟块拖进来。"""
    store = FakeVectorStore(
        blocks=[
            _block("w1", "如何评估 RAG 效果", "AI八股文.md"),
            _block("w2", "Java 垃圾回收机制", "AI八股文.md"),  # 同文档但无关
            _block("w3", "缓存 设计", "cache.md"),
        ],
        query_result=[
            _hit("w1", 0.66, "AI八股文.md"),  # 弱锚点：过了 0.60 门禁，但低于 0.68 强阈值
            _hit("w2", 0.40, "AI八股文.md"),
            _hit("w3", 0.50, "cache.md"),
        ],
    )
    retriever = _retriever(store, min_similarity=0.60, context_top_k=8)
    chunks = retriever.retrieve("Everything RAG 是做什么的", [1.0, 0.0, 0.0])
    ids = [chunk.block_id for chunk in chunks]
    assert ids == ["w1"]  # 只返回弱锚点自身，w2/w3 不进上下文


def test_parent_child_expands_to_parent_section_not_whole_doc() -> None:
    """父子检索：命中块所在父节（同一上级标题）的兄弟块被展开，跨父节的块不拖入。"""
    store = FakeVectorStore(
        blocks=[
            _block("p1", "父节A 的子块1 详细内容", "a.md", "父节A > 子1"),
            _block("p2", "父节A 的子块2 详细内容", "a.md", "父节A > 子2"),
            _block("q1", "父节B 的子块 完全无关", "a.md", "父节B > 子1"),
        ],
        query_result=[
            _hit("p1", 0.90, "a.md"),
            _hit("p2", 0.40, "a.md"),
            _hit("q1", 0.45, "a.md"),
        ],
    )
    retriever = _retriever(store, min_similarity=0.60, context_top_k=8)
    chunks = retriever.retrieve("父节A", [1.0, 0.0, 0.0])
    ids = [chunk.block_id for chunk in chunks]
    assert "p1" in ids  # 锚点
    assert "p2" in ids  # 同父节（父节A）的兄弟块被展开
    assert "q1" not in ids  # 跨父节（父节B）不拖入


def test_dense_only_chunk_requires_higher_similarity() -> None:
    """纯 dense 命中（无词法匹配）的块需更高相似度，过滤共享泛词的假阳性。"""
    store = FakeVectorStore(
        blocks=[
            _block("r1", "首次触点模型 归因方式 末次触点", "归因.md"),
            _block("n1", "首单加深离奇的问题", "首单.md"),
        ],
        query_result=[
            _hit("r1", 0.60, "归因.md"),  # 词法命中，0.60 过救援门槛
            _hit("n1", 0.64, "首单.md"),  # 仅 dense，0.64 < 0.65 → 过滤
        ],
    )
    retriever = _retriever(store, min_similarity=0.60, context_top_k=8)
    chunks = retriever.retrieve("什么是首次触点末次触点归因方式", [1.0, 0.0, 0.0])
    ids = [chunk.block_id for chunk in chunks]
    assert "r1" in ids  # 词法命中，救援
    assert "n1" not in ids  # 仅 dense，加罚后过滤


def test_no_bm25_and_low_sim_returns_empty() -> None:
    """无关查询：无 BM25 命中 + dense 全低于阈值 -> 返回 []（门禁拦截）。"""
    store = FakeVectorStore(
        blocks=[_block("g", "量子 物理 科普", "q.md")],
        query_result=[_hit("g", 0.45, "q.md")],
    )
    retriever = _retriever(store, min_similarity=0.55)
    assert retriever.retrieve("火星殖民地的生态系统", [1.0, 0.0, 0.0]) == []


# ---------------------------------------------------------------- 4. 守卫


@pytest.mark.parametrize("query_vector", [[], [0.0], [0.0, 0.0]])
def test_empty_query_raises_clear_error(query_vector: list[float]) -> None:
    store = FakeVectorStore(blocks=[_block("a", "内容", "a.md")])
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="空"):
        retriever.retrieve("问题", query_vector)
    assert store.query_calls == []


def test_unindexed_raises_clear_error() -> None:
    store = FakeVectorStore(blocks=[], count=0)
    retriever = _retriever(store)
    with pytest.raises(RetrievalError, match="索引"):
        retriever.retrieve("问题", [1.0, 0.0, 0.0])
    assert store.query_calls == []


def test_no_hits_returns_empty() -> None:
    store = FakeVectorStore(blocks=[_block("a", "内容", "a.md")], query_result=[])
    retriever = _retriever(store)
    assert retriever.retrieve("问题", [1.0, 0.0, 0.0]) == []


def test_returns_retrieved_chunk_with_metadata() -> None:
    store = FakeVectorStore(
        blocks=[_block("b1", "文本 内容", "a.md", "标题")],
        query_result=[_hit("b1", 0.9, "a.md")],
    )
    retriever = _retriever(store)
    chunk = retriever.retrieve("文本", [1.0, 0.0, 0.0])[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.block_id == "b1"
    assert chunk.source_file == "a.md"
    assert chunk.heading_path == "标题"


def test_identifier_query_rejects_generic_lexical_hits_without_identifier() -> None:
    """Warebase 问题不能用“公司/事故/介绍”等泛词召回其他文档。"""

    store = FakeVectorStore(
        blocks=[
            _block("relevant", "Warebase 迁移后出现定时任务性能事故", "Warebase记录.md"),
            _block("noise", "公司事故复盘与背景介绍", "其他公司.md"),
        ],
        query_result=[_hit("relevant", 0.50, "Warebase记录.md")],
    )
    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(
        "简单介绍一下warebase， 有没有什么事故出现在warebase身上",
        [1.0],
    )

    assert [chunk.block_id for chunk in result.chunks] == ["relevant"]
    noise = next(candidate for candidate in result.trace.candidates if candidate.block_id == "noise")
    assert noise.gate_reason == "rejected_missing_dense_weak_lexical"


def test_exact_identifier_does_not_override_contradictory_dense_score() -> None:
    store = FakeVectorStore(
        blocks=[_block("mention", "ADB 是一个历史数据系统", "历史系统.md")],
        query_result=[_hit("mention", 0.31, "历史系统.md")],
    )
    result = _retriever(store, min_similarity=0.60).retrieve_with_trace("ADB是什么", [1.0])

    assert result.chunks == ()
    assert result.trace.candidates[0].gate_reason == "rejected_lexical_dense_contradiction"


def test_two_term_lexical_overlap_below_refinement_floor_is_not_used_as_fill() -> None:
    store = FakeVectorStore(
        blocks=[_block("mention", "Redis key 问题的一般说明", "缓存.md")],
        query_result=[_hit("mention", 0.49, "缓存.md")],
    )

    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(
        "Redis 大 key 热 key 问题要怎么解决？", [1.0]
    )

    assert result.chunks == ()
    assert result.trace.candidates[0].gate_reason == "rejected_low_quality_lexical_fill"


def test_expansion_does_not_reintroduce_a_hard_vetoed_candidate() -> None:
    strong = _block("strong", "Redis 热 key 的处理方案", "缓存.md", "Redis > 性能")
    weak = _block("weak", "Redis key 问题的一般说明", "缓存.md", "Redis > 集群")
    store = FakeVectorStore(
        blocks=[strong, weak],
        query_result=[
            _hit("strong", 0.80, "缓存.md"),
            _hit("weak", 0.49, "缓存.md"),
        ],
    )

    result = _retriever(store, min_similarity=0.60).retrieve_with_trace(
        "Redis 大 key 热 key 问题要怎么解决？", [1.0]
    )

    assert [chunk.block_id for chunk in result.chunks] == ["strong"]
    weak_trace = next(item for item in result.trace.candidates if item.block_id == "weak")
    assert weak_trace.gate_reason == "rejected_low_quality_lexical_fill"
    assert weak_trace.selected is False


def test_identifier_definition_query_softly_demotes_missing_definition_evidence() -> None:
    store = FakeVectorStore(
        blocks=[_block("mention", "ERP 从 ADB 迁移到了 Warebase", "迁移记录.md")],
        query_result=[_hit("mention", 0.75, "迁移记录.md")],
    )
    result = _retriever(store, min_similarity=0.60).retrieve_with_trace("什么是adb", [1.0])

    assert [chunk.block_id for chunk in result.chunks] == ["mention"]
    assert result.trace.candidates[0].gate_reason.endswith("_definition_fallback")


def test_definition_subject_route_recalls_parenthetical_definition_without_extra_dense_route() -> None:
    definition = _block(
        "definition",
        "不得不说 MVCC（多版本并发控制）解决了读写互斥问题。",
        "加字段？锁表？.md",
        "并发控制",
    )
    store = FakeVectorStore(
        blocks=[definition, _block("noise", "数据库字段变更说明", "变更记录.md")],
        query_result=[_hit("noise", 0.62, "变更记录.md")],
    )

    result = _retriever(store, min_similarity=0.60, bm25_k=1).retrieve_with_trace(
        "什么是 MVCC？", [1.0]
    )

    assert result.trace.variant_kinds == ("original", "definition_subject")
    assert result.chunks[0].block_id == "definition"
    assert result.chunks[0].match_type == "lexical"


def test_identifier_definition_query_accepts_explicit_definition() -> None:
    store = FakeVectorStore(
        blocks=[_block("definition", "ADB 是一种分析型数据库。", "ADB简介.md")],
        query_result=[_hit("definition", 0.75, "ADB简介.md")],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve("什么是adb", [1.0])
    assert [chunk.block_id for chunk in chunks] == ["definition"]


def test_identifier_definition_query_accepts_bracketed_expansion() -> None:
    store = FakeVectorStore(
        blocks=[_block("definition", "AAB【Android APP Bundle】是一种发布格式。", "AAB.md")],
        query_result=[_hit("definition", 0.36, "AAB.md")],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve("请问一下 AAB 是什么？", [1.0])

    assert [chunk.block_id for chunk in chunks] == ["definition"]


def test_identifier_only_overlap_cannot_rescue_wrong_compound_intent() -> None:
    store = FakeVectorStore(
        blocks=[
            _block(
                "troubleshooting",
                "登录 K8S 机器查看日志，然后把 GET 改成 POST。",
                "排查bug的真实案例.md",
                "RPC 400 排查",
            ),
            _block(
                "incident",
                "K8S 灰度路由异常造成商品服务无法访问。",
                "K8S路由异常.md",
                "事故复盘",
            ),
        ],
        query_result=[],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve(
        "在知识库中，之前的 K8S 发生了什么事故", [1.0]
    )

    assert [chunk.block_id for chunk in chunks] == ["incident"]


def test_short_chunk_is_hydrated_with_same_parent_section_context() -> None:
    store = FakeVectorStore(
        blocks=[
            _block(
                "short",
                "第一天正常；第二天耗时变为三倍。",
                "Warebase记录.md",
                "1. 发现问题 > 问题现象",
            ),
            _block(
                "sibling",
                "ERP 业务原本运行在 ADB，五月底迁移到 Warebase；切换当天日志正常，次日定时任务开始明显变慢。",
                "Warebase记录.md",
                "1. 发现问题 > 初始状态",
            ),
            _block("other", "不应拼接的其他文档内容。", "其他.md", "1. 发现问题 > 初始状态"),
        ],
        query_result=[_hit("short", 0.70, "Warebase记录.md")],
    )

    chunks = _retriever(store, min_similarity=0.60).retrieve("Warebase 定时任务变慢", [1.0])
    short = next(chunk for chunk in chunks if chunk.block_id == "short")
    assert short.text == "第一天正常；第二天耗时变为三倍。"
    assert short.context_text is not None
    assert "问题现象" in short.context_text
    assert "ERP 业务原本运行在 ADB" in short.context_text
    assert "其他文档" not in short.context_text


def test_exact_heading_image_context_is_attached_without_consuming_another_slot() -> None:
    text_block = _block(
        "risk-text",
        "新加坡接口阈值调整后，CTR 从 358% 降到 30+%；同时降低商品详情、分页和"
        "收藏接口阈值，并结合误封率持续观察。",
        "行为风控数据分析.md",
        "2024-04-18 > 新加坡爬虫",
    )
    image_block = _block(
        "risk-image",
        "【图片描述】阈值调整前后 CTR 曲线及请求量对比。",
        "行为风控数据分析.md",
        "2024-04-18 > 新加坡爬虫",
    )
    image_block[2]["chunk_type"] = "image_description"
    image_block[2]["source_type"] = "image_description"
    unrelated_image = _block(
        "other-image",
        "【图片描述】另一个日期的图。",
        "行为风控数据分析.md",
        "2024-05-01 > 其他事项",
    )
    unrelated_image[2]["chunk_type"] = "image_description"
    unrelated_image[2]["source_type"] = "image_description"
    store = FakeVectorStore(
        blocks=[text_block, image_block, unrelated_image],
        query_result=[
            _hit("risk-text", 0.76, "行为风控数据分析.md"),
            _hit("risk-image", 0.30, "行为风控数据分析.md"),
            _hit("other-image", 0.29, "行为风控数据分析.md"),
        ],
    )

    result = _retriever(store, min_similarity=0.60, context_top_k=3).retrieve_with_trace(
        "风控优化99分位", [1.0]
    )

    assert [chunk.block_id for chunk in result.chunks] == ["risk-text"]
    assert result.chunks[0].context_text is not None
    assert "同标题图片信息" in result.chunks[0].context_text
    assert "CTR 曲线" in result.chunks[0].context_text


# ---------------------------------------------------------------- 5. 零出网


def test_zero_outbound_full_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected outbound network call")

    monkeypatch.setattr(socket.socket, "connect", deny)
    store = FakeVectorStore(blocks=_TITLE_BLOCKS, query_result=_TITLE_DENSE)
    retriever = _retriever(store)
    chunks = retriever.retrieve("实时风控方案", [1.0, 0.0, 0.0])
    assert chunks
