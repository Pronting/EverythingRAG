"""BM25 混合检索离线原型（一次性验证，不改生产代码）。

对同一批评测问题，对比三种策略在 hit@5 上的差异：
  A. 现状：纯向量 top-5
  B. 混合：向量 top-20 ∪ BM25 top-50，RRF 融合后取 top-5
  C. B + 相似度门禁（top1 sim 阈值，观察 precision 行为）

BM25 用字符 bigram 切词（无 jieba 依赖，中文近似可接受）。
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

POSITIVE = [
    ("「兵者，诡道也。故能而示之不能，用而示之不用」这句话出自哪部书？", "孙子兵法"),
    ("「上兵伐谋，其次伐交，其次伐兵，其下攻城」出自哪里？", "孙子兵法"),
    ("「不战而屈人之兵，善之善者也」这句话的出处？", "孙子兵法"),
    ("「春风得意马蹄疾，一日看尽长安花」是哪位诗人的名句？", "诗句收集"),
    ("罗曼罗兰说，世界上只有一种英雄主义，是什么？", "诗句收集"),
    ("中国的行政区划为什么按需求相似的人划分成区域？", "中国行政区域划分标准"),
    ("投递时 recruitment@ratingdog.cn 要求邮件标题格式是什么？", "春招邮箱"),
    ("中国政府官僚体系运转的两根传动轴是什么？", "中国政府的官僚体系详解"),
    ("为什么说谁离信息最近，谁就拥有决策优势？", "中国政府的官僚体系详解"),
    ("「富在术数，不在劳身，利在势居，不在力耕」出自哪部书？", "毛选"),
    ("Canal 和阿里云 DTS 这类工具在同步 DDL 表结构变更时有什么局限？", "8000万数据表"),
    ("自建 RocketMQ 时 nameserver 应该部署几台、如何分布在可用区？", "RocketMQ 迁移"),
    ("2024-09-05 推荐接口超时问题的最终根因是什么？", "推荐接口超时"),
    ("MySQL 的 MVCC 是用什么机制实现的？", "PostgreSQL 与 MySQL 差异"),
    ("PG 数据库的一个连接对应什么？", "PostgreSQL 与 MySQL 差异"),
    ("实时风控方案支持哪些维度的限流策略？", "实时风控方案"),
    ("商品订单创建事件里，list_id 参数在非 collection 场景下传什么值？", "商品订单创建事件"),
    ("墨西哥礼品卡支付金额错误这个事故记录在哪？", "墨西哥礼品卡支付金额错误"),
]

NEGATIVE = [
    "量子纠缠在量子计算退相干控制中的应用",
    "火星殖民地生态系统如何实现自维持",
    "巴赫平均律中的对位法与和声分析",
    "线粒体 DNA 在细胞衰老中的作用机制",
]


def tokenize(text: str) -> list[str]:
    """轻量中文切词：字母数字整体 + CJK 单字 + 相邻字 bigram。"""
    text = text.lower()
    toks: list[str] = []
    for chunk in re.findall(r"[a-z0-9_@.\-]+|[一-鿿]+", text):
        if re.match(r"[a-z0-9_@.\-]", chunk[0]):
            toks.append(chunk)
        else:
            toks.extend(chunk)
            if len(chunk) > 1:
                toks.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return toks


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._texts = corpus
        self._doc_len = [len(tokenize(d)) for d in corpus]
        self._avgdl = sum(self._doc_len) / max(len(corpus), 1)
        df: Counter[str] = Counter()
        for d in corpus:
            df.update(set(tokenize(d)))
        n = len(corpus)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items() if c > 0}
        self._k1 = k1
        self._b = b

    def score(self, query: str, doc_idx: int) -> float:
        doc_toks = Counter(tokenize(self._texts[doc_idx]))
        dl = self._doc_len[doc_idx]
        denom = 1 - self._b + self._b * (dl / self._avgdl) if self._avgdl else 1.0
        score = 0.0
        for t in set(tokenize(query)):
            tf = doc_toks.get(t, 0)
            if tf == 0 or t not in self._idf:
                continue
            score += self._idf[t] * (tf * (self._k1 + 1)) / (tf + self._k1 * denom)
        return score


def rrf_fuse(dense_hits: list[dict], bm25_idx: list[int], top_k: int = 5, k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, h in enumerate(dense_hits, start=1):
        gi = h["_gi"]
        if gi is not None:
            scores[gi] = scores.get(gi, 0.0) + 1.0 / (k + rank)
    for rank, doc_i in enumerate(bm25_idx, start=1):
        scores[doc_i] = scores.get(doc_i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:top_k]


def main() -> None:
    settings = get_settings()
    store = get_settings_store()
    app_settings = store.load()
    embedder = create_embedder_from_config(app_settings.embed)
    vs = create_vector_store(settings.data_dir, embedder=embedder)

    # 全量拉取文本与元数据
    res = vs._collection.get(include=["documents", "metadatas"])
    texts: list[str] = res.get("documents") or []
    metas: list[dict] = res.get("metadatas") or []
    ids: list[str] = res.get("ids") or []
    id2idx = {bid: i for i, bid in enumerate(ids)}
    # 关键改进：把源文件名(去后缀)注入可搜索文本 —— 标题常是唯一带关键字的信号
    titled = [
        text + " " + Path(m.get("source_file", "")).stem
        for text, m in zip(texts, metas)
    ]
    print(f"构建 BM25 索引：{len(texts)} 块（含标题注入）")

    bm25 = BM25(titled)

    def dense_top(q: str, top_k: int) -> list[dict]:
        vec = embedder.embed_texts([q])[0]
        hits = vs.query(vec, top_k)
        for h in hits:
            h["_gi"] = id2idx.get(str(h["block_id"]))
        return hits

    dense_hits5 = dense_h20 = hybrid_hits5 = 0
    print("\n" + "=" * 100)
    print(f"{'Q':<3} {'纯向量@5':<9} {'纯向量@20':<10} {'BM25+RRF@5':<11} 问题")
    print("=" * 100)
    for idx, (q, expect) in enumerate(POSITIVE, 1):
        d20 = dense_top(q, 20)
        hit_d5 = any(expect in d["metadata"].get("source_file", "") for d in d20[:5])
        hit_d20 = any(expect in d["metadata"].get("source_file", "") for d in d20)
        # BM25 全库打分，取 top 50
        scores = [(bm25.score(q, i), i) for i in range(len(texts))]
        scores.sort(reverse=True)
        bm25_top = [i for _, i in scores[:50]]
        fused = rrf_fuse(d20, bm25_top)
        fused_files = [metas[i].get("source_file", "") for i in fused]
        hit_h5 = any(expect in f for f in fused_files)
        dense_hits5 += hit_d5
        dense_h20 += hit_d20
        hybrid_hits5 += hit_h5
        flag = ""
        if hit_h5 and not hit_d5:
            flag = "  <-- 混合修复"
        elif not hit_h5 and not hit_d5:
            flag = "  <-- 双失败(可能未导入)"
        print(f"[{idx:02d}] {'Y' if hit_d5 else 'N':<9} {'Y' if hit_d20 else 'N':<10} "
              f"{'Y' if hit_h5 else 'N':<11} {q}{flag}")

    print("=" * 100)
    n = len(POSITIVE)
    print(f"纯向量 hit@5 = {dense_hits5}/{n} = {dense_hits5/n*100:.1f}%")
    print(f"纯向量 hit@20 = {dense_h20}/{n} = {dense_h20/n*100:.1f}%")
    print(f"BM25+RRF hit@5 = {hybrid_hits5}/{n} = {hybrid_hits5/n*100:.1f}%")

    # ---- 门禁探针：正向 vs 无关 的 top1 相似度分布 ----
    print("\n门禁探针（top1 cosine 相似度分布）:")
    pos_sims, neg_sims = [], []
    for q, _ in POSITIVE:
        pos_sims.append(dense_top(q, 1)[0]["similarity"])
    for q in NEGATIVE:
        neg_sims.append(dense_top(q, 1)[0]["similarity"])
    print(f"  正向 top1 sim: min={min(pos_sims):.3f} avg={sum(pos_sims)/len(pos_sims):.3f} max={max(pos_sims):.3f}")
    print(f"  无关 top1 sim: min={min(neg_sims):.3f} avg={sum(neg_sims)/len(neg_sims):.3f} max={max(neg_sims):.3f}")
    for thr in (0.45, 0.50, 0.55, 0.60):
        keep_pos = sum(s >= thr for s in pos_sims)
        keep_neg = sum(s >= thr for s in neg_sims)
        print(f"  门禁>={thr:.2f}: 保留正向 {keep_pos}/{len(pos_sims)}  漏放无关 {keep_neg}/{len(neg_sims)}")


if __name__ == "__main__":
    main()
