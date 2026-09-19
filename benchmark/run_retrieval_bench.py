"""RAG 检索基准跑分器。

对评测集逐条执行三条召回路径（dense-only / BM25-only / hybrid 生产链路），
计算 hit@K、MRR、NDCG、证据覆盖、上下文污染、负样本拒答、重复率与延迟，
并对 min_similarity 做消融。

嵌入按文本缓存，同一条 query 只请求一次；逐条落盘 checkpoint。
结果写入 benchmark/retrieval_results.json。
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "backend"))
sys.path.insert(0, str(BASE / "benchmark"))

socket.setdefaulttimeout(60)

import cases_loader

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import build_query_variants
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY, RetrievalError
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

OUT = BASE / "benchmark"
LOG = BASE / ".tmp_probe/bench_progress.txt"
KS = (1, 3, 5, 8, 10)
CTX_K = 8  # 生产 context_top_k
THRESHOLDS = (None, 0.50, 0.55, 0.60, 0.65, 0.70)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Everything RAG 检索基准")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="只运行指定 case id；可重复传入，用于快速回归",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="仅运行生产阈值 0.60，跳过其余阈值消融",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "retrieval_results.json",
        help="结果文件；相对路径按项目根目录解析",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="附带前 50 个候选的脱敏门控轨迹，用于定位召回与回填问题",
    )
    return parser.parse_args()


ARGS = _parse_args()
RESULT_PATH = ARGS.output if ARGS.output.is_absolute() else BASE / ARGS.output
ACTIVE_THRESHOLDS = (0.60,) if ARGS.skip_ablation else THRESHOLDS


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        fh.flush()


def norm(s: str) -> str:
    return "".join(s.split()).lower()


# ---------------------------------------------------------------- 装载
cases = cases_loader.load_cases(OUT / "cases.json")
samples = {sample["n"]: sample for sample in cases_loader.load_sample_chunks()}
if ARGS.case_ids:
    requested = set(ARGS.case_ids)
    known = {case["id"] for case in cases}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"未知 case id: {', '.join(unknown)}")
    cases = [case for case in cases if case["id"] in requested]

app_settings = get_settings_store().load()
embedder = create_embedder_from_config(app_settings.embed)
vs = create_vector_store(get_settings().data_dir, embedder=embedder)

log("装载索引…")
base = HybridRetriever(vs, min_similarity=DEFAULT_MIN_SIMILARITY)
base._ensure_bm25()
CANON: dict[str, str] = dict(base._canonical_by_id)
BLOCKS: dict[str, tuple[str, dict]] = dict(base._blocks)
IDX2ID: dict[int, str] = dict(base._index_to_id)
BM25: BM25Index = base._bm25
log(f"块数={len(BLOCKS)} 生产阈值={DEFAULT_MIN_SIMILARITY}")

VEC: dict[str, list[float]] = {}


def embed_cached(texts: list[str]) -> list[list[float]]:
    """按文本缓存，避免同一 query 在多路/消融下重复请求远端嵌入。"""
    missing = [t for t in dict.fromkeys(texts) if t not in VEC]
    if missing:
        for attempt in range(3):
            try:
                vecs = embedder.embed_queries(missing)
                break
            except Exception as exc:  # noqa: BLE001
                log(f"  embed 重试 {attempt+1}: {type(exc).__name__} {str(exc)[:80]}")
                time.sleep(2 + attempt * 3)
        else:
            raise RuntimeError("嵌入连续失败")
        for t, v in zip(missing, vecs):
            VEC[t] = v
    return [VEC[t] for t in texts]


def canon(bid: str) -> str:
    return CANON.get(bid, bid)


def make_retriever(min_sim):
    r = HybridRetriever(vs, min_similarity=min_sim)
    r._bm25, r._blocks, r._index_to_id, r._canonical_by_id = BM25, BLOCKS, IDX2ID, CANON
    return r


RETRIEVERS = {threshold: make_retriever(threshold) for threshold in ACTIVE_THRESHOLDS}


def src_key(bid: str) -> str:
    return str(BLOCKS.get(bid, ("", {}))[1].get("source_file", ""))


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked, gold, k):
    rels = [1 if b in gold else 0 for b in ranked[:k]]
    idcg = dcg([1] * min(len(gold), k))
    return (dcg(rels) / idcg) if idcg else 0.0


def rr_of(ranked, gold):
    return next((1.0 / (r + 1) for r, b in enumerate(ranked) if b in gold), 0.0)


def ev_cov(ranked, evidence, k, ctxmap=None):
    if not evidence:
        return None
    buf = []
    for bid in ranked[:k]:
        buf.append(ctxmap[bid] if (ctxmap and bid in ctxmap) else BLOCKS.get(bid, ("", {}))[0])
    blob = norm(" ".join(buf))
    return round(sum(1 for ev in evidence if norm(ev) in blob) / len(evidence), 3)


# ---------------------------------------------------------------- 主流程
per_case: list[dict] = []
t_start = time.perf_counter()
for i, c in enumerate(cases, 1):
    q = c["query"]
    gold = [canon(samples[n]["block_id"]) for n in c["gold"]]
    gset = set(gold)
    gold_files = {src_key(g) for g in gold}
    row = {"id": c["id"], "cat": c["cat"], "diff": c["diff"], "query": q, "gold": gold, "n_gold": len(gold)}

    variants = build_query_variants(q) or ()
    vtexts = [v.text for v in variants] or [q]
    vecs = embed_cached(vtexts)

    # ---- hybrid（生产阈值）
    t0 = time.perf_counter()
    try:
        if ARGS.trace:
            retrieval_result = RETRIEVERS[0.60].retrieve_variants_with_trace(
                variants, vecs
            )
            chunks = list(retrieval_result.chunks)
        else:
            retrieval_result = None
            chunks = RETRIEVERS[0.60].retrieve_variants(variants, vecs)
    except RetrievalError:
        retrieval_result = None
        chunks = []
    dt_h = (time.perf_counter() - t0) * 1000
    ranked = [canon(ch.block_id) for ch in chunks]
    ctxmap = {canon(ch.block_id): (ch.context_text or ch.text) for ch in chunks}
    sims = [ch.similarity for ch in chunks if ch.similarity is not None]
    row["hybrid"] = {
        "n_out": len(chunks), "latency_ms": round(dt_h, 1),
        "top1_sim": round(sims[0], 4) if sims else None,
        "hit": {k: int(bool(set(ranked[:k]) & gset)) for k in KS},
        "recall": {k: round(len(set(ranked[:k]) & gset) / len(gold), 3) if gold else None for k in KS},
        "rr": round(rr_of(ranked, gset), 4),
        "ndcg8": round(ndcg_at_k(ranked, gset, CTX_K), 3),
        "ev_cov_raw": ev_cov(ranked, c["evidence"], CTX_K),
        "ev_cov_ctx": ev_cov(ranked, c["evidence"], CTX_K, ctxmap),
        "pollution": round(sum(1 for b in ranked[:CTX_K] if src_key(b) not in gold_files) / len(ranked[:CTX_K]), 3) if ranked and gold_files else None,
        "dup": round(1 - len(set(ranked[:CTX_K])) / len(ranked[:CTX_K]), 3) if ranked else 0.0,
        "top_files": [Path(src_key(b)).name for b in ranked[:CTX_K]],
        "top_block_ids": ranked[:CTX_K],
    }
    if retrieval_result is not None:
        row["hybrid"]["trace"] = [
            {
                "block_id": candidate.block_id,
                "dense_similarity": candidate.dense_similarity,
                "gate_reason": candidate.gate_reason,
                "gate_threshold": candidate.gate_threshold,
                "selected": candidate.selected,
                "selection_reason": candidate.selection_reason,
                "signals": [
                    {
                        "route": signal.route,
                        "rank": signal.rank,
                        "coverage": signal.coverage,
                        "informative_matched_terms": signal.informative_matched_terms,
                        "informative_query_terms": signal.informative_query_terms,
                        "anchor_matched_terms": signal.anchor_matched_terms,
                        "anchor_query_terms": signal.anchor_query_terms,
                    }
                    for signal in candidate.signals
                ],
            }
            for candidate in retrieval_result.trace.candidates[:50]
        ]

    # ---- dense-only（无门控，top20）
    t0 = time.perf_counter()
    hits = vs.query(vecs[0], 20)
    dt_d = (time.perf_counter() - t0) * 1000
    d_ranked = [canon(h["block_id"]) for h in hits]
    row["dense"] = {
        "latency_ms": round(dt_d, 1),
        "top1_sim": round(hits[0]["similarity"], 4) if hits else None,
        "hit": {k: int(bool(set(d_ranked[:k]) & gset)) for k in KS},
        "rr": round(rr_of(d_ranked, gset), 4),
        "ndcg8": round(ndcg_at_k(d_ranked, gset, CTX_K), 3),
        "ev_cov_raw": ev_cov(d_ranked, c["evidence"], CTX_K),
    }

    # ---- bm25-only
    t0 = time.perf_counter()
    bhits = BM25.search_detailed(q, 50)
    dt_b = (time.perf_counter() - t0) * 1000
    b_ranked = [canon(IDX2ID[h.index]) for h in bhits if h.index in IDX2ID]
    row["bm25"] = {
        "latency_ms": round(dt_b, 1),
        "top1_score": round(bhits[0].score, 3) if bhits else None,
        "hit": {k: int(bool(set(b_ranked[:k]) & gset)) for k in KS},
        "rr": round(rr_of(b_ranked, gset), 4),
        "ndcg8": round(ndcg_at_k(b_ranked, gset, CTX_K), 3),
        "ev_cov_raw": ev_cov(b_ranked, c["evidence"], CTX_K),
    }

    # ---- 阈值消融
    abl = {}
    for thr in ACTIVE_THRESHOLDS:
        try:
            ch2 = RETRIEVERS[thr].retrieve_variants(variants, vecs)
        except RetrievalError:
            ch2 = []
        r2 = [canon(x.block_id) for x in ch2]
        c2 = {canon(x.block_id): (x.context_text or x.text) for x in ch2}
        key = "none" if thr is None else f"{thr:.2f}"
        abl[key] = {
            "n_out": len(ch2),
            "hit5": int(bool(set(r2[:5]) & gset)) if gold else None,
            "ev_cov_ctx": ev_cov(r2, c["evidence"], CTX_K, c2),
        }
    row["ablation"] = abl
    per_case.append(row)

    if i % 5 == 0 or i == len(cases):
        el = time.perf_counter() - t_start
        log(f"{i}/{len(cases)}  用时{el:.0f}s  预计剩余{el/i*(len(cases)-i):.0f}s")
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(
            json.dumps(per_case, ensure_ascii=False, indent=1), encoding="utf-8")

log(f"完成 {len(per_case)} 条，总耗时 {time.perf_counter()-t_start:.0f}s")
print("done")
