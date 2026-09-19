"""汇总检索基准：按路径/类别/难度/阈值输出能力表。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "benchmark"))

import cases_loader

parser = argparse.ArgumentParser(description="汇总 Everything RAG 检索基准")
parser.add_argument(
    "--input",
    type=Path,
    default=cases_loader.artifact_path("retrieval_results.json"),
)
parser.add_argument(
    "--output",
    type=Path,
    default=BASE / "benchmark/aggregate.json",
)
args = parser.parse_args()
input_path = args.input if args.input.is_absolute() else BASE / args.input
output_path = args.output if args.output.is_absolute() else BASE / args.output
rows = json.loads(input_path.read_text(encoding="utf-8"))
KS = ("1", "3", "5", "8", "10")

pos = [r for r in rows if r["n_gold"] > 0]
neg = [r for r in rows if r["n_gold"] == 0]
pure_neg = [r for r in neg if r["cat"] == "negative"]
adv_neg = [r for r in neg if r["cat"] == "negative_adversarial"]


def agg(subset, route, key):
    vals = [r[route][key] for r in subset if r[route].get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def hits(subset, route, k):
    return round(sum(r[route]["hit"][k] for r in subset) / len(subset), 4) if subset else None


def recall(subset, route, k):
    vals = [r[route]["recall"][k] for r in subset if r[route].get("recall", {}).get(k) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


out = {}
print("=" * 78)
print(f"A. 三路召回对比（正例 {len(pos)} 条）")
print("=" * 78)
print(f"{'路径':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@8':>8}{'hit@10':>10}{'MRR':>8}{'nDCG@8':>8}{'证据覆盖@8':>10}")
for route in ("hybrid", "dense", "bm25"):
    rk = "recall" if route == "hybrid" else "hit"
    print(f"{route:<10}" + "".join(f"{hits(pos, route, k):>8.3f}" for k in KS)
          + f"{agg(pos, route, 'rr'):>8.3f}{agg(pos, route, 'ndcg8'):>8.3f}"
          + f"{agg(pos, route, 'ev_cov_raw'):>10.3f}")
    out[route] = {"hit": {k: hits(pos, route, k) for k in KS},
                  "mrr": agg(pos, route, "rr"), "ndcg8": agg(pos, route, "ndcg8"),
                  "ev_cov": agg(pos, route, "ev_cov_raw")}

print("\n" + "=" * 78)
print("B. hybrid 按查询类别（hit@5 / MRR / 证据覆盖@8 / 上下文污染@8）")
print("=" * 78)
by_cat = defaultdict(list)
for r in pos:
    by_cat[r["cat"]].append(r)
print(f"{'类别':<20}{'n':>4}{'hit@1':>8}{'hit@5':>8}{'hit@8':>8}{'MRR':>8}{'证据覆盖':>9}{'污染率':>8}")
for cat, sub in sorted(by_cat.items()):
    print(f"{cat:<20}{len(sub):>4}{hits(sub,'hybrid','1'):>8.3f}{hits(sub,'hybrid','5'):>8.3f}"
          f"{hits(sub,'hybrid','8'):>8.3f}{agg(sub,'hybrid','rr'):>8.3f}"
          f"{agg(sub,'hybrid','ev_cov_ctx'):>9.3f}{agg(sub,'hybrid','pollution'):>8.3f}")
    out.setdefault("by_cat", {})[cat] = {"n": len(sub), "hit5": hits(sub, "hybrid", "5"),
                                         "mrr": agg(sub, "hybrid", "rr"),
                                         "ev_cov": agg(sub, "hybrid", "ev_cov_ctx"),
                                         "pollution": agg(sub, "hybrid", "pollution")}

print("\n" + "=" * 78)
print("C. hybrid 按难度")
print("=" * 78)
by_diff = defaultdict(list)
for r in pos:
    by_diff[r["diff"]].append(r)
print(f"{'难度':<12}{'n':>4}{'hit@1':>8}{'hit@5':>8}{'MRR':>8}{'证据覆盖':>9}{'污染率':>8}")
for d in ("easy", "medium", "hard"):
    if d not in by_diff:
        continue
    sub = by_diff[d]
    print(f"{d:<12}{len(sub):>4}{hits(sub,'hybrid','1'):>8.3f}{hits(sub,'hybrid','5'):>8.3f}"
          f"{agg(sub,'hybrid','rr'):>8.3f}{agg(sub,'hybrid','ev_cov_ctx'):>9.3f}"
          f"{agg(sub,'hybrid','pollution'):>8.3f}")
    out.setdefault("by_diff", {})[d] = {"n": len(sub), "hit5": hits(sub, "hybrid", "5"),
                                        "mrr": agg(sub, "hybrid", "rr"),
                                        "ev_cov": agg(sub, "hybrid", "ev_cov_ctx")}

print("\n" + "=" * 78)
print("D. min_similarity 阈值消融（正例命中 vs 负样本拒答）")
print("=" * 78)
print(f"{'阈值':<8}{'hit@5':>8}{'证据覆盖':>10}{'召回块数均值':>12}{'纯负样本拒答率':>14}{'召回=0占比':>10}")
available_thresholds = [
    threshold
    for threshold in ("none", "0.50", "0.55", "0.60", "0.65", "0.70")
    if rows and threshold in rows[0]["ablation"]
]
for thr in available_thresholds:
    h5 = sum(r["ablation"][thr]["hit5"] for r in pos) / len(pos)
    ev = [r["ablation"][thr]["ev_cov_ctx"] for r in pos if r["ablation"][thr]["ev_cov_ctx"] is not None]
    nout = statistics.mean(r["ablation"][thr]["n_out"] for r in rows)
    rej_pure = sum(1 for r in pure_neg if r["ablation"][thr]["n_out"] == 0) / len(pure_neg)
    zero_all = sum(1 for r in rows if r["ablation"][thr]["n_out"] == 0) / len(rows)
    print(f"{thr:<8}{h5:>8.3f}{sum(ev)/len(ev):>10.3f}{nout:>12.2f}{rej_pure:>14.3f}{zero_all:>10.3f}")
    out.setdefault("ablation", {})[thr] = {"hit5": round(h5, 4),
                                           "ev_cov": round(sum(ev) / len(ev), 4),
                                           "mean_n_out": round(nout, 2),
                                           "neg_reject": round(rej_pure, 4),
                                           "zero_all_rate": round(zero_all, 4)}

print("\n" + "=" * 78)
print("E. 负样本行为（库外问题，应少召回/不召回）")
print("=" * 78)
print(f"{'id':<6}{'类别':<22}{'召回块数':>8}{'top1相似度':>12}  问题")
for r in neg:
    print(f"{r['id']:<6}{r['cat']:<22}{r['hybrid']['n_out']:>8}"
          f"{r['hybrid']['top1_sim']!s:>12}  {r['query'][:36]}")
reject_rate = sum(1 for r in pure_neg if r["hybrid"]["n_out"] == 0) / len(pure_neg)
leak = [r for r in pure_neg if r["hybrid"]["n_out"] > 0]
print(f"\n纯库外负样本拒答率（召回=0）: {reject_rate:.1%}  ({len(pure_neg)-len(leak)}/{len(pure_neg)})")
if leak:
    sims = [r["hybrid"]["top1_sim"] for r in leak if r["hybrid"]["top1_sim"] is not None]
    if sims:
        print(f"漏过门控的 {len(leak)} 条，top1 相似度 {min(sims):.3f}~{max(sims):.3f} 均值 {statistics.mean(sims):.3f}")
    else:
        print(f"漏过门控的 {len(leak)} 条，均为纯词法救援（BM25-only，无 dense 相似度）")
    for r in leak:
        print(f"   {r['id']} n_out={r['hybrid']['n_out']}  {r['query'][:40]}")
out["negative"] = {"pure_count": len(pure_neg), "reject_rate": round(reject_rate, 4),
                   "leaked": len(leak),
                   "leak_top1_sim_mean": round(statistics.mean(sims), 4) if leak and sims else None}

print("\n" + "=" * 78)
print("F. 质量与效率")
print("=" * 78)
print(f"hybrid 上下文污染率@8      : {agg(pos,'hybrid','pollution'):.1%}（top8 中非黄金来源文件占比）")
print(f"hybrid top8 重复块率       : {agg(pos,'hybrid','dup'):.1%}")
print(f"证据完全覆盖(=1.0)比例     : "
      f"{sum(1 for r in pos if (r['hybrid']['ev_cov_ctx'] or 0) >= 0.999)/len(pos):.1%}")
print(f"证据零覆盖(=0.0)比例       : "
      f"{sum(1 for r in pos if (r['hybrid']['ev_cov_ctx'] or 0) == 0)/len(pos):.1%}")
lat_h = [r["hybrid"]["latency_ms"] for r in rows]
lat_d = [r["dense"]["latency_ms"] for r in rows]
lat_b = [r["bm25"]["latency_ms"] for r in rows]
print(f"延迟 hybrid 中位/均值      : {statistics.median(lat_h):.0f} / {statistics.mean(lat_h):.0f} ms")
print(f"延迟 dense  中位/均值      : {statistics.median(lat_d):.0f} / {statistics.mean(lat_d):.0f} ms")
print(f"延迟 bm25   中位/均值      : {statistics.median(lat_b):.0f} / {statistics.mean(lat_b):.0f} ms")
out["quality"] = {
    "pollution": agg(pos, "hybrid", "pollution"),
    "dup": agg(pos, "hybrid", "dup"),
    "ev_full": round(sum(1 for r in pos if (r["hybrid"]["ev_cov_ctx"] or 0) >= 0.999) / len(pos), 4),
    "ev_zero": round(sum(1 for r in pos if (r["hybrid"]["ev_cov_ctx"] or 0) == 0) / len(pos), 4),
    "lat_hybrid_ms_median": round(statistics.median(lat_h), 1),
    "lat_dense_ms_median": round(statistics.median(lat_d), 1),
    "lat_bm25_ms_median": round(statistics.median(lat_b), 1),
}

print("\n" + "=" * 78)
print("G. 完全未命中的用例（hit@10 = 0）")
print("=" * 78)
misses = [r for r in pos if r["hybrid"]["hit"]["10"] == 0]
for r in misses:
    print(f"  {r['id']} [{r['cat']}/{r['diff']}] {r['query'][:44]}")
    print(f"        top1_sim={r['hybrid']['top1_sim']}  gold={[Path(g).name for g in r.get('gold_files', [])]}")
print(f"总计 {len(misses)}/{len(pos)} = {len(misses)/len(pos):.1%}")
out["total_miss"] = len(misses)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n-> {output_path}")
