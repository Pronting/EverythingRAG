"""端到端答案质量汇总。"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"C:\Users\22331\Desktop\everything-rag")
rows = json.loads((BASE / "benchmark/answer_results.json").read_text(encoding="utf-8"))
cases = {c["id"]: c for c in json.loads((BASE / "benchmark/cases.json").read_text(encoding="utf-8"))["cases"]}

pos = [r for r in rows if cases[r["id"]]["gold"]]
neg = [r for r in rows if not cases[r["id"]]["gold"]]
pure_neg = [r for r in neg if r["cat"] == "negative"]
adv_neg = [r for r in neg if r["cat"] != "negative"]

errs = [r for r in rows if r.get("error")]
print(f"总用例 {len(rows)}，错误 {len(errs)}")
for r in errs[:5]:
    print(f"  {r['id']}: {r['error'][:80]}")

print("\n=== A. 正例：证据命中 / 来源命中 / 引用合规 ===")
ev_vals = [r["ev_hit"] for r in pos if r["ev_hit"] is not None]
src_hit = sum(1 for r in pos if r["source_hit"])
full_ev = sum(1 for r in pos if (r["ev_hit"] or 0) >= 0.999)
zero_ev = sum(1 for r in pos if r["ev_hit"] == 0)
bad_cite = [r for r in pos if r["bad_cite"]]
cited_any = [r for r in pos if r["cited"]]
print(f"平均证据命中率（答案含证据串比例）: {statistics.mean(ev_vals):.1%}")
print(f"证据完全命中 (=1.0)            : {full_ev}/{len(pos)} = {full_ev/len(pos):.1%}")
print(f"证据零命中 (=0.0)              : {zero_ev}/{len(pos)} = {zero_ev/len(pos):.1%}")
print(f"来源文件命中（引用了黄金来源）  : {src_hit}/{len(pos)} = {src_hit/len(pos):.1%}")
print(f"给出引用标注的比例             : {len(cited_any)}/{len(pos)} = {len(cited_any)/len(pos):.1%}")
print(f"虚构引用（引用不存在的 S 编号） : {len(bad_cite)}")
for r in bad_cite[:5]:
    print(f"    {r['id']}: {r['bad_cite']}")

print("\n=== B. 正例按类别（平均证据命中率）===")
by_cat = defaultdict(list)
for r in pos:
    by_cat[r["cat"]].append(r)
print(f"{'类别':<14}{'n':>4}{'平均证据命中':>12}{'来源命中':>10}{'平均来源数':>10}")
for cat, sub in sorted(by_cat.items()):
    ev = [r["ev_hit"] for r in sub if r["ev_hit"] is not None]
    print(f"{cat:<14}{len(sub):>4}{statistics.mean(ev):>12.1%}"
          f"{sum(1 for r in sub if r['source_hit'])/len(sub):>10.0%}"
          f"{statistics.mean(r['n_sources'] for r in sub):>10.1f}")

print("\n=== C. 负样本行为 ===")
print(f"{'id':<5}{'basis':<14}{'来源数':>6}  问题")
for r in neg:
    print(f"{r['id']:<5}{str(r['basis']):<14}{r['n_sources']:>6}  {r['query'][:38]}")
general = sum(1 for r in pure_neg if r["basis"] == "general")
insuff = sum(1 for r in pure_neg if r["basis"] == "insufficient")
print(f"\n纯负样本 {len(pure_neg)} 条: basis=general {general}, basis=insufficient {insuff}, 其他 {len(pure_neg)-general-insuff}")

print("\n--- 对抗性负样本（库内有相关内容）---")
for r in adv_neg:
    print(f"{r['id']:<5} basis={str(r['basis']):<14} src={r['n_sources']}  {r['query'][:40]}")

print("\n=== D. basis 分布（全部 87 条）===")
print(dict(Counter(r["basis"] for r in rows)))
print("mode 分布:", dict(Counter(r["mode"] for r in rows)))

print("\n=== E. 证据零命中的正例（回答未覆盖任何关键事实）===")
for r in pos:
    if r["ev_hit"] == 0:
        print(f"  {r['id']} [{r['cat']}] n_src={r['n_sources']} {r['query'][:44]}")
        print(f"        src_files={r['src_files'][:3]}")

print("\n=== F. 回答长度 ===")
lens = [r["answer_len"] for r in rows]
print(f"中位数 {statistics.median(lens):.0f} 字，均值 {statistics.mean(lens):.0f}，最大 {max(lens)}")

agg = {
    "n": len(rows), "errors": len(errs),
    "pos_avg_ev_hit": round(statistics.mean(ev_vals), 4),
    "pos_full_ev": full_ev, "pos_zero_ev": zero_ev, "pos_src_hit": src_hit,
    "neg_basis": dict(Counter(r["basis"] for r in neg)),
    "pure_neg_reject": round((general + insuff) / len(pure_neg), 4) if pure_neg else None,
}
(BASE / "benchmark/answer_aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n-> benchmark/answer_aggregate.json")
