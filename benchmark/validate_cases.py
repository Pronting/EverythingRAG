"""校验每条用例的证据串是否真实存在于 gold chunk 原文中。

用例（问题原文 + 证据串）与块样本都是**私人正文**，只保存在 Git 忽略的本地目录
（PRD §7.2），本脚本经 ``cases_loader`` 组装后校验。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases_loader

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub("", s).lower()


try:
    cases = cases_loader.load_cases()
    samples = {s["n"]: s for s in cases_loader.load_sample_chunks()}
except cases_loader.PrivateCasesMissing as exc:
    print(f"[skip] {exc}")
    raise SystemExit(2) from exc

bad = 0
for c in cases:
    if not c["gold"]:
        continue
    missing = [n for n in c["gold"] if n not in samples]
    if missing:
        print(f"[MISS] {c['id']}  gold 块序号不在样本里：{missing}")
        bad += 1
        continue
    texts = [norm(samples[n]["text"]) for n in c["gold"]]
    joined = " ".join(texts)
    for ev in c["evidence"]:
        if norm(ev) not in joined:
            bad += 1
            print(f"[MISS] {c['id']}  evidence=…(len={len(ev)})  not in gold={c['gold']}")
            print(f"        gold 块 n={c['gold'][0]}（来源/标题见本地私文）")

print(f"\n用例总数={len(cases)}  证据缺失={bad}")
pos = [c for c in cases if c["gold"]]
neg = [c for c in cases if not c["gold"]]
print(f"正例={len(pos)}  负例={len(neg)}")
print("类别分布:", dict(Counter(c["cat"] for c in cases)))
print("难度分布:", dict(Counter(c["diff"] for c in cases)))
