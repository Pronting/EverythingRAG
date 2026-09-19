"""校验每条用例的证据串是否真实存在于 gold chunk 原文中。"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"C:\Users\22331\Desktop\everything-rag")

cases = json.loads((BASE / "benchmark/cases.json").read_text(encoding="utf-8"))["cases"]
samples = {s["n"]: s for s in json.loads((BASE / ".tmp_probe/sample_chunks.json").read_text(encoding="utf-8"))}

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub("", s).lower()


bad = 0
for c in cases:
    if not c["gold"]:
        continue
    texts = [norm(samples[n]["text"]) for n in c["gold"]]
    joined = " ".join(texts)
    for ev in c["evidence"]:
        if norm(ev) not in joined:
            bad += 1
            print(f"[MISS] {c['id']}  evidence={ev!r}  not in gold={c['gold']}")
            print(f"        gold file: {samples[c['gold'][0]]['file']} :: {samples[c['gold'][0]]['heading']}")

print(f"\n用例总数={len(cases)}  证据缺失={bad}")
pos = [c for c in cases if c["gold"]]
neg = [c for c in cases if not c["gold"]]
print(f"正例={len(pos)}  负例={len(neg)}")
from collections import Counter
print("类别分布:", dict(Counter(c["cat"] for c in cases)))
print("难度分布:", dict(Counter(c["diff"] for c in cases)))
