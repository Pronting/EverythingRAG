"""索引覆盖率 + 索引新鲜度分析。

1) 语料里有多少 md 文件真正进了索引（覆盖率）
2) 对「已索引的那批文件」用当前 chunker 重切，与索引里的实际块长对比
   —— 分布差异大说明索引是用旧版切块建的（stale）
3) 剔除 image 块后的纯正文切块质量
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:/Users/22331/Desktop/everything-rag/backend")

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.ingestion.chunker import chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

BASE = Path(r"C:\Users\22331\Desktop\everything-rag")
DOCS = Path(r"C:\Users\22331\.everything-rag\documents")


def pct(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))]


def dist(name, lens):
    n = len(lens)
    return {
        "name": name, "n": n,
        "p10": pct(lens, 10), "p25": pct(lens, 25), "median": pct(lens, 50),
        "mean": round(statistics.mean(lens), 1), "p75": pct(lens, 75),
        "p90": pct(lens, 90), "max": max(lens),
        "lt50_pct": round(sum(1 for x in lens if x < 50) / n * 100, 2),
        "lt100_pct": round(sum(1 for x in lens if x < 100) / n * 100, 2),
        "lt150_pct": round(sum(1 for x in lens if x < 150) / n * 100, 2),
    }


app_settings = get_settings_store().load()
emb = create_embedder_from_config(app_settings.embed)
vs = create_vector_store(get_settings().data_dir, embedder=emb)
blocks = list(vs.list_blocks())

indexed_files = {b[2].get("source_file", "") for b in blocks}
indexed_files = {f for f in indexed_files if f}
corpus_files = {str(p) for p in DOCS.rglob("*.md") if p.is_file()}

in_index = indexed_files & corpus_files
missing = corpus_files - indexed_files
orphan = indexed_files - corpus_files

print(f"语料 md 文件数            : {len(corpus_files)}")
print(f"索引中出现的源文件数      : {len(indexed_files)}")
print(f"  其中仍存在于语料        : {len(in_index)}")
print(f"  源文件已不在语料（孤儿）: {len(orphan)}")
print(f"语料中未被索引的文件数    : {len(missing)}  ({len(missing)/len(corpus_files)*100:.1f}%)")

print("\n--- 未索引文件 TOP 目录分布 ---")
dirs = Counter()
for f in missing:
    rel = Path(f).relative_to(DOCS)
    dirs[str(rel.parent).split("\\")[0] if rel.parent != Path(".") else "(根)"] += 1
for d, c in dirs.most_common(12):
    print(f"   {c:5d}  {d}")

print("\n--- 未索引文件样例 ---")
for f in sorted(missing)[:12]:
    print(f"   {str(Path(f).relative_to(DOCS))}")

# ---- 对已索引文件用当前 chunker 重切，对比索引实况
print("\n=== 索引新鲜度：同一批文件，索引实况 vs 当前 chunker 重切 ===")
re_len, re_n, idx_len = [], 0, []
by_file_idx = {}
for bid, text, md in blocks:
    by_file_idx.setdefault(md.get("source_file", ""), []).append(len(text))
for f in in_index:
    idx_len.extend(by_file_idx.get(f, []))

for f in sorted(in_index):
    try:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if len(text.strip()) < 20:
        continue
    try:
        chunks = chunk_document(parse_markdown(text), f)
    except Exception:
        continue
    # 只统计已进入向量库的文本类块（image 需识图后才有描述）
    for c in chunks:
        if c.kind in {"text", "table", "code"}:
            re_len.append(len(c.text))
            re_n += 1

idx_text_len = [len(b[1]) for b in blocks if b[2].get("chunk_type") != "image_description"]

d1 = dist("索引实况（文本类块）", idx_text_len)
d2 = dist("当前chunker重切（同批文件,文本类）", re_len)
for d in (d1, d2):
    print(f"  {d['name']}: n={d['n']} 中位数={d['median']} 均值={d['mean']} "
          f"<100字={d['lt100_pct']}% <150字={d['lt150_pct']}%")

# 全语料、剔除 image 后的切块分布
print("\n=== 全语料当前 chunker 重切（剔除 image 块）===")
all_len, all_img, n_img = [], [], 0
per_file_chunks = {}
for p in DOCS.rglob("*.md"):
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if len(text.strip()) < 20:
        continue
    try:
        cs = chunk_document(parse_markdown(text), str(p))
    except Exception:
        continue
    per_file_chunks[str(p)] = len(cs)
    for c in cs:
        if c.kind == "image":
            all_img.append(len(c.text))
            n_img += 1
        else:
            all_len.append(len(c.text))
d3 = dist("全语料重切（剔除 image）", all_len)
print(f"  n={d3['n']} 中位数={d3['median']} 均值={d3['mean']} "
      f"<50字={d3['lt50_pct']}% <100字={d3['lt100_pct']}% <150字={d3['lt150_pct']}%")
print(f"  image 块: n={n_img} 中位数={pct(all_img,50) if all_img else 0}")
top = sorted(per_file_chunks.items(), key=lambda x: -x[1])[:10]
print("  单文件块数 TOP10:")
for f, c in top:
    print(f"     {c:5d}  {str(Path(f).relative_to(DOCS))}")

json.dump({
    "corpus_md_files": len(corpus_files),
    "indexed_source_files": len(indexed_files),
    "indexed_and_present": len(in_index),
    "indexed_orphan": len(orphan),
    "corpus_not_indexed": len(missing),
    "index_live_text": d1,
    "rechunk_indexed_files": d2,
    "rechunk_all_no_image": d3,
    "top_docs_by_chunks": [[str(Path(f).relative_to(DOCS)), c] for f, c in top],
}, open(BASE / "benchmark/index_coverage.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=1)
print("\n-> benchmark/index_coverage.json")
