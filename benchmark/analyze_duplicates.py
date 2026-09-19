"""未索引文件归因：区分「UUID 上传目录的历史重复副本」与「真正缺失的内容」。"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:/Users/22331/Desktop/everything-rag/backend")

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

BASE = Path(r"C:\Users\22331\Desktop\everything-rag")
DOCS = Path(r"C:\Users\22331\.everything-rag\documents")
UUID_SEG = __import__("re").compile(r"^[0-9a-f]{32}$")

app_settings = get_settings_store().load()
emb = create_embedder_from_config(app_settings.embed)
vs = create_vector_store(get_settings().data_dir, embedder=emb)
indexed = {b[2].get("source_file", "") for b in vs.list_blocks()}
indexed = {f for f in indexed if f}

corpus = [p for p in DOCS.rglob("*.md") if p.is_file()]
missing = [p for p in corpus if str(p) not in indexed]


def content_hash(p: Path) -> str:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()
    except Exception:
        return "ERR"


def is_uuid_dir(p: Path) -> bool:
    rel = p.relative_to(DOCS)
    return any(UUID_SEG.match(part) for part in rel.parts[:-1])


# 已索引文件的内容哈希集合
idx_hashes = defaultdict(list)
for f in indexed:
    fp = Path(f)
    if fp.exists():
        idx_hashes[content_hash(fp)].append(f)

uuid_dup, uuid_unique, real_missing, real_dup = [], [], [], []
for p in missing:
    h = content_hash(p)
    if is_uuid_dir(p):
        (uuid_dup if h in idx_hashes else uuid_unique).append(p)
    else:
        (real_dup if h in idx_hashes else real_missing).append(p)

print(f"语料 md 总数           : {len(corpus)}")
print(f"已索引                 : {len(indexed)}")
print(f"未索引                 : {len(missing)}")
print(f"  ├ UUID 上传目录重复副本: {len(uuid_dup)}   （内容已在库中，属冗余而非缺失）")
print(f"  ├ UUID 目录独有内容   : {len(uuid_unique)}   （★ 真正丢失）")
print(f"  ├ 规范目录重复副本    : {len(real_dup)}")
print(f"  └ 规范目录独有内容    : {len(real_missing)}   （★ 真正丢失）")

print(f"\n--- 规范目录下真正缺失的文件（{len(real_missing)} 个）---")
for p in sorted(real_missing)[:40]:
    try:
        size = p.stat().st_size
    except Exception:
        size = -1
    print(f"   {size:8d}B  {p.relative_to(DOCS)}")

print(f"\n--- UUID 目录中内容独有的文件（前 20 / {len(uuid_unique)}）---")
for p in sorted(uuid_unique)[:20]:
    print(f"   {p.relative_to(DOCS)}")

# 索引内的重复度：同一内容哈希出现在多个路径
print("\n--- 索引内重复内容（同一内容多个路径入库）---")
dup_groups = {h: fs for h, fs in idx_hashes.items() if len(fs) > 1}
print(f"重复内容组数: {len(dup_groups)}  涉及路径数: {sum(len(v) for v in dup_groups.values())}")
rows = sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:10]
for h, fs in rows:
    print(f"   x{len(fs)}  {Path(fs[0]).name}")
    for f in fs[:6]:
        print(f"        {f.replace(str(DOCS), '').lstrip(chr(92))}")

# UUID 目录的冗余规模
uu = [f for f in indexed if is_uuid_dir(Path(f))]
print(f"\n已索引路径中来自 UUID 上传目录的: {len(uu)} / {len(indexed)}")

json.dump({
    "corpus_md": len(corpus), "indexed": len(indexed), "missing": len(missing),
    "uuid_dir_duplicate": len(uuid_dup), "uuid_dir_unique": len(uuid_unique),
    "canonical_duplicate": len(real_dup), "canonical_unique_missing": len(real_missing),
    "missing_canonical_files": [str(p.relative_to(DOCS)) for p in sorted(real_missing)],
    "dup_content_groups": len(dup_groups),
    "indexed_from_uuid_dirs": len(uu),
}, open(BASE / "benchmark/coverage_attribution.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=1)
print("\n-> benchmark/coverage_attribution.json")
