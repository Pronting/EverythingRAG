"""RAG 分块质量基准：对真实语料全量重跑 chunk_document，量化切块质量。

指标覆盖：块长分布 / 碎片率 / 硬截断率 / 标题边界完整性 / 类型隔离 /
重复块 / 单文档块数倾斜 / 检索可用性代理。并做参数消融。
结果写入 benchmark/chunking_results.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "backend"))

from app.ingestion.chunker import chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.scanner import scan_directory

DOCS = Path(
    os.environ.get(
        "EVERYTHING_RAG_DOCUMENTS_DIR",
        str(Path.home() / ".everything-rag" / "documents"),
    )
)
OUT = BASE / "benchmark"

SENT_END = "。！？!?；;”’)]}》"
BULLET_START = re.compile(r"^([-*+•]|\d+[.)、]|[（(]\d+[）)])\s*")
DEFAULT_MAX = 2000
DEFAULT_MIN = 120


def pct(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100) * (len(s) - 1))))
    return s[k]


def hard_cut(text: str) -> bool:
    """块结尾既不是句末符也不是列表/标题行 —— 疑似被字符级硬切断。"""
    if text.endswith(("\n", "\r")):
        return False
    t = text.rstrip()
    if not t:
        return False
    if t[-1] in SENT_END:
        return False
    last_line = t.rsplit("\n", 1)[-1].strip()
    return not BULLET_START.match(last_line)


def analyze(chunks_by_doc: dict[str, list], tag: str, *, max_chars: int) -> dict:
    raw_chunks = [c for cs in chunks_by_doc.values() for c in cs]
    # Raw image references are pending vision tasks, not text/vector blocks.  Only a successful
    # image description is embedded by the production pipeline, so including empty image refs in
    # text-fragment metrics materially misstates the quality of the searchable index.
    allch = [c for c in raw_chunks if c.kind != "image"]
    lens = [len(c.text) for c in allch]
    n = len(lens)
    if n == 0:
        return {"tag": tag, "n_chunks": 0}

    types = Counter(c.kind for c in allch)
    docs = len(chunks_by_doc)
    per_doc = sorted((len(cs) for cs in chunks_by_doc.values()), reverse=True)
    # 块内重复（同文档完全相同的正文）
    dup_blocks = 0
    for cs in chunks_by_doc.values():
        seen = Counter(c.text.strip() for c in cs if len(c.text.strip()) > 4)
        dup_blocks += sum(v - 1 for v in seen.values() if v > 1)
    # 空/噪声块
    emptyish = sum(1 for c in allch if len(c.text.strip()) < 5)
    # 硬截断（只在较长块上统计，短块本身可能就是完整短句）
    # A complete paragraph pack can naturally end at 1800-1999 chars without terminal
    # punctuation. Fixed-width fallback pieces, by contrast, hit the configured ceiling exactly.
    # Restrict this proxy to exact-ceiling chunks so natural section endings are not mislabelled.
    limit_chunks = [c for c in allch if len(c.text) == max_chars]
    hard = sum(1 for c in limit_chunks if hard_cut(c.text))
    hard_by_type = {
        kind: {
            "ceiling": len(items),
            "hard_cut": sum(1 for chunk in items if hard_cut(chunk.text)),
        }
        for kind in sorted({chunk.kind for chunk in limit_chunks})
        if (items := [chunk for chunk in limit_chunks if chunk.kind == kind])
    }
    # 类型隔离：table/code/image 块里是否混入明显正文段落
    mixed = 0
    for c in allch:
        if c.kind in {"table", "code", "image"}:
            continue
        if "\n|" in c.text and c.text.count("|") > 6:
            mixed += 1
    # 标题覆盖
    no_heading = sum(1 for c in allch if not (c.heading_path or "").strip())
    # 超上限
    over_max = sum(1 for l in lens if l > DEFAULT_MAX)

    return {
        "tag": tag,
        "raw_chunks_including_image_tasks": len(raw_chunks),
        "pending_image_tasks": len(raw_chunks) - len(allch),
        "n_docs": docs,
        "n_chunks": n,
        "chunks_per_doc_mean": round(n / docs, 2),
        "chunks_per_doc_median": statistics.median(per_doc),
        "chunks_per_doc_max": per_doc[0],
        "top1_doc_share_pct": round(per_doc[0] / n * 100, 2),
        "top10_doc_share_pct": round(sum(per_doc[:10]) / n * 100, 2),
        "len_min": min(lens),
        "len_p10": pct(lens, 10),
        "len_p25": pct(lens, 25),
        "len_median": pct(lens, 50),
        "len_mean": round(statistics.mean(lens), 1),
        "len_p75": pct(lens, 75),
        "len_p90": pct(lens, 90),
        "len_max": max(lens),
        "frag_lt50_pct": round(sum(1 for l in lens if l < 50) / n * 100, 2),
        "frag_lt100_pct": round(sum(1 for l in lens if l < 100) / n * 100, 2),
        "frag_lt150_pct": round(sum(1 for l in lens if l < 150) / n * 100, 2),
        "frag_lt200_pct": round(sum(1 for l in lens if l < 200) / n * 100, 2),
        "empty_lt5_pct": round(emptyish / n * 100, 2),
        "over_max_pct": round(over_max / n * 100, 2),
        "hard_cut_pct_at_ceiling": (
            round(hard / len(limit_chunks) * 100, 2) if limit_chunks else 0.0
        ),
        "hard_cut_pct_all": round(hard / n * 100, 4),
        "ceiling_chunks": len(limit_chunks),
        "hard_cut_by_type": hard_by_type,
        "dup_text_blocks": dup_blocks,
        "dup_text_pct": round(dup_blocks / n * 100, 2),
        "no_heading_pct": round(no_heading / n * 100, 2),
        "text_mixed_with_table_pct": round(mixed / n * 100, 2),
        "types": dict(types),
    }


def run_param(max_chars: int, min_chars: int, parsed_map: dict) -> dict:
    by_doc: dict[str, list] = {}
    for path, parsed in parsed_map.items():
        by_doc[path] = chunk_document(parsed, path, max_chunk_chars=max_chars, min_chunk_chars=min_chars)
    return analyze(by_doc, f"max={max_chars},min={min_chars}", max_chars=max_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production Markdown chunking")
    parser.add_argument(
        "--output",
        default="chunking_results.json",
        help="Output filename under benchmark/ (default: chunking_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(p for p in DOCS.rglob("*.md") if p.is_file())
    production_paths = {entry.path for entry in scan_directory(DOCS)}
    print(f"语料 Markdown 文件数：{len(files)}", file=sys.stderr)
    print(f"生产扫描可见文件数：{len(production_paths)}", file=sys.stderr)

    parsed_map: dict[str, object] = {}
    unique_parsed_map: dict[str, object] = {}
    production_parsed_map: dict[str, object] = {}
    production_unique_parsed_map: dict[str, object] = {}
    seen_content: set[str] = set()
    production_seen_content: set[str] = set()
    skipped = 0
    t0 = time.perf_counter()
    for idx, p in enumerate(files, 1):
        try:
            raw = p.read_bytes()
            text = raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001 -- 单文件损坏不阻断全语料统计
            skipped += 1
            continue
        try:
            parsed = parse_markdown(text)
            content_hash = hashlib.sha256(raw).hexdigest()
            if p.resolve() in production_paths:
                production_parsed_map[str(p)] = parsed
                if content_hash not in production_seen_content:
                    production_seen_content.add(content_hash)
                    production_unique_parsed_map[str(p)] = parsed
            if len(text.strip()) < 20:
                skipped += 1
                continue
            parsed_map[str(p)] = parsed
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_parsed_map[str(p)] = parsed
        except Exception:  # noqa: BLE001 -- parser 对畸形导出文件可能抛多种异常
            skipped += 1
        if idx % 300 == 0:
            print(f"  解析 {idx}/{len(files)}", file=sys.stderr)
    parse_s = time.perf_counter() - t0
    print(f"解析完成 {len(parsed_map)} 篇，跳过 {skipped}，耗时 {parse_s:.1f}s", file=sys.stderr)

    # ---- 当前生产参数
    t1 = time.perf_counter()
    prod = run_param(DEFAULT_MAX, DEFAULT_MIN, parsed_map)
    prod["chunk_seconds"] = round(time.perf_counter() - t1, 2)
    prod["parse_seconds"] = round(parse_s, 2)
    exhaustive_unique = run_param(DEFAULT_MAX, DEFAULT_MIN, unique_parsed_map)
    production_scan = run_param(DEFAULT_MAX, DEFAULT_MIN, production_parsed_map)
    prod_unique = run_param(DEFAULT_MAX, DEFAULT_MIN, production_unique_parsed_map)

    # ---- 参数消融
    ablations = [
        run_param(2000, 120, parsed_map),
        run_param(2000, 300, parsed_map),
        run_param(1200, 200, parsed_map),
        run_param(800, 200, parsed_map),
        run_param(3000, 400, parsed_map),
    ]

    # ---- 碎片样例（生产参数下最糟的 25 个块）
    worst = []
    for path, parsed in list(parsed_map.items()):
        for c in chunk_document(parsed, path, DEFAULT_MAX, DEFAULT_MIN):
            if c.kind == "image":
                continue
            if len(c.text.strip()) <= 30:
                worst.append({"file": path.replace(str(DOCS), "").lstrip("\\/"),
                              "heading": c.heading_path, "kind": c.kind,
                              "text": c.text.strip()[:80]})
    worst.sort(key=lambda x: len(x["text"]))
    worst = worst[:30]

    # ---- 单文档块数 TOP（语料倾斜）
    per_doc_count = {}
    for path, parsed in parsed_map.items():
        per_doc_count[path.replace(str(DOCS), "").lstrip("\\/")] = len(
            chunk_document(parsed, path, DEFAULT_MAX, DEFAULT_MIN))
    top_docs = sorted(per_doc_count.items(), key=lambda x: -x[1])[:15]

    result = {
        "corpus_files": len(files),
        "parsed_files": len(parsed_map),
        "skipped_files": skipped,
        "production_scanned_files": len(production_parsed_map),
        "production": prod,
        "production_scan": production_scan,
        "production_unique_documents": prod_unique,
        "exhaustive_unique_documents": exhaustive_unique,
        "ablations": ablations,
        "worst_fragments": worst,
        "top_docs_by_chunks": top_docs,
    }
    output_path = OUT / Path(args.output).name
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 生产参数分块质量 ===", file=sys.stderr)
    for k, v in prod.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\n-> {output_path.relative_to(BASE)}", file=sys.stderr)


if __name__ == "__main__":
    main()
