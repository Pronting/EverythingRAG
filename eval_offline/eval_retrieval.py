"""离线 RAG 检索评测（不联网、不改业务代码）。

用途：
1. 用本地 bge-m3 ONNX + 仓库 docs/ 隔离索引，跑 HybridRetriever 全链路；
2. 计算 Hit@k / MRR / nDCG，并按阈值扫描观察门控与负例误放行；
3. 通过真实 `_search_knowledge` 工具入口复跑，校验工具契约与端到端检索耗时。

前置：先运行 build_index.py 生成 eval_offline/index 与 manifest.json。
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.retrieval.query_transform import build_query_variants  # noqa: E402
from app.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

from local_embedder import LocalBgeM3Embedder  # noqa: E402

COLLECTION = "chunks__local-bge-m3-onnx__v7"
PROD_THRESHOLD = 0.60
SWEEP = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def norm(text: str) -> str:
    return " ".join(text.split())


def load_manifest() -> list[dict]:
    path = HERE / "manifest.json"
    if not path.exists():
        raise SystemExit("缺少 manifest.json：请先完成 build_index.py")
    return json.loads(path.read_text(encoding="utf-8"))


def map_evidence(questions: list[dict], manifest: list[dict]) -> dict[str, list[set[str]]]:
    """把每条 evidence 片段映射为包含它的 block_id 集合（按 evidence 分组）。"""

    mapping: dict[str, list[set[str]]] = {}
    normalized = [(row["block_id"], norm(row["text"])) for row in manifest]
    for q in questions:
        groups: list[set[str]] = []
        for ev in q["evidence"]:
            snippet = norm(ev["snippet"])
            hits = {bid for bid, text in normalized if snippet in text}
            if not hits:
                # 回退：按标题 + 文档定位（片段跨块时仍能给出可核对的证据块）
                hits = {
                    row["block_id"]
                    for row in manifest
                    if row["rel"] == ev["doc"] and snippet[:18] in norm(row["text"])
                }
            groups.append(hits)
        mapping[q["id"]] = groups
    return mapping


def snapshot_rss_mb() -> float | None:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return round(counters.WorkingSetSize / 1024 / 1024, 1) if ok else None
    except Exception:
        return None


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_question(retrieved: list, groups: list[set[str]], k: int) -> dict:
    ids = [chunk.block_id for chunk in retrieved[:k]]
    relevant = set().union(*groups) if groups else set()
    ranks = [i + 1 for i, bid in enumerate(ids) if bid in relevant]
    hit = bool(ranks)
    full = all(any(bid in group for bid in ids) for group in groups) if groups else False
    gains = [1 if bid in relevant else 0 for bid in ids]
    ideal = [1] * min(len(relevant), k)
    ndcg = dcg(gains) / dcg(ideal) if ideal else 0.0
    return {
        "hit": hit,
        "full": full,
        "mrr": 1.0 / ranks[0] if ranks else 0.0,
        "ndcg": ndcg,
        "top1_relevant": bool(ids and ids[0] in relevant),
        "n_relevant_in_topk": sum(gains),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "hit@1": round(sum(r["top1_relevant"] for r in rows) / n, 4),
        "hit@5": round(sum(r["hit"] for r in rows) / n, 4),
        "hit@8": round(sum(r["hit"] for r in rows) / n, 4),
        "full@8": round(sum(r["full"] for r in rows) / n, 4),
        "mrr@8": round(sum(r["mrr"] for r in rows) / n, 4),
        "ndcg@8": round(sum(r["ndcg"] for r in rows) / n, 4),
    }


async def run_tool_checks(
    questions: list[dict], embedder, store, retriever, per_question: dict
) -> list[dict]:
    """走真实工具入口，验证契约与端到端检索耗时。"""

    from app.agent.tools import SearchKnowledgeParams, ToolContext, _search_knowledge

    workspace = SimpleNamespace(
        workspace_id="eval-offline",
        name="eval-offline",
        knowledge_revision=1,
        vector_store=store,
    )
    ctx = ToolContext(
        workspace=workspace,
        embedder=embedder,
        retriever=retriever,
        vector_store=store,
        rerank=None,
    )
    out = []
    for q in questions:
        params = SearchKnowledgeParams(query=q["question"], top_k=8)
        started = time.perf_counter()
        result = await _search_knowledge(params, ctx)
        elapsed = time.perf_counter() - started
        data = result.data or {}
        out.append(
            {
                "id": q["id"],
                "status": str(result.status),
                "count": data.get("count"),
                "candidate_count": data.get("candidate_count"),
                "tool_seconds": round(elapsed, 3),
                "timings": data.get("timings"),
                "payload_keys": sorted((data.get("chunks") or [{}])[0].keys()),
                "has_path_keys": any(
                    k in (data.get("chunks") or [{}])[0]
                    for k in ("source_file", "source_path", "file_path")
                ),
                "top_scores": [c.get("similarity") for c in data.get("chunks") or []][:3],
            }
        )
        if q["id"] in per_question:
            per_question[q["id"]]["tool_status"] = out[-1]["status"]
            per_question[q["id"]]["tool_seconds"] = out[-1]["tool_seconds"]
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=PROD_THRESHOLD)
    parser.add_argument("--skip-tool", action="store_true")
    args = parser.parse_args()

    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]
    manifest = load_manifest()
    evidence = map_evidence(questions, manifest)

    print(f"questions={len(questions)} blocks={len(manifest)}", flush=True)
    embedder = LocalBgeM3Embedder()
    store = ChromaVectorStore(HERE / "index", COLLECTION, embedder)
    print(f"indexed={store.count()} rss_mb={snapshot_rss_mb()}", flush=True)
    retriever = HybridRetriever(store, min_similarity=args.threshold)

    # 逐题编码一次，阈值扫描复用向量；词法规范化变体也各自编码（与工具层一致）
    vectors: dict[str, list[list[float]]] = {}
    variants_by_id: dict[str, list] = {}
    embed_seconds: dict[str, float] = {}
    for q in questions:
        variants = build_query_variants(q["question"])
        variants_by_id[q["id"]] = variants
        started = time.perf_counter()
        vectors[q["id"]] = [embedder.embed_query(v.text) for v in variants]
        embed_seconds[q["id"]] = time.perf_counter() - started
    print("embeddings ready", flush=True)

    sweep_results: dict[str, dict] = {}
    per_threshold_rows: dict[str, list[dict]] = {}
    for threshold in SWEEP:
        retriever._min_similarity = threshold
        rows = []
        retrieval_times = []
        for q in questions:
            started = time.perf_counter()
            chunks = retriever.retrieve_variants(
                variants_by_id[q["id"]], vectors[q["id"]]
            )
            retrieval_times.append(time.perf_counter() - started)
            if not q["answerable"]:
                rows.append({"id": q["id"], "negative": True, "accepted": len(chunks)})
                continue
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "negative": False,
                    **score_question(chunks, evidence[q["id"]], 8),
                    "retrieved": [
                        {
                            "block_id": c.block_id,
                            "rel": next(
                                (m["rel"] for m in manifest if m["block_id"] == c.block_id), ""
                            ),
                            "similarity": c.similarity,
                            "match_type": c.match_type,
                        }
                        for c in chunks
                    ],
                }
            )
        answerable = [r for r in rows if not r["negative"]]
        negatives = [r for r in rows if r["negative"]]
        sweep_results[str(threshold)] = {
            **summarize(answerable),
            "negatives": len(negatives),
            "negative_false_accept": sum(1 for r in negatives if r["accepted"] > 0),
            "retrieval_p50_ms": round(statistics.median(retrieval_times) * 1000, 1),
            "retrieval_p90_ms": round(
                statistics.quantiles(retrieval_times, n=10)[-1] * 1000, 1
            )
            if len(retrieval_times) >= 10
            else None,
        }
        per_threshold_rows[str(threshold)] = rows
        print(f"threshold={threshold} {sweep_results[str(threshold)]}", flush=True)

    rows = per_threshold_rows[str(args.threshold)]
    per_question = {r["id"]: r for r in rows if not r["negative"]}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["negative"]:
            by_type[r["type"]].append(r)

    tool_rows = []
    if not args.skip_tool:
        retriever._min_similarity = args.threshold
        tool_rows = await run_tool_checks(questions, embedder, store, retriever, per_question)
        print("tool checks done", flush=True)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "threshold": args.threshold,
        "collection": COLLECTION,
        "indexed_blocks": store.count(),
        "manifest_blocks": len(manifest),
        "embedder": "local bge-m3 ONNX (CLS, 1024d, max 512 tokens)",
        "rss_mb_after_run": snapshot_rss_mb(),
        "embed_p50_s": round(statistics.median(embed_seconds.values()), 3),
        "embed_p90_s": round(statistics.quantiles(list(embed_seconds.values()), n=10)[-1], 3),
        "sweep": sweep_results,
        "by_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "per_question": rows,
        "tool": tool_rows,
    }
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "retrieval_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 精简 markdown 摘要
    lines = [
        "# 离线检索评测结果",
        "",
        f"- 生成时间：{out['generated_at']}",
        f"- 语料：docs/ 34 篇 → {out['indexed_blocks']} 块（隔离索引）",
        f"- 嵌入：{out['embedder']}",
        f"- 生产阈值：{args.threshold}；embed p50 {out['embed_p50_s']}s / p90 {out['embed_p90_s']}s",
        f"- 进程 RSS：{out['rss_mb_after_run']} MB",
        "",
        "## 阈值扫描（36 题，含 2 条无答案）",
        "",
        "| 阈值 | Hit@1 | Hit@5 | Hit@8 | Full@8 | MRR@8 | nDCG@8 | 负例误放行 | 检索 p50 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for threshold, s in sweep_results.items():
        lines.append(
            f"| {threshold} | {s['hit@1']:.2%} | {s['hit@5']:.2%} | {s['hit@8']:.2%} | "
            f"{s['full@8']:.2%} | {s['mrr@8']:.3f} | {s['ndcg@8']:.3f} | "
            f"{s['negative_false_accept']}/{s['negatives']} | {s['retrieval_p50_ms']} ms |"
        )
    lines += ["", "## 分类（生产阈值）", "", "| 类型 | n | Hit@1 | Hit@5 | Hit@8 | MRR@8 |", "|---|---|---|---|---|---|"]
    for kind, s in out["by_type"].items():
        lines.append(
            f"| {kind} | {s['n']} | {s['hit@1']:.2%} | {s['hit@5']:.2%} | {s['hit@8']:.2%} | {s['mrr@8']:.3f} |"
        )
    (results_dir / "retrieval_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote results/retrieval_metrics.{json,md}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
