"""Run the block-level Everything RAG retrieval quality gate against the active index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.api.deps import get_embedder, get_vector_store
from app.evaluation.rag_quality import (
    evaluate_retrieval,
    load_eval_cases,
    quality_gate_passes,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY
from app.vectorstore.chroma_store import create_vector_store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Everything RAG 真实块级召回评估")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "scripts" / "rag_eval_cases.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--persist-dir",
        type=Path,
        help="显式评估一个历史 Chroma 目录（必须同时指定 --collection-name）",
    )
    parser.add_argument(
        "--collection-name",
        help="显式评估的历史 collection（必须同时指定 --persist-dir）",
    )
    parser.add_argument(
        "--enforce", action="store_true", help="低于门槛时返回非零退出码"
    )
    parser.add_argument("--min-hit-at-8", type=float, default=1.0)
    parser.add_argument("--max-negative-fcr", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.persist_dir) != bool(args.collection_name):
        print("--persist-dir 与 --collection-name 必须同时指定")
        return 2
    positives, negatives = load_eval_cases(args.dataset)
    embedder = get_embedder()
    vectorstore = (
        create_vector_store(
            args.persist_dir.expanduser().resolve(),
            collection_name=args.collection_name,
            embedder=embedder,
        )
        if args.persist_dir is not None
        else get_vector_store()
    )
    retriever = HybridRetriever(
        vectorstore,
        context_top_k=args.top_k,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )
    report = evaluate_retrieval(
        embedder=embedder,
        retriever=retriever,
        positives=positives,
        negatives=negatives,
        embed_batch_size=args.batch_size,
    )
    summary = {
        "positive_total": report.positive_total,
        "negative_total": report.negative_total,
        "hit_at_1": report.hit_at_1,
        "hit_at_5": report.hit_at_5,
        "hit_at_8": report.hit_at_8,
        "mrr_at_8": report.mrr_at_8,
        "negative_false_context_rate": report.negative_false_context_rate,
        "duration_seconds": report.duration_seconds,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = [item.id for item in report.positive_outcomes if item.matched_rank is None]
    noisy = [item.id for item in report.negative_outcomes if item.retrieved_count > 0]
    if failed:
        print(f"未在 Top-{args.top_k} 命中的正例 ({len(failed)}): {', '.join(failed)}")
    if noisy:
        print(f"误放行上下文的负例 ({len(noisy)}): {', '.join(noisy)}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"完整报告: {args.json_output.resolve()}")
    if args.enforce and not quality_gate_passes(
        report,
        min_hit_at_8=args.min_hit_at_8,
        max_negative_false_context_rate=args.max_negative_fcr,
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
