"""对一个黄金集 case 输出脱敏召回 trace；不打印块正文或绝对来源路径。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ORIGINAL_CWD = Path.cwd()
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from app.core.settings_store import get_settings_store
from app.evaluation.rag_quality import chunk_matches_expectation, load_eval_cases
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_transform import build_query_variants
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY, RetrievedChunk
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断单个 RAG 黄金集 case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--persist-dir", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "scripts" / "rag_eval_cases.json")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ORIGINAL_CWD / path).resolve()


def main() -> int:
    args = _args()
    positives, _negatives = load_eval_cases(_resolve(args.dataset))
    case = next((item for item in positives if item.id == args.case_id), None)
    if case is None:
        print("未知正例 case id")
        return 2
    embedder = create_embedder_from_config(get_settings_store().load().embed)
    vectorstore = create_vector_store(
        _resolve(args.persist_dir),
        collection_name=args.collection_name,
        embedder=embedder,
    )
    variants = build_query_variants(case.query)
    vectors = embedder.embed_queries([variant.text for variant in variants])
    result = HybridRetriever(
        vectorstore,
        context_top_k=8,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    ).retrieve_variants_with_trace(variants, vectors)
    metadata_by_id = {block_id: metadata for block_id, _text, metadata in vectorstore.list_blocks()}
    trace_by_id = {item.block_id: item for item in result.trace.candidates}

    def projection(block_id: str, *, selected: bool) -> dict[str, object]:
        metadata = metadata_by_id.get(block_id, {})
        source = PurePosixPath(str(metadata.get("source_file", "")).replace("\\", "/")).name
        trace = trace_by_id.get(block_id)
        return {
            "source": source,
            "heading": str(metadata.get("heading_path") or ""),
            "selected": selected,
            "dense_similarity": trace.dense_similarity if trace else None,
            "fused_score": trace.fused_score if trace else None,
            "gate_reason": trace.gate_reason if trace else None,
            "selection_reason": trace.selection_reason if trace else None,
            "signals": [signal.to_dict() for signal in trace.signals] if trace else [],
        }

    selected = [projection(chunk.block_id, selected=True) for chunk in result.chunks]
    expected: list[dict[str, object]] = []
    for block_id, metadata in metadata_by_id.items():
        probe = RetrievedChunk(
            block_id=block_id,
            text="",
            similarity=0.0,
            source_file=str(metadata.get("source_file") or ""),
            platform=str(metadata.get("platform") or "local"),
            chunk_type=str(metadata.get("chunk_type") or "text"),
            anchor=str(metadata.get("anchor") or "") or None,
            heading_path=str(metadata.get("heading_path") or "") or None,
            metadata=metadata,
        )
        if chunk_matches_expectation(probe, case):
            expected.append(
                projection(block_id, selected=block_id in {c.block_id for c in result.chunks})
            )
    print(
        json.dumps(
            {"case_id": case.id, "selected": selected, "expected_candidates": expected},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
