"""在隔离 Chroma 中按当前清洗/切片代码重建语料并运行块级召回评估。

脚本绝不删除或写入应用正在使用的数据目录。首次运行给一个全新的 ``--work-dir``；
后续只想复测召回参数时可传 ``--reuse-index``，避免再次嵌入全部文档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ORIGINAL_CWD = Path.cwd()
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.evaluation.index_manifest import (
    MANIFEST_VERSION,
    EvalIndexManifest,
    IndexManifestError,
    build_corpus_snapshot,
    load_index_manifest,
    verify_index_manifest,
    write_index_manifest,
)
from app.evaluation.rag_quality import (
    evaluate_retrieval,
    load_eval_cases,
    quality_gate_passes,
)
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION
from app.ingestion.state_store import DocumentStateStore
from app.ingestion.sync import SyncService
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="隔离重建索引并运行 Everything RAG 质量门禁")
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "scripts" / "rag_eval_cases.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--collection-name", help="复测历史隔离索引时显式指定 collection")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--min-hit-at-8", type=float, default=1.0)
    parser.add_argument("--max-negative-fcr", type=float, default=0.0)
    return parser.parse_args()


def _resolve_from_original(path: Path) -> Path:
    return (path if path.is_absolute() else ORIGINAL_CWD / path).resolve()


def main() -> int:
    args = _parse_args()
    docs = _resolve_from_original(args.docs)
    work_dir = _resolve_from_original(args.work_dir)
    live_dir = get_settings().data_dir.expanduser().resolve()
    if not docs.is_dir():
        print("文档目录不存在或不是目录")
        return 2
    if work_dir == live_dir or live_dir in work_dir.parents:
        print("拒绝运行：隔离 work-dir 不能等于或位于应用数据目录内")
        return 2

    app_settings = get_settings_store().load()
    embedder = create_embedder_from_config(app_settings.embed)
    collection_name = args.collection_name or (
        f"chunks__{embedder.fingerprint}__v{INGESTION_SCHEMA_VERSION}"
    )
    corpus = build_corpus_snapshot(docs)
    vectorstore = create_vector_store(
        work_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
    existing = vectorstore.count()
    if existing and not args.reuse_index:
        print("隔离索引已包含数据；请换一个全新目录，或显式传 --reuse-index")
        return 2
    if args.reuse_index and not existing:
        print("拒绝复用：指定的隔离索引为空")
        return 2

    ingest_summary: dict[str, bool | int | list[str]]
    ingestion_healthy = True
    if existing:
        try:
            manifest = load_index_manifest(work_dir)
            verify_index_manifest(
                manifest,
                corpus=corpus,
                index_schema_version=INGESTION_SCHEMA_VERSION,
                embedder_fingerprint=embedder.fingerprint,
                collection_name=collection_name,
                logical_count=existing,
            )
        except IndexManifestError as exc:
            print(f"拒绝复用：{exc}")
            return 2
        ingest_summary = {
            "reused_blocks": existing,
            "provenance_verified": True,
        }
    else:
        # Mirror the production sync path: exact-content aliases are not embedded repeatedly,
        # while content that exists only in a historical UUID directory is still indexed once.
        sync = SyncService(
            embedder=embedder,
            vectorstore=vectorstore,
            state_store=DocumentStateStore(work_dir / "state.db"),
            image_state_store=ImageStateStore(work_dir / "image_state.db"),
        )
        ingest = sync.ingest(docs)
        ingest_summary = {
            "files_scanned": ingest.files_scanned,
            "files_added": ingest.files_added,
            "files_updated": ingest.files_updated,
            "files_deleted": ingest.files_deleted,
            "files_unchanged": ingest.files_unchanged,
            "files_skipped": ingest.files_skipped,
            "blocks_upserted": ingest.blocks_upserted,
            "blocks_deleted": ingest.blocks_deleted,
            "errors": list(ingest.errors),
        }
        ingestion_healthy = (
            ingest.files_scanned > 0
            and ingest.files_skipped == 0
            and not ingest.errors
            and (
                ingest.files_added
                + ingest.files_updated
                + ingest.files_deleted
                + ingest.files_unchanged
                == ingest.files_scanned
            )
        )
    ingest_summary["healthy"] = ingestion_healthy

    integrity = vectorstore.inspect_integrity()
    if not existing and ingestion_healthy and integrity.healthy:
        write_index_manifest(
            work_dir,
            EvalIndexManifest(
                manifest_version=MANIFEST_VERSION,
                corpus_sha256=corpus.sha256,
                corpus_file_count=corpus.file_count,
                index_schema_version=INGESTION_SCHEMA_VERSION,
                embedder_fingerprint=embedder.fingerprint,
                collection_name=collection_name,
                logical_count=integrity.logical_count,
            ),
        )
        ingest_summary["provenance_manifest_written"] = True
    if args.integrity_only:
        print(json.dumps(integrity.to_dict(), ensure_ascii=False, indent=2))
        return 0 if integrity.healthy else 1

    dataset = _resolve_from_original(args.dataset)
    positives, negatives = load_eval_cases(dataset)
    report = evaluate_retrieval(
        embedder=embedder,
        retriever=HybridRetriever(
            vectorstore,
            context_top_k=8,
            min_similarity=DEFAULT_MIN_SIMILARITY,
        ),
        positives=positives,
        negatives=negatives,
        embed_batch_size=args.batch_size,
    )
    quality_healthy = quality_gate_passes(
        report,
        min_hit_at_8=args.min_hit_at_8,
        max_negative_false_context_rate=args.max_negative_fcr,
    )
    summary = {
        "isolated_index": str(work_dir),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": corpus.sha256,
        "corpus_file_count": corpus.file_count,
        "index_schema_version": INGESTION_SCHEMA_VERSION,
        "embedder_fingerprint": embedder.fingerprint,
        "collection_name": collection_name,
        "retriever": {
            "type": "HybridRetriever",
            "context_top_k": 8,
            "min_similarity": DEFAULT_MIN_SIMILARITY,
        },
        "ingestion": ingest_summary,
        "integrity": integrity.to_dict(),
        "quality": {
            "healthy": quality_healthy,
            "min_hit_at_8": args.min_hit_at_8,
            "max_negative_false_context_rate": args.max_negative_fcr,
            "positive_total": report.positive_total,
            "negative_total": report.negative_total,
            "hit_at_1": report.hit_at_1,
            "hit_at_5": report.hit_at_5,
            "hit_at_8": report.hit_at_8,
            "mrr_at_8": report.mrr_at_8,
            "negative_false_context_rate": report.negative_false_context_rate,
            "duration_seconds": report.duration_seconds,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output is not None:
        output = _resolve_from_original(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {**summary, "retrieval_report": report.to_dict()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"完整报告: {output}")
    return 0 if ingestion_healthy and integrity.healthy and quality_healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
