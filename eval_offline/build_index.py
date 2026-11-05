"""Build an isolated Chroma index over docs/ with the local BGE-M3 ONNX model.

Usage: backend/.venv/Scripts/python -u eval_offline/build_index.py
Writes eval_offline/index/ (delete to rebuild) and eval_offline/manifest.json.
No network. Does not touch ~/.everything-rag or repository data/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.ingestion.chunker import chunk_document
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.pipeline import chunk_embedding_text, make_block_id, _normalize_source_file
from app.models.schemas import BlockMetadata, SourceType
from app.vectorstore.chroma_store import ChromaVectorStore
from eval_offline.local_embedder import LocalBgeM3Embedder

INDEX_DIR = ROOT / "eval_offline" / "index"
MANIFEST = ROOT / "eval_offline" / "manifest.json"
COLLECTION = "chunks__local-bge-m3-onnx__v7"
BATCH = 8


def main() -> None:
    docs = sorted(glob.glob(str(ROOT / "docs" / "**" / "*.md"), recursive=True))
    embedder = LocalBgeM3Embedder()
    rows = []
    for path in docs:
        rel = Path(path).relative_to(ROOT).as_posix()
        text = Path(path).read_text(encoding="utf-8")
        chunks = chunk_document(parse_markdown(text), source_file=str(Path(path).resolve()))
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for order, chunk in enumerate(chunks):
            if chunk.kind == "image":
                continue
            block_id = make_block_id(content_hash, chunk.block_id, source_file=Path(path).resolve())
            meta = BlockMetadata(
                block_id=block_id,
                doc_id=content_hash,
                source_type=SourceType.DOCUMENT,
                source_file=str(Path(path).resolve()),
                platform="local",
                heading_path=chunk.heading_path or None,
                anchor=chunk.anchor or None,
                document_order=order,
                chunk_type=chunk.kind if chunk.kind in {"table", "code"} else "text",
            )
            rows.append({"rel": rel, "block_id": block_id, "text": chunk.text,
                         "embed_text": chunk_embedding_text(chunk), "meta": meta,
                         "heading": chunk.heading_path, "kind": chunk.kind})
    print(f"files={len(docs)} chunks={len(rows)}", flush=True)
    if INDEX_DIR.exists():
        import shutil
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True)
    store = ChromaVectorStore(INDEX_DIR, COLLECTION, embedder)
    started = time.time()
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        vectors = embedder.embed([r["embed_text"] for r in batch], batch_size=BATCH)
        store.upsert([(r["block_id"], r["text"], v, r["meta"]) for r, v in zip(batch, vectors)])
        done = start + len(batch)
        if done % 64 == 0 or done == len(rows):
            elapsed = time.time() - started
            eta = elapsed / done * (len(rows) - done)
            print(f"{done}/{len(rows)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    manifest = [{k: v for k, v in r.items() if k != "meta"} for r in rows]
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"done count={store.count()} total={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
