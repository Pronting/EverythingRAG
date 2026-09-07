"""Build a separate v7 index: copy text vectors, re-embed image summaries, preserve v6.

Run while imports are idle. This never deletes a collection or regenerates vision descriptions.
Manifest promotion is separate so the new collection can be evaluated before activation.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.deps import get_embedder
from app.core.config import settings
from app.ingestion.image_document import IMAGE_REPRESENTATION_VERSION, build_image_document
from app.ingestion.index_schema import make_index_fingerprint
from app.models.schemas import BlockMetadata
from app.vectorstore.chroma_store import create_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    embedder = get_embedder()
    source_name = f"chunks__{embedder.fingerprint}__v6"
    target_name = f"chunks__{embedder.fingerprint}__v7"
    source = create_vector_store(settings.data_dir, source_name, embedder)
    target = create_vector_store(settings.data_dir, target_name, embedder)
    count = source.count()
    if not count:
        raise RuntimeError("Source collection is empty; no migration performed")
    if args.promote:
        if target.count() != count:
            raise RuntimeError("Incomplete target; promotion refused")
        source_blocks = {bid: (body, meta) for bid, body, meta in source.list_blocks()}
        target_blocks = {bid: (body, meta) for bid, body, meta in target.list_blocks()}
        if source_blocks.keys() != target_blocks.keys():
            raise RuntimeError("Block identities differ; promotion refused")
        for bid, (body, meta) in source_blocks.items():
            new_body, new_meta = target_blocks[bid]
            if meta.get("chunk_type") == "image_description":
                if (new_meta.get("image_representation_version") != IMAGE_REPRESENTATION_VERSION
                        or not new_meta.get("image_index_text") or not new_body):
                    raise RuntimeError("Missing image representation; promotion refused")
            elif body != new_body or meta != new_meta:
                raise RuntimeError("Text representation changed; promotion refused")
        target.repair_ann_index()
        state = sqlite3.connect(settings.data_dir / "state.db")
        backup = settings.data_dir / "state.pre-image-v7.db"
        if not backup.exists():
            with sqlite3.connect(backup) as destination:
                state.backup(destination)
        with state:
            state.execute("UPDATE documents SET schema_version=7, index_fingerprint=? "
                          "WHERE schema_version=6 AND index_fingerprint=?",
                          (make_index_fingerprint(embedder.fingerprint, 7),
                           make_index_fingerprint(embedder.fingerprint, 6)))
        state.close()
        print("Manifest promoted; v6 collection and state backup preserved", flush=True)
        return
    # Resolve/validate current provider dimension before copying existing vectors.
    embedder.embed_queries(["知识库索引迁移"])
    images = []
    text_count = 0
    reused_images = 0
    for offset in range(0, count, 128):
        page = source._collection.get(limit=128, offset=offset,
                                      include=["documents", "metadatas", "embeddings"])
        records = []
        existing = target._collection.get(ids=page["ids"], include=["metadatas", "embeddings"])
        existing_vectors = existing["embeddings"]
        existing_images = {
            bid: (meta, vector) for bid, meta, vector in zip(
                existing["ids"], existing["metadatas"],
                existing_vectors if existing_vectors is not None else [], strict=True
            ) if meta.get("chunk_type") == "image_description"
        }
        for bid, body, meta, vector in zip(page["ids"], page["documents"], page["metadatas"], page["embeddings"], strict=True):
            if meta.get("chunk_type") == "image_description":
                document = build_image_document(body, meta.get("heading_path", ""),
                                                Path(meta.get("source_file", "")).stem)
                meta = dict(meta, image_index_text=document.index_text,
                            image_representation_version=IMAGE_REPRESENTATION_VERSION)
                prior = existing_images.get(bid)
                if prior and prior[0].get("image_index_text") == document.index_text:
                    records.append((bid, document.evidence_text, prior[1].tolist(), BlockMetadata(**meta)))
                    reused_images += 1
                else:
                    images.append((bid, document, BlockMetadata(**meta)))
            else:
                records.append((bid, body, vector.tolist(), BlockMetadata(**meta)))
                text_count += 1
        if records:
            target.upsert(records)
    print(f"Copied {text_count} text vectors; reused {reused_images} image vectors; preparing {len(images)} image vectors", flush=True)
    batches = [images[start:start + 16] for start in range(0, len(images), 16)]

    def prepare(batch):
        vectors = embedder.embed_texts([document.index_text for _, document, _ in batch])
        return [(bid, document.evidence_text, vector, meta)
                for (bid, document, meta), vector in zip(batch, vectors, strict=True)]

    done = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for records in pool.map(prepare, batches):
            target.upsert(records)
            done += len(records)
            if done % 160 == 0 or done == len(images):
                print(f"Images {done}/{len(images)}", flush=True)
    if source.count() != count or target.count() != count:
        raise RuntimeError("Collection counts changed; promotion refused")
    repaired = target.repair_ann_index()
    print(f"Complete: {count} blocks; ANN repaired={repaired}. Evaluate before --promote.", flush=True)


if __name__ == "__main__":
    main()
