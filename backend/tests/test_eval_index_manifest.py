"""隔离评测索引清单的来源核验测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.index_manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    EvalIndexManifest,
    IndexManifestError,
    build_corpus_snapshot,
    load_index_manifest,
    verify_index_manifest,
    write_index_manifest,
)
from app.ingestion.index_schema import INGESTION_SCHEMA_VERSION


def _manifest(corpus_sha256: str, file_count: int) -> EvalIndexManifest:
    return EvalIndexManifest(
        manifest_version=MANIFEST_VERSION,
        corpus_sha256=corpus_sha256,
        corpus_file_count=file_count,
        index_schema_version=INGESTION_SCHEMA_VERSION,
        embedder_fingerprint="cloud-Qwen",
        collection_name=f"chunks__cloud-Qwen__v{INGESTION_SCHEMA_VERSION}",
        logical_count=3,
    )


def test_corpus_snapshot_changes_for_path_or_content_and_ignores_hidden(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    hidden = tmp_path / ".obsidian"
    hidden.mkdir()
    (hidden / "ignored.md").write_text("ignored", encoding="utf-8")
    first = build_corpus_snapshot(tmp_path)

    (tmp_path / "a.md").write_text("B", encoding="utf-8")
    changed_content = build_corpus_snapshot(tmp_path)
    (tmp_path / "a.md").rename(tmp_path / "renamed.md")
    changed_path = build_corpus_snapshot(tmp_path)

    assert first.file_count == changed_content.file_count == changed_path.file_count == 1
    assert len({first.sha256, changed_content.sha256, changed_path.sha256}) == 3


def test_manifest_round_trip_and_exact_verification(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("正文", encoding="utf-8")
    corpus = build_corpus_snapshot(tmp_path)
    manifest = _manifest(corpus.sha256, corpus.file_count)

    write_index_manifest(tmp_path, manifest)
    loaded = load_index_manifest(tmp_path)
    verify_index_manifest(
        loaded,
        corpus=corpus,
        index_schema_version=INGESTION_SCHEMA_VERSION,
        embedder_fingerprint="cloud-Qwen",
        collection_name=f"chunks__cloud-Qwen__v{INGESTION_SCHEMA_VERSION}",
        logical_count=3,
    )
    assert loaded == manifest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_sha256", "f" * 64),
        ("corpus_file_count", 2),
        ("index_schema_version", 3),
        ("embedder_fingerprint", "other"),
        ("collection_name", "other"),
        ("logical_count", 4),
    ],
)
def test_manifest_rejects_every_provenance_mismatch(
    tmp_path: Path, field: str, value: str | int
) -> None:
    (tmp_path / "doc.md").write_text("正文", encoding="utf-8")
    corpus = build_corpus_snapshot(tmp_path)
    manifest = _manifest(corpus.sha256, corpus.file_count)
    payload = manifest.to_dict()
    payload[field] = value
    tampered = EvalIndexManifest(**payload)

    with pytest.raises(IndexManifestError, match=field):
        verify_index_manifest(
            tampered,
            corpus=corpus,
            index_schema_version=INGESTION_SCHEMA_VERSION,
            embedder_fingerprint="cloud-Qwen",
            collection_name=f"chunks__cloud-Qwen__v{INGESTION_SCHEMA_VERSION}",
            logical_count=3,
        )


def test_manifest_rejects_missing_unknown_and_invalid_fields(tmp_path: Path) -> None:
    with pytest.raises(IndexManifestError, match="缺少"):
        load_index_manifest(tmp_path)

    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    with pytest.raises(IndexManifestError, match="字段"):
        load_index_manifest(tmp_path)

    invalid = _manifest("x" * 64, 1).to_dict()
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(IndexManifestError, match="无效"):
        load_index_manifest(tmp_path)
