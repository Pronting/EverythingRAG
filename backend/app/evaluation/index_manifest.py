"""可复现的隔离评测索引清单；复用索引前必须逐项核验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.ingestion.scanner import scan_directory

MANIFEST_VERSION = 1
MANIFEST_FILENAME = ".everything-rag-eval-index.json"


class IndexManifestError(ValueError):
    """评测索引清单缺失、格式错误或与当前运行输入不一致。"""


@dataclass(frozen=True)
class CorpusSnapshot:
    sha256: str
    file_count: int


@dataclass(frozen=True)
class EvalIndexManifest:
    manifest_version: int
    corpus_sha256: str
    corpus_file_count: int
    index_schema_version: int
    embedder_fingerprint: str
    collection_name: str
    logical_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_corpus_snapshot(root: Path) -> CorpusSnapshot:
    """按扫描器的真实文件集合计算「相对路径 + 文件 SHA-256」清单摘要。"""

    resolved_root = root.expanduser().resolve()
    entries: list[tuple[str, str]] = []
    for discovered in scan_directory(resolved_root):
        relative = discovered.path.relative_to(resolved_root).as_posix()
        content_sha256 = hashlib.sha256(discovered.path.read_bytes()).hexdigest()
        entries.append((relative, content_sha256))
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return CorpusSnapshot(sha256=hashlib.sha256(canonical).hexdigest(), file_count=len(entries))


def write_index_manifest(work_dir: Path, manifest: EvalIndexManifest) -> Path:
    """原子写入 sidecar；只在一次完整且健康的新建索引后调用。"""

    path = work_dir / MANIFEST_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_index_manifest(work_dir: Path) -> EvalIndexManifest:
    """严格读取 sidecar；未知/缺失字段一律拒绝。"""

    path = work_dir / MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IndexManifestError("评测索引缺少语料清单，不能证明其来源") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexManifestError("评测索引语料清单不可读") from exc
    if not isinstance(payload, dict):
        raise IndexManifestError("评测索引语料清单格式错误")
    expected_fields = {
        "manifest_version",
        "corpus_sha256",
        "corpus_file_count",
        "index_schema_version",
        "embedder_fingerprint",
        "collection_name",
        "logical_count",
    }
    if set(payload) != expected_fields:
        raise IndexManifestError("评测索引语料清单字段不匹配")
    try:
        manifest = EvalIndexManifest(**payload)
    except TypeError as exc:
        raise IndexManifestError("评测索引语料清单字段类型错误") from exc
    if (
        manifest.manifest_version != MANIFEST_VERSION
        or not isinstance(manifest.corpus_sha256, str)
        or len(manifest.corpus_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest.corpus_sha256)
        or not isinstance(manifest.corpus_file_count, int)
        or manifest.corpus_file_count <= 0
        or not isinstance(manifest.index_schema_version, int)
        or not isinstance(manifest.embedder_fingerprint, str)
        or not manifest.embedder_fingerprint
        or not isinstance(manifest.collection_name, str)
        or not manifest.collection_name
        or not isinstance(manifest.logical_count, int)
        or manifest.logical_count <= 0
    ):
        raise IndexManifestError("评测索引语料清单内容无效")
    return manifest


def verify_index_manifest(
    manifest: EvalIndexManifest,
    *,
    corpus: CorpusSnapshot,
    index_schema_version: int,
    embedder_fingerprint: str,
    collection_name: str,
    logical_count: int,
) -> None:
    """复用前核验语料、切片结构、向量空间、collection 与记录数。"""

    checks = {
        "corpus_sha256": manifest.corpus_sha256 == corpus.sha256,
        "corpus_file_count": manifest.corpus_file_count == corpus.file_count,
        "index_schema_version": manifest.index_schema_version == index_schema_version,
        "embedder_fingerprint": manifest.embedder_fingerprint == embedder_fingerprint,
        "collection_name": manifest.collection_name == collection_name,
        "logical_count": manifest.logical_count == logical_count,
    }
    mismatches = [name for name, matches in checks.items() if not matches]
    if mismatches:
        raise IndexManifestError(f"评测索引来源不匹配：{', '.join(mismatches)}")
