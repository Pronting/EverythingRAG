"""ingestion 扫描器测试：递归目录扫描 + Markdown 发现（MVP 任务 1）。

验收断言对应任务规格 7 条；用 pytest tmp_path 动态构造目录树，保证隔离。
全部为纯本地文件系统操作，不产生任何出网。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import xxhash

from app.ingestion import DiscoveredFile, scan_directory


def _write(root: Path, rel: str, content: str = "") -> Path:
    """在 root 下按相对路径写文件，自动创建父目录。"""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_finds_all_md_recursively_ignores_non_md(tmp_path: Path) -> None:
    """嵌套子目录下所有 .md 均被发现，非 .md 文件被忽略。"""
    _write(tmp_path, "top.md", "# top")
    _write(tmp_path, "notes.txt", "plain text")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    _write(tmp_path, "sub/inner.md", "# inner")
    _write(tmp_path, "sub/deep/leaf.md", "# leaf")
    _write(tmp_path, "sub/deep/leaf.md.bak", "# not md")

    results = scan_directory(tmp_path)
    expected = {
        tmp_path / "top.md",
        tmp_path / "sub" / "inner.md",
        tmp_path / "sub" / "deep" / "leaf.md",
    }
    assert {r.path for r in results} == {p.resolve() for p in expected}


def test_skips_hidden_dirs_and_files(tmp_path: Path) -> None:
    """点前缀目录整棵跳过，点前缀文件忽略。"""
    _write(tmp_path, ".git/config.md", "# git config")
    _write(tmp_path, ".obsidian/nested/note.md", "# obsidian")
    _write(tmp_path, ".hidden.md", "# hidden file")
    _write(tmp_path, "visible.md", "# visible")

    results = scan_directory(tmp_path)
    assert [r.filename for r in results] == ["visible.md"]


def test_metadata_fields_and_hash_stability(tmp_path: Path) -> None:
    """每个文档返回结构化元数据；同一文件两次扫描 hash 一致。"""
    doc = tmp_path / "doc.md"
    doc.write_text("# hello world", encoding="utf-8")

    (results,) = scan_directory(tmp_path)
    assert isinstance(results, DiscoveredFile)
    assert results.filename == "doc.md"
    assert results.path.is_absolute()
    assert results.path == results.path.resolve()
    assert isinstance(results.mtime, float)
    assert isinstance(results.size, int)
    assert isinstance(results.content_hash, str)
    assert len(results.content_hash) == 16  # xxh64 hexdigest = 16 hex 字符
    assert results.content_hash == xxhash.xxh64(b"# hello world").hexdigest()

    (results2,) = scan_directory(tmp_path)
    assert results2 == results
    assert results2.content_hash == results.content_hash


def test_deterministic_order_stable_across_runs(tmp_path: Path) -> None:
    """返回顺序按 path 确定排序，两次扫描结果完全一致。"""
    for name in ("z.md", "a.md", "m.md"):
        (tmp_path / name).write_text(f"# {name}", encoding="utf-8")
    _write(tmp_path, "sub/b.md", "# b")

    first = scan_directory(tmp_path)
    second = scan_directory(tmp_path)
    assert first == second
    assert [r.path for r in first] == sorted(r.path for r in first)


def test_empty_md_is_discovered(tmp_path: Path) -> None:
    """空 .md 文件仍为合法文档，被发现且不抛异常。"""
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")

    (results,) = scan_directory(tmp_path)
    assert results.path == empty.resolve()
    assert results.size == 0
    assert results.content_hash == xxhash.xxh64(b"").hexdigest()


def test_binary_content_hashed_as_bytes(tmp_path: Path) -> None:
    """以字节流读文件计算指纹，二进制内容不因编码而损坏。"""
    raw = b"\x00\x01\xfe\xff # binary md"
    (tmp_path / "bin.md").write_bytes(raw)

    (results,) = scan_directory(tmp_path)
    assert results.content_hash == xxhash.xxh64(raw).hexdigest()


def test_skips_unreadable_file_without_interrupting(monkeypatch, tmp_path: Path) -> None:
    """单个文件读取失败：跳过该文件、记录日志，不中断整体扫描。"""
    (tmp_path / "good.md").write_text("# good", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("# bad", encoding="utf-8")
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        if self.name == "bad.md":
            raise PermissionError("simulated: permission denied")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    results = scan_directory(tmp_path)
    assert [r.filename for r in results] == ["good.md"]


def test_invalid_root_raises_value_error(tmp_path: Path) -> None:
    """根目录不存在或指向文件时快速失败，给调用方清晰错误。"""
    with pytest.raises(ValueError):
        scan_directory(tmp_path / "does-not-exist")
    (tmp_path / "afile.md").write_text("# x", encoding="utf-8")
    with pytest.raises(ValueError):
        scan_directory(tmp_path / "afile.md")


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    """空目录返回空列表，不抛异常。"""
    assert scan_directory(tmp_path) == []
