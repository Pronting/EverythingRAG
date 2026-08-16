"""SyncService 单测：指纹快路径 + 块级复用（新增/更新/删除/未变）。

用轻量 fake（FakeEmbedder + 记录型 FakeVectorStore + 真实 tmp SQLite），
零出网、不加载真实 Chroma / bge-m3。块级复用是核心验收：
「改一处只更新相关块」——未变块 block_id 复用、不重嵌入。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.scanner import DiscoveredFile, discover_file
from app.ingestion.state_store import DocumentStateStore
from app.ingestion.sync import SyncService


class FakeEmbedder:
    """确定性嵌入器：文本长度哈希到定长向量；记录每次 embed 的文本。"""

    fingerprint = "fake"
    dim = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)) % 7.0, 0.0, 0.0, 0.0] for text in texts]

    @property
    def total_embedded(self) -> int:
        return sum(len(call) for call in self.calls)


class FakeVectorStore:
    """记录型假向量库：upsert/delete/get_blocks_by_source 全真实内存模拟。"""

    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, list[float], object]] = {}

    def upsert(self, blocks: list[tuple[str, str, list[float], object]]) -> None:
        for block_id, text, vector, meta in blocks:
            self.blocks[block_id] = (text, vector, meta)

    def delete_by_ids(self, block_ids: list[str]) -> None:
        for block_id in block_ids:
            self.blocks.pop(block_id, None)

    def delete_by_where(self, where: dict[str, str]) -> None:
        for block_id, (_, _, meta) in list(self.blocks.items()):
            if where == {"source_file": meta.source_file}:
                self.blocks.pop(block_id, None)

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {
            block_id: text
            for block_id, (text, _, meta) in self.blocks.items()
            if meta.source_file == source_file
        }

    def count(self) -> int:
        return len(self.blocks)


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def harness(tmp_path: Path) -> tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore]:
    """装配 SyncService + fake，state 落在 tmp_path（测试后自动清理）。"""
    root = tmp_path / "docs"
    root.mkdir()
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    state = DocumentStateStore(tmp_path / "state.db")
    service = SyncService(embedder=embedder, vectorstore=store, state_store=state)
    yield service, embedder, store, state
    state.close()


# 注：fixture 段落内容需达到最小块长（120 字符），否则短段落会被合并、块数断言失效。
_FIXTURE = {
    "a.md": "# 标题一\n\n" + "第一段内容。" * 60 + "\n\n## 子标题\n\n" + "子内容。" * 60 + "\n",
    "b.md": "# B\n\n" + "B 的内容。" * 60 + "\n",
}


def _seed_docs(root: Path) -> None:
    for name, content in _FIXTURE.items():
        _write(root, name, content)


# ---------------------------------------------------------------- 1. 首次同步 = 全新增


def test_first_sync_adds_all_files(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """空状态首同步：全部文件判为新增，全块入库，来源根已登记。"""
    service, _embedder, store, state = harness
    root = harness_root(harness)
    _seed_docs(root)

    report = service.ingest(root)

    assert report.files_scanned == 2
    assert report.files_added == 2
    assert report.files_updated == 0
    assert report.files_deleted == 0
    assert report.files_unchanged == 0
    assert report.files_skipped == 0
    assert report.blocks_upserted == store.count() == 3  # a=2 块、b=1 块
    assert report.errors == ()
    # 指纹已持久化，来源根已登记
    assert len(state.load_all()) == 2
    assert state.list_sources() == [harness_root(harness)]


def harness_root(harness: tuple[SyncService, object, object, DocumentStateStore]) -> Path:
    """返回 SyncService 待同步的根目录（tmp_path/docs）。"""
    state_path = harness[3]._path
    return state_path.parent / "docs"


# ---------------------------------------------------------------- 2. 未变跳过（快路径零重嵌入）


def test_second_sync_unchanged_skips_everything(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """内容未变再同步：全部未变跳过，零嵌入（快路径生效）。"""
    service, embedder, store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)

    first = service.ingest(root)
    embedder.calls.clear()
    second = service.ingest(root)

    assert second.files_scanned == 2
    assert second.files_added == 0
    assert second.files_updated == 0
    assert second.files_deleted == 0
    assert second.files_unchanged == 2
    assert second.blocks_upserted == 0
    assert embedder.total_embedded == 0  # 未变 -> 零重嵌入
    assert store.count() == first.blocks_upserted  # 库不变


# ---------------------------------------------------------------- 3. 新增文件


def test_sync_detects_new_file(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """首同步后新增一个文件 -> 只处理新增，其余未变。"""
    service, _embedder, store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)

    _write(root, "c.md", "# C\n\nC 的新内容。\n")
    report = service.ingest(root)

    assert report.files_added == 1
    assert report.files_updated == 0
    assert report.files_unchanged == 2
    assert report.blocks_upserted == 1
    assert store.count() == 4  # 3 + 1


# ---------------------------------------------------------------- 4. 删除文件级联清块


def test_sync_detects_deleted_file(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """删除一个文件 -> 级联清掉其全部块 + 指纹，不残留孤儿。"""
    service, _embedder, store, state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)

    (root / "b.md").unlink()
    report = service.ingest(root)

    assert report.files_deleted == 1
    assert report.files_unchanged == 1  # 仅 a.md 未变
    assert report.blocks_deleted == 1  # b.md 的 1 块被清
    assert store.count() == 2  # 仅剩 a.md 的 2 块
    assert "b.md" not in state.load_all()


# ---------------------------------------------------------------- 5. 修改文件 -> 块级复用（核心）


def test_modified_file_reuses_unchanged_blocks(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """改一处只更新相关块：未变区 block_id 复用、零重嵌入；只嵌入变化块。

    a.md 结构：# 标题一（第1段） + ## 子标题（子内容）。改「第一段内容」
    -> 只第一段所在块变化，子标题块文本未变 -> 复用旧 block_id。
    """
    service, embedder, store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)

    before_ids = set(store.blocks)
    # 修改第一段内容（不动子标题）；段落保持实质长度以维持独立块
    _write(root, "a.md", "# 标题一\n\n" + "第一段已修改。" * 60 + "\n\n## 子标题\n\n" + "子内容。" * 60 + "\n")
    embedder.calls.clear()

    report = service.ingest(root)

    assert report.files_updated == 1
    assert report.files_unchanged == 1  # b.md 未变
    assert report.blocks_upserted == 1  # 只嵌入变化的那一块
    assert report.blocks_deleted == 1  # 旧的变化块被删

    # 关键：只调用了一次 embed，且只含变化块文本（未变块未重嵌入）
    assert embedder.total_embedded == 1
    assert embedder.calls == [["第一段已修改。" * 60]]

    # 库仍 3 块；a.md 未变的子标题块 block_id 复用（在 before_ids 中），文本未变
    assert store.count() == 3
    reused_a = {
        bid
        for bid in (before_ids & set(store.blocks))
        if store.blocks[bid][2].source_file.endswith("a.md")
    }
    assert len(reused_a) == 1
    assert store.blocks[reused_a.pop()][0] == "子内容。" * 60


def test_modified_file_adds_and_removes_blocks(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """新增一个标题区块 + 删除一个标题区块 -> 分别 upsert / delete。"""
    service, embedder, store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)
    before_ids = set(store.blocks)

    # 把子标题整个删掉（只留标题一），再新增一个「新增标题」
    _write(
        root,
        "a.md",
        "# 标题一\n\n" + "第一段内容。" * 60 + "\n\n## 新增标题\n\n" + "新增的内容。" * 60 + "\n",
    )
    embedder.calls.clear()
    report = service.ingest(root)

    assert report.files_updated == 1
    # 新增标题块嵌入 + 旧子标题块删除；第一段块与标题一块文本未变 -> 复用
    assert report.blocks_upserted == 1
    assert report.blocks_deleted == 1
    assert store.count() == 3  # 3 块不变（1 换 1）
    reused_a = {
        bid
        for bid in (before_ids & set(store.blocks))
        if store.blocks[bid][2].source_file.endswith("a.md")
    }
    assert len(reused_a) == 1  # a.md 标题一块复用
    # 新增内容已入库，被删的子内容不在库中
    texts = {text for text, _, _ in store.blocks.values()}
    assert any("新增的内容。" in text for text in texts)
    assert not any("子内容。" in text for text in texts)


# ---------------------------------------------------------------- 6. mtime 触碰但内容未变


def test_mtime_touch_only_counts_unchanged(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
    tmp_path: Path,
) -> None:
    """仅触碰 mtime（内容未变）-> 视为未变，零嵌入，指纹刷新 mtime。"""
    service, embedder, store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)

    import os

    path = root / "a.md"
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10.0))
    embedder.calls.clear()
    report = service.ingest(root)

    assert report.files_unchanged == 2
    assert report.files_updated == 0
    assert embedder.total_embedded == 0
    assert store.count() == 3


# ---------------------------------------------------------------- 7. ingest_many 聚合


def test_ingest_many_aggregates(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
    tmp_path: Path,
) -> None:
    """多根同步：报告聚合（计数求和），来源根逐一登记。"""
    service, _embedder, _store, state = harness
    root_a = harness_root(harness)
    root_b = tmp_path / "other"
    root_b.mkdir()
    _seed_docs(root_a)
    _write(root_b, "d.md", "# D\n\nD 内容。\n")

    report = service.ingest_many([root_a, root_b])

    assert report.files_added == 3
    assert report.files_scanned == 3
    assert report.blocks_upserted == 4  # a=2 + b=1 + d=1
    assert len(state.list_sources()) == 2


def test_ingest_many_skips_missing_root_and_cleans_registration(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
    tmp_path: Path,
) -> None:
    """失效来源根（目录已删除）-> 跳过并移除登记，不中断其余根，不抛错。

    回归防护：state_store.sources 里可能残留已失效的登记（如临时目录被清理），
    一键同步必须跳过而非整体失败——否则用户每次点「同步知识库」都报错。
    """
    service, _embedder, _store, state = harness
    root_a = harness_root(harness)
    root_b = tmp_path / "vanished"
    root_b.mkdir()
    _seed_docs(root_a)
    _write(root_b, "x.md", "# X\n\n内容。\n")
    # 模拟「曾登记但目录后来被删除」的失效状态
    state.register_source(root_a, synced_at="2026-08-12T00:00:00+00:00")
    state.register_source(root_b, synced_at="2026-08-12T00:00:00+00:00")
    (root_b / "x.md").unlink()
    root_b.rmdir()

    report = service.ingest_many([root_a, root_b])

    # 有效根照常同步；失效根记为跳过且不抛错
    assert report.files_added == 2  # root_a 的 a/b 两文件
    assert report.files_skipped == 1  # root_b 失效被跳过
    assert report.errors == ("DirectoryNotFound",)
    # 失效登记已移除，有效根保留
    assert state.list_sources() == [root_a]


# ---------------------------------------------------------------- 8. 单文件失败脱敏不中断


def test_sync_single_file_failure_sanitized(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单文件不可发现 -> files_skipped + 跳过，其余文件照常同步。"""
    service, _embedder, _store, _state = harness
    root = harness_root(harness)
    _seed_docs(root)
    service.ingest(root)

    # 让 c.md 在发现阶段失败（返回 None，模拟权限不可读）
    _write(root, "c.md", "# C\n\n内容。\n")

    from app.ingestion import sync as sync_mod

    real_discover = sync_mod.discover_file

    def failing_discover(path: Path) -> DiscoveredFile | None:
        if path.name == "c.md":
            return None
        return real_discover(path)

    monkeypatch.setattr(sync_mod, "discover_file", failing_discover)
    report = service.ingest(root)

    assert report.files_skipped == 1  # c.md 不可发现
    assert report.files_unchanged == 2
    assert report.errors == ()


# ---------------------------------------------------------------- 9. 上传式 sync_upload（多文件夹/散文件）


def _disc(path: Path) -> DiscoveredFile:
    """discover_file 的断言封装（读已落盘文件，模拟上传后的规范根文件）。"""
    disc = discover_file(path)
    assert disc is not None
    return disc


def _upload(root: Path, entries: dict[str, str]) -> list[DiscoveredFile]:
    """在 root 下按相对路径写文件并发现，返回 DiscoveredFile 列表。"""
    for rel, content in entries.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return [_disc(root / rel) for rel in entries]


def test_upload_first_sync_adds_all(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """上传首次（无已知指纹）-> 全部判为新增，文件夹+散文件一起入库。"""
    service, _embedder, store, _state = harness
    root = harness_root(harness)
    files = _upload(root, {"folder1/a.md": "# A\n\n内容 A\n", "note.md": "# N\n\n散文件\n"})

    report = service.sync_upload(files, deletion_scopes=[root / "folder1"])

    assert report.files_added == 2
    assert report.files_unchanged == 0
    assert report.files_deleted == 0
    assert store.count() == 2  # 两个文件各 1 块


def test_upload_reupload_unchanged_zero_embed(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """重传相同内容 -> 全部未变，零重嵌入（快路径生效）。"""
    service, embedder, store, _state = harness
    root = harness_root(harness)
    files = _upload(root, {"folder1/a.md": "# A\n\n内容 A\n"})
    service.sync_upload(files, deletion_scopes=[root / "folder1"])

    embedder.calls.clear()
    report = service.sync_upload(files, deletion_scopes=[root / "folder1"])

    assert report.files_unchanged == 1
    assert report.files_added == 0
    assert embedder.total_embedded == 0
    assert store.count() == 1


def test_upload_reupload_modified_reuses_block(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """重传已改文件 -> 块级复用：只重嵌入变化块，未变块复用旧 block_id。"""
    service, embedder, store, _state = harness
    root = harness_root(harness)
    path = root / "folder1" / "a.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# 标题一\n\n" + "第一段内容。" * 60 + "\n\n## 子标题\n\n" + "子内容。" * 60 + "\n", encoding="utf-8")
    service.sync_upload([_disc(path)], deletion_scopes=[root / "folder1"])
    before_ids = set(store.blocks)

    path.write_text("# 标题一\n\n" + "第一段已修改。" * 60 + "\n\n## 子标题\n\n" + "子内容。" * 60 + "\n", encoding="utf-8")
    embedder.calls.clear()
    report = service.sync_upload([_disc(path)], deletion_scopes=[root / "folder1"])

    assert report.files_updated == 1
    assert embedder.total_embedded == 1  # 只重嵌入变化块
    reused_a = {
        bid
        for bid in (before_ids & set(store.blocks))
        if store.blocks[bid][2].source_file == str(path.resolve())
    }
    assert len(reused_a) == 1  # 未变「子标题」块复用


def test_upload_deletes_removed_file_in_scope(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """重传某文件夹但少了一个文件 -> 该文件在 scope 内被删除（块 + 指纹清掉）。"""
    service, _embedder, store, state = harness
    root = harness_root(harness)
    a = root / "folder1" / "a.md"
    b = root / "folder1" / "b.md"
    _upload(root, {"folder1/a.md": "# A\n\n内容 A\n", "folder1/b.md": "# B\n\n内容 B\n"})
    service.sync_upload([_disc(a), _disc(b)], deletion_scopes=[root / "folder1"])
    assert store.count() == 2

    # 模拟 b.md 在本地被删：本次重传只有 a.md
    b.unlink()
    report = service.sync_upload([_disc(a)], deletion_scopes=[root / "folder1"])

    assert report.files_deleted == 1
    assert report.files_unchanged == 1
    assert store.count() == 1  # b 的块被级联清除
    assert str(b.resolve()) not in state.load_all()  # b 指纹移除


def test_upload_loose_file_not_deleted(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """散文件不属于任何 scope -> 未重传也不删除（只增/改）。"""
    service, _embedder, store, state = harness
    root = harness_root(harness)
    folder_a = root / "folder1" / "a.md"
    note = root / "note.md"
    _upload(root, {"folder1/a.md": "# A\n\n内容\n", "note.md": "# N\n\n散文件\n"})
    service.sync_upload([_disc(folder_a), _disc(note)], deletion_scopes=[root / "folder1"])

    # 只重传 folder1/a.md，散文件 note.md 未重传
    report = service.sync_upload([_disc(folder_a)], deletion_scopes=[root / "folder1"])

    assert report.files_deleted == 0  # note.md 不在 scope 内，不删
    assert str(note.resolve()) in state.load_all()  # 指纹保留
    assert store.count() == 2  # note 块仍在


def test_upload_new_folder_touches_only_it(
    harness: tuple[SyncService, FakeEmbedder, FakeVectorStore, DocumentStateStore],
) -> None:
    """重传时新增另一文件夹 -> 只处理新增范围，原范围未变不动。"""
    service, _embedder, store, _state = harness
    root = harness_root(harness)
    p = root / "folder1" / "a.md"
    _upload(root, {"folder1/a.md": "# A\n\n内容\n"})
    service.sync_upload([_disc(p)], deletion_scopes=[root / "folder1"])
    assert store.count() == 1

    q = root / "folder2" / "b.md"
    _upload(root, {"folder2/b.md": "# B\n\n内容\n"})
    report = service.sync_upload([_disc(p), _disc(q)], deletion_scopes=[root / "folder1", root / "folder2"])

    assert report.files_added == 1  # folder2/b.md
    assert report.files_unchanged == 1  # folder1/a.md
    assert report.files_deleted == 0
    assert store.count() == 2
