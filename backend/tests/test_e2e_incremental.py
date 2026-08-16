"""增量同步 E2E：真实 Chroma 闭环（全量首同步 -> 改/增/删 -> 增量同步）。

确定性 fake（FakeEmbedder）避免真实 bge-m3 下载；但扫描/解析/切块/Chroma/
SyncService/块级复用全真实。核心验收：
- 首同步 = 全新增；
- 改一处只更新相关块（未变块 block_id 复用、零重嵌入）；
- 新增文件只入新增块；删除文件级联清块无孤儿；
- 检索反映最新内容（删掉的不再命中、新增的能命中）；
- 零出网（全链 socket 阻断用例）。
"""

from __future__ import annotations

import math
import re
import socket
from hashlib import blake2b
from pathlib import Path

import pytest

from app.ingestion.scanner import discover_file
from app.ingestion.state_store import DocumentStateStore
from app.ingestion.sync import SyncService
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore

_CJK = re.compile(r"[一-鿿]+")
_LATIN = re.compile(r"[a-z0-9]+")


def _token_stream(text: str) -> list[str]:
    """确定性词元：中文按单字 + 相邻双字（bigram），拉丁按整词。"""
    lower = text.lower()
    tokens: list[str] = []
    for run in _CJK.findall(lower):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_LATIN.findall(lower))
    return tokens


def _hash_index(token: str, dim: int) -> int:
    """词元 -> [0, dim) 稳定索引（blake2b 哈希，跨进程确定）。"""
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化：余弦相似度即为归一化向量的点积。"""
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


class FakeEmbedder:
    """确定性嵌入器：词元哈希到定长向量，L2 归一化（满足 Embedder 协议）。"""

    fingerprint = "fake-e2e"
    dim = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _token_stream(text):
                vec[_hash_index(token, self.dim)] += 1.0
            out.append(_l2_normalize(vec))
        return out


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


# 注：段落内容需达到最小块长（120 字符），否则短段落被合并、块数/复用断言失效。
_FIXTURE = {
    "apple.md": (
        "# 水果笔记\n\n"
        "## 苹果\n\n" + "苹果是一种常见的水果，富含多种营养成分。" * 12 + "\n\n"
        "## 维生素\n\n" + "苹果富含维生素 C，对健康十分有益。" * 12 + "\n"
    ),
    "banana.md": "# 水果笔记\n\n## 香蕉\n\n" + "香蕉富含钾元素，是很好的能量来源。" * 12 + "\n",
    "polar.md": "# 动物\n\n" + "北极熊生活在北极，是最大的陆地食肉动物。" * 12 + "\n",
}


def _seed(root: Path) -> None:
    for name, content in _FIXTURE.items():
        _write(root, name, content)


def _harness(tmp_path: Path) -> tuple[SyncService, FakeEmbedder, ChromaVectorStore, DocumentStateStore]:
    """装配真实 Chroma + 确定性嵌入器 + tmp 状态库，返回 (service, embedder, store, state)。"""
    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-e2e__v1",
        embedder=embedder,
    )
    state = DocumentStateStore(tmp_path / "state.db")
    service = SyncService(embedder=embedder, vectorstore=store, state_store=state)
    return service, embedder, store, state


# ---------------------------------------------------------------- 1. 首同步 = 全新增


def test_first_sync_adds_all_files(tmp_path: Path) -> None:
    """空状态首同步：全部文件新增，块入库，来源根登记。"""
    root = tmp_path / "docs"
    root.mkdir()
    _seed(root)
    service, _embedder, store, state = _harness(tmp_path)
    try:
        report = service.ingest(root)

        assert report.files_added == 3
        assert report.files_unchanged == 0
        assert report.blocks_upserted == store.count() == 4  # apple=2、banana=1、polar=1
        assert report.errors == ()
        assert len(state.load_all()) == 3
        assert state.list_sources() == [root]
    finally:
        state.close()


# ---------------------------------------------------------------- 2. 改一处只更新相关块（核心）


def test_incremental_modified_file_reuses_unchanged_block(tmp_path: Path) -> None:
    """改 apple.md 的「苹果」段：只该块重嵌入，未变「维生素」块 block_id 复用。"""
    root = tmp_path / "docs"
    root.mkdir()
    _seed(root)
    service, _embedder, store, state = _harness(tmp_path)
    try:
        service.ingest(root)

        apple_path = str((root / "apple.md").resolve())
        before_ids = set(store.get_blocks_by_source(apple_path))

        # 只改「苹果」段正文，不动「维生素」段
        _write(
            root,
            "apple.md",
            "# 水果笔记\n\n## 苹果\n\n" + "苹果是蔷薇科植物的果实。" * 12 + "\n\n## 维生素\n\n" + "苹果富含维生素 C，对健康十分有益。" * 12 + "\n",
        )
        report = service.ingest(root)

        assert report.files_updated == 1
        assert report.files_unchanged == 2  # banana / polar
        # 未变「维生素」块 block_id 复用（在 before_ids 中），文本不变
        after_ids = set(store.get_blocks_by_source(apple_path))
        reused = before_ids & after_ids
        assert len(reused) == 1, "未变块应复用旧 block_id"
        # 变化块换成新 block_id，旧的变化块被清
        added_ids = after_ids - before_ids
        assert len(added_ids) == 1
        # 检索新内容命中 apple.md
        retriever = VectorRetriever(store)
        hits = retriever.retrieve("蔷薇科", service._embedder.embed_texts(["蔷薇科"])[0])
        assert hits and any(h.source_file.endswith("apple.md") for h in hits)
    finally:
        state.close()


# ---------------------------------------------------------------- 3. 增/删/改混合增量


def test_incremental_add_delete_modify_mixed(tmp_path: Path) -> None:
    """首同步后：新增 cherry + 删除 polar + 改 apple -> 增量报告与库状态一致、无孤儿。"""
    root = tmp_path / "docs"
    root.mkdir()
    _seed(root)
    service, _embedder, store, state = _harness(tmp_path)
    try:
        service.ingest(root)

        _write(root, "cherry.md", "# 水果笔记\n\n## 樱桃\n\n樱桃色泽鲜红。\n")
        _write(root, "apple.md", "# 水果笔记\n\n## 苹果\n\n" + "苹果是蔷薇科植物的果实。" * 12 + "\n\n## 维生素\n\n" + "苹果富含维生素 C，对健康十分有益。" * 12 + "\n")
        (root / "polar.md").unlink()

        report = service.ingest(root)

        assert report.files_added == 1  # cherry
        assert report.files_updated == 1  # apple
        assert report.files_deleted == 1  # polar
        assert report.files_unchanged == 1  # banana
        assert report.blocks_upserted == 2  # cherry 1 块 + apple 变化块 1 块
        assert report.blocks_deleted == 2  # polar 1 块 + apple 旧变化块 1 块
        assert store.count() == 4  # apple=2 + banana=1 + cherry=1（polar 已清、无孤儿）

        # 库中不存在 polar 的孤儿块
        polar_blocks = store.get_blocks_by_source(str((root / "polar.md").resolve()))
        assert polar_blocks == {}

        # 检索：新内容能命中，被删内容不再命中
        retriever = VectorRetriever(store)
        cherry_hits = retriever.retrieve("樱桃", service._embedder.embed_texts(["樱桃"])[0])
        assert cherry_hits and any(h.source_file.endswith("cherry.md") for h in cherry_hits)
        polar_hits = retriever.retrieve("北极熊", service._embedder.embed_texts(["北极熊"])[0])
        assert not any(h.source_file.endswith("polar.md") for h in polar_hits)
    finally:
        state.close()


# ---------------------------------------------------------------- 3b. 上传式增量 sync_upload 闭环


def test_upload_incremental_closed_loop(tmp_path: Path) -> None:
    """模拟上传式增量：写规范根 -> sync_upload 全新增 -> 改/删/增/散文件 -> 只处理变化。"""
    root = tmp_path / "docs"  # 模拟规范根
    root.mkdir()
    service, _embedder, store, state = _harness(tmp_path)
    try:
        # 首次上传：folder1 两个文件 + 一个散文件
        (root / "folder1").mkdir(parents=True, exist_ok=True)
        _write(root / "folder1", "a.md", "# A\n\n苹果是一种水果。\n")
        _write(root / "folder1", "b.md", "# B\n\n香蕉富含钾。\n")
        _write(root, "note.md", "# N\n\n散文件内容。\n")
        files1 = [
            d
            for p in (root / "folder1" / "a.md", root / "folder1" / "b.md", root / "note.md")
            if (d := discover_file(p)) is not None
        ]
        first = service.sync_upload(files1, deletion_scopes=[root / "folder1"])
        assert first.files_added == 3
        assert first.files_unchanged == 0
        assert store.count() == 3

        # 二轮重传：改 a.md、删 b.md、新增 c.md；散文件 note.md 未重传（不在 scope）
        (root / "folder1" / "a.md").write_text("# A\n\n苹果是蔷薇科果实。\n", encoding="utf-8")
        (root / "folder1" / "b.md").unlink()
        _write(root / "folder1", "c.md", "# C\n\n樱桃色泽鲜红。\n")
        files2 = [
            d
            for p in (root / "folder1" / "a.md", root / "folder1" / "c.md", root / "note.md")
            if (d := discover_file(p)) is not None
        ]
        second = service.sync_upload(files2, deletion_scopes=[root / "folder1"])

        assert second.files_updated == 1  # a.md
        assert second.files_deleted == 1  # b.md（scope 内被删）
        assert second.files_added == 1  # c.md
        assert second.files_unchanged == 1  # note.md 散文件
        assert store.count() == 3  # a(1) + c(1) + note(1)，b 已清、无孤儿

        # 检索反映最新内容：新增的樱桃能命中，被删的香蕉不再命中
        retriever = VectorRetriever(store)
        cherry_hits = retriever.retrieve("樱桃", service._embedder.embed_texts(["樱桃"])[0])
        assert cherry_hits and any(h.source_file.endswith("c.md") for h in cherry_hits)
        banana_hits = retriever.retrieve("香蕉", service._embedder.embed_texts(["香蕉"])[0])
        assert not any(h.source_file.endswith("b.md") for h in banana_hits)
    finally:
        state.close()


# ---------------------------------------------------------------- 4. 零出网


def test_incremental_zero_outbound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接，全量+增量同步仍成功 => 零出网。"""
    _block_non_loopback(monkeypatch)
    root = tmp_path / "docs"
    root.mkdir()
    _seed(root)
    service, _embedder, store, state = _harness(tmp_path)
    try:
        first = service.ingest(root)
        assert first.blocks_upserted == 4

        _write(root, "cherry.md", "# 水果笔记\n\n## 樱桃\n\n樱桃色泽鲜红。\n")
        (root / "polar.md").unlink()
        second = service.ingest(root)
        assert second.files_added == 1
        assert second.files_deleted == 1
        assert store.count() == 4  # 4 - polar(1) + cherry(1)
    finally:
        state.close()


def _block_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接（真实外呼）；回环目的地放行（事件循环自环所需）。"""
    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    original_connect = socket.socket.connect

    def guarded_connect(
        self: socket.socket,
        address: tuple | str | bytes | int,
        *args: object,
        **kwargs: object,
    ) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Unexpected outbound network call to {host!r}")
        return original_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")
