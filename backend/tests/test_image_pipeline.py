"""IngestionPipeline 图片集成单测：识图描述入向量库 + 内容哈希去重 + 失败跳过。

全部用 fake（FakeEmbedder / FakeVectorStore / FakeVision / FakeFetcher）+ 内存 PNG，
零出网、不加载真实 Chroma / VLM。识图与描述是否「正确」另由真实用例集成验证。
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from app.ingestion.image_fetch import ImageFetchError
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import SourceType


def _png_bytes(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, float(len(t))] for t in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, list[float], object]] = {}

    def upsert(self, blocks: list) -> None:
        for block_id, text, vector, meta in blocks:
            self.blocks[block_id] = (text, vector, meta)

    def delete_by_ids(self, block_ids: list[str]) -> None:
        pass

    def delete_by_where(self, where: dict) -> None:
        pass

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        return {}

    def query(self, vector, top_k, where=None) -> list:
        return []

    def count(self) -> int:
        return len(self.blocks)


class FakeVision:
    def __init__(self, description: str = "画面主体：架构图") -> None:
        self._description = description
        self.calls = 0

    def describe(self, image_bytes: bytes, prompt: str = "", mime: str = "image/jpeg") -> str:
        self.calls += 1
        return self._description


class FakeFetcher:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        return self._data


class BoomFetcher:
    def fetch(self, url: str) -> bytes:
        raise ImageFetchError("下载失败: HTTP 404")


class BoomVision:
    """describe 抛非 VisionProviderError 的意外异常（如嵌入/客户端崩溃）。"""

    def describe(self, image_bytes: bytes, prompt: str = "", mime: str = "image/jpeg") -> str:
        raise RuntimeError("unexpected vision failure")


def _pipeline(vision=None, fetcher=None) -> tuple[IngestionPipeline, FakeVectorStore]:
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        embedder=FakeEmbedder(),
        vectorstore=store,
        vision=vision,
        fetcher=fetcher,
    )
    return pipeline, store


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _image_blocks(store: FakeVectorStore) -> list:
    return [meta for _, _, meta in store.blocks.values() if meta.source_type == SourceType.IMAGE_DESCRIPTION]


def test_image_described_and_indexed(tmp_path: Path) -> None:
    """图片被识图并产出 image_description 块：元数据带 URL/内容哈希/来源继承。"""
    root = tmp_path / "docs"
    root.mkdir()
    url = "https://cdn.nlark.com/yuque/0/2025/png/1.png"
    _write(root, "a.md", f"# 架构\n\n正文说明内容段落。\n\n![]({url})")

    vision = FakeVision()
    fetcher = FakeFetcher(_png_bytes())
    pipeline, store = _pipeline(vision=vision, fetcher=fetcher)
    pipeline.ingest(root)

    assert vision.calls == 1
    assert fetcher.calls == [url]
    images = _image_blocks(store)
    assert len(images) == 1
    meta = images[0]
    assert meta.chunk_type == "image_description"
    assert meta.image_path == url
    assert meta.image_content_hash  # 非空内容哈希
    assert meta.heading_path == "架构"
    # 描述文本已入库（与文本同向量库）
    texts = [text for text, _, _ in store.blocks.values()]
    assert any("架构图" in text for text in texts)


def test_same_image_across_files_deduped(tmp_path: Path) -> None:
    """两文件引用同一图床图（同字节）-> 识图只算一次，各文件独立块。"""
    root = tmp_path / "docs"
    root.mkdir()
    url = "https://cdn.example.com/shared.png"
    _write(root, "a.md", f"# A\n\n![]({url})")
    _write(root, "b.md", f"# B\n\n![]({url})")

    vision = FakeVision()
    pipeline, store = _pipeline(vision=vision, fetcher=FakeFetcher(_png_bytes()))
    pipeline.ingest(root)

    assert vision.calls == 1  # 内容哈希去重：同图只识一次
    images = _image_blocks(store)
    assert len(images) == 2  # 两个引用位置各一个块
    hashes = {meta.image_content_hash for meta in images}
    assert len(hashes) == 1  # 共享同一内容哈希


def test_image_fetch_failure_skips_image_keeps_text(tmp_path: Path) -> None:
    """图片下载失败 -> 跳过图片（不产块），文本块照常入库、不中断。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# 正文\n\n正文说明内容段落。\n\n![](https://cdn.example.com/dead.png)")

    vision = FakeVision()
    pipeline, store = _pipeline(vision=vision, fetcher=BoomFetcher())
    report = pipeline.ingest(root)

    assert vision.calls == 0
    assert report.files_parsed == 1
    assert _image_blocks(store) == []
    # 文本块仍入库
    assert any(meta.source_type == SourceType.DOCUMENT for _, _, meta in store.blocks.values())


def test_vision_unconfigured_skips_images(tmp_path: Path) -> None:
    """未注入识图模型 -> 图片不产块，文本正常入库（PRD 4.3 验收 2）。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# 正文\n\n正文说明内容段落。\n\n![](https://cdn.example.com/a.png)")

    pipeline, store = _pipeline()  # 无 vision / fetcher
    report = pipeline.ingest(root)

    assert report.files_parsed == 1
    assert _image_blocks(store) == []
    assert any(meta.source_type == SourceType.DOCUMENT for _, _, meta in store.blocks.values())


def test_image_unexpected_exception_keeps_text(tmp_path: Path) -> None:
    """识图抛意外异常（非 VisionProviderError）-> 文本仍入库，文件不标记失败。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# 正文\n\n正文说明内容段落。\n\n![](https://cdn.example.com/a.png)")

    pipeline, store = _pipeline(vision=BoomVision(), fetcher=FakeFetcher(_png_bytes()))
    report = pipeline.ingest(root)

    assert report.files_parsed == 1  # 图片异常不拖累文本：文件不失败
    assert _image_blocks(store) == []
    assert any(meta.source_type == SourceType.DOCUMENT for _, _, meta in store.blocks.values())


def _image_texts(store: FakeVectorStore) -> list[str]:
    return [text for text, _, meta in store.blocks.values() if meta.source_type == SourceType.IMAGE_DESCRIPTION]


def test_image_block_injects_heading_context(tmp_path: Path) -> None:
    """图片描述块文本注入标题路径（主题上下文），使图片可被主题词检索命中。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# AB测试\n\n![](https://cdn.example.com/a.png)")

    pipeline, store = _pipeline(vision=FakeVision(), fetcher=FakeFetcher(_png_bytes()))
    pipeline.ingest(root)

    texts = _image_texts(store)
    assert len(texts) == 1
    assert "【所属主题】AB测试" in texts[0]  # 主题上下文已注入
    assert "架构图" in texts[0]  # 识图描述仍在


def test_image_block_headingless_no_theme_prefix(tmp_path: Path) -> None:
    """无标题文档的图片块不注入主题前缀（原样纯描述）。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "![](https://cdn.example.com/a.png)")

    pipeline, store = _pipeline(vision=FakeVision(), fetcher=FakeFetcher(_png_bytes()))
    pipeline.ingest(root)

    texts = _image_texts(store)
    assert len(texts) == 1
    assert "【所属主题】" not in texts[0]


def test_same_content_different_urls_deduped(tmp_path: Path) -> None:
    """不同图床 URL 但同一图片内容 -> 只识图一次（内容哈希去重，不依赖 URL）。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# A\n\n![](https://cdn.a.com/x.png)\n\n![](https://cdn.b.com/y.png)")

    vision = FakeVision()
    # FakeFetcher 对任何 URL 都返回同一张图 -> 内容哈希相同
    pipeline, store = _pipeline(vision=vision, fetcher=FakeFetcher(_png_bytes()))
    pipeline.ingest(root)

    assert vision.calls == 1  # 同内容只识一次
    images = _image_blocks(store)
    assert len(images) == 2  # 两个引用各一个块
    assert len({m.image_content_hash for m in images}) == 1  # 共享同一内容哈希


def test_persistent_dedup_across_imports(tmp_path: Path) -> None:
    """共享 state_store 的两次导入遇到同图 -> 第二次复用描述，不再识图。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write(root, "a.md", "# A\n\n![](https://cdn.example.com/a.png)")

    state_store = ImageStateStore(tmp_path / "img_state.db")
    vision = FakeVision()

    def make_pipeline() -> IngestionPipeline:
        return IngestionPipeline(
            embedder=FakeEmbedder(),
            vectorstore=FakeVectorStore(),
            vision=vision,
            fetcher=FakeFetcher(_png_bytes()),
            state_store=state_store,
        )

    make_pipeline().ingest(root)  # 第一次：识图 + 落描述缓存
    assert vision.calls == 1

    make_pipeline().ingest(root)  # 第二次：复用持久化描述，不再识图
    assert vision.calls == 1
