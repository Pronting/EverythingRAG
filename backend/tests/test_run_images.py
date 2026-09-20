"""run 图片引用测试（PRD 13.2「图片引用」/ 1.2 R07 / 7.1 ``verify_image``）。

覆盖三件事，缺一不可：

1. **不静默忽略**：每条引用都带 ``status`` + ``reason`` 落进本区图片库与用户消息元数据；
2. **本区隔离**：原图写进该工作区的 ``image_state.db``（跨区查不到）；
3. **与 ``verify_image`` 数据打通**：走真实 ``IngestionPipeline.process_image_task_and_upsert``
   （只把「视觉模型」和「向量库」换成确定性替身），产出的描述块能被真实
   ``verify_image`` 按 ``image_id`` 命中，并且读得到的正是本区缓存的原图字节。

零出网：URL 通道未注入 fetcher 时不会调用；data URI 路径本地解码。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.agent.tools import ToolContext, ToolName, ToolStatus, build_default_registry
from app.api.run_images import (
    MAX_RUN_IMAGES,
    LocalImageFetcher,
    persist_run_images,
)
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.pipeline import IngestionPipeline

__all__ = ["RecordingVectorStore", "StubVision", "png_data_uri"]


# ---------------------------------------------------------------- 替身与工具


def _png_bytes(color: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def png_data_uri(color: tuple[int, int, int] = (200, 40, 40)) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(color)).decode("ascii")


class RecordingVectorStore:
    """记录 upsert 的最小向量库替身（``verify_image`` 要能 list_blocks 到它）。"""

    def __init__(self) -> None:
        self.blocks: dict[str, tuple[str, dict[str, Any]]] = {}

    def upsert(self, blocks: list[tuple[str, str, list[float], Any]]) -> None:
        for block_id, text, _vector, metadata in blocks:
            payload = metadata.model_dump(mode="json") if hasattr(metadata, "model_dump") else dict(metadata)
            self.blocks[block_id] = (text, payload)

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [(block_id, text, meta) for block_id, (text, meta) in self.blocks.items()]

    def count(self) -> int:
        return len(self.blocks)

    def count_files(self) -> int:
        return 0

    def count_images(self) -> int:
        return len(self.blocks)


class StubVision:
    """确定性视觉模型替身（真实调用点是 ``describe(preprocessed_image)``）。"""

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, image: Any) -> str:
        del image
        self.calls += 1
        return "一张 2x2 的红色测试图片"


class StubEmbedder:
    fingerprint = "run-image-test"
    dim = 8

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 7)] * self.dim for text in texts]


class NoNetworkFetcher:
    """会在被使用时立刻失败的 fetcher：证明 data URI 路径零出网。"""

    def fetch(self, url: str) -> bytes:  # pragma: no cover - 断言用
        raise AssertionError(f"不应出网：{url}")


@pytest.fixture
def image_store(tmp_path: Path):
    store = ImageStateStore(tmp_path / "image_state.db")
    try:
        yield store
    finally:
        store.close()


# ---------------------------------------------------------------- 1. 不静默忽略


async def test_data_uri_is_cached_and_reported(image_store: ImageStateStore) -> None:
    """data URI：原图落本区库 + 任务登记 + status=cached（未配置视觉时如实说明）。"""

    refs = await persist_run_images([png_data_uri()], image_store=image_store)
    assert len(refs) == 1
    ref = refs[0]
    assert ref["status"] == "cached"
    assert ref["kind"] == "data_uri"
    assert ref["content_hash"] and ref["block_id"]
    assert "视觉模型未配置" in ref["reason"]
    # 原图真的在本区库里（可按内容哈希取回）
    assert image_store.has_image_bytes(ref["content_hash"])
    original = image_store.get_image_bytes(ref["content_hash"])
    assert original and original.startswith(b"\x89PNG")
    # 任务已登记（启动续跑/worker 能接手）
    task = image_store.get_task(ref["block_id"])
    assert task is not None and task.doc_id == ref["content_hash"]
    # 元数据里不留整段 base64（只保留 MIME 头 + 说明）
    assert len(ref["src_ref"]) < 200
    assert "data:image/png" in ref["src_ref"]


async def test_url_without_fetcher_is_marked_unavailable(image_store: ImageStateStore) -> None:
    """未注入下载通道时不抓取链接，但**必须**如实标记（不是静默忽略）。"""

    refs = await persist_run_images(["https://example.com/a.png"], image_store=image_store)
    assert refs[0]["status"] == "unavailable"
    assert refs[0]["kind"] == "url"
    assert "未配置图片下载通道" in refs[0]["reason"]
    assert image_store.count_status().get("pending", 0) == 0  # 没有任何任务被登记


async def test_invalid_reference_is_rejected(image_store: ImageStateStore) -> None:
    refs = await persist_run_images(["ftp://example.com/a.png"], image_store=image_store)
    assert refs[0]["status"] == "rejected"
    assert "data:image" in refs[0]["reason"]


async def test_too_many_refs_are_rejected_one_by_one(image_store: ImageStateStore) -> None:
    refs = await persist_run_images(
        [png_data_uri() for _ in range(MAX_RUN_IMAGES + 1)], image_store=image_store
    )
    assert len(refs) == MAX_RUN_IMAGES + 1
    assert refs[-1]["status"] == "rejected"
    assert "最多附带" in refs[-1]["reason"]


async def test_broken_data_uri_is_rejected_not_raised(image_store: ImageStateStore) -> None:
    refs = await persist_run_images(["data:image/png;base64,!!!not-base64!!!"], image_store=image_store)
    assert refs[0]["status"] == "rejected"
    assert refs[0]["reason"]


async def test_missing_image_store_does_not_pretend_success() -> None:
    refs = await persist_run_images([png_data_uri()], image_store=None)
    assert refs[0]["status"] == "unavailable"
    assert "图片库不可用" in refs[0]["reason"]


# ---------------------------------------------------------------- 2. 本区隔离


async def test_original_is_workspace_scoped(tmp_path: Path) -> None:
    """两个工作区各自的图片库互不可见（PRD 1.2 R07）。"""

    first = ImageStateStore(tmp_path / "ws1" / "image_state.db")
    second = ImageStateStore(tmp_path / "ws2" / "image_state.db")
    try:
        refs = await persist_run_images([png_data_uri()], image_store=first)
        content_hash = refs[0]["content_hash"]
        assert first.has_image_bytes(content_hash)
        assert not second.has_image_bytes(content_hash)
        assert second.get_task(refs[0]["block_id"]) is None
    finally:
        first.close()
        second.close()


# ---------------------------------------------------------------- 3. 与 verify_image 打通


async def test_described_image_is_verifiable_by_verify_image(
    tmp_path: Path, image_store: ImageStateStore
) -> None:
    """真实管线（假视觉 + 记录型向量库）：描述块可被真实 ``verify_image`` 命中。"""

    vector_store = RecordingVectorStore()
    vision = StubVision()
    pipeline = IngestionPipeline(
        embedder=StubEmbedder(),
        vectorstore=vector_store,
        vision=vision,
        fetcher=LocalImageFetcher(NoNetworkFetcher()),
        state_store=image_store,
    )
    refs = await persist_run_images(
        [png_data_uri()], image_store=image_store, fetcher=NoNetworkFetcher(), pipeline=pipeline
    )
    ref = refs[0]
    assert ref["status"] == "described", ref
    assert vision.calls == 1
    assert vector_store.list_blocks(), "描述块必须进向量库，否则 verify_image 找不到"

    # 真实 verify_image：按 content_hash 命中本区原图
    reader_inputs: list[dict[str, Any]] = []

    def reader(chunk: Any, question: str) -> str:
        """真实读取路径：按内容哈希从本区图片库取原图字节。"""
        content_hash = str(chunk.metadata.get("image_content_hash") or "")
        data = image_store.get_image_bytes(content_hash)
        assert data, "本区图片库必须能按内容哈希取回原图"
        reader_inputs.append({"hash": content_hash, "bytes": len(data), "question": question})
        return "原图显示：红色方块"

    context = ToolContext(
        workspace=_StubWorkspace(tmp_path / "ws1"),
        vector_store=vector_store,
        image_reader=reader,
    )
    registry = build_default_registry()
    result = await registry.execute(
        ToolName.VERIFY_IMAGE,
        {"image_id": ref["image_id"], "question": "这张图是什么颜色？"},
        context,
        record_operation=False,
    )
    assert result.status is ToolStatus.OK
    # 两个 hash 是两个轴（本用例把这条事实钉住，避免日后被当成同一个）：
    # - ref["content_hash"] = 请求侧缓存键（原始字节 sha256，用于去重与前端引用）；
    # - data["image_content_hash"] = 描述块自己的身份（既有管线对预处理后 JPEG 的 xxh64），
    #   本区图片库的 image_originals 就按它存放原图 —— 上面 reader 里已实测可取回。
    assert result.data["image_content_hash"]
    assert result.data["image_content_hash"] != ref["content_hash"]
    assert image_store.has_image_bytes(result.data["image_content_hash"])
    assert result.data["verification"] == "原图显示：红色方块"
    assert reader_inputs and reader_inputs[0]["bytes"] > 0

    # 也可按 block_id 命中（两种身份都接受，二者此处相同）
    by_block = await registry.execute(
        ToolName.VERIFY_IMAGE,
        {"image_id": ref["block_id"], "question": "再核对一次"},
        context,
        record_operation=False,
    )
    assert by_block.status is ToolStatus.OK


async def test_cross_workspace_verify_image_is_not_found(tmp_path: Path) -> None:
    """另一区的图片库与向量库都查不到该图片（跨区等同不存在）。"""

    first = ImageStateStore(tmp_path / "ws1" / "image_state.db")
    second = ImageStateStore(tmp_path / "ws2" / "image_state.db")
    try:
        refs = await persist_run_images([png_data_uri()], image_store=first)
        context = ToolContext(
            workspace=_StubWorkspace(tmp_path / "ws2"),
            vector_store=RecordingVectorStore(),  # B 区的空向量库
            image_reader=lambda chunk, question: "不应被调用",
        )
        result = await build_default_registry().execute(
            ToolName.VERIFY_IMAGE,
            {"image_id": refs[0]["image_id"], "question": "在吗"},
            context,
            record_operation=False,
        )
        assert result.status is ToolStatus.NOT_FOUND
        assert result.data["code"] == "image_not_found"
    finally:
        first.close()
        second.close()


class _StubWorkspace:
    """只带工具层需要的身份/版本属性。"""

    def __init__(self, root: Path) -> None:
        self.workspace_id = "ws-1"
        self.name = "测试工作区"
        self.knowledge_revision = 0
        self.artifacts_dir = root / "artifacts"
