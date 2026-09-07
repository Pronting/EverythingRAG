import io
from dataclasses import replace

import pytest
from PIL import Image

from app.generation.chat_service import ChatService
from app.generation.image_reader import KnowledgeImageReader
from app.ingestion.image_document import build_image_document
from app.ingestion.image_state_store import ImageStateStore
from tests.test_chat_service import FakeChatModel, FakeEmbedder, FakeRetriever, _chunk


def test_summary_and_full_table_are_independent_and_watermark_is_not_indexed():
    table = "|日期|GMV|\n|---|---|\n" + "|2024-12-02|4826万|\n" * 100
    raw = "【检索摘要】黑五每日GMV和转化率\n【原始证据】" + table
    doc = build_image_document(raw, "黑五成绩", "运营复盘")
    assert len(doc.index_text) < 100
    assert "4826万" not in doc.index_text
    assert table.strip() in doc.evidence_text
    legacy = build_image_document("【画面主体】运营表格。\n【关键实体与场景】GMV和日期。背景有水印姓名1234。\n【图中文字(OCR)】GMV：4826万")
    assert "姓名1234" not in legacy.index_text
    assert "姓名1234" in legacy.evidence_text  # Filtering the index never destroys source evidence.
    assert "4826万" in legacy.evidence_text


def test_reader_uses_cached_original_without_downloading(monkeypatch, tmp_path):
    cache = ImageStateStore(tmp_path / "images.db")
    output = io.BytesIO()
    Image.new("RGB", (1800, 120), "white").save(output, format="PNG")
    cache.save_image_bytes("hash", output.getvalue())
    from app.generation import image_reader
    monkeypatch.setattr(image_reader.ImageFetcher, "fetch", lambda *args: pytest.fail("unexpected download"))
    class Vision:
        def describe(self, data, prompt):
            assert Image.open(io.BytesIO(data)).width == 1800
            assert "2024-12-02" in prompt
            return "GMV=4826万"
    chunk = replace(_chunk("image"), metadata={"image_content_hash": "hash"})
    assert KnowledgeImageReader(Vision(), cache)(chunk, "2024-12-02的GMV是多少") == "GMV=4826万"
    cache.clear()
    assert cache.get_image_bytes("hash") is None
    cache.close()


def test_original_image_endpoint_returns_cached_bytes_and_handles_missing(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.deps import get_image_state_store
    from app.main import app
    cache = ImageStateStore(tmp_path / "images.db")
    output = io.BytesIO()
    Image.new("RGB", (64, 64)).save(output, format="PNG")
    cache.save_image_bytes("a" * 16, output.getvalue())
    app.dependency_overrides[get_image_state_store] = lambda: cache
    client = TestClient(app)
    try:
        response = client.get("/api/knowledge/images/" + "a" * 16)
        assert response.status_code == 200
        assert response.content == output.getvalue()
        assert response.headers["content-type"] == "image/png"
        assert client.get("/api/knowledge/images/" + "b" * 16).status_code == 404
        assert client.get("/api/knowledge/images/invalid").status_code == 404
    finally:
        app.dependency_overrides.pop(get_image_state_store, None)
        cache.close()


@pytest.mark.asyncio
async def test_verified_image_replaces_old_ocr_and_keeps_citation_identity():
    chunk = replace(_chunk("image", "旧识别：GMV=错误数字", chunk_type="image_description"),
                    metadata={"image_path": "https://example.com/table.png"})
    model = FakeChatModel()
    service = ChatService(FakeEmbedder(), FakeRetriever([chunk]), model,
                          image_reader=lambda *args: "原图：GMV=4826万")
    frames = [f async for f in service.stream_answer("图表数据是多少")]
    source = next(f for f in frames if f["type"] == "meta")["sources"][0]
    assert source["block_id"] == "image"
    assert source["image_url"] == "https://example.com/table.png"
    assert "4826万" in source["text"] and "错误数字" not in source["text"]
    assert frames[0] == {"type": "tool", "tool": "image_verify", "status": "running"}


@pytest.mark.asyncio
async def test_verification_failure_is_bounded_and_does_not_abort_answer():
    chunks = [replace(_chunk(str(i), "历史证据", chunk_type="image_description"),
                      metadata={"image_path": f"https://example.com/{i}.png"}) for i in range(4)]
    calls = []
    def fail(chunk, question):
        calls.append(chunk.block_id)
        raise RuntimeError("secret upstream error")
    service = ChatService(FakeEmbedder(), FakeRetriever(chunks), FakeChatModel(), image_reader=fail)
    frames = [f async for f in service.stream_answer("图片数据是多少")]
    assert len(calls) == 2
    assert frames[-1]["type"] == "done"
    assert "secret" not in str(frames)
    assert "无法核对" in next(f for f in frames if f["type"] == "meta")["sources"][0]["text"]


@pytest.mark.asyncio
async def test_ordinary_question_does_not_call_vision():
    service = ChatService(FakeEmbedder(), FakeRetriever([_chunk("image", chunk_type="image_description")]),
                          FakeChatModel(), image_reader=lambda *args: pytest.fail("unexpected vision call"))
    frames = [f async for f in service.stream_answer("风控的策略是什么")]
    assert not any(f["type"] == "tool" for f in frames)


@pytest.mark.asyncio
async def test_verification_updates_duplicate_images_and_removes_stale_hydrated_numbers():
    metadata = {"image_content_hash": "a" * 16}
    image = replace(_chunk("image", "旧数字", chunk_type="image_description"), metadata=metadata)
    alias = replace(image, block_id="alias")
    prose = replace(image, block_id="prose", chunk_type="text", text="正文事实",
                    context_text="正文事实\n\n同标题图片信息：旧数字")
    calls = []
    def verify(*args):
        calls.append(args)
        return "核对后的数字"
    service = ChatService(FakeEmbedder(), FakeRetriever([image, alias, prose]), FakeChatModel(),
                          image_reader=verify)
    frames = [f async for f in service.stream_answer("图片数据是多少")]
    sources = next(f for f in frames if f["type"] == "meta")["sources"]
    assert len(calls) == 1
    assert all("旧数字" not in s["text"] for s in sources)
    assert sources[0]["text"] == sources[1]["text"]
    assert sources[2]["text"] == "正文事实"
