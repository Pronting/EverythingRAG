"""E2E 最小闭环：目录 -> 管线导入 -> 向量检索 -> POST /api/chat SSE 答案（任务 9）。

确定性 fake（FakeEmbedder / FakeChatModel）避免真实 bge-m3 下载与真实云端出网；
但管线组件（scan / parse / chunk / Chroma / retriever / chat_service / HTTP / SSE）
全真实。隐私红线：测试零外发，另有全链 socket 阻断用例实证。
"""

from __future__ import annotations

import json
import math
import re
import socket
from hashlib import blake2b
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.chat import get_chat_service
from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.ingestion.pipeline import IngestionPipeline
from app.main import app
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore

_CJK = re.compile(r"[一-鿿]+")
_LATIN = re.compile(r"[a-z0-9]+")


def _token_stream(text: str) -> list[str]:
    """确定性词元：中文按单字 + 相邻双字（bigram），拉丁按整词。

    保证「关键词查询」能召回含该词的 chunk（余弦即词项重叠度量）。
    """
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
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


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


class FakeChatModel:
    """确定性对话模型：从 user 消息提取问题，产出 2 个增量 token（答案含关键字）。"""

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        question = _extract_question(messages[-1]["content"]) if messages else ""
        for token in ("答案是：", question):
            yield ChatChunk("content", token)


def _extract_question(user_content: str) -> str:
    """从 ChatService 组装的 user 内容中提取原问题（"问题：{msg}\n\n参考上下文..."）。"""
    head = user_content.split("\n", 1)[0]
    return head.removeprefix("问题：")


def _write_fixture(root: Path) -> list[Path]:
    """写 3 个含唯一关键字的 MD：嵌套标题 / 无标题 / 代码块。返回文件路径列表。"""
    files = {
        "apple.md": (
            "# 水果笔记\n\n"
            "## 苹果\n\n"
            "苹果是一种常见的水果，富含维生素 C 与膳食纤维。\n"
        ),
        "orange.md": (
            "橘子属于柑橘类水果，果皮可入药，果肉多汁。\n\n"
            "```python\nprint(\"orange\")\n```\n"
        ),
        "polar.md": (
            "# 动物\n\n"
            "北极熊生活在北极，是最大的陆地食肉动物。\n\n"
            "## 北极\n\n"
            "北极地区气候严寒，常年覆盖冰雪。\n"
        ),
    }
    paths: list[Path] = []
    for name, content in files.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """解析 SSE body：每行 'data: {json}' -> 帧 dict 列表。"""
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


# ---------------------------------------------------------------- 1. 管线集成


def test_pipeline_ingest_retrieval_closed_loop(tmp_path: Path) -> None:
    """IngestionPipeline 导入 fixture -> 计数正确 -> Chroma 检索命中带来源块。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write_fixture(root)

    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-e2e__v1",
        embedder=embedder,
    )
    report = IngestionPipeline(embedder=embedder, vectorstore=store).ingest(root)

    # 计数与 store 一致（apple=1、orange=2 含代码块、polar 两个标题区间各 1）
    assert report.files_scanned == 3
    assert report.files_parsed == 3
    assert report.files_skipped == 0
    assert report.chunks == 5
    assert report.blocks_upserted == 5
    assert report.errors == ()
    assert store.count() == 5

    # 检索：关键字「橘子」命中含该词的块，来源指向 fixture 文件
    retriever = VectorRetriever(store)
    hits = retriever.retrieve("橘子", embedder.embed_texts(["橘子"])[0])
    assert hits, "关键字查询应命中至少一个块"
    first = hits[0]
    assert "橘子" in first.text
    assert first.source_file.endswith("orange.md")
    assert Path(first.source_file).is_file()  # 来源是真实文件（可回溯）

    # 另一关键字「北极熊」同样可召回
    polar_hits = retriever.retrieve("北极熊", embedder.embed_texts(["北极熊"])[0])
    assert polar_hits and "北极熊" in polar_hits[0].text
    assert polar_hits[0].source_file.endswith("polar.md")


# ---------------------------------------------------------------- 2. HTTP 闭环


def test_http_closed_loop_sse_with_sources(tmp_path: Path) -> None:
    """POST /api/chat 全真实链路：meta -> token(s) -> done；sources 带 .md；答案含关键字。"""
    root = tmp_path / "docs"
    root.mkdir()
    _write_fixture(root)

    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-e2e__v1",
        embedder=embedder,
    )
    IngestionPipeline(embedder=embedder, vectorstore=store).ingest(root)

    retriever = VectorRetriever(store)
    service = ChatService(embedder=embedder, retriever=retriever, chat_model=FakeChatModel())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        with client.stream("POST", "/api/chat", json={"message": "橘子是什么？"}) as resp:
            status = resp.status_code
            body = resp.read().decode("utf-8")
    finally:
        app.dependency_overrides.clear()

    assert status == 200
    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["meta", "token", "token", "done"]

    sources = frames[0]["sources"]
    assert sources, "检索应命中至少一个来源"
    assert all(source["source_file"].endswith(".md") for source in sources)
    assert any("橘子" in source["text"] for source in sources)

    answer = "".join(frame["text"] for frame in frames if frame["type"] == "token")
    assert "橘子" in answer  # 答案含 fixture 关键字


# ---------------------------------------------------------------- 3. 零出网


def test_zero_outbound_full_closed_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接，全链（导入 -> 检索 -> SSE）仍成功 => 零出网。"""
    _block_non_loopback(monkeypatch)
    root = tmp_path / "docs"
    root.mkdir()
    _write_fixture(root)

    embedder = FakeEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-e2e__v1",
        embedder=embedder,
    )
    report = IngestionPipeline(embedder=embedder, vectorstore=store).ingest(root)
    assert report.blocks_upserted == 5

    retriever = VectorRetriever(store)
    service = ChatService(embedder=embedder, retriever=retriever, chat_model=FakeChatModel())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        with client.stream("POST", "/api/chat", json={"message": "北极熊生活在哪里？"}) as resp:
            body = resp.read().decode("utf-8")
    finally:
        app.dependency_overrides.clear()

    frames = _parse_sse(body)
    assert [frame["type"] for frame in frames] == ["meta", "token", "token", "done"]
    assert any(source["source_file"].endswith(".md") for source in frames[0]["sources"])


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
