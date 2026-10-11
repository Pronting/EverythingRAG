"""E2E 最小闭环：目录 -> 管线导入 -> 向量检索 -> **run API + v2 事件** 答案（任务 9 → T09 迁移）。

V1.1.0 的闭环走 ``POST /api/chat``（旧帧 ``meta/token/done``）。V1.2.0 改成 run API：
``POST /api/workspaces/{w}/runs`` 建任务 + ``GET .../runs/{r}/events`` 订阅 v2 事件。
**产品价值不变**：目录 -> 导入 -> 检索 -> 带来源的答案，全链路真实（真实 Chroma /
真实切块 / 真实检索 / 真实工具与核对），只有**模型**是确定性替身（识别/合成不调模型，
核对用假模型 Verifier）。

零出网：全链阻断非回环连接（PRD 隐私红线）。
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

from app.agent import events as agent_events
from app.agent.app_store import AppStore
from app.agent.contracts import EventType, RunStatus
from app.agent.runtime import RunManager
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore

_CJK = re.compile(r"[一-鿿]+")
_LATIN = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------- 确定性替身


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
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


class FakeEmbedder:
    """确定性嵌入器：词元哈希到定长向量并 L2 归一化。"""

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


class StubContext:
    """``RunManager`` 需要的只读上下文（真实布局：每区一个 checkpoints.db）。"""

    def __init__(self, workspace_id: str, root: Path) -> None:
        self.workspace_id = workspace_id
        self.root = root

    @property
    def knowledge_revision(self) -> int:
        return 0

    @property
    def checkpoints_db(self) -> Path:
        return self.root / "checkpoints.db"


class StubFactory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(self, workspace_id: str | None = None) -> StubContext:
        key = workspace_id or "ws-1"
        self.root.mkdir(parents=True, exist_ok=True)
        return StubContext(key, self.root)


def _write_fixture(root: Path) -> None:
    files = {
        "apple.md": "# 水果笔记\n\n## 苹果\n\n苹果是一种常见的水果，富含维生素 C 与膳食纤维。\n",
        "orange.md": "橘子属于柑橘类水果，果皮可入药，果肉多汁。\n",
        "polar.md": "# 动物\n\n北极熊生活在北极，是最大的陆地食肉动物。\n",
    }
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------- 夹具


class _EchoChatModel:
    """确定性离线对话模型：把提示词里给出的证据占位符原样抄成正文（零出网）。

    闭环用例必须真的走「模型撰写候选答案」这条链路：``synthesize_answer`` 在没有可用
    模型时抛 ``SynthesisUnavailable``（不再回落确定性拼接稿，见 ``verification`` 的说明），
    因此这里按真实用户已配置模型的情形注入一个假模型，而不是让链路走已删除的兜底路径。
    """

    supports_image_input = False

    async def stream_chat(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        from app.generation.base import ChatChunk

        text = "\n".join(str(item.get("content") or "") for item in messages)
        # 只认真实证据占位符（`ev-` + 24 位十六进制）：提示词里的示例 `{{ev-xxxx}}` 不算。
        evidence = [
            (match.group(1), match.group(2).strip())
            for match in re.finditer(
                r"\{\{(ev-[0-9a-f]{8,})\}\}\s*←\s*([^\n]+)", text
            )
        ]
        if evidence:
            # 正文确实用上召回原文（而不是只抄占位符），引用编号仍由程序分配。
            body = "；".join(f"{snippet}{{{{{eid}}}}}" for eid, snippet in evidence)
        else:
            body = "据通用知识作答。"
        yield ChatChunk("content", body)


@pytest.fixture(autouse=True)
def _offline_chat_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """闭环用例统一注入离线假对话模型（真实链路 + 零出网）。"""

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config",
        lambda _config: _EchoChatModel(),
    )


class ClosedLoop:
    """一次完整的「导入 -> 建 run -> 事件流」环境。"""

    def __init__(self, store: AppStore, manager: RunManager, vector_store: Any, root: Path) -> None:
        self.store = store
        self.manager = manager
        self.vector_store = vector_store
        self.root = root

    async def run_question(self, question: str) -> tuple[str, list[agent_events.StoredEvent]]:
        run = await self.manager.create_run(
            workspace_id="ws-1", conversation_id="conv-1", question=question
        )
        await self.manager.run_to_completion(run.id, timeout=120)
        return run.id, await self.manager.read_events(run.id)


async def _build(tmp_path: Path) -> ClosedLoop:
    root = tmp_path / "docs"
    root.mkdir()
    _write_fixture(root)

    embedder = FakeEmbedder()
    vector_store = ChromaVectorStore(
        persist_dir=tmp_path / "index",
        collection_name="chunks__fake-e2e__v1",
        embedder=embedder,
    )
    report = IngestionPipeline(embedder=embedder, vectorstore=vector_store).ingest(root)
    assert report.blocks_upserted > 0

    store = await AppStore.open(tmp_path / "app.db")
    await store.create_workspace("ws-1", "sk-ws-1", name="E2E 工作区")
    await store.create_conversation("conv-1", "ws-1")

    retriever = VectorRetriever(vector_store)

    def tool_services(context: Any) -> dict[str, Any]:
        del context
        return {
            "vector_store": vector_store,
            "retriever": retriever,
            "embedder": embedder,
            "extract_provider": None,
            "search_provider": None,
            "_catalog": None,
            "_memory": None,
        }

    manager = RunManager(
        store=store,
        context_factory=StubFactory(tmp_path / "ws-root"),
        registry=None,  # 用真实工具注册表（build_default_registry）
        tool_services=tool_services,
        private_names=(),
    )
    return ClosedLoop(store, manager, vector_store, root)


def _types(events: list[agent_events.StoredEvent]) -> list[str]:
    return [event.type.value for event in events]


# ---------------------------------------------------------------- 1. 管线集成（未改）


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
    assert report.files_scanned == 3
    assert report.errors == ()
    assert store.count() == report.chunks

    retriever = VectorRetriever(store)
    hits = retriever.retrieve("橘子", embedder.embed_texts(["橘子"])[0])
    assert hits and "橘子" in hits[0].text
    assert hits[0].source_file.endswith("orange.md")
    assert Path(hits[0].source_file).is_file()


# ---------------------------------------------------------------- 2. run API 闭环


async def test_run_api_closed_loop_with_v2_events(tmp_path: Path) -> None:
    """全真实链路（假模型）：run.status -> answer.meta -> answer.delta -> run.completed。"""

    env = await _build(tmp_path)
    try:
        run_id, events = await env.run_question("笔记里说橘子属于什么类水果？")
        types = _types(events)
        run = await env.store.get_run(run_id)
        assert run.status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_GAPS}, types

        # ① 首帧是 run.status（前端据此进入过程区域）
        assert types[0] == EventType.RUN_STATUS.value
        # ② 收尾帧是 run.completed（订阅据此关闭）
        assert types[-1] == EventType.RUN_COMPLETED.value
        # ③ 答案帧序：answer.meta → answer.delta* → run.completed
        meta_index = types.index(EventType.ANSWER_META.value)
        delta_index = types.index(EventType.ANSWER_DELTA.value)
        assert meta_index < delta_index < len(types) - 1

        # ④ 来源：证据来自真实检索（orange.md），引用编号已分配（PRD 8.3）
        artifact = await env.store.get_answer(run.answer_id)
        assert artifact is not None
        assert artifact.citations, "答案必须带引用编号"
        assert artifact.citations[0].label.startswith(("S", "W"))
        answer_text = "".join(
            event.payload.get("text", "")
            for event in events
            if event.type is EventType.ANSWER_DELTA
        )
        assert "[S" in answer_text or "[W" in answer_text
        assert answer_text.strip()

        # ⑤ 每条 SSE data 都是 v2 事件对象（可被前端解析器直接消费）
        for event in events:
            body = json.loads(agent_events.dumps_event(event))
            assert body["schema_version"] == agent_events.EVENT_SCHEMA_VERSION
            assert body["type"] == event.type.value
    finally:
        await env.manager.aclose()
        await env.store.close()


async def test_run_api_closed_loop_recalls_orange_chunk(tmp_path: Path) -> None:
    """检索确实命中了 fixture 的「橘子」块：证据 source_ref 指向 orange.md。"""

    env = await _build(tmp_path)
    try:
        run_id, events = await env.run_question("笔记里说橘子属于什么类水果？")
        evidence_events = [
            event for event in events if event.type is EventType.EVIDENCE_ADDED
        ]
        assert evidence_events, "研究阶段必须发布 evidence.added"
        items = [
            item for event in evidence_events for item in event.payload.get("items", [])
        ]
        # 知识来源的 source_ref 是「文档指纹 + 块位置」而不是文件名（PRD 8.1），
        # 因此召回的证据要按**正文**判定：片段必须来自 orange.md 的那句话。
        assert items, "evidence.added 必须带条目"
        assert any("橘子" in str(item.get("snippet") or "") for item in items), items
        assert all(item.get("source_ref") for item in items)
        artifact = await env.store.get_answer((await env.store.get_run(run_id)).answer_id)
        assert artifact is not None
        # 答案正文来自召回原文（不是模板空话），且引用编号指向该证据
        assert "橘子" in artifact.content
        assert artifact.citations and artifact.citations[0].source_ref
    finally:
        await env.manager.aclose()
        await env.store.close()


# ---------------------------------------------------------------- 3. 零出网


async def test_zero_outbound_full_closed_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接，全链（导入 -> 检索 -> run -> 事件）仍成功 => 零出网。"""

    _block_non_loopback(monkeypatch)
    env = await _build(tmp_path)
    try:
        run_id, events = await env.run_question("北极熊生活在哪里？笔记里有写吗？")
        types = _types(events)
        assert types[0] == EventType.RUN_STATUS.value
        assert types[-1] == EventType.RUN_COMPLETED.value
        run = await env.store.get_run(run_id)
        assert run.answer_id
    finally:
        await env.manager.aclose()
        await env.store.close()


def _block_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """阻断一切非回环连接（真实外呼）；回环目的地放行。"""

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
