"""T09 单测：来源序列化与历史兼容（PRD §8.3）。

覆盖：
* 「已筛选资料」的来源字段（relevance_checked / rerank_score / score_type / profile_id /
  evidence_hash / member_refs）随消息保存、重载、移动工作区与分支后保持不变；
* ``similarity`` 始终是原始向量相似度，重排分数只进独立字段，二者不互相覆盖；
* 旧记录（缺少新字段、或没有来源）正常读取，不被补造字段、不报错。
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.generation.chat_service import (
    _relevance_source_fields,
    _source_chunk,
)
from app.main import app
from app.retrieval.evidence_view import RetrievalScope, build_chunk_views
from app.retrieval.vector_retriever import RetrievedChunk

SCOPE = RetrievalScope(scope_id="global", index_revision="chunks__fake__v7")
RERANK_SCORE = 0.812
VECTOR_SIMILARITY = 0.42


def _client() -> TestClient:
    return TestClient(app)


def make_source(**overrides: Any) -> dict[str, Any]:
    """构造一条带相关性字段的来源（字段与 ``_meta_frame`` 的投影一致）。"""

    source: dict[str, Any] = {
        "source_id": "S1",
        "block_id": "blk-1",
        "text": "增量同步只重新处理发生变化的文档。",
        "source_file": "笔记.md",
        "similarity": VECTOR_SIMILARITY,
        "match_type": "rerank",
        "heading_path": "笔记.md > 同步",
        "anchor": None,
        "chunk_type": "text",
        "platform": "local",
        "source_type": "knowledge",
        "image_url": None,
        "relevance_checked": True,
        "rerank_score": RERANK_SCORE,
        "score_type": "rerank",
        "profile_id": "prof-validated",
        "evidence_hash": "a" * 64,
        "member_refs": [
            {
                "block_id": "blk-1",
                "content_hash": "b" * 64,
                "relation": "primary",
                "start": 0,
                "end": 20,
                "block_char_offset": 0,
            }
        ],
    }
    source.update(overrides)
    return source


def _new_conversation(client: TestClient) -> str:
    created = client.post("/api/conversations", json={})
    assert created.status_code == 201, created.text
    return created.json()["id"]


def _append(client: TestClient, conversation_id: str, sources: list[dict] | None) -> None:
    response = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "messages": [
                {"role": "user", "content": "问题"},
                {
                    "role": "assistant",
                    "content": "回答",
                    "sources": sources,
                    "answer_basis": "knowledge",
                    "policy_version": "rag-policy-v3",
                },
            ]
        },
    )
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------------------------
# 来源字段构造（与 ChatService 投影一致）
# --------------------------------------------------------------------------------------


def test_source_projection_keeps_vector_similarity_and_separate_rerank_score() -> None:
    chunk = RetrievedChunk(
        block_id="blk-1",
        text="原始整块正文",
        similarity=VECTOR_SIMILARITY,
        source_file="笔记.md",
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path="笔记.md > 同步",
        metadata={"source_file": "笔记.md"},
    )
    view = build_chunk_views(chunk, SCOPE, include_structure_views=False)[0]

    source = _source_chunk(chunk, view, profile_id="prof-validated", rerank_score=RERANK_SCORE)

    assert source.similarity == VECTOR_SIMILARITY  # 向量相似度不改写
    assert source.text == view.evidence_text  # 来源面板与生成上下文同一份文本
    assert source.context_text == view.evidence_text
    assert source.block_id == view.primary_block_id
    metadata = source.metadata
    assert metadata["relevance_checked"] is True
    assert metadata["rerank_score"] == RERANK_SCORE
    assert metadata["score_type"] == "rerank"
    assert metadata["profile_id"] == "prof-validated"
    assert metadata["evidence_hash"] == view.payload_hash
    assert isinstance(metadata["member_refs"], list) and metadata["member_refs"]
    # 图片原始路径不进入来源元数据（隐私红线）
    assert "image_path" not in metadata


def test_legacy_source_without_relevance_metadata_adds_no_fields() -> None:
    chunk = RetrievedChunk(
        block_id="blk-1",
        text="旧路径正文",
        similarity=0.5,
        source_file="笔记.md",
        platform="local",
        chunk_type="text",
        anchor=None,
        heading_path=None,
        metadata={},
    )
    assert _relevance_source_fields(chunk) == {}


# --------------------------------------------------------------------------------------
# 会话保存 / 重载 / 移动 / 分支
# --------------------------------------------------------------------------------------


def test_relevance_fields_survive_save_reload_move_and_fork() -> None:
    client = _client()
    conversation_id = _new_conversation(client)
    _append(client, conversation_id, [make_source()])

    loaded = client.get(f"/api/conversations/{conversation_id}").json()
    source = loaded["messages"][1]["sources"][0]
    assert source["relevance_checked"] is True
    assert source["rerank_score"] == RERANK_SCORE
    assert source["score_type"] == "rerank"
    assert source["profile_id"] == "prof-validated"
    assert source["evidence_hash"] == "a" * 64
    assert source["member_refs"][0]["relation"] == "primary"
    assert source["similarity"] == VECTOR_SIMILARITY
    assert "rerank_score" not in (source.get("similarity"),)

    # 移动到另一个工作区后字段保持不变。
    # V1.2.0（PRD 5.4）：携带历史的会话迁移必须显式 confirm，否则 409 要求二次确认。
    workspace = client.post("/api/workspaces", json={"name": "私密分区"}).json()
    moved = client.post(
        f"/api/conversations/{conversation_id}/workspace",
        json={"workspace_id": workspace["id"], "confirm": True},
    )
    assert moved.status_code == 200, moved.text
    after_move = client.get(f"/api/conversations/{conversation_id}").json()
    assert after_move["workspace_id"] == workspace["id"]
    assert after_move["messages"][1]["sources"] == loaded["messages"][1]["sources"]

    # 分支（fork）复制历史时同样保留新字段
    forked = client.post(f"/api/conversations/{conversation_id}/fork", json={})
    assert forked.status_code == 201, forked.text
    fork_body = client.get(f"/api/conversations/{forked.json()['id']}").json()
    assert fork_body["messages"][1]["sources"][0]["evidence_hash"] == "a" * 64


def test_legacy_sources_without_new_fields_load_unchanged() -> None:
    """旧会话记录（只有基础字段）读取时原样返回，不补造新字段。"""
    client = _client()
    conversation_id = _new_conversation(client)
    legacy = {
        "source_id": "S1",
        "block_id": "old-block",
        "text": "旧的来源文本",
        "source_file": "旧笔记.md",
        "similarity": 0.61,
        "match_type": "dense",
        "heading_path": None,
        "anchor": None,
        "chunk_type": "text",
        "platform": "local",
        "source_type": "knowledge",
    }
    _append(client, conversation_id, [legacy])

    source = client.get(f"/api/conversations/{conversation_id}").json()["messages"][1]["sources"][0]
    assert source == legacy
    for field in (
        "relevance_checked",
        "rerank_score",
        "score_type",
        "profile_id",
        "evidence_hash",
        "member_refs",
    ):
        assert field not in source


def test_message_without_sources_stays_null() -> None:
    client = _client()
    conversation_id = _new_conversation(client)
    _append(client, conversation_id, None)
    messages = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
    assert messages[1]["sources"] is None
