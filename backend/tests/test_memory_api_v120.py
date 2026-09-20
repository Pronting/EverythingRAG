"""工作区记忆与历史检索 API（PRD 13.2）。

覆盖六个端点的 HTTP 契约：状态码、负载形状、乐观并发 409、缺失工作区/条目 404、
关闭记忆后写入 409、历史检索的查询/日期/分页与 400/503 错误面。

conftest 已把 ``settings.data_dir`` 隔离到 tmp，因此这里既不触碰真实
``~/.everything-rag``，也不会读写用户的 ``memory.json``。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.app_store import SCHEMA_STATEMENTS
from app.core.config import settings
from app.core.workspace_context import _workspace_paths
from app.main import app
from app.memory.history import history_tokenizer, tokenize_for_index

TERMS = ("固定快照", "知识库管理系统")


def _client() -> TestClient:
    return TestClient(app)


def _memory_path() -> Path:
    return _workspace_paths(Path(settings.data_dir), "default").memory_json


def _raw() -> dict:
    return json.loads(_memory_path().read_text(encoding="utf-8"))


def _seed_history_db() -> None:
    """在隔离的 ``data_dir`` 里造一份真实的 ``app.db``（消息 + 分词索引）。

    这里用同步 ``sqlite3`` 直接落地：索引写入侧本身已由 ``test_history_search.py``
    端到端覆盖（经 ``AppStore``），本文件只验证 HTTP 层。
    """

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "app.db"
    if db.exists():
        db.unlink()
    paths = _workspace_paths(data_dir, "default")
    paths.jieba_cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = history_tokenizer(paths, terms=TERMS)

    connection = sqlite3.connect(str(db))
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        "INSERT INTO workspaces (id, storage_key, name, status, default_permission_mode,"
        " default_reasoning_level, knowledge_revision, created_at, updated_at)"
        " VALUES ('default','default','默认','active','ask','high',0,'t','t')"
    )
    connection.execute(
        "INSERT INTO conversations (id, workspace_id, title, status, permission_mode,"
        " reasoning_level, context_epoch, memory_revision, memory_invalidation_revision,"
        " memory_snapshot_text, memory_snapshot_hash, memory_snapshot_at, created_at, updated_at)"
        " VALUES ('conv-1','default','主会话','active','ask','high',1,0,0,'','','','t','t')"
    )
    rows = [
        ("msg-1", "你回答一律用中文", "user", "2026-09-01T00:00:00Z"),
        ("msg-2", "我们讨论过 固定快照 的失效机制", "user", "2026-09-02T00:00:00Z"),
        ("msg-3", "天气不错", "user", "2026-09-03T00:00:00Z"),
    ]
    for index, (message_id, content, role, created_at) in enumerate(rows, start=1):
        connection.execute(
            "INSERT INTO messages (id, conversation_id, sequence, role, content,"
            " source_snapshot, created_at) VALUES (?,?,?,?,?,'{}',?)",
            (message_id, "conv-1", index, role, content, created_at),
        )
        connection.execute(
            "INSERT INTO history_fts (message_id, conversation_id, workspace_id, tokens)"
            " VALUES (?,?,?,?)",
            (message_id, "conv-1", "default", tokenize_for_index(content, tokenizer)),
        )
    connection.commit()
    connection.close()


# ---------------------------------------------------------------- GET


def test_get_memory_returns_entries_revisions_and_switch() -> None:
    client = _client()
    created = client.post(
        "/api/workspaces/default/memory",
        json={"text": "回答一律用中文", "kind": "response_preference"},
    )
    assert created.status_code == 201

    response = client.get("/api/workspaces/default/memory")
    assert response.status_code == 200
    body = response.json()

    assert body["workspace_id"] == "default"
    assert body["schema_version"] == 1
    assert body["enabled"] is True
    assert body["revision"] >= 1
    assert body["invalidation_revision"] == 0
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["text"] == "回答一律用中文"
    assert entry["kind"] == "response_preference"
    assert entry["user_pinned"] is True
    assert entry["status"] == "active"
    assert body["snapshot_preview"]["chars"] > 0
    assert body["blocked_source_conversations"] == []


def test_responses_never_expose_absolute_paths() -> None:
    """与 ``test_overview_api_never_exposes_absolute_paths`` 同一约定：不回磁盘路径。"""

    client = _client()
    client.post("/api/workspaces/default/memory", json={"text": "某个主题"})

    for payload in (
        client.get("/api/workspaces/default/memory").text,
        client.post("/api/workspaces/default/memory", json={"text": "另一个主题"}).text,
    ):
        assert "memory.json" not in payload
        assert "app.db" not in payload
        assert str(Path(settings.data_dir)) not in payload
        assert "C:\\" not in payload and "\\\\" not in payload


# ---------------------------------------------------------------- POST


def test_post_memory_saves_pinned_entry_and_infers_kind() -> None:
    client = _client()

    requirement = client.post(
        "/api/workspaces/default/memory", json={"text": "以后回答请用中文，并先给结论"}
    )
    assert requirement.status_code == 201
    assert requirement.json()["entry"]["kind"] == "response_preference"
    assert requirement.json()["entry"]["user_pinned"] is True

    topic = client.post("/api/workspaces/default/memory", json={"text": "在调研 FTS5 方案"})
    assert topic.status_code == 201
    assert topic.json()["entry"]["kind"] == "topic"

    # 主动保存不递增 invalidation_revision（新增不是失效事件，PRD 11.3）。
    assert _raw()["invalidation_revision"] == 0
    assert _raw()["revision"] == 2


def test_post_memory_rejects_empty_text_and_unknown_kind() -> None:
    client = _client()
    assert client.post("/api/workspaces/default/memory", json={"text": "   "}).status_code == 422
    assert (
        client.post(
            "/api/workspaces/default/memory", json={"text": "x", "kind": "not-a-kind"}
        ).status_code
        == 422
    )


# ---------------------------------------------------------------- PATCH 开关


def test_patch_switch_toggles_memory_and_blocks_writes_while_disabled() -> None:
    client = _client()
    client.post("/api/workspaces/default/memory", json={"text": "回答一律用中文"})

    disabled = client.patch("/api/workspaces/default/memory", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert disabled.json()["invalidation_revision"] == 1

    blocked = client.post("/api/workspaces/default/memory", json={"text": "关闭后写入"})
    assert blocked.status_code == 409

    enabled = client.patch("/api/workspaces/default/memory", json={"enabled": True})
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True
    assert client.post("/api/workspaces/default/memory", json={"text": "重开后可写"}).status_code == 201


def test_patch_switch_requires_boolean_enabled() -> None:
    client = _client()
    assert client.patch("/api/workspaces/default/memory", json={}).status_code == 422
    assert (
        client.patch("/api/workspaces/default/memory", json={"enabled": "yes"}).status_code == 422
    )


def test_user_can_still_forget_and_correct_while_memory_is_off() -> None:
    """关闭记忆后仍可查看 / 纠正 / 遗忘（PRD 18.1「用户可以查看、修改和关闭」）。"""

    client = _client()
    requirement_id = client.post(
        "/api/workspaces/default/memory", json={"text": "回答一律用英文"}
    ).json()["entry"]["id"]
    conclusion_id = client.post(
        "/api/workspaces/default/memory", json={"text": "旧结论"}
    ).json()["entry"]["id"]
    assert client.patch("/api/workspaces/default/memory", json={"enabled": False}).status_code == 200

    assert client.get("/api/workspaces/default/memory").json()["enabled"] is False
    assert client.post("/api/workspaces/default/memory", json={"text": "新增"}).status_code == 409

    corrected = client.patch(
        f"/api/workspaces/default/memory/{conclusion_id}", json={"text": "修正后的结论"}
    )
    assert corrected.status_code == 200
    assert corrected.json()["entry"]["text"] == "修正后的结论"

    forgotten = client.delete(f"/api/workspaces/default/memory/{requirement_id}")
    assert forgotten.status_code == 200
    assert forgotten.json()["entry"]["status"] == "forgotten"


# ---------------------------------------------------------------- PATCH 条目


def test_patch_entry_corrects_text_and_records_invalidation() -> None:
    client = _client()
    entry_id = client.post(
        "/api/workspaces/default/memory", json={"text": "回答一律用中文"}
    ).json()["entry"]["id"]

    corrected = client.patch(
        f"/api/workspaces/default/memory/{entry_id}",
        json={"text": "回答一律用中文，并且先给结论"},
    )
    assert corrected.status_code == 200
    body = corrected.json()
    assert body["entry"]["text"] == "回答一律用中文，并且先给结论"
    assert body["invalidation_revision"] == 1
    assert body["applied"] is True

    invalidations = _raw()["invalidations"]
    assert len(invalidations) == 1
    assert invalidations[0]["kind"] == "corrected"
    assert invalidations[0]["quote"] == "回答一律用中文"
    assert invalidations[0]["replacement"] == "回答一律用中文，并且先给结论"


def test_patch_unknown_entry_returns_404_and_bad_kind_422() -> None:
    client = _client()
    assert (
        client.patch(
            "/api/workspaces/default/memory/mem-nope", json={"text": "x"}
        ).status_code
        == 404
    )
    entry_id = client.post("/api/workspaces/default/memory", json={"text": "主题"}).json()["entry"]["id"]
    assert (
        client.patch(
            f"/api/workspaces/default/memory/{entry_id}", json={"kind": "bogus"}
        ).status_code
        == 422
    )


# ---------------------------------------------------------------- DELETE


def test_delete_entry_forgets_softly_and_keeps_the_record() -> None:
    client = _client()
    entry_id = client.post(
        "/api/workspaces/default/memory", json={"text": "回答一律用英文"}
    ).json()["entry"]["id"]

    removed = client.delete(f"/api/workspaces/default/memory/{entry_id}")
    assert removed.status_code == 200
    assert removed.json()["entry"]["status"] == "forgotten"
    assert removed.json()["invalidation_revision"] == 1

    entries = client.get("/api/workspaces/default/memory").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["status"] == "forgotten"
    assert _raw()["invalidations"][0]["kind"] == "forgotten"


def test_delete_unknown_entry_returns_404() -> None:
    client = _client()
    assert client.delete("/api/workspaces/default/memory/mem-nope").status_code == 404


# ---------------------------------------------------------------- 乐观并发


def test_stale_expected_revision_returns_409() -> None:
    client = _client()
    entry_id = client.post(
        "/api/workspaces/default/memory", json={"text": "第一条", "expected_revision": 0}
    ).json()["entry"]["id"]

    failure = client.patch(
        f"/api/workspaces/default/memory/{entry_id}",
        json={"text": "并发纠正", "expected_revision": 0},
    )
    assert failure.status_code == 409
    assert client.delete(
        f"/api/workspaces/default/memory/{entry_id}", params={"expected_revision": 0}
    ).status_code == 409
    assert client.patch(
        "/api/workspaces/default/memory", json={"enabled": False, "expected_revision": 0}
    ).status_code == 409


# ---------------------------------------------------------------- 归属


def test_unknown_workspace_returns_404_on_every_endpoint() -> None:
    client = _client()
    assert client.get("/api/workspaces/nope/memory").status_code == 404
    assert client.post("/api/workspaces/nope/memory", json={"text": "x"}).status_code == 404
    assert client.patch("/api/workspaces/nope/memory", json={"enabled": True}).status_code == 404
    assert client.patch("/api/workspaces/nope/memory/mem-1", json={"text": "x"}).status_code == 404
    assert client.delete("/api/workspaces/nope/memory/mem-1").status_code == 404
    assert client.get("/api/workspaces/nope/history", params={"q": "中文"}).status_code == 404


# ---------------------------------------------------------------- 历史检索


def test_history_endpoint_returns_original_messages() -> None:
    _seed_history_db()
    client = _client()

    response = client.get("/api/workspaces/default/history", params={"q": "固定快照"})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "固定快照"
    assert len(body["results"]) == 1
    hit = body["results"][0]
    assert hit["message_id"] == "msg-2"
    assert hit["content"] == "我们讨论过 固定快照 的失效机制"
    assert "固定快照" in hit["snippet"]
    assert body["workspace_id"] == "default"
    # 诊断字段不出现在响应里（不回磁盘路径）。
    assert "db" not in body


def test_history_endpoint_supports_dates_and_pagination() -> None:
    _seed_history_db()
    client = _client()

    filtered = client.get(
        "/api/workspaces/default/history",
        params={"q": "中文", "date_from": "1999-01-01", "date_to": "2999-01-01"},
    )
    assert filtered.status_code == 200
    assert len(filtered.json()["results"]) == 1

    assert (
        client.get(
            "/api/workspaces/default/history",
            params={"q": "中文", "date_from": "2999-01-01"},
        ).json()["results"]
        == []
    )

    paged = client.get(
        "/api/workspaces/default/history", params={"q": "固定快照", "limit": 1, "offset": 1}
    )
    assert paged.status_code == 200
    assert paged.json()["results"] == []
    assert paged.json()["limit"] == 1 and paged.json()["offset"] == 1


def test_history_endpoint_error_faces() -> None:
    client = _client()
    # 空查询：可检索词项为空 → 400（不是 500）。
    assert client.get("/api/workspaces/default/history", params={"q": "，。"}).status_code == 400
    # 隔离目录里还没有 app.db → 503（明确不可用，而不是假装没有结果）。
    response = client.get("/api/workspaces/default/history", params={"q": "中文"})
    assert response.status_code == 503

    _seed_history_db()
    assert client.get("/api/workspaces/default/history", params={"q": "中文"}).status_code == 200


def test_history_limit_is_validated() -> None:
    _seed_history_db()
    client = _client()
    assert (
        client.get("/api/workspaces/default/history", params={"q": "中文", "limit": 0}).status_code
        == 422
    )
    assert (
        client.get(
            "/api/workspaces/default/history", params={"q": "中文", "limit": 999}
        ).status_code
        == 422
    )


def test_endpoints_are_registered() -> None:
    """六个端点确实挂在 app 上（而不只是在 memory.py 里定义了 router）。"""

    paths = TestClient(app).get("/openapi.json").json()["paths"]
    assert set(paths["/api/workspaces/{workspace_id}/memory"]) >= {"get", "patch", "post"}
    assert set(paths["/api/workspaces/{workspace_id}/memory/{memory_id}"]) >= {"delete", "patch"}
    assert "get" in paths["/api/workspaces/{workspace_id}/history"]


def test_workspace_memory_path_is_isolated_to_tmp() -> None:
    """本文件写入的 memory.json 必须在 conftest 的 tmp 数据目录里。"""

    client = _client()
    client.post("/api/workspaces/default/memory", json={"text": "隔离检查"})

    assert _memory_path().is_file()
    assert "pytest-of-" in str(settings.data_dir) or "tmp" in str(settings.data_dir).lower()
