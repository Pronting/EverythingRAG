"""会话 API 测试：列表 / 新建 / 读取 / 追加 / 标题（AI 生成用 fake 模型，零出网）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import conversations as conversations_mod
from app.generation.base import ChatChunk
from app.main import app


def _client() -> TestClient:
    return TestClient(app)


class FakeTitleModel:
    """对话模型替身：stream 产出「测试标题」token。"""

    @property
    def supports_image_input(self) -> bool:
        return False

    async def stream_chat(self, messages: list, **kwargs: object):  # type: ignore[no-untyped-def]
        for token in ("测试", "标题"):
            yield ChatChunk("content", token)


def test_list_empty_and_create() -> None:
    client = _client()
    assert client.get("/api/conversations").json() == []

    resp = client.post("/api/conversations")
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"]
    assert data["title"] == "新对话"
    # 空会话（未交换过消息）不进侧边栏列表，避免堆满「新对话」
    assert client.get("/api/conversations").json() == []
    assert client.get(f"/api/conversations/{data['id']}").status_code == 200


def test_get_unknown_404() -> None:
    assert _client().get("/api/conversations/nope").status_code == 404


def test_append_messages_and_read_back() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]

    resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["message_count"] == 2

    conv = client.get(f"/api/conversations/{conv_id}").json()
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["content"] == "你好"


def test_reasoning_is_persisted_alongside_answer_metadata() -> None:
    """思维链随消息一起落盘（前端默认折叠展示），回答元数据不受影响。"""

    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "content": "最终答案",
                    "reasoning": "先确认检索命中，再得出结论",
                    "answer_basis": "knowledge",
                    "policy_version": "rag-policy-v3",
                }
            ]
        },
    )
    assert resp.status_code == 200
    stored = client.get(f"/api/conversations/{conv_id}").json()["messages"][0]
    assert stored["reasoning"] == "先确认检索命中，再得出结论"
    assert stored["answer_basis"] == "knowledge"
    assert stored["policy_version"] == "rag-policy-v3"


def test_messages_without_reasoning_default_to_null() -> None:
    """未带思维链的消息 reasoning 缺省为 null（前端据此不渲染折叠块）。"""

    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "assistant", "content": "普通回答"}]},
    )
    stored = client.get(f"/api/conversations/{conv_id}").json()["messages"][0]
    assert stored["reasoning"] is None


def test_set_title_manual() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    resp = client.put(f"/api/conversations/{conv_id}/title", json={"title": "手动标题"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "手动标题"


def test_generate_title_uses_chat_model() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "如何配置对话模型"}]},
    )

    # 用 fake 对话模型替换（避免真实出网）
    conversations_mod.create_chat_model_from_config = lambda _chat: FakeTitleModel()
    try:
        resp = client.post(f"/api/conversations/{conv_id}/generate-title")
        assert resp.status_code == 200
        assert resp.json()["title"] == "测试标题"
        assert client.get(f"/api/conversations/{conv_id}").json()["title"] == "测试标题"
    finally:
        # 还原（其他用例需要真实实现）
        from app.generation import providers

        conversations_mod.create_chat_model_from_config = providers.create_chat_model_from_config


def test_generate_title_empty_conv_fallback() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    resp = client.post(f"/api/conversations/{conv_id}/generate-title")
    assert resp.status_code == 200
    assert resp.json()["title"] == "新对话"


def test_delete_conversation() -> None:
    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    assert client.delete(f"/api/conversations/{conv_id}").status_code == 204
    assert client.get(f"/api/conversations/{conv_id}").status_code == 404
    assert client.delete(f"/api/conversations/{conv_id}").status_code == 404


def test_search_matches_message_content_and_returns_excerpt():
    client = _client()
    conv = client.post("/api/conversations").json()
    client.put(f"/api/conversations/{conv['id']}/title", json={"title": "项目讨论"})
    client.post(f"/api/conversations/{conv['id']}/messages", json={"messages": [
        {"role": "user", "content": "需要优化上传速度"},
        {"role": "assistant", "content": "使用 ThreadPool 并行导入可以改善速度"},
    ]})
    for query in ["项目", "上传", "threadpool", "项目 并行"]:
        results = client.get("/api/conversations", params={"q": query}).json()
        assert len(results) == 1
        assert results[0]["id"] == conv["id"]
        assert "messages" not in results[0]
    results = client.get("/api/conversations", params={"q": "threadpool"}).json()
    assert "ThreadPool" in results[0]["match_excerpt"]
    assert client.get("/api/conversations", params={"q": "不存在的内容"}).json() == []
    assert len(client.get("/api/conversations", params={"q": "  "}).json()) == 1


def test_truncate_from_index_drops_that_message_and_after() -> None:
    """编辑重发：从被编辑的那条起截断，后面的轮次作废（不重复、不错位）。"""

    client = _client()
    conv_id = client.post("/api/conversations").json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "问题二"},
            {"role": "assistant", "content": "回答二"},
        ]},
    )

    resp = client.post(f"/api/conversations/{conv_id}/truncate", json={"index": 2})
    assert resp.status_code == 200
    assert resp.json() == {"message_count": 2}

    contents = [m["content"] for m in client.get(f"/api/conversations/{conv_id}").json()["messages"]]
    assert contents == ["问题一", "回答一"]

    # 重发编辑后的内容：直接追加，历史里不会残留旧版本
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [
            {"role": "user", "content": "问题二（改写）"},
            {"role": "assistant", "content": "回答二新"},
        ]},
    )
    contents = [m["content"] for m in client.get(f"/api/conversations/{conv_id}").json()["messages"]]
    assert contents == ["问题一", "回答一", "问题二（改写）", "回答二新"]


def test_truncate_first_message_clears_history() -> None:
    client = _client()
    conv_id = _conversation_with_messages(client, "清空测试")
    assert client.post(f"/api/conversations/{conv_id}/truncate", json={"index": 0}).json() == {"message_count": 0}
    assert client.get(f"/api/conversations/{conv_id}").json()["messages"] == []


def test_truncate_beyond_length_is_noop_and_unknown_404() -> None:
    client = _client()
    conv_id = _conversation_with_messages(client, "越界测试")
    assert client.post(f"/api/conversations/{conv_id}/truncate", json={"index": 99}).json() == {"message_count": 2}
    assert len(client.get(f"/api/conversations/{conv_id}").json()["messages"]) == 2
    assert client.post("/api/conversations/nope/truncate", json={"index": 0}).status_code == 404
    assert client.post(f"/api/conversations/{conv_id}/truncate", json={"index": -1}).status_code == 422


def _conversation_with_messages(client: TestClient, title: str) -> str:
    """建一个「已交换过消息」的会话并返回 id（空会话不进侧边栏列表）。"""

    conv_id = client.post("/api/conversations").json()["id"]
    client.put(f"/api/conversations/{conv_id}/title", json={"title": title})
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [
            {"role": "user", "content": f"{title} 的问题"},
            {"role": "assistant", "content": f"{title} 的回答", "answer_basis": "knowledge"},
        ]},
    )
    return conv_id


def test_archive_hides_from_sidebar_and_restore_brings_it_back() -> None:
    client = _client()
    conv_id = _conversation_with_messages(client, "归档测试")
    assert [item["id"] for item in client.get("/api/conversations").json()] == [conv_id]

    resp = client.post(f"/api/conversations/{conv_id}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived"] is True
    assert resp.json()["archived_at"] is not None
    # 归档后：侧边栏列表为空，但归档列表可查，搜索也搜不到
    assert client.get("/api/conversations").json() == []
    assert client.get("/api/conversations", params={"q": "归档测试"}).json() == []
    archived = client.get("/api/conversations/archived").json()
    assert [item["id"] for item in archived] == [conv_id]
    assert archived[0]["message_count"] == 2

    restored = client.post(f"/api/conversations/{conv_id}/restore")
    assert restored.json()["archived"] is False
    assert restored.json()["archived_at"] is None
    assert client.get("/api/conversations/archived").json() == []
    assert [item["id"] for item in client.get("/api/conversations").json()] == [conv_id]


def test_archive_and_restore_unknown_404() -> None:
    client = _client()
    assert client.post("/api/conversations/nope/archive").status_code == 404
    assert client.post("/api/conversations/nope/restore").status_code == 404


def test_fork_copies_messages_and_links_to_parent() -> None:
    client = _client()
    parent_id = _conversation_with_messages(client, "原对话")

    resp = client.post(f"/api/conversations/{parent_id}/fork")
    assert resp.status_code == 201
    branch = resp.json()
    assert branch["id"] != parent_id
    assert branch["title"] == "原对话 · 分支"
    assert branch["parent_id"] == parent_id
    assert branch["forked_from_title"] == "原对话"
    assert branch["message_count"] == 2

    # 分支继承全部消息（含来源与依据），是可继续追问的独立会话
    parent = client.get(f"/api/conversations/{parent_id}").json()
    forked = client.get(f"/api/conversations/{branch['id']}").json()
    assert [m["content"] for m in forked["messages"]] == [m["content"] for m in parent["messages"]]

    # 侧边栏同时列出父与分支（分支带 parent_id 供嵌套展示）
    listed = {item["id"]: item for item in client.get("/api/conversations").json()}
    assert set(listed) == {parent_id, branch["id"]}
    assert listed[parent_id]["parent_id"] is None
    assert listed[branch["id"]]["parent_id"] == parent_id

    # 分支是独立副本：父对话继续追加不影响分支
    client.post(f"/api/conversations/{parent_id}/messages", json={"messages": [
        {"role": "user", "content": "父对话追加"},
    ]})
    assert len(client.get(f"/api/conversations/{branch['id']}").json()["messages"]) == 2


def test_fork_unknown_404() -> None:
    assert _client().post("/api/conversations/nope/fork").status_code == 404


def test_archived_parent_hides_but_branch_keeps_its_link() -> None:
    """归档父对话：父从列表消失，分支仍保留 parent_id（前端提升为顶层展示）。"""

    client = _client()
    parent_id = _conversation_with_messages(client, "父对话")
    branch_id = client.post(f"/api/conversations/{parent_id}/fork").json()["id"]

    client.post(f"/api/conversations/{parent_id}/archive")
    listed = client.get("/api/conversations").json()
    assert [item["id"] for item in listed] == [branch_id]
    assert listed[0]["parent_id"] == parent_id


def test_delete_parent_promotes_branch_to_top_level() -> None:
    client = _client()
    parent_id = _conversation_with_messages(client, "父对话")
    branch_id = client.post(f"/api/conversations/{parent_id}/fork").json()["id"]

    assert client.delete(f"/api/conversations/{parent_id}").status_code == 204
    # 分支不被级联删除，且不再悬挂在不存在的父上
    branch = client.get(f"/api/conversations/{branch_id}").json()
    assert branch["parent_id"] is None
    assert branch["messages"] != []


# ------------------------------------------------------------------ 工作区（对话分区）


def test_workspaces_list_starts_with_default() -> None:
    client = _client()
    workspaces = client.get("/api/workspaces").json()
    assert workspaces[0]["id"] == "default"
    assert workspaces[0]["is_default"] is True
    assert workspaces[0]["conversation_count"] == 0


def test_create_rename_and_delete_workspace() -> None:
    client = _client()
    created = client.post("/api/workspaces", json={"name": "工作笔记"})
    assert created.status_code == 201
    workspace_id = created.json()["id"]
    assert created.json()["name"] == "工作笔记"
    assert created.json()["is_default"] is False

    renamed = client.put(f"/api/workspaces/{workspace_id}", json={"name": "长期项目"})
    assert renamed.json()["name"] == "长期项目"

    assert client.delete(f"/api/workspaces/{workspace_id}").status_code == 200
    assert [item["id"] for item in client.get("/api/workspaces").json()] == ["default"]


def test_default_workspace_cannot_be_deleted() -> None:
    client = _client()
    resp = client.delete("/api/workspaces/default")
    assert resp.status_code == 400
    assert "默认" in resp.json()["detail"]
    assert client.delete("/api/workspaces/nope").status_code == 404


def test_new_conversation_lands_in_requested_workspace() -> None:
    client = _client()
    workspace_id = client.post("/api/workspaces", json={"name": "研究"}).json()["id"]

    conv_id = client.post("/api/conversations", params={"workspace": workspace_id}).json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "分区里的问题"}]},
    )
    assert client.get(f"/api/conversations/{conv_id}").json()["workspace_id"] == workspace_id

    # 侧边栏按分区过滤：默认分区看不到它（不带 workspace 则不带过滤，返回全部）
    assert client.get("/api/conversations", params={"workspace": "default"}).json() == []
    scoped = client.get("/api/conversations", params={"workspace": workspace_id}).json()
    assert [item["id"] for item in scoped] == [conv_id]
    assert [item["id"] for item in client.get("/api/conversations").json()] == [conv_id]
    # 工作区列表带出会话数
    counts = {item["id"]: item["conversation_count"] for item in client.get("/api/workspaces").json()}
    assert counts == {"default": 0, workspace_id: 1}


def test_search_is_scoped_by_workspace() -> None:
    client = _client()
    workspace_id = client.post("/api/workspaces", json={"name": "研究"}).json()["id"]
    default_conv = _conversation_with_messages(client, "默认区对话")
    scoped_conv_id = client.post("/api/conversations", params={"workspace": workspace_id}).json()["id"]
    client.post(
        f"/api/conversations/{scoped_conv_id}/messages",
        json={"messages": [{"role": "user", "content": "共同关键词 只在研究区"}]},
    )

    default_hits = client.get("/api/conversations", params={"q": "共同关键词", "workspace": "default"}).json()
    assert [item["id"] for item in default_hits] == []
    scoped = client.get("/api/conversations", params={"q": "共同关键词", "workspace": workspace_id}).json()
    assert [item["id"] for item in scoped] == [scoped_conv_id]
    assert default_conv not in [item["id"] for item in scoped]
    # 不带 workspace：跨分区搜索
    assert len(client.get("/api/conversations", params={"q": "共同关键词"}).json()) == 1


def test_move_conversation_between_workspaces() -> None:
    client = _client()
    workspace_id = client.post("/api/workspaces", json={"name": "研究"}).json()["id"]
    conv_id = _conversation_with_messages(client, "待归类")
    default_scope = {"workspace": "default"}

    assert client.post(f"/api/conversations/{conv_id}/workspace", json={"workspace_id": workspace_id}).status_code == 200
    assert client.get("/api/conversations", params=default_scope).json() == []
    assert [item["id"] for item in client.get("/api/conversations", params={"workspace": workspace_id}).json()] == [conv_id]
    # 移回默认分区
    client.post(f"/api/conversations/{conv_id}/workspace", json={"workspace_id": "default"})
    assert [item["id"] for item in client.get("/api/conversations", params=default_scope).json()] == [conv_id]


def test_move_conversation_unknown_workspace_404() -> None:
    client = _client()
    conv_id = _conversation_with_messages(client, "待归类")
    assert client.post(f"/api/conversations/{conv_id}/workspace", json={"workspace_id": "nope"}).status_code == 404
    assert client.post("/api/conversations/nope/workspace", json={"workspace_id": "default"}).status_code == 404


def test_fork_stays_in_same_workspace() -> None:
    client = _client()
    workspace_id = client.post("/api/workspaces", json={"name": "研究"}).json()["id"]
    conv_id = client.post("/api/conversations", params={"workspace": workspace_id}).json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "分区里的问题"}]},
    )

    branch = client.post(f"/api/conversations/{conv_id}/fork").json()
    assert branch["workspace_id"] == workspace_id


def test_delete_workspace_archives_its_conversations() -> None:
    """删分区不删对话：其中对话全部归档，可在归档列表找回。"""

    client = _client()
    workspace_id = client.post("/api/workspaces", json={"name": "临时"}).json()["id"]
    conv_id = client.post("/api/conversations", params={"workspace": workspace_id}).json()["id"]
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"messages": [{"role": "user", "content": "分区里的问题"}]},
    )

    resp = client.delete(f"/api/workspaces/{workspace_id}")
    assert resp.status_code == 200
    assert resp.json() == {"archived": 1}
    assert client.get("/api/conversations", params={"workspace": workspace_id}).json() == []
    archived = client.get("/api/conversations/archived").json()
    assert [item["id"] for item in archived] == [conv_id]
    # 归档项仍记得原分区，恢复后回到默认分区列表
    assert archived[0]["workspace_id"] == workspace_id
    assert client.post(f"/api/conversations/{conv_id}/restore").status_code == 200


def test_legacy_conversations_without_workspace_default() -> None:
    """旧 conversations.json（无 workspace_id）读回后落到默认分区。"""

    client = _client()
    assert client.get("/api/conversations").json() == []
    conv = client.post("/api/conversations").json()
    assert conv["workspace_id"] == "default"
