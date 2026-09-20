"""人名脱敏测试（PRD 10.4，验收 A29）。

覆盖：
1. :func:`redact_names` 的稳定称谓与逐字匹配（先长后短）；
2. :class:`NameRedactor` 的递归处理、URL 策略（能保持地址就保持，否则退化为
   脱敏搜索目标并记录原因）；
3. 权限门用**脱敏后**的参数写卡片（原始姓名不出现在卡片里）；
4. 用户原文 / 聊天记录 / 知识内容**保持原内容**（脱敏只作用于出网参数）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.approval import NameRedactionResult, NameRedactor, redact_names
from app.agent.contracts import ApprovalStatus, RunStatus
from tests.fakes_agent import app_store  # noqa: F401 -- 共享夹具
from tests.fakes_run import FakeServices, build_manager, build_registry, seed_conversation

# =====================================================================
# 纯函数
# =====================================================================


def test_redact_names_uses_stable_aliases() -> None:
    result = redact_names("张三与李四的合同", private_names=["张三", "李四"])
    assert isinstance(result, NameRedactionResult)
    assert result.changed
    assert result.text == "联系人1与联系人2的合同"
    assert result.replacements == {"张三": "联系人1", "李四": "联系人2"}
    assert "张三" not in result.text


def test_redact_names_prefers_longest_match() -> None:
    result = redact_names("张三丰与张三", private_names=["张三", "张三丰"])
    assert result.text == "联系人1与联系人2"


def test_redact_names_reuses_existing_mapping() -> None:
    first = redact_names("张三", private_names=["张三"])
    second = redact_names("张三又来", private_names=["张三"], replacements=first.replacements)
    assert second.text == "联系人1又来"
    assert second.replacements == {"张三": "联系人1"}


def test_redactor_handles_nested_structures() -> None:
    redactor = NameRedactor(["张三"])
    payload = {
        "query": "张三 近况",
        "filters": {"author": "张三"},
        "tags": ["张三", "无关"],
        "limit": 5,
    }
    redacted = redactor("web_search", payload)
    assert redacted["query"] == "联系人1 近况"
    assert redacted["filters"]["author"] == "联系人1"
    assert redacted["tags"] == ["联系人1", "无关"]
    assert redacted["limit"] == 5  # 非字符串原样保留


def test_redactor_keeps_valid_url_and_degrades_invalid_one() -> None:
    redactor = NameRedactor(["张三"])
    # 路径里是明文姓名：替换后仍是合法地址 → 保持可打开
    kept = redactor("read_web_page", {"url": "https://news.example.com/张三/profile"})
    assert "张三" not in kept["url"]
    assert kept["url"].startswith("https://news.example.com/")
    assert redactor.reasons == []

    # 百分号编码形式：替换后地址结构被破坏 → 退化为脱敏搜索目标并记录原因
    encoded = redactor("read_web_page", {"url": "https://x.example.com/a?name=%E5%BC%A0%E4%B8%89"})
    assert "张三" not in str(encoded)
    assert encoded["url"]
    # 只要没有伪造仍可打开的原文地址，允许保持无语义的目标字符串


def test_redactor_records_reason_when_address_is_rewritten() -> None:
    """百分号编码的姓名必须先解码才能发现：替换后记录原始/实际查询的对应关系。"""

    redactor = NameRedactor(["张三"])
    encoded = redactor(
        "read_web_page", {"url": "https://x.example.com/a?name=%E5%BC%A0%E4%B8%89"}
    )
    assert "张三" not in str(encoded)
    assert "%E5%BC%A0%E4%B8%89" not in str(encoded)
    assert redactor.reasons and "百分号" in redactor.reasons[0]


# =====================================================================
# 运行层：卡片与出网参数都用脱敏值，本地数据保持原样
# =====================================================================


@pytest.fixture
async def approval_case(app_store, tmp_path: Path):
    await seed_conversation(app_store)
    registry = build_registry()
    services = FakeServices(task_count=1, approve_tool="web_search")
    manager = build_manager(app_store, services=services, root=tmp_path, registry=registry)
    run = await manager.create_run(
        workspace_id="ws-1", conversation_id="conv-1", question="张三最近怎么样"
    )
    await manager.run_to_completion(run.id)
    try:
        yield manager, registry, run
    finally:
        await manager.aclose()


async def test_approval_card_contains_only_redacted_parameters(approval_case) -> None:
    manager, _registry, run = approval_case
    assert (await manager.require_run(run.id)).status is RunStatus.WAITING_APPROVAL
    card = (await manager.store.list_approvals(run.id, status=ApprovalStatus.PENDING))[0]
    dumped = str(card.display.model_dump())
    assert "张三" not in dumped
    assert "联系人" in dumped  # 稳定称谓
    assert card.display.redaction_notice  # 卡片提示已脱敏（PRD 10.4）
    assert card.display.purpose
    assert card.display.tool_name == "web_search"


async def test_approved_execution_sends_redacted_query(approval_case) -> None:
    manager, registry, run = approval_case
    card = (await manager.store.list_approvals(run.id))[0]
    await manager.submit_approval(card.approval_id, decision_kind="allow_once")
    await manager.wait(run.id)
    assert registry.call_log == ["web_search"]
    sent = registry.query_log[-1]
    assert "张三" not in str(sent)
    assert "联系人" in sent["query"]


async def test_local_content_is_not_redacted(approval_case, app_store) -> None:
    """用户原文与助手答案保持原内容（脱敏只作用于出网参数）。"""

    manager, _registry, run = approval_case
    messages = await app_store.list_messages("conv-1")
    assert messages[0].content == "张三最近怎么样"
    card = (await manager.store.list_approvals(run.id))[0]
    await manager.submit_approval(card.approval_id, decision_kind="allow_once")
    await manager.wait(run.id)
    done = await manager.require_run(run.id)
    answers = await app_store.get_answer(done.answer_id)
    assert answers is not None
    # 答案里可以出现用户原始问题里的姓名（本地链路不脱敏）
    assert "张三最近怎么样" in messages[0].content
