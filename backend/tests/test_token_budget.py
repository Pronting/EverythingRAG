"""本地 token 计数与上下文预算测试（PRD 9.3 / 18.1）。Owner: T03。

零外呼：官方编码路径用 ``sys.modules`` 注入的假 ``deepseek_recipe`` 模块驱动，
tokenizer 资源用 tmp_path 合成文件（真的走 SHA-256 校验），不发任何真实请求。
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app.agent.context import (
    DEFAULT_CONTEXT_COMPACTION_RATIO,
    AgentContext,
    BudgetStatus,
    ContextBudget,
)
from app.agent.contracts import ReasoningLevel
from app.generation.capabilities import resolve_capabilities
from app.generation.context import (
    DEFAULT_PROTOCOL_MARGIN_TOKENS,
    build_context,
    compact_context,
    evaluate_context_budget,
    should_compact,
)
from app.generation.tokenizer import (
    TOKENIZER_DIR_ENV_VAR,
    TOKENIZER_MANIFEST,
    TOKENIZER_PINNED_SHA256,
    TOKENIZER_PINNED_SIZE,
    TOKENIZER_RESOURCE_FILENAME,
    PromptCounter,
    TokenizerManifest,
    TokenizerResourceError,
    count_prompt_tokens,
    count_reasoning_messages,
    official_message_payload,
    official_prompt,
    resolve_pinned_encoder,
    sha256_of_file,
    to_official_messages,
    verify_resource,
)

OFFICIAL = "https://api.deepseek.com"


# ---------------------------------------------------------------- 假编码器


class InMemoryEncoder:
    """按「每个字符 1 个 token」计数的内存编码器（不发网络、结果确定性）。

    用于验证计数口径与聚合逻辑；真实计数走 :func:`resolve_pinned_encoder`。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> list[int]:
        self.calls.append(payload)
        size = len(official_prompt(payload))
        return list(range(size))


def _fake_recipe_module(char_count_per_token: int = 1) -> types.ModuleType:
    """构造假 ``deepseek_recipe`` 模块（模拟官方 Python API 形状）。"""
    module = types.ModuleType("deepseek_recipe")

    class _Tokenizer:
        @staticmethod
        def from_file(path: str) -> str:
            return f"tokenizer:{path}"

    class _Encoding:
        def with_tokenizer(self, tokenizer: Any) -> _Encoding:
            self.tokenizer = tokenizer
            return self

        def encode(self, conversation: Any) -> list[int]:
            return list(range(len(str(conversation)) // char_count_per_token))

    class _Converted:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.conversation = payload

    class _ConversionOptions:
        pass

    class _ChatCompletionRequest:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def convert(self, options: Any) -> _Converted:
            return _Converted(self._payload)

    module.Tokenizer = _Tokenizer
    module.DeepseekV41Encoding = _Encoding
    module.ConversionOptions = _ConversionOptions
    module.ChatCompletionRequest = _ChatCompletionRequest
    return module


# ---------------------------------------------------------------- 1. 官方消息格式


def test_tool_call_reasoning_and_tool_result_are_converted() -> None:
    """完整请求 → 官方消息格式：保留 reasoning_content、tool_calls、tool_call_id。"""
    request = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": "职责"},
            {"role": "user", "content": "退款政策？"},
            {
                "role": "assistant",
                "content": "先查资料",
                "reasoning_content": "需要检索内部文档",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "search_knowledge", "arguments": '{"query": "退款"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "退款 7 天"},
            {
                "role": "assistant",
                "content": "结论",
                "reasoning_content": "未调用工具的轮次也要保留",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "检索",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
    }
    payload = official_message_payload(request)
    messages = payload["messages"]
    assert messages[2]["reasoning_content"] == "需要检索内部文档"
    assert messages[2]["tool_calls"][0]["id"] == "call_a"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"query": "退款"}'
    assert messages[3]["tool_call_id"] == "call_a"
    assert messages[4]["reasoning_content"] == "未调用工具的轮次也要保留"
    assert payload["tools"][0]["function"]["name"] == "search_knowledge"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["model"] == "deepseek-flash"
    # 计数口径不含 stream / max_tokens 等不影响 prompt 长度的字段
    assert "stream" not in payload
    assert "max_tokens" not in payload


def test_official_message_payload_keeps_only_official_fields() -> None:
    """工具定义只保留官方字段，本地字段不污染计数口径。"""
    payload = official_message_payload(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "description": "d",
                        "parameters": {},
                        "local_only": True,
                    },
                    "workspace_id": "ws",
                }
            ],
        }
    )
    assert set(payload["tools"][0]) == {"type", "function"}
    assert set(payload["tools"][0]["function"]) == {"name", "parameters", "description"}


def test_tool_message_without_call_id_is_rejected() -> None:
    with pytest.raises(TokenizerResourceError):
        to_official_messages([{"role": "tool", "content": "x"}])


def test_unknown_role_is_rejected_in_official_conversion() -> None:
    with pytest.raises(TokenizerResourceError):
        to_official_messages([{"role": "observer", "content": "x"}])


def test_multimodal_content_preserved() -> None:
    """多模态分段内容按官方形状保留（识图路径的计数口径）。"""
    content = [
        {"type": "text", "text": "这是什么"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    messages = to_official_messages([{"role": "user", "content": content}])
    assert messages[0]["content"] == content


def test_reasoning_message_count() -> None:
    """需要保留推理内容的消息条数会计入报告（PRD 9.3）。"""
    assert (
        count_reasoning_messages(
            [
                {"role": "assistant", "reasoning_content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "assistant", "reasoning_content": "c"},
            ]
        )
        == 2
    )


# ---------------------------------------------------------------- 2. 计数报告


def test_prompt_counter_counts_token_ids_and_reports_provenance() -> None:
    """计数来自 token ID 数量；报告带 tokenizer 版本与 SHA-256（PRD 9.3）。"""
    encoder = InMemoryEncoder()
    counter = PromptCounter(encoder)
    request = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "你好"}],
        "tools": [
            {"type": "function", "function": {"name": "t", "parameters": {}}},
        ],
    }
    encoding = counter.encode(request)
    assert encoding.token_count == len(encoding.token_ids)
    assert encoding.token_count == len(official_prompt(request))
    report = counter.report(request)
    assert report.local_prompt_tokens == encoding.token_count
    assert report.model == "deepseek-flash"
    assert report.tool_count == 1
    provenance = report.tokenizer
    assert provenance["tokenizer_id"] == TOKENIZER_MANIFEST.tokenizer_id
    assert provenance["tokenizer_version"] == TOKENIZER_MANIFEST.tokenizer_version
    assert provenance["sha256"] == TOKENIZER_PINNED_SHA256
    assert len(provenance["sha256"]) == 64


def test_counter_includes_tools_and_reasoning_in_request_payload() -> None:
    """计数请求里必须包含 tools 与需保留的推理内容（PRD 9.3）。"""
    encoder = InMemoryEncoder()
    counter = PromptCounter(encoder)
    counter.encode(
        {
            "model": "deepseek-flash",
            "messages": [
                {"role": "assistant", "content": "x", "reasoning_content": "保留的推理"},
            ],
            "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
        }
    )
    _messages, payload = encoder.calls[-1]["messages"], encoder.calls[-1]
    assert payload["tools"]
    assert any(item.get("reasoning_content") for item in payload["messages"])


def test_counter_without_resources_raises_instead_of_estimating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有 tokenizer 资源 -> 明确错误，绝不按字符数估算、绝不伪造计数。"""
    monkeypatch.delenv(TOKENIZER_DIR_ENV_VAR, raising=False)
    counter = PromptCounter()
    with pytest.raises(TokenizerResourceError) as exc:
        counter.count({"model": "deepseek-flash", "messages": [{"role": "user", "content": "x"}]})
    assert TOKENIZER_DIR_ENV_VAR in str(exc.value)
    assert "不按字符数估算" in str(exc.value)


def test_module_level_count_prompt_tokens_uses_default_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模块级入口与默认计数器同源（运行层唯一调用点）。"""
    from app.generation import tokenizer as tokenizer_module

    monkeypatch.setattr(
        tokenizer_module, "_DEFAULT_COUNTER", PromptCounter(InMemoryEncoder())
    )
    count = count_prompt_tokens(
        {"model": "deepseek-flash", "messages": [{"role": "user", "content": "你好"}]}
    )
    assert count > 0


# ---------------------------------------------------------------- 3. tokenizer 资源校验


def test_pinned_manifest_matches_official_artifact() -> None:
    """清单固定的是官方 v41/tokenizer.json 的大小与 SHA-256（PRD 9.3）。"""
    assert TOKENIZER_PINNED_SHA256 == (
        "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
    )
    assert TOKENIZER_PINNED_SIZE == 6_367_247
    assert TOKENIZER_MANIFEST.sha256_pinned is True
    assert TOKENIZER_MANIFEST.filename == TOKENIZER_RESOURCE_FILENAME


def test_resolve_requires_configured_resource_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """未配置资源目录 -> 明确错误并说明获取方式。"""
    monkeypatch.delenv(TOKENIZER_DIR_ENV_VAR, raising=False)
    with pytest.raises(TokenizerResourceError) as exc:
        resolve_pinned_encoder()
    assert TOKENIZER_DIR_ENV_VAR in str(exc.value)


def test_resolve_reads_env_directory_and_verifies_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recwarn: pytest.WarningsChecker
) -> None:
    """资源目录来自环境变量；文件通过 SHA-256 后走官方编码路径（假模块驱动）。"""
    content = b'{"model": {"type": "BPE"}}'
    resource = tmp_path / TOKENIZER_RESOURCE_FILENAME
    resource.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = TokenizerManifest(
        tokenizer_id="test/v41",
        tokenizer_version="v4.1",
        filename=TOKENIZER_RESOURCE_FILENAME,
        revision="test",
        sha256=digest,
        size=len(content),
    )
    monkeypatch.setattr(sys, "modules", {**sys.modules, "deepseek_recipe": _fake_recipe_module()})
    monkeypatch.setenv(TOKENIZER_DIR_ENV_VAR, str(tmp_path))
    counter = PromptCounter(manifest=manifest)
    encoding = counter.encode(
        {"model": "deepseek-flash", "messages": [{"role": "user", "content": "abcd"}]}
    )
    assert encoding.token_count > 0
    assert verify_resource(resource, manifest) == digest


def test_resolve_reports_missing_official_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """文件通过校验但未安装官方包 -> 明确错误（不退回估算）。"""
    content = b"{}"
    resource = tmp_path / TOKENIZER_RESOURCE_FILENAME
    resource.write_bytes(content)
    manifest = TokenizerManifest(
        tokenizer_id="test/v41",
        tokenizer_version="v4.1",
        filename=TOKENIZER_RESOURCE_FILENAME,
        revision="test",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    monkeypatch.setitem(sys.modules, "deepseek_recipe", None)  # 触发 ImportError
    with pytest.raises(TokenizerResourceError) as exc:
        resolve_pinned_encoder(manifest, resource_dir=tmp_path)
    assert "deepseek-recipe" in str(exc.value)


def test_verify_resource_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TokenizerResourceError) as exc:
        verify_resource(tmp_path / TOKENIZER_RESOURCE_FILENAME)
    assert "不存在" in str(exc.value)


def test_verify_resource_rejects_unpinned_hash(tmp_path: Path) -> None:
    """未固定 SHA-256 的清单必须拒绝（不"跳过校验"）。"""
    resource = tmp_path / TOKENIZER_RESOURCE_FILENAME
    resource.write_bytes(b"{}")
    manifest = TokenizerManifest(
        tokenizer_id="test",
        tokenizer_version="v0",
        filename=TOKENIZER_RESOURCE_FILENAME,
        revision="test",
        sha256="",
    )
    with pytest.raises(TokenizerResourceError) as exc:
        verify_resource(resource, manifest)
    assert "未固定" in str(exc.value)


def test_verify_resource_rejects_hash_mismatch(tmp_path: Path) -> None:
    resource = tmp_path / TOKENIZER_RESOURCE_FILENAME
    resource.write_bytes(b"{}")
    manifest = TokenizerManifest(
        tokenizer_id="test",
        tokenizer_version="v0",
        filename=TOKENIZER_RESOURCE_FILENAME,
        revision="test",
        sha256="0" * 64,
        size=2,
    )
    with pytest.raises(TokenizerResourceError) as exc:
        verify_resource(resource, manifest)
    assert "不匹配" in str(exc.value)


def test_verify_resource_rejects_size_mismatch(tmp_path: Path) -> None:
    """字节数不符（截断下载）即拒绝，不看哈希。"""
    content = b"{}"
    resource = tmp_path / TOKENIZER_RESOURCE_FILENAME
    resource.write_bytes(content)
    manifest = TokenizerManifest(
        tokenizer_id="test",
        tokenizer_version="v0",
        filename=TOKENIZER_RESOURCE_FILENAME,
        revision="test",
        sha256=hashlib.sha256(content).hexdigest(),
        size=999,
    )
    with pytest.raises(TokenizerResourceError) as exc:
        verify_resource(resource, manifest)
    assert "字节" in str(exc.value)


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    resource = tmp_path / "x.bin"
    payload = b"abc" * 1000
    resource.write_bytes(payload)
    assert sha256_of_file(resource) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------- 4. 预算公式


def _caps(context_limit: int, max_output_tokens: int = 4096):
    return resolve_capabilities(
        "budget-model",
        "https://compat.example.test/v1",
        declared={
            "supports_tools": True,
            "supports_thinking_toggle": True,
            "reasoning_levels": ["off", "low", "high", "max"],
            "supports_image_input": False,
            "context_limit": context_limit,
            "max_output_tokens": max_output_tokens,
            "history_protocol": "deepseek-chat-completions-reasoning-replay",
        },
    )


def test_budget_exact_boundary_fits() -> None:
    """等于上限可提交（PRD 9.3 的 <= 判定）。"""
    caps = _caps(10_000)
    budget = evaluate_context_budget(
        caps, local_prompt_tokens=7_000, reserved_output_tokens=2_500, protocol_margin=500
    )
    assert budget.required_tokens == 10_000
    assert budget.remaining_tokens == 0
    assert budget.fits is True
    assert budget.status in {BudgetStatus.OK, BudgetStatus.COMPACT}


def test_budget_one_token_over_limit_is_rejected() -> None:
    """超出上限 1 个 token 即不得提交。"""
    caps = _caps(10_000)
    budget = evaluate_context_budget(
        caps, local_prompt_tokens=7_001, reserved_output_tokens=2_500, protocol_margin=500
    )
    assert budget.required_tokens == 10_001
    assert budget.remaining_tokens == -1
    assert budget.fits is False
    assert budget.status is BudgetStatus.OVER_LIMIT


def test_budget_compaction_threshold_is_80_percent_of_usable_input() -> None:
    """可用输入空间（上限 - 预留输出 - 协议余量）达到 80% -> 安排整理（PRD 18.1）。"""
    caps = _caps(10_000)
    usable = 10_000 - 2_000 - DEFAULT_PROTOCOL_MARGIN_TOKENS  # = 5952
    threshold = int(usable * DEFAULT_CONTEXT_COMPACTION_RATIO)
    below = evaluate_context_budget(
        caps,
        local_prompt_tokens=threshold - 1,
        reserved_output_tokens=2_000,
    )
    at_threshold = evaluate_context_budget(
        caps,
        local_prompt_tokens=threshold,
        reserved_output_tokens=2_000,
    )
    assert DEFAULT_CONTEXT_COMPACTION_RATIO == 0.8
    assert below.needs_compaction is False
    assert below.status is BudgetStatus.OK
    assert at_threshold.needs_compaction is True
    assert at_threshold.status is BudgetStatus.COMPACT
    assert at_threshold.compaction_threshold_tokens == threshold
    assert at_threshold.available_input_tokens == usable


def test_budget_rejects_negative_inputs() -> None:
    caps = _caps(10_000)
    with pytest.raises(ValueError):
        evaluate_context_budget(caps, local_prompt_tokens=-1, reserved_output_tokens=0)
    with pytest.raises(ValueError):
        evaluate_context_budget(caps, local_prompt_tokens=1, reserved_output_tokens=-5)


def test_budget_rejects_invalid_compaction_ratio() -> None:
    caps = _caps(10_000)
    with pytest.raises(ValueError):
        evaluate_context_budget(
            caps, local_prompt_tokens=1, reserved_output_tokens=1, compaction_ratio=0
        )


def test_budget_as_dict_reports_all_fields() -> None:
    caps = _caps(10_000)
    payload = evaluate_context_budget(
        caps, local_prompt_tokens=100, reserved_output_tokens=50, protocol_margin=10
    ).as_dict()
    assert payload["provider_context_limit"] == 10_000
    assert payload["required_tokens"] == 160
    assert payload["fits"] is True
    assert payload["status"] == "ok"


def test_should_compact_waits_for_complete_tool_round() -> None:
    """本组工具结果未完整返回时不得整理上下文（PRD 9.3）。"""
    caps = _caps(10_000)
    budget = evaluate_context_budget(
        caps, local_prompt_tokens=6_500, reserved_output_tokens=1_000
    )
    assert budget.needs_compaction is True
    assert should_compact(budget, tool_round_complete=True) is True
    assert should_compact(budget, tool_round_complete=False) is False
    small = evaluate_context_budget(
        caps, local_prompt_tokens=10, reserved_output_tokens=1_000
    )
    assert should_compact(small, tool_round_complete=True) is False


# ---------------------------------------------------------------- 5. 上下文顺序与压缩


def test_build_context_uses_fixed_section_order() -> None:
    """输入顺序固定（PRD 9.3）：职责 → 工具定义 → 快照 → 历史 → 最新 → 动态。"""
    messages = build_context(
        duty_prompt="职责提示",
        tool_definitions=[{"type": "function", "function": {"name": "t", "description": "d"}}],
        snapshot="工作区快照",
        history=[{"role": "user", "content": "旧问题"}],
        latest=[{"role": "user", "content": "新问题"}],
        dynamic=[{"role": "user", "content": "检索结果"}],
    )
    contents = [item["content"] for item in messages]
    assert contents[0] == "职责提示"
    assert "t" in contents[1]
    assert contents[2] == "工作区快照"
    assert contents[3] == "旧问题"
    assert contents[4] == "新问题"
    assert contents[5] == "检索结果"


def test_snapshot_truncates_at_default_limit() -> None:
    """快照正文长度默认 8000 字符（PRD 18.1），截断时标注而不静默丢弃。"""
    context = AgentContext(context_id="c1", snapshot="x" * 9000)
    assert context.snapshot.truncated is True
    assert context.snapshot.text.startswith("x" * 100)
    assert "截断" in context.snapshot.text
    assert len(context.snapshot.text) < 8100


def test_agent_context_reports_epochs_and_request() -> None:
    context = AgentContext(
        context_id="c1",
        duty_prompt="职责",
        history=({"role": "user", "content": "q"},),
        latest=({"role": "user", "content": "q2"},),
    )
    request = context.request("deepseek-flash")
    assert request["model"] == "deepseek-flash"
    assert len(request["messages"]) == 3
    assert context.agent_context_epoch == 1
    assert context.context_epoch == 1


def test_compact_context_only_bumps_agent_epoch() -> None:
    """压缩只增加该 Agent 的 agent_context_epoch，会话 context_epoch 不变（PRD 9.3）。"""
    context = AgentContext(
        context_id="c1",
        session_epoch=4,
        duty_prompt="职责",
        history=({"role": "user", "content": "很长的一段历史"},),
    )
    compacted = compact_context(context, "用户目标/已核实结论/未解决问题")
    assert compacted.agent_context_epoch == 2
    assert compacted.context_epoch == 4
    assert compacted.history == ()
    assert compacted.summaries == ("用户目标/已核实结论/未解决问题",)
    contents = [item["content"] for item in compacted.messages()]
    assert any("历史摘要" in item and "参考数据" in item for item in contents)
    # 原对象不变
    assert context.agent_context_epoch == 1


def test_compacted_context_never_invents_tool_calls() -> None:
    """压缩摘要只作为 system 参考数据，不制造 assistant 工具调用（PRD 9.2 末段）。"""
    compacted = compact_context(AgentContext(context_id="c"), "摘要")
    for message in compacted.messages():
        assert message["role"] in {"system", "user"}
        assert "tool_calls" not in message


def test_agent_context_budget_uses_counter() -> None:
    """运行层接口：AgentContext.budget(model, capabilities) 给预算判定。"""
    caps = _caps(10_000)
    context = AgentContext(
        context_id="c1",
        history=({"role": "user", "content": "q" * 50},),
        reserved_output_tokens=1_000,
    )
    budget = context.budget("deepseek-flash", caps, counter=PromptCounter(InMemoryEncoder()))
    assert isinstance(budget, ContextBudget)
    assert budget.local_prompt_tokens > 0
    assert budget.reserved_output_tokens == 1_000


def test_context_budget_reserved_output_can_follow_capability() -> None:
    """预留输出缺省值不小于 floor，且不裁剪模型能力声明。"""
    from app.generation.context import default_reserved_output_tokens

    assert default_reserved_output_tokens(_caps(10_000, max_output_tokens=512)) == 4_096
    assert default_reserved_output_tokens(_caps(1_000_000, max_output_tokens=393_216)) == 393_216


def test_deepseek_default_capability_budget_is_sane() -> None:
    """官方模型能力下的预算示例（运行层调用样例，供 runtime/routing 参考）。"""
    caps = resolve_capabilities("deepseek-flash", OFFICIAL)
    budget = evaluate_context_budget(
        caps,
        local_prompt_tokens=100_000,
        reserved_output_tokens=65_536,
        protocol_margin=2_048,
    )
    assert budget.fits is True
    assert budget.status is BudgetStatus.OK
    assert budget.compaction_threshold_tokens == int(
        (caps.context_limit - 65_536 - 2_048) * 0.8
    )
    assert ReasoningLevel.HIGH.value == "high"
