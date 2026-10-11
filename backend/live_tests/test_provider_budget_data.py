import pytest

from app.core.outbound import OutboundClient
from app.generation.providers import ContextWindowExceededError, OpenAICompatChatModel
from app.generation.tokenizer import PromptCounter


def flash_model(outbound=None):
    return OpenAICompatChatModel(
        base_url="https://api.commandcode.ai/v1",
        model="deepseek/deepseek-v4.1-flash",
        outbound=outbound,
    )


def test_final_request_counts_system_tools_and_tool_results():
    model = flash_model()
    request = model.build_request([
        {"role": "system", "content": "回答需要依据已读取的资料。"},
        {"role": "user", "content": "请查找会话记忆的更新条件。"},
        {"role": "assistant", "content": "", "reasoning_content": "需要读取资料。",
         "tool_calls": [{"id": "read-1", "type": "function", "function": {
             "name": "read_document", "arguments": '{"document_id":"memory"}'
         }}]},
        {"role": "tool", "tool_call_id": "read-1", "content": "人工修改会增加记忆版本。"},
    ], stream=True, reasoning_level="low", max_tokens=4096, tools=[{
        "type": "function", "function": {
            "name": "read_document", "description": "读取资料正文",
            "parameters": {"type": "object", "properties": {
                "document_id": {"type": "string"}
            }, "required": ["document_id"]},
        },
    }])
    budget = model.context_budget(request)
    assert budget.local_prompt_tokens == PromptCounter().count(request)
    assert budget.local_prompt_tokens > PromptCounter().count({
        "model": request["model"], "messages": [request["messages"][1]],
    })
    assert budget.provider_context_limit == 1_000_000
    assert budget.reserved_output_tokens == 4096
    assert budget.protocol_margin == 0
    assert budget.fits


def test_default_output_reservation_and_extra_body_use_final_payload():
    model = flash_model()
    request = model.build_request([
        {"role": "user", "content": "请解释记忆。"},
    ], stream=False)
    budget = model.context_budget(request)
    assert budget.reserved_output_tokens == model.capabilities.max_output_tokens
    request["extra_body"] = {
        "messages": [{"role": "user", "content": "请说明最终请求中的工具结果。"}],
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
    }
    final_request = {**request, **request["extra_body"]}
    final_request.pop("extra_body")
    final_budget = model.context_budget(request)
    assert final_budget.local_prompt_tokens == PromptCounter().count(final_request)
    assert final_budget.reserved_output_tokens == 2048


def test_other_models_do_not_require_deepseek_counter():
    for base_url, model_id in (
        ("https://example.com/v1", "other-model"),
        ("https://example.com/v1", "deepseek-flash"),
        ("https://api.deepseek.com/v1", "deepseek-v4-flash"),
        ("https://api.deepseek.com/v1", "deepseek-v4-pro"),
    ):
        model = OpenAICompatChatModel(base_url=base_url, model=model_id)
        request = model.build_request([{"role": "user", "content": "你好"}], stream=True)
        assert model.context_budget(request) is None
        assert model._prompt_counter is None


@pytest.fixture(scope="module")
def over_limit_request():
    outbound = OutboundClient()
    model = flash_model(outbound)
    content = "上下文预算核验。" * 150_000
    request = model.build_request([{"role": "user", "content": content}], stream=True)
    budget = model.context_budget(request)
    assert not budget.fits, budget.as_dict()
    return model, outbound, request, budget


@pytest.mark.parametrize("method", ["stream_chat", "stream_result", "complete"])
async def test_over_limit_request_stops_before_network(over_limit_request, method):
    model, outbound, request, budget = over_limit_request
    with pytest.raises(ContextWindowExceededError) as raised:
        if method == "complete":
            await model.complete(request["messages"])
        else:
            async for _chunk in getattr(model, method)(request["messages"]):
                pass
    assert raised.value.budget.local_prompt_tokens == budget.local_prompt_tokens
    assert raised.value.budget.protocol_margin == 0
    assert outbound.entries() == []
