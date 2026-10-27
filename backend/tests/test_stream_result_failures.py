import pytest

from app.generation.base import ModelResponse, StreamResult, classify_finish_reason


@pytest.mark.parametrize("finish_reason", [
    "length", "content_filter", "aborted", "insufficient_system_resource",
])
def test_incomplete_response_explains_provider_finish_reason(finish_reason: str) -> None:
    response = ModelResponse(
        finish_reason=finish_reason, status=classify_finish_reason(finish_reason),
    )
    result = StreamResult.from_response(response)
    assert not result.succeeded
    assert result.failure_reasons
    assert f"finish_reason={finish_reason}" in result.failure_reasons[0]


def test_output_limit_preserves_additional_provider_notes() -> None:
    response = ModelResponse(
        content="部分正文", finish_reason="length", status=classify_finish_reason("length"),
        notes=["工具参数未完整到达"],
    )
    result = StreamResult.from_response(response)
    assert len(result.failure_reasons) == 2
    assert "输出 token 上限" in result.failure_reasons[0]
    assert result.failure_reasons[1] == response.notes[0]


def test_complete_response_has_no_failure_reasons() -> None:
    response = ModelResponse(
        content="正文", finish_reason="stop", status=classify_finish_reason("stop"),
    )
    result = StreamResult.from_response(response)
    assert result.succeeded
    assert result.failure_reasons == []
