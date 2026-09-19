"""T03 单测：重排内部对象、档案与错误类型（PRD §5.3/§6/§8.4/§9）。

纯类型与校验检查：不调用任何模型服务、不起 HTTP 服务、不读知识库。
覆盖每个错误类型（PRD §8.4 八个用户错误码 + 服务不可用 + 内部契约违例）与
非法输入拒绝（空候选、重复候选、scope 越界、版本变化、缺哈希、非有限分数、
top_n 不符、档案字段与能力限制非法）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, datetime

import pytest

from app.core.outbound import outbound_client
from app.retrieval import reranker
from app.retrieval.reranker import (
    CANDIDATE_WINDOW_CHOICES,
    ERROR_USER_MESSAGES,
    EVIDENCE_CHAR_BUDGET,
    MAX_SELECTED_SOURCES,
    EvidenceVersionChangedError,
    ExclusionReason,
    ExclusionRecord,
    ProfileStatus,
    RerankAuthFailedError,
    RerankCandidate,
    RerankError,
    RerankErrorCode,
    RerankInputTooLongError,
    RerankInvalidResponseError,
    RerankNotConfiguredError,
    RerankProfile,
    RerankProfileMismatchError,
    RerankProvider,
    RerankProviderCapabilities,
    RerankProviderType,
    RerankRateLimitedError,
    RerankRequest,
    RerankResult,
    RerankScorable,
    RerankScore,
    RerankServiceUnavailableError,
    RerankTimeoutError,
    RerankUsage,
    RerankValidationError,
    RouteSignals,
    ScoreScope,
    ScoreType,
    SelectionResult,
    SelectionStatus,
    ViewKind,
    assert_within_token_limits,
    error_for_http_status,
    sha256_hex,
)

SHA = sha256_hex("evidence")
TEXT = "证据原文片段"


# --------------------------------------------------------------------------------------
# 夹具构造
# --------------------------------------------------------------------------------------


def make_candidate(**overrides) -> RerankCandidate:
    values = {
        "candidate_id": "cand-1",
        "scope_id": "global",
        "index_revision": "rev-1",
        "ranking_text": f"笔记.md > 标题\n{TEXT}",
        "payload_hash": SHA,
    }
    values.update(overrides)
    return RerankCandidate(**values)


def make_request(**overrides) -> RerankRequest:
    values = {
        "query": "为什么 AI 提效没有落地？",
        "scope_id": "global",
        "index_revision": "rev-1",
        "documents": (make_candidate(),),
        "profile_id": "prof-1",
        "instruction_id": "rerank-task-v1",
        "instruction_text": "判断内容是否提供回答原问题所需的事实",
    }
    values.update(overrides)
    return RerankRequest(**values)


def make_capabilities(**overrides) -> RerankProviderCapabilities:
    values = {
        "provider_type": RerankProviderType.SILICONFLOW_RERANK,
        "model": "Qwen/Qwen3-Reranker-8B",
        "tokenizer_id": "Qwen/Qwen3-Reranker-8B@transformers",
        "max_documents": 100,
        "score_min": 0.0,
        "score_max": 1.0,
        "supports_instruction": True,
    }
    values.update(overrides)
    return RerankProviderCapabilities(**values)


def make_profile(**overrides) -> RerankProfile:
    values = {
        "profile_id": "prof-1",
        "status": ProfileStatus.VALIDATED,
        "provider_type": RerankProviderType.SILICONFLOW_RERANK,
        "model": "Qwen/Qwen3-Reranker-8B",
        "endpoint_fingerprint": "https://api.siliconflow.cn|https",
        "instruction_id": "rerank-task-v1",
        "instruction_hash": sha256_hex("task instruction v1"),
        "tokenizer_id": "Qwen/Qwen3-Reranker-8B@transformers",
        "view_version": "evidence-view-v1",
        "scoring_protocol_version": "rerank-protocol-v1",
        "score_threshold": 0.35,
        "max_candidates": 50,
        "min_candidate_count": 20,
        "generated_at": datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return RerankProfile(**values)


class StubView:
    """结构上满足 ``RerankScorable`` 的证据视图替身（模拟 evidence_view.EvidenceView）。"""

    def __init__(self, candidate_id: str, text: str, view_kind: str = "block") -> None:
        self.candidate_id = candidate_id
        self.scope_id = "global"
        self.index_revision = "rev-1"
        self.ranking_text = text
        self.evidence_text = text
        self.payload_hash = sha256_hex(text)
        self.view_kind = view_kind


# --------------------------------------------------------------------------------------
# 错误类型（PRD §8.4）
# --------------------------------------------------------------------------------------

PRD_ERROR_CONTRACT = {
    RerankErrorCode.NOT_CONFIGURED: (RerankNotConfiguredError, "筛选服务尚未配置，请先在设置中完成配置"),
    RerankErrorCode.PROFILE_MISMATCH: (
        RerankProfileMismatchError,
        (
            "当前模型或设置与已验收档案不匹配：请在设置页查看档案状态与验证日期"
            "（档案未通过验收、或模型/指令/tokenizer/视图版本/评分协议与档案不一致）"
        ),
    ),
    RerankErrorCode.AUTH_FAILED: (RerankAuthFailedError, "筛选服务认证失败"),
    RerankErrorCode.RATE_LIMITED: (RerankRateLimitedError, "筛选服务请求受限，请稍后重试"),
    RerankErrorCode.TIMEOUT: (RerankTimeoutError, "筛选服务响应超时"),
    RerankErrorCode.INPUT_TOO_LONG: (RerankInputTooLongError, "待检查资料超过服务输入限制"),
    RerankErrorCode.INVALID_RESPONSE: (
        RerankInvalidResponseError,
        "筛选服务返回的数据不完整或无效",
    ),
    RerankErrorCode.EVIDENCE_VERSION_CHANGED: (
        EvidenceVersionChangedError,
        "资料在本轮处理期间发生变化，请重新提问",
    ),
    RerankErrorCode.SERVICE_UNAVAILABLE: (
        RerankServiceUnavailableError,
        "筛选服务暂时不可用，请稍后重试",
    ),
    RerankErrorCode.INVALID_INPUT: (RerankValidationError, "重排输入不满足内部契约"),
}

#: PRD §8.4 固定错误码的原始字符串，字面比对防止改名。
PRD_ERROR_CODE_VALUES = (
    "RERANK_NOT_CONFIGURED",
    "RERANK_PROFILE_MISMATCH",
    "RERANK_AUTH_FAILED",
    "RERANK_RATE_LIMITED",
    "RERANK_TIMEOUT",
    "RERANK_INPUT_TOO_LONG",
    "RERANK_INVALID_RESPONSE",
    "EVIDENCE_VERSION_CHANGED",
)


def test_prd_error_codes_keep_exact_string_values():
    values = {code.value for code in RerankErrorCode}
    assert set(PRD_ERROR_CODE_VALUES) <= values
    # 别名与早期引用方式保持一致
    assert reranker.EVIDENCE_VERSION_CHANGED is RerankErrorCode.EVIDENCE_VERSION_CHANGED


@pytest.mark.parametrize("code", list(PRD_ERROR_CONTRACT))
def test_each_error_type_has_code_and_fixed_user_message(code):
    error_type, message = PRD_ERROR_CONTRACT[code]
    error = error_type("HTTP 429")
    assert isinstance(error, RerankError)
    assert error.code is code
    assert error.user_message == message == ERROR_USER_MESSAGES[code]
    assert code.value in str(error)
    assert error.http_status is None


@pytest.mark.parametrize("code", list(PRD_ERROR_CONTRACT))
def test_user_event_exposes_only_code_and_message(code):
    error_type, message = PRD_ERROR_CONTRACT[code]
    # detail 形如「服务原始响应体」，绝不能出现在用户事件里
    error = error_type("raw provider body: secret", http_status=429)
    event = error.to_event()
    assert event == {"type": "error", "code": code.value, "message": message}
    assert "raw provider body" not in json.dumps(event, ensure_ascii=False)
    assert error.http_status == 429


@pytest.mark.parametrize(
    ("status", "expected_type"),
    [
        (401, RerankAuthFailedError),
        (403, RerankAuthFailedError),
        (429, RerankRateLimitedError),
        (413, RerankInputTooLongError),
        (502, RerankServiceUnavailableError),
        (503, RerankServiceUnavailableError),
        (504, RerankServiceUnavailableError),
        (404, RerankProfileMismatchError),
        (400, RerankValidationError),
        (422, RerankValidationError),
        (500, RerankInvalidResponseError),
    ],
)
def test_http_status_maps_to_explicit_error_type(status, expected_type):
    error = error_for_http_status(status)
    assert type(error) is expected_type
    assert error.http_status == status
    assert error.detail == f"HTTP {status}"


def test_http_status_helper_rejects_non_status_values():
    with pytest.raises(RerankValidationError):
        error_for_http_status(True)


def test_error_can_carry_audited_http_status_and_sanitized_detail():
    error = error_for_http_status(503, "model overloaded")
    assert error.code is RerankErrorCode.SERVICE_UNAVAILABLE
    assert error.http_status == 503
    assert "model overloaded" in error.detail


# --------------------------------------------------------------------------------------
# 类型层不外呼
# --------------------------------------------------------------------------------------


def test_type_layer_records_no_outbound_traffic():
    outbound_client.clear()
    make_request(documents=(make_candidate(), make_candidate(candidate_id="cand-2")))
    make_profile()
    make_capabilities()
    assert outbound_client.is_empty()
    assert outbound_client.outbound_state == "local-only"


# --------------------------------------------------------------------------------------
# 候选与请求校验
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", ""),
        ("candidate_id", "   "),
        ("scope_id", ""),
        ("index_revision", ""),
        ("ranking_text", ""),
        ("payload_hash", ""),
        ("payload_hash", "abc"),
        ("payload_hash", "z" * 64),
        ("view_kind", "unknown-kind"),
    ],
)
def test_candidate_rejects_invalid_fields(field, value):
    with pytest.raises(RerankValidationError):
        make_candidate(**{field: value})


def test_candidate_accepts_full_shape_and_serializes_signals():
    candidate = make_candidate(
        view_kind="support_bundle",
        primary_block_id="blk-1",
        display_name="笔记.md",
        heading_path="笔记.md > 标题",
        route_signals=RouteSignals(dense_score=0.71, dense_rank=1, rrf_score=0.03, rrf_rank=1),
    )
    assert candidate.view_kind is ViewKind.SUPPORT_BUNDLE
    assert candidate.display_name == "笔记.md"
    assert candidate.route_signals.to_dict()["dense_rank"] == 1


@pytest.mark.parametrize(
    "signals",
    [
        {"dense_score": float("nan")},
        {"bm25_score": float("inf")},
        {"rrf_score": float("-inf")},
        {"dense_rank": 0},
        {"bm25_rank": -1},
        {"rrf_rank": True},
        {"lexical_only": "yes"},
    ],
)
def test_route_signals_reject_non_finite_scores_and_bad_ranks(signals):
    with pytest.raises(RerankValidationError):
        RouteSignals(**signals)


def test_request_requires_non_empty_query():
    with pytest.raises(RerankValidationError):
        make_request(query="  ")


def test_request_rejects_empty_candidate_list():
    with pytest.raises(RerankValidationError, match="候选列表不能为空"):
        make_request(documents=())


def test_request_rejects_duplicate_candidates():
    duplicate = make_candidate()
    with pytest.raises(RerankValidationError, match="重复"):
        make_request(documents=(duplicate, duplicate))


def test_request_rejects_candidate_from_other_scope():
    other = StubView("cand-2", TEXT)
    other.scope_id = "workspace-9"
    with pytest.raises(RerankValidationError, match="scope"):
        make_request(documents=(make_candidate(), other))


def test_request_rejects_candidate_from_other_revision():
    stale = StubView("cand-2", TEXT)
    stale.index_revision = "rev-0"
    with pytest.raises(EvidenceVersionChangedError) as excinfo:
        make_request(documents=(make_candidate(), stale))
    assert excinfo.value.code is RerankErrorCode.EVIDENCE_VERSION_CHANGED
    assert "rev-0" not in str(excinfo.value)


def test_request_rejects_candidate_missing_ranking_text():
    broken = StubView("cand-2", TEXT)
    del broken.ranking_text
    with pytest.raises(RerankValidationError, match="缺少字段"):
        make_request(documents=(make_candidate(), broken))


def test_request_rejects_candidate_missing_payload_hash():
    broken = StubView("cand-2", TEXT)
    del broken.payload_hash
    with pytest.raises(RerankValidationError, match="缺少字段"):
        make_request(documents=(make_candidate(), broken))


def test_request_rejects_top_n_not_equal_to_candidate_count():
    with pytest.raises(RerankValidationError, match="top_n"):
        make_request(top_n=5)
    with pytest.raises(RerankValidationError, match="top_n"):
        make_request(top_n=0)


def test_request_top_n_matches_candidate_count_when_given():
    request = make_request(
        documents=(make_candidate(), make_candidate(candidate_id="cand-2")),
        top_n=2,
    )
    assert request.top_n_value == 2
    assert request.candidate_ids == ("cand-1", "cand-2")


def test_request_accepts_structurally_compatible_evidence_views():
    view = StubView("cand-1", TEXT)
    assert isinstance(view, RerankScorable)
    assert isinstance(make_candidate(), RerankScorable)
    request = make_request(documents=(view,))
    assert request.candidate_ids == ("cand-1",)


def test_request_audit_dict_contains_no_question_or_evidence_text():
    request = make_request()
    audit = json.dumps(request.to_audit_dict(), ensure_ascii=False)
    assert "AI 提效" not in audit
    assert TEXT not in audit
    assert audit.count("global") == 1
    assert request.to_audit_dict()["candidate_count"] == 1


# --------------------------------------------------------------------------------------
# 分数、用量与 Provider 响应
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf"), "0.9", None, True])
def test_score_rejects_non_finite_values(bad_score):
    with pytest.raises(RerankInvalidResponseError):
        RerankScore(candidate_id="cand-1", index=0, score=bad_score)


@pytest.mark.parametrize("bad_index", [-1, 1.5, "0", True, None])
def test_score_rejects_invalid_index(bad_index):
    with pytest.raises(RerankInvalidResponseError):
        RerankScore(candidate_id="cand-1", index=bad_index, score=0.9)


def test_score_keeps_score_type_separate_from_vector_similarity():
    score = RerankScore(candidate_id="cand-1", index=0, score=0.812)
    assert score.score_type is ScoreType.RERANK
    assert score.to_dict()["score_type"] == "rerank"
    assert ScoreType.DENSE_SIMILARITY.value == "dense_similarity"
    with pytest.raises(RerankInvalidResponseError):
        RerankScore(candidate_id="cand-1", index=0, score=0.9, score_type="similarity")


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1},
        {"output_tokens": 1.5},
        {"image_tokens": True},
        {"billed_units": float("nan")},
    ],
)
def test_usage_rejects_invalid_metering(usage):
    with pytest.raises(RerankInvalidResponseError):
        RerankUsage(**usage)


def test_usage_reported_flag_distinguishes_missing_metering():
    assert RerankUsage().reported is False
    assert RerankUsage(input_tokens=120, output_tokens=0).reported is True


def make_result(**overrides) -> RerankResult:
    values = {
        "profile_id": "prof-1",
        "provider_type": RerankProviderType.SILICONFLOW_RERANK,
        "model": "Qwen/Qwen3-Reranker-8B",
        "document_count": 2,
        "scores": (
            RerankScore(candidate_id="cand-1", index=0, score=0.97),
            RerankScore(candidate_id="cand-2", index=1, score=0.00022),
        ),
        "request_id": "req-1",
        "trace_id": "ti_5zbcks2o1el7jm7z6l",
        "usage": RerankUsage(input_tokens=88, output_tokens=0),
    }
    values.update(overrides)
    return RerankResult(**values)


def test_result_accepts_complete_provider_payload():
    result = make_result()
    assert result.score_for("cand-1") == 0.97
    assert result.score_for("missing") is None
    assert result.usage_reported is True
    assert result.score_scope is ScoreScope.REQUEST_RELATIVE


def test_result_requires_one_finite_score_per_candidate():
    with pytest.raises(RerankInvalidResponseError, match="数量"):
        make_result(scores=(RerankScore(candidate_id="cand-1", index=0, score=0.97),))


def test_result_rejects_duplicate_candidate_scores():
    with pytest.raises(RerankInvalidResponseError, match="重复"):
        make_result(
            scores=(
                RerankScore(candidate_id="cand-1", index=0, score=0.97),
                RerankScore(candidate_id="cand-1", index=1, score=0.1),
            )
        )


def test_result_rejects_duplicate_and_out_of_range_index():
    with pytest.raises(RerankInvalidResponseError, match="index 重复"):
        make_result(
            scores=(
                RerankScore(candidate_id="cand-1", index=0, score=0.97),
                RerankScore(candidate_id="cand-2", index=0, score=0.1),
            )
        )
    with pytest.raises(RerankInvalidResponseError, match="越界"):
        make_result(
            scores=(
                RerankScore(candidate_id="cand-1", index=0, score=0.97),
                RerankScore(candidate_id="cand-2", index=7, score=0.1),
            )
        )


@pytest.mark.parametrize("bad", [{"document_count": 0}, {"latency_ms": -1.0}, {"usage": "none"}])
def test_result_rejects_invalid_metadata(bad):
    with pytest.raises(RerankInvalidResponseError):
        make_result(**bad)


def test_result_rejects_scores_outside_declared_range():
    capabilities = make_capabilities(score_min=0.0, score_max=1.0)
    make_result().assert_scores_in_range(capabilities)
    with pytest.raises(RerankInvalidResponseError, match="分数范围"):
        make_result(
            scores=(
                RerankScore(candidate_id="cand-1", index=0, score=1.4),
                RerankScore(candidate_id="cand-2", index=1, score=0.2),
            )
        ).assert_scores_in_range(capabilities)


def test_result_assert_matches_request_checks_index_mapping_and_profile():
    request = make_request(
        documents=(make_candidate(), make_candidate(candidate_id="cand-2")),
        top_n=2,
    )
    make_result().assert_matches_request(request)

    with pytest.raises(RerankInvalidResponseError, match="不对应"):
        make_result(
            scores=(
                RerankScore(candidate_id="cand-2", index=0, score=0.97),
                RerankScore(candidate_id="cand-1", index=1, score=0.1),
            )
        ).assert_matches_request(request)

    with pytest.raises(RerankProfileMismatchError):
        make_result(profile_id="other-profile").assert_matches_request(request)

    single = RerankResult(
        profile_id="prof-1",
        provider_type=RerankProviderType.SILICONFLOW_RERANK,
        model="Qwen/Qwen3-Reranker-8B",
        document_count=1,
        scores=(RerankScore(candidate_id="cand-1", index=0, score=0.97),),
    )
    with pytest.raises(RerankInvalidResponseError, match="候选数"):
        single.assert_matches_request(request)


def test_result_dict_exposes_no_text():
    payload = json.dumps(make_result().to_dict(), ensure_ascii=False)
    assert TEXT not in payload
    assert "ranking_text" not in payload


# --------------------------------------------------------------------------------------
# Provider 能力与抽象接口
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"provider_type": ""},
        {"model": ""},
        {"tokenizer_id": ""},
        {"max_documents": 0},
        {"max_document_tokens": -1},
        {"score_min": float("nan")},
        {"score_min": 0.5, "score_max": 0.5},
        {"score_max": -0.1},
        {"supports_instruction": "yes"},
    ],
)
def test_capabilities_reject_invalid_values(bad):
    with pytest.raises(RerankValidationError):
        make_capabilities(**bad)


def test_capabilities_allow_unmeasured_limits_and_reject_oversized_batch():
    unmeasured = make_capabilities(max_documents=None, max_document_tokens=None)
    unmeasured.assert_document_count(1000)
    with pytest.raises(RerankInputTooLongError, match="候选数"):
        make_capabilities(max_documents=50).assert_document_count(51)


def test_capabilities_reject_unsupported_model_features_explicitly():
    supported = make_capabilities(supports_instruction=True, supports_chunk_options=True)
    supported.assert_instruction_supported()
    supported.assert_chunk_options_supported()
    without_instruction = make_capabilities(supports_instruction=False)
    with pytest.raises(RerankProfileMismatchError, match="instruction"):
        without_instruction.assert_instruction_supported()
    with pytest.raises(RerankProfileMismatchError, match="max_chunks_per_doc"):
        without_instruction.assert_chunk_options_supported()


def test_token_limits_reject_oversized_query_document_and_batch():
    capabilities = make_capabilities(
        max_query_tokens=100, max_document_tokens=200, max_request_tokens=500
    )
    assert_within_token_limits(
        capabilities, query_tokens=100, document_tokens=[200, 199], request_tokens=500
    )
    with pytest.raises(RerankInputTooLongError, match="问题长度"):
        assert_within_token_limits(capabilities, query_tokens=101, document_tokens=[10])
    with pytest.raises(RerankInputTooLongError, match="位置 1"):
        assert_within_token_limits(capabilities, query_tokens=10, document_tokens=[200, 201])
    with pytest.raises(RerankInputTooLongError, match="请求上限"):
        assert_within_token_limits(
            capabilities, query_tokens=10, document_tokens=[10], request_tokens=501
        )


def test_token_limits_skip_unmeasured_limits():
    capabilities = make_capabilities(max_query_tokens=None, max_document_tokens=None)
    assert_within_token_limits(capabilities, query_tokens=10_000, document_tokens=[10_000])


def test_token_limits_reject_negative_counts():
    with pytest.raises(RerankValidationError):
        assert_within_token_limits(make_capabilities(), query_tokens=-1, document_tokens=[1])


class StubProvider(RerankProvider):
    """最小可实例化子类：只验证抽象契约，不产生任何评分结果。"""

    def __init__(self) -> None:
        self._capabilities = make_capabilities()

    @property
    def provider_type(self) -> str:
        return RerankProviderType.SILICONFLOW_RERANK

    @property
    def capabilities(self) -> RerankProviderCapabilities:
        return self._capabilities

    async def rerank(self, request: RerankRequest) -> RerankResult:
        raise NotImplementedError


def test_provider_interface_is_abstract_and_requires_all_members():
    with pytest.raises(TypeError):
        RerankProvider()

    class _Incomplete(RerankProvider):
        @property
        def provider_type(self) -> str:
            return "siliconflow_rerank"

    with pytest.raises(TypeError):
        _Incomplete()


def test_provider_contract_is_async_and_closable():
    provider = StubProvider()
    assert inspect.iscoroutinefunction(provider.rerank)
    assert provider.provider_type == "siliconflow_rerank"
    assert provider.capabilities.model == "Qwen/Qwen3-Reranker-8B"
    assert asyncio.run(provider.aclose()) is None


# --------------------------------------------------------------------------------------
# 排除原因与最终选择结果
# --------------------------------------------------------------------------------------


def test_exclusion_reasons_are_the_fixed_seven():
    assert [reason.value for reason in ExclusionReason] == [
        "below_threshold",
        "duplicate",
        "covered_by_bundle",
        "outside_candidate_window",
        "budget_exceeded",
        "scope_mismatch",
        "version_mismatch",
    ]


@pytest.mark.parametrize("reason", list(ExclusionReason))
def test_exclusion_record_accepts_every_fixed_reason(reason):
    record = ExclusionRecord(candidate_id="cand-1", reason=reason, stage="select")
    assert record.reason is reason
    assert record.to_dict()["reason"] == reason.value


def test_exclusion_record_rejects_unknown_reason():
    with pytest.raises(RerankValidationError):
        ExclusionRecord(candidate_id="cand-1", reason="lexical_reserve")
    with pytest.raises(RerankValidationError):
        ExclusionRecord(candidate_id="", reason=ExclusionReason.DUPLICATE)


def test_selection_status_covers_the_four_states():
    assert [status.value for status in SelectionStatus] == [
        "selected",
        "empty",
        "failed",
        "cancelled",
    ]


def test_selection_result_accepts_selected_empty_failed_and_cancelled():
    selected = SelectionResult(
        status=SelectionStatus.SELECTED,
        profile_id="prof-1",
        selected=(StubView("cand-1", TEXT),),
        exclusions=(ExclusionRecord(candidate_id="cand-2", reason=ExclusionReason.BELOW_THRESHOLD),),
        candidate_count=2,
    )
    assert selected.evidence_char_count == len(TEXT)
    assert selected.to_dict()["error_code"] is None

    empty = SelectionResult(
        status=SelectionStatus.EMPTY,
        profile_id="prof-1",
        candidate_count=3,
        exclusions=(ExclusionRecord(candidate_id="cand-2", reason=ExclusionReason.DUPLICATE),),
    )
    assert empty.selected == ()

    failed = SelectionResult(
        status=SelectionStatus.FAILED,
        profile_id="prof-1",
        error=RerankTimeoutError("读取超时"),
        candidate_count=3,
    )
    assert failed.to_dict()["error_code"] == "RERANK_TIMEOUT"

    cancelled = SelectionResult(status=SelectionStatus.CANCELLED, profile_id="prof-1")
    assert cancelled.to_dict()["status"] == "cancelled"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": SelectionStatus.SELECTED, "selected": ()},
        {"status": SelectionStatus.EMPTY, "selected": (StubView("cand-1", TEXT),)},
        {"status": SelectionStatus.FAILED},
        {"status": SelectionStatus.FAILED, "error": RerankTimeoutError(), "selected": (StubView("c", "t"),)},
        {"status": SelectionStatus.CANCELLED, "error": RerankTimeoutError()},
        {"status": "unknown"},
        {"status": SelectionStatus.SELECTED, "selected": (StubView("cand-1", TEXT), RerankCandidate(
            candidate_id="cand-1", scope_id="global", index_revision="rev-1",
            ranking_text=TEXT, payload_hash=sha256_hex(TEXT),
        ))},
    ],
)
def test_selection_result_rejects_inconsistent_states(kwargs):
    with pytest.raises(RerankValidationError):
        SelectionResult(profile_id="prof-1", candidate_count=2, **kwargs)


def test_selection_result_enforces_source_cap_and_character_budget():
    too_many = tuple(StubView(f"cand-{index}", TEXT) for index in range(MAX_SELECTED_SOURCES + 1))
    with pytest.raises(RerankValidationError, match="超过上限"):
        SelectionResult(
            status=SelectionStatus.SELECTED,
            profile_id="prof-1",
            selected=too_many,
            candidate_count=len(too_many),
        )
    oversized = StubView("cand-1", "字" * (EVIDENCE_CHAR_BUDGET + 1))
    with pytest.raises(RerankValidationError, match="预算"):
        SelectionResult(
            status=SelectionStatus.SELECTED,
            profile_id="prof-1",
            selected=(oversized,),
            candidate_count=1,
        )


def test_selection_result_rejects_duplicate_exclusion_records():
    with pytest.raises(RerankValidationError, match="排除记录"):
        SelectionResult(
            status=SelectionStatus.EMPTY,
            profile_id="prof-1",
            candidate_count=2,
            exclusions=(
                ExclusionRecord(candidate_id="cand-1", reason=ExclusionReason.DUPLICATE),
                ExclusionRecord(candidate_id="cand-1", reason=ExclusionReason.BUDGET_EXCEEDED),
            ),
        )


# --------------------------------------------------------------------------------------
# 校准档案
# --------------------------------------------------------------------------------------


def test_profile_accepts_calibrated_artifact():
    profile = make_profile(metrics={"returned_precision": 0.93}, notes="T30 校准")
    assert profile.production_ready is True
    assert profile.compatibility_key()[1] == "Qwen/Qwen3-Reranker-8B"
    assert profile.to_dict()["status"] == "validated"


@pytest.mark.parametrize(
    "bad",
    [
        {"profile_id": ""},
        {"status": "released"},
        {"provider_type": ""},
        {"model": ""},
        {"endpoint_fingerprint": ""},
        {"instruction_id": ""},
        {"instruction_hash": "deadbeef"},
        {"tokenizer_id": ""},
        {"view_version": ""},
        {"scoring_protocol_version": ""},
        {"score_threshold": float("nan")},
        {"score_threshold": 1.5},
        {"max_candidates": 42},
        {"max_candidates": 0},
        {"min_candidate_count": 0},
        {"min_candidate_count": 51},
        {"max_selected": 0},
        {"max_selected": MAX_SELECTED_SOURCES + 1},
        {"evidence_char_budget": 0},
        {"max_document_tokens": 0},
        {"corpus_manifest_hash": "abc"},
        {"generated_at": "2026-09-19"},
        {"needs_validation": "yes"},
        {"metrics": {"precision": float("inf")}},
    ],
)
def test_profile_rejects_invalid_fields(bad):
    with pytest.raises(RerankValidationError):
        make_profile(**bad)


def test_profile_candidate_window_values_are_the_calibrated_choices():
    for window in CANDIDATE_WINDOW_CHOICES:
        assert make_profile(max_candidates=window).max_candidates == window


def test_profile_production_gate_requires_validated_status():
    with pytest.raises(RerankProfileMismatchError, match="未通过验收"):
        make_profile(status=ProfileStatus.CANDIDATE).assert_production_ready()
    with pytest.raises(RerankProfileMismatchError, match="未通过验收"):
        make_profile(status=ProfileStatus.QUALITY_PASSED).assert_production_ready()
    assert make_profile(status=ProfileStatus.VALIDATED).production_ready is True
    make_profile(status=ProfileStatus.VALIDATED).assert_production_ready()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_type", "dashscope_qwen3"),
        ("model", "Qwen/Qwen3-Reranker-4B"),
        ("endpoint_fingerprint", "https://example.invalid|https"),
        ("instruction_hash", sha256_hex("other instruction")),
        ("tokenizer_id", "BAAI/bge-reranker-v2-m3@transformers"),
        ("view_version", "evidence-view-v2"),
        ("scoring_protocol_version", "rerank-protocol-v2"),
    ],
)
def test_profile_assert_compatible_detects_every_identity_change(field, value):
    profile = make_profile()
    profile.assert_compatible(**{field: getattr(profile, field)})
    with pytest.raises(RerankProfileMismatchError) as excinfo:
        profile.assert_compatible(**{field: value})
    assert field in str(excinfo.value)


def test_profile_assert_compatible_ignores_unspecified_fields():
    profile = make_profile()
    profile.assert_compatible()
    profile.assert_compatible(model=profile.model, instruction_hash=None)


def test_profile_projection_contains_no_credentials():
    payload = json.dumps(make_profile().to_dict(), ensure_ascii=False).lower()
    for forbidden in ("api_key", "apikey", "secret", "password", "authorization"):
        assert forbidden not in payload


def test_sha256_helper_is_stable_and_separates_parts():
    digest = sha256_hex("a", "b")
    assert digest == sha256_hex("a", "b")
    assert digest != sha256_hex("ab", "")
    assert reranker._SHA256_HEX_RE.match(digest)
    with pytest.raises(RerankValidationError):
        sha256_hex("a", 1)  # type: ignore[arg-type]
