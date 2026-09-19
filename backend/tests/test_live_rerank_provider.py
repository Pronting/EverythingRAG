"""T06 单测：真实 HTTP 重排 Provider 与完整响应校验（PRD §6.1/§6.2/§6.3/§8.4）。

覆盖三块：
1. 真实服务响应副本的解析（``tests/fixtures/rerank/siliconflow_rerank_response.json``，
   2026-09-19 真实调用存档）：index ↔ 候选逐位对应、排序分数、``id``/trace/usage 提取。
2. 传输层管线：出网审计、HTTP 状态映射、超时、连接失败、异步取消。
3. 非法响应与非法配置的拒绝：缺项、数量不符、重复/越界 index、非有限分数、非 JSON、
   不支持的 instruction/chunk 选项、超长输入在调用前失败。

说明（PRD §10.3 与任务清单 §1.5）：本文件通过 ``httpx.MockTransport`` 与真实响应副本只验证
协议解析和错误处理，不产生任何模型质量结论，也不替代 T30–T32 的真实服务验收。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from app.core.outbound import OutboundClient
from app.retrieval.rerank_provider import (
    CONNECTION_TEST_DOCUMENTS,
    DEFAULT_ENDPOINT,
    SiliconFlowRerankProvider,
    capabilities_for_model,
    model_supports_chunk_options,
    model_supports_instruction,
    validate_endpoint,
)
from app.retrieval.reranker import (
    RerankAuthFailedError,
    RerankCandidate,
    RerankErrorCode,
    RerankInputTooLongError,
    RerankInvalidResponseError,
    RerankNotConfiguredError,
    RerankProfileMismatchError,
    RerankRateLimitedError,
    RerankRequest,
    RerankResult,
    RerankServiceUnavailableError,
    RerankTimeoutError,
    RerankValidationError,
    ScoreType,
    sha256_hex,
)

MODEL = "Qwen/Qwen3-Reranker-8B"
API_KEY = "test-key-not-a-real-credential"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rerank" / "siliconflow_rerank_response.json"
TRACE_HEADER = "x-siliconcloud-trace-id"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


FIXTURE = load_fixture()
REAL_BODY = FIXTURE["response_body"]
REAL_HEADERS = FIXTURE["response_headers"]
REAL_REQUEST = FIXTURE["request"]


def make_request(
    documents: tuple[str, ...] | None = None,
    *,
    query: str | None = None,
    instruction_text: str = "",
) -> RerankRequest:
    texts = documents if documents is not None else tuple(REAL_REQUEST["documents"])
    return RerankRequest(
        query=query or REAL_REQUEST["query"],
        scope_id="global",
        index_revision="rev-1",
        documents=tuple(
            RerankCandidate(
                candidate_id=f"cand-{index}",
                scope_id="global",
                index_revision="rev-1",
                ranking_text=text,
                payload_hash=sha256_hex(text),
            )
            for index, text in enumerate(texts)
        ),
        profile_id="prof-1",
        instruction_id="rerank-task-v1",
        instruction_text=instruction_text,
    )


def json_handler(
    body: object,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
):
    """构造返回给定 JSON 的同步 handler（body 为字符串时原样返回，用于 NaN 等原始文本）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        return httpx.Response(
            status,
            headers={"content-type": "application/json; charset=utf-8", **(headers or {})},
            content=text.encode("utf-8"),
        )

    return handler


def _capturing(handler, captured: list[httpx.Request]):
    """包一层请求捕获（同步与异步 handler 都支持）。"""

    if inspect.iscoroutinefunction(handler):

        async def async_wrapper(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return await handler(request)

        return async_wrapper

    def sync_wrapper(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    return sync_wrapper


_ISOLATED = object()


def make_provider(
    handler,
    *,
    model: str = MODEL,
    outbound: OutboundClient | None | object = _ISOLATED,
    captured: list[httpx.Request] | None = None,
    **kwargs,
) -> tuple[SiliconFlowRerankProvider, httpx.AsyncClient]:
    """构造 Provider。

    ``outbound`` 缺省为隔离的审计记录器（避免测试互相污染）；传 ``None`` 表示明确关闭审计；
    传具体实例则使用该实例。全局单例路径由 ``test_default_outbound_client_is_used`` 直接覆盖。
    """
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            _capturing(handler, captured) if captured is not None else handler
        )
    )
    audit: object = _ISOLATED if outbound is _ISOLATED else outbound
    provider = SiliconFlowRerankProvider(
        model=model,
        api_key=API_KEY,
        client=client,
        outbound=OutboundClient() if audit is _ISOLATED else audit,  # type: ignore[arg-type]
        **kwargs,
    )
    return provider, client


# --------------------------------------------------------------------------------------
# 真实响应副本：解析、映射、计量
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_concurrency_is_capped_at_two_requests_across_providers():
    """PRD §9：同时执行的重排请求至多 2 个，且闸门跨 Provider 实例生效（进程级）。"""

    from app.retrieval.rerank_provider import MAX_CONCURRENT_REQUESTS, request_gate

    assert MAX_CONCURRENT_REQUESTS == 2
    live = 0
    peak = 0
    finished = 0
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal live, peak, finished
        live += 1
        peak = max(peak, live)
        try:
            # 让出事件循环：闸门若失效，其余任务会同时进入并把 peak 推高
            for _ in range(5):
                await asyncio.sleep(0)
            return json_handler(REAL_BODY)(request)
        finally:
            live -= 1
            finished += 1
            if finished >= MAX_CONCURRENT_REQUESTS:
                release.set()

    providers = [make_provider(handler) for _ in range(4)]
    assert providers[0][0] is not providers[1][0]  # 每个实例独立，闸门仍是进程级
    try:
        tasks = [asyncio.create_task(provider.rerank(make_request())) for provider, _client in providers]
        # 先放行两个：此时第三个必须还在闸门外排队
        await asyncio.wait_for(release.wait(), timeout=5)
        await asyncio.sleep(0)
        assert live <= MAX_CONCURRENT_REQUESTS
        results = await asyncio.gather(*tasks)
    finally:
        for provider, client in providers:
            await provider.aclose()
            await client.aclose()

    assert len(results) == 4
    assert peak == MAX_CONCURRENT_REQUESTS  # 恰好用满 2，且从未超过
    assert request_gate()._value >= 0  # 全部释放，没有泄漏信号量


@pytest.mark.asyncio
async def test_cancelled_waiter_never_sends_its_request():
    """取消等待中的任务：请求不发出（PRD §9「取消不继续外呼」）。"""

    started = asyncio.Event()
    release = asyncio.Event()
    sent: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("x-task", ""))
        started.set()
        await release.wait()
        return json_handler(REAL_BODY)(request)

    blocker_provider, blocker_client = make_provider(handler)
    queued_provider, queued_client = make_provider(handler)
    try:
        blocker = asyncio.create_task(blocker_provider.rerank(make_request()))
        await asyncio.wait_for(started.wait(), timeout=5)
        # 闸门已被占满 2 个中的一个，还需要第二个占位者才能让 queued 排队
        second_provider, second_client = make_provider(handler)
        second = asyncio.create_task(second_provider.rerank(make_request()))
        await asyncio.sleep(0)
        queued = asyncio.create_task(queued_provider.rerank(make_request()))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await asyncio.gather(blocker, second)
    finally:
        release.set()
        for provider, client in (
            (blocker_provider, blocker_client),
            (queued_provider, queued_client),
            (second_provider, second_client),
        ):
            await provider.aclose()
            await client.aclose()

    assert len(sent) == 2  # 被取消的第三个请求从未发出


async def test_real_response_copy_parses_scores_mapping_and_metering():
    outbound = OutboundClient()
    provider, client = make_provider(json_handler(REAL_BODY, headers=REAL_HEADERS), outbound=outbound)
    try:
        result = await provider.rerank(make_request())
    finally:
        await client.aclose()

    assert isinstance(result, RerankResult)
    assert result.provider_type == "siliconflow_rerank"
    assert result.model == MODEL
    assert result.document_count == 3
    assert [score.candidate_id for score in result.scores] == ["cand-0", "cand-1", "cand-2"]
    assert [score.index for score in result.scores] == [0, 1, 2]  # 响应按分数降序，本地按 index 归档
    assert result.score_for("cand-0") == pytest.approx(0.9701576828956604)
    assert result.score_for("cand-1") == pytest.approx(1.2911860949316178e-06)
    assert result.score_for("cand-2") == pytest.approx(0.00021110977104399353)
    assert all(score.score_type is ScoreType.RERANK for score in result.scores)
    assert result.request_id == REAL_BODY["id"]
    assert result.trace_id == REAL_HEADERS[TRACE_HEADER]
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (347, 0)
    assert result.usage.billing_breakdown["search_units"] == 0.0
    assert result.usage_reported is True
    assert result.latency_ms is not None and math.isfinite(result.latency_ms)


async def test_real_response_order_is_mapped_by_index_not_position():
    """服务按分数降序返回，index 必须回到原候选位置（PRD §6.2）。"""
    body = {
        "id": "req-1",
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ],
        "meta": {},
    }
    provider, client = make_provider(json_handler(body))
    try:
        result = await provider.rerank(make_request(documents=("短文甲", "短文乙")))
    finally:
        await client.aclose()
    by_id = {score.candidate_id: score.score for score in result.scores}
    assert by_id == {"cand-0": 0.1, "cand-1": 0.9}


async def test_request_payload_is_complete_utf8_and_top_n_matches_candidates():
    captured: list[httpx.Request] = []
    provider, client = make_provider(
        json_handler(REAL_BODY, headers=REAL_HEADERS), captured=captured
    )
    try:
        await provider.rerank(make_request(instruction_text="任务指令：判断是否回答问题"))
    finally:
        await client.aclose()

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == DEFAULT_ENDPOINT
    assert request.headers["authorization"].startswith("Bearer ")
    assert "charset=utf-8" in request.headers["content-type"]
    # 中文必须按 UTF-8 原样发送（PowerShell 直发会打成 ?，因此断言原始字节可复原）
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["query"] == REAL_REQUEST["query"]
    assert payload["documents"] == REAL_REQUEST["documents"]
    assert payload["top_n"] == len(payload["documents"]) == 3
    assert payload["model"] == MODEL
    assert payload["instruction"] == "任务指令：判断是否回答问题"
    assert "return_documents" not in payload  # 默认不要求回传正文（本地已有 evidence_text）
    assert str(request.content, "utf-8").find("增量同步") > 0


async def test_outbound_audit_records_only_metadata():
    outbound = OutboundClient()
    provider, client = make_provider(
        json_handler(REAL_BODY, headers=REAL_HEADERS), outbound=outbound
    )
    try:
        await provider.rerank(make_request())
    finally:
        await client.aclose()

    entries = outbound.entries()
    assert len(entries) == 1
    event = entries[0]
    assert event.provider == "rerank"
    assert event.destination == "api.siliconflow.cn"  # 只有 host，无路径/query/key
    assert event.method == "rerank"
    assert event.status == 200
    assert isinstance(event.started_at, datetime) and event.started_at.tzinfo is not None
    dumped = repr(event.__dict__)
    assert API_KEY not in dumped
    assert REAL_REQUEST["query"] not in dumped
    assert REAL_REQUEST["documents"][0] not in dumped


async def test_default_outbound_client_is_used_when_not_injected():
    """生产缺省接全局单例：真实出网必须进入网络审计，不能静默失效。"""
    from app.core.outbound import outbound_client

    outbound_client.clear()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(json_handler(REAL_BODY, headers=REAL_HEADERS))
    )
    provider = SiliconFlowRerankProvider(model=MODEL, api_key=API_KEY, client=client)
    try:
        await provider.rerank(make_request())
        assert len(outbound_client.entries()) == 1
    finally:
        await client.aclose()
        outbound_client.clear()


async def test_audit_can_be_disabled_explicitly_for_offline_use():
    provider, client = make_provider(json_handler(REAL_BODY), outbound=None)
    try:
        await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert provider._outbound is None


# --------------------------------------------------------------------------------------
# 非法响应：立即失败，绝不静默补零
# --------------------------------------------------------------------------------------


def _body_with(results: object, *, meta: object | None = None) -> dict:
    return {"id": "req-1", "results": results, "meta": meta if meta is not None else {}}


def _score_items(*pairs: tuple[int, float]) -> list[dict]:
    return [{"index": index, "relevance_score": score} for index, score in pairs]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"id": "req-1", "meta": {}}, "缺少 results"),
        ({"id": "req-1", "results": {"index": 0}, "meta": {}}, "results 不是数组"),
        (_body_with(_score_items((0, 0.9))), "数量少于候选数"),
        (_body_with(_score_items((0, 0.9), (1, 0.2), (2, 0.1), (3, 0.05))), "数量多于候选数"),
        (_body_with(_score_items((0, 0.9), (0, 0.2), (1, 0.1))), "重复 index"),
        (_body_with(_score_items((0, 0.9), (1, 0.2), (7, 0.1))), "越界 index"),
        (_body_with(_score_items((0, 0.9), ("1", 0.2), (2, 0.1))), "index 非整数"),
        (_body_with(_score_items((0, 0.9), (1.5, 0.2), (2, 0.1))), "index 为浮点"),
        (_body_with(_score_items((0, 0.9), (True, 0.2), (2, 0.1))), "index 为布尔"),
        (_body_with(_score_items((0, 0.9), (1, "high"), (2, 0.1))), "分数非数值"),
        (_body_with(_score_items((0, 0.9), (1, None), (2, 0.1))), "分数缺失"),
        (_body_with(_score_items((0, 0.9), (1, 1.4), (2, 0.1))), "分数越界"),
        (_body_with([{"index": 0, "relevance_score": 0.9}, "x", {"index": 2, "relevance_score": 0.1}]), "项不是对象"),
    ],
)
async def test_invalid_responses_are_rejected(body, reason):
    provider, client = make_provider(json_handler(body))
    try:
        with pytest.raises(RerankInvalidResponseError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("{not json", "非 JSON"),
        ("[]", "顶层不是对象"),
        ('{"results": [{"index": 0, "relevance_score": NaN}, {"index": 1, "relevance_score": 0.1}]}', "NaN 分数"),
        ('{"results": [{"index": 0, "relevance_score": Infinity}, {"index": 1, "relevance_score": 0.1}]}', "Infinity 分数"),
    ],
)
async def test_invalid_raw_bodies_are_rejected(raw, reason):
    provider, client = make_provider(json_handler(raw))
    try:
        with pytest.raises(RerankInvalidResponseError):
            await provider.rerank(make_request(documents=("甲", "乙")))
    finally:
        await client.aclose()


async def test_result_failure_never_returns_success_with_empty_scores():
    provider, client = make_provider(json_handler({"results": []}))
    try:
        with pytest.raises(RerankInvalidResponseError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()


# --------------------------------------------------------------------------------------
# HTTP 状态、超时、连接失败、取消
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_type"),
    [
        (401, RerankAuthFailedError),
        (403, RerankAuthFailedError),
        (429, RerankRateLimitedError),
        (413, RerankInputTooLongError),
        (503, RerankServiceUnavailableError),
        (504, RerankServiceUnavailableError),
        (404, RerankProfileMismatchError),
        (400, RerankValidationError),
        (422, RerankValidationError),
        (500, RerankInvalidResponseError),
        (418, RerankInvalidResponseError),
    ],
)
async def test_http_status_maps_to_explicit_error_type(status, expected_type):
    outbound = OutboundClient()
    body = {"code": 50505, "message": "Model service overloaded, secret provider body"}
    provider, client = make_provider(
        json_handler(body, status=status), outbound=outbound
    )
    try:
        with pytest.raises(expected_type) as excinfo:
            await provider.rerank(make_request())
    finally:
        await client.aclose()
    error = excinfo.value
    assert error.http_status == status
    # 服务错误体原文不得进入错误详情或用户事件
    assert "Model service overloaded" not in str(error)
    assert "secret provider body" not in json.dumps(error.to_event(), ensure_ascii=False)
    assert outbound.entries()[0].status == status


async def test_plain_text_error_body_is_not_leaked():
    provider, client = make_provider(json_handler("Invalid token: sk-secret-value", status=401))
    try:
        with pytest.raises(RerankAuthFailedError) as excinfo:
            await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert "sk-secret-value" not in str(excinfo.value)
    assert excinfo.value.code is RerankErrorCode.AUTH_FAILED


async def test_timeout_maps_to_timeout_error_and_is_audited():
    outbound = OutboundClient()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    provider, client = make_provider(handler, outbound=outbound)
    try:
        with pytest.raises(RerankTimeoutError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert outbound.entries()[0].status is None


async def test_connection_failure_maps_to_service_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, client = make_provider(handler)
    try:
        with pytest.raises(RerankServiceUnavailableError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()


async def test_cancellation_propagates_and_is_audited():
    outbound = OutboundClient()

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json=REAL_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SiliconFlowRerankProvider(
        model=MODEL, api_key=API_KEY, client=client, outbound=outbound
    )
    task = asyncio.create_task(provider.rerank(make_request()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    entries = outbound.entries()
    assert len(entries) == 1 and entries[0].status is None


# --------------------------------------------------------------------------------------
# 配置与能力门控（不支持的字段在调用前显式拒绝）
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", ["", "   ", None])
def test_missing_credentials_raise_not_configured(bad_key):
    with pytest.raises(RerankNotConfiguredError):
        SiliconFlowRerankProvider(model=MODEL, api_key=bad_key, client=httpx.AsyncClient())


def test_missing_model_raises_not_configured():
    with pytest.raises(RerankNotConfiguredError):
        SiliconFlowRerankProvider(model="", api_key=API_KEY, client=httpx.AsyncClient())


@pytest.mark.parametrize(
    ("endpoint", "ok"),
    [
        ("https://api.siliconflow.cn/v1/rerank", True),
        ("http://127.0.0.1:9000/v1/rerank", True),
        ("http://localhost:9000/v1/rerank", True),
        ("http://api.siliconflow.cn/v1/rerank", False),
        ("ftp://api.siliconflow.cn/rerank", False),
        ("api.siliconflow.cn/v1/rerank", False),
        ("", False),
    ],
)
def test_endpoint_validation(endpoint, ok):
    if ok:
        assert validate_endpoint(endpoint) == endpoint
    else:
        with pytest.raises(RerankNotConfiguredError):
            validate_endpoint(endpoint)


@pytest.mark.parametrize(
    ("model", "instruction", "chunk_options"),
    [
        ("Qwen/Qwen3-Reranker-8B", True, False),
        ("Qwen/Qwen3-Reranker-4B", True, False),
        ("BAAI/bge-reranker-v2-m3", False, True),
        ("netease-youdao/bce-reranker-base_v1", False, True),
        ("some/unknown-reranker", False, False),
    ],
)
def test_capabilities_table_matches_measured_model_families(model, instruction, chunk_options):
    assert model_supports_instruction(model) is instruction
    assert model_supports_chunk_options(model) is chunk_options
    capabilities = capabilities_for_model(model)
    assert capabilities.supports_instruction is instruction
    assert capabilities.supports_chunk_options is chunk_options
    # 未实测的限制必须保持未返回状态，不得用其他厂商的值填充
    assert capabilities.max_documents is None
    assert capabilities.max_document_tokens is None
    assert capabilities.max_request_tokens is None


async def test_instruction_rejected_for_model_without_support_before_http():
    captured: list[httpx.Request] = []
    provider, client = make_provider(
        json_handler(REAL_BODY), model="BAAI/bge-reranker-v2-m3", captured=captured
    )
    try:
        with pytest.raises(RerankProfileMismatchError, match="instruction"):
            await provider.rerank(make_request(instruction_text="任务指令"))
    finally:
        await client.aclose()
    assert captured == []


async def test_instruction_conflict_between_profile_and_request_is_rejected():
    provider, client = make_provider(
        json_handler(REAL_BODY), instruction_text="档案固定指令"
    )
    try:
        with pytest.raises(RerankProfileMismatchError):
            await provider.rerank(make_request(instruction_text="另一段指令"))
    finally:
        await client.aclose()


async def test_chunk_options_gated_by_model_support():
    with pytest.raises(RerankProfileMismatchError, match="max_chunks_per_doc"):
        make_provider(json_handler(REAL_BODY), max_chunks_per_doc=2)

    captured: list[httpx.Request] = []
    provider, client = make_provider(
        json_handler(REAL_BODY),
        model="BAAI/bge-reranker-v2-m3",
        captured=captured,
        max_chunks_per_doc=2,
        overlap_tokens=64,
    )
    try:
        await provider.rerank(make_request())
    finally:
        await client.aclose()
    payload = json.loads(captured[0].content.decode("utf-8"))
    assert payload["max_chunks_per_doc"] == 2
    assert payload["overlap_tokens"] == 64


def test_capabilities_model_must_match_configured_model():
    capabilities = capabilities_for_model("Qwen/Qwen3-Reranker-4B")
    with pytest.raises(RerankProfileMismatchError):
        SiliconFlowRerankProvider(
            model=MODEL, api_key=API_KEY, capabilities=capabilities, client=httpx.AsyncClient()
        )


def test_declared_token_limits_require_a_tokenizer_counter():
    capabilities = capabilities_for_model(MODEL, max_document_tokens=100)
    with pytest.raises(RerankProfileMismatchError, match="tokenizer"):
        SiliconFlowRerankProvider(
            model=MODEL, api_key=API_KEY, capabilities=capabilities, client=httpx.AsyncClient()
        )


async def test_oversized_input_fails_before_http_call():
    captured: list[httpx.Request] = []
    capabilities = capabilities_for_model(MODEL, max_query_tokens=4, max_document_tokens=8)
    provider, client = make_provider(
        json_handler(REAL_BODY),
        captured=captured,
        capabilities=capabilities,
        token_counter=len,  # 测试替身：仅验证长度门控管线，不代表真实 tokenizer
    )
    try:
        with pytest.raises(RerankInputTooLongError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert captured == []


async def test_length_check_passes_within_declared_limits():
    capabilities = capabilities_for_model(
        MODEL, max_query_tokens=1000, max_document_tokens=1000, max_request_tokens=5000
    )
    provider, client = make_provider(
        json_handler(REAL_BODY, headers=REAL_HEADERS),
        capabilities=capabilities,
        token_counter=len,
    )
    try:
        result = await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert result.document_count == 3


async def test_document_count_over_service_limit_rejected_before_http():
    captured: list[httpx.Request] = []
    capabilities = capabilities_for_model(MODEL, max_documents=2)
    provider, client = make_provider(
        json_handler(REAL_BODY), captured=captured, capabilities=capabilities
    )
    try:
        with pytest.raises(RerankInputTooLongError):
            await provider.rerank(make_request())
    finally:
        await client.aclose()
    assert captured == []


# --------------------------------------------------------------------------------------
# 连接测试与生命周期
# --------------------------------------------------------------------------------------


async def test_check_connection_reports_connectivity_without_quality_claim():
    body = _body_with(_score_items((0, 0.9), (1, 0.1)), meta={"tokens": {"input_tokens": 12}})
    provider, client = make_provider(json_handler(body, headers={TRACE_HEADER: "ti_test"}))
    try:
        status = await provider.check_connection()
    finally:
        await client.aclose()
    assert status["success"] is True
    assert status["model"] == MODEL
    assert status["endpoint_host"] == "api.siliconflow.cn"
    assert status["scored_count"] == len(CONNECTION_TEST_DOCUMENTS)
    assert status["quality_checked"] is False
    assert status["trace_id"] == "ti_test"


async def test_check_connection_reports_error_code_without_provider_body():
    provider, client = make_provider(json_handler({"message": "TPM limit reached"}, status=429))
    try:
        status = await provider.check_connection()
    finally:
        await client.aclose()
    assert status["success"] is False
    assert status["code"] == "RERANK_RATE_LIMITED"
    assert "TPM limit reached" not in json.dumps(status, ensure_ascii=False)
    assert status["quality_checked"] is False


async def test_aclose_only_closes_self_created_client(monkeypatch):
    injected = httpx.AsyncClient(transport=httpx.MockTransport(json_handler(REAL_BODY)))
    provider = SiliconFlowRerankProvider(model=MODEL, api_key=API_KEY, client=injected)
    assert provider._owns_client is False
    await provider.aclose()
    assert injected.is_closed is False
    await injected.aclose()

    import app.retrieval.rerank_provider as provider_module

    real_client_cls = httpx.AsyncClient
    created: list[httpx.AsyncClient] = []

    def fake_client(**kwargs) -> httpx.AsyncClient:
        client = real_client_cls(transport=httpx.MockTransport(json_handler(REAL_BODY)))
        created.append(client)
        return client

    monkeypatch.setattr(provider_module.httpx, "AsyncClient", fake_client)
    owned = SiliconFlowRerankProvider(model=MODEL, api_key=API_KEY, outbound=None)
    assert owned._owns_client is True
    assert len(created) == 1
    await owned.aclose()
    assert created[0].is_closed is True
