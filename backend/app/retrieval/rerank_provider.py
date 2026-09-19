"""真实 HTTP 重排 Provider（SiliconFlow ``/v1/rerank``）+ 完整响应校验。

对齐 PRD RAG-REL-001 §6.1/§6.2/§6.3 与 T06：

- 出网只经现有 ``httpx``（隐式重试关闭）与唯一收敛点 ``outbound_client``；审计只记
  provider / host / 方法 / 状态 / 耗时，绝不含问题、正文、绝对路径、密钥或响应体。
- 一次请求提交完整候选集合，``top_n`` 等于候选数，要求返回每个候选的分数。
- 响应校验：``results`` 为数组、数量与候选数一致、``index`` 为整数且在范围内、无重复、
  ``relevance_score`` 为有限数值并落在 Provider 声明范围内；任何缺失 / NaN / 越界 /
  结构错误立即失败，绝不静默补零或截断。
- ``usage`` 取 ``meta.tokens`` 与 ``meta.billed_units``；服务未返回时保持未返回状态，
  不估算费用（PRD §9）。
- 连接超时 5 秒、读取超时 60 秒（PRD §6.2）；超时、认证失败、限流、服务不可用、协议
  异常分别映射到 PRD §8.4 的类型明确错误，不重试、不换模型、不当成「无相关资料」。
- 长度限制按 Provider 实测值（``None`` = 未实测），token 数只能由注入的成熟 tokenizer
  计数，禁止用字符数替代；超限抛 ``RERANK_INPUT_TOO_LONG``。

Provider 中立部分留在 ``retrieval/reranker.py``；本模块只实现 SiliconFlow 协议细节，
新增厂家只需再写一个 ``RerankProvider`` 子类，不影响 ``relevance_selector``。
"""

from __future__ import annotations

import asyncio
import json
import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from app.core.outbound import OutboundClient, OutboundEvent, outbound_client
from app.retrieval.reranker import (
    RerankCandidate,
    RerankError,
    RerankInvalidResponseError,
    RerankNotConfiguredError,
    RerankProfileMismatchError,
    RerankProvider,
    RerankProviderCapabilities,
    RerankProviderType,
    RerankRequest,
    RerankResult,
    RerankScore,
    RerankServiceUnavailableError,
    RerankTimeoutError,
    RerankUsage,
    ScoreScope,
    assert_within_token_limits,
    error_for_http_status,
    sha256_hex,
)

__all__ = [
    "CONNECTION_TEST_DOCUMENTS",
    "CONNECTION_TEST_QUERY",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL_PREFIX_HINTS",
    "MAX_CONCURRENT_REQUESTS",
    "SiliconFlowRerankProvider",
    "SiliconFlowUsage",
    "capabilities_for_model",
    "model_supports_chunk_options",
    "model_supports_instruction",
    "request_gate",
    "validate_endpoint",
]

#: SiliconFlow 重排完整 endpoint（PRD §6.5：endpoint 由用户在设置中确认）。
DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1/rerank"
PROVIDER_TYPE = RerankProviderType.SILICONFLOW_RERANK

#: PRD §9：同时执行的 HTTP 请求上限为 2。闸门必须是**进程级**的——同一进程里可以同时存在
#: 多个 Provider 实例（每个设置快照、每次连接测试各一个），挂在实例上等于没有上限。
MAX_CONCURRENT_REQUESTS = 2

#: ``asyncio.Semaphore`` 绑定事件循环，因此按运行中的循环惰性创建：线上单循环 ⇒ 全局上限 2；
#: 测试里的多个事件循环也不会互相污染（跨循环复用同一个 Semaphore 会直接报错）。
_REQUEST_GATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def request_gate() -> asyncio.Semaphore:
    """当前事件循环的重排请求闸门（至多 :data:`MAX_CONCURRENT_REQUESTS` 个同时外呼）。"""

    loop = asyncio.get_running_loop()
    gate = _REQUEST_GATES.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        _REQUEST_GATES[loop] = gate
    return gate

#: 连接测试使用的固定公开文本（PRD §8.1）：不含任何知识库内容。
CONNECTION_TEST_QUERY = "连接测试：判断资料是否回答了一个明确问题。"
CONNECTION_TEST_DOCUMENTS = (
    "固定公开文本：增量同步只重新处理发生变化的文档。",
    "固定公开文本：今天天气很好，适合出门散步。",
)

#: 只允许 HTTPS；本地测试服务只允许显式配置的回环 HTTP（PRD §6.5）。
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: 实测能力表（T11 探针，2026-09-19）：instruction 只对 Qwen3-Reranker 系列生效。
#: 服务对不支持的模型会静默忽略 instruction，因此必须由客户端显式拒绝。
_INSTRUCTION_MODEL_HINTS = ("qwen3-reranker",)
#: max_chunks_per_doc / overlap_tokens 只对 bge-reranker-v2-m3、Pro 版与 bce 系列生效。
_CHUNK_OPTION_MODEL_HINTS = ("bge-reranker-v2-m3", "bge-reranker-v2-m3-pro", "bce-reranker")
#: 上述两组提示是实测观测到的模型族；未知模型一律按「两项都不支持」保守处理。
DEFAULT_MODEL_PREFIX_HINTS = (_INSTRUCTION_MODEL_HINTS, _CHUNK_OPTION_MODEL_HINTS)


def validate_endpoint(endpoint: str) -> str:
    """校验 endpoint：必须是完整 HTTPS，或显式配置的回环 HTTP（PRD §6.5）。"""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RerankNotConfiguredError("筛选服务 endpoint 未配置")
    parts = urlsplit(endpoint.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise RerankNotConfiguredError("筛选服务 endpoint 必须是完整的 http(s) 地址")
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise RerankNotConfiguredError("筛选服务只允许 HTTPS；HTTP 仅限本机回环测试地址")
    return endpoint.strip()


def model_supports_instruction(model: str) -> bool:
    """模型是否支持 ``instruction``（实测：Qwen3-Reranker 系列）。"""
    lowered = (model or "").casefold()
    return any(hint in lowered for hint in _INSTRUCTION_MODEL_HINTS)


def model_supports_chunk_options(model: str) -> bool:
    """模型是否支持 ``max_chunks_per_doc`` / ``overlap_tokens``（实测：bge/bce 系列）。"""
    lowered = (model or "").casefold()
    return any(hint in lowered for hint in _CHUNK_OPTION_MODEL_HINTS)


def capabilities_for_model(
    model: str,
    *,
    tokenizer_id: str | None = None,
    provider_type: str = PROVIDER_TYPE,
    max_documents: int | None = None,
    max_query_tokens: int | None = None,
    max_document_tokens: int | None = None,
    max_request_tokens: int | None = None,
    score_min: float = 0.0,
    score_max: float = 1.0,
) -> RerankProviderCapabilities:
    """按模型族生成能力描述。

    长度与数量限制留空（``None``）表示尚未实测：不得用其他模型或其他厂商的数值填充；
    T11 实测后由调用方传入。tokenizer 身份必须由 T11 确认，未确认时按模型名占位并在
    ``notes`` 中显式标记。
    """
    resolved_tokenizer = tokenizer_id or model
    notes = "模型能力来自 2026-09-19 真实探针观测；长度/数量限制未实测时保持未返回状态"
    if tokenizer_id is None:
        notes = f"{notes}；tokenizer 身份待 T11 实测确认（当前占位：{model}）"
    return RerankProviderCapabilities(
        provider_type=str(provider_type),
        model=model,
        tokenizer_id=resolved_tokenizer,
        max_documents=max_documents,
        max_query_tokens=max_query_tokens,
        max_document_tokens=max_document_tokens,
        max_request_tokens=max_request_tokens,
        score_min=score_min,
        score_max=score_max,
        supports_instruction=model_supports_instruction(model),
        supports_chunk_options=model_supports_chunk_options(model),
        notes=notes,
    )


@dataclass(frozen=True)
class SiliconFlowUsage(RerankUsage):
    """SiliconFlow 计量：``meta.tokens`` 与 ``meta.billed_units`` 明细（原样保存，不做换算）。

    实测响应形如::

        "meta": {"tokens": {"input_tokens": 347, "output_tokens": 0, "image_tokens": 0},
                 "billed_units": {"input_tokens": 347, ..., "search_units": 0,
                                  "classifications": 0}}

    ``billed_units`` 是明细对象而不是标量，因此标量字段只在服务返回标量时填写；明细全部
    保留在 ``billing_breakdown``，不把明细求和当成费用。
    """

    billing_breakdown: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        breakdown = self.billing_breakdown
        if not isinstance(breakdown, Mapping):
            raise RerankInvalidResponseError("billing_breakdown 必须是映射")
        for key, value in breakdown.items():
            if not isinstance(key, str):
                raise RerankInvalidResponseError("billing_breakdown 键必须是字符串")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RerankInvalidResponseError("billing_breakdown 值必须是数值")
            if not math.isfinite(float(value)):
                raise RerankInvalidResponseError("billing_breakdown 值必须是有限数值")

    @property
    def reported(self) -> bool:
        return super().reported or bool(self.billing_breakdown)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["billing_breakdown"] = dict(self.billing_breakdown)
        return payload


class SiliconFlowRerankProvider(RerankProvider):
    """SiliconFlow ``POST /v1/rerank`` 异步 Provider（PRD §6.1/§6.2/§6.3）。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | SecretStr,
        endpoint: str = DEFAULT_ENDPOINT,
        instruction_text: str = "",
        capabilities: RerankProviderCapabilities | None = None,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 60.0,
        token_counter: Callable[[str], int] | None = None,
        return_documents: bool = False,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
        outbound: OutboundClient | None = outbound_client,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise RerankNotConfiguredError("筛选服务模型未配置")
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise RerankNotConfiguredError("筛选服务 API Key 未配置")
        self._model = model.strip()
        self._api_key = resolved_key
        self._endpoint = validate_endpoint(endpoint)
        self._destination = _destination_host(self._endpoint)
        self._instruction_text = instruction_text or ""
        self._capabilities = capabilities or capabilities_for_model(self._model)
        if self._capabilities.model != self._model:
            raise RerankProfileMismatchError(
                f"能力描述模型（{self._capabilities.model}）与配置模型（{self._model}）不一致"
            )
        if _has_token_limits(self._capabilities) and token_counter is None:
            raise RerankProfileMismatchError(
                "能力描述声明了 token 限制，但未提供 tokenizer 计数函数："
                "token 数不能用字符数替代"
            )
        self._token_counter = token_counter
        self._return_documents = bool(return_documents)
        self._max_chunks_per_doc = _optional_positive_int(max_chunks_per_doc, "max_chunks_per_doc")
        self._overlap_tokens = _optional_positive_int(overlap_tokens, "overlap_tokens")
        if (self._max_chunks_per_doc is not None or self._overlap_tokens is not None) and (
            not self._capabilities.supports_chunk_options
        ):
            self._capabilities.assert_chunk_options_supported()
        self._outbound = outbound
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=read_timeout_seconds,
                pool=connect_timeout_seconds,
            ),
            transport=httpx.AsyncHTTPTransport(retries=0),  # 隐式重试关闭（PRD §6.2）
        )

    # -- 身份与能力 ----------------------------------------------------------------

    @property
    def provider_type(self) -> str:
        return str(PROVIDER_TYPE)

    @property
    def capabilities(self) -> RerankProviderCapabilities:
        return self._capabilities

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint_host(self) -> str:
        """审计/展示用的 host（绝不含路径与查询）。"""
        return self._destination

    # -- 调用 ----------------------------------------------------------------------

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """提交完整候选集合并校验全部返回分数（PRD §6.2/§6.3）。"""
        instruction = self._resolve_instruction(request)
        self._assert_input_supported(request, instruction)
        payload = self._build_payload(request, instruction)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        started_at = datetime.now(UTC)
        timer = perf_counter()
        status: int | None = None
        try:
            response = await _post_gated(
                self._client,
                self._endpoint,
                headers=headers,
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
            status = response.status_code
            if status != httpx.codes.OK:
                raise error_for_http_status(status, _status_detail(status, response))
            result = self._parse_response(request, response, _elapsed_ms(timer))
        except RerankError:
            raise
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise RerankTimeoutError(f"{type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            # 连接被拒绝 / DNS 失败 / 连接中断：服务不可用，不是「无相关资料」
            raise RerankServiceUnavailableError(f"{type(exc).__name__}") from exc
        finally:
            self._record_outbound(started_at, status, timer)
        return result

    def _resolve_instruction(self, request: RerankRequest) -> str:
        """档案固定发送文本；请求与构造参数不一致说明档案失配。"""
        configured = self._instruction_text
        requested = request.instruction_text
        if configured and requested and configured != requested:
            raise RerankProfileMismatchError("请求 instruction 与档案固定文本不一致")
        return requested or configured

    def _assert_input_supported(self, request: RerankRequest, instruction: str) -> None:
        """调用前的能力与长度校验：不支持的字段显式拒绝，超限不静默截断。"""
        if instruction:
            self._capabilities.assert_instruction_supported()
        self._capabilities.assert_document_count(request.top_n_value)
        if self._token_counter is None:
            return
        document_tokens = [self._token_counter(document.ranking_text) for document in request.documents]
        assert_within_token_limits(
            self._capabilities,
            query_tokens=self._token_counter(request.query),
            document_tokens=document_tokens,
            request_tokens=sum(document_tokens),
        )

    def _build_payload(self, request: RerankRequest, instruction: str) -> dict[str, Any]:
        """构造请求体：``top_n`` 等于候选数，强制返回全部分数。"""
        payload: dict[str, Any] = {
            "model": self._model,
            "query": request.query,
            "documents": [document.ranking_text for document in request.documents],
            "top_n": request.top_n_value,
        }
        if self._return_documents:
            payload["return_documents"] = True
        if instruction:
            payload["instruction"] = instruction
        if self._max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = self._max_chunks_per_doc
        if self._overlap_tokens is not None:
            payload["overlap_tokens"] = self._overlap_tokens
        return payload

    def _parse_response(
        self, request: RerankRequest, response: httpx.Response, latency_ms: float
    ) -> RerankResult:
        """严格校验响应结构与分数；任何缺失/重复/越界/非有限值立即失败。"""
        try:
            body = response.json()
        except ValueError as exc:
            raise RerankInvalidResponseError("响应不是合法 JSON") from exc
        if not isinstance(body, dict):
            raise RerankInvalidResponseError("响应顶层必须是 JSON 对象")
        raw_results = body.get("results")
        if not isinstance(raw_results, list):
            raise RerankInvalidResponseError("响应缺少 results 数组")
        expected = request.top_n_value
        if len(raw_results) != expected:
            raise RerankInvalidResponseError(
                f"分数数量（{len(raw_results)}）与候选数（{expected}）不一致"
            )
        candidate_ids = request.candidate_ids
        scores: list[RerankScore] = []
        for position, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise RerankInvalidResponseError(f"results[{position}] 必须是对象")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankInvalidResponseError(f"results[{position}].index 必须是整数")
            if index < 0 or index >= expected:
                raise RerankInvalidResponseError(
                    f"results[{position}].index 越界（{index} >= {expected}）"
                )
            score = item.get("relevance_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RerankInvalidResponseError(
                    f"results[{position}].relevance_score 必须是有限数值"
                )
            scores.append(
                RerankScore(
                    candidate_id=candidate_ids[index],
                    index=index,
                    score=float(score),
                )
            )
        scores.sort(key=lambda item: item.index)
        result = RerankResult(
            profile_id=request.profile_id,
            provider_type=self.provider_type,
            model=self._model,
            document_count=expected,
            scores=tuple(scores),
            request_id=_optional_str(body.get("id")),
            trace_id=_optional_str(response.headers.get("x-siliconcloud-trace-id")),
            usage=_parse_usage(body.get("meta")),
            latency_ms=latency_ms,
            score_scope=ScoreScope.REQUEST_RELATIVE,
        )
        result.assert_matches_request(request)
        result.assert_scores_in_range(self._capabilities)
        return result

    # -- 连接测试（供设置页 T20 复用；只报告连通性，不代表质量验收） ---------------

    async def check_connection(self) -> dict[str, Any]:
        """用固定公开文本检查连通、模型与响应结构；不读取知识库、不持久化任何配置。

        连通成功只说明服务可达且协议匹配，**不代表**质量验收通过（PRD §8.1）。
        """
        documents = tuple(
            _probe_candidate(index, text) for index, text in enumerate(CONNECTION_TEST_DOCUMENTS)
        )
        request = RerankRequest(
            query=CONNECTION_TEST_QUERY,
            scope_id="connection-test",
            index_revision="connection-test",
            documents=documents,
            profile_id="connection-test",
            instruction_id="connection-test",
            instruction_text=self._instruction_text,
        )
        try:
            result = await self.rerank(request)
        except RerankError as exc:
            return {
                "success": False,
                "model": self._model,
                "endpoint_host": self._destination,
                "code": exc.code.value,
                "message": exc.user_message,
                "quality_checked": False,
            }
        return {
            "success": True,
            "model": result.model,
            "endpoint_host": self._destination,
            "latency_ms": result.latency_ms,
            "candidate_count": result.document_count,
            "scored_count": len(result.scores),
            "trace_id": result.trace_id,
            "usage_reported": result.usage_reported,
            "quality_checked": False,
        }

    # -- 生命周期 ------------------------------------------------------------------

    async def aclose(self) -> None:
        """关闭自建连接池；注入的客户端由调用方管理。"""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _record_outbound(
        self, started_at: datetime, status: int | None, timer: float
    ) -> None:
        """记录一条脱敏审计事件（provider/host/方法/状态/耗时，无正文无密钥）。"""
        if self._outbound is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="rerank",
                destination=self._destination,
                method="rerank",
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(timer),
            )
        )


async def _post_gated(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    content: bytes,
) -> httpx.Response:
    """经进程级闸门发起 HTTP 请求（PRD §9：同时执行上限 2）。

    取消语义：任务在**排队等待**期间被取消时，``Semaphore.acquire`` 直接抛
    ``CancelledError``，请求根本不会发出（「取消不继续外呼」）；已发出的请求由 httpx 自身
    响应取消。闸门只覆盖真实出网那一段，等待期间不占用事件循环。
    """

    async with request_gate():
        return await client.post(url, headers=dict(headers), content=content)


def _probe_candidate(index: int, text: str) -> RerankCandidate:
    return RerankCandidate(
        candidate_id=f"connection-test-{index}",
        scope_id="connection-test",
        index_revision="connection-test",
        ranking_text=text,
        payload_hash=sha256_hex(text),
    )


def _resolve_api_key(api_key: str | SecretStr) -> str:
    """解析凭证：SecretStr 取明文；只在内存中使用，绝不进入日志、审计或档案。"""
    if api_key is None:
        return ""
    if isinstance(api_key, SecretStr):
        return api_key.get_secret_value().strip()
    return str(api_key).strip()


def _has_token_limits(capabilities: RerankProviderCapabilities) -> bool:
    return any(
        limit is not None
        for limit in (
            capabilities.max_query_tokens,
            capabilities.max_document_tokens,
            capabilities.max_request_tokens,
        )
    )


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RerankProfileMismatchError(f"{name} 必须是正整数")
    return value


def _status_detail(status: int, response: httpx.Response) -> str:
    """状态详情：只带 HTTP 状态与服务返回的数字错误码，绝不含响应体文本。"""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {status}"
    if isinstance(body, dict):
        code = body.get("code")
        if isinstance(code, int) and not isinstance(code, bool):
            return f"HTTP {status} (service code {code})"
    return f"HTTP {status}"


def _parse_usage(meta: object) -> SiliconFlowUsage | None:
    """解析 ``meta.tokens`` / ``meta.billed_units``；未返回时返回 ``None``，不估算。"""
    if not isinstance(meta, dict):
        return None
    tokens = meta.get("tokens")
    breakdown: dict[str, float] = {}
    billed = meta.get("billed_units")
    scalar_billed: float | None = None
    if isinstance(billed, dict):
        for key, value in billed.items():
            if (
                isinstance(key, str)
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                breakdown[key] = float(value)
    elif (
        isinstance(billed, (int, float))
        and not isinstance(billed, bool)
        and math.isfinite(float(billed))
    ):
        scalar_billed = float(billed)
    if not isinstance(tokens, dict) and not breakdown and scalar_billed is None:
        return None
    token_map = tokens if isinstance(tokens, dict) else {}
    return SiliconFlowUsage(
        input_tokens=_optional_count(token_map.get("input_tokens")),
        output_tokens=_optional_count(token_map.get("output_tokens")),
        image_tokens=_optional_count(token_map.get("image_tokens")),
        billed_units=scalar_billed,
        billing_breakdown=breakdown,
    )


def _optional_count(value: object) -> int | None:
    """计量字段：非负整数原样保留，其余视为「服务未返回」。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _destination_host(endpoint: str) -> str:
    """审计目的地只记 host（绝不含路径 / query / key）。"""
    hostname = urlsplit(endpoint).hostname
    if hostname:
        return hostname
    head = endpoint.split("/", 1)[0]
    return head.split("?", 1)[0].split("#", 1)[0] or "unknown"


def _elapsed_ms(timer: float) -> float:
    return max(0.0, (perf_counter() - timer) * 1000.0)
