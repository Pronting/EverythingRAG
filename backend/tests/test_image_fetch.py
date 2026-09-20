"""图床图片下载单测：成功 / 非 200 / 超体积 / 非 http(s) / 空内容 + 审计。

全部零真实出网；审计走真实 OutboundClient 断言记录。

T04 顺手修复（captain 裁示，限定本文件 + ``ImageFetcher`` 的注入点）：原先用
``monkeypatch.setattr(mod.httpx, "Client", ...)`` 替换 **全局 httpx 同步客户端**，
那是进程级篡改，会影响并行执行的所有 httpx 消费者（与 ``tests/test_search_provider.py``
里修掉的 5 处同源）。现改为注入 ``httpx.MockTransport``（``ImageFetcher(transport=...)``，
默认 ``None`` 时行为不变）。

等价性说明：旧 fake 用两个 chunk（``b"abc"``/``b"def"``）喂 ``iter_bytes``，本文件
断言的是**聚合后的字节**与**累计体积上限**，两处在 MockTransport 下语义相同（仍走
``client.stream`` + ``iter_bytes``）。唯一未保留的是「响应按多个 chunk 产出」这一实现
细节——原用例本来也没有断言 chunk 切分形状。所有原有断言一条未减，另补两条更硬的
断言（请求方法与 URL、非 http(s) 时**零请求**）。
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core.outbound import OutboundClient
from app.ingestion import image_fetch as mod
from app.ingestion.image_fetch import ImageFetcher, ImageFetchError


class _Recorder:
    """``httpx.MockTransport`` 处理器：记录请求，按配置返回响应。"""

    def __init__(self, *, status_code: int = 200, body: bytes = b"") -> None:
        self.status_code = status_code
        self.body = body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, content=self.body)

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


def _fetcher(outbound: OutboundClient, recorder: _Recorder, **kwargs: Any) -> ImageFetcher:
    """构造 ImageFetcher 并注入 MockTransport（不触碰全局 ``httpx.Client``）。"""

    return ImageFetcher(outbound=outbound, transport=httpx.MockTransport(recorder), **kwargs)


def test_fetch_success_records_audit() -> None:
    """200 + 有内容 -> 返回字节，审计 provider=image_fetch 且 status=200。"""

    outbound = OutboundClient()
    recorder = _Recorder(status_code=200, body=b"abcdef")
    fetcher = _fetcher(outbound, recorder)

    data = fetcher.fetch("https://cdn.example.com/a.png")

    assert data == b"abcdef"
    assert recorder.urls == ["https://cdn.example.com/a.png"]
    assert recorder.requests[0].method == "GET"
    events = outbound.entries()
    assert len(events) == 1
    assert events[0].provider == "image_fetch"
    assert events[0].destination == "cdn.example.com"
    assert events[0].status == 200


def test_fetch_non_200_raises() -> None:
    """非 200 -> ImageFetchError，审计 status=None。"""

    outbound = OutboundClient()
    recorder = _Recorder(status_code=404, body=b"")
    fetcher = _fetcher(outbound, recorder)

    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/missing.png")

    assert outbound.entries()[0].status is None


def test_fetch_non_http_scheme_raises() -> None:
    """非 http(s) 链接 -> ImageFetchError，且不出网不记审计。"""

    outbound = OutboundClient()
    recorder = _Recorder()
    fetcher = _fetcher(outbound, recorder)

    with pytest.raises(ImageFetchError):
        fetcher.fetch("file:///etc/passwd")

    assert outbound.is_empty()
    assert recorder.requests == []  # 参数校验先于任何出网


def test_fetch_empty_body_raises() -> None:
    """200 但空内容 -> ImageFetchError。"""

    outbound = OutboundClient()
    recorder = _Recorder(status_code=200, body=b"")
    fetcher = _fetcher(outbound, recorder)

    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/empty.png")


def test_fetch_exceeds_size_cap_raises() -> None:
    """累计体积超过上限 -> ImageFetchError（流式读、不一次性吃进内存）。"""

    outbound = OutboundClient()
    recorder = _Recorder(status_code=200, body=b"x" * 8 + b"y" * 8)
    fetcher = _fetcher(outbound, recorder)
    fetcher._max_bytes = 10  # 测试直接压小上限

    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/big.png")


def test_fetch_construction_is_zero_outbound() -> None:
    """构造 ImageFetcher 零联网：不发请求、不记审计。"""

    outbound = OutboundClient()
    recorder = _Recorder(status_code=200, body=b"abc")
    fetcher = _fetcher(outbound, recorder)

    assert recorder.requests == []
    assert outbound.is_empty()
    assert fetcher._transport is not None  # 注入点生效，但构造期不使用


def test_fetcher_default_transport_is_none() -> None:
    """默认 ``transport=None``：生产路径行为不变（httpx.Client 用自带传输）。"""

    fetcher = ImageFetcher(outbound=None)
    assert fetcher._transport is None


def test_resolve_image_bytes_decodes_data_uri() -> None:
    """resolve_image_bytes：data URI 内联 base64 解码为原始字节。"""

    payload = b"fake-image-bytes"
    data_uri = "data:image/png;base64," + __import__("base64").b64encode(payload).decode("ascii")
    assert mod.resolve_image_bytes(data_uri) == payload


def test_resolve_image_bytes_invalid_data_uri_raises() -> None:
    """resolve_image_bytes：缺逗号分隔的 data URI -> ImageFetchError。"""

    with pytest.raises(ImageFetchError):
        mod.resolve_image_bytes("data:image/png;base64")
