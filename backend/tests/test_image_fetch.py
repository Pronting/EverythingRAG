"""图床图片下载单测：成功 / 非 200 / 超体积 / 非 http(s) / 空内容 + 审计。

全部用 fake httpx client，零真实出网；审计走真实 OutboundClient 断言记录。
"""

from __future__ import annotations

from typing import Self

import pytest

from app.core.outbound import OutboundClient
from app.ingestion import image_fetch as mod
from app.ingestion.image_fetch import ImageFetcher, ImageFetchError


class _FakeResponse:
    """httpx.stream 返回的响应替身（context manager + status_code + iter_bytes）。"""

    def __init__(self, status_code: int = 200, chunks: tuple[bytes, ...] = (b"",)) -> None:
        self.status_code = status_code
        self._chunks = list(chunks)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def iter_bytes(self):
        yield from self._chunks


class _FakeClient:
    """httpx.Client 替身：__enter__/__exit__ + stream 返回给定响应。"""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.url: str | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def stream(self, method: str, url: str) -> _FakeResponse:
        self.url = url
        return self._response


def _fetcher(outbound: OutboundClient, response: _FakeResponse, monkeypatch: pytest.MonkeyPatch) -> ImageFetcher:
    """构造 ImageFetcher 并把 httpx.Client 替换为返回给定响应的 fake。"""
    monkeypatch.setattr(mod.httpx, "Client", lambda *a, **kw: _FakeClient(response))
    return ImageFetcher(outbound=outbound)


def test_fetch_success_records_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """200 + 有内容 -> 返回字节，审计 provider=image_fetch 且 status=200。"""
    outbound = OutboundClient()
    fetcher = _fetcher(outbound, _FakeResponse(200, (b"abc", b"def")), monkeypatch)
    data = fetcher.fetch("https://cdn.example.com/a.png")
    assert data == b"abcdef"
    events = outbound.entries()
    assert len(events) == 1
    assert events[0].provider == "image_fetch"
    assert events[0].destination == "cdn.example.com"
    assert events[0].status == 200


def test_fetch_non_200_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 200 -> ImageFetchError，审计 status=None。"""
    outbound = OutboundClient()
    fetcher = _fetcher(outbound, _FakeResponse(404, (b"",)), monkeypatch)
    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/missing.png")
    assert outbound.entries()[0].status is None


def test_fetch_non_http_scheme_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 http(s) 链接 -> ImageFetchError，且不出网不记审计。"""
    outbound = OutboundClient()
    fetcher = _fetcher(outbound, _FakeResponse(), monkeypatch)
    with pytest.raises(ImageFetchError):
        fetcher.fetch("file:///etc/passwd")
    assert outbound.is_empty()


def test_fetch_empty_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """200 但空内容 -> ImageFetchError。"""
    outbound = OutboundClient()
    fetcher = _fetcher(outbound, _FakeResponse(200, (b"",)), monkeypatch)
    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/empty.png")


def test_fetch_exceeds_size_cap_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """累计体积超过上限 -> ImageFetchError（流式读、不一次性吃进内存）。"""
    outbound = OutboundClient()
    fetcher = _fetcher(outbound, _FakeResponse(200, (b"x" * 8, b"y" * 8)), monkeypatch)
    fetcher._max_bytes = 10  # 测试直接压小上限
    with pytest.raises(ImageFetchError):
        fetcher.fetch("https://cdn.example.com/big.png")


def test_resolve_image_bytes_decodes_data_uri() -> None:
    """resolve_image_bytes：data URI 内联 base64 解码为原始字节。"""
    payload = b"fake-image-bytes"
    data_uri = "data:image/png;base64," + __import__("base64").b64encode(payload).decode("ascii")
    assert mod.resolve_image_bytes(data_uri) == payload


def test_resolve_image_bytes_invalid_data_uri_raises() -> None:
    """resolve_image_bytes：缺逗号分隔的 data URI -> ImageFetchError。"""
    with pytest.raises(ImageFetchError):
        mod.resolve_image_bytes("data:image/png;base64")
