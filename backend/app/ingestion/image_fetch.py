"""ingestion：图床图片下载 —— httpx GET 下载远程图片字节（方案 B 远程图床链接）。

隐私：这是识图链路唯一的「抓取内容」出网点，经 OutboundClient 记审计
（provider="image_fetch"，destination 只取 host，绝不含路径/query）。
仅 http/https；超时 / 体积上限 / 空内容兜底；非图内容由 image_preprocess 的
Pillow verify 兜底。失败抛 ImageFetchError（可读原因，不含 URL 敏感片段）。

同步 httpx 模型：与 CloudEmbedder 一致，供导入后台线程调用（不占事件循环）。
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from app.core.outbound import OutboundClient, OutboundEvent

DEFAULT_TIMEOUT_S = 30.0
MAX_BYTES = 20 * 1024 * 1024  # 20MB


class ImageFetchError(Exception):
    """图片下载失败（非 200 / 超时 / 超体积 / 非 http(s) 等）。"""


class ImageFetcher:
    """同步 httpx 图片下载器：流式读取 + 体积上限 + follow 重定向。"""

    def __init__(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_bytes: int = MAX_BYTES,
        outbound: OutboundClient | None = None,
    ) -> None:
        self._timeout = timeout_s
        self._max_bytes = max(1, int(max_bytes))
        self._outbound = outbound

    def fetch(self, url: str) -> bytes:
        """下载 ``url`` 的图片字节；失败抛 ImageFetchError（含可读原因）。"""
        if not url.startswith(("http://", "https://")):
            raise ImageFetchError("仅支持 http(s) 图片链接")
        host = _destination_host(url)
        started_at = _utcnow() if self._outbound is not None else None
        try:
            with (
                httpx.Client(timeout=self._timeout, follow_redirects=True) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code != 200:
                    raise ImageFetchError(f"下载失败: HTTP {response.status_code}")
                data = _read_limited(response, self._max_bytes)
        except ImageFetchError:
            self._record(started_at, host, None)
            raise
        except Exception as exc:  # 网络/超时等统一收敛为可读错误
            self._record(started_at, host, None)
            raise ImageFetchError(f"下载失败: {type(exc).__name__}") from exc
        else:
            self._record(started_at, host, 200)
            return data

    def _record(self, started_at: datetime | None, host: str, status: int | None) -> None:
        """记录一条出网审计事件；未注入 outbound / 未开始计时则跳过。"""
        if self._outbound is None or started_at is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="image_fetch",
                destination=host,
                method="GET",
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(started_at),
            )
        )


def resolve_image_bytes(src: str, fetcher: ImageFetcher | None = None) -> bytes:
    """把图片来源解析为原始字节：data URI（base64 内联）或 http(s) 图床 URL。

    供聊天「识图代理」复用：前端贴图以 data URI 上送，粘贴链接以 URL 上送，
    统一在此归一为字节，再交给 image_preprocess 降采样 / 识图模型 describe。
    """
    if src.startswith("data:"):
        if "," not in src:
            raise ImageFetchError("无效的 data URI 图片")
        _, payload = src.split(",", 1)
        try:
            return base64.b64decode(payload)
        except Exception as exc:  # base64 解码失败收敛为可读错误
            raise ImageFetchError(f"data URI 解码失败: {type(exc).__name__}") from exc
    return (fetcher if fetcher is not None else ImageFetcher()).fetch(src)


def _read_limited(response: httpx.Response, max_bytes: int) -> bytes:
    """流式读满响应体，超上限 / 空内容抛 ImageFetchError。"""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ImageFetchError(f"图片超过体积上限 {max_bytes} 字节")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise ImageFetchError("下载内容为空")
    return data


def _destination_host(url: str) -> str:
    """从 URL 提取仅 host 的审计目的地（绝不含路径 / query / 凭证）。"""
    hostname = urlsplit(url).hostname
    if hostname:
        return hostname
    head = url.split("/", 1)[0]
    return head.split("?", 1)[0].split("#", 1)[0] or "unknown"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _elapsed_ms(started_at: datetime) -> float:
    return max(0.0, (datetime.now(UTC) - started_at).total_seconds() * 1000.0)
