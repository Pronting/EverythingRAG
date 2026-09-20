"""run 附带的**图片引用**：落本区图片库、与 ``verify_image`` 的数据路径打通（PRD 13.2 / 1.2 R07）。

PRD 13.2 首行的请求参数里有「图片引用」，PRD 1.2 R07 要求「图片及内容缓存」按工作区
隔离，PRD 7.1 的 ``verify_image`` 工具按「本区图片 ID」核对原图。本模块负责把一次 run
带的图片引用变成该工作区里**可核对**的数据：

1. 解析——``data:`` URI（前端贴图内联）本地解码；``http(s)`` 链接只在注入了
   ``ImageFetcher`` 时下载（**没有注入就明确标记 unavailable，绝不自建 fetcher 偷偷出网**）。
2. 落库——原始字节写进本区 ``image_state.db`` 的 ``image_originals``（内容 hash 去重），
   并登记一条 ``ImageTask``，身份与导入路径**同一套** ``make_block_id`` 约定。
3. 描述——注入了 ``IngestionPipeline``（即配置了视觉模型）时调用既有的
   ``process_image_task_and_upsert``：描述块进本区向量库，``metadata.image_content_hash``
   即本次的 ``content_hash``，因此 ``verify_image(image_id=<block_id 或 content_hash>)``
   能直接命中原图；未配置视觉时保持 ``pending_vision``，由现有启动续跑路径补做。

**不静默忽略**：每个引用都会带着 ``status`` + ``reason`` 持久化，并由 run 详情返回给前端。
``status`` 取值：

- ``described``：原图已缓存 + 描述已入库，``verify_image`` 可核对（``image_id`` 即块 ID）；
- ``cached``：原图已缓存、``ImageTask`` 已登记，但未识图（视觉未配置），待续跑；
- ``pending_vision``：登记成功但流水线报告处理失败（例如识图模型报错）；
- ``unavailable``：引用形态可识别但当前环境无法取得字节（未配置下载 / 视觉未接）；
- ``rejected``：引用本身不可用（非法 data URI、非 http(s)、超出张数上限）。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final

from app.agent.contracts import utc_now_iso
from app.ingestion.image_fetch import ImageFetcher, ImageFetchError, resolve_image_bytes
from app.ingestion.image_state_store import ImageStateStore, ImageTask
from app.ingestion.pipeline import IngestionPipeline, make_block_id

__all__ = [
    "MAX_RUN_IMAGES",
    "RUN_IMAGE_SOURCE_PREFIX",
    "LocalImageFetcher",
    "RunImageRef",
    "persist_run_images",
    "summarize_refs",
]

#: 一次 run 允许附带的图片张数（与 V1.1.0 ``ChatRequest.images`` 的上限一致）。
MAX_RUN_IMAGES: Final[int] = 4

#: 运行期图片引用的伪来源前缀（块 ID 里只出现内容哈希，不泄露本机路径，PRD 12.5）。
RUN_IMAGE_SOURCE_PREFIX: Final[str] = "run-image/"

#: ``status`` 取值（见模块 docstring）。
STATUS_DESCRIBED: Final[str] = "described"
STATUS_CACHED: Final[str] = "cached"
STATUS_PENDING_VISION: Final[str] = "pending_vision"
STATUS_UNAVAILABLE: Final[str] = "unavailable"
STATUS_REJECTED: Final[str] = "rejected"


class LocalImageFetcher:
    """包一层 ``ImageFetcher``：``data:`` URI 本地解码，http(s) 才委托出去。

    为什么需要它：``IngestionPipeline.process_image_task`` 自己调 ``fetcher.fetch(task.src)``
    重新解析图片字节，而既有 ``ImageFetcher`` 只走 httpx。前端贴图是 data URI，
    不包这一层就会在建 run 时被判成死链（图片静默失效）。**委托为 None 时绝不
    自建 fetcher 出网**：http(s) 直接抛 :class:`ImageFetchError`，由调用方标记 unavailable。
    """

    def __init__(self, delegate: Any = None) -> None:
        self._delegate = delegate

    def fetch(self, src: str) -> bytes:
        if str(src).startswith("data:"):
            return resolve_image_bytes(src)
        if self._delegate is None:
            raise ImageFetchError("未配置图片下载通道（出网必须先经审计）")
        return self._delegate.fetch(src)

    def __getattr__(self, item: str) -> Any:
        if self._delegate is None:  # pragma: no cover - 仅委托存在时才有其它成员
            raise AttributeError(item)
        return getattr(self._delegate, item)


class RunImageRef:
    """一条图片引用的处理结果（JSON 安全的 dict 构造器，避免与 API DTO 耦合）。"""

    @staticmethod
    def build(
        *,
        image_id: str,
        block_id: str = "",
        content_hash: str = "",
        kind: str = "",
        status: str,
        reason: str = "",
        src_ref: str = "",
    ) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "block_id": block_id,
            "content_hash": content_hash,
            "kind": kind,
            "status": status,
            "reason": reason,
            "src_ref": src_ref,
            "recorded_at": utc_now_iso(),
        }


async def persist_run_images(
    images: Sequence[str],
    *,
    image_store: ImageStateStore | None = None,
    fetcher: ImageFetcher | None = None,
    pipeline: IngestionPipeline | None = None,
) -> list[dict[str, Any]]:
    """把一次 run 的图片引用落进**本工作区**图片库，返回逐条结果（永不抛错）。

    参数是注入式的：``image_store`` 必须是本区的库；``fetcher`` 为 None 时**不下载**
    http(s) 图片（避免未经审计的出网）；``pipeline`` 为 None 时只缓存原图 + 登记任务，
    不调识图模型。
    """

    refs: list[dict[str, Any]] = []
    for index, raw in enumerate(images):
        ref = str(raw or "")
        if index >= MAX_RUN_IMAGES:
            refs.append(
                RunImageRef.build(
                    image_id="",
                    kind=_kind_of(ref),
                    status=STATUS_REJECTED,
                    reason=f"一次最多附带 {MAX_RUN_IMAGES} 张图片",
                    src_ref=_short_ref(ref),
                )
            )
            continue
        refs.append(
            await _persist_one(
                ref, image_store=image_store, fetcher=fetcher, pipeline=pipeline
            )
        )
    return refs


async def _persist_one(
    ref: str,
    *,
    image_store: ImageStateStore | None,
    fetcher: ImageFetcher | None,
    pipeline: IngestionPipeline | None,
) -> dict[str, Any]:
    kind = _kind_of(ref)
    short = _short_ref(ref)
    if image_store is None:
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_UNAVAILABLE,
            reason="本工作区图片库不可用，引用未落库",
            src_ref=short,
        )
    if kind == "invalid":
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_REJECTED,
            reason="只接受 data:image/... 内联图片或 http(s) 图片链接",
            src_ref=short,
        )
    if kind == "url" and fetcher is None:
        # 不静默忽略、也不偷偷出网：明确标记需要审计过的下载通道
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_UNAVAILABLE,
            reason="未配置图片下载通道（出网必须先经审计），链接未抓取",
            src_ref=short,
        )
    try:
        data = await asyncio.to_thread(resolve_image_bytes, ref, fetcher)
    except ImageFetchError as exc:
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_REJECTED,
            reason=f"图片无法解析：{type(exc).__name__}",
            src_ref=short,
        )
    except Exception as exc:  # noqa: BLE001 - 任何解析异常都如实记录，不吞掉
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_REJECTED,
            reason=f"图片无法解析：{type(exc).__name__}",
            src_ref=short,
        )

    content_hash = hashlib.sha256(data).hexdigest()
    try:
        image_store.save_image_bytes(content_hash, data)
        task = ImageTask(
            block_id=make_block_id(
                content_hash, "run-image", f"{RUN_IMAGE_SOURCE_PREFIX}{content_hash}"
            ),
            src=ref,
            doc_id=content_hash,
            source_file=f"{RUN_IMAGE_SOURCE_PREFIX}{content_hash}",
            heading_path="",
            anchor="",
        )
        image_store.add_task(task)
    except Exception as exc:  # noqa: BLE001 - 缓存失败也必须如实上报
        return RunImageRef.build(
            image_id="",
            kind=kind,
            status=STATUS_UNAVAILABLE,
            reason=f"原图缓存失败：{type(exc).__name__}",
            src_ref=short,
        )

    if pipeline is None:
        return RunImageRef.build(
            image_id=task.block_id,
            block_id=task.block_id,
            content_hash=content_hash,
            kind=kind,
            status=STATUS_CACHED,
            reason="原图已缓存、任务已登记；视觉模型未配置，识图后由启动续跑补做",
            src_ref=short,
        )
    try:
        described = await asyncio.to_thread(pipeline.process_image_task_and_upsert, task)
    except Exception as exc:  # noqa: BLE001 - 识图失败不拖垮建 run
        return RunImageRef.build(
            image_id=task.block_id,
            block_id=task.block_id,
            content_hash=content_hash,
            kind=kind,
            status=STATUS_PENDING_VISION,
            reason=f"识图失败：{type(exc).__name__}",
            src_ref=short,
        )
    if not described:
        return RunImageRef.build(
            image_id=task.block_id,
            block_id=task.block_id,
            content_hash=content_hash,
            kind=kind,
            status=STATUS_PENDING_VISION,
            reason="识图模型未返回可用描述，任务保留待续跑",
            src_ref=short,
        )
    return RunImageRef.build(
        image_id=task.block_id,
        block_id=task.block_id,
        content_hash=content_hash,
        kind=kind,
        status=STATUS_DESCRIBED,
        reason="",
        src_ref=short,
    )


def _kind_of(ref: str) -> str:
    if ref.startswith("data:"):
        return "data_uri"
    if ref.startswith(("http://", "https://")):
        return "url"
    return "invalid"


def _short_ref(ref: str) -> str:
    """元数据里保存的短引用：data URI 只留前缀（避免把整段 base64 写进消息元数据）。"""

    if ref.startswith("data:"):
        head = ref.split(",", 1)[0]
        return f"{head},…（内联图片已缓存，按内容哈希引用）"
    return ref[:2048]


def summarize_refs(refs: Sequence[Mapping[str, Any]]) -> str:
    """给日志/诊断用的一行摘要（只含状态计数，不含任何图片内容）。"""

    counts: dict[str, int] = {}
    for ref in refs:
        key = str(ref.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
