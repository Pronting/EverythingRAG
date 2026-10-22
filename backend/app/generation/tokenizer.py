from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Protocol

__all__ = [
    "DEEPSEEK_RECIPE_PACKAGE",
    "DEEPSEEK_RECIPE_REVISION",
    "TOKENIZER_DIR_ENV_VAR",
    "TOKENIZER_ID",
    "TOKENIZER_MANIFEST",
    "TOKENIZER_PINNED_SHA256",
    "TOKENIZER_PINNED_SIZE",
    "TOKENIZER_RESOURCE_FILENAME",
    "TOKENIZER_VERSION",
    "MessageEncoder",
    "PromptCounter",
    "PromptEncoding",
    "RequestTokenReport",
    "TokenizerManifest",
    "TokenizerResourceError",
    "collect_prompt_token_ids",
    "count_prompt_tokens",
    "default_prompt_counter",
    "official_message_payload",
    "official_prompt",
    "resolve_pinned_encoder",
    "sha256_of_file",
    "to_official_messages",
    "verify_resource",
]

#: 官方 tokenizer 身份（PRD 9.3 的「tokenizer 版本」）。
TOKENIZER_ID: Final[str] = "deepseek-recipe/static/tokenizers/v41"
TOKENIZER_VERSION: Final[str] = "v4.1"
TOKENIZER_RESOURCE_FILENAME: Final[str] = "tokenizer.json"

#: 官方仓库修订（取值来源与复核依据）。
DEEPSEEK_RECIPE_REVISION: Final[str] = "deepseek-ai/deepseek-recipe@8cadfede7063c896b944e7bae05daa3549ae97ea"
DEEPSEEK_RECIPE_PACKAGE: Final[str] = "deepseek-recipe"

#: tokenizer 文件 SHA-256（PRD 9.3 要求固定）。
#: 官方仓库 ``static/tokenizers/v41/tokenizer.json`` 的固定文件身份。
TOKENIZER_PINNED_SHA256: Final[str] = (
    "df385ab3ba4bbb1815327f3e71bba5b263263f5d015a838e6ff295ca69b3198e"
)

#: 官方文件字节数（与 SHA-256 一起做二次身份校验，防截断下载）。
TOKENIZER_PINNED_SIZE: Final[int] = 6_634_606

#: 哈希固定状态说明；写入证据与错误消息。
TOKENIZER_SHA256_STATE: Final[str] = "已固定（官方 v41/tokenizer.json，6,634,606 字节）"

TOKENIZER_DIR_ENV_VAR: Final[str] = "EVERYTHING_RAG_TOKENIZER_DIR"
TOKENIZER_BUNDLED_DIR: Final[Path] = Path(__file__).resolve().parent / "resources" / "deepseek-v41"


class TokenizerResourceError(RuntimeError):
    """tokenizer 资源缺失 / 版本与哈希不匹配 / 编码失败：明确错误，绝不伪造计数。"""


class MessageEncoder(Protocol):
    """tokenizer 编码器协议：把官方消息格式编码为 token ID。"""

    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> list[int]:
        """``messages`` 为官方消息列表，``payload`` 为含 ``tools``/思考参数的完整请求。"""
        ...


@dataclass(frozen=True)
class TokenizerManifest:
    """tokenizer 资源清单：标识、版本、文件与 SHA-256（PRD 9.3 三个固定项）。"""

    tokenizer_id: str
    tokenizer_version: str
    filename: str
    revision: str
    sha256: str = ""
    size: int = 0
    path: Path | None = None

    @property
    def sha256_pinned(self) -> bool:
        return bool(self.sha256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_version": self.tokenizer_version,
            "filename": self.filename,
            "revision": self.revision,
            "sha256": self.sha256,
            "sha256_pinned": self.sha256_pinned,
            "size": self.size,
            "path": str(self.path) if self.path else "",
        }


#: 全局清单（软件版本 + tokenizer 版本 + 文件 SHA-256 三联，PRD 9.3）。
TOKENIZER_MANIFEST: Final[TokenizerManifest] = TokenizerManifest(
    tokenizer_id=TOKENIZER_ID,
    tokenizer_version=TOKENIZER_VERSION,
    filename=TOKENIZER_RESOURCE_FILENAME,
    revision=DEEPSEEK_RECIPE_REVISION,
    sha256=TOKENIZER_PINNED_SHA256,
    size=TOKENIZER_PINNED_SIZE,
)


@dataclass(frozen=True)
class PromptEncoding:
    """一次 prompt 编码结果（token ID 数量是唯一计数依据）。"""

    prompt: str
    token_ids: tuple[int, ...]
    model: str = ""

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class RequestTokenReport:
    """一次本地计数的完整报告（含 tokenizer 出处，供核验与落库）。"""

    local_prompt_tokens: int
    model: str = ""
    tool_count: int = 0
    reasoning_messages: int = 0
    tokenizer: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_prompt_tokens": self.local_prompt_tokens,
            "model": self.model,
            "tool_count": self.tool_count,
            "reasoning_messages": self.reasoning_messages,
            "tokenizer": dict(self.tokenizer),
        }


def sha256_of_file(path: Path) -> str:
    """流式计算文件 SHA-256（十六进制小写，不把整文件读进内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resource(path: Path, manifest: TokenizerManifest = TOKENIZER_MANIFEST) -> str:
    """校验 tokenizer 文件（存在性 + 字节数 + SHA-256）；返回实际哈希。

    SHA-256 与官方字节数一起校验：任一不符即拒绝，不用未知版本计数（PRD 9.3）。
    """
    if not path.is_file():
        raise TokenizerResourceError(
            f"tokenizer 文件不存在: {path}（PRD 9.3 要求固定 tokenizer 版本与文件 SHA-256）"
        )
    size = path.stat().st_size
    if manifest.size and size != manifest.size:
        raise TokenizerResourceError(
            "tokenizer 文件大小与官方固定值不符（可能是截断/损坏下载）: "
            f"期望 {manifest.size} 字节，实际 {size} 字节（{path}）"
        )
    actual = sha256_of_file(path)
    if not manifest.sha256:
        raise TokenizerResourceError(
            f"tokenizer 文件的 SHA-256 未固定（{manifest.revision}）。"
            f"已就位文件 {path} 的实际 SHA-256 = {actual}，"
            "确认来自官方仓库后填入 app/generation/tokenizer.py 的 TOKENIZER_PINNED_SHA256。"
        )
    if actual != manifest.sha256:
        raise TokenizerResourceError(
            "tokenizer 文件 SHA-256 不匹配（拒绝用未知版本计数）: "
            f"期望 {manifest.sha256}，实际 {actual}（{path}）"
        )
    return actual


class _RecipeEncoder:
    """``deepseek-recipe`` 官方编码路径的薄适配（惰性导入，不新增硬依赖）。"""

    def __init__(self, encoding: Any, manifest: TokenizerManifest) -> None:
        self._encoding = encoding
        self._manifest = manifest

    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> list[int]:
        from deepseek_recipe import (  # type: ignore[import-not-found]
            ChatCompletionRequest,
            ConversionOptions,
        )
        try:
            converted = ChatCompletionRequest(payload).convert(ConversionOptions())
            raw_ids = self._encoding.encode(converted.conversation)
        except TokenizerResourceError:
            raise
        except Exception as exc:
            raise TokenizerResourceError(f"官方 deepseek-recipe 编码失败: {exc}") from exc
        return [int(item) for item in raw_ids]


def resolve_pinned_encoder(
    manifest: TokenizerManifest = TOKENIZER_MANIFEST,
    *,
    resource_dir: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> MessageEncoder:
    """从指定目录、环境变量或随包资源加载并校验官方 tokenizer。"""
    source = os.environ if env is None else env
    directory: Path | str = resource_dir or source.get(TOKENIZER_DIR_ENV_VAR) or TOKENIZER_BUNDLED_DIR
    path = Path(directory) / manifest.filename
    verify_resource(path, manifest)
    from deepseek_recipe import DeepseekV41Encoding, Tokenizer  # type: ignore[import-not-found]
    tokenizer = Tokenizer.from_file(str(path))
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    return _RecipeEncoder(encoding=encoding, manifest=manifest)


class PromptCounter:
    """本地 token 计数器：编码器缺省惰性解析（首次使用时才找资源）。"""

    def __init__(
        self,
        encoder: MessageEncoder | None = None,
        *,
        manifest: TokenizerManifest = TOKENIZER_MANIFEST,
        resource_dir: Path | str | None = None,
    ) -> None:
        self._encoder = encoder
        self._manifest = manifest
        self._resource_dir = resource_dir
        self._resolved: MessageEncoder | None = encoder

    @property
    def encoder(self) -> MessageEncoder:
        """当前编码器；未注入时惰性解析（失败抛 :class:`TokenizerResourceError`）。"""
        if self._resolved is None:
            self._resolved = resolve_pinned_encoder(
                self._manifest, resource_dir=self._resource_dir
            )
        return self._resolved

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode(self, request: Mapping[str, Any]) -> PromptEncoding:
        """把完整 Chat Completions 请求编码为官方 prompt token（PRD 9.3）。"""
        payload = official_message_payload(request)
        token_ids = self.encoder.encode_messages(payload["messages"], payload)
        return PromptEncoding(
            prompt=official_prompt(request),
            token_ids=tuple(token_ids),
            model=str(request.get("model") or ""),
        )

    def count(self, request: Mapping[str, Any]) -> int:
        """仅要数量时的便捷入口。"""
        return self.encode(request).token_count

    def report(self, request: Mapping[str, Any]) -> RequestTokenReport:
        """完整报告：数量 + tokenizer 出处 + 工具/推理消息统计。"""
        encoding = self.encode(request)
        return RequestTokenReport(
            local_prompt_tokens=encoding.token_count,
            model=encoding.model,
            tool_count=len(list(request.get("tools") or [])),
            reasoning_messages=count_reasoning_messages(request.get("messages") or []),
            tokenizer=self.tokenizer_provenance(),
        )

    def tokenizer_provenance(self) -> dict[str, Any]:
        """tokenizer 出处：清单 + 实际安装的 deepseek-recipe 版本（缺失留空）。"""
        payload = self._manifest.as_dict()
        try:
            payload["package_version"] = metadata.version(DEEPSEEK_RECIPE_PACKAGE)
        except metadata.PackageNotFoundError:
            payload["package_version"] = ""
        return payload


_DEFAULT_COUNTER: PromptCounter | None = None


def default_prompt_counter() -> PromptCounter:
    """进程内默认计数器（编码器惰性解析，import 与首次调用均零联网）。"""
    global _DEFAULT_COUNTER
    if _DEFAULT_COUNTER is None:
        _DEFAULT_COUNTER = PromptCounter()
    return _DEFAULT_COUNTER


def count_prompt_tokens(request: Mapping[str, Any]) -> int:
    """完整 Chat Completions 请求 → 官方消息格式 → token ID 数量（PRD 9.3）。

    资源缺失或 tokenizer 无法取得时抛 :class:`TokenizerResourceError`，不伪造计数。
    """
    return default_prompt_counter().count(request)


def collect_prompt_token_ids(request: Mapping[str, Any]) -> list[int]:
    """:func:`count_prompt_tokens` 的 token ID 形态（核验与调参用）。"""
    return list(default_prompt_counter().encode(request).token_ids)


def count_reasoning_messages(messages: Sequence[Mapping[str, Any]]) -> int:
    """统计需要保留推理内容的消息条数（PRD 9.3：含要求保留的推理内容）。"""
    return sum(1 for item in messages if item.get("reasoning_content"))


def to_official_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把历史消息转换为官方消息格式（纯函数，PRD 9.3）。

    - ``content`` 支持字符串与多模态分段（分段内容按官方形状原样保留）；
    - assistant 的 ``reasoning_content`` 只要存在就保留（含未调用工具的轮次）；
    - assistant 的 ``tool_calls`` → 官方 ``{id, type, function:{name, arguments}}``，
      ``arguments`` 保持调用产生的原始文本；
    - ``tool`` 消息必须带 ``tool_call_id``（否则抛错，不静默丢掉关联）；
    - 不接受未知 role（避免本地状态字段混进请求口径）。
    """
    official: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        item: dict[str, Any] = {"role": role}
        if role == "assistant":
            item["content"] = _content_of(message)
            reasoning = message.get("reasoning_content")
            if reasoning:
                item["reasoning_content"] = reasoning
            tool_calls = message.get("tool_calls")
            if tool_calls:
                item["tool_calls"] = [_official_tool_call(call) for call in tool_calls]
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise TokenizerResourceError(
                    f"第 {index} 条 tool 消息缺少 tool_call_id，无法按官方格式计数"
                )
            item["tool_call_id"] = call_id
            item["content"] = _content_of(message)
        elif role in {"system", "user", "developer"}:
            item["content"] = _content_of(message)
        else:
            raise TokenizerResourceError(
                f"第 {index} 条消息 role={role!r} 不在官方 Chat Completions 消息范围内"
            )
        official.append(item)
    return official


def official_message_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    """官方格式请求体：``model`` + 官方消息 + ``tools``（+ 思考参数，PRD 9.3）。

    只保留影响 prompt token 数量的字段：``model`` / ``messages`` / ``tools`` /
    ``thinking`` / ``reasoning_effort``；``stream``、``max_tokens`` 等不参与计数。
    """
    payload: dict[str, Any] = {
        "model": str(request.get("model") or ""),
        "messages": to_official_messages(request.get("messages") or []),
    }
    tools = request.get("tools")
    if tools:
        payload["tools"] = [_official_tool(item) for item in tools]
    thinking = request.get("thinking")
    if thinking is None:
        extra_body = request.get("extra_body")
        if isinstance(extra_body, Mapping):
            thinking = extra_body.get("thinking")
    if thinking is not None:
        payload["thinking"] = thinking
    effort = request.get("reasoning_effort")
    if effort is not None:
        # 官方消息模板使用 none 表达网关 off 所指定的关闭推理状态。
        payload["reasoning_effort"] = "none" if effort == "off" else effort
    return payload


def official_prompt(request: Mapping[str, Any]) -> str:
    """官方消息格式的确定性 JSON 文本（UTF-8、保留键序、不转义非 ASCII）。

    仅用于日志 / 去重 / 核验展示；token 计数走编码器的 token ID 数量。
    """
    return json.dumps(
        official_message_payload(request),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def _official_tool(item: Mapping[str, Any]) -> dict[str, Any]:
    """工具定义规范化（只保留官方字段，避免本地字段污染计数口径）。"""
    function = item.get("function") or {}
    payload: dict[str, Any] = {"type": str(item.get("type") or "function")}
    normalized: dict[str, Any] = {
        "name": str(function.get("name") or ""),
        "parameters": function.get("parameters") or {},
    }
    if function.get("description"):
        normalized["description"] = function["description"]
    if function.get("strict") is not None:
        normalized["strict"] = bool(function["strict"])
    payload["function"] = normalized
    return payload


def _official_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """工具调用规范化：``id`` 透传、``arguments`` 保持调用产生的原始文本。"""
    function = call.get("function") or {}
    raw_arguments = function.get("arguments")
    if raw_arguments is None:
        raw_arguments = call.get("raw_arguments")
    if raw_arguments is None:
        raw_arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
    return {
        "id": str(call.get("id") or call.get("call_id") or ""),
        "type": str(call.get("type") or "function"),
        "function": {
            "name": str(function.get("name") or call.get("name") or ""),
            "arguments": str(raw_arguments),
        },
    }


def _content_of(message: Mapping[str, Any]) -> Any:
    """消息内容：字符串原样；分段列表按官方形状透传；缺失给空串。"""
    content = message.get("content")
    if content is None:
        return ""
    return content
