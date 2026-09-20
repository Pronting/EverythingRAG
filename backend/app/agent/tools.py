"""ToolRegistry 与 Agent 工具集（PRD 7.1）。Owner: T04。

职责：
- 注册 10 个工具（``lookup_catalog`` / ``get_workspace_overview`` /
  ``search_knowledge`` / ``read_document`` / ``search_history`` / ``read_history`` /
  ``web_search`` / ``read_web_page`` / ``verify_image`` / ``ask_user``），
  记录 Pydantic 参数类型、处理函数、是否联网与允许使用的 Agent。
- 工具参数使用标准 JSON 解析，校验通过后执行；``workspace_id`` 由已绑定的
  ``WorkspaceContext`` 提供，不暴露给模型自由指定（PRD 5.2）。
- 网页正文与知识原文作为参考数据进入上下文，不得改变工具权限或系统规则。
- 联网工具统一走 ``OutboundClient`` 审计入口（CLAUDE.md 隐私红线）。

T04 实现要点（PRD 7.1 / 7.3 / 7.4 / 12.1）：

1. **工具名单一权威**：:class:`ToolName` 是 10 个工具名的唯一来源。
2. **执行必过注册表**：本模块**不**导出模块级 ``web_search(...)`` 之类的函数；消费方
   （T07 Web/RAG 分支）用 ``registry.get(ToolName.WEB_SEARCH)`` 或
   ``await registry.execute(...)``，使权限校验与 ``operation_id`` 登记无法被绕过
   （PRD 7.1 末段 + 12.1）。
3. **稳定 operation_id + 重放复用**：``operation_id = op-<sha256(run|task|tool|args)>``；
   同一 run/任务下同参数（且知识版本一致）的重放直接复用已保存结果。
4. **Agent 门控**：``ToolSpec.allowed_agents`` 在任何检索 / 出网之前校验。A01 要求的
   「Web 路径不得调用 ``search_knowledge``（本地知识正文检索次数为 0）」由
   ``AgentKind.WEB`` 不在 ``search_knowledge`` 的允许集合里结构性保证。
5. **脱敏挂载点**：:meth:`ToolContext.redact_arguments` 与
   :meth:`ToolRegistry.redact_arguments` 是「已脱敏参数」入口。联网工具把脱敏后的
   参数真正发出去；redactor 由 T09 注入（本模块不实现姓名识别）。
6. **artifacts 路径规则**（工作区 ``artifacts/`` 内，绝对路径不外泄）：
   ``operations/<operation_id>.json``（操作结果缓存 / 重放依据）、
   ``web/<content_hash>.md``（长网页完整提取结果，Agent 分批读取并记录位置）。
7. **历史检索走可注入接缝**：``ToolContext.history_search``（T09 装配 T08 的 FTS5 实现）
   非 None 时 ``search_history`` 完全委托；为 None 时退化为**有界扫描**（会话 / 消息 /
   命中数三重上限，触顶截断并在 message 里说明）。``agent`` 层因此**不 import
   ``memory`` 层**，符合架构 §5 的依赖方向，且两条路径对外 schema 一致。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from app.agent.app_store import AppStore, new_id
from app.agent.contracts import (
    DEFAULT_CONTEXT_TOP_K,
    AgentKind,
    ContractModel,
    InputRequest,
    InputRequestStatus,
    OperationStatus,
    WebPageEvidence,
    WebReadRange,
    utc_now_iso,
)
from app.core.outbound import OutboundClient
from app.generation.knowledge_catalog import KnowledgeOverview, build_knowledge_overview
from app.retrieval import rerank_pipeline
from app.retrieval.query_transform import QueryVariant, build_query_variants
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk
from app.search.base import SearchError

__all__ = [
    "CATALOG_DEFAULT_PAGE_SIZE",
    "CATALOG_MAX_PAGE_SIZE",
    "HISTORY_FALLBACK_MAX_CONVERSATIONS",
    "HISTORY_FALLBACK_MAX_MATCHES",
    "HISTORY_FALLBACK_MAX_MESSAGES",
    "HISTORY_SCAN_LIMIT",
    "READ_DOCUMENT_DEFAULT_CHARS",
    "READ_DOCUMENT_MAX_CHARS",
    "READ_WEB_PAGE_DEFAULT_CHARS",
    "READ_WEB_PAGE_MAX_CHARS",
    "TOOL_FORBIDDEN_PATH_KEYS",
    "WEB_ARTIFACT_MIN_CHARS",
    "AskUserParams",
    "GetWorkspaceOverviewParams",
    "HistorySearchCallable",
    "LookupCatalogParams",
    "ReadDocumentParams",
    "ReadHistoryParams",
    "ReadWebPageParams",
    "SearchHistoryParams",
    "SearchKnowledgeParams",
    "ToolArgumentError",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolName",
    "ToolPermissionError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
    "ToolUnavailableError",
    "VerifyImageParams",
    "WebPageEvidence",
    "WebReadRange",
    "WebSearchParams",
    "arguments_hash",
    "artifact_path",
    "build_default_registry",
    "document_id_of_block",
    "read_artifact",
    "resolve_document_source",
    "stable_operation_id",
    "write_artifact",
]


# =====================================================================
# 0. 常量
# =====================================================================

#: artifacts 子目录（白名单：只允许写这两个目录）。
OPERATIONS_ARTIFACT_DIR: Final[str] = "operations"
WEB_ARTIFACT_DIR: Final[str] = "web"

#: 超过该字符数的网页正文写入 ``artifacts/web/`` 并改为分批读取（PRD 7.3）。
WEB_ARTIFACT_MIN_CHARS: Final[int] = 2000

#: 单次网页正文读取的默认 / 上限字符数。
READ_WEB_PAGE_DEFAULT_CHARS: Final[int] = 20000
READ_WEB_PAGE_MAX_CHARS: Final[int] = 200000

#: ``read_document`` 单次读取的默认 / 上限字符数。
READ_DOCUMENT_DEFAULT_CHARS: Final[int] = 4000
READ_DOCUMENT_MAX_CHARS: Final[int] = 40000

#: 目录分页（``lookup_catalog``）。
CATALOG_DEFAULT_PAGE_SIZE: Final[int] = 10
CATALOG_MAX_PAGE_SIZE: Final[int] = 50

#: 单个会话最多扫描的历史消息数（有界，避免一次工具调用扫全库）。
HISTORY_SCAN_LIMIT: Final[int] = 2000

#: 回退扫描的**跨会话**上限：会话数 / 累计消息数 / 收集到的命中数。
#: 任一触顶即停止扫描，并在副结果 ``message`` 里说明「结果可能不完整」。
#: 接入 FTS 索引（``ToolContext.history_search``）后这些上限不再生效。
HISTORY_FALLBACK_MAX_CONVERSATIONS: Final[int] = 200
HISTORY_FALLBACK_MAX_MESSAGES: Final[int] = 5000
HISTORY_FALLBACK_MAX_MATCHES: Final[int] = 500

#: FTS 历史检索接缝的签名（异步、纯数据进出；见 :attr:`ToolContext.history_search`）。
HistorySearchCallable = Callable[[str, Mapping[str, Any]], "Awaitable[Mapping[str, Any]]"]

#: 工具返回数据里**禁止出现**的键（这些键可能承载绝对路径 / 原图路径）。
#: 测试据此断言工具结果不泄露本机路径。
TOOL_FORBIDDEN_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {"source_file", "source_path", "abs_path", "image_path", "root_dir", "documents_dir"}
)

_SAFE_ARTIFACT_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_ARTIFACT_DIRS: Final[frozenset[str]] = frozenset({OPERATIONS_ARTIFACT_DIR, WEB_ARTIFACT_DIR})
_CLEAN_LABEL: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


# =====================================================================
# 1. 工具名（唯一权威）、状态与错误
# =====================================================================


class ToolName(StrEnum):
    """PRD 7.1 的 10 个工具名——**唯一权威来源**。

    其它模块（工具消费方、事件名、测试）一律引用本枚举，不要再写字面量。
    """

    LOOKUP_CATALOG = "lookup_catalog"
    GET_WORKSPACE_OVERVIEW = "get_workspace_overview"
    SEARCH_KNOWLEDGE = "search_knowledge"
    READ_DOCUMENT = "read_document"
    SEARCH_HISTORY = "search_history"
    READ_HISTORY = "read_history"
    WEB_SEARCH = "web_search"
    READ_WEB_PAGE = "read_web_page"
    VERIFY_IMAGE = "verify_image"
    ASK_USER = "ask_user"


class ToolStatus(StrEnum):
    """工具返回状态（PRD 7.2 末段：空 / 无关 / 部分 / 待澄清分别如实上报）。

    ``empty`` / ``irrelevant`` / ``not_found`` / ``partial`` 都是**正常状态**而非错误：
    调用方据此决定下一步（换查询 / 放弃本地路径 / 转 Web），不需要异常控制流。
    """

    OK = "ok"
    EMPTY = "empty"  # 查询结果为空（本区无命中 / 未建索引 / 目录无匹配）
    IRRELEVANT = "irrelevant"  # 有候选资料但没有达到相关度的内容
    NOT_FOUND = "not_found"  # 指定 ID / 会话 / 章节在本工作区不存在（跨区等同不存在）
    PARTIAL = "partial"  # 部分成功（多 URL 逐项失败 / 正文不完整）
    WAITING_INPUT = "waiting_input"  # ask_user 已创建澄清请求，等待用户回答


class ToolErrorCode(StrEnum):
    """工具错误码（PRD 12.5 ``AgentError.error_code`` 的来源之一）。"""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_JSON = "invalid_json"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_PERMITTED = "not_permitted"
    WORKSPACE_MISMATCH = "workspace_mismatch"
    NOT_CONFIGURED = "not_configured"
    SERVICE_ERROR = "service_error"
    INTERNAL = "internal"


class ToolError(Exception):
    """工具层领域错误基类：携带错误码、工具名与**脱敏**的用户可读消息。"""

    default_code: ToolErrorCode = ToolErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode | None = None,
        tool: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.tool = tool
        self.message = message

    def to_error_payload(self) -> dict[str, str]:
        """错误对象（PRD 12.5）：错误码 + 用户说明 + 失败工具。"""

        return {"error_code": str(self.code), "message": self.message, "tool": self.tool}


class ToolArgumentError(ToolError):
    """参数无法解析或不符合 schema（标准 JSON 解析 + Pydantic 校验失败）。"""

    default_code = ToolErrorCode.INVALID_ARGUMENTS


class ToolPermissionError(ToolError):
    """当前 Agent 不允许使用该工具，或检索面不属于当前工作区。"""

    default_code = ToolErrorCode.NOT_PERMITTED


class ToolUnavailableError(ToolError):
    """依赖未配置（嵌入器 / 联网 provider / 识图模型 / run 上下文 / artifacts 目录）。"""

    default_code = ToolErrorCode.NOT_CONFIGURED


class ToolExecutionError(ToolError):
    """执行期错误（服务失败 / 未预期异常），在发生位置抛出（PRD 7.4）。"""

    default_code = ToolErrorCode.SERVICE_ERROR


# =====================================================================
# 2. 工具参数（标准 JSON 校验后就地失败，PRD 7.1 / 12.5）
# =====================================================================


class LookupCatalogParams(ContractModel):
    """``lookup_catalog`` 参数：在本工作区目录中查找文档、术语与标题位置。"""

    query: str = Field(min_length=1, description="要查找的术语、主题或文档线索，非空。")
    cursor: str | None = Field(default=None, description="上一页返回的 next_cursor；首页省略。")
    limit: int = Field(
        default=CATALOG_DEFAULT_PAGE_SIZE,
        ge=1,
        le=CATALOG_MAX_PAGE_SIZE,
        description="本页返回的文档条数上限。",
    )


class GetWorkspaceOverviewParams(ContractModel):
    """``get_workspace_overview`` 参数：按统计类别取本区概览。"""

    categories: list[Literal["knowledge", "documents", "topics", "images", "tasks"]] = Field(
        default_factory=list,
        description=(
            "统计类别；省略 = 全部。knowledge=文件/块统计，documents=已登记与已索引文档数，"
            "topics=主题，images=图片块，tasks=当前 run 的任务状态。"
        ),
    )


class SearchKnowledgeParams(ContractModel):
    """``search_knowledge`` 参数：本工作区知识正文检索（PRD 7.2）。"""

    query: str = Field(min_length=1, description="检索查询，保留原问题与内部术语。")
    top_k: int = Field(
        default=DEFAULT_CONTEXT_TOP_K,
        ge=1,
        le=50,
        description="返回片段数上限；默认 8（PRD 7.2 / 18.1 的 context_top_k）。",
    )
    document_scope: list[str] = Field(
        default_factory=list,
        description="可选文档范围：lookup_catalog 返回的 document_id 列表；省略 = 全工作区。",
    )
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="相关度门控：低于该值的命中记为「资料无关」而不返回。省略 = 不额外门控。",
    )


class ReadDocumentParams(ContractModel):
    """``read_document`` 参数：读取指定文档的原文位置。"""

    document_id: str = Field(min_length=1, description="lookup_catalog 返回的 document_id。")
    heading: str | None = Field(
        default=None,
        description="章节标题片段（大小写不敏感子串匹配）；省略 = 全文按标题顺序拼接。",
    )
    offset: int = Field(default=0, ge=0, description="从该字符位置开始读取（续读用）。")
    limit: int = Field(
        default=READ_DOCUMENT_DEFAULT_CHARS,
        ge=1,
        le=READ_DOCUMENT_MAX_CHARS,
        description="本次读取的最大字符数，默认 4000。",
    )


class SearchHistoryParams(ContractModel):
    """``search_history`` 参数：检索本工作区历史会话消息（PRD 7.1）。"""

    query: str = Field(
        min_length=1,
        description="检索词；空白分隔多词时**任一命中即为候选**，命中词数多者排前（OR + 打分）。",
    )
    start_date: str | None = Field(
        default=None, description="起始日期 YYYY-MM-DD（UTC 日粒度，含当日）；省略 = 不限。"
    )
    end_date: str | None = Field(
        default=None, description="结束日期 YYYY-MM-DD（UTC 日粒度，含当日）；省略 = 不限。"
    )
    cursor: str | None = Field(default=None, description="上一页返回的 next_cursor；首页省略。")
    limit: int = Field(default=10, ge=1, le=50, description="本页返回的消息条数上限。")


class ReadHistoryParams(ContractModel):
    """``read_history`` 参数：读取命中位置前后的历史消息（PRD 7.1）。"""

    conversation_id: str = Field(min_length=1, description="本工作区的会话 ID。")
    around_sequence: int | None = Field(
        default=None,
        ge=0,
        description="以该消息序号为中心读前后各半窗口；与 after_sequence 二选一。",
    )
    after_sequence: int = Field(default=0, ge=0, description="从该序号之后继续读取（续读用）。")
    limit: int = Field(default=20, ge=1, le=100, description="本次返回的消息条数上限。")


class WebSearchParams(ContractModel):
    """``web_search`` 参数（PRD 7.3）：Search 线索，不含网页原文。"""

    query: str = Field(min_length=1, description="检索查询；发出前经过「已脱敏参数」入口。")
    max_results: int = Field(default=5, ge=1, le=10, description="返回线索条数上限。")
    topic: Literal["general", "news"] | None = Field(
        default=None, description="检索主题；news 时按新闻场景传日期范围。"
    )
    start_date: str | None = Field(default=None, description="起始日期 YYYY-MM-DD；省略 = 不限。")
    end_date: str | None = Field(default=None, description="结束日期 YYYY-MM-DD；省略 = 不限。")
    include_published_date: bool | None = Field(
        default=None, description="是否要求提供商返回发布时间；省略 = 提供商默认。"
    )
    search_depth: Literal["basic", "advanced"] | None = Field(
        default=None, description="检索深度；省略 = PRD 7.3 默认 advanced。"
    )
    include_answer: bool | None = Field(
        default=None, description="是否要提供商生成的答案；省略 = false（PRD 7.3）。"
    )


class ReadWebPageParams(ContractModel):
    """``read_web_page`` 参数（PRD 7.3）：读取网页原文，长页分批读取。"""

    urls: list[str] = Field(
        min_length=1, max_length=5, description="要读取的 URL 列表，单次最多 5 个。"
    )
    query: str | None = Field(
        default=None, description="定位用查询；省略 = 要求整页内容（PRD 7.3）。"
    )
    offset: int = Field(default=0, ge=0, description="从该字符位置开始读取（长页续读用）。")
    limit: int = Field(
        default=READ_WEB_PAGE_DEFAULT_CHARS,
        ge=1,
        le=READ_WEB_PAGE_MAX_CHARS,
        description="本次每页读取的最大字符数，默认 20000。",
    )


class VerifyImageParams(ContractModel):
    """``verify_image`` 参数：用原图核对具体问题（PRD 7.1）。"""

    image_id: str = Field(
        min_length=1, description="本区图片 ID（图片内容哈希或图片块 block_id），跨区不可解析。"
    )
    question: str = Field(min_length=1, description="要依据原图核对的具体问题。")


class AskUserParams(ContractModel):
    """``ask_user`` 参数：向用户提出澄清问题（PRD 5.4 / 13.2）。"""

    question: str = Field(min_length=1, description="要问用户的问题，必须具体、可回答。")
    options: list[str] = Field(default_factory=list, description="可选选项；空 = 自由回答。")
    reason: str = Field(default="", description="为什么要问（阻塞了哪一步）。")


# =====================================================================
# 3. 网页证据与 artifacts
# =====================================================================


# 网页证据结构已升格为共享契约（PRD 7.3；架构文档 §6 D32），**定义在
# ``app/agent/contracts.py``**，本模块只做别名再导出（见上方导入块），不再重复定义：
#
# - T07 的 Web Agent 产出它、Synthesis / Verifier 读它、前端引用卡片展示它、
#   t11 的 A01 / A11 验收断言它 —— 都是跨模块数据，不应让下游 import 工具层内部结构；
# - 既有 ``from app.agent.tools import WebPageEvidence / WebReadRange`` 调用方**零改动**，
#   且 ``tools.WebPageEvidence is contracts.WebPageEvidence`` 成立（别名收口，已实测）。


@dataclass(frozen=True)
class ArtifactWrite:
    """一次 artifacts 落盘的结果（``ref`` 是相对 artifacts 目录的 POSIX 路径）。"""

    artifact_id: str
    ref: str
    path: Path
    content_hash: str
    size: int


def artifact_path(
    context: ToolContext,
    directory: str,
    artifact_id: str,
    *,
    suffix: str = ".md",
) -> Path:
    """工作区 artifacts 目录下的文件路径（白名单目录 + 安全 ID 校验）。"""

    return _require_artifacts_dir(context) / _require_artifact_dir(directory) / (
        _require_safe_artifact_id(artifact_id) + suffix
    )


def write_artifact(
    context: ToolContext,
    directory: str,
    artifact_id: str,
    text: str,
    *,
    suffix: str = ".md",
) -> ArtifactWrite:
    """原子写入一个 artifact（先写 ``.tmp`` 再替换，避免半截文件被读到）。"""

    path = artifact_path(context, directory, artifact_id, suffix=suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return ArtifactWrite(
        artifact_id=artifact_id,
        ref=f"{directory}/{path.name}",
        path=path,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        size=len(text),
    )


def read_artifact(context: ToolContext, ref: str) -> str:
    """读取 artifacts 内的文件（``ref`` 必须是相对路径，拒绝目录穿越）。"""

    base = _require_artifacts_dir(context)
    relative = PurePosixPath(str(ref))
    if relative.is_absolute() or ".." in relative.parts:
        raise ToolExecutionError("artifact 引用非法", code=ToolErrorCode.INTERNAL)
    candidate = (base / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ToolExecutionError("artifact 引用越界", code=ToolErrorCode.INTERNAL)
    if not candidate.is_file():
        raise ToolUnavailableError("artifact 不存在或已被清理")
    return candidate.read_text(encoding="utf-8")


def _require_artifacts_dir(context: ToolContext) -> Path:
    directory = context.artifacts_dir
    if directory is None:
        raise ToolUnavailableError("工作区上下文缺少 artifacts 目录")
    return directory


def _require_artifact_dir(directory: str) -> str:
    if directory not in _ARTIFACT_DIRS:
        raise ToolExecutionError(
            f"未知 artifacts 子目录：{directory}", code=ToolErrorCode.INTERNAL
        )
    return directory


def _require_safe_artifact_id(artifact_id: str) -> str:
    if not _SAFE_ARTIFACT_ID.match(str(artifact_id)):
        raise ToolExecutionError(
            "artifact ID 只允许 A-Za-z0-9._- 且不超过 80 字符",
            code=ToolErrorCode.INTERNAL,
        )
    return str(artifact_id)


# =====================================================================
# 4. 稳定标识与文档身份
# =====================================================================


def arguments_hash(arguments: Mapping[str, Any]) -> str:
    """参数指纹：键排序 + 紧凑分隔符的 sha256（对模型无意义，只用于身份与复用）。"""

    return hashlib.sha256(_canonical_json(arguments).encode("utf-8")).hexdigest()


def stable_operation_id(
    run_id: str | None,
    task_id: str | None,
    tool_name: str,
    args_hash: str,
) -> str:
    """稳定 ``operation_id``（PRD 12.1）：同一 run/任务 + 同工具 + 同参数 → 同一个 ID。

    重放（用户继续 / 恢复运行）因此天然命中同一条 ``operations`` 记录并复用结果。
    """

    seed = f"{run_id or ''}|{task_id or ''}|{tool_name}|{args_hash}"
    return f"op-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]}"


def document_id_of_block(block_id: str) -> str:
    """块 ID → 文档 ID（``make_block_id`` 的 ``source_hash`` 段，已脱敏、不含路径）。"""

    text = str(block_id)
    return text.split("::", 1)[0] if "::" in text else text


def resolve_document_source(vector_store: Any, document_id: str) -> str:
    """``document_id`` → 内部 ``source_file``（**绝不**交给模型 / 前端）。

    T07 组装 ``Evidence.source_ref`` 时用它把工具层的脱敏 ID 映射回真实来源。
    """

    for block_id, _text, metadata in _list_blocks(vector_store):
        if document_id_of_block(block_id) != document_id:
            continue
        source = (metadata or {}).get("source_file")
        if source:
            return str(source)
    return ""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


# =====================================================================
# 5. 执行上下文、结果与工具规格
# =====================================================================


@dataclass(frozen=True)
class ToolContext:
    """一次工具执行所绑定的**不可变**上下文（PRD 5.2）。

    ``workspace`` 是 T01 的运行期 ``app.core.workspace_context.WorkspaceContext``
    （frozen+slots，字段 ``view`` / ``paths`` / ``app_store`` / ``vector_store`` …）；
    工具从这里取 ``workspace_id``，模型可见的参数里**不出现** ``workspace_id``。

    服务句柄既可以显式传入（``vector_store`` / ``retriever`` / ``knowledge_area`` …），
    也可以留空由工作区上下文兜底解析；显式传入的 ``knowledge_area`` 若属于别的
    工作区，解析阶段直接拒绝（跨区读取的硬门）。
    """

    workspace: Any
    app_store: AppStore | None = None
    outbound: OutboundClient | None = None
    embedder: Any = None
    retriever: Any = None
    knowledge_area: Any = None
    vector_store: Any = None
    state_store: Any = None
    image_store: Any = None
    image_reader: Callable[[Any, str], str] | None = None
    search_provider: Any = None
    extract_provider: Any = None
    run_id: str | None = None
    task_id: str | None = None
    agent_kind: AgentKind | None = None
    #: 「已脱敏参数」入口的实现（T09 注入）；None = 原样返回副本。
    redactor: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None
    #: FTS 历史检索接缝（T09 装配 T08 的实现）。None = 用下面的有界回退扫描。
    #:
    #: 契约（纯数据进出，``agent`` 层不 import ``memory`` 层）::
    #:
    #:     await history_search(query, {
    #:         "workspace_id": str,          # 必给：当前工作区（来自绑定的上下文）
    #:         "limit": int,                 # 必给：本页条数上限
    #:         "cursor": str | None,         # 可选：上一页返回的 next_cursor（语义由实现方定义）
    #:         "start_date": str | None,     # 可选：YYYY-MM-DD（含当日）
    #:         "end_date": str | None,       # 可选：YYYY-MM-DD（含当日）
    #:     }) -> {
    #:         "messages": [ {message_id, conversation_id, sequence, role, created_at,
    #:                        content | excerpt | snippet, conversation_title?,
    #:                        matched_terms?} ],
    #:         "next_cursor": str | None,    # 无更多页时为 None
    #:         "total": int | None,          # 可选：分页前候选总数；未知传 None
    #:         "conversation_count": int | None,  # 可选：诊断用
    #:     }
    #:
    #: 也接受直接返回命中列表（等价于 ``{"messages": [...]}``）。
    history_search: HistorySearchCallable | None = None
    history_scan_limit: int = HISTORY_SCAN_LIMIT
    #: 回退扫描的跨会话上限（接缝为 None 时生效）；触顶即截断并在 message 中说明。
    history_max_conversations: int = HISTORY_FALLBACK_MAX_CONVERSATIONS
    history_max_messages: int = HISTORY_FALLBACK_MAX_MESSAGES
    history_max_matches: int = HISTORY_FALLBACK_MAX_MATCHES
    #: 相关性筛选运行态来源（``Callable[[], Any]`` 或运行态对象本身；集成接线）。
    #:
    #: 运行态需要三个属性：``provider`` / ``profile`` / ``config_error_code``（可选
    #: ``config_error_message``），即 ``app.api.deps.build_rerank_runtime`` 的返回值形状。
    #: ``None`` = 本次装配未接筛选 —— 与「用户未启用」等价，检索走旧路径。
    rerank: Any = None
    #: 本轮资料范围（``RetrievalScope`` 或返回它的可调用）；省略时按本区索引显式标记。
    rerank_scope: Any = None
    #: 筛选进度上报：``await rerank_progress(state, error_code)``，
    #: ``state`` ∈ ``{"running", "done", "error"}``。``None`` = 不上报（单测直连路径）。
    #: 运行层把它接成 v2 的 ``tool.started`` / ``tool.completed``（``tool_name=rerank``）。
    #:
    #: ⚠️ 装配约束：只注入本接缝（不注入 :attr:`rerank_outcome`）时，``done`` 事件的两个
    #: 计数取 payload 默认 0，界面会把「已筛选出资料」误读成 ``empty``。生产装配在
    #: ``app.agent.runtime.tool_context_for`` 里**同时**注入两个接缝，因此不会出现该状态。
    rerank_progress: Callable[[str, str], Awaitable[None]] | None = None
    #: 筛选**结果**上报（纯追加，集成 t5）：``await rerank_outcome(payload)``，
    #: ``payload`` = ``{"status", "result_count", "source_count", "sources"}`` ——
    #: ``result_count`` 是入选来源数（0 ⇒ 筛选后 0 条）、``source_count`` 是候选数、
    #: ``sources`` 是入选来源的相关性记录（键为 ``source_ref``）。运行层用它把**真实计数**
    #: 填进 ``tool.completed``（前端据此区分 ``filtered`` / ``empty``），并把 ``sources``
    #: 记下来供 ``answer.meta.citations`` 的「已筛选资料」字段使用。
    #: ``None`` = 未注入 ⇒ 退回 ``rerank_progress("done")``（接线前行为不变）。
    rerank_outcome: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None

    # ---- 身份 ----

    @property
    def workspace_id(self) -> str:
        return str(getattr(self.workspace, "workspace_id", "") or "")

    @property
    def workspace_name(self) -> str:
        return str(getattr(self.workspace, "name", "") or "")

    @property
    def knowledge_revision(self) -> int:
        value = getattr(self.workspace, "knowledge_revision", 0)
        return int(value or 0)

    @property
    def artifacts_dir(self) -> Path | None:
        value = getattr(self.workspace, "artifacts_dir", None)
        return None if value is None else Path(value)

    @property
    def permission_mode(self) -> str:
        value = getattr(self.workspace, "default_permission_mode", "")
        return str(value or "")

    @property
    def layout(self) -> str:
        value = getattr(self.workspace, "layout", "")
        return str(value or "")

    # ---- 服务句柄解析（含跨工作区硬门）----

    def resolve_knowledge_area(self) -> Any:
        """本区检索面；传入的面属于其它工作区时拒绝（跨区即不存在）。"""

        area = self.knowledge_area
        if area is None:
            return None
        area_workspace = str(getattr(getattr(area, "context", None), "workspace_id", "") or "")
        if area_workspace and self.workspace_id and area_workspace != self.workspace_id:
            raise ToolPermissionError(
                "检索面属于其它工作区，拒绝跨区读取",
                code=ToolErrorCode.WORKSPACE_MISMATCH,
            )
        return area

    def require_vector_store(self) -> Any:
        area = self.resolve_knowledge_area()
        store = self.vector_store or getattr(area, "vector_store", None)
        if store is None:
            store = getattr(self.workspace, "vector_store", None)
        if store is None:
            raise ToolUnavailableError("本工作区未绑定向量库，无法读取知识")
        return store

    def require_retriever(self) -> Any:
        area = self.resolve_knowledge_area()
        retriever = self.retriever or getattr(area, "retriever", None)
        if retriever is None:
            raise ToolUnavailableError("本工作区未绑定检索器，无法检索知识正文")
        return retriever

    def resolve_state_store(self) -> Any:
        return self.state_store or getattr(self.workspace, "state_store", None)

    def require_image_reader(self) -> Callable[[Any, str], str]:
        reader = self.image_reader
        if reader is None:
            reader = getattr(self.workspace, "image_reader", None)
        if reader is None:
            raise ToolUnavailableError("未配置识图模型，无法核对原图")
        return reader

    def require_search_provider(self) -> Any:
        if self.search_provider is None:
            raise ToolUnavailableError("联网搜索未配置或未启用")
        return self.search_provider

    def require_extract_provider(self) -> Any:
        provider = self.extract_provider or self.search_provider
        if provider is None or not callable(getattr(provider, "extract", None)):
            raise ToolUnavailableError("网页原文读取未配置或未启用")
        return provider

    # ---- 脱敏挂载点 ----

    def redact_arguments(self, tool: str | ToolName, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """「已脱敏参数」入口：返回**实际发送/落库**的参数。

        未注入 redactor 时返回浅拷贝，调用方仍可安全地按脱敏后的值出网。
        """

        raw = dict(arguments)
        if self.redactor is None:
            return raw
        redacted = self.redactor(str(tool), raw)
        return dict(redacted)


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结果（状态 + 数据 + 身份 + artifacts 引用）。"""

    tool: str
    status: ToolStatus
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    operation_id: str | None = None
    artifact_id: str = ""
    artifact_ref: str = ""
    reused: bool = False
    error_code: str = ""
    permission_checked: bool = False
    network: bool = False

    def to_payload(self) -> dict[str, Any]:
        """JSON 安全的投影（事件负载 / 模型上下文 / 操作缓存共用同一形状）。"""

        return {
            "status": str(self.status),
            "data": self.data,
            "message": self.message,
            "operation_id": self.operation_id,
            "artifact_id": self.artifact_id,
            "artifact_ref": self.artifact_ref,
            "reused": self.reused,
            "error_code": self.error_code,
            "permission_checked": self.permission_checked,
            "network": self.network,
        }

    @classmethod
    def from_payload(cls, tool: str, payload: Mapping[str, Any]) -> ToolResult:
        """从缓存负载还原（重放复用路径）。"""

        return cls(
            tool=tool,
            status=ToolStatus(str(payload.get("status") or ToolStatus.OK)),
            data=dict(payload.get("data") or {}),
            message=str(payload.get("message") or ""),
            operation_id=payload.get("operation_id"),
            artifact_id=str(payload.get("artifact_id") or ""),
            artifact_ref=str(payload.get("artifact_ref") or ""),
            reused=True,
            error_code=str(payload.get("error_code") or ""),
            permission_checked=bool(payload.get("permission_checked")),
            network=bool(payload.get("network")),
        )


ToolHandler = Callable[[Any, ToolContext], "ToolResult | Awaitable[ToolResult]"]
"""工具处理函数：接收校验后的参数模型与上下文，返回 ``ToolResult``（可为 awaitable）。"""


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的全部元信息（PRD 7.1：参数类型 / 处理函数 / 是否联网 / 允许的 Agent）。"""

    name: ToolName
    params_model: type[ContractModel]
    handler: ToolHandler
    description: str
    returns: str
    networked: bool = False
    requires_approval: bool = False
    #: 知识正文类工具：重放复用要看知识版本（PRD 7.4「同一资料版本」）。
    knowledge_scoped: bool = False
    #: 空集合 = 不限制（仅理论上用于内部工具）。
    allowed_agents: frozenset[AgentKind] = frozenset()

    def allows(self, agent_kind: AgentKind | str | None) -> bool:
        if not self.allowed_agents:
            return True
        if agent_kind is None:
            return False
        try:
            kind = AgentKind(str(agent_kind))
        except ValueError:
            return False
        return kind in self.allowed_agents

    def describe(self) -> dict[str, Any]:
        """给模型 / 事件用的工具说明（参数含义、返回内容与来源范围）。"""

        return {
            "name": str(self.name),
            "description": self.description,
            "returns": self.returns,
            "networked": self.networked,
            "requires_approval": self.requires_approval,
            "allowed_agents": sorted(str(kind) for kind in self.allowed_agents),
            "parameters": self.params_model.model_json_schema(),
        }


# =====================================================================
# 6. 处理函数（PRD 7.1 的 10 个工具）
# =====================================================================

#: Extract 的固定参数（PRD 7.3）：markdown 正文 + advanced 深度。
WEB_EXTRACT_DEPTH: Final[str] = "advanced"
WEB_EXTRACT_FORMAT: Final[str] = "markdown"

#: 工具 → 允许的 Agent 职责（PRD 4.2）。**Web Agent 只拿得到联网工具**：
#: A01 要求「Web 路径不得调用 search_knowledge（本地知识正文检索次数为 0）」，
#: 这条结构性约束就写在这里。
_READ_ONLY_AGENTS: Final[frozenset[AgentKind]] = frozenset(
    {
        AgentKind.INTENT,
        AgentKind.PLANNER,
        AgentKind.GENERAL,
        AgentKind.RAG,
        AgentKind.SYNTHESIS,
        AgentKind.VERIFIER,
    }
)
_BODY_TEXT_AGENTS: Final[frozenset[AgentKind]] = frozenset(
    {AgentKind.RAG, AgentKind.SYNTHESIS, AgentKind.VERIFIER}
)
_WEB_AGENTS: Final[frozenset[AgentKind]] = frozenset(
    {AgentKind.PLANNER, AgentKind.WEB, AgentKind.VERIFIER}
)
_IMAGE_AGENTS: Final[frozenset[AgentKind]] = frozenset({AgentKind.RAG, AgentKind.VERIFIER})
_ALL_AGENTS: Final[frozenset[AgentKind]] = frozenset(AgentKind)


@dataclass(frozen=True)
class _DocumentEntry:
    """目录里的一条文档线索（只含脱敏字段，绝不带绝对路径）。"""

    document_id: str
    name: str
    topic: str
    chunk_count: int
    document_revision: str
    headings: tuple[str, ...]
    positions: tuple[dict[str, str], ...]
    terms: tuple[str, ...]


# ---------------------------------------------------------------- 公共小工具


def _list_blocks(vector_store: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """枚举本区全部块（异常转工具错误，消息只带异常类名）。"""

    try:
        raw = vector_store.list_blocks()
    except Exception as exc:
        raise ToolExecutionError(f"读取本区索引失败：{exc.__class__.__name__}") from exc
    blocks: list[tuple[str, str, dict[str, Any]]] = []
    for item in raw or []:
        parts = list(item) + [None, None, None]
        metadata = parts[2]
        blocks.append((str(parts[0]), str(parts[1] or ""), dict(metadata or {})))
    return blocks


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meets_similarity(chunk: Any, threshold: float) -> bool:
    """命中是否达到相关度门控；缺相似度（BM25-only）视为未达标。"""

    similarity = _optional_float(getattr(chunk, "similarity", None))
    return similarity is not None and similarity >= threshold


def _clean_label(value: Any, fallback: str) -> str:
    text = _CLEAN_LABEL.sub("", str(value or "")).strip()
    return text[:120] or fallback


def _public_source_parts(source_file: Any, metadata: Mapping[str, Any]) -> tuple[str, str]:
    """source_file → (公开文档名, 主题)：只保留 basename 与直接父主题。"""

    normalized = str(source_file or "").replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("/", ".", "..")]
    name = _clean_label(parts[-1] if parts else source_file, "未命名文档")
    if len(parts) >= 2:
        topic = _clean_label(parts[-2], "未分类")
    else:
        platform = str(metadata.get("platform") or "")
        topic = _clean_label(platform, "未分类") if platform and platform != "local" else "未分类"
    return name, topic


def _query_tokens(query: str) -> list[str]:
    tokens = [token for token in _WHITESPACE.split(str(query).casefold()) if token]
    if tokens:
        return tokens
    stripped = str(query).strip().casefold()
    return [stripped] if stripped else []


def _parse_cursor(cursor: str | None) -> int:
    if cursor is None or str(cursor) == "":
        return 0
    try:
        value = int(str(cursor))
    except ValueError as exc:
        raise ToolArgumentError(
            "cursor 必须是上一页返回的 next_cursor",
            code=ToolErrorCode.INVALID_ARGUMENTS,
        ) from exc
    if value < 0:
        raise ToolArgumentError(
            "cursor 不能为负数", code=ToolErrorCode.INVALID_ARGUMENTS
        )
    return value


def _embed_texts(embedder: Any, texts: list[str]) -> list[list[float]]:
    """嵌入查询文本；按 ``embed_queries`` → ``embed_query`` → ``embed_texts`` 依次回退。"""

    embed_queries = getattr(embedder, "embed_queries", None)
    if callable(embed_queries):
        vectors = embed_queries(texts)
    else:
        embed_query = getattr(embedder, "embed_query", None)
        if callable(embed_query):
            vectors = [embed_query(text) for text in texts]
        else:
            embed_texts = getattr(embedder, "embed_texts", None)
            if not callable(embed_texts):
                raise ToolUnavailableError("嵌入模型缺少可用接口（embed_queries / embed_query）")
            vectors = embed_texts(texts)
    return [list(vector) for vector in vectors]


def _embed_and_retrieve(embedder: Any, retriever: Any, query: str) -> list[Any]:
    """查询变体嵌入 + 混合检索（与线上 Chat 走同一条多变体路径）。"""

    variants = build_query_variants(query) or (QueryVariant(kind="original", text=query),)
    vectors = _embed_texts(embedder, [variant.text for variant in variants])
    retrieve_variants = getattr(retriever, "retrieve_variants", None)
    if callable(retrieve_variants):
        return list(retrieve_variants(variants, vectors))
    return list(retriever.retrieve(variants[0].text, vectors[0]))


def _count_or_none(vector_store: Any) -> int | None:
    try:
        return int(vector_store.count())
    except Exception:  # noqa: BLE001 -- 索引句柄不可用时降级为「未知」，不阻断工具
        return None


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    """检索命中 → 工具结果（**不含** source_file 等可能承载绝对路径的键）。

    已筛选来源额外带 ``relevance_checked`` / ``rerank_score`` / ``profile_id`` /
    ``evidence_hash`` 等字段（PRD §8.3/§9）：同一批分数既用于最终选择也用于来源展示，
    未启用筛选的来源保持原字段不变（不假装筛选过）。
    """

    metadata = dict(getattr(chunk, "metadata", None) or {})
    block_id = str(getattr(chunk, "block_id", ""))
    name, topic = _public_source_parts(metadata.get("source_file"), metadata)
    payload = {
        "block_id": block_id,
        "document_id": document_id_of_block(block_id),
        "document_name": name,
        "topic": topic,
        "text": str(getattr(chunk, "context_text", None) or getattr(chunk, "text", "") or ""),
        "similarity": _optional_float(getattr(chunk, "similarity", None)),
        "match_type": str(getattr(chunk, "match_type", "") or ""),
        "heading_path": str(getattr(chunk, "heading_path", None) or ""),
        "anchor": str(getattr(chunk, "anchor", None) or ""),
        "chunk_type": str(getattr(chunk, "chunk_type", "") or ""),
        "platform": str(getattr(chunk, "platform", "") or ""),
    }
    if metadata.get("relevance_checked"):
        payload.update(
            {key: metadata.get(key) for key in rerank_pipeline.RELEVANCE_METADATA_KEYS}
        )
    return payload


def _catalog_documents(
    blocks: Sequence[tuple[str, str, dict[str, Any]]],
) -> dict[str, _DocumentEntry]:
    raw: dict[str, dict[str, Any]] = {}
    for block_id, _text, metadata in blocks:
        document_id = document_id_of_block(block_id)
        if not document_id:
            continue
        entry = raw.setdefault(
            document_id,
            {"headings": set(), "positions": [], "count": 0, "revision": "", "source": "", "platform": ""},
        )
        entry["count"] += 1
        entry["revision"] = entry["revision"] or str(metadata.get("doc_id") or "")
        entry["source"] = entry["source"] or str(metadata.get("source_file") or "")
        entry["platform"] = entry["platform"] or str(metadata.get("platform") or "")
        heading = str(metadata.get("heading_path") or "")
        anchor = str(metadata.get("anchor") or "")
        if heading:
            entry["headings"].add(heading)
        if heading or anchor:
            position = {"heading_path": heading, "anchor": anchor}
            if position not in entry["positions"] and len(entry["positions"]) < 10:
                entry["positions"].append(position)

    documents: dict[str, _DocumentEntry] = {}
    for document_id, entry in raw.items():
        name, topic = _public_source_parts(entry["source"], {"platform": entry["platform"]})
        headings = tuple(sorted(entry["headings"]))
        documents[document_id] = _DocumentEntry(
            document_id=document_id,
            name=name,
            topic=topic,
            chunk_count=entry["count"],
            document_revision=entry["revision"],
            headings=headings,
            positions=tuple(entry["positions"]),
            terms=headings,
        )
    return documents


def _catalog_match(entry: _DocumentEntry, tokens: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    haystack = " ".join([entry.name, entry.topic, *entry.headings]).casefold()
    matched = tuple(sorted({token for token in tokens if token in haystack}))
    return len(matched), matched


def _history_excerpt(content: str, tokens: Sequence[str]) -> str:
    text = _WHITESPACE.sub(" ", str(content or "")).strip()
    lowered = text.casefold()
    hits = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    start = max(0, (min(hits) if hits else 0) - 35)
    excerpt = text[start : start + 200]
    prefix = "…" if start else ""
    suffix = "…" if start + 200 < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"


def _date_in_range(created_at: str, start_date: str | None, end_date: str | None) -> bool:
    """按 UTC 日粒度（YYYY-MM-DD）过滤；时间戳为 ISO-8601 字符串，前缀比较即日期比较。"""

    day = str(created_at or "")[:10]
    if not day:
        return False
    if start_date and day < str(start_date):
        return False
    return not (end_date and day > str(end_date))


def _hit_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("snippet") or ""),
            "published_at": str(item.get("published_at") or ""),
            "score": _optional_float(item.get("score")),
        }
    return {
        "title": str(getattr(item, "title", "") or ""),
        "url": str(getattr(item, "url", "") or ""),
        "snippet": str(getattr(item, "snippet", "") or ""),
        "published_at": str(getattr(item, "published_at", "") or ""),
        "score": _optional_float(getattr(item, "score", None)),
    }


# ---------------------------------------------------------------- 1/10 目录


async def _lookup_catalog(params: LookupCatalogParams, context: ToolContext) -> ToolResult:
    """在本工作区目录（文档名 / 主题 / 标题）中查找线索，分页返回脱敏来源位置。"""

    tool = str(ToolName.LOOKUP_CATALOG)
    vector_store = context.require_vector_store()
    blocks = await asyncio.to_thread(_list_blocks, vector_store)
    documents = _catalog_documents(blocks)
    if not documents:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="本工作区目录为空（尚未登记任何文档）",
            data={
                "documents": [],
                "matched_terms": [],
                "catalog_terms": [],
                "total": 0,
                "next_cursor": None,
                "knowledge_revision": context.knowledge_revision,
            },
        )

    offset = _parse_cursor(params.cursor)
    tokens = _query_tokens(params.query)
    scored: list[tuple[int, tuple[str, ...], _DocumentEntry]] = []
    terms: set[str] = set()
    for entry in documents.values():
        terms.update(entry.terms)
        score, matched = _catalog_match(entry, tokens)
        if score:
            scored.append((score, matched, entry))
    catalog_terms = sorted(
        {term for term in terms if any(token in term.casefold() or term.casefold() in token for token in tokens)}
    )[:20]

    if not scored:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="目录中没有匹配的文档或术语",
            data={
                "documents": [],
                "matched_terms": list(tokens),
                "catalog_terms": catalog_terms,
                "total": 0,
                "next_cursor": None,
                "knowledge_revision": context.knowledge_revision,
            },
        )

    scored.sort(key=lambda item: (-item[0], item[2].name.casefold(), item[2].document_id))
    page = scored[offset : offset + params.limit]
    next_offset = offset + params.limit
    next_cursor = str(next_offset) if next_offset < len(scored) else None
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data={
            "documents": [
                {
                    "document_id": entry.document_id,
                    "name": entry.name,
                    "topic": entry.topic,
                    "chunk_count": entry.chunk_count,
                    "document_revision": entry.document_revision,
                    "headings": list(entry.headings),
                    "positions": [dict(position) for position in entry.positions],
                    "matched_terms": list(matched),
                    "score": score,
                }
                for score, matched, entry in page
            ],
            "matched_terms": sorted({token for _s, matched, _e in page for token in matched}),
            "catalog_terms": catalog_terms,
            "total": len(scored),
            "returned": len(page),
            "next_cursor": next_cursor,
            "knowledge_revision": context.knowledge_revision,
        },
        message=f"命中 {len(scored)} 个文档线索",
    )


# ---------------------------------------------------------------- 2/10 概览


async def _get_workspace_overview(
    params: GetWorkspaceOverviewParams, context: ToolContext
) -> ToolResult:
    """本工作区概览：已登记 / 已索引文档数、主题、图片与当前 run 的任务状态。"""

    tool = str(ToolName.GET_WORKSPACE_OVERVIEW)
    categories = list(params.categories) or ["knowledge", "documents", "topics", "images", "tasks"]
    vector_store = context.require_vector_store()
    overview = await asyncio.to_thread(_build_overview, vector_store)

    data: dict[str, Any] = {
        "workspace": {
            "workspace_id": context.workspace_id,
            "name": context.workspace_name,
            "knowledge_revision": context.knowledge_revision,
            "permission_mode": context.permission_mode,
            "layout": context.layout,
        },
        "categories": categories,
    }
    if "knowledge" in categories:
        data["knowledge"] = {
            "file_count": overview.file_count,
            "chunk_count": overview.chunk_count,
            "text_chunk_count": overview.text_chunk_count,
            "image_chunk_count": overview.image_chunk_count,
            "chunk_types": dict(overview.chunk_types),
        }
    if "documents" in categories:
        data["documents"] = {
            "registered": _registered_documents(context),
            "indexed": overview.file_count,
        }
    if "topics" in categories:
        data["topics"] = [topic.model_dump() for topic in overview.topics]
        data["representative_documents"] = [
            document.model_dump() for document in overview.representative_documents
        ]
    if "images" in categories:
        data["images"] = {
            "image_chunk_count": overview.image_chunk_count,
            "indexed_images": _count_or_none_image(vector_store),
        }
    if "tasks" in categories:
        data["tasks"] = await _task_overview(context)

    return ToolResult(tool=tool, status=ToolStatus.OK, data=data)


def _build_overview(vector_store: Any) -> KnowledgeOverview:
    try:
        return build_knowledge_overview(vector_store)
    except Exception as exc:
        raise ToolExecutionError(f"生成知识概览失败：{exc.__class__.__name__}") from exc


def _count_or_none_image(vector_store: Any) -> int | None:
    try:
        return int(vector_store.count_images())
    except Exception:  # noqa: BLE001 -- 同上：拿不到图片计数时如实返回「未知」
        return None


def _registered_documents(context: ToolContext) -> int | None:
    """已登记文档数（state.db 指纹表）；拿不到时如实返回 None，不猜。"""

    store = context.resolve_state_store()
    load_all = getattr(store, "load_all", None)
    if not callable(load_all):
        return None
    try:
        return len(load_all())
    except Exception:  # noqa: BLE001 -- 指纹表不可用时返回 None（不猜数字）
        return None


async def _task_overview(context: ToolContext) -> dict[str, Any]:
    """当前 run 的任务状态分布（无 run 上下文时如实标注不可用）。"""

    if context.app_store is None or not context.run_id:
        return {"available": False, "run_id": context.run_id, "by_status": {}, "total": 0}
    tasks = await context.app_store.list_tasks(context.run_id)
    counts: dict[str, int] = {}
    for task in tasks:
        key = str(task.status)
        counts[key] = counts.get(key, 0) + 1
    return {
        "available": True,
        "run_id": context.run_id,
        "by_status": dict(sorted(counts.items())),
        "total": len(tasks),
    }


# ---------------------------------------------------------------- 3/10 知识检索


async def _search_knowledge(params: SearchKnowledgeParams, context: ToolContext) -> ToolResult:
    """本工作区知识正文检索：返回片段 / 分数 / 来源 ID / 知识版本（PRD 7.2）。

    启用**相关性筛选**时，候选块在工具内完成评分与筛选（集成接线：main 的筛选语义从
    V1.1.0 的 ``/api/chat`` 迁到 V1.2.0 的 run 管线，核心算法见
    :mod:`app.retrieval.rerank_pipeline`）。四类状态与 main 的 ChatService 路径逐条对齐：

    * 未启用 → 旧路径不变（不产出任何 rerank 进度事件）；
    * 已筛选 → 只返回入选来源，``chunks`` 携带 ``relevance_checked`` 等元数据；
    * 有候选但全未达阈值 → ``irrelevant``（**不**回落未筛选候选，PRD §8.4）；
    * 已启用但配置不可用 / 评分失败 → 抛工具错误终止本轮并指向设置，绝不静默返回未筛选内容。

    进度走运行层注入的 ``context.rerank_progress``（v2 ``tool.started`` /
    ``tool.completed``，``tool_name=rerank``）；筛选**结束后**若注入了
    ``context.rerank_outcome``，则改由它上报**真实计数与入选来源**
    （``result_count`` = 入选数、``source_count`` = 候选数 ⇒ 前端得以区分
    ``filtered`` 与 ``empty``；同批记录供 ``answer.meta.citations`` 的「已筛选资料」
    字段使用）。两个接缝都未注入时（单测直连）静默跳过。
    """

    tool = str(ToolName.SEARCH_KNOWLEDGE)
    vector_store = context.require_vector_store()
    retriever = context.require_retriever()
    if context.embedder is None:
        raise ToolUnavailableError("未配置嵌入模型，无法检索知识正文")

    if _count_or_none(vector_store) == 0:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="本工作区尚未建立索引",
            data={"code": "index_empty", "chunks": [], "count": 0, "knowledge_revision": context.knowledge_revision},
        )

    # 运行态快照先解析：`configured`（用户已启用，无论配置是否可用）决定是否上报进度。
    snapshot = rerank_pipeline.resolve_runtime(context.rerank)
    filtering = rerank_pipeline.runtime_configured(snapshot)
    if filtering:
        # 评分期间不得生成正式答案：先上报进度，再执行筛选。
        await _report_rerank_progress(context, "running")

    try:
        chunks = await asyncio.to_thread(
            _embed_and_retrieve, context.embedder, retriever, params.query
        )
    except RetrievalError as exc:
        raise ToolExecutionError(str(exc)) from exc

    raw_count = len(chunks)
    if params.document_scope:
        scope = {str(item) for item in params.document_scope}
        chunks = [chunk for chunk in chunks if document_id_of_block(getattr(chunk, "block_id", "")) in scope]
    if params.min_similarity is not None:
        chunks = [chunk for chunk in chunks if _meets_similarity(chunk, params.min_similarity)]

    outcome: rerank_pipeline.RerankOutcome | None = None
    if filtering:
        # 评分与选择只看到**调用方真正要的那批候选**（范围/门控先收窄，再排序）。
        outcome = await rerank_pipeline.filter_chunks(
            chunks=chunks,
            query=params.query,
            runtime=snapshot,
            vectorstore=vector_store,
            scope=context.rerank_scope,
        )
        if outcome.failed:
            await _report_rerank_progress(context, "error", outcome.error_code)
            raise _rerank_failure(outcome)
        chunks = list(outcome.chunks)

    if not chunks:
        if outcome is not None:
            # 真实计数必须先上报：``result_count == 0`` 正是前端判「筛选后 0 条」的依据。
            await _report_rerank_outcome(context, outcome)
            if not outcome.candidate_count:
                return ToolResult(
                    tool=tool,
                    status=ToolStatus.EMPTY,
                    message="知识检索没有命中任何片段",
                    data={
                        "code": "no_results",
                        "chunks": [],
                        "count": 0,
                        "candidate_count": 0,
                        "retrieval": outcome.to_dict(),
                        "knowledge_revision": context.knowledge_revision,
                    },
                )
            return ToolResult(
                tool=tool,
                status=ToolStatus.IRRELEVANT,
                message="候选片段均未通过相关性筛选，视为资料无关",
                data={
                    "code": "relevance_empty",
                    "chunks": [],
                    "count": 0,
                    "candidate_count": outcome.candidate_count,
                    "retrieval": outcome.to_dict(),
                    "knowledge_revision": context.knowledge_revision,
                },
            )
        status = ToolStatus.IRRELEVANT if raw_count else ToolStatus.EMPTY
        message = (
            "有候选片段但均低于相关度门控，视为资料无关"
            if raw_count
            else "知识检索没有命中任何片段"
        )
        return ToolResult(
            tool=tool,
            status=status,
            message=message,
            data={
                "code": "irrelevant" if raw_count else "no_results",
                "chunks": [],
                "count": 0,
                "candidate_count": raw_count,
                "knowledge_revision": context.knowledge_revision,
            },
        )

    selected = chunks[: params.top_k]
    data: dict[str, Any] = {
        "query": params.query,
        "chunks": [_chunk_payload(chunk) for chunk in selected],
        "count": len(selected),
        "candidate_count": raw_count if outcome is None else outcome.candidate_count,
        "knowledge_revision": context.knowledge_revision,
    }
    if outcome is not None:
        data["retrieval"] = outcome.to_dict()
        # 入选数 > 0 ⇒ 前端可判 ``filtered``；同批记录供引用卡片使用。
        await _report_rerank_outcome(context, outcome)
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data=data,
        message=f"命中 {len(selected)} 个知识片段",
    )


async def _report_rerank_progress(context: ToolContext, state: str, error_code: str = "") -> None:
    """上报一次筛选进度（未注入接缝时什么都不做）。

    刻意**不吞**发布异常：进度事件是用户可见状态，写不进去就是运行层故障，必须如实失败，
    不能让一次「看起来成功」的检索掩盖事件流断点。
    """

    sink = context.rerank_progress
    if sink is None:
        return
    await sink(state, error_code)


async def _report_rerank_outcome(
    context: ToolContext, outcome: rerank_pipeline.RerankOutcome
) -> None:
    """上报一次筛选**结果**：真实计数（可判 filtered / empty）+ 入选来源的相关性记录。

    * 注入了 ``context.rerank_outcome``（运行层装配）→ 走它：运行层把 ``result_count`` /
      ``source_count`` 填进**同一个已声明事件** ``tool.completed``（不新增事件类型），
      并把 ``sources`` 记下来，供 ``answer.meta.citations`` 的「已筛选资料」字段使用；
    * 未注入（单测直连 / 旧装配）→ 退回 ``rerank_progress("done")``，行为与接线前一致。
    """

    sink = context.rerank_outcome
    if sink is None:
        await _report_rerank_progress(context, "done")
        return
    await sink(
        {
            "status": outcome.status,
            "result_count": int(outcome.selected_count),
            "source_count": int(outcome.candidate_count),
            "sources": _rerank_source_records(outcome.chunks),
        }
    )


def _rerank_source_records(chunks: Sequence[Any]) -> list[dict[str, Any]]:
    """入选来源 → 相关性记录（键为 ``source_ref``，与 ``Evidence.source_ref`` 同一口径）。

    ``source_ref`` 必须是 ``f"{document_id}#{block_id}"``：``RagResearcher._collect_retrieval``
    构造证据时用的就是这个口径，运行层因此能按 ``evidence.source_ref`` 把记录接到引用上
    （PRD §8.3）。值全部取自**本次真实的筛选产物**（视图元数据），不做任何补造。
    """

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        block_id = str(getattr(chunk, "block_id", "") or "")
        if not block_id:
            continue
        metadata = dict(getattr(chunk, "metadata", None) or {})
        records.append(
            {
                "source_ref": f"{document_id_of_block(block_id)}#{block_id}",
                "relevance_checked": bool(metadata.get("relevance_checked")),
                "rerank_score": _optional_float(metadata.get("rerank_score")),
                "score_type": str(metadata.get("score_type") or ""),
                "profile_id": str(metadata.get("profile_id") or ""),
                "evidence_hash": str(metadata.get("evidence_hash") or ""),
                "member_refs": [
                    dict(item)
                    for item in (metadata.get("member_refs") or [])
                    if isinstance(item, Mapping)
                ],
            }
        )
    return records


def _rerank_failure(outcome: rerank_pipeline.RerankOutcome) -> ToolError:
    """筛选失败 → 工具错误（终止本轮；PRD §8.4 禁止静默返回未筛选内容）。

    具体的筛选错误码（如 ``RERANK_AUTH_FAILED``）由进度事件 ``tool.completed.error_code``
    透出；工具错误码用共用的分类口径（``not_configured`` / ``service_error``），
    让运行层按既有语义判定「可继续 / 不可继续」，不新增枚举取值。
    """

    message = str(outcome.error_message or rerank_pipeline.RERANK_FAILED_MESSAGE)
    if outcome.error_code in _RERANK_CONFIG_ERROR_CODES:
        return ToolUnavailableError(message, tool=str(ToolName.SEARCH_KNOWLEDGE))
    return ToolExecutionError(message, tool=str(ToolName.SEARCH_KNOWLEDGE))


#: 「已启用但配置不可用」类错误码：按「未配置」上报（可修复后重跑）。
_RERANK_CONFIG_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"RERANK_NOT_CONFIGURED", "RERANK_PROFILE_MISMATCH", "RERANK_NOT_ENABLED", "RERANK_DISABLED"}
)


# ---------------------------------------------------------------- 4/10 读原文


async def _read_document(params: ReadDocumentParams, context: ToolContext) -> ToolResult:
    """读取指定文档的原文位置（标题 / 锚点 / 可继续读取的位置）。"""

    tool = str(ToolName.READ_DOCUMENT)
    vector_store = context.require_vector_store()
    blocks = await asyncio.to_thread(_list_blocks, vector_store)
    mine = [
        (block_id, text, metadata)
        for block_id, text, metadata in blocks
        if document_id_of_block(block_id) == params.document_id
    ]
    if not mine:
        return ToolResult(
            tool=tool,
            status=ToolStatus.NOT_FOUND,
            message="本工作区没有该文档",
            data={"code": "document_not_found", "document_id": params.document_id},
        )

    mine.sort(key=lambda item: (str(item[2].get("heading_path") or ""), item[0]))
    if params.heading:
        needle = params.heading.casefold()
        mine = [
            item
            for item in mine
            if needle in str(item[2].get("heading_path") or "").casefold()
        ]
        if not mine:
            return ToolResult(
                tool=tool,
                status=ToolStatus.EMPTY,
                message="该文档没有匹配的章节",
                data={
                    "code": "heading_not_found",
                    "document_id": params.document_id,
                    "heading": params.heading,
                },
            )

    full = "\n\n".join(text for _block_id, text, _metadata in mine)
    total = len(full)
    if params.offset >= total and total:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="已读到文档末尾",
            data={
                "code": "end_of_document",
                "document_id": params.document_id,
                "total_chars": total,
                "offset": params.offset,
            },
        )
    window = full[params.offset : params.offset + params.limit]
    read_end = params.offset + len(window)
    name, topic = _public_source_parts(mine[0][2].get("source_file"), mine[0][2])
    headings = tuple(
        dict.fromkeys(str(metadata.get("heading_path") or "") for *_rest, metadata in mine)
    )
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data={
            "document_id": params.document_id,
            "document_name": name,
            "topic": topic,
            "document_revision": str(mine[0][2].get("doc_id") or ""),
            "block_count": len(mine),
            "total_chars": total,
            "offset": params.offset,
            "text": window,
            "headings": [heading for heading in headings if heading],
            "next_offset": read_end if read_end < total else None,
            "knowledge_revision": context.knowledge_revision,
        },
    )


# ---------------------------------------------------------------- 5/10 历史检索


async def _search_history(params: SearchHistoryParams, context: ToolContext) -> ToolResult:
    """检索本工作区历史会话消息（本区数据 + 来源）。

    两条实现**对外 schema 完全一致**（``messages[] / total / returned / next_cursor /
    conversation_count``）：

    1. 接了 FTS 接缝（``context.history_search``，T09 装配 T08 的
       ``app.memory.history``）→ 完全委托，本层不再遍历会话；
    2. 未接 → **有界回退扫描**（每会话 + 跨会话总量 + 命中数三重上限），
       触顶即截断并在 ``message`` 中说明。
    """

    if not context.workspace_id:
        raise ToolUnavailableError("执行上下文缺少 workspace_id，无法检索历史")
    if context.history_search is not None:
        return await _search_history_via_index(params, context)
    return await _search_history_fallback(params, context)


async def _search_history_via_index(
    params: SearchHistoryParams, context: ToolContext
) -> ToolResult:
    """委托给 FTS 接缝（PRD 11.5：FTS5 检索原文；分词/AND/bm25 排序由实现方决定）。"""

    tool = str(ToolName.SEARCH_HISTORY)
    seam = context.history_search
    assert seam is not None  # 调用方已判空；仅为类型收窄
    request: dict[str, Any] = {
        "workspace_id": context.workspace_id,
        "limit": params.limit,
        "cursor": params.cursor,
        "start_date": params.start_date,
        "end_date": params.end_date,
    }
    try:
        payload = await seam(params.query, request)
    except ToolError:
        raise
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise ToolExecutionError(
            f"历史索引检索失败：{detail}",
            code=ToolErrorCode.SERVICE_ERROR,
            tool=tool,
        ) from exc

    raw_messages, next_cursor, total, conversation_count = _unpack_history_index_result(payload)
    tokens = _query_tokens(params.query)
    messages = [_history_message_payload(item, tokens) for item in raw_messages]
    await _fill_conversation_titles(messages, context)

    if not messages:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="本工作区历史中没有匹配的消息（FTS 索引检索）",
            data={
                "messages": [],
                "total": 0 if total is None else total,
                "next_cursor": next_cursor,
                "conversation_count": conversation_count,
            },
        )

    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data={
            "messages": messages,
            "total": total,
            "returned": len(messages),
            "next_cursor": next_cursor,
            "conversation_count": conversation_count,
        },
        message=f"命中 {len(messages)} 条历史消息（FTS 索引检索）",
    )


async def _search_history_fallback(
    params: SearchHistoryParams, context: ToolContext
) -> ToolResult:
    """未接 FTS 时的**有界**回退扫描（会话数 / 消息数 / 命中数三重上限）。"""

    tool = str(ToolName.SEARCH_HISTORY)
    store = context.app_store
    if store is None:
        raise ToolUnavailableError("未绑定 app.db，且未接 FTS 历史检索接缝，无法检索历史")
    workspace_id = context.workspace_id
    if not workspace_id:
        raise ToolUnavailableError("执行上下文缺少 workspace_id，无法检索历史")

    offset = _parse_cursor(params.cursor)
    tokens = _query_tokens(params.query)
    conversations = await store.list_conversations(workspace_id)
    matches: list[dict[str, Any]] = []
    scanned_conversations = 0
    scanned_messages = 0
    truncated = False

    for conversation in conversations:
        if scanned_conversations >= context.history_max_conversations:
            truncated = True  # 跨会话上限
            break
        if scanned_messages >= context.history_max_messages:
            truncated = True  # 跨会话消息总量上限
            break
        scanned_conversations += 1
        messages = await store.list_messages(
            conversation.id, limit=context.history_scan_limit
        )
        if len(messages) >= context.history_scan_limit:
            # 保守判定：本会话可能还有更多消息未读（不额外查一次 COUNT）。
            truncated = True
        for message in messages:
            if scanned_messages >= context.history_max_messages:
                truncated = True  # 跨会话消息总量上限（会话内同样生效）
                break
            scanned_messages += 1
            if message.role not in ("user", "assistant"):
                continue
            if not _date_in_range(message.created_at, params.start_date, params.end_date):
                continue
            matched = tuple(
                token for token in tokens if token in str(message.content or "").casefold()
            )
            if not matched:
                continue
            matches.append(
                {
                    "conversation_id": conversation.id,
                    "conversation_title": conversation.title,
                    "message_id": message.id,
                    "sequence": message.sequence,
                    "role": message.role,
                    "created_at": message.created_at,
                    "excerpt": _history_excerpt(message.content, matched),
                    "matched_terms": list(matched),
                }
            )
            if len(matches) >= context.history_max_matches:
                truncated = True  # 命中数上限：够分页了，停止继续扫
                break
        if len(matches) >= context.history_max_matches:
            break
        if scanned_messages >= context.history_max_messages:
            break

    truncation_note = (
        "（回退扫描已达上限，结果可能不完整；接入 FTS 索引后无该上限）" if truncated else ""
    )

    if not matches:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message=f"本工作区历史中没有匹配的消息{truncation_note}",
            data={
                "messages": [],
                "total": 0,
                "next_cursor": None,
                "conversation_count": scanned_conversations,
            },
        )

    # 三轮**稳定**排序：命中词数降序 → 时间降序 → message_id 升序（完全确定）。
    matches.sort(key=lambda item: str(item["message_id"]))
    matches.sort(key=lambda item: str(item["created_at"]), reverse=True)
    matches.sort(key=lambda item: -len(item["matched_terms"]))

    page = matches[offset : offset + params.limit]
    next_offset = offset + params.limit
    next_cursor = str(next_offset) if next_offset < len(matches) else None
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data={
            "messages": page,
            "total": len(matches),
            "returned": len(page),
            "next_cursor": next_cursor,
            "conversation_count": scanned_conversations,
        },
        message=f"命中 {len(matches)} 条历史消息（回退扫描）{truncation_note}",
    )


def _unpack_history_index_result(
    payload: Any,
) -> tuple[list[Any], str | None, int | None, int | None]:
    """解析 FTS 接缝的返回值（接受 ``{messages,...}`` 或直接给命中列表）。"""

    if isinstance(payload, Mapping):
        raw_messages = payload.get("messages")
        next_cursor = payload.get("next_cursor")
        total = payload.get("total")
        conversation_count = payload.get("conversation_count")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        raw_messages, next_cursor, total, conversation_count = payload, None, None, None
    else:
        raise ToolExecutionError(
            "历史索引接缝返回值不是消息列表或对象",
            code=ToolErrorCode.INTERNAL,
            tool=str(ToolName.SEARCH_HISTORY),
        )
    messages = list(raw_messages or [])
    if not all(isinstance(item, Mapping) for item in messages):
        raise ToolExecutionError(
            "历史索引接缝返回的消息项不是对象",
            code=ToolErrorCode.INTERNAL,
            tool=str(ToolName.SEARCH_HISTORY),
        )
    return (
        messages,
        None if next_cursor is None else str(next_cursor),
        None if total is None else int(total),
        None if conversation_count is None else int(conversation_count),
    )


def _history_message_payload(item: Mapping[str, Any], tokens: Sequence[str]) -> dict[str, Any]:
    """把接缝返回的一条命中映射成工具 schema（键与回退路径完全一致）。"""

    content = str(item.get("content") or "")
    excerpt = str(item.get("excerpt") or item.get("snippet") or "")
    if not excerpt:
        excerpt = _history_excerpt(content, tokens)
    matched = item.get("matched_terms")
    if matched:
        matched_terms = [str(term) for term in matched]
    else:
        haystack = (content or excerpt).casefold()
        matched_terms = [token for token in tokens if token in haystack]
    return {
        "conversation_id": str(item.get("conversation_id") or ""),
        "conversation_title": str(item.get("conversation_title") or ""),
        "message_id": str(item.get("message_id") or ""),
        "sequence": int(item.get("sequence") or 0),
        "role": str(item.get("role") or ""),
        "created_at": str(item.get("created_at") or ""),
        "excerpt": excerpt,
        "matched_terms": matched_terms,
    }


async def _fill_conversation_titles(
    messages: list[dict[str, Any]], context: ToolContext
) -> None:
    """补会话标题（接缝未提供时，按页内去重后的会话 ID 做主键查询）；失败不影响检索。"""

    store = context.app_store
    if store is None:
        return
    missing = {
        str(message["conversation_id"])
        for message in messages
        if message["conversation_id"] and not message["conversation_title"]
    }
    titles: dict[str, str] = {}
    for conversation_id in sorted(missing):
        try:
            conversation = await store.get_conversation(conversation_id)
        except Exception:  # noqa: BLE001 -- 标题只是展示增益，绝不阻断检索
            return
        if conversation is None or conversation.workspace_id != context.workspace_id:
            continue
        titles[conversation_id] = conversation.title
    for message in messages:
        if not message["conversation_title"]:
            message["conversation_title"] = titles.get(message["conversation_id"], "")


# ---------------------------------------------------------------- 6/10 历史原文


async def _read_history(params: ReadHistoryParams, context: ToolContext) -> ToolResult:
    """读取命中位置前后的历史消息（跨区会话等同不存在）。"""

    tool = str(ToolName.READ_HISTORY)
    store = context.app_store
    if store is None:
        raise ToolUnavailableError("未绑定 app.db，无法读取历史")
    conversation = await store.get_conversation(params.conversation_id)
    if conversation is None or conversation.workspace_id != context.workspace_id:
        return ToolResult(
            tool=tool,
            status=ToolStatus.NOT_FOUND,
            message="本工作区没有该会话",
            data={"code": "conversation_not_found", "conversation_id": params.conversation_id},
        )

    if params.around_sequence is not None:
        half = max(1, params.limit // 2)
        after = max(0, params.around_sequence - half - 1)
    else:
        after = params.after_sequence

    messages = await store.list_messages(
        params.conversation_id, after_sequence=after, limit=params.limit
    )
    if not messages:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="该位置之后没有更多消息",
            data={
                "code": "no_more_messages",
                "conversation_id": params.conversation_id,
                "after_sequence": after,
            },
        )

    last_sequence = messages[-1].sequence
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        data={
            "conversation_id": conversation.id,
            "conversation_title": conversation.title,
            "workspace_id": conversation.workspace_id,
            "after_sequence": after,
            "messages": [
                {
                    "message_id": message.id,
                    "sequence": message.sequence,
                    "role": message.role,
                    "created_at": message.created_at,
                    "content": message.content,
                }
                for message in messages
            ],
            "returned": len(messages),
            "next_after_sequence": last_sequence if len(messages) == params.limit else None,
        },
    )


# ---------------------------------------------------------------- 7/10 联网搜索


async def _web_search(params: WebSearchParams, context: ToolContext) -> ToolResult:
    """联网搜索线索（PRD 7.3）：过「已脱敏参数」后出网，返回 URL / 摘要 / 日期 / 请求 ID。"""

    tool = str(ToolName.WEB_SEARCH)
    provider = context.require_search_provider()
    redacted = context.redact_arguments(ToolName.WEB_SEARCH, params.model_dump(mode="json"))
    query = str(redacted.get("query") or params.query)
    kwargs: dict[str, Any] = {
        "topic": redacted.get("topic"),
        "start_date": redacted.get("start_date"),
        "end_date": redacted.get("end_date"),
        "include_published_date": redacted.get("include_published_date"),
    }
    if redacted.get("search_depth") is not None:
        kwargs["search_depth"] = redacted["search_depth"]
    if redacted.get("include_answer") is not None:
        kwargs["include_answer"] = redacted["include_answer"]

    search_web = getattr(provider, "search_web", None)
    try:
        if callable(search_web):
            response = await search_web(query, params.max_results, **kwargs)
            request_id = str(getattr(response, "request_id", "") or "")
            raw_results = list(getattr(response, "results", ()) or ())
        else:
            request_id = ""
            raw_results = list(await provider.search(query, params.max_results))
    except SearchError as exc:
        raise ToolExecutionError(
            str(exc), code=ToolErrorCode.SERVICE_ERROR, tool=tool
        ) from exc

    hits = [_hit_payload(item) for item in raw_results]
    data = {
        "query": query,
        "request_id": request_id,
        "max_results": params.max_results,
        "results": hits,
        "count": len(hits),
        "redacted": context.redactor is not None,
    }
    if not hits:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="联网搜索没有返回结果",
            data=data | {"code": "no_results"},
        )
    return ToolResult(tool=tool, status=ToolStatus.OK, data=data, network=True)


# ---------------------------------------------------------------- 8/10 网页原文


async def _read_web_page(params: ReadWebPageParams, context: ToolContext) -> ToolResult:
    """读取网页原文（PRD 7.3）：逐项检查 results / failed_results，长页写 artifacts。"""

    tool = str(ToolName.READ_WEB_PAGE)
    provider = context.require_extract_provider()
    redacted = context.redact_arguments(ToolName.READ_WEB_PAGE, params.model_dump(mode="json"))
    urls = [str(url) for url in (redacted.get("urls") or params.urls)]
    query = redacted.get("query", params.query)
    if query is not None and not str(query).strip():
        query = None

    try:
        response = await provider.extract(
            urls,
            query=None if query is None else str(query),
            extract_depth=WEB_EXTRACT_DEPTH,
            format=WEB_EXTRACT_FORMAT,
        )
    except SearchError as exc:
        raise ToolExecutionError(str(exc), code=ToolErrorCode.SERVICE_ERROR, tool=tool) from exc

    retrieved_at = utc_now_iso()
    evidences: list[WebPageEvidence] = []
    for page in list(getattr(response, "pages", ()) or ()):
        content = str(getattr(page, "content", "") or "")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""
        artifact_id = ""
        if len(content) > WEB_ARTIFACT_MIN_CHARS:
            written = write_artifact(
                context, WEB_ARTIFACT_DIR, content_hash, content, suffix=".md"
            )
            artifact_id = written.artifact_id
        window = content[params.offset : params.offset + params.limit]
        covers_all = params.offset == 0 and len(window) == len(content)
        evidences.append(
            WebPageEvidence(
                url=str(getattr(page, "url", "") or ""),
                final_url=str(getattr(page, "final_url", "") or ""),
                title=str(getattr(page, "title", "") or ""),
                published_at=str(getattr(page, "published_at", "") or ""),
                retrieved_at=retrieved_at,
                content_hash=content_hash,
                artifact_id=artifact_id,
                read_range=WebReadRange(
                    offset=params.offset,
                    length=len(window),
                    total=len(content),
                    complete=covers_all,
                ),
                content_complete=bool(getattr(page, "content_complete", False)) and covers_all,
                content=window,
            )
        )

    failed = [
        {
            "url": str(getattr(item, "url", "") or ""),
            "error": str(getattr(item, "error", "") or ""),
        }
        for item in list(getattr(response, "failed", ()) or ())
    ]
    artifact_ids = [evidence.artifact_id for evidence in evidences if evidence.artifact_id]
    data = {
        "request_id": str(getattr(response, "request_id", "") or ""),
        "pages": [evidence.model_dump() for evidence in evidences],
        "failed": failed,
        "count": len(evidences),
        "failed_count": len(failed),
        "offset": params.offset,
        "limit": params.limit,
        "network": True,
    }
    if not evidences:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message=(
                "逐项内容检查：所有 URL 都没有读到正文"
                if failed
                else "网页读取没有返回正文"
            ),
            data=data | {"code": "no_content"},
            artifact_id=",".join(artifact_ids),
            network=True,
        )
    incomplete = bool(failed) or any(not evidence.content_complete for evidence in evidences)
    return ToolResult(
        tool=tool,
        status=ToolStatus.PARTIAL if incomplete else ToolStatus.OK,
        message="部分网页读取失败或正文不完整，按逐项结果使用" if incomplete else "",
        data=data,
        artifact_id=",".join(artifact_ids),
        artifact_ref=f"{WEB_ARTIFACT_DIR}/{artifact_ids[0]}.md" if artifact_ids else "",
        network=True,
    )


# ---------------------------------------------------------------- 9/10 原图核对


async def _verify_image(params: VerifyImageParams, context: ToolContext) -> ToolResult:
    """用本区原图核对具体问题（跨区图片等同不存在）。"""

    tool = str(ToolName.VERIFY_IMAGE)
    vector_store = context.require_vector_store()
    blocks = await asyncio.to_thread(_list_blocks, vector_store)
    match: tuple[str, str, dict[str, Any]] | None = None
    for block_id, text, metadata in blocks:
        kind = str(metadata.get("chunk_type") or "")
        source_kind = str(metadata.get("source_type") or "")
        if kind != "image_description" and source_kind != "image_description":
            continue
        image_hash = str(metadata.get("image_content_hash") or "")
        if block_id == params.image_id or (image_hash and image_hash == params.image_id):
            match = (block_id, text, metadata)
            break
    if match is None:
        return ToolResult(
            tool=tool,
            status=ToolStatus.NOT_FOUND,
            message="本工作区没有该图片",
            data={"code": "image_not_found", "image_id": params.image_id},
        )

    reader = context.require_image_reader()
    block_id, text, metadata = match
    redacted = context.redact_arguments(
        ToolName.VERIFY_IMAGE, {"image_id": params.image_id, "question": params.question}
    )
    question = str(redacted.get("question") or params.question)
    chunk = RetrievedChunk(
        block_id=block_id,
        text=text,
        similarity=None,
        source_file=str(metadata.get("source_file") or ""),
        platform=str(metadata.get("platform") or "local"),
        chunk_type="image_description",
        anchor=metadata.get("anchor"),
        heading_path=metadata.get("heading_path"),
        metadata=metadata,
    )
    try:
        verification = await asyncio.to_thread(reader, chunk, question)
    except Exception as exc:
        raise ToolExecutionError(
            f"原图核对失败：{exc.__class__.__name__}", tool=tool
        ) from exc

    text_out = str(verification or "").strip()
    name, _topic = _public_source_parts(metadata.get("source_file"), metadata)
    data = {
        "image_id": params.image_id,
        "image_content_hash": str(metadata.get("image_content_hash") or ""),
        "document_id": document_id_of_block(block_id),
        "document_name": name,
        "heading_path": str(metadata.get("heading_path") or ""),
        "question": question,
        "verification": text_out,
        "retrieved_at": utc_now_iso(),
        "knowledge_revision": context.knowledge_revision,
    }
    if not text_out:
        return ToolResult(
            tool=tool,
            status=ToolStatus.EMPTY,
            message="原图核对没有返回可用内容",
            data=data | {"code": "empty_verification"},
        )
    return ToolResult(tool=tool, status=ToolStatus.OK, data=data)


# ---------------------------------------------------------------- 10/10 澄清


async def _ask_user(params: AskUserParams, context: ToolContext) -> ToolResult:
    """向用户提出澄清问题：登记 ``input_requests`` 并返回待澄清状态（PRD 5.4 / 13.2）。"""

    tool = str(ToolName.ASK_USER)
    store = context.app_store
    if store is None:
        raise ToolUnavailableError("未绑定 app.db，无法创建澄清请求")
    if not context.run_id:
        raise ToolUnavailableError("执行上下文缺少 run_id，无法创建澄清请求")

    request = InputRequest(
        input_request_id=new_id("input"),
        run_id=context.run_id,
        task_id=context.task_id,
        question=params.question,
        options=list(params.options),
        reason=params.reason,
        status=InputRequestStatus.PENDING,
        question_version=1,
        created_at=utc_now_iso(),
    )
    saved = await store.request_input(request)
    return ToolResult(
        tool=tool,
        status=ToolStatus.WAITING_INPUT,
        message="已向用户提出澄清问题，等待回答",
        data={
            "input_request_id": saved.input_request_id,
            "run_id": saved.run_id,
            "task_id": saved.task_id,
            "question": saved.question,
            "options": list(saved.options),
            "reason": saved.reason,
            "question_version": saved.question_version,
            "status": str(saved.status),
        },
    )


# =====================================================================
# 7. 注册表（PRD 7.1：JSON 解析 → 校验 → 权限 → operation_id → 执行）
# =====================================================================


class ToolRegistry:
    """工具注册表：注册元信息、解析参数、门控 Agent、登记操作与复用结果。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ---- 注册与查询 ----

    def register(self, spec: ToolSpec) -> ToolSpec:
        key = str(spec.name)
        if key in self._specs:
            raise ValueError(f"工具名重复注册：{key}")
        self._specs[key] = spec
        return spec

    def get(self, name: str | ToolName) -> ToolSpec:
        spec = self._specs.get(str(name))
        if spec is None:
            raise ToolArgumentError(
                f"未知工具：{name}", code=ToolErrorCode.UNKNOWN_TOOL, tool=str(name)
            )
        return spec

    def has(self, name: str | ToolName) -> bool:
        return str(name) in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def describe(self) -> list[dict[str, Any]]:
        """工具说明全集（交给模型 / 事件层；含参数 schema 与来源范围）。"""

        return [spec.describe() for spec in self._specs.values()]

    def allows(self, name: str | ToolName, agent_kind: AgentKind | str | None) -> bool:
        return self.get(name).allows(agent_kind)

    # ---- 参数解析 ----

    def parse_arguments(
        self, name: str | ToolName | ToolSpec, arguments: str | Mapping[str, Any]
    ) -> Any:
        """标准 JSON 解析 + Pydantic 校验（失败抛 :class:`ToolArgumentError`）。"""

        spec = name if isinstance(name, ToolSpec) else self.get(name)
        if isinstance(arguments, str):
            raw = arguments.strip() or "{}"
            try:
                parsed: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolArgumentError(
                    f"{spec.name} 的参数不是合法 JSON：{exc.msg}",
                    code=ToolErrorCode.INVALID_JSON,
                    tool=str(spec.name),
                ) from exc
        else:
            parsed = dict(arguments)
        if not isinstance(parsed, Mapping):
            raise ToolArgumentError(
                f"{spec.name} 的参数必须是 JSON 对象",
                code=ToolErrorCode.INVALID_JSON,
                tool=str(spec.name),
            )
        try:
            return spec.params_model.model_validate(dict(parsed))
        except ValidationError as exc:
            raise ToolArgumentError(
                f"{spec.name} 的参数不符合 schema：{_validation_summary(exc)}",
                code=ToolErrorCode.INVALID_ARGUMENTS,
                tool=str(spec.name),
            ) from exc

    def redact_arguments(
        self,
        name: str | ToolName,
        arguments: str | Mapping[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        """「已脱敏参数」公开入口：解析 + 校验 + 脱敏后的参数（供批准卡片等展示/绑定）。"""

        spec = self.get(name)
        params = self.parse_arguments(spec, arguments)
        return context.redact_arguments(spec.name, params.model_dump(mode="json"))

    # ---- 执行 ----

    async def execute(
        self,
        name: str | ToolName,
        arguments: str | Mapping[str, Any],
        context: ToolContext,
        *,
        reuse: bool = True,
        record_operation: bool = True,
    ) -> ToolResult:
        """执行一次工具调用（PRD 7.1 / 12.1）。

        顺序刻意固定：解析 → 校验 → **权限门控** → 登记 ``operation_id`` → 重放复用 →
        执行 → 落盘结果。权限门控在任何检索 / 出网之前，A01 的「Web 路径 0 次本地正文
        检索」因此不可能被绕过。
        """

        spec = self.get(name)
        params = self.parse_arguments(spec, arguments)
        args_hash = arguments_hash(params.model_dump(mode="json"))
        operation_id = stable_operation_id(
            context.run_id, context.task_id, str(spec.name), args_hash
        )
        permission_checked = self._check_permission(spec, context)
        store = context.app_store if record_operation else None

        if store is not None:
            existing = await store.register_operation(
                operation_id=operation_id,
                tool_name=str(spec.name),
                args_hash=args_hash,
                run_id=context.run_id,
                task_id=context.task_id,
            )
            if reuse and existing.status is OperationStatus.SUCCEEDED:
                cached = self._load_cached(context, spec, operation_id, args_hash)
                if cached is not None:
                    return replace(cached, permission_checked=permission_checked)
            await store.start_operation_attempt(operation_id)

        try:
            produced = spec.handler(params, context)
            result = await produced if inspect.isawaitable(produced) else produced
        except ToolError as exc:
            await self._finish_failed(store, operation_id)
            if not exc.tool:
                exc.tool = str(spec.name)
            raise
        except Exception as exc:
            await self._finish_failed(store, operation_id)
            raise ToolExecutionError(
                f"{spec.name} 执行失败：{exc.__class__.__name__}",
                code=ToolErrorCode.INTERNAL,
                tool=str(spec.name),
            ) from exc

        if not isinstance(result, ToolResult):
            await self._finish_failed(store, operation_id)
            raise ToolExecutionError(
                f"{spec.name} 没有返回 ToolResult",
                code=ToolErrorCode.INTERNAL,
                tool=str(spec.name),
            )

        final = replace(
            result,
            tool=str(spec.name),
            operation_id=operation_id,
            permission_checked=permission_checked,
            network=spec.networked,
        )
        ref = self._store_result(context, spec, operation_id, args_hash, final)
        if store is not None:
            await store.finish_operation(
                operation_id, OperationStatus.SUCCEEDED, result_ref=ref or None
            )
        return final

    # ---- 内部 ----

    def _check_permission(self, spec: ToolSpec, context: ToolContext) -> bool:
        """返回是否真的校验过 Agent 身份；未绑定身份时如实返回 False。"""

        kind = context.agent_kind
        if kind is None:
            return False
        if not spec.allows(kind):
            raise ToolPermissionError(
                f"Agent {kind} 不允许使用工具 {spec.name}",
                code=ToolErrorCode.NOT_PERMITTED,
                tool=str(spec.name),
            )
        return True

    def _cache_ref(self, operation_id: str) -> str:
        return f"{OPERATIONS_ARTIFACT_DIR}/{operation_id}.json"

    def _store_result(
        self,
        context: ToolContext,
        spec: ToolSpec,
        operation_id: str,
        args_hash: str,
        result: ToolResult,
    ) -> str:
        """把结果写进操作缓存；返回相对 artifacts 的引用（写不了就返回空串，不阻断）。"""

        if context.artifacts_dir is None:
            return ""
        payload = {
            "tool": str(spec.name),
            "args_hash": args_hash,
            "knowledge_revision": context.knowledge_revision,
            "created_at": utc_now_iso(),
            "result": result.to_payload(),
        }
        try:
            written = write_artifact(
                context,
                OPERATIONS_ARTIFACT_DIR,
                operation_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                suffix=".json",
            )
        except (ToolError, OSError):
            return ""
        return written.ref

    def _load_cached(
        self,
        context: ToolContext,
        spec: ToolSpec,
        operation_id: str,
        args_hash: str,
    ) -> ToolResult | None:
        """读取可复用的操作结果；参数或知识版本不一致时返回 None（重新执行）。"""

        if context.artifacts_dir is None:
            return None
        try:
            text = read_artifact(context, self._cache_ref(operation_id))
        except ToolError:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("args_hash") or "") != args_hash:
            return None
        if spec.knowledge_scoped and int(payload.get("knowledge_revision") or 0) != context.knowledge_revision:
            return None
        cached = ToolResult.from_payload(
            str(spec.name), payload.get("result") or {}
        )
        return replace(cached, operation_id=operation_id)

    async def _finish_failed(self, store: AppStore | None, operation_id: str) -> None:
        if store is None:
            return
        try:
            await store.finish_operation(operation_id, OperationStatus.FAILED)
        except Exception:  # noqa: BLE001 -- 失败标记本身不能再抛出，否则会盖掉原始错误
            return


def _validation_summary(exc: ValidationError) -> str:
    """把 Pydantic 错误压成简短、无路径的说明（最多 3 条）。"""

    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "参数"
        parts.append(f"{location}: {error.get('msg', '')}")
    return "；".join(parts) or "参数不合法"


# =====================================================================
# 8. 默认注册表（PRD 7.1 的 10 个工具）
# =====================================================================


def build_default_registry() -> ToolRegistry:
    """构建带 10 个工具的标准注册表（每次调用返回独立实例）。"""

    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name=ToolName.LOOKUP_CATALOG,
            params_model=LookupCatalogParams,
            handler=_lookup_catalog,
            description=(
                "在本工作区知识目录中查找文档、术语与标题线索（不含正文）。"
                "query=要查找的线索；cursor=上一页的 next_cursor；limit=本页条数。"
                "返回 document_id / 公开文档名 / 主题 / 标题位置；"
                "空表示目录无匹配，不是错误。"
            ),
            returns="documents[]（document_id、name、topic、headings、positions）、next_cursor",
            knowledge_scoped=True,
            allowed_agents=_READ_ONLY_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.GET_WORKSPACE_OVERVIEW,
            params_model=GetWorkspaceOverviewParams,
            handler=_get_workspace_overview,
            description=(
                "取本工作区概览。categories 取 knowledge / documents / topics / images / tasks；"
                "省略 = 全部。documents.registered 来自 state.db 指纹表，indexed 来自本区索引；"
                "tasks 只统计当前 run 的任务状态。"
            ),
            returns="workspace / knowledge / documents / topics / images / tasks",
            allowed_agents=_READ_ONLY_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.SEARCH_KNOWLEDGE,
            params_model=SearchKnowledgeParams,
            handler=_search_knowledge,
            description=(
                "检索**本工作区知识正文**。query=检索查询；top_k=返回片段数上限（默认 8）；"
                "document_scope=限定 document_id 列表；min_similarity=相关度门控。"
                "返回片段、相似度、document_id 与知识版本；no_results / irrelevant 是状态。"
                "Web Agent 不允许调用本工具（A01）。"
            ),
            returns="chunks[]（text、similarity、document_id、heading_path）、knowledge_revision",
            knowledge_scoped=True,
            allowed_agents=_BODY_TEXT_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.READ_DOCUMENT,
            params_model=ReadDocumentParams,
            handler=_read_document,
            description=(
                "读取指定文档的原文位置。document_id=目录返回的 ID；heading=章节标题片段；"
                "offset/limit=从哪读多少字符（续读用 next_offset）。"
                "返回原文窗口、文档版本、标题清单与可继续读取的位置。"
            ),
            returns="text、document_revision、headings、next_offset",
            knowledge_scoped=True,
            allowed_agents=_BODY_TEXT_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.SEARCH_HISTORY,
            params_model=SearchHistoryParams,
            handler=_search_history,
            description=(
                "检索**本工作区**历史会话消息。query=检索词；start_date/end_date=YYYY-MM-DD 日期范围；"
                "cursor=分页。已接 FTS5 索引时走索引检索（jieba cut_for_search 分词 + 相关性排序）；"
                "未接索引时回退为**有界扫描**（每会话 2000 条、最多 200 个会话 / 5000 条消息 / 500 条命中），"
                "触顶即截断并在 message 里说明；回退态下多词为任一命中即候选、命中词多者排前。"
                "返回会话 / 消息 ID、角色、时间与摘要片段。"
            ),
            returns="messages[]（conversation_id、message_id、role、created_at、excerpt）",
            allowed_agents=_READ_ONLY_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.READ_HISTORY,
            params_model=ReadHistoryParams,
            handler=_read_history,
            description=(
                "读取历史消息上下文。conversation_id=会话 ID；around_sequence=以该序号为中心读前后；"
                "after_sequence=向后继续读；limit=条数。跨工作区会话返回 not_found。"
            ),
            returns="messages[]（sequence、role、content）、next_after_sequence",
            allowed_agents=_READ_ONLY_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.WEB_SEARCH,
            params_model=WebSearchParams,
            handler=_web_search,
            description=(
                "联网搜索线索（不是原文）。query=检索查询；max_results=条数；"
                "topic=general/news；start_date/end_date=日期范围；include_published_date=是否要发布时间；"
                "search_depth / include_answer 省略时按 PRD 7.3 用 advanced / false。"
                "参数先过「已脱敏参数」入口再出网；返回 URL、摘要、日期与搜索请求 ID。"
            ),
            returns="results[]（title、url、snippet、published_at、score）、request_id",
            networked=True,
            requires_approval=True,
            allowed_agents=_WEB_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.READ_WEB_PAGE,
            params_model=ReadWebPageParams,
            handler=_read_web_page,
            description=(
                "读取网页原文。urls=URL 列表（≤5）；query 省略 = 要求整页内容；"
                "offset/limit=长页分批读取。逐项检查 results 与 failed_results；"
                "超长正文写入工作区 artifacts 并返回 artifact_id 与读取范围。"
            ),
            returns="pages[]（url、final_url、title、published_at、content_hash、artifact_id、read_range）、failed[]",
            networked=True,
            requires_approval=True,
            allowed_agents=_WEB_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.VERIFY_IMAGE,
            params_model=VerifyImageParams,
            handler=_verify_image,
            description=(
                "用本区原图核对具体问题。image_id=本区图片 ID（内容哈希或图片块 ID）；"
                "question=要核对的问题。返回原图核对结果与来源；跨区图片返回 not_found。"
            ),
            returns="verification、document_id、retrieved_at",
            allowed_agents=_IMAGE_AGENTS,
        )
    )
    registry.register(
        ToolSpec(
            name=ToolName.ASK_USER,
            params_model=AskUserParams,
            handler=_ask_user,
            description=(
                "向用户提出澄清问题并登记 input_requests。question=问题；options=可选项；"
                "reason=为什么问。返回 waiting_input 与 input_request_id，由运行层暂停等待回答。"
            ),
            returns="input_request_id、question、options、status=waiting_input",
            allowed_agents=_ALL_AGENTS,
        )
    )
    return registry
