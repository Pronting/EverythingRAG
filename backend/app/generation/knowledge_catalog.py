"""工作区知识目录与知识概览（PRD 6.1、5.3.4）。Owner: T06（routing）。

本模块有两层，职责不同但同源（都从**本区**向量库块枚举出来）：

1. **知识概览** :class:`KnowledgeOverview` / :func:`build_knowledge_overview`
   —— 面向前端的公开统计投影（文件、块、图片、主题、代表文档），不含任何路径。
2. **知识目录** :class:`KnowledgeCatalog` / :func:`build_knowledge_catalog`
   —— 面向路由与 `lookup_catalog` 的定位索引（PRD 6.1）：全部文档的标题、标题
   层级、主题、术语、别名与来源 ID；查询支持精确词、别名与全文检索。

四条硬约束（PRD 6.1）：

- 目录只用于**寻找原文**，本身不构成答案依据；内部术语条目必须附实际文档位置
  （``document_id`` + ``heading_path``）。
- 缺少明确解释的术语标 ``definition_unknown``，不得当成已定义概念。
- 模型提取的术语与别名必须由**程序**校验引用片段确实存在
  （:func:`apply_term_candidates`），片段不存在即拒绝整条候选。
- 完整目录保存在本地；进入模型提示的只有与本轮相关的结果
  （:meth:`KnowledgeCatalog.prompt_projection`）。

缓存范围按 PRD 5.3.4：``workspace_id`` + ``knowledge_revision`` + 索引指纹 +
生成器版本（由 ``WorkspaceContext.index_cache_key`` 提供），因此导入 / 同步 / 删除
提交后递增 ``knowledge_revision`` 即自动失效重建，无需调用方手动清理。

绝对路径只存在于私有时刻映射 ``_source_files``（用于按来源 ID 读原文）；任何对外
投影（提示、工具结果、数据集、报告）都不含路径。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

__all__ = [
    "CATALOG_DEFAULT_LIMIT",
    "CATALOG_GENERATOR_VERSION",
    "CATALOG_SCHEMA_VERSION",
    "CatalogDocument",
    "CatalogHeading",
    "CatalogMatch",
    "CatalogPromptProjection",
    "CatalogQueryResult",
    "CatalogTerm",
    "KnowledgeCatalog",
    "KnowledgeOverview",
    "KnowledgeTopic",
    "RepresentativeDocument",
    "TermApplicationResult",
    "TermCandidate",
    "TermRejection",
    "apply_term_candidates",
    "build_knowledge_catalog",
    "build_knowledge_overview",
    "build_term_extraction_prompt",
    "catalog_cache_key",
    "catalog_cache_size",
    "catalog_for_area",
    "clear_catalog_cache",
    "invalidate_catalog",
    "lookup_catalog_query",
    "normalize_catalog_text",
    "parse_term_candidates",
]


# =====================================================================
# 1. 知识概览（既有公开投影，行为保持不变）
# =====================================================================


class KnowledgeTopic(BaseModel):
    name: str
    file_count: int
    chunk_count: int


class RepresentativeDocument(BaseModel):
    name: str
    topic: str
    chunk_count: int


class KnowledgeOverview(BaseModel):
    file_count: int = 0
    chunk_count: int = 0
    text_chunk_count: int = 0
    image_chunk_count: int = 0
    chunk_types: dict[str, int] = Field(default_factory=dict)
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    representative_documents: list[RepresentativeDocument] = Field(default_factory=list)

    def to_prompt_context(self) -> str:
        """模型只接收公开投影；原始 source_file 永远不会进入概览上下文。"""

        return (
            json.dumps(self.model_dump(), ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )


def build_knowledge_overview(vectorstore: Any) -> KnowledgeOverview:
    """单次枚举块并聚合文件、块类型、主题和代表文档，排序结果完全确定。"""

    blocks = vectorstore.list_blocks()
    type_counts: dict[str, int] = {}
    file_counts: dict[str, int] = {}
    file_names: dict[str, str] = {}
    file_topics: dict[str, str] = {}
    topic_chunks: dict[str, int] = {}
    topic_files: dict[str, set[str]] = {}
    image_chunks = 0

    for _block_id, _text, metadata in blocks:
        meta = metadata if isinstance(metadata, dict) else {}
        chunk_type = _clean_label(meta.get("chunk_type"), fallback="text")
        source_type = str(meta.get("source_type") or "")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        if source_type == "image_description" or chunk_type == "image_description":
            image_chunks += 1

        raw_source = str(meta.get("source_file") or "")
        if not raw_source:
            continue
        document_name, topic = _public_source_parts(raw_source, meta)
        file_counts[raw_source] = file_counts.get(raw_source, 0) + 1
        file_names[raw_source] = document_name
        file_topics[raw_source] = topic
        topic_chunks[topic] = topic_chunks.get(topic, 0) + 1
        topic_files.setdefault(topic, set()).add(raw_source)

    topics = [
        KnowledgeTopic(
            name=name,
            file_count=len(topic_files[name]),
            chunk_count=topic_chunks[name],
        )
        for name in sorted(topic_chunks, key=lambda item: (-topic_chunks[item], item.casefold()))[:12]
    ]
    representatives = [
        RepresentativeDocument(
            name=file_names[source],
            topic=file_topics[source],
            chunk_count=count,
        )
        for source, count in sorted(
            file_counts.items(),
            key=lambda item: (
                -item[1],
                file_names[item[0]].casefold(),
                file_topics[item[0]].casefold(),
            ),
        )[:12]
    ]

    return KnowledgeOverview(
        file_count=len(file_counts),
        chunk_count=len(blocks),
        text_chunk_count=max(0, len(blocks) - image_chunks),
        image_chunk_count=image_chunks,
        chunk_types=dict(sorted(type_counts.items(), key=lambda item: item[0].casefold())),
        topics=topics,
        representative_documents=representatives,
    )


#: 不能作为「主题」暴露的路径片段（盘符、家目录、用户目录）。
_PRIVATE_TOPIC_RE = re.compile(r"^(?:[a-z]:|~|users|home)$", re.IGNORECASE)


def _public_source_parts(
    raw_source: str,
    metadata: Mapping[str, Any],
    *,
    strip_extension: bool = False,
) -> tuple[str, str]:
    """只返回 basename 与直接父主题；盘符、用户目录和完整路径不会向外暴露。

    ``strip_extension=True``（目录用）再去掉 ``.md`` 一类来源后缀：目录标题是**匹配
    与提示**用的名字，不是文件清单。保留后缀会让「整串等于标题」这条线索永远无法成立
    （用户不会在提问里打 ``.md``），等于把最可靠的一条线索作废。
    """

    normalized = raw_source.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("/", ".", "..")]
    name = _clean_label(parts[-1] if parts else raw_source, fallback="未命名文档")
    if strip_extension:
        name = _DOCUMENT_EXTENSION_RE.sub("", name) or name
    topic = ""
    if len(parts) >= 2 and not _PRIVATE_TOPIC_RE.match(parts[-2].strip()):
        topic = _clean_label(parts[-2], fallback="")
    if not topic:
        platform = str(metadata.get("platform") or "")
        topic = _clean_label(platform, fallback="未分类") if platform != "local" else "未分类"
    return name, topic


def _clean_label(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", "", str(value or "")).strip()
    return text[:120] or fallback


# =====================================================================
# 2. 目录常量与离线助手
# =====================================================================

#: 目录数据自身的版本（结构或派生规则变化时递增，参与缓存键）。
CATALOG_SCHEMA_VERSION: Final[int] = 1

#: 目录生成器版本；与 ``knowledge_revision``、索引指纹一起决定缓存失效（PRD 5.3.4）。
CATALOG_GENERATOR_VERSION: Final[str] = "catalog-v1"

#: 单个文档保留的正文样例长度（仅用于离线派生术语；不进入任何对外投影）。
CATALOG_SAMPLE_MAX_CHARS: Final[int] = 4_000

#: 目录上限，保证提示与内存可控。
CATALOG_MAX_TERMS: Final[int] = 400
CATALOG_MAX_HEADINGS_PER_DOC: Final[int] = 64

CATALOG_DEFAULT_LIMIT: Final[int] = 8
CATALOG_MAX_LIMIT: Final[int] = 50
CATALOG_SNIPPET_MAX_CHARS: Final[int] = 240

#: 判定「标题下是否给出明确解释」的句式标记。
_DEFINITION_MARKERS: Final[tuple[str, ...]] = (
    "是",
    "指",
    "表示",
    "含义",
    "定义为",
    "称为",
    "叫做",
    "即",
    "：",
    ":",
)

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_WS_RE = re.compile(r"\s+")
#: 目录文档标题要去掉的来源文件后缀（标题用于匹配与提示，不是文件清单）。
_DOCUMENT_EXTENSION_RE = re.compile(r"\.(?:md|markdown|mdx|txt)$", re.IGNORECASE)
#: 「标签整体出现在问题里」这一档（800/700）要求的最小标签长度：1–2 字的泛化标题
#: （「方案」「总结」）本身没有区分度，授权「本区有这份资料」的依据不成立。
_MIN_SUBSTRING_LABEL_CHARS = 3
#: 提问句式标签：聊天导出里常出现把提问句当小节的标题（「怎么做」）。它们**不指代
#: 任何资料主题**，因此即便长度够也不许走 800/700 档（否则「X 怎么做」会凭空获得
#: 「本区有资料」的授权）。真实主题（「召回率」「实体链接」）不在此列。
_LOW_INFORMATION_LABELS: Final[frozenset[str]] = frozenset(
    {
        "是什么",
        "什么是",
        "为什么",
        "为何",
        "怎么",
        "怎么样",
        "怎么做",
        "怎么实现",
        "如何",
        "如何做",
        "如何实现",
        "怎样",
        "哪些",
        "有哪些",
        "多少",
        "什么",
    }
)
#: 部分标题命中的最小公共中文串长度（问题与标题/小节标题的连续公共子串）。
_MIN_SHARED_RUN_CHARS = 4
#: 部分标题命中的基准分（= 路由的线索门槛，见 ``agent/routing.py::CATALOG_CLUE_MIN_SCORE``，
#: 两处必须保持一致：低于门槛的命中不会授权 R1）。
_PARTIAL_TITLE_CLUE_SCORE = 400
#: 覆盖强度加成上限（公共中文二元组个数）：整体仍低于「整串等于标题」的 500 档。
_MAX_CLUE_COVERAGE_BONUS = 30
#: 线索排序门槛：非术语条目落在 ``[门槛, 强命中)`` 这一带时属于「定位线索」，在 ``search``
#: 里排在**其它非线索条目**之前。为什么需要：术语带 +50 加成且数量远超文档/标题，会把
#: 真正的 400 分文档线索挤出调用方的 top-K（``IntentAgent.enrich`` 只取 6 条 ⇒ 该走知识库
#: 的问题落回通用知识；实测「物化视图新鲜度怎么保证」）。
_CLUE_RANK_MIN_SCORE = 400
#: 强命中下限（精确词 / 别名 / 整串等于标题）：这一档**不动**原有排序语义（分数高者先，
#: 同分按 kind），避免把「精确术语命中」挤到标题条目之后（``lookup_catalog`` 的既有约定）。
_STRONG_MATCH_SCORE = 500


def _has_shared_cjk_run(query_norm: str, label_norm: str, size: int) -> bool:
    """问题与标签是否存在长度 ≥ ``size`` 的**连续公共中文串**（部分标题命中）。

    用连续串而不是词元集合：词元（bigram/trigram）重叠在长标题上极易偶然成立，
    而「≥4 个连续汉字同时出现在问题与标题里」几乎只在用户真的点到某份资料时发生
    （「货拉拉的记忆系统…」↔「货拉拉大模型记忆系统（一）…」的公共串「记忆系统」）。
    """

    if not query_norm or not label_norm or size <= 0:
        return False
    for run in _CJK_RUN_RE.findall(query_norm):
        if len(run) < size:
            continue
        for start in range(len(run) - size + 1):
            if run[start : start + size] in label_norm:
                return True
    return False


def _shared_cjk_bigram_count(query_norm: str, label_norm: str) -> int:
    """问题与标签之间**不同**的公共中文二元组数量 —— 部分标题命中的强度。

    为什么需要强度而不是「命中/不命中」：同一个问题常常与**多份**资料都有 ≥4 字公共串
    （「记忆系统」这种领域词同时出现在若干标题/小节里）。命中的公共二元组越多，说明用户
    点的越可能是这份资料（例：「货拉拉的记忆系统…」与目标标题有 货拉/拉拉/记忆/忆系/系统
    五个，与另一篇只有 记忆/忆系/系统 三个）—— 用它把**真正被问到的那份**排到最前。
    """

    if not query_norm or not label_norm:
        return 0
    grams: set[str] = set()
    for run in _CJK_RUN_RE.findall(query_norm):
        grams.update(run[start : start + 2] for start in range(len(run) - 1))
    return sum(1 for gram in grams if gram in label_norm)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_catalog_text(value: Any) -> str:
    """匹配用规范化：NFKC + casefold + 空白折叠（不改变对外展示文本）。"""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _WS_RE.sub(" ", text).strip()


def _display_text(value: Any, *, limit: int = CATALOG_SNIPPET_MAX_CHARS) -> str:
    text = _WS_RE.sub(" ", str(value or "")).strip()
    return text[:limit]


def _tokens(text: str) -> set[str]:
    """确定性的全文检索词集合：ASCII 词 + CJK 二元/三元组（不依赖分词器）。"""

    normalized = normalize_catalog_text(text)
    tokens = set(_ASCII_TOKEN_RE.findall(normalized))
    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) == 1:
            tokens.add(run)
            continue
        for size in (2, 3):
            for start in range(len(run) - size + 1):
                tokens.add(run[start : start + size])
    return tokens


def _document_id_for(source_file: str) -> str:
    """稳定的来源 ID：规范化路径的 SHA-256 前 16 位（路径本身不外泄）。"""

    normalized = unicodedata.normalize("NFC", str(source_file).replace("\\", "/")).casefold()
    return "doc-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _sentence_at(text: str, normalized_position: int) -> str:
    """按规范化位置近似定位原句（规范化不改变长度比例，只做近似切片）。"""

    offset = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not sentence:
            continue
        plain = _display_text(sentence)
        if not plain:
            continue
        normalized = normalize_catalog_text(sentence)
        if offset <= normalized_position < offset + max(len(normalized), 1):
            return plain
        offset += len(normalized) + 1
    return ""


def _find_definition(term: str, aliases: Sequence[str], text: str) -> tuple[str, bool]:
    """在正文里寻找术语的明确解释；找不到返回 ``("", False)``。

    判定条件（可离线复核）：正文出现该术语或别名，其所在句子含解释性句式标记，且
    句子长度明显超过术语本身。
    """

    if not text:
        return "", False
    haystack = normalize_catalog_text(text)
    for needle in (term, *aliases):
        wanted = normalize_catalog_text(needle)
        if len(wanted) < 2:
            continue
        position = haystack.find(wanted)
        if position < 0:
            continue
        plain = _sentence_at(text, position)
        if len(plain) <= len(needle):
            continue
        if any(marker in plain for marker in _DEFINITION_MARKERS):
            return plain, True
    return "", False


def _parse_aliases(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, (list, tuple)):
        return [_display_text(item, limit=300) for item in raw if str(item).strip()]
    text = str(raw)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return [_display_text(text, limit=300)] if text.strip() else []
    if isinstance(decoded, list):
        return [_display_text(item, limit=300) for item in decoded if str(item).strip()]
    return []


def _dedupe_labels(values: Sequence[str], *, exclude: str = "") -> list[str]:
    wanted = normalize_catalog_text(exclude)
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        label = _display_text(value, limit=120)
        key = normalize_catalog_text(label)
        if not label or key == wanted or key in seen:
            continue
        seen.add(key)
        result.append(label)
    return result


def _term_key(term: CatalogTerm) -> tuple[str, str, str]:
    return (normalize_catalog_text(term.term), term.document_id, term.heading_path)


def _dedupe_terms(terms: Sequence[CatalogTerm]) -> list[CatalogTerm]:
    seen: set[tuple[str, str, str]] = set()
    result: list[CatalogTerm] = []
    for term in sorted(terms, key=lambda item: _term_key(item)):
        key = _term_key(term)
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


# =====================================================================
# 3. 目录模型（extra="forbid"：与共享契约一致）
# =====================================================================


class _CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogHeading(_CatalogModel):
    """文档内的一个标题节点（层级 + 路径 + 锚点）。"""

    path: str
    level: int = Field(default=1, ge=1, le=6)
    title: str = ""
    anchor: str = ""
    chunk_count: int = Field(default=0, ge=0)
    aliases: list[str] = Field(default_factory=list)


class CatalogDocument(_CatalogModel):
    """目录里的一个文档（不含任何绝对路径）。"""

    document_id: str
    title: str
    topic: str = "未分类"
    source_type: str = "document"
    platform: str = "local"
    chunk_count: int = Field(default=0, ge=0)
    content_id: str = ""
    headings: list[CatalogHeading] = Field(default_factory=list)


class CatalogTerm(_CatalogModel):
    """内部术语条目：必须能定位到实际文档位置（PRD 6.1）。"""

    term: str
    aliases: list[str] = Field(default_factory=list)
    document_id: str
    heading_path: str = ""
    snippet: str = ""
    definition_unknown: bool = True
    evidence_verified: bool = False
    origin: Literal["heading", "model", "manual"] = "heading"

    def definition_text(self) -> str:
        """术语条目的展示片段（缺明确解释时给位置说明）。"""

        if self.snippet:
            return _display_text(self.snippet)
        return f"{self.term}（位置：{self.heading_path or '（无标题）'}）"


class CatalogMatch(_CatalogModel):
    """一次目录查询命中的条目（按用途区分 term / heading / document）。"""

    kind: Literal["term", "heading", "document"]
    document_id: str
    title: str = ""
    heading_path: str = ""
    snippet: str = ""
    score: int = Field(default=0, ge=0)
    matched_on: str = ""
    definition_unknown: bool = False
    aliases: list[str] = Field(default_factory=list)


class CatalogQueryResult(_CatalogModel):
    """目录查询结果（分页稳定：同分按来源 ID 与标题排序）。"""

    query: str
    matches: list[CatalogMatch] = Field(default_factory=list)
    total: int = 0
    next_cursor: int | None = None


class CatalogPromptProjection(_CatalogModel):
    """进入模型提示的**相关**目录片段；完整目录留在本地（PRD 6.1）。"""

    text: str = ""
    document_ids: list[str] = Field(default_factory=list)
    term_count: int = 0
    truncated: bool = False


class TermCandidate(_CatalogModel):
    """模型给出的术语候选（未校验）。"""

    term: str
    aliases: list[str] = Field(default_factory=list)
    document_id: str = ""
    heading_path: str = ""
    snippet: str = ""
    definition: str = ""


class TermRejection(_CatalogModel):
    """被程序拒绝的术语候选及原因（可审计，不静默丢弃）。"""

    term: str
    reason: str


class TermApplicationResult(_CatalogModel):
    """术语候选校验结果：接受 / 拒绝 / 标记为缺少解释。"""

    accepted: list[CatalogTerm] = Field(default_factory=list)
    rejected: list[TermRejection] = Field(default_factory=list)
    definition_unknown: list[str] = Field(default_factory=list)


# =====================================================================
# 4. KnowledgeCatalog
# =====================================================================


class _CatalogContext(Protocol):
    """目录缓存只依赖这几个成员（不导入 T01 模块，避免层次倒置）。"""

    @property
    def workspace_id(self) -> str: ...

    @property
    def storage_key(self) -> str: ...

    @property
    def paths(self) -> Any: ...

    def index_cache_key(self, *, index_fingerprint: str, generator_version: str) -> str: ...


class KnowledgeCatalog(_CatalogModel):
    """一个工作区的知识目录（PRD 6.1）。

    ``documents`` / ``terms`` 是唯一对外投影；绝对路径与正文样例保存在私有属性里
    （``_source_files`` / ``_document_text``），只用于按来源 ID 读原文与离线派生术语。
    """

    schema_version: int = CATALOG_SCHEMA_VERSION
    generator_version: str = CATALOG_GENERATOR_VERSION
    workspace_id: str = ""
    knowledge_revision: int = Field(default=0, ge=0)
    built_at: str = ""
    documents: list[CatalogDocument] = Field(default_factory=list)
    terms: list[CatalogTerm] = Field(default_factory=list)

    _source_files: dict[str, str] = PrivateAttr(default_factory=dict)
    _document_text: dict[str, str] = PrivateAttr(default_factory=dict)
    _dirty: bool = PrivateAttr(default=True)
    _index: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    # ---------------------------------------------------------------- 内部索引

    def mark_dirty(self) -> None:
        """术语或文档变化后重建查询索引（幂等）。"""

        self._dirty = True

    def _ensure_index(self) -> None:
        entries: list[dict[str, Any]] = []
        titles = {document.document_id: document.title for document in self.documents}
        for term in self.terms:
            blob = " ".join([term.term, *term.aliases, term.heading_path])
            entries.append(
                {
                    "kind": "term",
                    "document_id": term.document_id,
                    "title": titles.get(term.document_id, ""),
                    "heading_path": term.heading_path,
                    "matched_on": term.term,
                    "snippet": term.definition_text(),
                    "definition_unknown": term.definition_unknown,
                    "aliases": list(term.aliases),
                    "tokens": _tokens(blob),
                }
            )
        for document in self.documents:
            entries.append(
                {
                    "kind": "document",
                    "document_id": document.document_id,
                    "title": document.title,
                    "heading_path": "",
                    "matched_on": document.title,
                    "snippet": f"{document.title}（{document.topic}，{document.chunk_count} 块）",
                    "definition_unknown": False,
                    "aliases": [],
                    "tokens": _tokens(f"{document.title} {document.topic}"),
                }
            )
            for heading in document.headings:
                entries.append(
                    {
                        "kind": "heading",
                        "document_id": document.document_id,
                        "title": document.title,
                        "heading_path": heading.path,
                        "matched_on": heading.title or heading.path,
                        "snippet": f"{document.title} > {heading.path}",
                        "definition_unknown": False,
                        "aliases": list(heading.aliases),
                        "tokens": _tokens(" ".join([heading.path, *heading.aliases])),
                    }
                )
        self._index = entries
        self._dirty = False

    def _check_dirty(self) -> None:
        if self._dirty:
            self._ensure_index()

    # ---------------------------------------------------------------- 基本访问

    def is_empty(self) -> bool:
        return not self.documents

    def document(self, document_id: str) -> CatalogDocument | None:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        return None

    def source_file_of(self, document_id: str) -> str:
        """内部使用：来源 ID → 块元数据里的 source_file（**不对外投影**）。"""

        return self._source_files.get(document_id, "")

    def text_for(self, document_id: str) -> str:
        """离线正文样例（可能被截断；引用片段校验请用 :meth:`internal_text_lookup`）。"""

        return self._document_text.get(document_id, "")

    def set_document_text(self, document_id: str, text: str) -> None:
        """写入/替换某文档的正文样例（测试与按需补充用）。"""

        self._document_text[document_id] = str(text or "")
        self.mark_dirty()

    def internal_text_lookup(self, vector_store: Any) -> Callable[[str], str]:
        """按来源 ID 读取**全文**的访问器（引用片段存在性校验用）。

        只在本进程内把来源 ID 映射回块的 source_file；调用方拿不到路径。
        """

        def lookup(document_id: str) -> str:
            source = self._source_files.get(document_id)
            if not source:
                return ""
            blocks = vector_store.get_blocks_by_source(source)
            if isinstance(blocks, Mapping):
                return "\n".join(str(text) for text in blocks.values())
            return ""

        return lookup

    def summary(self) -> dict[str, int]:
        """目录规模（报告与工具统计用）。"""

        return {
            "documents": len(self.documents),
            "headings": sum(len(document.headings) for document in self.documents),
            "terms": len(self.terms),
            "definition_unknown": sum(1 for term in self.terms if term.definition_unknown),
        }

    # ---------------------------------------------------------------- 查询

    def lookup_term(self, value: str, *, limit: int = CATALOG_DEFAULT_LIMIT) -> list[CatalogTerm]:
        """精确词 / 别名查询（大小写与全半角已规范化）。"""

        wanted = normalize_catalog_text(value)
        if not wanted:
            return []
        hits = [
            term
            for term in self.terms
            if normalize_catalog_text(term.term) == wanted
            or any(normalize_catalog_text(alias) == wanted for alias in term.aliases)
        ]
        return hits[: max(1, limit)]

    def search(
        self,
        query: str,
        *,
        limit: int = CATALOG_DEFAULT_LIMIT,
        cursor: int = 0,
        kinds: Sequence[str] | None = None,
    ) -> CatalogQueryResult:
        """精确词 / 别名 / 全文检索统一入口（``lookup_catalog`` 直接消费）。

        - 排序分三档：**强命中**（≥500：精确词 / 别名 / 整串等于标题）→ **定位线索**
          （非术语、400 ≤ 分 < 500：文档或小节的**部分**标题命中）→ 其余；档内按分数、
          再按 ``kind`` → ``document_id`` → ``heading_path`` 稳定排序。
          中间那一档单独提前，是因为术语带 +50 加成且数量远超文档/标题，会把真正的 400 分
          文档线索挤出调用方的 top-K（``IntentAgent.enrich`` 只取 6 条 ⇒ 本该走知识库的
          问题落回通用知识）；强命中档保持原语义，精确术语命中仍排在小节条目之前。
        - ``next_cursor`` 为 ``None`` 表示已到末页。
        """

        normalized = normalize_catalog_text(query)
        if not normalized:
            return CatalogQueryResult(query=query, matches=[], total=0, next_cursor=None)
        self._check_dirty()
        wanted_kinds = {kind for kind in (kinds or ()) if kind in {"term", "heading", "document"}}
        query_tokens = _tokens(query)

        scored: list[tuple[int, int, int, str, str, str, dict[str, Any]]] = []
        for entry in self._index:
            if wanted_kinds and entry["kind"] not in wanted_kinds:
                continue
            score = self._score_entry(entry, normalized, query_tokens)
            if score <= 0:
                continue
            if score >= _STRONG_MATCH_SCORE:
                band = 0
            elif entry["kind"] != "term" and score >= _CLUE_RANK_MIN_SCORE:
                band = 1
            else:
                band = 2
            kind_rank = {"term": 0, "heading": 1, "document": 2}[entry["kind"]]
            scored.append(
                (
                    band,
                    -score,
                    kind_rank,
                    entry["document_id"],
                    entry["heading_path"],
                    entry["snippet"],
                    entry,
                )
            )
        scored.sort(key=lambda item: item[:6])
        total = len(scored)
        safe_limit = max(1, min(int(limit), CATALOG_MAX_LIMIT))
        start = max(0, int(cursor))
        matches = [
            CatalogMatch(
                kind=item[6]["kind"],
                document_id=item[6]["document_id"],
                title=item[6]["title"],
                heading_path=item[6]["heading_path"],
                snippet=_display_text(item[6]["snippet"]),
                score=-item[1],
                matched_on=item[6]["matched_on"],
                definition_unknown=bool(item[6]["definition_unknown"]),
                aliases=list(item[6]["aliases"]),
            )
            for item in scored[start : start + safe_limit]
        ]
        next_cursor = start + safe_limit if start + safe_limit < total else None
        return CatalogQueryResult(
            query=query, matches=matches, total=total, next_cursor=next_cursor
        )

    def _score_entry(
        self, entry: Mapping[str, Any], normalized_query: str, query_tokens: set[str]
    ) -> int:
        term_norm = normalize_catalog_text(entry["matched_on"])
        alias_norms = [normalize_catalog_text(alias) for alias in entry.get("aliases", ())]
        heading_norm = normalize_catalog_text(entry["heading_path"])
        title_norm = normalize_catalog_text(entry.get("title", ""))
        is_document = entry["kind"] == "document"
        # 「某个标签整体出现在问题里」这一档（800/700）只对**有区分度的标签**成立：
        # 1–2 字的泛化标题（「方案」「问题」「总结」）本来就出现在大量提问里，让它拿到
        # 800 分就会把「…那种方案」这类泛指问题误判成「本工作区有这份资料」
        # （真实缺陷：两个无关文档里的小节标题「方案」曾单独授权 R1）。文档标题天然
        # 有区分度（整串等于标题名），因此不受这条限制。
        label_long_enough = (
            is_document
            or len(term_norm) >= _MIN_SUBSTRING_LABEL_CHARS
        ) and term_norm not in _LOW_INFORMATION_LABELS
        score = 0
        if term_norm and term_norm == normalized_query:
            score = 1_000
        elif any(alias and alias == normalized_query for alias in alias_norms):
            score = 900
        elif (
            term_norm
            and len(normalized_query) >= 2
            and term_norm in normalized_query
            and label_long_enough
        ):
            score = 800
        elif any(
            alias and len(alias) >= _MIN_SUBSTRING_LABEL_CHARS and alias in normalized_query
            for alias in alias_norms
        ):
            score = 700
        elif title_norm and (title_norm in normalized_query or normalized_query in title_norm):
            score = 500
        elif (
            heading_norm
            and (heading_norm in normalized_query or normalized_query in heading_norm)
            and (
                is_document
                or (
                    len(heading_norm) >= _MIN_SUBSTRING_LABEL_CHARS
                    and heading_norm not in _LOW_INFORMATION_LABELS
                )
            )
        ):
            score = 400
        elif (
            _has_shared_cjk_run(
                normalized_query,
                title_norm if is_document else (term_norm or heading_norm),
                _MIN_SHARED_RUN_CHARS,
            )
            and (
                is_document
                or (term_norm or heading_norm) not in _LOW_INFORMATION_LABELS
            )
        ):
            # 「用户点到了这份资料的标题/小节」的部分命中：问题与文档标题（或小节标题）
            # 存在 ≥4 字的公共中文串（如「记忆系统」「从提取到召回」）。整串包含太严
            # （自然提问几乎不会原样打出全标题），只看词元重叠又太松（子串级噪音会盖过
            # 它），这条介于两者之间，且**只有 >=4 字的连续公共串**才算 —— 2 字泛化词
            # （「方案」）永远够不到。
            #
            # 得分不是常数：同一问题常与多份资料都有 4 字公共串（「记忆系统」这种领域词
            # 出现在多篇标题/小节里），用一个**覆盖强度**（公共中文二元组个数，封顶 30）
            # 把真正被问到的那份排到最前 —— 否则先按 kind_rank 排，文档级线索会被另一篇
            # 资料的小节标题挤到第 2/3 位（真实缺陷：R1 的证据指向了别篇的小节）。
            label = title_norm if is_document else (term_norm or heading_norm)
            score = _PARTIAL_TITLE_CLUE_SCORE + min(
                _shared_cjk_bigram_count(normalized_query, label), _MAX_CLUE_COVERAGE_BONUS
            )
        elif any(token and token in heading_norm for token in query_tokens):
            score = 300
        if score == 0:
            overlap = len(query_tokens & set(entry.get("tokens", ())))
            if overlap:
                score = 100 + min(overlap, 99)
        if score and entry["kind"] == "term":
            # 术语命中是路由判断「内部术语」的直接线索，排在标题命中之前。
            score += 50
        return score

    def prompt_projection(self, query: str, *, limit: int = 6) -> CatalogPromptProjection:
        """只把**与本轮相关**的目录条目渲染进提示（PRD 6.1）。

        来源 ID 在提示里以 ``D1`` / ``D2`` 短标签出现，标签与真实 ID 的对应关系由
        ``document_ids`` 保留，便于把决定写回 ``RouteDecision.scope_evidence_ids``。
        完整目录不进入提示。
        """

        if self.is_empty():
            return CatalogPromptProjection(text="", document_ids=[], term_count=0)
        result = self.search(query, limit=max(1, limit))
        if not result.matches:
            return CatalogPromptProjection(text="", document_ids=[], term_count=0)

        labels: dict[str, str] = {}
        lines: list[str] = []
        terms = 0
        for match in result.matches:
            label = labels.setdefault(match.document_id, f"D{len(labels) + 1}")
            if match.kind == "term":
                terms += 1
                suffix = "（缺少明确解释，只能用于定位原文）" if match.definition_unknown else ""
                lines.append(
                    f"- [{label}] 术语：{match.matched_on}{suffix}"
                    f"；位置：{match.heading_path or '（无标题）'}"
                )
                continue
            lines.append(f"- [{label}] {match.snippet}；命中：{match.heading_path or '（文档级）'}")
        payload = {
            "note": "本区知识目录的相关条目；只用于定位原文，不能作为事实依据",
            "entries": lines,
        }
        text = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
        return CatalogPromptProjection(
            text=text,
            document_ids=list(labels),
            term_count=terms,
            truncated=result.next_cursor is not None,
        )


def build_knowledge_catalog(
    blocks: Iterable[tuple[str, str, Mapping[str, Any]]],
    *,
    workspace_id: str = "",
    knowledge_revision: int = 0,
    generator_version: str = CATALOG_GENERATOR_VERSION,
    max_terms: int = CATALOG_MAX_TERMS,
) -> KnowledgeCatalog:
    """从**本区**块枚举构建知识目录（标题、层级、主题、术语、别名、来源 ID）。

    术语派生是确定性的离线规则：标题（含聚合短节的别名路径）即候选术语，标题所在
    正文样例中能找到解释性句子时给出 ``snippet``，否则标 ``definition_unknown``。
    模型提取的术语走 :func:`apply_term_candidates` 的独立校验通道。
    """

    block_list = [(block_id, text, metadata) for block_id, text, metadata in blocks]
    documents: dict[str, dict[str, Any]] = {}
    heading_seen: dict[str, set[str]] = {}

    for _block_id, text, metadata in block_list:
        meta = dict(metadata) if isinstance(metadata, Mapping) else {}
        raw_source = str(meta.get("source_file") or "")
        if not raw_source:
            continue
        document_id = _document_id_for(raw_source)
        entry = documents.get(document_id)
        if entry is None:
            title, topic = _public_source_parts(raw_source, meta, strip_extension=True)
            entry = {
                "document_id": document_id,
                "title": title,
                "topic": topic,
                "source_type": str(meta.get("source_type") or "document"),
                "platform": str(meta.get("platform") or "local"),
                "chunk_count": 0,
                "content_id": str(meta.get("doc_id") or ""),
                "headings": [],
                "_source_file": raw_source,
                "_text": "",
                "_is_image": False,
            }
            documents[document_id] = entry
            heading_seen[document_id] = set()
        entry["chunk_count"] += 1
        is_image = entry["source_type"] == "image_description" or str(
            meta.get("chunk_type") or ""
        ) == "image_description"
        entry["_is_image"] = bool(entry["_is_image"]) or is_image
        if not entry["content_id"]:
            entry["content_id"] = str(meta.get("doc_id") or "")

        heading_path = _display_text(meta.get("heading_path"), limit=300)
        if heading_path and heading_path not in heading_seen[document_id]:
            heading_seen[document_id].add(heading_path)
            aliases = _parse_aliases(meta.get("heading_aliases"))
            entry["headings"].append(
                {
                    "path": heading_path,
                    "level": max(1, min(6, heading_path.count(" > ") + 1)),
                    "title": heading_path.split(" > ")[-1].strip(),
                    "anchor": _display_text(meta.get("anchor"), limit=120),
                    "chunk_count": 0,
                    "aliases": aliases,
                }
            )
        for heading in entry["headings"]:
            if heading["path"] == heading_path:
                heading["chunk_count"] += 1
                break
        if not is_image:
            sample = entry["_text"]
            if len(sample) < CATALOG_SAMPLE_MAX_CHARS and text:
                entry["_text"] = (sample + "\n" + str(text))[:CATALOG_SAMPLE_MAX_CHARS]

    catalog_documents: list[CatalogDocument] = []
    terms: list[CatalogTerm] = []
    source_files: dict[str, str] = {}
    document_text: dict[str, str] = {}

    for document_id, entry in documents.items():
        headings = [
            CatalogHeading(**heading)
            for heading in entry["headings"][:CATALOG_MAX_HEADINGS_PER_DOC]
        ]
        catalog_documents.append(
            CatalogDocument(
                document_id=document_id,
                title=entry["title"],
                topic=entry["topic"],
                source_type=entry["source_type"],
                platform=entry["platform"],
                chunk_count=entry["chunk_count"],
                content_id=entry["content_id"],
                headings=headings,
            )
        )
        source_files[document_id] = str(entry["_source_file"])
        document_text[document_id] = str(entry["_text"])
        if entry["_is_image"]:
            continue
        for heading in headings:
            snippet, defined = _find_definition(
                heading.title, heading.aliases, document_text[document_id]
            )
            terms.append(
                CatalogTerm(
                    term=heading.title,
                    aliases=list(heading.aliases),
                    document_id=document_id,
                    heading_path=heading.path,
                    snippet=snippet,
                    definition_unknown=not defined,
                    evidence_verified=bool(snippet),
                    origin="heading",
                )
            )

    catalog_documents.sort(key=lambda item: item.document_id)
    catalog = KnowledgeCatalog(
        generator_version=generator_version,
        workspace_id=workspace_id,
        knowledge_revision=max(0, int(knowledge_revision)),
        built_at=_now_iso(),
        documents=catalog_documents,
        terms=_dedupe_terms(terms)[:max_terms],
    )
    catalog._source_files = source_files
    catalog._document_text = document_text
    catalog.mark_dirty()
    return catalog


# ---------------------------------------------------------------------
# 5. 模型术语提取：提示 + 严格校验（PRD 6.1）
# ---------------------------------------------------------------------


def build_term_extraction_prompt(
    documents: Sequence[CatalogDocument],
    *,
    max_terms: int = 40,
) -> str:
    """构造术语提取提示：只给标题线索，要求模型附文档位置与引用片段。"""

    lines = [
        "任务：从下面给出的文档标题结构中提取本工作区**内部术语**及其别名。",
        f"最多输出 {max_terms} 条；每条必须给出 document_id、heading_path 与原文片段。",
        "找不到明确解释时把 snippet 留空（程序会标记 definition_unknown）。",
        "只输出 JSON 数组，元素字段：term / aliases / document_id / heading_path / snippet。",
        "",
        "文档标题结构：",
    ]
    for document in documents:
        lines.append(f"- document_id={document.document_id} 标题={document.title}")
        for heading in document.headings[:CATALOG_MAX_HEADINGS_PER_DOC]:
            lines.append(f"    - {heading.path}")
    return "\n".join(lines)


def parse_term_candidates(raw: str | Sequence[Mapping[str, Any]]) -> list[TermCandidate]:
    """解析模型输出的术语候选；结构不合法的元素直接跳过（不猜测）。"""

    payload: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"术语候选不是合法 JSON: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("terms", [])
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise TypeError("术语候选必须是 JSON 数组")

    candidates: list[TermCandidate] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        candidate = _validate_term_candidate(dict(item))
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _validate_term_candidate(data: dict[str, Any]) -> TermCandidate | None:
    """单条候选校验：不合法返回 ``None``（不猜测、不修补）。"""

    try:
        return TermCandidate.model_validate(data)
    except ValidationError:
        return None


def apply_term_candidates(
    catalog: KnowledgeCatalog,
    candidates: Sequence[TermCandidate],
    *,
    text_lookup: Callable[[str], str] | None = None,
    max_terms: int = CATALOG_MAX_TERMS,
) -> TermApplicationResult:
    """程序校验模型术语候选，只保留能落到实际文档位置的条目（PRD 6.1）。

    拒绝条件（任一命中即整条拒绝）：术语为空、文档 ID 不在目录、标题位置不存在、
    引用片段在原文里找不到。片段为空（或不是解释句）时保留条目但标
    ``definition_unknown`` —— 「缺少明确解释」不是编造。
    """

    accepted: list[CatalogTerm] = []
    rejected: list[TermRejection] = []
    unknown: list[str] = []

    for candidate in candidates:
        term_text = _display_text(candidate.term, limit=120)
        if not term_text:
            rejected.append(TermRejection(term="", reason="术语为空"))
            continue
        document = catalog.document(candidate.document_id)
        if document is None:
            rejected.append(
                TermRejection(term=term_text, reason=f"文档 ID 不存在: {candidate.document_id!r}")
            )
            continue
        heading_path = _display_text(candidate.heading_path, limit=300)
        known_headings = {heading.path for heading in document.headings}
        if heading_path and known_headings and heading_path not in known_headings:
            rejected.append(
                TermRejection(term=term_text, reason=f"标题位置不存在: {heading_path!r}")
            )
            continue
        body = ""
        if text_lookup is not None:
            body = text_lookup(candidate.document_id) or ""
        if not body:
            # 全文访问器缺失或返回空时退回离线样例（可能被截断），避免把真实片段误拒。
            body = catalog.text_for(candidate.document_id)
        reference = candidate.snippet or candidate.definition
        snippet = _display_text(reference, limit=300)
        if snippet and normalize_catalog_text(snippet) not in normalize_catalog_text(body):
            rejected.append(TermRejection(term=term_text, reason="引用片段不存在于原文"))
            continue
        defined = bool(snippet) and len(normalize_catalog_text(snippet)) > len(
            normalize_catalog_text(term_text)
        )
        accepted.append(
            CatalogTerm(
                term=term_text,
                aliases=_dedupe_labels(candidate.aliases, exclude=term_text),
                document_id=document.document_id,
                heading_path=heading_path,
                snippet=snippet if defined else "",
                definition_unknown=not defined,
                evidence_verified=bool(snippet),
                origin="model",
            )
        )
        if not defined:
            unknown.append(term_text)

    if accepted:
        accepted_keys = {_term_key(term) for term in accepted}
        merged = [term for term in catalog.terms if _term_key(term) not in accepted_keys]
        catalog.terms = _dedupe_terms([*merged, *accepted])[:max_terms]
        catalog.mark_dirty()

    return TermApplicationResult(
        accepted=accepted, rejected=rejected, definition_unknown=unknown
    )


def lookup_catalog_query(
    catalog: KnowledgeCatalog,
    query: str,
    *,
    cursor: int = 0,
    limit: int = CATALOG_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """`lookup_catalog` 工具（T04 注册）直接消费的结果投影：无路径、可分页。"""

    result = catalog.search(query, limit=limit, cursor=cursor)
    return {
        "query": result.query,
        "total": result.total,
        "next_cursor": result.next_cursor,
        "items": [
            {
                "kind": match.kind,
                "document_id": match.document_id,
                "title": match.title,
                "heading_path": match.heading_path,
                "snippet": match.snippet,
                "definition_unknown": match.definition_unknown,
            }
            for match in result.matches
        ],
    }


# ---------------------------------------------------------------------
# 6. 每区目录缓存（PRD 5.3.4）
# ---------------------------------------------------------------------

_CATALOGS: dict[str, KnowledgeCatalog] = {}


def clear_catalog_cache() -> None:
    """清空进程内目录缓存（测试与迁移后调用）。"""

    _CATALOGS.clear()


def catalog_cache_size() -> int:
    return len(_CATALOGS)


def invalidate_catalog(workspace_id: str) -> int:
    """按工作区失效目录缓存（知识版本递增已是自动失效，这里是显式兜底）。"""

    removed = 0
    for key in list(_CATALOGS):
        if key.endswith(f"|ws={workspace_id}"):
            _CATALOGS.pop(key, None)
            removed += 1
    return removed


def catalog_cache_key(
    context: _CatalogContext,
    *,
    index_fingerprint: str | None = None,
    generator_version: str = CATALOG_GENERATOR_VERSION,
) -> str:
    """缓存键 = workspace_id + knowledge_revision + 索引指纹 + 生成器版本（PRD 5.3.4）。

    目录身份用**共享数据根**（D-KB：知识库全局共享，语料只有一份），避免同一
    ``workspace_id`` 在不同数据目录下互相命中。
    """

    fingerprint = index_fingerprint or context.storage_key
    key = context.index_cache_key(
        index_fingerprint=fingerprint,
        generator_version=f"{CATALOG_SCHEMA_VERSION}+{generator_version}",
    )
    root = getattr(getattr(context, "paths", None), "data_dir", "")
    return f"{root}|{key}|ws={context.workspace_id}"


def catalog_for_area(area: Any, *, max_terms: int = CATALOG_MAX_TERMS) -> KnowledgeCatalog:
    """返回**共享知识库**的目录（按 :func:`catalog_cache_key` 缓存）。

    ``area`` 是 ``app.retrieval.workspace_retrieval.WorkspaceKnowledgeArea``
    （鸭子类型：``context`` / ``index_fingerprint`` / ``blocks()``）。导入、同步、删除
    提交后 ``knowledge_revision`` 递增（D-KB：**所有**工作区一起递增），旧键自然失效
    并重建（PRD 5.3.4）。
    """

    context = area.context
    key = catalog_cache_key(
        context,
        index_fingerprint=getattr(area, "index_fingerprint", None),
    )
    cached = _CATALOGS.get(key)
    if cached is not None:
        return cached
    catalog = build_knowledge_catalog(
        area.blocks(),
        workspace_id=context.workspace_id,
        knowledge_revision=context.knowledge_revision,
        max_terms=max_terms,
    )
    _CATALOGS[key] = catalog
    return catalog
