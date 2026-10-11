"""查询规划 Agent：把用户问题整理成**可独立检索**的查询（PRD 7.2 第 1 步 / 6.4）。

职责与边界：

- **合并同义补充**：同一个问题被切成多句时，只在「后续句自带研究对象」时才单独成一条
  查询；否则与上一句合并成完整问题。这样不会单独检索「有哪些维度值得考虑」这类缺少
  对象的片段（用户实测缺陷 C02/C03）。
- **补齐研究对象与关注主题**：每条查询单独交给检索系统时，都必须能确定「在问什么
  对象、关心什么主题」。
- **承接多轮上下文**：相关历史的主题属于本轮问题的一部分；出现指代/承接表达时，把
  历史主题带进查询（不改变用户本轮的原话）。
- **允许一般知识参与术语理解**：模型可以给出术语的原始名称、中文名称与同义/相关概念
  表达；这些表达**只是检索线索** —— 不能据此断言文档讨论过该概念，文档事实仍需证据。
- **保持用户要求**：查询保留研究对象、来源角色和交付要求，原始问题用于核对范围。

无模型时给出确定性结果（纯函数、可离线单测）；模型输出不合法时同样回落到确定性结果。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import urlsplit

__all__ = [
    "MAX_QUERIES",
    "MAX_QUERY_CHARS",
    "MODEL_QUERY_COUNT",
    "QueryPlan",
    "QueryPlannerAgent",
    "StableQueryIdentity",
    "canonical_query",
    "clause_is_standalone",
    "compose_research_query",
    "confirmed_document_scope",
    "research_document_scope",
    "content_terms",
    "distinct_queries",
    "generic_document_content_ask",
    "generic_document_scope",
    "has_research_object",
    "history_topic",
    "is_fragment_follow_up",
    "is_instruction_only",
    "is_referential",
    "merge_context_queries",
    "merge_queries",
    "merge_semantic_queries",
    "model_plan_needed",
    "parse_query_plan",
    "stable_query_identity",
    "topic_phrase",
]

#: 一次 run 最多执行的检索查询条数（每条独立查询都会真的打一次检索）。
MAX_QUERIES: Final[int] = 3

#: 单条查询的长度上限（超出说明模型在复述问题，不再当作查询）。
MAX_QUERY_CHARS: Final[int] = 120

#: 请求模型整理查询时，模型最多给出的查询条数。
MODEL_QUERY_COUNT: Final[int] = 3

#: 句子级切分：与 Planner 的 ``_SENTENCE_SPLIT`` 同源（换行 / 句末标点），
#: 逗号不切 —— 逗号两侧通常属于同一项要求。
_CLAUSE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^\n。；;!?！？]+")

#: ASCII / 数字形式的术语（研究对象最常见的可识别形态，例如 Agent、OpenClaw、P99）。
_ASCII_TERM: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{1,}")
_DIGIT_RUN: Final[re.Pattern[str]] = re.compile(r"\d{2,}")
_QUOTED_TERM: Final[re.Pattern[str]] = re.compile(r"[“\"「『]([^”\"」』]{2,20})[”\"」』]")

#: 提问 / 连接 / 泛指词：这些词元不携带研究对象，不能作为「可独立检索」的依据。
_QUESTION_FORM_WORDS: Final[tuple[str, ...]] = (
    "哪些", "哪个", "什么", "怎么", "怎样", "如何", "为什么", "是否", "能否",
    "可以", "可以吗", "需要", "值得", "考虑", "有什", "有哪", "么样", "一个",
    "这个", "那个", "这些", "那些", "的话", "来说", "关于", "对于", "如果",
    "就是", "也是", "以及", "还有", "另外", "同样", "请问", "一下", "我们",
    "你们", "他们", "自己", "有没有", "多少", "多久", "介绍", "说说", "讲讲",
)

#: 领域名词（研究对象的中心词）。只有「修饰语 + 领域名词」这种名词短语才算自带对象，
#: 避免把「有哪些维度值得考虑」里的泛指名词当成研究对象。
_DOMAIN_HEADS: Final[tuple[str, ...]] = (
    "系统", "方案", "机制", "策略", "模块", "流程", "架构", "模型", "平台",
    "服务", "接口", "组件", "链路", "引擎", "框架", "工具", "能力", "算法",
    "文档", "文章", "报告", "手册", "规范", "标准", "笔记", "资料库",
    "库", "表", "集群", "索引", "队列", "缓存", "网关", "任务", "技能",
    "日志", "配置", "权限", "指标", "管道", "预算", "实验", "看板", "报表",
)

#: 承接 / 指代 / 同义补充表达：出现即说明该句依附上文，不能单独成查询。
_REFERENTIAL_MARKERS: Final[tuple[str, ...]] = (
    "类似", "这种", "那种", "这类", "那类", "该方案", "该文档", "该机制",
    "这个", "那个", "这些", "那些", "上述", "如上", "前者", "后者", "它",
    "也就是说", "换句话说", "换个说法", "换种说法", "同样", "以及", "还有",
    "另外", "再说", "再找", "再查", "再看", "的话", "呢",
)

#: 修饰语不是研究对象的情况（代词 / 泛指 / 容器）。
_GENERIC_MODIFIERS: Final[tuple[str, ...]] = (
    "我们", "我", "你们", "你", "大家", "本人", "自己", "上面", "下面", "以下",
    "上述", "相关", "这些", "那些", "各种", "很多", "一些",
)

#: `X的…领域名词` 形态：X 为 2–12 个汉字，中间最多 6 个汉字（「货拉拉的记忆系统」）。
_MODIFIER_HEAD: Final[re.Pattern[str]] = re.compile(
    r"([\u4e00-\u9fff]{2,12})的[\u4e00-\u9fff]{0,6}?(" + "|".join(_DOMAIN_HEADS) + r")"
)

#: 术语扩展的提示词（模型职责提示，system 消息）。
_QUERY_DUTY: Final[str] = (
    "你是检索查询规划器。你的职责是把用户问题整理成可独立检索的查询，并给出术语理解；"
    "不回答问题本身、不编造资料内容。"
)

_QUERY_TEMPLATE: Final[str] = """把下面的用户问题整理成 {count} 条以内的检索查询。只输出 JSON 对象。

硬性要求：
1. 每条查询单独交给检索系统时，都必须能确定「研究对象 + 关注主题 + 必要条件」；
   不要输出「有哪些维度值得考虑」这类缺少研究对象的片段。
2. 合并重复或同义的补充说法（例如同一个问题的两种说法只保留一条查询）。
3. 保持用户的研究对象、文档来源和交付要求。用户比较多个对象或多篇文章时，按每个对象分别
   给出完整查询，每条都保留该文章负责的问题；不能把多篇来源混在每条查询里。
4. 可以用你自己的通用知识理解术语，给出**原始名称、中文名称、同义或相关概念表达**
   （写进 terms）。这些表达只是检索线索：即使检索命中，也不能据此断言文档讨论过该概念。
5. 多轮追问：把相关历史的主题补进查询（历史主题属于本轮问题的一部分）。
6. 不要引入用户没有提到的研究对象；不要回答问题。
7. 同一对象的中文、英文和同义表达只构成一个查询，其他表达写进terms。
8. 原问题没有要求官方、最新或特定日期时，不得增加这些来源和时间条件。

输出字段：
- interpretation：一句话说明你如何理解这个问题（术语含义、研究对象、关注主题）。
- terms：术语的其他名称 / 同义表达（数组，可为空）。
- queries：检索查询（数组，1–{count} 条，中文或中英混合均可）。

问题：{question}
{history}{clauses}"""


@dataclass(frozen=True)
class QueryPlan:
    """一次查询规划的结果（原问题 + 追加的检索线索）。"""

    queries: list[str] = field(default_factory=list)
    interpretation: str = ""
    terms: list[str] = field(default_factory=list)
    source: str = "rules"
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        return "；".join(self.queries)


# =====================================================================
# 1. 词元与判据（纯函数）
# =====================================================================


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


#: 查询去重用的规范形：去掉**所有**标点 / 空白并统一大小写。
#:
#: 规划与执行必须共用同一个函数（用户实测缺陷 C21.4）：旧实现规划侧只去首尾标点，
#: 执行侧（``research._query_key``）也只统一空白 / 大小写 / 首尾标点，于是
#: 「文档中，比如记忆，成本这些维度。都是怎么样设计的？」与
#: 「文档中  比如记忆，成本这些维度。都是怎么样设计的？」被当成两条查询，
#: 返回完全相同的 8 个片段。
#: 去重时需要忽略的标点字符（用 Unicode 码点声明，避免引号 / 反斜杠转义错误）。
#: 「文档中，」与「文档中  」（C21.4）、《书名号》与中文弯引号都必须真的被消掉。
_CANONICAL_PUNCTUATION: Final[tuple[str, ...]] = tuple(chr(cp) for cp in (
    0x0020, 0x0009, 0x000A, 0x000D, 0x3002, 0xFF1B, 0x003B, 0x0021, 0x003F,
    0xFF01, 0xFF1F, 0xFF0C, 0x002C, 0x3001, 0xFF1A, 0x003A, 0x0022, 0x0027,
    0x201C, 0x201D, 0x2018, 0x2019, 0x0028, 0x0029, 0xFF08, 0xFF09, 0x005B,
    0x005D, 0x3010, 0x3011, 0x300C, 0x300D, 0x300E, 0x300F, 0x003C, 0x003E,
    0x300A, 0x300B, 0x002D, 0x2014, 0x005F, 0x00B7, 0x007E, 0x002F, 0x005C,
))
_CANONICAL_STRIP: Final[re.Pattern[str]] = re.compile(
    "[" + "".join(re.escape(item) for item in _CANONICAL_PUNCTUATION) + "]+"
)


def canonical_query(text: Any) -> str:
    """查询的规范形（标点 / 空白 / 大小写归一）：规划与执行共用的唯一去重口径。"""

    return _CANONICAL_STRIP.sub("", str(text or "")).casefold()


@dataclass(frozen=True)
class StableQueryIdentity:
    key: str
    objects: tuple[str, ...]
    directions: tuple[str, ...]
    source_scope: tuple[str, ...]
    deliverables: tuple[str, ...]

    def reuse_reason(self) -> str:
        parts = [
            "研究对象=" + "/".join(self.objects),
            "研究方向=" + "/".join(self.directions),
            "来源范围=" + "/".join(self.source_scope),
            "交付要求=" + "/".join(self.deliverables),
        ]
        return "语义查询身份一致（" + "；".join(parts) + "）"


_IDENTITY_ASCII: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[/._#+-][A-Za-z0-9]+)*"
)
_IDENTITY_DOMAIN: Final[re.Pattern[str]] = re.compile(
    r"(?:(?P<prefix>https?://|site:))?(?P<host>[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:[/\s]|$)",
    re.IGNORECASE,
)
_IDENTITY_TLDS: Final[frozenset[str]] = frozenset({
    "ai", "app", "cloud", "cn", "co", "com", "dev", "edu", "gov", "io", "jp",
    "net", "org", "tech", "uk",
})
_IDENTITY_STOP: Final[frozenset[str]] = frozenset({
    "api", "reference", "official", "document", "docs", "page", "parameter",
    "parameters", "valid", "range", "minimum", "maximum", "min", "max", "days",
    "day", "site", "http", "https", "com", "latest", "the", "and", "for", "with",
    "allowed", "request", "unit", "value", "values",
})
_IDENTITY_DIRECTION_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("create", ("创建", "新增", "create", "creation")),
    ("update", ("更新", "修改", "update", "modify")),
    ("delete", ("删除", "移除", "delete", "remove")),
    ("read", ("读取", "查询", "获取", "read", "get", "retrieve")),
    ("compare", ("比较", "差异", "区别", "compare", "difference")),
    ("explain", ("说明", "解释", "原理", "explain", "definition")),
)
_IDENTITY_DELIVERY_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("bounds", ("最小", "最大", "上下限", "范围", "边界", "minimum", "maximum", "valid range", "bounds")),
    ("source", ("官网", "官方", "链接", "页面", "出处", "source", "reference")),
    ("unit", ("单位", "天数", "days", "unit")),
    ("example", ("示例", "例子", "example")),
    ("steps", ("步骤", "流程", "step", "procedure")),
)


def _identity_parts(token: str) -> list[str]:
    value = token.casefold().strip("._/+-#")
    if not value:
        return []
    value = {"amazon": "aws"}.get(value, value)
    parts = [value] if not re.search(r"[._+#-]", token) else []
    separated = re.sub(r"[/._#+-]+", " ", token).casefold()
    camel_parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).casefold().split()
    if camel_parts and camel_parts[0] in {"create", "update", "delete", "read", "get"}:
        separated += " " + " ".join(camel_parts)
    for item in separated.split():
        normalized = {"amazon": "aws"}.get(item, item)
        if normalized and normalized not in parts:
            parts.append(normalized)
    return parts


def _identity_source_scope(text: str, source_kind: str) -> tuple[str, ...]:
    folded = text.casefold()
    scopes: list[str] = []
    for match in _IDENTITY_DOMAIN.finditer(text):
        candidate = match.group("host")
        # 无 URL 或 site: 前缀时，CamelCase 标识符里的点号不表示域名。
        if not match.group("prefix"):
            suffix = candidate.rsplit(".", 1)[-1].casefold()
            if any(char.isupper() for char in candidate) or suffix not in _IDENTITY_TLDS:
                continue
        host = (urlsplit("https://" + candidate).hostname or "").casefold()
        if host.endswith("aws.amazon.com") or host == "amazon.com":
            scope = "aws-official"
        elif host.startswith("docs.") and len(host.split(".")) >= 3:
            scope = host.split(".")[1] + "-official"
        else:
            scope = host
        if scope and scope not in scopes:
            scopes.append(scope)
    if not scopes and re.search(r"\b(?:aws|amazon)\b", folded) and re.search(
        r"官方|官网|official|\bsite:", folded
    ):
        scopes.append("aws-official")
    if not scopes:
        product_match = re.search(
            r"\b([a-z][a-z0-9-]+)\s+(?:official\s+)?(?:docs|documentation)\b|"
            r"\b([a-z][a-z0-9-]+)\s*(?:官方文档|官方网站|官网)",
            folded,
        )
        if product_match:
            product = product_match.group(1) or product_match.group(2)
            scopes.append(product + "-official")
    if source_kind:
        scopes.append(str(source_kind).casefold())
    return tuple(sorted(dict.fromkeys(scopes)))


def stable_query_identity(
    text: str, *, source_kind: str = "", context_text: str = ""
) -> StableQueryIdentity:
    folded = _normalize(f"{text or ''} {context_text or ''}").casefold()
    source_scope = _identity_source_scope(
        f"{text or ''} {context_text or ''}", source_kind
    )
    object_text = re.sub(r"(?:https?://|site:)\S+", " ", str(text or ""), flags=re.IGNORECASE)
    objects: list[str] = []
    compound_objects: list[str] = []
    camel_objects: list[str] = []
    context_compounds: list[str] = []
    for match in _IDENTITY_ASCII.finditer(str(context_text or "")):
        raw_token = match.group(0)
        normalized_token = raw_token.casefold().strip("._/+-#")
        if (
            normalized_token
            and re.search(r"[/._#+-]", raw_token)
            and normalized_token not in context_compounds
        ):
            context_compounds.append(normalized_token)
    for match in _IDENTITY_ASCII.finditer(object_text):
        raw_token = match.group(0)
        normalized_token = raw_token.casefold().strip("._/+-#")
        normalized_token = re.sub(r"^api[._/#+-]+", "", normalized_token)
        if (
            normalized_token
            and re.search(r"[a-z][A-Z]", raw_token)
            and normalized_token not in camel_objects
        ):
            camel_objects.append(normalized_token)
        if (
            normalized_token
            and (
                re.search(r"[a-z][A-Z]", raw_token)
                or re.search(r"[/._#+-]", raw_token)
            )
            and normalized_token not in compound_objects
        ):
            compound_objects.append(normalized_token)
        for token in _identity_parts(raw_token):
            if token in _IDENTITY_STOP or re.fullmatch(r"\d+", token):
                continue
            if token == "aws" and "aws-official" in source_scope:
                continue
            if token not in objects:
                objects.append(token)
    for match in re.finditer(r"\b([a-z][A-Za-z0-9]*)\s+([A-Z][A-Za-z0-9]*)\b", object_text):
        namespace = match.group(1).casefold()
        member = match.group(2).casefold()
        if namespace in _IDENTITY_STOP or member in _IDENTITY_STOP:
            continue
        if member.startswith(("create", "update", "delete", "read", "get")):
            continue
        combined = f"{namespace}.{member}"
        if combined not in context_compounds:
            continue
        camel_objects = [item for item in camel_objects if item != member]
        compound_objects = [item for item in compound_objects if item != member]
        camel_objects.append(combined)
        compound_objects.append(combined)
    direction_names = [
        name for name, markers in _IDENTITY_DIRECTION_MARKERS
        if any(marker in folded for marker in markers)
    ]
    version_metadata = bool(re.search(
        r"(?:新增|added|new|versionadded).{0,12}(?:版本|version)|"
        r"(?:版本|version).{0,12}(?:新增|added|new)",
        folded,
    )) or bool(
        re.search(r"新增|added|new|versionadded", str(text or "").casefold())
        and re.search(r"版本|version", str(context_text or "").casefold())
    )
    if (
        version_metadata
        and "create" in direction_names
        and "创建" not in folded
        and not re.search(r"\bcreate", folded)
    ):
        direction_names.remove("create")
    directions = tuple(direction_names)
    delivery_names = [
        name for name, markers in _IDENTITY_DELIVERY_MARKERS
        if any(marker in folded for marker in markers)
    ]
    explicit_source_scope = _identity_source_scope(
        f"{text or ''} {context_text or ''}", ""
    )
    if explicit_source_scope and "source" not in delivery_names:
        delivery_names.append("source")
    deliverables = tuple(delivery_names)
    focused_compounds = sorted(
        camel_objects if len(camel_objects) >= 2 else compound_objects,
        key=lambda item: (-len(item), item),
    )[:2]
    qualified_compounds: list[str] = []
    for item in focused_compounds:
        matches = [
            candidate for candidate in context_compounds
            if candidate.endswith("." + item)
        ]
        qualified = matches[0] if len(matches) == 1 else item
        if qualified not in qualified_compounds:
            qualified_compounds.append(qualified)
    focused_compounds = qualified_compounds
    focused_compounds = [
        item for item in focused_compounds
        if not any(other != item and other.endswith("." + item) for other in focused_compounds)
    ]
    if any(scope.endswith("-official") for scope in source_scope):
        identity_objects = tuple(sorted(focused_compounds or objects))
    else:
        compound_parts = {part for item in focused_compounds for part in _identity_parts(item)}
        identity_objects = tuple(sorted(set(focused_compounds) | {item for item in objects if item not in compound_parts}))
    payload = json.dumps(
        {
            "objects": identity_objects,
            "directions": sorted(directions),
            "source_scope": source_scope,
            "deliverables": sorted(deliverables),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return StableQueryIdentity(
        key=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32],
        objects=identity_objects,
        directions=tuple(sorted(directions)),
        source_scope=source_scope,
        deliverables=tuple(sorted(deliverables)),
    )


def merge_semantic_queries(
    queries: Sequence[str], *, source_kind: str = "", context_text: str = ""
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    identities: dict[str, tuple[str, StableQueryIdentity]] = {}
    notes: list[str] = []
    for query in distinct_queries(queries):
        identity = stable_query_identity(
            query, source_kind=source_kind, context_text=context_text
        )
        existing = identities.get(identity.key)
        if existing is None or not identity.objects:
            identities[identity.key] = (query, identity)
            kept.append(query)
            continue
        original, _ = existing
        notes.append(
            f"复用查询「{original}」；省略语义等价表达「{query}」；{identity.reuse_reason()}"
        )
    return kept, notes


def _canonical(text: Any) -> str:
    return canonical_query(text)


def is_instruction_only(text: str) -> bool:
    value = _normalize(text).strip("。！？?!；;，, ")
    return bool(re.fullmatch(
        r"(?:请|务必|仅|只|不要|不必|无需|禁止|必须|可以)*"
        r"(?:联网|上网|搜索网页|查网络|使用网络|提供引用|附上出处|给出来源|列出引用)"
        r"(?:搜索|查询|检索|回答|即可)?", value
    )) or value in {"文档中", "资料里", "文章中"}


def distinct_queries(queries: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for query in queries:
        text = _normalize(query)
        if not text or is_instruction_only(text):
            continue
        key = canonical_query(text)
        if any(key == canonical_query(other) for other in kept):
            continue
        kept.append(text)
    return kept


def _model_queries_are_facets(anchor: str, queries: Sequence[str]) -> bool:
    """模型只把同一个完整问题改写成多个关注面时，保留完整问题一次执行。"""

    if len(queries) < 2:
        return False
    anchor_terms = set(content_terms(anchor))
    if len(anchor_terms) < 4:
        return False
    novel: set[str] = set()
    for query in queries:
        terms = set(content_terms(query))
        if not terms or len(terms & anchor_terms) / len(terms) < 0.55:
            return False
        novel.update(terms - anchor_terms)
    return len(novel) <= max(3, len(anchor_terms) // 5)


def confirmed_document_scope(question: str, catalog: Any) -> list[str]:
    documents = list(getattr(catalog, "documents", ()) or ())

    def resolve(text: str) -> list[str]:
        normalized = canonical_query(text)
        quoted = [canonical_query(value) for value in re.findall(r"《([^》]+)》", text)]
        confirmed: list[str] = []
        for document in documents:
            title = re.sub(r"\.md$", "", str(document.title), flags=re.IGNORECASE)
            key = canonical_query(title)
            if len(key) >= 4 and key in normalized:
                confirmed.append(str(document.document_id))
        for fragment in quoted:
            if not fragment:
                continue
            exact = [str(document.document_id) for document in documents
                     if fragment == canonical_query(re.sub(r"\.md$", "", str(document.title), flags=re.IGNORECASE))]
            if exact:
                confirmed.extend(exact)
                continue
            if len(fragment) < 4:
                continue
            matches = [str(document.document_id) for document in documents
                       if fragment in canonical_query(document.title)]
            if len(set(matches)) == 1:
                confirmed.extend(matches)
        for phrase in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Za-z][A-Za-z0-9_.-]*)+", text):
            key = canonical_query(phrase)
            if len(key) < 8:
                continue
            matches = [str(document.document_id) for document in documents
                       if key in canonical_query(document.title)]
            if len(set(matches)) == 1:
                confirmed.extend(matches)
        for fragment in re.split(r"[，。！？；：、]|以及|结合|根据|依据|参考|对比|和|与", text):
            reference = re.search(r"(.{2,100}?)(?:文章|文档|上篇|下篇)", fragment)
            if reference is None:
                continue
            description = reference.group(1).casefold()
            ascii_terms = {term for term in _ASCII_TERM.findall(description) if len(term) >= 3}
            chinese_runs = re.findall(r"[\u4e00-\u9fff]+", description)
            chinese_terms = {run[index:index + 2] for run in chinese_runs for index in range(len(run) - 1)}
            ranked: list[tuple[int, str]] = []
            for document in documents:
                title = str(document.title).casefold()
                ascii_matches = sum(term in title for term in ascii_terms)
                chinese_anchor = any(run[index:index + 4] in title for run in chinese_runs for index in range(len(run) - 3))
                if not chinese_anchor and ascii_matches < 2:
                    continue
                score = ascii_matches * 4 + sum(term in title for term in chinese_terms)
                ranked.append((score, str(document.document_id)))
            ranked.sort(reverse=True)
            if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
                confirmed.append(ranked[0][1])
        return list(dict.fromkeys(confirmed))

    return resolve(question)


def research_document_scope(
    original_question: str, task_question: str, catalog: Any, *, scoped_query: bool,
) -> list[str]:
    original_scope = confirmed_document_scope(original_question, catalog)
    task_scope = confirmed_document_scope(task_question, catalog)
    exclusive = bool(re.search(r"(?:仅|只)(?:根据|依据|使用|参考|按照)", original_question))
    if exclusive and original_scope:
        return [document_id for document_id in task_scope if document_id in original_scope] or original_scope
    if task_scope:
        return task_scope
    return [] if scoped_query else original_scope


def _is_question_form(token: str) -> bool:
    return any(token in word or word in token for word in _QUESTION_FORM_WORDS)


def content_terms(text: str) -> list[str]:
    """文本里携带信息量的词元（ASCII 术语 + 非提问用语的 CJK 二元组）。

    提问用语（哪些 / 怎么 / 值得 / 考虑…）被剔除：它们出现在几乎每个问题里，
    既不能说明研究对象，也不能作为「这条查询自成一体」的依据。
    """

    cleaned = _normalize(text).casefold()
    terms: list[str] = []
    for match in _ASCII_TERM.finditer(cleaned):
        token = match.group(0)
        if len(token) >= 2 and token not in {"the", "and", "for", "with"}:
            terms.append(token)
    digits = _DIGIT_RUN.findall(cleaned)
    terms.extend(digits)
    for run in re.findall(r"[\u4e00-\u9fff]+", cleaned):
        for index in range(len(run) - 1):
            bigram = run[index : index + 2]
            if _is_question_form(bigram):
                continue
            terms.append(bigram)
    return list(dict.fromkeys(terms))


def is_referential(clause: str) -> bool:
    """该句是否依附上文（指代 / 类比 / 同义补充），因而不能单独成为一条查询。"""

    text = _normalize(clause)
    return any(marker in text for marker in _REFERENTIAL_MARKERS)


def _has_subject_anchor(clause: str) -> bool:
    """该句是否自带研究对象。

    三种可复核的形态：明确的 ASCII / 数字术语、引号术语，或「修饰语 + 领域名词」
    （例如「货拉拉的记忆系统」）。泛指名词（维度、因素、思路）不算研究对象。
    """

    text = _normalize(clause)
    if _ASCII_TERM.search(text) or _DIGIT_RUN.search(text) or _QUOTED_TERM.search(text):
        return True
    for match in _MODIFIER_HEAD.finditer(text):
        modifier = match.group(1)
        if any(modifier.startswith(item) for item in _GENERIC_MODIFIERS):
            continue
        if all(_is_question_form(bigram) for bigram in content_terms(modifier)):
            continue
        return True
    return False


def clause_is_standalone(clause: str) -> bool:
    """这一句能否独立成为一条检索查询（研究对象 + 关注主题都在句内）。"""

    text = _normalize(clause)
    if not text:
        return False
    if is_referential(text):
        return False
    if len(content_terms(text)) < 2:
        return False
    return _has_subject_anchor(text)


# =====================================================================
# 2. 多轮上下文（历史主题）
# =====================================================================


_TOPIC_LEAD: Final[re.Pattern[str]] = re.compile(
    r"^(?:有一个|有个|一个|经典|面试题|请问|问一下|我想问|我想知道|那么|所以|然后|"
    r"另外|如果|关于|对于|顺便|你好|嗨)[：:，,、\s]*"
)
_TOPIC_TAIL: Final[re.Pattern[str]] = re.compile(r"[。？！?!，,、；;\s]+$")
_TOPIC_CUT: Final[tuple[str, ...]] = (
    "？", "?", "。", "，", ",", "是", "怎么", "如何", "为什么", "有哪些", "哪些",
    "多少", "是否", "有没有", "吗",
)
#: 泛指文档的空间指向（「文档中 / 资料里 / 原文里」）：用户点名「知识库里的资料」
#: 但**没有给出唯一文章名**（用户实测缺陷 C21.3「关于文档中，他们是怎么设计 agent 的」）。
#: 这种话要按「查本地资料」处理，而不是当成没有来源的通用问题。
#: 只匹配「文档 / 资料 / 文章 / 笔记 + 空间词」的**指向形态**，不把「文档中台」这类
#: 复合名词误判成文档指向（中台 / 中心 / 中控 / 中断 / 中文 / 中间）。
_GENERIC_DOC_SCOPE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:知识库|文档库|文档|资料|文章|笔记)(?:中|里|内)(?!(?:台|心|控|断|文|间|转|介))"
    r"|(?:文档|资料|文章|笔记)上面"
    r"|(?:按|根据|按照|依据)[^。！？?，,；;\s]{0,4}?(?:文档|资料|文章|笔记)"
    r"|(?:继续|接着|再|还有|另外)(?:按|根据|按照|依据)?[^。！？?，,；;\s]{0,4}?(?:讲|说|聊|谈)"
)
#: ^ 「文档中」后面紧跟「台 / 心 / 控 / 断 / 文 / 间 / 转 / 介」时是复合名词
#: （文档中台 / 文档中心 / 文档中文），不是「文档里的内容」这一空间指向。
#: 「问文档里的内容」的意图形态（设计 / 做法 / 实现 / 说明 / 依据）。泛指文档指向
#: 必须与它同时出现才算「请查文档里的内容」，避免把「上传到文档中」这类动作误判。
_DOC_CONTENT_ASK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:怎么(?:样)?(?:设计|做|实现|写|处理|算)|如何(?:设计|实现|做|处理|计算|说明)|"
    r"设计了|设计的|实现(?:了|的)|说了什么|讲了什么|是怎么|依据|出处|原文|维度)"
)
#: 「文档里有没有 / 找一下那份资料」这类**文档寻址**意图（寻址动词 + 文档对象）。
#: 关注维度（与路由层 ``_CONTENT_ASPECT`` 同源口径）。
_CONTENT_ASPECT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:职责|作用|区别|差异|关系|原理|机制|流程|步骤|公式|口径|边界|维度|因素|"
    r"取舍|权衡|设计|实践|依据|例子|数量|范围|阈值|配置|实现|成本|记忆)"
)
_DOC_SEEK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:在哪(?:里)?|哪里|哪个文档|哪个文件|哪一份|找一下|找找|查一下|翻一下|有没有|有吗|"
    r"是否有|收录)[^。！？?；;\n]{0,10}?"
    r"(?:手册|文档|文件|资料|笔记|记录|报告|规范|说明书|wiki|知识库|文档库|文章)"
)
#: 「他们 / 大家 + 怎么(样) + <对象> + 的」形态：主题截取会只剩「文档中」这类前置词，
#: 因此单独抽出 **「设计 + 对象」** 当主题（C21.3  C21.4 的 Agent 对象）。
#: 「他们 / 大家 + 怎么(样) + <对象> + 的」形态：主题截取会只剩「文档中」这类前置词，
#: 因此单独抽出 **「设计 + 对象」** 当主题（C21.3  C21.4 的 Agent 对象）。
#: 「他们 / 大家 + 怎么(样) + 设计/做 + <对象>」形态：沿用主题截取只会剩「文档中」这类
#: 前置词，因此单独抽出 **「设计 + 对象」** 当主题（C21.3  C21.4 的 Agent 对象）。
#: 「他们 / 大家 + 怎么(样) + 设计/做 + <对象> + 的」形态：沿用主题截取只会剩「文档中」
#: 这类前置词，因此单独抽出 **「设计 + 对象」** 当主题（C21.3  C21.4 的 Agent 对象）。
#: 对象只取到句末的「的 / 标点 / 补充提问」之前（「设计 agent 的？应该有好几个维度吧」
#: 的对象是 agent，不是整句）。
_TOPIC_DESIGN_OBJECT: Final[re.Pattern[str]] = re.compile(
    r"(?:他们|你们|我们|大家).*?(?:怎么|怎样|如何)\s*(?:样)?\s*"
    r"(?:设计|做|实现|写)\s*(?P<object>.+?)\s*(?:的|吗|呢)?"
    r"(?:[。！？?！，,、；;]|\s+(?=还有|以及|另外|其他|别的)|$)"
)
#: 研究对象候选（ASCII 术语 + 中文词）。
_TOPIC_OBJECT: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-]*|[\u4e00-\u9fff]{2,}"
)
_TOPIC_FUNCTION_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "文档", "资料", "文章", "笔记", "文档中", "资料中", "文章中", "文档里",
        "资料里", "文章里", "他们", "你们", "我们", "这些", "那些", "这个", "那个",
        "维度", "方面", "什么", "怎么", "如何", "设计", "内容", "东西", "地方",
    }
)


def generic_document_scope(text: str) -> str:
    """文本里对「资料 / 文档」的**泛指空间指向**（未点名唯一文章）。

    返回命中的短语（如「文档中」），未命中返回空串。用户实测缺陷 C21.3/C21.4：
    第三轮已明确说「关于文档中怎么设计 agent」，规则层却没有把它当成查本地资料，
    于是用通用解释回答；第四轮历史主题又被截成「文档中」单独检索。
    """

    value = _normalize(text)
    if not value:
        return ""
    match = _GENERIC_DOC_SCOPE_RE.search(value)
    return match.group(0) if match is not None else ""


def generic_document_content_ask(text: str) -> str:
    """问题是否在**问泛指文档里的内容**（泛指文档指向 + 内容 / 寻址意图）。

    命中的是「请查文档内容」这一语义，不是单一关键词：只有空间指向、没有内容意图
    （「把结果写进文档中」）或只有内容词、没有文档指向（「怎么设计 Agent」）都不算。
    """

    value = _normalize(text)
    if not value or not generic_document_scope(value):
        return ""
    match = _DOC_CONTENT_ASK_RE.search(value) or _DOC_SEEK_RE.search(value)
    if match is None:
        # 「继续按文档讲其他设计因素」这类没有「怎么/依据」等固定问法词，但有关注维度
        # （因素 / 维度 / 设计 / 做法）：同样是请查文档内容（C22.3）。
        match = _CONTENT_ASPECT_RE.search(value)
    return match.group(0) if match is not None else ""


def has_research_object(topic: str) -> bool:
    """主题里是否还有**可用的研究对象**（能独立成为检索对象）。

    判据沿用「一句能否独立成查询」的同一套口径（``_has_subject_anchor`` + 信息词），
    因此「文档中」（泛指指向）与「比如记忆」（只有关注维度、对象被省略）都判否。用途：
    多轮承接时挑「含对象的上一轮主题」，主题只剩泛指指向时沿历史继续往前找（C21.4 的
    Agent 对象在更早一轮）；同时不会把只是「碰巧在附近」的上一轮完整问题硬塞进新查询。
    """

    value = _normalize(topic)
    if not value or not _has_subject_anchor(value):
        return False
    return bool(content_terms(value))


def is_fragment_follow_up(text: str) -> bool:
    """本轮问题是否**只是上文的一部分**（对象被省略的片段式追问）。

    片段 = 承接 / 指代表达打头（「那成本呢」「这些维度呢」「还有别的吗」），或者
    实际内容词只有 1 个。用途：只有片段才把上文对象前置；用户本轮原话本身是完整
    问题时（C21.4「比如记忆，成本这些维度。都是怎么样设计的？」、C04 的面试题追问）
    保持原话在最前，事实对象补充在后。
    """

    value = _normalize(text)
    if not value:
        return True
    if re.search(r"刚才|上一(?:轮|条|次)|前面(?:的|提到|取得)|此前(?:的|提到)|上文", value):
        return True
    if re.match(r"^(?:请|麻烦|帮我)?(?:继续|其中)", value):
        return True
    stripped = value
    guard = 0
    while guard < 4:
        shorter = _TOPIC_LEAD.sub("", stripped)
        if shorter == stripped:
            break
        stripped = shorter
        guard += 1
    if any(stripped.startswith(marker) for marker in _REFERENTIAL_MARKERS):
        return True
    if any(marker in stripped for marker in ("呢", "还有", "另外", "别的", "其他")):
        return True
    return len(content_terms(stripped)) < 3


#: 句首连接词（去掉泛指文档指向后可能悬空）。
_LEAD_CONNECTOR: Final[re.Pattern[str]] = re.compile(
    r"^(?:关于|对于|至于|针对|那(?:么)?|所以|然后|另外|还有)+"
)


def compose_research_query(context: str, question: str) -> str:
    """把上一轮研究对象与本轮问题合成一条**完整研究查询**（两者都保留）。

    用户实测缺陷 C21.4：历史主题「设计 agent」被当成一条独立查询（最后甚至只剩「文档中」），
    本轮真正要问的记忆 / 成本维度与研究对象被拆散。这里按「含文档指向的问题先去掉泛指
    指向、再把上文对象放在最前面」合并，用户本轮原话一字不改。
    """

    value = _normalize(question)
    if not value:
        return _normalize(context)
    position = _GENERIC_DOC_SCOPE_RE.search(value)
    if position is not None:
        value = _normalize(value[: position.start()] + value[position.end() :])
        # 泛指指向去掉后可能留下悬空的连接词（「关于，他们」「他们」）或前导标点，
        # 拼进检索查询会变成噪声。
        value = _LEAD_CONNECTOR.sub("", value).lstrip("，,、；;：:。 ") 
        value = _normalize(value)
    if not value:
        # 去掉泛指指向后没有剩余内容（「继续按文档讲」）：直接用本轮原问题，
        # 由调用方把上文对象拼上。
        value = _normalize(question)
    prefix = _normalize(context)
    if not prefix or prefix.casefold() in value.casefold():
        return value
    # 用户本轮原话是完整问题（自带对象 + 关注主题）时**原话仍在最前**（既有行为不变），
    # 只把上文对象作为补充追在后面；只有本轮只是片段（「记忆和成本呢」）时才把上文对象
    # 前置。
    if not is_fragment_follow_up(question):
        return _normalize(f"{value}  {prefix}")
    return _normalize(f"{prefix}  {value}")


def merge_context_queries(
    queries: Sequence[str], topic: str, *, question: str = ""
) -> tuple[list[str], bool]:
    """把「上一轮主题」合进相关查询；返回 (查询列表, 是否改写了列表)。

    只有当本轮查询**离不开**上文主题时才合并（泛指文档指向 / 指代 / 同义补充）；
    否则主题仍作为独立查询追加（保持既有行为：主题与新一轮问题各自可独立检索）。
    """

    merged = list(queries)
    prefix = _normalize(topic)
    if not prefix:
        return merged, False
    changed = False
    folded: list[str] = []
    for query in merged:
        text = _normalize(query)
        fold_needed = (
            generic_document_scope(text) != ""
            or is_referential(text)
            or generic_document_scope(question) != ""
        )
        if (
            fold_needed
            and prefix.casefold() not in text.casefold()
            and text not in folded
        ):
            folded.append(compose_research_query(prefix, text))
            changed = True
        else:
            folded.append(text)
    return folded, changed


def topic_phrase(text: str) -> str:
    """从一个问题里取「上文主题」（可读、有限长度、可复核）。

    去掉开场套话（「有一个经典面试题：」）与提问尾巴（「是怎么设计的？」），
    只留下研究对象本身；取不到时返回空串（调用方不猜测对象）。

    两种主题形态分开处理（用户实测缺陷 C21.2  C21.3 的连续追问）：

    - **名词短语型**（「面试要求设计一个 Agent」「货拉拉的记忆系统有淘汰策略」） 沿用
      提问尾巴截取，主题保持原话；
    - **「他们 + 设计 + 对象」型**（「文档中他们是怎么设计 agent 的」） 沿用截取会把
      主题切成「文档中」，Agent 对象丢失。这里按 `_TOPIC_DESIGN_OBJECT` 抽成
      「设计 agent」，让对象能传给下一轮。
    """

    value = _normalize(text)
    guard = 0
    while guard < 4:
        stripped = _TOPIC_LEAD.sub("", value)
        if stripped == value:
            break
        value = stripped
        guard += 1
    object_match = _TOPIC_DESIGN_OBJECT.search(value)
    if object_match is not None:
        target = _TOPIC_TAIL.sub("", object_match.group("object")).strip("的了吗呢吧 ")
        if target:
            return f"设计 {target}"[:40]
    cut = len(value)
    for marker in _TOPIC_CUT:
        position = value.find(marker)
        if 0 < position < cut:
            cut = position
    topic = _TOPIC_TAIL.sub("", value[:cut]).strip("的了吗呢吧 ")
    return topic[:40]


def history_topic(history: Sequence[Mapping[str, Any]]) -> str:
    """从最近几轮用户提问里取「上文主题」。

    只取用户消息：助手回答的开头往往是客套话，取它会把噪音带进查询。连续追问里
    上一轮可能只剩「文档中」这类泛指指向（C21.4 的上一轮是「关于文档中，他们是怎么
    设计 agent 的」），因此**优先取含实质研究对象的最近一轮**；只有实在没有时才退回
    最近一轮的非空主题。
    """

    turns = [str(turn.get("content") or "") for turn in history if turn.get("role") == "user"]
    if not turns:
        return ""
    fallback = ""
    for content in reversed(turns):
        topic = topic_phrase(content)
        if not topic:
            continue
        if not fallback:
            fallback = topic
        if has_research_object(topic):
            return topic
    return fallback


# =====================================================================
# 3. 合并与查询生成（确定性）
# =====================================================================


def _clause_spans(question: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _CLAUSE_SPLIT.finditer(question)]


def merge_queries(question: str) -> tuple[list[str], list[str]]:
    """把问题整理成可独立检索的查询；返回 ``(queries, notes)``。

    规则：按句末标点切分，只有「自带研究对象」的句子才另起一条查询；依附上文的
    补充句（类似…的那种、也就是说…、哪些维度值得考虑）与上一句合并，合并结果按
    **原问题的连续片段**取回，因此不会把用户的话改写成另一种说法。
    """

    text = _normalize(question)
    notes: list[str] = []
    if not text:
        return [], notes
    spans = _clause_spans(text)
    if not spans:
        return [text], notes

    groups: list[list[tuple[int, int]]] = []
    for span in spans:
        clause = text[span[0]:span[1]].strip()
        if not clause:
            continue
        if groups and not clause_is_standalone(clause):
            groups[-1].append(span)
            notes.append(f"补充句与上一句合并为一条查询：{clause[:30]}")
            continue
        groups.append([span])
    queries: list[str] = []
    for group in groups:
        start = group[0][0]
        end = group[-1][1]
        merged = _normalize(text[start:end])
        if merged and merged not in queries:
            queries.append(merged)
    if not queries:
        queries = [text]
    if len(queries) > 1:
        notes.append(f"识别出 {len(queries)} 个自带研究对象的主题，分别检索")
    return queries[:MAX_QUERIES], notes


#: 需要请模型整理查询的信号：多条要求、术语 + 概念意图、或明显的多轮承接。
_CONCEPT_INTENT: Final[tuple[str, ...]] = (
    "是什么", "什么意思", "含义", "定义", "解释", "概念", "原理", "机制",
    "场景", "用途", "用法", "怎么用", "如何使用", "有哪些", "举例", "例子",
)


def model_plan_needed(
    question: str, *, history: Sequence[Mapping[str, Any]] = (), needs_web: bool = False,
) -> bool:
    """是否需要请模型整理查询。

    只在「机器整理不出好查询」时才付出一次模型调用的代价：多个句子、含术语且问的是
    概念/场景、或本轮依赖上文。简单单句问题直接用原话检索，不额外增加延迟。
    """

    text = _normalize(question)
    if needs_web:
        return True
    if len(_clause_spans(text)) > 1:
        return True
    if _ASCII_TERM.search(text) and any(marker in text for marker in _CONCEPT_INTENT):
        return True
    return bool(history) and is_referential(text)


# =====================================================================
# 4. 模型输出解析（严格，不合法就回落）
# =====================================================================


def parse_query_plan(raw: Any, *, question: str, terms_hint: Sequence[str] = ()) -> QueryPlan:
    """解析模型输出；结构不合法时抛 :class:`ValueError`（调用方回落确定性结果）。

    ``queries`` 会被逐条校验：太长、空、纯提问用语、或完全没有出现在原问题 / 历史主题 /
    术语扩展里的查询，一律丢弃 —— 模型不能借「整理查询」把研究对象换掉。
    """

    payload: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise TypeError("查询规划结论必须是 JSON 对象")

    interpretation = _normalize(payload.get("interpretation"))
    raw_terms = payload.get("terms") or []
    terms = [_normalize(item) for item in raw_terms if _normalize(item)]
    raw_queries = payload.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    known = " ".join([question, *terms_hint, *terms]).casefold()
    known_terms = set(content_terms(known))
    queries: list[str] = []
    for item in raw_queries:
        candidate = _normalize(item)
        if not candidate or len(candidate) > MAX_QUERY_CHARS or is_instruction_only(candidate):
            continue
        if candidate in queries:
            continue
        own_terms = set(content_terms(candidate))
        if not own_terms & known_terms:
            # 查询与问题 / 历史主题 / 术语扩展没有任何信息重叠 -> 换了研究对象，丢弃。
            continue
        queries.append(candidate)
    if not queries:
        raise ValueError("模型没有给出可用查询")
    return QueryPlan(
        queries=queries[:MODEL_QUERY_COUNT],
        interpretation=interpretation,
        terms=terms,
        source="model",
        notes=["查询由模型结合一般知识与上下文整理"],
    )


# =====================================================================
# 5. 查询规划 Agent
# =====================================================================


class QueryPlannerAgent:
    """查询规划门面：确定性合并 + 可选的模型整理。

    ``model`` 是**同步**可调用（``(duty_prompt, prompt) -> str``），由运行层注入；
    未注入时完全走确定性路径（离线可用，且不产生任何出网）。
    """

    def __init__(
        self,
        model: Any = None,
        *,
        max_queries: int = MAX_QUERIES,
        use_model: bool = True,
    ) -> None:
        self.model = model
        self.max_queries = max(1, int(max_queries))
        self.use_model = bool(use_model)

    # ------------------------------------------------------------ 确定性

    def decide(self, question: str, *, history: Sequence[Mapping[str, Any]] = ()) -> QueryPlan:
        """不调用模型：合并补充句 + 带上文主题。"""

        queries, notes = merge_queries(question)
        topic = history_topic(history)
        if topic and (
            is_referential(question)
            or generic_document_scope(question) != ""
            or re.search(r"呢[？?。！!]*$", question)
            or _has_subject_anchor(question)
        ):
            folded, changed = merge_context_queries(queries, topic, question=question)
            if changed:
                notes.append(f"把上文主题合进本轮查询：「{topic}」")
                queries = folded
            elif topic.casefold() not in " ".join(queries).casefold():
                queries = [*queries, topic]
                notes.append(f"把上文主题带进检索：「{topic}」")
        return QueryPlan(queries=queries[: self.max_queries], source="rules", notes=notes)

    # ------------------------------------------------------------ 提示

    def build_prompt(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] = (),
        clauses: Sequence[str] = (),
        needs_web: bool = False,
    ) -> str:
        history_text = ""
        topic = history_topic(history)
        if topic:
            history_text = f"\n相关历史主题（上一轮用户问题）：{topic}\n"
        clause_text = ""
        if clauses:
            clause_text = "\n问题被切分出的片段：" + " | ".join(str(item) for item in clauses) + "\n"
        prompt = _QUERY_TEMPLATE.format(
            count=min(MODEL_QUERY_COUNT, self.max_queries),
            question=_normalize(question),
            history=history_text,
            clauses=clause_text,
        )
        if needs_web:
            prompt += (
                "\n本任务需要公开来源查证。请把自然语言对象解析成可搜索的官方产品、"
                "操作名称、参数名称与常用英文术语，生成能定位参数说明或规范正文的查询。"
                "保留创建/更新/返回值、版本、上下限等条件；各查询应定位不同的必要证据。"
                "推断的参数名只是检索线索，必须由后续官方原文验证，禁止提供未经查证的数值答案。"
            )
        return prompt

    # ------------------------------------------------------------ 主入口

    async def plan(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] = (),
        clauses: Sequence[str] = (),
        run_blocking: Any = None,
        needs_web: bool = False,
    ) -> QueryPlan:
        """整理查询：模型提供完整独立查询，未提供有效结果时使用确定性规划。"""

        base = self.decide(question, history=history)
        if self.model is None or not self.use_model:
            return base
        if not model_plan_needed(question, history=history, needs_web=needs_web):
            return base

        import asyncio

        prompt = self.build_prompt(
            question, history=history, clauses=clauses, needs_web=needs_web,
        )
        notes = list(base.notes)
        try:
            if run_blocking is not None:
                raw = await run_blocking(prompt)
            else:
                raw = await asyncio.to_thread(self.model, _QUERY_DUTY, prompt)
        except Exception:  # noqa: BLE001 - 模型不可用时确定性结果仍然可用
            notes.append("查询整理模型不可用，使用按原问题整理的查询")
            return QueryPlan(queries=base.queries, source="rules", notes=notes)

        try:
            planned = parse_query_plan(raw, question=question, terms_hint=base.queries)
        except (ValueError, TypeError, json.JSONDecodeError):
            notes.append("查询整理模型输出不合法，使用按原问题整理的查询")
            return QueryPlan(queries=base.queries, source="rules", notes=notes)

        if (
            len(base.queries) == 1
            and not re.search(r"对比|比较", question)
            and (
                _model_queries_are_facets(base.queries[0], planned.queries)
                or not re.search(r"比较|对比|分别|以及|和|与|同时|《.+》.+《", question)
            )
        ):
            return QueryPlan(
                queries=list(base.queries) if _model_queries_are_facets(base.queries[0], planned.queries) else [planned.queries[0]],
                interpretation=planned.interpretation,
                terms=planned.terms,
                source="model",
                notes=[*notes, *planned.notes, "多个关注面共用一条完整问题查询"],
            )

        queries = []
        seen = {_canonical(query) for query in queries}
        for query in planned.queries:
            key = _canonical(query)
            if key in seen:
                # 与已有查询只是标点 / 大小写不同 -> 同一条查询，不重复检索（合并重复意图）。
                continue
            seen.add(key)
            queries.append(query)
        merged, reuse_notes = merge_semantic_queries(
            queries,
            source_kind="web" if needs_web else "knowledge",
            context_text=question,
        )
        return QueryPlan(
            queries=merged[: self.max_queries],
            interpretation=planned.interpretation,
            terms=planned.terms,
            source="model",
            notes=[*notes, *planned.notes, *reuse_notes],
        )
