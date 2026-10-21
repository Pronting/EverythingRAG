"""Intent Agent、范围检查与路由规则（PRD 6.1–6.3）。Owner: T06（routing）。

本模块是**意图识别层**的唯一实现，三件事：

1. **路由输入组装**（PRD 6.1）：当前问题 + 当前日期与用户时区 + 必要最近历史 +
   本会话固定记忆快照 + 当前工作区主题概览 + 目录查询得到的术语与文件线索。
   提示走 PRD 9.3 的固定顺序（复用 ``app.agent.context.build_context``）。
2. **规则引擎**（PRD 6.3 九条规则）：确定性、可离线单测；模型只做补充，
   自报置信度仅进诊断，不参与验收。硬规则（产品/概览、时效、内部引用、空知识库）
   在合并时覆盖模型输出。
3. **范围检查与调整**（PRD 6.1 / 6.3 末段）：目录命中只用于定位原文；缺明确解释的
   术语标 ``definition_unknown`` 并进入 ``unresolved_terms``；Planner 可在研究过程中
   通过 :func:`adjust_route_decision` 调整来源需求。

``RouteDecision`` 字段定义在 :mod:`app.agent.contracts`（T00b 冻结，PRD 6.2 全 12
字段），本模块不重新定义模型，也不改契约。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent import query_plan
from app.agent.contracts import Freshness, IntentKind, RouteDecision
from app.agent.events import EventType, StoredEvent, build_event
from app.agent.web_budget import local_only_question
from app.generation.knowledge_catalog import (
    CatalogMatch,
    CatalogPromptProjection,
    KnowledgeCatalog,
)
from app.generation.prompt_policy import QueryMode, classify_query

__all__ = [
    "CLARIFICATION_MAX_QUESTION_CHARS",
    "DEFAULT_CATALOG_HIT_LIMIT",
    "MAX_RECENT_HISTORY",
    "ROUTING_RULES_VERSION",
    "RULE_DESCRIPTIONS",
    "IntentAgent",
    "RouteDiagnostics",
    "RouteParseError",
    "RoutePrompt",
    "RouteResult",
    "RoutingHistoryTurn",
    "RoutingInput",
    "RuleHit",
    "RuleOutcome",
    "ScopeCheckResult",
    "adjust_route_decision",
    "build_route_decided_event",
    "build_route_prompt",
    "check_scope",
    "evaluate_route_rules",
    "resolve_reference",
    "route_decided_payload",
    "route_question",
]

#: 路由规则版本（规则层语义变化时递增；路由评测报告会记录它）。
ROUTING_RULES_VERSION: Final[str] = "routing-rules-v2"

#: 进入路由提示的目录条目数上限（完整目录留在本地，PRD 6.1）。
DEFAULT_CATALOG_HIT_LIMIT: Final[int] = 6

#: 路由输入里最多带的最近历史轮数（必要历史，非全量）。
MAX_RECENT_HISTORY: Final[int] = 8

#: 短问题 + 含义不明术语 → 主动澄清（PRD 6.3 R4 / 13 章）。
CLARIFICATION_MAX_QUESTION_CHARS: Final[int] = 24

#: 目录命中的最低相关性：只有标题 / 术语 / 别名级别的**子串级**命中才算「线索」。
#: 仅词元重叠（100–199）或词元恰好出现在标题里（300）不算，避免把「这个方案呢？」
#: 这类泛指问题误判成内部资料问题。
CATALOG_CLUE_MIN_SCORE: Final[int] = 400

#: PRD 6.3 决策规则表（规则 id 与描述一一对应；R11 为 2026-10 评测后新增的默认检索兜底）。
RULE_DESCRIPTIONS: Final[dict[str, str]] = {
    "R1": "用户明确引用笔记、内部文档、既有方案或当前项目 → 查询知识库",
    "R2": "对话已围绕某内部资料展开，用户继续追问 → 历史补全对象并继续知识查询",
    "R3": "内部术语目录命中且问题涉及其在本系统中的定义或实现 → 查询对应原文",
    "R4": "术语含义不明确、可能指内部对象 → 范围检查后决定路径",
    "R5": "用户明确要求今天、最新、当前版本等信息 → 联网查询并检查时间",
    "R6": "通用解释同时涉及已有内部方案 → General 与 RAG 分工",
    "R7": "仅询问产品功能或文档数量 → 使用产品说明或结构化统计",
    "R8": "用户回忆以前的讨论 → 查询工作区记忆与历史记录",
    "R9": "知识库为空 → 通用与网页问题仍可处理；内部资料问题说明当前资料情况",
    "R10": "围绕已有对话补充观点或评价 → 使用当前上下文和通用知识",
    "R11": "无法确定是否与本区资料相关 → 默认检索一次，检索不到再退回通用知识",
}

_INTERNAL_MARKERS: Final[tuple[str, ...]] = (
    "我的笔记",
    "我笔记",
    "笔记里",
    "我的文档",
    "我的资料",
    "本地文档",
    "本地资料",
    "本地笔记",
    "本地知识",
    "我们内部",
    "内部资料",
    "内部文档",
    "内部方案",
    "内部系统",
    "内部术语",
    "我们的项目",
    "我们项目",
    "本项目",
    "当前项目",
    "我们的方案",
    "我们方案",
    "既有方案",
    "现有方案",
    "我们的实现",
    "我们实现",
    "我们的代码",
    "知识库",
    "资料库",
    "导入的",
    "工作区",
    "公司内部",
)

_FRESHNESS_CURRENT_MARKERS: Final[tuple[str, ...]] = (
    "今天",
    "今日",
    "最新",
    "最新版",
    "最近",
    "现在",
    "实时",
    "刚刚",
    "刚发布",
    "当下",
    "如今",
    "目前最新",
    "这几天",
    "这两天",
    "当前版本",
    "最新版本",
)

_FRESHNESS_RANGE_MARKERS: Final[tuple[str, ...]] = (
    "本周",
    "这周",
    "上周",
    "本月",
    "这个月",
    "上个月",
    "本季度",
    "上个季度",
    "今年",
    "去年",
    "近一周",
    "近一个月",
    "近三个月",
    "近半年",
    "近一年",
    "最近一周",
    "最近一个月",
    "最近几天",
    "近期",
    "过去一周",
    "过去一个月",
    "过去一年",
)

_FRESHNESS_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:从|自)\s*\d{4}\s*[-/年]|近\s*\d+\s*(?:天|周|个月|年)|"
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*(?:到|至|-|~)"
)

_PRONOUNS: Final[tuple[str, ...]] = (
    "这个方案",
    "那个方案",
    "这个文档",
    "那个文档",
    "这个机制",
    "那个机制",
    "这种方式",
    "那种方式",
    "第二种方式",
    "第一种方式",
    "该方案",
    "该文档",
    "该机制",
    "那套",
    "这套",
    "那种",
    "这种",
    "这篇",
    "那篇",
    "上述",
    "前者",
    "后者",
    "这些",
    "那些",
    "这个",
    "那个",
    "它",
)

#: 自带名词的指代表达：就地替换成语义完整（「这个方案」→「内部风控 P99 方案」）。
_PHRASE_PRONOUNS: Final[tuple[str, ...]] = tuple(
    item for item in _PRONOUNS if item[-1] in "案档制式套篇"
)

#: 裸指代（这个 / 它 / 上述…）：它后面往往还跟着名词（「这个面试题」），就地替换会把
#: 两段话说在一起（用户实测缺陷：「回答有一个很经典的一个面试题面试题」）。因此裸指代
#: 用**前置上文主题**的方式补全 —— 用户本轮的原话一字不改。
_BARE_PRONOUNS: Final[tuple[str, ...]] = tuple(
    item for item in _PRONOUNS if item not in _PHRASE_PRONOUNS
)

_MEMORY_MARKERS: Final[tuple[str, ...]] = (
    "搜索以前的对话",
    "搜索之前的对话",
    "查询历史对话",
    "检索历史会话",
    "查找以前的对话",
    "以前的对话",
    "之前的会话",
    "历史聊天记录",
    "还记得",
    "我之前问过",
    "我以前问过",
    "我们之前聊过",
    "之前讨论过",
    "之前我们",
    "上次我们",
    "上次提到",
    "之前提到过",
    "以前说过",
    "之前说过",
    "我提过",
    "我们聊过",
    "回忆一下",
    "帮我回忆",
)


def source_exclusions(question: str) -> tuple[bool, bool]:
    search_denial = re.search(
        r"(?:不要|不用|无需|禁止|不必|别)(?:再)?(?:检索|搜索|查询|查找|查)(?:本地)?(?:资料|文档|知识库)?(?:或|和|及)?(?:联网|上网)?",
        question,
    )
    no_search = search_denial is not None
    local_only = bool(search_denial and re.search(r"本地|知识库|内部", search_denial.group(0)))
    no_web = bool(re.search(r"(?:不要|不用|无需|禁止|不必|别)[^。！？]{0,15}(?:联网|上网)", question))
    return no_search, bool(no_web or no_search and not local_only and not explicit_web_request(question))


def _apply_source_exclusions(decision: RouteDecision, question: str) -> RouteDecision:
    if history_only_request(question):
        decision.intent = IntentKind.MEMORY
        decision.needs_history = True
        decision.needs_knowledge = False
        decision.needs_web = False
        decision.needs_general = False
        decision.freshness = Freshness.NONE
        decision.reason_summary = "查询以前的对话并读取原始消息"
    no_search, no_web = source_exclusions(question)
    if no_search:
        decision.needs_knowledge = False
    if no_web:
        decision.needs_web = False
        decision.freshness = Freshness.NONE
    if no_search and not decision.needs_history and not decision.needs_web:
        decision.intent = IntentKind.GENERAL
        decision.needs_general = True
        decision.needs_source_search = False
        decision.reason_summary = "按当前会话要求直接回复"
    return decision


def history_only_request(question: str) -> bool:
    if re.search(
        r"(?:同时|然后|另外|并且|再|并现在|并当前)(?:请)?"
        r"(?:联网|检索本地|查询本地|搜索本地|(?:核实|核对|查看|查询|检索|搜索|确认)[^。；！？]{0,30}(?:官方|官网|网页))",
        question,
    ):
        return False
    return bool(re.search(
        r"(?:查询|搜索|检索|查找|回顾)(?:一下)?(?:我|我们)?(?:以前|之前|历史)(?:的)?(?:一段|一次|一条)?(?:对话|会话|聊天记录)",
        question,
    ))


def discussion_request(question: str, *, has_history: bool) -> bool:
    if not has_history:
        return False
    if _detect_freshness(question) is not Freshness.NONE or source_requirement(question):
        return False
    if re.search(
        r"搜索|检索|查找|查询|联网|上网|核实|核查|核对|查证|验证|最新|今天|当前版本|"
        r"引用|出处|证据|原文依据|补充(?:一些|更多|相关|的)?(?:资料|数据|事实|来源)|"
        r"(?:根据|依据|按照|只用|仅用).{0,12}(?:原文|文档|资料|笔记)|"
        r"(?:原文|文中|资料里|笔记里).{0,12}(?:说|写|提|补充|还有)|"
        r"(?:叫什么|怎么做的|如何实现|具体做法|实际效果|量化结果)|"
        r"\b(?:search|look up|verify|latest|sources?|citations?)\b",
        question, re.IGNORECASE,
    ):
        return False
    return bool(re.search(
        r"你(?:自己)?(?:怎么(?:看|想|评价)|有(?:什么|哪些)(?:看法|观点|建议)|觉得|认为)|"
        r"你(?:自己)?的(?:看法|观点|评价|建议)|"
        r"(?:谈谈|说说|给出|补充).{0,8}(?:看法|观点|评价|建议)|"
        r"(?:有(?:什么|哪些)|还(?:有|能)).{0,5}补充(?:的)?(?:说法|观点|看法|建议)?|"
        r"\b(?:what do you think|your (?:opinion|thoughts)|anything to add)\b",
        question, re.IGNORECASE,
    ))

_GENERAL_MARKERS: Final[tuple[str, ...]] = (
    "是什么",
    "为什么",
    "原理",
    "概念",
    "怎么理解",
    "如何理解",
    "一般",
    "通常",
    "常见的",
    "最佳实践",
    "设计建议",
    "架构建议",
    "有什么通用方法",
    "怎么排查",
)

_DEFINITION_INTENT_MARKERS: Final[tuple[str, ...]] = (
    "定义",
    "是什么",
    "什么意思",
    "含义",
    "怎么实现",
    "如何实现",
    "实现方式",
    "原理",
    "机制",
    "怎么做的",
    "如何落地",
)

#: 「一般知识可以参与理解」的概念 / 场景意图（PRD 6.3 R6 的术语分支）。
#: 出现这些问法时，模型可以先用一般知识理解术语，再据此组织检索表达。
_CONCEPT_INTENT_MARKERS: Final[tuple[str, ...]] = (
    "是什么",
    "什么意思",
    "含义",
    "定义",
    "解释",
    "概念",
    "原理",
    "机制",
    "场景",
    "用途",
    "用法",
    "怎么用",
    "如何使用",
    "有哪些",
    "举例",
    "例子",
    "介绍",
    "怎么做",
    "怎么做的",
    "如何做",
    "一回事",
    "一样吗",
    "有什么区别",
)

#: 术语形态（一个或多个连续英文词）：研究对象最容易被一般知识识别的形态。
_TERM_SPAN: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-]*(?:\s+[A-Za-z][A-Za-z0-9_.\-]*)*"
)
_TERM_SPAN_STOP: Final[frozenset[str]] = frozenset(
    {"a", "an", "and", "or", "the", "is", "are", "of", "to", "in", "on", "for", "with", "how", "what"}
)

#: 「某份文档在哪里 / 有没有」这类**文档寻址**问法：问的是本工作区有没有这份资料，
#: 必须先真的检索一次再回答（否则「没找到」是没检索过就下的结论）。
_DOC_SEEK_INTENT: Final[re.Pattern[str]] = re.compile(
    r"(?:在哪|在哪里|哪里|哪个文档|哪个文件|哪一份|找一下|找找|查一下|翻一下|"
    r"有没有|有吗|是否有|收录)",
)
_DOC_SEEK_OBJECT: Final[re.Pattern[str]] = re.compile(
    r"(?:手册|文档|文件|资料|笔记|记录|报告|规范|说明书|wiki|知识库|文档库)",
)

#: 指向**知识库里某一篇具体资料**的定指标记（「那篇文章」「这篇文章」「文中」「原文」）。
_DOC_POINTER: Final[re.Pattern[str]] = re.compile(
    r"(?:那篇|这篇|该篇|这一篇|那一篇|此篇|那篇文章|这篇文章|该文章|本文|文中|原文|"
    r"文章里|文章中的|给出的文章|指定的文章|那本书|这篇文章中)"
)
#: 「给出**文中的依据**」这类**求文档论据**的要求：用户点名的是知识库里的既有原文，
#: 必须真的去读，不能用通用解释顶替（用户实测缺陷 F14）。
_DOC_EVIDENCE_DEMAND: Final[re.Pattern[str]] = re.compile(
    r"(?:文中的|原文的|文章中的|文档中的|该书中的|文里)"
    r"[^。！？?]{0,8}"
    r"(?:依据|出处|原文|论断|证据|说明|理由|例子|数据|结论|观点|怎么(?:说|写|讲))"
)
#: 问「某篇文章/某份资料说明了什么」的内容问法（不是问清单，也不是问产品）。
_DOC_CONTENT_ASK: Final[re.Pattern[str]] = re.compile(
    r"(?:如何(?:说明|解释|描述|阐述|处理|计算|实现|规定|定义)|怎么(?:说明|解释|写|讲)|"
    r"为什么|说明了什么|讲了什么|说了什么|介绍了(?:什么|哪些)|有哪些(?:做法|办法|措施|方案|依据|内容)|"
    r"怎么(?:做|算|处理))"
)
#: 「介绍 X 的那篇文章/那份资料」这类**由内容反指文档**的说法（用户实测 T12：
#: 「知识库中介绍 K8S 十四个基础概念的文章，如何说明 Deployment 和 Service 各自的
#: 职责？」）。它不是问清单，而是问那篇文章里的具体说法。
_CONTENT_DESCRIBED_DOC: Final[re.Pattern[str]] = re.compile(
    r"(?:介绍|讲|写|谈|说)[^。！？?，,]{0,24}?(?:的)?(?:那|这|该)?"
    r"(?:篇|份|个)?(?:文章|文档|资料|笔记|实践|文章里|报告|规范)"
)
#: 求「某个具体方面的说明」的内容要素：出现即说明问的是正文内容，不是清单
#: （「各自的职责」「具体实践」「怎么计算」…）。
_CONTENT_ASPECT: Final[re.Pattern[str]] = re.compile(
    r"(?:职责|作用|区别|差异|关系|原理|机制|流程|步骤|公式|口径|定义|边界|"
    r"如何|怎么|为什么|具体|实践|依据|例子|举例|数量|范围|阈值|配置|实现)"
)


def document_intent(question: str) -> str:
    """问题是否在**指向知识库里某一篇具体资料的内容**；是则返回可读线索。

    与「知识库有哪些文章」区分开（用户实测缺陷 F15）：这里问的是某篇文章说了什么、
    文中依据是什么，因此必须走正文研究，而不是只输出结构化清单或通用解释。
    覆盖四类可复核问法：
    ``某篇文章 + 内容问法``、``文中依据类要求``、``文档寻址``、
    ``介绍 X 的文章 + 内容要素``。
    """

    text = " ".join(str(question or "").split())
    if not text:
        return ""
    named_document = re.search(r"《[^》]+》", text)
    if named_document is not None and re.search(r"多少|哪些|什么|如何|为什么|怎样|说明|列出|介绍|总结|概述|引用|依据", text):
        return named_document.group(0)
    pointer = _DOC_POINTER.search(text)
    if pointer is not None and (_DOC_CONTENT_ASK.search(text) or _DOC_SEEK_OBJECT.search(text)):
        return pointer.group(0)
    demand = _DOC_EVIDENCE_DEMAND.search(text)
    if demand is not None:
        return demand.group(0)
    seek = _DOC_SEEK_INTENT.search(text)
    if seek is not None and _DOC_SEEK_OBJECT.search(text):
        return seek.group(0)
    described = _CONTENT_DESCRIBED_DOC.search(text)
    if described is not None and (
        _DOC_CONTENT_ASK.search(text) or _CONTENT_ASPECT.search(text)
    ):
        return described.group(0)
    # 用户点名「资料 / 文档」但**没有给出唯一文章名**（C21.3「关于文档中，他们是怎么
    # 设计 agent 的」）：语义上就是「查本地资料里的内容」，与上面「点名某篇文章」同属
    # 内容问法。判定在 query_plan 里唯一实现（泛指文档指向 + 内容 / 寻址意图），这里只取
    # 它的结果，避免两处各写一套「什么算问文档内容」。
    return query_plan.generic_document_content_ask(text)


def _document_intent(question: str) -> str:
    """内部别名（规则层用；与 :func:`document_intent` 同一实现）。"""

    return document_intent(question)
#: 联网动作的表达（动词短语）。用户实测缺陷 F05：``联网搜索`` 能触发网络工具，而
#: 近义的 ``联网查找`` 不动 —— 固定表达清单只能补个别用例，因此这里覆盖**动作短语**
#: 而不是逐句枚举：``联网/上网/在线`` + ``搜/查/找/检索/核实/核对/验证/求证``，
#: 以及 ``网页/网上/公开资料`` 这类明确指向外部来源的表达。
_WEB_ACTION: Final[str] = r"(?:搜索|搜一下|搜一搜|查询|查一下|查找|查查|检索|寻找|找一下|找找|找|核实|核对|验证|求证|确认|看看|了解一下|浏览)"
_WEB_SOURCE: Final[str] = r"(?:联网|上网|在线|网上|网页|互联网|公开资料|公开来源|外部资料|外部来源|官方网站|官网)"
#: 动作 + 紧邻的来源要求（「查一下官方文档」「搜索官方链接」）：两者之间允许一段
#: 对象名（「查一下 Kubernetes 官方文档」），但不能跨句（间隔里出现句末标点即不算）。
_OFFICIAL_ASK: Final[str] = r"(?:官方网站|官网|官方文档|官方资料|官方来源|官方页面)"
#: 三段拆分 + VERBOSE：把带量词的片段各自独立，避免交替 / 量词的优先级歧义。
_EXPLICIT_WEB_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
      (?P<src_then_action> {_WEB_SOURCE} \s* {_WEB_ACTION} )
    | (?P<action_then_src> {_WEB_ACTION} \s*{_WEB_SOURCE} )
    | (?P<action_then_official> {_WEB_ACTION} [^。！？?；;\n]{{0,20}}? {_OFFICIAL_ASK} )
    """,
    re.VERBOSE,
)
#: 否定词（出现在联网动作**紧前方**的窗口里即视为否定该动作）。
#: 只看紧邻窗口，因此不会因为句首有「不用」就否定后面的联网要求
#: （「文档里没有就说，不用联网搜索」这类就近判定仍然正确）。
_WEB_NEGATION_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:不要|不用|无需|不必|不得|禁止|不允许|请勿|别去|别)"
)
#: 明确要求「公开可核实的来源 / 官方链接」：这是**交付要求**，不只是检索动作。
_SOURCE_REQUIREMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:官方(?:文档|资料|来源|网站|链接|页面|说明|原文)|公开(?:来源|资料|信息)|"
    r"官网|原始出处|一手来源)"
)


def explicit_web_request(question: str) -> str:
    """明确要求联网的表达（无 / 有则返回命中的动作短语）。

    否定表达（「不要联网查找」）优先判定，命中即返回空串 —— 用户明确排除的能力
    不能被当成要求。联网意图的识别同时供路由规则与联网查询构造使用，避免两处各写
    一套「什么算联网」。
    """

    text = " ".join(str(question or "").split())
    if not text:
        return ""
    for match in _EXPLICIT_WEB_RE.finditer(text):
        prefix = text[: match.start()]
        # 只看**紧邻动作之前**的否定窗口（去掉标点空白后约 12 字）：既覆盖
        # 「不要联网查找」这种紧贴否定，也不会把远处的「不用」误判成对本次动作的否定。
        window = re.sub(r"[\s，,、：:；;。！!？?]", "", prefix)[-12:]
        if _WEB_NEGATION_WORD_RE.search(window):
            continue
        return match.group(0)
    return ""


#: 明确指定机构的检索目标 → 官方域名（只收**确定**的官方文档域名，不猜）。
#: 用户实测缺陷 F17：T19/T20 要求「官方来源 / 官方链接」，但请求并未体现这一点。
OFFICIAL_DOMAINS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("kubernetes", ("kubernetes.io",)),
    ("k8s", ("kubernetes.io",)),
    ("bedrock", ("docs.aws.amazon.com", "aws.amazon.com")),
    ("agentcore", ("docs.aws.amazon.com", "aws.amazon.com")),
    ("aws", ("docs.aws.amazon.com", "aws.amazon.com")),
    ("amazon", ("docs.aws.amazon.com", "aws.amazon.com")),
    ("azure", ("learn.microsoft.com", "azure.microsoft.com")),
    ("microsoft", ("learn.microsoft.com",)),
    ("google cloud", ("cloud.google.com",)),
    ("阿里云", ("help.aliyun.com",)),
    ("腾讯云", ("cloud.tencent.com",)),
    ("docker", ("docs.docker.com",)),
    ("python", ("docs.python.org",)),
    ("react", ("react.dev",)),
    ("openai", ("platform.openai.com",)),
    ("openclaw", ("docs.openclaw.ai",)),
    ("anthropic", ("docs.anthropic.com",)),
    ("claude", ("docs.anthropic.com",)),
    ("postgres", ("www.postgresql.org", "postgresql.org")),
    ("mysql", ("dev.mysql.com",)),
    ("redis", ("redis.io",)),
    ("nginx", ("nginx.org",)),
    ("linux", ("www.kernel.org", "man7.org")),
)


def official_domains(question: str) -> list[str]:
    """问题里点名的机构 → 可信官方域名（用于限定查询与选页，缺陷 F17）。

    只返回**确定的**官方文档域名；无法确定时返回空列表 —— 不猜域名，也不因此拒绝
    回答（用户裁定：额度是上限，信息不足时如实说明缺口）。
    """

    text = str(question or "").casefold()
    domains: list[str] = []
    for marker, values in OFFICIAL_DOMAINS:
        if marker in text:
            for domain in values:
                if domain not in domains:
                    domains.append(domain)
    return domains


def source_requirement(question: str) -> str:
    """问题里对**来源类型**的明确要求（官方文档 / 公开来源 / 官网 / 一手来源）。

    用户实测缺陷 F17：T19、T20 都要求「官方来源 / 官方链接」，但搜索仍直接发送用户
    整句问题、选页直接取前三条第三方结果，最终没有官方依据却登记 completed。
    本函数把这个要求提取成**可执行的检索与验收条件**。
    """

    if history_only_request(question):
        return ""
    text = " ".join(str(question or "").split())
    match = _SOURCE_REQUIREMENT_RE.search(text)
    return match.group(0) if match is not None else ""


def concept_terms(question: str) -> list[str]:
    """问题里的术语表达（仅在问概念/场景/定义时返回）。

    用途：让「用一般知识理解术语」这一步**真的参与检索** —— 术语被填进
    ``RouteDecision.entities``，随后成为检索查询；但术语扩展只是**检索线索**，
    不能据此断言文档讨论过该概念（route prompt 与综合提示都写明这一点）。
    """

    text = " ".join(str(question or "").split())
    if not any(marker in text.casefold() for marker in _CONCEPT_INTENT_MARKERS):
        return []
    terms: list[str] = []
    for match in _TERM_SPAN.finditer(text):
        term = match.group(0).strip(" .-_")
        if len(term) < 2 or term.casefold() in _TERM_SPAN_STOP:
            continue
        if term not in terms:
            terms.append(term)
    # 短语里的短词只是长术语的前缀（「AI」⊂「AI Native」）：只保留更具体的表达，
    # 避免把「AI」这种超泛词当成一条检索查询。
    return [
        term
        for term in terms
        if not any(other != term and other.casefold().startswith(term.casefold()) for other in terms)
    ]

_TOPIC_CUT_MARKERS: Final[tuple[str, ...]] = (
    "？",
    "?",
    "。",
    "，",
    ",",
    "是",
    "怎么",
    "如何",
    "为什么",
    "有哪些",
    "哪些",
    "多少",
)

#: 明确的非知识对话（问候/致谢/翻译/代写/闲聊）：这些不该触发默认检索。
_NON_KNOWLEDGE_CHAT: Final[re.Pattern[str]] = re.compile(
    r"^(?:你好|hi|hello|在吗|谢谢|感谢|再见|你是谁|你能做(?:什么|啥)|"
    r"(?:把|帮我把).{0,20}翻译|翻译(?:成|为)|帮我写(?:一段|一个|个|份)|润色|改写(?:这段|一下)|"
    r"讲个笑话|随便聊聊|今天天气)",
    re.IGNORECASE,
)

#: 默认检索兜底的最短问题长度：太短的追问（「好」「继续」）交给历史与澄清路径。
_DEFAULT_RETRIEVAL_MIN_CHARS: Final[int] = 8

_ROUTE_DECISION_FIELDS: Final[frozenset[str]] = frozenset(RouteDecision.model_fields)


class _RoutingModel(BaseModel):
    """路由模型的共同基类（与共享契约一致：多传字段直接报错）。"""

    model_config = ConfigDict(extra="forbid")


class RouteParseError(ValueError):
    """模型返回的路由结论结构不合法（未知 intent、非法 JSON 等）。"""


# =====================================================================
# 1. 路由输入（PRD 6.1）
# =====================================================================


class RoutingHistoryTurn(_RoutingModel):
    """必要的最近历史轮次（路由只带必要历史，不带全量会话）。"""

    role: Literal["user", "assistant"]
    content: str
    created_at: str = ""


class RoutingInput(_RoutingModel):
    """Intent Agent 的全部输入（PRD 6.1）。

    ``catalog_hits`` 由 :meth:`KnowledgeCatalog.search` 产出；``catalog_clues``
    是更轻的线索文本（例如工作区主题概览里的词），供规则层在无目录实例时使用。
    """

    question: str
    now: str = ""
    timezone: str = ""
    recent_history: list[RoutingHistoryTurn] = Field(default_factory=list)
    memory_snapshot: str = ""
    workspace_overview: str = ""
    catalog_hits: list[CatalogMatch] = Field(default_factory=list)
    catalog_clues: list[str] = Field(default_factory=list)
    file_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    workspace_id: str = ""
    knowledge_revision: int = Field(default=0, ge=0)

    @property
    def knowledge_available(self) -> bool:
        """知识库是否非空（PRD 6.3 R9 的判定依据）。"""

        return self.file_count > 0

    def recent_user_turns(self) -> list[RoutingHistoryTurn]:
        return [turn for turn in self.recent_history if turn.role == "user"]


# =====================================================================
# 2. 规则层（PRD 6.3）
# =====================================================================


class RuleHit(_RoutingModel):
    """命中的一条决策规则及可复核的证据串。"""

    rule_id: str
    description: str
    evidence: str = ""


class RuleOutcome(_RoutingModel):
    """规则层输出：结论 + 命中规则 + 说明（不含模型调用）。"""

    decision: RouteDecision
    hits: list[RuleHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def rule_ids(self) -> list[str]:
        return [hit.rule_id for hit in self.hits]


class RouteDiagnostics(_RoutingModel):
    """诊断信息：只用于排查，不作为验收依据（PRD 6.2 末段）。"""

    model_used: bool = False
    confidence: float | None = None
    dropped_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ScopeCheckResult(_RoutingModel):
    """范围检查结果（PRD 6.1）：目录只用于定位原文，不构成依据。"""

    decision: RouteDecision
    knowledge_required: bool = False
    web_required: bool = False
    catalog_empty: bool = True
    scope_evidence_ids: list[str] = Field(default_factory=list)
    resolved_terms: list[str] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    definition_unknown_terms: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RouteResult(_RoutingModel):
    """一次路由的完整结果（图节点消费 ``decision``，评测消费全部字段）。"""

    decision: RouteDecision
    scope: ScopeCheckResult
    rules: list[RuleHit] = Field(default_factory=list)
    diagnostics: RouteDiagnostics = Field(default_factory=RouteDiagnostics)


def _contains(text: str, markers: Sequence[str]) -> str:
    """返回第一个命中的标记（便于记录可复核证据），未命中返回空串。"""

    for marker in markers:
        if marker in text:
            return marker
    return ""


def _detect_freshness(question: str) -> Freshness:
    if _FRESHNESS_RANGE_RE.search(question):
        return Freshness.DATE_RANGE
    if _contains(question, _FRESHNESS_RANGE_MARKERS):
        return Freshness.DATE_RANGE
    if _contains(question, _FRESHNESS_CURRENT_MARKERS):
        return Freshness.CURRENT
    return Freshness.NONE


def _topic_of(text: str) -> str:
    """从上一轮问题里取一个可替换指代的对象（唯一实现见 ``query_plan.topic_phrase``）。"""

    return query_plan.topic_phrase(text)


def _default_retrieval_queries(question: str) -> list[str]:
    """默认检索兜底的查询：**只发原句**，不做同义扩展，也不再拼主题短语。

    实测依据（2026-10-20/21 评测）：原句在 BM25 上已能把金标文档排到 1–2 名；把同义词、
    文档词汇拼进多查询融合反而给别的文档加词法分，让金标掉到 6–8 名。

    再补一条「去掉问句尾巴的主题短语」同样有害：它与原句命中同一批文档（sem_daemon
    两种发法都是 6 块 / 2 文档），却把上下文从 5–7 块顶到 8 块（sem_cache 5→8、
    sem_attr2 7→8），多出来的是同一文档的相邻块，直接摊薄合成阶段的证据密度。
    兜底只负责「把检索发出去」，扩词留给有 rerank 的场景。
    """

    return [" ".join(question.split())]


def _longest_pronoun(question: str) -> str:
    for pronoun in sorted(_PRONOUNS, key=len, reverse=True):
        if pronoun in question:
            return pronoun
    return ""


def _reference_topic(previous: Sequence[RoutingHistoryTurn]) -> str:
    """上一轮里可用来补全指代的对象（连续追问的保护层）。

    用户实测缺陷 C21.4：上一轮主题只剩「文档中」这类泛指指向时，直接把「文档中」
    填进本轮会把研究对象丢掉（Agent 对象在更早一轮）。因此**优先取含实质研究对象的
    最近一轮**（``query_plan.has_research_object``），只有实在没有时才退回最近一轮的
    非空主题，语义与旧实现一致（旧行为仍然覆盖）。
    """

    latest = ""
    for turn in reversed(previous):
        topic = _topic_of(turn.content)
        if not topic:
            continue
        if not latest:
            latest = topic
        if query_plan.has_research_object(topic):
            return topic
    return latest


def resolve_reference(
    question: str, history: Sequence[RoutingHistoryTurn]
) -> tuple[str, str]:
    """按历史补全指代（PRD 6.3 R2）：返回 ``(resolved_question, note)``。

    两类指代分开处理（用户实测缺陷 C04/C05）：

    - **自带名词的指代**（「这个方案」「这个文档」「这篇」）→ 就地替换成上一轮的对象，
      语义完整且不留指代；
    - **裸指代**（「这个」「它」「上述」）→ 把上一轮主题作为**上下文前置**，本轮原话
      一字不改。早期实现对裸指代也做就地替换，于是「如果让你回答这个面试题的话」被写成
      「回答有一个很经典的一个面试题面试题」这类重复表达，检索查询也跟着被污染。

    没有指代词或没有可用历史时原样返回，并把原因写进 note，不猜测对象。
    """

    matched = _longest_pronoun(question)
    if not matched:
        return question, ""
    previous = [turn for turn in history if turn.role == "user"]
    if not previous:
        return question, "出现指代但本轮没有可用历史，未补全对象"
    topic = _reference_topic(previous)
    if not topic:
        return question, "上一轮问题没有可提取的对象，未补全指代"
    if matched in _PHRASE_PRONOUNS:
        return question.replace(matched, topic, 1), f"指代补全：{matched} → {topic}"
    return f"{topic} —— {question}", f"指代补全（上文主题）：{matched} → {topic}"


def _catalog_term_matches(input: RoutingInput) -> list[CatalogMatch]:
    return [match for match in input.catalog_hits if match.kind == "term"]


def _catalog_document_matches(input: RoutingInput) -> list[CatalogMatch]:
    return [
        match
        for match in input.catalog_hits
        if match.kind != "term" and match.score >= CATALOG_CLUE_MIN_SCORE
    ]


def _mentioned_in_question(question: str, match: CatalogMatch) -> bool:
    """术语/别名是否真的出现在问题里（弱命中不能当「本轮相关术语」）。"""

    normalized = question.casefold()
    for label in (match.matched_on, *match.aliases):
        wanted = str(label or "").casefold().strip()
        if len(wanted) >= 2 and wanted in normalized:
            return True
    return False


def _mentioned_term_matches(question: str, input: RoutingInput) -> list[CatalogMatch]:
    return [
        match for match in _catalog_term_matches(input) if _mentioned_in_question(question, match)
    ]


def evaluate_route_rules(input: RoutingInput) -> RuleOutcome:
    """纯规则路由（无模型调用）：落地 PRD 6.3 九条规则。

    规则按固定优先级求值；所有命中规则都记进 ``hits``（不合并成单一原因），
    因此评测可以逐条核对「哪条规则把问题定向到了哪个来源」。
    """

    question = " ".join(str(input.question or "").split())
    resolved_question, reference_note = resolve_reference(question, input.recent_history)
    hits: list[RuleHit] = []
    notes: list[str] = []

    def hit(rule_id: str, evidence: str) -> None:
        hits.append(
            RuleHit(
                rule_id=rule_id,
                description=RULE_DESCRIPTIONS[rule_id],
                evidence=evidence,
            )
        )

    decision = RouteDecision(
        question=question,
        resolved_question=resolved_question or question,
        intent=IntentKind.RESEARCH,
    )
    if reference_note:
        notes.append(reference_note)

    if discussion_request(question, has_history=bool(input.recent_history)):
        hit("R10", "本轮请求基于已有对话表达观点")
        decision.intent = IntentKind.GENERAL
        decision.needs_general = True
        decision.needs_source_search = False
        decision.reason_summary = "结合当前对话和通用知识补充观点"
        return RuleOutcome(decision=decision, hits=hits, notes=notes)

    # ---- 内容线索（先算，R7 的早退要用它判断「产品/概览」是否其实是内容问题）----
    internal_marker = _contains(question, _INTERNAL_MARKERS)
    term_matches = _mentioned_term_matches(question, input)
    document_matches = _catalog_document_matches(input)
    unknown_terms = [match for match in term_matches if match.definition_unknown]
    document_intent = _document_intent(question)
    generic_document_intent = query_plan.generic_document_content_ask(question)
    explicit_document_intent = (
        document_intent if document_intent != generic_document_intent else ""
    )
    # 泛指文档内容（「文档中，他们是怎么设计的」）比「指定资料的内容问法」更精确：
    # C21.3 会同时匹配后者（正则过宽），因此只把它作为 R1 的证据文案，避免同一条规则
    # 出现两条语义不同的命中。是否只有泛指、没有明确文章名，由 R1 分支单独判定。
    generic_document_only = False

    # ---- R7：仅询问产品功能或资料概览
    # 早退的前提是「问题确实只是在问产品/概览」。用户实测缺陷 F11/F15：T12 问的是
    # 「知识库中介绍 K8S 十四个基础概念的文章，如何说明 Deployment 和 Service 各自的
    # 职责」—— 这是**指定文章的内容问题**，却因为句中出现「知识库」被归入 catalog，
    # needs_knowledge=False，联网额度/查询规划都被跳过；T14 问「某篇文章…文中的依据」
    # 连一次检索都没有执行。因此：命中内容线索（点名文档 / 求文中依据 / 目录文件线索）
    # 时保留 R7 的意图标签，但**同时**要求走正文研究。
    mode = classify_query(question)
    content_question = document_intent or generic_document_intent or bool(document_matches)
    if mode is QueryMode.PRODUCT_HELP:
        hit("R7", "产品功能问题标记命中")
        decision.intent = IntentKind.PRODUCT
        decision.reason_summary = "询问产品功能，使用产品说明回答"
        if content_question:
            decision.needs_knowledge = True
            notes.append(
                "问题同时点名了具体资料内容：保留产品意图，但必须按正文研究回答（不只看产品说明）"
            )
        else:
            return RuleOutcome(decision=decision, hits=hits, notes=notes)
    elif mode is QueryMode.KNOWLEDGE_OVERVIEW:
        hit("R7", "文档数量 / 概览问题标记命中")
        decision.intent = IntentKind.CATALOG
        decision.reason_summary = "询问工作区资料概览，使用结构化统计回答"
        if content_question:
            decision.needs_knowledge = True
            decision.intent = IntentKind.RESEARCH
            decision.reason_summary = "询问知识库中的内容，检索并读取相关正文"
            notes.append(
                "问题包含资料内容要求，进入正文研究"
            )
        else:
            return RuleOutcome(decision=decision, hits=hits, notes=notes)

    # ---- R8：回忆以前的讨论
    memory_marker = _contains(question, _MEMORY_MARKERS)
    if memory_marker:
        hit("R8", f"回忆标记：{memory_marker}")
        decision.intent = IntentKind.MEMORY
        decision.needs_history = True

    # ---- R1：明确引用内部资料 / 项目；泛指文档内容优先归类 ----
    if internal_marker:
        hit("R1", f"内部引用标记：{internal_marker}")
        decision.needs_knowledge = True
    elif generic_document_intent:
        hit("R1", f"泛指文档的内容问法：{generic_document_intent}")
        decision.needs_knowledge = True
        # 只有「泛指」、没有明确文章名时，不再叠加通用解释（避免用通用解释顶替文档内容）。
        generic_document_only = not (
            internal_marker or document_matches or explicit_document_intent
        )
    elif document_intent:
        # 文档寻址问法（「的运维手册在哪里？」）与文档内容问法（「那篇文章为什么强调」
        # 「请给出文中的依据」）：先检索再回答，不能没查就说没有、也不能要求用户
        # 再提供知识库里已有的原文（用户实测缺陷 F14/F15）。
        hit("R1", f"指定资料的内容问法：{document_intent}")
        decision.needs_knowledge = True
    elif document_matches:
        hit("R1", f"目录文件线索：{document_matches[0].matched_on}")
        decision.needs_knowledge = True

    # ---- R2：围绕内部资料继续追问（指代 + 历史）
    if reference_note and (
        decision.needs_knowledge or reference_note.startswith("指代补全")
    ):
        hit("R2", reference_note)
        decision.needs_knowledge = True

    # ---- R3：内部术语目录命中 + 问其定义或实现
    definition_marker = _contains(question, _DEFINITION_INTENT_MARKERS)
    if term_matches and definition_marker:
        hit("R3", f"术语命中 {term_matches[0].matched_on} + 定义意图 {definition_marker}")
        decision.needs_knowledge = True
        decision.scope_evidence_ids = _merge_unique(
            decision.scope_evidence_ids,
            [match.document_id for match in term_matches if match.document_id],
        )

    # ---- R4：术语含义不明确，可能指内部对象
    if unknown_terms:
        names = [match.matched_on for match in unknown_terms]
        hit("R4", f"含义不明术语：{'、'.join(names)}")
        decision.needs_knowledge = True
        decision.unresolved_terms = _merge_unique(decision.unresolved_terms, names)
        notes.append("目录命中但缺少明确解释：先做范围检查，只用于定位原文")
        if len(question) <= CLARIFICATION_MAX_QUESTION_CHARS and not decision.needs_history:
            decision.intent = IntentKind.CLARIFICATION
            notes.append("问题很短且术语不明，先请用户确认对象")

    # ---- R5：明确要求今天 / 最新 / 当前版本，或明确要求联网
    # 联网表达用**动作短语**识别（见 :func:`explicit_web_request`）：旧实现是一张固定
    # 动词清单（「联网搜索 / 联网查询 / 上网搜索 / 搜索网页 / 上网查」），于是
    # 「联网查找」静默不动、行为与「联网搜索」不一致（用户实测缺陷 F05/T20）。
    freshness = _detect_freshness(question)
    explicit_web = explicit_web_request(question)
    official_web = bool(source_requirement(question)) and not memory_marker
    general_only = bool(re.search(r"(?:仅|只|用|使用|根据)通用知识|无需检索|不(?:要|用)检索", question))
    product_web = bool(
        official_domains(question)
        and not (internal_marker or memory_marker or document_intent or generic_document_intent or document_matches)
        and not general_only and not local_only_question(question)
        and any(marker in question.casefold() for marker in (*_CONCEPT_INTENT_MARKERS, "什么是", "默认", "参数", "配置"))
    )
    if freshness is not Freshness.NONE or explicit_web or official_web or product_web:
        marker = (
            explicit_web
            or ("指定机构官方来源" if official_web else "")
            or ("产品机制优先查询官方文档" if product_web else "")
            or _contains(question, _FRESHNESS_RANGE_MARKERS)
            or _contains(question, _FRESHNESS_CURRENT_MARKERS)
            or "时间范围表达"
        )
        hit("R5", f"联网要求：{marker}")
        decision.needs_web = True
        decision.freshness = freshness
        if official_web and not (internal_marker or memory_marker):
            decision.needs_knowledge = False
            notes.append("用户指定官方来源，本轮以其官方网页为准")
        elif product_web and input.knowledge_available:
            # 产品/厂商名命中只代表「优先官方来源」，不能反过来禁掉本地检索：
            # 联网被策略关闭（关闭思考）时，否则既不查库也不查网，只剩模型常识。
            decision.needs_knowledge = True
            notes.append("产品机制题同时查本工作区资料，官方网页作补充")
        elif not decision.needs_knowledge and bool(explicit_web):
            decision.needs_knowledge = True
            notes.append("用户明确要求联网核实，同时也查本工作区资料作为对照")
        if not decision.needs_knowledge and bool(explicit_web or official_web):
            decision.reason_summary = "根据请求联网查询公开资料"

    # ---- R6：通用解释同时涉及已有内部方案 / 术语需要先理解再检索
    general_marker = _contains(question, _GENERAL_MARKERS)
    terms = concept_terms(question)
    if general_marker and (decision.needs_knowledge or bool(input.catalog_clues)):
        hit("R6", f"通用解释标记：{general_marker}，同时有内部线索")
        decision.needs_general = True
    elif terms and not generic_document_only and input.knowledge_available:
        # 术语 + 概念/场景意图：允许先用一般知识理解术语（General），同时把术语作为
        # **检索线索**去查本区资料（RAG）。这样「AI Native 的使用场景」第一次提问就能
        # 既解释概念又去找资料，而不是反复说「文档没有定义」（用户实测缺陷 C01）。
        hit("R6", f"术语 + 概念意图：{'、'.join(terms[:3])}")
        decision.needs_knowledge = True
        decision.needs_general = True
        decision.entities = _merge_unique(decision.entities, terms)
        notes.append(
            "术语先用一般知识理解并生成检索表达；术语扩展只是检索线索，"
            "文档观点与案例仍需原文证据"
        )

    if official_web and not (internal_marker or memory_marker):
        decision.needs_knowledge = False

    # ---- R9：知识库为空
    if not input.knowledge_available:
        hit("R9", "本区已登记文档数为 0")
        if decision.needs_knowledge:
            notes.append("当前工作区没有已登记资料：内部资料问题需说明资料情况")
        elif not decision.needs_web:
            notes.append("当前工作区没有已登记资料：通用或联网问题仍可处理")

    # ---- R11：默认检索兜底（不确定与本区资料是否相关 → 先查一次）
    # 实测缺陷 D2（2026-10-20 评测）：路由把「要不要检索」实现成「有没有明说文档/术语」，
    # 口语改写与概念问法一律不出检索，答案退回模型自带知识（sem_cache/sem_coase/
    # sem_attr2 三题 0 工具）。目录只按标题与术语做字面匹配，「没命中」不能当作
    # 「本区没有资料」的证据；除明确闲聊外，默认先查一次，查不到再退回通用作答。
    if (
        not decision.needs_knowledge
        and not decision.needs_web
        and not decision.needs_history
        and not discussion_request(question, has_history=bool(input.recent_history))
        and not _NON_KNOWLEDGE_CHAT.match(question.strip())
        and input.knowledge_available
        and len(question.strip()) >= _DEFAULT_RETRIEVAL_MIN_CHARS
    ):
        hit("R11", "没有命中内部引用规则，按默认检索兜底")
        decision.needs_knowledge = True
        decision.needs_general = True
        decision.needs_source_search = True
        decision.retrieval_queries = _default_retrieval_queries(question)
        notes.append("目录未命中只说明标题/术语没有字面命中，不构成跳过检索的理由")

    # ---- 兜底意图：没有内部/联网要求时按通用解释
    if (
        not decision.needs_knowledge
        and not decision.needs_web
        and decision.intent is IntentKind.RESEARCH
    ):
        decision.intent = IntentKind.GENERAL
        decision.needs_general = True

    if input.catalog_clues:
        clues = [clue for clue in input.catalog_clues[:3] if clue]
        if clues:
            decision.scope_evidence_ids = _merge_unique(decision.scope_evidence_ids, clues)
            notes.append(f"工作区主题线索：{'、'.join(clues)}")

    if not decision.reason_summary:
        decision.reason_summary = _reason_summary(decision)
    return RuleOutcome(decision=_apply_source_exclusions(decision, question), hits=hits, notes=notes)


def _reason_summary(decision: RouteDecision) -> str:
    parts: list[str] = []
    if decision.needs_knowledge:
        parts.append("查本工作区资料")
    if decision.needs_web:
        parts.append("联网核实最新信息")
    if decision.needs_history:
        parts.append("查历史讨论")
    if decision.needs_general and not parts:
        parts.append("用通用知识解释")
    if not parts:
        parts.append("按通用路径处理")
    return "；".join(parts)


def _merge_unique(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    seen = {str(item) for item in existing}
    result = [str(item) for item in existing]
    for item in extra:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _unique(values: Sequence[str]) -> list[str]:
    return _merge_unique([], values)


# =====================================================================
# 3. 范围检查与调整（PRD 6.1 / 6.3 末段）
# =====================================================================


def check_scope(
    decision: RouteDecision,
    *,
    catalog: KnowledgeCatalog | None = None,
    input: RoutingInput | None = None,
    clue_terms: Sequence[str] | None = None,
    text_lookup: Callable[[str], str] | None = None,
) -> ScopeCheckResult:
    """对路由结论做范围检查（PRD 6.1）：

    - 目录命中只用于定位原文，结果写进 ``scope_evidence_ids``；
    - 缺明确解释的术语进 ``unresolved_terms``（不当作已定义概念）；
    - 引用片段必须真的存在于原文：给 ``text_lookup`` 时逐条校验，校验不通过的术语
      进 ``unresolved_terms`` 并记 note。
    """

    catalog_empty = True
    scope_ids: list[str] = []
    resolved_terms: list[str] = []
    unknown_terms: list[str] = []
    notes: list[str] = []

    candidate_terms = [str(term) for term in (clue_terms or ())]
    if not candidate_terms and input is not None:
        candidate_terms = [match.matched_on for match in _catalog_term_matches(input)]

    if catalog is not None:
        catalog_empty = catalog.is_empty()
        question = decision.resolved_question or decision.question
        result = catalog.search(question, limit=8)
        for match in result.matches:
            if match.document_id and match.score >= CATALOG_CLUE_MIN_SCORE:
                scope_ids.append(match.document_id)
            if match.kind != "term" or not _mentioned_in_question(question, match):
                continue
            if match.definition_unknown:
                unknown_terms.append(match.matched_on)
                continue
            entry = _term_entry(catalog, match.matched_on, match.document_id)
            if text_lookup is not None and entry is not None:
                body = text_lookup(entry.document_id) or ""
                if entry.snippet and entry.snippet not in body:
                    unknown_terms.append(match.matched_on)
                    notes.append(f"术语 {match.matched_on} 的引用片段未在原文复现，转为待确认")
                    continue
            resolved_terms.append(match.matched_on)
    elif input is not None:
        catalog_empty = not input.knowledge_available
        scope_ids.extend(
            match.document_id
            for match in input.catalog_hits
            if match.score >= CATALOG_CLUE_MIN_SCORE
        )
        for match in _mentioned_term_matches(decision.resolved_question or decision.question, input):
            if match.definition_unknown:
                unknown_terms.append(match.matched_on)
            else:
                resolved_terms.append(match.matched_on)

    unresolved = _merge_unique(
        [*decision.unresolved_terms, *unknown_terms],
        [term for term in candidate_terms if term in unknown_terms],
    )
    updated = decision
    if scope_ids:
        updated = adjust_route_decision(
            decision,
            scope_evidence_ids=_merge_unique(decision.scope_evidence_ids, _unique(scope_ids)),
        )
        notes.append("目录命中仅用于定位原文，不作为答案依据")
    if catalog_empty:
        notes.append("本区知识目录为空")

    return ScopeCheckResult(
        decision=updated,
        knowledge_required=updated.needs_knowledge,
        web_required=updated.needs_web,
        catalog_empty=catalog_empty,
        scope_evidence_ids=list(updated.scope_evidence_ids),
        resolved_terms=_unique(resolved_terms),
        unresolved_terms=unresolved,
        definition_unknown_terms=_unique(unknown_terms),
        rules=[],
        notes=notes,
    )


def _term_entry(catalog: KnowledgeCatalog, term: str, document_id: str) -> Any:
    for entry in catalog.terms:
        if entry.term == term and (not document_id or entry.document_id == document_id):
            return entry
    return None


def adjust_route_decision(
    decision: RouteDecision,
    *,
    resolved_question: str | None = None,
    needs_knowledge: bool | None = None,
    needs_web: bool | None = None,
    needs_history: bool | None = None,
    needs_general: bool | None = None,
    freshness: Freshness | str | None = None,
    entities: Sequence[str] | None = None,
    scope_evidence_ids: Sequence[str] | None = None,
    unresolved_terms: Sequence[str] | None = None,
    reason_summary: str | None = None,
    intent: IntentKind | str | None = None,
    needs_source_search: bool | None = None,
    retrieval_queries: Sequence[str] | None = None,
) -> RouteDecision:
    """研究过程中调整来源需求（PRD 6.3 末段，Planner 使用）。

    只允许显式传入的字段变化；返回**新对象**（原结论保持可追溯）。
    """

    payload = decision.model_dump(mode="json")
    updates: dict[str, Any] = {
        "resolved_question": resolved_question,
        "needs_knowledge": needs_knowledge,
        "needs_web": needs_web,
        "needs_history": needs_history,
        "needs_general": needs_general,
        "freshness": freshness,
        "entities": list(entities) if entities is not None else None,
        "scope_evidence_ids": list(scope_evidence_ids) if scope_evidence_ids is not None else None,
        "unresolved_terms": list(unresolved_terms) if unresolved_terms is not None else None,
        "reason_summary": reason_summary,
        "intent": intent,
        "needs_source_search": needs_source_search,
        "retrieval_queries": list(retrieval_queries) if retrieval_queries is not None else None,
    }
    for key, value in updates.items():
        if value is None:
            continue
        payload[key] = value
    if not str(payload.get("resolved_question") or "").strip():
        payload["resolved_question"] = payload.get("question", "")
    return RouteDecision.model_validate(payload)


# =====================================================================
# 4. 提示组装（PRD 6.1 + 9.3 固定顺序）
# =====================================================================

_DUTY_PROMPT_HEADER: Final[str] = (
    "你是 Everything RAG 的意图识别 Agent（Intent Agent）。"
    "你的唯一职责是判断这个问题需要哪些来源，并输出结构化结论；不要回答问题本身。"
)

_DUTY_PROMPT_RULES: Final[str] = """判断规则（与产品决策规则一致，按优先级）：
R1 明确引用笔记、内部文档、既有方案或当前项目 → 需要本工作区资料
R2 对话已围绕某内部资料展开、用户用“这个方案/第二种方式”等继续追问 → 用历史补全对象后查资料
R3 内部术语在目录里命中，且问题问它的定义或实现 → 查对应原文
R4 术语含义不明确、可能指内部对象 → 标记待确认术语，先做范围检查
R5 明确要求今天、最新、当前版本 → 需要联网并检查时间
R6 通用解释同时涉及已有内部方案 → 通用解释与工作区资料都要
R7 仅询问产品功能或文档数量 → 用产品说明或结构化统计，不要检索正文
R8 用户回忆以前的讨论 → 查工作区记忆与历史记录
R9 知识库为空 → 通用和联网问题仍可处理；内部资料问题要说明当前资料情况
R10 用户邀请补充观点、表达看法或评价已有做法 → 结合当前对话和通用知识回答；沿用讨论对象无需重新检索。
R11 无法确定是否与本区资料相关时 → 默认检索一次（needs_knowledge=true，retrieval_queries 给 1–2 条），
    同时把 needs_general 设为 true；检索不到再退回通用知识作答。

要求：
- needs_knowledge 与 needs_web 可以同时为 true。
- 当前对话已经提供的背景可直接用于讨论。needs_history只表示需要搜索其他历史记录。
- 区分询问资料中的事实与邀请你分析评价；“你有什么补充的说法”“你怎么看他们的做法”属于观点讨论。
- 用户同时明确要求新增来源、核实事实或查询最新情况时，保留相应查询任务。
- retrieval_queries 是检索表达：每个用户明确询问的研究对象至少保留一条，覆盖该对象所问的全部维度；
  每条包含专有名词、产品名、文件名与关键实体，同一对象的同义表达共用研究过程；
  需要历史或通用来源、不需要检索时给空数组。
- intent 只能取 general / research / product / catalog / memory / clarification。
- freshness 只能取 none / current / date_range。
- 目录片段只是定位线索，不能当作事实依据；不要引用其中的原文当答案。
- **目录未命中不等于本区没有相关资料**：目录只按标题与术语做字面匹配，口语改写、同义表达、
  概念问法（「为什么…」「怎么…」）本来就不会命中。判定 needs_knowledge=false 时，
  reason_summary 必须给出「问题与本地资料无关」的具体理由，不能以「目录未命中」为由跳过检索。
- 只有明确的问候、致谢、翻译、代写、闲聊才可以 needs_knowledge=false 且 retrieval_queries 为空。
- reuse_evidence_ids逐份选择上一轮已完整阅读、且包含本轮所问事实的原文ID；另一份来源尚需补充时仍保留这些ID。
  read_evidence_ids选择上一轮尚未完整阅读、需要按其章节位置补全的知识片段ID；已经有完整原文的相同章节无需再次选择。
  needs_source_search表示是否还需搜索尚未定位的来源。既有完整原文与待补读章节已经定位本轮所需全部来源时为false，否则为true。
  用户要求当前状态或重新联网读取时，needs_source_search为true。核对刚才原文的含义、版本、来源可以复用；来源类型仍在needs_knowledge/needs_web中声明。
- 只输出 JSON 对象，字段：question / resolved_question / intent / needs_knowledge /
  needs_web / needs_history / needs_general / freshness / entities /
  scope_evidence_ids / reuse_evidence_ids / read_evidence_ids / needs_source_search / unresolved_terms / reason_summary / retrieval_queries。不要输出额外字段、不要输出置信度。"""


class RoutePrompt(_RoutingModel):
    """一次路由请求的完整提示（已按 PRD 9.3 固定顺序组装）。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    embedded_chars: int = Field(default=0, ge=0)
    catalog_terms: int = Field(default=0, ge=0)
    injected_documents: int = Field(default=0, ge=0)


def build_route_prompt(
    input: RoutingInput,
    *,
    catalog: KnowledgeCatalog | None = None,
    projection: CatalogPromptProjection | None = None,
    catalog_hit_limit: int = DEFAULT_CATALOG_HIT_LIMIT,
) -> RoutePrompt:
    """组装 Intent Agent 的模型输入（PRD 6.1 路由输入 + 9.3 固定顺序）。

    只注入与本轮相关的目录条目（``projection``），完整目录留在本地。
    """

    from app.agent.context import ContextSnapshot, build_context

    query = input.question
    if projection is None and catalog is not None:
        projection = catalog.prompt_projection(query, limit=catalog_hit_limit)

    dynamic_lines: list[str] = []
    if input.now:
        dynamic_lines.append(f"当前时间：{input.now}")
    if input.timezone:
        dynamic_lines.append(f"用户时区：{input.timezone}")
    dynamic_lines.append(f"当前工作区：{input.workspace_id or '（默认）'}")
    dynamic_lines.append(
        f"本区资料规模：{input.file_count} 个文档 / {input.chunk_count} 个片段"
        f"（知识版本 {input.knowledge_revision}）"
    )
    if input.workspace_overview:
        dynamic_lines.append(f"工作区主题概览：{input.workspace_overview}")
    if input.catalog_clues:
        dynamic_lines.append("工作区主题线索：" + "、".join(input.catalog_clues[:8]))
    if projection is not None and projection.text:
        dynamic_lines.append(
            "本轮相关目录条目（字面匹配，只用于定位原文，不是相关性判定）：" + projection.text
        )
    else:
        dynamic_lines.append(
            "本轮相关目录条目：无。目录按标题与术语做字面匹配，未命中只说明问法没对上字面，"
            "不代表本区没有相关资料，也不构成跳过检索的理由。"
        )

    history_messages = [
        {"role": turn.role, "content": turn.content}
        for turn in input.recent_history[-MAX_RECENT_HISTORY:]
    ]
    messages = build_context(
        duty_prompt=f"{_DUTY_PROMPT_HEADER}\n\n{_DUTY_PROMPT_RULES}",
        snapshot=ContextSnapshot(input.memory_snapshot or ""),
        history=history_messages,
        latest=[{"role": "user", "content": query}],
        dynamic=[{"role": "user", "content": "\n".join(dynamic_lines)}],
    )
    embedded = sum(len(str(message.get("content") or "")) for message in messages)
    return RoutePrompt(
        messages=messages,
        document_ids=list(projection.document_ids) if projection is not None else [],
        embedded_chars=embedded,
        catalog_terms=projection.term_count if projection is not None else 0,
        injected_documents=len(projection.document_ids) if projection is not None else 0,
    )


# =====================================================================
# 5. 模型输出解析（严格、不过度信任）
# =====================================================================


_ROUTE_QUERY_LIMIT: Final[int] = 3


def _normalize_queries(raw: Any) -> list[str]:
    """把模型给的检索查询归一成「最多 3 条、去空、去重、去掉列表符号」的字符串列表。"""

    if isinstance(raw, str):
        items: Sequence[Any] = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return []
    queries: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = re.sub(r"^\s*(?:[-*·]|\d+[.)、])\s*", "", item).strip()
        if not text or text in queries:
            continue
        queries.append(text)
        if len(queries) >= _ROUTE_QUERY_LIMIT:
            break
    return queries


def parse_route_decision(
    raw: str | Mapping[str, Any], *, question: str = ""
) -> tuple[RouteDecision, RouteDiagnostics]:
    """解析并校验模型输出的 ``RouteDecision``。

    未知字段只进诊断（模型的额外字段不改变契约）；``intent`` / ``freshness`` 取值
    非法时抛 :class:`RouteParseError`，由调用方回退到规则层结论。
    """

    payload: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RouteParseError(f"路由结论不是合法 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RouteParseError("路由结论必须是 JSON 对象")

    dropped = [
        str(key)
        for key in payload
        if key not in _ROUTE_DECISION_FIELDS and key != "confidence"
    ]
    confidence_raw = payload.get("confidence")
    confidence: float | None = None
    if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool):
        confidence = float(confidence_raw)

    data = {key: value for key, value in payload.items() if key in _ROUTE_DECISION_FIELDS}
    if not str(data.get("question") or "").strip():
        data["question"] = question
    if not str(data.get("resolved_question") or "").strip():
        data["resolved_question"] = data["question"]
    if "intent" not in data:
        raise RouteParseError("路由结论缺少 intent")
    # retrieval_queries 是可选附加字段：模型偶尔会写成字符串或塞进非字符串项，
    # 这里先做一次宽松归一，避免因为一个附加字段把整条路由结论判为协议错误。
    data["retrieval_queries"] = _normalize_queries(data.get("retrieval_queries"))
    try:
        decision = RouteDecision.model_validate(data)
    except Exception as exc:  # pydantic 校验失败统一转成路由协议错误
        raise RouteParseError(f"路由结论校验失败: {exc}") from exc
    return decision, RouteDiagnostics(
        model_used=True,
        confidence=confidence,
        dropped_fields=dropped,
    )


# =====================================================================
# 6. Intent Agent
# =====================================================================


class _RouteModel(Protocol):
    """只依赖非流式 ``complete``（T03 的 ChatModel 已提供）。"""

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class IntentAgent:
    """意图识别 Agent：规则层 + 可选的模型补充（PRD 4.2 / 6.1–6.3）。

    规则层始终先跑，并作为**硬约束**覆盖模型输出：产品/概览问题不查正文、时效问题
    必须联网、内部引用必须查资料、空知识库要说明资料情况。模型的 ``confidence``
    只写进诊断字段，不作为验收依据。
    """

    def __init__(
        self,
        *,
        model: _RouteModel | None = None,
        catalog: KnowledgeCatalog | None = None,
        text_lookup: Callable[[str], str] | None = None,
        rules_version: str = ROUTING_RULES_VERSION,
        catalog_hit_limit: int = DEFAULT_CATALOG_HIT_LIMIT,
    ) -> None:
        self.model = model
        self.catalog = catalog
        self.text_lookup = text_lookup
        self.rules_version = rules_version
        self.catalog_hit_limit = catalog_hit_limit

    # ------------------------------------------------------------ 规则层

    def enrich(self, input: RoutingInput) -> RoutingInput:
        """把目录线索补进路由输入（调用方没给 hits 时由本 Agent 查询目录）。"""

        if input.catalog_hits or self.catalog is None or self.catalog.is_empty():
            return input
        result = self.catalog.search(input.question, limit=self.catalog_hit_limit)
        if not result.matches:
            return input
        return input.model_copy(update={"catalog_hits": list(result.matches)})

    def decide(self, input: RoutingInput) -> RouteResult:
        """纯规则路由（离线可用，不调用模型）。"""

        enriched = self.enrich(input)
        outcome = evaluate_route_rules(enriched)
        scope = check_scope(
            outcome.decision,
            catalog=self.catalog,
            input=enriched,
            text_lookup=self.text_lookup,
        )
        scope.rules = outcome.rule_ids
        return RouteResult(
            decision=scope.decision,
            scope=scope,
            rules=outcome.hits,
            diagnostics=RouteDiagnostics(
                model_used=False,
                confidence=None,
                notes=list(outcome.notes),
            ),
        )

    def build_prompt(self, input: RoutingInput) -> RoutePrompt:
        enriched = self.enrich(input)
        return build_route_prompt(
            enriched,
            catalog=self.catalog,
            catalog_hit_limit=self.catalog_hit_limit,
        )

    # ------------------------------------------------------------ 模型层

    async def decide_with_model(
        self, input: RoutingInput, *, model: _RouteModel | None = None
    ) -> RouteResult:
        """模型补充路由：模型输出只做补充，硬规则仍然由规则层掌权。"""

        active_model = model or self.model
        base = self.decide(input)
        if active_model is None:
            base.diagnostics.notes.append("未注入模型，使用规则层结论")
            return base

        prompt = self.build_prompt(input)
        try:
            response = await active_model.complete(prompt.messages)
            model_decision, diagnostics = parse_route_decision(
                _response_content(response), question=input.question
            )
        except RouteParseError as exc:
            base.diagnostics.notes.append(f"模型结论不可用，回退规则层：{exc}")
            return base

        merged = self._merge(base, model_decision)
        scope = check_scope(
            merged,
            catalog=self.catalog,
            input=self.enrich(input),
            text_lookup=self.text_lookup,
        )
        scope.rules = base.scope.rules
        diagnostics.notes = list(base.diagnostics.notes)
        return RouteResult(
            decision=scope.decision,
            scope=scope,
            rules=base.rules,
            diagnostics=diagnostics,
        )

    def _merge(self, base: RouteResult, model_decision: RouteDecision) -> RouteDecision:
        """合并规则层与模型结论：硬规则优先，其余以模型补充。"""

        rule_decision = base.decision
        rule_ids = set(base.scope.rules)
        merged = model_decision

        if "R10" in rule_ids:
            return rule_decision.model_copy(deep=True)

        # 产品 / 概览问题（R7）不接受模型的检索要求。
        if "R7" in rule_ids:
            merged = adjust_route_decision(
                merged,
                intent=rule_decision.intent,
                needs_knowledge=False,
                needs_web=False,
                needs_general=False,
                needs_history=False,
                freshness=Freshness.NONE,
            )
        # 时效要求（R5）必须联网。
        if "R5" in rule_ids:
            merged = adjust_route_decision(
                merged, needs_web=True, freshness=rule_decision.freshness
            )
        # 内部引用（R1/R2/R3）必须查资料。
        if rule_ids & {"R1", "R2", "R3"}:
            merged = adjust_route_decision(merged, needs_knowledge=True)
        # 默认检索兜底（R11）同为硬规则：模型把 needs_knowledge 判成 false 不算通过。
        # R7（产品/概览）优先，不抢它的否定结论。
        if "R11" in rule_ids and "R7" not in rule_ids:
            merged = adjust_route_decision(
                merged,
                needs_knowledge=True,
                needs_source_search=True,
                needs_general=True,
            )
            if not merged.retrieval_queries:
                merged = adjust_route_decision(
                    merged, retrieval_queries=rule_decision.retrieval_queries
                )
        # 回忆讨论（R8）必须查历史。
        if "R8" in rule_ids:
            merged = adjust_route_decision(merged, needs_history=True)
            if not rule_decision.needs_knowledge:
                merged = adjust_route_decision(merged, needs_knowledge=False)
            if not rule_decision.needs_web:
                merged = adjust_route_decision(merged, needs_web=False)
        # 空知识库（R9）保持规则层的资料情况说明。
        if "R9" in rule_ids and rule_decision.needs_knowledge:
            merged = adjust_route_decision(merged, needs_knowledge=True)
        # 规则层识别出的待确认术语必须保留（模型漏报不算通过）。
        if rule_decision.unresolved_terms:
            merged = adjust_route_decision(
                merged,
                unresolved_terms=_merge_unique(
                    merged.unresolved_terms, rule_decision.unresolved_terms
                ),
            )
        if not merged.reason_summary:
            merged = adjust_route_decision(merged, reason_summary=rule_decision.reason_summary)
        return _apply_source_exclusions(merged, rule_decision.question)


def _response_content(response: Any) -> Any:
    """从模型响应里取正文（兼容 str / dict / 带 content 属性的对象）。"""

    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        return response.get("content") or ""
    return getattr(response, "content", "") or ""


def route_question(
    input: RoutingInput,
    *,
    agent: IntentAgent | None = None,
    catalog: KnowledgeCatalog | None = None,
    text_lookup: Callable[[str], str] | None = None,
) -> RouteDecision:
    """离线规则层路由入口（图节点与离线评测使用；需要模型时用 ``IntentAgent``）。"""

    active = agent or IntentAgent(catalog=catalog, text_lookup=text_lookup)
    return active.decide(input).decision


# =====================================================================
# 7. route.decided 事件（契约由 T00b 冻结，本模块只按模型构造）
# =====================================================================


def route_decided_payload(decision: RouteDecision) -> dict[str, Any]:
    """``route.decided`` 的 payload：形状固定为 ``{"decision": <RouteDecision>}``。"""

    return {"decision": decision.model_dump(mode="json")}


def build_route_decided_event(
    *,
    run_id: str,
    event_seq: int,
    decision: RouteDecision,
    execution_id: str = "",
    task_id: str | None = None,
    created_at: str = "",
) -> StoredEvent:
    """按冻结契约构造 ``route.decided`` 事件（多一个字段就会抛协议错误）。"""

    return build_event(
        run_id=run_id,
        event_seq=event_seq,
        type=EventType.ROUTE_DECIDED,
        payload=route_decided_payload(decision),
        execution_id=execution_id,
        task_id=task_id,
        created_at=created_at,
    )
