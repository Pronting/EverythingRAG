"""Deterministic query variants for conversational knowledge-base questions.

The transformer is deliberately conservative: it removes only well-known conversational
scaffolding and never asks a model to invent missing context.  The original query is always
kept as the first variant, so product names, identifiers and other entities cannot be lost by
normalisation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.retrieval.bm25 import tokenize
from app.retrieval.percentiles import percentile_usage_subject

QueryVariantKind = Literal[
    "original",
    "normalized",
    "definition_subject",
]


@dataclass(frozen=True)
class QueryVariant:
    """One retrieval query variant without any generated or inferred facts."""

    kind: QueryVariantKind
    text: str


_LITERAL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # Example markers are retrieval scaffolding, not entities.  A separator keeps the examples
    # apart without adding an unrelated ``例如`` bigram to the lexical query.
    ("比如说", ""),
    ("比方说", ""),
    ("我想问一下", ""),
    ("我想问问", ""),
    ("我想了解一下", ""),
    ("我想了解", ""),
    ("想问一下", ""),
    ("想了解一下", ""),
    ("能不能帮我", ""),
    ("可以帮我", ""),
    ("麻烦帮我", ""),
    ("请帮我", ""),
    ("请问一下", ""),
    ("请问", ""),
)

_LEADING_UNCERTAINTY_RE = re.compile(r"^(?:(?:我)?(?:好像|似乎|印象中|记得|感觉)\s*[，,、]?\s*)+")
_LEADING_QUANTIFIER_RE = re.compile(r"^(?:(?:有用?|存在)?一些)\s*")
_LEADING_TOPIC_RE = re.compile(r"^(?:对于|关于)\s*")
_QUESTION_METHOD_RE = re.compile(r"(?:团队|我们|他们)?(?:是)?(?:怎么|如何)(?=[一-鿿A-Za-z0-9])")
_DETAIL_SUFFIX_RE = re.compile(
    r"(?P<detail>的?(?:具体)?详情)\s*(?:是|为)?\s*(?:什么样|怎么样)(?:的)?\s*[?？。！!]*$"
)
_GENERIC_SUFFIX_RE = re.compile(r"\s*(?:是|为)?\s*(?:什么样|怎么样)(?:的)?\s*[?？。！!]*$")
_DEMONSTRATIVE_DETAIL_RE = re.compile(
    r"(?:这些|上述|以上)(?:对比|方案|内容|情况)?的?(?:具体)?详情"
)
_STRUCTURAL_PARTICLE_RE = re.compile(
    r"的(?=(?:分表|对比|详情|方案|规则|机制|流程|配置|实现|区别|优缺点|内容))"
)
_EXAMPLE_COMPARISON_TAIL_RE = re.compile(r"等对比")
_SEGMENT_CONNECTIVE_RE = re.compile(r"(^|[，,；;])和(?=[一-鿿])")
_TRAILING_DETAIL_SEPARATOR_RE = re.compile(r"[，,；;]+(?=详情\s*$)")
_TRAILING_PARTICLE_RE = re.compile(r"的\s*(?=[?？。！!]*$)")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s*([，,、；;：:？?。！!])\s*")
_REPEATED_PUNCT_RE = re.compile(r"([，,、；;：:？?。！!])\1+")
# Replace whole grammatical phrases with separators, never individual Chinese characters:
# deleting particles character-by-character would corrupt entities such as 京东/存在性检测.
_CONVERSATIONAL_SCAFFOLD_RE = re.compile(
    r"(?:应该|应当|其实)是|(?:它|他|她|它们|他们)的|"
    r"(?:是)?(?:什么样|怎么样|怎样)的?|(?:是)?如何在|"
    r"(?:对于|以及)|(?:中存在的)|(?:意义何在)"
)
_DEFINITION_QUERY_PATTERNS = (
    re.compile(
        r"^(?:请)?(?:问)?(?:一下)?(?:什么是|何为)\s*"
        r"(?P<subject>[^，,；;。！!？?]{1,80})\s*[？?]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<subject>[^，,；;。！!？?]{1,80}?)\s*"
        r"(?:是什么|什么意思|的定义(?:是什么)?|的全称(?:是什么)?)\s*[？?]?$",
        re.IGNORECASE,
    ),
)
_POLITE_DEFINITION_PREFIX_RE = re.compile(
    r"^(?:(?:请)?问(?:一下)?|我想(?:问|了解)(?:一下)?)\s*[,，:：]?\s*",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Remove conversational filler while preserving the user's factual tokens.

    No synonym expansion or pronoun resolution happens here: those operations can fabricate
    intent.  Unicode compatibility normalisation makes full-width ASCII identifiers searchable,
    and the original query remains available through :func:`build_query_variants`.
    """

    original = query.strip()
    if not original:
        return ""

    normalized = unicodedata.normalize("NFKC", original)
    normalized = _LEADING_UNCERTAINTY_RE.sub("", normalized)
    normalized = _LEADING_QUANTIFIER_RE.sub("", normalized)
    normalized = _LEADING_TOPIC_RE.sub("", normalized)
    for source, replacement in _LITERAL_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    subject = percentile_usage_subject(normalized)
    if subject is not None:
        return subject

    # Turn "这些对比的详情是什么样的" into "这些对比的详情" instead of deleting
    # the information-bearing word "详情".
    normalized = _DETAIL_SUFFIX_RE.sub(lambda match: match.group("detail"), normalized)
    normalized = _GENERIC_SUFFIX_RE.sub("", normalized)
    # Do this before the method rule: otherwise 怎么样 becomes the meaningless fragment 样.
    normalized = _CONVERSATIONAL_SCAFFOLD_RE.sub(" ", normalized)
    normalized = _QUESTION_METHOD_RE.sub("", normalized)
    normalized = _DEMONSTRATIVE_DETAIL_RE.sub("详情", normalized)
    normalized = _STRUCTURAL_PARTICLE_RE.sub("", normalized)
    normalized = _EXAMPLE_COMPARISON_TAIL_RE.sub("", normalized)
    normalized = _SEGMENT_CONNECTIVE_RE.sub(r"\1", normalized)
    normalized = _TRAILING_DETAIL_SEPARATOR_RE.sub("", normalized)
    normalized = _TRAILING_PARTICLE_RE.sub("", normalized)
    normalized = _SPACE_RE.sub(" ", normalized)
    normalized = _PUNCT_SPACE_RE.sub(r"\1", normalized)
    normalized = _REPEATED_PUNCT_RE.sub(r"\1", normalized)
    normalized = normalized.strip(" \t\r\n，,、；;：:？?。！!")
    return normalized or unicodedata.normalize("NFKC", original)


def build_query_variants(query: str) -> tuple[QueryVariant, ...]:
    """Return the exact original query plus at most one deterministic normalised variant."""

    original = query.strip()
    if not original:
        return ()
    variants = [QueryVariant(kind="original", text=original)]
    normalized = normalize_query(original)
    # NFKC and punctuation cleanup can change the string without changing a single lexical
    # token. Treating that as a second route double-counts identical BM25 evidence and, more
    # importantly, lets a punctuation-only copy take a different rescue path.
    if (
        normalized
        and normalized != original
        and set(tokenize(normalized)) != set(tokenize(original))
    ):
        variants.append(QueryVariant(kind="normalized", text=normalized))
    return tuple(variants)


def extract_identifier_definition_subjects(query: str) -> tuple[str, ...]:
    """Extract a concrete ASCII identifier from a narrow definition-only question."""

    compact = _POLITE_DEFINITION_PREFIX_RE.sub("", query.strip())
    for pattern in _DEFINITION_QUERY_PATTERNS:
        match = pattern.fullmatch(compact)
        if not match:
            continue
        subject = match.group("subject").strip(" \t\r\n，,、；;：:？?。！!")
        # The hard evidence gate is only for a *bare* ASCII identifier. A compound question such
        # as “APISIX 事故的根因是什么” or an enumerated comparison ending in “分别是什么” is
        # content QA, not a definition lookup merely because one ASCII product name appears.
        if subject and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Za-z][A-Za-z0-9_.-]*){0,3}",
            subject,
        ):
            return (subject,)
    return ()
