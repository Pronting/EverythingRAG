"""Exact percentile aliases; P99, 99%, and 99.9 remain distinct metrics."""
import re
import unicodedata
from decimal import Decimal

PERCENTILE_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?:p(?P<ascii>\d{1,2}(?:\.\d+)?)(?![\d.])|"
    r"(?P<number>\d{1,2}(?:\.\d+)?|九十九|九九)\s*%?\s*(?:百分位|分位)(?:数)?)",
    re.IGNORECASE,
)


def canonical_percentiles(text: str) -> str:
    def replace(match: re.Match) -> str:
        number = match.group("ascii") or match.group("number")
        number = "99" if number in ("九九", "九十九") else number
        return f" p{Decimal(number).normalize():f} "
    return PERCENTILE_RE.sub(replace, unicodedata.normalize("NFKC", text))


# Only discard an entirely generic usage question. Any unknown qualifier survives unchanged.
_CONTEXT_WORDS = re.compile(
    r"什么样|怎么样|如何|怎么|哪些|什么|公司|业务|项目|作为|指标|进行|调优|优化|"
    r"场景|作用|应用|使用|这个|这种|它们|他们|以及|意义|价值|实际|具体|一下|"
    r"在|的|中|是|去|下|他|它|对|又|有|呢|吗|和|与|及|了|用|到|做|上|什么|[\s，,。？?、；;！!]+"
)


def percentile_usage_subject(text: str) -> str | None:
    canonical = canonical_percentiles(text).strip()
    match = re.match(r"p\d{1,2}(?:\.\d+)?(?=\s|$)", canonical)
    if not match:
        return None
    subject = match.group()
    tail = canonical[match.end():].replace(subject, "")
    if re.search(r"场景|业务|项目|作用|调优|应用|意义", tail) and not _CONTEXT_WORDS.sub("", tail):
        return subject[1:] + "分位"
    return None
