"""原文结构分段：证据视图的可靠结构边界（PRD RAG-REL-001 §5.3）。

只回答一个问题——「这段 chunk 原文能否在不破坏结构的前提下切成若干连续区间？」。
它是**纯文本操作**，不解析、不改写、不归一化正文：

* 表格、围栏代码、列表、问答对保持完整，绝不落单或在中间切开；
* 返回的分段互不重叠、按原文顺序、拼接后逐字等于输入原文（含分段之间的空行）；
* 结构无法确定时返回整段（调用方据此保留完整分块，按 PRD 的「不能猜测边界」处理）。

本模块不依赖向量库或模型，零出网，可在任何进程内重复运行得到相同结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "MIN_SEGMENT_CHARS",
    "is_question_prompt",
    "structure_segments",
]

#: 一个引用视图至少要有的字符数：过短的分段没有引用价值，还会挤占候选窗口。
MIN_SEGMENT_CHARS = 200

_FENCE_RE = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}\s+\S")
_TABLE_ROW_RE = re.compile(r"^[ \t]{0,3}\|.*\|[ \t]*$")
_LIST_MARKER_RE = re.compile(r"^[ \t]{0,3}(?:[-*+]|\d{1,9}[.)])[ \t]+\S")
_QUOTE_MARKER_RE = re.compile(r"^[ \t]{0,3}>[ \t]?\S")

#: 问答标记：这些行开启的单元必须与其后的回答一起保留。
_QUESTION_RE = re.compile(
    r"^[ \t]{0,3}(?:"
    r"[Qq][:：.、)]|"
    r"[Aa][:：.、)]|"
    r"问[:：.、)]|答[:：.、)]|"
    r"【问】|【答】|"
    r"问题[:：]|回答[:：]|"
    r"###?\s*[Qq]\d"
    r")"
)

#: 必须整体保留的结构类型：表格 / 围栏代码 / 列表。
_ATOMIC_KINDS = frozenset({"code", "table", "list"})
_HEADING_KIND = "heading"
_NARRATIVE_KIND = "narrative"


@dataclass(frozen=True)
class _Line:
    """一行原文：原始偏移、结构类型，以及它前面是否有空行。"""

    start: int
    end: int
    text: str
    kind: str
    blank_before: bool = False


def is_question_prompt(line: str) -> bool:
    """该行是否是问答对的提问行（需要与紧随其后的回答一起保留）。"""

    return _QUESTION_RE.match(line) is not None


def structure_segments(
    text: str, *, min_chars: int = MIN_SEGMENT_CHARS
) -> tuple[tuple[int, int], ...]:
    """把原文分成结构完整、互不重叠且覆盖全文的连续区间 ``[start, end)``。

    返回 ``((0, len(text)),)`` 表示「没有可靠边界，请保留完整分块」。
    """

    if not isinstance(text, str) or not text:
        return ()
    length = len(text)
    if length < max(1, min_chars):
        return ((0, length),)

    lines = _classify_lines(text)
    units = _group_lines(lines)
    segments = _merge_units(units, lines, length, min_chars)
    if len(segments) < 2:
        return ((0, length),)
    return tuple(segments)


def _classify_lines(text: str) -> list[_Line]:
    """逐行判定结构类型并保留原始偏移。"""

    lines: list[_Line] = []
    fence: str | None = None
    offset = 0
    blank_run = 0
    for raw in text.split("\n"):
        start = offset
        end = offset + len(raw)
        offset = end + 1
        if not raw.strip():
            blank_run += 1
            continue
        line = raw.strip()
        kind = _line_kind(line, fence_is_open=fence is not None)
        lines.append(
            _Line(
                start=start,
                end=end,
                text=line,
                kind=kind,
                blank_before=blank_run > 0,
            )
        )
        blank_run = 0
        if kind == "code":
            fence_match = _FENCE_RE.match(line)
            if fence is None and fence_match:
                fence = fence_match.group("fence")
            elif fence is not None and line.startswith(fence):
                fence = None
    return lines


def _line_kind(line: str, *, fence_is_open: bool) -> str:
    if fence_is_open or _FENCE_RE.match(line):
        return "code"
    if _TABLE_ROW_RE.match(line):
        return "table"
    if _HEADING_RE.match(line):
        return _HEADING_KIND
    if _LIST_MARKER_RE.match(line):
        return "list"
    if _QUOTE_MARKER_RE.match(line):
        return "quote"
    return _NARRATIVE_KIND


def _group_lines(lines: list[_Line]) -> list[tuple[int, int, str]]:
    """把行归并成「不可拆」单元 ``(起始行下标, 结束行下标含, 类型)``。

    边界只来自真正的结构变化：新标题、新表格、新代码块、空行隔开的列表，以及问答对
    结束后的空行。标题归入它引导的那个单元（因此不会把上一段正文吞进自己的范围）；
    问答对的提问与回答始终留在同一单元里，空行不会把它们分开。
    """

    units: list[tuple[int, int, str]] = []
    start_index: int | None = None
    pending_heading: int | None = None
    has_question = False

    def flush(end_index: int) -> None:
        nonlocal start_index, pending_heading, has_question
        if start_index is not None:
            units.append((start_index, end_index, _kind_label(lines, start_index)))
        start_index = None
        pending_heading = None
        has_question = False

    for index, line in enumerate(lines):
        if start_index is None:
            # 标题归入它引导的那个单元：标题行开启单元，正文行结束「待归入标题」状态。
            if line.kind == _HEADING_KIND:
                pending_heading = index
                continue
            start_index = pending_heading if pending_heading is not None else index
            pending_heading = None
            has_question = is_question_prompt(line.text)
            continue
        if _starts_new_unit(
            lines,
            index,
            unit_start=start_index,
            pending_heading=pending_heading,
            has_question=has_question,
        ):
            flush(index - 1)
            if line.kind == _HEADING_KIND:
                pending_heading = index
                continue
            start_index = index
            has_question = is_question_prompt(line.text)
            continue
        if pending_heading is not None:
            pending_heading = None
        if is_question_prompt(line.text):
            has_question = True
    if start_index is None and pending_heading is not None:
        start_index = pending_heading
    flush(len(lines) - 1)
    return units


def _answer_kind(text: str) -> bool:
    """该行是否是问答对里的回答行（``A:`` / ``答:`` / ``【答】``）。"""

    return bool(re.match(r"^[ \t]{0,3}(?:[Aa][:：.、)]|答[:：.、)]|【答】|回答[:：])", text))


def _kind_label(lines: list[_Line], start_index: int) -> str:
    """单元类型：标题开头记为 heading，否则取首行类型。"""

    return lines[start_index].kind


def _starts_new_unit(
    lines: list[_Line],
    index: int,
    *,
    unit_start: int,
    pending_heading: int | None,
    has_question: bool,
) -> bool:
    """当前行是否开启新的不可拆单元。"""

    line = lines[index]
    if line.kind == _HEADING_KIND:
        return True
    if pending_heading is not None:
        return False  # 标题引导的正文，属于同一单元
    current_kind = lines[unit_start].kind
    if line.kind in _ATOMIC_KINDS or current_kind in _ATOMIC_KINDS:
        if current_kind in _ATOMIC_KINDS and line.kind in _ATOMIC_KINDS:
            return line.kind != current_kind or _has_blank_line(lines, index)
        return True
    if has_question and line.blank_before and line.kind == _NARRATIVE_KIND:
        # 问答对已经完整，空行之后的新段落不再属于它
        return True
    if {current_kind, line.kind} <= {_NARRATIVE_KIND, "answer"}:
        return False
    return line.kind != current_kind


def _has_blank_line(lines: list[_Line], index: int) -> bool:
    """该行与上一行之间是否存在空行（列表可由空行分成两组）。"""

    return index > 0 and lines[index - 1].end + 1 < lines[index].start


def _to_spans(
    units: list[tuple[int, int]],
    lines: list[_Line],
    length: int,
) -> list[tuple[int, int]]:
    """把行单元转换成原文区间。

    区间覆盖「本单元的行 + 其后的间隔行」，分段之间只由换行分隔、不存在空隙，
    因此 ``''.join(text[start:end] for ...)`` 逐字等于原文。
    """

    spans: list[tuple[int, int]] = []
    count = len(units)
    for position, (start_index, end_index) in enumerate(units):
        start = lines[start_index].start
        end = length if end_index + 1 >= len(lines) else lines[end_index + 1].start
        spans.append((start, length if position == count - 1 else end))
    return spans


def _merge_units(
    units: list[tuple[int, int, str]],
    lines: list[_Line],
    length: int,
    min_chars: int,
) -> list[tuple[int, int]]:
    """把行单元转换成原文区间，并合并过短的相邻区间。"""

    if not units:
        return [(0, length)]
    spans = _to_spans([(start, end) for start, end, _kind in units], lines, length)
    if len(spans) == 1:
        return spans
    keep = [position for position, (start, end) in enumerate(spans) if end - start >= min_chars]
    if not keep:
        return [(0, length)]
    merged: list[tuple[int, int]] = []
    for index, position in enumerate(keep):
        start = 0 if index == 0 else spans[keep[index - 1]][1]
        end = length if index == len(keep) - 1 else spans[position][1]
        merged.append((start, end))
    if len(merged) < 2 or merged[-1][1] - merged[-1][0] < min_chars:
        return [(0, length)]
    return merged
