"""Read-time image evidence projection; stored descriptions and vectors stay intact."""

import re

_SECTION = re.compile(r"【([^】\n]{1,40})】")
_EMPTY = re.compile(r"^(?:无|不适用|未提供上下文|未提供|无上下文|无明显文字)[。；;\s]*$")


def image_facts(text: str) -> str:
    """Separate actual visual observations from copied document context and generated tags.

    Keep OCR, numbers, units and relationships verbatim. Unknown/legacy formats fail open.
    This is a retrieval/context view, never a destructive migration of cached descriptions.
    """
    description = text.split("【图片描述】", 1)[-1].strip()
    sections = list(_SECTION.finditer(description))
    if not sections:
        return description
    parts = [description[:sections[0].start()].strip()]
    seen: set[str] = set()
    for i, section in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(description)
        body = description[section.end():end].strip()
        if section.group(1) in {"所属主题", "正文上下文", "与上下文的关系", "可检索标签"}:
            continue
        if not body or _EMPTY.fullmatch(body) or body in seen:
            continue
        seen.add(body)
        parts.append(section.group() + body)
    return "\n".join(part for part in parts if part) or description
