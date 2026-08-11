"""ingestion：Markdown 解析 —— 标题树 + 归一化正文（MVP 任务 2）。

用 markdown-it-py 的 token 流（不经过 HTML 渲染）把 Markdown 文档解析为
「标题树 + 归一化正文」，供任务 3 语义切块使用。纯内存文本操作，零出网。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

#: 使用 js-default 预设：启用 GFM 表格 / 删除线，不启用 linkify（无需额外依赖）。
_MD = MarkdownIt("default")

#: slug 用：保留字母/数字/中文/空白/连字符/下划线，其余标点剔除。
_PUNCT_RE = re.compile(r"[^\w\s-]")
_SPACE_RUN_RE = re.compile(r"\s+")
_INLINE_SPACES_RE = re.compile(r"\s{2,}")

_PLACEHOLDER_ANCHOR = "section"


@dataclass(frozen=True)
class HeadingNode:
    """标题树节点：层级 / 归一化文本 / 稳定唯一锚点 / 嵌套子标题。"""

    level: int  # 1-6
    text: str
    anchor: str
    children: tuple[HeadingNode, ...] = ()


@dataclass(frozen=True)
class ParsedMarkdown:
    """Markdown 解析结果：标题 + 标题树 + 归一化正文。"""

    title: str | None  # frontmatter title 或首个 H1（无则 None）
    heading_tree: tuple[HeadingNode, ...]
    normalized_text: str


@dataclass
class _MutableNode:
    """构建标题树时用的可变中间节点，最终冻结为 HeadingNode。"""

    level: int
    text: str
    anchor: str
    children: list[_MutableNode] = field(default_factory=list)


def parse_markdown(text: str) -> ParsedMarkdown:
    """把 Markdown 文本解析为标题树 + 归一化正文（纯函数，零出网）。"""
    text = _to_lf(text).lstrip("﻿")
    meta, body = _split_frontmatter(text)
    tokens = _MD.parse(body)
    return ParsedMarkdown(
        title=_resolve_title(meta, tokens),
        heading_tree=_build_heading_tree(tokens),
        normalized_text=_build_normalized_text(tokens),
    )


def _to_lf(text: str) -> str:
    """统一换行为 LF（CRLF / 独立 CR → LF）。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """提取文档开头的 YAML frontmatter，返回 (meta, 正文)；无则 ( {}, 原文 )。

    手写简单解析：仅 ``key: value`` 行，不做完整 YAML。
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() != "---":
            continue
        meta: dict[str, str] = {}
        for line in lines[1:i]:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
        return meta, "\n".join(lines[i + 1 :])
    return {}, text


def _iter_headings(tokens: list) -> list[tuple[int, str]]:
    """按文档顺序收集 (level, 归一化标题文本)；空标题文本为 ""。"""
    headings: list[tuple[int, str]] = []
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        level = int(tok.tag[1:])
        text = ""
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
            text = _inline_token_text(tokens[i + 1])
        headings.append((level, text))
    return headings


def _resolve_title(meta: dict[str, str], tokens: list) -> str | None:
    """标题优先级：frontmatter title > 首个非空 H1；均无则 None。"""
    fm_title = meta.get("title")
    if fm_title is not None and fm_title.strip():
        return fm_title.strip()
    for level, text in _iter_headings(tokens):
        if level == 1 and text:
            return text
    return None


def _build_heading_tree(tokens: list) -> tuple[HeadingNode, ...]:
    """由 token 流组装嵌套标题树：子标题挂在最近前驱上级下。"""
    roots: list[_MutableNode] = []
    stack: list[_MutableNode] = []
    seen: dict[str, int] = {}
    for level, text in _iter_headings(tokens):
        node = _MutableNode(level=level, text=text, anchor=_make_anchor(text, seen))
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return _freeze_nodes(roots)


def _make_anchor(text: str, seen: dict[str, int]) -> str:
    """生成稳定唯一锚点：slug 为基址，同级/全局重复加 -1/-2 后缀。"""
    base = _slugify(text) or _PLACEHOLDER_ANCHOR
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def _slugify(text: str) -> str:
    """GitHub 风格 slug：小写、去标点、空白转连字符、折叠连字符。"""
    slug = _PUNCT_RE.sub("", text.strip().lower())
    slug = _SPACE_RUN_RE.sub("-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _freeze_nodes(nodes: list[_MutableNode]) -> tuple[HeadingNode, ...]:
    """把可变中间树递归冻结为不可变 HeadingNode 树。"""
    return tuple(
        HeadingNode(level=n.level, text=n.text, anchor=n.anchor, children=_freeze_nodes(n.children))
        for n in nodes
    )


def _build_normalized_text(tokens: list) -> str:
    """拼接非标题块为归一化正文；标题文本单独成块保证可检索。"""
    blocks: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type == "heading_open":
            if i + 1 < n and tokens[i + 1].type == "inline":
                text = _inline_token_text(tokens[i + 1])
                if text:
                    blocks.append(text)
                i += 1  # 跳过该标题的 inline token，避免正文重复
        elif tok.type == "inline":
            text = _inline_token_text(tok)
            if text:
                blocks.append(text)
        elif tok.type in ("fence", "code_block"):
            text = tok.content.rstrip("\n")
            if text:
                blocks.append(text)
        i += 1
    return _normalize_text("\n".join(blocks))


def _inline_token_text(tok: object) -> str:
    """从 inline token 的子 token 提取纯文本：去标记、保可读内容。"""
    parts: list[str] = []
    for child in tok.children or []:  # type: ignore[attr-defined]
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            alt = _image_alt(child)
            if alt:
                parts.append(alt)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return _INLINE_SPACES_RE.sub(" ", "".join(parts)).strip()


def _image_alt(tok: object) -> str:
    """图片 token 的 alt 文本（无 alt 返回空串）。"""
    if tok.children:  # type: ignore[attr-defined]
        return "".join(c.content for c in tok.children)  # type: ignore[attr-defined]
    return tok.content  # type: ignore[attr-defined]


def _normalize_text(text: str) -> str:
    """折叠连续空行至多留一个、去行尾空白、去首尾空行。"""
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if line == "":
            blank += 1
            if blank == 1:
                out.append(line)
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip("\n")
