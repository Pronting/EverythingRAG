"""ingestion：Markdown 解析 —— 块级结构 + 标题树 + 归一化正文（MVP 任务 2/3）。

用 markdown-it-py 的 token 流（不经过 HTML 渲染）把 Markdown 文档解析为
有序块级结构 ``blocks``（单一数据源），标题树与归一化正文均由 blocks 派生，
供任务 3 语义切块使用。纯内存文本操作，零出网。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token

#: 使用 js-default 预设：启用 GFM 表格 / 删除线，不启用 linkify（无需额外依赖）。
#: disable("code")：关闭「4 空格缩进 = 代码块」——Yuque/Obsidian 导出的工程文档
#: 常用缩进做视觉排版（如加粗标签行、eventType 列表），被误判成代码块会产出
#: 「只有标签、零信息」的噪声块，且代码块在切块时永不被合并/去 ** 标记。
_MD = MarkdownIt("default").disable("code")

#: 表格单元格里作者手写的假标题标记（如 ``| ##### 事件 |``）——单元格内 ``#``
#: 不是真标题（标题是块级），纯噪声，切块时剥离。
_HEADING_MARKER_RE = re.compile(r"^#{1,6}\s*")

#: slug 用：保留字母/数字/中文/空白/连字符/下划线，其余标点剔除。
_PUNCT_RE = re.compile(r"[^\w\s-]")
_SPACE_RUN_RE = re.compile(r"\s+")
_INLINE_SPACES_RE = re.compile(r"\s{2,}")

#: 噪声行识别（优化①：低信息块不污染向量检索）。
#: 只作用于「顶层段落」，不碰 list/quote/表格项（避免链接收藏节整节消失）。
#: 纯标签行（Obsidian `#c端 #埋点`）、纯 URL、纯标点/表情行均无检索价值；
#: 标签/URL 后无空格粘连中文正文（`#c端：这是正文` / `https://x.com，参见`）不是噪声。
_TAG_TOKEN = r"#[0-9A-Za-z_一-鿿][0-9A-Za-z_一-鿿/-]*"
_TAG_ONLY_RE = re.compile(rf"^(?:{_TAG_TOKEN})(?:[ \t]+{_TAG_TOKEN})*$")
_URL_ONLY_RE = re.compile(r"^https?://[^\s一-鿿，。！？；：、]+[.。]?$")
_PUNCT_EMOJI_ONLY_RE = re.compile(r"^[\W_]+$")

_PLACEHOLDER_ANCHOR = "section"


@dataclass(frozen=True)
class HeadingNode:
    """标题树节点：层级 / 归一化文本 / 稳定唯一锚点 / 嵌套子标题。"""

    level: int  # 1-6
    text: str
    anchor: str
    children: tuple[HeadingNode, ...] = ()


@dataclass(frozen=True)
class MDBlock:
    """Markdown 块级结构单元：kind + 归一化文本（heading 另带 level/anchor）。"""

    kind: str  # "heading" | "paragraph" | "code" | "list" | "table" | "quote" | "hr" | "image" | "frontmatter"
    text: str  # 归一化文本（heading 即标题文本；hr/frontmatter 为空）
    level: int | None = None  # 仅 heading：1-6
    anchor: str | None = None  # 仅 heading：文档唯一，与 heading_tree 一致
    src: str | None = None  # 仅 image：图片引用地址（图床 URL / 本地路径）


@dataclass(frozen=True)
class ParsedMarkdown:
    """Markdown 解析结果：标题 + 标题树 + 归一化正文 + 有序块级结构。"""

    title: str | None  # frontmatter title 或首个 H1（无则 None）
    heading_tree: tuple[HeadingNode, ...]
    normalized_text: str
    blocks: tuple[MDBlock, ...]  # 单一数据源：文档顺序块级结构


@dataclass
class _MutableNode:
    """构建标题树时用的可变中间节点，最终冻结为 HeadingNode。"""

    level: int
    text: str
    anchor: str
    children: list[_MutableNode] = field(default_factory=list)


def parse_markdown(text: str) -> ParsedMarkdown:
    """把 Markdown 文本解析为块结构 + 标题树 + 归一化正文（纯函数，零出网）。

    ``_build_blocks`` 是主 pass：按 token 流产出有序 MDBlock；标题树与
    归一化正文均由 blocks 派生，保证单一数据源。
    """
    text = _to_lf(text).lstrip("﻿")
    meta, body = _split_frontmatter(text)
    tokens = _MD.parse(body)
    blocks = _build_blocks(tokens, has_frontmatter=body is not text)
    return ParsedMarkdown(
        title=_resolve_title(meta, tokens),
        heading_tree=_heading_tree_from_blocks(blocks),
        normalized_text=_normalized_text_from_blocks(blocks),
        blocks=blocks,
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


def _iter_headings(tokens: list[Token]) -> list[tuple[int, str]]:
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


def _resolve_title(meta: dict[str, str], tokens: list[Token]) -> str | None:
    """标题优先级：frontmatter title > 首个非空 H1；均无则 None。"""
    fm_title = meta.get("title")
    if fm_title is not None and fm_title.strip():
        return fm_title.strip()
    for level, text in _iter_headings(tokens):
        if level == 1 and text:
            return text
    return None


def _build_blocks(tokens: list[Token], has_frontmatter: bool) -> tuple[MDBlock, ...]:
    """按 token 流产出文档顺序的块级结构（单一数据源，主 pass）。

    每个非标题 inline 文本为独立块（list/table/quote 内同理，保持块级粒度
    以可靠区分代码块与段落边界）；代码块（fence/code_block）永远独立成块；
    标题块的 level/anchor 与旧标题树使用同一 ``_make_anchor`` + ``seen``
    序列，保证锚点一致。
    """
    blocks: list[MDBlock] = []
    if has_frontmatter:
        blocks.append(MDBlock(kind="frontmatter", text=""))
    seen: set[str] = set()
    context: list[str] = []  # 容器栈：list / quote / table，决定 inline 的 kind
    table_row: list[str] = []  # 当前表格行的单元格累计（table 上下文内用）
    table_header: list[str] = []  # 当前表头单元格（首行）
    table_row_count = 0  # 当前表已产出的行数（首行为表头）
    table_prev_cells: list[str] = []  # 上一数据行单元格（合并单元格继承用）
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        t = tok.type
        if t == "heading_open":
            text = ""
            if i + 1 < n and tokens[i + 1].type == "inline":
                text = _inline_token_text(tokens[i + 1])
                i += 1  # 跳过该标题的 inline，避免重复
            blocks.append(
                MDBlock(kind="heading", text=text, level=int(tok.tag[1:]), anchor=_make_anchor(text, seen))
            )
        elif t == "inline":
            if context and context[-1] == "table":
                # 表格单元格：空单元格也保留 "" 占位，维持列对齐（rowspan 继承需要）
                table_row.append(_strip_heading_markers(_inline_token_text(tok)))
            else:
                # 图片单独成块（每个 image 一个 block，携带 src）；一行多图逐个产出。
                for alt, src in _image_refs(tok):
                    blocks.append(MDBlock(kind="image", text=alt, src=src))
                text = _inline_text_no_images(tok)
                if not text:
                    pass
                elif not context and _is_noise_text(text):
                    pass  # 顶层噪声行（纯标签/URL/标点）丢弃；list/quote 项保留
                else:
                    kind = context[-1] if context else "paragraph"
                    blocks.append(MDBlock(kind=kind, text=text))
        elif t in ("fence", "code_block"):
            text = _normalize_text(tok.content.rstrip("\n"))
            if not text:
                continue
            info = (getattr(tok, "info", "") or "").strip().lower()
            # 纯文本围栏（```````plain```` / ```` ```text`````）按正文处理：
            # Yuque 常把 canned 回复/转移文案包成 ```plain```，实为文本而非代码，
            # 当正文可参与短碎片合并，避免「几个字」的孤立块。
            if t == "fence" and info in ("plain", "text"):
                blocks.append(MDBlock(kind="paragraph", text=text))
            else:
                blocks.append(MDBlock(kind="code", text=text))
        elif t == "hr":
            blocks.append(MDBlock(kind="hr", text=""))
        elif t in ("bullet_list_open", "ordered_list_open"):
            context.append("list")
        elif t == "blockquote_open":
            context.append("quote")
        elif t == "table_open":
            context.append("table")
            table_header = []
            table_row_count = 0
            table_prev_cells = []
        elif t == "tr_open":
            table_row = []
        elif t == "tr_close":
            table_row_count += 1
            if not table_row:
                continue
            if table_row_count == 1:
                table_header = table_row
            else:
                # 合并单元格（rowspan）：仅当本行首列为空（是上一行的延续）时才继承
                # 上一行同列的单元格；首列非空 = 新一组，不继承（避免跨组串值）。
                cells = list(table_row)
                if cells and not cells[0]:
                    for idx in range(len(cells)):
                        if not cells[idx] and idx < len(table_prev_cells):
                            cells[idx] = table_prev_cells[idx]
                table_prev_cells = cells
                if table_header and any(table_header):
                    # 表头感知：header：value 对，跳过空表头/空值，语义化而非 cell | cell
                    pairs = [
                        f"{table_header[idx]}：{cells[idx]}"
                        for idx in range(min(len(table_header), len(cells)))
                        if table_header[idx] and cells[idx]
                    ]
                    blocks.append(MDBlock(kind="table", text="；".join(pairs)))
                else:
                    blocks.append(MDBlock(kind="table", text=" | ".join(table_row)))
        elif t in ("bullet_list_close", "ordered_list_close", "blockquote_close", "table_close"):
            if context:
                context.pop()
        i += 1
    return tuple(blocks)


def _image_refs(tok: Token) -> list[tuple[str, str]]:
    """inline token 内全部图片引用 (alt, src)，文档顺序；无图片返回 []。"""
    return [(_image_alt(child), _image_src(child)) for child in (tok.children or []) if child.type == "image"]


def _image_src(tok: Token) -> str:
    """图片 token 的 src（图床 URL / 本地路径）；缺省返回空串。"""
    return tok.attrGet("src") or ""


def _inline_text_no_images(tok: Token) -> str:
    """inline token 的非图片文本（图片 alt 不进正文，图片已单独成块）。"""
    parts: list[str] = []
    for child in tok.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            continue
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return _INLINE_SPACES_RE.sub(" ", "".join(parts)).strip()


def _heading_tree_from_blocks(blocks: tuple[MDBlock, ...]) -> tuple[HeadingNode, ...]:
    """由 heading blocks 重建嵌套标题树：子标题挂在最近前驱上级下。"""
    roots: list[_MutableNode] = []
    stack: list[_MutableNode] = []
    for block in blocks:
        if block.kind != "heading" or block.level is None:
            continue
        node = _MutableNode(level=block.level, text=block.text, anchor=block.anchor or "")
        while stack and stack[-1].level >= block.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return _freeze_nodes(roots)


def _normalized_text_from_blocks(blocks: tuple[MDBlock, ...]) -> str:
    """由 blocks 的文本以单个空行连接派生归一化正文（标题文本含在内）。"""
    return _normalize_text("\n\n".join(b.text for b in blocks if b.text))


def _make_anchor(text: str, seen: set[str]) -> str:
    """生成稳定唯一锚点：slug 为基址，撞车则递增后缀直到全局唯一。"""
    base = _slugify(text) or _PLACEHOLDER_ANCHOR
    candidate = base
    n = 1
    while candidate in seen:
        candidate = f"{base}-{n}"
        n += 1
    seen.add(candidate)
    return candidate


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


def _inline_token_text(tok: Token) -> str:
    """从 inline token 的子 token 提取纯文本：去标记、保可读内容。"""
    parts: list[str] = []
    for child in tok.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            alt = _image_alt(child)
            if alt:
                parts.append(alt)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return _INLINE_SPACES_RE.sub(" ", "".join(parts)).strip()


def _strip_heading_markers(text: str) -> str:
    """剥离表格单元格里作者手写的假标题标记（``##### 事件`` -> ``事件``）。"""
    return _HEADING_MARKER_RE.sub("", text)


def _is_noise_text(text: str) -> bool:
    """判断归一化文本是否为无检索价值的噪声行（纯标签 / 纯 URL / 纯标点表情）。

    优化①：Obsidian 标签行（``#c端 #埋点``）、纯 URL、纯标点/表情行会被整行
    丢弃，避免这类低信息块以「与查询同词」的姿态高排位污染向量检索。
    标签与正文混排的行（``#c端 这是正文``）不是噪声，保留。
    """
    s = text.strip()
    if not s:
        return True
    return (
        _TAG_ONLY_RE.match(s) is not None
        or _URL_ONLY_RE.match(s) is not None
        or _PUNCT_EMOJI_ONLY_RE.match(s) is not None
    )


def _image_alt(tok: Token) -> str:
    """图片 token 的 alt 文本（无 alt 返回空串）。"""
    if tok.children:
        return "".join(c.content for c in tok.children)
    return tok.content


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
