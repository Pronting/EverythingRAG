"""ingestion：语义切块 —— 标题锚点分层 + 防边界漂移（MVP 任务 3）。

把任务 2 的 ``ParsedMarkdown.blocks`` 按标题锚点分层切成带来源/锚点元数据的
文本块，供任务 4 向量化入库。以标题为一级边界；代码块永远独立成块；
block_id 基于标题锚点路径链 + 区间内序号，局部编辑不改变标题结构时
未受影响区间的 block_id 保持不变（防边界漂移核心）。纯内存计算，零出网。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.ingestion.markdown_parser import MDBlock, ParsedMarkdown

#: 可累计进标题区间的内容块类型；代码块与图片块单独处理，hr/frontmatter 无内容。
_CONTENT_KINDS = frozenset({"paragraph", "list", "table", "quote"})

#: 句子切分用：贪婪匹配「非句末符若干 + 一个句末符」，保留句末符便于拼接还原。
_SENTENCE_RE = re.compile(r"[^。！？.!?；;]*[。！？.!?；;]")
_HEADING_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_HEADING_INLINE_VALUE_RE = re.compile(r"^.{1,80}?[：:]\s*(?P<value>\S.+)$")

@dataclass(frozen=True)
class Chunk:
    """单个切块：稳定锚点路径链 + 序号 / 文本 / 来源 / 标题路径 / 锚点 / 序号 / 类型。"""

    block_id: str  # 如 "/H1/{anchor}/H2/{anchor}#1"；无标题文档 "#1"、"#2"…
    text: str
    source_file: str
    heading_path: str  # 标题路径文本，如 "笔记方法 > 检索"；无标题则 ""
    anchor: str  # 所属标题锚点链末端；无标题则 ""
    seq: int  # 该锚点区间内块序号（1 起）
    kind: str = "text"  # text | table | code | image；table/code/image 不与正文混合
    src: str | None = None  # 仅 image：图片引用地址（图床 URL / 本地路径）
    heading_aliases: tuple[str, ...] = ()  # 聚合短兄弟节时保留全部原始标题路径


@dataclass(frozen=True)
class _Region:
    """一个标题区间：标题锚点链 + 内容块文本 + 代码块文本。"""

    chain: tuple[tuple[int, str, str], ...]  # (level, text, anchor) 文档顺序
    content: tuple[tuple[str, str], ...]  # (paragraph/list/table/quote, 归一化文本)
    code: tuple[str, ...]  # 独立成块的代码块文本
    images: tuple[tuple[str, str], ...] = ()  # (alt, src) 图片引用，独立成块


def chunk_document(
    parsed: ParsedMarkdown,
    source_file: str,
    max_chunk_chars: int = 2000,  # PRD 4.3：默认约 2000 字符
    min_chunk_chars: int = 120,  # 优化②：短碎片合并的下限（低于该长度的文本块不独立成块）
) -> list[Chunk]:
    """把解析结果按标题锚点分层切成文本块（纯函数，零出网）。

    - 标题区间默认合并为一个块；超长则按段落/句子/字符边界逐级拆分。
    - 代码块永远独立成块，即使超过上限也不拆散（PRD 规则 7）。
    - 无标题文档按段落切块（``#1``、``#2``…）。
    - HTML/Markdown 展示噪声已由 parser 清洗；低于 ``min_chunk_chars`` 的同标题
      相邻正文前向合并，绝不跨标题、表格或代码边界。
    """
    max_chars = max(1, int(max_chunk_chars))
    min_chars = max(0, int(min_chunk_chars))
    chunks: list[Chunk] = []
    doc_seq = 0  # 无标题区间用文档级序号，保证 block_id 全局唯一
    for region in _build_regions(parsed.blocks):
        if region.chain:
            _emit_heading_region(region, source_file, max_chars, chunks)
        else:
            doc_seq = _emit_headingless_region(region, source_file, max_chars, doc_seq, chunks)
    chunks = _coalesce_short_chunks(chunks, min_chars, max_chars)
    return _coalesce_short_sibling_chunks(chunks, min_chars, max_chars)


def _build_regions(blocks: tuple[MDBlock, ...]) -> list[_Region]:
    """把块流切分为标题区间：标题开新区间，内容累计进当前区间。"""
    regions: list[_Region] = []
    chain: list[tuple[int, str, str]] = []
    content: list[tuple[str, str]] = []
    code: list[str] = []
    images: list[tuple[str, str]] = []
    for block in blocks:
        if block.kind == "heading" and block.level is not None:
            _flush_region(regions, chain, content, code, images)
            while chain and chain[-1][0] >= block.level:
                chain.pop()
            chain.append((block.level, block.text, block.anchor or ""))
        elif block.kind == "code":
            if block.text:
                code.append(block.text)
        elif block.kind == "image":
            images.append((block.text, block.src or ""))
        elif block.kind in _CONTENT_KINDS and block.text:
            content.append((block.kind, block.text))
    _flush_region(regions, chain, content, code, images)
    return regions


def _flush_region(
    regions: list[_Region],
    chain: list[tuple[int, str, str]],
    content: list[tuple[str, str]],
    code: list[str],
    images: list[tuple[str, str]],
) -> None:
    """把当前累计区间封存入 regions，并保住写在叶子标题里的事实。"""

    effective_content = list(content)
    if not content and not code and chain:
        inline_heading = _informative_heading_content(chain[-1][1])
        if inline_heading is not None:
            effective_content.append(("paragraph", inline_heading))
    if effective_content or code or images:
        regions.append(
            _Region(
                chain=tuple(chain),
                content=tuple(effective_content),
                code=tuple(code),
                images=tuple(images),
            )
        )
    content.clear()
    code.clear()
    images.clear()


def _informative_heading_content(heading: str) -> str | None:
    """Return a leaf heading that carries a URL or ``label: value`` fact by itself.

    Rich-text exports often encode the entire fact in a heading and put only an image—or nothing—
    below it. Generic structural headings remain metadata-only, so this does not create a chunk
    for every empty section.
    """

    normalized = heading.strip()
    if _HEADING_URL_RE.search(normalized):
        return normalized
    match = _HEADING_INLINE_VALUE_RE.match(normalized)
    if match and len(match.group("value").strip()) >= 2:
        return normalized
    return None


def _emit_heading_region(region: _Region, source_file: str, max_chars: int, chunks: list[Chunk]) -> None:
    """标题区间：内容合并成一块（超长逐级拆），代码块独立成块，seq 递增。"""
    prefix = _block_id_prefix(region.chain)
    heading_path = " > ".join(text for _, text, _ in region.chain)
    anchor = region.chain[-1][2]
    seq = 0
    for kind, part in _split_content_blocks(region.content, max_chars):
        seq += 1
        chunk_kind = "table" if kind == "table" else "text"
        chunks.append(
            Chunk(f"{prefix}#{seq}", part, source_file, heading_path, anchor, seq, kind=chunk_kind)
        )
    for alt, src in region.images:
        seq += 1
        chunks.append(Chunk(f"{prefix}#{seq}", alt, source_file, heading_path, anchor, seq, kind="image", src=src))
    for text in region.code:
        # 优化① 提速配套：代码块超上限按字符边界拆分（_split_chars 短块原样返回）。
        # 巨型代码块（如 JVM 配置转储）若整块保留，会 OOM 且无法作上下文喂给 LLM。
        for part in _split_chars(text, max_chars):
            seq += 1
            chunks.append(Chunk(f"{prefix}#{seq}", part, source_file, heading_path, anchor, seq, kind="code"))


def _emit_headingless_region(
    region: _Region,
    source_file: str,
    max_chars: int,
    start_seq: int,
    chunks: list[Chunk],
) -> int:
    """无标题区间：每个段落为最小单元切块，seq 全局递增；段落超长同规则拆。"""
    seq = start_seq
    for kind, part in _split_content_blocks(region.content, max_chars):
        seq += 1
        chunk_kind = "table" if kind == "table" else "text"
        chunks.append(Chunk(f"#{seq}", part, source_file, "", "", seq, kind=chunk_kind))
    for alt, src in region.images:
        seq += 1
        chunks.append(Chunk(f"#{seq}", alt, source_file, "", "", seq, kind="image", src=src))
    for text in region.code:
        # 优化① 提速配套：超长代码块按字符边界拆分（同带标题分支）。
        for part in _split_chars(text, max_chars):
            seq += 1
            chunks.append(Chunk(f"#{seq}", part, source_file, "", "", seq, kind="code"))
    return seq


def _block_id_prefix(chain: tuple[tuple[int, str, str], ...]) -> str:
    """block_id 的稳定前缀：标题锚点路径链，如 "/H1/a/H2/b"。"""
    return "".join(f"/H{level}/{anchor}" for level, _, anchor in chain)


def _split_content_blocks(
    blocks: tuple[tuple[str, str], ...], max_chars: int
) -> list[tuple[str, str]]:
    """按内容类型和语义边界切片；表格行不与普通正文混成同一块。

    连续表格行以换行打包并标记为 ``table``；普通段落仍以空行打包。类型切换本身
    就是边界，避免统计表被正文掩盖，也使后端能给表格写入正确 ``chunk_type``。
    """
    result: list[tuple[str, str]] = []
    run_kind = ""
    run_texts: list[str] = []

    def flush_run() -> None:
        nonlocal run_kind, run_texts
        if not run_texts:
            return
        if run_kind == "table":
            result.extend(("table", part) for part in _split_table_rows(tuple(run_texts), max_chars))
        else:
            result.extend(("text", part) for part in _split_text_blocks(tuple(run_texts), max_chars))
        run_kind = ""
        run_texts = []

    for block_kind, text in blocks:
        normalized_kind = "table" if block_kind == "table" else "text"
        if run_texts and normalized_kind != run_kind:
            flush_run()
        run_kind = normalized_kind
        run_texts.append(text)
    flush_run()
    return result


def _split_text_blocks(blocks: tuple[str, ...], max_chars: int) -> list[str]:
    """普通正文按段落边界打包；单段过长再按句子/字符边界拆分。"""
    return _pack_units(blocks, max_chars, separator="\n\n", oversize=_split_sentence_text)


def _split_table_rows(rows: tuple[str, ...], max_chars: int) -> list[str]:
    """表格按完整语义行打包；超长单行优先在「；」单元格边界拆分。"""
    return _pack_units(rows, max_chars, separator="\n\n", oversize=_split_table_row)


def _pack_units(
    units: tuple[str, ...],
    max_chars: int,
    separator: str,
    oversize: Callable[[str, int], list[str]],
) -> list[str]:
    """按完整单元打包到上限；超长单元交给对应语义 splitter。"""
    result: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_parts = [unit] if len(unit) <= max_chars else oversize(unit, max_chars)
        for part in unit_parts:
            extra = len(part) + (len(separator) if current else 0)
            if current and current_len + extra > max_chars:
                result.append(separator.join(current))
                current = []
                current_len = 0
                extra = len(part)
            current.append(part)
            current_len += extra
    if current:
        result.append(separator.join(current))
    return result


def _split_table_row(text: str, max_chars: int) -> list[str]:
    """超长表格行按字段边界拆，字段自身过长才退化为字符切分。"""
    fields = [field.strip() for field in text.split("；") if field.strip()]
    if len(fields) <= 1:
        return _split_chars(text, max_chars)
    return _pack_units(tuple(fields), max_chars, separator="；", oversize=_split_chars)


def _split_sentence_text(text: str, max_chars: int) -> list[str]:
    """按句子（。！？.!?）拆超长文本为 ≤ 上限的子块；句子仍超长则按字符拆。"""
    if len(text) <= max_chars:
        return [text]
    parts = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if not current:
            current = part
        elif len(current) + len(part) <= max_chars:
            current += part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_split_chars(chunk, max_chars))
    return result


def _split_sentences(text: str) -> list[str]:
    """按句末符拆分并保留句末符（拼接可还原原文）。"""
    parts = _SENTENCE_RE.findall(text)
    tail = _SENTENCE_RE.sub("", text)
    if tail:
        parts.append(tail)
    return parts or [text]


def _split_chars(text: str, max_chars: int) -> list[str]:
    """最终兜底切分：优先完整行、标点和空白，实在没有才按固定字符截断。

    所有分隔符都保留在前一块中，因此把结果直接拼接可逐字还原原文。只在窗口后
    45% 搜索软边界，避免为了追求标点把块切得过短。
    """

    if len(text) <= max_chars:
        return [text]
    result: list[str] = []
    remaining = text
    min_soft_cut = max(1, int(max_chars * 0.55))
    boundary_groups = ("\n", "。！？.!?；;", "，,：:、", " \t")
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max_chars
        for boundaries in boundary_groups:
            candidate = max((window.rfind(char) for char in boundaries), default=-1)
            if candidate >= min_soft_cut:
                cut = candidate + 1
                break
        result.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        result.append(remaining)
    return result


def _coalesce_short_chunks(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """把低于最小长度的相邻文本块前向合并，使每块达到语义完整长度。

    - 低于 ``min_chars`` 的碎片被并入后续块（直到 ≥min 或触及上限）；已达标且
      后续块也达标的相邻块保持独立，不过度粘连。
    - 前向合并不掉的短碎片（位于文档末尾、或后面紧跟代码块）会**向后并入前一
      个文本块**，避免「几个字 / 一行代码」的孤立碎片块污染检索。
    - 标题路径/锚点是硬边界，任何短块都不得跨 heading 合并。
    - 表格、代码与图片恒独立，不参与正文合并。
    - 合并块保留首个块的 block_id/锚点身份（内容归属其起点的标题区间）。
    """
    if min_chars <= 0 or not chunks:
        return chunks
    out: list[Chunk] = []
    pending: list[Chunk] = []
    pending_len = 0

    def _flush() -> None:
        nonlocal pending, pending_len
        if not pending:
            return
        # 短碎片只可向后并入同一标题身份的正文块，禁止跨标题/代码/表格粘连。
        if (
            pending_len < min_chars
            and out
            and out[-1].kind == "text"
            and _same_chunk_scope(out[-1], pending[0])
        ):
            prev = out[-1]
            if len(prev.text) + 2 + pending_len <= max_chars:
                out[-1] = Chunk(
                    block_id=prev.block_id,
                    text=prev.text + "\n\n" + "\n\n".join(c.text for c in pending),
                    source_file=prev.source_file,
                    heading_path=prev.heading_path,
                    anchor=prev.anchor,
                    seq=prev.seq,
                    kind="text",
                )
                pending, pending_len = [], 0
                return
        if len(pending) > 1:
            first = pending[0]
            merged = Chunk(
                block_id=first.block_id,
                text="\n\n".join(c.text for c in pending),
                source_file=first.source_file,
                heading_path=first.heading_path,
                anchor=first.anchor,
                seq=first.seq,
                kind="text",
            )
            out.append(merged)
        else:
            out.append(pending[0])
        pending, pending_len = [], 0

    for chunk in chunks:
        if chunk.kind != "text":
            _flush()
            out.append(chunk)
            continue
        if pending and not _same_chunk_scope(pending[0], chunk):
            _flush()
        # 已达最小长度且下一块也达标 -> 独立成块（不过度粘连）
        if pending_len >= min_chars and len(chunk.text) >= min_chars:
            _flush()
        # 追加下一块会超过上限 -> 先落盘当前累积
        if pending and pending_len + 2 + len(chunk.text) > max_chars:
            _flush()
        pending.append(chunk)
        pending_len += (2 if pending_len else 0) + len(chunk.text)
    _flush()
    return out


def _same_chunk_scope(left: Chunk, right: Chunk) -> bool:
    """正文可合并范围：同文件、同 heading_path、同 anchor，且双方都是 text。"""
    return (
        left.kind == right.kind == "text"
        and left.source_file == right.source_file
        and left.heading_path == right.heading_path
        and left.anchor == right.anchor
    )


def _coalesce_short_sibling_chunks(
    chunks: list[Chunk], min_chars: int, max_chars: int
) -> list[Chunk]:
    """聚合同一父标题下连续的短叶子节，并把每个叶子标题写回正文。

    标题仍然是强语义边界，但不是所有标题都值得单独占一个向量。事故记录和方案文档
    常把「现象 / 原因 / 处理」各写成十几个字的小节；单独嵌入信息不足，检索时也难以
    一次拿齐答案。这里仅聚合 **同层、同父级、同类型且均低于 min** 的连续兄弟节，
    顶级标题、正文/表格类型切换、代码和图片仍是硬边界。显式 ``【叶子标题】`` 标签
    保证合并后每条事实的原始归属仍可被 BM25、向量模型和回答模型识别。
    """

    if min_chars <= 0 or not chunks:
        return chunks
    out: list[Chunk] = []
    pending: list[Chunk] = []
    pending_scope: tuple[str, str, int, str] | None = None
    pending_len = 0

    def flush() -> None:
        nonlocal pending, pending_scope, pending_len
        if len(pending) < 2:
            out.extend(pending)
        else:
            first = pending[0]
            parent = _heading_parent(first.heading_path)
            leaves = [_heading_leaf(chunk.heading_path) for chunk in pending]
            text = "\n\n".join(
                f"【{leaf}】\n{chunk.text}" for leaf, chunk in zip(leaves, pending, strict=True)
            )
            out.append(
                Chunk(
                    block_id=first.block_id,
                    text=text,
                    source_file=first.source_file,
                    heading_path=f"{parent} > {' / '.join(leaves)}",
                    anchor=first.anchor,
                    seq=first.seq,
                    kind=first.kind,
                    heading_aliases=tuple(chunk.heading_path for chunk in pending),
                )
            )
        pending = []
        pending_scope = None
        pending_len = 0

    for chunk in chunks:
        scope = _short_sibling_scope(chunk, min_chars)
        leaf = _heading_leaf(chunk.heading_path)
        labelled_len = len(chunk.text) + len(leaf) + 3
        if scope is None:
            flush()
            out.append(chunk)
            continue
        if pending and pending_len >= min_chars:
            # 达到最小有效长度就落块，避免把大量弱关联兄弟小节一直装到上限。
            flush()
        if pending and (
            scope != pending_scope
            or leaf == _heading_leaf(pending[-1].heading_path)
            or pending_len + 2 + labelled_len > max_chars
        ):
            flush()
        pending.append(chunk)
        pending_scope = scope
        pending_len += (2 if pending_len else 0) + labelled_len
    flush()
    return out


def _short_sibling_scope(chunk: Chunk, min_chars: int) -> tuple[str, str, int, str] | None:
    if chunk.kind not in {"text", "table"} or len(chunk.text) >= min_chars:
        return None
    parts = _heading_parts(chunk.heading_path)
    if len(parts) < 2:
        return None
    return (chunk.source_file, " > ".join(parts[:-1]), len(parts), chunk.kind)


def _heading_parts(heading_path: str) -> list[str]:
    return [part.strip() for part in heading_path.split(" > ") if part.strip()]


def _heading_parent(heading_path: str) -> str:
    return " > ".join(_heading_parts(heading_path)[:-1])


def _heading_leaf(heading_path: str) -> str:
    parts = _heading_parts(heading_path)
    return parts[-1] if parts else heading_path.strip()
