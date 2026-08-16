"""ingestion：语义切块 —— 标题锚点分层 + 防边界漂移（MVP 任务 3）。

把任务 2 的 ``ParsedMarkdown.blocks`` 按标题锚点分层切成带来源/锚点元数据的
文本块，供任务 4 向量化入库。以标题为一级边界；代码块永远独立成块；
block_id 基于标题锚点路径链 + 区间内序号，局部编辑不改变标题结构时
未受影响区间的 block_id 保持不变（防边界漂移核心）。纯内存计算，零出网。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.markdown_parser import MDBlock, ParsedMarkdown

#: 可累计进标题区间的内容块类型；代码块与图片块单独处理，hr/frontmatter 无内容。
_CONTENT_KINDS = frozenset({"paragraph", "list", "table", "quote"})

#: 句子切分用：贪婪匹配「非句末符若干 + 一个句末符」，保留句末符便于拼接还原。
_SENTENCE_RE = re.compile(r"[^。！？.!?]*[。！？.!?]")

#: 内联 HTML 标签（markdown-it html=False 下作为字面文本进正文，如 <font style="...">）。
#: 标签本身无检索价值且污染嵌入/上下文，切块时从内容块剥离（代码块保留原样）。
_INLINE_HTML_RE = re.compile(r"<[^>]*>")

#: 残留的 markdown 加粗/斜体标记（如 ``** 规则`` 前导空格、``***`` 三个星号这类
#: markdown-it 未识别的边界情况，会作为字面文本留下），一并剥离；单个 ``*`` 保留。
_BOLD_MARKER_RE = re.compile(r"\*{2,}")


@dataclass(frozen=True)
class Chunk:
    """单个切块：稳定锚点路径链 + 序号 / 文本 / 来源 / 标题路径 / 锚点 / 序号 / 类型。"""

    block_id: str  # 如 "/H1/{anchor}/H2/{anchor}#1"；无标题文档 "#1"、"#2"…
    text: str
    source_file: str
    heading_path: str  # 标题路径文本，如 "笔记方法 > 检索"；无标题则 ""
    anchor: str  # 所属标题锚点链末端；无标题则 ""
    seq: int  # 该锚点区间内块序号（1 起）
    kind: str = "text"  # "text" | "code" | "image"：code/image 恒独立、不参与短块合并
    src: str | None = None  # 仅 image：图片引用地址（图床 URL / 本地路径）


@dataclass(frozen=True)
class _Region:
    """一个标题区间：标题锚点链 + 内容块文本 + 代码块文本。"""

    chain: tuple[tuple[int, str, str], ...]  # (level, text, anchor) 文档顺序
    content: tuple[str, ...]  # 累计的内容块文本（块间以 \\n\\n 连接）
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
    - 优化②：正文内联 HTML 标签（<font> 等）剥离；低于 ``min_chunk_chars`` 的
      相邻短文本块前向合并，使检索/嵌入面对语义完整的块而非残句。
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
    return _drop_numeric_noise(chunks)


#: 结构化数值噪声块（定价/统计表）的「数字+标点」占比阈值：超过即视为语义密度过低，
#: 是 dense 检索的噪声磁铁（对无关查询虚高相似度，如「大模型价格表」），入库前丢弃。
_NUMERIC_NOISE_RATIO = 0.45
_NUMERIC_NOISE_MIN_LEN = 50
_NUMERIC_PUNCT = frozenset("；：，。、()（）$.,/[]{}:;|")


def _drop_numeric_noise(
    chunks: list[Chunk],
    ratio: float = _NUMERIC_NOISE_RATIO,
    min_len: int = _NUMERIC_NOISE_MIN_LEN,
) -> list[Chunk]:
    """丢弃数字+标点占比过高的结构化数值噪声块（定价/统计表）；代码块与短块恒保留。"""

    def _noise_ratio(text: str) -> float:
        n = sum(1 for c in text if c.isdigit() or c in _NUMERIC_PUNCT)
        return n / max(len(text), 1)

    return [
        chunk
        for chunk in chunks
        if chunk.kind == "code" or len(chunk.text) < min_len or _noise_ratio(chunk.text) <= ratio
    ]


def _build_regions(blocks: tuple[MDBlock, ...]) -> list[_Region]:
    """把块流切分为标题区间：标题开新区间，内容累计进当前区间。"""
    regions: list[_Region] = []
    chain: list[tuple[int, str, str]] = []
    content: list[str] = []
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
            content.append(_strip_inline_html(block.text))
    _flush_region(regions, chain, content, code, images)
    return regions


def _flush_region(
    regions: list[_Region],
    chain: list[tuple[int, str, str]],
    content: list[str],
    code: list[str],
    images: list[tuple[str, str]],
) -> None:
    """把当前累计区间封存入 regions（无内容则不产块），并清空累计。"""
    if content or code or images:
        regions.append(
            _Region(
                chain=tuple(chain),
                content=tuple(content),
                code=tuple(code),
                images=tuple(images),
            )
        )
    content.clear()
    code.clear()
    images.clear()


def _emit_heading_region(region: _Region, source_file: str, max_chars: int, chunks: list[Chunk]) -> None:
    """标题区间：内容合并成一块（超长逐级拆），代码块独立成块，seq 递增。"""
    prefix = _block_id_prefix(region.chain)
    heading_path = " > ".join(text for _, text, _ in region.chain)
    anchor = region.chain[-1][2]
    seq = 0
    for part in _split_content_blocks(region.content, max_chars):
        seq += 1
        chunks.append(Chunk(f"{prefix}#{seq}", part, source_file, heading_path, anchor, seq))
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
    for text in region.content:
        for part in _split_sentence_text(text, max_chars):
            seq += 1
            chunks.append(Chunk(f"#{seq}", part, source_file, "", "", seq))
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


def _split_content_blocks(blocks: tuple[str, ...], max_chars: int) -> list[str]:
    """标题区间内容按段落边界拆成 ≤ 上限的子块；单段仍超长则逐级再拆。"""
    if not blocks:
        return []
    if len("\n\n".join(blocks)) <= max_chars:
        return ["\n\n".join(blocks)]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for text in blocks:
        if not current:
            current = [text]
            current_len = len(text)
        elif current_len + 2 + len(text) <= max_chars:
            current.append(text)
            current_len += 2 + len(text)
        else:
            parts.append("\n\n".join(current))
            current = [text]
            current_len = len(text)
    if current:
        parts.append("\n\n".join(current))
    result: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            result.append(part)
        else:
            result.extend(_split_sentence_text(part, max_chars))
    return result


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
    """按字符边界拆（允许）：最终兜底，保证每块 ≤ 上限。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _strip_inline_html(text: str) -> str:
    """剥离内联 HTML 标签（<font> 等）与残留加粗/斜体标记；内容块专用，代码块不调用。"""
    text = _INLINE_HTML_RE.sub("", text)
    return _BOLD_MARKER_RE.sub("", text)


def _coalesce_short_chunks(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """把低于最小长度的相邻文本块前向合并，使每块达到语义完整长度。

    - 低于 ``min_chars`` 的碎片被并入后续块（直到 ≥min 或触及上限）；已达标且
      后续块也达标的相邻块保持独立，不过度粘连。
    - 前向合并不掉的短碎片（位于文档末尾、或后面紧跟代码块）会**向后并入前一
      个文本块**，避免「几个字 / 一行代码」的孤立碎片块污染检索。
    - 代码块恒独立，不参与合并、也不被短正文并入（代码与正文分离）。
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
        # 短碎片向后并入前一个块（文本/代码块，若未超上限），消除孤立碎片；
        # 但不并入图片块（图片 alt 描述应独立，避免混入正文）；合并后按正文处理。
        if pending_len < min_chars and out and out[-1].kind != "image":
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
        if chunk.kind == "code":
            _flush()
            out.append(chunk)
            continue
        if chunk.kind == "image":
            # 图片不打断正文累积：步骤式文档里截图夹在短句之间，若在此 flush 会把
            # 每步短句冲成孤立碎块；图片独立成块即可，正文继续累积跨图合并。
            out.append(chunk)
            continue
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
