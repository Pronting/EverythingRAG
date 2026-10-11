"""证据文本的展示形态：把「标题：… / 内容：…」包装拆开，供引用面板、结论与提示词共用。

检索层（``retrieval/hybrid_retriever.py``）会给部分片段拼接包装文本：

* 短片段补上下文：``标题：<heading_path>\\n内容：<短正文>\\n\\n同章节上下文：…``
* 同标题补充信息 / 图片证据：``标题：…\\n内容：…\\n\\n同标题补充信息：…``

V1.2.0 的引用处理直接对这段拼好的文本「取首句」，于是把完整证据截成了**只剩标题**，
正文里也把「标题：/内容：」这类内部形状暴露给模型，模型再原样抄进答案。本模块提供
唯一的拆解实现（**只有一处**，避免每个调用点各拆一遍、口径分叉）：

* :func:`split_evidence_content` → ``(章节标题, 正文)``；
* :func:`evidence_body` → 只取正文（引用面板的「证据正文」）；
* :func:`evidence_display_text` → 给模型看的「章节标题：正文」（不含任何内部标识）。

没有该包装时标题为空串、正文原样返回 —— **绝不猜标题**。
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "evidence_body",
    "evidence_display_text",
    "evidence_location_text",
    "split_evidence_content",
]

#: 检索层拼的包装：``标题：<heading_path>\n内容：<正文>``（正文可含补充段）。
_WRAPPER: Final[re.Pattern[str]] = re.compile(
    r"^标题：(?P<heading>[^\n]*)\n+内容：(?P<body>.*)$", re.DOTALL
)


#: 原文里的噪声串（OCR 残留 / 误打的 Markdown 分隔符）：证据正文里出现时会被模型原样
#: 抄进答案（实测 T12 出现过「研究过的软件组件==[S2]」）。这里只在**展示与引用**时
#: 去掉连续 2 个以上的 ``=`` / ``*`` / ``-`` / ``_`` / ``~``，不改动原文事实与其它字符。
_NOISE_RUN: Final[re.Pattern[str]] = re.compile(r"[=*_~]{2,}|-{4,}")


def split_evidence_content(content: str) -> tuple[str, str]:
    """把证据内容拆成 ``(章节标题, 正文)``；没有包装时标题为 ``""``。"""

    text = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    match = _WRAPPER.match(text)
    if match is None:
        return "", text
    heading = " ".join(match.group("heading").split())
    body = match.group("body").strip()
    return heading, body or text


def evidence_body(content: str) -> str:
    """证据正文（只去掉「标题：/内容：」包装标签；补充段内容一并保留）。"""

    _heading, body = split_evidence_content(content)
    return body


def evidence_display_text(content: str, *, document_title: str = "") -> str:
    """给模型看的证据文本：``《文档标题》 章节标题：正文``（**不含**内部定位标识）。

    ``document_title`` 来自 :attr:`app.agent.contracts.Evidence.document_title`。
    带上它之后，模型能够区分「这份资料属于哪篇文章的哪一节」，而不是只拿到一个章节名 ——
    只给章节名时，跨文档比较会把 A 文章的章节当成 B 文章的观点（用户实测缺陷 F13）。
    没有标题时行为与引入该参数前逐字一致（不猜标题）。
    """

    heading, body = split_evidence_content(content)
    body_text = body or heading
    return evidence_location_text(
        document_title=document_title, heading_path=heading, body=body_text
    )


def evidence_location_text(
    *, document_title: str = "", heading_path: str = "", body: str = ""
) -> str:
    """把「文档标题 + 章节位置 + 正文」拼成一行可核对文本（唯一实现，避免口径分叉）。

    - 只有标题：返回 ``《标题》``；
    - 标题 + 章节：``《标题》 章节：正文``；
    - 只有章节：``章节：正文``（与引入文档标题前的行为一致）。
    """

    title = " ".join(str(document_title or "").split()).strip()
    heading = " ".join(str(heading_path or "").split()).strip()
    text = _NOISE_RUN.sub(" ", str(body or "")).strip()
    heading = _NOISE_RUN.sub(" ", heading).strip()
    prefix = f"《{title}》" if title else ""
    location = heading or ""
    if location and text:
        located = f"{location}：{text}"
    else:
        located = text or location
    if prefix and located:
        return f"{prefix} {located}"
    return prefix or located
