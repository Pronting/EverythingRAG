"""流式正文过滤器：硬性剔除「割裂」措辞（代码级约束，不依赖模型自觉）。

当知识库无相关语料时，模型可能（尤其在使用旧版系统提示词时）输出
「通用知识」「未检索到」「没有直接关联」等过程性、割裂性措辞，影响回答的
自然度与用户信任。本过滤器在流式 token 层剔除这些短语。

实现为流式缓冲匹配：feed 喂增量文本，flush 取出尾部余量；短语可跨 token 匹配。
只缓冲「可能是某 banned 短语前缀」的尾部字符，正常正文即时透传，几乎不破坏
流式节奏。
"""

from __future__ import annotations

#: 需剔除的「割裂」措辞（按长度降序，避免前缀子串被误删）。
BANNED_PHRASES: list[str] = sorted(
    [
        "以下为通用知识，非你的资料库内容",
        "以下为通用知识，非你资料库内容",
        "非你的资料库内容",
        "非你资料库内容",
        "以下为通用知识",
        "知识库中未检索到",
        "知识库中未检索",
        "未检索到相关",
        "没有直接关联",
        "无法提取有效信息",
        "无法从这些片段中提取",
    ],
    key=len,
    reverse=True,
)


class PhraseFilter:
    """流式剔除 banned 短语。

    - ``feed(text)``：喂入增量文本，返回可安全透传的净文本（可能为空）。
    - ``flush()``：流结束时取出缓冲余量（已剔除短语）。
    """

    def __init__(self, phrases: list[str] | None = None) -> None:
        self._phrases = phrases if phrases is not None else BANNED_PHRASES
        self._buffer = ""

    def feed(self, text: str) -> str:
        """喂入增量文本，返回可安全透传的净文本。"""
        if not text:
            return ""
        self._buffer += text
        self._strip_phrases()
        pending = self._pending_len()
        if len(self._buffer) <= pending:
            return ""
        out = self._buffer[: len(self._buffer) - pending]
        self._buffer = self._buffer[len(self._buffer) - pending :]
        return out

    def flush(self) -> str:
        """流结束：取出缓冲余量（已剔除短语）。"""
        self._strip_phrases()
        out = self._buffer
        self._buffer = ""
        return out

    def _strip_phrases(self) -> None:
        """反复替换直至稳定，处理「删除后相邻拼接又成短语」的罕见情况。"""
        changed = True
        while changed:
            changed = False
            for phrase in self._phrases:
                if phrase in self._buffer:
                    self._buffer = self._buffer.replace(phrase, "")
                    changed = True

    def _pending_len(self) -> int:
        """返回 buffer 末尾「可能是某 banned 短语前缀」的最长后缀长度。

        这些字符可能是短语的开头、其后续字符将在未来 token 到达，故暂缓透传；
        其余字符安全，立即透传，保持流式节奏。
        """
        pending = 0
        for phrase in self._phrases:
            max_k = min(len(self._buffer), len(phrase))
            for k in range(1, max_k + 1):
                if phrase.startswith(self._buffer[-k:]):
                    pending = max(pending, k)
        return pending
