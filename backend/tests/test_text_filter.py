"""PhraseFilter 单测：流式剔除「割裂」措辞（含跨 token 匹配 + flush 余量）。"""

from __future__ import annotations

from app.generation.text_filter import BANNED_PHRASES, PhraseFilter


def test_strips_exact_phrase() -> None:
    """整段命中即剔除。"""
    f = PhraseFilter()
    assert f.feed("以下为通用知识，非你的资料库内容") == ""
    assert f.flush() == ""


def test_strips_phrase_across_tokens() -> None:
    """短语被切在多个 token 之间也能被完整剔除（流式缓冲匹配）。"""
    f = PhraseFilter()
    text = "".join(
        [
            f.feed("以下为通用"),
            f.feed("知识，非你的资料库内容"),
            f.feed("：Everything RAG 是一款本地知识助手。"),
            f.flush(),
        ]
    )
    assert "非你的资料库内容" not in text
    assert "以下为通用知识" not in text
    assert "Everything RAG" in text


def test_keeps_normal_text_unchanged() -> None:
    """不含 banned 短语的正文原样透传。"""
    f = PhraseFilter()
    text = f.feed("Everything RAG 是一款运行在本地的个人知识助手。") + f.flush()
    assert text == "Everything RAG 是一款运行在本地的个人知识助手。"


def test_strips_marker_without_trailing_punctuation() -> None:
    """「没有直接关联」「无法提取有效信息」等标记词被剔除，正文保留。"""
    f = PhraseFilter()
    text = "".join(
        [f.feed("两者没有直接关联，无法提取有效信息。"), f.feed("下面正式回答。"), f.flush()]
    )
    assert "没有直接关联" not in text
    assert "无法提取有效信息" not in text
    assert "下面正式回答" in text


def test_banned_phrases_sorted_by_length_desc() -> None:
    """短语按长度降序，保证长短语先匹配、前缀子串不误删。"""
    assert BANNED_PHRASES == sorted(BANNED_PHRASES, key=len, reverse=True)
