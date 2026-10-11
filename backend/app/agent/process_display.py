from collections.abc import Mapping
from typing import Any

from app.agent.evidence_text import evidence_body


def tool_start_summary(name: str, arguments: Mapping[str, Any]) -> str:
    query = str(arguments.get("query") or "").strip()
    heading = str(arguments.get("heading") or "").strip()
    if name == "read_document":
        return f"正在读取章节：{heading}" if heading else "正在读取文档正文"
    if name == "read_web_page":
        return "正在读取网页正文"
    if query:
        action = "搜索网页" if name == "web_search" else "搜索资料"
        return f"{action}：{query}"
    return ""


def _web_budget_details(data: Mapping[str, Any]) -> list[str]:
    """联网额度的一行摘要（思考等级 / 权限 / 已用额度 / 阶段）：进过程记录。

    ``tool.completed.details`` 是契约里的 ``list[str]``，因此额度状态不需要新增事件字段
    就能被如实记录与展示。
    """

    snapshot = data.get("web_budget")
    if not isinstance(snapshot, Mapping):
        return []
    if not snapshot.get("allowed"):
        reason = str(snapshot.get("reason") or "本轮不联网")
        return [f"联网额度：不可用（{reason}）"]
    lines = [
        "联网额度：查询 {queries_used}/{queries_limit}、网页 {pages_used}/{pages_limit}、"
        "用时 {elapsed_seconds:.0f}/{seconds_limit:.0f}s（思考等级 {reasoning_level}）".format(
            queries_used=snapshot.get("queries_used", 0),
            queries_limit=snapshot.get("queries_limit", 0),
            pages_used=snapshot.get("pages_used", 0),
            pages_limit=snapshot.get("pages_limit", 0),
            elapsed_seconds=float(snapshot.get("elapsed_seconds") or 0.0),
            seconds_limit=float(snapshot.get("seconds_limit") or 0.0),
            reasoning_level=snapshot.get("reasoning_level", ""),
        )
    ]
    note = str(data.get("budget_note") or "")
    if note:
        lines.append(note)
    return lines


def tool_result_display(name: str, result: Any) -> dict[str, Any]:
    data = result.data
    status = str(result.status)
    budget_details = _web_budget_details(data)
    if status not in {"ok", "partial"}:
        payload: dict[str, Any] = {
            "summary": result.message or {
                "empty": "本次查询没有找到资料",
                "not_found": "没有找到指定资料",
                "irrelevant": "本次候选资料与问题不够相关",
            }.get(status, "本次操作未返回可用内容")
        }
        if budget_details:
            payload["details"] = budget_details
        return payload

    if name == "read_document":
        text = str(data.get("text") or "")
        title = str(data.get("document_name") or "文档正文")
        headings = data.get("headings") or []
        details = [f"章节：{heading}" for heading in headings]
        if data.get("next_offset") is not None:
            details.append("当前仅展示本次读取的内容，文档仍有后续正文。")
        return {
            "summary": f"已读取《{title}》，共 {len(text)} 个字符" if text else "本次读取没有返回正文",
            "details": details,
            "sources": [{"title": title, "text": text}] if text else [],
        }

    sources = []
    for key in ("chunks", "pages", "hits", "results", "messages"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("document_name") or item.get("title") or
                        item.get("conversation_title") or item.get("url") or "相关资料")
            heading = str(item.get("heading_path") or "")
            if heading:
                title = f"{title} · {heading}"
            text = str(item.get("text") or item.get("content") or
                       item.get("snippet") or item.get("excerpt") or "")
            sources.append({"title": title, "text": evidence_body(text)})
        break
    if sources:
        action = "已读取" if name.startswith("read_") else "找到"
        details = [result.message] if result.message else []
        timings = data.get("timings")
        if isinstance(timings, Mapping):
            for key, label in (
                ("embedding_seconds", "查询嵌入"),
                ("local_retrieval_seconds", "本地检索"),
                ("rerank_seconds", "相关性筛选"),
            ):
                seconds = timings.get(key)
                if isinstance(seconds, (int, float)):
                    details.append(f"{label}耗时 {seconds:.2f} 秒")
        return {
            "summary": f"{action} {len(sources)} 条资料",
            "sources": sources,
            "details": details + budget_details,
        }
    fallback: dict[str, Any] = {"summary": result.message or "操作已完成"}
    if budget_details:
        fallback["details"] = budget_details
    return fallback
