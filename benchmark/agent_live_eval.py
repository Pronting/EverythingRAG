from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import statistics
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import xxhash


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".tmp_probe" / "agent-live-eval-20260929"
CASES_PATH = PRIVATE / "cases.json"
RESULTS = PRIVATE / "results"
TERMINAL = {"completed", "completed_with_gaps", "failed", "cancelled", "waiting_approval", "waiting_input"}
LABEL_PATTERN = re.compile(r"\[([SHMW]\d+)\]")


def load_cases() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def compact(value: str) -> str:
    return "".join(value.lower().split())


def source_locations(catalog: dict, data_dir: Path) -> dict[str, dict]:
    state_db = data_dir / "state.db"
    app_db = data_dir / "app.db"
    if not state_db.is_file() or not app_db.is_file():
        raise FileNotFoundError("本地真实索引或会话数据库不存在")
    locations = {}
    with sqlite3.connect(state_db) as index, sqlite3.connect(app_db) as app:
        for source_id, source in catalog["sources"].items():
            quote = source["quote"]
            if source["kind"] == "document":
                document = data_dir / "documents" / source["file"]
                content = document.read_text(encoding="utf-8")
                if quote not in content:
                    raise ValueError(f"文档短引与原文不符：{source_id}")
                record = index.execute(
                    "SELECT chunk_count,content_hash FROM documents WHERE path = ?", (str(document),)
                ).fetchone()
                if record is None or record[0] < 1:
                    raise ValueError(f"来源尚未进入真实索引：{source_id}")
                if record[1] != xxhash.xxh64(document.read_bytes()).hexdigest():
                    raise ValueError(f"来源文档与索引版本不一致：{source_id}")
                line = content[: content.index(quote)].count("\n") + 1
                locations[source_id] = {
                    "kind": "document", "path": str(document), "line": line,
                    "indexed_chunks": record[0], "quote": quote,
                }
            elif source["kind"] == "history":
                row = app.execute(
                    "SELECT m.content,c.workspace_id FROM messages m "
                    "JOIN conversations c ON c.id=m.conversation_id "
                    "WHERE m.id=? AND m.conversation_id=? AND m.role='user'",
                    (source["message_id"], source["conversation_id"]),
                ).fetchone()
                if row is None or quote not in row[0] or row[1] != catalog["workspace_id"]:
                    raise ValueError(f"历史消息与真实会话不符：{source_id}")
                locations[source_id] = {
                    "kind": "history", "conversation_id": source["conversation_id"],
                    "message_id": source["message_id"], "quote": quote,
                }
            else:
                raise ValueError(f"未知来源类别：{source_id}")
        for case in catalog["cases"]:
            if case["category"] == "insufficient":
                missing = data_dir / "documents" / case["missing_file"]
                if missing.exists() or index.execute(
                    "SELECT 1 FROM documents WHERE path LIKE ? LIMIT 1",
                    ("%" + case["missing_file"],),
                ).fetchone():
                    raise ValueError(f"资料不足题的目标文档实际存在：{case['id']}")
    return locations


def validate(catalog: dict, data_dir: Path) -> dict[str, dict]:
    cases = catalog["cases"]
    identifiers = [case["id"] for case in cases]
    if len(cases) < 50 or len(identifiers) != len(set(identifiers)):
        raise ValueError("题目必须至少 50 道，且 ID 唯一")
    if set(collections.Counter(case["category"] for case in cases)) != {
        "fact", "cross", "history", "insufficient"
    }:
        raise ValueError("四类题目均须存在")
    for case in cases:
        if not case["question"] or not case["facts"]:
            raise ValueError(f"题目或关键事实为空：{case['id']}")
        if not all(group and all(term for term in group) for group in case["facts"]):
            raise ValueError(f"关键事实规则无效：{case['id']}")
        if not all(source_id in catalog["sources"] for source_id in case["sources"]):
            raise ValueError(f"来源 ID 不存在：{case['id']}")
        source_kinds = {catalog["sources"][key]["kind"] for key in case["sources"]}
        if case["category"] == "history" and source_kinds != {"history"}:
            raise ValueError(f"历史题来源类别错误：{case['id']}")
        if case["category"] in {"fact", "cross"} and source_kinds != {"document"}:
            raise ValueError(f"文档题来源类别错误：{case['id']}")
        if case["category"] == "cross" and len(case["sources"]) < 2:
            raise ValueError(f"跨文档题来源数量不足：{case['id']}")
        if case["category"] == "insufficient" and case["sources"]:
            raise ValueError(f"资料不足题含有效来源：{case['id']}")
    return source_locations(catalog, data_dir)


def request_json(base_url: str, method: str, path: str, body: dict | None = None) -> dict:
    url = base_url.rstrip("/") + path
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = Request(
        url, data=encoded, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def evaluate(case: dict, catalog: dict, answer: str, citations: list[dict], events: list[dict], status: str) -> dict:
    normalized = compact(answer)
    facts = [any(compact(term) in normalized for term in alternatives) for alternatives in case["facts"]]
    labels = set(LABEL_PATTERN.findall(answer))
    available_labels = {entry["label"] for entry in citations}
    citation_labels_valid = labels <= available_labels
    citation_targets = {}
    citation_quote_visible = {}
    for source_id in case["sources"]:
        source = catalog["sources"][source_id]
        if source["kind"] == "document":
            citation_targets[source_id] = any(
                entry.get("document_title") == source["file"] and entry.get("label") in labels
                for entry in citations
            )
        else:
            citation_targets[source_id] = any(
                entry.get("source_type") == "history" and entry.get("label") in labels and (
                    source["message_id"] in entry.get("source_ref", "")
                    or source["conversation_id"] in entry.get("source_ref", "")
                    or source["quote"] in entry.get("snippet", "")
                ) for entry in citations
            )
        citation_quote_visible[source_id] = any(
            entry.get("label") in labels
            and compact(source["quote"]) in compact(entry.get("snippet", ""))
            for entry in citations
        )
    tools = [
        event.get("payload", {}).get("tool_name", "")
        for event in events if event["type"] == "tool.started"
    ]
    if case["category"] == "history":
        required_tools = {"search_history"}
        forbidden_tools = {"search_knowledge", "read_document", "web_search", "search_web", "read_web", "web_read"}
    elif case["category"] == "insufficient":
        required_tools = set()
        forbidden_tools = {"search_history", "read_history", "web_search", "search_web", "read_web", "web_read"}
    else:
        required_tools = {"search_knowledge"}
        forbidden_tools = {"search_history", "read_history", "web_search", "search_web", "read_web", "web_read"}
    tool_scope_valid = required_tools.issubset(tools) and not forbidden_tools.intersection(tools)
    if case["category"] == "insufficient":
        source_valid = not any(entry.get("source_type") == "knowledge" for entry in citations)
    else:
        source_valid = all(citation_targets.values())
    return {
        "facts_present": facts,
        "citation_labels_valid": citation_labels_valid,
        "target_citations_present": citation_targets,
        "source_quote_visible_in_citation": citation_quote_visible,
        "source_scope_valid": source_valid,
        "tool_scope_valid": tool_scope_valid,
        "tools": tools,
        "machine_checks_pass": status in {"completed", "completed_with_gaps"}
        and all(facts) and citation_labels_valid and source_valid and tool_scope_valid,
        "manual_fact_and_support_review_required": True,
    }


def run_case(case: dict, catalog: dict, locations: dict, base_url: str, timeout: float) -> dict:
    workspace = catalog["workspace_id"]
    started = time.perf_counter()
    conversation = request_json(base_url, "POST", f"/api/conversations?workspace={workspace}")
    run = request_json(base_url, "POST", f"/api/workspaces/{workspace}/runs", {
        "conversation_id": conversation["id"],
        "message": case["question"],
        "client_request_id": "agent-live-eval-" + case["id"] + "-" + uuid.uuid4().hex,
        "permission_mode": "full", "reasoning_level": "off",
    })
    run_id = run["run_id"]
    while run["status"] not in TERMINAL:
        if time.perf_counter() - started >= timeout:
            request_json(base_url, "POST", f"/api/workspaces/{workspace}/runs/{run_id}/cancel")
            run = request_json(base_url, "GET", f"/api/workspaces/{workspace}/runs/{run_id}")
            break
        time.sleep(1)
        run = request_json(base_url, "GET", f"/api/workspaces/{workspace}/runs/{run_id}")
    events = request_json(base_url, "GET", f"/api/workspaces/{workspace}/runs/{run_id}/event-log")["events"]
    deltas = {
        event["payload"]["index"]: event["payload"]["text"]
        for event in events if event["type"] == "answer.delta"
    }
    answer = "".join(deltas[index] for index in sorted(deltas))
    metas = [event["payload"] for event in events if event["type"] == "answer.meta"]
    citations = metas[-1]["citations"] if metas else []
    assessment = evaluate(case, catalog, answer, citations, events, run["status"])
    return {
        "case_id": case["id"], "category": case["category"], "question": case["question"],
        "expected_facts": case["facts"],
        "source_locations": {source_id: locations[source_id] for source_id in case["sources"]},
        "conversation_id": conversation["id"], "run_id": run_id, "status": run["status"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "answer": answer, "citations": citations, "events": events,
        "assessment": assessment,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="当前 Agent 的真实来源答案评测")
    parser.add_argument("command", choices=("validate", "run", "report", "latency"))
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".everything-rag")
    parser.add_argument("--base-url", default="http://127.0.0.1:9999")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--result-set", default="baseline")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.result_set):
        raise ValueError("结果组名称只能包含英文字母、数字、连字符和下划线")
    results_dir = RESULTS if args.result_set == "baseline" else PRIVATE / f"results-{args.result_set}"
    catalog = load_cases()
    locations = validate(catalog, args.data_dir)
    counts = dict(collections.Counter(case["category"] for case in catalog["cases"]))
    if args.command == "validate":
        print(json.dumps({"cases": len(catalog["cases"]), "categories": counts,
                          "sources": len(locations)}, ensure_ascii=False))
        return
    if args.command == "report":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))]
        by_category = {}
        for category in counts:
            rows = [row for row in results if row["category"] == category]
            by_category[category] = {"run": len(rows), "machine_checks_pass": sum(
                row["assessment"]["machine_checks_pass"] for row in rows
            )}
        print(json.dumps({"planned": len(catalog["cases"]), "results": len(results),
                          "by_category": by_category,
                          "manual_fact_and_support_review_required": len(results)}, ensure_ascii=False, indent=2))
        return
    if args.command == "latency":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))]
        calls = {}
        for result in results:
            for event in result["events"]:
                model_call = event.get("payload", {}).get("model_call")
                if model_call and model_call["outcome"] == "completed":
                    calls[model_call["call_id"]] = model_call
        by_stage = {}
        for stage in sorted({call["stage"] for call in calls.values()}):
            seconds = sorted(call["total_seconds"] for call in calls.values() if call["stage"] == stage)
            by_stage[stage] = {
                "calls": len(seconds), "total_seconds": round(sum(seconds), 3),
                "mean_seconds": round(sum(seconds) / len(seconds), 3),
                "median_seconds": round(statistics.median(seconds), 3),
                "p95_seconds": round(seconds[(95 * len(seconds) + 99) // 100 - 1], 3),
            }
        wall = sorted(result["elapsed_seconds"] for result in results)
        print(json.dumps({
            "results": len(results), "model_calls": len(calls), "by_stage": by_stage,
            "wall_seconds": {"total": round(sum(wall), 2),
                             "median": round(statistics.median(wall), 2) if wall else None,
                             "p95": wall[(95 * len(wall) + 99) // 100 - 1] if wall else None},
            "reasoning_observed_while_off_calls": sum(
                call["reasoning_observed_while_off"] for call in calls.values()
            ),
        }, ensure_ascii=False, indent=2))
        return
    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("评测服务地址必须是本机 HTTP 地址")
    selected = [case for case in catalog["cases"] if not args.ids or case["id"] in args.ids]
    if args.ids and len(selected) != len(set(args.ids)):
        raise ValueError("指定的题目 ID 不完整或重复")
    if args.start < 1:
        raise ValueError("--start 必须大于零")
    selected = selected[args.start - 1:]
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.reverse:
        selected.reverse()
    results_dir.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(selected, start=1):
        path = results_dir / f"{case['id']}.json"
        if path.exists() and not args.rerun:
            print(f"{number}/{len(selected)} {case['id']} 已有结果")
            continue
        result = run_case(case, catalog, locations, args.base_url, args.timeout)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{number}/{len(selected)} {case['id']} {result['status']} "
              f"{result['elapsed_seconds']}s machine={result['assessment']['machine_checks_pass']}", flush=True)


if __name__ == "__main__":
    main()
