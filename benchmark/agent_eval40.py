from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "completed_with_gaps", "failed", "cancelled", "waiting_approval", "waiting_input", "interrupted"}


def request(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def summarize(events: list[dict]) -> dict:
    deltas = {event["payload"]["index"]: event["payload"]["text"] for event in events if event["type"] == "answer.delta"}
    metadata = [event["payload"] for event in events if event["type"] == "answer.meta"]
    calls = {}
    tools = []
    for event in events:
        payload = event.get("payload", {})
        call = payload.get("model_call")
        if call:
            calls[call["call_id"]] = call
        if event["type"].startswith("tool."):
            tools.append(event)
    return {"answer": "".join(deltas[index] for index in sorted(deltas)),
            "citations": metadata[-1].get("citations", []) if metadata else [],
            "model_calls": list(calls.values()), "tool_events": tools}


def run_case(case: dict, args: argparse.Namespace, conversation_id: str | None) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    observer_path = ROOT / ".tmp_probe" / "agent-eval40-20260930" / "observe_ui.cjs"
    if not args.no_ui and not observer_path.is_file():
        raise FileNotFoundError(observer_path)
    started = time.perf_counter()
    if conversation_id is None:
        conversation_id = request(base, "POST", f"/api/conversations?workspace={args.workspace}")["id"]
    run = request(base, "POST", f"/api/workspaces/{args.workspace}/runs", {
        "conversation_id": conversation_id, "message": case["question"],
        "client_request_id": "eval40-" + case["id"] + "-" + uuid.uuid4().hex,
    })
    result = {"case": case, "case_id": case["id"], "port": args.port,
              "conversation_id": conversation_id, "run_id": run["run_id"],
              "run": run, "status": run["status"], "events": [],
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "manual_review_required": True, "harness_timeout": False}
    output = args.output_dir / f"{case['id']}.json"
    save(output, result)
    save(args.output_dir / "progress.json", result)
    print(json.dumps({key: result[key] for key in ("case_id", "port", "conversation_id", "run_id")}), flush=True)
    observer = None
    observer_log = None
    if not args.no_ui:
        observer_log = (args.output_dir / f"{case['id']}-ui.log").open("wb")
        observer = subprocess.Popen([
            "node", str(observer_path), "--port", str(args.port), "--conversation", conversation_id,
            "--run", run["run_id"], "--output", str(args.output_dir / f"{case['id']}-ui.json"),
            "--timeout", str(args.timeout + 60),
        ], cwd=ROOT, stdout=observer_log, stderr=observer_log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    previous_phase = None
    try:
        while True:
            run = request(base, "GET", f"/api/workspaces/{args.workspace}/runs/{run['run_id']}")
            events = request(base, "GET", f"/api/workspaces/{args.workspace}/runs/{run['run_id']}/event-log")["events"]
            result.update({"run": run, "status": run["status"], "events": events,
                           "elapsed_seconds": round(time.perf_counter() - started, 3), **summarize(events)})
            save(output, result)
            phase = (run["status"], len(result["model_calls"]), len(result["tool_events"]))
            progress = {key: value for key, value in result.items() if key not in {"events", "case", "answer", "citations", "tool_events"}}
            progress["event_count"] = len(events)
            save(args.output_dir / "progress.json", progress)
            if phase != previous_phase:
                print(json.dumps({"id": case["id"], "status": run["status"],
                                  "elapsed": result["elapsed_seconds"], "events": len(events),
                                  "model_stages": [call.get("stage") for call in result["model_calls"]],
                                  "tool_events": len(result["tool_events"])}, ensure_ascii=False), flush=True)
                previous_phase = phase
            if run["status"] in TERMINAL:
                break
            if result["elapsed_seconds"] >= args.timeout:
                request(base, "POST", f"/api/workspaces/{args.workspace}/runs/{run['run_id']}/cancel")
                result["harness_timeout"] = True
            time.sleep(5)
        if observer is not None:
            observer.wait(timeout=60)
            result["ui_process_exit_code"] = observer.returncode
    finally:
        if observer is not None and observer.poll() is None:
            observer.terminate()
            observer.wait(timeout=15)
            result["ui_process_terminated"] = True
        if observer_log is not None:
            observer_log.close()
        save(output, result)
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="独立应用实例的真实 Agent 问答与页面观察")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", default=900, type=float)
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("题目 ID 重复")
    selected = [case for case in cases if not args.ids or case["id"] in args.ids]
    if args.ids and len(selected) != len(set(args.ids)):
        raise ValueError("部分题目 ID 不存在")
    for group in {case.get("conversation_group") for case in selected} - {None}:
        original_group = [case for case in cases if case.get("conversation_group") == group]
        selected_group = [case for case in selected if case.get("conversation_group") == group]
        if len(selected_group) != len(original_group):
            raise ValueError("同一会话的问题必须分配给同一个执行器")
        if [case["order"] for case in selected_group] != sorted(case["order"] for case in selected_group):
            raise ValueError("同一会话的问题顺序不正确")
    if not selected:
        raise ValueError("没有选中题目")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conversations = {}
    for case in selected:
        group = case.get("conversation_group")
        output = args.output_dir / f"{case['id']}.json"
        if output.exists():
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous["status"] not in TERMINAL:
                raise RuntimeError(f"已有题目仍在执行，需要检查其进程：{case['id']}")
            if group:
                conversations[group] = previous["conversation_id"]
            continue
        result = run_case(case, args, conversations.get(group) if group else None)
        if group:
            conversations[group] = result["conversation_id"]
        print(json.dumps({"finished": case["id"], "status": result["status"], "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
