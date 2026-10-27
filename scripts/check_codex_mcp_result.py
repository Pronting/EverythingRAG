import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("answer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    calls = [event["item"] for event in events if event.get("type") == "item.completed"
             and event.get("item", {}).get("type") == "mcp_tool_call"
             and event["item"].get("server") == "everything_rag"]
    assert calls and all(call["status"] == "completed" and not call.get("error") for call in calls)
    names = {call["tool"] for call in calls}
    assert {"get_knowledge_overview", "lookup_catalog", "search_knowledge", "read_document"} <= names
    read = next(call["result"]["structured_content"] for call in reversed(calls)
                if call["tool"] == "read_document")
    answer = args.answer.read_text(encoding="utf-8-sig")
    assert read["text"] and read["document_name"] in answer and read["document_id"] in answer
    assert any(span["heading_path"] in answer for span in read["blocks"] if span["heading_path"])
    assert any(event.get("type") == "turn.completed" for event in events)
    report = {"verified": True, "tools": sorted(names), "successful_calls": len(calls),
              "failed_calls": 0, "citation_checked": True}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
