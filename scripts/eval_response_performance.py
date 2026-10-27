import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

CASES = [
    {"id": "definition", "level": "off", "question": "什么是银弹？"},
    {"id": "http", "level": "off", "question": "HTTP 请求 429 是什么意思？"},
    {"id": "recall", "level": "off", "question": "仅依据本地资料，货拉拉记忆系统的三路召回分别是什么？它们怎样配合？"},
    {"id": "layers", "level": "off", "question": "仅依据本地资料，TencentDB Agent Memory 的 L0 到 L3 四层分别保存什么？请逐层说明并给出引用。"},
    {"id": "comparison", "level": "low", "question": "仅依据本地资料，比较货拉拉、TencentDB Agent Memory 和 OpenClaw 的记忆设计，说明各自如何存储、召回与更新记忆，以及设计的适用条件。"},
    {"id": "numbers", "level": "low", "question": "仅依据本地资料，OpenClaw Dreaming 的 Deep 阶段六维打分中权重最高的维度是哪个？权重是多少？除了分数还要满足哪三重门槛？"},
    {"id": "absence", "level": "off", "question": "知识库文档里有没有 bge-m3 在 CPU 上的实测首 token 延迟数据？具体是多少？"},
    {"id": "followup", "level": "off", "turns": [
        "仅依据本地资料，货拉拉的三路召回分别承担什么职责？",
        "那如果只用向量召回，会漏掉哪类信息？请继续依据刚才的资料说明。",
    ]},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cases", nargs="*")
    return parser.parse_args()


def prepare_data(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["chroma.sqlite3", "state.db", "image_state.db"]:
        original = sqlite3.connect(f"{(source / name).as_uri()}?mode=ro", uri=True)
        copied = sqlite3.connect(destination / name)
        original.backup(copied)
        copied.close()
        original.close()
    shutil.copy2(source / "config.json", destination / "config.json")
    connection = sqlite3.connect(f"{(source / 'chroma.sqlite3').as_uri()}?mode=ro", uri=True)
    for segment, in connection.execute("SELECT id FROM segments WHERE scope='VECTOR'"):
        if (source / segment).is_dir():
            shutil.copytree(source / segment, destination / segment)
    connection.close()


def seconds_between(start, end):
    parse = lambda value: datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return round((parse(end) - parse(start)).total_seconds(), 3)


async def evaluate(args, data):
    from app.agent.app_store import AppStore
    from app.api import deps
    from app.api.run_services import build_run_manager

    store = await AppStore.open(data / "app.db")
    deps.register_app_store(store)
    manager = build_run_manager(store)
    results = []
    try:
        await store.create_workspace("default", "default")
        selected = [case for case in CASES if not args.cases or case["id"] in args.cases]
        for repeat in range(args.rounds):
            ordered = selected if repeat % 2 == 0 else list(reversed(selected))
            for case in ordered:
                conversation = f"performance-{uuid4().hex}"
                await store.create_conversation(conversation, "default")
                for turn, question in enumerate(case.get("turns", [case.get("question")]), 1):
                    started = time.perf_counter()
                    run = await manager.create_run(
                        workspace_id="default", conversation_id=conversation,
                        question=question, reasoning_level=case["level"], permission_mode="auto",
                    )
                    settled = await manager.run_to_completion(run.id, timeout=600)
                    elapsed = time.perf_counter() - started
                    events = await manager.read_events(run.id)
                    answer_event = next((event for event in events if event.type == "answer.meta"), None)
                    first = next((event for event in events if event.type == "answer.delta"), None)
                    messages = await store.list_messages(conversation)
                    answer = next((message for message in reversed(messages) if message.run_id == run.id and message.role == "assistant"), None)
                    calls = {}
                    for event in events:
                        record = event.payload.get("model_call")
                        if record:
                            calls[record["call_id"]] = record
                    record = {
                        "case": case["id"], "repeat": repeat + 1, "turn": turn,
                        "question": question, "level": case["level"],
                        "seconds": round(elapsed, 3), "status": settled.status.value,
                        "first_answer_seconds": seconds_between(run.created_at, first.created_at) if first else None,
                        "answer": answer.content if answer else "",
                        "source_snapshot": answer.source_snapshot if answer else {},
                        "answer_meta": answer_event.payload if answer_event else None,
                        "model_calls": list(calls.values()),
                        "errors": [event.payload for event in events if event.type == "run.failed"],
                        "state": manager._states.get(run.id, {}),
                        "events": [event.model_dump(mode="json") for event in events],
                    }
                    results.append(record)
                    (args.output_dir / "results.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
                    )
                    print(json.dumps({
                        key: record[key] for key in ["case", "repeat", "turn", "seconds", "first_answer_seconds", "status"]
                    }, ensure_ascii=False), flush=True)
    finally:
        await manager.aclose()
        await store.close()
        deps.register_app_store(None)


def main():
    args = parse_args()
    args.code_root = args.code_root.resolve()
    args.data_source = args.data_source.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = args.output_dir / uuid4().hex
    prepare_data(args.data_source, data)
    os.environ["EVERYTHING_RAG_DATA_DIR"] = str(data)
    sys.path.insert(0, str(args.code_root))
    logging.basicConfig(filename=args.output_dir / "execution.log", level=logging.WARNING, encoding="utf-8")
    fingerprint = hashlib.sha256()
    for path in sorted((args.code_root / "app").rglob("*.py")):
        fingerprint.update(str(path.relative_to(args.code_root)).encode())
        fingerprint.update(path.read_bytes())
    (args.output_dir / "environment.json").write_text(json.dumps({
        "code_sha256": fingerprint.hexdigest(),
        "process_id": os.getpid(),
        "config_sha256": hashlib.sha256((data / "config.json").read_bytes()).hexdigest(),
        "data_source": str(args.data_source), "data_copy": str(data),
        "rounds": args.rounds, "cases": args.cases,
    }, indent=2), encoding="utf-8")
    asyncio.run(evaluate(args, data))


if __name__ == "__main__":
    main()
