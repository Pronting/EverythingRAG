"""端到端答案准确度评测：走真实 ChatService 生产链路（检索→门控→提示词→生成）。

对每条用例调用 stream_answer，收集 meta（mode/basis/sources）与完整回答，
再按「证据命中 / 引用合规 / 来源正确 / 负样本诚实度」打分。
结果写入 benchmark/answer_results.json。
"""
from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:/Users/22331/Desktop/everything-rag/backend")
sys.path.insert(0, str(Path(__file__).resolve().parent))

socket.setdefaulttimeout(120)

import cases_loader

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.generation.chat_service import ChatService
from app.generation.providers import create_chat_model_from_config
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import DEFAULT_MIN_SIMILARITY
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

BASE = Path(r"C:\Users\22331\Desktop\everything-rag")
OUT = BASE / "benchmark"
LOG = BASE / ".tmp_probe/answer_progress.txt"
CONCURRENCY = 4
MAX_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        fh.flush()


def norm(s: str) -> str:
    return "".join(s.split()).lower()


cases = cases_loader.load_cases(BASE / "benchmark/cases.json")
samples = {s["n"]: s for s in cases_loader.load_sample_chunks()}

app_settings = get_settings_store().load()
embedder = create_embedder_from_config(app_settings.embed)
vs = create_vector_store(get_settings().data_dir, embedder=embedder)
retriever = HybridRetriever(vs, min_similarity=DEFAULT_MIN_SIMILARITY)
chat_model = create_chat_model_from_config(app_settings.chat)
service = ChatService(
    embedder=embedder,
    retriever=retriever,
    chat_model=chat_model,
    search_provider=None,
    search_max_results=0,
    vision=None,
    preferences=app_settings.answer_preferences,
    knowledge_overview=lambda: None,
)
log("ChatService 就绪")

CITE_RE = re.compile(r"\[(S\d+|W\d+)\]")


async def run_one(case: dict) -> dict:
    frames = []
    try:
        async for fr in service.stream_answer(case["query"], [], web_search=False, images=[]):
            frames.append(fr)
    except Exception as exc:  # noqa: BLE001
        return {"id": case["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    meta = next((f for f in frames if f.get("type") == "meta"), {})
    text = "".join(f.get("text", "") for f in frames if f.get("type") == "token")
    err = next((f for f in frames if f.get("type") == "error"), None)

    gold_files = {Path(samples[n]["file"]).name for n in case["gold"]}
    gold_heads = [samples[n]["heading"] for n in case["gold"]]
    sources = meta.get("sources", [])

    blob = norm(text)
    ev = case["evidence"]
    ev_hit = round(sum(1 for e in ev if norm(e) in blob) / len(ev), 3) if ev else None

    cited = set(CITE_RE.findall(text))
    allowed = {s.get("source_id", "") for s in sources}
    bad_cite = sorted(c for c in cited if c.startswith("S") and c not in allowed)

    src_files = {Path(s.get("source_file", "")).name for s in sources}
    src_hit = bool(gold_files & src_files) if gold_files else None

    return {
        "id": case["id"], "cat": case["cat"], "diff": case["diff"], "query": case["query"],
        "mode": meta.get("query_mode"), "basis": meta.get("answer_basis"),
        "n_sources": len(sources),
        "src_files": sorted(src_files)[:8],
        "gold_files": sorted(gold_files),
        "source_hit": src_hit,
        "ev_hit": ev_hit,
        "n_evidence": len(ev),
        "cited": sorted(cited), "bad_cite": bad_cite,
        "answer_len": len(text),
        "answer": text[:2500],
        "error": err.get("message") if err else None,
    }


async def main() -> None:
    todo = [c for c in cases][:MAX_CASES]
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = []

    async def worker(c):
        async with sem:
            r = await run_one(c)
            results.append(r)
            log(f"{len(results)}/{len(todo)} {c['id']} basis={r.get('basis')} "
                f"src={r.get('n_sources')} ev={r.get('ev_hit')}")
            return r

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(c) for c in todo))
    results.sort(key=lambda r: r["id"])
    (OUT / "answer_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"完成 {len(results)} 条，耗时 {time.perf_counter()-t0:.0f}s -> benchmark/answer_results.json")


asyncio.run(main())
