"""检索准确率评测脚本（一次性工具，不入测试套件）。

评测当前纯向量检索链路（bge-m3 嵌入 -> Chroma cosine top-20）：
- hit@5 / hit@20（文档级）：正向问题是否把「含答案的文档」召回进 top-N
- 无关问题的 top-5 行为 + 相似度分布（precision 探针）

用法：backend venv 下运行
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ../scripts/eval_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# (问题, 应命中文件的独特子串)。事实均逐条核对过语料原文。
from local_eval_cases import load_case_tables

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

POSITIVE, NEGATIVE = load_case_tables("eval_retrieval")


def main() -> None:
    settings = get_settings()
    store = get_settings_store()
    app_settings = store.load()
    embedder = create_embedder_from_config(app_settings.embed)
    vs = create_vector_store(settings.data_dir, embedder=embedder)

    print(f"嵌入器={embedder.fingerprint} dim={embedder.dim} 块总数={vs.count()}\n")

    def run_query(q: str, top_k: int = 20) -> list[dict]:
        vec = embedder.embed_texts([q])[0]
        return vs.query(vec, top_k)

    # ---- 正向问题：hit@5 / hit@20 ----
    hits5 = hits20 = 0
    print("=" * 78)
    print("正向问题（已知答案文档）")
    print("=" * 78)
    for idx, (q, expect) in enumerate(POSITIVE, 1):
        hits = run_query(q)
        files5 = [h["metadata"].get("source_file", "") for h in hits[:5]]
        files20 = [h["metadata"].get("source_file", "") for h in hits]
        hit5 = any(expect in f for f in files5)
        hit20 = any(expect in f for f in files20)
        hits5 += hit5
        hits20 += hit20
        top1_name = Path(files5[0]).name if files5 else "-"
        sims = ", ".join(f"{h['similarity']:.3f}" for h in hits[:5])
        mark = "✔" if hit5 else ("△hit@20" if hit20 else "✘MISS")
        print(f"[{idx:02d}] {mark} hit@5={hit5} hit@20={hit20}  top1={top1_name}")
        print(f"     sim(top5)=[{sims}]")
        print(f"     Q: {q}")
        if not hit5:
            print(f"     expect<...{expect}>  got files(top5)=")
            for f in files5[:5]:
                print(f"        - {Path(f).name}")
    print(f"\n正向结果: hit@5 = {hits5}/{len(POSITIVE)} = {hits5/len(POSITIVE)*100:.1f}%  "
          f"hit@20 = {hits20}/{len(POSITIVE)} = {hits20/len(POSITIVE)*100:.1f}%")

    # ---- 无关问题：precision 探针 ----
    print("\n" + "=" * 78)
    print("无关/跨域问题（语料无答案，观察被迫检索行为）")
    print("=" * 78)
    for idx, q in enumerate(NEGATIVE, 1):
        hits = run_query(q)
        top5_sims = [round(h["similarity"], 3) for h in hits[:5]]
        print(f"\n[{idx}] {q}")
        print(f"    top5 sim: {top5_sims}")
        for h in hits[:5]:
            print(f"      {h['similarity']:.3f}  {Path(h['metadata'].get('source_file','')).name}  | {h['text'][:40].strip()!r}")

    # ---- 相似度整体分布（top-1 命中问题的相似度）----
    print("\n" + "=" * 78)
    print("top-1 相似度分布（正向问题）")
    print("=" * 78)
    sims = []
    for q, _ in POSITIVE:
        vec = embedder.embed_texts([q])[0]
        top = vs.query(vec, 1)[0]
        sims.append(top["similarity"])
        print(f"  {top['similarity']:.3f}  {Path(top['metadata'].get('source_file','')).name}")
    print(f"  avg top1-sim = {sum(sims)/len(sims):.3f}  min={min(sims):.3f}  max={max(sims):.3f}")


if __name__ == "__main__":
    main()
