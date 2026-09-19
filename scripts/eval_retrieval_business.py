"""业务文档检索质量评测（一次性工具）。

针对刚导入的 doc/ 业务文件夹，构造「已知答案文档」的查询，验证 hit@5 / hit@20。
每条问题的答案都逐条核对过对应文档原文。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:/Users/22331/Desktop/everything-rag/backend")

from local_eval_cases import load_case_tables

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

POSITIVE, NEGATIVE = load_case_tables("eval_retrieval_business")


def main() -> None:
    settings = get_settings()
    app_settings = get_settings_store().load()
    embedder = create_embedder_from_config(app_settings.embed)
    vs = create_vector_store(settings.data_dir, embedder=embedder)
    print(f"嵌入器={embedder.fingerprint} 块总数={vs.count()}\n")

    hits5 = hits20 = 0
    print("=" * 90)
    for idx, (q, expect) in enumerate(POSITIVE, 1):
        vec = embedder.embed_texts([q])[0]
        hits = vs.query(vec, 20)
        files5 = [h["metadata"].get("source_file", "") for h in hits[:5]]
        files20 = [h["metadata"].get("source_file", "") for h in hits]
        hit5 = any(expect in f for f in files5)
        hit20 = any(expect in f for f in files20)
        hits5 += hit5
        hits20 += hit20
        top1 = Path(files5[0]).name if files5 else "-"
        sims = ", ".join(f"{h['similarity']:.3f}" for h in hits[:5])
        mark = "✔" if hit5 else ("△@20" if hit20 else "✘MISS")
        print(f"[{idx:02d}] {mark}  top1={top1}")
        print(f"     sim=[{sims}]  Q: {q}")
        if not hit5:
            print(f"     expect<{expect}> top5=")
            for f in files5[:5]:
                print(f"        - {Path(f).name}")
    print(f"\n业务文档评测: hit@5 = {hits5}/{len(POSITIVE)} = {hits5/len(POSITIVE)*100:.1f}%  "
          f"hit@20 = {hits20}/{len(POSITIVE)} = {hits20/len(POSITIVE)*100:.1f}%")

    print("\n--- 无关问题 top-5 行为 ---")
    for idx, q in enumerate(NEGATIVE, 1):
        vec = embedder.embed_texts([q])[0]
        hits = vs.query(vec, 5)
        sims = [round(h["similarity"], 3) for h in hits]
        names = [Path(h["metadata"].get("source_file", "")).name for h in hits]
        print(f"[{idx}] sim={sims}")
        for n in names:
            print(f"      {n}")


if __name__ == "__main__":
    main()
