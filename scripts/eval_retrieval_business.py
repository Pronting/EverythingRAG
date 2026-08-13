"""业务文档检索质量评测（一次性工具）。

针对刚导入的 doc/ 业务文件夹，构造「已知答案文档」的查询，验证 hit@5 / hit@20。
每条问题的答案都逐条核对过对应文档原文。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:/Users/22331/Desktop/everything-rag/backend")

from app.core.config import get_settings
from app.core.settings_store import get_settings_store
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import create_embedder_from_config

POSITIVE = [
    ("同步 DDL 表结构变更时，Canal 和阿里云 DTS 这类工具有什么局限？", "8000万数据表"),
    ("自建 RocketMQ 时，nameserver 应该部署几台、如何分布在可用区？", "RocketMQ 迁移"),
    ("2024-09-05 推荐接口超时问题的最终根因是什么？", "推荐接口超时"),
    ("MySQL 的 MVCC 是用什么机制实现的？", "PostgreSQL 与 MySQL 差异"),
    ("PG 数据库的一个连接对应什么？", "PostgreSQL 与 MySQL 差异"),
    ("实时风控方案支持哪些维度的限流策略？", "实时风控方案"),
    ("商品订单创建事件里，list_id 参数在非 collection 场景下传什么值？", "商品订单创建事件"),
    ("埋点公共参数里，platform 参数支持哪些取值？", "Cider_埋点公共参数"),
    ("墨西哥礼品卡支付金额错误这个事故记录在哪份文档？", "墨西哥礼品卡支付金额错误"),
    ("验证码邮件重复发送的事故复盘文档是哪个？", "验证码邮件重复发送"),
    ("搜索无结果的技术方案是怎么做的？", "搜索无结果"),
    ("AB 实验的观察指标和分流规则是什么？", "AB实验 观察指标"),
    ("如何评估 RAG 效果？指标是什么？", "AI 八股文"),
    ("埋点平台如何管理埋点的生命周期？", "SPM"),
    ("ApiSix 网关的作用是什么？", "ApiSix 网关"),
    ("阿里云 RocketMQ 迁移时 Broker 机器怎么配置？", "RocketMQ 迁移"),
    ("埋点2.0设计文档里埋点管理后台的地址是什么？", "埋点2.0设计文档"),
    ("线上死循环导致 MQ 消息堆积问题怎么排查？", "死循环导致MQ消息堆积"),
]

NEGATIVE = [
    "量子纠缠在量子计算退相干控制中的应用",
    "火星殖民地生态系统如何实现自维持",
    "巴赫平均律中的对位法与和声分析",
]


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
