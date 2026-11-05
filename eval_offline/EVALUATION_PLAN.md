# Everything RAG — RAG Agent 评测方案（v1，离线可复现）

> 目标：对当前仓库的 RAG Agent 做一次可复现、可证伪的系统评测：检索质量、上下文相关性、答案忠实度/准确率、拒答、延迟、吞吐、内存，并沉淀问题清单。
>
> 边界：本轮**不联网、不调用任何外部 API**，不修改业务代码；所有新增脚本位于 `eval_offline/`，只读业务模块。

## 1. 被测对象：真实链路（已按代码核对）

V1.2.0 的问答入口不是旧的 `POST /api/chat`（该路由已返回 **410 Gone**，`backend/app/api/routes/chat.py`），而是 Agent run：

```
POST /api/workspaces/{workspace_id}/runs
  → RunManager（backend/app/agent/runtime.py）
  → 路由 planner（agent/routing.py / planner.py）
  → 工具循环（agent/tools.py）
      search_knowledge → _embed_and_retrieve → HybridRetriever.retrieve_variants
      read_document    → 全文/章节续读
  → 证据核对（agent/verification.py）
  → 生成（generation/providers.py + generation/prompt_policy.py）
  → SSE 事件：GET .../runs/{run_id}/events
```

检索装配与门控（关键事实）：

| 环节 | 代码位置 | 事实 |
|---|---|---|
| 检索器装配 | `backend/app/api/run_services.py:144` | 每次 `tool_services_for()` 都**新建** `HybridRetriever(vector_store, min_similarity=DEFAULT_MIN_SIMILARITY)` |
| 门控默认值 | `backend/app/retrieval/vector_retriever.py:52` | `DEFAULT_MIN_SIMILARITY = 0.60`；注释记录 0.55 为 bge-m3 校准，切到 Qwen3-8B 后上调到 0.60 |
| 混合检索 | `backend/app/retrieval/hybrid_retriever.py:139-209` | dense + BM25 + RRF(k=60) + 章节扩展；`context_top_k=8`、`candidate_k=100` |
| 门控 | 同上 `:905-1024` | `min_similarity=None` 时**不设阈值**；否则 dense-only 需 `阈值+0.05`，数字噪声再 `+0.10`，词法救援可用 0.45 覆盖但 dense<0.35 否决 |
| 工具 | `backend/app/agent/tools.py:1426` | `search_knowledge` 默认 `top_k=8`；全被门控 → `IRRELEVANT/irrelevant`，不回落未筛选候选 |
| 拼上下文 | `backend/app/agent/tools.py:1105-1133` | 片段载荷含 block_id/document_name/text/similarity/heading，**不含路径键** |
| 提示词 | `backend/app/generation/prompt_policy.py` | `rag-policy-v3`；只依据上下文、引用必须来自本轮 source_id、无依据不得断言用户没有资料 |

评测直接覆盖「检索 + 工具契约 + 上下文拼装」；生成质量需要真实 LLM，本轮用本地假模型验证契约（见 §5）。

## 2. 维度、指标与判分口径

### 2.1 检索层（离线可跑）

| 指标 | 定义 | 判分口径 |
|---|---|---|
| Hit@1 / Hit@5 / Hit@8 | 前 k 个块中是否含任一「证据块」 | 证据块 = 包含标注原文片段的块（多跳题每组证据各算一组） |
| Full@8 | 多跳题的全部证据组是否都进前 8 | 缺任一组即 0 |
| MRR@8 | 首个证据块排名的倒数 | 无命中记 0 |
| nDCG@8 | 二值增益 nDCG | `DCG=Σ rel_i/log2(i+1)`，IDCG 按证据块数截断 |
| 负例误放行 | 无答案题是否仍返回片段 | 返回 ≥1 块即误放行 |
| 门控通过率 | 有证据题的证据块中，被 gate 接受的比例 | 从 trace 的候选决策统计 |

### 2.2 上下文相关性（离线可跑）

- **证据覆盖率**：返回的 top-8 块中，包含标注证据的块数 / 该题证据总数。
- **上下文信噪比**：证据块字符数 / 全部入上下文块字符数。
- **冗余度**：同 heading / 同 parent 的重复块占比（检索器已按 heading≤2、parent≤3 限流，用于验证限流是否生效）。
- **路径泄露**：工具返回载荷是否出现 `source_file/source_path` 等键（隐私契约，应为 0）。

### 2.3 生成层（需真实 LLM；本轮只验证契约）

| 指标 | 定义 | 判分 |
|---|---|---|
| 答案准确率 | 回答要点命中标注 `answer_points` 的比例 | 逐点 0/1，按题加权；≥80% 为 PRD 目标 |
| 忠实度 | 回答中的事实陈述能否在引用片段中找到依据 | 逐句人工/LLM 裁判；不支持的事实计 hallucination |
| 引用正确率 | `[S1]` 等引用是否指向真实入上下文且支持该句的块 | 错误引用数 / 引用总数 |
| 拒答正确率 | 无答案题是否明确说明资料不足 | 编造即失败；有答案题误拒也失败 |
| 引用格式合规 | 是否只使用本轮 source_id | 出现未提供编号即失败 |

### 2.4 系统层

| 指标 | 定义 | 采集方式 |
|---|---|---|
| 首 token 延迟 TTFT | 请求 → 首个 token 帧 | SSE 埋点（本轮无法真跑，给替代验证） |
| 检索延迟 | 查询嵌入 + dense + BM25 + 融合 + 门控 | 工具 `timings` / 进程内计时 |
| 生成吞吐 | 输出 token / 生成秒 | `done.tok_per_s`（本轮无法真跑） |
| 内存 | 进程 WorkingSet（不含本地 LLM） | Windows `GetProcessMemoryInfo` |
| 索引构建吞吐 | 块/秒、总耗时 | `build_index.py` 日志 |

### 2.5 验收阈值（来自 PRD，用于对照而非自定）

- Top-5 命中 ≥80%（`docs/PRD/PRD.md:460`）
- 答案准确率 ≥80%（`:461`）
- 首 token <3s（本地模型，`:463`）；v0.2 含检索 <5s（`:1549`）
- 检索 P95 <1s（`:1583`）
- 常驻内存 ≤2GB（不含本地大模型，`:1530`）

## 3. 评测集设计

- **语料**：仓库 `docs/` 全部 34 篇 Markdown（PRD、tech、诊断报告等，约 1.03MB），在 `eval_offline/index` 建隔离索引，不触碰 `~/.everything-rag`。
- **题量**：36 题 = 单跳 19 + 术语 3 + 多跳 3 + 跨文档 4 + 表格 3 + 无答案 2 + 负向有据 2。
- **证据标注**：每题给 `evidence: [{doc, snippet}]`，snippet 必须能在原文中逐字命中（已用脚本校验 36/36 通过）；索引后映射为「包含该片段的块 ID」，作为相关性金标准。
- **多跳**：M01（LangChain 决策反转）、M02（记忆路线图 vs 已交付）、M03（知识库隔离 vs D-KB）、M04（导入性能自相矛盾）要求**全部**证据组命中才算 Full。
- **故意无答案**：N01（iOS 客户端）、N02（单次问答人民币成本）；文档中确无依据，用于测拒答。
- **负向有据**：N03（xlsx 不支持）、N04（不做云同步）要求答「否」并给出文档依据，用于测「能否用证据否证」。

> 隐私：`questions.json` 只引用项目自身的公开设计文档，不含真实用户语料或个人数据。

## 4. 可离线跑 vs 需模型/授权

| 部分 | 本轮状态 | 说明 |
|---|---|---|
| 切块、解析 | ✅ 可跑 | 纯函数；已产出 1282 块统计 |
| 本地嵌入 | ✅ 可跑（替代） | 用 HF 缓存里的 bge-m3 ONNX（CLS+归一化，1024 维，截断 512），**不是生产 Qwen3-8B** |
| 混合检索全链路 | ✅ 可跑 | 真实 `HybridRetriever`，含 BM25/RRF/门控/扩展 |
| 工具层 `search_knowledge` | ✅ 可跑 | 真实函数 + 真实上下文对象 |
| 上下文拼装/引用契约 | ✅ 可跑 | 校验载荷字段与路径脱敏 |
| 答案质量/忠实度/拒答 | ❌ 不可离线 | 无本地 LLM（无 ollama/llama.cpp）；需显式 opt-in 云端或本地模型 |
| 首 token / tok/s | ❌ 不可离线 | 同上；仅能给检索段延迟与埋点位置 |
| 生产向量库对比 | ⚠️ 只读检查 | `~/.everything-rag` 有 4096 维 Qwen collection，但缺少对应云端 API key 时无法查询嵌入 |

## 5. 执行与复现

```bash
# 1) 组装本地 bge-m3（硬链接，不复制 2.2GB）
#    已生成 eval_tmp_model/：model.onnx + model.onnx_data + tokenizer.json
# 2) 建索引（约 17 分钟，1282 块）
backend/.venv/Scripts/python eval_offline/build_index.py
# 3) 跑检索评测（阈值扫描 + 工具契约）
backend/.venv/Scripts/python eval_offline/eval_retrieval.py
```

产物：`eval_offline/manifest.json`、`eval_offline/results/retrieval_metrics.{json,md}`。

## 6. 已知偏差（结论适用范围）

1. **嵌入模型不同**：本地 bge-m3 1024 维 vs 生产 Qwen3-Embedding-8B（2048 请求维/4096 历史 collection）。相似度分布不同，**0.60 阈值对 bge-m3 偏严**，因此报告做阈值扫描而非只报单点。
2. **截断**：本地适配器按 512 token 截断；语料 8.4% 的块超长（max 1246），这部分嵌入只看到前半段。生产端若同样截断，问题一致；若 Qwen3 用更长上下文，本报告会低估长块召回。
3. **无生成**：不评答案准确率/忠实度/拒答的真实表现，只评「检索能否把证据送进上下文」和「工具/提示词契约是否成立」。
4. **Agent 编排未跑**：planner/verification 需要 LLM；因此不覆盖路由错判、工具选择错误、多轮循环开销。
5. **语料是项目文档**：领域偏技术文档，结论不能直接外推到对话导出/图片语料。
