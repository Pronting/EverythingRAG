# Agentic 缺陷修复与回归（2026-10-21）

> 范围：在**隔离副本**中对 `everything-rag` HEAD（`a87afc9 feat: V1.2.0`）修复端到端评测暴露的
> 智能体层缺陷，并做单测 + 端到端回归。主工作区未改动，补丁见
> [`fix-agentic-2026-10-21.patch`](fix-agentic-2026-10-21.patch)。
>
> 隔离原因：主仓库 `.git` 被 ACL 锁（`git worktree add` 无法创建 refs 锁），且其他 agent 正在
> 修改同一工作区。副本由 `git archive HEAD` 导出到 `%TEMP%\dsh-7vyviK\erag-fix\repo`，
> `git init` 后基线提交 `e5f0d62`。

## 1. 修了什么（12 项）

### 第一轮：评测直接暴露的 P0

| # | 缺陷 | 位置（HEAD） | 修法 |
|---|---|---|---|
| 1 | 数值严格集合比较：原文 `96.1%` 与正文 `96%` 判为不符，核验失败**吞掉整篇正文** | `verification.py:491-506`；`runtime.py:1536` | `_numbers_equal/_unsupported_numbers`：四舍五入、百分比↔小数、±1% 容差（年份仍严格）；纠正额度用尽时降级投递已写好的正文并标注「未通过来源核验」 |
| 2 | 产品/厂商名（Kubernetes、腾讯云）命中 R5 即 `needs_knowledge=False`，关闭思考时既不查库也不查网 | `routing.py:985-992`、`1013-1014` | 只有**用户指定官方来源**才清零检索；产品名仅表示来源偏好，仍查本地库 |
| 3 | `reason_summary` 虚构「根据请求联网查询公开资料」 | `routing.py:991` | 仅 `explicit_web`/`official_web` 时写该理由 |
| 4 | 中文问句 vs 英文官方页无共同词元，已读到的官方页被整段丢弃 | `research.py:3601-3613` | 新增 `_web_identity_fallback`：用标题/URL 的专有名词确认研究对象；逐段约束不变 |
| 5 | `plan.updated` 硬编码 `covered=[]`、`uncovered=dict键` | `graph.py:198-199` | covered 由 `tasks[*].requirement_ids` 计算；uncovered 输出要求文本 |
| 6 | 产品自述意图被规划成 RAG 任务，0 命中后拒答 | `planner.py:228-231` | `TaskSource.PRODUCT → AgentKind.GENERAL` |

### 第二轮：超时 / 契约 / 检索门控

| # | 缺陷 | 修法 |
|---|---|---|
| 7 | 关闭思考时 `OFF_RUN_TIMEOUT_SECONDS=30`，4 次模型往返必超时（首次复测 4/5 条在 30.4s 被截断、0 字） | 改为环境变量 `EVERYTHING_RAG_OFF_RUN_TIMEOUT_SECONDS`，默认 **90s** |
| 8 | `reasoning_level=medium` 直接 422（合法值只有 off/low/high/max） | `ReasoningLevel._missing_` 把 medium/mid 映射为 high |
| 9 | `GET /api/status` 500：`main.py:221` 调用不存在的 `api_deps.workspace_vector_store` | 改为 `get_workspace_vector_store`（实测 200） |
| 10 | 中文口语问句被 `min_similarity=0.60` 判死、0 命中后直接降级通用知识 | 检索循环遇 `EMPTY/IRRELEVANT` 且未显式传参时，用 `min_similarity=0.35` 重试一次（`GATE_FALLBACK_SIMILARITY`） |
| 11 | 目录盘点死链：`lookup_catalog` 已实现但 research 从不调用；「有哪几篇文章」退化成 1 次语义检索 | 新增 `_CATALOG_LIST_PATTERN` + `_catalog_listing`：命中「哪几篇/哪些文章/列出文档…」时先调 `lookup_catalog(limit=20)`，把目录条目转成 CATALOG 证据并挂 requirement；查询词失败时用去修饰的 core 再试一次 |

### 第三轮：跨文档补读 + 模型失败降级

| # | 缺陷 | 修法 |
|---|---|---|
| 12 | 对比题只选中一侧文档（货拉拉 vs TencentDB 只答 TencentDB）；语义复核模型偶发「推理吃满 4096 输出额度、0 正文」时整轮 `failed`；候选答案模型同类失败无重试 | ① 对比类问题（`区别/差异/对比/相比/不同/各自/vs`）在章节选择后检查覆盖文档数，不足 2 篇则用 `_auto_read_targets` 补另一侧；② `_select_sections` 模型异常/解析失败退回默认阅读目标，不再打死整轮；③ 机械核验已产出正文时，语义复核失败改为**降级投递 + 明示「语义未二次确认」**；④ 候选答案模型调用失败重试一次 |

补丁同时把两条断言旧行为的测试改为新契约（`backend/tests/test_agent_planner.py`），并新增
`backend/tests/test_agentic_fixes_2026_10_21.py`（13 条）。

## 2. 回归结果

### 2.1 单测

隔离副本 `pytest --noconftest`（`conftest.py` 导入 `app.api.deps` 时读取 10MB
`conversations.json` 触发本机内存上限 `MemoryError`，与本次改动无关）：

| 范围 | 基线（HEAD） | 打补丁后 |
|---|---|---|
| 5 文件（新增测试 + routing_rules / verification / agent_planner / routing_eval_cases） | 10 failed / 104 passed / 4 errors | **10 failed（同一批）/ 117 passed / 4 errors** |

失败清单与基线逐条一致（3 条 `SynthesisTests`、6 条 `test_agent_planner` 分源/合并类、
1 条 `test_routing_eval_cases` M-007）。`compileall` 通过。

### 2.2 端到端（隔离数据 241 文件 / 4727 块，commandcode `deepseek/deepseek-v4.1-flash`，关闭思考）

| 用例 | 修复前 | 最终 | 判定 |
|---|---|---|---|
| 产品自述「你好，简单介绍一下你自己」 | 8.3s，68 字「无法确认我的身份」，0 工具 | 15.3s，374 字完整自述，0 工具 | ✅ 不再拒答 |
| 记忆系统「有哪几篇文章」 | 6.2s，1 次检索，只看到 1 篇，未列目录 | 64.4s，11 工具（目录 + 多轮检索），749 字；仍只确认 1 篇、其余登记 gaps | ⚠️ 目录已触发，但标题匹配不到其余文档 |
| 记忆系统泛问题 | 12.3s，401 字，只 1 篇 | 24.3s，581 字，1 引用，仍只《存之有序》 | ⚠️ 受目录/召回限制 |
| 货拉拉 vs TencentDB | 25.6s，609 字，两方都答 | 51.7s，4 工具，690 字，**2 引用**，两侧设计差异讲清 | ✅ 跨文档补读生效 |
| K8s 最小调度单位（关闭思考） | 6.1s，0 工具，209 字，0 引用 | 14.8s，2 工具，257 字，1 引用（本地文档）；明确说明原文用的是「最小单位」而非「调度单位」 | ✅ 引用 + 边界说明 |
| arthas CPU 96.1%（核验） | `failed`，0 字，`synthesis_model_unavailable` | 68.2s，777 字，`completed_with_gaps`：给出 thread→堆栈→源码定位链，并声明 96.1% 无原文依据 | ✅ 不再吞正文，缺口显式 |
| `plan.updated` 覆盖率 | `covered=[]`、`uncovered=["R1","R2"]` | `covered=["R1","R2"]`、`uncovered=[]` | ✅ |

`run_id` / 事件原文：`%TEMP%\dsh-7vyviK\erag-agentic\ag_events\`；结果
`ag_results_round3d.json`（6 例）、`ag_results_num.json`、`ag_results_round3c.json`。

## 3. 未修 / 需要产品决策

1. **目录盘点的标题匹配**：`lookup_catalog` 按标题字面匹配，问句「记忆系统有哪几篇文章」与
   另外两篇的标题（《苦练基本功——写作》《货拉拉…》）没有共同词，只能靠语义检索命中其中一篇。
   要真正列全，需要给文档加主题标签或让目录工具支持语义检索（数据/索引层改动）。
2. **延迟**：6 例中位 38s（15–68s）。主因仍是模型往返次数（目录盘点多 3–4 次检索、对比题补读、
   语义复核 + 纠正），以及推理模型在「关闭思考」下仍消耗 reasoning tokens（日志
   `reasoning_observed_while_off=true`），单次 4–33s。
3. **推理模型吃满输出额度**：`completion_tokens=4096` 全为 reasoning、`content_characters=0`
   的失败在复测中出现 3 次（section_selection / semantic_review / 候选答案）。已加降级与重试，
   但根因在模型/网关；建议提高 `max_tokens` 或在 provider 侧限制 reasoning。
4. **拆分粒度**：三侧/四问类问题仍只建 1 个知识任务（`decomp_three_docs`、`decomp_list` 的
   Service/Ingress 缺查询），gaps 由系统自认。需要 planner 按「子问题 → 要求」显式拆分。
5. **检索门控**：0.35 重试只救「整条 0 命中」；混合检索里 `_is_lexical_rescue` 对未命中的 ASCII
   identifier（如 DTS）整条否决、`max_chunks_per_source=None`（789 块的长文档可占满上下文）
   仍在。
6. 其他既有项：API key 明文、语料覆盖 74.8%、`chunk` 严重不均、`source_ref` 键空间不一致。

## 4. 如何合入

```bash
# 在干净 HEAD 工作区（当前其他 agent 改动请先提交/暂存）
git apply --check docs/tech/fix-agentic-2026-10-21.patch   # 已验证可干净应用
git apply docs/tech/fix-agentic-2026-10-21.patch

cd backend
.venv/Scripts/python -m pytest tests/test_agentic_fixes_2026_10_21.py -q
.venv/Scripts/python -m pytest tests/test_routing_rules.py tests/test_verification.py \
    tests/test_agent_planner.py tests/test_routing_eval_cases.py -q   # 期望：无新增失败
```

隔离副本路径：`%TEMP%\dsh-7vyviK\erag-fix\repo`（含完整可运行环境与 `git diff` 历史）。

## 5. 合入结果（2026-10-21 已在 main 落地）

- 提交：`339d246 fix(agent): 修复 agentic 评测暴露的路由/检索/核验缺陷`（10 文件，+1558/−212）。
- 合入方式：主工作区在补丁生成后已被并行 agent 改动，`git apply --3way` 因索引与补丁前像不一致无法直接应用，改为逐文件 `git merge-file`（base=`a87afc9`、ours=当前工作区、theirs=补丁后副本）三方合并，4 处冲突按语义手工解决：
  - `backend/app/agent/research.py`：`_select_sections` 已被并行改成「按文档并发选章节」，保留新结构，把 try/except 提到 `asyncio.gather` 外层（模型失败/输出不可解析 → 记 warning 并返回 None，退回默认阅读目标）；`_clean_text` 保留并行版的「不折叠空白」实现。
  - `backend/app/agent/runtime.py`：并行版已删掉旧的 `OFF_RUN_TIMEOUT_SECONDS`（改用 `RUN_ACTIVE_TIMEOUT_SECONDS=900` + `_EXECUTION_DEADLINE`），因此丢弃本补丁的 30s→90s 环境变量常量；保留 `factual_validation` 与本补丁的 `last_bound/last_unbound/review` 降级投递逻辑。
  - `backend/app/agent/verification.py`：容差函数 `_numbers_equal/_unsupported_numbers` 与并行版新增的 `source_numbers` 取并集后一起参与校验。
  - `backend/tests/test_agent_planner.py`：`TaskSource.PRODUCT` 期望改为 `AgentKind.GENERAL`。
- 回归（合入后，`pytest --noconftest`）：`7 failed / 124 passed`，失败仍是基线同一批（3 条 SynthesisTests、3 条 planner 分源/合并、1 条 routing_eval M-007），且比基线少 3 条——并行改动已顺手修掉部分 planner 用例；新增 13 条测试全部通过。
- 前端构建：`npm run build`（tsc -b + vite）通过，146 模块，产物 `dist/assets/index-*.js` 522 kB（仅 chunk 体积提示）。
- 后端启动：真实数据目录启动成功，`/api/status` 200（`get_workspace_vector_store` 生效）、`/api/knowledge/overview` 200（241 文件 / 4727 块）；验证后服务已停止。
- 启动期观察：
  - 用隔离数据副本启动时，3 条历史记忆任务恢复报 `WorkspaceNotFoundError`——`app.db` 的 `workspaces` 表有这些 id，但 `workspaces.json` 没有，属两套工作区记录不一致；真实数据目录启动日志干净，未复现。
  - 冒烟 2 题均 completed（12.94s / 13.15s）；「介绍一下你自己」现在被路由成 `clarification`，检索 0 命中后用雷姆人设作答而非产品自述，仍是待改的产品口径问题（不影响本次合并）。
