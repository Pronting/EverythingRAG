# Loop 状态机 — MVP 最小闭环任务队列

> 本文件是 `/loop-plan` 循环控制器的**唯一事实来源**：一轮任务做完 → 更新状态 → 进下一轮。
> 状态图例：⬜ 待做 ｜ 🔄 进行中（进程在 sub-agent 里，主会话只留摘要）｜ ✅ 通过 ｜ ⚠️ 需人工 ｜ ❌ 打回重做（≤2 次后转 ⚠️）
> 驱动方式：输入 `/loop-plan`（可带任务号，如 `/loop-plan 3`）。规范见 `docs/loop/loop-workflow.md`。

## MVP 最小闭环（当前循环）

目标：打通 **目录扫描 → MD 解析 → 切块 → 嵌入 → Chroma → 检索 → SSE 问答** 完整链路（PRD §MVP 第一块拼图）。
起点：脚手架已就绪（commit `7e8bcfa`：backend 骨架 + smoke 测试 + frontend Vite + scripts/run.py）。

| # | 任务 | 验收标准（可执行断言） | 依赖 | Gate | 状态 |
|---|---|---|---|---|---|
| 1 | ingestion：目录扫描 + MD 发现 | pytest：fixture 目录（嵌套 .md / 非 md / 隐藏目录）→ 返回正确 md 文档 + 元数据（路径/文件名/mtime/xxhash 指纹）；隐藏与无关文件被忽略 | - | 无 | ✅ |
| 2 | ingestion：MD 解析（markdown-it-py） | pytest：fixture MD → 标题树 + 归一化正文；空/畸形文档不崩（返回空结构，不抛异常） | 1 | 无 | ✅ |
| 3 | ingestion：语义切块（防边界漂移） | pytest：长文按标题层级切块，块长 ≤ 上限、标题边界不漂移、块带来源路径与锚点 | 2 | 无 | ✅ |
| 4 | vectorstore：Chroma 适配 + bge-m3 嵌入 | 单测用 fake embedder（快、零网络）：upsert + query top-k 返回带元数据块；collection 命名 `chunks__bge-m3__v1`；真模型 bge-m3 **惰性加载**（下载出网已获用户授权） | 3 | 无（下载已授权） | ✅ |
| 5 | retrieval：检索管线 | pytest：mock vectorstore，query → top-k 带来源块；空查询 / 未建索引返回清晰错误而非崩溃 | 3,4 | 无 | ✅ |
| 6 | generation + API：SSE 问答 `POST /api/chat` | pytest+httpx：SSE 帧按 meta→token(s)→done 顺序输出（协议用 fake provider 断言）；error 路径发 error 帧不崩；对话 provider 走 OpenAI 兼容接口、可插拔，配置取自**用户输入的 provider 信息** | 5 | 已确认：云端 OpenAI 兼容 + env 占位（2026-08-12 用户拍板） | ✅ |
| 7 | 隐私审计：出网收敛 + 网络审计 | pytest：纯文本查询链路零出网（mock 断言 OutboundClient 未调用）；审计面板数据结构落位 | 5,6 | 无 | ❌ |
| 8 | frontend：Chat UI（手写 SSE 解析） | Playwright：打开首页 → 发消息 → 断言流式 token 渲染 + 来源块展示 | 6 | 无 | ⬜ |
| 9 | 端到端最小闭环 | E2E：导入 fixture 语料 → 问答返回带来源答案；`scripts/run.py` 一键拉起全流程 | 1-8 | 无 | ⬜ |

## 任务执行摘要（每轮完成后控制器留痕）

- **任务 1 ✅ 2026-08-11**：`scan_directory` → `DiscoveredFile`（path/filename/mtime/size/content_hash=xxh64）；隐藏目录/文件跳过、失败容错不中断、确定性排序、零出网。**11 passed**，ruff 全绿，commit `d00bc97`。评审 APPROVE。遗留 MEDIUM（不阻塞，留待后续）：① 跳过日志为 debug——评审建议升 warning，但 PRD 日志脱敏红线「日志不得含文件路径」，保持 debug 正确 ② 目录不可读分支 `_on_walk_error` 缺单测 ③ `.md` 大小写敏感（`NOTE.MD` 忽略）④ 符号链接可能产生重复条目（增量同步去重任务需注意）。
- **任务 2 ✅ 2026-08-11**：`parse_markdown` → `ParsedMarkdown`（title/heading_tree/normalized_text），markdown-it-py token 流（非 HTML）、frontmatter 剔除、GitHub slug 全局唯一锚点、段落边界保留、空/畸形容错、零出网。首轮 34 passed（commit `8e5c973`）→ 评审 REVISE（HIGH 锚点唯一性：`# a-b/# a b/# a b 1` 重复 `a-b-1`；MEDIUM 段落边界折叠）→ 修复 `ffe6cc1`（锚点候选冲突递增唯一 + 块间 `\n\n` 保留）→ **复审 APPROVE**。最终 **36 passed**、ruff 全绿。
- **任务 3 ✅ 2026-08-11**：解析器增量扩展（`MDBlock` + `ParsedMarkdown.blocks` 单一数据源，输出逐字节不变）+ `chunker.chunk_document`（标题锚点分层、block_id=锚点路径链+区间序号、代码块整体成块不拆散、超长按段/句/字逐级拆、无标题段落切块、边界不漂移）。**62 passed**、ruff 全绿，commit `baf42ef`。评审 APPROVE（27 输入对旧实现字节级回归一致）。遗留 MEDIUM（不阻塞）：① 防漂移测试可加 block_id 集合相等断言 ② frontmatter 判定用对象身份（`body is not text`）脆弱 ③ `_iter_headings` 与 blocks 重复可 DRY ④ 结构性插入同名标题会使后续锚点漂移（设计边界，建议 docs 注明）。
- **任务 4 ✅ 2026-08-11**：`FastEmbedEmbedder`（bge-m3 惰性加载，fastembed 0.8 未内置 → `add_custom_model` 回退注册 CLS 池化+归一化 dim1024，对照官方仓库+源码核实）+ `ChromaVectorStore`（cosine space、`chunks__bge-m3__v1`、`anonymized_telemetry=False`、upsert 幂等带 text、query where 过滤、维度 fail-fast）。真实 bge-m3 **未触发下载**（惰性到首次真实 embed）。**76 passed**、ruff 全绿，commit `04de046`。评审 APPROVE。遗留 MEDIUM（不阻塞）：① 回退注册参数无测试保护（建议断言 pooling/dim/normalization/model_file）② similarity 可为负（cosine∈[-1,1]，下游业务层须知）③ query 向量维度未 fail-fast ④ `_ensure_model` 非线程安全 ⑤ 真实下载后建议一次性向量一致性冒烟。
- **任务 5 ✅ 2026-08-11**：`VectorRetriever`（candidate_k=20 召回 → similarity 降序 → context_top_k=5），`RetrievalError`（空查询/未建索引清晰错误，均在 store.query 前短路）、无匹配返 `[]`、where 透传、同分按 block_id 稳定。VectorStore Protocol 新增 `count()`（additive）。**88 passed**、ruff 全绿，commit `78b9e62`。评审 APPROVE。遗留 LOW（不阻塞）：① `count()` 缺直接单测 ② `float(similarity)` 遇 None 会崩（建议防御）③ 每查询一次 count O(n)（MVP 个人库量级可接受）。
- **任务 6 ✅ 2026-08-12**：SSE 问答 `POST /api/chat`（OpenAI 兼容云端 provider，key 仅环境变量）+ `ChatService`（嵌入→检索→上下文→流式）+ `providers.create_chat_model`（未配置 → `UnconfiguredChatModel` 延迟报错帧）。SSE 帧 meta→token(s)→done / error。**110 passed**、ruff 全绿，commit `7b019d1`。评审 APPROVE。遗留 MEDIUM（不阻塞）：① provider 流 `choices` 空时 IndexError ② `stream_chat` 在 try 外 ③ 嵌入+检索同步阻塞事件循环 ④ SDK 401/429 未映射受控文案。（① ② 已由 task 7 顺手修复）
- **任务 7 ❌ 首轮 128 passed（commit `be6bc4a`）→ 评审 REVISE（第 1 次打回）**：**CRITICAL = 生产链路未注入 `outbound_client`**（`create_chat_model` 从不传全局单例 → 真实云端调用零审计、`/api/audit` 与 `/api/status` 恒 `local-only` 误报隐私状态；单测全手动注入故 128 全绿未暴露）。MEDIUM = GeneratorExit（SSE 断开）不记审计事件；choices 空时未查 `chunk.error`。已按评审方案重派修复实施器（`create_chat_model` 默认接全局单例 + 补接线测试 + 两 MEDIUM 防御）。

## 半自动 / 需人工（设计决策 Gate，见 docs/loop/loop-workflow.md §5）

- **任务 4（已授权，不再询问）**：bge-m3 模型下载出网已获用户显式授权（2026-08-11），任务 4 允许出网；**仅限该模型下载**，其余模块仍保持「默认零外发」红线。
- **任务 6（已拍板 2026-08-12）**：用户选择**云端 OpenAI 兼容 API**；具体字段先用环境变量占位——`EVERYTHING_RAG_CHAT_BASE_URL` / `EVERYTHING_RAG_CHAT_MODEL` / `EVERYTHING_RAG_CHAT_API_KEY`（API key 走环境变量，不落明文 config）。任务 6 按此落配置结构；协议测试用 fake provider，不触发真实出网。
- **切块算法 / 对话去重键规范**：若子 agent 方案与 PRD 12.2 待拍板项冲突，控制器必须停下报用户，不得自行拍板。

## 循环完成报告（最后一个任务 ✅ 后，控制器在此留痕）

（首轮循环完成后填写：完成的任务 / 测试通过情况 / 遗留需人工项）
