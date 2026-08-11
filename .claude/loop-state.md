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
| 4 | vectorstore：Chroma 适配 + bge-m3 嵌入 | 单测用 fake embedder（快、零网络）：upsert + query top-k 返回带元数据块；collection 命名 `chunks__bge-m3__v1`；真模型 bge-m3 **惰性加载**（下载出网已获用户授权） | 3 | 无（下载已授权） | ⬜ |
| 5 | retrieval：检索管线 | pytest：mock vectorstore，query → top-k 带来源块；空查询 / 未建索引返回清晰错误而非崩溃 | 3,4 | 无 | ⬜ |
| 6 | generation + API：SSE 问答 `POST /api/chat` | pytest+httpx：SSE 帧按 meta→token(s)→done 顺序输出（协议用 fake provider 断言）；error 路径发 error 帧不崩；对话 provider 走 OpenAI 兼容接口、可插拔，配置取自**用户输入的 provider 信息** | 5 | 需用户输入 provider 信息（类型/base_url/模型名/云端 API key） | ⬜ |
| 7 | 隐私审计：出网收敛 + 网络审计 | pytest：纯文本查询链路零出网（mock 断言 OutboundClient 未调用）；审计面板数据结构落位 | 5,6 | 无 | ⬜ |
| 8 | frontend：Chat UI（手写 SSE 解析） | Playwright：打开首页 → 发消息 → 断言流式 token 渲染 + 来源块展示 | 6 | 无 | ⬜ |
| 9 | 端到端最小闭环 | E2E：导入 fixture 语料 → 问答返回带来源答案；`scripts/run.py` 一键拉起全流程 | 1-8 | 无 | ⬜ |

## 任务执行摘要（每轮完成后控制器留痕）

- **任务 1 ✅ 2026-08-11**：`scan_directory` → `DiscoveredFile`（path/filename/mtime/size/content_hash=xxh64）；隐藏目录/文件跳过、失败容错不中断、确定性排序、零出网。**11 passed**，ruff 全绿，commit `d00bc97`。评审 APPROVE。遗留 MEDIUM（不阻塞，留待后续）：① 跳过日志为 debug——评审建议升 warning，但 PRD 日志脱敏红线「日志不得含文件路径」，保持 debug 正确 ② 目录不可读分支 `_on_walk_error` 缺单测 ③ `.md` 大小写敏感（`NOTE.MD` 忽略）④ 符号链接可能产生重复条目（增量同步去重任务需注意）。
- **任务 2 ✅ 2026-08-11**：`parse_markdown` → `ParsedMarkdown`（title/heading_tree/normalized_text），markdown-it-py token 流（非 HTML）、frontmatter 剔除、GitHub slug 全局唯一锚点、段落边界保留、空/畸形容错、零出网。首轮 34 passed（commit `8e5c973`）→ 评审 REVISE（HIGH 锚点唯一性：`# a-b/# a b/# a b 1` 重复 `a-b-1`；MEDIUM 段落边界折叠）→ 修复 `ffe6cc1`（锚点候选冲突递增唯一 + 块间 `\n\n` 保留）→ **复审 APPROVE**。最终 **36 passed**、ruff 全绿。
- **任务 3 ✅ 2026-08-11**：解析器增量扩展（`MDBlock` + `ParsedMarkdown.blocks` 单一数据源，输出逐字节不变）+ `chunker.chunk_document`（标题锚点分层、block_id=锚点路径链+区间序号、代码块整体成块不拆散、超长按段/句/字逐级拆、无标题段落切块、边界不漂移）。**62 passed**、ruff 全绿，commit `baf42ef`。评审 APPROVE（27 输入对旧实现字节级回归一致）。遗留 MEDIUM（不阻塞）：① 防漂移测试可加 block_id 集合相等断言 ② frontmatter 判定用对象身份（`body is not text`）脆弱 ③ `_iter_headings` 与 blocks 重复可 DRY ④ 结构性插入同名标题会使后续锚点漂移（设计边界，建议 docs 注明）。

## 半自动 / 需人工（设计决策 Gate，见 docs/loop/loop-workflow.md §5）

- **任务 4（已授权，不再询问）**：bge-m3 模型下载出网已获用户显式授权（2026-08-11），任务 4 允许出网；**仅限该模型下载**，其余模块仍保持「默认零外发」红线。
- **任务 6（用户输入项）**：任务 6 开始时，控制器必须**向用户收集对话 provider 信息**（本地 Ollama vs 云端、base_url、模型名、云端时 API key），用户输入后才落为配置并继续。
- **切块算法 / 对话去重键规范**：若子 agent 方案与 PRD 12.2 待拍板项冲突，控制器必须停下报用户，不得自行拍板。

## 循环完成报告（最后一个任务 ✅ 后，控制器在此留痕）

（首轮循环完成后填写：完成的任务 / 测试通过情况 / 遗留需人工项）
