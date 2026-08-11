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
| 2 | ingestion：MD 解析（markdown-it-py） | pytest：fixture MD → 标题树 + 归一化正文；空/畸形文档不崩（返回空结构，不抛异常） | 1 | 无 | 🔄 |
| 3 | ingestion：语义切块（防边界漂移） | pytest：长文按标题层级切块，块长 ≤ 上限、标题边界不漂移、块带来源路径与锚点 | 2 | 无 | ⬜ |
| 4 | vectorstore：Chroma 适配 + bge-m3 嵌入 | 单测用 fake embedder：upsert + query top-k 返回带元数据块；collection 命名 `chunks__bge-m3__v1`；真模型**惰性加载** | 3 | ⚠️ 首次需下载 bge-m3 ONNX（~600MB，属显式 opt-in 出网，**需人工确认**） | ⬜ |
| 5 | retrieval：检索管线 | pytest：mock vectorstore，query → top-k 带来源块；空查询 / 未建索引返回清晰错误而非崩溃 | 3,4 | 无 | ⬜ |
| 6 | generation + API：SSE 问答 `POST /api/chat` | pytest+httpx：SSE 帧按 meta→token(s)→done 顺序输出（协议用 fake provider 断言）；error 路径发 error 帧不崩；对话 provider 走 OpenAI 兼容接口、可插拔 | 5 | 对话 provider 默认值（本地 Ollama 地址）需确认 | ⬜ |
| 7 | 隐私审计：出网收敛 + 网络审计 | pytest：纯文本查询链路零出网（mock 断言 OutboundClient 未调用）；审计面板数据结构落位 | 5,6 | 无 | ⬜ |
| 8 | frontend：Chat UI（手写 SSE 解析） | Playwright：打开首页 → 发消息 → 断言流式 token 渲染 + 来源块展示 | 6 | 无 | ⬜ |
| 9 | 端到端最小闭环 | E2E：导入 fixture 语料 → 问答返回带来源答案；`scripts/run.py` 一键拉起全流程 | 1-8 | 无 | ⬜ |

## 任务执行摘要（每轮完成后控制器留痕）

- **任务 1 ✅ 2026-08-11**：`scan_directory` → `DiscoveredFile`（path/filename/mtime/size/content_hash=xxh64）；隐藏目录/文件跳过、失败容错不中断、确定性排序、零出网。**11 passed**，ruff 全绿，commit `d00bc97`。评审 APPROVE。遗留 MEDIUM（不阻塞，留待后续）：① 跳过日志为 debug——评审建议升 warning，但 PRD 日志脱敏红线「日志不得含文件路径」，保持 debug 正确 ② 目录不可读分支 `_on_walk_error` 缺单测 ③ `.md` 大小写敏感（`NOTE.MD` 忽略）④ 符号链接可能产生重复条目（增量同步去重任务需注意）。

## 半自动 / 需人工（设计决策 Gate，见 docs/loop/loop-workflow.md §5）

- **任务 4**：bge-m3 模型下载属**出网**，违反「默认零外发」→ 必须用户显式确认后才允许触发下载；未确认前用 fake embedder 完成单测。
- **任务 6**：对话 provider 的默认值（本地 Ollama 地址 / 云端 endpoint）需用户确认后才落为默认配置。
- **切块算法 / 对话去重键规范**：若子 agent 方案与 PRD 12.2 待拍板项冲突，控制器必须停下报用户，不得自行拍板。

## 循环完成报告（最后一个任务 ✅ 后，控制器在此留痕）

（首轮循环完成后填写：完成的任务 / 测试通过情况 / 遗留需人工项）
