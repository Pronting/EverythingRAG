# T2-Dashboard 前端 UI 技术选型细节与边界

> 生成日期：2026-08-11 ｜ 状态：技术选型细节 v0.1 ｜ 角色：前端架构师
> 承接：`PRD/sections/04-技术架构.md`（§3.4 流式、§3.5 部署形态、§5 隐私安全）与 `PRD/sections/03-功能需求.md`（4.1/4.5/4.6/4.7/4.8）
> 范围：**仅细化 Dashboard 前端 UI 的落地细节与边界**，不推翻任何既有决策。后端语言/框架（Python + FastAPI）、SSE 而非 WebSocket、本地 Web 服务形态、纯本地与数据不出机、v1.0 后 Tauri 套壳复用，均为既定约束。

---

## 0. 决策总览（一页速览）

| # | 决策点 | 推荐 | 备选 | 一句话理由 |
|---|---|---|---|---|
| 1 | 前端框架 | **React + Vite + TypeScript**（构建产物为纯静态资源） | Vue 3 + Vite；原生 HTML+少量 JS | 功能面大（5 视图 + SSE + 多轮 + i18n + 向导），组件化收益远超构建成本；Vite 把构建变成"一条命令"；纯静态产物为 Tauri 复用留路 |
| 2 | 路由 | **Hash 路由**（`#/chat` 等） | Browser 路由 + FastAPI SPA fallback | 静态托管无需后端 catch-all，Tauri 本地资源加载更稳 |
| 3 | 状态管理 | React 内置（useState/useReducer/Context） | Zustand（v0.2 需要时引入） | YAGNI；MVP 无跨组件全局共享复杂状态 |
| 4 | Markdown 渲染 | **markdown-it（`html:false`）+ DOMPurify 兜底** | 自研极简渲染器 | 不可信文本必须白名单渲染，见 §6 |
| 5 | SSE 消费 | **原生 fetch + ReadableStream 手写解析**（POST + AbortController） | @microsoft/fetch-event-source | 原生 EventSource 只支持 GET，长上下文超 URL 限制；手写 ~80 行换取消能力与零依赖 |
| 6 | 样式方案 | **自研 CSS 变量设计令牌 + 语义化组件类**（无 UI 库） | Tailwind CSS v4；shadcn/ui | 5 个视图都是聊天/表单/列表类，重组件库 80% 用不上；零依赖 + 最利于严格 CSP |
| 7 | 暗色模式 | CSS 变量 + `[data-theme]`，默认跟随系统 | 仅固定亮/暗 | 几行 CSS 实现，个人工具克制即可 |
| 8 | i18n | **自研极简 i18n**（messages/zh.ts + en.ts + `t()` + Context） | react-i18next | 中文首发、英文仅预留，自研 ~100 行足够且不引入运行时 |
| 9 | 导入形态边界 | **路径模式（后端扫描、就地引用）为主 + 拖拽上传（内容进受管目录）为辅** | 仅路径模式 | 浏览器无法读取本地绝对路径，"拖拽"只能拿到文件内容，必须落受管目录，见 §5.1 |
| 10 | 前端安全 | 全文本渲染 + Markdown 白名单 + 严格 CSP（`script-src 'self'`） | 无 | 检索上下文与模型输出不可信；CSP 与 Vite 产物天然兼容（见 §6 兼容注意） |

---

## 1. 前端框架选型

### 1.1 三案权衡

| 维度 | React 18/19 + Vite + TS | Vue 3 + Vite | 原生 HTML + 少量 JS |
|---|---|---|---|
| 组件化维护性 | **强**：函数组件 + hooks，聊天/来源卡片/向导天然组件化 | 强：SFC 模板直观 | 弱：DOM 拼接 + 事件手动管理，功能累积后失控 |
| 构建复杂度 | **低**：Vite dev 即时热更（无手动构建感）；`npm run build` 单命令，由发布脚本执行 | 同左 | 零构建（唯一优势） |
| SSE 流式支持 | 成熟：state 增量 + 局部渲染模式文档多 | 成熟 | 需手写 DOM 更新，重复劳动 |
| 状态管理（多轮对话） | hooks 原生支持 | 组合式 API 支持 | 手写 store，易出 bug |
| i18n / 暗色 / CSP | 纯静态产物，CSP 友好 | 同左 | 同左，但代码组织差 |
| 生态/社区 | **最大**（未来开源阶段贡献者最熟） | 中文社区友好 | 无生态可言 |
| Tauri 复用（v1.0） | 纯静态 dist/ 直接收编 | 同左 | 也能进 Tauri，但维护性差 |
| 风险 | 需 Node 工具链（开发期一次安装） | 同左 | 无工具链 |

### 1.2 推荐：React + Vite + TypeScript

理由（按权重排序）：

1. **功能面与"无构建偏好"的冲突是伪冲突**。MVP 前端功能面不小：5 个视图、SSE 流式、多轮对话、来源标注、配置向导、i18n、暗色、CSP。这套需求用原生 JS 做，跨组件状态、流式缓冲、事件清理会快速腐化；而 Vite 的开发体验（`npm run dev` + HMR，保存即热更新）本质上就是"无手动构建"，生产 `build` 一条命令且只在发布时执行，对日常开发零负担。
2. **纯静态产物，彻底对齐既定架构**。Vite build 产出 `dist/`（index.html + assets），由 FastAPI `StaticFiles` 同源托管；未来 Tauri 壳直接指向 dist/ 或加载 `127.0.0.1:port`，前端不锁死后端形态。
3. **生态与社区**。聊天/流式/来源标注类交互在 React 生态有大量可参考实现；开源阶段全球贡献者对 React 熟悉度最高，符合"个人工具 → 开源"演进。
4. **TypeScript 与后端契约天然配套**。后端 FastAPI 自动生成 OpenAPI（`/openapi.json`），前端可用 openapi-typescript 生成类型（见 §5.5），前后端接口边界有类型级保障——这是原生 JS 完全不具备的。

**备选**：
- **Vue 3 + Vite**：完全可用，与 React 同级别选型，中文社区模板多。若开发者在 Vue 上生产力更高，可选它，不影响任何架构决策（报告其余章节均与框架无关）。
- **原生 HTML + 少量 JS**：仅当 v0.1 想"两三天出原型"时作为起步过渡，**不建议作为正式实现基线**；从原型到正式版的重写成本会吃掉省下的构建成本。若坚持原生路线，至少用原生 Web Components 封装聊天/来源卡片等复用单元，但流式与状态管理仍需手写。

**版本口径**：React 19（以官方文档为准）、Vite 7（以官方文档为准）、TypeScript 5.x strict 模式。构建产物 target 设为 `es2020+` 即可覆盖 WebView（WebView2/WKWebView）与现代浏览器。

### 1.3 构建与部署流程

```
开发期（日常）：
  终端 A：uvicorn backend.main:app --port 8765        # FastAPI，含 /api/*
  终端 B：npm run dev                                   # Vite dev server :5173
  前端通过 Vite proxy 把 /api/* 转发到 8765（见 vite.config.ts server.proxy），
  前端所有请求同源书写（fetch("/api/...")），dev/prod 无需分支。

生产/本地启动：
  npm run build  →  dist/（纯静态）
  FastAPI 挂载 StaticFiles(html=True) 于 "/"，API 于 "/api/*"，
  启动后自动打开浏览器访问 http://127.0.0.1:<随机端口>。
```

关键点：**前端代码里所有请求一律用相对路径 `/api/*`，不写死主机/端口**。这是 dev（Vite proxy）与 prod（同源托管）能零切换的根本约定，也是未来 Tauri 复用（URL 或相对资源）的前提。

### 1.4 状态管理与数据层

- 全局状态（MVP）：一个 `AppContext`（导航 + 当前会话 id + 服务状态 + 全局 toast）。
- 问答会话状态：`useChat` hook（自研），内部用 `useRef` 维护累积文本缓冲 + `useState` 暴露渲染用快照；SSE 解析、AbortController、节流渲染都封装在此 hook 内。**不引入 Redux/Zustand**（YAGNI）。
- 任务进度（导入/同步）：`useTaskPolling(taskId)` hook，2s 轮询后端任务端点（见 §5.2）。任务轮询与聊天 SSE 分工明确：**低延迟链路（聊天）用 SSE 推送，分钟级操作（导入/同步）用轮询**，避免两套实时通道。

### 1.5 边界（v1.0 前不做）

- 不做 PWA / Service Worker / 离线缓存策略（本地服务本身离线，无需）。
- 不做前端本地 AI（向量/嵌入/切块全部在后端，前端零 AI 逻辑）。
- 不做多标签页协同 / BroadcastChannel 同步（单窗口个人工具）。
- 不做移动端完整适配（见 §4.4）。

---

## 2. 页面/视图结构（信息架构）

### 2.1 整体布局

```
┌─────────────────────────────────────────────────────────┐
│ 顶栏：品牌名 / 同步状态徽标（上次同步时间+差异） / 暗色切换 / 语言  │
├──────────┬──────────────────────────────────────────────┤
│ 左导航    │  主内容区（按路由渲染 5 个视图之一）             │
│ ● 问答    │                                              │
│ ● 导入    │                                              │
│ ● 同步    │                                              │
│ ● 设置    │                                              │
│ ● 隐私    │                                              │
├──────────┴──────────────────────────────────────────────┤
│ 页脚（仅状态类页面）：数据目录路径 / 服务 PID / 端口          │
└─────────────────────────────────────────────────────────┘
```

- 顶栏"同步状态徽标"：展示最近一次同步结果（如"3 分钟前 · 新增 2/更新 1/删除 0"），点击跳转同步视图。对齐 4.5 验收标准 1（启动自动检查后 UI 提示状态与结果）。
- 左导航每项带小徽标：如导入视图在有失败项时标红点、设置视图在配置不完整时标黄点（对齐 4.7 验收标准 3/4/10 的"缺失提示"）。

### 2.2 路由

Hash 路由（`#/chat`、`#/import`、`#/sync`、`#/settings`、`#/privacy`、`#/setup`）。理由：静态托管无需后端 SPA fallback；Tauri 加载本地资源时 hash 路由不依赖路径分发。实现可用 React Router（hashHistory）或手写 ~30 行 location.hash 监听——**MVP 建议手写极简路由**（5 个静态视图，无需嵌套/守卫），少一个依赖。

首次启动强制进入 `#/setup` 配置向导（由 `GET /api/status` 的 `config.wizard_completed=false` 驱动，对齐 4.7 验收标准 10：不进入半配置状态）。

### 2.3 各视图详设

#### 2.3.1 问答视图 `#/chat`（默认落地页）

核心组件清单：

| 组件 | 职责 | 关键交互 |
|---|---|---|
| `MessageList` | 渲染消息流，自动滚动到底部 | 流式时"吸底"滚动；用户上滚暂停吸底（防止中途查看历史被拉走） |
| `MessageItem`（user/assistant 两态） | 单条消息气泡 | 用户消息纯文本；助手消息走 Markdown 渲染（§3.2） |
| `AssistantAnswer` | 助手答案：Markdown 正文 + 来源卡片 + 状态 | 见 §3 |
| `SourceCard` | 来源标注卡片 | 点击 → 打开 `SourceDrawer`（见 §3.3） |
| `SourceDrawer`（侧滑抽屉） | 展示原文块 + 上下文 + 图片缩略图 | 数据来自 `GET /api/sources/{block_id}` |
| `ChatInput` | 多行输入框 | Enter 发送 / Shift+Enter 换行；发送中禁用；支持粘贴 |
| `StopButton` | 流式中显示"停止生成" | AbortController 断开 SSE，后端检测断开停止（对齐 4.6 第 9 条保留已流出部分） |
| `EmptyState` | 空库/无历史引导 | 知识库为空时展示"去导入数据"按钮（对齐 4.6 第 6 条） |
| `NoResultCard` | 低置信引导卡 | 展示"未检索到足够相关的知识" + 最相关片段 + 去同步按钮（对齐 4.6 第 5 条） |

关键交互：发送 → 消息上屏（用户）→ 请求 `POST /api/chat`（SSE）→ 助手消息占位"思考中…" → 流式增量渲染 → `sources` 事件落地来源卡片 → `done` 结束。会话为单会话（刷新后从 `GET /api/sessions/latest` 恢复，MVP 不做多会话列表，v0.2 再加会话侧栏）。

#### 2.3.2 导入视图 `#/import`

对齐 4.1 验收标准 1/2/3/8/9/10/11/12。两栏结构：**来源区** + **任务区**。

| 组件 | 职责 | 关键交互 |
|---|---|---|
| `SourcePicker` | 两种来源入口 | ①「选择目录」→ 打开 `DirectoryBrowser`；②「拖拽区」→ 拖入 `.md` 文件（可多个） |
| `DirectoryBrowser`（模态） | 目录树选择器 | 数据来自 `GET /api/fs/browse`；展示层级、可展开；选择后确认回填路径；非 MD 目录给提示 |
| `DragDropZone` | 拖拽上传 | 过滤非 `.md`（对齐 4.1 验收 11：提示"当前版本仅支持 Markdown"且不中断批次）；上传即入 `POST /api/import/upload` |
| `ImportProgress` | 任务进度条 | 轮询任务端点；展示 已完成/总数、成功/失败/跳过、当前文件名 |
| `ImportReport` | 完成报告 | 成功/失败/跳过统计 + 失败项表格（文件名 + 可读原因）；"重试失败项"按钮 → `POST /api/import/task/{id}/retry`（对齐 4.1 验收 9） |
| `DedupNotice` | 去重提示 | 重复导入提示"已存在，跳过"（对齐 4.1 验收 10） |

关键交互：选目录 → 展示"将扫描 N 个 .md 文件（含子目录），并启用增量同步"确认 → 注册目录并触发导入 → 进度 → 报告。拖拽文件 → 上传（分块进度）→ 入受管目录 → 同管线。

#### 2.3.3 同步视图 `#/sync`

对齐 4.5 验收标准 2/12。核心是**差异预览 → 确认执行 → 结果**三步。

| 组件 | 职责 | 关键交互 |
|---|---|---|
| `SyncActionBar` | 手动触发同步 | "同步预览"按钮 → `POST /api/sync/dry-run` |
| `SyncDiffPreview` | 差异预览 | 展示 **新增 N / 更新 M / 删除 K** 三组文件列表；删除组红显（对齐 4.5 验收 12 差异摘要） |
| `SyncDiffItem` | 单个差异项 | 文件路径 + 变更类型徽标；路径移动项标注"原路径移除 + 新路径新增"（对齐验收 13） |
| `SyncConfirmDialog` | 确认执行 | 展示差异摘要 → 确认 → `POST /api/sync/run` |
| `SyncProgress` | 同步进度 | 与 ImportProgress 复用同一 `TaskProgress` 组件（统一任务模型，见 §5.2） |
| `SyncReport` | 结果 + 失败重试 | 新增/更新/删除/跳过计数 + 失败清单 + 重试按钮（对齐验收 11） |
| `SyncStatusBadge`（顶栏） | 常驻同步状态 | 启动时自动检查（对齐验收 1）的异步结果在此展示 |

关键交互：首次进入先调 `GET /api/status` 拿 `last_sync_at`；用户点预览 → dry-run（毫秒级指纹比对）→ 差异表 → 确认 → run → 进度 → 报告。**dry-run 与 run 分离**确保"预览不落库、确认才变更"。

#### 2.3.4 设置视图 `#/settings`（含配置向导）

两块：**首次配置向导（`#/setup` 全屏）** 与 **模型配置管理（设置页内）**。

配置向导步骤（对齐 4.7 验收 1）：① 数据接入（选目录/拖拽）→ ② 嵌入模型 → ③ 对话模型 → ④（可选）识图模型 → ⑤ 完成预览。

| 组件 | 职责 | 关键交互 |
|---|---|---|
| `SetupWizard`（全屏步骤容器） | 步骤导航 + 每步校验 | 顶部步骤条；上一步/下一步；未完成阻塞进入下一步 |
| `StepDataSource` | 目录选择 + 拖拽（复用导入视图组件） | 同 §2.3.2 |
| `ModelConfigForm` | 单个模型配置：provider 切换（本地/云端）+ base_url + model + key + 开关 | provider 切换处展示隐私影响提示（见下） |
| `ProviderToggle` | 本地/云端切换位 | 切换云端 → 弹隐私提示 + 显式确认（对齐 4.7 验收 6/7、4.8 验收 3） |
| `KeyInput` | API Key 输入（掩码显示） | 仅存内存 + 掩码；后端不落明文（对齐技术架构 §5.3 配置脱敏） |
| `TestConnectionButton` | 校验可达性 | `POST /api/config/test`；失败给出可读错误并阻止保存（对齐 4.7 验收 8） |
| `ModelImpactNotice` | 变更连带影响提示 | 改嵌入模型 → 提示需重建索引（对齐验收 9） |
| `VisionSwitch` | 识图功能开关 | 开启 → 要求配置识图模型；关闭 → 跳过该步（对齐验收 3/4/5） |
| `SetupCompletePreview` | 完成预览 | 汇总：数据目录 + 三模型 provider + 出网状态 | 

#### 2.3.5 隐私/数据流向视图 `#/privacy`

对齐 4.8 验收标准 1/3/4/5/6/8/9/10。**网络审计面板**在此落地。

| 组件 | 职责 | 关键交互 |
|---|---|---|
| `PrivacyStatementCard` | 首次启动隐私承诺 | 首次进入展示数据本地化原则/例外场景/删除能力，需滚动到底确认（对齐验收 10） |
| `DataFlowPanel`（核心） | 数据流向视图 | 可视化：本机数据目录 → 各处理环节（切块/嵌入/向量库/检索/生成）→ 各模型出口；每出口标注本地/云端 + 是否外发 |
| `OutboundAuditTable` | 出网清单 | 表格：配置场景 / 出网数据 / 出网时机 / 默认 / 当前状态（对齐技术架构 §5.2 审计表） |
| `ProviderStatusList` | 各模型 provider 状态 | 嵌入/对话/识图各自 本地或云端，云端时红色"出网"徽标 |
| `KnowledgeExportButton` | 一键导出备份 | `POST /api/knowledge/export`（对齐验收 5） |
| `KnowledgeDeleteButton` | 一键删除知识库 | 二次确认对话框（输入"DELETE"或勾选确认）→ `POST /api/knowledge/delete`（对齐验收 4） |
| `LogClearButton` | 清除日志 | `POST /api/logs/clear`（对齐验收 7） |

关键交互：`DataFlowPanel` 的箭头在"全本地"时全绿，任一模型切云端后对应箭头变红并出现"云端发送 X"标签。数据来自 `GET /api/privacy/audit`（单端点聚合，前端只负责呈现）。

---

## 3. 问答界面细节

### 3.1 消息列表

- 用户消息（右对齐气泡）：纯文本，React 文本节点渲染（自动转义）。
- 助手消息（左对齐宽气泡）：Markdown 渲染 + 底部来源卡片区。
- 系统消息（居中窄条）：如"正在同步知识库…"、"未检索到相关内容"等非问答提示。
- 消息结构（前端模型）：
  ```ts
  type ChatMessage =
    | { id: string; role: "user"; content: string; created_at: string }
    | { id: string; role: "assistant"; content: string; sources: SourceInfo[]; status: "done" | "error" | "no_result"; created_at: string };
  type SourceInfo = { block_id: string; file_path: string; title_path?: string; platform?: string; time?: string; snippet: string; relevance?: number };
  ```
- 自动滚动：`MessageList` 监听新内容追加，`scrollToBottom` 平滑滚动；用户手动上滚超过阈值（如 80px）暂停吸底，恢复吸底需点"回到底部"浮动按钮。

### 3.2 流式渲染方案

**传输**：`POST /api/chat`（非 GET），响应 `text/event-stream`，用 `fetch + ReadableStream` 手写 SSE 解析（原生 `EventSource` 只支持 GET，多轮对话上下文可能很长，GET query 有 URL 长度限制；POST 同时天然获得 `AbortController` 取消能力）。

解析器要点（自研 `parseSSE.ts`，~80 行）：按行读取 `event:` / `data:`，`data:` 按换行拼接后 `JSON.parse`；`retry:`/`id:` 字段忽略；连接关闭时清理。事件协议见 §5.3。

**渲染策略（防卡顿三档，逐级启用）**：

1. **MVP 基线**：流式期间把当前累积文本放进独立 `StreamingMessage` 组件的局部 state，**只重渲染该组件**（历史消息用 `React.memo` 隔离），约 100–150ms 节流更新一次；用 markdown-it 渲染整段（答案通常 <5KB，个人机器可承受）。
2. 若实测吐字渲染掉帧（对齐技术架构 §7 "性能假设实测校准"原则）：流式期间改纯文本渲染（不做 Markdown 解析），`done` 后一次性渲染。
3. 长答案再优化：按段落缓存已完成段落的 HTML（只重新渲染当前活跃段）。

不做"虚拟列表"（消息量小）；不逐 token 更新整个列表状态。

**流式中断**：模型/服务异常时后端发 `event: error`（见 §5.3），前端保留已流出文本、在消息下方展示错误条 + "重试"按钮（对齐 4.6 验收 9）。

### 3.3 来源标注卡片

**设计**：强制出现在每个助手答案底部（对齐 4.6 验收 3）。卡片展示：来源类型徽标（文件/对话/平台）+ 文件名或对话标题 + 标题路径 + 平台 + 时间 + 相似度（可选）。**点击卡片 → 打开 `SourceDrawer` 展示原文块与上下文**——注意"跳转原文"是应用内抽屉，不是打开系统文件（浏览器无法 `file://` 打开本地文件；见 §6.1）。

```
┌─ SourceDrawer ────────────────────────────┐
│ [文件] 笔记/项目/架构决策.md                │
│ 路径: ~/notes/项目/架构决策.md             │
│ 平台: 本地文档 ｜ 时间: 2026-08-01         │
│ ───────────────────────────────────────── │
│ （原文块文本，Markdown 渲染）              │
│ ┌─────── 上下文（上一段）────────────────┐ │
│ ┌─────── 上下文（下一段）────────────────┐ │
│ [图片缩略图]（若有，点击看大图）             │
└───────────────────────────────────────────┘
```

**数据流**：`sources` 事件携带 `SourceInfo[]` → 渲染卡片（仅展示字段文本，全部转义）；点击 → `GET /api/sources/{block_id}` → 抽屉渲染原文 + `context.prev/next` + 图片缩略图（`/api/media/thumbnail/{image_block_id}`）。

**内嵌引用标记（可选增强，v0.2）**：答案正文中 `[n]` 引用模型是否内嵌由提示词驱动，后端做**规范化校验**（正则提取 `[n]`，n 必须落在来源列表范围内才渲染成可点击徽标，越界/伪造的 `[n]` 降级为纯文本），防止 LLM 编造引用。MVP 不强依赖内嵌标记，**卡片列表是强制、可靠的主标注**。

### 3.4 多轮对话上下文

- 前端职责：维护 `messages[]`，把**完整历史**随 `POST /api/chat` 的 `messages` 字段发给后端，不做摘要（对齐 4.6 验收 7）。
- 后端职责（边界明确在前端之外）：历史截断/摘要策略由后端决定（保留相关历史或最近 N 轮，N 可配置，对齐验收 8），前端不感知。
- 会话恢复：`GET /api/sessions/latest` 取最近会话的消息历史，刷新页面后还原对话现场；会话持久化由后端 SQLite 承担（对齐技术架构 §1.4 `state.db`）。
- 边界：MVP 单会话（无会话列表/重命名/删除入口），多会话管理 v0.2。

### 3.5 空结果/低置信引导

后端在检索阶段判定（对齐 4.6 验收 5/6）：知识库为空、检索结果为空、或 top-k 相关度低于阈值时，不发 `delta`，改发 `status`（no_result）+ `sources`（携带最相关片段）→ `done`。

前端对应渲染 `NoResultCard`（非编造答案）：
- 主文案："未检索到足够相关的知识"。
- 附最相关片段（`best_snippets`，供用户自行判断）。
- 操作：`[去同步数据]` 跳转同步视图、`[去导入]` 跳转导入视图。
- 知识库为空时由前端 `EmptyState` 提前拦截（从 `GET /api/status` 的 `knowledge.file_count === 0` 判断），甚至不发起生成请求。

---

## 4. 组件/样式方案

### 4.1 样式策略：自研 CSS（设计令牌 + 语义化类）

**推荐**：CSS 变量设计令牌 + 语义化组件类 + 每视图一个样式文件。不用 CSS-in-JS。

理由：
1. **CSP 友好是硬约束**：严格 CSP 的 `style-src 'self'` 与 CSS-in-JS（运行时注入 `<style>`）冲突，需要 `'unsafe-inline'`。独立 CSS 文件天然兼容（见 §6.2）。
2. **组件库 80% 能力用不上**：本 UI 只有聊天、表单、列表、对话框、进度条、徽标，无复杂表格/日期选择器/虚拟滚动需求；引入 Ant Design 等主要获得包体负担（~1MB gzip）与强绑定设计风格。
3. **零第三方依赖 + 离线优先**：符合"数据不出机"与供应链最小化（技术架构 §5.4 依赖锁定）。
4. **个人工具审美**：自绘可控的克制风格，比套通用组件库更贴合"本地第二大脑"的质感。

设计令牌（`tokens.css`）示例：
```css
:root {
  --color-bg: #fafafa; --color-bg-elevated: #ffffff; --color-bg-chat: #f2f2f5;
  --color-text: #1a1a1a; --color-text-secondary: #6b6b70;
  --color-border: #e5e5e8; --color-primary: #4f6ef7; --color-primary-hover: #3d5ce5;
  --color-success: #22a06b; --color-warning: #f5a623; --color-danger: #e5484d;
  --radius-sm: 6px; --radius-md: 10px; --radius-lg: 14px;
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px; --space-6: 24px;
  --font-ui: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
```

**备选**：Tailwind CSS v4（Vite 插件，构建期生成 CSS、无运行时、CSP 兼容）——若开发者偏好 utility-first 快速堆叠，可用它替代手写原子类，但需接受 class 冗长与版本迭代成本。shadcn/ui 仅为风格备选（源码复制 + Tailwind，无运行时），不推荐引入完整组件库。

### 4.2 自研 UI 原子（`ui/` 目录）

| 原子 | 用途 |
|---|---|
| `Button`（primary/secondary/danger/ghost + loading） | 全站按钮 |
| `Input` / `Textarea` | 表单与 ChatInput |
| `Card` / `CardHeader` | 卡片容器 |
| `Dialog`（模态 + 遮罩 + ESC 关闭） | 确认框、目录选择器 |
| `Drawer`（侧滑） | SourceDrawer |
| `Toast` | 全局轻提示 |
| `Tag` / `Badge` | 来源类型徽标、同步状态点 |
| `Progress` | 导入/同步进度 |
| `Tabs`（同步/设置视图内） | 差异分组切换 |
| `EmptyState` | 空库引导 |
| `Icon`（内联 SVG 组件集） | 图标（避免 iconfont 依赖；`font-src 'self'` 下零风险） |

### 4.3 暗色模式

- 实现：`[data-theme="dark"]` 覆盖同名 CSS 变量（复用 token，仅换色值）。
- 默认：跟随系统 `prefers-color-scheme`；提供手动切换（顶栏按钮），选择存 `localStorage`；`<html>` 上写 `data-theme`。
- 渲染前避免闪烁：`index.html` 内嵌一行 `?theme=` 读取 localStorage 的脚本（该脚本是唯一允许的内联脚本，见 §6.2 CSP 兼容处理——用非阻塞策略或放行该单一脚本）。

### 4.4 响应式（桌面为主）

- 基线：≥1024px 桌面窗口全功能布局。
- 窄窗口断点（≤768px）：左导航折叠为窄图标栏（或抽屉）；聊天消息气泡全宽；SourceDrawer 全屏化。仅保证"可用"，不做精致移动适配。
- 非目标（显式）：手机端、触控手势、PWA 安装。边界写进 v1.0 不做清单。

### 4.5 中文 i18n（首发）与英文预留

**推荐自研极简 i18n**（不引入 react-i18next）。结构：

```
src/i18n/
  index.ts         // t() 查表 + useI18n() hook + I18nProvider + locale 切换
  messages/zh.ts   // 中文全量（MVP 首发完整）
  messages/en.ts   // 英文（v0.1 骨架：核心文案补齐，未翻译 key 回退 zh 并 console.warn 标记）
```

- key 约定：`namespace.key.sub`，如 `chat.input.placeholder`、`import.drag.tip`、`privacy.audit.table.header.outbound_data`。
- **硬约束：所有用户可见字符串必须走 `t()`，禁止硬编码**——这是"英文预留"的真正意义（key 化 + 资源文件结构，而非翻译完整性）。
- 语言切换：顶栏下拉（中文/English），选择存 localStorage + 写入 `<html lang>`。
- 特殊处理：
  - 来源相关信息（文件名、路径、平台名、对话标题）**不翻译**，保持原文展示。
  - 时间格式用 `Intl.DateTimeFormat(locale)` + 中文习惯（"今天 14:32"、"8月1日"）；英文 locale 用 `Intl.RelativeTimeFormat`。
  - 数字/千分位用 `Intl.NumberFormat(locale)`。
  - SSE 流式、进度等动态文案同样 key 化。

---

## 5. 与后端接口边界（API 契约草案）

### 5.1 前后端解耦与托管

- **同源**：前端相对路径 `/api/*`，dev 走 Vite proxy、prod 同源静态托管，前端代码零分支。
- **静态托管**：FastAPI `StaticFiles(html=True)` 挂 `/`，服务 `dist/`；API 全在 `/api/*` 前缀下。端点只返回数据，不渲染 HTML 片段（前后端彻底分离，利于 Tauri 复用）。
- **一个必须说清的架构边界：浏览器读不到本地绝对路径，因此"选目录"与"拖拽"是两条语义不同的导入路径**：

| 导入方式 | 谁能拿到什么 | 落地语义 | 与"不拷贝原始文件"的关系 |
|---|---|---|---|
| **路径模式（选目录）** | 后端扫描用户输入/选择的**绝对路径**目录 | 就地引用、路径入元数据 | ✅ 完全一致（推荐主路径，也是增量同步的对象） |
| **拖拽/上传模式** | 浏览器只能拿**文件内容**（File API），拿不到绝对路径 | 内容 POST 上传 → 后端写入受管目录（如 `~/.everything-rag/uploads/`） | ⚠️ 本质是"复制进受管空间"，非就地引用 |

  落地结论：MVP **两种都做**（4.1 验收标准 1/2 明确要求拖拽与选目录）。目录选择器必须由后端枚举文件系统（`GET /api/fs/browse`），前端展示树；拖拽走 multipart 分片上传。此边界须同步给后端，避免"拖拽后文件到底存哪"的歧义。

### 5.2 REST 端点清单（草案）

统一响应约定：成功直接返回数据体；错误统一 `{ "error": { code, message, detail?, request_id } }`（见 §5.4）。

| 方法 | 路径 | 请求/响应要点 | 用途 |
|---|---|---|---|
| GET | `/api/status` | → `{ app:{version,pid,started_at}, config:{wizard_completed, embed_configured, chat_configured, vision_configured, vision_enabled}, knowledge:{file_count, chunk_count, image_count, last_sync_at, index_version, needs_rebuild}, privacy:{outbound_state, providers} }` | 首屏、向导入口、空库判断、索引重建提示 |
| GET | `/api/config` | → 脱敏配置（key 全掩码为占位符） | 设置页回显 |
| PUT | `/api/config` | body 模型配置 + 开关；校验失败 422 带可读错误 | 保存配置 |
| POST | `/api/config/test` | body `{model_role:"embed"\|"chat"\|"vision"}` → `{ok, latency_ms?, error?}` | 连接可达性测试 |
| GET | `/api/fs/browse?path=` | → `{path, name, is_dir, children:[...]}`（层级懒加载） | 目录选择器 |
| POST | `/api/import/path` | body `{paths:["/abs/dir"]}` → `{task_id}` | 注册目录 + 触发导入/扫描 |
| POST | `/api/import/upload` | multipart（`files[]`，分片可选）→ `{task_id, accepted, rejected:[{name,reason}]}` | 拖拽上传到受管目录 |
| GET | `/api/import/task/{id}` | → 任务进度（见下方统一任务模型） | 轮询 |
| GET | `/api/import/task/{id}/report` | → `{success:[...], failed:[{file,reason,retryable}], skipped:[{file,reason}]}` | 完成报告 |
| POST | `/api/import/task/{id}/retry` | → 新任务 `{task_id}` | 仅重试失败项 |
| GET | `/api/knowledge/trees` | → 已注册知识目录列表（路径、状态、文件数） | 导入/同步/隐私视图共用 |
| POST | `/api/sync/dry-run` | → `{task_id}`（或同步返回 `{added:[],updated:[],deleted:[],moved:[],stats}`） | 差异预览 |
| POST | `/api/sync/run` | → `{task_id}` | 确认执行同步 |
| GET | `/api/sync/task/{id}` | → 同步任务进度 + 差异结果 | 轮询 |
| POST | `/api/sync/task/{id}/retry` | → 新任务 | 重试失败项 |
| POST | `/api/chat` | body `{messages:[...], session_id?}` → SSE（§5.3） | 问答主链路 |
| GET | `/api/sessions/latest` | → 最近会话消息历史（或空） | 刷新恢复对话现场 |
| GET | `/api/sources/{block_id}` | → `{block_id, text, context:{prev,next}, meta:{file_path,title_path,platform,time,role}, images:[{block_id, thumbnail_url}]}` | 来源跳转原文抽屉 |
| GET | `/api/media/thumbnail/{image_block_id}` | → 图片（缩略图） | 原文抽屉缩略图 |
| GET | `/api/media/image/{image_block_id}` | → 原图 | 点击看大图（校验 image_path 在已注册目录内，防任意文件读取） |
| GET | `/api/privacy/audit` | → `{data_dir, models:[{role, provider, local\|cloud, outbound}], outbound_scenarios:[{scenario, data, timing, default, current}], has_cloud_call_history}` | 数据流向审计面板 |
| POST | `/api/knowledge/export` | → `{export_path}`（服务端生成可移植包） | 备份/迁移 |
| POST | `/api/knowledge/delete` | body `{confirm:true}` → `{ok}` | 删除知识库（二次确认参数） |
| POST | `/api/logs/clear` | → `{ok}` | 清除日志 |

**统一任务模型**（导入/同步共用）：
```json
{
  "task_id": "uuid",
  "type": "import|sync",
  "status": "pending|running|done|failed|cancelled",
  "created_at": "ISO8601",
  "progress": { "done": 12, "total": 40, "success": 11, "failed": 1, "skipped": 0, "current_file": "notes/a.md" },
  "report": null
}
```
前端 `useTaskPolling` 2s 轮询；`status === "done"` 后拉取 `.../report`。

### 5.3 SSE 聊天协议（`POST /api/chat`）

请求：
```json
{
  "messages": [
    { "role": "user", "content": "我上次关于迁移数据库说了什么？" }
  ],
  "session_id": "uuid (可选)"
}
```
响应：`Content-Type: text/event-stream; charset=utf-8`。事件序列（按需发生）：

| event | data 内容 | 说明 |
|---|---|---|
| `meta` | `{ "chunk_count": 5, "query_embed_ms": 120 }` | 检索元信息（首事件） |
| `status` | `{ "code": "generating" \| "no_result" \| "low_confidence", "message": "...", "best_snippets": [...] }` | 生成状态切换；no_result/low_confidence 时**不发 delta**，随后直接 done |
| `delta` | `{ "text": "你" }` | 文本增量，逐段推送 |
| `sources` | `SourceInfo[]` | 最终来源标注（答案流结束前必发一次，对齐"强制标注来源"） |
| `done` | `{ "answer_id": "...", "session_id": "...", "usage": {...}, "truncated_history": false }` | 正常结束 |
| `error` | `{ "code": "model_unavailable" \| "generation_failed", "message": "...", "request_id": "..." }` | 异常结束；已流出的 delta 前端保留 |

- **取消**：客户端 `AbortController.abort()` 断开连接；后端检测客户端断开即停止生成（对齐技术架构 §3.4"SSE 连接断开即停止生成"）。
- **no_result 判定在后端**（检索空/低阈值），前端只按事件渲染。
- **来源必达保障**：前端约定 `done` 事件前如未见 `sources` 则标记该答案"来源缺失"并显示警示（防后端遗漏造成"无来源答案"，兜底 4.6 验收 3）。

### 5.4 错误格式

统一错误体（REST 与 SSE `error` 事件一致）：
```json
{
  "error": {
    "code": "EMBED_MODEL_NOT_CONFIGURED",
    "message": "嵌入模型未配置，请前往设置完成配置",
    "detail": { "hint": "/#/settings" },
    "request_id": "abc123"
  }
}
```
HTTP 状态码约定：
| 码 | 场景 |
|---|---|
| 400 | 参数错误（缺字段/类型错误） |
| 404 | 资源不存在（task/block/session） |
| 409 | 冲突（同步进行中再次触发、上传与路径去重冲突） |
| 422 | 配置校验失败（含可达性校验失败） |
| 500 | 内部错误（detail 仅含非敏感结构信息，对齐 4.8 日志脱敏） |
| 503 | 依赖不可用（本地模型未启动/云端端点不可达） |

前端 `api.ts` 封装统一处理：`request_id` 展示在错误提示角标，供排障（对齐技术架构 §5.3 错误信息不泄露敏感数据）。

### 5.5 契约同步（OpenAPI → TypeScript）

- 后端 FastAPI 自动产出 `/openapi.json`。
- 前端用 **openapi-typescript** 在 CI/发布脚本中从 `/openapi.json` 生成 `src/api/schema.d.ts`，SSE 事件类型在 `schema.d.ts` 外单独手写（SSE 不属 OpenAPI 标准，约定在契约文档即本文件 §5.3）。
- 分工：**REST 端点类型由后端产出驱动，SSE 事件协议由本文件作为唯一权威**，前后端均引用。可避免两端对端点字段漂移。

---

## 6. 前端安全

> 基线（技术架构 §5.4）：本地 Web UI 防 XSS——检索上下文与模型输出可能包含不可信文本（来源是用户自己的文件/第三方对话导出，可能是恶意 HTML/脚本），前端渲染必须转义，Markdown 用白名单解析器。

### 6.1 XSS 防护

**渲染原则（三条铁律）**：
1. **一切文本走 React 文本节点渲染**（`{text}`），默认自动转义。来源卡片中的文件路径、对话标题、平台名、摘要文本全部按此处理——即使路径含 `<script>` 也只是一串文本。
2. **Markdown 用白名单渲染**：markdown-it 开启 `html: false`（丢弃内联 HTML）；链接只保留 `http(s)` 协议（`linkify` 自定义校验 + 渲染为 `target="_blank" rel="noopener noreferrer"`）；即使关闭 HTML，仍过一层 **DOMPurify**（`ALLOWED_TAGS` 白名单：p/strong/em/code/pre/blockquote/ul/ol/li/a/table/thead/tbody/tr/td/th/h1-h6/img 等）作为纵深防御。**永不把未经 DOMPurify 消毒的内容喂给 `dangerouslySetInnerHTML`**。
3. **图片不直连本地路径**：不渲染 `file://`（浏览器禁），不渲染任意 `http(s)` 远程图（违反不出机 + 引入跟踪像素/XSS 面）。图片一律走同源代理 `/api/media/image/{block_id}`（`<img src="/api/media/...">`），由后端按元数据读取并校验路径在已注册知识目录内——同时杜绝任意文件读取（路径穿越）。

**其他**：
- 用户文档中的外链 URL：仅存入元数据并展示为可点击链接；点击用 `http(s)` 协议白名单校验，其他协议（`file:`/`javascript:`）降级为不可点击纯文本。
- 文档内容中出现 `<img src="http://...">` 等内联引用：markdown-it `html:false` + DOMPurify 白名单过滤后不会渲染远程图；图片引用（`![](...)`）经图片管线后统一走 `/api/media/`。
- 来源卡片点击打开 `SourceDrawer`，原文块文本同样走第 1、2 条渲染路径。

### 6.2 CSP 建议

推荐指令集（本地同源静态，极简且强）：
```
default-src 'self';
script-src 'self';
style-src 'self';
img-src 'self' data:;
connect-src 'self';
font-src 'self';
object-src 'none';
base-uri 'none';
frame-ancestors 'none';
form-action 'self';
upgrade-insecure-requests
```
通过响应头下发（FastAPI `Response` header 或 Starlette 中间件），对 `/api/*` 与静态资源统一生效。SSE 属 `connect-src 'self'`，同源天然放行。

**与工具链的兼容注意（必须落实，否则上线即破 CSP）**：
1. **Vite 默认的 modulepreload polyfill 是内联脚本**，会违反 `script-src 'self'`。处理：`build.modulePreload.polyfill = false`（产物面向现代浏览器/WebView，无需 polyfill），即可保持 `script-src 'self'` 不含 `'unsafe-inline'`。
2. **禁用 CSS-in-JS**（§4.1），否则需 `style-src 'unsafe-inline'`。
3. **避免 `eval()`/`new Function`**：现代 React/markdown-it/DOMPurify 生产构建均不使用；注意不引入需要 eval 的库。React 开发模式有 eval（sourcemap），生产无碍——CSP 仅对生产生效。
4. 暗色主题防闪烁的 `<html>` 内联脚本（§4.3）：与第 1 条冲突。两种处理任选：a) 该脚本改为 `src` 引用的外置脚本（首屏就绪前执行，`<script src="/theme.js" async>`）；b) 接受 `script-src 'self' 'unsafe-inline'` 并只允许该单点内联（不推荐，弱化基线）。**推荐 a**。
5. 若未来 Tauri 复用：Tauri WebView 同样遵守注入的 CSP；capability 配置需允许加载同源/本地资源。

**CSP 之外**：
- 依赖锁定（lockfile）+ 发布前漏洞扫描（技术架构 §5.4）。
- 不引入任何第三方 CDN 运行时资源（离线优先，也避免扩大 CSP 白名单）。
- 隐私：`/api/media/*` 与 `/api/sources/*` 均校验数据目录归属，杜绝"前端任意路径读取本机文件"的本地攻击面（浏览器同源沙箱 + 后端路径校验双保险）。

---

## 7. 决策边界与风险清单

| 风险/边界 | 说明 | 缓解 |
|---|---|---|
| **拖拽导入 ≠ 就地引用** | 浏览器拿不到绝对路径，拖拽必然"内容上传到受管目录" | 已在 §5.1 明确双路径语义；需与后端对齐"受管目录"路径与清理策略 |
| **来源跳转是应用内抽屉** | 浏览器无法打开本地文件 | 已在 §3.3 定义 SourceDrawer + `/api/sources/{block_id}`，v1.0 套 Tauri 后可加"在文件管理器中显示"（需 Tauri 插件，届时再议） |
| **CSP 与工具链** | Vite modulepreload polyfill / CSS-in-JS / 内联 theme 脚本都会破 CSP | §6.2 三条兼容注意落实进脚手架 |
| **构建工具链门槛** | 引入 Node 工具链，与"无构建复杂度偏好"张力 | Vite dev 无手动构建感；`build` 仅发布时执行；`package.json` 脚本固化 |
| **流式渲染性能** | 长答案 + 高吐字速率可能掉帧 | §3.2 三档降级策略 + 架构 §7 实测校准原则 |
| **SSE 来源必达** | 后端遗漏 sources 事件 → 无来源答案 | 前端 done 前校验 + 警示兜底（§5.3） |
| **远程图片/外链** | 文档内容可能引用远程资源 | 一律不过网：图片走本地代理，链接 http(s) 白名单 + rel 安全 |
| **Tauri 复用** | 浏览器特定 API 可能不适用于 WebView | 只用标准 Web API（fetch/EventSource/File）；hash 路由；相对路径请求 |

**MVP 明确不做（前端侧）**：多会话列表、答案内嵌 `[n]` 引用（v0.2）、PWA、移动端适配、远程访问鉴权 UI、WebSocket 通道。
