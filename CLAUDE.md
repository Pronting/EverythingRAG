# CLAUDE.md — Everything RAG 个人知识第二大脑

> 本文件是项目**总纲**，让新 Agent 打开即能掌握全貌。技术细节以链接文档为准，本文件只给"地图 + 规矩"。

## 一、项目一句话

把「本地文档 + 各 AI 平台（ChatGPT / Kimi / DeepSeek）对话记录」统一向量化，用自然语言问答检索的个人知识第二大脑；**纯本地运行、答案永远带来源**。

## 二、关键文档（新 Agent 必读，按顺序）

| 优先级 | 文档 | 内容 |
|---|---|---|
| ★必读 | `docs/PRD/PRD.md` | 产品需求文档（13 章定稿）：背景/定位/功能/技术架构/MVP/路线图/验收/风险/决策 |
| ★必读 | `docs/tech/00-技术选型决策文档.md` | 技术选型底稿：栈总表 + 6 路细节 + 冲突裁决 + 边界清单 |
| 按需 | `docs/tech/T1~T6-*.md` | 各技术域详细报告（RAG框架/前端/流式/向量库/识图/配置部署） |
| 存档 | `docs/PRD/sections/`、`docs/PRD/PRD-base.md` | sub-agent 原始产出与基础框架，一般不必读 |

## 三、产品要点（快速理解）

- **定位**：个人知识第二大脑；MVP 个人使用 → 开源产品。
- **部署**：纯本地（数据不出本机）。
- **核心场景**：统一可检索、可回溯的问答（跨平台对话 + 本地文档），答案带来源。
- **诚实定位**：「检索找回我的一切」，不是「AI 知道我的一切」。
- **目标用户**：多平台重度 AI 用户 + 有本地文档的知识工作者（先服务作者这一类人）。

### 8 条既定关键决策（详见 PRD 13.1，不得违背）
1. 纯本地部署，隐私是信任分水岭
2. 数据接入 = 第三方导出 → **Markdown 归一化**；不绑 token/API/具体工具；外链仅存元数据不抓取
3. 图片**方案 B**：识图 → 文字描述 → 文本嵌入（检索主干）；原图路径存元数据；问答默认不调识图
4. 增量同步：手动「同步」按钮 + 启动检查；指纹比对；**对话级去重键**（平台+规范化内容，非文件 hash）；语义切块防边界漂移
5. 三模型配置：嵌入/对话必配，识图条件性配置（功能开关联动）
6. MVP 只认 Markdown
7. 诚实定位（见上）
8. MVP 个人 → 开源

### 4 项待拍板（PRD 12.2，不阻塞 MVP 关键路径）
① 识图本地 vs 云端（技术推荐：本地默认）② 对话边界/去重键规范 ③ Dashboard 形态（技术推荐：本地 Web）④ 开源许可（技术推荐：Apache-2.0）

## 四、技术栈（已锁定，详见技术选型决策文档）

| 层 | 选型 |
|---|---|
| 后端 | **Python 3.11 + FastAPI + asyncio** |
| 向量库 | **Chroma**（进程内，单一 collection 含模型指纹，如 `chunks__bge-m3__v1`） |
| 嵌入 | **bge-m3 本地 fastembed**（ONNX，int8 约 600MB，惰性加载）；备选 gte-multilingual-base |
| 对话/识图模型 | **OpenAI 兼容可插拔**（默认本地 Ollama/llama.cpp，可选云端）；识图 = 本地 VLM（Qwen2-VL/MiniCPM-V 方向） |
| 前端 | **React 19 + Vite + TypeScript**，Hash 路由，静态 `dist/` 由 FastAPI 托管 |
| 流式 | **SSE**：`POST /api/chat` + 前端 `fetch/ReadableStream` 手写解析；帧契约 meta/token/done/error |
| 状态存储 | **SQLite（aiosqlite）**：指纹表/去重键表/切块清单/会话 |
| 部署/交付 | 本地 Web 服务 `127.0.0.1:随机端口`（MVP）；**开源交付 Tauri v2 桌面客户端（Windows/macOS）+ Python sidecar**（见技术选型决策文档 §7） |
| 隐私 | **唯一 `OutboundClient`** 收敛所有出网 + 网络审计面板 + 日志脱敏 |

**RAG 框架**：**自研轻量管线**（markdown-it-py 解析 + 自研切块/同步/去重），**不引入** langchain / llama-index。

## 五、目录结构

**当前结构**（脚手架已搭建，2026-08）：
```
everything-rag/
├── CLAUDE.md
├── .claude/
│   ├── settings.json            # hooks / auto-memory / 插件（可提交）
│   ├── settings.local.json      # 个人权限覆盖（gitignore）
│   ├── project-state.md         # 当前阶段 + 验收点（SessionStart 自动注入）
│   └── hooks/                   # session_start / test_gate / stop_check
├── backend/                     # Python 后端（FastAPI）
│   ├── app/                     # api/core/ingestion/models/vectorstore/retrieval/generation/util
│   ├── tests/                   # 冒烟测试
│   ├── pyproject.toml
│   └── .venv/                   # 项目虚拟环境（gitignore）
├── frontend/                    # React 19 + Vite + TS
│   ├── src/
│   └── package.json
├── scripts/run.py               # 启动：动态端口 + 自动开浏览器
└── docs/
    ├── PRD/  tech/              # 见第二节
```

**目标结构**（完整实现后）：
```
backend/            # Python 后端
├── app/
│   ├── api/        # 接口层（含 SSE）
│   ├── core/       # 配置中心 / 日志 / 审计
│   ├── ingestion/  # 数据接入 / 切块 / 识图 / 增量同步
│   ├── models/     # 数据模型 / 元数据 schema
│   ├── vectorstore/# Chroma 适配
│   ├── retrieval/  # 检索
│   ├── generation/ # 问答生成（流式）
│   └── util/
├── tests/
└── pyproject.toml
frontend/           # React 前端
└── src/
scripts/            # 启动 / 安装脚本
```

## 六、开发工作流（TDD 强制）

1. **先写测试（RED）** → 最小实现（GREEN）→ 重构（IMPROVE）。每个功能都必须有**单元测试**，关键流程配 **E2E**。
2. 后端单测用 **pytest**；前端组件测试 + E2E 用 **Playwright**。
3. 前端调试：**用 Playwright MCP**（已启用）打开 `http://127.0.0.1:<port>`，用 snapshot/find/click 验证界面交互，用 evaluate 断言状态。
4. 代码完成后用 **code-reviewer** agent 审查，CRITICAL/HIGH 必须修复。
5. 提交用 conventional commits（feat/fix/refactor/docs/test/chore/perf/ci）。
6. **测试命令**：后端 `(cd backend && .venv/Scripts/python -m pytest)`（Windows）或 `bin/python`（mac/linux）；前端 `cd frontend && npm test`（脚手架阶段暂无 test script，后续补）；E2E `npx playwright test`。**测试门钩子在写代码后自动用项目 venv 运行相关测试，无测试框架则跳过。**

## 七、可观测指标（验收看这些，见 PRD 11 章）

| 指标 | 目标（假设，需实测） | 验证方式 |
|---|---|---|
| 测试通过率 | 100%（单测 + E2E） | pytest / playwright |
| 检索质量 | Top-5 命中率 ≥80%；答案准确率 ≥80% | 自建评测集（≥50 问答对） |
| 导入耗时 | 100 个 MD ≤5-10 分钟 | 基准测试 |
| 增量同步 | 改一处只更新相关块；未变文件 <10ms 跳过 | 一致性测试 |
| 流式性能 | 首 token <3s；≥20 tok/s（<10 视为不可用） | SSE 埋点 |
| 隐私 | 默认**零外发**（纯文本路径） | 网络审计断言 |
| 常驻内存 | ≤2GB（不含本地大模型进程） | 资源测量 |

> 所有数值为假设，须在 v0.1 用作者真实语料实测校准后回填 PRD。

## 八、阶段验收钩子（harness 自动执行）

- **SessionStart**：自动读取 `.claude/project-state.md` 注入上下文——新 Agent 开会话就知道「当前阶段 + 本阶段验收点」。
- **PostToolUse(Write|Edit)**：写代码后自动尝试运行相关测试；无测试则跳过（不阻塞）。
- **Stop**：结束时提醒核对当前阶段验收点（看 `.claude/project-state.md` 的 checklist）。
- 进度推进后**必须更新** `project-state.md`（阶段 → 下一步 → 验收 checklist）。

## 九、项目记忆（auto-memory）

- 记忆目录：`~/.claude/projects/<本路径>/memory/`，索引在 `MEMORY.md`。
- **何时写记忆**：新 Agent 完成一个阶段/模块后，把「非显而易见的事实 + 决策 + 踩坑」写入记忆文件并更新 `MEMORY.md` 一行索引。不写代码结构本身（仓库已记录）。
- 记忆类型：`user`（用户偏好）/ `feedback`（工作方式纠偏）/ `project`（项目约束与目标）/ `reference`（外部资源指针）。

## 十、Agent 协作与可插拔适配器

- 多 Agent 并行时按模块解耦（ingestion / vectorstore / retrieval / generation / frontend / config-privacy），每个 Agent 只改自己模块，通过接口契约协作，避免共享文件冲突。
- **可插拔适配器**（目标，开源后正式化）：导入器、嵌入 provider、对话模型 provider、识图模型 provider、向量库后端，均走薄接口 + 工厂注册，新实现不改业务层。
- 出网只允许出现在模型 provider 的客户端内（`OutboundClient`），其余模块禁止发外呼。

## 十一、隐私红线（开发时必须遵守）

- 纯文本路径默认零外发；任何出网都必须是显式 opt-in + 二次确认。
- API key 不落 config 明文（环境变量 / 内存掩码）。
- 日志不得含对话正文 / 文件路径 / 文件名 / 密钥；错误不泄露响应体。
- 本地服务只绑定 `127.0.0.1`，不开放远程访问。
