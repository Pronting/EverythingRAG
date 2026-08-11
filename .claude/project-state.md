# 项目状态 — Everything RAG

> 本文件由 **SessionStart 钩子自动注入**每个会话上下文，让 Agent 开会话即知「现在在哪、下一步做什么、验收什么」。
> 进度推进后必须更新「当前阶段 / 下一步 / 验收 checklist」。最后更新：2026-08-11。

## 当前阶段

**Stage 1：MVP 最小闭环（✅ 脚手架完成 + loop harness 就绪，⬜ 构建中）**
- ✅ 脚手架：git init + 目录结构 + FastAPI 骨架（/api/status + 模块分层 + Protocol 契约）+ 前端 React19/Vite/TS + pytest 冒烟 **2 passed** + scripts/run.py
- ✅ **loop harness 三件套**：`.claude/loop-state.md`（任务队列）· `.claude/commands/loop-plan.md`（循环入口）· `docs/loop/loop-workflow.md`（规范）
- ⬜ **下一步：输入 `/loop-plan` 启动循环**，逐轮打通最小闭环（扫描→解析→切块→嵌入→Chroma→检索→SSE 问答）

## 下一步动作

1. 输入 `/loop-plan`（或 `/loop-plan 任务号`）启动循环，控制器按 `.claude/loop-state.md` 队列逐轮推进
2. 遇人工 Gate（任务 4 模型下载 / 任务 6 对话 provider / 设计决策）停下确认
3. 队列清空 → 更新本文件 checklist 打勾 + 里程碑回填实测指标

## 本阶段验收 checklist（MVP 最小闭环，任务粒度见 .claude/loop-state.md）

- [x] Loop harness：loop-state.md / loop-plan.md / loop-workflow.md 已落盘
- [ ] 目录扫描 + MD 解析 + 语义切块（loop 任务 1-3）
- [ ] Chroma + bge-m3 嵌入（loop 任务 4，真模型下载需人工确认）
- [ ] 检索管线（loop 任务 5）+ SSE 问答 POST /api/chat（loop 任务 6）
- [ ] 隐私审计：纯文本路径零外发断言（loop 任务 7）
- [ ] 前端 Chat UI + 手写 SSE 解析（loop 任务 8）
- [ ] 端到端最小闭环：导入语料 → 问答带来源（loop 任务 9）

## 本阶段验收 checklist（脚手架搭建 — 已完成 ✅）

- [x] 仓库初始化：git init、.gitignore、README、依赖声明齐全
- [x] 目录结构：backend/app（api/core/ingestion/models/vectorstore/retrieval/generation/util）+ frontend/src + scripts
- [x] 后端可启动：`python -m uvicorn ...` 拉起本地 Web 服务，`GET /api/status` 返回 200
- [x] 前端可访问：`npm run dev` 或 FastAPI 托管静态页面可打开
- [x] 冒烟测试通过：pytest 至少 1 个冒烟用例（**2 passed**）
- [x] 隐私基线：启动时无任何出网调用（默认零外发）
- [x] 阶段产物落盘：脚手架改动已提交（`7e8bcfa`）

## 里程碑（后续）

| 里程碑 | 内容 | 验收标准（见 PRD 11 章） |
|---|---|---|
| v0.1 MVP | 目录扫描/导入/图片/增量同步/问答 | 100 MD 导入、改一处只更新相关块、答案带来源、数据不出机 |
| v0.2 | 喂图增强、混合检索+重排、更多格式 | 评测集 Top-5 ≥ v0.1+10pp |
| v1.0 | 记忆画像、融合排序、多平台导入器 | 跨会话一致性、≥3 平台导入器 |
| 开源 | **Tauri v2 桌面客户端（Win/macOS）** + 适配器规范 + 社区 | 新人 30 分钟跑通 |

## 快速命令

- 后端测试（搭好后）：`cd backend && pytest`
- 前端测试（搭好后）：`cd frontend && npm test`
- E2E：`npx playwright test`
- 启动（搭好后）：见 `scripts/`（`run.py`，127.0.0.1 随机端口）
