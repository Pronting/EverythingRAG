# 项目状态 — Everything RAG

> 本文件由 **SessionStart 钩子自动注入**每个会话上下文，让 Agent 开会话即知「现在在哪、下一步做什么、验收什么」。
> 进度推进后必须更新「当前阶段 / 下一步 / 验收 checklist」。最后更新：2026-08-11。

## 当前阶段

**Stage 1：脚手架搭建（✅ 已完成）**
- ✅ git init + .gitignore + 目录结构（backend/frontend/scripts）
- ✅ 后端骨架：FastAPI + `/api/status` + 模块分层（api/core/ingestion/models/vectorstore/retrieval/generation/util）+ 接口契约（Protocol 可插拔）
- ✅ 前端骨架：React 19 + Vite + TS，`npm run build` 通过（742ms）
- ✅ 测试基座：后端 pytest 冒烟 **2 passed**（TestClient + 真实 uvicorn 启动均验证）
- ✅ `scripts/run.py`（动态端口 + 自动开浏览器）
- ⬜ **下一步：实现最小闭环**（目录扫描 → MD 解析 → 切块 → 嵌入 → Chroma → 检索 → SSE 问答）

## 下一步动作

1. 实现数据接入 + 文档管线：目录扫描、MD 解析（markdown-it-py）、语义切块（标题锚点）
2. 实现向量化 + 向量库：bge-m3 fastembed 加载 + Chroma 持久化
3. 实现增量同步：指纹比对 + 对话级去重
4. 实现问答：检索 → SSE 流式生成 → 来源标注
5. 前端：问答视图 + 配置向导

## 本阶段验收 checklist（最小闭环 v0.1 骨架）

- [ ] 目录扫描 → 切块 → 嵌入 → Chroma 入库可跑通（冒烟）
- [ ] 问答接口 POST /api/chat 返回 SSE 流（meta/token/done）
- [ ] 前端可发起问答并渲染来源
- [ ] 隐私基线：默认零外发断言保持通过

## 本阶段验收 checklist（脚手架搭建的验收点）

- [ ] 仓库初始化：git init、.gitignore、README、依赖声明齐全
- [ ] 目录结构：backend/app（api/core/ingestion/models/vectorstore/retrieval/generation/util）+ frontend/src + scripts
- [ ] 后端可启动：`python -m uvicorn ...` 拉起本地 Web 服务，`GET /api/status` 返回 200
- [ ] 前端可访问：`npm run dev` 或 FastAPI 托管静态页面可打开
- [ ] 冒烟测试通过：pytest 至少 1 个冒烟用例、Playwright 至少打开首页断言
- [ ] 隐私基线：启动时无任何出网调用（默认零外发）
- [ ] 阶段产物落盘：本次脚手架改动提交（conventional commit）

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
