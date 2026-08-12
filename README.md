# Everything RAG — 个人知识第二大脑

纯本地 RAG 检索问答：把「本地文档 + 各 AI 平台对话记录」统一向量化，用自然语言检索，**答案永远带来源，数据不出本机**。

> 项目总纲与工作流见 [CLAUDE.md](CLAUDE.md) ｜ 产品需求见 [docs/PRD/PRD.md](docs/PRD/PRD.md) ｜ 技术选型见 [docs/tech/](docs/tech/)

## 快速开始

### 1. 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.11+ | 后端（Windows 建议用 `py -3.11`） |
| Node | 18+ | 仅构建前端时需要 |
| 网络 | 首次需联网 | 下载 bge-m3 嵌入模型（约 2.5GB，一次性） |

### 2. 安装依赖

```bash
# 后端（Windows：backend\.venv\Scripts\python）
cd backend
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

# 前端（可选，不构建也能启动——后端直接托管已有 dist/）
cd ../frontend
npm install
```

### 3. 配置对话模型（必做）

复制配置模板并填写：

```bash
cd backend
cp .env.example .env        # git-bash；PowerShell 用 Copy-Item .env.example .env
```

编辑 `.env`，填写 `EVERYTHING_RAG_CHAT_BASE_URL` / `EVERYTHING_RAG_CHAT_MODEL` / `EVERYTHING_RAG_CHAT_API_KEY`（云端必填）。**嵌入模型不需要配置**：本地 bge-m3 首次使用时自动下载，无需 API Key。

### 4. 启动

```bash
# 回到仓库根目录
backend/.venv/Scripts/python scripts/run.py        # 随机端口 + 自动开浏览器
backend/.venv/Scripts/python scripts/run.py --no-browser   # 自动化/CI 用
```

## 配置说明

所有配置走**环境变量**（pydantic-settings 读取，前缀 `EVERYTHING_RAG_`），可写在 `backend/.env` 或真实环境变量中。完整模板见 [backend/.env.example](backend/.env.example)。

### 对话模型（必配）

| 变量 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `EVERYTHING_RAG_CHAT_BASE_URL` | ✅ | OpenAI 兼容端点地址 | `https://api.deepseek.com/v1` |
| `EVERYTHING_RAG_CHAT_MODEL` | ✅ | 模型 id | `gpt-4o-mini` / `deepseek-chat` |
| `EVERYTHING_RAG_CHAT_API_KEY` | 云端必填 | API Key，只填值不带 `Bearer` | `sk-...` |
| `EVERYTHING_RAG_CHAT_PROVIDER_TYPE` | ✳ | Provider 类型，默认 `openai_compatible` | `openai_compatible` |

### 嵌入模型（自动，无需配置）

- 模型固定为本地 **bge-m3**（fastembed/ONNX），首次真实嵌入时自动从 Hugging Face 下载约 2.5GB，之后离线可用。
- 无需 API Key；换模型能力在 v0.1 配置向导中落地。

### 其他可选

| 变量 | 说明 | 默认 |
|---|---|---|
| `EVERYTHING_RAG_DATA_DIR` | 数据目录（向量库/state.db/配置，不出本机） | `~/.everything-rag` |
| `EVERYTHING_RAG_VISION_ENABLED` | 识图开关（v0.1 前保持关） | `false` |
| `EVERYTHING_RAG_HOST` / `EVERYTHING_RAG_PORT` | 绑定地址/端口（run.py 实际用随机端口） | `127.0.0.1:8000` |
| `EVERYTHING_RAG_ENABLE_CORS` / `EVERYTHING_RAG_CORS_ORIGINS` | 跨域开关 + 白名单（JSON 数组） | `false` / `[]` |

> 隐私红线：API key 只运行期从环境读取，绝不写入配置/日志；出网仅经唯一 `OutboundClient` 并记入审计（`GET /api/audit`）。

## 常用命令

```bash
# 后端测试（Windows）
cd backend && .venv/Scripts/python -m pytest

# 前端构建
cd frontend && npm run build

# E2E（Playwright，需先启动服务）
cd frontend && npx playwright test

# 网络审计
curl http://127.0.0.1:<port>/api/audit
```

## 文档导航

- `docs/PRD/PRD.md` — 产品需求（13 章定稿）
- `docs/tech/00-技术选型决策文档.md` — 技术栈总表与决策
- `docs/tech/T1~T6-*.md` — RAG 框架/前端/流式/向量库/识图/配置隐私部署
