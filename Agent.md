# Everything RAG：Agent 速览

## 项目定位

面向个人知识库的本地 Web RAG：导入 Markdown（可含图片），进行增量索引、混合检索和流式问答；内容问答尽可能附带可追溯来源，并显式标识本轮答案依据。知识库和运行状态保存在本机；对话、嵌入、识图及联网搜索会按用户配置访问外部服务。

## 当前技术与入口

- 后端：Python 3.11+、FastAPI、Chroma、SQLite，入口 `backend/app/main.py`。
- 前端：React 19、TypeScript、Vite；`frontend/dist/` 由 FastAPI 同源托管。
- 启动：根目录双击 `EverythingRagStart.bat`；底层入口为 `backend/.venv/Scripts/python scripts/run.py`，固定绑定 `127.0.0.1:9999`并自动打开浏览器。
- 配置：优先使用页面设置，保存到 `EVERYTHING_RAG_DATA_DIR/config.json`；该文件存在后即为权威配置。首次运行才回退 `backend/.env`，变量模板见 `backend/.env.example`。
- 默认数据目录：`~/.everything-rag/`，包含 Chroma、`state.db`、`image_state.db`、`conversations.json`、配置、上传文档和头像等运行数据；不得提交。

## 核心数据流

1. `ingestion/` 扫描 Markdown，解析标题/段落/列表/表格/代码/图片，清洗噪声并按标题、长度和语义边界切块。
2. 增量同步以 mtime+size 快判、内容哈希确认；修改文件按块文本哈希复用旧向量，仅更新变化块。
3. 识图配置完整时，图片由后台任务执行下载、内容去重、高分辨率识图，分别保存检索摘要和详细证据；只嵌入摘要、文件名和标题，完整证据参与关键词检索并用于引用。原图本地缓存；视觉/数值问题最多并行核对两张已召回图片。失败不阻塞文本链路，未完成任务可重启续跑。v7 迁移脚本复用正文向量与历史识图缓存，重建图片摘要向量；评测后单独 promote，保留 v6 和状态备份。
4. 文档向量使用“文件名 + 完整标题路径 + 清洗后正文”，引用仍保存纯正文；查询向量使用独立 query 编码，Qwen3 自动加检索 instruct。Chroma collection 同时按模型、端点 fingerprint 与入库结构版本隔离。
5. `HybridRetriever` 使用 dense + BM25 + RRF，保留原问题并增加至多一个确定性去口语变体；经过信息词/覆盖率/稀有度、低 dense 反证、词法保留位和标题多样性门禁后生成上下文。
6. `generation/` 通过 `POST /api/chat` 输出 SSE：`meta/token/tool/done/error`。原始思维链不发送、不持久化、不展示；引用使用稳定的 `S1...` / `W1...` 来源 ID。

## 目录地图

- `backend/app/api/routes/`：status、chat、import/sync、settings、conversations、blocks、avatars、audit API。
- `backend/app/ingestion/`：Markdown 解析、切块、导入、增量同步、图片管线。
- `backend/app/vectorstore/`、`retrieval/`、`generation/`：嵌入与 Chroma、召回、模型/SSE。
- `backend/app/core/`：基础配置、页面设置、会话存储、统一出网审计。
- `frontend/src/`：页面、组件、API 客户端与 SSE 解析；`frontend/e2e/` 为 Playwright 用例。
- `scripts/`：启动、命令行导入、检索评测与识图检查。
- `docs/PRD/PRD.md`、`docs/tech/`：产品和技术背景。`CLAUDE.md` 部分早期技术描述已过时，冲突时以当前代码、`README.md`、`pyproject.toml` 和 `package.json` 为准。

## 开发约束

- 保持轻量自研 RAG 管线，不引入 LangChain/LlamaIndex；新 provider 通过现有接口/工厂接入。
- Agent 身份、grounding、防提示注入和引用规则由 `generation/prompt_policy.py` 固定；设置页支持结构化表达偏好及 `custom_instructions` 附加要求（最多 4000 字符，可清空），随每轮问答发送，不能替换核心 system policy。
- 本地服务只绑定 `127.0.0.1`；外呼统一经过出网审计，日志不得记录正文、路径、密钥或响应体。
- 解析/切片/向量表示有版本与索引 fingerprint；切片版本或嵌入模型变化会切换到新 collection，并在状态接口标记需要重建。用户主动同步后才会重新嵌入，可能产生模型调用费用；旧 collection 保留，不原地破坏。不得让多个进程绕过 `.chroma-write.lock` 并发写同一 Chroma 数据目录。
- 全量导入/增量同步结束必须执行 ANN 全库可达性修复；评测复用索引必须同时匹配语料 SHA-256、schema、嵌入 fingerprint、collection 和逻辑块数。
- 旧配置中的自由文本 `system_prompt` 首次加载时只迁移到只读备份，不再参与执行；核心 Agent 策略始终由代码固定。
- API、前端 `src/types.ts` 与 SSE 帧契约必须同步修改；嵌入器和向量库必须共享同 fingerprint 的进程内实例。
- 导入单文件失败应脱敏并继续；图片失败不得拖垮文本链路；运行数据、密钥、`.env`、`dist`、虚拟环境和依赖目录不入库。
- 延续现有风格：后端测试优先、类型注解、Ruff 100 字符行宽；前端保持严格 TypeScript，并为关键交互补 E2E。

## 验证命令（Windows）

```bat
cd backend
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check app tests

cd ..
backend\.venv\Scripts\python scripts\eval_rag_quality.py --enforce

cd frontend
npm run build
npx playwright test
```

检索验收集在 `scripts/rag_eval_cases.json`（72 个块级正例 + 10 个库外负例，共 82 个独立单例）；正例必须同时命中文件 basename、完整标题路径和正文证据，不能用同文档/同标题的错误块充数。重建后的真实语料评估使用 `scripts/eval_reindexed_quality.py`，并先通过向量/ANN 完整性门禁。修改启动或静态托管时，至少运行 `backend/.venv/Scripts/python -m pytest backend/tests/test_run_py.py`；完整改动按影响范围运行后端全测、前端构建及 E2E。
