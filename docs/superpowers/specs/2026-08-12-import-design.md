# 知识库导入功能 — 设计规格

> 日期：2026-08-12 ｜ 状态：已确认 ｜ 关联：MVP 任务 9 的对外化、v0.1 导入 UI 的前置

## 1. 背景与业务逻辑

用户选择一个本地目录，系统把目录下**所有 Markdown 文件**向量化写入本地向量库，成为可自然语言问答的"记忆"。这是 MVP「目录扫描 → 解析 → 切块 → 嵌入 → 检索 → 问答」链路中**导入环节的对外化**：此前 `IngestionPipeline` 只能被 Python/测试调用，没有 HTTP 入口和界面。

核心业务逻辑：
```
用户选目录 → 后台线程执行 IngestionPipeline.ingest(dir)
           → 逐文件解析/切块/嵌入/upsert
           → 前端轮询实时进度 → 完成后展示报告 → /api/status 计数更新
```

**格式边界（MVP 决策 8）**：只处理 `.md`；目录下非 Markdown 文件跳过（计入 `files_skipped`，报告可见）。全格式支持（PDF/docx 等）属 v0.2。

## 2. 已确认决策

| 决策点 | 结论 |
|---|---|
| 执行模型 | **异步 + 进度轮询**（非同步阻塞、非 SSE） |
| 增量同步位 | **预留 `mode` 字段**，现仅支持 `"full"`；`"incremental"` 返回清晰 400「未实现」 |

## 3. 后端 API

### 3.1 `POST /api/import`

请求体：
```json
{ "dir": "C:/Users/me/Documents/knowledge", "mode": "full" }
```
- `dir` 必填：服务器本地目录路径（浏览器无法读服务器文件系统，只填路径）。
- `mode` 可选，默认 `"full"`；仅 `"full"` 合法，`"incremental"` → 400 `{"detail": "增量同步未实现，当前仅支持 mode=full"}`，其他值 → 422。
- `dir` 不存在或不是目录 → 400 `{"detail": "目录不存在: <dir>"}`（脱敏，不含敏感路径细节）。
- 成功 → `202 Accepted`，立即返回 `{"task_id": "<uuid>"}`（不等导入）。

### 3.2 `GET /api/import/status/{task_id}`

返回任务状态：
```json
{
  "task_id": "<uuid>",
  "status": "running" | "done" | "error",
  "progress": { "files_scanned": 10, "files_parsed": 5, "files_skipped": 0, "chunks": 12 },
  "report": null | { "files_scanned": 10, "files_parsed": 10, "files_skipped": 0, "chunks": 23, "blocks_upserted": 23, "errors": [] }
}
```
- `running`：progress 为实时快照，report 为 null。
- `done`：progress 为最终值，report 为完整 `IngestReport`。
- `error`：非目录/解析级灾难错误；message 字段给可读原因。
- 未知 task_id → 404。

## 4. 任务存储与执行

- **`ImportTaskStore`**：进程内 dict + `threading.Lock`，键 `task_id`，值 `ImportTask`（含 status / progress / report / error）。单用户本地足够；重启丢失（status 返回 404，可接受）。接口独立成模块便于测试。
- **执行**：`IngestionPipeline.ingest()` 为同步阻塞（chroma/embed），放在**后台线程**运行（`threading.Thread` 或 executor），事件循环不被阻塞。线程内逐步更新共享 task 状态。

## 5. 进度机制（管线改造）

当前 `ingest()` 仅末尾返回报告，无中间进度。新增**可选回调**（向后兼容，默认 None 零开销、现有测试不受影响）：

```python
def ingest(
    self,
    root_dir: Path,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
) -> IngestReport
```

每处理一个文件后调用一次 `on_progress(snapshot)`，`ProgressSnapshot` 含运行中的 `files_scanned / files_parsed / files_skipped / chunks`。后台线程把快照写入 task 状态。

## 6. `/api/status` 知识计数修复

`knowledge` 段从硬编码 0 改为真实值（status 处理器引入 vector store 依赖）：
- `chunk_count` = `vectorstore.count()`
- `file_count` = 元数据中 `source_file` 去重数
- `image_count` = 0（MVP 无识图）
- `last_sync_at` = 最近成功导入时间（内存态；重启回 None）
- `needs_rebuild` = false

## 7. 前端

- **「导入知识库」面板**（头部与聊天区之间）：目录路径输入框 + 「开始导入」按钮 + 进度/报告区。
- 流程：点击 → `startImport(dir)` → 拿 `task_id` → 每 ~1s `getImportStatus(task_id)` → 显示 `扫描 X · 解析 Y · 写入 Z` → 完成展示报告（含失败原因）→ 刷新 `/api/status` 计数显示。
- 新增 `frontend/src/api/importApi.ts`（`startImport` / `getImportStatus`）；扩展 `types.ts` 的 `StatusResponse`（config / knowledge 字段）；`App.tsx` 状态栏增加「知识库：N 文档 / M 块」。
- 样式遵循现有 Chat UI（`App.css`）。

## 8. 错误处理

| 场景 | 行为 |
|---|---|
| `dir` 不存在/非目录 | 400，任务不启动 |
| `mode` 非法 | 400/422 清晰报错 |
| 导入中单文件失败 | 聚合脱敏原因（仅异常类型名），计入 report.errors，不入任务 error |
| 任务内灾难错误（罕见） | 任务 status=error + 可读 message |
| 未知 task_id | 404 |
| 前端轮询超时/网络断 | 显示「状态获取失败」，可手动重新导入 |

## 9. 测试策略（TDD）

后端（pytest）：
1. **管线进度回调**：`ingest(..., on_progress=...)` 被逐文件调用，快照递增、终值等于报告。
2. **ImportTaskStore**：状态机 running→done/error；并发写线程安全。
3. **import API**：POST 启动（202 + task_id）、轮询到 done、404、mode=incremental→400、坏目录→400。
4. **status 计数**：种入 vector store 后 `/api/status` 反映真实 file/chunk 数。

前端（Playwright E2E）：输入路径 → 点导入 → 看到进度与报告。

## 10. 非目标（明确不做）

- 完整增量同步（指纹快路径/对话级去重键/防漂移）— v0.1 单独块，本需求仅预留 `mode` 位。
- 浏览器文件上传（只填目录路径）。
- 任务持久化到磁盘（重启丢失）。
- 非 Markdown 格式解析（v0.2）。
- 图片入库（方案 B，v0.1 后续）。

## 11. 风险与注意

- **轮询间隔**：1s 对本地导入足够；导入很快时前端可能看到 running 一闪而过，需处理「已完成再轮询」的竞态（status 返回 done 即停）。
- **大目录**：`file_count` 用元数据去重，需一次 query 全量元数据，百级文件可接受。
- **Embedding 首次加载**：bge-m3 已持久缓存（`~/.everything-rag/models`），导入无需下载。
