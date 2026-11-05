# MCP Server 验收记录

验证日期：2026-10-27。运行环境：Windows、Python 3.11、官方 `mcp==2.3.0`、Codex CLI 0.156.1。

## 已实现内容

- 在 FastAPI 生命周期中启动 Streamable HTTP MCP Server，地址为 `http://127.0.0.1:9999/mcp/`。
- 提供 `get_knowledge_overview`、`lookup_catalog`、`search_knowledge`、`read_document` 四个只读工具。
- 内部 Agent 与 MCP 共用知识服务、混合检索和 rerank 处理；内部已有的文档范围 ID 继续有效。
- 提供 Bearer 鉴权、Host/Origin 验证、输入限制、输出预算、固定错误代码和日志脱敏。
- 通过内容版本检测同数量内容更新，校验目录分页、连续阅读和检索结束时的版本。
- 通过进程内读取顺序及跨进程维护锁协调读取、导入、同步、上传、删除和图片索引迁移。
- 已配置本机 Codex，使用 `scripts/mcp_headers.ps1` 从用户环境变量读取凭证。

## 实际知识库验证

实际知识库包含 241 个文档、4,727 个内容块。检查过程中保持这些文档和索引内容不变。

运行命令：

```powershell
$env:EVERYTHING_RAG_MCP_TOKEN = [Environment]::GetEnvironmentVariable('EVERYTHING_RAG_MCP_TOKEN', 'User')
backend/.venv/Scripts/python scripts/check_mcp_live.py --output .tmp_probe/mcp-live-result.json
```

| 检查项目 | 结果 |
| --- | --- |
| `Client(mode="auto")` | 四个工具发现成功，正文检索返回 3 条实际证据 |
| `Client(mode="legacy")` | 四个工具发现成功，正文检索返回 3 条实际证据 |
| 结构化输出与 JSON 文本 | 内容一致，响应大小符合限制 |
| 文档范围 | 返回证据均属于指定 document_id |
| 连续阅读 | 两页内容与相同范围的一次读取内容一致 |
| 版本核对 | 目录、搜索与读取返回相同文档版本 |
| 无凭证 GET / POST / DELETE | 401 |
| 非本机 Host / Origin | 403 |
| 缺少连续阅读版本、失效版本、非法 cursor、空查询、超限 top_k、未知文档范围 | 每种客户端模式下的六项错误检查全部通过 |

当前真实配置没有启用 rerank，也没有配置 rerank 模型、凭证或校准档案，因此真实知识库搜索返回 `rerank_status=disabled`。隔离测试已验证启用但配置不完整时返回错误并终止检索；真实 rerank 服务成功评分的路径未执行。

## 隔离测试

测试文件：`backend/live_tests/test_mcp_server.py`。使用实际 `Agent.md` 和 MCP 设计文档，由项目导入管线调用真实 embedding 服务建立独立 Chroma 索引。测试数据位于 `.tmp_probe/mcp-acceptance/`，没有使用 mock 或替代模型响应。

```powershell
Set-Location backend
.venv/Scripts/python -m pytest live_tests/test_mcp_server.py -q --basetemp ../.tmp_probe/pytest-mcp
```

结果：**9 passed**。

| 检查 | 验证内容 |
| --- | --- |
| 实际导入、HTTP 与内部工具 | 目录分页、真实检索、原文阅读、内部文档范围 ID、两种协议连接方式 |
| 同数量内容更新 | 实际更新隔离索引中的一个块，旧文档版本与旧 cursor 失效，BM25 重新构建 |
| 维护锁、超时和并发名额 | 维护期间明确拒绝；超时后继续占用名额，完成后释放 |
| rerank 配置错误 | 启用但缺少校准档案时返回 `RERANK_FAILED` |
| 空索引与输出预算 | 空状态准确，限制返回字符数并保留连续读取位置 |
| HTTP 调用性能 | 与公共服务直接调用的结果一致，记录各六次调用耗时 |
| 跨进程维护 | 独立进程取得维护锁后，读取受到保护；释放后恢复 |
| 失效 collection 读取 | 读取失败，未自动创建 collection |
| 并行调用 | 四个工具通过实际 HTTP 同时调用，全部成功且 index_revision 一致 |

## 应用生命周期与现有入口

```powershell
backend/.venv/Scripts/python scripts/check_mcp_lifespan.py
backend/.venv/Scripts/python scripts/check_mcp_lifespan.py --enabled
```

启用和关闭两种配置各运行两次完整应用生命周期，全部通过。验证了关闭时 `/mcp` 与 `/mcp/` 返回 404，启用时检查凭证，重新启动时创建新的 SDK 会话管理器，结束时释放 MCP 实例。未配置 embedding 时仍可发现工具，查询操作返回 `EMBEDDING_NOT_CONFIGURED`。

以下现有测试通过，结果为 **2 passed**：

```powershell
Set-Location backend
.venv/Scripts/python -m pytest --noconftest tests/test_run_py.py -q -k 'parse_args_accepts_no_browser or static_hosting_serves_index_when_dist_exists' --basetemp ../.tmp_probe/pytest-mcp-startup-real
```

所有修改涉及的 Python 文件通过 Ruff 检查，`pip check` 通过，`git diff --check` 通过。

## Codex 实际调用

通过已经写入用户配置的 `everything_rag` 服务执行 `codex exec --ephemeral --json --sandbox read-only`。

第一组明确要求完成四个工具的调用；四项调用均成功，文档与引用内容一致。

第二组使用知识库中实际文档的自然语言问题，要求查询资料并附文档名称、章节和 document_id，提示中没有列出 MCP 工具名称。Codex 自动调用了四个工具，结果为：

```json
{
  "verified": true,
  "tools": [
    "get_knowledge_overview",
    "lookup_catalog",
    "read_document",
    "search_knowledge"
  ],
  "successful_calls": 4,
  "failed_calls": 0,
  "citation_checked": true
}
```

核查直接读取 Codex 的 MCP 调用事件，并检查最终回答包含实际读取结果中的文档名称、文档 ID 和章节。可重复核查：

```powershell
backend/.venv/Scripts/python scripts/check_codex_mcp_result.py .tmp_probe/mcp-codex-natural-events.jsonl .tmp_probe/mcp-codex-natural-answer.txt --output .tmp_probe/mcp-codex-verification.json
```

CLI 原始事件和回答包含知识库内容，保存在忽略目录 `.tmp_probe/`。本次实际客户端验证范围为 Codex；Claude Code、DSH 的配置示例保存在设计文档中，其客户端对话流程未执行。

## 调用耗时

测量语料为两个实际项目文档，各路径六次调用，embedding 查询缓存已经预热，rerank 未启用。p95 按样本的最近秩计算，六个样本时取最大值。

| 调用方式 | p50 | p95 | 失败数 |
| --- | --- | --- | --- |
| 直接调用公共知识服务 | 41.37 ms | 52.38 ms | 0 |
| MCP HTTP | 46.61 ms | 49.93 ms | 0 |

这些数字反映当前本机、隔离语料与缓存状态。完整知识库的首次 embedding 请求和模型服务延迟需要单独测量。

## 当前运行状态

后端已在本机 9999 端口启用 MCP，Codex 用户配置已保存。Token 存储在 Windows 用户环境变量中，仓库文件和验收记录不包含 Token。接入与启动步骤见 [使用说明](mcp-server-usage.md)。
