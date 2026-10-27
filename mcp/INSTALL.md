# 为当前 Agent 安装 Everything RAG MCP

把下面这句话交给能够操作本机文件和终端的 Agent：

> 按照这个项目的 mcp/INSTALL.md，为你自己安装 Everything RAG MCP，并完成真实连接验证。

## Agent 执行指令

你的任务是将本项目知识库接入当前 Agent。完成客户端配置、启动服务和真实验证，报告检查结果。使用本目录现有脚本处理配置与凭证。

### 1. 确认运行环境

- 根据当前会话和可用命令，识别客户端为 Codex、Claude Code 或 DSH；仅安装到当前客户端。对应参数依次为 `codex`、`claude`、`dsh`。
- 通过本文件的位置确定项目根目录，命令中的项目路径使用实际绝对路径，正确处理路径中的空格。
- 确认项目的 Python 环境已安装后端依赖。Windows 默认解释器为 `backend/.venv/Scripts/python.exe`；macOS 和 Linux 通常为 `backend/.venv/bin/python`。
- 知识库沿用项目已有的 embedding 配置和索引。缺少这些配置时明确说明缺少什么，不填写未经用户提供的模型地址或凭证。
- 服务使用本机 `http://127.0.0.1:9999/mcp/`。客户端与服务需要运行在同一台电脑上。Claude 网页及其他云端 Agent 无法直接访问该地址。

### 2. 执行安装

Windows 在项目根目录执行以下命令，将 `codex` 替换为当前客户端：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\mcp\setup.ps1 -Client codex -Action install
```

`setup.ps1` 使用项目 Python 环境安装本目录依赖，然后执行 `manage.py`。

macOS 或 Linux 使用项目解释器执行：

```bash
backend/.venv/bin/python -m pip install -r mcp/requirements.txt
backend/.venv/bin/python mcp/manage.py install --client codex
```

脚本会自动完成以下操作：

1. 读取已有 MCP 凭证；不存在时生成随机凭证。
2. 解析客户端配置，检查 `everything_rag` 是否存在名称冲突。
3. 启动后台服务，或者检查并复用正在运行的服务；不会打开 WebUI。
4. 获取工具列表并查询真实知识库概览。
5. 备份需要修改的配置，写入当前客户端的 MCP 条目。
6. 将同一凭证和 MCP 启用状态保存到项目 `backend/.env`，保留其他配置。

重复安装更新同一个条目，不增加重复条目。原有其他服务、客户端设置和工具权限保持原值；Codex 的 `everything_rag` 条目会设为启用。

如果已有进程占用 9999 端口且 MCP 尚未启用，脚本会停止并报告连接错误。确认该进程属于本项目且可以停止后，通过正常关闭方式结束服务，再执行安装。不要终止身份不明的进程，也不要绕过鉴权。

### 3. 验证结果

Windows 执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\mcp\setup.ps1 -Action check
```

macOS 或 Linux 执行：

```bash
backend/.venv/bin/python mcp/manage.py check
```

确认脚本成功退出，输出包含 `tools: 4` 和实际 `file_count`。客户端需要重新连接 MCP 或启动新会话才能加载新配置；客户端自己的目录信任和工具批准规则继续生效。

当前 Agent 能够调用新工具时，继续调用 `get_knowledge_overview`。有文档时，选择一篇通过 `lookup_catalog` 定位，再通过 `read_document` 读取正文，确认调用成功。如果当前会话不能重新加载工具，明确区分脚本连接验证与当前会话的工具加载状态，不声称已经完成尚未执行的调用。

最终报告安装的客户端、验证结果和是否需要重新连接。不要输出 Token、请求头、客户端配置全文、环境变量全文或私人文档正文。

## 后续启动

电脑重启后可以使用同一脚本启动后台服务：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\mcp\setup.ps1 -Action start
```

macOS 或 Linux 使用 `backend/.venv/bin/python mcp/manage.py start`。

## 配置与文件位置

| 客户端 | 默认文件 | 支持的自定义目录 |
| --- | --- | --- |
| Codex | `~/.codex/config.toml` | `CODEX_HOME` |
| Claude Code | `~/.claude.json` | `CLAUDE_CONFIG_DIR` |
| DSH | `~/.dsh/cordis.patch.yml` | `DSH_HOME` |

DSH 使用用户级 patch，各个 profile 都可以读取。当前进程环境中的 MCP 配置优先，其次为 Windows 用户环境，最后为项目 `backend/.env`。

知识库数据目录沿用 `EVERYTHING_RAG_DATA_DIR`，默认 `~/.everything-rag/`。配置备份位于该目录的 `mcp/backups/`，后台服务日志位于 `mcp/server.log`。凭证保存在本机客户端配置、`backend/.env` 及备份中，不得提交到代码仓库或发送给他人。

## 其他 Agent

支持 Streamable HTTP 与 Bearer Token 的本机客户端可以连接此服务。使用 `setup.ps1 -Client generic` 或 `manage.py install --client generic`，生成 `~/.everything-rag/mcp-client.json`。

根据目标客户端提供的配置或导入功能完成接入。这个 JSON 包含本机凭证，仅供本机使用。未知客户端需要核对其正式配置说明，不能假定它使用 Codex、Claude Code 或 DSH 的文件格式。

## 开发验证

测试使用实际客户端、现有配置的隔离副本及正在运行的 MCP 服务，检查配置保留、重复安装和真实连接。需要本机安装三个客户端及本目录依赖：

```powershell
backend/.venv/Scripts/python.exe -m pytest --noconftest backend/live_tests/test_mcp_setup.py -q --basetemp .tmp_probe/pytest-mcp-setup
```

测试产物保存在项目忽略目录 `.tmp_probe/mcp-onboarding/`。
