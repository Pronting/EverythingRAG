# 当前 Agent 真实资料评测

运行器：`benchmark/agent_live_eval.py`。私有题目保存在 `.tmp_probe/agent-live-eval-20260929/cases.json`，逐题结果保存在同目录的 `results/`；这些路径已由 `.gitignore` 排除。题目包含个人文档短引和历史会话消息 ID，不能加入版本控制。

题目共 50 道：文档事实 27 道、跨文档关联 8 道、历史会话回忆 10 道、指定资料不存在 5 道。每道正例记录问题、至少一处真实来源定位、需要回答的关键事实。运行前校验文档短引、索引内容哈希、历史消息 ID 和原话；目标文档不存在的题目会检查指定文件确实没有进入本机索引。

使用项目虚拟环境运行：

```powershell
backend/.venv/Scripts/python.exe benchmark/agent_live_eval.py validate
backend/.venv/Scripts/python.exe benchmark/agent_live_eval.py run --ids F01
backend/.venv/Scripts/python.exe benchmark/agent_live_eval.py run
backend/.venv/Scripts/python.exe benchmark/agent_live_eval.py report
```

`run` 通过 `127.0.0.1:9999` 的现有会话与 run 接口执行真实 Agent。每道题建立单独会话，读取最终任务状态和完整事件记录，保存答案、引用、工具调用顺序及耗时。中断后再次运行会跳过已有结果；需要重新执行指定题目时使用 `--ids F01 --rerun`。`--limit` 可以只运行前几题。

修复后的复测使用独立结果组，例如 `run --result-set after-fix-1`，结果会保存在 `.tmp_probe/agent-live-eval-20260929/results-after-fix-1/`。首轮默认结果组为 `baseline`，保存在 `results/`。用 `report --result-set after-fix-1` 查看指定结果组；不要覆盖首轮记录。

逐题 `assessment` 的 `facts_present` 只检查关键词或候选表达是否出现；`target_citations_present` 检查答案引用对应文档或历史消息；`source_quote_visible_in_citation` 表明引用片段里能否直接看到题目短引；`tool_scope_valid` 检查工具类别，并对只查询历史的题目禁止文档工具。`machine_checks_pass` 仅表示这些机器规则均满足。语义是否正确、引用内容是否真正支持各项陈述、遗漏的限定条件与错误推断均需人工核对。短引未完整出现在引用片段时，仍可能存在足以支持答案的其他原文，人工核对必须阅读相应来源。运行状态失败、等待输入或等待批准会保留在逐题结果中，不能计为答案通过。
