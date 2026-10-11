# 处理过程展示

过程区域按执行顺序展示动作与结果。同一次工具调用更新同一个条目，查询的问题保留在可展开的任务列表中。

- `tool.started.summary` 描述本次查询或阅读的对象。
- `tool.completed.summary` 描述工具实际返回的结果；`details` 保存补充说明，`sources` 保存资料名称与正文。
- 阅读文档根据返回正文计算字符数量。搜索资料根据返回条目展示数量。
- 原始参数、内部任务编号和来源定位标识保存在后台记录中。

## 模型思考

`model.reasoning` 使用现有事件保存与 SSE 传输机制。负载包含 `step_id`、`stream_id`、`title` 和 `text`。一次模型调用使用一个独立的 `stream_id`，并行任务按 `step_id` 分别追加思考文本。引用格式纠正产生的再次调用独立展示。

Provider 接收 `reasoning_content` 或 `reasoning` 字段。运行层合并短增量后保存事件，前端按 `event_seq` 去重并追加。刷新后读取事件记录即可恢复。关闭思考时不发布模型思考事件。

Command Code 的 `deepseek/deepseek-v4.1-flash` 纯文本调用支持传递思考档位。该端点实际返回 `reasoning` 字段；即使请求 `thinking.type=disabled`，也可能返回思考文本，界面仍遵守本次会话的关闭设置。此适配仅处理已验证的文本请求，不声明该端点的工具调用能力或输出长度上限。

接口资料：[Provider API](https://commandcode.ai/docs/provider)、[模型页面](https://commandcode.ai/models/deepseek-v4-1-flash)。

## 验证

`backend/live_tests/test_process_stream.py` 使用当前配置的真实模型和知识库，验证文档阅读结果、模型思考先于答案到达、并行任务分别记录、事件补读及关闭思考。测试数据库位于项目的 `.tmp_probe` 中，测试结束后清理。
