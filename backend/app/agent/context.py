"""模型上下文组装、token 预算与压缩（PRD 9.2、9.3）。Owner: T03（model-provider）。

本模块是**门面（facade）**：`app/agent/context.py` 归 model-provider 所有，
所以实现落在 :mod:`app.generation.context`（与 generation 其他模块内聚），
这里只导出运行层需要的名字，避免出现第二份上下文实现。

- 固定输入顺序（PRD 9.3）：稳定职责提示 → 稳定工具定义 → 工作区与会话固定快照 →
  当前上下文段历史 → 最新问题或工具结果；动态时间、检索结果与批准结果追加在
  动态部分。
- 上下文复用与压缩：完整保留 tool_calls 与结果组，压缩前等待本组工具结果完整
  返回，只增加该 Agent 的 ``agent_context_epoch``（PRD 9.3）。
- token 预算：``local_prompt_tokens + reserved_output_tokens + protocol_margin
  <= provider_context_limit``，可用输入空间达 80% 时安排整理（PRD 9.3 / 18.1）。
- DeepSeek 完整协议历史回传：带 ``tools`` 的会话必须保留并回传此前 assistant
  消息的完整 ``reasoning_content``（含未调用工具的轮次），工具结果携带
  ``tool_call_id``（PRD 9.2）。
- 本地计数走 :mod:`app.generation.tokenizer`（官方 tokenizer + 文件 SHA-256）；
  资源缺失时抛明确错误，不按字符数估算。

``ModelCapabilities`` 等跨工作流字段定义在 :mod:`app.agent.contracts`。
"""

from __future__ import annotations

from app.generation.context import (
    DEFAULT_COMPACTION_RATIO as DEFAULT_CONTEXT_COMPACTION_RATIO,
)
from app.generation.context import (
    AgentContext,
    BudgetStatus,
    ContextBudget,
    ContextSection,
    ContextSnapshot,
    TokenCounter,
    budget_for_request,
    build_context,
    compact_context,
    count_prompt_tokens,
    count_reasoning_messages,
    default_reserved_output_tokens,
    evaluate_context_budget,
    should_compact,
)
from app.generation.tokenizer import TokenizerResourceError

__all__ = [
    "DEFAULT_CONTEXT_COMPACTION_RATIO",
    "AgentContext",
    "BudgetStatus",
    "ContextBudget",
    "ContextSection",
    "ContextSnapshot",
    "TokenCounter",
    "TokenizerResourceError",
    "budget_for_request",
    "build_context",
    "compact_context",
    "count_prompt_tokens",
    "count_reasoning_messages",
    "default_reserved_output_tokens",
    "evaluate_context_budget",
    "should_compact",
]
