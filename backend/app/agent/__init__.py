"""V1.2.0 Agent 编排包：LangGraph 主图 + 研究子图 + RunManager 业务状态。

本包按 PRD 17 的建议结构建立，职责边界以 `docs/tech/v1.2.0-architecture.md`
的「文件归属表」为准：**每个模块只有一个拥有者工作流**，跨工作流协作走
`app/agent/contracts.py` 的冻结契约，不通过改同一文件实现。

T00a 阶段只建立包与模块骨架；除 :mod:`app.agent.contracts`（枚举与状态机）外
各模块不含实现，由后续工作流认领填充。
"""

from __future__ import annotations

from app.agent import contracts

__all__ = ["contracts"]
