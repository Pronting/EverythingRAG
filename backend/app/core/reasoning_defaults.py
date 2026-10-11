"""历史工作区默认档位的一次性对账：旧默认值 → 当前程序默认值。

**为什么需要它（用户实测复现）**：默认思考强度从 `high` 改为 `off` 后，新建对话读的是
**工作区已落库的默认值**（`frontend/src/App.tsx` 用
`workspace.default_reasoning_level ?? DEFAULT_REASONING_LEVEL`），因此旧工作区里那批
「按旧程序默认值写入的 `high`」会继续覆盖代码默认值 —— 界面上仍然显示「深度思考」。
只改代码常量与 DDL 默认值**不足以**修复，必须把已落库的旧默认值对账掉。

关键事实（决定了本对账是安全的）：设置页与侧边栏**没有**任何入口能修改工作区默认档位
（前端只消费 `default_permission_mode` / `default_reasoning_level`；唯一的写端点
``PUT /api/workspaces/{id}/defaults`` 无 UI 调用方）。用户改档位走的是**会话级**写入。
因此库里出现的 ``high`` 只可能来自「旧程序默认值」，不可能是用户的显式选择 —— 对账只把
**旧程序默认值**精确改写成新默认值，其它取值（`off` / `low` / `max`）一字不动。

两个硬约束：

1. **写库必须走 ``AppStore``**（D19/A3：任何模块都不得绕过它直写 ``workspaces`` 表），
   因此批量改写实现在 :meth:`app.agent.app_store.AppStore.reconcile_default_reasoning_level`；
2. **必须在事件循环内执行**（lifespan 的异步阶段）：早期版本在同步阶段用 ``asyncio.run``
   ⇒ ``RuntimeError`` 被宽泛的 except 吞掉 ⇒ 重启后对账静默不执行、界面照旧显示「深度思考」。

对账只跑一次：完成标记落在设置存储（``data_dir/config.json``），与 app.db 同属一个数据目录。
方向永远是 app.db → ``workspaces.json`` 存档（架构 §6.4：app.db 是唯一权威）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.agent.contracts import DEFAULT_REASONING_LEVEL, ReasoningLevel
from app.core.workspace_store import WorkspaceStore

logger = logging.getLogger(__name__)

#: 当前对账版本。``_LEGACY_DEFAULTS`` 列的是**历史默认值**，不要写成当前默认值。
WORKSPACE_REASONING_DEFAULTS_VERSION = 1

#: 本版本要改写的「历史程序默认值」→ 目标值（当前程序默认值）。
_LEGACY_DEFAULTS: Mapping[str, ReasoningLevel] = {
    ReasoningLevel.HIGH.value: DEFAULT_REASONING_LEVEL,
}


async def reconcile_workspace_default_reasoning(
    settings_store: Any,
    app_store: Any,
    *,
    archive_store: WorkspaceStore | None = None,
) -> int:
    """把仍是旧程序默认档位的工作区改写为当前程序默认档位；返回改写条数。

    - 幂等：同一版本只执行一次（标记在 ``config.json``）；
    - 只动**旧默认值**，不动用户/测试显式写过的档位；
    - app.db 是权威（D28）；``workspaces.json`` 存档行按返回的权威行镜像，避免状态分叉。
    """

    settings = settings_store.load()
    if settings.workspace_reasoning_preferences_version >= WORKSPACE_REASONING_DEFAULTS_VERSION:
        return 0

    changed, updated = await app_store.reconcile_default_reasoning_level(_LEGACY_DEFAULTS)
    if archive_store is not None:
        for record in updated:
            archive_store.adopt_record(record)

    settings_store.mark_workspace_reasoning_preferences_reconciled(
        WORKSPACE_REASONING_DEFAULTS_VERSION
    )
    if changed:
        logger.info(
            "工作区默认思考档位对账完成：改写 %d 个工作区（步骤=reasoning_defaults）", changed
        )
    return changed
