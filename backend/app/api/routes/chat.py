"""旧问答入口 ``POST /api/chat`` 已下线：返回 **410 Gone**（PRD 13.2）。Owner: T09。

V1.1.0 的 ``/api/chat``（帧契约 ``meta/reasoning/token/tool/done/error``）是单轮 SSE
问答入口。V1.2.0 起问答统一走**持久任务**：

- 建 run：``POST /api/workspaces/{workspace_id}/runs``
- 订阅事件：``GET  /api/workspaces/{workspace_id}/runs/{run_id}/events``（v2 事件，``id`` + JSON ``data``）
- 批准 / 澄清 / 停止 / 继续：``/api/workspaces/{workspace_id}/...`` 下的对应端点

为什么保留本文件而不是删掉：410 是**产品语义**的一部分 —— 老前端拿到 410 就知道必须迁移，
拿到 404 会误判为「服务没起来」。``Allow`` / ``Link`` 头给出新入口，便于日志与排障。
本路由不再注册任何 ``Depends``：旧链路（全局向量库 / 全局图片库 / ChatService）因此
**结构性不可达**，不会再有「按区隔离漏改」的隐患。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

#: 旧入口的下线原因（不泄露内部结构，只说明迁移目标）。
_GONE_DETAIL = "POST /api/chat 已在 V1.2.0 下线，请改用 run API 建任务并订阅事件"

#: 新入口模板（前端据此迁移，见回报里给 t17 的 7 处耦合点）。
_REPLACEMENT = "/api/workspaces/{workspace_id}/runs"


@router.post("/api/chat")
async def chat_gone() -> JSONResponse:
    """旧入口：410 Gone + 新入口提示（不解析请求体，避免把旧格式当合法输入）。"""

    return JSONResponse(
        status_code=410,
        content={
            "error_code": "chat_endpoint_removed",
            "detail": _GONE_DETAIL,
            "replacement": _REPLACEMENT,
        },
        headers={"Link": f'<{_REPLACEMENT}>; rel="successor-version"'},
    )
