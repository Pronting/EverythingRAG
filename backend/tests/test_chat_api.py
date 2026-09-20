"""``POST /api/chat`` 下线验证（PRD 13.2）：旧入口**必须** 410，新入口必须在位。

V1.2.0 把单轮 SSE 问答换成持久任务（run API + v2 事件）。本文件保留「证明旧入口
下线」的用例，而不是删掉整个文件：

1. 旧入口对任何请求体都返回 **410 Gone**（不是 404 —— 404 会被前端误判成「服务没起来」）；
2. 旧链路的依赖（``get_chat_service`` / ChatService 注入点）**结构性不可达**：
   路由模块里不再存在这个 provider，``dependency_overrides`` 也就无从挂上；
3. 410 路径**零出网**（阻断非回环连接后仍然正常返回）；
4. 替代入口在 OpenAPI 里真实存在（前端照此迁移，耦合点清单见 T09 回报）。
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from app.api.routes import chat as chat_module
from app.main import app


def test_old_endpoint_returns_410_gone() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "你好"})
    assert response.status_code == 410
    body = response.json()
    assert body["error_code"] == "chat_endpoint_removed"
    assert body["replacement"] == "/api/workspaces/{workspace_id}/runs"
    assert "run API" in body["detail"] or "runs" in body["detail"]


def test_old_endpoint_is_410_even_with_legacy_payload() -> None:
    """旧格式（history / web_search / images）也返回 410：不再解析旧请求体。"""

    client = TestClient(app)
    legacy = {
        "message": "你好",
        "history": [{"role": "user", "content": "上一轮"}],
        "web_search": True,
        "images": ["data:image/png;base64,AAAA"],
    }
    response = client.post("/api/chat", json=legacy)
    assert response.status_code == 410


def test_old_service_dependency_is_unreachable() -> None:
    """旧链路的依赖注入点已随路由一起移除（不会被 dependency_overrides 复活）。"""

    assert not hasattr(chat_module, "get_chat_service")
    assert "GET" not in {method for route in chat_module.router.routes for method in route.methods}


def test_old_endpoint_does_not_touch_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """410 路径零外发：阻断一切非回环连接后仍然正常返回。"""

    monkeypatch.setattr(socket, "create_connection", _raise_outbound)
    original_connect = socket.socket.connect

    def guarded_connect(
        self: socket.socket,
        address: tuple | str | bytes | int,
        *args: object,
        **kwargs: object,
    ) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Unexpected outbound network call to {host!r}")
        return original_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    client = TestClient(app)
    assert client.post("/api/chat", json={"message": "你好"}).status_code == 410


def _raise_outbound(*args: object, **kwargs: object) -> None:
    raise AssertionError("Unexpected outbound network call")


def test_replacement_run_endpoints_are_registered() -> None:
    """替代入口必须在位：建 run、订阅事件、批准、澄清、停止、继续（PRD 13.2）。"""

    paths = set(app.openapi()["paths"])
    for path in (
        "/api/workspaces/{workspace_id}/runs",
        "/api/workspaces/{workspace_id}/runs/{run_id}",
        "/api/workspaces/{workspace_id}/runs/{run_id}/events",
        "/api/workspaces/{workspace_id}/runs/{run_id}/cancel",
        "/api/workspaces/{workspace_id}/runs/{run_id}/resume",
        "/api/workspaces/{workspace_id}/approvals/{approval_id}",
        "/api/workspaces/{workspace_id}/inputs/{input_request_id}",
    ):
        assert path in paths, path
