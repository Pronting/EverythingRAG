"""冒烟测试：验证本地 Web 服务可启动、/api/status 正常、隐私基线为 local-only。

依赖：fastapi / uvicorn / pytest / httpx（最小集）。
"""
from fastapi.testclient import TestClient

from app.main import app


def test_status_ok() -> None:
    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app"]["name"] == "everything-rag"
    # 隐私基线：默认 local-only（纯本地，零外发）
    assert data["privacy"]["outbound_state"] == "local-only"


def test_unknown_route_404() -> None:
    client = TestClient(app)
    assert client.get("/api/nonexistent").status_code == 404
