"""头像 API 测试：上传 / 删除 / 读取 + 类型校验 + 路径穿越防护。

conftest 注入 tmp 设置存储与 data_dir，不触碰真实 ~/.everything-rag。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_GIF = b"GIF89a" + b"\x00" * 16


def _client() -> TestClient:
    return TestClient(app)


def test_upload_and_serve_user_avatar() -> None:
    """上传用户头像 -> 设置视图回显 URL -> GET 读取字节一致。"""
    client = _client()
    resp = client.post("/api/avatars/user", files={"file": ("me.png", _PNG, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "user"
    assert data["url"].startswith("/api/avatars/user-")
    assert data["url"].endswith(".png")

    settings = client.get("/api/settings").json()
    assert settings["avatars"]["user"] == data["url"]
    assert settings["avatars"]["agent"] is None

    got = client.get(data["url"])
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.content == _PNG


def test_replace_avatar_deletes_old_file() -> None:
    """重复上传替换旧头像：旧 URL 404，新 URL 可读。"""
    client = _client()
    first = client.post("/api/avatars/agent", files={"file": ("a.gif", _GIF, "image/gif")}).json()
    second = client.post("/api/avatars/agent", files={"file": ("b.png", _PNG, "image/png")}).json()
    assert first["url"] != second["url"]
    assert client.get(first["url"]).status_code == 404
    assert client.get(second["url"]).status_code == 200


def test_delete_avatar_restores_default() -> None:
    """删除头像 -> 设置视图回 null，文件 404。"""
    client = _client()
    up = client.post("/api/avatars/user", files={"file": ("me.png", _PNG, "image/png")}).json()
    resp = client.delete("/api/avatars/user")
    assert resp.status_code == 200
    assert client.get("/api/settings").json()["avatars"]["user"] is None
    assert client.get(up["url"]).status_code == 404


def test_upload_rejects_bad_kind() -> None:
    """非法头像类型 -> 400。"""
    client = _client()
    resp = client.post("/api/avatars/team", files={"file": ("a.png", _PNG, "image/png")})
    assert resp.status_code == 400


def test_upload_rejects_bad_extension() -> None:
    """非图片扩展名 -> 400。"""
    client = _client()
    resp = client.post("/api/avatars/user", files={"file": ("a.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_non_image_bytes() -> None:
    """扩展名合法但内容非图片（魔数不符）-> 400。"""
    client = _client()
    resp = client.post("/api/avatars/user", files={"file": ("a.png", b"not an image", "image/png")})
    assert resp.status_code == 400


def test_serve_rejects_traversal_and_missing() -> None:
    """路径穿越 / 不存在文件 -> 404（或路由不匹配 404）。"""
    client = _client()
    assert client.get("/api/avatars/..%2F..%2Fconfig.json").status_code in (400, 404)
    assert client.get("/api/avatars/user-0000000000000000.png").status_code == 404


def test_put_settings_preserves_avatar() -> None:
    """PUT 其它设置（覆盖式更新）不得清空已上传头像。"""
    client = _client()
    up = client.post("/api/avatars/user", files={"file": ("me.png", _PNG, "image/png")}).json()
    resp = client.put("/api/settings", json={"answer_preferences": {"verbosity": "concise"}})
    assert resp.status_code == 200
    settings = client.get("/api/settings").json()
    assert settings["avatars"]["user"] == up["url"]
