"""头像接口：上传 / 删除 / 读取（纯本地，不涉及任何出网）。

- POST   /api/avatars/{kind}    multipart 上传用户/Agent 头像（PNG/JPG/JPEG/WebP/GIF ≤5MB），
                               落盘 data_dir/avatars/，更新 config.json 并删除旧文件。
- DELETE /api/avatars/{kind}    恢复默认（删除头像文件并清空 config）。
- GET    /api/avatars/{filename} 读取已上传头像（严格 basename 校验，防路径穿越）。

隐私：头像文件仅存本地 data_dir，服务只绑定 127.0.0.1；本模块无任何外部调用。
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.settings_store import SettingsStore, get_settings_store

router = APIRouter()

#: 合法的头像类型（user=用户，agent=Agent 助手）
_AVATAR_KINDS = {"user", "agent"}

#: 允许的图片扩展名 -> MIME
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: 头像大小上限（5MB，纯本地可放宽，但防误传超大文件）
_MAX_AVATAR_BYTES = 5 * 1024 * 1024


def _avatars_dir() -> Path:
    """头像文件目录（data_dir/avatars，惰性创建）。"""
    return get_settings().data_dir / "avatars"


def _looks_like_image(data: bytes) -> bool:
    """魔数校验：确认上传内容确为图片（不依赖第三方图像库）。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return True
    if data.startswith((b"GIF87a", b"GIF89a")):
        return True
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _validate_kind(kind: str) -> None:
    if kind not in _AVATAR_KINDS:
        raise HTTPException(status_code=400, detail="头像类型需为 user 或 agent")


def _delete_file(filename: str | None, keep: str | None = None) -> None:
    """删除旧头像文件；``keep`` 用于跳过同名文件（理论上不会发生，防御性处理）。"""
    if filename is None or filename == keep:
        return
    path = _avatars_dir() / Path(filename).name
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # 删除失败不影响主流程（本地遗留文件可接受）


def _set_avatar(store: SettingsStore, kind: str, filename: str | None) -> str | None:
    """更新 config 中的头像字段，返回被替换的旧文件名。"""
    settings = store.load()
    old = settings.avatars.user if kind == "user" else settings.avatars.agent
    if kind == "user":
        settings.avatars.user = filename
    else:
        settings.avatars.agent = filename
    store.save(settings)
    return old


@router.post("/api/avatars/{kind}")
async def upload_avatar(
    kind: str,
    file: UploadFile = File(...),  # noqa: B008
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """上传头像：校验类型/大小/魔数后落盘，替换旧头像并更新 config。"""
    _validate_kind(kind)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PNG / JPG / JPEG / WebP / GIF 图片")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="头像文件为空")
    if len(data) > _MAX_AVATAR_BYTES:
        raise HTTPException(status_code=400, detail="头像图片不能超过 5MB")
    if not _looks_like_image(data):
        raise HTTPException(status_code=400, detail="文件内容不是有效图片")

    avatars_dir = _avatars_dir()
    avatars_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{kind}-{secrets.token_hex(8)}{ext}"
    (avatars_dir / filename).write_bytes(data)

    old = _set_avatar(store, kind, filename)
    _delete_file(old, keep=filename)

    return {"kind": kind, "url": f"/api/avatars/{filename}", "filename": filename}


@router.delete("/api/avatars/{kind}")
async def delete_avatar(
    kind: str,
    store: SettingsStore = Depends(get_settings_store),  # noqa: B008
) -> dict:
    """恢复默认头像：删除本地文件并清空 config 对应字段。"""
    _validate_kind(kind)
    old = _set_avatar(store, kind, None)
    _delete_file(old)
    return {"kind": kind, "url": None}


@router.get("/api/avatars/{filename}")
async def get_avatar(filename: str) -> FileResponse:
    """读取已上传头像：仅接受 basename 形式的合法文件名，防路径穿越。"""
    name = Path(filename).name
    if name != filename or Path(name).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=404, detail="头像不存在")
    path = _avatars_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="头像不存在")
    media_type = _MEDIA_TYPES[Path(name).suffix.lower()]
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
