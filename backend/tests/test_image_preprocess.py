"""图片预处理单测：解码校验 / 降采样 / JPEG 重编码 / 失败收敛。

全部内存内构造 Pillow 图片，零出网、不落盘、不依赖模型。
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.ingestion.image_preprocess import (
    MAX_EDGE,
    MAX_PIXELS,
    ImageDecodeError,
    preprocess_image,
)


def _png_bytes(size: tuple[int, int] = (100, 80), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """内存内构造一张纯色 PNG。"""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_returns_jpeg() -> None:
    """任意输入 -> 输出为可解码的 JPEG 字节。"""
    out = preprocess_image(_png_bytes())
    with Image.open(io.BytesIO(out)) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_small_image_keeps_size() -> None:
    """达标小图（≤MAX_EDGE 且 ≤MAX_PIXELS）原尺寸保留。"""
    out = preprocess_image(_png_bytes((100, 80)))
    with Image.open(io.BytesIO(out)) as img:
        assert img.size == (100, 80)


def test_large_image_downsampled() -> None:
    """超大图（边长与总像素均超上限）被降采样到约束内。"""
    out = preprocess_image(_png_bytes((2000, 1500)))
    with Image.open(io.BytesIO(out)) as img:
        w, h = img.size
        assert max(w, h) <= MAX_EDGE
        assert w * h <= MAX_PIXELS


def test_wide_image_downsampled_by_edge() -> None:
    """超宽图（如 4000x500）边长超上限 -> 最长边压到 ≤MAX_EDGE。"""
    out = preprocess_image(_png_bytes((4000, 500)))
    with Image.open(io.BytesIO(out)) as img:
        assert max(img.size) <= MAX_EDGE


def test_non_image_bytes_raise_decode_error() -> None:
    """非图片字节 -> ImageDecodeError（不中断整批，由调用方跳过）。"""
    with pytest.raises(ImageDecodeError):
        preprocess_image(b"this is not an image")


def test_empty_bytes_raise_decode_error() -> None:
    """空内容 -> ImageDecodeError。"""
    with pytest.raises(ImageDecodeError):
        preprocess_image(b"")
