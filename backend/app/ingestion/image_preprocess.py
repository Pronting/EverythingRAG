"""ingestion：图片预处理 —— 解码校验 / EXIF 转正 / 降采样 / JPEG 重编码（方案 B 入库前）。

对齐 T5 §5.2：Pillow 打开 -> verify() 损坏校验 -> exif_transpose() 方向转正
-> convert("RGB")（剥离 alpha/CMYK）-> 降采样（最长边 ≤1280px 且总像素 ≤130 万）
-> JPEG q85 编码。纯本地 CPU 计算，零出网。解码失败抛 ImageDecodeError，
由调用方标记「图片不可读」并跳过（PRD §4.3 验收 7，不中断整批）。
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps

#: 降采样上限：最长边（px）与总像素，取 T5 §5.2 推荐常量，兼顾 Qwen2-VL/MiniCPM-V。
MAX_EDGE = 1280
MAX_PIXELS = 1_300_000
JPEG_QUALITY = 85


class ImageDecodeError(Exception):
    """图片无法解码 / 损坏 / 空内容（调用方据此标记「不可读」并跳过）。"""


def preprocess_image(
    data: bytes, *, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS,
) -> bytes:
    """把任意图片字节解码并重编码为降采样后的 JPEG（纯函数，零出网）。

    - 先 verify() 校验完整性（半截文件 / 非图字节在此即抛，而非转换期才炸）；
    - 重新 open 后 exif_transpose 转正方向、convert("RGB") 剥离 alpha/CMYK；
    - 降采样到 MAX_EDGE × MAX_PIXELS 内，JPEG q85 编码输出。
    """
    if not data:
        raise ImageDecodeError("图片内容为空")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception as exc:  # 解码/校验失败统一收敛为可读错误
        raise ImageDecodeError(f"图片解码失败: {type(exc).__name__}") from exc

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = _downsample(img, max_edge=max_edge, max_pixels=max_pixels)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()


def _downsample(
    img: Image.Image, *, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS,
) -> Image.Image:
    """降采样到最长边 ≤ MAX_EDGE 且总像素 ≤ MAX_PIXELS；已达标则原样返回。"""
    width, height = img.size
    longest = max(width, height)
    pixels = width * height
    if longest <= max_edge and pixels <= max_pixels:
        return img
    scale = min(1.0, max_edge / longest, (max_pixels / pixels) ** 0.5)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return img.resize((new_width, new_height), Image.LANCZOS)
