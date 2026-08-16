"""识图链路真实用例自检脚本：下载图床图 → 预处理 → 识图 → 打印描述。

用途：配好识图模型后，用真实图床链接验证「抓取 + 识别」整条链路是否正常，
再决定是否重导文档（避免盲目全量重导）。

模型配置读取顺序与设置页一致：data_dir/config.json 的 vision 段（base_url/model/api_key）；
未配置时脚本直接报错退出。也可用环境变量覆盖（与后端 Settings 同前缀）：
  EVERYTHING_RAG_VISION_BASE_URL / EVERYTHING_RAG_VISION_MODEL / EVERYTHING_RAG_VISION_API_KEY

用法（从仓库根或 backend 目录运行均可）：
  python scripts/vision_check.py <url1> [url2 ...]           # 直接测图床 URL
  python scripts/vision_check.py --md <doc.md> --limit 3     # 从 Markdown 抽图批量测
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: 让脚本从仓库根 / backend 目录都能 import app.*（backend 目录加入 sys.path）。
_BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(_BACKEND))

from app.core.settings_store import get_settings_store, is_vision_configured  # noqa: E402
from app.generation.vision import create_vision_model_from_config  # noqa: E402
from app.ingestion.image_fetch import ImageFetcher  # noqa: E402
from app.ingestion.image_preprocess import preprocess_image  # noqa: E402
from app.ingestion.markdown_parser import parse_markdown  # noqa: E402


def _extract_urls_from_md(path: Path) -> list[str]:
    """从 Markdown 抽取全部 http(s) 图片链接（复用入库同一解析器，保证口径一致）。"""
    parsed = parse_markdown(path.read_text(encoding="utf-8"))
    return [
        block.src
        for block in parsed.blocks
        if block.kind == "image" and block.src and block.src.startswith("http")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="识图链路真实用例自检")
    parser.add_argument("urls", nargs="*", help="图床图片 URL（可多个）")
    parser.add_argument("--md", help="从该 Markdown 文件抽取图片链接")
    parser.add_argument("--limit", type=int, default=3, help="--md 时最多测几张（默认 3）")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.md:
        md_path = Path(args.md)
        if not md_path.is_file():
            print(f"文件不存在: {args.md}")
            return 2
        urls = _extract_urls_from_md(md_path)[: max(1, args.limit)]
    if not urls:
        print("未提供任何图片 URL（--md 里也没找到 http 图片链接）")
        return 2

    vision_cfg = get_settings_store().load().vision
    if not is_vision_configured(vision_cfg):
        print("识图模型未配置：请先在设置页 / config.json 配好 vision 的 base_url 与 model")
        return 2

    vision = create_vision_model_from_config(vision_cfg)
    fetcher = ImageFetcher()
    ok = 0
    for url in urls:
        try:
            raw = fetcher.fetch(url)
            jpeg = preprocess_image(raw)
            description = vision.describe(jpeg)
        except Exception as exc:  # 真实用例自检：打印可读原因，不吞掉异常类型
            print(f"✗ {url}\n  失败: {type(exc).__name__}: {exc}")
            continue
        tail = "…" if len(description) > 200 else ""
        print(f"✓ {url}\n  描述({len(description)} 字): {description[:200]}{tail}")
        ok += 1

    print(f"\n结果: {ok}/{len(urls)} 张成功")
    return 0 if ok == len(urls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
