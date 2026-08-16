"""识图批量自检：扫描知识库图床链接 → 批量 fetch + 识图 → 汇总报告（含计时）。

用途：配好识图模型后，批量验证「抓取 + 识别」质量与速度；可用 --model/--base-url
覆盖模型名做 A/B 对比（不改 config.json）。

用法：
  python scripts/vision_batch_check.py --limit 100
  python scripts/vision_batch_check.py --limit 40 --model Qwen/Qwen2.5-VL-7B-Instruct --out report_7b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: 让脚本从仓库根 / backend 目录都能 import app.*。
_BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(_BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.core.settings_store import get_settings_store, is_vision_configured  # noqa: E402
from app.generation.vision import DESCRIBE_PROMPT, create_vision_model_from_config  # noqa: E402
from app.ingestion.image_fetch import ImageFetcher  # noqa: E402
from app.ingestion.image_preprocess import preprocess_image  # noqa: E402
from app.ingestion.markdown_parser import parse_markdown  # noqa: E402
from app.ingestion.scanner import scan_directory  # noqa: E402


def _collect_urls(root: Path) -> list[str]:
    """扫描目录下全部 .md，去重收集 http(s) 图片链接（复用入库同一解析器）。"""
    urls: list[str] = []
    seen: set[str] = set()
    for file in scan_directory(root):
        try:
            parsed = parse_markdown(file.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- 单文件解析失败跳过，不中断扫描
            continue
        for block in parsed.blocks:
            src = block.src
            if block.kind == "image" and src and src.startswith("http") and src not in seen:
                seen.add(src)
                urls.append(src)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description="识图批量自检")
    parser.add_argument("--dir", default=None, help="知识库目录（默认 data_dir/documents）")
    parser.add_argument("--limit", type=int, default=100, help="最多测几张（默认 100）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认 data_dir/vision_batch_report.json）")
    parser.add_argument("--model", default=None, help="覆盖识图模型名（A/B 对比用，不改 config）")
    parser.add_argument("--base-url", default=None, help="覆盖 base_url（A/B 对比用）")
    args = parser.parse_args()

    root = Path(args.dir) if args.dir else get_settings().data_dir / "documents"
    # Windows 控制台默认 GBK：重配为 UTF-8（errors=replace 兜底），避免特殊字符编码崩溃
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- stdout 为管道时 reconfigure 可能不可用，忽略
        pass
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 2

    vision_cfg = get_settings_store().load().vision
    if args.model:
        vision_cfg = vision_cfg.model_copy(update={"model": args.model})
    if args.base_url:
        vision_cfg = vision_cfg.model_copy(update={"base_url": args.base_url})
    if not is_vision_configured(vision_cfg):
        print("识图模型未配置：请先在设置页配好 vision 的 base_url 与 model")
        return 2

    urls = _collect_urls(root)
    total = min(max(1, args.limit), len(urls))
    print(f"知识库扫描到 {len(urls)} 张唯一图床图，开始测试前 {total} 张（model={vision_cfg.model}）…", flush=True)

    vision = create_vision_model_from_config(vision_cfg)
    fetcher = ImageFetcher()
    results: list[dict] = []
    ok = 0
    start = time.monotonic()
    for i, url in enumerate(urls[:total], 1):
        t0 = time.monotonic()
        try:
            raw = fetcher.fetch(url)
            jpeg = preprocess_image(raw)
            desc = vision.describe(jpeg, DESCRIBE_PROMPT)
            secs = round(time.monotonic() - t0, 1)
            results.append({"url": url, "ok": True, "chars": len(desc), "seconds": secs, "desc": desc})
            ok += 1
            print(f"[{i}/{total}] OK {len(desc)}字 {secs:.1f}s {url[:80]}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- 单张失败记录原因并继续
            secs = round(time.monotonic() - t0, 1)
            reason = f"{type(exc).__name__}: {exc}"
            results.append({"url": url, "ok": False, "seconds": secs, "error": reason})
            print(f"[{i}/{total}] FAIL {type(exc).__name__} {secs:.1f}s {url[:80]}", flush=True)

    elapsed = round(time.monotonic() - start, 1)
    lens = [r["chars"] for r in results if r["ok"]]
    secs = [r["seconds"] for r in results]
    avg_chars = round(sum(lens) / len(lens)) if lens else 0
    avg_secs = round(sum(secs) / len(secs), 1) if secs else 0
    summary = {
        "model": vision_cfg.model,
        "total_unique_urls": len(urls),
        "tested": total,
        "ok": ok,
        "failed": total - ok,
        "avg_desc_chars": avg_chars,
        "avg_seconds_per_image": avg_secs,
        "total_seconds": elapsed,
        "results": results,
    }
    out_path = Path(args.out) if args.out else get_settings().data_dir / "vision_batch_report.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 汇总 ===", flush=True)
    print(f"成功 {ok}/{total}，失败 {total - ok}", flush=True)
    print(f"平均描述 {avg_chars} 字，平均 {avg_secs}s/张，总耗时 {elapsed}s", flush=True)
    print(f"报告已写入: {out_path}", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
