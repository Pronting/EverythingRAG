"""启动本地 Web 服务（MVP 形态）。对齐技术选型决策文档 §7.3：动态端口 + 数据目录可指定。

用法：backend/.venv/Scripts/python scripts/run.py [--no-browser]
启动后默认自动打开浏览器访问 127.0.0.1:<随机端口>；--no-browser（或环境变量
EVERYTHING_RAG_NO_BROWSER=1）跳过开浏览器，便于自动化/CI 一键拉起验证。
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"


def pick_free_port() -> int:
    """动态端口：让 OS 分配一个空闲端口，避免写死。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def should_open_browser(no_browser: bool) -> bool:
    """--no-browser 优先；未指定时看环境变量 EVERYTHING_RAG_NO_BROWSER=1。"""
    if no_browser:
        return False
    return os.environ.get("EVERYTHING_RAG_NO_BROWSER", "0") != "1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Everything RAG 本地后端")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="跳过自动打开浏览器（自动化/CI 用）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    port = pick_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Everything RAG 后端启动：{url}")
    # 前端构建产物若存在（frontend/dist），由后端静态托管（见 app/main.py）
    if should_open_browser(args.no_browser):
        webbrowser.open(url)
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=BACKEND_DIR,
    )


if __name__ == "__main__":
    sys.exit(main())
