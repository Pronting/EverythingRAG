"""启动本地 Web 服务（MVP 形态）。对齐技术选型决策文档 §7.3：动态端口 + 数据目录可指定。

用法：backend/.venv/Scripts/python scripts/run.py
启动后自动打开浏览器访问 127.0.0.1:<随机端口>。
"""
from __future__ import annotations

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


def main() -> int:
    port = pick_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Everything RAG 后端启动：{url}")
    webbrowser.open(url)
    # 前端构建产物若存在（frontend/dist），可在此由后端静态托管（MVP 占位）
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
