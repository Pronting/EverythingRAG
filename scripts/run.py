"""可靠启动 Everything RAG：固定端口 9999、等待健康检查，再打开浏览器。

用法：backend/.venv/Scripts/python scripts/run.py [--no-browser]

``EverythingRagStart.bat`` 还通过本文件的隐藏 stamp 参数判断依赖和前端产物是否过期。stamp
只使用 Python 标准库，因此新虚拟环境尚未安装项目依赖时也能执行。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
HOST = "127.0.0.1"
PORT = 9999
DEFAULT_READY_TIMEOUT_SECONDS = 30.0


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def bind_server_socket(host: str = HOST) -> socket.socket:
    """持续持有固定端口的 socket，端口占用时明确报错，不切换随机端口。"""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, PORT))
        listener.set_inheritable(True)
        return listener
    except OSError:
        listener.close()
        raise


def should_open_browser(no_browser: bool) -> bool:
    """--no-browser 优先；未指定时看环境变量 EVERYTHING_RAG_NO_BROWSER=1。"""

    if no_browser:
        return False
    return os.environ.get("EVERYTHING_RAG_NO_BROWSER", "0") != "1"


def wait_for_status(url: str, timeout: float = DEFAULT_READY_TIMEOUT_SECONDS) -> bool:
    """轮询 ``/api/status``，只有收到 HTTP 200 才认为应用已经可交互。"""

    deadline = time.monotonic() + max(0.0, timeout)
    status_url = f"{url.rstrip('/')}/api/status"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.2)
    return False


def open_browser_when_ready(url: str, timeout: float = DEFAULT_READY_TIMEOUT_SECONDS) -> None:
    """健康检查成功后打开浏览器；超时只报错，不把用户带到未就绪页面。"""

    if wait_for_status(url, timeout=timeout):
        webbrowser.open(url)
        return
    print(
        f"[启动警告] {timeout:g} 秒内未通过 /api/status 健康检查，浏览器未自动打开。"
        f"请查看上方服务日志，或稍后手动访问 {url}",
        file=sys.stderr,
        flush=True,
    )


def calculate_stamp(inputs: list[Path], base_dir: Path | None = None) -> str:
    """对文件/目录的相对名称与内容做稳定 SHA-256；缺失路径也进入摘要。"""

    base = (base_dir or Path.cwd()).resolve()
    digest = hashlib.sha256()
    for raw in inputs:
        path = raw if raw.is_absolute() else base / raw
        label = raw.as_posix()
        if path.is_dir():
            files = sorted(
                (candidate for candidate in path.rglob("*") if candidate.is_file()),
                key=lambda candidate: candidate.relative_to(path).as_posix(),
            )
            _update_digest(digest, f"D:{label}", b"")
            for file in files:
                relative = file.relative_to(path).as_posix()
                _update_digest(digest, f"F:{label}/{relative}", file.read_bytes())
        elif path.is_file():
            _update_digest(digest, f"F:{label}", path.read_bytes())
        else:
            _update_digest(digest, f"M:{label}", b"")
    return digest.hexdigest()


def stamp_matches(stamp_file: Path, inputs: list[Path], base_dir: Path | None = None) -> bool:
    """stamp 不存在、不可读或内容变化时返回 False。"""

    try:
        recorded = stamp_file.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == calculate_stamp(inputs, base_dir=base_dir)


def write_stamp(stamp_file: Path, inputs: list[Path], base_dir: Path | None = None) -> None:
    """原子写 stamp，避免安装/构建中断留下“已完成”的假标记。"""

    value = calculate_stamp(inputs, base_dir=base_dir)
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = stamp_file.with_name(f"{stamp_file.name}.tmp")
    temporary.write_text(f"{value}\n", encoding="utf-8")
    os.replace(temporary, stamp_file)


def _update_digest(digest: _Digest, label: str, payload: bytes) -> None:
    """给摘要加入无歧义的长度前缀，避免路径/内容拼接碰撞。"""

    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Everything RAG 本地后端")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="跳过自动打开浏览器（自动化/CI 用）",
    )
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument("--check-stamp", type=Path, help=argparse.SUPPRESS)
    maintenance.add_argument("--write-stamp", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--stamp-input",
        action="append",
        type=Path,
        default=[],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp_result = _handle_stamp_command(args)
    if stamp_result is not None:
        return stamp_result
    return _serve(no_browser=args.no_browser)


def _handle_stamp_command(args: argparse.Namespace) -> int | None:
    stamp_file = args.check_stamp or args.write_stamp
    if stamp_file is None:
        return None
    if not args.stamp_input:
        print("[启动检查失败] stamp 至少需要一个 --stamp-input", file=sys.stderr)
        return 2
    try:
        if args.check_stamp is not None:
            return 0 if stamp_matches(stamp_file, args.stamp_input) else 1
        write_stamp(stamp_file, args.stamp_input)
    except OSError as exc:
        print(f"[启动检查失败] 无法更新环境 stamp（{type(exc).__name__}）", file=sys.stderr)
        return 2
    return 0


def _serve(*, no_browser: bool) -> int:
    try:
        listener = bind_server_socket()
    except OSError as exc:
        print(
            f"[启动失败] 无法绑定 {HOST}:{PORT}（{type(exc).__name__}）。"
            "请检查端口是否已被占用；不会自动切换端口。",
            file=sys.stderr,
        )
        return 1

    port = int(listener.getsockname()[1])
    url = f"http://{HOST}:{port}"
    print(f"Everything RAG 后端准备启动：{url}", flush=True)

    if should_open_browser(no_browser):
        threading.Thread(
            target=open_browser_when_ready,
            args=(url,),
            daemon=True,
            name="open-browser-when-ready",
        ).start()

    try:
        # 延迟导入：stamp 子命令必须能在尚未安装依赖的新 venv 中运行。
        import uvicorn

        os.chdir(BACKEND_DIR)
        backend_path = str(BACKEND_DIR)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        config = uvicorn.Config("app.main:app", host=HOST, port=port)
        server = uvicorn.Server(config)
        server.run(sockets=[listener])
        return 0 if server.started else 1
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print("[启动失败] 后端初始化失败，请查看上方日志。", file=sys.stderr)
        return code or 1
    except Exception as exc:  # noqa: BLE001 -- 启动边界需转成面向用户的稳定错误
        print(
            f"[启动失败] 后端未能启动（{type(exc).__name__}），请查看上方日志。",
            file=sys.stderr,
        )
        return 1
    finally:
        listener.close()


if __name__ == "__main__":
    sys.exit(main())
