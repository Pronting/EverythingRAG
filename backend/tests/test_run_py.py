"""scripts/run.py 一键拉起验证 + 静态托管（任务 9 交付 C）。

- 单元：--no-browser、健康检查后开浏览器、端口预占和源码 stamp。
- 集成：subprocess 拉起 ``run.py --no-browser`` → 解析 stdout 端口 → GET /api/status 200
  → 终止进程（Windows 下连带清理 uvicorn 子进程，taskkill 进程树）。
- 静态托管：frontend/dist 存在时 GET / 返回 200 HTML，且 /api/status 200。
"""

from __future__ import annotations

import importlib.util
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[2]
RUN_PY = ROOT / "scripts" / "run.py"

_STARTUP_LINE_RE = re.compile(r"http://127\.0\.0\.1:(\d+)")


def _load_run_module() -> ModuleType:
    """经 importlib 加载 scripts/run.py（不在 backend sys.path，避免路径注入）。"""
    spec = importlib.util.spec_from_file_location("scripts_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_mod = _load_run_module()


# ---------------------------------------------------------------- 1. run.py 逻辑单测


def test_no_browser_flag_skips_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-browser=True -> 不开浏览器（不论环境变量）。"""
    monkeypatch.setenv("EVERYTHING_RAG_NO_BROWSER", "0")
    assert run_mod.should_open_browser(no_browser=True) is False


def test_env_var_skips_browser_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """未指定 --no-browser 但环境变量 EVERYTHING_RAG_NO_BROWSER=1 -> 不开浏览器。"""
    monkeypatch.setenv("EVERYTHING_RAG_NO_BROWSER", "1")
    assert run_mod.should_open_browser(no_browser=False) is False


def test_default_opens_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 --no-browser 且环境变量未设 -> 打开浏览器。"""
    monkeypatch.delenv("EVERYTHING_RAG_NO_BROWSER", raising=False)
    assert run_mod.should_open_browser(no_browser=False) is True


def test_parse_args_accepts_no_browser() -> None:
    """--no-browser 解析为 True；缺省为 False。"""
    assert run_mod.parse_args(["--no-browser"]).no_browser is True
    assert run_mod.parse_args([]).no_browser is False


def test_bound_server_socket_reserves_fixed_port() -> None:
    """固定端口在交给 uvicorn 前持续被占用，不自动换端口。"""
    listener = run_mod.bind_server_socket()
    port = listener.getsockname()[1]
    assert port == 9999
    contender = socket.socket()
    try:
        with pytest.raises(OSError):
            contender.bind(("127.0.0.1", port))
    finally:
        contender.close()
        listener.close()


def test_browser_opens_only_after_status_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def ready(_url: str, timeout: float) -> bool:
        events.append(f"ready:{timeout}")
        return True

    monkeypatch.setattr(run_mod, "wait_for_status", ready)
    monkeypatch.setattr(run_mod.webbrowser, "open", lambda _url: events.append("browser"))
    run_mod.open_browser_when_ready("http://127.0.0.1:1234", timeout=1.5)
    assert events == ["ready:1.5", "browser"]


def test_browser_is_not_opened_when_status_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(run_mod, "wait_for_status", lambda _url, timeout: False)
    monkeypatch.setattr(run_mod.webbrowser, "open", opened.append)
    run_mod.open_browser_when_ready("http://127.0.0.1:1234", timeout=0.0)
    assert opened == []


def test_stamp_detects_file_and_directory_changes(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    file = source / "app.ts"
    file.write_text("v1", encoding="utf-8")
    config = tmp_path / "package.json"
    config.write_text("{}", encoding="utf-8")
    stamp_file = tmp_path / "dist" / ".build.stamp"
    inputs = [source, config]

    run_mod.write_stamp(stamp_file, inputs)
    assert run_mod.stamp_matches(stamp_file, inputs) is True
    file.write_text("v2", encoding="utf-8")
    assert run_mod.stamp_matches(stamp_file, inputs) is False


def test_stamp_tracks_missing_path_becoming_present(tmp_path: Path) -> None:
    optional = tmp_path / ".env.production"
    stamp_file = tmp_path / ".stamp"
    run_mod.write_stamp(stamp_file, [optional])
    assert run_mod.stamp_matches(stamp_file, [optional]) is True
    optional.write_text("VITE_FLAG=1", encoding="utf-8")
    assert run_mod.stamp_matches(stamp_file, [optional]) is False


# ---------------------------------------------------------------- 2. 静态托管


def test_static_hosting_serves_index_when_dist_exists() -> None:
    """frontend/dist 存在时 GET / 返回 200 HTML，/api/status 不受静态托管影响。"""
    dist = ROOT / "frontend" / "dist"
    if not dist.is_dir():
        pytest.skip("frontend/dist 不存在，静态托管未启用")
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert resp.text.lstrip().lower().startswith("<!doctype html>")
    assert client.get("/api/status").status_code == 200


# ---------------------------------------------------------------- 3. run.py 拉起冒烟


def test_run_py_no_browser_starts_and_serves_status(tmp_path: Path) -> None:
    """subprocess 拉起 run.py --no-browser → 解析端口 → /api/status 200 → 干净终止。"""
    # 子进程用独立数据目录：避免与真实 ~/.everything-rag 的 Chroma 锁冲突
    # （/api/status 现在会打开向量库；隔离后不依赖/不干扰任何运行中的服务）。
    data_dir = tmp_path / "data"
    env = {**os.environ, "EVERYTHING_RAG_DATA_DIR": str(data_dir)}
    proc = subprocess.Popen(
        [sys.executable, "-u", str(RUN_PY), "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        port = _wait_for_startup_line(proc, timeout=30.0)
        assert port is not None, "未读到 run.py 启动行（见下输出）：\n" + _drain_stdout(proc)
        assert port == 9999
        url = f"http://127.0.0.1:{port}/api/status"
        body, status = _wait_for_status_200(url, timeout=30.0)
        assert status == 200
        assert "everything-rag" in body
    finally:
        _terminate_tree(proc)


def _wait_for_startup_line(proc: subprocess.Popen, timeout: float) -> int | None:
    """后台线程读 stdout，等待启动行并解析端口（避免管道缓冲/死锁）。"""
    lines: queue.Queue[str] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)

    threading.Thread(target=_reader, daemon=True).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        match = _STARTUP_LINE_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def _wait_for_status_200(url: str, timeout: float) -> tuple[str, int]:
    """轮询 GET url 直到 200（uvicorn 启动后路由才就绪）。"""
    deadline = time.monotonic() + timeout
    last: tuple[str, int] = ("", 0)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return resp.read().decode("utf-8"), resp.status
        except (urllib.error.URLError, OSError) as exc:
            last = ("", getattr(exc, "code", 0))
            time.sleep(0.3)
    return last


def _terminate_tree(proc: subprocess.Popen) -> None:
    """终止启动进程；Windows 用进程树清理兜底，避免测试失败遗留服务。"""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
                check=False,
            )
            proc.wait(timeout=10)
            return
        except (subprocess.TimeoutExpired, OSError):
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _drain_stdout(proc: subprocess.Popen) -> str:
    """超时失败时的诊断：尽量读出剩余 stdout（尽力而为，不阻塞）。"""
    assert proc.stdout is not None
    return proc.stdout.read() or ""
