import argparse
import asyncio
import io
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import tomlkit
from dotenv import dotenv_values, set_key
from mcp.client.streamable_http import streamable_http_client
from ruamel.yaml import YAML

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:9999/mcp/"
NAME = "everything_rag"
TOKEN_KEY = "EVERYTHING_RAG_MCP_TOKEN"
TOOLS = {"get_knowledge_overview", "lookup_catalog", "search_knowledge", "read_document"}


def settings():
    values = dict(dotenv_values(ROOT / "backend" / ".env"))
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for index in range(winreg.QueryInfoKey(key)[1]):
                name, value, _ = winreg.EnumValue(key, index)
                if name.startswith("EVERYTHING_RAG_"):
                    values[name] = value
    values.update({key: value for key, value in os.environ.items()
                   if key.startswith("EVERYTHING_RAG_")})
    return values


def private_write(path, content, backup_dir):
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        destination = backup_dir / f"{path.parent.name}-{path.name}-{secrets.token_hex(4)}"
        shutil.copy2(path, destination)
        destination.chmod(0o600)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.new")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
    os.replace(temporary, path)


def check_existing(entry):
    if entry and (entry.get("url") != URL or entry.get("command")):
        raise RuntimeError("everything_rag 名称已被其他连接使用，请检查对应客户端配置。")


def config_updates(clients, token, home):
    headers = {"Authorization": f"Bearer {token}"}
    updates = []
    if "codex" in clients:
        path = home / ".codex" / "config.toml" if home else (
            Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
        )
        config = tomlkit.parse(path.read_text(encoding="utf-8-sig")) if path.exists() else (
            tomlkit.document()
        )
        servers = config.setdefault("mcp_servers", {})
        entry = servers.setdefault(NAME, {})
        check_existing(entry)
        for key in ("bearer_token_env_var", "env_http_headers", "http_headers_helper"):
            entry.pop(key, None)
        entry.update(url=URL, http_headers=headers, startup_timeout_sec=30,
                     tool_timeout_sec=60, enabled=True)
        updates.append((path, tomlkit.dumps(config)))
    if "claude" in clients:
        base = home if home else Path.home()
        path = (base / ".claude.json") if home or not os.environ.get("CLAUDE_CONFIG_DIR") else (
            Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".claude.json"
        )
        config = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        entry = config.setdefault("mcpServers", {}).setdefault(NAME, {})
        check_existing(entry)
        entry.pop("headersHelper", None)
        entry.update(type="http", url=URL, headers=headers)
        updates.append((path, json.dumps(config, ensure_ascii=False, indent=2) + "\n"))
    if "dsh" in clients:
        base = home / ".dsh" if home else Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
        path = base / "cordis.patch.yml"
        yaml = YAML()
        yaml.preserve_quotes = True
        patches = yaml.load(path.read_text(encoding="utf-8-sig")) if path.exists() else []
        if patches is None:
            patches = []
        if not isinstance(patches, list):
            raise RuntimeError("DSH cordis.patch.yml 必须包含 YAML 列表。")
        matches = [entry for patch in patches for entry in patch.get("insert", [])
                   if entry.get("id") == "everything-rag-mcp"
                   or entry.get("config", {}).get("serverName") == NAME]
        if len(matches) > 1:
            raise RuntimeError("DSH 存在多个 everything_rag 配置，请检查重复条目。")
        if matches:
            entry = matches[0]
            if entry.get("name") != "@deepseek-ai/dsh-mcp-client":
                raise RuntimeError("DSH everything-rag-mcp 条目已被其他插件使用。")
            check_existing(entry.get("config", {}))
        else:
            entry = {"id": "everything-rag-mcp", "name": "@deepseek-ai/dsh-mcp-client"}
            patches.append({"insert": [entry]})
        entry.setdefault("config", {}).update(
            serverName=NAME, transport="streamable-http", url=URL, headers=headers,
            failOnStartupError=True, toolCallTimeoutMs=60000,
        )
        output = io.StringIO()
        yaml.dump(patches, output)
        updates.append((path, output.getvalue()))
    if "generic" in clients:
        base = home if home else Path.home()
        path = base / ".everything-rag" / "mcp-client.json"
        config = {"mcpServers": {NAME: {"type": "http", "url": URL, "headers": headers}}}
        updates.append((path, json.dumps(config, ensure_ascii=False, indent=2) + "\n"))
    return updates


async def verify(token, read_overview=True):
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=60) as http,
        Client(streamable_http_client(URL, http_client=http), read_timeout_seconds=60) as client,
    ):
        result = await client.list_tools()
        if {tool.name for tool in result.tools} != TOOLS:
            raise RuntimeError("MCP 工具列表与 Everything RAG 不一致。")
        if read_overview:
            result = await client.call_tool("get_knowledge_overview", {"limit": 1})
            if result.is_error:
                raise RuntimeError("知识库概览查询失败，请检查 embedding 配置和索引状态。")
            return {"tools": len(TOOLS), "file_count": result.structured_content["file_count"]}
        return {"tools": len(TOOLS)}


def listening():
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", 9999)) == 0


def start(token, values, runtime_dir):
    if listening():
        asyncio.run(verify(token, read_overview=False))
        return
    runtime_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {key: str(value) for key, value in values.items() if value is not None}
    environment.update({TOKEN_KEY: token, "EVERYTHING_RAG_MCP_ENABLED": "true"})
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    with (runtime_dir / "server.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "run.py"), "--no-browser"],
            cwd=ROOT / "backend", env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, **options,
        )
    deadline = time.monotonic() + 60
    while not listening():
        if process.poll() is not None:
            raise RuntimeError(f"后台服务启动失败，请查看 {runtime_dir / 'server.log'}。")
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait(timeout=10)
            raise TimeoutError("后台服务未在 60 秒内启动。")
        time.sleep(0.2)
    # TCP 建立后，MCP initialize 会等待应用完成启动。
    verified = False
    try:
        asyncio.run(verify(token, read_overview=False))
        verified = True
    finally:
        if not verified and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    print("Everything RAG 后台服务已启动。")


def main():
    parser = argparse.ArgumentParser(description="Everything RAG MCP 自动接入")
    parser.add_argument("action", choices=["install", "start", "check"])
    parser.add_argument("--client", choices=["auto", "all", "codex", "claude", "dsh", "generic"],
                        default="auto")
    parser.add_argument("--config-home", type=Path, help="将客户端配置写入指定目录")
    args = parser.parse_args()
    values = settings()
    token = values.get(TOKEN_KEY)
    if not token and args.action == "install":
        token = secrets.token_urlsafe(48)
    if not token or len(token) < 32:
        raise RuntimeError("MCP 凭证尚未配置，请按照 mcp/INSTALL.md 执行安装。")
    runtime_dir = Path(values.get("EVERYTHING_RAG_DATA_DIR") or Path.home() / ".everything-rag") / "mcp"
    if args.action == "check":
        print(json.dumps(asyncio.run(verify(token)), ensure_ascii=False))
        return
    if args.action == "start":
        start(token, values, runtime_dir)
        print(json.dumps(asyncio.run(verify(token)), ensure_ascii=False))
        return
    clients = ["codex", "claude", "dsh"] if args.client == "all" else [args.client]
    if args.client == "auto":
        clients = [name for name in ("codex", "claude", "dsh") if shutil.which(name)]
    if not clients:
        raise RuntimeError("未检测到 Agent 命令。可使用 --client 指定客户端。")
    updates = config_updates(clients, token, args.config_home)
    start(token, values, runtime_dir)
    report = asyncio.run(verify(token))
    backup_dir = runtime_dir / "backups" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for path, content in updates:
        private_write(path, content, backup_dir)
        print(f"已配置 {path}")
    # 服务与客户端共用凭证；dotenv 保留其他配置及注释。
    env_path = ROOT / "backend" / ".env"
    persisted = dotenv_values(env_path)
    changed = persisted.get(TOKEN_KEY) != token or (
        persisted.get("EVERYTHING_RAG_MCP_ENABLED") != "true"
    )
    if changed and env_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_path, backup_dir / "backend.env")
        (backup_dir / "backend.env").chmod(0o600)
    if changed:
        set_key(env_path, TOKEN_KEY, token)
        set_key(env_path, "EVERYTHING_RAG_MCP_ENABLED", "true")
    print(json.dumps(report, ensure_ascii=False))
    print("配置完成。重新连接 MCP 或重新启动 Agent 后即可使用。")


if __name__ == "__main__":
    main()
