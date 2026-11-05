#!/bin/bash
set -euo pipefail

on_exit() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        printf '\n启动失败，请查看上方错误及 mcp/INSTALL.md。\n' >&2
        if [ -t 0 ]; then
            read -r -p '按回车关闭窗口。' answer || :
        fi
    fi
    exit "$status"
}
trap on_exit EXIT

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
python="$project_root/backend/.venv/bin/python"
export PYTHONUTF8=1

if [ ! -x "$python" ]; then
    printf '未找到项目 Python 环境：%s\n请按照 mcp/INSTALL.md 完成项目安装。\n' "$python" >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" -r "$script_dir/requirements.txt"
else
    "$python" -m pip install -r "$script_dir/requirements.txt"
fi

"$python" "$script_dir/manage.py" start
printf '\n服务状态：MCP 连接检查通过，服务正在后台运行。\n'
printf '服务地址：http://127.0.0.1:9999/mcp/\n'
printf '工具数量和文档数量见上方检查结果。\n'
printf '关闭此窗口后，后台服务继续运行。\n'
if [ -t 0 ]; then
    read -r -p '按回车关闭窗口。' answer || :
fi
