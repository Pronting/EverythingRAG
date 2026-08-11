"""SessionStart 钩子：把 .claude/project-state.md 注入会话上下文。

让每个新会话的 Agent 开会话即知「当前阶段 / 下一步 / 验收 checklist」。
输出 JSON 的 hookSpecificOutput.additionalContext 会被注入模型上下文。
"""
import json
import pathlib
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出（project-state.md 含 emoji/中文）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 定位项目根：<root>/.claude/hooks/session_start.py -> parents[2] = root
ROOT = pathlib.Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / ".claude" / "project-state.md"

content = STATE_FILE.read_text(encoding="utf-8") if STATE_FILE.exists() else "（project-state.md 缺失）"

output = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": "【项目状态】\n" + content,
    }
}
json.dump(output, sys.stdout, ensure_ascii=False)
