"""Stop 钩子：会话结束时提醒核对当前阶段验收点。

输出 systemMessage（显示给用户），不阻断。引导 Agent/用户检查
.claude/project-state.md 中当前阶段的验收 checklist 并更新状态。
"""
import json
import pathlib
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出（含中文）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / ".claude" / "project-state.md"

# 从 project-state.md 提取「当前阶段」标题行作为提示上下文
current_stage = ""
if STATE_FILE.exists():
    for line in STATE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("**Stage") or line.startswith("## 当前阶段"):
            current_stage = line.lstrip("#* ").strip()
            break

output = {
    "systemMessage": (
        "阶段验收提醒：本轮工作完成。\n"
        f"当前阶段：{current_stage or '见 .claude/project-state.md'}\n"
        "请核对 project-state.md 中该阶段的验收 checklist；"
        "若已达成，更新「当前阶段 / 下一步 / 最后更新」，再做下一阶段。"
    )
}
json.dump(output, sys.stdout, ensure_ascii=False)
