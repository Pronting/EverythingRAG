"""PostToolUse 测试门钩子：写代码后自动尝试运行相关测试。

规则（渐进式，脚手架搭好后自动生效）：
- 仅当改动文件位于 backend/ 或 frontend/ 内、且扩展名是代码文件时触发；
- 仅当对应项目已配置测试框架（backend/pyproject.toml + tests/、frontend/package.json）才真正运行；
- 全部通过 -> 静默（不输出，不干扰）；有失败 -> 输出 additionalContext 提示修复；
- 任何异常/超时都 exit 0，绝不影响主流程。
"""
import json
import pathlib
import subprocess
import sys

# Windows 控制台默认 GBK，强制 UTF-8（stdin 输入 JSON 与 stdout 输出均可能含中文/emoji）
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".css"}

try:
    data = json.load(sys.stdin)
    fp = (data.get("tool_input") or {}).get("file_path", "")
except Exception:
    sys.exit(0)

if not fp:
    sys.exit(0)

path = pathlib.Path(fp)
if path.suffix.lower() not in CODE_EXTS:
    sys.exit(0)
if path.name == "__init__.py":  # 包元文件不算实质变更，跳过测试门
    sys.exit(0)

# 仅 backend/ 或 frontend/ 内的代码文件触发
try:
    rel = path.resolve().relative_to(ROOT.resolve())
except ValueError:
    sys.exit(0)
rel_str = rel.as_posix()
if not (rel_str.startswith("backend/") or rel_str.startswith("frontend/")):
    sys.exit(0)


def run(cmd, cwd, timeout=150):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + "\n" + r.stderr).strip()
    except FileNotFoundError:
        return -1, f"命令不存在: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, f"超时(>{timeout}s): {cmd[0]}"
    except Exception as e:  # noqa: BLE001
        return -3, str(e)


fails = []

# 后端（仅当存在测试文件时才跑，避免空 tests/ 或无依赖时报错）
backend = ROOT / "backend"
if (backend / "pyproject.toml").exists() and (backend / "tests").is_dir():
    has_tests = any(
        p.suffix == ".py" and (p.name.startswith("test_") or p.name.endswith("_test.py"))
        for p in (backend / "tests").rglob("*.py")
    )
    if has_tests:
        # 优先用项目 venv 的 python 跑 pytest，避免 PATH 里的系统 pytest 版本/依赖不一致
        venv_py = backend / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
        cmd = [str(venv_py), "-m", "pytest", "-q", "--no-header"] if venv_py.exists() else ["pytest", "-q", "--no-header"]
        rc, msg = run(cmd, backend)
        # 0=通过；-1=命令不存在(依赖未就绪)——均不视为失败
        if rc not in (0, -1):
            fails.append(f"[后端 pytest FAIL exit={rc}]\n{msg[-1500:]}")

# 前端（仅当 package.json 声明了 test script 时才跑）
frontend = ROOT / "frontend"
if (frontend / "package.json").exists():
    import json as _json
    try:
        pkg = _json.loads((frontend / "package.json").read_text(encoding="utf-8"))
        has_test_script = bool((pkg.get("scripts") or {}).get("test"))
    except Exception:
        has_test_script = False
    if has_test_script:
        rc, msg = run(["npm", "test", "--", "--run"], frontend)
        if rc != 0:
            fails.append(f"[前端 npm test FAIL exit={rc}]\n{msg[-1500:]}")

if not fails:
    sys.exit(0)  # 全通过或无可跑测试 -> 静默

output = {
    "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": "【测试门】修改 " + path.name + " 后测试有失败，请修复：\n" + "\n".join(fails),
    }
}
json.dump(output, sys.stdout, ensure_ascii=False)
