"""一次性评测脚本的本地用例表加载器（PRD §7.2 脱敏）。

三个历史脚本（``eval_retrieval.py`` / ``eval_retrieval_hybrid.py`` /
``eval_retrieval_business.py``）原本把**真实提问 + 期望文档名**内联在源码里，
属于私人正文。现在这些表只保存在 Git 忽略的本地文件：

    .tmp_probe/rag-relevance/local-eval-cases.json
    {"<脚本名>": {"positive": [[问题, 期望命中子串], ...], "negative": [...]}}

本模块负责读取（路径可用 ``EVERYTHING_RAG_LOCAL_EVAL_CASES`` 覆盖）。
本地文件缺失时抛 ``LocalEvalCasesMissing``，并在消息里写明如何恢复，
**不会**用空表静默跑出虚假结论。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_PATH = REPO_ROOT / ".tmp_probe" / "rag-relevance" / "local-eval-cases.json"
PRIVATE_PATH_ENV = "EVERYTHING_RAG_LOCAL_EVAL_CASES"

SCRIPT_STEMS = (
    "eval_retrieval",
    "eval_retrieval_hybrid",
    "eval_retrieval_business",
)


class LocalEvalCasesMissing(FileNotFoundError):
    """本地用例表不存在（Git 忽略，只在本机保存）。"""


def resolve_private_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    from_env = os.environ.get(PRIVATE_PATH_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_PRIVATE_PATH


def has_local_tables(path: Path | str | None = None) -> bool:
    target = resolve_private_path(path)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(stem in payload for stem in SCRIPT_STEMS)


def load_case_tables(
    script_name: str,
    *,
    path: Path | str | None = None,
) -> tuple[list[list[str]], list[str]]:
    """返回 ``(POSITIVE, NEGATIVE)``；表缺失或格式不符时抛异常。"""

    target = resolve_private_path(path)
    if not target.is_file():
        raise LocalEvalCasesMissing(
            f"本地评测用例表不存在：{target}；"
            "这些表含真实提问与文档名（PRD §7.2 不入仓库），"
            f"可用环境变量 {PRIVATE_PATH_ENV} 指向别处；"
            "本机请从备份恢复该文件后重跑"
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or script_name not in payload:
        raise LocalEvalCasesMissing(
            f"本地评测用例表缺少 {script_name!r} 段：{target}"
        )
    entry = payload[script_name]
    if not isinstance(entry, dict):
        raise TypeError(f"本地评测用例表的 {script_name!r} 段必须是 JSON object")
    positive = entry.get("positive")
    negative = entry.get("negative")
    if not isinstance(positive, list) or not positive:
        raise ValueError(f"本地评测用例表的 {script_name!r}.positive 必须是非空数组")
    if not isinstance(negative, list):
        raise TypeError(f"本地评测用例表的 {script_name!r}.negative 必须是数组")
    tables: list[list[str]] = []
    for item in positive:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{script_name!r}.positive 的每条必须是 [问题, 期望子串]")
        tables.append([str(item[0]), str(item[1])])
    return tables, [str(item) for item in negative]


def summary(path: Path | str | None = None) -> dict[str, Any]:
    target = resolve_private_path(path)
    payload = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    return {
        "path": str(target),
        "available": has_local_tables(path),
        "counts": {
            str(stem): {
                "positive": len(entry.get("positive") or []),
                "negative": len(entry.get("negative") or []),
            }
            for stem, entry in sorted(payload.items())
            if isinstance(entry, dict)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一次性评测脚本的本地用例表（PRD §7.2）")
    parser.add_argument("--path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = summary(args.path)
    except json.JSONDecodeError as exc:
        print(f"[error] 本地用例表不是合法 JSON：{exc}")
        return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"path={data['path']} available={data['available']}")
        for stem, counts in data["counts"].items():
            print(f"  {stem}: positive={counts['positive']} negative={counts['negative']}")
    return 0 if data["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
