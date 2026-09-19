"""benchmark 用例集的本地加载器（PRD §7.2 脱敏）。

`benchmark/cases.json` 在仓库里只是**脱敏结构**：`id / cat / diff / gold`（gold 是块序号，
不含来源名）+ 私文内容的 SHA-256 + 计数。真实问题原文与证据串保存在 Git 忽略的本地文件：

    .tmp_probe/rag-relevance/benchmark-private/cases-private.json

消费者（`run_answer_bench.py`、`validate_cases.py`）必须经本模块取完整用例；
本地私文缺失时抛 ``PrivateCasesMissing``，脚本会给出明确提示而不是静默跑空集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = REPO_ROOT / "benchmark" / "cases.json"
DEFAULT_PRIVATE_PATH = (
    REPO_ROOT / ".tmp_probe" / "rag-relevance" / "benchmark-private" / "cases-private.json"
)
PRIVATE_PATH_ENV = "EVERYTHING_RAG_BENCH_PRIVATE_CASES"

_HASH_FIELDS = ("id", "cat", "diff", "gold", "query", "evidence")


class PrivateCasesMissing(FileNotFoundError):
    """本地私文用例缺失（Git 忽略，只在本机保存）。"""


BENCH_PRIVATE_DIR_ENV = "EVERYTHING_RAG_BENCH_PRIVATE_DIR"


def private_dir(path: Path | str | None = None) -> Path:
    """benchmark 私文目录（块样本、结果产物都放这里，`benchmark/*.json` 已被 ignore）。"""

    if path is not None:
        return Path(path).expanduser()
    from_env = os.environ.get(BENCH_PRIVATE_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return REPO_ROOT / ".tmp_probe" / "rag-relevance" / "benchmark-private"


def sample_chunks_path(path: Path | str | None = None) -> Path:
    """块样本路径：优先 gitignored 私文目录，其次 benchmark/（本机重跑产物，同样被 ignore）。"""

    if path is not None:
        return Path(path)
    candidate = private_dir() / "sample_chunks.json"
    if candidate.is_file():
        return candidate
    generated = REPO_ROOT / "benchmark" / "sample_chunks.json"
    if generated.is_file():
        return generated
    return candidate


def artifact_path(name: str, *, fallback_dir: Path | str | None = None) -> Path:
    """数据产物路径：优先 gitignored 私文目录，其次 benchmark/（本机重跑产物，同样被 ignore）。

    这些产物含真实分块正文/来源名，仓库只保留生成脚本与聚合结果（PRD §7.2 / T22）。
    """

    candidate = private_dir() / name
    if candidate.is_file():
        return candidate
    generated = (Path(fallback_dir) if fallback_dir is not None else REPO_ROOT / "benchmark") / name
    if generated.is_file():
        return generated
    return candidate


def load_sample_chunks(path: Path | str | None = None) -> list[dict[str, Any]]:
    target = sample_chunks_path(path)
    if not target.is_file():
        raise PrivateCasesMissing(
            f"块样本不存在：{target}；它含真实分块正文与来源名（PRD §7.2 不入仓库），"
            f"可用环境变量 {BENCH_PRIVATE_DIR_ENV} 指向本地私文目录，"
            "或用 benchmark/run_retrieval_bench.py 在本机重新导出"
        )
    return json.loads(target.read_text(encoding="utf-8"))


def resolve(private_path: Path | str | None = None) -> Path:
    if private_path is not None:
        return Path(private_path).expanduser()
    from_env = os.environ.get(PRIVATE_PATH_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_PRIVATE_PATH


def has_private_cases(private_path: Path | str | None = None) -> bool:
    return resolve(private_path).is_file()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def case_digest(case: dict[str, Any]) -> str:
    """与仓库结构里的 ``case_sha256`` 同一口径（字段固定 + 规范 JSON）。"""

    payload = {field: case.get(field) for field in _HASH_FIELDS}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(blob)


def load_document(
    cases_path: Path | str | None = None,
    *,
    private_path: Path | str | None = None,
) -> dict[str, Any]:
    """返回完整 cases payload（结构与私文组装后），并逐题校验哈希。"""

    path = Path(cases_path) if cases_path is not None else DEFAULT_CASES_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("redaction"):
        # 兼容未脱敏的旧副本（例如主仓尚未合并本次改动时的 benchmark/cases.json）：
        # 那时 query/evidence 已内联，直接用。
        return payload
    private_file = resolve(private_path)
    if not private_file.is_file():
        raise PrivateCasesMissing(
            f"benchmark 用例的本地私文不存在：{private_file}；"
            "问题原文与证据串属私人正文（PRD §7.2 不入仓库），"
            f"可用环境变量 {PRIVATE_PATH_ENV} 指向别处；本机请从备份恢复后重跑"
        )
    private = json.loads(private_file.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in private.get("cases", [])}
    assembled = []
    for case in payload["cases"]:
        local = by_id.get(case["id"])
        if local is None:
            raise ValueError(f"本地私文缺少用例：{case['id']!r}")
        if case.get("query_sha256") and _sha256(local["query"]) != case["query_sha256"]:
            raise ValueError(f"本地私文与仓库结构不匹配（query 哈希不同）：{case['id']!r}")
        merged = {
            "id": case["id"],
            "cat": case["cat"],
            "diff": case["diff"],
            "gold": list(case["gold"]),
            "query": local["query"],
            "evidence": list(local["evidence"]),
        }
        if case.get("case_sha256") and case_digest(merged) != case["case_sha256"]:
            raise ValueError(f"本地私文与仓库结构不匹配（case_sha256 不同）：{case['id']!r}")
        assembled.append(merged)
    extra = sorted(set(by_id) - {case["id"] for case in payload["cases"]})
    if extra:
        raise ValueError(f"本地私文含仓库结构里没有的用例：{extra!r}")
    return {**payload, "cases": assembled}


def load_cases(
    cases_path: Path | str | None = None,
    *,
    private_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    return list(load_document(cases_path, private_path=private_path)["cases"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="benchmark 用例结构 + 本地私文组装")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--private", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = load_document(args.cases, private_path=args.private)
    except PrivateCasesMissing as exc:
        print(f"[skip] {exc}")
        return 2
    summary = {
        "cases": len(document["cases"]),
        "counts": document.get("counts", {}),
        "private": str(resolve(args.private)),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"cases={summary['cases']} counts={summary['counts']}")
        print(f"private={summary['private']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
