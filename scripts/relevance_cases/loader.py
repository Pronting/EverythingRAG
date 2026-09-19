"""分级相关性数据集的本地加载器（PRD RAG-REL-001 §7.2）。

仓库内 ``cases-C0X.json`` 只是**脱敏结构**：问题原文、来源 basename、标题、证据片段
正文一律不入库，只保留非私密字段与私文内容的 SHA-256。完整数据集由本模块在**本机**
把「结构 + 本地私文」按 ``case_id`` / ``target_id`` 组装出来：

```
结构：scripts/relevance_cases/cases-C01.json          （随仓库交付）
私文：.tmp_probe/rag-relevance/relevance-private/private-C01.json （.gitignore 覆盖）
```

组装后逐字段校验：query 哈希、每题的 ``case_sha256``（覆盖全部私文字段）、来源 basename、
必要证据等级、两级分组规则。私文缺失时（CI / 公开环境）：

- ``load_cases(..., allow_redacted=True)`` 仍可读取结构（供一致性用例使用）；
- ``load_cases`` 默认直接抛 ``PrivateDatasetMissing``，绝不静默返回半成品。

命令行（只打印统计与路径，不打印私人正文）：::

    python scripts/relevance_cases/loader.py --cases-dir scripts/relevance_cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

#: 仓库根（scripts/relevance_cases/loader.py → 上两级）。
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

if TYPE_CHECKING:  # 仅类型标注；运行时按需导入，避免把 ``app`` 绑到调用方的 sys.path 解析上
    from app.evaluation.rag_quality import RelevanceCase


def _quality_module():
    """按需导入评测核心模块：优先用**本仓库**的 backend（不受调用方 sys.path 顺序影响）。

    调用方（例如固定旧代码快照的评测脚本）可能已经在 ``sys.modules`` 里放了**别的**
    ``app.evaluation.rag_quality``：那种情况下按文件路径直接加载本仓库的实现，保证
    「结构 + 私文」的组装口径永远来自当前仓库，而不是调用方的搜索路径。
    """

    if str(BACKEND_ROOT) not in sys.path:
        sys.path.append(str(BACKEND_ROOT))
    module = sys.modules.get("app.evaluation.rag_quality")
    if module is None or not hasattr(module, "load_relevance_cases"):
        import importlib.util

        target = BACKEND_ROOT / "app" / "evaluation" / "rag_quality.py"
        spec = importlib.util.spec_from_file_location(
            "everything_rag_active_rag_quality", target
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # dataclass 需要能在 sys.modules 里找回本模块（__module__ 解析）。
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.load_relevance_cases, module.validate_relevance_cases

#: 私文（问题原文/来源名/标题/证据片段）的默认本地目录；被 ``.gitignore`` 的 ``.tmp_probe/`` 覆盖。
DEFAULT_PRIVATE_DIR = REPO_ROOT / ".tmp_probe" / "rag-relevance" / "relevance-private"
DEFAULT_CASES_DIR = REPO_ROOT / "scripts" / "relevance_cases"
PRIVATE_DIR_ENV = "EVERYTHING_RAG_RELEVANCE_PRIVATE_DIR"
CASES_DIR_ENV = "EVERYTHING_RAG_RELEVANCE_CASES_DIR"

#: 私文文件里允许出现的字段（用于格式校验）。
_PRIVATE_CASE_FIELDS = {"case_id", "query", "note", "targets"}
_PRIVATE_TARGET_FIELDS = {
    "target_id",
    "source",
    "heading",
    "required_evidence",
    "irrelevant_spans",
    "relevant_spans",
    "note",
}


class PrivateDatasetMissing(FileNotFoundError):
    """本地私文数据集不存在（CI / 公开环境属预期，测试应据此 skip）。"""


def resolve_cases_dir(cases_dir: Path | str | None = None) -> Path:
    if cases_dir is not None:
        return Path(cases_dir).expanduser()
    from_env = os.environ.get(CASES_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_CASES_DIR


def resolve_private_dir(private_dir: Path | str | None = None) -> Path:
    if private_dir is not None:
        return Path(private_dir).expanduser()
    from_env = os.environ.get(PRIVATE_DIR_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_PRIVATE_DIR


def private_path(family: str, *, private_dir: Path | str | None = None) -> Path:
    return resolve_private_dir(private_dir) / f"private-{family}.json"


def case_files(cases_dir: Path | str | None = None) -> tuple[Path, ...]:
    """仓库内的脱敏结构文件（按 family 名排序）。"""

    return tuple(sorted(resolve_cases_dir(cases_dir).glob("cases-C*.json")))


def family_of(path: Path) -> str:
    return Path(path).stem.removeprefix("cases-")


def has_private_dataset(*, private_dir: Path | str | None = None) -> bool:
    """六类私文是否齐备。"""

    directory = resolve_private_dir(private_dir)
    return all(private_path(family, private_dir=directory).is_file() for family in _families())


def _families() -> tuple[str, ...]:
    return tuple(family_of(path) for path in case_files()) or (
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
    )


def load_private_family(
    family: str,
    *,
    private_dir: Path | str | None = None,
) -> dict[str, Any]:
    path = private_path(family, private_dir=private_dir)
    if not path.is_file():
        raise PrivateDatasetMissing(
            f"本地私文数据集不存在：{path}；"
            "公开/CI 环境属预期（相关用例应 skip），本机请从备份恢复该目录"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"本地私文数据集顶层必须是 JSON object：{path}")
    if payload.get("family") != family:
        raise ValueError(f"本地私文 family 与文件名不一致：{path}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"本地私文 cases 必须是非空数组：{path}")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError(f"本地私文案例必须是 JSON object：{path}")
        unknown = set(case) - _PRIVATE_CASE_FIELDS
        if unknown:
            raise ValueError(f"本地私文案例含未知字段：{sorted(unknown)!r}")
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"本地私文案例 query 不能为空：{case.get('case_id')!r}")
        targets = case.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"本地私文案例 targets 必须是非空数组：{case.get('case_id')!r}")
        for target in targets:
            if not isinstance(target, dict):
                raise TypeError(f"本地私文目标必须是 JSON object：{case.get('case_id')!r}")
            target_unknown = set(target) - _PRIVATE_TARGET_FIELDS
            if target_unknown:
                raise ValueError(f"本地私文目标含未知字段：{sorted(target_unknown)!r}")
            if not isinstance(target.get("source"), str) or not target["source"].strip():
                raise ValueError(
                    f"本地私文目标 source 不能为空：{target.get('target_id')!r}"
                )
    return payload


def load_private_dataset(
    *,
    private_dir: Path | str | None = None,
    families: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """读取全部 family 私文；缺任何一个都抛 ``PrivateDatasetMissing``。"""

    names = tuple(families) if families is not None else _families()
    return {family: load_private_family(family, private_dir=private_dir) for family in names}


def load_cases(
    path: Path | str,
    *,
    private_dir: Path | str | None = None,
    expected_corpus_sha256: str | None = None,
    allow_redacted: bool = False,
) -> tuple[RelevanceCase, ...]:
    """组装单个 family：结构文件 + 本地私文。

    ``allow_redacted=True`` 且私文缺失时返回未组装的脱敏结构（只含非私密字段），
    供「仓库契约」类用例使用；默认（``False``）在私文缺失时抛 ``PrivateDatasetMissing``。
    """

    path = Path(path)
    family = family_of(path)
    try:
        private = load_private_family(family, private_dir=private_dir)
    except PrivateDatasetMissing:
        if not allow_redacted:
            raise
        private = None
    load_relevance_cases, _validate = _quality_module()
    return load_relevance_cases(
        path,
        expected_corpus_sha256=expected_corpus_sha256,
        private_dataset=private,
        allow_redacted=allow_redacted,
    )


def load_dataset(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
    expected_corpus_sha256: str | None = None,
    allow_redacted: bool = False,
) -> tuple[RelevanceCase, ...]:
    """组装六类完整数据集（120 题 / 729 目标），并跨 family 校验唯一性。"""

    merged: list[RelevanceCase] = []
    for path in case_files(cases_dir):
        merged.extend(
            load_cases(
                path,
                private_dir=private_dir,
                expected_corpus_sha256=expected_corpus_sha256,
                allow_redacted=allow_redacted,
            )
        )
    ids = [case.case_id for case in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("跨 family 合并后 case_id 必须唯一")
    _, validate_relevance_cases = _quality_module()
    validate_relevance_cases(tuple(merged), allow_redacted=allow_redacted)
    return tuple(merged)


def load_all_queries(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
) -> tuple[str, ...]:
    """全部问题原文（本地评测用；只在内存中，不落仓库）。"""

    return tuple(case.query or "" for case in load_dataset(cases_dir=cases_dir, private_dir=private_dir))


def assemble_family_payload(
    path: Path | str,
    *,
    private_dir: Path | str | None = None,
) -> dict[str, Any]:
    """把「脱敏结构 + 本地私文」组装回一份完整 family payload（逐字校验工具用）。

    私文缺失时抛 ``PrivateDatasetMissing``；哈希不一致时抛 ``ValueError``。
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    family = payload["family"]
    private = load_private_family(family, private_dir=private_dir)
    private_cases = {case["case_id"]: case for case in private["cases"]}
    verify_structure_hashes(payload["cases"], private_cases)
    assembled_cases: list[dict[str, Any]] = []
    for raw in payload["cases"]:
        local = private_cases[raw["case_id"]]
        local_targets = {target["target_id"]: target for target in local["targets"]}
        assembled_cases.append(
            {
                **{
                    key: value
                    for key, value in raw.items()
                    if key not in {"targets", "query_sha256"}
                },
                "query": local["query"],
                "targets": [
                    {**target, **local_targets[target["target_id"]]} for target in raw["targets"]
                ],
            }
        )
    return {**payload, "cases": assembled_cases}


def private_string_index(
    *,
    private_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """收集本地私文里的全部私密字符串（防回归自检用；只在内存中）。"""

    index: dict[str, list[str]] = {
        "sources": [],
        "headings": [],
        "queries": [],
        "evidence": [],
        "span_texts": [],
        "notes": [],
    }
    for payload in load_private_dataset(private_dir=private_dir).values():
        for case in payload["cases"]:
            index["queries"].append(case["query"])
            if case.get("note"):
                index["notes"].append(case["note"])
            for target in case["targets"]:
                index["sources"].append(target["source"])
                if (target.get("heading") or "").strip():
                    index["headings"].append(target["heading"])
                index["evidence"].extend(target.get("required_evidence") or [])
                if target.get("note"):
                    index["notes"].append(target["note"])
                for span in list(target.get("irrelevant_spans") or []) + list(
                    target.get("relevant_spans") or []
                ):
                    if span.get("text"):
                        index["span_texts"].append(span["text"])
    return index


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def case_digest(case: Mapping[str, Any]) -> str:
    """私文内容的规范哈希：必须与结构文件里的 ``case_sha256`` 完全一致。

    覆盖 ``case_id/group_id/split/query/needs_answer/scope_id/review_status`` 与每个目标的
    来源、标题、证据原文、范围、note；目标按 ``target_id`` 排序，因此与文件内顺序无关。
    既接受仓库结构里「结构字段 + 私文字段」合并后的整题，也接受本地私文单文件里的案例
    （缺失的结构字段按结构文件默认值补齐）。
    """

    hash_fields = (
        "case_id",
        "group_id",
        "split",
        "query",
        "needs_answer",
        "scope_id",
        "review_status",
    )
    defaults: dict[str, Any] = {
        "group_id": "",
        "split": "",
        "query": "",
        "needs_answer": True,
        "scope_id": "global",
        "review_status": "reviewed",
    }
    payload: dict[str, Any] = {
        field: case[field] if field in case else defaults.get(field) for field in hash_fields
    }
    payload["targets"] = [
        {
            "target_id": target.get("target_id"),
            "source": target.get("source") or "",
            "heading": target.get("heading") or "",
            "required_evidence": list(target.get("required_evidence") or []),
            "irrelevant_spans": [
                {
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "sha256": span.get("sha256"),
                    "text": span.get("text"),
                }
                for span in target.get("irrelevant_spans") or []
            ],
            "relevant_spans": [
                {
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "sha256": span.get("sha256"),
                    "text": span.get("text"),
                }
                for span in target.get("relevant_spans") or []
            ],
            "note": target.get("note") or "",
        }
        for target in sorted(case.get("targets") or [], key=lambda item: str(item.get("target_id")))
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(blob)


#: ``case_sha256`` 覆盖的目标级私文字段（字段名参与哈希，改动等于更换哈希口径）。
DIGEST_TARGET_FIELDS = (
    "source",
    "heading",
    "required_evidence",
    "irrelevant_spans",
    "relevant_spans",
    "note",
)


def merge_for_digest(
    structure_case: Mapping[str, Any],
    private_case: Mapping[str, Any],
) -> dict[str, Any]:
    """结构案例 + 私文案例 → 可用于 ``case_digest`` 的整题（结构字段取结构文件）。"""

    local_targets = {str(target.get("target_id")): target for target in private_case.get("targets") or []}
    digest_targets: list[dict[str, Any]] = []
    for structure_target in structure_case.get("targets") or []:
        private_target = local_targets.get(str(structure_target.get("target_id")))
        if private_target is None:
            raise ValueError(
                f"本地私文缺少目标：{structure_case.get('case_id')!r}/"
                f"{structure_target.get('target_id')!r}"
            )
        digest_targets.append(
            {
                "target_id": structure_target.get("target_id"),
                **{field: private_target.get(field) for field in DIGEST_TARGET_FIELDS},
            }
        )
    return {
        **{key: value for key, value in structure_case.items() if key != "targets"},
        "query": private_case.get("query"),
        "targets": digest_targets,
    }


def verify_structure_hashes(
    structure_cases: Sequence[Mapping[str, Any]],
    private_cases: Mapping[str, Mapping[str, Any]],
) -> None:
    """逐题校验结构文件里的 ``query_sha256`` / ``case_sha256`` 与本地私文一致。"""

    for case in structure_cases:
        case_id = str(case.get("case_id"))
        local = private_cases.get(case_id)
        if local is None:
            raise ValueError(f"本地私文缺少案例：{case_id!r}")
        query = str(local.get("query") or "")
        if case.get("query_sha256") and _sha256_text(query) != case["query_sha256"]:
            raise ValueError(f"本地私文与结构文件不匹配（query 哈希不同）：{case_id!r}")
        if not case.get("case_sha256"):
            continue
        if case_digest(merge_for_digest(case, local)) != case["case_sha256"]:
            raise ValueError(f"本地私文与结构文件不匹配（case_sha256 不同）：{case_id!r}")


def structure_summary(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
) -> dict[str, Any]:
    """只读结构 + 统计摘要（不返回任何私人正文），供 CLI / 一致性断言使用。"""

    directory = resolve_cases_dir(cases_dir)
    families: list[dict[str, Any]] = []
    totals = {"cases": 0, "targets": 0, "calibration": 0, "test": 0}
    hashes: set[str] = set()
    for path in case_files(directory):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = dict(payload.get("counts") or {})
        cases = payload["cases"]
        family_targets = totals_targets(cases)
        totals["cases"] += len(cases)
        totals["targets"] += family_targets
        totals["calibration"] += sum(1 for case in cases if case["split"] == "calibration")
        totals["test"] += sum(1 for case in cases if case["split"] == "test")
        hashes.add(payload["corpus_sha256"])
        families.append(
            {
                "family": payload["family"],
                "file": path.name,
                "redacted": bool(payload.get("redaction")),
                "cases": len(cases),
                "targets": family_targets,
                "counts": counts,
            }
        )
    return {
        "cases_dir": str(directory),
        "private_dir": str(resolve_private_dir(private_dir)),
        "private_available": has_private_dataset(private_dir=private_dir),
        "corpus_sha256": min(hashes) if len(hashes) == 1 else None,
        "totals": totals,
        "families": families,
    }


def totals_targets(cases: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(case.get("targets") or []) for case in cases)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分级相关性数据集本地加载器（PRD §7.2）")
    parser.add_argument("--cases-dir", default=None, help="仓库内脱敏结构目录")
    parser.add_argument("--private-dir", default=None, help="本地私文目录（gitignored）")
    parser.add_argument("--assemble", action="store_true", help="实际组装六类数据集并报告统计")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    args = parser.parse_args(argv)

    summary = structure_summary(cases_dir=args.cases_dir, private_dir=args.private_dir)
    if args.assemble:
        try:
            cases = load_dataset(cases_dir=args.cases_dir, private_dir=args.private_dir)
        except (PrivateDatasetMissing, ValueError) as exc:
            print(f"[assemble] 失败：{exc}")
            return 1
        summary["assembled"] = {
            "cases": len(cases),
            "targets": sum(len(case.targets) for case in cases),
            "required_targets": sum(len(case.required_targets) for case in cases),
            "queries_loaded": sum(1 for case in cases if case.query),
            "redacted_remaining": sum(1 for case in cases if case.is_redacted),
        }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"cases_dir={summary['cases_dir']}\n"
            f"private_dir={summary['private_dir']} available={summary['private_available']}\n"
            f"totals={summary['totals']}"
        )
        for entry in summary["families"]:
            print(
                f"  {entry['family']}: cases={entry['cases']} targets={entry['targets']} "
                f"redacted={entry['redacted']}"
            )
        if "assembled" in summary:
            print(f"assembled={summary['assembled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
