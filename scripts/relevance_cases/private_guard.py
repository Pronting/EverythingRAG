"""防回归自检：本地私文里的任何私人正文都不得出现在仓库文件里（PRD §7.2）。

两种扫描模式：

- **数据集目录模式**（``scan`` / ``--dataset-dir``）：只扫 ``scripts/relevance_cases/``，
  needle 全部来自本地私文数据集，阈值最严（去重后每个字符串全文搜索）。T21 的验收用例。
- **仓库模式**（``scan_repo`` / 默认 CLI）：扫整仓文本文件（跳过 ``.git``/``.tmp_probe``/
  ``.venv`` 等），needle 来自这些**本地私文**来源：
  1. RAG-REL 分级数据集私文（问题/来源名/标题/证据/note）；
  2. 语料清单 ``corpus-inventory.json``（全部来源 basename + 标题 + 正文预览）；
  3. ``benchmark/`` 私文（用例问题与证据串，``benchmark-private/cases-private.json``）；
  4. 三个一次性评测脚本的本地用例表（``local-eval-cases.json``）；
  5. 可选 ``sensitive-needles.json``：人名、可识别时间戳等点名敏感串。

  阈值按类型区分（来源 basename/标题是精确标识，用短阈值；通用中文短语用长阈值），
  并跳过「仓库自身同名文件」这类歧义 needle（如 ``CLAUDE.md``/``README.md``）。

**发布阻断项**：``scripts/rag_eval_cases.json``（检索黄金集）按用户决定留到校准批次处理，
它仍含真实来源名与标题。仓库模式把它单列进 ``release_blockers``，**不并入 leaks 静默放过**：
只要它还在，CLI 退出码为 2 并打印 ``[RELEASE BLOCKER]``；``leaks`` 只统计未被豁免的命中。

用法::

    python scripts/relevance_cases/private_guard.py                 # 整仓模式
    python scripts/relevance_cases/private_guard.py --dataset-dir   # 只扫数据集目录（最严）
    python scripts/relevance_cases/private_guard.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# 优先按包路径复用调用方已加载的 loader（否则 ``loader`` 与 ``relevance_cases.loader``
# 会被加载成两个模块对象，``PrivateDatasetMissing`` 变成两个不同的类，
# 调用方的 ``except`` 抓不住）；作为脚本直接运行时回退到顶层导入。
try:  # pragma: no cover - 取决于导入方式
    from relevance_cases.loader import (
        REPO_ROOT,
        PrivateDatasetMissing,
        private_string_index,
        resolve_cases_dir,
        resolve_private_dir,
    )
except ImportError:  # pragma: no cover - 脚本直跑路径
    from loader import (  # type: ignore[no-redef]
        REPO_ROOT,
        PrivateDatasetMissing,
        private_string_index,
        resolve_cases_dir,
        resolve_private_dir,
    )

#: 数据集目录模式忽略的极短字符串（避免把空格、单字母当成泄漏证据）。
MIN_NEEDLE_CHARS = 2

#: 仓库模式：按类型区分最小长度（来源/标题是精确标识；通用短语容易与无关文案撞词）。
#:
#: **阈值口径与理由**（每个 kind 为什么是这个数、低于它怎么处理）：
#: - ``sources``/``corpus_sources`` = 4：来源 basename 自带 ``.md`` 后缀，≥4 字符即精确标识，
#:   与无关文案几乎不可能撞词；低于 4 的（如 ``a.md``）是测试夹具里的通用短名，属假阳性。
#: - ``source_stems``/``corpus_stems`` = 5：去掉扩展名后 4 字的中文名（如「公司介绍」）是通用短语，
#:   会与 PRD/文档正文撞词，因此 bare stem 只在 ≥5 字时才算标识；``.md`` 全名仍按 4 字算。
#: - ``headings`` = 12 / ``corpus_headings`` = 16：章节标题多是「序号 + 小标题」这类通用短语，
#:   短标题在中文文档里撞词概率很高；语料标题取 16 字进一步保守（只保留真正独特的标题）。
#: - ``queries``/``benchmark_queries``/``local_eval_questions`` = 12：真实提问是整句，
#:   ≥12 字即足以作为身份指纹；更短的多为「什么是 adb」这类通用问法。
#: - ``evidence``/``benchmark_evidence``/``span_texts`` = 16：证据片段是**逐字原文**，
#:   短片段极易与通用叙述撞词（本次 T23-F4 的 12 字片段就落在这个缝里）；
#:   因此短于 16 字的已知私文片段**不得靠阈值蒙混**——必须显式登记进
#:   ``sensitive-needles.json`` 的 ``short_fragments``（见下），该 kind 无阈值豁免。
#: - ``notes`` = 24：标注依据是自然语言，取 24 字避免与报告正文撞词。
#: - ``corpus_previews`` = 24：语料清单里的正文预览，取 24 字。
#: - ``local_eval_anchors`` = 5：一次性评测脚本的「期望命中子串」（文档名片段）。
#: - ``sensitive``/``declared_short_fragments`` = 2：**点名列举**的串（人名、可识别时间戳、
#:   已知短私文片段），不做通用短语过滤——它们是显式声明，命中即真实泄漏。
MIN_CHARS_BY_KIND = {
    "sources": 4,
    "source_stems": 5,
    "corpus_sources": 4,
    "corpus_stems": 5,
    "headings": 12,
    "corpus_headings": 16,
    "corpus_previews": 24,
    "queries": 12,
    "benchmark_queries": 12,
    "evidence": 16,
    "benchmark_evidence": 16,
    "local_eval_questions": 12,
    "local_eval_anchors": 5,
    "span_texts": 16,
    "notes": 24,
    "sensitive": 2,
    "declared_short_fragments": 2,
}

#: 各 kind 的判定理由（随报告一起输出，避免「阈值挡住了」成为不透明解释）。
THRESHOLD_RATIONALE = {
    "sources": "来源 basename（含 .md）是精确标识，≥4 字符即不可与通用文案混淆",
    "source_stems": "去掉扩展名的中文名 4 字常是通用短语（如「公司介绍」），故 ≥5 字才算标识",
    "corpus_sources": "同 sources（语料清单里的来源 basename）",
    "corpus_stems": "同 source_stems",
    "headings": "章节标题多为通用短语，<12 字撞词概率高",
    "corpus_headings": "语料标题取 16 字更保守，只保留独特标题",
    "corpus_previews": "语料正文预览，≥24 字才具备身份唯一性",
    "queries": "真实提问是整句，≥12 字可作身份指纹",
    "benchmark_queries": "同 queries（benchmark 用例的问题原文）",
    "evidence": "逐字证据片段短片段易撞词，≥16 字；短于此的已知片段必须走 declared_short_fragments",
    "benchmark_evidence": "同 evidence",
    "span_texts": "标注范围文本，同 evidence",
    "notes": "标注依据为自然语言，≥24 字避免与报告正文撞词",
    "local_eval_questions": "一次性评测脚本的问题原文，同 queries",
    "local_eval_anchors": "期待命中子串（文档名片段），≥5 字",
    "sensitive": "点名列举的敏感串（人名/时间戳），无通用短语过滤",
    "declared_short_fragments": "显式登记的短私文片段（低于同类阈值者），无阈值豁免，命中即泄漏",
}

#: 只检查文本文件；这些后缀之外一律跳过。
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".html",
}

#: 不进入仓库交付范围的目录（虚拟环境/构建产物/本地私文目录/团队信箱）。
SKIPPED_DIRS = {
    ".git",
    ".tmp_probe",
    ".agent-teams",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".mypy_cache",
    "htmlcov",
}

#: 单文件扫描上限（跳过数据库/大样本）。
MAX_FILE_BYTES = 4 * 1024 * 1024

#: 经用户决定暂缓、**必须显式报出**的例外（不是「没问题」）。
RELEASE_BLOCKERS: dict[str, str] = {
    "scripts/rag_eval_cases.json": (
        "检索黄金集（72 正例/10 负例）仍含真实来源名与标题；"
        "按用户决定留到校准批次处理，在此之前本项构成发布阻断"
    ),
}


def _walk(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:  # pragma: no cover - 文件消失/权限
            continue
        files.append(path)
    return tuple(files)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _unique(values: Sequence[str]) -> set[str]:
    return {item for item in values if item}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# needle 来源（全部来自 Git 忽略的本地文件）
# ---------------------------------------------------------------------------


def _inventory_path(private_dir: Path) -> Path:
    """语料清单与私文目录同级：``.tmp_probe/rag-relevance/corpus-inventory.json``。"""

    return private_dir.parent / "corpus-inventory.json"


def repo_needle_pools(
    *,
    private_dir: Path | str | None = None,
) -> dict[str, set[str]]:
    """仓库模式的 needle 集合。"""

    directory = resolve_private_dir(private_dir)
    index = private_string_index(private_dir=directory)
    pools: dict[str, set[str]] = {
        "sources": _unique(index["sources"]),
        "headings": _unique(index["headings"]),
        "queries": _unique(index["queries"]),
        "evidence": _unique(index["evidence"]),
        "span_texts": _unique(index["span_texts"]),
        "notes": _unique(index["notes"]),
    }
    pools["source_stems"] = {Path(name).stem for name in pools["sources"]}

    inventory = _inventory_path(directory)
    if inventory.is_file():
        payload = _read_json(inventory)
        sources = _unique([entry.get("source") or "" for entry in payload.get("sources", [])])
        headings = _unique(
            [
                heading.get("heading") or ""
                for entry in payload.get("sources", [])
                for heading in entry.get("headings", [])
            ]
        )
        previews = _unique(
            [
                heading.get("preview") or ""
                for entry in payload.get("sources", [])
                for heading in entry.get("headings", [])
            ]
        )
        pools["corpus_sources"] = sources
        pools["corpus_stems"] = {Path(name).stem for name in sources}
        pools["corpus_headings"] = headings
        pools["corpus_previews"] = previews

    benchmark_private = directory.parent / "benchmark-private" / "cases-private.json"
    if benchmark_private.is_file():
        payload = _read_json(benchmark_private)
        pools["benchmark_queries"] = _unique(
            [case.get("query") or "" for case in payload.get("cases", [])]
        )
        pools["benchmark_evidence"] = _unique(
            [item for case in payload.get("cases", []) for item in case.get("evidence") or []]
        )

    local_cases = directory.parent / "local-eval-cases.json"
    if local_cases.is_file():
        payload = _read_json(local_cases)
        questions: list[str] = []
        anchors: list[str] = []
        for entry in payload.values():
            if not isinstance(entry, Mapping):
                continue
            for item in entry.get("positive") or []:
                if isinstance(item, list) and len(item) == 2:
                    questions.append(str(item[0]))
                    anchors.append(str(item[1]))
            for item in entry.get("negative") or []:
                questions.append(str(item))
        pools["local_eval_questions"] = _unique(questions)
        pools["local_eval_anchors"] = _unique(anchors)

    sensitive = directory / "sensitive-needles.json"
    if sensitive.is_file():
        payload = _read_json(sensitive)
        items = payload.get("needles") if isinstance(payload, Mapping) else payload
        if isinstance(items, list):
            pools["sensitive"] = _unique([str(item) for item in items])
        # 显式登记的「低于同类阈值」的短私文片段（T23-F4）：无阈值豁免，命中即泄漏。
        short = payload.get("short_fragments") if isinstance(payload, Mapping) else None
        if isinstance(short, list):
            texts: list[str] = []
            for item in short:
                if isinstance(item, Mapping):
                    texts.append(str(item.get("text") or ""))
                else:
                    texts.append(str(item))
            pools["declared_short_fragments"] = _unique(texts)

    return {kind: value for kind, value in pools.items() if value}


def declared_short_fragments(
    *,
    private_dir: Path | str | None = None,
) -> tuple[str, ...]:
    """本地登记的短私文片段（含来历说明），供测试与 CLI 直接读取。

    本地 needle 文件整体缺失时抛 ``PrivateDatasetMissing``（调用方据此 skip，
    避免把「本机没有私文」误读成「没有短片段要查」）；文件在但未登记短片段时返回空元组。
    """

    directory = resolve_private_dir(private_dir)
    path = directory / "sensitive-needles.json"
    if not path.is_file():
        raise PrivateDatasetMissing(
            f"本地敏感串文件不存在：{path}；公开/CI 环境属预期（相关用例应 skip），"
            "本机请从备份恢复该文件"
        )
    payload = _read_json(path)
    short = payload.get("short_fragments") if isinstance(payload, Mapping) else None
    if not isinstance(short, list):
        return ()
    texts: list[str] = []
    for item in short:
        if isinstance(item, Mapping):
            texts.append(str(item.get("text") or ""))
        else:
            texts.append(str(item))
    return tuple(item for item in texts if item)


def thresholds_by_kind(min_chars: int | None = None) -> dict[str, dict[str, Any]]:
    """把「kind → 阈值 + 判定理由」做成可机读的说明（随报告输出）。

    ``min_chars`` 非空表示数据集目录模式：所有 kind 统一用最严阈值（2 字符），
    不做通用短语过滤——该模式只扫 ``scripts/relevance_cases/``，命中的都当泄漏。
    """

    if min_chars is not None:
        return {
            kind: {
                "min_chars": min_chars,
                "mode": "dataset_dir",
                "rationale": (
                    "数据集目录模式：统一最严阈值（不做通用短语过滤，命中即泄漏）；"
                    f"仓库模式的分类阈值为 {threshold}"
                ),
            }
            for kind, threshold in sorted(MIN_CHARS_BY_KIND.items())
        }
    return {
        kind: {
            "min_chars": threshold,
            "mode": "repo",
            "rationale": THRESHOLD_RATIONALE.get(kind, ""),
        }
        for kind, threshold in sorted(MIN_CHARS_BY_KIND.items())
    }


def _repo_basenames(root: Path) -> set[str]:
    """仓库自身的文件名（含去扩展名的 stem）。

    语料里也有叫 ``CLAUDE.md``/``README.md`` 的文档，这类 needle 与仓库自身文件同名，
    无法区分「私文泄漏」与「通用文件名」，因此不参与断言（在报告里单列）。
    """

    names: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        names.add(path.name)
        if path.suffix:
            names.add(path.stem)
    return names


def _select_needles(
    pools: Mapping[str, set[str]],
    *,
    min_chars: int | None = None,
    ambiguous: set[str] | None = None,
) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for kind, values in pools.items():
        threshold = min_chars if min_chars is not None else MIN_CHARS_BY_KIND.get(kind, 16)
        kept = {
            item
            for item in values
            if item
            and len(item.strip()) >= threshold
            and not (ambiguous and item in ambiguous)
        }
        if kept:
            selected[kind] = kept
    return selected


def _collect_files(roots: Sequence[Path]) -> tuple[list[Path], dict[Path, str]]:
    seen: set[Path] = set()
    files: list[Path] = []
    for base in roots:
        for path in _walk(base):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files, {path: path.read_text(encoding="utf-8", errors="replace") for path in files}


def _find_leaks(
    selected: Mapping[str, set[str]],
    haystacks: Mapping[Path, str],
) -> list[dict[str, str]]:
    leaks: list[dict[str, str]] = []
    for kind, needles in sorted(selected.items()):
        for needle in sorted(needles):
            for path, haystack in haystacks.items():
                if needle in haystack:
                    leaks.append(
                        {
                            "kind": kind,
                            "file": str(path),
                            "needle_sha256": _sha256_text(needle),
                            "needle_length": str(len(needle)),
                        }
                    )
    return leaks


# ---------------------------------------------------------------------------
# 两种模式
# ---------------------------------------------------------------------------


def scan(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
    scan_roots: Sequence[Path | str] | None = None,
    min_chars: int = MIN_NEEDLE_CHARS,
    label: str = "dataset_dir",
) -> dict[str, Any]:
    """数据集目录模式：needle 全部来自本地私文数据集，阈值最严。"""

    directory = resolve_cases_dir(cases_dir)
    index = private_string_index(private_dir=private_dir)
    roots = [Path(item) for item in scan_roots] if scan_roots is not None else [directory]
    selected = _select_needles(index, min_chars=min_chars)
    files, haystacks = _collect_files(roots)

    leaks = _find_leaks(selected, haystacks)
    return {
        "label": label,
        "scan_roots": [str(item) for item in roots],
        "cases_dir": str(directory),
        "private_dir": str(resolve_private_dir(private_dir)),
        "min_needle_chars": min_chars,
        "files_scanned": len(files),
        "needles_checked": sum(len(values) for values in selected.values()),
        "needle_counts": {kind: len(values) for kind, values in sorted(selected.items())},
        "thresholds": thresholds_by_kind(min_chars),
        "ambiguous_needles_skipped": [],
        "leaks": leaks,
        "unexpected_leaks": leaks,
        "blockers": [],
        "release_blockers": [],
        "release_blocked": False,
        "ok": not leaks,
    }


def scan_repo(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
    scan_roots: Sequence[Path | str] | None = None,
    min_chars: int | None = None,
    label: str = "repo",
) -> dict[str, Any]:
    """仓库模式：needle 来自全部本地私文来源，按类型阈值，豁免声明过的发布阻断项。"""

    directory = resolve_cases_dir(cases_dir)
    pools = repo_needle_pools(private_dir=private_dir)
    roots = [Path(item) for item in scan_roots] if scan_roots is not None else [REPO_ROOT]
    ambiguous = _repo_basenames(REPO_ROOT)
    selected = _select_needles(pools, min_chars=min_chars, ambiguous=ambiguous)
    files, haystacks = _collect_files(roots)

    all_leaks = _find_leaks(selected, haystacks)
    blocker_files = {str(REPO_ROOT / name) for name in RELEASE_BLOCKERS}
    blockers = [item for item in all_leaks if item["file"] in blocker_files]
    leaks = [item for item in all_leaks if item["file"] not in blocker_files]
    blocker_report = [
        {
            "file": name,
            "reason": RELEASE_BLOCKERS[name],
            "hits": sum(1 for item in blockers if item["file"] == str(REPO_ROOT / name)),
            "blocks_release": True,
        }
        for name in sorted(RELEASE_BLOCKERS)
    ]
    return {
        "label": label,
        "scan_roots": [str(item) for item in roots],
        "cases_dir": str(directory),
        "private_dir": str(resolve_private_dir(private_dir)),
        "min_needle_chars": min_chars,
        "files_scanned": len(files),
        "needles_checked": sum(len(values) for values in selected.values()),
        "needle_counts": {kind: len(values) for kind, values in sorted(selected.items())},
        "thresholds": thresholds_by_kind(min_chars),
        "declared_short_fragments": sorted(selected.get("declared_short_fragments", ())),
        "ambiguous_needles_skipped": sorted(
            {item for values in pools.values() for item in values if item in ambiguous}
        ),
        "leaks": leaks,
        "unexpected_leaks": leaks,
        "blockers": blockers,
        "release_blockers": blocker_report,
        "release_blocked": any(entry["hits"] for entry in blocker_report),
        "ok": not leaks,
    }


def assert_no_leaks(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
    scan_roots: Sequence[Path | str] | None = None,
    min_chars: int = MIN_NEEDLE_CHARS,
) -> Mapping[str, Any]:
    """数据集目录模式断言：任何命中即失败（T21 验收核心）。"""

    report = scan(
        cases_dir=cases_dir,
        private_dir=private_dir,
        scan_roots=scan_roots,
        min_chars=min_chars,
    )
    if report["leaks"]:
        sample = report["leaks"][:5]
        raise AssertionError(f"仓库出现私人正文：{len(report['leaks'])} 处命中；示例 {sample!r}")
    return report


def assert_repo_clean(
    *,
    cases_dir: Path | str | None = None,
    private_dir: Path | str | None = None,
) -> Mapping[str, Any]:
    """仓库模式断言：未被豁免的文件必须 0 命中（发布阻断项单独报出）。"""

    report = scan_repo(cases_dir=cases_dir, private_dir=private_dir)
    if report["leaks"]:
        sample = [
            {**item, "file": Path(item["file"]).relative_to(REPO_ROOT).as_posix()}
            for item in report["leaks"][:5]
        ]
        raise AssertionError(
            f"仓库出现私人正文：{len(report['leaks'])} 处命中（发布阻断项除外）；示例 {sample!r}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据集私文泄漏自检（PRD §7.2）")
    parser.add_argument("--cases-dir", default=None)
    parser.add_argument("--private-dir", default=None)
    parser.add_argument(
        "--dataset-dir",
        action="store_true",
        help="只扫 scripts/relevance_cases/（最严阈值）；默认整仓模式",
    )
    parser.add_argument("--root", action="append", default=None, help="仓库模式下的扫描根（可多次）")
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.dataset_dir:
            report = scan(
                cases_dir=args.cases_dir,
                private_dir=args.private_dir,
                min_chars=args.min_chars if args.min_chars is not None else MIN_NEEDLE_CHARS,
            )
        else:
            report = scan_repo(
                cases_dir=args.cases_dir,
                private_dir=args.private_dir,
                scan_roots=args.root,
                min_chars=args.min_chars,
            )
    except PrivateDatasetMissing as exc:
        print(f"[skip] 本地私文不可用：{exc}")
        return 0

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"[{report['label']}] roots={report['scan_roots']} files={report['files_scanned']} "
            f"needles={report['needles_checked']} leaks={len(report['leaks'])}"
        )
        for kind, count in report["needle_counts"].items():
            threshold = report["thresholds"].get(kind, {})
            print(
                f"  needle[{kind}]={count}  min_chars={threshold.get('min_chars')} "
                f"— {threshold.get('rationale', '')}"
            )
        short = report.get("declared_short_fragments") or []
        short_hashes = [_sha256_text(item)[:8] for item in short]
        print(
            f"  显式登记的短片段（无阈值豁免，命中即泄漏）：{len(short)} 条"
            + (f"，sha256[:8]={short_hashes}" if short_hashes else "")
        )
        for item in report["leaks"][:20]:
            print(
                f" - {item['kind']} in {item['file']} "
                f"(sha256={item['needle_sha256'][:12]} len={item['needle_length']})"
            )
        for entry in report.get("release_blockers", []):
            print(
                f" [RELEASE BLOCKER] {entry['file']}：命中 {entry['hits']} 处 — {entry['reason']}"
            )
    if report["leaks"]:
        return 1
    return 2 if report.get("release_blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
