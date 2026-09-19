"""benchmark 用例集脱敏契约测试（PRD §7.2，T22）。

- 仓库内 ``benchmark/cases.json`` 只保留结构（id/cat/diff/gold + 哈希 + 计数），
  不得含问题原文或证据串；
- 完整用例由 ``benchmark/cases_loader.py`` 从 Git 忽略的本地私文组装；
- 数据产物（``*_results*.json`` 等）不进仓库（``.gitignore`` 忽略，只留本地）。

无私文时（CI / 公开环境）需要组装的用例 skip 并说明原因；结构契约始终生效。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
CASES_PATH = BENCH / "cases.json"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import cases_loader

PRIVATE_ABSENT_REASON = (
    "private dataset not present: 本地 benchmark 私文 "
    f"{cases_loader.resolve()} 缺失（Git 忽略，仅本机存在）；"
    "结构契约用例仍在本环境执行"
)

#: 仓库允许保留的非源码文件（生成脚本 + 聚合结果 + 脱敏结构）。
ALLOWED_REPO_JSON = {
    "cases.json",
    "aggregate.json",
    "aggregate_optimized.json",
    "answer_aggregate.json",
}


def _structure() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_cases_structure_declares_redaction_and_counts() -> None:
    payload = _structure()

    redaction = payload.get("redaction")
    assert isinstance(redaction, dict), "benchmark 用例集必须声明脱敏口径（PRD §7.2）"
    assert redaction["loader"] == "benchmark/cases_loader.py"
    assert redaction["private_dataset"].endswith("cases-private.json")
    counts = payload["counts"]
    assert counts["cases"] == len(payload["cases"]) == 87
    assert counts["evidence_items"] > 0
    assert sum(counts["by_cat"].values()) == counts["cases"]
    assert sum(counts["by_diff"].values()) == counts["cases"]


def test_cases_structure_has_no_private_text() -> None:
    payload = _structure()

    for case in payload["cases"]:
        assert set(case) == {
            "id",
            "cat",
            "diff",
            "gold",
            "n_evidence",
            "query_sha256",
            "evidence_sha256",
            "case_sha256",
        }, f"{case['id']} 出现了结构外的字段（可能是私文）"
        assert isinstance(case["gold"], list)
        assert case["n_evidence"] == len(case["evidence_sha256"])
        assert all(len(item) == 64 for item in case["evidence_sha256"])
        assert len(case["query_sha256"]) == 64
        assert len(case["case_sha256"]) == 64
    assert all("query" not in case and "evidence" not in case for case in payload["cases"])


def test_loader_assembles_full_cases_locally() -> None:
    if not cases_loader.has_private_cases():
        pytest.skip(PRIVATE_ABSENT_REASON)

    cases = cases_loader.load_cases()

    assert len(cases) == 87
    assert all(case["query"].strip() for case in cases)
    assert all("evidence" in case for case in cases)
    structure = {case["id"]: case for case in _structure()["cases"]}
    for case in cases:
        assert len(structure[case["id"]]["query_sha256"]) == 64
        assert cases_loader.case_digest(case) == structure[case["id"]]["case_sha256"]


def test_loader_rejects_mismatched_private_cases(tmp_path: Path) -> None:
    if not cases_loader.has_private_cases():
        pytest.skip(PRIVATE_ABSENT_REASON)

    payload = json.loads(cases_loader.resolve().read_text(encoding="utf-8"))
    payload["cases"][0]["query"] = payload["cases"][0]["query"] + "（已篡改）"
    mutated = tmp_path / "cases-private.json"
    mutated.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="哈希不同"):
        cases_loader.load_cases(private_path=mutated)


def test_loader_reports_missing_private_cases(tmp_path: Path) -> None:
    with pytest.raises(cases_loader.PrivateCasesMissing, match="本地私文不存在"):
        cases_loader.load_cases(private_path=tmp_path / "absent.json")


def test_data_products_are_gitignored() -> None:
    """数据产物必须被 .gitignore 忽略；三个聚合结果与脱敏结构必须不被忽略。"""

    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "benchmark/*.json" in rules
    for name in sorted(ALLOWED_REPO_JSON):
        assert f"!benchmark/{name}" in rules, f"benchmark/{name} 必须显式放行"

    try:
        subprocess.run(
            ["git", "--version"], cwd=ROOT, capture_output=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 无 git 环境
        pytest.skip("git 不可用，无法校验 ignore 规则的行为")

    def ignored(relative: str) -> bool:
        """未跟踪路径的忽略状态（已跟踪文件 git check-ignore 会一律报未忽略）。"""

        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode not in (0, 1):
            pytest.skip("git check-ignore 不可用")
        return result.returncode == 0

    # 未来重跑生成的数据产物同样必须被忽略。
    # 注意：`git check-ignore` 对**已跟踪**文件一律报「未忽略」，因此这里用未被跟踪过的
    # 路径验证规则本身；已被删除的历史产物由 test_repo_has_no_stray_data_products 兜底。
    for name in ("retrieval_results_v7.json", "chunking_results_v3.json", "aggregate_v2.json"):
        assert ignored(f"benchmark/{name}"), f"benchmark/{name} 必须被 ignore 规则覆盖"
    assert not ignored("benchmark/cases.json")
    for name in ("sample_chunks.json", "answer_results.json", "reindexed_quality_v6.json"):
        assert not (BENCH / name).exists(), f"benchmark/{name} 必须已从工作树移除"


def test_repo_has_no_stray_data_products() -> None:
    """工作树里不得残留未被忽略的 benchmark JSON（除白名单）。"""

    present = {path.name for path in BENCH.glob("*.json")} - ALLOWED_REPO_JSON
    assert not present, f"benchmark/ 出现未纳入白名单的数据文件：{sorted(present)}"


def test_repo_copy_is_preserved_for_local_reruns() -> None:
    """本地私文副本仍可用于重跑（缺私文时 skip）。"""

    if not cases_loader.has_private_cases():
        pytest.skip(PRIVATE_ABSENT_REASON)

    private_dir = cases_loader.private_dir()
    expected = {
        "cases-private.json",
        "sample_chunks.json",
        "answer_results.json",
        "retrieval_results.json",
    }
    missing = sorted(name for name in expected if not (private_dir / name).is_file())
    assert not missing, f"本地私文目录缺少重跑所需文件：{missing}"


def test_validate_cases_script_still_assembles() -> None:
    """消费者可用性：validate_cases.py 依赖的组装路径必须返回完整用例。"""

    if not cases_loader.has_private_cases():
        pytest.skip(PRIVATE_ABSENT_REASON)

    cases = cases_loader.load_cases(CASES_PATH)
    samples = {sample["n"]: sample for sample in cases_loader.load_sample_chunks()}
    positives = [case for case in cases if case["gold"]]
    assert positives
    for case in positives[:5]:
        assert all(n in samples for n in case["gold"])


def test_private_dir_can_be_relocated(tmp_path: Path) -> None:
    if not cases_loader.has_private_cases():
        pytest.skip(PRIVATE_ABSENT_REASON)

    copied = tmp_path / "bench-private"
    shutil.copytree(cases_loader.private_dir(), copied)
    cases = cases_loader.load_cases(private_path=copied / "cases-private.json")
    assert len(cases) == 87
