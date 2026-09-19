"""分级相关性数据集契约测试（C01–C06，PRD 7.1/7.2）。

仓库内 ``scripts/relevance_cases/cases-C*.json`` 是**脱敏结构**（PRD §7.2）：
问题原文、来源 basename、标题、证据片段正文一律不入库，只留非私密字段与私文哈希。

- **不需要私文的用例**（结构、分组、manifest、脱敏契约）在任何环境都跑，包括 CI。
- **需要私人正文的用例**（组装、逐字校验、防回归自检）用 ``scripts/relevance_cases/loader.py``
  读取 Git 忽略的本地私文 ``.tmp_probe/rag-relevance/relevance-private/``；
  私文缺失时这些用例 **skip 并给出明确原因**（"private dataset not present"），
  既不静默通过，也不报成失败而与真实缺陷混淆。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "scripts" / "relevance_cases"
CASE_FILES = sorted(CASES_DIR.glob("cases-C*.json"))
MANIFEST_PATH = CASES_DIR / "manifest.json"
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from relevance_cases import loader, private_guard
from relevance_cases.loader import PrivateDatasetMissing

from app.evaluation.rag_quality import (
    RELEVANCE_SCHEMA_VERSION,
    RelevanceCase,
    load_relevance_cases,
    validate_relevance_cases,
)

PRIVATE_ABSENT_REASON = (
    "private dataset not present: 本地私文 "
    f"{loader.resolve_private_dir()} 缺失（Git 忽略，仅本机存在）；"
    "结构/manifest 一致性用例仍在本环境执行"
)

#: 结构文件里**绝不出现**的私密字段（PRD §7.2 的脱敏面）。
FORBIDDEN_PRIVATE_FIELDS = {
    "query",
    "source",
    "heading",
    "required_evidence",
    "irrelevant_spans",
    "relevant_spans",
    "note",
}


def _structure(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _redacted_cases(path: Path):
    """只用仓库结构（无需私文）。"""

    return load_relevance_cases(path, allow_redacted=True)


def _assembled_cases(path: Path):
    """结构 + 本地私文；私文缺失时 skip。"""

    try:
        return loader.load_cases(path)
    except PrivateDatasetMissing as exc:
        pytest.skip(f"{PRIVATE_ABSENT_REASON}（{exc}）")


# ---------------------------------------------------------------------------
# 仓库契约（无需私文，任何环境都跑）
# ---------------------------------------------------------------------------


def test_dataset_directory_is_committed_and_versioned() -> None:
    assert CASES_DIR.is_dir(), "分级相关性数据集必须随仓库交付"
    assert CASE_FILES, "至少要有 C01 数据集文件"
    assert MANIFEST_PATH.is_file(), "必须随数据集交付脱敏 manifest"


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda item: item.stem)
def test_family_structure_matches_schema_and_split_contract(path: Path) -> None:
    payload = _structure(path)
    assert payload["schema_version"] == RELEVANCE_SCHEMA_VERSION

    cases = _redacted_cases(path)

    assert len(cases) >= 20, f"{path.stem} 需要至少 20 题"
    calibration = [case for case in cases if case.split == "calibration"]
    test = [case for case in cases if case.split == "test"]
    assert len(calibration) == 10, f"{path.stem} 校准集必须是 10 题"
    assert len(test) == 10, f"{path.stem} 独立测试集必须是 10 题"
    assert {case.family for case in cases} == {payload["family"]}
    validate_relevance_cases(cases, allow_redacted=True)
    assert all(case.case_id for case in cases)
    assert all(case.query is None for case in cases), "脱敏结构不得携带问题原文"
    assert all(case.targets for case in cases), f"{path.stem} 每题都要有标注目标"


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda item: item.stem)
def test_structure_declares_redaction_and_carries_hashes(path: Path) -> None:
    payload = _structure(path)

    redaction = payload.get("redaction")
    assert isinstance(redaction, dict), f"{path.stem} 必须声明脱敏口径（PRD §7.2）"
    assert redaction["private_dataset"].endswith(f"private-{payload['family']}.json")
    assert redaction["loader"] == "scripts/relevance_cases/loader.py"
    counts = payload.get("counts")
    assert isinstance(counts, dict) and counts["cases"] == len(payload["cases"])
    assert counts["targets"] == sum(len(case["targets"]) for case in payload["cases"])
    assert counts["grades"], "统计必须包含等级分布"
    assert counts["roles"], "统计必须包含角色分布"

    for case in payload["cases"]:
        assert set(case) & FORBIDDEN_PRIVATE_FIELDS == set(), (
            f"{case['case_id']} 的案例结构里出现了私密字段："
            f"{sorted(set(case) & FORBIDDEN_PRIVATE_FIELDS)}"
        )
        assert len(case["query_sha256"]) == 64, f"{case['case_id']} 必须给出 query 哈希"
        assert len(case["case_sha256"]) == 64, f"{case['case_id']} 必须给出整题私文哈希"
        for target in case["targets"]:
            leaked = set(target) & FORBIDDEN_PRIVATE_FIELDS
            assert leaked == set(), (
                f"{target['target_id']} 的目标结构里出现了私密字段：{sorted(leaked)}"
            )
            assert set(target) == {"target_id", "role", "grade", "group"}


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda item: item.stem)
def test_redacted_structure_requires_local_private_dataset(path: Path) -> None:
    """未组装私文的结构案例不得当作可评测数据集（避免「静默当无关」）。"""

    with pytest.raises(ValueError, match="脱敏结构"):
        load_relevance_cases(path)


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda item: item.stem)
def test_every_case_declares_evidence_and_hard_negatives(path: Path) -> None:
    cases = _assembled_cases(path)

    for case in cases:
        assert case.query, f"{case.case_id} 组装后必须有问题原文"
        assert case.targets, f"{case.case_id} 没有任何标注目标"
        assert case.forbidden_targets, f"{case.case_id} 必须标注至少一个明确排除的难负例"
        if case.needs_answer:
            assert case.required_targets, f"{case.case_id} 有答案问题必须标注必要证据"
            for target in case.required_targets:
                assert target.grade in (2, 3), f"{case.case_id} 必要证据等级必须为 2/3"
                assert target.required_evidence, f"{case.case_id} 必要证据必须给出原文片段"
                assert target.source, f"{case.case_id} 组装后必须带来源"
            assert case.required_groups(), f"{case.case_id} 必要证据必须能按证据族分组"
        else:
            assert not case.required_targets, f"{case.case_id} 无证据问题不得标注必要证据"
        for target in case.forbidden_targets:
            assert target.grade in (0, 1), f"{case.case_id} 排除目标等级必须是 0/1"
        assert not case.has_unjudged_targets, f"{case.case_id} 交付的标注不允许残留 unjudged 目标"
        assert not case.is_redacted, f"{case.case_id} 组装后不得残留脱敏缺口"


def test_groups_do_not_cross_splits_and_corpus_hash_is_consistent() -> None:
    all_cases: list[RelevanceCase] = []
    hashes = set()
    for path in CASE_FILES:
        payload = _structure(path)
        hashes.add(payload["corpus_sha256"])
        all_cases.extend(_redacted_cases(path))

    assert len(hashes) == 1, "所有家族数据集必须来自同一份语料快照"
    validate_relevance_cases(all_cases, allow_redacted=True)
    ids = [case.case_id for case in all_cases]
    assert len(ids) == len(set(ids)), "跨家族 case_id 必须唯一"
    assert len(ids) == 120, "六类数据集共 120 题"


def test_target_level_groups_are_case_scoped_and_never_cross_splits() -> None:
    """target 级 group 是证据族：只能在同一道题内复用，绝不能跨题/跨集合。"""

    seen: dict[str, tuple[str, str]] = {}
    for path in CASE_FILES:
        for case in _redacted_cases(path):
            for target in case.targets:
                if not target.group:
                    continue
                assert target.group.startswith(f"{case.case_id}:"), (
                    f"{case.case_id} 的 target 级证据族 {target.group!r} 没有按案例限定"
                )
                previous = seen.setdefault(target.group, (case.case_id, case.split))
                assert previous == (case.case_id, case.split), (
                    f"证据族 {target.group!r} 被多道题/多个集合复用：{previous} vs "
                    f"{(case.case_id, case.split)}"
                )
                members = [item for item in case.targets if item.group == target.group]
                assert len(members) >= 2, f"证据族 {target.group!r} 只有一个成员，等价性不成立"


def test_target_level_group_members_come_from_distinct_blocks() -> None:
    """块的等价性（来源 + 标题）需要私人正文，无私文时 skip。"""

    for path in CASE_FILES:
        for case in _assembled_cases(path):
            for target in case.targets:
                if not target.group:
                    continue
                members = [item for item in case.targets if item.group == target.group]
                blocks = {(item.source, item.heading) for item in members}
                assert len(blocks) == len(members), f"证据族 {target.group!r} 的成员来自同一个块"


def test_required_evidence_groups_count_each_target_once() -> None:
    """没有等价证据时，每个必要证据目标必须自成一族（族数 == 必要目标数）。"""

    for path in CASE_FILES:
        cases = _redacted_cases(path)
        total_targets = sum(len(case.required_targets) for case in cases)
        total_groups = sum(len(case.required_groups()) for case in cases)
        declared = [target for case in cases for target in case.required_targets if target.group]
        if not declared:
            assert total_groups == total_targets, (
                f"{path.stem} 未声明等价证据，但必要证据族数 {total_groups} "
                f"!= 必要目标数 {total_targets}"
            )
        else:
            assert total_groups <= total_targets


def _required_target(target_id: str, group: str, source: str, heading: str):
    from app.evaluation.rag_quality import RelevanceTarget

    return RelevanceTarget(
        target_id=target_id,
        role="required",
        grade=3,
        source=source,
        heading=heading,
        required_evidence=(f"{target_id} 的证据片段",),
        group=group,
    )


def test_loader_rejects_cross_case_reuse_of_target_level_group() -> None:
    """复现本次缺陷：同一个 target 级族名跨题/跨集合复用必须被拒绝。"""

    calibration = RelevanceCase(
        case_id="C02-001",
        family="C02",
        group_id="C02-g-001",
        split="calibration",
        query="q1",
        targets=(_required_target("C02-001-T1", "g1", "a.md", "A"),),
    )
    test = RelevanceCase(
        case_id="C02-002",
        family="C02",
        group_id="C02-g-002",
        split="test",
        query="q2",
        targets=(_required_target("C02-002-T1", "g1", "b.md", "B"),),
    )

    with pytest.raises(ValueError, match="必须按案例限定"):
        validate_relevance_cases((calibration, test))


def test_loader_rejects_single_member_and_same_block_evidence_group() -> None:
    single = RelevanceCase(
        case_id="C02-003",
        family="C02",
        group_id="C02-g-003",
        split="test",
        query="q",
        targets=(_required_target("C02-003-T1", "C02-003:eq", "a.md", "A"),),
    )
    with pytest.raises(ValueError, match="只有 1 个成员"):
        validate_relevance_cases((single,))

    same_block = RelevanceCase(
        case_id="C02-004",
        family="C02",
        group_id="C02-g-004",
        split="test",
        query="q",
        targets=(
            _required_target("C02-004-T1", "C02-004:eq", "a.md", "A"),
            _required_target("C02-004-T2", "C02-004:eq", "a.md", "A"),
        ),
    )
    with pytest.raises(ValueError, match="成员必须来自不同的块"):
        validate_relevance_cases((same_block,))


# ---------------------------------------------------------------------------
# 脱敏 manifest
# ---------------------------------------------------------------------------


def test_manifest_statistics_match_case_files() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["redacted"] is True
    assert manifest["corpus_sha256"]
    families = {entry["family"]: entry for entry in manifest["families"]}
    totals_cases = 0
    totals_targets = 0
    for path in CASE_FILES:
        payload = _structure(path)
        family = payload["family"]
        assert family in families, f"manifest 缺少 {family}"
        entry = families[family]
        assert entry["cases"] == len(payload["cases"])
        assert entry["calibration"] == 10
        assert entry["test"] == 10
        assert entry["targets"] == sum(len(case["targets"]) for case in payload["cases"])
        assert entry["targets"] == payload["counts"]["targets"]
        assert entry["grades"] == payload["counts"]["grades"]
        assert entry["roles"] == payload["counts"]["roles"]
        assert sum(entry["grades"].values()) == entry["targets"]
        totals_cases += entry["cases"]
        totals_targets += entry["targets"]

    assert manifest["totals"]["cases"] == totals_cases == 120
    assert manifest["totals"]["targets"] == totals_targets == 729
    assert manifest["totals"]["families"] == len(CASE_FILES) == 6
    assert manifest["totals"]["calibration"] == 60
    assert manifest["totals"]["test"] == 60


def test_manifest_contains_only_statistics_and_hashes() -> None:
    """manifest 不得含来源名、标题或逐题私人正文（脱敏 manifest 只放统计）。"""

    payload = _structure(MANIFEST_PATH)
    # ``annotation_protocol`` 是标注口径的说明文字（含「证据片段」等术语），不是数据。
    data = {key: value for key, value in payload.items() if key != "annotation_protocol"}
    # ``totals`` / ``families[*]`` 里的 ``targets`` 是计数，不是逐目标数组；数组本身由
    # 下面「不得出现 dict 元素」的断言兜住。
    forbidden_keys = {
        "source",
        "sources",
        "heading",
        "headings",
        "query",
        "queries",
        "target",
        "required_evidence",
        "note",
    }

    def _walk(node, trail: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            assert not (set(node) & forbidden_keys), f"manifest 出现私密字段：{trail} {sorted(node)}"
            assert "case_id" not in node, f"manifest 出现逐题数据：{trail}"
            for key, value in node.items():
                _walk(value, (*trail, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, (*trail, str(index)))

    _walk(data)
    assert payload["redacted"] is True
    assert payload["privacy"] and payload["redaction_policy"]
    assert "loader" in payload["verification"]["local_assembly"]


def test_manifest_never_contains_local_private_strings() -> None:
    """manifest 里不得出现本地私文的任何来源名/标题/原文/片段（需要私文）。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    text = MANIFEST_PATH.read_text(encoding="utf-8")
    index = loader.private_string_index()
    for kind in ("sources", "headings", "queries", "evidence", "span_texts"):
        for needle in set(index[kind]):
            assert needle not in text, f"manifest 出现私文（{kind}）：sha256={loader._sha256_text(needle)[:12]}"


# ---------------------------------------------------------------------------
# 本地组装（需要私文）
# ---------------------------------------------------------------------------


def test_loader_assembles_full_dataset_locally() -> None:
    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    cases = loader.load_dataset()

    assert len(cases) == 120, "六类数据集共 120 题"
    assert sum(len(case.targets) for case in cases) == 729, "共 729 个标注目标"
    assert sum(1 for case in cases if case.split == "calibration") == 60
    assert sum(1 for case in cases if case.split == "test") == 60
    assert all(case.query for case in cases), "组装后每题必须有问题原文"
    assert not any(case.is_redacted for case in cases)


def test_assembled_queries_match_structure_hashes() -> None:
    """组装后的问题原文必须与仓库结构里的 ``query_sha256`` 逐题一致。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    hashes = {
        case["case_id"]: case["query_sha256"]
        for path in CASE_FILES
        for case in _structure(path)["cases"]
    }
    cases = loader.load_dataset()
    assert len(cases) == len(hashes)
    for case in cases:
        assert loader._sha256_text(case.query or "") == hashes[case.case_id], case.case_id


def test_private_dataset_hashes_match_repo_structure() -> None:
    """结构文件里的哈希必须逐字对得上本地私文（错配即拒绝加载）。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    for path in CASE_FILES:
        payload = _structure(path)
        private = loader.load_private_family(payload["family"])
        private_by_id = {case["case_id"]: case for case in private["cases"]}
        for case in payload["cases"]:
            local = private_by_id[case["case_id"]]
            assert case["query_sha256"] == loader._sha256_text(local["query"])
            merged = {
                **{key: value for key, value in case.items() if key != "targets"},
                "query": local["query"],
                "targets": (
                    [{**structure_target, **private_target} for structure_target, private_target in zip(
                        case["targets"], local["targets"], strict=True
                    )]
                ),
            }
            assert case["case_sha256"] == loader.case_digest(merged), (
                f"{case['case_id']} 的私文与仓库结构不一致（case_sha256 不同）"
            )


def test_loader_rejects_private_dataset_that_does_not_match_structure() -> None:
    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    family = "C01"
    private = loader.load_private_family(family)
    private["cases"][0]["query"] = private["cases"][0]["query"] + "（已篡改）"

    with pytest.raises(ValueError, match="hash|哈希"):
        load_relevance_cases(
            CASES_DIR / f"cases-{family}.json",
            private_dataset=private,
        )


def test_loader_reports_missing_private_dataset(tmp_path: Path) -> None:
    """私文缺失必须显式报错（skip 由调用方决定），不得静默返回半成品。"""

    with pytest.raises(PrivateDatasetMissing, match="本地私文数据集不存在"):
        loader.load_cases(CASE_FILES[0], private_dir=tmp_path / "absent")


def test_loader_supports_relocated_dataset_directories(tmp_path: Path) -> None:
    """加载器目录可重定位：把结构目录复制到别处后照样能读出结构（无需私文）。"""

    cases_copy = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases_copy)

    cases = loader.load_cases(
        cases_copy / "cases-C01.json",
        private_dir=tmp_path / "absent-private",
        allow_redacted=True,
    )
    assert len(cases) == 20
    assert all(case.is_redacted for case in cases)


def test_loader_supports_relocated_private_directory(tmp_path: Path) -> None:
    """私文目录可重定位（自定义环境用参数或环境变量）；无私文时 skip。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    cases_copy = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases_copy)
    private_copy = tmp_path / "private"
    shutil.copytree(loader.resolve_private_dir(), private_copy)

    cases = loader.load_cases(cases_copy / "cases-C01.json", private_dir=private_copy)
    assert len(cases) == 20
    assert all(case.query and not case.is_redacted for case in cases)


# ---------------------------------------------------------------------------
# 防回归自检：私文不得出现在仓库数据集里（本项验收核心）
# ---------------------------------------------------------------------------


def test_private_strings_never_leak_into_repo_dataset() -> None:
    """本地私文里的来源名/标题/问题原文/证据片段不得出现在 scripts/relevance_cases/。"""

    try:
        loader.private_string_index()  # 先确认本地私文齐备（缺失即 skip，不是失败）
    except PrivateDatasetMissing as exc:
        pytest.skip(f"{PRIVATE_ABSENT_REASON}（{exc}）")

    report = private_guard.assert_no_leaks(cases_dir=CASES_DIR)

    assert report["ok"], f"仓库数据集出现私人正文：{report['leaks'][:5]!r}"
    counts = report["needle_counts"]
    assert counts["queries"] >= 100, "自检必须真的覆盖全部问题原文"
    assert counts["sources"] >= 20, "自检必须真的覆盖来源 basename"
    assert counts["headings"] >= 100, "自检必须真的覆盖标题"
    assert counts["evidence"] >= 100, "自检必须真的覆盖证据片段"
    assert report["leaks"] == []


def test_private_guard_detects_a_planted_leak(tmp_path: Path) -> None:
    """反向自检：把私文片段写进数据集目录时必须被抓到（证明上面的用例不是空转）。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    cases_copy = tmp_path / "cases"
    shutil.copytree(CASES_DIR, cases_copy)
    private_copy = tmp_path / "private"
    shutil.copytree(loader.resolve_private_dir(), private_copy)

    index = loader.private_string_index(private_dir=private_copy)
    planted = max(index["evidence"], key=len)
    (cases_copy / "planted-leak.md").write_text(f"泄漏片段：{planted}", encoding="utf-8")

    report = private_guard.scan(cases_dir=cases_copy, private_dir=private_copy)
    assert not report["ok"], "防回归自检必须能抓到植入的私文片段"
    assert any(item["file"].endswith("planted-leak.md") for item in report["leaks"])


def test_private_strings_absent_from_whole_repo() -> None:
    """T22：整仓（benchmark/、backend/tests/、scripts/、docs/…）不得承载私人正文。

    唯一豁免是**发布阻断项** ``scripts/rag_eval_cases.json``（黄金集，留待校准批次），
    它由 ``release_blockers`` 单独报出且 ``release_blocked=True``，绝不静默放过。
    """

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    report = private_guard.assert_repo_clean(cases_dir=CASES_DIR)

    assert report["leaks"] == [], f"仓库出现私人正文：{report['leaks'][:5]!r}"
    assert report["ok"] is True

    blockers = {entry["file"]: entry for entry in report["release_blockers"]}
    assert "scripts/rag_eval_cases.json" in blockers, "发布阻断项必须被显式列出"
    blocker = blockers["scripts/rag_eval_cases.json"]
    assert blocker["blocks_release"] is True
    assert blocker["hits"] > 0, "黄金集仍含真实来源名/标题，命中数必须 > 0（不是「没问题」）"
    assert report["release_blocked"] is True
    assert all(
        Path(item["file"]).as_posix().endswith("scripts/rag_eval_cases.json")
        for item in report["blockers"]
    )


def test_repo_guard_detects_a_planted_leak(tmp_path: Path) -> None:
    """反向自检：仓库模式也必须能抓到植入的私文片段（证明扫描不是空转）。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    index = loader.private_string_index()
    pools = private_guard.repo_needle_pools()
    planted = max(pools.get("benchmark_evidence", index["evidence"]), key=len)

    planted_dir = tmp_path / "repo-like"
    planted_dir.mkdir()
    (planted_dir / "planted-leak.md").write_text(f"泄漏片段：{planted}", encoding="utf-8")

    report = private_guard.scan_repo(cases_dir=CASES_DIR, scan_roots=[planted_dir])
    assert report["ok"] is False, "仓库模式必须能抓到植入的私文片段"
    assert any(item["file"].endswith("planted-leak.md") for item in report["leaks"])


def test_repo_guard_skips_repo_own_filenames(tmp_path: Path) -> None:
    """与仓库自身同名的语料文件名（CLAUDE.md/README.md）不构成泄漏证据，必须被豁免。"""

    if not loader.has_private_dataset():
        pytest.skip(PRIVATE_ABSENT_REASON)

    report = private_guard.scan_repo(cases_dir=CASES_DIR)
    skipped = set(report["ambiguous_needles_skipped"])
    assert "CLAUDE.md" in skipped and "README.md" in skipped
    assert not any(item["file"].endswith("CLAUDE.md") for item in report["leaks"])


def _local_short_fragments() -> tuple[str, ...]:
    """读取本地登记的短片段；本地私文/敏感串文件缺失时 skip（原因统一、不混淆）。"""

    sensitive = loader.resolve_private_dir() / "sensitive-needles.json"
    if not loader.has_private_dataset() or not sensitive.is_file():
        pytest.skip(PRIVATE_ABSENT_REASON)
    fragments = private_guard.declared_short_fragments()
    if not fragments:
        pytest.skip("本地私文未登记 short_fragments（T23-F4 的 12 字语料片段）")
    return fragments


def test_declared_short_fragments_absent_from_repo() -> None:
    """T23-F4 防回归：低于同类阈值、已显式登记的短私文片段不得出现在仓库文件里。"""

    fragments = _local_short_fragments()

    report = private_guard.scan_repo(cases_dir=CASES_DIR)
    assert report["declared_short_fragments"], "显式短片段必须真的进入 needle 集合（不得空转）"
    assert not [item for item in report["leaks"] if item["kind"] == "declared_short_fragments"]
    assert not [item for item in report["blockers"] if item["kind"] == "declared_short_fragments"]

    # 点名文件：T23-F4 的片段来自诊断报告，必须已被中性文字替代
    doc = ROOT / "docs" / "诊断报告-检索质量与生成行为.md"
    text = doc.read_text(encoding="utf-8")
    for fragment in fragments:
        assert fragment not in text, (
            f"诊断报告不得保留语料原文片段（{len(fragment)} 字，sha256="
            f"{loader._sha256_text(fragment)[:12]}）"
        )


def test_declared_short_fragment_needle_is_enforced(tmp_path: Path) -> None:
    """反向自检：把已登记的短片段再写回仓库式目录时必须被抓到（证明阈值不豁免它）。"""

    fragments = _local_short_fragments()

    planted_dir = tmp_path / "repo-like"
    planted_dir.mkdir()
    (planted_dir / "planted-short.md").write_text(f"片段：{fragments[0]}", encoding="utf-8")

    report = private_guard.scan_repo(cases_dir=CASES_DIR, scan_roots=[planted_dir])
    assert any(
        item["kind"] == "declared_short_fragments" and item["file"].endswith("planted-short.md")
        for item in report["leaks"]
    ), "显式登记的短片段必须按最严阈值命中"


def test_guard_documents_threshold_rationale() -> None:
    """阈值口径必须可机读：每个 kind 都要有阈值 + 理由（不再用「阈值挡住了」解释遗漏）。"""

    thresholds = private_guard.thresholds_by_kind()

    for kind in (
        "sources",
        "corpus_stems",
        "headings",
        "corpus_headings",
        "queries",
        "evidence",
        "notes",
        "sensitive",
        "declared_short_fragments",
    ):
        assert thresholds[kind]["min_chars"] > 0, kind
        assert thresholds[kind]["rationale"], f"{kind} 缺少阈值理由"

    assert thresholds["evidence"]["min_chars"] == 16
    assert "declared_short_fragments" in thresholds["evidence"]["rationale"], (
        "evidence 类的理由必须写明短片段改走 declared_short_fragments"
    )
    assert thresholds["declared_short_fragments"]["min_chars"] == 2
    assert "无阈值豁免" in thresholds["declared_short_fragments"]["rationale"]

    # 报告里也要带上口径（CLI 打印的就是这份数据）
    report = private_guard.scan_repo(cases_dir=CASES_DIR) if loader.has_private_dataset() else None
    if report is not None:
        assert report["thresholds"]["evidence"]["rationale"]
