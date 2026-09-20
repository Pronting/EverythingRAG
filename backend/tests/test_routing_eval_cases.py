"""路由评测集测试（PRD 16.3）。Owner: T06（routing）。

覆盖：评测集结构与配额校验、真实条目与 `benchmark/cases.json` 逐条一致的来源纪律、
真实/构造账目、family 与 split 一致性、`--dry-run` 输出、六项离线指标阈值、
逐条证据（RouteDecision / 范围检查 / 计划工具）落盘。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_routing import (
    CATEGORIES,
    DEFAULT_CASES,
    QUOTAS,
    SOURCE_CASES,
    distribution,
    load_dataset,
    main,
    query_digest,
    run,
    source_case_digests,
    validate_dataset,
)

REQUIRED_CASE_KEYS = {
    "id",
    "category",
    "question",
    "source",
    "family",
    "split",
    "required_sources",
    "acceptable_intents",
    "freshness_required",
    "forbidden_misstatements",
    "catalog_hints",
    "catalog_hint_origin",
}


@pytest.fixture(scope="module")
def dataset() -> dict:
    return load_dataset(DEFAULT_CASES)


def test_dataset_validates_without_problems(dataset: dict) -> None:
    assert validate_dataset(dataset) == []


def test_real_questions_are_traceable_to_benchmark_cases(dataset: dict) -> None:
    """真实条目必须与 ``benchmark/cases.json`` 逐字一致（集成后按 SHA-256 指纹比对）。

    ``benchmark/cases.json`` 集成后是**脱敏结构**（main 的 ``chore(privacy)``：
    问题原文与证据串不入仓库，只保留 ``query_sha256``）。指纹相等 ⇔ 文本逐字相等，
    因此这条纪律的强度不变，且不再依赖本机是否存有私文副本。
    """

    source = json.loads(Path(SOURCE_CASES).read_text(encoding="utf-8"))
    assert source.get("redaction"), "benchmark/cases.json 应为脱敏结构（集成后口径）"
    lookup = source_case_digests()
    assert lookup, "benchmark/cases.json 必须给出可比对的问题指纹"
    real = [case for case in dataset["cases"] if case["source"] == "real"]
    assert len(real) == 87
    for case in real:
        assert case["reused_case_id"] in lookup
        assert lookup[case["reused_case_id"]] == query_digest(case["question"])
        assert case["constructed_parts"] == []
    constructed = [case for case in dataset["cases"] if case["source"] == "constructed"]
    assert len(constructed) == 68
    for case in constructed:
        assert case["reused_case_id"] == ""
        assert case["constructed_reason"].strip()
        assert "question" in case["constructed_parts"]


def test_counts_quotas_and_splits(dataset: dict) -> None:
    stats = distribution(dataset, path=DEFAULT_CASES)
    assert stats["total"] == 155
    assert stats["real"] + stats["constructed"] == stats["total"]
    for category in CATEGORIES:
        assert stats["by_category"][category] >= QUOTAS[category], category
    assert stats["by_category"]["internal"] == 65
    assert stats["by_category"]["general"] == 30
    assert stats["by_category"]["freshness"] == 20
    assert stats["by_category"]["product"] == 10
    assert stats["by_category"]["multi_source"] == 15
    assert stats["by_category"]["multi_turn"] == 15
    # 开发集与保留集都必须非空，且保留集里有内部问题可供 M2 统计。
    assert stats["by_split"]["dev"] > 0
    assert stats["by_split"]["holdout"] > 0
    holdout_internal = [
        case
        for case in dataset["cases"]
        if case["split"] == "holdout" and case["category"] == "internal"
    ]
    assert holdout_internal
    assert stats["sha256"] and len(stats["sha256"]) == 64


def test_every_case_is_fully_annotated(dataset: dict) -> None:
    for case in dataset["cases"]:
        missing = REQUIRED_CASE_KEYS - set(case)
        assert not missing, f"{case['id']} 缺少字段 {missing}"
        assert case["forbidden_misstatements"], case["id"]
        assert case["catalog_hint_origin"] in {"cases_evidence", "none"}
        if case["catalog_hint_origin"] == "cases_evidence":
            assert case["catalog_hints"], case["id"]
            assert case["source"] == "real", case["id"]
        if case["category"] == "multi_turn":
            assert any(turn["role"] == "user" for turn in case["turns"]), case["id"]


def test_constructed_samples_are_explicitly_labeled(dataset: dict) -> None:
    """PRD 16.1：构造样例必须显式标注，不得混成真实语料。"""

    assert dataset["label_disclaimer"]
    assert dataset["label_origin"] == "authored_by_t6_from_PRD_6.3_and_16.3"
    assert "constructed_reason" in dataset["field_guide"]
    constructed_freshness = [
        case
        for case in dataset["cases"]
        if case["category"] == "freshness" and case["source"] == "constructed"
    ]
    assert len(constructed_freshness) == 19
    assert all("构造" in case["constructed_reason"] for case in constructed_freshness)


def _minimal_dataset(entries: list[dict]) -> dict:
    """自足的最小评测集：校验逻辑不依赖真实数据文件的顺序或规模。"""

    return {
        "schema_version": 1,
        "counts": {"total": len(entries)},
        "cases": entries,
    }


def _minimal_case(case_id: str, family: str, split: str, category: str = "internal") -> dict:
    return {
        "id": case_id,
        "category": category,
        "question": f"问题 {case_id}",
        "source": "constructed",
        "reused_case_id": "",
        "reused_case_cat": "",
        "constructed_parts": ["question", "labels"],
        "constructed_reason": "自足校验用例",
        "family": family,
        "split": split,
        "required_sources": ["knowledge"],
        "acceptable_intents": ["research"],
        "freshness_required": "none",
        "forbidden_misstatements": ["不得编造"],
        "catalog_hints": [],
        "catalog_hint_origin": "none",
        "gold": [],
        "evidence": [],
    }


def test_validate_detects_missing_quota_and_split_conflict(dataset: dict) -> None:
    # --- 配额缺失（多轮类被改成普通类 → multi_turn 配额不足）
    mutated = json.loads(json.dumps(dataset))
    for case in mutated["cases"]:
        if case["category"] == "multi_turn":
            case["category"] = "general"
    problems = validate_dataset(mutated)
    assert any("multi_turn" in problem for problem in problems)

    # --- family 冲突（自足数据，不依赖真实文件的顺序或规模）
    synthetic = _minimal_dataset(
        [
            _minimal_case("X-1", "family-a", "dev"),
            _minimal_case("X-2", "family-a", "holdout"),
        ]
    )
    synthetic_problems = validate_dataset(synthetic)
    assert any("family" in problem for problem in synthetic_problems)

    # --- 真实评测集里确实存在「改写族」，且把族内一条翻到另一集合会被判定冲突
    family_sizes: dict[str, int] = {}
    for case in dataset["cases"]:
        family_sizes[case["family"]] = family_sizes.get(case["family"], 0) + 1
    duplicated = sorted(family for family, size in family_sizes.items() if size > 1)
    assert duplicated, "评测集应包含改写族（同一 gold 组多条）"

    conflicted = json.loads(json.dumps(dataset))
    family_members = [case for case in conflicted["cases"] if case["family"] == duplicated[0]]
    family_members[0]["split"] = "holdout" if family_members[0]["split"] == "dev" else "dev"
    real_problems = validate_dataset(conflicted)
    assert any("family" in problem for problem in real_problems), (
        f"family {duplicated[0]} 的 split 被改后应判冲突，实际：{real_problems}"
    )


def test_validate_detects_real_question_drift(dataset: dict) -> None:
    mutated = json.loads(json.dumps(dataset))
    real = next(case for case in mutated["cases"] if case["source"] == "real")
    real["question"] = real["question"] + "（被改过）"
    problems = validate_dataset(mutated)
    assert any("文本不一致" in problem for problem in problems)


def test_dry_run_prints_distribution_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "路由评测集" in output
    assert "条目总数       : 155" in output
    assert "内部" not in output  # 分布行用类别英文名，避免与正文统计混读
    for category in CATEGORIES:
        assert category in output
    assert "评测集校验：通过" in output
    assert "构建" not in output


def test_dry_run_reports_missing_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.json"
    assert main(["--cases", str(missing), "--dry-run"]) == 2
    assert "评测集错误" in capsys.readouterr().err


def test_rule_layer_meets_prd_16_3_offline_metrics(dataset: dict) -> None:
    """离线规则层指标回归门槛：M1–M6 全部达标（M7 需真实工作区，见报告说明）。"""

    report = run(dataset, catalog=None, split="all", limit=None)
    metrics = {item["id"]: item for item in report["metrics"]}
    assert report["model_calls"] == 0
    assert report["catalog"] == "hint-proxy"
    assert metrics["M1"]["rate"] == 1.0
    assert metrics["M2"]["rate"] == 1.0
    assert metrics["M3"]["passed"] is True
    assert metrics["M3"]["rate"] is not None and metrics["M3"]["rate"] >= 0.95
    assert metrics["M4"]["rate"] == 1.0
    assert metrics["M5"]["rate"] == 1.0
    assert metrics["M6"]["passed"] is True
    assert report["failures"] == []
    assert report["scope_checks"]["cases_with_scope_evidence"] > 0
    assert "M7" in report["not_covered_here"]


def test_report_writes_per_case_evidence(tmp_path: Path) -> None:
    output = tmp_path / "routing-report.json"
    assert main(["--limit", "12", "--report", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["total"] == 155
    assert payload["cases"] == 12
    assert payload["records"]
    record = payload["records"][0]
    assert set(record["decision"]) >= {"question", "intent", "needs_knowledge", "freshness"}
    assert "scope_check" in record
    assert "planned_tools_not_executed" in record
    assert record["prompt_evidence"]["model_used"] is False


def test_split_filter_limits_evaluation(dataset: dict) -> None:
    dev = run(dataset, catalog=None, split="dev", limit=None)
    holdout = run(dataset, catalog=None, split="holdout", limit=None)
    assert dev["cases"] + holdout["cases"] == 155
    assert dev["cases"] != holdout["cases"]


def test_catalog_mode_consumes_real_workspace_catalog(
    tmp_path: Path, dataset: dict
) -> None:
    """`--catalog` 模式：用真实工作区导出的目录替代离线线索代理（T_verify 复用点）。"""

    from app.generation.knowledge_catalog import build_knowledge_catalog

    blocks = [
        (
            "b1",
            "内部风控 P99 方案：P99 表示第 99 百分位延迟。",
            {
                "source_file": r"C:\eval\风控方案.md",
                "chunk_type": "text",
                "source_type": "document",
                "heading_path": "风控 > P99 方案",
            },
        )
    ]
    catalog = build_knowledge_catalog(blocks, workspace_id="ws-eval", knowledge_revision=3)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(catalog.model_dump_json(), encoding="utf-8")

    report_path = tmp_path / "report.json"
    assert main(["--catalog", str(catalog_path), "--limit", "8", "--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["catalog"] == "real-catalog"
    assert payload["records"] and len(payload["records"]) == 8
    assert payload["dataset"]["total"] == 155


def test_catalog_mode_reports_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--catalog", str(tmp_path / "missing.json"), "--limit", "1"]) == 2
    assert "目录文件不存在" in capsys.readouterr().err
