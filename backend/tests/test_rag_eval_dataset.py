"""检索黄金集契约测试。

这些测试只保证每条回归样例都可独立执行、期望来源足够精确；真实命中率由
``scripts/eval_rag_quality.py`` 通过生产 ``HybridRetriever`` 计算，二者不能混为一谈。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.rag_quality import load_eval_cases

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "scripts" / "rag_eval_cases.json"
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
POSITIVE_CASES = DATASET["positive_cases"]
NEGATIVE_CASES = DATASET["negative_cases"]

#: 形如 ``…V2.0.1`` 的版本尾巴：用于断言「两版本只差版本号」，不暴露文档名本身。
_VERSION_RE = re.compile(r"^(?P<base>.*?)[ _-]?[vV]?(?P<version>\d+(?:[._-]\d+)+)$")


def _version_match(stem: str) -> re.Match[str]:
    match = _VERSION_RE.match(stem)
    assert match, f"多版本文档名必须带可解析的版本号：{stem!r}"
    return match


def _version_base(stem: str) -> str:
    return _version_match(stem).group("base")


def _version_suffix(stem: str) -> int:
    version = _version_match(stem).group("version")
    return int(re.split(r"[._-]", version)[-1])


def _case_id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _targets(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "acceptable_targets" in case:
        return case["acceptable_targets"]
    if "required_targets" in case:
        return case["required_targets"]
    return [
        {
            "source": case["expected_source"],
            "heading": case["expected_heading"],
            "required_evidence": case["required_evidence"],
        }
    ]


def test_eval_dataset_has_at_least_fifty_positive_cases() -> None:
    assert len(POSITIVE_CASES) >= 50


def test_version_helpers_are_not_vacuous() -> None:
    """多版本断言依赖的解析必须真的会挑错（避免「换成结构断言后空转」）。"""

    assert _version_base("设计文档V2.0") == "设计文档"
    assert _version_base("设计文档V2.0.1") == "设计文档"
    assert _version_suffix("设计文档V2.0.1") == 1
    assert _version_suffix("设计文档V2.0.2") == 2
    with pytest.raises(AssertionError):
        _version_base("没有版本号的文档")


def test_eval_dataset_uses_strict_schema_and_loads_as_independent_cases() -> None:
    positives, negatives = load_eval_cases(DATASET_PATH)

    assert set(DATASET) == {"schema_version", "positive_cases", "negative_cases"}
    assert DATASET["schema_version"] == 3
    assert len(positives) == len(POSITIVE_CASES) == 72
    assert len(negatives) == len(NEGATIVE_CASES) == 10


def test_eval_dataset_ids_are_unique() -> None:
    ids = [case["id"] for case in [*POSITIVE_CASES, *NEGATIVE_CASES]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=_case_id)
def test_each_positive_case_has_block_level_expectation(case: dict[str, Any]) -> None:
    """每个正例必须同时约束文件和标题，防止“命中文档但答错章节”虚高。"""

    assert case["query"].strip()
    assert case["category"].strip()
    modes = sum(
        (
            "expected_source" in case or "expected_heading" in case,
            "acceptable_targets" in case,
            "required_targets" in case,
        )
    )
    assert modes == 1
    for target in _targets(case):
        assert set(target) == {"source", "heading", "required_evidence"}
        assert target["source"].strip()
        assert Path(target["source"]).name == target["source"]
        assert "/" not in target["source"] and "\\" not in target["source"]
        assert target["heading"].strip()
        assert target["required_evidence"]
        assert all(
            isinstance(evidence, str) and evidence.strip()
            for evidence in target["required_evidence"]
        )


def test_tracking_versions_and_compound_answer_are_explicit() -> None:
    """多版本文档必须显式登记为等价目标；复合答案必须逐条给出目标。

    具体文件/标题属于私人语料身份（PRD §7.2 不落仓库），这里断言**结构不变式**：
    - `tracking-001`：两个 acceptable target，同一标题、不同来源，且来源只是版本号不同；
    - `tracking-011`：两个 required target，标题互不相同（复合答案分属两处）。
    """

    by_id = {case["id"]: case for case in POSITIVE_CASES}

    versions = by_id["tracking-001"]["acceptable_targets"]
    assert len(versions) == 2, "多版本文档必须显式列出两个等价版本"
    assert len({target["heading"] for target in versions}) == 1, "两版本必须同一章节标题"
    stems = {Path(target["source"]).stem for target in versions}
    assert len(stems) == 2, "两版本必须来自不同文件"
    bases = {_version_base(stem) for stem in stems}
    assert len(bases) == 1, "两版本文档名必须只差版本号"
    assert {_version_suffix(stem) for stem in stems} == {1, 2}

    compound = by_id["tracking-011"]["required_targets"]
    assert len(compound) == 2
    headings = {target["heading"] for target in compound}
    assert len(headings) == 2, "复合答案必须分属两个不同标题"


def test_original_colloquial_cart_question_is_an_independent_case() -> None:
    """原始口语化问题必须是独立案例，且期望来源/标题/证据齐备。"""

    by_id = {case["id"]: case for case in POSITIVE_CASES}
    case = by_id["cart-020"]

    assert case["query"].strip()
    assert _case_id(case) == "cart-020"
    assert Path(case["expected_source"]).name == case["expected_source"]
    assert case["expected_heading"].strip()
    assert case["required_evidence"]
    assert any("cider" in evidence.lower() for evidence in case["required_evidence"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(description="不允许的顶层字段"), "顶层"),
        (lambda payload: payload.update(positive_cases=payload["positive_cases"][:49]), "正例不足"),
        (lambda payload: payload.update(negative_cases=payload["negative_cases"][:9]), "负例不足"),
    ],
)
def test_loader_rejects_schema_drift_and_insufficient_negatives(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_eval_cases(path)


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=_case_id)
def test_each_negative_case_has_no_leaked_positive_expectation(case: dict[str, Any]) -> None:
    """负例只描述问题，不能意外携带答案目标而弱化越界召回检查。"""

    assert case["query"].strip()
    assert "expected_source" not in case
    assert "expected_heading" not in case
    assert "acceptable_targets" not in case
    assert "required_targets" not in case
