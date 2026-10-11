import json
from pathlib import Path

from app.agent.research import MaterialGap
from app.agent.verification import render_gap_section


BASE = Path(__file__).resolve().parents[2] / ".tmp_probe" / "agent-eval40-20260930" / "results-10001"


def test_real_comparison_gap_has_readable_scope_and_single_punctuation():
    record = json.loads((BASE / "Q10.json").read_text(encoding="utf-8"))
    completed = next(event["payload"] for event in record["events"] if event["type"] == "run.completed")
    gaps = [MaterialGap(missing=missing) for missing in completed["gaps"]]
    rendered = render_gap_section(gaps, {})
    assert "原文场景3仅描述到 Policy Engine 返回 ALLOW 决策与自动完成退款审批" in rendered
    assert "未描述该决策之后另行调用的工具" in rendered
    assert "confirmed_source_scope" not in rendered
    assert "provided_dimensions" not in rendered
    assert "。。" not in rendered


def test_real_missing_table_gap_remains_explicit():
    record = json.loads((BASE / "Q11.json").read_text(encoding="utf-8"))
    completed = next(event["payload"] for event in record["events"] if event["type"] == "run.completed")
    rendered = render_gap_section([MaterialGap(missing=missing) for missing in completed["gaps"]], {})
    assert "四类配置" in rendered
    assert "未" in rendered
    assert "。。" not in rendered
