import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


def identity(record):
    return record["case"], record["repeat"], record["turn"]


def phase_results(directory, overrides=()):
    records = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    quality = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    case_overrides = []
    for override in overrides:
        replacement = json.loads((override / "results.json").read_text(encoding="utf-8"))
        replacement_quality = json.loads((override / "quality.json").read_text(encoding="utf-8"))
        replacement_environment = json.loads((override / "environment.json").read_text(encoding="utf-8"))
        if any(environment[name] != replacement_environment[name] for name in ["config_sha256", "data_source"]):
            raise ValueError("定向回归使用了不同设置或知识库")
        cases = {row["case"] for row in replacement}
        original = {identity(row): row["question"] for row in records if row["case"] in cases}
        if not original or original != {identity(row): row["question"] for row in replacement}:
            raise ValueError("定向回归没有覆盖该题全部测试次数与原始问题")
        if {identity(row) for row in replacement_quality} != set(original):
            raise ValueError("定向回归缺少逐项质量验收")
        records = [row for row in records if row["case"] not in cases] + replacement
        quality = [row for row in quality if row["case"] not in cases] + replacement_quality
        case_overrides.append({"directory": str(override.resolve()), "cases": sorted(cases),
                               "environment": replacement_environment})
    indexed = {identity(record): record for record in quality}
    if len(indexed) != len(quality) or set(indexed) != {identity(record) for record in records}:
        raise ValueError("质量记录与速度记录没有逐项对应")
    groups = defaultdict(list)
    for record in records:
        groups[record["case"]].append(record)
    cases = {}
    for name, rows in groups.items():
        times = [row["seconds"] for row in rows]
        target = 60 if name == "comparison" else 20
        reviews = [indexed[identity(row)] for row in rows]
        cases[name] = {
            "runs": len(rows), "median_seconds": round(median(times), 3),
            "min_seconds": min(times), "max_seconds": max(times),
            "published": sum(bool(row["answer"]) for row in rows),
            "quality_passed": sum(row["complete"] for row in reviews),
            "within_target": sum(row["seconds"] <= target for row in rows),
            "quality_passed_within_target": sum(
                row["seconds"] <= target and indexed[identity(row)]["complete"] for row in rows
            ),
            "target_seconds": target,
            "requirements": {
                status: sum(
                    item["status"] == status
                    for review in reviews if review["review"]
                    for item in review["review"]["requirements"]
                ) for status in ["answered", "legitimate_gap", "missing", "incorrect"]
            },
            "citation_errors": sum(len(row["review"]["citation_errors"]) for row in reviews if row["review"]),
            "factual_errors": sum(len(row["review"]["factual_errors"]) for row in reviews if row["review"]),
            "unsupported_absences": sum(
                len(row["review"]["unsupported_absences"]) for row in reviews if row["review"]
            ),
        }
    calls = defaultdict(list)
    for row in records:
        for call in row["model_calls"]:
            if call["total_seconds"] is not None:
                calls[call["stage"]].append(call["total_seconds"])
    return {
        "environment": {**environment, "case_overrides": case_overrides},
        "runs": len(records), "published": sum(bool(row["answer"]) for row in records),
        "quality_passed": sum(row["complete"] for row in quality),
        "cases": cases,
        "model_calls": {
            stage: {"count": len(times), "median_seconds": round(median(times), 3), "max_seconds": max(times)}
            for stage, times in calls.items()
        },
    }, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--after-case-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, before_rows = phase_results(args.before)
    after, after_rows = phase_results(args.after, args.after_case_dir)
    if before["environment"]["config_sha256"] != after["environment"]["config_sha256"]:
        raise ValueError("前后测试模型设置不同")
    if before["environment"]["data_source"] != after["environment"]["data_source"]:
        raise ValueError("前后测试知识库快照不同")
    before_index = {identity(record): record for record in before_rows}
    after_index = {identity(record): record for record in after_rows}
    if before_index.keys() != after_index.keys() or any(
        before_index[key]["question"] != after_index[key]["question"] for key in before_index
    ):
        raise ValueError("前后测试问题或执行次数不同")
    comparison = {
        name: {
            "before": before["cases"][name], "after": after["cases"][name],
            "median_reduction_percent": round(100 * (
                1 - after["cases"][name]["median_seconds"] / before["cases"][name]["median_seconds"]
            ), 1),
        } for name in before["cases"]
    }
    output = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "before": before, "after": after, "comparison": comparison,
        "limits": ["每个问题执行两轮，结果表示本次真实服务表现，不代表长期百分位耗时。", "质量验收保留独立模型的逐项判断与原始回答，须结合原文复查争议项。"],
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in comparison.items():
        print(json.dumps({"case": name, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
