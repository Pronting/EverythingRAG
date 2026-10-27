import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    from app.agent.runtime import _merge_source_scope_subjects
    from app.agent.verification import SourceScopeReview

    checked = []
    for path in args.artifacts.rglob("review-*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("stage") != "source_scope_initial":
            continue
        original = SourceScopeReview.model_validate_json(record["raw"])
        merged = _merge_source_scope_subjects(original.model_copy(deep=True))
        assert {key for row in original.source_scope for key in row.evidence_ids} == {
            key for row in merged.source_scope for key in row.evidence_ids
        }
        for row in original.source_scope:
            target = next(item for item in merged.source_scope
                          if " ".join(item.subject.split()).casefold() == " ".join(row.subject.split()).casefold())
            assert set(row.requested_dimensions) <= set(target.requested_dimensions)
            assert row.documented_operation in target.documented_operation
            if row.matches_request:
                assert set(row.provided_dimensions) <= set(target.provided_dimensions)
        checked.append({"artifact": str(path), "before": len(original.source_scope),
                        "after": len(merged.source_scope), "evidence_retained": True})
    if not checked:
        raise ValueError("没有找到真实来源核对记录")
    (args.artifacts / "scope-merge-audit.json").write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checked, ensure_ascii=False))


if __name__ == "__main__":
    main()
