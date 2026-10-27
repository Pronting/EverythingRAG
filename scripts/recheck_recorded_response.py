import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-artifacts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    from app.agent.contracts import Evidence
    from app.agent.runtime import _structured_json
    from app.agent.verification import (
        ParagraphCorrections,
        ParagraphSynthesisReview,
        SourceScopeReview,
        SynthesisReview,
        apply_paragraph_corrections,
        bind_review_paragraphs,
    )

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_artifacts.glob("review-*.json")]
    failed = sorted((row for row in records if row.get("accepted") is False), key=lambda row: row["attempt"])
    initial = failed[0]
    review = SynthesisReview.model_validate(
        ParagraphSynthesisReview.model_validate_json(_structured_json(initial["raw"])).model_dump(),
    )
    scope = next(row for row in records if row.get("stage") == "source_scope")
    review.source_scope = SourceScopeReview.model_validate_json(scope["raw"]).source_scope
    for row in failed[1:]:
        corrections = ParagraphCorrections.model_validate_json(_structured_json(row["raw"]))
        review = apply_paragraph_corrections(
            review, corrections, {item.index for item in corrections.corrections}, set(initial["requirements"]),
        )
    evidence = [Evidence.model_validate(item) for item in initial["evidence"]]
    content = bind_review_paragraphs(review, evidence, question=initial["question"])
    output = {"passed": True, "paragraphs": len(review.paragraphs), "content": content,
              "review": review.model_dump(mode="json"), "input_directory": str(args.run_artifacts.resolve())}
    (args.run_artifacts / "rechecked-review.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "paragraphs": len(review.paragraphs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
