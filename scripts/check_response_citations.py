import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--exclude-cases", nargs="*", default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    from app.agent.contracts import Evidence
    from app.agent.verification import SynthesisReview, bind_review_paragraphs

    environment = json.loads((args.results_dir / "environment.json").read_text(encoding="utf-8"))
    records = json.loads((args.results_dir / "results.json").read_text(encoding="utf-8"))
    artifacts = Path(environment["data_copy"]) / "workspaces" / "default" / "artifacts" / "synthesis"
    results = []
    for record in records:
        if record["case"] in args.exclude_cases:
            continue
        if not record["answer"] or not (record["answer_meta"] or {}).get("citations"):
            continue
        directory = artifacts / record["state"]["run_id"]
        accepted = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("review-*.json")]
        accepted = [entry for entry in accepted if entry.get("accepted") is True]
        if len(accepted) != 1:
            raise ValueError(f"{record['case']}缺少唯一的引用验收记录")
        entry = accepted[0]
        review = SynthesisReview.model_validate(entry["review"])
        evidence = [Evidence.model_validate(item) for item in entry["evidence"]]
        bind_review_paragraphs(review, evidence, question=record["question"])
        results.append({
            "case": record["case"], "repeat": record["repeat"], "turn": record["turn"],
            "paragraphs": len(review.paragraphs),
            "citation_spans": sum(len(paragraph.citations) for paragraph in review.paragraphs),
            "source_ids": sorted({span.evidence_id for paragraph in review.paragraphs for span in paragraph.citations}),
            "passed": True,
        })
    output = args.results_dir / "citation-audit.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"answers": len(results), "citation_spans": sum(row["citation_spans"] for row in results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
