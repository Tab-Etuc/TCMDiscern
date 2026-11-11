"""Write original TCM-SD Dev cases as Jeff decision rows (no gold in prompt).

This is inference input preparation only. Test is not read. For K>16,
Jeff.Decider.decide_wide must group the options (not semif_messages).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_format import decision_row


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export TCM-SD Dev decisions in Jeff format")
    parser.add_argument("--k", type=int, default=5, help="2..35 (K=1 uses BM25 directly)")
    parser.add_argument("--limit", type=int, default=None, help="Smoke tests only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 2 <= args.k <= 35:
        parser.error("K must be 2..35; for K=1 choose the BM25 rank-1 result")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    dev = load_records(project_root / "data/tcm-sd/dev.json")
    predictions = read_predictions(project_root / "data/candidates/bm25_dev.jsonl")
    if len(dev) != len(predictions) or len(dev) != 5486:
        raise ValueError("Requires the complete original 5,486-case Dev and BM25 rankings")
    basename = f"dev_k{args.k:02d}"
    if args.limit is not None:
        basename += f"_smoke_{args.limit}"
    output_path = args.output or project_root / "data/jeff" / f"{basename}.jsonl"
    if not output_path.resolve().is_relative_to((project_root / "data").resolve()):
        parser.error("Clinical data outputs must stay under the git-ignored data/ directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate the full selected subset *before* creating the output file.
    # This prevents a mismatched gold/candidate row from leaving a partial export.
    serialized_rows: list[str] = []
    for index, (case, result) in enumerate(zip(dev, predictions, strict=True)):
        gold = str(case["norm_syndrome"])
        if result.case_id != f"dev-{index:06d}" or result.gold_labels != (gold,):
            raise ValueError(f"Official Dev and BM25 rankings disagree at row {index}")
        if len(result.candidate_labels) < args.k:
            raise ValueError(f"BM25 candidate list too short at row {index}")
        row = decision_row(
            case,
            case_id=result.case_id,
            candidates=result.candidate_labels[: args.k],
            allow_grouped=args.k > 16,
        )
        serialized_rows.append(json.dumps(row, ensure_ascii=False))
        if args.limit is not None and len(serialized_rows) >= args.limit:
            break
    with output_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(serialized_rows) + "\n")
    exported = len(serialized_rows)
    print(
        json.dumps(
            {
                "split": "official_dev",
                "k": args.k,
                "exported": exported,
                "gold_labels_in_prompt": False,
                "format": "Jeff/SemIf id,state,question,options",
                "requires_grouped_decision": args.k > 16,
                "path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
