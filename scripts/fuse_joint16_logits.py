"""Join two independent Jeff K16 decisions into one disease-syndrome prediction.

Reads only frozen raw logits and candidate metadata: no 31B forward, no retraining.
Unconstrained product and Train-seen pair constraint are both reported, so
unseen pair exclusions cannot masquerade as a clinical improvement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from tcmdiscern.datasets import load_records
from tcmdiscern.disease_study import disease_name, pair_key, pair_parts, syndrome_name
from tcmdiscern.joint_fusion import decode

ROOT = Path(__file__).resolve().parents[1]


def read_logits(path: Path, *, n: int, split: str) -> list[list[float]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            record = json.loads(line)
            if (
                record.get("case_id") != f"{split}-{i:06d}"
                or set(record) != {"case_id", "logits"}
                or len(record["logits"]) != 16
                or any(not math.isfinite(float(value)) for value in record["logits"])
            ):
                raise ValueError(f"Unaligned raw model logits at {path}:{i}")
            rows.append([float(v) for v in record["logits"]])
    if len(rows) != n:
        raise ValueError(f"Incomplete model output: {path}")
    return rows


def read_candidates(path: Path, *, n: int, split: str, disease: bool) -> list[list[str]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            item = json.loads(line)
            options = item["candidates"] if disease else item["candidate_labels"][:16]
            if (
                item.get("case_id") != f"{split}-{i:06d}"
                or len(options) != 16
                or len(set(options)) != 16
            ):
                raise ValueError(f"Unaligned label candidate file: {path}:{i}")
            rows.append(options)
    if len(rows) != n:
        raise ValueError(f"Incomplete candidate labels: {path}")
    return rows


def score_fusion(
    train: list[dict], cases: list[dict], predictions: list[str],
    *, fallback_count: int, constrained: bool,
) -> dict:
    if len(cases) != len(predictions):
        raise ValueError("Prediction count mismatch")
    truth = [pair_key(disease_name(case), syndrome_name(case)) for case in cases]
    known = frozenset(pair_key(disease_name(case), syndrome_name(case)) for case in train)
    labels = sorted(known | set(truth))
    gold_disease = [pair_parts(p)[0] for p in truth]
    gold_syndrome = [pair_parts(p)[1] for p in truth]
    pred_disease = [pair_parts(p)[0] for p in predictions]
    pred_syndrome = [pair_parts(p)[1] for p in predictions]
    return {
        "n_cases": len(cases),
        "label_classes": len(labels),
        "pair_top1": float(accuracy_score(truth, predictions)),
        "pair_macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "disease_top1": float(accuracy_score(gold_disease, pred_disease)),
        "syndrome_top1": float(accuracy_score(gold_syndrome, pred_syndrome)),
        "model_forward_passes_per_case": 2,
        "train_pair_constraint": constrained,
        "predicted_train_unseen_pair_count": sum(p not in known for p in predictions),
        "gold_train_unseen_pair_count": sum(p not in known for p in truth),
        "no_compatible_pair_fallback_count": fallback_count,
    }


def run(
    *,
    split: str, disease_logits: Path, syndrome_logits: Path,
    syndrome_candidates: Path, disease_weight: float, run_id: str,
) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("Official Dev/Test only")
    if split == "test" and not (ROOT / "experiments/joint16_dev_selection.json").exists():
        raise ValueError("Dev joint model selection must be frozen before Test")
    if not run_id.isascii() or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Safe ASCII run_id required")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    cases = load_records(ROOT / f"data/tcm-sd/{split}.json")
    if len(cases) != 5486:
        raise ValueError("Complete original official split required")
    base = ROOT / "data/joint16"
    disease_candidates = read_candidates(
        base / f"{split}_disease_labels.jsonl", n=len(cases),
        split=split, disease=True,
    )
    syndrome_choices = read_candidates(
        syndrome_candidates, n=len(cases), split=split, disease=False,
    )
    d_logits = read_logits(disease_logits, n=len(cases), split=split)
    s_logits = read_logits(syndrome_logits, n=len(cases), split=split)
    known_pairs = frozenset(pair_key(disease_name(c), syndrome_name(c)) for c in train)

    reports = {}
    for constrained in (False, True):
        predictions = []
        fallback = 0
        for d_labels, s_labels, dz, sz in zip(
            disease_candidates, syndrome_choices, d_logits, s_logits, strict=True
        ):
            prediction, used_fallback = decode(
                d_labels, s_labels, dz, sz,
                known_pairs=known_pairs if constrained else None,
                disease_weight=disease_weight,
            )
            predictions.append(prediction)
            fallback += used_fallback
        name = "train_constrained" if constrained else "unconstrained"
        reports[name] = score_fusion(
            train, cases, predictions,
            fallback_count=fallback, constrained=constrained,
        )
        output = base / f"{split}_factorized_{run_id}_{name}_predictions.jsonl"
        with output.open("x", encoding="utf-8") as stream:
            for i, p in enumerate(predictions):
                stream.write(json.dumps(
                    {"case_id": f"{split}-{i:06d}", "prediction": p},
                    ensure_ascii=False,
                ) + "\n")
    result = {
        "status": "FACTORIZED_TWO_JEFF_DECISIONS",
        "split": split, "run_id": run_id,
        "disease_logit_source": str(disease_logits),
        "syndrome_logit_source": str(syndrome_logits),
        "disease_weight": disease_weight,
        "results": reports,
        "gold_disease_used_for_joint_prediction": False,
    }
    destination = ROOT / "experiments" / f"joint16_{split}_factorized_{run_id}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--disease-logits", type=Path, required=True)
    parser.add_argument("--syndrome-logits", type=Path, required=True)
    parser.add_argument("--syndrome-candidates", type=Path,
                        default=ROOT / "data/candidates/bm25_dev.jsonl")
    parser.add_argument("--disease-weight", type=float, default=0.5)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run(
        split=args.split, disease_logits=args.disease_logits,
        syndrome_logits=args.syndrome_logits,
        syndrome_candidates=args.syndrome_candidates,
        disease_weight=args.disease_weight, run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
