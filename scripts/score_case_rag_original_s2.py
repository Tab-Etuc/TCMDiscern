"""Score the immutable original Jeff S2 logits on the fixed Case-RAG pilot IDs."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_case_rag_pilot import load_candidates, make_selection
from sklearn.metrics import f1_score

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_scoring import read_model_logits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl"


def main() -> None:
    selected = make_selection(300, 20261116)
    raw = read_model_logits(SOURCE, num_cases=5486)
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    candidate_lists = load_candidates(ROOT / "data/candidates/bm25_dev.jsonl")
    labels = sorted({str(r["norm_syndrome"]) for r in train})
    observed = [str(dev[i]["norm_syndrome"]) for i in selected]
    bm25 = [candidate_lists[i][0] for i in selected]
    s2 = [candidate_lists[i][int(raw[i].argmax())] for i in selected]
    overlap = {clinical_text(r) for r in train}
    unseen = [j for j, i in enumerate(selected) if clinical_text(dev[i]) not in overlap]
    report = {
        "scope": "Original, previously frozen real Jeff S2 logits; no Case-RAG inference yet",
        "sample": 300,
        "selection_seed": 20261116,
        "sample_no_train_exact_text_overlap": len(unseen),
        "bm25_top1_percent": round(
            100 * sum(g == p for g, p in zip(observed, bm25, strict=True))/300, 4
        ),
        "jeff_s2_raw_top1_percent": round(
            100 * sum(g == p for g, p in zip(observed, s2, strict=True))/300, 4
        ),
        "jeff_s2_raw_macro_f1_148_percent": round(
            100 * float(f1_score(observed, s2, labels=labels,
                                 average="macro", zero_division=0)), 4
        ),
        "jeff_s2_raw_no_train_overlap_top1_percent": round(
            100 * sum(observed[j] == s2[j] for j in unseen)/len(unseen), 4
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "new_condition_scores": "Not yet measured; requires original H200 checkpoint",
        "test_used": False,
    }
    dest = ROOT / "experiments/case_rag_jeff_s2_original_sample_2026-11-16.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
