"""Audit the zero-retraining Case-RAG pilot without running Jeff.

All retrieval records must come from the official TCM-SD Train. Only summary
statistics (no patient text or individual predictions) are published.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from tcmdiscern.case_rag import CONDITIONS, CONDITIONS_2, _view, transitive_group_ids
from tcmdiscern.case_rag_excerpts import focused_case_view, verify_extractive
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records, normalized_input
from tcmdiscern.jeff_scoring import read_model_logits
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "pilotclean_0300_20261116"
OUTPUT = ROOT / "experiments/case_rag_input_audit_2026-11-16.json"


def jaccard(a: dict, b: dict) -> float:
    text_a = " ".join(str(a.get(k) or "") for k in CLINICAL_FIELDS)
    text_b = " ".join(str(b.get(k) or "") for k in CLINICAL_FIELDS)
    aa = set(chinese_bigrams(text_a))
    bb = set(chinese_bigrams(text_b))
    return len(aa & bb) / len(aa | bb) if (aa or bb) else 0.0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(tag: str = DEFAULT_TAG) -> dict:
    folder = ROOT / "data/case_rag" / tag
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    arms = tuple(manifest["conditions"])
    if arms not in (CONDITIONS, CONDITIONS_2):
        raise ValueError("Unexpected Case-RAG condition definitions")
    budget = 2 if arms == CONDITIONS_2 else 4
    if manifest.get("evidence_budget", budget) != budget:
        raise ValueError("Evidence-budget mismatch")
    n = int(manifest["samples"])
    data = {condition: jsonl(folder / f"{condition}.jsonl") for condition in arms}
    provenance = jsonl(folder / "provenance.jsonl")
    if len(provenance) != n or any(len(data[c]) != n for c in arms):
        raise ValueError("Missing or duplicated pilot rows")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if (len(train), len(dev)) != (43180, 5486):
        raise ValueError("Original TCM-SD split counts changed")
    training_groups = transitive_group_ids(train)
    group_labels: dict[int, set[str]] = {}
    for row, group in zip(train, training_groups, strict=True):
        group_labels.setdefault(group, set()).add(str(row["norm_syndrome"]))
    frozen = {
        row["id"]: row for row in jsonl(ROOT / "data/jeff/dev_k16.jsonl")
    }
    if len(frozen) != len(dev):
        raise ValueError("Frozen Jeff Dev inputs are incomplete")
    baseline_logits = read_model_logits(
        ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl",
        num_cases=len(dev),
    )
    exact_train = {clinical_text(row) for row in train}
    counters = Counter()
    condition_stats: dict[str, Counter] = {c: Counter() for c in arms if c != "baseline"}
    condition_scores: dict[str, list[float]] = {c: [] for c in condition_stats}
    pair_similarity: dict[str, list[float]] = {
        name: [] for name in (f"balanced{budget}", f"paired{budget}", f"random{budget}")
    }
    ids_seen: set[str] = set()
    for i in range(n):
        original = data["baseline"][i]
        cid = original["id"]
        if cid in ids_seen:
            raise ValueError("Duplicate case ID")
        ids_seen.add(cid)
        expected = frozen.get(cid)
        if expected is None or original != expected:
            raise ValueError(f"Baseline prompt mismatches published Jeff input at {cid}")
        if provenance[i]["case_id"] != cid:
            raise ValueError("Provenance case order differs")
        index = int(cid.removeprefix("dev-"))
        if cid != f"dev-{index:06d}":
            raise ValueError("Invalid original Dev ID")
        gold = str(dev[index]["norm_syndrome"])
        options = [opt["id"] for opt in original["options"]]
        if len(options) != 16 or len(set(options)) != 16:
            raise ValueError("Incorrect BM25 Top-16 candidate count")
        if gold in options[:2]:
            counters["gold_in_top2"] += 1
        if gold in options:
            counters["gold_in_top16"] += 1
        if clinical_text(dev[index]) in exact_train:
            counters["exact_overlap_queries"] += 1
        baseline_prediction = options[int(np.argmax(baseline_logits[index]))]
        counters["bm25_baseline_correct"] += options[0] == gold
        counters["jeff_s2_saved_correct"] += baseline_prediction == gold
        counters["jeff_s2_saved_wrong_gold_top2"] += (
            baseline_prediction != gold and gold in options[:2]
        )
        counters["jeff_s2_saved_wrong_gold_3to16"] += (
            baseline_prediction != gold and gold in options[2:]
        )
        counters["paired_differs_from_balanced"] += (
            provenance[i]["conditions"][f"paired{budget}"]["source_train_positions"]
            != provenance[i]["conditions"][f"balanced{budget}"]["source_train_positions"]
        )
        for condition in arms[1:]:
            row = data[condition][i]
            if (
                row["id"] != cid
                or row["options"] != original["options"]
                or row["question"] != original["question"]
                or row["qtype"] != original["qtype"]
            ):
                raise ValueError(f"{condition} changed the candidate decision contract")
            if set(row["state"]) != set(CLINICAL_FIELDS) | {
                "retrieval_note", "retrieved_cases"
            }:
                raise ValueError(f"{condition} introduced unexpected prompt fields")
            if any(row["state"][field] != original["state"][field] for field in CLINICAL_FIELDS):
                raise ValueError(f"{condition} modified the original clinical input")
            examples = row["state"]["retrieved_cases"]
            source = provenance[i]["conditions"][condition]
            positions = source["source_train_positions"]
            groups = source["support_groups"]
            labels = source["support_labels"]
            scores = source["support_scores"]
            if not all(len(x) == budget for x in (examples, positions, groups, labels, scores)):
                raise ValueError("RAG input does not match evidence budget")
            if len(set(groups)) != budget:
                raise ValueError("Dependent duplicated support group")
            if set(labels) - set(options):
                raise ValueError("RAG support is outside the fixed 16 candidates")
            if condition in pair_similarity:
                if Counter(labels) != Counter({options[0]: budget // 2, options[1]: budget // 2}):
                    raise ValueError("Top-2 condition has unbalanced evidence labels")
                pair_indices = tuple((j, j + 1) for j in range(0, budget, 2))
                for left, right in pair_indices:
                    if labels[left] != options[0] or labels[right] != options[1]:
                        raise ValueError("Top-2 comparison must alternate A/B")
                pair_similarity[condition].append(
                    mean(
                        jaccard(train[positions[left]], train[positions[right]])
                        for left, right in pair_indices
                    )
                )
            for ex, pos, group, label, score in zip(
                examples, positions, groups, labels, scores, strict=True
            ):
                if not 0 <= pos < len(train) or group != training_groups[pos]:
                    raise ValueError("Invalid Train support provenance")
                if len(group_labels[group]) != 1:
                    raise ValueError("Ambiguously annotated support used")
                if train[pos]["norm_syndrome"] != label:
                    raise ValueError("Retrieved label differs from Train gold")
                query_uid = str(dev[index].get("user_id") or "").strip()
                exemplar_uid = str(train[pos].get("user_id") or "").strip()
                if query_uid and query_uid == exemplar_uid:
                    raise ValueError("Same-patient support case leaked into query")
                if normalized_input(train[pos], full=True) == normalized_input(
                    dev[index], full=True,
                ):
                    raise ValueError("Identical patient evidence leaked into support cases")
                focused = manifest.get("evidence_style") == "single-contiguous-query-window"
                expected_view = (
                    focused_case_view(train[pos], dev[index]) if focused
                    else _view(train[pos])
                )
                if focused:
                    verify_extractive(train[pos], expected_view)
                if ex != {"candidate_syndrome": label, **expected_view}:
                    raise ValueError("Retrieved case has incorrect content or extra fields")
                if len(ex["chief_complaint"]) > 58 or len(ex["description"]) > 90:
                    raise ValueError("Truncation bounds exceeded")
                condition_scores[condition].append(float(score))
            condition_stats[condition]["queries_with_gold_support"] += gold in labels
            condition_stats[condition]["queries_with_original_top1_support"] += (
                options[0] in labels
            )
            condition_stats[condition]["queries_with_original_top2_support"] += (
                options[1] in labels
            )
            condition_stats[condition]["support_label_diversity_sum"] += len(set(labels))
            condition_stats[condition]["max_source_score"] += max(scores)
    summary = {
        "status": "INPUT_AUDITED_NO_RAG_JEFF_INFERENCE",
        "scope": "Official TCM-SD Train supports and existing original Dev 300-case prompt set",
        "samples": n,
        "evidence_budget": budget,
        "evidence_style": manifest.get("evidence_style", "prefix"),
        "sample_seed": manifest["seed"],
        "paired_vs_balanced_source_changed_n": counters["paired_differs_from_balanced"],
        "paired_vs_balanced_source_changed_percent": round(100 * counters["paired_differs_from_balanced"] / n, 4),
        "gold_in_bm25_top2": counters["gold_in_top2"],
        "gold_in_bm25_top16": counters["gold_in_top16"],
        "same_text_as_train_samples": counters["exact_overlap_queries"],
        "saved_existing_jeff_s2_baseline_correct": counters["jeff_s2_saved_correct"],
        "saved_existing_jeff_s2_baseline_top1_percent": round(
            100 * counters["jeff_s2_saved_correct"] / n, 4
        ),
        "saved_existing_bm25_baseline_correct": counters["bm25_baseline_correct"],
        "saved_jeff_s2_wrong_but_gold_top2": counters["jeff_s2_saved_wrong_gold_top2"],
        "saved_jeff_s2_wrong_but_gold_rank3to16": counters[
            "jeff_s2_saved_wrong_gold_3to16"
        ],
        "support_groups_distinct": True,
        "support_gold_from_original_train": True,
        "baseline_equals_frozen_original_jeff_input": True,
        "same_decision_candidates_all_conditions": True,
        "no_prompt_gold_injection": True,
        "condition_summaries": {
            condition: {
                "gold_support_rate_percent": round(
                    100 * stats["queries_with_gold_support"] / n, 4
                ),
                "top1_support_rate_percent": round(
                    100 * stats["queries_with_original_top1_support"] / n, 4
                ),
                "top2_support_rate_percent": round(
                    100 * stats["queries_with_original_top2_support"] / n, 4
                ),
                "mean_distinct_support_labels": round(
                    stats["support_label_diversity_sum"] / n, 4
                ),
                "mean_train_support_bm25": round(
                    mean(condition_scores[condition]), 4
                ),
                **(
                    {"mean_cross_label_pair_jaccard": round(
                        mean(pair_similarity[condition]), 6
                    )}
                    if condition in pair_similarity else {}
                ),
                "mean_prompt_tokens": manifest["input_tokens"][condition]["mean"],
            }
            for condition, stats in condition_stats.items()
        },
        "mean_baseline_prompt_tokens": manifest["input_tokens"]["baseline"]["mean"],
        "input_token_limit_verified_by_preparation": manifest["all_within_8192"],
        "official_dev_score_comparison_possible": False,
        "test_used": False,
    }
    if manifest.get("evidence_style") == "single-contiguous-query-window":
        destination = ROOT / f"experiments/case_rag_focused{budget}_input_audit_2026-11-16.json"
    else:
        destination = (
            OUTPUT if budget == 4
            else ROOT / "experiments/case_rag_budget2_input_audit_2026-11-16.json"
        )
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    arguments = parser.parse_args()
    run(tag=arguments.tag)
