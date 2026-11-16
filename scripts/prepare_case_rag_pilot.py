"""Prepare auditable clinical-case retrieval inputs for Jeff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from tcmdiscern.case_rag import CONDITIONS, CONDITIONS_2, CaseRAGIndex
from tcmdiscern.case_rag_tokenizer import load_jeff_token_counter
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_format import semif_messages

ROOT = Path(__file__).resolve().parents[1]


def load_candidates(path: Path) -> list[list[str]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            value = json.loads(line)
            if value["case_id"] != f"dev-{i:06d}":
                raise ValueError("Dev candidates out of original order")
            items = value["candidate_labels"][:16]
            if len(items) != 16 or len(set(items)) != 16:
                raise ValueError("Invalid original BM25 candidates")
            result.append(items)
    return result


def make_selection(size: int, seed: int) -> list[int]:
    if not 1 <= size <= 5486:
        raise ValueError("Size must be within original Dev split")
    return sorted(Random(seed).sample(range(5486), size))


def run(size: int, seed: int, tag: str, *, evidence_budget: int = 4) -> dict:
    if evidence_budget not in (2, 4):
        raise ValueError("Evidence budget must be two or four cases")
    conditions = CONDITIONS if evidence_budget == 4 else CONDITIONS_2

    started = perf_counter()
    selected = make_selection(size, seed)
    if not tag.isascii() or not tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Unsafe result tag")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if (len(train), len(dev)) != (43180, 5486):
        raise ValueError("Unexpected original Train/Dev counts")
    candidates = load_candidates(ROOT / "data/candidates/bm25_dev.jsonl")
    if len(candidates) != len(dev):
        raise ValueError("Missing original BM25 candidates")
    index = CaseRAGIndex(train)
    generated = index.model.rank([clinical_text(dev[i]) for i in selected], max_k=16)
    if any(generated[j] != candidates[i] for j, i in enumerate(selected)):
        raise ValueError("BM25 candidate parity failed")
    output = ROOT / "data/case_rag" / f"{tag}_{size:04d}_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    streams = {
        name: (output / f"{name}.jsonl").open("x", encoding="utf-8")
        for name in conditions
    }
    trace = (output / "provenance.jsonl").open("x", encoding="utf-8")
    lengths: dict[str, list[int]] = {name: [] for name in conditions}
    original_texts = {clinical_text(row) for row in train}
    duplicate_count = 0
    tok, _ = load_jeff_token_counter(ROOT)
    try:
        for count, i in enumerate(selected, 1):
            row = dev[i]
            options = candidates[i]
            pool = index._supports(row, options)
            duplicate_count += int(clinical_text(row) in original_texts)
            provenance = {"case_id": f"dev-{i:06d}", "conditions": {}}
            for condition in conditions:
                prompt, detail = index.make_row(
                    row, options, case_id=f"dev-{i:06d}",
                    condition=condition, seed=seed + i, precomputed_items=pool,
                )
                if [item["id"] for item in prompt["options"]] != options:
                    raise AssertionError("RAG changed the original candidate order")
                n = len(tok.apply_chat_template(
                    semif_messages(prompt),
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ))
                if n > 8192:
                    raise ValueError("Jeff maximum input exceeded")
                lengths[condition].append(n)
                detail["input_tokens"] = n
                provenance["conditions"][condition] = detail
                streams[condition].write(json.dumps(prompt, ensure_ascii=False) + "\n")
            trace.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            if count % 50 == 0 or count == size:
                print(f"[case-rag] {count}/{size}", flush=True)
    finally:
        for handle in streams.values():
            handle.close()
        trace.close()
    with (ROOT / "data/jeff/dev_k16.jsonl").open(encoding="utf-8") as source:
        frozen = {item["id"]: item for item in (json.loads(line) for line in source)}
    with (output / "baseline.jsonl").open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item != frozen[item["id"]]:
                raise AssertionError("Baseline row differs from official Jeff input")
    report = {
        "study": "Differential Case-RAG zero-retraining input pilot",
        "status": "PROMPTS_READY_NO_NEW_JEFF_SCORES",
        "samples": size,
        "seed": seed,
        "train_rows": len(train),
        "original_dev_rows": len(dev),
        "duplicate_train_text_samples": duplicate_count,
        "candidate_top16_parity": True,
        "baseline_prompt_parity": True,
        "conditions": list(conditions),
        "evidence_budget": evidence_budget,
        "input_tokens": {
            name: {"minimum": min(values), "mean": round(mean(values), 1),
                   "maximum": max(values)}
            for name, values in lengths.items()
        },
        "all_within_8192": True,
        "private_directory": str(output.relative_to(ROOT)),
        "test_used": False,
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--evidence-budget", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    run(args.samples, args.seed, args.tag, evidence_budget=args.evidence_budget)
