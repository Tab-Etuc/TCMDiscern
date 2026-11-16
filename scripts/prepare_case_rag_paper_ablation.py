"""Create full-Dev medical-evidence control inputs using frozen Global2 cases."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "paper_5486_20261116"
ARMS = ("global2_labelhide", "global2_labelonly")


def transform(row: dict, mode: str) -> dict:
    from tcmdiscern.case_rag_counterfactual import HIDDEN_LABEL

    if mode not in ARMS:
        raise ValueError("Unknown full-Dev evidence control")
    updated = deepcopy(row)
    for item in updated["state"]["retrieved_cases"]:
        if mode == "global2_labelhide":
            item["candidate_syndrome"] = HIDDEN_LABEL
        else:
            for field in ("chief_complaint", "description", "detection"):
                item[field] = ""
    return updated


def prepare(root: Path = ROOT) -> dict:
    """Prepare frozen label-only/hidden-label Global2 on all official Dev cases."""
    from contextlib import ExitStack
    from statistics import mean

    from sklearn.metrics import f1_score

    from tcmdiscern.case_rag_tokenizer import (
        count_jeff_prompt_tokens,
        load_jeff_token_counter,
    )
    from tcmdiscern.datasets import clinical_text, load_records

    folder = root / "data/case_rag" / TAG
    original_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        original_manifest.get("samples") != 5486
        or not original_manifest.get("baseline_prompt_parity")
        or not original_manifest.get("candidate_top16_parity")
    ):
        raise ValueError("Original 5486 Dev prompt parity failed")
    artifact = folder / "paper_ablations.json"
    if artifact.exists():
        value = json.loads(artifact.read_text(encoding="utf-8"))
        if value.get("samples") != 5486 or any(
            not (folder / f"{arm}.jsonl").is_file() for arm in ARMS
        ):
            raise ValueError("Frozen paper ablations are incomplete")
        return value
    if any((folder / f"{arm}.jsonl").exists() for arm in ARMS):
        raise ValueError("Partial paper ablation artifacts need inspection")
    prior = root / "data/case_rag/budget2_0300_20261116/global2.jsonl"
    old = {}
    if prior.is_file():
        with prior.open(encoding="utf-8") as stream:
            old = {v["id"]: v for v in map(json.loads, stream)}
        if len(old) != 300:
            raise ValueError("Old Pilot reference was incomplete")
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Official Train/Dev counts differ")
    codec, _ = load_jeff_token_counter(root)
    counts = {name: [] for name in ARMS}
    vote, gold, unseen = [], [], []
    original_texts = {clinical_text(row) for row in train}
    with ExitStack() as stack:
        source = stack.enter_context((folder / "global2.jsonl").open(encoding="utf-8"))
        baseline = stack.enter_context((folder / "baseline.jsonl").open(encoding="utf-8"))
        original = stack.enter_context((root / "data/jeff/dev_k16.jsonl").open(encoding="utf-8"))
        outputs = {
            name: stack.enter_context((folder / f"{name}.jsonl").open("x", encoding="utf-8"))
            for name in ARMS
        }
        for i, (source_line, base_line, orig_line) in enumerate(
            zip(source, baseline, original, strict=True)
        ):
            case = json.loads(source_line)
            b = json.loads(base_line)
            if b != json.loads(orig_line) or case["id"] != f"dev-{i:06d}":
                raise ValueError("Original model input / Dev ordering changed")
            if case["id"] in old and case != old[case["id"]]:
                raise ValueError("Full-Dev Global2 differs from original 300-case Pilot")
            for name in ARMS:
                altered = transform(case, name)
                if any(
                    altered["state"][key] != b["state"][key]
                    for key in ("chief_complaint", "description", "detection")
                ):
                    raise ValueError("Source ablation changed patient evidence")
                n = count_jeff_prompt_tokens(codec, altered)
                if n > 8192:
                    raise ValueError(f"{name} prompt too long at {i}")
                counts[name].append(n)
                outputs[name].write(json.dumps(altered, ensure_ascii=False) + "\n")
            vote.append(case["state"]["retrieved_cases"][0]["candidate_syndrome"])
            gold.append(str(dev[i]["norm_syndrome"]))
            unseen.append(clinical_text(dev[i]) not in original_texts)
            if (i + 1) % 500 == 0:
                print(f"[PAPER ABLATION] {i + 1}/5486", flush=True)
    if len(gold) != 5486:
        raise ValueError("Incomplete original Dev")
    classes = sorted({str(item["norm_syndrome"]) for item in train})
    if len(classes) != 148:
        raise ValueError("Missing original 148 labels")
    matched = [a == b for a, b in zip(vote, gold, strict=True)]
    clean = sum(unseen)
    shortcut = {
        "status": "PAPER_CPU_RETRIEVED_LABEL_CONTROL",
        "samples": len(gold),
        "selection": "Use the highest-BM25-score historical Global2 case diagnosis",
        "top1_percent": round(100 * mean(matched), 4),
        "macro_f1_148_percent": round(
            100 * f1_score(gold, vote, labels=classes, average="macro", zero_division=0),
            4,
        ),
        "without_exact_train_text_n": clean,
        "without_exact_train_text_top1_percent": round(
            100 * sum(m for m, u in zip(matched, unseen, strict=True) if u) / clean,
            4,
        ),
        "test_used": False,
    }
    report = {
        "status": "PAPER_5486_ABLATIONS_VERIFIED",
        "samples": len(gold),
        "conditions": list(ARMS),
        "source_tag": TAG,
        "pilot_300_global2_parity": len(old) == 300,
        "input_tokens": {
            mode: {"minimum": min(vals), "mean": round(mean(vals), 2), "maximum": max(vals)}
            for mode, vals in counts.items()
        },
        "cpu_retrieved_label_control": shortcut,
        "test_used": False,
    }
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "experiments/case_rag_paper_cpu_source_label_20261117.json").write_text(
        json.dumps(shortcut, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
