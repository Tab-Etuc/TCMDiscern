"""Export only BM25 retrieval Recall@K for every K=1..35.

Inputs: immutable official TCM-SD Dev, cached BM25 train-only candidate rankings.
No classifier, reranker, LLM, deduplication or Test predictions.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score

from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions

MAX_K = 35
OFFICIAL_TRAIN_N = 43180
OFFICIAL_DEV_N = 5486


def evaluate(root: Path) -> dict:
    baseline = json.loads(
        (root / "experiments/bm25_dev_official_2026-11-11.json").read_text(encoding="utf-8")
    )
    dev = load_records(root / "data/tcm-sd/dev.json")
    candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(dev) != len(candidates) or len(dev) != OFFICIAL_DEV_N:
        raise ValueError("The complete original TCM-SD Dev split is required")
    if (
        baseline["official_split"]
        != {"train": OFFICIAL_TRAIN_N, "dev": OFFICIAL_DEV_N, "test": OFFICIAL_DEV_N}
        or baseline["excluded_cases"] != 0
    ):
        raise ValueError("Frozen official BM25 protocol does not match")

    gold: list[str] = []
    first_choice: list[str] = []
    hit_ranks: list[int] = []
    for i, (row, candidate) in enumerate(zip(dev, candidates, strict=True)):
        correct_label = str(row["norm_syndrome"])
        if candidate.case_id != f"dev-{i:06d}" or candidate.gold_labels != (correct_label,):
            raise ValueError(f"Candidate/gold mismatch at Dev row {i}")
        if len(candidate.candidate_labels) != 50 or len(set(candidate.candidate_labels)) != 50:
            raise ValueError(f"Dev row {i} requires 50 distinct BM25 syndrome candidates")
        if candidate.predicted_label is not None:
            raise ValueError("The cached file must contain only BM25 candidates")
        gold.append(correct_label)
        first_choice.append(candidate.candidate_labels[0])
        try:
            rank = candidate.candidate_labels.index(correct_label) + 1
        except ValueError:
            rank = 51
        hit_ranks.append(rank)

    labels = sorted(set(gold))
    if len(labels) != 148:
        raise ValueError("Expected the original 148 syndrome labels")
    support = Counter(gold)
    hit_count = Counter(hit_ranks)
    bm25_top1 = sum(rank == 1 for rank in hit_ranks) / OFFICIAL_DEV_N
    bm25_macro_f1 = float(f1_score(gold, first_choice, labels=labels, average="macro"))
    per_class_hits: dict[str, list[int]] = {label: [] for label in labels}
    for label, rank in zip(gold, hit_ranks, strict=True):
        per_class_hits[label].append(rank)

    rows: list[dict] = []
    cumulative = 0
    for k in range(1, MAX_K + 1):
        added = hit_count[k]
        cumulative += added
        micro = cumulative / OFFICIAL_DEV_N
        macro = sum(
            sum(rank <= k for rank in per_class_hits[label]) / support[label] for label in labels
        ) / len(labels)
        if abs(micro - baseline["micro"]["recall_at_each_k"][str(k)]) > 1e-12:
            raise ValueError(f"Frozen BM25 Micro Recall differs at K={k}")
        if abs(macro - baseline["macro"]["recall_at_each_k"][str(k)]) > 1e-12:
            raise ValueError(f"Frozen BM25 Macro Recall differs at K={k}")
        if cumulative != baseline["micro"]["correct_counts_at_each_k"][str(k)]:
            raise ValueError(f"Frozen BM25 case count differs at K={k}")
        rows.append(
            {
                "k": k,
                "cases": OFFICIAL_DEV_N,
                "candidate_hit_n": cumulative,
                "new_candidate_hits_at_k": added,
                "missed_cases": OFFICIAL_DEV_N - cumulative,
                "micro_recall": micro,
                "macro_recall": macro,
                "bm25_top1_accuracy": bm25_top1,
                "bm25_top1_macro_f1": bm25_macro_f1,
            }
        )

    return {
        "study": "TCMDiscern",
        "date": "2026-11-11",
        "status": "BM25_ONLY_COMPLETE_K1_TO_K35",
        "dataset": "TCM-SD",
        "index": "43,180 original Train clinical cases",
        "query": "5,486 original Dev clinical cases",
        "label_space": 148,
        "excluded_records": 0,
        "duplicate_policy": "retain all original official records",
        "retriever": "BM25 CJK Bigram; per-syndrome max training-case score",
        "evaluation_methods": [
            "BM25 candidate Micro Recall@K",
            "BM25 candidate Macro Recall@K",
            "BM25 rank-1 accuracy and Macro-F1",
        ],
        "not_executed": ["Gemma", "LoRA", "Embedding", "Hybrid"],
        "results": rows,
        "per_syndrome_recall": [
            {
                "syndrome": label,
                "dev_n": support[label],
                "recall_at_k": {
                    str(k): sum(rank <= k for rank in per_class_hits[label]) / support[label]
                    for k in range(1, MAX_K + 1)
                },
            }
            for label in labels
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = evaluate(root)
    rows = result["results"]
    path = root / "experiments/bm25_k1_35_dev_2026-11-11"
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    table = "\n".join(
        f"| {row['k']} | {row['candidate_hit_n']} | {row['new_candidate_hits_at_k']} "
        f"| {row['missed_cases']} | {pct(row['micro_recall'])} "
        f"| {pct(row['macro_recall'])} |"
        for row in rows
    )
    document = f"""# TCM-SD：BM25 K=1～35 完整結果

**2026-11-11｜僅 BM25｜已完整實測｜未使用其他分類器。**

- 43,180 筆原始 Train 建立 BM25 資料庫；5,486 筆完整 Dev 作查詢，148 類證型。
- 官方 Train／Dev／Test 全部依原始切分，**不刪除重複病例、不變更標籤、不重新切分**。
- 查詢使用 `chief_complaint`、`description`、`detection`；Train 標籤僅用於病例最高分的證型聚合。
- 中文 Bigram BM25（`k1=1.5`、`b=0.75`、`min_df=2`）；每個病例保存前 50 個互異證型，本次評估 K=1～35。
- 測試集 5,486 筆尚未進行模型評估。
- 這份報告**只有 BM25**。沒有加入其他分類器或重排序器；Embedding、Hybrid、Gemma 均尚未實測，不得將召回率寫成最終 LLM 診斷準確率。

## K=1～35 實測表

| K | 候選包含正確證型（/ 5486） | 此 K 新增命中 | 候選漏掉病例 | Micro Recall@K | Macro Recall@K |
|---:|---:|---:|---:|---:|---:|
{table}

## 指標解讀與決策

- **Micro Recall@K**：所有 Dev 病例中，正確證型出現在前 K 個候選的比例。
- **Macro Recall@K**：148 個證型各自候選命中率的等權平均，避免只看多數類別。
- **新增命中**：K 比 K−1 多找到的病例數；例如此列為 0，代表該排名沒有新病例首次命中。
- **BM25 直接取第一名的 Top-1**：{pct(rows[0]["bm25_top1_accuracy"])}（所有 K 不變）。
- **BM25 直接取第一名的 Macro-F1**：{pct(rows[0]["bm25_top1_macro_f1"])}（所有 K 不變）。
- 因為 BM25 排名固定，增加可查看的 K 不會改變第一名；若想提升最終 Top-1，需額外的重排序方法，但不屬於本次 BM25 實驗。

| K | Micro Recall | Macro Recall | 說明 |
|---:|---:|---:|---|
| 1 | {pct(rows[0]["micro_recall"])} | {pct(rows[0]["macro_recall"])} | 直接採用 BM25 第一名 |
| 5 | {pct(rows[4]["micro_recall"])} | {pct(rows[4]["macro_recall"])} | Micro 超過 95% 的最小 K |
| 17 | {pct(rows[16]["micro_recall"])} | {pct(rows[16]["macro_recall"])} | Macro 超過 90% 的最小 K，成本／候選涵蓋的折衷 |
| 34 | {pct(rows[33]["micro_recall"])} | {pct(rows[33]["macro_recall"])} | Macro 超過 95% 的最小 K |
| 35 | {pct(rows[34]["micro_recall"])} | {pct(rows[34]["macro_recall"])} | 本次 K 範圍中的最高候選涵蓋率 |

若目標是 **候選 Micro Recall ≥95%、Macro Recall ≥90%**，本輪 BM25 的最小 K 是 **17**；若只關心 BM25 第一名分類正確率，K=1 已經是相同的決策結果。

## 檔案與重現

- [CSV：BM25 K=1～35](../experiments/bm25_k1_35_dev_2026-11-11.csv)
- [JSON：完整精度、35 組結果與 148 類證型分項召回](../experiments/bm25_k1_35_dev_2026-11-11.json)
- [原始 BM25 K=1～50 研究快照](../experiments/bm25_dev_official_2026-11-11.json)
- [研究資料規則](research_protocol.md)

```bash
uv sync --python 3.12 --dev
uv run python scripts/auto_k_bm25.py --max-k 50 --targets 0.80 0.85 0.90 0.95 0.98
uv run python scripts/export_bm25_k_grid.py
uv run pytest -q
```

原始病例、Dev 個別候選資料存放於 Git 忽略的 `data/`。公開成果僅包含彙總指標、每證型召回及必要方法說明。
"""
    (root / "docs/bm25_k_1_35.md").write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "k_count": len(rows),
                "dev_cases": result["results"][0]["cases"],
                "bm25_top1": rows[0]["bm25_top1_accuracy"],
                "bm25_macro_f1": rows[0]["bm25_top1_macro_f1"],
                "micro_recall_at_17": rows[16]["micro_recall"],
                "macro_recall_at_17": rows[16]["macro_recall"],
                "excluded": result["excluded_records"],
                "outputs": [
                    str(path.with_suffix(".csv")),
                    str(path.with_suffix(".json")),
                    str(root / "docs/bm25_k_1_35.md"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
