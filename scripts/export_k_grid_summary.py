"""Export human-readable K=1..35 tables from full official Dev results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments/k1_35_cpu_selector_dev_2026-11-11.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    methods = report["methods"]
    ks = list(range(1, 36))
    if report["k_values"] != ks or report["excluded_records"] != 0:
        raise ValueError("K grid or official no-dedup policy mismatch")

    for model in ("bm25_rank1", "multinomial_nb", "complement_nb"):
        results = methods[model]["results"]
        if [item["k"] for item in results] != ks:
            raise ValueError(f"{model} missing K=1..35")

    table = []
    for idx, k in enumerate(ks):
        top1 = methods["bm25_rank1"]["results"][idx]
        nb = methods["multinomial_nb"]["results"][idx]
        cnb = methods["complement_nb"]["results"][idx]
        if not (
            top1["oracle_micro_recall"] == nb["oracle_micro_recall"] == cnb["oracle_micro_recall"]
        ):
            raise ValueError(f"Different candidate oracle coverage at K={k}")
        table.append(
            {
                "k": k,
                "n": top1["n"],
                "oracle_covered_n": top1["oracle_covered_n"],
                "oracle_micro_recall": top1["oracle_micro_recall"],
                "oracle_macro_recall": top1["oracle_macro_recall"],
                "bm25_rank1_top1": top1["top1"],
                "bm25_rank1_macro_f1": top1["macro_f1"],
                "multinomial_nb_top1": nb["top1"],
                "multinomial_nb_macro_f1": nb["macro_f1"],
                "multinomial_nb_correct_n": nb["correct_n"],
                "multinomial_nb_conditional_accuracy": nb[
                    "conditional_accuracy_given_gold_in_candidates"
                ],
                "complement_nb_top1": cnb["top1"],
                "complement_nb_macro_f1": cnb["macro_f1"],
                "complement_nb_correct_n": cnb["correct_n"],
                "complement_nb_conditional_accuracy": cnb[
                    "conditional_accuracy_given_gold_in_candidates"
                ],
            }
        )
    csv_path = root / "experiments/k1_35_cpu_selector_dev_2026-11-11.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    mdrows = "\n".join(
        f"| {x['k']} | {x['oracle_covered_n']} | "
        f"{pct(x['oracle_micro_recall'])} | {pct(x['oracle_macro_recall'])} | "
        f"{pct(x['bm25_rank1_top1'])} | "
        f"{pct(x['multinomial_nb_top1'])} | {pct(x['multinomial_nb_macro_f1'])} | "
        f"{pct(x['complement_nb_top1'])} | {pct(x['complement_nb_macro_f1'])} |"
        for x in table
    )
    best_nb_top1 = max(table, key=lambda x: (x["multinomial_nb_top1"], -x["k"]))
    best_nb_macro = max(table, key=lambda x: (x["multinomial_nb_macro_f1"], -x["k"]))
    best_cnb_top1 = max(table, key=lambda x: (x["complement_nb_top1"], -x["k"]))
    best_cnb_macro = max(table, key=lambda x: (x["complement_nb_macro_f1"], -x["k"]))
    times = report["timing_seconds"]
    nb_time = methods["multinomial_nb"]["timing_seconds"]
    cnb_time = methods["complement_nb"]["timing_seconds"]
    text = f"""# TCM-SD K=1～35 全數實測：BM25 與 CPU 候選重排序基準

**日期：2026-11-11｜狀態：35 個 K 的 BM25 + 輕量 CPU 模型已實測；Gemma LoRA 尚未執行**

## 研究設計

- **Train：43,180 筆，全部保留**；用於 BM25 候選資料庫及兩個簡單 CPU 文字分類器。
- **Dev：5,486 筆，全部保留**；每個 K 都使用**同一批完整官方 Dev**。不刪除重複病例、不重新切分、不改標籤。
- **Test：5,486 筆，保留最終實驗**；本輪不執行 Test 模型評估。
- 類別：148 個官方 `norm_syndrome` 證型。查詢／分類器輸入限主訴、病情描述、四診資訊。
- 每個 Dev 病例的 BM25 **Top-50 互異證型排名**是預先完成且固定的。取前 K（1～35）作分類器可選的候選。
- **BM25 rank-1**：只選排名第一的標籤，當成完全不重排的參考基準，其 Top-1 對 K 必然不變。
- **MultinomialNB 與 ComplementNB**：只使用 Train 以 `CountVectorizer` CJK Bigram（`min_df=2`）、`alpha=1` 訓練兩個文字分類器；對 Dev 一次輸出全部 148 類別分數。在 K 個 BM25 候選內選擇分類器分數最高的標籤。
- **固定同一組分類器權重及 Dev 分數**，所有 35 個 K 不重新訓練，以排除訓練隨機性。模型的類別分數不含候選列表，因此這是**固定分數、候選限制的 CPU 重排序基準**。
- **每個 K 都實際根據固定模型分數，重新選出 5,486 個最終標籤並計算 Top-1、Macro-F1**，總共兩個模型各 35 組。
- 資料來源、重複病例政策與跨切分重複稽核：[研究定稿](research_protocol.md)。

以上 CPU 方法是額外的初步基準；**Gemma 4 31B／Gemma LoRA 目前沒有完成訓練或推論，不能把下列 CPU 準確率當作 Gemma 分數**。

## K=1～35 完整表格

| K | 候選命中病例數 | 候選 Micro Recall | 候選 Macro Recall | BM25 Top-1 | MultinomialNB Top-1 | MultinomialNB Macro-F1 | ComplementNB Top-1 | ComplementNB Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{mdrows}

此表的「候選 Micro/Macro Recall」是檢索候選涵蓋率；後續 Top-1 和 Macro-F1 是分類器實際選擇後的成績。35 個 K 的 **Conditional Accuracy、正確病例數、每類 precision/recall/F1** 保存在詳細 JSON（包含逐類別結果）及 CSV（包含各 K 彙總）中。

## 初步結論：候選 Recall 與分類 Top-1 可能反向

| 評估方法 | Dev Top-1 最佳 K | 最佳 Top-1 | Macro-F1 最佳 K | 最佳 Macro-F1 |
|---|---:|---:|---:|---:|
| BM25 rank-1（不重排） | 1（所有 K 相同） | {pct(table[0]["bm25_rank1_top1"])} | 1（所有 K 相同） | {pct(table[0]["bm25_rank1_macro_f1"])} |
| MultinomialNB（固定分數） | {best_nb_top1["k"]} | {pct(best_nb_top1["multinomial_nb_top1"])} | {best_nb_macro["k"]} | {pct(best_nb_macro["multinomial_nb_macro_f1"])} |
| ComplementNB（固定分數） | {best_cnb_top1["k"]} | {pct(best_cnb_top1["complement_nb_top1"])} | {best_cnb_macro["k"]} | {pct(best_cnb_macro["complement_nb_macro_f1"])} |

候選 Recall@K 隨 K 增加只會上升或持平；固定 CPU 模型的 Top-1 可能因為更多難辨識候選加入而下降。**上述結果只代表這兩個簡單的 CPU 分類器，不能推論 Gemma 會有相同的最佳 K。**

**正式方法已改成全部測試 K=1～35**，對每個 K 使用同一個 Gemma LoRA checkpoint、相同 Dev 病例、相同檢索排名；以 Dev Top-1、Macro-F1 及真實推論成本決定最佳 K。原先 K=17 與 5／35 是舊版預選值，現在只作參考刻度。

### 執行時間與環境

- CPU，本輪**未使用 H200 或 RTX 4060**。
- Python {report["environment"]["python"]}、NumPy {report["environment"]["numpy"]}、SciPy {report["environment"]["scipy"]}、scikit-learn {report["environment"]["scikit_learn"]}。
- Train/Dev 特徵建置：**{times["vectorize_train_and_dev"]:.3f} 秒**。
- MultinomialNB 訓練：**{nb_time["fit_model"]:.3f} 秒**；Dev 全類分數計算 **{nb_time["dev_score_once"]:.3f} 秒**；K 全評估 **{nb_time["all_k_evaluation"]:.3f} 秒**。
- ComplementNB 訓練：**{cnb_time["fit_model"]:.3f} 秒**；Dev 全類分數計算 **{cnb_time["dev_score_once"]:.3f} 秒**；K 全評估 **{cnb_time["all_k_evaluation"]:.3f} 秒**。
- 上述秒數**不含首次 BM25 建索引／候選檢索約 22.36 秒**，也不含 Gemma 推論。

## 研究資料與重現方式

- 原始資料：`data/tcm-sd/`，Git 忽略。
- BM25 排名／Dev 金標：`data/candidates/bm25_dev.jsonl`，Git 忽略。
- 全部 K 的 CPU 結果（包括每類 F1）：[詳細 JSON](../experiments/k1_35_cpu_selector_dev_2026-11-11.json)。
- 全部 K 的比較數字（適合 Excel、圖表及論文表格）：[CSV](../experiments/k1_35_cpu_selector_dev_2026-11-11.csv)。
- 原始 BM25 檢索與二分搜尋：[實驗快照](../experiments/bm25_dev_official_2026-11-11.json)。
- 正式研究規範：[research_protocol.md](research_protocol.md)。

```bash
uv sync --python 3.12 --dev
uv run python scripts/fetch_tcmsd.py
uv run python scripts/auto_k_bm25.py --max-k 50 --targets 0.80 0.85 0.90 0.95 0.98
uv run python scripts/evaluate_k_grid.py --max-k 35
uv run python scripts/export_k_grid_summary.py
uv run pytest -q
```

上述執行程式採只建立新快照的原則，重現時需使用新的輸出路徑，避免覆蓋已封存的原始實驗結果。

## Gemma LoRA：尚未執行的 35 組正式實驗

目前連線的開發環境只有 **NVIDIA RTX 4060 8GB**，未找到 Gemma 4 31B 本地模型權重或可用的 H200 執行節點。

需在配置 Gemma 4 31B LoRA checkpoint 及 H200 後，對全部 5,486 筆 Dev 產生 **35 組 K=1～35 的模型預測**，逐組評分 Top-1、Macro-F1、Conditional Accuracy、延遲及顯存。採 **同一 LoRA checkpoint** 比較所有 K，防止不同訓練模型掩蓋候選數量效應。

應注意：如果改用一次取得所有候選獨立分數、再以 Top-K 子集合取最高分的策略，能高效率評估所有 K，但這與**每個 K 都把 K 個候選放入 Gemma listwise prompt 重新推論**屬不同的實驗方法；未來須明確分別報告。
"""
    text += '\n\n### Gemma 預測檔案格式與驗證指令\n\n輸入格式為 JSONL，**每筆 Dev 病例在每個 K 都需要一筆預測**；例如下列示意中的選項必須與 `data/candidates/bm25_dev.jsonl` 該病例的前 K 名一致：\n\n```json\n{"case_id":"dev-000000","k":1,"predicted_label":"<前1候選證型>","latency_ms":123.4}\n{"case_id":"dev-000000","k":2,"predicted_label":"<前2候選中的某個證型>","latency_ms":139.1}\n```\n\n正式檔案應包含 **35 × 5,486 = 192,010 筆**有效預測，且每個 `(case_id, k)` 只能出現一次。模型實際執行完成後，可用以下指令產生正式模型評分；缺少任何 K／病例、重複輸出、預測超出前 K 名，都會拒絕評估：\n\n```bash\nuv run python scripts/score_llm_k_grid.py data/predictions/gemma_35k_dev.jsonl \\\n  --model "實際 checkpoint 識別碼" --method listwise \\\n  --output experiments/gemma_listwise_k1_35_dev.json\n```\n\n上述指令是**未來驗證實際模型結果**，目前沒有執行 Gemma，也沒有生成 192,010 筆 Gemma 預測。\n'
    doc = root / "docs/k_1_35_experiment.md"
    doc.write_text(text, encoding="utf-8")
    print(f"Generated {csv_path} and {doc}")
    print(
        json.dumps(
            {
                "best_nb_top1_k": best_nb_top1["k"],
                "best_nb_top1": best_nb_top1["multinomial_nb_top1"],
                "best_nb_macro_k": best_nb_macro["k"],
                "best_cnb_top1_k": best_cnb_top1["k"],
                "best_cnb_top1": best_cnb_top1["complement_nb_top1"],
                "best_cnb_macro_k": best_cnb_macro["k"],
                "n_rows": len(table),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
