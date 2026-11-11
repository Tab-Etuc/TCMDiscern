# TCMDiscern

中醫證型辨別研究：檢索候選生成、難負例訓練、Gemma LoRA 重排序與自適應候選數量。

## 研究目標

使用 TCM-SD 病例，比較 BM25、Embedding、Hybrid 的候選覆蓋率，以及 Gemma 4 31B 在不同候選數量下的端到端證型辨別表現。

研究依序分為：

1. 查核 TCM-SD 官方來源、標籤、切分與重複病例；依定稿保留官方全部病例，不執行去重或重新切分。
2. 在訓練資料中建立檢索索引，產生 **互異證型** 的候選；訓練樣本使用分組 Out-of-Fold 檢索。
3. 完整評估 BM25 Recall@K（K=1～35），後續才進行 Embedding、Hybrid 比較。
4. 進行 Gemma Zero-shot 與 148 類直接分類基準實驗。
5. 比較隨機、困難、混合負例 LoRA；驗證集選擇模型，測試集僅作最終評估。
6. 比較固定 K 與 Adaptive K，報告 Top-1、Macro-F1、延遲、顯存與統計不確定性。

> 已完成 TCM-SD 原始資料稽核及 BM25 + 自動 K 的完整 Dev 基準；Embedding／Hybrid／Gemma LoRA 尚待實驗。正式採用**原始切分、零刪除**，詳見 [研究定稿](docs/research_protocol.md)。

## 開發環境

先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/)，其可自行安裝受支援的 Python：

```bash
uv python install 3.12
uv sync --python 3.12 --dev
uv run pytest
uv run ruff check .
```

一般 CPU 環境即可進行資料處理與評分測試。GPU／Gemma 訓練依賴待實測 H200 環境後獨立加入，避免本地安裝大型套件。

## 研究資料

已於研究工作機取得原作者公開的 **TCM-SD 54,152 筆病例**（Train 43,180／Dev 5,486／Test 5,486）、148 類詞表，以及 1,027 條證型知識；另備有 819 筆獨立研究集、MTCMB、RobustSDT、BEST4SDT 評估資產。資料儲存在本機 `data/`，**不隨 Git 提交**。

```bash
uv run python scripts/fetch_tcmsd.py          # 在新環境下載官方 TCM-SD
uv run python scripts/sync_existing_data.py   # 選用：匯入本機既有次要評估資料
uv run python scripts/audit_datasets.py       # 核對筆數、標籤、重複病例
```

原始格式為 JSONL（副檔名 `.json`），臨床輸入欄位是 `chief_complaint`、`description`、`detection`；標準證型是 `norm_syndrome`。完整資料來源、授權與已確認的 Train/Test 重複問題，請參閱 [研究資料清冊](docs/data_sources.md)。

## 已完成：BM25 自動選 K

可在 TCM-SD 官方 Dev 上計算 BM25 的互異證型候選排名，並以**二分搜尋**找出達成 Micro / Macro Recall 門檻的最小 K：

```bash
uv run python scripts/auto_k_bm25.py --max-k 50 --targets 0.80 0.85 0.90 0.95 0.98
```

完整官方 Dev 5,486 筆：Micro Recall ≥95% 最小 K=5、Macro Recall ≥90% 最小 K=17、Macro Recall ≥95% 最小 K=34。**已完整跑完 BM25 K=1～35，沒有加入其他分類器。** Gemma／Embedding／Hybrid 都尚未測試。完整方法與二分搜尋見 [原 BM25 實驗報告](docs/auto_k_experiment.md)。

## BM25 K=1～35 完整結果

**只用 BM25**，對官方 5,486 筆 Dev 評估每個 K 的候選 Micro Recall、Macro Recall、命中病例數、新增命中與漏掉病例數。BM25 第一名的 Top-1 固定為 79.58%，Macro-F1 為 54.85%；候選數增加不會改變第一名，K=17 是 Macro Recall ≥90% 的最小 K。

- [BM25 K=1～35 完整表格與說明](docs/bm25_k_1_35.md)
- [CSV：全部 35 組 BM25 成績](experiments/bm25_k1_35_dev_2026-11-11.csv)
- [JSON：全精度數據與 148 類別的逐 K 召回](experiments/bm25_k1_35_dev_2026-11-11.json)

```bash
uv run python scripts/export_bm25_k_grid.py
```

K=1～35 的結果由已完成的 BM25 Dev 排名逐一核算，對照 [原始 BM25 K=1～50 快照](experiments/bm25_dev_official_2026-11-11.json) 驗證一致性。Gemma 4 31B 尚無實測成績。

## 專案結構

```text
configs/                    研究設定
src/tcmdiscern/             共用研究程式
src/tcmdiscern/evaluation.py  候選召回率與端到端 Top-1 評分
tests/                      單元測試
data/                       本地資料（不提交 Git）
results/                    本地實驗輸出（不提交 Git）
```

## 評分資料格式

每個 JSONL 物件代表一筆病例，標籤使用正規化後的 canonical ID：

```json
{"case_id":"example-001","gold_labels":["zheng-001"],"candidate_labels":["zheng-003","zheng-001","zheng-002"],"predicted_label":"zheng-001"}
```

`gold_labels` 支援多標籤；Recall@K 是「至少一個金標出現在前 K 個不同證型中」；Top-1 是預測是否命中任一金標。選擇器**只能**預測候選內證型，正式測試不得補入金標。

以 JSONL 檔案評分：

```bash
uv run python -m tcmdiscern.evaluation data/predictions.jsonl --ks 5 7 10 15 20 25
```

評分程式將檢查空金標、重複候選、重複病例 ID 與候選外預測。資料不足 K 筆時仍以現有候選計算，實際正式評估應先固定候選生成策略。

## 研究規範

- 正式 Test 不用於參數選擇、訓練或 hard-negative mining。
- **研究定稿：不刪除重複病例、不因相同文字或 user_id 排除樣本、不重新切分。** 跨切分重複及標籤衝突如實記錄，論文揭露可能偏差。
- 正規化同義證型，避免把同義或其他正確標籤當作負例。
- 不提交病例原文、模型權重、API 金鑰與大型實驗輸出。
- 顯式區分「候選生成召回率」與「候選命中條件下的決策準確率」。
- 現有 CMU_AITCM 的 819 筆結果只作獨立研究參考，與 TCM-SD 數值不能直接混用。
