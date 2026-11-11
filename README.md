# TCMDiscern

中醫證型辨別研究：BM25 候選檢索、Jeff-1.1-Large 中醫 LoRA + Decision Head，**主要 K=16**、全 K=1～35 評估。

## 研究目標

使用 TCM-SD 病例，以 **BM25 + Jeff-1.1-Large（Gemma 4 31B-it backbone）領域微調**為主要方法。BM25 提供候選，Jeff 透過 LoRA 與 Decision Head 在 K 個候選中選出正確證型。原始 Gemma 直接微調為備選對照。

研究依序分為：

1. 查核 TCM-SD 官方來源、標籤、切分與重複病例；依定稿保留官方全部病例，不執行去重或重新切分。
2. 在訓練資料中建立檢索索引，產生 **互異證型** 的候選；訓練樣本使用分組 Out-of-Fold 檢索。
3. 完整評估 BM25 Recall@K（K=1～35），後續才進行 Embedding、Hybrid 比較。
4. 進行 Jeff-1.1-Large 未微調基準，再以 TCM-SD Train 訓練 Jeff LoRA + Decision Head；原始 Gemma 直接微調僅為備選對照。
5. 多答案／標註歧義採預先固定的對照：安全單正解 CE、Masked CE、Multi-positive Set Loss，均勻 Soft-target CE 為次要消融。保持 K=16、相同 Train OOF 候選設定與完整官方評估。
6. 使用官方完整 Dev 比較 **K=1～35** 的真實 Jeff Top-1、Macro-F1、延遲、顯存與統計不確定性，選擇最佳 K。

> **2026-11-12 方法定稿：** Jeff 1.1 + BM25 K=16；主研究假設採 **Multi-positive Set Loss** 處理 Train 同文異標的標註歧義，安全單正解 CE 與 Masked CE 作對照，均勻 Soft CE 作次要消融。完整規格請先讀 [標註歧義與 Jeff 多正解方法定稿](docs/jeff_ambiguity_final_protocol_2026-11-12.md)。已完成原始 BM25 Dev 基準、Train Group-aware OOF 與安全單正解 K=16 43,180 筆資料 QA；**Set Loss 訓練資料與 GPU LoRA 尚未實作，沒有 Jeff 診斷成績。** 原始官方病例、標籤與切分完全保留。

## 開發環境

先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/)，其可自行安裝受支援的 Python：

```bash
uv python install 3.12
uv sync --python 3.12 --dev
uv run pytest
uv run ruff check .
```

一般 CPU 環境可進行資料處理與評分；Jeff-1.1-Large 微調需另外在 H200 驗證 GPU、PyTorch／Transformers 與顯存需求。本機暫不安裝大型訓練依賴。

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

K=1～35 的結果由已完成的 BM25 Dev 排名逐一核算，對照 [原始 BM25 K=1～50 快照](experiments/bm25_dev_official_2026-11-11.json) 驗證一致性。**Jeff Zero-shot 與 LoRA 尚無實測成績。**

## Jeff LoRA 研究主線

主模型：`jgeuter/Jeff-1.1-Large`。**主要 K=16（1 正解＋15 個 BM25 OOF 困難負例）**，對應原生 A～P Decision Head，**一次 Forward Pass** 即可決策。官方 Dev BM25 Recall@16：Micro **98.60%**（5,409/5,486）、Macro **89.91%**；比 K=17 少 3 筆候選命中。保留 K=1～35 的端到端評估，K=17～35 才採作者分組決策。已依原作者的 **SemIf `state/question/options` → `evidence/criterion/A～P`** 格式實作病例轉換器，訓練選項 1 正解＋15 困難負例，正式 Dev 不補入正解。詳見 [Jeff 微調計畫](docs/jeff_finetune_plan.md)。

```bash
uv run python scripts/export_jeff_dev.py          # 預設主要 K=16
uv run python scripts/export_jeff_dev.py --k 5    # 低 K 對照
uv run python scripts/export_jeff_dev.py --k 17   # 分組決策對照
```

已產生主要 Dev `data/jeff/dev_k16.jsonl`（5,486 筆），並保留舊 K=5、17 候選作對照，全部存放在 Git 忽略的 `data/jeff/`。Train OOF 已全部完成；Jeff LoRA 尚未實際訓練。

### 已完成：Train OOF BM25 與 Jeff 監督資料

已在完整官方 Train **43,180 筆**上執行 5-fold OOF BM25（每折索引 34,544、查詢 8,636），每筆產生 50 個互異證型候選；再建立以下三套 **每套 43,180 筆**的 Jeff 監督資料：

- `train_oof_k05_epoch00.jsonl`：1 正確＋4 個困難負例。
- `train_oof_k16_epoch00.jsonl`：1 正確＋15 個困難負例。
- `train_oof_mixed_k02_16_epoch00.jsonl`：混合 2～16 個選項。

原先 **129,540 筆題目全部逐筆通過驗證**，保留重複病例、記錄跨折 6,054 筆完全相同臨床文字查詢。**改為 K=16 後，已另用預設設定重建 Train 43,180 筆（epoch 1）與 Dev 5,486 筆，逐筆驗證全數通過**；結果見 [K=16 主設定驗證](experiments/jeff_primary_k16_2026-11-11.json)。Train OOF K=1～50、五折及 148 類分項統計也已封存。詳見 [Train OOF 研究報告](docs/jeff_train_data.md)、[全 K 統計](experiments/bm25_train_oof_all_k_2026-11-11.json)。

```bash
uv run python scripts/mine_jeff_train_oof.py
uv run python scripts/export_jeff_train.py                     # 預設只產生主要 K=16
uv run python -m scripts.verify_jeff_train                     # QA 封存資料
uv run python -m scripts.summarize_jeff_oof                    # OOF 全 K 統計
# 另行產生消融資料（須選擇未使用的 epoch）：
uv run python scripts/export_jeff_train.py --ks 5 16 --mixed --epoch 3
```

### 最新研究定稿：多答案與單答案公平比較

**S1 單金標／M1 多觀察金標 Soft-target CE／M2 相同候選 Set Loss**，全部使用 Jeff-1.1-Large、Train 43,180、Group-aware OOF、主要 K=16，正式 Dev／Test 不補金標。**已定稿方法，M1／M2 尚未實作與訓練。** 詳見 [完整研究定稿](docs/jeff_training_final_protocol_2026-11-12.md) 與 [多標註上界彙總](experiments/jeff_multi_annotation_bounds_2026-11-12.json)。

### 已完成：訓練前完整審查與矛盾負例過濾

- **全量 Tokenizer 檢查：** Train 43,180、Dev 5,486 筆 K=16 Prompt 都低於 Jeff 8,192 token 限制。
- **重複病例檢索風險：** 原 OOF 6,054 筆查詢能命中其他折完全相同的文字；新 Exact-text Group-aware OOF 降為 0，官方資料與切分不變。
- **明確矛盾負例：** 原 OOF 的 817 筆、Group-aware OOF 原本 793 筆，在 K=16 把相同病例的另一個官方金標列為錯誤選項。現在以同文不同金標集合過濾負例，另外產生 `epoch02` **43,180 筆**，共替換 797 個矛盾負例，完整 QA **0 個**已知矛盾負例。
- **仍阻擋大型微調：** 31B LoRA + Decision Head 訓練器、H200 記憶體實測與 Jeff Zero-shot Dev 分數尚未完成。需先確認 GPU 反向傳播、Head 正確讀取最後有效 Token、Loss／梯度、checkpoint 與原始 Jeff 逐 logit 一致。

```bash
# 以下為完整的產生步驟，已生成的快照採不覆寫政策
uv run python scripts/mine_jeff_train_oof.py --fold-policy exact_text_grouped
uv run python scripts/export_jeff_train.py \
  --candidates data/candidates/bm25_train_text_grouped.jsonl \
  --oof-report experiments/bm25_train_text_grouped_2026-11-11.json \
  --epoch 2
uv run python -m scripts.verify_jeff_grouped_filtered
uv run python scripts/preflight_jeff_training.py
```

新 K=16 訓練題位於 Git 忽略的 `data/jeff/train_oof_k16_epoch02.jsonl`，並由 [修正後的完整 QA 快照](experiments/jeff_grouped_filtered_k16_verification_2026-11-12.json)與 [訓練前完整風險審查](docs/jeff_pretraining_review_2026-11-11.md)記錄。**已有檔案不可直接重複執行輸出；要產生新排列須改用尚未使用的 `--epoch`。** 原始病例與逐筆訓練題目一律存於 Git 忽略的 `data/`，Git 僅存彙總統計。

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
