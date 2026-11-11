# Jeff-1.1-Large：TCM-SD Train OOF 候選與監督資料（2026-11-11）

**狀態：資料產生與完整性驗證已完成。Jeff 模型 LoRA/Decision Head 微調尚未執行。**

本次將 TCMDiscern 的全部官方 **Train 43,180 筆**建立為與 Jeff/SemIf 相容的病例多選題，使用 **5-fold Out-of-Fold (OOF) BM25** 挖掘難負例。這是依 [Jeff 官方 Prompt 格式](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py)進行的 **TCM 領域資料轉換**；上游公開儲存庫主要包含推論與 Head 實作，未提供整套原始訓練腳本，因此外層監督欄位是 TCMDiscern 的自訂格式。

## 1. 資料邊界與不去重政策

| 資料 | 筆數 | 本次使用 |
|---|---:|---|
| 官方 Train | **43,180** | 全部納入 OOF 檢索與訓練題目 |
| 官方 Dev | 5,486 | 不參與 OOF、訓練負例或訓練題目 |
| 官方 Test | 5,486 | 不參與 OOF、訓練負例或訓練題目 |
| 證型 | 148 類 | 使用官方 `norm_syndrome` 正規化標籤 |
| 刪除或排除 Train | **0** | 依 2026-11-11 定稿保留全部病例 |

訓練查詢只使用 `chief_complaint`、`description`、`detection`。病例 `user_id`、`lcd_id/lcd_name`、`syndrome`、`norm_syndrome` 等標註欄位均未當成查詢特徵。**Gold label 只用於建立訓練多選題與計算監督 Loss**；每題有且只有一個正確選項。

## 2. OOF BM25 的完整設定

- 固定 `seed=20261111`、`StratifiedKFold(n_splits=5, shuffle=True)`，依官方 Train 的 `norm_syndrome` 做類別分層。
- 每折：34,544 筆索引、8,636 筆查詢；五折合計每筆 Train **恰好查詢一次**。
- 每筆查詢只用**另外四折**建立 BM25 索引。查詢的原始資料列不會進入自己的檢索索引。
- 與正式 Dev BM25 相同：CJK Bigram；`k1=1.5`、`b=0.75`、`min_df=2`；同證型取訓練病例最高分。
- 每筆保存前 **50 個互異證型候選**，保留 BM25 排名及各列 `fold`，以供之後按 K 組成訓練題目。
- **不做病人分組、去重或跨折相似病例刪除**。這符合已凍結的原始資料政策，也代表其他折可能包含相同文字的病例，需在論文限制揭露。

### 五折實際執行紀錄

| 折 | 索引筆數 | 查詢筆數 | 耗時 |
|---|---:|---:|---:|
| 1 | 34,544 | 8,636 | 26.456 秒 |
| 2 | 34,544 | 8,636 | 26.315 秒 |
| 3 | 34,544 | 8,636 | 26.348 秒 |
| 4 | 34,544 | 8,636 | 26.331 秒 |
| 5 | 34,544 | 8,636 | 26.293 秒 |
| **合計** | 5 個獨立四折索引 | **43,180** | **131.743 秒** |

### Train OOF 候選覆蓋率（真實量測）

| K | OOF Micro Recall | OOF Macro Recall |
|---:|---:|---:|
| 1 | 79.85% | 51.25% |
| 3 | 93.40% | 70.63% |
| 5 | **95.90%** | **77.62%** |
| 10 | 97.80% | 85.39% |
| 16 | 98.62% | 90.39% |
| 17 | 98.70% | 90.78% |
| 35 | 99.46% | 94.98% |
| 50 | 99.69% | 96.38% |

此表僅是 **Train OOF 的 BM25 檢索結果**，不能與官方 Dev 的 BM25 或 Jeff 最終辨證準確率混淆。原始完整精度、每折時間與其他統計保存在 [OOF 彙整報表](../experiments/bm25_train_oof_2026-11-11.json)。另外已從完整 43,180 筆 OOF 候選**重新交叉驗算 K=1～50 全曲線、5 折個別 Recall、148 類各自候選命中分布**，見 [OOF 完整統計 JSON](../experiments/bm25_train_oof_all_k_2026-11-11.json) 與 [OOF K=1～35 CSV](../experiments/bm25_train_oof_k1_35_2026-11-11.csv)。這些彙總不含病例文字或患者 ID。

### 保留的跨折重複情況

- **6,054 筆查詢**在其他折存在完全相同的主訴／描述／四診文字。
- 其中 **381 筆查詢**，其標籤未出現在其他折那些完全相同文字病例的標籤集合中（此條件下的衝突統計）。
- 全部保留；OOF 僅保證排除**同一原始資料列**，並未保證跨折患者或文字不重複。
- 原始 Train 有 **1 筆**臨床文字直接包含自己的正規化證型名稱。保留原文且記錄；儘管模型 evidence 不含 `norm_syndrome` 欄位，仍不能宣稱所有病例原文都不含答案字串。

## 3. 已建立的 Jeff 監督題目

| 訓練檔（本機 `data/jeff/`） | 題型 | 題目數 |
|---|---|---:|
| `train_oof_k05_epoch00.jsonl` | 1 個金標＋4 個 OOF 高分錯誤證型 | 43,180 |
| `train_oof_k16_epoch00.jsonl` | 1 個金標＋15 個 OOF 高分錯誤證型 | 43,180 |
| `train_oof_mixed_k02_16_epoch00.jsonl` | 每筆 K 隨機抽樣 2～16，1 個金標＋K−1 個 OOF 負例 | 43,180 |
| **合計** | **三套替代訓練資料** | **129,540** |

**三套是不同訓練設定，不能直接把 129,540 筆當成 129,540 個獨立患者。** 每套涵蓋相同的 43,180 筆官方 Train；比較模型時應控制相同訓練步數、樣本與計算預算。

混合 K=2～16 的每個 K 在第一個 epoch 約有 2,800～3,000 筆；K=5 的正確答案在 A～E 的位置約各占 20%，K=16 的正確答案在 A～P 的位置約各占 6.25%。

### 實際 JSONL 結構

```json
{
  "row": {
    "id": "train-000001",
    "state": {
      "chief_complaint": "（官方原始主訴）",
      "description": "（官方原始病情描述）",
      "detection": "（官方原始四診）"
    },
    "question": "根據患者主訴、病情與四診資訊，選出最符合的中醫證型。",
    "options": [
      {"id": "證型甲", "description": "證型甲"},
      {"id": "證型乙", "description": "證型乙"}
    ],
    "qtype": "choice"
  },
  "target_index": 1,
  "target_letter": "B"
}
```

以上為**格式示意**，並未引用任何真實患者病例。只將內部 `row` 交由 Jeff 的 `prompting.messages` 或 `render_prompt` 轉換成真正的模型輸入；外層 `target_index`、`target_letter` 僅供監督損失使用。每次用新的 `--epoch` 重新產生題目時，會重新打亂答案位置，混合版也會重新抽 K；已選取的前 K−1 個 BM25 困難負例保持按排名選取。

### Jeff 格式與選項數量限制

- `row` 必須包含 `id/state/question/options`，並使用官方固定 System Message。
- 模型讀到的 User JSON 僅包含 `evidence/criterion/options`，選項用 **A～P** 標示。
- 透過官方模型 tokenizer 執行 Chat Template，設定 `add_generation_prompt=True`、`enable_thinking=False`。
- **2026-11-11 決策更新：主要訓練 K=16（1 正解＋15 困難負例），採一次 Forward Pass。** K=5、混合 2～16 保留為已有的消融資料。Jeff 原生 Decision Head 一次能處理 2～16 個選項。
- 正式 Dev **K=17～35** 使用作者原生的 **balanced grouping + winners 決選**推論；K=1 直接採用 BM25 第一名。

## 4. 完整資料 QA 已通過

已另外**逐筆驗證全部 129,540 筆訓練題目**，包含：病例順序／ID、官方 Train 金標、BM25 前 K−1 困難負例、正解唯一性、選項字母與 `target_index` 一致、病例欄位白名單、Jeff/SemIf Prompt 結構、各 K 筆數及不同答案位置的分布。

| QA 項目 | 結果 |
|---|---|
| 全部訓練題目逐筆檢核 | **129,540 / 129,540 通過** |
| 原始病例未被排除 | **43,180 / 43,180 保留** |
| 正解選項每題僅一個 | 通過 |
| 正解標籤欄位未進 evidence | 通過 |
| 監督目標未進 Prompt | 通過 |
| K 與 A～P 選項對應 | 通過 |
| 訓練候選來自 OOF BM25 | 通過 |
| 需 H200 實際量測的 Gemma tokenizer token 數 | **尚未量測** |

詳細彙整：[訓練資料品質報表](../experiments/jeff_training_quality_2026-11-11.json)。**檔案字元長度已記錄，仍需用釘選版本的 Gemma 4 tokenizer 確認是否超過 Jeff 8,192 token 上限**。不以字元數假裝 token 長度已驗證。

## 5. 本機可重現流程

從 TCMDiscern 專案根目錄執行：

```bash
uv sync --python 3.12 --dev

# 完整官方 Train 上執行五折 OOF（約 132 秒，需原始資料存在）
uv run python scripts/mine_jeff_train_oof.py

# 主要 K=16（首次產生新 epoch，僅輸出 K16）
uv run python scripts/export_jeff_train.py

# 如需重現原本三套 K=5／16／混合 2～16 的消融資料：
# 在對應 epoch 尚未有輸出的情況下執行
uv run python scripts/export_jeff_train.py --ks 5 16 --mixed --epoch 2

# 逐筆驗證全部訓練題目並輸出只有彙總統計的 QA 報表
uv run python -m scripts.verify_jeff_train

# 重新交叉驗算所有 Train OOF K=1～50 與五折分項結果
uv run python -m scripts.summarize_jeff_oof

uv run pytest -q
uv run ruff check .
```

每個輸出腳本採**不覆寫既有資料**的政策。要在**同一份 OOF 候選**上產生新 epoch 的不同選項順序與 K 分布，只執行：

```bash
uv run python scripts/export_jeff_train.py --epoch 3
```

如果 H200 訓練器會執行多個 epoch，**不能重複讀取同一個 `epoch00` 檔案並宣稱每個 epoch 都重新打亂**；需由 DataLoader 在每個 epoch 直接呼叫 `make_train_example`，或預先輸出每個 epoch 對應的資料檔。

本機逐筆訓練資料位於 `data/candidates/`、`data/jeff/`，均受 `.gitignore` 保護；Git 只提交方法、程式、測試與無病例原文的 Aggregate JSON。

## 6. 下一個實際訓練階段

**目前沒有 H200 上的 Jeff LoRA checkpoint 或實測準確率。**

後續使用固定版本的 `jgeuter/Jeff-1.1-Large`：

1. 用官方 tokenizer 與 `render_prompt` 逐筆檢查最大 8,192 tokens 的上下文限制。
2. 從 Jeff 發布模型與原作者 Decision Head 開始，凍結主要模型、另加新 LoRA 並訓練 Head。
3. 以 BM25 硬負例與固定訓練預算完成模型訓練；利用完整 Dev 5,486 筆 **K=1～35** 評估真正的 Top-1、Macro-F1、延遲與峰值顯存。
4. 在 Dev 決定模型 checkpoint 和最終 K，官方 Test 5,486 筆僅作最終正式測試。

**本次的 OOF Recall 與監督資料檢核是準備階段的成果，不能當成 Jeff LoRA 診斷準確率。**
