# Case-RAG：H200 上立即執行（2026-11-16）

## 最短指令

在原本已完成 Jeff 31B、S2／S3 訓練的 H200 專案中執行：

```bash
cd TCMDiscern
git pull --ff-only origin main
bash run_case_rag_h200.sh s2 --check
bash run_case_rag_h200.sh s2 --limit 5
bash run_case_rag_h200.sh s2
```

`--limit 5` 是真正 5 筆 GPU 煙霧測試，輸出獨立儲存，
預設 run ID 為 `case-rag-s2-core-n5`；
隨後 300 筆正式 Pilot 的 run ID 為 `case-rag-s2-core`。
不必重新微調 Jeff；沿用已有 S2 checkpoint、相同 16 個
BM25 候選及原始 300 筆 Dev。

預設會自動選取 `data/runs/` 底下**完整、指定 loss method、
global update 最大且進度唯一**的 checkpoint。
若有多個同進度 checkpoint，或模型位於不同路徑，
使用明確參數：

```bash
bash run_case_rag_h200.sh s2 \
  --checkpoint data/runs/s2-seed20261112-e2/resume-0001350 \
  --model-dir data/models/Jeff-1.1-Large
```

以上 checkpoint 位置僅為範例；`--check` 會列出本機可用選項。
若這一台使用既有的 H200 別名或不同路徑，可設定
`H200_MODEL_DIR=/實際/Jeff-1.1-Large`，也會自動用其
`tokenizer.json` 與 `chat_template.jinja`。

## 當 H200 尚缺必需資料

```bash
bash run_h200.sh prepare
```

會使用官方 TCM-SD Train、Dev 重建候選與 Jeff Dev 輸入。
若正式 Jeff 31B 權重／環境尚未安裝，執行一次：

```bash
bash run_h200.sh pilot
```

模型初始化可能耗時，使用已有完整模型及 checkpoint 的主機
可以跳過這兩步。Case-RAG 的一鍵程式只使用專案既有
`.venv/bin/python`，避免重新安裝套件改動 GPU 相依版本。

只驗證官方資料和原文來源，不碰 GPU：

```bash
bash run_case_rag_h200.sh s2 --prepare-only
```

它會在 `data/case_rag/` 建立並驗證固定種子兩套實驗輸入：

- `budget2_0300_20261116`：
  `baseline/global2/balanced2/paired2/random2`
- `excerpt2_0300_20261116`：
  `baseline/paired2/querywindow2/randomwindow2/fullcase2`

合計 10 個控制條件，經**整份 Prompt 完全相同**去重後是
8 個獨立輸入；300 筆患者共 2,400 次 Jeff Decision Head
forward，模型只載入一次。對照病歷與原本 Jeff Input 相同，
證據由官方 Train 提供並排除來源同病人／完全同文等群組。
完整病例版本 `fullcase2` 300/300 筆均低於 8192 tokens。

## 如果 S3 也完成

```bash
bash run_case_rag_h200.sh s3 --check
bash run_case_rag_h200.sh s3 --limit 5
bash run_case_rag_h200.sh s3
```

S2／S3 使用相同患者、檢索醫案與對照條件，
依 checkpoint 方法自動使用獨立 run ID。
若 S2 的 checkpoint 正是產生已保存的 Epoch2 Dev logits 的版本，
可以加上 `--require-baseline-parity` 檢查原始決策一致。

## 更完整的方法學反事實控制（選用）

```bash
bash run_case_rag_h200.sh s2 --all
```

包含歷史醫案證型標籤交換、隱藏、展示順序翻轉、
來源警示等 7 組反事實對照。總計 17 條件，
去重後 13 組、300 筆共 3,900 次模型 forward。
*標籤交換*僅作科研負控制，絕不可用於正式臨床決策。

## 推論安全與結果位置

- 嚴格核對 checkpoint 完整性、訓練方法、Jeff 發布版本、
  模型可見 Prompt 的內容指紋。
- 每筆 16-slot raw logits 及耗時／CUDA peak memory 逐筆記錄，
  中斷可從有效前綴繼續；`--limit 5` 使用獨立 run ID。
- 同一 Prompt 的兩組試驗共用 logits 和 timing，避免重算。
- 300 筆 Pilot 的正式研究報告採配對救回／破壞、
  Top-1、候選召回、Train 重複子集；Macro-F1
  同時報固定 148 類與 Pilot 已出現的 56 類供探索，
  **不和完整 Dev 148 類 Macro-F1 直接比較**。
- 原始預測（私有）：
  `data/case_rag/<實驗套件>/scores/<run-id>/`
- 研究摘要：
  `experiments/case_rag_batch_<實驗套件>_<run-id>.json`

研究主結論仍需完整 5,486 筆 Dev 驗證，正式 Test 保持封存。

## 故障時最快的定位

| 提示 | 對應處理 |
|---|---|
| 缺少 `data/candidates/bm25_dev.jsonl` | `bash run_h200.sh prepare` |
| 缺少 Jeff model/tokenizer | `bash run_h200.sh pilot`，或指定 `H200_MODEL_DIR` |
| 找不到 S2/S3 checkpoint | `--check` 查看清單，必要時加 `--checkpoint` |
| 出現多個同進度 checkpoint | 明確指定 checkpoint；避免意外選錯 Seed |
| `run_manifest.json` 不一致 | 使用新的 `--run-id` 分開實驗 |
| GPU 推論中斷 | 重跑原始相同指令，從有一致來源／耗時的有效紀錄續跑 |
| `runner.lock` 被占用 | 等現有 H200 工作結束再執行 |

本機驗證：`bash run_case_rag_h200.sh s2 --check`、
`--prepare-only`、四組 H200 入口單元測試、
Case-RAG 批次／來源原文測試與全專案 pytest。
