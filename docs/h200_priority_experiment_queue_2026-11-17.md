# H200 優先實驗一鍵執行（2026-11-17）

## 一條指令

將本地 TCMDiscern 的程式變更同步到 H200 的專案資料夾後：

```bash
cd /home/quanta/TCMDiscern-git
bash run_h200_priority_experiments.sh --dry-run
bash run_h200_priority_experiments.sh
```

建議在 `tmux`/終端機中執行。腳本按以下順序運作：

| 優先 | 實驗 | 既有資料可重用 | 新 GPU 計算 |
|---|---|---|---|
| 0 | 固定 SGD+Jeff 的先驗、Gate、Pairwise 消融及 0–100% Jeff 呼叫率曲線 | 是 | 0 |
| 0 | 固定強 SGD+Jeff 的 Case-RAG 全條件對照 | 是 | 0 |
| 1 | H200 實際選擇性呼叫 20% 病例（原 S1 checkpoint）與 logits 同步驗證 | 是 | 1,097 次 forward |
| 2 | Head-only S1（1-step smoke + 2 Epoch） | 否 | 新訓練 |
| 2 | LoRA-only S1（1-step smoke + 2 Epoch） | 否 | 新訓練 |
| 3 | Random-negative S1（1-step smoke + 2 Epoch） | 否 | 新訓練 |
| 5 | S1 第二與第三 Seed（2 Epoch） | 部分 | 新訓練 |
| 6 | S2/S3 第二與第三 Seed（2 Epoch） | 部分 | 新訓練 |
| 最後 | 彙整對照表與 CPU 效率報告 | 是 | 0 |

執行 `bash run_h200_priority_experiments.sh --skip-seeds` 可只執行優先實驗；預設會依序完成全部。

## 必要前提

- 使用現有釘選的 H200 Python 3.12 `.venv`、Jeff v1.3 與 31B 權重。
- 已完成第一 Seed 的 S1/S2/S3 兩個 Epoch，以及官方 Dev K=16 logits。
- 原始 `data/runs/s1-seed20261112-e2/resume-0001350` 可載入。
- 官方 Test 需維持封存。此腳本只讀 Train/Dev。
- 新的 Random-negative 正例、正例位置、同文已知正例集合完全保留；僅抽換負例證型。
- 所有訓練使用相同 `43,180 Train × 2 Epoch` 與 1,350 optimizer updates，獨立 run-id。
- H200 已有的 GPU runner lock 會阻止本腳本與其他 H200 研究同時寫入 checkpoint。
- 若已有尚未完成的同名 run-id，從最後完整 checkpoint 續跑；已完成項目跳過。版本／run 設定不符則中止，供人工核對。
- 新訓練的 micro-batch、注意力後端、LoRA rank/alpha、學習率與 optimizer 超參數對齊已完成的原始 S1 訓練設定；如不一致，GPU 訓練前中止。
- 每次啟動會在當前 H200 Python/NumPy/scikit-learn 環境重新計算 SGD 路由索引，避免沿用不同環境的數值結果。歷史 Case-RAG 若缺資料則標記為可選階段，核心 GPU 訓練可繼續。
- 實際 GPU 20% 推論每筆儲存無病歷內容的進度紀錄，可以從中斷前最後一筆接續；每一筆都驗證與原始 S1 Dev logits 的 Top-1 一致。
- 確認已完成的結果時，會驗證訓練設定、epoch/checkpoint、Dev 原始 logits 5486 筆及模型來源資訊；檔案存在本身不足以判定完成。
- 原本所有 Seed 跑完而延遲測試中斷時，續跑仍會執行遺漏的延遲測試。
- `H200_MODEL_DIR` 沿用既有 `run_h200.sh` 的模型權重設定；未指定時會依序尋找目前 Git checkout、`TCMDISCERN_LEGACY_ROOT` 與相鄰 `TCMDiscern-main` 的原始 Jeff 權重。
- H200 曾發生 `ModuleNotFoundError: tcmdiscern.case_rag`：啟動器現在強制將此 Git checkout 的 `src/` 放在 `PYTHONPATH` 最前面並在 CPU 計算前檢查來源；核心 CPU 消融則已解耦 Case-RAG 歷史實驗匯入，使用獨立且已和固定基準逐函式比對的評分／SGD 常數。
- 原始第一 Seed 的 LoRA、Head 與完整 Dev logits 可以仍留在相鄰 `TCMDiscern-main`；腳本會自動尋找與驗證，所有新優先實驗仍寫在當前 `TCMDiscern-git/data/`。

## 主要輸出

- `experiments/h200_priority_cpu_ablation.json`：CPU 消融、固定 148 類 Macro-F1、37 個低頻類別及 Jeff 呼叫率曲線。
- `experiments/h200_priority_real_gpu_latency.json`：實際呼叫 20% Dev 病例的 mean/p50/p95、token 數、原 checkpoint logits parity。
- `experiments/jeff_priority-..._k16_dev.json`：各個訓練消融在完整 Dev 上的結果。
- `experiments/h200_priority_experiments_summary.json`：一份完整比較資料表。
- `data/h200/priority_experiment_queue.json`：進度與出錯原因。
- `data/runs/priority-*/`：可繼續訓練的 optimizer/adapter/Head checkpoint。

## 重要研究解讀

固定 gate 最強基準於本地 Python 3.12.13：Top-1 85.0711%、148 類 Macro-F1 60.9597%；同機 H200 Python 3.12.3 已測到 Top-1 85.1805%、Macro-F1 61.2118%。跨環境差異需分開紀錄。所有新方法應和**同一台 H200、同一個固定規則的 Base** 比較。

目前的 20% 選擇性呼叫於 Dev 得到 Top-1 85.1440%、Macro-F1 61.0744%。這是 Dev 探索性結果；CPU 評估中的低信心排序使用已保存 Jeff logits 模擬選擇性計算。新增 GPU 延遲階段才會實際只呼叫所選的 1,097 筆。

本腳本完成 Dev 階段後產出研究比較資料。**正式 Test 需在方法選定、決策規則封存後，再啟動獨立評測**，以免在多輪研究中將 Test 變成調參依據。

以上 CPU 消融優先，隨後針對微調內部的 Head/LoRA 及困難負例做因果消融；種子複驗順序在最後。
