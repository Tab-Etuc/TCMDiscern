# H200 一鍵部署與實驗執行

此文件說明原證型 Top-16 執行器。2026-11-13 三任務的優先實驗規範為 [中醫病名、證型與聯合診斷](joint_disease_syndrome_experiment_protocol.md)，CPU 來源審查依 [完整資料稽核](h200_final_data_and_paper_review_2026-11-12.md)。

## 1. 一鍵入口

在 Linux H200 主機取得包含 `run_h200.sh` 的程式碼，然後執行：

```bash
cd TCMDiscern
bash run_h200.sh prepare
bash run_h200.sh pilot
bash run_h200.sh full --confirm-full
```

- `prepare`：自動建立 Python 3.12 開發虛擬環境（使用 frozen `uv.lock`），僅下載官方 TCM-SD Train/Dev，重建 **43,180 筆 Group-aware OOF BM25**、**5,486 筆 Dev BM25**、S1 單正解、S2/S3/S4 多正解 K=16，以及 Dev K=5/16/17，最後全部逐筆核對。已存在的私有資產會重用，但驗收仍需重新檢查；不通過立即停止。沒有下載、訓練 31B 模型。
- `pilot`：準備資料並安裝 Jeff v1.3.0、官方 CUDA 13.0 版依賴及 PEFT 0.21.1。使用官方釘選 revision 的模型權重、頭部和 Tokenizer；依序執行原始模型 2 筆 Dev、安全注意力檢查、LoRA Step-0 一致性、1,024 次病例曝光的 16-update Pilot（accumulation 64）。無法通過 GPU/版本/梯度/續訓驗證即停止。
- `full --confirm-full`：先確保 Pilot 完成；跑完正式 Zero-shot 5,486 Dev；執行 9 組 S1/S2/S3 × 3 Seeds × 2 Epoch、各 1,350 Updates；每組在 675 與 1350 Updates 的 checkpoint 對官方 Dev 完整推論。最後自動產生 `experiments/jeff_h200_dev_selection.json`，按三 Seed 平均 Accuracy 選方法和 Epoch、Macro-F1 平手判定。**正式 Test 保留封存，不自動測試。**
- `full --confirm-full --include-exploratory`：額外增加 S4、S3 的 Train OOF 困難病例 ×2 加權，各單 Seed、1 Epoch（探索性比較）。
- `status`：只讀階段狀態；`plan`：列印所有方法與計算預算；`full --dry-run`：列印全部訓練／評估指令，不下載、不啟動 GPU、不寫資料。這三個模式可在未安裝 PyTorch 的機器使用。

若 Hugging Face 對模型設有存取同意，必須先用同一使用者完成模型存取授權，並在 H200 端執行 `hf auth login`；部署腳本不保存金鑰至程式庫。

## 2. 目錄及架構

| 位置 | 責任 |
|---|---|
| `run_h200.sh` | 唯一外部啟動入口；建立 Python、GPU 套件、取得 Jeff 原版及模型 |
| `scripts/h200.py` | CPU/GPU 命令路由，全面禁用未確認的大規模訓練 |
| `src/tcmdiscern/h200/plan.py` | 不可變的 Methods、Seeds、Epoch 和完整 Updates 規格 |
| `src/tcmdiscern/h200/artifacts.py` | Clone 後重建來源資料／候選／Prompt，避免覆寫 |
| `src/tcmdiscern/h200/workflow.py` | 階段日誌、結果檢查、續訓、回復與自動評估 |
| `src/tcmdiscern/h200/results.py` | 三 Seed 完整 Dev 指標彙整與預定選模 |
| `scripts/audit_h200_payload.py` | 本機/H200 重複執行的全量病例來源比對 |
| `scripts/audit_jeff_training_controls.py` | 全量 A～P 位置、S1/S3 配對、多正標、OOF 難度驗證 |
| `src/tcmdiscern/jeff_train_loop.py` | 保持既有 H200 梯度／Optimizer／Checkpoint 訓練核心 |
| `scripts/run_jeff_zeroshot.py` | 真實發布模型及指定 Checkpoint 的 Dev 推論與打分 |

所有病例及逐筆 logits、Adapter、Optimizers 都在 `data/`（Git 忽略），僅可公開不含病例的彙總。Run 元資料與日誌在 `data/h200/state`、`data/h200/logs`。

## 3. 重跑、暫停與資料遺失

反覆執行同一個 `bash run_h200.sh full --confirm-full` 會依完成標記和完整輸出驗證跳過已通過階段。若訓練被中斷，先確認 `status`，原命令會自動選最新完整 `resume-NNNNNNN`；檔案沒有完成保存或設定不同，會停止並提示。不得直接刪除、編輯或覆寫 `data/runs/` 下的續訓檔。

**限制：** 如果 process 在最近一次完整 checkpoint 後失敗，已完成但尚未 checkpoint 的 Optimizer 更新需要從上一個已保存 checkpoint 重做；依確定性 Seed 和 Cursor 保持同一資料順序。若出現 OOM、NaN、檔案缺漏、釘選模型版本不符或前綴遮罩失敗，將中止長訓練。重新下載或轉移模型時應確認磁碟容量及 Hugging Face 權限。

## 4. 重要驗證界線

- 本機可驗證：官方 Train/Dev 全量列數與每列來源、Group-aware OOF、S1～S4 金標／候選／Prompt、A～P 排列、K=1～35 模擬打分及研究統計。
- **只有 H200 能驗證：** 31B merged backbone + FP32 Decision Head 真正 Forward、BF16 backward、LoRA 實際更新、Optimizer/Checkpoint 續訓、GPU 峰值記憶體、全套 6 次訓練的耗時與成績。
- 目前 H200 安裝／模型下載命令已按 Jeff v1.3.0 原版 requirements 編排，但仍待第一次在目標主機上從乾淨 clone 實際通過。**不要把 CPU 模擬測試當成 GPU 上線驗證。**

## 5. 直接查問題

```bash
bash run_h200.sh status
bash run_h200.sh plan
bash run_h200.sh full --dry-run
```

每個階段失敗記錄 `data/h200/state/<階段>.json`，詳細 stdout/stderr 留在對應 `data/h200/logs/<階段>.log`。用同一命令重新執行，保留紀錄供審查，不自行抹除。
