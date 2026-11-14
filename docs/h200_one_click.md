# H200 一鍵部署與實驗執行

此文件說明原證型 Top-16 執行器。2026-11-13 三任務的優先實驗規範為 [中醫病名、證型與聯合診斷](joint_disease_syndrome_experiment_protocol.md)，CPU 來源審查依 [完整資料稽核](h200_final_data_and_paper_review_2026-11-12.md)。

## 1. 一鍵入口

在 Linux H200 主機取得包含 `run_h200.sh` 的程式碼，然後執行：

```bash
cd TCMDiscern
bash run_h200.sh prepare
bash run_h200.sh micro          # 快速四層實體 LoRA／Checkpoint 測試
bash run_h200.sh micro --deep   # 保留 31B 的 60 層注意力排列，但矩陣極小
bash run_h200.sh pilot          # 先自動執行深度 Micro，再載入 31B
bash run_h200.sh full --confirm-full
```

- `prepare`：自動建立 Python 3.12 開發虛擬環境（使用 frozen `uv.lock`），僅下載官方 TCM-SD Train/Dev，重建 **43,180 筆 Group-aware OOF BM25**、**5,486 筆 Dev BM25**、S1 歷史安全單正解資料、S2/S3/S4 多正解 K=16，以及 Dev K=5/16/17，最後全部逐筆核對。正式訓練 S1 會使用多正解資料的**相同候選題目**並只保留單金標 Loss，確保方法比較的題目相同。已存在的私有資產會重用，但驗收仍需重新檢查；不通過立即停止。沒有下載、訓練 31B 模型。
- `micro`：**快速端到端 GPU 驗收**。先讀取 Jeff-1.1-Large 的 `config.json` 與 `jeff_config.json`，用四層隨機初始化的 Gemma 4（含全域／滑動注意力、K=V、Shared-KV、PLE）和真實原始 Train 題目，執行 Jeff 官方 Prompt／16-slot Head、LoRA Step-0、S3 Set Loss、2 次反向傳播與 Optimizer 更新、儲存 PEFT/Head/Optimizer、重新載入比對 logits。使用既有 `.venv`，不下載模型、**不讀取 31B 權重**，結果記錄於 `data/h200/micro_smoke.json`。可先用 `bash run_h200.sh micro --metadata-only` 只查注意力模式。加上 `--deep` 時保留官方設定的 Decoder 深度、各層 Full/Sliding Attention 排列及 K/V 共用層數，額外使用較長序列（最高 256 tokens）進行同樣的兩次參數更新和 Checkpoint 重載，報表寫入 `data/h200/micro_deep.json`。正式 Pilot 在載入 31B 前自動執行深度 Micro。這是軟體整合驗證，隨機小模型的分數不可用於論文。
- `pilot`：Driver R580+ 使用 Jeff 官方測試的 torch 2.13.0/CUDA 13.0；Driver R525～R579 選擇獨立 PyTorch 2.11.0+cu128/CUDA 12.8（支援 Linux ARM64/GH200）。只修改專案 `.venv`，先驗證 CUDA BF16，再取得 Jeff v1.3.0 模型/Head。**整個證型 Pilot 只讀取 31B 權重一次**：在同一 Python 工作階段連續驗證注意力、兩筆原版 Dev、LoRA Step-0、2 次真正的 S3 Optimizer 更新（累積 64，128 筆曝光）、保存 Optimizer/Head/Adapter，最後卸下訓練 Adapter 並從 Checkpoint 重新載入做兩筆 Dev 推論。任何 checkpoint 遺失、已有 logits 與目前後端不一致，會安全停止。CUDA 12.8 屬相容性配置，應在論文記錄與上游 CUDA 13.0 的版本差異。
- `screen --confirm-screen`：先驗證 Pilot，使用單一**常駐 GPU Python 工作階段**完成原版 Zero-shot 5,486 Dev 與第一個 Seed 的 S1/S2/S3 各兩 Epoch 微調及 Epoch 1/2 完整 Dev 推論。各組開始前重新載入原版 Head，上一組 LoRA 直接卸載、不合併至 Backbone；評估則重新載入指定訓練 checkpoint。
- `full --confirm-full`：沿用上述常駐 GPU 流程，執行 9 組 S1/S2/S3 × 3 Seeds × 2 Epoch、各 1,350 Updates；每組在 675 與 1350 Updates 評估完整 Dev，依完成標記跳過已驗證階段。選模前核對全部 Dev 報表都使用同一已驗證的注意力後端，然後才產生 `experiments/jeff_h200_dev_selection.json`，按三 Seed 平均 Accuracy 選方法與 Epoch。**正式 Test 封存。**
- `full --confirm-full --include-exploratory`：額外增加 S4、S3 的 Train OOF 困難病例 ×2 加權，各單 Seed、1 Epoch（探索性比較）。
- `status`：只讀階段狀態；`plan`：列印所有方法與計算預算；`full --dry-run`：列印全部訓練／評估指令，不下載、不啟動 GPU、不寫資料。這三個模式可在未安裝 PyTorch 的機器使用。

若 Hugging Face 對模型設有存取同意，必須先用同一使用者完成模型存取授權，並在 H200 端執行 `hf auth login`；部署腳本不保存金鑰至程式庫。

**GH200 Driver 550 / CUDA 12.8 的實際失敗紀錄：** 31B 權重完整載入後，預設 SDPA 的前綴不變性檢查有 82,308 個 mismatch，切換 Eager 仍有 112,791 個 mismatch，兩種後端都中止於訓練前。新版先由輕量設定檔判斷模型預期是因果或雙向注意力；只有真正宣告雙向的發布模型才採用重複推論一致性檢查，因果模型則仍嚴格要求前綴不受後綴影響。下一次完整 31B 測試前，先讓 `micro` 重現同架構的注意力和 LoRA 路徑；如果小模型失敗，就直接停在前置驗收。若 31B 仍失敗，會在**已載入模型的同一個程序**再次檢查每層輸出，記錄首個違反因果前綴的 Decoder 層、層類別與特徵差異，供後續定位。Gemma 4 31B 的全域層使用 K=V、後段 Shared-KV 層連 k_proj/v_proj 也沒有；LoRA 逐層按 Transformers 5.17 真實模組配置驗證，僅作用於文字 Decoder，保留 vision tower 原始權重。已下載模型和 CPU 資產保留；Pilot 中斷後可原命令續跑。此修正仍須經目標 GH200 重新測試驗證。

## 2. 目錄及架構

| 位置 | 責任 |
|---|---|
| `run_h200.sh` | GPU 模式、快速 `micro` 預檢、唯一執行鎖、正式 31B 入口 |
| `scripts/smoke_tiny_jeff.py` | 四層 Gemma 4 / Jeff 實體 LoRA、Loss、Optimizer、Checkpoint 端到端驗收 |
| `src/tcmdiscern/jeff_release_attention.py` | 在載入權重前檢查官方 Jeff/Gemma 4 注意力設定 |
| `scripts/h200.py` | CPU/GPU 命令路由，全面禁用未確認的大規模訓練 |
| `src/tcmdiscern/h200/plan.py` | 不可變的 Methods、Seeds、Epoch 和完整 Updates 規格 |
| `src/tcmdiscern/h200/artifacts.py` | Clone 後重建來源資料／候選／Prompt，避免覆寫 |
| `src/tcmdiscern/h200/workflow.py` | 單程序 Stage 日誌、結果完整性、續訓及自動評估 |
| `src/tcmdiscern/h200/session.py` | GH200 常駐 Backbone、隔離每組 LoRA/Head、避免重複載入權重 |
| `scripts/run_jeff_pilot_once.py` | 一次載入完成證型 Pilot、真實 2 次更新、權重變化驗證及 Checkpoint 重載 |
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
- **只有 GH200 能驗證：** 已知 Driver 550/CUDA12.8 能載入完整 31B 權重，但先前的預設 SDPA 前綴因果檢查失敗；新的 eager 回退、全量 BF16 Backward、真正 LoRA/Head 梯度、Checkpoint 重載、GPU 峰值顯存及實際效能須在 H200 再次驗證。
- 模型來源與提示格式對齊 Jeff 官方 v1.3.0 和〈Releasing the Inner Jev〉所述 16-slot Decision Head、BF16 Backbone、LoRA rank32/alpha64。GH200 CUDA12.8 分支是相容性配置；每個模型輸出包含 `attention_backend`，後續彙整會拒絕混用後端。模型是否可完成單張 GH200 微調，仍以實際 Pilot 為準。

## 5. 直接查問題

```bash
bash run_h200.sh status
bash run_h200.sh plan
bash run_h200.sh full --dry-run
```

每個階段失敗記錄 `data/h200/state/<階段>.json`，詳細 stdout/stderr 留在對應 `data/h200/logs/<階段>.log`。用同一命令重新執行，保留紀錄供審查，不自行抹除。
