# H200 Case-RAG 完整 5,486 筆 Dev：一次載入三個模型

本次固定使用官方 TCM-SD Train／Dev（Train 43,180、Dev 5,486），BM25 Top-16，Jeff-1.1-Large S2／S1／S3 Epoch 2 checkpoint。官方 Test 保持封存。

## 啟動（直接在 H200 上執行）

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main

# CPU：準備完整 Dev Prompt 和兩組來源標籤對照
bash run_case_rag_paper_h200.sh --prepare-only

# CPU：先檢查全部 5,486 筆原始資料、真實 Token IDs、
# 8192 上限、三組 checkpoint、attention/FA2、既有結果及續跑狀態
bash run_case_rag_paper_h200.sh --check

# 必須看到 CASE_RAG_PAPER_ALL_PREFLIGHTS_PASSED 再啟動
mkdir -p case_rag_logs
nohup bash run_case_rag_paper_h200.sh \
  > case_rag_logs/full_dev.log 2>&1 < /dev/null &
echo "Case-RAG PID: $!"
```

追蹤進度：

```bash
tail -n 30 /home/quanta/TCMDiscern-git/case_rag_logs/full_dev.log
```

完成訊號：`CASE_RAG_PAPER_FULL_DEV_COMPLETE`，其中 `backbone_loads` 必須為 1。

## 真正執行的 GPU 條件

| Checkpoint | 新推論 | 參考資料 |
|---|---|---|
| S2 | Global2、Random2、Global2 隱藏來源標籤、Global2 僅來源標籤 | 重用既有完整 Dev S2 logits |
| S1 | Global2 | 重用既有完整 Dev S1 logits |
| S3 | Global2 | 重用既有完整 Dev S3 logits |

最多 6 × 5,486 = **32,916 個新 forward**，約數小時；不重跑原始 Jeff 的 16 槽 Baseline。單次載入 Jeff 31B（過往約 43 分鐘），後續僅切換 LoRA／Head；已完成的 logits 與同步 latency 逐行續跑，斷線後可使用同一命令繼續。

## 實驗保護機制

- **預先拒絕截斷**：用 pinned Jeff tokenizer 逐筆重新取得真實 input IDs，檢查 Token 長度（8192 以內）；H200 上的 inference 使用 `long_input=refuse`，每次 forward 再確認 `shortened=false`、無思考生成。
- **預先驗證注意力**：先核對每組 manifest 的 `axolotl_hybrid_fa2_sdpa`、released metadata 及 FA2 kernel，與原始訓練載入程式完全相同；檢查在讀 31B 權重前完成。
- **固定資料與頭部**：所有輸入保持原始 Dev 5,486 順序與 16 個 BM25 候選；S2／S1／S3 每個 checkpoint 使用訓練時的 LoRA、FP32 Head 與 probe logits，切換前重設先前 adapter。
- **檢查續跑完整性**：每一 arm 的 `run_manifest.json` 鎖定模型來源、checkpoint 與 Prompt 指紋；在載入大模型前檢查已完成及部分輸出的 logits／timings。未完成記錄恢復為最短一致前綴。
- **安全性**：查詢 gold 僅用於評測；資料由 Train 檢索；所有官方 Test 維持封存；flock 避免同一 GPU runner 重複啟動。

## 結果

各組報告（CPU 自動算分）：

- `experiments/case_rag_paper_full_dev_s2_20261117.json`
- `experiments/case_rag_paper_full_dev_s1_20261117.json`
- `experiments/case_rag_paper_full_dev_s3_20261117.json`

報告包含 Top-1、148 類 Macro-F1、37 個低頻證型 Macro-F1、群組 Bootstrap、McNemar、原始 Jeff／BM25／來源標籤的 CPU 基線及模型推論延遲。原始 logits 和每筆 latency 保存在 `data/case_rag/paper_5486_20261116/scores/`。

允許透過純 CPU 模式重新產生評分：

```bash
.venv/bin/python -m scripts.evaluate_case_rag_paper_dev
```

提醒：`--check` 是 GPU 前的正式檢查關卡；在不同硬體執行未經驗證的完整推論前，請確認所有 PASS。基於資料集已有 Train/Dev 重複病例，正式報告需同時比較排除 529 筆同文案例的結果。
