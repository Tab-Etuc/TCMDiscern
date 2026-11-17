# Case-RAG 下一批 H200：共用一次 Jeff 權重載入

日期：2026-11-17

## 決策

已完成的完整 5,486 筆 Dev：S1/S2/S3 的原始 Jeff、Global2；S2 的 Random2、Global2 Label-Hide、Global2 Label-Only。

**下一批只補 S3、S1 的跨模型來源歸因驗證**：各執行 Random2、Global2 Label-Hide、Global2 Label-Only，共六個條件、最多 32,916 個新 forward（每個模型 16 個候選）。優先順序：S3（目前最高完整 Dev Top-1）→ S1（Pilot 最大改善）。共享單個 resident Jeff 31B GPU session，跨 checkpoint 卸載前一組 LoRA 並還原訓練 Head。

Seed 2/3 訓練與重新微調先暫緩，這個批次只使用已存在的 Seed 1 S1/S3 checkpoint。

## H200 啟動

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main

# 唯讀 CPU/注意力前檢：三組原始報告、每個病例、Prompt tokens、
# CUDA FA2、checkpoint、之前已完成/部分完成的 logits/timings
bash run_case_rag_followup_h200.sh --check

# 輸出 CASE_RAG_FOLLOWUP_PREFLIGHT_PASSED 後執行
mkdir -p case_rag_logs
nohup bash run_case_rag_followup_h200.sh > case_rag_logs/followup.log 2>&1 < /dev/null &
echo "PID: $!"
```

進度與中斷後續跑：

```bash
tail -n 40 /home/quanta/TCMDiscern-git/case_rag_logs/followup.log
```

要續跑，直接再執行上述 `nohup bash ...` 命令。runner 根據完整檔與部分前綴計算剩餘 forward；完成的 arm 整組跳過，所有工作完成時跳過整個 31B 權重載入。

完成標記：`CASE_RAG_FOLLOWUP_COMPLETED`，`backbone_loads: 1`（若完全續跑完成則為 0）。

## 測試與保存

- `run_case_rag_followup_h200.sh` 使用與原本全 Dev 相同的 GPU 排他 lock、PINNED hybrid FA2/SDPA attention、Python environment。
- `scripts.run_case_rag_paper_h200.preflight` 再驗證原始三組完整 Dev checkpoint、既有 logits、所有 frozen Prompt 真實 token ID（最大 token < 8,192）。
- 所有重用、續跑、原始身份標記位於 `data/case_rag/paper_5486_20261116/scores/paper-{s1,s3}-full/`；輸出是 JSONL 一例一筆、原生 16 個 logits。
- 完成後更新 `experiments/case_rag_paper_full_dev_s1_20261117.json` 與 `experiments/case_rag_paper_full_dev_s3_20261117.json`，包含跨模型消融的 Top-1、148 類 Macro-F1、末 37 類 F1、配對 McNemar、群組 Bootstrap、耗時。
- 不依靠 Train/Dev gold 構造模型輸入；未使用官方 Test。

下一步可用已保存 logits 做 **CPU** 配對差異、長尾診斷和來源標籤敏感度分析，先決定是否有理由引入新演算法後，再設計下一批 GPU 計算。
