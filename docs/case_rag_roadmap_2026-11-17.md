# TCMDiscern Case-RAG 下一步工作清單與研究決策（2026-11-17 鎖定）

主研究紀錄：`docs/case_rag_research_record_2026-11-17.md`。
實驗前協定：`docs/case_rag_jbhi_experiment_lock_2026-11-17.md`（保留原版以追蹤選擇偏差）。
GPU 程式：`run_case_rag_followup_h200.sh`。

## P0 — 目前唯一安排的 GPU 工作：S3/S1 跨模型消融

**狀態：程式與 CPU 測試已完成、待 H200 跑。**

- `S3`：Global2 Label-Hide、Label-Only、Random2，各 5,486 筆。
- `S1`：同上。已完成的 S2 消融、S3/S1 Global2 與全部 Baseline 都重用。
- `Seed 2／3`：明確暫緩。只操作現有 Seed 1 模型的 S3→S1 LoRA/Head，主模型最多載入一次。
- CPU 前檢 `bash run_case_rag_followup_h200.sh --check` 必須通過後才執行正式 runner。
- **完成標準**：`CASE_RAG_FOLLOWUP_COMPLETED`；`backbone_loads=1`（若全部重用則 0）；`new_forwards_planned ≤ 32916`；S1/S3 報告新增三個 control 的 Top-1、Macro-F1、Tail-F1 與配對統計；官方 Test 未讀取。
- **預估**：基於歷史 243 ms/forward、一次權重 43 min、32,916 forwards，中心約 3 小時，規劃抓 **3～3.5 小時**，負載波動可到 4 小時。CPU 預檢可以在前面先跑，用 `planned_new_forwards` 取得真實剩餘工作量。

### H200 指令

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main
bash run_case_rag_followup_h200.sh --check
mkdir -p case_rag_logs
nohup bash run_case_rag_followup_h200.sh > case_rag_logs/followup.log 2>&1 < /dev/null &
echo "PID: $!"
```

## P1 — 完成 GPU 後立即整理（CPU、無模型載入）

1. **歸檔 H200 完成版**：同步三組 `experiments/case_rag_paper_full_dev_{s1,s2,s3}_20261117.json` 至本地；保留 `data/case_rag/paper_5486_20261116/scores/` 的逐病例原始 logits 與 timings，在 private data 下追蹤，避免把病歷推到公開 GitHub。
2. **重新核對內容一致**：三模型 5,486 筆、16 logits、輸入候選順序、資料指紋、Checkpoint／Hybrid FA2、Mean/p50/p95、run 完成狀態；禁止用本地舊 `PENDING_GPU` 結果取代正式 H200 的完成版。
3. **三模型矩陣**：Baseline、BM25、source-label CPU、Global2、Random2、Label-Hide、Label-Only。必須有 Top-1、148 類 Macro-F1、Tail-F1、rescue/break、McNemar、群組 Bootstrap。
4. **證據互補評估**：比較 `Global2 vs Label-Only`、`Global2 vs Label-Hide`、`Global2 vs Random2`，S1/S2/S3 都檢驗方向是否一致。必要時將 S2 當事先固定之主要消融，S1/S3 作探索性跨模型複現；對多重比較給出完整校正策略。
5. **長尾錯誤對照**：逐類列出 148 類樣本數與 F1；聚焦最低頻 37 類、83 筆 gold，辨識 BM25 正確／Jeff Global2 錯誤、反之的病歷與候選差異；同時列出整體 Top-1 與 Macro-F1 的 trade-off。
6. **重複案例敏感度**：除原始 5,486 筆外，再報排除 529 個 Train/Dev 同文後的 4,957 筆，另外盤點 20 個同文但異標註與 19 個患者 ID 重疊；避免把清潔子集視為真正外部測試。
7. **效能報告**：從真實 timings 得出每條 arm 平均、p50、p95、peak VRAM，額外列出一次模型載入耗時，區隔 GPU 決策 latency 與檢索/服務端到端 latency。

**P1 完成標準**：一份可重現的 CPU 統計表、前後一致的報告、長尾逐類分析、異常病例清單與明確方法決策；無需新的大型 GPU 工作。

## P2 — 新技術機制決策（先 CPU，小規模 GPU 僅視結果決定）

核心待解技術問題：**Jeff+Global2 的整體 Top-1 高於 BM25，但 148 類和長尾 Macro-F1 仍輸給 BM25／來源標籤 CPU 基準。**

1. 先評估 **只靠 CPU 能否改善**：從固定的各候選 logits、BM25 rank/score、訓練類別頻率、來源診斷一致度與已檢索醫案的臨床支持特徵中，建立可驗證的可靠度／衝突特徵。先嘗試清楚可重現的固定規則、再做小模型，對照 BM25/Global2/Label-Only；避免直接在全 Dev 上試參數後宣稱無偏效能。
2. 如需擬合參數，使用 **Train OOF** 或適當分組的 fit/cal 切分；Dev 已被大量查看，只能作探索性比較，須清楚標示選模偏差。與已有 `fusion_s1_cpu`、`screen_gate_before_jeff_retraining.py` 既存檢查結果比較，避免重複失敗的方案。
3. 若 CPU 規則無法保留 Top-1 又改善長尾，才設計明確假說的新機制（例如來源標籤可信度校準、矛盾證據診斷、長尾候選證據差分）及必要的最小 GPU Ablation。
4. 評估門檻：**(i)** 與 Global2 相比 Top-1 下降幅度有可接受的預先定義界線；**(ii)** Macro-F1／Tail Macro-F1 有真實增益與不確定度；**(iii)** 排除重複病例仍一致；**(iv)** 推論時間不因新演算法失控；**(v)** 強基準（BM25、來源標籤 CPU）完全保留。若無法同時滿足則保留 Global2 作為主要比較基線，不急於再訓練。

## P3 — 論文寫作（可以現在同步進行）

期刊目標：IEEE Journal of Biomedical and Health Informatics（J-BHI）。目前屬**證據歸因完整、演算法創新仍需建立**的技術研究準備階段。

- **Introduction**：TCM 病例證據如何改善辨證，以及目前 83.45% Top-1 vs BM25 更好的長尾 Macro-F1 之衝突。
- **Methods**：TCM-SD 148 類、BM25 Top16、Jeff S1/S2/S3 的固定 LoRA/Head、Global2 Top2 病例、Random2／Label-Hide／Label-Only、顯存/延遲流程、預檢與續跑。
- **Experimental Setup**：43,180／5,486 官方 Train/Dev，Token 8,192 上限、同一候選集合、個案配對檢定／群組 Bootstrap、長尾 37 類定義、資料品質。
- **Tables/Figures**：資料集與樣本品質、主模型比較、三模型來源歸因、長尾與錯誤表、Train/Dev 同文敏感度、推論成本、決策流程。
- **Discussion**：醫案標籤捷徑、臨床文字增量、重複病例、類別不平衡、全 Dev 開發選擇偏差與部署效率。
- **Novelty & Ablations**：P2 新方法確定且有效後，將其寫作核心技術貢獻；保留純 Global2 為有效基準。
- **Final validation**：全文定稿、實驗協定鎖定後執行一次獨立 Test／外部驗證；必要時補 2～3 Seeds 估計訓練隨機性。此時才評估投稿版本的穩定性。

### 不變的研究原則

- 不填寫尚未取得的測試分數；不要以 Dev 搜尋後的最佳結果充當獨立 Test。
- 比較文獻時必須確認 TCM-SD 切分、K、標籤與 Macro-F1 定義相容。
- 在投稿中如實保留 BM25 高 Macro-F1 的強基準與消融負面結果。
- 團隊後續修改研究協定時新開補充紀錄，不追溯修改原先已鎖定的方法及原始結果。
