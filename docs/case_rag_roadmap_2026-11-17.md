# TCMDiscern Case-RAG 後續研究路線圖（2026-11-17，CPU-first 修訂）

**目前優先順序：P0 已執行部分 CPU 分析 → P1 取得 H200 現有 Global2 logits 再做 CPU 配對分析 → P2 決定技術方法與最小必要 GPU 消融 → P3 寫論文／獨立驗證。**

主研究紀錄：`docs/case_rag_research_record_2026-11-17.md`
CPU 第一輪結果：`docs/case_rag_cpu_first_2026-11-17.md`；CPU 分組路由與最高歷史成績核對：`docs/case_rag_cpu_router_2026-11-17.md`
最初事先鎖定實驗：`docs/case_rag_jbhi_experiment_lock_2026-11-17.md`（原文保留以追蹤決策歷程）
GPU 跨模型補強腳本：`run_case_rag_followup_h200.sh`（**目前暫緩**）

## P0 — CPU 已完成的原始模型逐病例分析

**狀態：完成。**

- 直接使用本地完整 Dev S1/S2/S3 原始 Jeff logits、BM25 Top-16 與 Global2 來源標籤，分析 5,486 筆病例。
- 重算 Top-1／148 類 Macro-F1／37 尾類 Macro-F1，與原 H200 基準成績一致。
- 將 gold 依 Train 類別頻率固定分桶，證實低頻 1–199 筆/類的 S3 原始模型明顯弱於 BM25；高頻 200+ 的 Jeff 較佳。
- 分析 S3 Jeff 與 BM25 各自獨對：485／457 筆；兩者挑對的一方 Oracle 上限 88.4251%，僅為互補診斷。
- 結果：`experiments/case_rag_cpu_first_2026-11-17.json`；可重現程式：`scripts/analyze_case_rag_cpu_first.py`。
- 新增 5 折分組 Logistic 路由（Dev 開發性分析）：S3 原模型 + BM25 在 threshold 0.5 得 **84.543% Top-1 / 58.376% Macro-F1 / 44.659% Tail-F1（37 類）**，紀錄在 `experiments/case_rag_cpu_router_2026-11-17.json`。
- **全面重估強基準：先前 2026-11-16 完整 Dev 的 Joint+Jeff S2 達 85.180% Top-1；SGD+Jeff S1 達 60.960% Macro-F1（該組 Top-1 85.071%）。** 下一方法要和這兩項正式比較。

## P1 — H200 現有 Global2 logits 的 CPU 分析（現在要做）

**狀態：等待取得已完成的原始 logits；不需要訓練，也不需要 Jeff 31B GPU 載入。**

H200 上已有 S1/S2/S3 的 Global2 完整 5,486 筆 logits，及 S2 的 Random2／Label-Hide／Label-Only。原始聚合成績已在主研究紀錄。

直接在 H200 執行：

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main
.venv/bin/python -m scripts.analyze_case_rag_cpu_first --strict-global2
.venv/bin/python -m scripts.analyze_case_rag_cpu_router --strict-global2
```

報告輸出 `experiments/case_rag_cpu_first_2026-11-17.json`。分析下列固定問題：

1. **S3 + Global2 vs BM25／來源標籤**：各自獨對、雙對、雙錯、Oracle 上限及排除 Train 同文後的結果。
2. **頻率偏差是否改善**：依 Train gold 頻率 1–9、10–49、50–199、200+ 檢查 S3 + Global2。確認 S3+Global2 是否仍系統性偏向高頻。
3. **尾類候選衝突**：BM25 預測尾類但 Global2 不預測時，BM25／Global2 誰比較常正確；檢查單純依頻率切換是否有淨優勢。
4. **來源歸因差分**：S2 Global2 vs Random2／Label-Hide／Label-Only 的逐病例救回／破壞與類別差異。
5. **統計與資料品質**：完整與排除 529 筆同文後的 Top-1、配對 McNemar、群組 Bootstrap；避免把 Dev 上的擇優規則當成獨立驗證。

**H200 進度補充（2026-11-17）**：六模型 Global2 CPU 路由已完成。最高路由 Top-1 為 **S3 Base 84.543%**；Global2 的最高路由 Macro-F1 為 **S1 58.800%**。兩者仍低於既有完整 Dev 強基準的 85.180% Top-1 與 60.960% Macro-F1。H200 的 `analyze_case_rag_cpu_first --strict-global2` 曾遇到 `s3_base_vs_bm25` KeyError，原因已修正並測試。

**P1 尚待確認**：在 H200 拉取最新修正後，重新執行 `analyze_case_rag_cpu_first --strict-global2`，取得三模型 Global2 的逐病例 rescue/break、Oracle、Train 分桶與固定頻率保護成績。保存完成的 H200 原始 JSONL 與聚合報告；新 GPU 工作維持暫緩。

## P2 — 決定新方法與是否需要下一批 GPU（條件式）

**先設計 CPU 規則，再決定 GPU 是否值得執行。**

- 比較 Jeff 決策 logits/margin、BM25 排名與候選差距、來源病例證型一致度、Train 類別頻率與臨床描述的證據衝突。
- 先評估固定規則或簡單、可重現的路由器，在 Train OOF 或分組 fit/cal 上訓練與調參。Dev 已反覆用於方法研究，只能作探索性結果。
- 對照 BM25、source-label CPU、S3+Global2、S2+Global2，並納入 **既存 Joint+Jeff S2 85.180% Top-1 與 SGD+Jeff S1 60.960% Macro-F1** 作更強的完整 Dev 比較；追求維持整體 Top-1 優勢，同時縮小低頻 Macro-F1 落差。
- 若 P1 顯示需要跨模型證據歸因，才執行已寫好的 `run_case_rag_followup_h200.sh`：S3/S1 各三個消融，共最多 **32,916** 新 forward，**單次** Jeff 31B 權重載入；歷史量測估計約 3～3.5 小時。
- 若無額外科研價值，取消這六組 GPU 重複驗證，把算力留給真正新方法。**Seed 2／Seed 3 的訓練繼續延後**。

P2 完成標準：有清楚技術創新、強基準與必要消融；整體 Top-1／Macro-F1／長尾改善經配對統計與重複病例敏感度驗證。

## P3 — IEEE J-BHI 論文寫作與最後驗證

1. Introduction：從 S3+Global2 的 Top-1 **83.45%** 與 BM25 長尾優勢切入臨床可靠度問題。
2. Related Work：TCM-SD 使用者、傳統與神經檢索、Jeff / LoRA、Case-RAG、長尾辨證與證據衝突方法；標註可比切分與指標。
3. Methods：BM25 K16 + Jeff 原模型 + Global2、病例證據文本與診斷標籤分離、後續新方法及其演算法細節（取得證據後寫入）。
4. Experiments：Train 43,180／Dev 5,486／148 證型、Train 低頻 37 類、K16 固定候選、同一 checkpoint、Token 8,192 上限、Hybrid FA2/SDPA；未讀取 Test。
5. Results：主基準表、S1/S2/S3 Global2 增益、來源標籤/臨床文字消融、逐類/長尾分析、配對差異／Bootstrap、排除 529 同文後結果、推論成本／顯存。
6. Discussion：來源標籤捷徑可能性、臨床證據歸因、樣本重複、Dev 方法選擇偏差、尾類僅 83 gold 所帶來的不確定性。
7. 正式方法/實驗協定鎖定後，才進行一次獨立 Test／外部資料評估，再決定是否補 Seed 2/3 作訓練穩定性驗證。

**論文結論需以已確認數據為準，尚未完成的方法結果保持待測，真實方法創新與可重現性優先於增加無效 GPU 實驗組數。**
