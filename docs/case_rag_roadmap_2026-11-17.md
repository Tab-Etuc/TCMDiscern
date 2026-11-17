# TCMDiscern Case-RAG 後續研究路線圖（2026-11-17，CPU-first 修訂）

**目前優先順序：P0／P1 已完成所有 H200 logits 本地匯入、全量 CPU 消融與強融合重現 → P2 使用 Train OOF／固定 fit-cal 研發新的證據可靠度方法 → P3 論文／最終獨立驗證。**

主研究紀錄：`docs/case_rag_research_record_2026-11-17.md`
CPU 第一輪結果：`docs/case_rag_cpu_first_2026-11-17.md`；CPU 分組路由與最高歷史成績核對：`docs/case_rag_cpu_router_2026-11-17.md`；**完整 H200 聚合結果與新 CPU 強基準**：`docs/case_rag_cpu_complete_review_2026-11-17.md`
最初事先鎖定實驗：`docs/case_rag_jbhi_experiment_lock_2026-11-17.md`（原文保留以追蹤決策歷程）
完整壓縮包匯入及所有 CPU 分析：`docs/case_rag_h200_archive_complete_2026-11-17.md`。GPU 跨模型補強腳本：`run_case_rag_followup_h200.sh`（**暫緩**）

## P0 — CPU 已完成的原始模型逐病例分析

**狀態：完成。**

- 直接使用本地完整 Dev S1/S2/S3 原始 Jeff logits、BM25 Top-16 與 Global2 來源標籤，分析 5,486 筆病例。
- 重算 Top-1／148 類 Macro-F1／37 尾類 Macro-F1，與原 H200 基準成績一致。
- 將 gold 依 Train 類別頻率固定分桶，證實低頻 1–199 筆/類的 S3 原始模型明顯弱於 BM25；高頻 200+ 的 Jeff 較佳。
- 分析 S3 Jeff 與 BM25 各自獨對：485／457 筆；兩者挑對的一方 Oracle 上限 88.4251%，僅為互補診斷。
- 結果：`experiments/case_rag_cpu_first_2026-11-17.json`；可重現程式：`scripts/analyze_case_rag_cpu_first.py`。
- 新增 5 折分組 Logistic 路由（Dev 開發性分析）：S3 原模型 + BM25 在 threshold 0.5 得 **84.543% Top-1 / 58.376% Macro-F1 / 44.659% Tail-F1（37 類）**，紀錄在 `experiments/case_rag_cpu_router_2026-11-17.json`。
- **全面重估強基準：先前 2026-11-16 完整 Dev 的 Joint+Jeff S2 達 85.180% Top-1；SGD+Jeff S1 達 60.960% Macro-F1（該組 Top-1 85.071%）。** 下一方法要和這兩項正式比較。

## P1 — H200 歷史 logits 匯入、原始與強融合 CPU 比較（已完成）

**狀態：完成。** H200 全量壓縮檔已進入本地 TCMDiscern，五個固定輸入的 SHA-256 與本地完全一致；已補入 9 個完整 Case-RAG 推論條件及逐病例 timings，保留原有程式與私人病歷。完整記錄：`docs/case_rag_h200_archive_complete_2026-11-17.md`。

- 已重算原始 Jeff + Global2 的 Top-1、Macro-F1、Tail-F1，和 H200 原始 12 組共有方法全數一致（僅配對 p 值有浮點末位差）。
- H200 同機最佳 SGD + S2 Global2 為 **85.3081% Top-1／59.8629% Macro-F1**，相較同機 S2 Base 多 8 筆、McNemar p=0.341、群組 Bootstrap 95% CI **[-0.1102,+0.4042] pp**；Global2 的強融合增益目前不穩健。
- 本地同機 SGD + S1 Base 為 **85.0711% Top-1／60.9597% Macro-F1**；擴充至 S2/S3 Random2、Label-Hide、Label-Only，發現 **S3 Label-Only + SGD 85.2351%／59.8835%**，Top-1 比 S3 完整 Global2 + SGD **85.1075%** 稍高，但此差異的配對 p=0.392。
- H200 Python 3.12.3、本地 Python 3.12.13；同樣 NumPy／scikit-learn／SciPy 版本、相同資料與固定程式時 SGD 分數仍差 6 筆。所有模型提升以**同一環境的 paired Base** 為主，禁止將跨環境分數差當演算法改善。
- H200 S3 Global2 的逐病例 forward 中位 **225.845 ms**；Label-Only **174.980 ms**。完整端到端部署延遲需另計檢索與冷啟動。

已完成的 CPU 工作與輸出：

```bash
.venv/bin/python -m scripts.import_case_rag_h200_archive
.venv/bin/python -m scripts.analyze_case_rag_cpu_first --strict-global2 \
  --output experiments/case_rag_cpu_first_local_recomputed_2026-11-17.json
.venv/bin/python -m scripts.analyze_case_rag_cpu_router --strict-global2 \
  --output experiments/case_rag_cpu_router_local_recomputed_2026-11-17.json
.venv/bin/python -m scripts.evaluate_case_rag_strong_sgd_fusion_cpu --strict-global2 --all-arms \
  --output experiments/case_rag_strong_sgd_all_arms_local_2026-11-17.json
```

**下一階段改用目前可用的完整逐病例 logits 研究 Train OOF／固定 fit-cal 的診斷可靠度與證據衝突路由，先建立能公平超越 85% Top-1／61% Macro-F1 的新機制。** 官方 Test 保持封存；追加 GPU 消融與訓練維持暫緩。

## P2 — 決定新方法與是否需要下一批 GPU（條件式）

**先設計 CPU 規則，再決定 GPU 是否值得執行。**

- 比較 Jeff 決策 logits/margin、BM25 排名與候選差距、來源病例證型一致度、Train 類別頻率與臨床描述的證據衝突。
- 先評估固定規則或簡單、可重現的路由器，在 Train OOF 或分組 fit/cal 上訓練與調參。Dev 已反覆用於方法研究，只能作探索性結果。
- 對照 BM25、source-label CPU、S3+Global2、S2+Global2，並納入 **既存 Joint+Jeff S2 85.180% Top-1 與 SGD+Jeff S1 60.960% Macro-F1** 作更強的完整 Dev 比較；追求維持整體 Top-1 優勢，同時縮小低頻 Macro-F1 落差。
- 已完成 S2/S3 的 Random2／Label-Hide／Label-Only 完整消融，且強 SGD 下 Global2 增益較小。只有新方法明確需要尚缺的 S1 對照組且預期研究結論會改變時，才啟用 `run_case_rag_followup_h200.sh`；成本上限 **32,916** 新 forward。
- 現階段先使用本地完整 logits 做 CPU 差分分析與固定 fit-cal 驗證，對照歷史 SGD+S1／Joint+S2 強基準。**Seed 2／Seed 3 的 GPU 訓練繼續延後**。

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
