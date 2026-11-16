# TCMDiscern Case-RAG：J-BHI 研究實驗鎖定（2026-11-17）

## 研究問題與論文主張

**研究問題：** 在微調後的 Jeff K=16 中醫證型辨識器中，加入來自官方 Train 的兩筆標註歷史醫案，對辨證準確率、148 類 Macro-F1、尾類辨識與效率有何影響？改善來自臨床文字、歷史醫案的標註證型，或二者共同作用？

本研究定位為**有嚴格證據歸因的病例檢索決策研究**。目前 Global2 屬強基準，直接接上 Jeff 尚不足以宣稱新演算法。若正式 5,486 Dev 的歸因消融確認獨立臨床文字的增益，再設計有技術內容的可靠度加權或證據衝突解法；有新的訓練或選擇機制才主張演算法原創性。

所有 300 筆 Pilot 曾用來選擇 Global2；後續 5,486 Dev 屬**延伸開發評估**，不視作完全獨立的最終驗證。原始官方 Test 保持封存。

## 論文正文表格／圖表

| 編號 | 實驗／圖表 | 資料量 | 執行狀態 |
|---|---|---:|---|
| Table I | TCM-SD Train/Dev、148 類、標籤分布、Train/Dev 完全同文與患者重複 | Train 43,180；Dev 5,486 | CPU 完成 |
| Table II | BM25、Jeff S1/S2/S3、來源標籤 CPU 基準、S1/S2/S3 + Global2 | 全 Dev | 基準完成；Global2 GPU 待跑 |
| Table III | S2 + Global2、Random2、Label-Hide、Label-Only 對照 | 全 Dev | 輸入已完成；GPU 待跑 |
| Table IV | Top-1、148 類 Macro-F1、末 37 類 Macro-F1、Tail Top-1、救回／破壞與 McNemar 檢定 | 全 Dev | CPU 評分器完成；GPU 待跑 |
| Table V | 排除 Train 完全同文病例的敏感度與相同患者 ID 風險 | 全 Dev | CPU 部分完成 |
| Table VI | baseline／Global2／消融的 token、GPU 推論 mean／p50／p95、VRAM 與單次載入成本 | 全 Dev | Pilot 完成；全 Dev GPU 待跑 |
| Figure 1 | 輸入→BM25 Top-16→Train 病例 Top-2→Jeff Decision Head→證型；兩組歸因消融 | 架構 | 可依固定流程製作 |
| Figure 2 | S1/S2/S3 Top-1、Macro-F1 的 Base vs Global2 配對差及信賴區間 | 全 Dev | GPU 待跑 |
| Figure 3 | Global2／Label-Hide／Label-Only／Random2 的準確率與來源標籤敏感性 | 全 Dev | GPU 待跑 |
| Supplement | 300 筆 Pilot 17 條件、完整醫案、Paired2、Balanced2、標籤調換／次序反轉 | Pilot 300 | 全部完成 |

### 固定原始模型結果（完整 Dev）

| 基準 | Top-1 | Macro-F1 |
|---|---:|---:|
| BM25 Top-1 | 79.5844% | 54.8496% |
| Jeff S1 | 79.2198% | 40.3404% |
| Jeff S2 | 80.3500% | 42.6654% |
| Jeff S3 | 80.0948% | 41.2788% |
| 來源病例 Top-1 診斷標籤（CPU） | 80.2953% | 55.1720% |

來源標籤 CPU 基準直接取 Global2 排名第一的 Train 病例之標註證型，明確保留為重要強基準。它不呼叫 Jeff。

## 鎖定方法（Full Dev）

1. **主要比較（S2）**：原始 S2 完整 Dev logits 對 S2 + Global2。同一患者、同一候選 Top-16，同一批 Train 索引、相同 LoRA；只增加兩筆來源醫案。
2. **跨損失函數驗證**：S1/S3 對各自 Global2，比較病例檢索對不同微調策略的穩定性。正式比較使用多重檢定校正；三組共用 5,486 筆患者。
3. **隨機病例控制（S2）**：Random2 使用來源自候選 Top-2 證型的兩筆隨機 Train 醫案，固定原本患者、候選與證型來源池。
4. **臨床證據歸因（S2）**：
   - Global2：兩筆來源病例的證型標籤與三個臨床欄位都保留。
   - Global2 Label-Hide：三個臨床欄位保持完全一致，只將歷史病例證型標籤替換成「未提供」。
   - Global2 Label-Only：歷史來源證型標籤保持完全一致，只移除歷史來源病例的主訴、描述、四診內容。原始患者三個臨床欄位保留。
   - 同時報告來源病例 Top-1 診斷標籤 CPU 基準。Label-Only 與 Global2 的 Token 數不同，解釋效果時必須考慮 Prompt 長度。
5. **排除重複病例**：全文與主表附列 5,486 筆及排除所有 Train/Dev 完全同文病例後的 Top-1。患者 ID 重複、診斷標籤衝突亦以獨立資料品質指標報告。
6. **效率比較**：推論的同步 wall-clock mean、p50、p95 與顯存峰值；另列索引檢索、Prompt 構造與模型載入的時間範圍，避免將 GPU `decide` 延遲當作服務端到端延遲。

### 完整 Dev 現有資料品質核對

- Train：43,180，Dev：5,486，Train 證型：148 類。
- **完整 Dev 有 148/148 類 gold**；其中 28 類 gold 僅有 1–2 筆，尾類仍有很大統計不確定性。
- Train/Dev 標準化後完全同文：**529** 筆（9.64%）。
- Train/Dev 同文但證型 gold 不一致：**20** 筆。
- Train/Dev 的患者 ID 重疊：**19** 筆。
- 排除完全同文後剩下 **4,957** 筆 Dev。
- 索引生成中已排除相同正規化病例文字與同患者 group 的來源病例；資料品質重疊指標仍須報告，不能把這個子集視為外部驗證。

### 指標與統計規格

- 主要：Top-1 準確率與配對增益；同時報告**148 類 Macro-F1**，檢查性能有否集中於高頻證型。
- 尾類：以 Train 診斷證型頻率最低的 **37 類**（148 類的四分之一）固定為長尾組，報告 Macro-F1、尾類 gold 筆數和 Top-1。
- 類別覆蓋：保留 BM25 Top-16 Recall 上限與候選不包含 gold 的病例數；確保所有方法受相同候選限制。
- 同一患者的 rescue/break 與 exact McNemar，1,000 次**患者 ID／相同病歷文字群組 Bootstrap** 95% CI。
- 主要 S2 比較是事先於此完整 Dev 推論前指定；S1/S3 為複製驗證。若同時提出三組顯著性主張，對 Global2 vs baseline 的三個檢定使用 Holm 修正。
- 300 筆 Pilot 用來挑選 Global2，因此 5,486 Dev 仍受到方法選擇影響。最終 Test／外部資料應由完全定稿的實驗方案做一次鎖定評測。

## H200 執行預算與資料安全

**GPU 需要新增 6 組推論**：S2（Global2、Random2、Label-Hide、Label-Only）× 5,486；S1、S3 各 Global2 × 5,486。總共 **32,916** 筆新推論，使用 **1 次** Jeff 31B 權重載入。過去的三組全 Dev baseline logits 完整重用。輸入所有 Token 數均小於 8,192，產生與檢查只需 CPU。單例推論透過 `run_resumable` 保留 partial、計時與顯存資訊。

GPU runner：`run_case_rag_paper_h200.sh`；先執行 `--prepare-only`，再執行 `--check` 驗證所有三組 checkpoint 與 CUDA/FA2，最後執行正式完整推論。

完成後輸出：
- `experiments/case_rag_paper_full_dev_s2_20261117.json`
- `experiments/case_rag_paper_full_dev_s1_20261117.json`
- `experiments/case_rag_paper_full_dev_s3_20261117.json`

模型原始 logits 和患者病例維持於忽略版本控制的 `data/`；公開報告只包含聚合結果。

## 投稿判斷門檻

若 Global2 穩定改善三模型 Top-1，但尾類與 Macro-F1 退步，應如實轉成「效率／Top-1」的病例增強研究，並針對來源標籤捷徑及尾類退步提出新方法。若 Label-Only 接近 Global2，應優先研發具有真正**臨床證據作用**的選擇或可靠度判斷機制，而非以單純加入歷史標註病例作為 J-BHI 的演算法創新主張。

相關現有研究包含 TCM-CoT-RAG（2026，https://pubmed.ncbi.nlm.nih.gov/42656911/）、病例及指引共同檢索的 MECR-RAG（2026，https://pmc.ncbi.nlm.nih.gov/articles/PMC12887567/）；論文須明確區隔既有案例檢索與本文的新方法貢獻。
