# Jeff + BM25 延伸研究重新審視：檢索錨定決策（2026-11-12）

**地位：研究假說修訂；正式 S1/S3 與 H200 一鍵主流程保持不變。此文件只定義後續擴展，不授權任何未經 Pilot 的大型訓練。**

## 一、為什麼需要改變上一版方向

- 單純對 Gold vs BM25 Top1 加 Pairwise Margin 與既有 S3 listwise Set Loss 有相當大的監督重複；方法原創性薄弱，容易造成高頻混淆類別的梯度偏重。
- BM25＋知識文字的簡單固定 RRF/重排在 5,486 筆 Dev 大幅退步；該結果否證直接採用這種詞彙排序，無法否證 Jeff 使用證型定義的可能性。
- TCM-SD DCKA 已使用知識注意力、Agent-GNN 已用知識圖改善長尾、NAACL 2024 已用 BM25＋對比重排。單純再增加 knowledge attention / graph / pairwise loss 難以成立核心新穎性。
- Jeff 原本使用 A～P 決策槽位，屬有選項排列敏感性風險的模型架構。既有多選題 position-bias 研究（ICLR 2024、NAACL 2024、2026）證明此議題已有先行研究；不能將「發現選項偏差」宣稱首創。
- 上游 Jeff 模型公開說明其 calibration fit 在其他 heldout task；TCM-SD 是明顯的 domain shift，必須重新分析 confidence / trust。

## 二、重新核對的量化問題

完整官方 Train 43,180／Dev 5,486，再讀原始 OOF 50 名、S3 訓練 Prompt、Dev 固定候選、Train/Dev 原始輸入：

| 事實 | 已重新確認數字 |
|---|---:|
| S3 Train 正解在 A 槽 | 2,646/43,180 = 6.13% |
| 正式 Dev Gold 在 BM25 Rank1（A） | 4,366/5,486 = 79.58% |
| Dev Gold 原 BM25 Rank2～16 | 1,043/5,486 |
| Dev Gold 未進 Top16 | 77/5,486 |
| Train OOF Gold 位於 Rank1 | 34,656 |
| Train OOF Gold 位於 Rank2～16 | 7,950 |
| Train OOF Gold 位於 Rank17～50 / >50 | 434 / 140 |
| **S3 訓練題選項集合恰等於原 OOF Top16** | **42,579/43,180** |
| **S3 選項集合經插入／替換** | **601**（其中 Gold 原本不在 Top16：574；Gold 在 Top16 但其他觀察正標插入：27） |
| S1 選項集合恰等於原 OOF Top16 | 41,838/43,180 |
| 官方 Dev 與 Train 原始三欄臨床文字完全一致 | 529/5,486（其中 20 筆金標與相同訓練文字的標註集合不同） |
| 無相同 Train 文本的 Dev | 4,957 筆，BM25 Top1 78.56%、Recall@16 98.49% |
| 與 Train 完全相同的 Dev | 529 筆，BM25 Top1 89.22%、Recall@16 99.62% |

**解讀：** 訓練時打亂選項可避免直接記住 A 槽偏差，但 Jeff 也無法從打亂後位置直接利用 BM25 原始證型排序。若直接將 Rank17+ 當成新增正解的特徵，574 筆會存在被插入正解線索的風險；另 27 筆也不能硬賦假裝存在的前 16 名排名。

完整可重跑聚合報告：`scripts/audit_retrieval_rank_shift.py`、`experiments/retrieval_rank_shift_2026-11-12.json`；S3 候選集合對位數字也已由 Train OOF 與完整原 Prompt 逐筆比較。

## 三、新主張：Retrieval-Anchored Ambiguity-Safe Decision（暫稱）

**BM25 → 固定 Top16 → Jeff 16 logits** 完全不變。新增的是 **與候選 label-ID 對齊、位於 prompt 之外的原始 BM25 rank prior**，以及僅在合法原始 OOF candidate set 上啟用的決策監督。

對輸入病例 x、候選集合 C、BM25 原始排名 r 和 Jeff 16 個 logits z：

```
q_B(y) ∝ (rank_B(y) + c)^(-alpha)
s(y | x,C) = log(q_B(y)) + g_phi(x,C) * (z_theta(y)-z_theta(y_BM25_top1))/T
p(y | x,C) = softmax_y(s(y | x,C))
```

其中 rank 不出現在 Jeff Prompt A～P 選項的文字中，**只在送進 Head 後按 label-ID 對回數值向量**。正確 mapping 不受選項排列變動影響；g=0 回到 BM25 Rank1，g 大時 Jeff 有能力推翻 Rank1。這個 prior 加權本身有大量 IR 先行工作，必須以臨床歧義保護與可控的更正決策訓練機制作為額外研究貢獻，避免冒稱純融合為首創。

**訓練設計：**
1. 保持原 S3 listwise Set Loss 全 43,180 筆；正標仍只是 observed label 集合。
2. 只在 **42,579** 筆「訓練 16 選項集合恰等於 Train OOF 原 Top16」的病例上新增 retrieval-anchored listwise loss。這些病例 gold 原本就被 OOF 召回，不用人工注入 rank。
3. 對其餘 **601** 筆仍只做原 S3 Loss；不得把人工補入標籤的 Rank17／Rank51 當作診斷線索。
4. 選項每 Epoch 照舊隨機排列；原始 rank 則以 label-ID map 接回正確選項 logit 位置。任何違反 mapping 的樣本立即中止。
5. 逐病例混合原 S3 與 retrieval-aware Loss，固定同一 43,180 筆／相同 Batch/Seed/LoRA/Optimizer Updates；額外計算僅作用於既有 16 logits，**不增加 31B Forward**。
6. 多觀察正標 S3 採 `P={所有已觀察標籤}`；若 Gold 原始 rank1 也在觀察正標中，不把該標籤當成必錯負例。需要分開報官方單金標／觀察正標多答案評估，避免假設臨床等價。
7. 新增對照：固定 rank prior + 現成 Jeff logits（無訓練）；以及單一 confidence gate / threshold（無訓練）。新方法至少要優於**最強的簡單後處理 baseline**，才能稱得上新的技術增益。

**可選後續：**
- Jeff 並非原生 permutation equivariant；在固定樣本對 3 種選項順序測試 ID 對齊後的預測穩定性。若變化大，再加入 permutation-consistency loss，附 GPU 額外 Forward 成本。已有先行研究，屬輔助模組。
- 知識只在證據可比且打亂知識與正確知識的負控制出現明顯差異後導入。保持 16 候選、禁用以「不」「無」推理症狀不存在，防止 cmu_aitcm 的 negation/Graph No-Go 再次發生。
- 原先 7,717 筆 Train OOF 弱負例可作 diagnostics；Pairwise Margin 先放在附加消融，不作論文主貢獻。

## 四、按成本排序的決策門檻

| 階段 | 實驗 | 為什麼先做 |
|---|---|---|
| 0 | 真正 Jeff zero-shot/S1/S3 GPU，完整 Dev 5,486，保存原 16 logits | 沒有真實模型結果，所有新技術的改善預期都只是推測 |
| 1 | **後處理零成本 prior／gate 基準**：BM25 rank-only，Jeff-only，數值 rank prior + Jeff，固定閾值保守切換 | 如簡單融合已足夠，避免浪費數個 31B model-epochs |
| 2 | 500 筆固定 Dev，3 個無標註依賴選項排列，檢查 permutation consistency | 驗證 A～P 結構的脆弱點；Permutation 已有先行工作，僅作控制 |
| 3 | 原 S3 vs retrieval-anchored S3，所有 43,180 原始列、相同 Seed/Updates，額外 prior Loss 僅作用 42,579 真 Top16 rows | 真正測量 retriever-conditioned supervised improvement |
| 4 | 若階段 3 > 最強後處理基準，再做 3 Seed，含原始 Dev／4,957 無相同文字病例／長尾／歧義／救回 vs 改錯 | 支撐論文實質效果及泛化 |
| 5 | 只有階段 4 明確成功，才新增知識差異模組，配正確／亂配／等長 placebo 控制 | 避免重複 DCKA、Agent-GNN 的知識模組，或被知識雜訊拖累 |
| 6 | 全部方法與 K、gate、Seed 選擇凍結，最後一次執行 Test | 減少重複 Dev 模型選擇造成的偏差 |

**模型選擇公平性：** Dev 已被用於探索 BM25 與多種知識變體，不能稱它為完全未見的驗證集。若在 Dev 上擬合小型 gate，應使用按原始臨床文字分組的交叉驗證報開發分數，完成選模後只在封存 Test 做一次最終確認；不得把在同一 Dev 上調出最佳參數的數字當無偏結果。

**明確失敗判斷：** 若新模型僅比 S3 單獨好、卻與 BM25+Jeff 的簡單後處理效果相當，則主張限縮為工程融合與消融；若同 Seeds 的 paired difference、Macro-F1 和無相同文本子群沒有持續增益，則不強行宣稱研究創新成立。

## 五、論文貢獻的可驗證表述

1. **診斷分布落差分析**：逐筆揭露金標強制插入、位置排列與真 BM25 原始 Rank distribution 的差距；建立 leakage-safe rank-conditioned training gate。
2. **歧義安全的 BM25 對齊殘差決策**：保留原 Jeff 和 BM25，在真正 OOF 候選集合上學習「何時足以推翻第一名」的更正訊號。
3. **公平且可重現的臨床評估**：固定候選、相同 31B Forward/Seed/Updates、原始 vs 無完全相同病例子群、148 類 Macro-F1、rescue/broken、permutation stability、Latency、Test 一次性評估。

現有相關研究：
- Jeff 原版 16-slot Head（https://huggingface.co/jgeuter/Jeff-1.1-Large）
- TCM-SD DCKA 知識注意力（https://doi.org/10.1038/s41598-025-96404-w）
- 2026 Agent-GNN（https://doi.org/10.1016/j.dcmed.2026.02.005）
- NAACL 2024 Retrieve-and-Re-rank ICD（https://aclanthology.org/2024.naacl-long.273/）
- ICLR 2024 MCQ Selection Bias（https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html）
- 2024 第一階段 retriever score 顯式注入（https://link.springer.com/article/10.1007/s10791-024-09435-8）

**不變的界線：本文件沒有修改正式 S1/S3 H200 主程式、沒驗證真實 GPU 改善，且尚未將 rank-conditioned loss 接到 GPU 訓練。**
