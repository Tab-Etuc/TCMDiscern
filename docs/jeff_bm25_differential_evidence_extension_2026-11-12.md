# TCM-SD：保留 Jeff + BM25 的鑑別式證據延伸研究（2026-11-12）

**狀態：CPU 探索完成；新訓練目標與知識增強尚未做 H200。本頁為早期研究候選，最新優先決策見 [檢索錨定決策重新審視](jeff_bm25_rank_anchored_research_decision_2026-11-12.md)。正式 S1/S3 訓練規範保持不變。**

## 一、不可變的正式主幹

保留既有 [Jeff + BM25 預先定稿規格](jeff_experiment_preregistration_2026-11-12.md)：

1. TCM-SD 官方 Train 43,180／Dev 5,486／Test 5,486，148 證型。
2. Train-only、Group-aware OOF 的中文 bigram BM25；Dev 固定全 Train index，**原始 Top-16 候選、順序與原始排名都不可覆寫**。
3. Jeff-1.1-Large，原 Decision Head + rank32/alpha64 LoRA，固定 16-choice SemIf 輸入、三個 Seed 與原版 S1/S3。
4. 正式 Test 直到確定擴展方法及所有 Dev 決策才一次性執行。Dev 已用於本次方法探索，因此不能稱為未見方法選擇集。
5. 所有知識只作**標籤側特徵**。不得將 Dev gold、Test gold、病因／病名推測或相似病例的診斷帶入患者病例來源。

## 二、重新閱讀 cmu_aitcm 的 No-Go：哪些可重啟，哪些不可重複

| 歷史 No-Go（來源） | 已觀察失敗 | 本研究可能真正不同的條件 |
|---|---|---|
| `full-four-source-lora-ablation-2026-10-15.md` | 院內 3,811／459，InfoNCE 單負例、rank8 LoRA；原 106/459，完整四診 LoRA 102/459 | 官方 148 類、固定 OOF 候選、Jeff listwise Decision Head，S3 的目標與原 InfoNCE 不同 |
| `syndrome-evidence-model-decision-2026-10-15.md` | Zero-shot Qwen Reranker，300 評分 133→74；LightGBM OOF 僅 1674→1682/3806 | Jeff 是已訓練的選項決策模型，進一步做同域 Train LoRA，且同候選下比較 |
| `tcmsd-source-aware-evidence-transfer-2026-10-15.md` | 1,481 外部病例與院內 235 類術語映射、分布落差；跨來源融合 459 退 106→101 | 此次全部病例在官方 TCM-SD、148 類同標籤空間，**不混院內病例或類別** |
| `setfit-class-weight-ablation-2026-10-29.md` | 單一大類降權僅 356→358/737，Macro-F1 +0.0052 | 對 Train OOF 的候選競爭關係做成對監督，保持全類別而不調單類先驗 |
| `fast-kg-ppr-shadow-2026-10-26.md` | 圖約束過度依賴缺正向極性的來源，Flat/PPR 都 0/88 合法封包 | 本文**不建患者因果 Graph**；知識是可選 label-side 特徵，不升格為臨床主張 |
| `tcmsd-cheap-span-negation-study-2026-10-16.md` | 簡單字首肯否易把「不寐」「不時」當陰性；20/24 是非獨立小反例 | 不從「不／無」推患者不存在的症狀，只做不改醫療事實的欄位視圖一致性控制 |
| `la-scl-original-reproduction-and-label-audit-2026-10-29.md` | LA-SCL 原作者完整評分器復刻未達標、標籤關係不能視為臨床等價 | 不宣稱原作者模型復刻，不把同文異標直接當多名醫師確證的共同真值 |

歷史 No-Go 都受各自病例、標籤、模型、指標與候選建構限制；必須使用與現有 S3 完全相同的 Train、Dev、Seed、Token 環境，才能聲稱新增技術改善。

## 三、2026-11-12 本機完整 CPU 否證

本機新建**可重跑**的 `scripts/probe_syndrome_knowledge_top16.py`，使用 TCM-SD 官方 148 條證型知識的 Name + Definition + Typical_performance；Knowledge BM25 的 k1=1.5、b=0.75、min_df=1，獨立對全部 5,486 Dev 問診文字排名。Dev gold 僅用於計分；Knowledge 排序在固定 Top16 裡重排，沒有加減原 BM25 候選。

| 方法 | Dev Top-1 | 固定 148 類 Macro-F1 |
|---|---:|---:|
| 原始病例 BM25，Top1 | **4366/5486 = 79.5844%** | **0.54850** |
| 知識 BM25 在全 148 類直接 Top1 | 120/5486 = 2.1874% | 0.01476 |
| 知識 BM25 在病例 Top16 重新排序 | 418/5486 = 7.6194% | 0.05796 |
| 原 Top16 同池固定 RRF(k=60) | 832/5486 = 15.1659% | 0.13028 |

- 兩個 Top1 排序有 **5,080/5,486** 筆不同；知識重排僅救回 **99 筆**，卻改錯 **4,047 筆**。
- 原病例 BM25 的 **77** 筆 Top16 缺 Gold，獨立知識 BM25 Top16 涵蓋其中僅 **18** 筆，屬補充候選的探索訊號，不能冒稱 18 筆已診對。
- 由此判斷：**用症候文字知識檢索得分直接修改 Jeff 的候選排名，現有做法 No-Go**。詞彙重疊低分不能推論知識對語意決策無幫助，也不能拿這些數字冒稱 Jeff 成績。
- 結果：`experiments/knowledge_top16_cpu_probe_2026-11-12.json`。

## 四、Train OOF 歧義與候選鑑別差異的可用性

獨立執行 `scripts/audit_jeff_oof_confusions.py`，僅讀 Train-only grouped OOF 和經原文件稽核的多觀察正標：

- **34,656/43,180** 個 Train OOF 的 BM25 Rank-1 與原始 Gold 一致。
- **7,950** 個 BM25 Rank-1 錯誤，但 Gold 仍在 K16。
- 其中 **233** 個 BM25 Top1 其實也在同文病例的**其他觀察正標**集合。這些 Top1 絕不可拿來作必錯的 hard-negative。
- 扣除可觀察正標衝突，剩 **7,717** 個 OOF Top1 可作「目前沒有證據是另一觀察正標」的**弱負例**候選；缺標註可能仍有其他臨床合理答案，不能稱其為醫師確認的絕對錯誤。
- 7,717 個弱負例涵蓋 **1,412 種有方向性的 Gold→BM25 Top1 混淆組合**，其中前 50 種僅占 **41.65%**。因此只為熱門 10～50 對硬寫規則、或只調某類先驗，覆蓋度不足，應採用對每個 Query 及候選集合動態生效的 loss。
- 結果：`experiments/jeff_oof_confusion_2026-11-12.json`。

## 五、主要技術假說：Retrieve-Conditioned Differential Evidence（RCDE，暫定）

**保持 BM25 提供固定 Top16、Jeff 負責一次性 16 logits。** 本方法核心：將最難的「與原始 BM25 Top1 的候選競爭」視為額外訓練監督，並在必要時提供帶來源標記、候選專屬的證型定義供 Jeff 辨別。

- **基底 loss**：沿用 S3 的 multi-positive Set Loss，不重設金標、替換模型或刪掉 820 筆同文異標原始病例。
- **Pairwise hard-negative margin（候選）**：當 Gold 仍在 BM25 Top16、BM25 Top1 與所有已觀察正標不同時，增加 `softplus(m - (logsumexp(z[P]) - z[h]))`；`h` 是 OOF BM25 Top1，`P` 是觀察正標索引集合。這是待驗證的附加損失，不能把常見 margin loss 宣稱首創；在 GPU 以 FP32 計算，且只利用 Train-only 固定 fold 產生，不使用 Dev 金標挖掘。
- **Knowledge context（候選）**：獨立輸入臂將官方證型定義與典型表現加到 Jeff 原本的 `options.description`，保持 labels、K16 和 Head 不變。擷取知識內容須註明原始文件欄位、可重播的文字長度與 Token 上限，不可補寫患者症狀。
- **Controlled-view consistency（後續）**：只對等價的患者輸入視圖（欄位順序交換、標題移除且所有 source chars 保留）施加 KL／配對一致性；不能將明確肯否翻轉、當下／過去互換當成同標籤樣本。
- **Knowledge placebo（必要負控制）**：與增加相同長度的亂配證型知識、全無知識文字，配對使用同 Seed、同 Steps、同候選、同 148 類 macro scorer。只有正確知識優於錯配知識，才可能歸因知識本身。
- **訓練成本**：Pairwise loss 可直接用同一次 Jeff forward 的 16 logits 計算，沒有第二次 31B forward；完整語義視圖一致性通常需要額外 forward，須獨立報告 GPU 整體成本。

注意：傳統成對 margin 與知識注意力均有既有研究；**「只加 Pairwise loss」不能直接自稱強創新**。要主張可投稿的新方法，需證明至少一個可驗證的新增機制：例如歧義標註安全的動態負例選擇、候選對差異知識的有效使用、跨記錄格式一致性改善，以及比隨機難例／錯配知識更好的結果。

## 六、最小且足以判斷的實驗矩陣

| 臂 | 與正式 Jeff S3 的唯一差別 | 檢驗假說 |
|---|---|---|
| A：S3 固定主基準 | 無 | 保留原研究結果 |
| B：S3 + Pairwise OOF | 新增成對 Hard Negative loss，不增加資料曝光次數 | 難負例是否有價值 |
| C：S3 + Label Knowledge | 只加入正確的 148 類標籤知識作候選描述 | 有效知識能否被 Jeff 使用 |
| D：S3 + Pairwise + Knowledge | 結合 B/C | 是否存在互補交互作用 |
| E：S3 + Pairwise + Shuffled Knowledge | 只打亂標籤知識對應，長度分布盡量匹配 | 排除提示變長／無關資訊造成的假增益 |
| F：S3 + Random Hard Pair | 按同一 OOF 樣本／類別頻率隨機替代 Hard Pair | 排除只是多一項 loss 的效果 |

優先在小範圍、單 Seed、同樣 GPU Token/Optimizer 預算下執行 B/C/D/E/F 的可行性篩選，然後**預先固定**一項新方法對 A 做三 Seed 完整配對。所有額外臂都需報 Gold 保留率、BM25 原錯救回率、原對改錯率、全 148 類 Macro-F1、長尾類別、等價輸入翻轉率及 GPU 成本。**沒有獨立成績時不把方法提升寫成論文事實**。

現有 IEEE/醫療相關競爭研究已包含：
- [DCKA（Scientific Reports 2025）](https://pmc.ncbi.nlm.nih.gov/articles/PMC12008396/)：TCM-SD 知識注意力，Test Accuracy 84.01%、Macro-F1 54.25%；知識注意力本身已不是新方向。
- [Agent-GNN（Digital Chinese Medicine 2026）](https://doi.org/10.1016/j.dcmed.2026.02.005)：TCM-SD 長尾知識圖；「中醫圖＋長尾」亦已有先行研究。
- [Multi-stage Retrieve and Re-rank（NAACL 2024）](https://aclanthology.org/2024.naacl-long.273/)：ICD 多標籤任務 BM25、code graph 與對比學習。其共現監督不能無條件搬到 148 類單標籤證型。
- [Jeff（2026）](https://arxiv.org/abs/2610.17205)：選項決策 Head 原理與 LoRA 已公開。
- [LLM2Jev（2026）](https://www.microsoft.com/en-us/research/publication/llm2jev-llms-are-already-jev-style-decision-models-when-and-how-to-fine-tune-them/)：listwise loss 及 KL anchoring 已公開，因此一般性的 listwise／KL 不能冒稱首創。

**研究狀態與程式界線：** CPU 基準／混淆圖已重跑；B～F 尚未實作或測出 Jeff 成績。建議先跑既定 A，取得真實 31B 分數與成本，再用 B 的同一次 forward 新 loss 試作第一個研究延伸。完整 Full Train／Dev／Test 原始欄位、候選與正式實驗流程均保持不變。
