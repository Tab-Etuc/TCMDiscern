# TCMDiscern：Jeff 多答案與訓練方法最終研究規範（2026-11-12）

**文件性質：預先固定的研究方法與審稿檢核清單。已完成資料稽核；尚未訓練或測量 Jeff。**
**適用範圍：TCM-SD 官方 Train 43,180／Dev 5,486／Test 5,486，148 類；主模型 Jeff-1.1-Large + LoRA + Decision Head，主要 K=16。**

## 0. 研究決策（優先於較早的候選規劃）

1. **主研究任務維持官方單標籤 Top-1 診斷**；臨床上可能有多個合理證型，但 TCM-SD 的同文異標**只能稱為「觀察到的標註歧義」**，未經醫師審查不能宣稱全部屬多個正確臨床診斷。
2. **主路線：** BM25 生成前 16 個候選 → Jeff-1.1-Large 已發布權重上的新 LoRA + 原作者 Decision Head → 單次選項決策。Train 的 BM25 難負例由 **Train-only Exact-text Group-aware 5-fold OOF** 產生；官方 Train/Dev/Test 全部原始病例與標籤保留，不刪除、不重新切分。
3. **多標籤歧義作為預先指定的核心消融假設：** 與安全單正解基準比較 **Multi-positive Set Loss（部分標籤集合的邊際概似）**；再比較 **Masked CE**（控制組）與 **均勻 Soft-target CE**（較強的正解等價假設）。優先測 Set Loss，避免未經證實的「多正解各占 50%」成為主假設。
4. 對有 2～3 個*已觀察標註*的相同臨床文字，選項要同時放入全部觀察標註，剩餘位置由高分 BM25 負例補滿；Gold Set **只能從完整官方 Train 建立**，絕不能讀取 Dev/Test 標籤構建 Prompt 或評分用可接受集合。
5. **主評估完全使用官方單一 `norm_syndrome`**：Dev/Test 的 Top-1 Accuracy、Macro-F1、各類別 support。候選 Recall@16（Dev Micro 98.60%、Macro 89.91%）只是一個受候選限制的上限，**不能報成 Jeff Accuracy**。
6. **先做 Zero-shot、模型算子與訓練梯度驗收，再花 H200 時間。** 若無穩定提升，保留負面結果，不為了提高數字而改 Test、重切資料或在 Test 上選 K。

## 1. 為什麼不直接把兩個觀察標註各設 50%？

已驗證官方 Train 有 **38,599 組**不同的完整臨床輸入文字：其中 **38,347 組一種標註、251 組兩種標註、1 組三種標註**，總共 **252 組、涉及 820/43,180=1.90% 的原始病例**。同一文字不同標註可能是多種醫學上合理證型、不同專家傾向，或標註誤差；資料未提供獨立醫師對同一病例的效度投票。

官方 Dev 完整 5,486 筆僅有 **29 個同文異標群組、涉及 64 筆**，表示這個子群的可觀察樣本很少。其置信區間與隨機種子波動需要單獨呈現；**不能單憑這 64 筆就聲稱模型具備廣泛臨床多證型診斷能力**。用 Dev 比較方法時，應以**完整臨床輸入文字群組**為重抽樣單位做配對 Group Bootstrap，避免重複病歷被誤當成獨立樣本；Macro-F1 需另報 148 類 support 與極少類別的不確定性。

Jeff 原作者訓練使用 `target_distribution`，但來源包括**實際人工票數分布、教師模型的預測分布及程式產生的標籤**；並未說明可把「重複資料列出現不同金標」直接轉成可靠的醫學機率。

使用均分 1/2、1/3 會**主動假設全部標註等價且都確定正確**；以資料列出現頻次加權則可能放大重複匯入的紀錄，而非醫師共識。兩者均缺充分證據。

參考來源：
- [Jeff 原始論文](https://arxiv.org/abs/2610.17205)
- [Jeff DecisionBank 官方資料說明](https://huggingface.co/datasets/jgeuter/DecisionBank)
- [Partial Multi-Label Learning with Label Distribution, AAAI 2020](https://doi.org/10.1609/aaai.v34i04.6124)
- [Partial Multi-Label Learning With Noisy Label Identification, IEEE TPAMI 2022](https://doi.org/10.1109/TPAMI.2021.3059290)
- [When Multiple Choices Are Correct (2026 預印本)](https://arxiv.org/abs/2610.29217)
- [Towards Unbiased Exploration in Partial Label Learning, JMLR 2024](https://www.jmlr.org/papers/v25/23-0868.html)

## 2. 選項、標籤及 Loss 的定義

對 Train 病例 `x`，定義 `P(x)` 為**官方 Train 中，完整 `chief_complaint + description + detection` 與 `x` 完全相同**的所有互異 `norm_syndrome`，基數 `m=1..3`。不對近似病例、同病名、同 user_id 自動合併標籤；不動官方各列原金標。

1. 建構 16 個不重複證型候選：先加入 `P(x)` 中所有觀察標註，然後從對應 Group-aware OOF BM25 互異候選排名依序加入 `16-m` 個**不在 P(x)** 的高分證型。
2. 從每筆官方 Train 列得到本列 `y`（`y ∈ P(x)`），同時保存 `positive_indices`、`target_index`；**僅作外部 Loss 標籤**，不進模型 Prompt。
3. 使用 Jeff 官方 `state/question/options` → `evidence/criterion/options A..P` 與 Gemma Chat Template，`enable_thinking=False`。選項順序每個 epoch 隨機打亂，**目標位置與多正解 mask 一起跟著排列**。
4. 真實 Dev/Test 推論只取 BM25 前 16 名，沒有強制加入正確標籤，沒有 `positive_indices`，沒有 train label-group 查詢；只選 1 個 Top-1。

令 `z_j` 是 Jeff 對候選 `j` 的 logit，`p_j=softmax(z)_j`，`N` 為非觀察正解的候選集合。

| 實驗方法 | Loss | 用途與限制 |
| --- | --- | --- |
| **S1 安全單正解 CE（已完成資料基準）** | `-log p_y` | 只有本列 `y` 進選項，其他觀察到的金標**排除於負例池**。主流 One-hot CE baseline；與多正解方案的 Prompt 候選集合不同，不能把 S1→S3 變化全歸因於 Loss。 |
| **S2 Masked CE（相同候選控制組）** | `-log[e^{z_y}/(e^{z_y}+Σ_{n∈N} e^{z_n})]` | 選項同時包含全部 `P(x)`，本列原金標之外的其他觀察正解對 Loss **不加分也不扣分**。與 S3/S4 用相同 16 個選項，隔離 Loss 因素。 |
| **S3 Multi-positive Set Loss（主要假設）** | `-log Σ_{j∈P} p_j` | 對候選標註集合分配總機率，不強迫等分；當 `|P|=1`，嚴格退化成原 CE。**不保證所有正解都高分**，適合目前單一 Top-1 與部分標籤不確定性。 |
| **S4 均勻 Soft-target CE（次要消融）** | `-Σ_j q_j log p_j, q_j=1/|P|\ (j∈P), 0\ (j∉P)` | 參考 Jeff DecisionBank 的分布監督，但額外假設每個觀察標註等權有效。須由實驗證明，**預先不指定為主方法**。 |

- **重要理論限制：** JMLR 2024 *Towards Unbiased Exploration in Partial Label Learning* 指出，Softmax 部分標籤學習可能因初始化偏差過早集中在某一候選。`Set Loss` 只約束正解集合的機率總量，**可能只學會其中一個觀察標籤**。因此必須同時檢查組內機率集中程度、正解相對負例的順位，並用 S2 Masked CE 評估是否更能保留所有觀察正解；不得宣稱 Set Loss 已解決「兩個都要判對」。
- **工程先決條件：** 所有 Loss 先用合成 logits 做有限值、梯度、選項置換與數值穩定性測試；使用 `logsumexp` 避免 underflow，沒有正解進 16 個選項時必須報錯。所有方法 `K=16`、相同 Prompt 模板、相同候選 OOF 索引與相同 Train／Dev 分母。
- **訓練方案不能直接套用目前的 `epoch02` 到 S2～S4。** `epoch02` 是「其他標註排除於負例」的 S1 設定；S2～S4 需另建立**包含全部觀察標註的相同候選題目**、另存新的資料路徑及 epoch，且不可覆寫歷史檔案。
- 若要在部署時同時輸出多個證型，應另建獨立的多標籤目標與經醫師確認的資料；目前 S3 的 Softmax 只提供相對選項機率，不能用固定機率門檻宣稱已找齊所有正解。
- **防假效果：** 43,180 原始資料列全部保留；同文字重複次數作為分層報告。同群組加權 `1/n_group` 可在 S1～S4 **統一使用作獨立消融**；不能只替其中一組加權，否則 Loss 效果與訓練分布效果混雜。

## 3. 審稿人最可能質疑的問題與已定下的應對

| 問題 | 決策／必要證據 |
|---|---|
| **只有 1.90% Train 涉及歧義，是否足以構成主要貢獻？** | 本論文主體定位為「快速 BM25 候選→Jeff 領域決策」；歧義感知是**預先指定的增量研究假設／消融**。沒有可重現提升時不得單獨宣稱創新或醫學價值。 |
| **BM25＋JEV/Jeff 已有先例，IEEE 新穎性是否足夠？** | SAG2 已公開 BM25＋JEV 的檢索／評分架構；Jeff 原作者也公開 LoRA＋Decision Head。本文不能宣稱「首先結合 BM25 和決策模型」。應將貢獻限縮為**中醫症狀檢索的 148 類固定基準、臨床標註歧義感知的候選與監督處理、困難負例領域適應，以及效能／長尾／成本的完整驗證**。只有在同預算消融及文獻可比條件下取得實證優勢，才可主張新方法有效。引用 [SAG2](https://github.com/Zleap-AI/SAG-Benchmark/tree/main/sag2)、[Jeff](https://github.com/j-geuter/Jeff)。 |
| **訓練必定放入正解，正式檢索可能缺正解** | 原始 Train OOF 約 1.38% 的金標不在 Top-16；Dev **77/5,486** 缺正解。這是 Listwise 監督選擇的分布落差，須揭露；正式 Dev/Test **不補金標**。 |
| **即使候選含正解也不一定能答對官方標籤** | 實際在完整 Dev 按 `(完整臨床輸入, 固定排序的 Top-16 候選)` 分組，存在相同模型輸入卻對應不同官方標註。對**決定性、只輸出一個選項的相同 Prompt**，候選含金標的 **5,409** 筆中仍有至少 **24 筆**因標註互斥而不可同時答對。故此固定決策設定的理論 Top-1 上界是 **5,385/5,486=98.16%**，比單看 Recall@16 的 98.60% 更嚴格。這是使用 Dev 金標的**分析上界**，不可用來調整模型 Prompt 或稱為實測模型表現。 |
| **BM25 第一名原本已有 79.58% 準確率，重排有可能退化** | Dev K16 **4,366 筆**原本正確、**1,043 筆**正解在排名 2～16 可救、**77 筆**候選未命中無法救。逐例統計 `rescued` 與 `broken`，淨改善 `rescued - broken`；只報綜合 Accuracy 不足以解釋模型效果。 |
| **訓練選項打亂，Dev A 固定為 BM25 Top-1** | 訓練保持 Jeff 原作者的順序隨機化，防止答案位置捷徑。**BM25 排名資訊另在推論後做有無 Late Fusion 消融**（Rank prior／標準化分數），不可在訓練 Prompt 塞入「這是補進來的 Gold」或暗示答案。要額外檢驗不同選項排列的穩健性。 |
| **同文重複／跨切分洩漏** | 主 Train OOF 用相同全文同折，避免同一文字跨 Train 內折出現在檢索索引。官方 Train↔Dev **529 筆**、Train↔Test **553 筆**完全同文字原樣保留，在結果報告分別列重疊與非重疊子群；正式主分數維持官方完整樣本，**不可宣稱獨立患者泛化**。 |
| **少數證型被 15 個困難負例反覆壓制** | Train 類別 support **7～7,380（比值約 1,054）**。彙報正負曝光、每類 F1、Macro-F1。先固定自然分布做 S1～S4 比較；類別平衡採樣／權重另作消融，不與 Loss 比較混在一起。 |
| **Jeff Soft-label 的分布來源和我們不同** | 官方 DecisionBank 部分是真實多標註票數或教師機率；我們是單份病例多筆資料列的觀察標籤集合。方法只稱 *ambiguity-aware*；均勻 Soft CE 只是次要假設。 |
| **沒有醫學知識只用標籤名稱** | 初期固定 `option.description=證型名稱`，保持所有模型同一 Prompt；訓練與推論同時改為「名稱＋經核對的證型定義」可作未來消融，不在首輪混入。 |
| **模型軟機率不等於每個證型的獨立診斷可信度** | 把 `p_j` 說成候選內相對機率。若要同時輸出多證型，必須新增醫師確認的多標籤評估及 BCE／Binary `noul` 或等價方案，並重新校準。 |
| **結果是否有統計可靠性** | 先單種子確認可訓練；最終核心 S1 與 S3，及有助隔離因素的 S2，盡量使用**至少三個獨立訓練 seed**；報平均±標準差、樣本成對信賴區間／McNemar Accuracy 對照、各類別 support 與分層指標。不以 Dev 多輪試錯後最低 p-value 當證據。 |

## 4. 經費與研究優先順序：避免一次做過多無效消融

**階段 A：執行前驗收（P0）**
- 固定 Jeff-1.1-Large 的模型 revision、Tokenizer、Head、Prompt 與 train/eval env。
- 先取得完整 Dev 的 Jeff **未微調 Zero-shot Top-1、Macro-F1**；同時比較 BM25 既有基準。
- H200 跑最小反傳：BF16 backbone + FP32 Decision Head、PEFT LoRA 參數數量、最後一個有效 Token、padding mask、K <16 的 mask，`step0` 與官方 Jeff runner logits 等價；確認 checkpoint 存取後結果一致。
- 先做好 S1/S2/S3/S4 的合成資料 Loss 單元測試，不得僅以 Prompt JSON 相同就宣稱訓練器可用。

**階段 B：主要實驗（P1）**
- 用 Group-aware OOF、K=16、相同 43,180 Train 預算。至少：Jeff Zero-shot、S1 safe CE、S2 masked CE、S3 Set Loss。
- S2 與 S3 **同一候選集合、同一 shuffling、同一 seed／batch budget**；兩者差異才能合理歸因於 Loss。S1 對照既有修正好的單正解資料。
- Dev 5,486 筆（**真實 BM25 前 16、不補正解**）做 checkpoint 選擇，優先依 Macro-F1，同時設 Accuracy 不劣於 BM25 基準的檢核；若以多指標進行選模，事先公開規則。

**階段 C：限制性消融（P2）**
- 如要對外宣稱「確有多個臨床正確證型」，優先邀請至少兩位獨立中醫師對 **252 個同文異標群組**盲審、記錄一致性、分歧原因及最終審定標籤；未完成專家複核之前，只能研究觀察到的資料標註歧義。
- S4 均勻 Soft CE、BM25 rank-based Late Fusion、同群組重新加權、隨機負例、候選名稱＋定義，依剩餘資源一次挑一個變因。
- 充分驗證後再比較 K=5、16、17、35；17～35 是 Jeff 分組推論，多次 forward cost 需明列，不可拿純 Recall 增加當 accuracy 提升。
- 最佳設定定案後**一次性執行官方 Test 5,486**，不以 Test 選權重、K 或閾值。

## 5. 評估與發表的不可逾越界線

- 正式主結果使用**原始單一官方標籤**，Accuracy、Macro-F1、per-class 指標；一律同時報 BM25、Jeff Zero-shot 與訓練後版本。另報候選 Recall@K、`gold-in-top16` 條件成功率、rescue/breakdown、延遲、峰值顯存、訓練步數／tokens／seed。
- **沒有獨立醫師複核的多標籤 Dev/Test**，不能把「預測命中 Train 歧義集合之一」冒充正式 TCM-SD 多答案準確率或新 SOTA。
- Train/Dev 精確同文重複及 Test/Train 重疊要揭露，並做原始 benchmark 完整結果之外的重疊／非重疊分層。主表不刪除任何官方病例。
- 類別長尾和 Dev 每類最少僅兩筆：Macro-F1 的變化需有 support、seed 方差及有條件的統計檢定；不要為小數點幾個百分點誇大方法優勢。
- 若預先指定的 S3 Set Loss 沒有改善官方 Dev 與最終 Test，保持結果並報告 S2/S4 的差異，不事後捏造「全多正解有效」結論。
- 引用 Jeff 的官方訓練資料說明、Prompt 與 Decision Head；SAG2 的多 `gold_docs`、Recall/AllEvidence 用於**多證據檢索**，只作方法背景，不能直接視為本題多個替代證型的訓練公式。

## 6. 現有完成狀態（禁止誤述）

**已完成：** 官方資料稽核、原始 BM25 Dev K=1～35、Train 分組 OOF 43,180 筆、保留每列原始標籤的單正解安全負例 epoch02、逐筆 QA、官方 Jeff K16 Prompt Tokenizer（全 Train/Dev 無超過 8,192 tokens）、65 項以上 CPU 回歸測試。

**2026-11-12 執行狀態更新：** S2/S3/S4 共用的多正解 K=16 Train 43,180 筆已生成並完成逐筆 QA，四組 Loss 已通過解析梯度與 CPU PyTorch autograd，Jeff tokenizer 長度及 Dev 評分模擬通過。**仍未執行 Jeff GPU 反傳、Zero-shot Dev、LoRA、正式 Test。** [本機執行紀錄](jeff_local_preflight_2026-11-12.md)。研究假設及 Loss 優先順序不因這些檢查更動。

> 審稿立場：將「使用正確選項集合，合理處理標註歧義」當成**可證偽的訓練假說**，而不宣稱它已經帶來提升；只要單標籤資料的既有基準不能超越或結果無法重現，研究就必須如實報告。
