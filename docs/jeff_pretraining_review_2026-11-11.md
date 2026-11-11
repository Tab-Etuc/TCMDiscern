# Jeff-1.1-Large 微調前完整技術審查（2026-11-11）

**對象：** TCMDiscern，TCM-SD 官方 Train 43,180／Dev 5,486／Test 5,486，148 類證型。主要實驗 K=16，Jeff-1.1-Large + 領域 LoRA + 原始 Decision Head。**目前模型未訓練，未取得 Jeff 的中醫 Top-1／Macro-F1。**

## 摘要決策

**現階段不啟動正式 31B LoRA 訓練。** 目前已確認病例資料轉成 Jeff/SemIf 的原始 Prompt 格式，且完整 Train/Dev 的 token 長度均合法；剩下最重要的問題在 **微調 Head 的計算路徑、負例正負分布、長尾類別、重複文字跨折、訓練與真實檢索落差**。應先完成負例對照與 31B 一步反傳的真實環境測試。

Jeff 官方可核對資料：[Jeff 服務／Prompt／Decision Head 程式](https://github.com/j-geuter/Jeff)、[Jeff-1.1-Large 模型卡](https://huggingface.co/jgeuter/Jeff-1.1-Large)、[研究報告](https://arxiv.org/abs/2610.17205)。模型完整權重與 Head 的實際數值均以公開 release 為準；原作者公開 repository **沒有完整的 Jeff-1.1 訓練程式**，本專案應明確標示訓練器是自行實作。

## 一、已通過的客觀驗證

| 檢查 | 實測結果 |
|---|---|
| 官方病例切分 | Train 43,180、Dev 5,486、Test 保留 5,486；無刪除、無重新切分 |
| Jeff/SemIf Prompt | 固定 System Message、`evidence/criterion/options` JSON、A～P、Chat Template／無 Thinking；已與作者 `prompting.py` 核對 |
| 實際模型 Tokenizer | 使用 Jeff-1.1-Large 正式發布的 GemmaTokenizer，Transformers 5.17.0 |
| Train K=16 prompt 長度 | 中位 658、P99 1,343、最大 **3,577** tokens |
| Dev K=16 prompt 長度 | 中位 666、P99 1,580、最大 **3,578** tokens |
| 8,192 tokens 上限 | **0 筆超限**，無需截斷病例文字 |
| K=16 Train gold 選項 | 1 正確＋15 OOF BM25 標籤負例，43,180 筆皆建立；答案 slot 隨機打亂 |
| K=16 Dev 候選 | 5,486 筆真實 BM25 Top-16，**不補入**金標，與官方 BM25 frozen report 一致 |

Tokenizer 紀錄（彙總、無病例原文）：[完整 Token 審核](../experiments/jeff_k16_tokenizer_2026-11-11.json)。資料完整性請參閱 [K16 主要輸入報告](../experiments/jeff_primary_k16_2026-11-11.json)。

## 二、優先修正清單

| 優先 | 議題與實際證據 | 對訓練的風險／下一個動作 |
|---|---|---|
| **P0** | **31B 訓練路徑尚未存在**；已發布 Jeff 含合併過的 LoRA 權重及獨立 Decision Head | 必須用發布權重載入既有 `head.safetensors`，在合併後 backbone 上**新增** LoRA；凍結原本模型，訓練新增 LoRA 與 `delta.weight/bias`；檢查梯度和重載。不可誤用 `adapter/` 再疊一次原版 LoRA。 |
| **P0** | **模型前向不可直接使用常見 LM Trainer** | Jeff 使用 backbone 最後**post-final-norm**的 hidden state，套用 Letter Readout + softcap + 已訓練 Residual Head；不計算整個 Vocabulary 的自回歸 LM Loss。**右側 padding 必須按每筆最後有效 token 取 hidden state**；batch 中少於 16 選項需遮罩剩餘 logits。 |
| **P0** | **訓練前必須證明 zero-step parity** | 在固定病例＋選項上，用官方 Jeff runner 比對自行實作的未訓練 LoRA／Head 輸出，在精度容許下近似一致；再跑小批次 loss 下降、梯度非零、checkpoint 保存／讀回一致。這些都尚未完成，不能宣稱可正式訓練。 |
| **P1** | **595／43,180 Train 的金標不在 OOF Top-16**（其中 136 不在 Top-50） | 目前訓練一律插入金標，部分多選題與正式搜尋情境有落差。**不能刪除病例**；保留全部並標記「原本搜尋未命中」供分層報告／必要時調整損失權重，不用 Dev 金標補候選。 |
| **P1** | **長尾極不均衡**：最大類 7,380、最小 7，比例約 1,054:1；28 類不足 16 筆、66 類不足 32 筆 | 比較自然分布、sqrt-inverse-frequency 或有上限的 class-aware sampler（相同總更新數）；不能僅報 Micro／Accuracy。報全 148 類 Macro-F1、每類 support 與信賴區間。 |
| **P1** | **同一證型被當負例的頻率失衡**：不同類別的負例/正例暴露比例差距很大，部分稀有類型常被當成錯誤候選 | 對 BM25 負例池做類別曝光稽核，設計負例曝光上限或分組採樣消融，避免模型學到「少數證型總是錯」。不擅自修改正式標籤。 |
| **P1** | **原 StratifiedKFold 有 6,054 筆查詢可在其他折找到完全相同文字** | OOF 避開自身資料列仍可能碰到其他折複本；已驗證**相同全文同折的 StratifiedGroupKFold** 可保留 43,180 筆並達成跨折同文字 **0**。保留原始 OOF 及此替代 OOF 作訓練資料消融；不刪病例，不動官方切分。 |
| **P1（已修正可程式判定部分）** | **同一臨床文字出現不同標籤**：原始 Train 有 820 筆涉及衝突群組，舊訓練題有 817 筆把另一個金標當負例 | 保留病例與原始標籤；**新 Group-aware K=16 epoch02 已過濾全部 793 筆可確認矛盾負例**。尚需醫學專家確認非完全相同病歷的語意同義證型；應在論文呈現衝突病例分層分析。 |
| **P1** | **Jeff 1.1 原訓練含教師 soft targets；我們只有 one-hot 硬標** | One-hot CE 可以作 baseline，但需對照小幅 label smoothing／可信教師分布（若能合法取得），觀察 Macro-F1、NLL／ECE；公開模型已以通用任務訓練，不能直接套原作者的 1,500 steps 不檢查是否過擬合。 |
| **P1** | **未做原始 Jeff Zero-shot 基準** | 在同一批完整 Dev 的真實 Top-16 上，先跑未增量微調的 Jeff；再與 LoRA 比。不能用 BM25 Recall@16（98.60%）當作模型準確率。 |
| **P2** | **僅用證型名稱作為候選描述** | 官方 `syndrome_knowledge.json` 可映射**148／148 類**且都有證型定義。另做「只用名稱」與「名稱＋定義／典型症狀」消融，**訓練和推論必須同樣格式**；目前主基準先不改選項描述，避免實驗混因。 |
| **P2** | **語意同義證型尚未人工審核** | 字面 NFKC／標點正規化的名稱衝突為 0，但這不等於醫學語意沒有重疊。不能將等義有效證型當成絕對錯誤標籤卻不揭露；保留原題，抽樣做臨床檢核。 |
| **P2** | **Jeff 原先 calibration 不是中醫 Dev** | 發布 choice temperature 約 1.49；另在 Dev 作 NLL／ECE 校準，記錄校準前後。溫度縮放不會改善固定 K 的 Argmax Accuracy，但會影響機率可信度。 |
| **P2** | **Train–Dev 原始資料有文字重複** | 正式 Dev 的 **529 筆**與 Train 全文相同（含 20 筆標籤衝突），全部保留；應附完整組與重複／非重複分層成績，避免將官方 benchmark 誤寫成完全未見患者表現。 |
| **P2** | **H200 記憶體與訓練環境未實測** | BF16 31B 約 62GB 權重；先跑單筆及 microbatch=1 真實反傳測峰值顯存，確認 Gradient Checkpointing、PEFT rank=32/alpha=64、PyTorch／Transformers 版本及 optimizer。80GB 只是作者**推論**最低需求，不能當成訓練保證。 |

### 重要實驗控制

1. 同一 **K=16**、相同 43,180 Train、相同 5,486 Dev、相同選項 Prompt、固定訓練預算，比較：Jeff 不微調、Jeff LoRA（一般 OOF）、Jeff LoRA（完全相同文字同折 OOF）。若資源允許，再比較少數類別加權與定義增強。
2. **每個 epoch 重新洗牌選項並更新 target_index**。不能跨 epoch 一直讀 `epoch01`，卻聲稱資料有持續增強。
3. 使用真實、未補答案的 Dev 候選，算 **Top-1、Macro-F1、per-class 指標、候選覆蓋條件下的正確率、平均／P95 延遲、顯存、ECE**。K=1～35 的 BM25 曲線保持不動；Jeff 的 K=17～35 沿用分組決策。
4. checkpoint、最佳 K、Loss 或機率校準只以 Dev 決定。**Test 5,486 筆最後一次性評估**。官方資料切分、重複病例與金標一律保留。
5. 追蹤**每類正例數、被選作負例次數、正確證型原 BM25 排名、是否完整重複文字、是否標籤衝突**等分層指標；所有逐筆病例與標籤仍在 Git 忽略的 `data/`，不公開病例。

## 三、新增的 Group-aware OOF：已完整重算

Group-aware OOF 使用 **Train 內部相同臨床文字同折**，和原 OOF 使用相同的 Bigram BM25、資料量、5 折及原始標籤；只是分折方式不同。

| 指標（Train OOF，非 Dev 成績） | 原始 StratifiedKFold | Exact-text Grouped |
|---|---:|---:|
| Train 原始病例 | 43,180 | 43,180 |
| 跨折完全相同文字查詢 | **6,054** | **0** |
| Micro Recall@1 | 79.85% | 80.26% |
| Micro Recall@16 | 98.62% | **98.67%** |
| Macro Recall@16 | **90.39%** | 90.35% |
| 官方切分／原始病例 | 原樣保留 | 原樣保留 |

Grouped 的 K=16 Micro 增加約 **0.05 個百分點**，Macro 減少約 **0.04 個百分點**。這是**Train OOF 的候選覆蓋**，無法證明 Jeff 訓練後會更準；主要貢獻是排除同文字跨折捷徑。兩個候選排名分別保存，不會覆蓋原 BM25 Dev 或原 Train OOF。

[原始 Train OOF 實驗](../experiments/bm25_train_oof_2026-11-11.json) ｜ [Group-aware Train OOF 實驗](../experiments/bm25_train_text_grouped_2026-11-11.json)

## 四、已修正：同一病歷不同金標被當成困難負例

**這是針對原先訓練資料找到的實際標註衝突，已另外製作安全訓練版本。** 舊資料／舊統計不覆寫，保留作方法消融。

- Train 官方資料有 **820 筆**屬於相同完整臨床文字卻對應不同證型的群組（共 252 個衝突群組）。全部保留。
- 原 Stratified OOF 的 **817 筆 K=16** 訓練題目，把該病歷的另一個官方金標放進負例。
- 新 Group-aware OOF 在**未過濾**時也有 **793 筆**相同風險。
- 現在預設將「相同臨床文字在 Train 曾被標為正解的其他證型」從 50 個 BM25 候選的負例池排除；依序往後補滿 **15 個真正可訓練的負例**。病例本身及本列官方金標完全不改。
- 正式產生 `epoch02`：**43,180 筆、每題 K=16**；共 **793 筆題目**避免矛盾，其中 **797 個負例被替換**，實際逐筆驗證已知衝突 **0 個**，保留全部 43,180 筆。
- 這項過濾只修正**相同臨床文字下明確可證實的矛盾金標**；不同病歷的同義證型或醫學上可並存的證型**尚未人工核對**。同一病例出現不同官方金標的訓練列仍依原規範保留，故也不能聲稱所有醫學標註矛盾已消除。

| 版本 | 五折策略 | 負例策略 | Train 筆數 |
|---|---|---|---:|
| 原始 `epoch01` | StratifiedKFold | 只排除本列金標（歷史資料） | 43,180 |
| 新 `epoch02` | 相同臨床全文同折 | 另排除相同全文的其他官方金標（**建議主訓練資料**） | **43,180** |

生成新版本的可重現指令（輸出路徑採 `open("x")` 不覆寫原始資料）：

```bash
uv run python scripts/mine_jeff_train_oof.py --fold-policy exact_text_grouped
uv run python scripts/export_jeff_train.py \
  --candidates data/candidates/bm25_train_text_grouped.jsonl \
  --oof-report experiments/bm25_train_text_grouped_2026-11-11.json \
  --epoch 2
uv run python -m scripts.verify_jeff_grouped_filtered
```

該執行紀錄已封存；若要重新產生不同選項排列，改用未使用的 `--epoch`。逐筆病例保存在 Git 忽略的 `data/jeff/`，僅提交無病例原文的 [完整驗證報告](../experiments/jeff_grouped_filtered_k16_verification_2026-11-12.json)。真正模型訓練前，須在正式 DataLoader **每個 epoch** 重建選項排列，不能將目前的 `epoch02` 複製重播卻宣稱有動態增強。

### 五、真正允許啟動 H200 LoRA 的驗收門檻

- [x] 官方 Train／Dev 分母、K=16 資料格式與逐筆答案位置核對。
- [x] 使用發布版 GemmaTokenizer 對全部 K=16 Train／Dev 實際統計長度，**沒有超長 prompt**。
- [x] 原始／group-aware OOF 各產生 43,180 筆，兩者跨折同文字查核並記錄。
- [ ] 官方 Jeff 發布權重與 Head 在 H200 成功載入，版本可重現。
- [ ] 相同輸入下，新訓練器 **step 0 機率與官方 runner 一致**；驗證 BF16 Backbone／FP32 Head。
- [ ] 帶 padding 的 microbatch 取正確最後有效 Token；K<16 logits 正確遮罩；只將 target_index 供 Loss，輸入 Prompt 不含金標。
- [ ] 訓練一次步驟確認 LoRA 與 Decision Head 有梯度，Backbone 凍結，Loss 有限且下降，保存／讀回答案一致。
- [ ] 用實際 H200 確認峰值顯存、訓練吞吐與 Token／step，再決定合理的 step、warmup／scheduler、早停。
- [ ] 先封存完整 Dev 的 **Jeff 原版 Zero-shot Top-1、Macro-F1**，再開始領域微調，以辨別性能倒退或真正提升。

**資料已準備好，GPU 模型訓練器仍未完成。以上尚未執行項目不得勾選或稱為實測。**
