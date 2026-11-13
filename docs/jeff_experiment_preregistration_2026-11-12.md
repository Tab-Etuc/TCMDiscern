# Jeff + BM25／TCM-SD 預先定稿實驗規格（2026-11-12）

**狀態：實驗方案已預先固定；截至本次檢查，Jeff-1.1-Large 尚無真實 H200 訓練或 Dev 成績。** 本文件作為後續 H200 訓練的優先研究契約。任何偏離應記錄原因、發生日期與受影響實驗，不能事後用 Test 修正。

## 一、核心主張與不能混淆的效果

1. **H1 領域微調是否有幫助（主要）**：BM25 Top-16 → 已發布 Jeff-1.1-Large zero-shot，與相同 BM25 Top-16 → Jeff 新 LoRA + 已發布 Head 在官方 148 類單金標 Dev/Test 的 Top-1、Macro-F1 比較。原始 BM25 Top-1 為必要基準。
2. **H2 標註歧義（次要）**：S2 Masked CE 與 S3 Set Loss **使用完全相同的候選及病例順序**，才能直接測量 Loss 的作用。S1 單金標基準在 820 筆同文異標病例上與 S3 的候選集合不同，所以 S1→S3 的差異同時包含候選改變及 Loss 改變。S4 均勻 Soft CE 只作對照。
3. **H3 可救回的困難病例（探索）**：使用 Train-only Group-aware OOF 的真實金標排名 2～16 作難例，Loss 權重 ×2；第一名正確的病例和排名 17 以後的病例保持 ×1。**不增加訓練樣本曝光次數或優化器更新**。與完全相同 Seed、Epoch、K、Loss 的 S3 uniform 作成對比較。
4. 本研究只把 **148 類證型**定義為主要任務。中醫／西醫病名缺乏經確認的同版標準標籤，此次不混算；論文不將候選 Recall@K 稱為最終 Top-1。

## 二、唯一資料來源及凍結內容

- 官方 TCM-SD Train **43,180**／Dev **5,486**／Test **5,486**、148 類；不刪除重複患者、同文異標、不重新切分。
- 主模型 **jgeuter/Jeff-1.1-Large**（發布版 31B 已合併 backbone）＋既有 Head，新的 rank32/alpha64 LoRA，reasoning 關閉，使用上游 v1.3.0 的 SemIf Prompt，官方輸入 Token 上限 8192。
- 特徵：`chief_complaint`／`description`／`detection`；只使用病例輸入與 BM25 候選。已做的「主訴＋病史」BM25 消融作為與 TCM-SD 原論文較接近的公平對照。
- BM25 候選固定為 **Train 全部病例索引，中文 bigram，k1=1.5，b=0.75，min_df=2，每證型取最高病例分數**。Dev 每筆 Top-50 已凍結。訓練負例來自 Train 內 Group-aware 五折 OOF；**Dev/Test 金標不參與 BM25 索引、候選挖掘或訓練**。
- 主要 **K=16**；Dev 另比較 K=1（BM25）、K=5、17、35；真正全面 K=1～35 只對最後選出的模型做成本評估。K>16 使用 Jeff 官方 grouped decision，增加 forward 次數，不能與單次 K=16 的 latency 混算。
- 已有 252 組／820 列同文異標 Train；42,360 列只有單一觀察金標（98.1%）。「多標註」屬弱監督證據，未經醫師裁決，不能稱每個候選都是臨床真值。
- 最新逐筆審查：S1/S3 在 **42,360 筆單觀察正標**使用相同 16 個候選；新的訓練前選項 canonicalization 讓**同 Seed／Epoch 的 A～P 排列完全相同**。S2/S3/S4 共用單一訓練資料來源，差別只在 Loss。
- 難例 OOF 排名已全量重新數出：Rank-1 **34,656**、Rank 2～16 **7,950**、Rank 17～50 **434**、超過 50 **140**，全部原始 Train 列仍保留。K16 推論無法從後兩類救回金標，因此不對其加重。

## 三、H200 計算預算與固定超參數

**主實驗設定（對所有 S1/S3、全部 Seed 完全相同）：**

| 項目 | 固定數值 |
|---|---|
| 主模型 | Jeff-1.1-Large + new LoRA + original FP32 Head |
| 主要 K | 16（單次 Head 決策） |
| Seed | 20261112、20261113、20261114 |
| LoRA | rank 32、alpha 64、dropout 0，七種 attention/MLP projections |
| LoRA LR／Head LR | 2e-5／1e-4 |
| Optimizer／Weight decay | AdamW／0.01 |
| 梯度裁切 | max norm 1.0 |
| Gradient accumulation | **64 病例／update**（每病例各一個 forward/backward） |
| Scheduler | 40 updates linear warmup，之後 constant |
| Checkpoint | 每 300 updates，另於第 675／1350 次更新的 Epoch 邊界強制保存 |
| Epoch | **2 完整輪**，保存 **Epoch 1 與 Epoch 2** 的 checkpoint |
| 每 Epoch updates | ceil(43180/64) = **675** |
| 兩 Epoch updates | **1,350** |
| 每 Epoch 真實輸入 Token | 約 30.33M（未含 padding；需以 GPU 實際計數再確認） |
| 兩 Epoch 病例曝光 | 86,360；有效輸入約 60.66M tokens |
| 主評分 | 官方完整 Dev 的 Top-1／Macro-F1，固定 148 類 |

這個更新預算與原 Jeff 作者的 **每 seed 約 89.2M tokens／1,500 steps** 不一樣；僅用來說明規模，不聲稱直接重現原作者私有訓練。上游不同問題長度與 padding、真正模型計算量都不同，應另報每 GPU-step 真實 tokens、累積病例、耗時和記憶體。H200 第一次跑 1 update 和短時 16 updates，依實測吞吐先核算整批成本；若無法負擔，必須在**正式實驗開始前**一致調整所有方法的預算並留下新決策紀錄。

**正式主實驗，共 6 次訓練（12 model-epochs）：**

- S1（安全 one-hot CE）× 3 個相同 Seed × 2 Epoch。
- S3（multi-positive Set Loss）× 同 3 Seed × 2 Epoch。
- 每次保留 Epoch 1 與 Epoch 2 checkpoint，使用相同候選來源／模型初始化／訓練順序／選項 seed；所有測試進行成對分析。

**預先列出的額外消融，共 3 次訓練（3 model-epochs）：**

- S2 Masked CE，Seed 20261112，1 Epoch：與 S3 同 Seed／同 Epoch 比較，固定相同病例與全部 16 候選。
- S4 Soft CE，Seed 20261112，1 Epoch：同條件。
- S3 + Train OOF recoverable-rank 2～16 權重 ×2，Seed 20261112，1 Epoch：對照 S3 uniform 同 Seed／同 Epoch。
- **這些單 Seed 數據只是探索性比較。** 若準備宣稱 S3 在 Loss 因果因素上顯著優於 S2，**必須補做 S2 的另外 2 Seed**，維持同 Epoch/資源，再進行成對統計；不得僅憑一個 Seed 的差異寫成確認性結論。

若全部規劃執行，主線 12 model-epochs，額外 3 model-epochs，共約 **455M 真實輸入 tokens**（粗估，不含變長模型計算及其他 forward）；實際耗時只能由 H200 Pilot 測定。

**第三個 Epoch**只列為預先指定的延伸學習曲線，主要表以 Epoch 1/2 比較。若需做 Epoch 3，S1/S3 同 Seed/方法同時補跑，報告所有 checkpoint；未延伸的方法不能拿 Epoch 3 與只訓練一個 Epoch 的對照直接宣稱 LoRA/Loss 優勢。

## 四、H200 必須遵循的執行順序

1. 本機先跑完整 source/payload 稽核，確認所有正式 Train/Dev/OOF/候選/Prompt 逐筆對齊、無外部欄位、所有金標及多答案 mask 保留。傳輸後 H200 **再跑同一套檢查**。Test 原始檔不傳入訓練環境。
2. 在 H200 上確認 Jeff 原始釘選版本（Tokenizer、merged 31B、original FP32 Head）；先驗證 causal prefix、不思考的模型輸出、原版 logits；比對注入新 LoRA **Step 0 仍與原版 logits 一致**。
3. 只做 **1 update**，驗證梯度流到 LoRA 與 Head、finite loss、checkpoint 保存及重新載入 logits／optimizer／scheduler／RNG 與中斷續訓的一致性。
4. **16 updates、accumulation 64** 做正式吞吐／顯存 Pilot；由真實 updates/sec、tokens/sec 和 checkpoint I/O 推算整個 15 model-epochs 所需成本，確認硬體和期限可負擔。
5. 用相同超參數開始 S1/S3 三組成對 Seed，各訓 2 Epoch；**只在 epoch 邊界**評估完整 Dev（epoch1=675、epoch2=1350 更新），訓練迴圈不讀 Dev/Test。中斷必須使用原 run-id／相同設定恢復，不得默默重新初始化。
6. 等主研究有完整結果，再做 S2、S4、OOF 困難病例加權；如需擴張成為主研究，只可依上面預先列出的多 Seed 補測規則執行。
7. **完全凍結模型選擇與 K，再一次性執行完整官方 Test**。正式 Test 只能是最終報告，不能用來改模型、K、threshold、資料、Seed 或重新跑超參數。

### 執行例子（本身不會自動開始；H200 人工執行）

```bash
python -m scripts.audit_h200_payload --check-only
python -m scripts.audit_jeff_training_controls --check-only

# 真實 BF16 GPU 安全檢查（優先於全量訓練）
python -m scripts.train_jeff_lora --model-dir /models/Jeff-1.1-Large --method s3
bash run_h200.sh pilot  # 2026-11-13 修訂：單次模型載入，2 updates × 64 cases，驗證真實 LoRA／Head 更新及 Checkpoint 重載

# 主線 S3 第一 Seed，與 S1 使用完全相同的設定
python -m scripts.run_jeff_finetune --model-dir /models/Jeff-1.1-Large --method s3 --run-id s3-seed20261112-e2 --steps 1350 --epochs 2 --accumulation 64 --seed 20261112 --warmup-steps 40 --checkpoint-every 300

# 次要消融：只加權原 Train OOF 排名 2～16 的病例
python -m scripts.run_jeff_finetune --model-dir /models/Jeff-1.1-Large --method s3 --run-id s3-seed20261112-e1-hard2 --steps 675 --epochs 1 --accumulation 64 --seed 20261112 --warmup-steps 40 --checkpoint-every 300 --weighting recoverable_oof_x2
```

S1 換 `--method s1`；S2/S4 換 `--method s2/s4`。其他 Seed 僅換 Seed 及 run-id，禁止同時改 LR、accumulation、Token 長度、候選檔或 K。

## 五、可投稿的主要評分、救回分析與統計

- **主要端到端指標：** official single-gold Top-1 Accuracy，固定 148 類 Macro-F1。公開來源論文的 Dev 與 Test 分開列表，與我們自己的成績只比較同一 split；輸入欄位不同時必須顯著標示。
- **必附：** BM25 Top1；BM25 Recall@K（micro/macro）；zero-shot Jeff K16；每個 LoRA loss/checkpoint 的 Top1、Macro-F1；各類別 support、Recall、F1；模型峰值 GPU 記憶體、GPU 训练秒數、每筆推論秒數。
- **救回／改錯：** K16 的 Dev BM25 baseline 為 **4,366 正確**；另 **1,043** 筆第一名錯但正解仍在前 16；**77** 筆完全不在前 16。每個 Jeff 要同時報（a）保住 4,366 正確病例的比例，（b）救回 1,043 筆的比例，（c）改錯數量，與結果 Top1，不能只報「救回案例」。
- **長尾分析：** 按 Train 支援度先固定分層：1～10、11～20、21～50、51～100、101～500、>500。Macro-F1 列 148 類，不能事後只選提升的類別。
- **歧義分析：** 原 Train 同文異標 820 列只是觀察標註；Dev 相同輸入衝突子群獨立報告，但類別和數量有限，信賴區間需要保守解讀。
- **多 Seed：** S1/S3 同 3 Seeds 的成對比較，報每 Seed、均值、標準差；Dev 選**方法及 epoch**依三 Seed 平均 Top1，Macro-F1 當平手裁決，不挑單次分數最漂亮的 Seed。正式 Test 針對選中的方法與 Epoch，使用**預先固定的第一 Seed**，避免以 Dev 選「幸運 Seed」。
- **不確定性：** 用原始完整病例臨床文字作群組，計算**成對 Group Bootstrap 2,000 次**的差值 95% 百分位信賴區間；保存原始樣本數、群組數、seed 與計算方法。對 Macro-F1 用固定 148 類名單計算，避免採樣恰無稀有類別時分母變動。考慮多種方法／多 checkpoint 的選擇後，Dev p-value 只作探索，Test 才作最終確認。
- **選項排列魯棒性：** 主表固定原始 BM25 排名，不在 Dev/Test 換金標位置。追加控制性重排實驗可取固定樣本做 3 次 deterministic option permutations，按 ID 反向對位後檢查 Top1 一致率；此為穩定性診斷，不混入主要 Accuracy。

## 六、研究變更停止規則

發生任何一項必須停止或回到前一驗收階段：

- 原始 Train/Dev、OOF 候選或兩套 JSONL 訓練檔的列數、對齊、正標、已知衝突負例有異常。
- Gemma4 causality/prefix、原版 16 logits vs Step0、LoRA 或 Head 非零梯度、BF16 值域、checkpoint 續訓、顯存不可通過。
- S1/S3 在 42,360 個單觀察正解病例上，對應 seed/epoch 的完整候選 A～P 排列不同。
- S2/S3/S4 未採完全相同候選集；困難病例權重讀到 Dev/Test，或錯把排名 17 以後列為可救回難例。
- 研究者根據 Test 決定 K、Epoch、Loss、Seed、訓練權重、Prompt。
- 某次實驗超出預先固定步數、提前停止後沒有註明，或使用不同 Token/病例曝光卻宣稱同預算公平比較。

## 七、已驗證的本機資料

- 已在 CPU 重新執行 **43,180 Train × 2 種資料 × 2 Epoch = 172,720 次**訓練題重排與正解對位：全部通過。
- 其中 **42,360 個單觀察標註病例**在每個 Epoch 的 S1/S3 完整選項位置一致；其餘 **820 個多觀察標註病例**的候選變化已明確隔離。
- Train-only Group-aware OOF 7,950 個金標排名 2～16，可用於固定 ×2 難例加權；其餘 35,230 筆維持 ×1。
- 原始 BM25 Dev：Top1 **79.5844%**、Macro-F1 **0.548496**、Recall@16 **98.5964%**。尚無任何真實 Jeff 微調數字。
- 論文對照：TCM-SD 原始 CCL 2022（https://aclanthology.org/2022.ccl-1.80/）；Jeff 原研究（https://arxiv.org/abs/2610.17205）；Jeff v1.3.0 官方推論實作（https://github.com/j-geuter/Jeff/tree/v1.3.0）；部分標籤學習（https://jmlr.org/beta/papers/v25/23-0868.html）。

**注意：這個方案不聲稱 43,180 筆人工標註都經臨床確認，也不將 31B Zero-shot 的未知分數編造成已測結果。**
