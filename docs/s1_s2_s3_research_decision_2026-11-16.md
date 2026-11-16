# TCMDiscern：S1／S2／S3 真實 Dev 比較與後續研究決策（2026-11-16）

## 決策

**下一階段 Case-RAG 的唯一首輪主模型：S2 Epoch 2（Seed 20261112）。** 沿用已完成的 Jeff LoRA／Head checkpoint，以原本 BM25 Top-16 的 16-slot raw logits 作為推論基準，無須重訓。

- **S2**：未額外校準的 Top-1 及 Macro-F1 皆為 S1／S2／S3 最高，作為 Case-RAG 原始輸出和新醫案效果的主實驗。
- **S1**：固定 Prior+Gate 的 148 類 Macro-F1 最好，保留為長尾診斷強基準；可對照新 RAG 是否真正克服長尾問題。
- **S3**：固定 Gate Top-1 較高，亦提供有限的不同答對病例；保留 checkpoint 和 logits 作獨立消融，待 S2 的 Case-RAG 醫案效果有訊號後再比較 S3，停止新的 S3 長訓練。
- **三模型等權融合**：目前屬推論成本增加的探索性上界對照。沒有顯示 Macro-F1 同時超過 S1 Gate 的穩健效果。

此決策是單 Seed 的下一階段**工程／研究優先順序**，不能取代原預註冊的三 Seed 平均正式選模。完成第二、第三 Seed 與獨立評分後才能做確認性結論；既有 Dev 已多次用於校準探索。

## 已驗證資料來源

- 官方 TCM-SD Train 43,180，Dev 5,486、固定 148 類；BM25 Top-16 固定順序，沒有 Dev 金標補入候選。
- 三組皆為同一個 Seed 20261112、Epoch2（1,350 updates）、Jeff 模型 revision 970c21af246a44c5527f7109fca129ab6015c941、attention_backend=axolotl_hybrid_fa2_sdpa。
- 已逐列確認三組 Dev logits 各 5,486×16、case_id 同序、數值有限，原始 Top-1／Macro-F1 與 H200 評估 JSON 完全相同。
- Dev 有 529 筆臨床文字和 Train 完全相同；Dev 臨床文字共有 5,139 個群組，統計重抽樣按完整文字分組。
- Test 保持封存。

## 同口徑指標

| 方法 | Top-1 | 固定 148 類 Macro-F1 | Train tail 11 類 Macro-F1 | vs BM25 救回／破壞 |
|---|---:|---:|---:|---:|
| BM25 | 79.584% | 54.850% | 40.000% | — |
| S1 raw | 79.220% | 40.340% | 13.333% | 491 / 511 |
| **S2 raw** | **80.350%** | **42.665%** | 19.394% | 465 / 423 |
| S3 raw | 80.095% | 41.279% | 13.333% | 485 / 457 |
| **S1 fixed Prior+Gate** | 82.829% | **59.312%** | 38.225% | 252 / 74 |
| S2 fixed Prior+Gate | 82.975% | 58.110% | 35.758% | 242 / 56 |
| S3 fixed Prior+Gate | **83.194%** | 58.483% | 37.922% | 284 / 86 |
| S1+S2+S3 等權 logits + 相同 Gate | 83.339% | 59.034% | 38.225% | 261 / 55 |

固定 Gate 沿用 S1 曾經在 Dev 探索的超參數：先對各證型減去 `0.6*log(Train 類別數)`，於前 10 個 BM25 候選中挑戰第一名，改判須達到 logit 差距 3。三模型等權是後續追加的簡單融合探索，無權重網格搜尋；**表中 Gate 及融合數字皆屬已反覆觀察的 Dev 描述性結果**。

對照排除 Train 完全同文後的 4,957 筆：

| 方法 | Top-1 | Macro-F1 |
|---|---:|---:|
| S1 fixed Gate | 81.703% | 58.520% |
| S2 fixed Gate | 81.763% | 57.219% |
| S3 fixed Gate | 81.945% | 57.615% |
| 三模型等權 fixed Gate | 82.126% | 58.110% |

Train tail 11 類在 Dev 僅有極少病例，該項 Macro-F1 波動大；即使數值略高，也應以固定全 148 類的正式指標判斷。

## 是否支持 S3 取代 S2？

逐病例比較 S3 與 S2 raw：

- S3 獨自答對、S2 答錯 **163 例**；S2 獨自答對、S3 答錯 **177 例**。淨差 **-14 例**。
- 精確 McNemar 雙尾 p = 0.481；同臨床文字 cluster paired bootstrap 2,000 次，Top-1 差 **-0.255 pp**，95% CI **[-0.919, +0.404] pp**。
- S3 相對 S2 raw Macro-F1 差 **-1.387 pp**；cluster bootstrap CI **[-3.672, +0.992] pp**。
- S3 fixed Gate 相對 S1 fixed Gate，正確病例 **+20**，Top-1 差 +0.365 pp，bootstrap CI **[-0.091, +0.873] pp**；Macro-F1 差 -0.829 pp，CI **[-2.128, +0.778] pp**。

**S3 有互補性，但現在無法支持「Set Loss 一定比 Masked CE 更好」。**

事後利用金標挑選模型的 oracle（僅為分析上界）：

- BM25 + S1 raw + S2 raw：**4,902/5,486 = 89.355%**。
- 再加入 S3 raw：**4,928/5,486 = 89.829%**，僅新增 **26 例**。
- BM25 + S1/S2 fixed Gate 的 oracle：4,667 例；再加 S3 fixed Gate：4,703 例，新增 **36 例**。
- 三模型等權 fixed Gate 相對 S1 fixed Gate 淨多正確 **28 例**，Top-1 差 +0.510 pp、bootstrap CI **[+0.127, +0.898] pp**，Macro-F1 差 -0.278 pp、CI **[-1.699,+1.147] pp**。這是看過 Dev 後追加的探索對照，須獨立資料驗證。

## 對三組使用相同 S2 Listwise 校準器（追加 CPU 比較）

為排除 S2、S3 僅因採用 S1 Gate 而排序不同的疑問，再將原本在 S2 探索過的 `rare_k16_d0.2` 規則**原樣套用到三組 logits**，重做相同 group-aware 5-fold cross-fit。每個 fold 的 LR 只使用其他四折的 Dev 標註做 fitting，特徵和超參數固定為 `StandardScaler + LogisticRegression(C=0.05)`、Train 類別權重 `clip(sqrt(100/count),0.5,3)`、K=16、margin=0.2。

| 方法 | 同設定五折 Top-1 | 固定 148 類 Macro-F1 | Train tail F1 | 排除 Train 完全同文 Top-1 / Macro-F1 |
|---|---:|---:|---:|---:|
| S1 | 83.266% | 57.726% | 31.515% | 82.167% / 56.969% |
| **S2** | **83.558%** | **58.392%** | 38.182% | 82.389% / **57.537%** |
| S3 | 83.613% | 57.271% | 39.091% | 82.429% / 56.505% |

- S3 比 S2 只多對 **3/5,486 例**（S3 單獨答對 74 例，S2 單獨答對 71 例）；精確 McNemar p=0.868，Top-1 cluster bootstrap 95% CI **[-0.402,+0.495] pp**。
- S3 Macro-F1 比 S2 低 **1.121 pp**，bootstrap CI **[-2.420,+0.717] pp**；S2 在這套校準器下的整體 Macro-F1 仍較好。
- 校準器由 Dev 五折訓練，原先的設定也曾在這一份 Dev 上探索，故此處為相同口徑下的*探索性比較*；沒有可直接用於獨立 Test 的固定 Train-only 校準器。

輸出：`experiments/s1_s2_s3_cpu/listwise_control.json`；程式：`scripts/compare_s1_s2_s3_listwise.py`。本次檢查維持 S2 為 Case-RAG 下一個主模型。

## 下一輪研究：S2 原始 checkpoint 的 Case-RAG 因果對照

針對目前 Jeff 僅看到 16 個證型名稱、無法比較真實醫案的問題，下一輪先固定 S2 checkpoint，檢驗「增加歷史病例是否改善新患者的辨證」，並控制：

1. `baseline`：沒有歷史醫案，確認原始 S2 logits／預測和先前 Dev 一致。
2. `balanced2` 與 `paired2`：相同兩個醫案預算，分離證型配對的來源選擇效應。
3. `querywindow2`、`fullcase2` 與 `randomwindow2`：同一批歷史醫案，比較文字選取、完整病史與隨機截取控制。
4. `random2`／`global2`：控制單純增加文字或檢索案例帶來的偽改善。

固定 Pilot 300 例（僅 56 個證型）可辨認效果方向、失敗模式和 H200 推論成本；其長尾覆蓋不足以代表完整 Dev 148 類 Macro-F1。若有一致的正向救回／破壞訊號，再於完整 5,486 Dev 重做凍結對照；負控制應一併保留。

### H200 指令

在**H200 已有 S2 Epoch2 checkpoint** 的專案目錄執行：

```bash
bash run_case_rag_h200.sh s2 --check
bash run_case_rag_h200.sh s2 --prepare-only

# 先驗證 5 個病例，和完整 Pilot 使用不同 run ID
bash run_case_rag_h200.sh s2 \
  --checkpoint data/runs/s2-seed20261112-e2/resume-0001350 \
  --require-baseline-parity --limit 5

# 300 個病例、10 個控制條件，獨立輸入 8 組，共 2400 次 forward
bash run_case_rag_h200.sh s2 \
  --checkpoint data/runs/s2-seed20261112-e2/resume-0001350 \
  --require-baseline-parity
```

如 H200 的實際 checkpoint 資料夾不同，以 `--check` 顯示的完整路徑替換；`--require-baseline-parity` 需要同一模型產生的 `data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl`。

**本機完成驗收**：`bash run_case_rag_h200.sh s2 --prepare-only` 顯示 `CASE_RAG_INPUTS_VERIFIED`、10 arms／8 unique，來源原文、病例順序及標註通過；本機可檢查資料與計算統計。H200 31B GPU 模型與 LoRA 權重目前留在 H200。

## 再現方法和輸出

```bash
cd /home/tabetuc/dev/TCMDiscern
.venv/bin/python -m scripts.analyze_s1_s2_s3
.venv/bin/python -m scripts.compare_s1_s2_s3_listwise
.venv/bin/pytest -q tests/test_s1_s2_s3.py tests/test_compare_s1_s2_s3_listwise.py
```

報告：`experiments/s1_s2_s3_cpu/summary.json`。完整三組 5,486×16 logits 位於私人 `data/jeff/`；來源 `tcmdiscern_s3_export.tar.gz` 已驗證並解壓縮。程式 `scripts/analyze_s1_s2_s3.py` 僅讀官方 Train、Dev 與候選，不開啟 Test，使用固定 148 類計分、同文本群組 bootstrap 2,000 次和精確 McNemar。

研究紀錄和單 Seed 探索有助決定 GPU 成本優先順序。後續正式論文須明列原註冊的三 Seed 設計以及此次改為單 Seed Pilot 的偏離，再用獨立留出／多 Seed 驗證核心結論。
