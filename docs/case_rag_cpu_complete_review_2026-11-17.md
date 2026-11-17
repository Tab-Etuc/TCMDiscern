# TCMDiscern：完整 Case-RAG H200 CPU 診斷定稿（2026-11-17）

## 來源與資料狀態

使用者已將 H200 產出的 `case_rag_cpu_first_2026-11-17.json` 放入本地專案。已核對 JSON 內容並歸檔到：

- `experiments/case_rag_cpu_first_2026-11-17.json`（此檔為**H200 完整版聚合報告**）
- H200 原始執行：`CASE_RAG_CPU_DIAGNOSTICS_COMPLETED`、官方 Test 未使用。
- 官方 Train **43,180**、Dev **5,486**、**148** 證型、**37** 個最低頻類別、尾類 gold **83** 筆；排除 Train/Dev 完全同文後 **4,957** 筆。
- 聚合報告包含 **12 組方法的指標、19 個逐病例配對統計、4 個 Train 頻率分桶、固定頻率覆寫分析**。可分析全體錯誤交集、淨增益與上限，但原始逐病例 logits 仍在 H200；本地只有 H200 聚合統計。
- 7 組已完成的 Case-RAG arms：`s1_global2`、`s2_global2`、`s2_random2`、`s2_global2_labelhide`、`s2_global2_labelonly`、`s3_global2`、`s3_global2_labelhide`。
- 原先 `analyze_case_rag_cpu_first` 在 H200 遇到的 `s3_base_vs_bm25` KeyError 已於 Git commit `3c30ffc` 修正；此份 H200 JSON 確實同時保留原始與 Global2 的 baseline 配對鍵值，符合修正後行為。

本報告數字來自上述完整聚合 JSON，百分點（pp）與 148 類 Macro-F1、37 類 Tail-F1 維持原始計算定義。

## 一、完整 Dev 成績

| 方法 | Top-1 | Macro-F1（148） | Tail-F1（37） | 無 Train 完全同文 Top-1（n=4957） |
|---|---:|---:|---:|---:|
| BM25 | 79.5844% | 54.8496% | 41.7568% | 78.5556% |
| 來源病例最高排名標籤（CPU） | 80.2953% | 55.1720% | 41.7568% | 78.7977% |
| S1 原始 | 79.2198% | 40.3404% | 18.6486% | 77.7285% |
| S1 + Global2 | 82.9019% | 50.9572% | 30.7207% | 81.6623% |
| S2 原始 | 80.3500% | 42.6654% | 22.1879% | 79.1406% |
| S2 + Global2 | 82.5921% | 50.0111% | **31.0553%** | 81.3597% |
| S3 原始 | 80.0948% | 41.2788% | 20.1802% | 78.7170% |
| **S3 + Global2** | **83.4488%** | **51.4706%** | 29.4595% | **82.2877%** |

S3 + Global2 比原始 S3：Top-1 **+3.3540 pp**（+184 筆）；Macro-F1 **+10.1918 pp**；Tail-F1 **+9.2793 pp**。在排除 Train/Dev 完全同文後的 4,957 筆，Top-1 為 **82.2877%**，增益方向一致。

## 二、Global2 與 BM25 的真正互補範圍

| S3 對 BM25 | Jeff 獨對 | BM25 獨對 | BM25 + Jeff Oracle Top-1 |
|---|---:|---:|---:|
| S3 原始 | 485 | 457 | **88.4251%** |
| **S3 + Global2** | 429 | 217 | **87.4043%** |

- S3 + Global2 比 BM25 多正確 **212 筆**，Top-1 +3.8644 pp；未校正 exact McNemar p = **5.41×10⁻¹⁷**。
- 對各自原始 S3，Global2 救回 **269** 筆、破壞 **85** 筆，淨增 **184**；未校正 exact McNemar p = **2.18×10⁻²³**。
- **檢索增強提高單一 Jeff 的分數，卻使它與 BM25 的理想二選一上限由 88.4251% 降為 87.4043%**（約 −1.021 pp）。可保留 Jeff 類強判斷與 BM25 少數類正確判斷的機會仍存在，但比原始 S3 略小。
- 上述 Oracle 用 gold 在事後選擇正確預測，**只作為可挽救空間上界**，絕非可部署模型分數。

## 三、來源病例的診斷標籤依賴

| 方法 | Top-1 | Macro-F1（148） | Tail-F1（37） |
|---|---:|---:|---:|
| S2 + Global2 | **82.5921%** | **50.0111%** | **31.0553%** |
| S2 + Random2 | 80.3500% | 42.4292% | 19.1892% |
| S2 + Label-Hide | 80.0401% | 42.3871% | 22.7928% |
| S2 + Label-Only | 81.1703% | 45.0105% | 22.9730% |
| S3 + Global2 | **83.4488%** | **51.4706%** | **29.4595%** |
| **S3 + Label-Hide** | 80.1495% | 41.3615% | 18.3784% |

- S3 隱藏來源標籤後，比完整 Global2 的 Top-1 下降 **3.2993 pp**；配對統計顯示 Global2 多救回 **256**、多破壞 **75**，未校正 p = **2.83×10⁻²⁴**。
- S2 也相同：Global2 比 Label-Hide **+2.5520 pp**；比 Label-Only **+1.4218 pp**。
- 兩個微調方法都呈現來源標籤高度影響 Jeff 判斷；臨床文字仍可能提供互補價值，但標籤污染／捷徑是論文需要明確分析的研究問題。

## 四、長尾分析：簡單頻率保護只能形成取捨

以 **S3 + Global2** 作為基準，當 BM25 預測的證型 Train 頻率不超過閾值、Jeff 預測的證型頻率高於閾值且二者衝突時，改採 BM25。這些閾值是已檢視 Dev 上的描述性探索。

| BM25 頻率上限 | 改判病例 | Top-1 | Macro-F1（148） | Tail-F1（37） |
|---|---:|---:|---:|---:|
| 原始 S3 + Global2 | 0 | **83.4488%** | 51.4706% | 29.4595% |
| ≤9 | 15 | **83.4488%** | 53.2987% | 36.3964% |
| ≤49 | 117 | 83.1936% | **55.9641%** | **42.2072%** |
| ≤199 | 285 | 82.9384% | 55.5612% | 35.9009% |

- ≤9 的 15 次切換救回 6、破壞 6，整體 Top-1 不變；固定 148 類 Macro-F1 提升約 **1.8281 pp**。
- ≤49 的 117 次切換救回 37、破壞 51，整體 Top-1 −0.2552 pp，148 類 Macro-F1 +4.4935 pp。
- **BM25 預測尾類但 S3 + Global2 不預測尾類** 的 39 筆分歧：BM25 正確 14、Global2 正確 18、兩者同錯 7。尾類本身不足以判斷何時改判；需要更強的臨床證據可靠性特徵。

## 五、同切分歷史強基準與 CPU 路由

早於這批 Case-RAG 實驗，專案於同一官方 Dev 已取得：

- **Joint Clinical Signature SGD + Jeff S2**：**85.180% Top-1**、59.387% Macro-F1。
- **Plain SGD + Jeff S1**：**85.071% Top-1**、**60.960% Macro-F1**。

目前單純 S3 + Global2 83.4488%／51.4706%，以及使用者先前提供的 H200 **五折分組 CPU 路由**結果：

| CPU 路由門檻 0.5 | Top-1 | Macro-F1 | Tail-F1（37） |
|---|---:|---:|---:|
| S1 + Global2 | 84.451% | **58.800%** | **45.109%** |
| S2 + Global2 | 84.160% | 57.461% | 40.245% |
| S3 + Global2 | 84.397% | 58.061% | 42.136% |
| S3 原始 + 路由 | **84.543%** | 58.376% | 44.659% |

路由結果來自已反覆研究的 Dev 內分組交叉擬合，需標記為探索性。歷史 85% 融合分數亦受先前 Dev 門檻選擇影響。這些都是開發集比較，**不能直接宣稱獨立泛化證據**。

> 不同歷史報告的 Tail 定義可能是 11 類，而本次 37 類。只有固定 148 類 Macro-F1 和相同 Dev Top-1 可直接橫向比較；未核對的 Tail 數字不合併。

## 六、定稿研究決策（依現有資料）

1. **CPU 分析階段已具備基本判斷**：Global2 有穩定增益，來源證型標籤貢獻大，對 BM25 的互補空間縮小，長尾與 Top-1 存在取捨。
2. **下一個最值得做的是既有強方法融合**：採已固定的 SGD＋Jeff 成對融合，將 Jeff scores 從 Base 換成已保存的 Global2 logits，原封沿用固定權重／成對門檻，再檢驗能否超越 **85.180% Top-1／60.960% Macro-F1**。此工作只需 H200 的現有 logits 與 CPU 重新評分；在本地完整 source logits 尚未同步前只能先完成方案定稿。
3. 任何新融合規則如需學習，優先改用官方 Train OOF／預先分組 fit/cal 訓練，Dev 先前已被多次查看，新的 Dev 增益只可作探索性。
4. 先保留 S3/S1 其餘三組來源消融 GPU 批次與 Seed 2、3 的暫緩狀態。若 CPU 現有強方法融合有明顯改善，再決定是否需要額外 GPU 訓練／消融。
5. 論文 Methods/Results 必須記下 Global2＋BM25 的 Oracle 上限下降、S3 Label-Hide 的性能回落、歷史高分強基準和 Train/Dev 同文敏感度；單純「加入兩筆醫案」尚不足以當作 J-BHI 的原創方法。

## 七、固定 SGD + Jeff 成對融合已在本地完成強基準重現

已新增 `scripts/evaluate_case_rag_strong_sgd_fusion_cpu.py`。此程式只使用專案已有的 Python/sklearn/TCM-SD，完全複製先前的字元 TF-IDF、Class-balanced SGD、固定 Jeff Top10/Gate 和 pairwise_margin1（閾值 1.0）規則，唯一待比較的變因是原始 Jeff logits 與 Global2 logits。**原始 Jeff S1 結果在本地完整 Train→Dev 重現後才會進入 Global2 評分**。

本地 CPU 實際執行結果已保存於 `experiments/case_rag_strong_sgd_baselines_cpu_2026-11-17.json`（只包含聚合數值、原始強基準，後續 H200 Global2 評分另存新檔）：

| 強融合方法（同樣 5,486 筆 Dev） | Top-1 | 148 類 Macro-F1 | 最低頻 37 類 Macro-F1 |
|---|---:|---:|---:|
| **Plain SGD + Jeff S1 原始** | **85.0711%（4667 正確）** | **60.9597%** | 45.9202% |
| Plain SGD + Jeff S2 原始 | 84.9617% | 59.4761% | 42.4067% |
| Plain SGD + Jeff S3 原始 | 84.9253% | 59.6804% | 43.1274% |

其中 S1 的 85.0711%／60.9597% 與 `experiments/linear_jeff_fusion_paired_diagnostics_2026-11-16.json` **完全一致**，因此這套沒有依賴本地 untracked 研究程式的固定 SGD+Jeff CPU 重現流程可以交付 H200。上述三組完整 Dev 的原始模型強融合是本地已完成結果；Global2 版本還需 H200 已有的 score JSONL。

H200 **下一個唯一需要執行的 CPU 工作**：

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main
.venv/bin/python -m scripts.evaluate_case_rag_strong_sgd_fusion_cpu --strict-global2
```

此命令會先重現 S1 Baseline（4667/5486、60.9597% Macro-F1），如未重現即直接中止；成功後一次評分 S1/S2/S3 原始與 Global2 共六組，輸出聚合報告 `experiments/case_rag_strong_sgd_fusion_cpu_2026-11-17.json`。完全在 CPU 上執行；不載入 Jeff 31B、不讀 Test、不改動任何 GPU checkpoint、不在 Dev 上重新搜尋閾值。最終分數與 pairwise rescue/break 將直接用來判定 Case-RAG 能否提升歷史強基準。

**完成條件**：本次 12 組方法、19 組配對分析與長尾頻率保護均已從 H200 聚合 JSON 稽核並歸檔；新的強融合 CPU 基準也已在本地完整重現。接下來只需要 H200 保存的 Global2 logits 給相同固定規則評分。
