# TCMDiscern Case-RAG 研究紀錄（2026-11-17 定稿快照）

> 本文件記錄截至 2026-11-17 已確認的實驗成果、研究問題、論文規劃及待執行批次；新增實驗保留附錄與版本紀錄，避免事後改寫既有結果。
> 實驗前方法鎖定版本：`docs/case_rag_jbhi_experiment_lock_2026-11-17.md`。
> 後續具體工作：`docs/case_rag_roadmap_2026-11-17.md`。

## 一、研究目標與定義

研究名稱：**TCMDiscern**。主題：以固定 BM25 Top-16 候選、LoRA 微調後的 Jeff-1.1-Large（31B）決策頭辨識中醫證型，分析加入兩筆歷史 Train 醫案的辨證效果、證型分布公平性、病例證據與標籤歸因、執行成本。

- Dataset：官方 TCM-SD Train **43,180**、Dev **5,486**，共 **148** 個證型；官方 Test 保持封存。
- K=16，原始 BM25 候選與原始患者臨床資訊保持不變。歷史醫案自 Train 檢索；評估 gold 只能用於分數與資料稽核。
- S1／S2／S3：三個不同**微調方法**。本輪使用已完成的同一個 seed 的 Epoch 2 checkpoint。Seed 2、Seed 3 尚待後續驗證，**暫緩**；不要混淆 S2 與 Seed 2。
- Global2：選擇兩筆排名較高的 Train 歷史醫案，保留來源病例的證型與臨床文字。
- 主要指標：Top-1、固定 148 類 Macro-F1；尾類為**Train 頻率最低 37 個證型**，另報 Tail Macro-F1、Tail Top-1。跨模型／條件使用配對檢定及依重複病歷群組 Bootstrap；同一 5,486 筆病例不能視為多模型間的獨立樣本。

## 二、實驗完成進度

| 批次 | 狀態 | 可稽核結果 |
|---|---|---|
| 300 筆 Dev Pilot，S1/S2/S3，17 arms 與反事實 | **完成** | `case_rag_reports.tar.gz`；`docs/case_rag_three_model_results_2026-11-16.md` |
| 5,486 筆 Dev，S2：Global2、Random2、Label-Hide、Label-Only | **H200 完成** | `experiments/case_rag_paper_full_dev_s2_20261117.json` |
| 5,486 筆 Dev，S1/S3：Global2 | **H200 完成** | `experiments/case_rag_paper_full_dev_s1_20261117.json`、`...s3...` |
| 5,486 筆 Dev，S3/S1：Random2、Label-Hide、Label-Only | **已寫好、待 H200 執行** | `run_case_rag_followup_h200.sh` |
| Seed 2／Seed 3 的重新訓練與驗證 | **延後** | 先完成歸因分析與方法決策 |

完成的 H200 原始執行回報：`CASE_RAG_PAPER_FULL_DEV_COMPLETE`，`backbone_loads=1`，`official_test_used=false`。

**資料來源注意：** 以下完整 Dev 的 Global2 與消融數字取自 **H200 終端機已印出的正式報告**（使用者於 2026-11-17 提供）；本地專案先前的 `experiments/case_rag_paper_full_dev_*.json` 仍可能是 `PENDING_GPU` 的 CPU 基準版本。H200 逐病例 logits、完整 timing 與完成版 JSON 應在下一個同步階段歸檔，維持這份快照數字的可驗證性。

## 三、已確認的完整 5,486 筆 Dev 數據

百分比；pp 表示百分點。

| 方法 | Top-1 | Macro-F1（148） | Tail Macro-F1（37） |
|---|---:|---:|---:|
| BM25 Top-1 | 79.58% | **54.85%** | **41.76%** |
| 來源醫案最高排名標籤（CPU） | 80.30% | **55.17%** | **41.76%** |
| S1 Base | 79.22% | 40.34% | 18.65% |
| **S1 + Global2** | **82.90%** | 50.96% | 30.72% |
| S2 Base | 80.35% | 42.67% | 22.19% |
| **S2 + Global2** | **82.59%** | 50.01% | **31.06%** |
| S3 Base | 80.09% | 41.28% | 20.18% |
| **S3 + Global2** | **83.45%** | **51.47%** | 29.46% |

- Global2 對各自 Base：S1 Top-1 **+3.68 pp**、S2 **+2.24 pp**、S3 **+3.35 pp**；相應 Macro-F1 **+10.62、+7.34、+10.19 pp**。
- 本批 H200 配對 McNemar 印出的三模型 Global2 vs Base p 值均為 `0.000000`（僅六位小數呈現，實際值仍為正數）；完整統計應以原 JSON 的未四捨五入數值為準。
- **S3 + Global2 目前最高 Top-1 與 Jeff 系列 Macro-F1；S2 + Global2 的 Tail Macro-F1 在 Jeff 系列最高。**
- BM25、來源標籤 CPU 基準在 148 類 Macro-F1 均高於所有 Jeff + Global2；BM25 的長尾 Macro-F1 亦高約 10–12 pp。這是研究真正需要改善的臨床技術缺口。

### S2 完整 Dev 的來源歸因消融（已完成）

| 條件 | Top-1 | Macro-F1 | Tail Macro-F1 |
|---|---:|---:|---:|
| S2 Base | 80.35% | 42.67% | 22.19% |
| Global2 | **82.59%** | **50.01%** | **31.06%** |
| Random2 | 80.35% | 42.43% | 19.19% |
| Global2 Label-Hide（保留歷史臨床文字） | 80.04% | 42.39% | 22.79% |
| Global2 Label-Only（保留歷史證型標籤） | 81.17% | 45.01% | 22.97% |

- Global2 優於 Label-Only **+1.42 pp Top-1**，優於 Label-Hide **+2.55 pp**。
- 這支持「**來源標籤與歷史臨床內容共同提供有效訊號**」的研究假說；仍須控制 Prompt token、選例與標籤偏差才能形成因果或演算法主張。
- Random2 與 Base 同 Top-1，說明來源病例選擇品質重要；尚不可直接等同於已證明「醫理推理」。

### 樣本品質與研究限制

- Train/Dev 標準化完全同文：**529/5,486**（9.64%），排除後 Dev **4,957** 筆；Train/Dev 同文而 gold 相異 **20** 筆，患者 ID 重疊 **19** 筆。
- 長尾 37 類僅 **83 筆 Dev gold**，尾類結果須報每類樣本數與不確定性。
- 300 筆 Pilot 已用於挑選 Global2，而完整 Dev 又用於觀察最優 S3。因此目前是**開發集結果**；完整方法與參數定稿後，才進行一次獨立 Test／外部驗證。
- 已完成的「訓練加病例」只是強比較基準。論文若要主張有新方法，仍須新增可被消融證明的 **可靠度估計、臨床證據衝突處理、長尾保護或其他技術機制**。

## 四、目前唯一排程中的 GPU 批次（未完成）

`scripts/run_case_rag_followup_h200.py` 會以 S3→S1 順序補完：

| 模型 | 新增 forward | 條件 |
|---|---:|---|
| S3 | 16,458 | Global2 Label-Hide、Label-Only、Random2 |
| S1 | 16,458 | Global2 Label-Hide、Label-Only、Random2 |
| **共計** | **32,916** | **單次 Jeff 31B 主權重載入** |

原始三模型 Base、三模型 Global2 與 S2 全套消融既已完成，均重用；由檔案核對決定實際剩餘 forward。輸入 Token、checkpoint、注意力後端與前次 logits 都在 GPU 前預檢；輸出以病例順序斷點續跑。Seed 2、3 不參與。

**執行：**

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main
bash run_case_rag_followup_h200.sh --check
mkdir -p case_rag_logs
nohup bash run_case_rag_followup_h200.sh > case_rag_logs/followup.log 2>&1 < /dev/null &
echo "PID: $!"
```

預期先出現 `CASE_RAG_FOLLOWUP_PREFLIGHT_PASSED`，最後 `CASE_RAG_FOLLOWUP_COMPLETED`。全新執行時 `backbone_loads=1`；已完成後再執行應是 0。

## 五、時間預估與計算依據

前次 **300 筆 Pilot** 的 Jeff + 兩筆歷史醫案推論均值約 **243 ms/forward**，既往 H200 Jeff 權重冷載入約 **43 分鐘**。本批最多 **32,916** 新 forward：

- 推論中位情境：`32,916 × 0.243 秒 = 7,998.6 秒 ≈ 133.3 分鐘（2 小時 13 分）`。
- 加一次權重冷載入：`≈ 176 分鐘（2 小時 56 分）`。
- 再加輸入驗證、Checkpoint 切換、JSONL I/O、算分、H200 環境波動：預留額外約 10–35 分鐘。
- **實務安排抓約 3～3.5 小時；合理波動範圍約 2.5～4 小時。** Label-Only Token 較少，實際可能更快；需以 H200 `followup.log` 與輸出 `timings.jsonl` 的真實吞吐修正。

以上屬規劃估算，非已量到的此次 H200 實際總耗時。排除 Model 冷載入的單例 latency 不能當作端到端總延遲。

## 六、論文要寫什麼（IEEE J-BHI 稿件方向）

**暫定技術命題：** Evidence-attributed Case Retrieval for Efficient TCM Syndrome Differentiation（研究題名仍待新方法確立後定稿）。

1. **Introduction**：TCM 辨證的長尾證型與病例資料稀疏問題；檢索式方法整體準確率與尾類 Macro-F1 的實際衝突；具體研究問題與臨床價值。
2. **Related Work**：TCM-SD 系列、BM25 檢索、Jeff／LoRA decision head、retrieval augmented classification／case-based clinical reasoning、長尾學習、來源標籤偏差。相關工作需按可比資料與評分協定清楚對照。
3. **Methods**：固定 Train/Dev、BM25 K16、Jeff S1/S2/S3 的固定 checkpoint、Global2 排序、來源病例內容表示、Random2／Label-Hide／Label-Only 的因果對照設計；新增方法未確立前只寫已實作流程。
4. **Experimental Setup**：43,180/5,486、148 證型、官方 Test 封存；GPU／FA2、token 上限 8,192、一次載入模型；148 類 Macro-F1、Top-1、37 類尾類 Macro-F1、病例群組 Bootstrap、配對 McNemar、Train/Dev 重複病例。
5. **Results**：Table I 資料特徵與品質；Table II BM25／來源標籤 CPU／Jeff 原始與 Global2；Table III 三模型的 Random2／Label-Hide／Label-Only；Table IV 長尾與逐類 rescue/break；Table V 無完全同文子集；Table VI 延遲與顯存。主圖呈現三模型 Top-1/Macro-F1、證據消融與錯誤分布。
6. **Discussion**：來源標籤與臨床內容的相依性、負向結果、長尾落差、重複病例、驗證集方法選擇偏差與推論成本；臨床安全性與外部泛化限制。
7. **Novel Method / Ablation（待研究）**：以臨床證據衝突、來源標籤可靠度或候選差異設計真正新機制，對 BM25、簡易投票、固定 Global2、參數匹配的保守對照驗證；論文能否作為方法研究取決於此。
8. **Final Validation（最後才做）**：方法與參數正式鎖定後再做獨立 Test 或外部病例評估；完成可重現性與統計報告。

**論文書寫原則**：已驗證結果直接呈現；未完成分數留空，方法尚未實作不得寫成已完成；保留相同資料／候選／指標的可比對照；維持精簡且無防禦式措辭。
