# TCMDiscern：CPU-only 路由與歷史最高 Dev 成績核對（2026-11-17）

## 既有同一官方 Dev 的強基準

Train 43,180、Dev 5,486、148 證型、BM25 候選 K16，官方 Test 封存。

| 方法 | Top-1 | 148 類 Macro-F1 | 實驗來源 |
|---|---:|---:|---|
| **Joint 臨床結構 SGD + Jeff S2 成對融合** | **85.180%（4673/5486）** | 59.387% | `experiments/joint_clinical_signature_jeff_fixed_2026-11-16.json` |
| Joint 臨床結構 SGD + Jeff S1 成對融合 | 85.144% | 60.861% | 同上 |
| **文字 SGD + Jeff S1 成對融合** | 85.071% | **60.960%** | `experiments/linear_jeff_fusion_paired_diagnostics_2026-11-16.json` |
| SGD（文字＋結構證據） | 84.451% | 58.195% | `experiments/clinical_signature_frozen_dev_2026-11-16.json` |
| Class-balanced SGD | 84.141% | 58.391% | `experiments/linear_syndrome_balanced_dev_2026-11-16.json` |
| S3 + Global2 Case-RAG | 83.45% | 51.47% | 完整 H200 Dev，使用者已提供完成報告 |

**主研究實際需要超越的比較強基準是 Top-1 85.180% 與 Macro-F1 60.960%**，兩數字來自不同融合方法，應分別對照。上述含 Jeff 的 SGD／Joint 融合曾在 Dev 選用或調整門檻，屬開發集探索性結果，正式成效需以鎖定規則後的最終測試確認。舊報告的最低頻 11 類 Tail-F1 與 Case-RAG 的最低頻 37 類 Tail-F1 定義不同。

## 本輪新增 CPU-only 五折分組路由實驗

程式：`scripts/analyze_case_rag_cpu_router.py`；報告：`experiments/case_rag_cpu_router_2026-11-17.json`。

路由根據推論當下可用的 17 個特徵，包括 Jeff 16 類 logits、與 BM25 候選的分數差、Train 類別出現頻率、Global2 來源病例標籤一致度及檢索分數。採用 GroupKFold(5) 分組，訓練組標籤用來學習在 Jeff 與 BM25 分歧且只有一方正確時該信任哪一方；驗證組全數保留。模型為 StandardScaler + L2 LogisticRegression。所列三個固定決策門檻 0.5／0.65／0.8 全部報告，不在 Dev 上挑單個最高值宣稱獨立測試。

| 方法 | Dev Top-1 | Macro-F1（148） | Tail-F1（37） |
|---|---:|---:|---:|
| S1 原始 Jeff | 79.220% | 40.340% | 18.649% |
| S1 + OOF 路由（0.5） | 84.397% | 57.578% | 41.956% |
| S2 原始 Jeff | 80.350% | 42.665% | 22.188% |
| S2 + OOF 路由（0.5） | 84.123% | 57.093% | 38.919% |
| S3 原始 Jeff | 80.095% | 41.279% | 20.180% |
| **S3 + OOF 路由（0.5）** | **84.543%** | **58.376%** | **44.659%** |
| S3 + OOF 路由（0.65） | 84.160% | 56.602% | 41.055% |
| S3 + OOF 路由（0.8） | 82.865% | 51.359% | 30.811% |

此結果為**Dev 內分組交叉擬合之探索性診斷**，因為此前多輪研究已檢視 Dev；推論時只用可取得的特徵，但也不應視為經 Train OOF→獨立 Dev 選模的正式演算法增益。完整 Global2 的逐病例 logits 尚未同步本地，因此本輪路由結果主要作用在原始 S1/S2/S3。

結果表明 CPU 決策機制能改善原始模型的長尾，但更強的既有 `Joint+Jeff S2 85.180/59.387`、`Plain SGD+Jeff S1 85.071/60.960` 仍是必須超越或獨立比較的標準。尤其目前 **84.543%** 尚未超越 **85.180%**，Macro-F1 **58.376%** 尚未超越 **60.960%**。

## H200 已取得的六模型分組路由結果（2026-11-17 終端輸出）

使用者在 H200 成功執行 `scripts.analyze_case_rag_cpu_router`，狀態 `CASE_RAG_CPU_ROUTER_EXPLORATORY_COMPLETE models=6`；以下為 H200 終端提供的 5,486 筆 Dev 分組五折探索性結果（門檻 0.5）：

| 使用的 Jeff logits | 原始 Top-1 | CPU 路由 Top-1 | 路由 Macro-F1（148） | 路由 Tail-F1（37） |
|---|---:|---:|---:|---:|
| S1 Base | 79.220% | 84.397% | 57.578% | 41.956% |
| **S1 + Global2** | 82.902% | **84.451%** | **58.800%** | **45.109%** |
| S2 Base | 80.350% | 84.123% | 57.093% | 38.919% |
| S2 + Global2 | 82.592% | 84.160% | 57.461% | 40.245% |
| **S3 Base** | 80.095% | **84.543%** | 58.376% | 44.659% |
| S3 + Global2 | 83.449% | 84.397% | 58.061% | 42.136% |

- **最高路由 Top-1：S3 Base + 路由 84.543%。最高路由 Macro-F1：S1 + Global2 + 路由 58.800%。**
- 即使加入新路由，仍低於先前完整 Dev 的 **Joint+Jeff S2 85.180% Top-1** 與 **Plain SGD+Jeff S1 60.960% Macro-F1**。
- S3 + Global2 經路由後的 84.397% 比 S3 Base 經相同路由的 84.543% 略低；進一步投入 GPU 前先確認 Global2 與 BM25 的逐病例互補性。
- 這批 Fold 是從**已多次觀察的 Dev** 進行交叉擬合，僅代表探索性結果。新的方法仍須 Train OOF 或獨立資料驗證。

同一批執行中，`analyze_case_rag_cpu_first --strict-global2` 因先前程式在取得 Global2 時覆蓋了 `s3_base_vs_bm25` 配對 key，觸發 `KeyError`。該程式已修正，會同時保留 Base 和 Global2 的配對表與 Oracle；回歸測試涵蓋三模型 Global2 存在的實際分支。

## 下一個 CPU 工作（已具備程式，待 H200 現有檔案）

在 H200 已完成的 `data/case_rag/paper_5486_20261116/scores/paper-{s1,s2,s3}-full/global2.jsonl` 旁直接執行：

```bash
cd /home/quanta/TCMDiscern-git
git pull --ff-only origin main
.venv/bin/python -m scripts.analyze_case_rag_cpu_first --strict-global2
.venv/bin/python -m scripts.analyze_case_rag_cpu_router --strict-global2
```

兩條命令均用 CPU 讀取現有 logits，前者計算 Global2 與 BM25 的逐病例救回／破壞、長尾差分及既定 Train 頻率保護規則；後者將同樣特徵應用於 Global2，產生三模型五折分組探索性路由矩陣。兩者**無需載入 31B 模型／新增 GPU forward**。官方 Test 持續封存。

Seed 2／3 及 S1／S3 額外 GPU Label-Hide／Label-Only／Random2 消融保持暫緩；先從這些現有計算判斷是否有值得發展的新增技術機制。
