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
