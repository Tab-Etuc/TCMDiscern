# TCMDiscern：H200 全量研究檔案匯入與 CPU 後續分析（2026-11-17）

## 一、匯入與實驗可重現性

來源：`TCMDiscern-H200-research-20261117.tar.gz`（39,631,939 bytes、414 個 tar 成員，未壓縮合計 259,872,112 bytes）。

使用 `scripts/import_case_rag_h200_archive.py` 安全匯入：逐一驗證 Train／Dev、BM25 Dev 候選、原始 Jeff prompt、Global2 prompt 的 SHA-256 均和現有工作目錄相同；**僅補入 H200 推論 scores、timings、run manifests 與完成版的強融合聚合報告**。首次新增 20 個檔案，另有 2 個內容相同；再次執行確認冪等（0 新增、22 相同）。兩個 S1 Label-Hide `.partial` 未完成檔保留於原始壓縮包，不參與正式分析。原始病歷、程式、其他未追蹤工作均維持原狀。

匯入後 H200 已完成的 **9 個** Case-RAG arms：

- S1：Global2
- S2：Global2、Random2、Global2 Label-Hide、Global2 Label-Only
- S3：Global2、Random2、Global2 Label-Hide、Global2 Label-Only

完整研究檔案與原始逐病例 logits 僅保存在本地 `data/`；推送 GitHub 時只包含聚合統計、測試、方法紀錄與程式。

已將 H200 原始完整強融合聚合結果另存為 `experiments/case_rag_strong_sgd_fusion_h200_imported_2026-11-17.json`，以區分本地重算，並避免 H200 原本的同名輸出在未來 `git pull` 時再次衝突。

本地執行：

```bash
.venv/bin/python -m scripts.import_case_rag_h200_archive
.venv/bin/python -m scripts.analyze_case_rag_cpu_first --strict-global2 \
  --output experiments/case_rag_cpu_first_local_recomputed_2026-11-17.json
.venv/bin/python -m scripts.analyze_case_rag_cpu_router --strict-global2 \
  --output experiments/case_rag_cpu_router_local_recomputed_2026-11-17.json
.venv/bin/python -m scripts.evaluate_case_rag_strong_sgd_fusion_cpu --strict-global2 --all-arms \
  --output experiments/case_rag_strong_sgd_all_arms_local_2026-11-17.json
```

使用 H200 原始 logits 重新計算：既有 12 個方法的 Top-1、Macro-F1、37 類 Tail-F1 **全部與 H200 聚合報告相同**。對應配對的 rescue／break／Oracle 完全相同，McNemar p 值只在浮點數末位有微差。

## 二、完整 Dev 原始 Jeff Case-RAG 消融

官方 TCM-SD Train 43,180、Dev 5,486、148 證型，BM25 K16，Test 封存。

| 評測條件 | Top-1 | Macro-F1（148） | Tail-F1（37） |
|---|---:|---:|---:|
| S3 Base | 80.0948% | 41.2788% | 20.1802% |
| **S3 Global2** | **83.4488%** | **51.4706%** | 29.4595% |
| S3 Random2 | 80.6963% | 43.4075% | 24.8649% |
| S3 Global2 Label-Hide | 80.1495% | 41.3615% | 18.3784% |
| S3 Global2 Label-Only | 82.3733% | 47.4542% | 29.7297% |

Global2 對原始 S3 增加 **184 筆**正確案例。Label-Hide 幾乎回到 Base；Label-Only 保留部分增益。**原始 Jeff 很依賴檢索到的證型標籤，臨床文字加上標籤對原始 Jeff 仍有額外作用。** S3 Global2 相對 BM25 的互補 Oracle 為 **87.4043%**，低於 S3 Base 的 **88.4251%**。

## 三、使用已驗證的強 SGD 固定融合

同一套 SGD 訓練、Train prior 與 pairwise_margin1 規則；本地 H200 logits 全部在同一次 CPU fitting 後評分。

| 固定強融合方法 | 本地 Top-1 | Macro-F1（148） | Tail-F1（37） |
|---|---:|---:|---:|
| **SGD + S1 Base** | 85.0711% | **60.9597%** | **45.9202%** |
| SGD + S1 Global2 | 85.1440% | 60.4736% | 45.2896% |
| SGD + S2 Base | 84.9617% | 59.4761% | 42.4067% |
| SGD + S2 Global2 | 85.0711% | 59.3008% | 41.5058% |
| SGD + S2 Label-Only | 85.0711% | 59.8687% | 42.4067% |
| SGD + S2 Label-Hide | 84.8706% | 58.9551% | 41.9562% |
| SGD + S2 Random2 | 84.8524% | 59.1362% | 41.5058% |
| SGD + S3 Base | 84.9253% | 59.6804% | 43.1274% |
| SGD + S3 Global2 | 85.1075% | 59.4063% | 41.9562% |
| **SGD + S3 Label-Only** | **85.2351%** | 59.8835% | 42.4067% |
| SGD + S3 Label-Hide | 84.9617% | 59.5613% | 43.4878% |
| SGD + S3 Random2 | 84.9982% | 59.7215% | 43.5779% |

S3 Label-Only 相比同機 S3 Base：救回 **38**、破壞 **21**，Top-1 **+0.3099 pp**，配對 McNemar **p=0.0363（未校正多重比較）**；群組 Bootstrap 95% CI **[+0.0364,+0.5848] pp**。但相比同機完整 S3 Global2 只多 **7 筆**，配對 p=**0.3916**。多個方法均已在反覆查看的 Dev 中探索，需嚴格視為**探索性、尚待獨立驗證**。

固定強 SGD 融合後，**S3 Label-Only Top-1 高於完整 Global2**；然而原始 Jeff 的全套 Global2 反而更好。這是新研究要正面處理的條件效應：檢索醫案文字對弱決策頭有用，和強監督式 SGD 融合時，來源標籤的增益可能已涵蓋大部分貢獻。眼前的原始醫案檢索不保證提高已很強的診斷器。

## 四、H200 原始強融合及環境差異

| H200 同機強融合 | Top-1 | Macro-F1 | 對同機 Base Top-1 增益 | 未校正 McNemar p |
|---|---:|---:|---:|---:|
| S1 Base | 85.1805% | **61.2118%** | — | — |
| S1 Global2 | 85.1987% | 60.6532% | +0.0182 pp | 1.000 |
| S2 Base | 85.1622% | 59.8474% | — | — |
| **S2 Global2** | **85.3081%** | 59.8629% | +0.1458 pp | 0.341 |
| S3 Base | 85.0346% | 60.1116% | — | — |
| S3 Global2 | 85.2351% | 59.9029% | +0.2005 pp | 0.272 |

H200 SGD + S2 Global2 最高 **85.3081%**，比同次 S2 Base 多 **8 筆**，群組 Bootstrap 95% CI **[−0.1102,+0.4042] pp**，跨過零。本地相同架構 S2 + Global2 多 **6 筆**，95% CI **[−0.1456,+0.3643] pp**。本次資料尚未提供高信心的 Global2 強融合改善證據。

H200：Python **3.12.3**；本地：Python **3.12.13**。兩者 NumPy **2.5.4**、scikit-learn **1.9.1**、SciPy **1.18.1** 相同；H200 S1 Base **4,673** 正確，本地 **4,667**。Train／Dev、prompt、BM25 候選逐位元相同。數值計算環境是合理的待查方向，**尚無證據可指定 BLAS 或 Python 微版本為唯一根因**。論文主張以同機 Base／Global2 配對差為準，跨機分數列作重現敏感度。

## 五、H200 推論耗時（完整輸出）

| 來源條件 | 中位 forward 時間 | 平均 forward 時間 | P95 forward 時間 |
|---|---:|---:|---:|
| S3 Global2 | **225.845 ms** | 239.013 ms | 304.833 ms |
| S3 Label-Only | 174.980 ms | 184.496 ms | 244.620 ms |
| S2 Global2 | 225.760 ms | 238.895 ms | 305.453 ms |

以上為逐病例模型 forward，計入實驗內的單次推論時間；整體流程仍須另計檢索、Token 處理、網路與模型冷啟動。Label-Only 比完整 Global2 的輸入短（中位 779 vs 1083 tokens），速度較快。

## 六、研究與工程決策

1. **CPU-only 全量分析已完成**：資料、logits、消融與 H200 timing 均在本地可讀，本階段可直接完成後續配對統計和方法測試。
2. **Global2 能讓原始 Jeff 變好，但目前對強 SGD+Jeff 的淨增益小且欠缺穩健統計證據**。優先保留 **SGD+S1 60.9597%** 作為本地 Macro-F1 強基準；H200 宏觀最佳是 **61.2118%**。
3. **值得研究的真實問題是來源證據可靠性與判斷互補性**：模型何時依賴檢索標籤、何時應讀取醫案文字、何時採用本身辨證分數。可先用 Train OOF／獨立 fit-cal 做可泛化的可靠度決策，再到官方 Test 一次驗證。
4. 新的 S1/S3 GPU 消融、Seed 2/3 微調繼續暫緩；現有資料已足以否決「只增加兩筆案例就能大幅超越 85%」的初步假說。
5. 部分 S1 Label-Hide `.partial` 的存在僅代表未完成的 GPU 作業快照；單獨歸檔，不冒充完整消融。此次完整 S2／S3 控制組足以分析目前問題。
