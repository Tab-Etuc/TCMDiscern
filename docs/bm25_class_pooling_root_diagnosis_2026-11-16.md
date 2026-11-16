# 根本性 BM25 聚合診斷（2026-11-16）

## 核心發現

現有 `CaseBM25` 將每個證型的全部病例 BM25 相似度取最大值。
當類別有 7 到 7,380 筆病例時，這項決策既受到極值偏差影響，
也把「一筆偶然相似病例」等同「多筆一致相似病例」。

在原有 TCM-SD Train 分組 Fit=30,226、Cal=6,477，使用同一個
BM25 字元 bigram 索引、相同 16 個候選證型，僅變更證型彙總：

| 聚合方式 | Top-1 | Macro-F1 | Tail Macro-F1 |
|---|---:|---:|---:|
| 原本每類 max | 79.1107 | 50.6443 | 34.3287 |
| **Top-3 mean** | **82.2603** | **52.7227** | **34.4180** |
| Top-5 mean | 81.9670 | 49.2177 | 23.8426 |
| LogMeanExp T=4 | 75.5442 | 50.6087 | 33.6138 |
| LogMeanExp T=12 | 67.8709 | 46.0753 | 26.1356 |
| 減去 25% 查詢條件類別最大值 null quantile | 65.0301 | 46.5902 | 30.3580 |

**Top-3 vs max**：救回 329、破壞 125，正確淨增加 204，Top-1
+3.1496 percentage points，Macro-F1 +2.0784 pp。此數據為 Cal 開發性測試，
Cal 曾多次探索；獨立驗證、置信區間和強分類器比較仍待完成。

## 下一個演算法假設

**Reliability-conditioned Evidence Mass (REM)**：對每個查詢與證型，
依據正例相似度尾端間隔、近鄰獨立性及競爭類別的分布，
估計有效支持病例數；讓聚合可從 `max` 過渡到 `Top-k mean`。
適合研究的推導方式是：
1. BM25 定義原始單病例證據（可保持不變）。
2. 群組相似度 `s_(1)>=...>=s_(m)` 加入去重/群組獨立性權重。
3. 定義單筆證據超越查詢條件背景的 reliability probability
   `r_j = sigmoid((s_j - b_q)/tau)`，以訓練資料 OOF 校準。
4. 定義支持強度 `M_c = sum_j r_j * independence_j`，
   讓分類損失同時優化 Top-1、Macro-F1/長尾與校準。
5. 依據可靠證據質量決定自適應讀取數量，保留單筆高品質
   病例的尾類回退機制。

這些是研究假設，尚非獲驗證的數學創新。自適應 kNN、MIL pooling、
class-balanced nearest-neighbor、類別機率校準已有大量先例。要提出
J-BHI 方法貢獻，必須對這些技術進行嚴格基準對照，並將病例
依賴與長尾限制納入聯合目標。

### 預期可驗證原理

- `max`：過度依賴單筆偶然高分。
- `Top-3`：減少 outlier，初步增益顯著。
- `Top-5`：要求過多支持病例時，罕見證型嚴重失分。
- 直接減去統計 null quantile：在現有分布下明顯失敗；
  不是有統計意義就一定有效，需尊重多來源病歷差異。

## 執行與重現

```bash
uv run --no-sync python scripts/probe_bm25_pooling.py
uv run --no-sync ruff check scripts/probe_bm25_pooling.py
```

輸出：`experiments/bm25_pooling_root_probe_2026-11-16.json`。

既有較強的 TF-IDF SGD Top16／Jeff 融合對照也必須納入後續
最終評估，目前這份原始 BM25 聚合改善不能等同端到端效益。
正式 Test 未使用，官方 Dev 未使用。
