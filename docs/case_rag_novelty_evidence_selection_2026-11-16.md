# TCMDiscern：Case-RAG 原創性查核、臨床證據截斷與反事實實驗（2026-11-16）

## 研究決策摘要

完成資料準備、程式實作、CPU Pilot 與文獻查核。**新增醫案對 Jeff-31B 的
Top-1 / Macro-F1 / 長尾 F1 仍待 H200 真實推論。**本報告的提升僅涉及
檢索速度與原始臨床文字資訊保留。正式 Test 持續封存。

目前最值得驗證的組合：
1. 保留 BM25 自然 Top-16 與已微調 Jeff 決策頭。
2. 對 Top-2 混淆證型，各取一筆具來源群組隔離的歷史醫案。
3. 讓歷史病例內容從「固定取開頭」變成「患者相關、原文可追溯、固定預算的片段」。
4. 使用來源標籤／順序的 2×2 反事實測試，檢查模型究竟仰賴臨床內容、
   歷史標籤語意，還是示範順序；加測來源界線提示。
5. 待確認正向增益，再建立訓練資料獨立的錯誤風險與推論預算路由。

## CPU 新增結果：資料截斷問題

沿用固定 Dev 300 例及官方 Train 43,180 例；比較 balanced2、
paired2 共 1,200 次病例來源引用。已驗證原始來源和病例標籤均正確。

| 醫案欄位 | 平均來源字數 | 前綴保留上限 | 有截斷的引用比例 |
|---|---:|---:|---:|
| 主訴 | 14.14 | 58 | 0.1667% |
| 現病史 | 323.389 | 90 | **99.5%** |
| 四診 | 132.953 | 70 | **99.9167%** |

1,200/1,200 筆引用至少一個欄位被截斷。被刪的文字中存在大量
與當前患者重疊的字元 bigram。此項只反映文字證據的損失，尚未證明
其中都是臨床關鍵特徵。

新實驗固定兩筆來源病例、其標籤及 16 候選，對照以下不同片段：

| 條件 | 平均 Jeff tokens | 最大 | 平均每筆支持醫案的患者重疊 bigrams |
|---|---:|---:|---:|
| baseline（無支持醫案） | 742.47 | 2100 | — |
| paired2（前綴裁切） | 1160.63 | 2533 | 82.328 |
| querywindow2（患者相關的原文視窗） | **1183.44** | 2553 | **93.655** |
| randomwindow2（同樣長度的隨機原文視窗） | 1178.48 | 2549 | 82.475 |
| fullcase2（兩筆醫案全部原始內容） | 1691.16 | 5526 | 170.868 |

在保留相同 2 筆支持醫案與標籤下：
- querywindow2 比 paired2 增加 **13.76%** 字元 bigram 重疊，
  平均 Prompt 增加約 **1.97%** tokens。
- querywindow2 與 randomwindow2 都是來源病例連續原文片段；
  不會把缺少的症狀補成「沒有」，也不會生成新症狀。
- fullcase2 對這 300 筆均低於 8192 tokens，但成本較高。
- querywindow2 依當前患者字面重疊優化，重疊增加部分是
  演算法目標所致，**不得當作準確率、因果證據或原創性提升**。

實作：
`src/tcmdiscern/case_rag_excerpt.py`、
`scripts/prepare_case_rag_excerpt.py`、
`scripts/audit_case_rag_truncation.py`、
`tests/test_case_rag_excerpt.py`。
保存的私有輸入：
`data/case_rag/excerpt2_0300_20261116/`；
匯總：`experiments/case_rag_excerpt_cpu_2026-11-16.json`、
`experiments/case_rag_clinical_truncation_2026-11-16.json`。

## 免訓練反事實歸因：資料與評分已準備

兩案相同 300 筆原始患者與來源病例，7 種條件：
- `baseline`：原始模型輸入；
- `paired2`：兩筆真實 Train 醫案、來源標籤正確；
- `labelswap2`：只交換兩筆歷史醫案的來源標籤；**故意造錯的壓力測試**；
- `labelhide2`：保留來源病例文字，隱藏歷史證型標籤；
- `orderflip2`：正確標籤與病例一起互換呈現順序；
- `doubleflip2`：交換歷史標籤之後再調換呈現順序；
- `sourceguard2`：保留正確歷史醫案，強調其他患者症狀不得被視為
  當前患者的觀察結果。

`paired2`、`labelswap2`、`orderflip2`、`doubleflip2`
在全部 300 筆中平均均為 **1160.63 tokens**，輸入長度相同；
模型仍面對原始 16 候選。

2×2 因子可將標籤與排列干擾分離。令
`m(E)=logit_候選A(E)-logit_候選B(E)`，
並令 `m00/m10/m01/m11` 分別代表原樣、標籤交換、
呈現順序交換、雙交換：

- `LabelEffect = 0.5 * [(m10-m00)+(m11-m01)]`
- `OrderEffect = 0.5 * [(m01-m00)+(m11-m10)]`
- `Interaction = m11-m10-m01+m00`

該設計是已知的 2×2 因子實驗與 ICL 標籤反事實測試延伸，
學術原創性不能建立在這組公式本身。
實際讀到 H200 31B logits 之後，評分程式會產生上述
logit margin 分解、預測翻轉率、148 類 Macro-F1、
Top-1、McNemar 及患者群組 Bootstrap CI。

`src/tcmdiscern/case_rag_counterfactual.py`、
`scripts/prepare_case_rag_counterfactual.py`、
`tests/test_case_rag_counterfactual.py`；
私有輸入：
`data/case_rag/cf2factorial_0300_20261116/`。
所有控制只用於科研評測，**labelswap2 / doubleflip2 不允許作為臨床決策證據**。

## 同一 BM25 查詢計算可共用

在 `src/tcmdiscern/case_rag.py` 新增
`rank_and_supports(query,max_k=16)`，一次矩陣乘積產出
Top-16 證型及各類支持病例，取代兩次 query-score 前向運算。
300 筆患者的 BM25 Top-16 與支持醫案均與獨立路徑一致。

在同機 CPU 300 例的平均時間：
- 舊兩段排序＋支持醫案：4.114 + 6.894 = **11.008 ms**；
- 共用單次分數：**7.222 ms**（兩階段計算少約 34%）。
- 加上配對選擇 4.843ms，共用路徑從 BM25 到配對完成約
  **12.064 ms**（數值只代表本機 CPU）。
- 既有四醫案 Pilot 300×5 來源組合：1500 項逐筆對照 0 差異。

`scripts/benchmark_case_rag_latency_cpu.py` 與
`experiments/case_rag_cpu_latency_2026-11-16.json` 可重現。

## 文獻比對：哪些方向已有人做過

以下為截至 2026-11-16 可查證的重要近似研究；不聲稱涵蓋所有論文。

| 研究（年份） | 既有貢獻 | 我們的重疊與差異 |
|---|---|---|
| TCM-DiffRAG (Frontiers in Medicine, 2026) | 中醫 KG＋CoT＋RAG | 中醫 RAG 整體已有先例；我們是固定 16 類分類 Head、醫案證據 |
| Ruling Out to Rule In / CHR (2026 預印本) | 正、負診斷假說的對比檢索 | Top-2 對照本身高度接近既有概念 |
| Medical Knowledge-Driven Contrastive Learning (IEEE J-BHI, 2026) | ICD／外部知識困難負例、相似患者檢索 | 以困難負例改善檢索已在目標期刊發表 |
| Counterfactual RAG (ICLR 2026) | 反事實查詢、相衝突證據裁決 | 不能以「反事實 RAG」一詞主張原創 |
| Retrieval-Induced Mirage (AACL Findings, 2026) | 其他病人來源證據混入目標病例 | 重要的新臨床風險；其主要評估模態是醫療影像語言生成 |
| Rectifying Demonstration Shortcut (NAACL 2024) | ICL 示範捷徑、校準 | 錯誤示範標籤檢測已有前例 |
| Where to show Demos (EMNLP 2025) | Prompt 示範的位置偏差 | 必須對案例呈現順序控制 |
| SeleCom (WWW 2026) | query-conditioned relevance 選擇式壓縮 | 依查詢挑片段並非原創概念 |
| Evidence Fragmentation (Information, 2026-11-10) | 片段證據分散、連續跨度補救 | 固定前綴裁切有一般性資料遺失問題 |
| EnComp (AACL 2026) | 用反事實信號選取重要句子並壓縮 Context | query 片段＋反事實選擇的直接相關先例 |
| When Knowledge Is Not Free (EMNLP Findings, 2026) | 顯式證據取得成本與動態預算 | 2/4/全案預算並非單獨的新貢獻 |

參考來源：
- https://doi.org/10.3389/fmed.2026.1804478
- https://arxiv.org/abs/2604.04593
- https://doi.org/10.1109/JBHI.2026.3690515
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/1c078897dc08d46091d0d361d9955c6b-Abstract-Conference.html
- https://aclanthology.org/2026.findings-aacl.108/
- https://aclanthology.org/2024.naacl-long.242/
- https://aclanthology.org/2025.emnlp-main.1503/
- https://doi.org/10.1145/3774904.3792700
- https://doi.org/10.3390/info17111129
- https://aclanthology.org/2026.aacl-long.39/
- https://aclanthology.org/2026.findings-emnlp.413/

## 可能建立的原創技術：待證的整合假說

候選主張：
**Candidate-Conditioned Source-Robust Differential Retrieval**
（候選條件化、來源可靠性約束的鑑別式檢索）。

重點應放在「選擇哪些歷史醫案片段才會改變本患者辨證，
而且改變能受來源標籤干擾風險與 Token 成本約束」。
配對醫案、查詢視窗及來源歸因是必要組件，但單獨都已有先例。

概念目標（尚未訓練成方法）：
`E* = argmax_E [ estimated_paired_diagnostic_gain(E)
  - lambda * inference_cost(E)
  - gamma * risk_of_source_label_shortcut(E)]`

其中 risk 須從固定 Train OOF 的歷史標籤反事實 logits
學出；**不能使用 Dev gold 或 Test 來調整選例／路由**。
部署時由輕量 selector 先決定 0/2/4 支持醫案及其視窗，
Jeff 只執行一次 16-slot 決策；在 H200 前無法證實選取效益。

這是**具體的研究方向**，並未查證「全球無人提出同一組合」。
是否可成為 IEEE J-BHI 主方法，依賴：
- 同一 31B Jeff 與相同原候選，querywindow2 > paired2 / randomwindow2
  的真實 Top-1、148 類 Macro-F1 及長尾；
- 證據來源標籤／順序壓力測試量化錯誤改判，sourceguard2 降低風險；
- 2 case vs fullcase 是否有診斷收益與延遲最優選擇；
- 跑過 SGD＋Jeff 強基準、傳統對比檢索及足夠公平的消融；
- 預定稿、獨立 Train OOF 與封存 Test，或外部多中心及醫師評估。

## H200 下一步（模型本機未提供）

此 CPU 開發環境未安裝完整 Jeff 31B 與 S2/S3 checkpoint。
H200 使用 `scripts.run_case_rag_jeff` 既有的單次載入、
中斷續跑、case-id parity 驗證與固定條件 raw logits 統計。

```bash
python -m scripts.run_case_rag_jeff --tag excerpt2_0300_20261116 --preflight
python -m scripts.run_case_rag_jeff --tag cf2factorial_0300_20261116 --preflight

python -m scripts.run_case_rag_jeff \
  --tag excerpt2_0300_20261116 \
  --model-dir MODEL_DIR --checkpoint S2_CHECKPOINT \
  --expected-method s2 --run-id excerpt-s2-300

python -m scripts.run_case_rag_jeff \
  --tag cf2factorial_0300_20261116 \
  --model-dir MODEL_DIR --checkpoint S2_CHECKPOINT \
  --expected-method s2 --run-id attribution-s2-300
```

首次在 H200 沒有 `data/case_rag/` 私有輸入時，
依序生成固定兩案 `budget2_0300_20261116`，
再執行 `scripts.prepare_case_rag_excerpt` 與
`scripts.prepare_case_rag_counterfactual`。
不同模型與條件使用獨立 `--run-id`。
正式 Test 不能被用於選方法或設門檻。

## H200 單次載入與重複推論移除

新增 `scripts/run_case_rag_batch_jeff.py`，一次載入 Jeff 31B
與單一 S2/S3 checkpoint，然後連續驗證並計分
`budget2_0300_20261116`、
`excerpt2_0300_20261116`、
`cf2factorial_0300_20261116`。

CPU Preflight 真實核對所有完整 Prompt：
共 17 組條件、其中 13 組互不相同；
`baseline` 與 `paired2` 各在三個實驗中重複。
因此正式 300 筆實驗預計使用 **3900 次** Jeff forward，
相對分別推論 5100 次減少 1200 次（約 23.53%），
31B 模型只載入一次。程式會在完整 Prompt 相等、
來源清單與 checkpoint 身分一致時，才共用既有 logits；
單病例 .partial 中斷續跑與原生 16-slot logits 均維持。
本機使用假 Decider 驗證跨組重用及再次執行時 0 次額外 forward。

```bash
python -m scripts.run_case_rag_batch_jeff --preflight
python -m scripts.run_case_rag_batch_jeff \
  --model-dir MODEL_DIR --checkpoint S2_CHECKPOINT \
  --expected-method s2 --require-baseline-parity \
  --run-id case-rag-joint-s2
```

最新完整專案測試 **449 passed、1 skipped**，
新 Case-RAG 相關 Ruff 全數通過；
固定的四醫案來源 1500 項比對為 0 差異。

本地結果只用於描述性研究與 H200 Pilot 前準備；
實際準確率、GPU 速度變化及原創貢獻尚待完整驗證。
