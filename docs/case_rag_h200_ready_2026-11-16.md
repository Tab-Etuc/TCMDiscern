# Case-RAG 本地完成紀錄與 H200 驗證程序（2026-11-16）

## 定位與目前結果

TCMDiscern 仍以官方 BM25 Top-16 → Jeff 16-slot Decision Head 為主線。
Case-RAG 讓 Jeff 在單次 forward 同時讀取原始患者病歷、固定 16 候選
以及官方 TCM-SD Train 的 2 或 4 筆有標註歷史醫案。

**狀態：CPU 工程／檢索／資料安全／模型輸入／評分入口完成；
新增醫案後的 31B Jeff 診斷成績尚未取得。**
原始 S2 checkpoint 的 300 例 raw Top-1 為 245/300 = 81.6667%，
排除 33 筆 Train 完全同文後為 79.7753%（267 例）。
此數字不能當作新增 Case-RAG 的成績。

### 固定 300 例、官方 Train 記憶庫 43,180 例

| 條件 | 平均 Jeff tokens | 最大 tokens | 與同預算 balanced 相比，來源選擇改變 |
|---|---:|---:|---:|
| baseline | 742.5 | 2100 | — |
| global2 | 1161.2 | 2539 | — |
| balanced2 | 1160.8 | 2533 | 基準 |
| paired2 | **1160.6** | **2533** | **90/300（30%）** |
| random2 | 1160.0 | 2524 | — |
| global4 | 1524.0 | 2916 | — |
| balanced4 | 1522.9 | 2917 | 基準 |
| paired4 | **1522.9** | **2917** | **192/300（64%）** |
| random4 | 1521.6 | 2906 | — |

兩筆 paired2 相對四筆 paired4 的平均輸入減少 362.3 tokens（約 23.8%），
相對原版仍增加約 56.3% tokens。沒有任何 Prompt 超過 8192 tokens。

每個配對預算分別固定 Top-2 證型各半數的醫案標籤；
對照組 global/balanced/paired/random 使用原始病例來源。
全部保留相同 16 候選、相同原患者三欄、相同種子與病例集合。
模型未接收查詢金標或 Train 的病人識別碼。
同病人、完全同文及歧義標註的來源群組被排除。
正確選例不等於正確診斷。

CPU 支持病例跨證型 Jaccard：
- 2 案：balanced2 0.513851 → paired2 **0.540784**；
- 4 案：balanced4 0.482404 → paired4 **0.519247**。
這些是受控病例文字相似度，尚不能當作臨床證據忠實性。

## 檢索效能與保真

使用 300 筆固定 Dev，CPU 單查詢測量：
- BM25 自然 Top-16 排序：平均 4.096 ms，P95 4.873 ms；
- 新增支持醫案檢索：平均 **7.022 ms**，P95 **8.432 ms**；
- 配對醫案選取：平均 **4.824 ms**，P95 **8.333 ms**；
- Case-RAG 增量 CPU 檢索＋配對：平均 **11.846 ms**，P95 **15.902 ms**；
- 索引啟動建立約 3.691 秒（服務啟動成本）。
- 前版逐筆掃描支持醫案的平均約 36 ms；現版支持醫案檢索約降 80%。
- 300 × 5 = 1500 筆固定來源位置、群組及標籤逐項校對：
  **0 筆差異**。候選也逐筆一致。

實作於 `src/tcmdiscern/case_rag.py`：
預先索引各證型的無歧義病例，使用 NumPy 群組去重與穩定排序，
並只保留每類前 32 個獨立支持病例；五組既有選取結果不變。

統計：
`experiments/case_rag_cpu_latency_2026-11-16.json`、
`experiments/case_rag_retrieval_parity_2026-11-16.json`、
`experiments/case_rag_input_audit_2026-11-16.json`、
`experiments/case_rag_budget2_input_audit_2026-11-16.json`。

## Jeff 實測入口已完成的保護

`scripts/run_case_rag_jeff.py` 可載入 S1、S2、S3 或 S4 的既有
Jeff checkpoint，使用一次模型載入、同一 Prompt 排序與
raw 16 logits。 `--expected-method` 強制 checkpoint 類型。

`src/tcmdiscern/case_rag_runtime.py`：
- 每一個病例 forward 後立即寫入私人 `.jsonl.partial`；
- 中斷後逐筆核對 Case ID、16 維有限 logits，再從剩餘病例續跑；
- 中斷於最後一列時，安全捨棄未完成的最後一列；
- 完成後原子發布 `.jsonl`，保留原檔不可覆蓋；
- 紀錄 Prompt 來源、案例數、模型路徑、checkpoint 路徑／方法，
  防止復用錯誤模型的輸出；
- 強制檢查 max_tokens、禁止輸入被縮短或切換思考生成；
- 可選 `--require-baseline-parity`：S2 首先與本地保存的原始
  S2 epoch2 Dev logits 驗證預測一致，再執行其他醫案組。

評分以 148 類固定 Macro-F1、Top-1、排除 Train 完全同文後 Top-1、
配對 Rescue/Break、McNemar exact p、Dev 患者群組 bootstrap 95% CI
以及 **paired2/4 vs balanced2/4** 同預算比較為主。
隨機醫案及一般全域醫案是必要負控制。

## H200 執行指令

在 H200／GH200 的 TCMDiscern 工作目錄，使用已能載入
Jeff 1.3.0 與 31B 權重的 Python 環境。
`MODEL_DIR` 與 `CHECKPOINT` 改成該主機實際位置。

首次在 H200 建立配對輸入（會重算官方 BM25 並核對原候選）：

```bash
python -m scripts.prepare_case_rag_pilot --samples 300 --seed 20261116 --tag pilotclean --evidence-budget 4
python -m scripts.prepare_case_rag_pilot --samples 300 --seed 20261116 --tag budget2 --evidence-budget 2
python -m scripts.run_case_rag_jeff --tag pilotclean_0300_20261116 --preflight
python -m scripts.run_case_rag_jeff --tag budget2_0300_20261116 --preflight
```

如來源目錄已存在，使用現有生成內容並核對 manifest；不覆寫舊 Pilot。

### 2026-11-16 新增：證據原文摘錄對照

前綴截斷會丟失多數病例病史與四診後段文字。新增的
`focused2/focused4` 保持 **相同來源病例、候選標籤、患者原文、
樣本與種子**，只將三個臨床欄位的單一連續原文摘錄位置改為
查詢詞項覆蓋較高的片段。這是受控摘錄基準，尚未證實臨床效果。

需先具備上述固定 prefix 來源，才產生：

```bash
python -m scripts.prepare_case_rag_excerpt_pilot --evidence-budget 2
python -m scripts.prepare_case_rag_excerpt_pilot --evidence-budget 4
python -m scripts.audit_case_rag_pilot --tag focused2_0300_20261116
python -m scripts.audit_case_rag_pilot --tag focused4_0300_20261116
python -m scripts.run_case_rag_jeff --tag focused2_0300_20261116 --preflight
python -m scripts.run_case_rag_jeff --tag focused4_0300_20261116 --preflight
```

對原始 Prefix 和新 Focused **使用同一個 S2 checkpoint**：

```bash
python -m scripts.run_case_rag_jeff \
  --tag budget2_0300_20261116 --model-dir MODEL_DIR \
  --checkpoint CHECKPOINT --expected-method s2 \
  --run-id case-rag-s2-prefix2
python -m scripts.run_case_rag_jeff \
  --tag focused2_0300_20261116 --model-dir MODEL_DIR \
  --checkpoint CHECKPOINT --expected-method s2 \
  --run-id case-rag-s2-focused2
python -m scripts.compare_case_rag_evidence_styles \
  --evidence-budget 2 --reference-run case-rag-s2-prefix2 \
  --focused-run case-rag-s2-focused2
```

跨風格評分先核對 Checkpoint、Train 來源群組、患者文字及
原始 16 候選是否相同，再計算同批病例的 Top-1、Macro-F1、
Rescue/Break、McNemar、分組 bootstrap 與排除同文的表現。
正式測試仍保持封存。也可以用相同流程替換為四案或 S3，
但必須在兩個風格使用**同一** checkpoint、不同 run-id。
完整方法與文獻原創性評估已寫入
`ieee-jbhi-paper/docs/case-rag-originality-review-and-evidence-optimization-2026-11-16.md`。

S2 首次正式五組輸入（300 例）：

```bash
python -m scripts.run_case_rag_jeff \
  --tag pilotclean_0300_20261116 \
  --model-dir MODEL_DIR \
  --checkpoint CHECKPOINT \
  --expected-method s2 \
  --require-baseline-parity \
  --run-id case-rag-s2-four
```

上式的 `--require-baseline-parity` 需要 H200 上
`data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl`
存在，且 checkpoint 正是產生該檔案的同一 Epoch2 S2。
不符合時先調整 checkpoint、重建基準或移除這項選擇性核對，
仍保留 checkpoint 類型與來源清單檢查。

S3 完成後直接用現有 S3 checkpoint：

```bash
python -m scripts.run_case_rag_jeff \
  --tag budget2_0300_20261116 \
  --model-dir MODEL_DIR \
  --checkpoint CHECKPOINT \
  --expected-method s3 \
  --run-id case-rag-s3-two
```

請用不同 `--run-id` 隔離 2 案、4 案、S2、S3 與任何 `--limit`
煙霧測試；輸出位於
`data/case_rag/<tag>/scores/<run-id>/`，每條件逐筆 logits 為私有。
只公開 `experiments/case_rag_jeff_<run-id>.json` 彙總。
首次 10 例 smoke 與正式 300 例必須使用不同 run-id。
重新執行同一 run-id 可從有效的 partial prefix 續跑。

### 本機新增：完整醫案與來源歸因實驗（11/16）

已完成以下附加稽核，全部限定原始 300 筆 Dev 與官方 Train：

- `excerpt2_0300_20261116`：5 組 `baseline/paired2/querywindow2/randomwindow2/fullcase2`，
  同一批固定 Top-2 來源醫案，逐項比較保留開頭、患者相關視窗、
  隨機視窗及完整病例。四種病例來源及候選皆相同。
- 完整兩筆醫案平均 **1691.2 tokens**、P95 **2945.1**、
  最大 **5526**，300/300 低於 Jeff 8192 上限。
  四筆完整醫案平均 **2555.5**、P95 **4634.6**，
  但有 **1/300** 筆超出上限（最大 8231），因此四筆全文
  需額外 Token 容量控制才能成為完整對照。
- `scripts/audit_case_rag_excerpt_provenance.py` 已以官方 Train
  原文重新建立三組摘錄，**3×600=1800 筆來源曝光全部通過**
  候選、來源病例、原始證型及原文一致性核對。
- `scripts/audit_case_rag_negation_boundary.py` 的否定詞切割風險代理：
  患者相關 `focused2` 1800 個臨床欄位摘錄有 3 個緊接
  「無／未／不」等字元之後開始、2 個在否定字元處截尾，
  並有 11 個位於較寬的四字上下文風險範圍；
  此指標只是需要人工確認的語意風險，不能視為診斷錯誤率。
- 原文過度截斷的代理訊號有改善，但對 Jeff Top-1／Macro-F1
  的實質增益、延遲仍待 H200 取得真實 logits。

H200 先用相同 checkpoint 跑最有鑑別力的**兩案完整對照**：

```bash
python -m scripts.audit_case_rag_excerpt_provenance
python -m scripts.run_case_rag_jeff --tag excerpt2_0300_20261116 --preflight
python -m scripts.run_case_rag_jeff \
  --tag excerpt2_0300_20261116 \
  --model-dir MODEL_DIR \
  --checkpoint CHECKPOINT \
  --expected-method s2 \
  --run-id case-rag-s2-excerpt2
```

同一次模型載入產生的統計會直接比較
`paired2`、`querywindow2`、`randomwindow2`、`fullcase2`
與 `baseline`，報救回／破壞、Macro-F1 與配對檢定。
只有 `fullcase2` 確實提高診斷效果時，才需要考慮新增
高成本的四筆全文實驗；`randomwindow2` 必須保留為摘錄負控制。

### 推論結果的可靠性與實際成本

新增 `src/tcmdiscern/case_rag_identity.py`，保留**模型可見的每筆
Prompt、16 個候選順序、病例原文、原始支援醫案的內容指紋**，
並記錄 checkpoint manifest、probe、Head、LoRA 權重檔版本。
首次執行與續跑都要核對 `run_manifest.json`；同一 run-id
若 Prompt、checkpoint 或其版本改變，會在 Jeff 31B 載入前拒絕。
`--score-only` 也會核對原始輸入指紋。
已產生的舊式推論 manifest 必須使用新的 run-id 重新建立紀錄。

`case_rag_runtime.py` 保留原始 16 logits 的 JSONL 外，
另為每個條件逐病例記錄 `<arm>.timings.jsonl`（私有）：
CUDA 同步後的 Jeff `decide` 耗時、Token 數、峰值 CUDA
已配置顯存。彙總提供 mean、P50、P95、第一筆與穩態 P95。
記錄範圍包括 Jeff tokenizer／forward 但不包括 BM25 與索引初始化。
若發生中斷，logits/timings 同時續跑並修復單筆不完整寫入；
先前無耗時紀錄的舊版 partial 會標記為未知，不以虛構耗時填補。
所有逐筆資訊仍只寫入被 Git 忽略的 `data/`。

CPU 檢索增量平均 11.846 ms、P95 15.902 ms 可與實際 GPU
推論時間分別報告。正式端到端總延遲需另加服務載入、
檢索與傳輸開銷。

## 2026-11-16：Pilot 類別覆蓋與正式評分防誤判

固定 300 筆 Dev 是先導實驗（Pilot），僅用來低成本檢查
Case-RAG 是否產生救回／破壞訊號與實際推論耗時。
此抽樣由病例隨機抽取，未按證型分層。

`uv run --no-sync python -m scripts.audit_case_rag_pilot_coverage`
已完成原始 Train／Dev 的資料稽核：

- 148 個 Train 證型中，300 筆只涵蓋 **56 類**，
  另 **92 類完全沒有 gold**。固定 148 類 Macro-F1 的
  **理論上限只有 37.8378%**，不能與完整 Dev 的 Macro-F1
  直接比較。
- 25 類只有 1 筆、33 類只有 2 筆以下。
- 依 Train 類別支持數分成 49 類 tail（≤23）、
  49 類 mid（24–79）、50 類 head（≥80）。
  Pilot 僅有 **4 筆 tail、14 筆 mid、282 筆 head**。
  尾類 F1 在這批資料上不具可靠估計性。
- Train／Dev 同臨床文本 **33 筆**，其中 **2 筆 gold 衝突**；
  另有 **2 筆同患者 ID**。應另報排除重複原文的子集。
- `scripts/run_case_rag_jeff.py` 三種評分路徑現在另外輸出
  `pilot_label_coverage`、固定 148 類 Macro-F1
  與只針對實際 56 類的探索性 Macro-F1。
  不把後者冒充完整 148 類成績。

**論文正式主指標**：完成參數與研究決策鎖定後，
在完整官方 Dev **5,486 筆**執行對照，計算固定 148 類
Macro-F1、Top-1、長尾 F1、群組化不確定區間及 GPU latency。
目前 300 筆僅供方法淘汰與工程檢查；官方 Test 繼續封存。

稽核輸出：`experiments/case_rag_pilot_coverage_2026-11-16.json`。
病例 gold 只用於指標及資料稽核，未寫入 Jeff Prompt 或檢索索引。

## 科學決策與測試

1. 第一關：原始 300 例上的 paired vs balanced 是否有正向 rescue/break；
   另外報 267 筆排除 Train 同文與相同 tokens/case 數的比較。
2. 第二關：比較 paired2、paired4 的額外醫案收益及 H200
   真正秒數／記憶體成本。僅在第一關有正向訊號時擴大資料量。
3. 第三關：在預先定稿的乾淨留出上比較固定 148 類 Macro-F1、
   Top-1、不同類別支持度、SGD＋Jeff 強基準及完整文獻消融。
   300 例僅作探索性 Pilot；148 類 Macro-F1 很不穩定。
4. 正式 Test 持續封存；研究資料全部只用官方 TCM-SD 原始
   Train／Dev，沒有外部醫案或新醫師標註。
5. 現有二證型配對與字元 Jaccard 已有相關學術前例；
   **只有模型增益與嚴格消融成立，才開始討論論文方法創新。**

驗證：最新完整本機 pytest **451 passed、1 skipped**（2026-11-16）；
相關 Ruff 檢查全部通過。
略過項為此 CPU 環境缺 Gemma 4 模組的 GPU 專用微型模型測試。

## 2026-11-16 新增：醫案截斷、反事實控制與單次載入批次執行

先在 CPU 生成三組凍結 Pilot；若目錄已有原始同名資料，使用既有檔案，
並以 `--preflight` 確認來源內容與候選一致。

```bash
python -m scripts.prepare_case_rag_pilot --samples 300 --seed 20261116 --tag budget2 --evidence-budget 2
python -m scripts.prepare_case_rag_excerpt --source-tag budget2_0300_20261116 --tag excerpt2_0300_20261116
python -m scripts.prepare_case_rag_counterfactual --source-tag budget2_0300_20261116 --tag cf2factorial_0300_20261116
python -m scripts.run_case_rag_batch_jeff --preflight
```

正式 H200 推論以**單次載入同一 Jeff 與同一 checkpoint** 的批次入口為主：

```bash
python -m scripts.run_case_rag_batch_jeff \
  --model-dir MODEL_DIR \
  --checkpoint S2_CHECKPOINT \
  --expected-method s2 \
  --require-baseline-parity \
  --run-id case-rag-joint-s2
```

批次含三組、共 17 個條件。以**完整 Prompt 逐筆相等**為前提，
共用 baseline 和 paired2 已產生的 16-slot raw logits，
故唯一輸入共 13 組、300 例時需 **3900 次 Jeff forward**
（逐組獨立跑 17×300 則需 5100 次），且 31B 模型只載入一次。
中斷時各條件續跑、模型與來源固定，完成後才計算彙總。

首次驗證可先使用 `--limit 10 --run-id case-rag-smoke-s2`；
正式 300 例使用新的 run-id。每組的私人 scores 依來源目錄隔離，
公開彙總輸出為 `experiments/case_rag_batch_<tag>_<run-id>.json`。
S3 checkpoint 完成後把 expected-method 改為 s3，使用新的 run-id。

新 CPU 報告：`docs/case_rag_novelty_evidence_selection_2026-11-16.md`。
它確認歷史病例現病史及四診幾乎總是被原本的 90/70 字前綴截斷，
querywindow2 在約多 2% Prompt tokens 下增加 13.76% 文字重疊；
這是**臨床文字保留指標**，仍需真實 Jeff accuracy/Macro-F1 才能評估醫療效益。

反事實 `labelswap2`/`doubleflip2` 故意把歷史來源標註改錯，
只能在研究隔離環境做壓力測試，不能用於臨床診斷建議。
