# Differential Case-RAG × Jeff S2：免微調 Pilot（2026-11-16）

## 狀態

**實際完成：** 官方 Train 醫案檢索、五組候選相同的 Jeff/SemIf Prompt、300 筆原始 Dev 抽樣、病例來源與去重 QA、token 預算核對、已保存 S2 logits 的原版成績，以及可在 H200 載入 S2 checkpoint 的評分入口。

**尚待 H200：** 四種新增醫案 Prompt 的真正 31B Jeff logits、相對原版 Top-1／Macro-F1 效益與延遲。這台開發機僅有 RTX 4060，並未安裝完整 31B Jeff 權重，故未宣稱新 RAG 診斷增益。正式 Test 持續封存。

## 科學假說

原版 TCMDiscern BM25 先找出 16 個不同證型，Jeff 只看到 16 個**證型名稱**（不是檢索病例全文）。DiffRAG 的實際貢獻啟示在於知識／病例需進入推理階段；本 pilot 檢查在不改候選、不微調 Jeff 的條件下，**對比支持醫案是否能提供額外資訊**。

五個條件全部使用**相同原始 16 候選名稱及位置**：

1. `baseline`：原版 Jeff Prompt，逐筆對照 `data/jeff/dev_k16.jsonl` 完全一致。
2. `global4`：全 16 個候選證型中 BM25 最高的 4 個相異病例群組。
3. `balanced4`：BM25 Top-2 兩個競爭證型，各取最接近患者的 2 個獨立醫案群組。
4. `paired4`：同樣 Top-2、各 2 個病例，聯合最大化查詢相似度與正反兩個標籤案例的彼此相似度，建立兩組對照醫案。
5. `random4`：同樣 Top-2、各隨機 2 個候選支持群組，固定種子。

後四者均提供 4 份有標註的官方 Train 歷史醫案；檢索文字只使用原始主訴／描述／四診欄位，來源標註與金標限於醫案本身。所有病歷取不同病患／文字傳遞群組，明確標註衝突群組排除。對查詢病患 `user_id` 相同或三欄文字完全相同的 Train 病例，只從支持例檢索排除，**BM25 原始 Top-16 排序維持不動**。

## 實跑結果：輸入與檢索層

- 固定 seed 20261116，原始 Dev 均勻無標籤抽樣 300/5486；官方 Train 記憶庫 43180 筆。
- 全 300 筆 x 5 組生成成功，候選與原版 Prompt 一致，0 筆超過 8192 token。
- Baseline 平均 742.5 tokens，global4 1524.0，balanced4 1522.9，paired4 1522.9，random4 1521.6。最高 2917 tokens。
- 33/300 筆 Dev 查詢與 Train 原文完全重複；在新支持醫案檢索裡已排除這些同文群組與同病人群組，評分仍另報去除原文重複的 267 筆。
- `paired4` 相對 `balanced4` 有 192/300（**64.0%**）改變支持醫案的選擇；兩個不同證型醫案的 lexical Jaccard 相似度平均 **0.4887 → 0.5254**，支持例平均 BM25 由 **280.431 → 278.921**。證明配對選取確實有不同證據，但**沒有證明診斷改善**。
- `global4` 每案例平均涵蓋 1.45 個證型；`balanced4` / `paired4` / `random4` 均為 2 個。
- 使用真正已保存的 S2 epoch2 16-slot Dev logits 在這 300 筆取 baseline：原始 BM25 Top-1 **80.6667%**，S2 raw Top-1 **81.6667%**；在 267 筆未與 Train 同文的病例上 S2 raw Top-1 **79.7753%**。300 筆並未覆蓋所有 148 類，個別小樣本 Macro-F1 不宜與完整 Dev 分數比較。

## 重現與 H200 執行

```bash
cd TCMDiscern

# CPU：每條件 300 筆、原候選一致、離線 tokenizer 長度與來源 QA
uv run --no-sync python scripts/prepare_case_rag_pilot.py --samples 300 --tag pilotclean
uv run --no-sync python scripts/audit_case_rag_input_pilot.py
uv run --no-sync python scripts/score_case_rag_original_s2.py
uv run --no-sync python scripts/evaluate_case_rag_jeff.py --dry-run
uv run --no-sync pytest -q tests/test_case_rag.py tests/test_jeff_format.py

# H200/GH200：使用已驗證的 Jeff v1.3.0、31B 官方基座及既有 S2 checkpoint
uv run --no-sync python scripts/evaluate_case_rag_jeff.py \
  --source data/case_rag/pilotclean_0300_20261116 \
  --model-dir /path/to/Jeff-1.1-Large \
  --checkpoint /path/to/s2-checkpoint
```

五組模型推論共用單次載入的 Jeff S2 權重。每個條件的 logits 以原子寫入存於忽略版控的 `data/case_rag/pilotclean_0300_20261116/jeff_s2_inference/`；中途失敗可重新執行命令，先驗證再續用已完成的組別。報告包含全體 Top-1、固定 148 類 Macro-F1、排除 Train 同文後 Top-1、rescue/break、McNemar p 與每條件時間。

H200 若模型/tokenizer 路徑不同，先確保 `data/models/Jeff-1.1-Large-tokenizer` 有原始 tokenizer.json / chat_template.jinja；報告中的新 RAG Prompt 在 H200 要實際確認 tokenizer 不觸發縮短。

## 判準與限制

- **主比較：** `paired4` 必須超過同等 4 醫案、同樣 Top-2 曝光的 `balanced4`；優於 `global4` 才能說明競爭條件檢索的額外價值；`random4` 是非相似支持的負控制。
- 報整體與排除精確重複病例、rescue/break、單次推論延遲；若新 Prompt 造成破壞個案多於救回，停止免微調擴大。
- 先用不改任何 checkpoint 的受控實驗驗證是否有訊號；只有新增病例確實提供模型能力之外的增益，才考慮額外 LoRA 訓練。
- 使用固定 4 個支持病例但文字長度有細微變化，後續可做完全等 token 長度控制；小樣本設計屬探索性，不可宣稱正式獨立測試。
- 配對選取採用現有 BM25 和字元相似度，技術本身已有先例，**現在無創新性或 IEEE 錄取結論**。

程式入口：
`src/tcmdiscern/case_rag.py`、
`scripts/prepare_case_rag_pilot.py`、
`scripts/audit_case_rag_input_pilot.py`、
`scripts/score_case_rag_original_s2.py`、
`scripts/evaluate_case_rag_jeff.py`；
測試 `tests/test_case_rag.py`；
結果 `experiments/case_rag_input_pilot_2026-11-16.json` 與
`experiments/case_rag_jeff_s2_original_sample_2026-11-16.json`。
