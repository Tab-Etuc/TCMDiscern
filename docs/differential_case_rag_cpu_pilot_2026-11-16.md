# Differential Case-RAG：免重訓對照醫案 Pilot（2026-11-16）

## 研究決策

已在官方 TCM-SD Train 43,180／Dev 5,486 上完成 300 筆 Dev 固定抽樣
（seed=20261116）的醫案檢索、5 組 Jeff SemIf Prompt 產生、原始候選
與原 Prompt parity、模型原始 S2 logits 配對診斷、醫案來源／群組／
患者文字隔離、Token 上限及測試。尚缺**同一 S2 checkpoint 下新增
RAG Prompt 的真實 Jeff GPU inference**，因此目前不能判斷新醫案
是否提升準確率、F1 或值得額外 LoRA 微調。

本機 CodexPro 無 CUDA 裝置，也沒有 Jeff 31B 權重或 S2 checkpoint；
完整模型資產位於另一台 GH200，且目前沒有可用 SSH host 連線資料。

## 固定比較

| 模式 | 輸入額外歷史醫案 | 300 例平均 tokens | 最大 tokens |
|---|---|---:|---:|
| baseline | 無 | 742.5 | 2100 |
| global4 | 全部 Top-16 類別中的高相似醫案 ×4 | 1524.0 | 2916 |
| balanced4 | BM25 Top-2 各取 2 筆 | 1522.9 | 2917 |
| paired4 | BM25 Top-2 對照匹配，各取 2 筆 | 1522.9 | 2917 |
| random4 | Top-2 各隨機抽 2 筆作負控制 | 1521.6 | 2906 |

每個 RAG Prompt 均使用完整原病歷、原先 16 個候選名稱與相同
模型推論結構，僅替換四筆輔助醫案。證據來自官方 Train，
病患／完全相同文字傳遞群組只允許獨立且不含歧義的來源；
最新版 `case_rag.py` 也禁止與查詢完全同文或同患者的支持醫案。
完整正式 Test 從未用於本實驗。

為評估消融，`paired4` vs `balanced4`：
- 192/300（64%）查詢的來源醫案集合確實改變；
- 跨證型配對病例的平均字元 bigram Jaccard
  從 0.482404 提升至 0.519247；
- 配對醫案平均 BM25 278.921，平衡醫案 280.431；
  Token 基本相同。這些只反映檢索證據不同，
  **不能當作診斷正確率提升**。

## 同一 300 筆原始 Jeff S2 已有 logits（零新增模型 forward）

- BM25 baseline 正確 242/300 = 80.6667%；
- S2 原始 logits 正確 245/300 = 81.6667%；排除 Train 完全同文 33 筆後，在 267 筆中 Top-1 = 79.7753%；
- 金標在原 BM25 Top-2：272/300；
- 金標在原 BM25 Top-16：297/300；
- S2 錯而金標在 Top-2：37 筆；第 3～16：15 筆。
- 300 筆中 33 筆與 Train 完全相同文字；
  正式比較須額外報告排除這些病例的結果。
- 所有 RAG 模式的 300 筆真實 Jeff output **尚未產生**。

目前 Top-2 對照醫案特別瞄準原 S2 的 37 個有救回空間錯誤，
同時有破壞原 S2 245 個正確病例的風險。需要 paired rescue/break、
固定 148 類 Macro-F1、排除重複子集、分組 bootstrap、
推論耗時及負控制共同判斷。

## 本機來源與重現

核心程式：
- `src/tcmdiscern/case_rag.py`
- `scripts/prepare_case_rag_pilot.py`
- `scripts/audit_case_rag_pilot.py`
- `scripts/audit_case_rag_input_pilot.py`
- `scripts/run_case_rag_jeff.py`
- `tests/test_case_rag.py`

輸入位於被 Git 忽略的
`data/case_rag/pilotclean_0300_20261116/`。
統計報告位於 `experiments/case_rag_input_audit_2026-11-16.json`
與 `experiments/case_rag_input_pilot_2026-11-16.json`。

```bash
cd TCMDiscern
uv run --no-sync python scripts/prepare_case_rag_pilot.py --samples 300 --seed 20261116 --tag pilotclean
uv run --no-sync python scripts/audit_case_rag_pilot.py --tag pilotclean_0300_20261116
uv run --no-sync python scripts/run_case_rag_jeff.py --tag pilotclean_0300_20261116 --preflight
uv run --no-sync pytest -q tests/test_case_rag.py tests/test_jeff_format.py tests/test_jeff_zeroshot_runner.py
```

H200（使用實際官方 Jeff 模型與 S2 訓練 checkpoint）：
```bash
cd /home/quanta/TCMDiscern-main
# 需確保此目錄已同步新程式、官方 Train/Dev 與原 BM25 候選
python -m scripts.prepare_case_rag_pilot --samples 300 --seed 20261116 --tag pilotclean
python -m scripts.run_case_rag_jeff \
  --tag pilotclean_0300_20261116 \
  --model-dir data/models/Jeff-1.1-Large \
  --checkpoint /path/to/s2/checkpoint \
  --run-id case-rag-s2-300
```

測試：Case-RAG 及相關專項 21/21 PASS，相關 Ruff PASS。完整測試 411/412 PASS，
1 FAIL 為本機已知的 Transformers 缺 `transformers.models.gemma4`
環境問題，與 Case-RAG 無關。

## 下一階段決策

在同一真實 S2 checkpoint 及相同 300 筆 Prompt 上完成五模式推論，
比對 `paired4` 相對 `balanced4`、`global4`、`random4` 與
`baseline` 的淨救回／破壞。配對法必須在相同 Token／醫案個數下
提升 Top-1 和 Macro-F1，且無同文與非重複子集有一致趨勢，
才值得考慮 LoRA。若結果缺乏增量，停止醫案匹配法。
