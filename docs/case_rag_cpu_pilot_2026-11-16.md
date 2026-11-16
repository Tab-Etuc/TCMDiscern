# Differential Case-RAG：零微調 CPU 前置實驗（2026-11-16）

## 決策狀態

**已完成可重現的檢索、提示組建、資料隔離及評估接口；實際 Jeff S2 加入醫案後的準確率尚未測得。**

主假說：既有 BM25 → 16 個證型名稱的流程丟失了相似病例原文；
保留相同候選後，提供兩個競爭證型各兩份標註醫案，
是否改善 Jeff-1.1-Large S2 原本無法解決的鑑別診斷？

這是**檢索整合技術假說**，需要實際模型推論證明增益。

## 資料與對照

- 唯一病例來源：官方 TCM-SD Train 43,180 筆；
  官方 Dev 固定隨機抽樣 300/5,486，seed=20261116。
- 固定原始 BM25 Top-16 候選，患者三欄全文完全一致。
- baseline：原始 Jeff 輸入，平均 742.5 tokens。
- global4：BM25 相似度最高的四筆獨立醫案，平均 1,524 tokens。
- balanced4：BM25 前兩個候選，各取兩筆最相似醫案，平均 1,522.9 tokens。
- paired4：前兩個候選，各兩筆；用病例間症狀相似性選擇跨證型匹配對，
  平均 1,522.9 tokens。
- random4：兩個候選各兩筆，從高排名候選醫案中以固定種子隨機抽取，
  平均 1,521.6 tokens。
- 每個醫案含固定長度三欄病歷、Train 原標註證型；來源位置/群組只留在私有 provenance。
  同一病人、同文、衝突標註群組禁止作為患者自身的檢索證據。
- 全部輸入通過原版 Jeff tokenizer 的 8192-token 以及 16 選項完整性檢查。
- 私有輸入：`data/case_rag/pilotclean_0300_20261116/`。
  原始 pilot 仍保留以供核查，但新增的 patient/text 排除策略以 **pilotclean** 為準。

## 已有數字（檢索／基準，非新增 Jeff 診斷成績）

| 指標 | 結果 |
|---|---:|
| 固定抽樣 | 300 筆 |
| 原 BM25 正解在 Top16 | 297/300 |
| 原 BM25 正解在 Top2 | 272/300 |
| 原 Jeff S2 的已保存 logits：Top1 | 245/300 = 81.6667% |
| 原 BM25 Top1 | 242/300 |
| 原 Jeff S2 判錯而正解在 Top2 | 37 筆 |
| 原 Jeff S2 判錯而正解在第 3～16 名 | 15 筆 |
| Dev 與 Train 完全同文的查詢 | 33/300（最終另外報排除子集） |
| balanced4 與 paired4 使用不同醫案 | 192/300 = 64% |
| balanced4 跨證型匹配詞彙 Jaccard | 0.4887 |
| paired4 跨證型匹配詞彙 Jaccard | **0.5254** |

balanced4、paired4、random4 三種策略，各使用兩類各兩筆病例，具有相同候選曝光與相近 token 預算。
配對策略確實找到更相似的跨證型病例，但這是檢索方法的直接結果，
**不能將 Jaccard 增加視為診斷能力增加**。
一般 Top-4 只涵蓋平均 1.45 個不同證型；另外三策略各涵蓋固定兩類。

## 驗證

```bash
uv run --no-sync python scripts/audit_case_rag_input_pilot.py
uv run --no-sync python scripts/audit_case_rag_pilot.py
uv run --no-sync python -m scripts.run_case_rag_jeff --preflight
uv run --no-sync pytest -q tests/test_case_rag.py tests/test_case_rag_runner.py
uv run --no-sync ruff check src/tcmdiscern/case_rag.py scripts/prepare_case_rag_pilot.py scripts/audit_case_rag_input_pilot.py scripts/audit_case_rag_pilot.py scripts/run_case_rag_jeff.py scripts/evaluate_case_rag_jeff.py
```

本機實測：10 個測試通過，Ruff 通過。評估程式 preflight 顯示
`READY_FOR_H200`。

現有 CodexPro 工作機為 RTX 4060 8GB，Python torch CUDA unavailable，
`data/models/Jeff-1.1-Large` 僅有 tokenizer；
沒有 31B 實際 Jeff 權重與 S2 checkpoint。
也沒有可直接連線的 H200 計算工作區。真實推論呼叫已測試，
在 CUDA 檢查階段按預期停止。

## H200 的實際零微調測試

在已有原版 Jeff-1.1-Large 與 S2 LoRA/Head checkpoint 的 H200，
將程式同步後，官方 Train/Dev/BM25 候選準備齊全：

```bash
uv run --no-sync python scripts/prepare_case_rag_pilot.py \
    --samples 300 --seed 20261116 --tag pilotclean
uv run --no-sync python -m scripts.run_case_rag_jeff \
    --tag pilotclean_0300_20261116 \
    --run-id case-rag-s2-300 \
    --model-dir /path/to/Jeff-1.1-Large \
    --checkpoint /path/to/actual/s2-checkpoint
```

注意：若輸入資料夾已存在，準備工具會阻止覆寫；直接用原檔。
H200 需以實際存在的權重與 adapter 路徑取代示例路徑。

真實推論一次載入原版 Jeff，恢復已微調 S2，
分別對五條件各 300 筆做原始 16-way logits；支持中斷後逐條件續跑，
模型無新增訓練。私有 logits 保存在 data/；公用報告只保存 aggregate，
包含 Top-1、固定 148 類 Macro-F1、排除 Train 同文後 Top-1、
成對 rescue/break、group bootstrap 區間與 McNemar 統計。

## 決策準則

- paired4 要在相同 4 醫案與 token 預算下，優於 balanced4 和 random4；
  若只有一般 RAG 有效，就只採用一般 RAG。
- 強基準仍包括無醫案的 Jeff S2、Top-3 BM25、平衡 TF-IDF SGD、
  Jeff S1＋SGD 融合。
- 300 筆僅為探索性 pilot，長尾 Macro-F1 波動很大。
  官方 Test 維持封存，未取得真正 Jeff RAG logits 前不宣稱增益或 IEEE 可投稿性。
