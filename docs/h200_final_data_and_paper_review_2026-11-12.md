# H200 前：論文方法、傳輸資料與逐筆檢核報告（2026-11-12）

## 結論

已重新開啟並逐筆比對官方 Train 43,180、Dev 5,486、有效的 Group-aware BM25 OOF 43,180、S1 訓練題 43,180、S2/S3/S4 共用訓練題 43,180、Dev BM25 5,486、Dev K=5/16/17 每組 5,486。首次在本機另對封存 Test 5,486 筆只做原始 schema/字元掃描；**H200 預設資料稽核完全不開 Test，也不傳 Test 檔**。既有報表不當作驗收替代品，驗收程式重新讀每一列 JSONL 並與來源病例、候選和標籤核對。

**本機驗收命令**（使用已產生的資料、唯讀、不覆寫正式研究快照）：

```bash
uv run python -m scripts.audit_h200_payload --check-only
uv run python -m scripts.run_jeff_kgrid --dry-run --all-ks
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## 送 H200 的唯一正式工作資料清單

正式研究先傳下表；上傳前應重新跑完整本機稽核，在 H200 解壓後再跑同一指令。原始 Test、歷史舊版 OOF、舊訓練 epoch、TCM_SD_train_dev.zip 和補充資料都**不要傳入訓練環境**。

| 路徑 | 正式用途 | 驗證 |
|---|---|---|
| `data/tcm-sd/train.json` | 稽核訓練來源；重建資料必要 | 每筆原始臨床欄位、148 類合法金標、原始順序 |
| `data/tcm-sd/dev.json` | 驗證分母與模型選擇；只於推論後評分讀金標 | 5,486 筆來源與候選對應 |
| `data/tcm-sd/syndrome_vocab.txt` | 148 類正式定義 | 全部互異且與 Train/Dev 對照 |
| `data/candidates/bm25_train_text_grouped.jsonl` | Group-aware OOF 候選來源追溯 | 43,180 筆、5 folds、每筆 50 個互異證型 |
| `data/candidates/bm25_dev.jsonl` | 正式 Dev 候選 | 5,486 筆、每筆 50 個互異證型，對應金標僅供評分 |
| `data/jeff/train_oof_k16_epoch02.jsonl` | S1 安全單金標對照 | 43,180 題、每題 16 選項、gold 正確，無已知同文衝突負例 |
| `data/jeff/train_oof_multipos_k16_epoch03.jsonl` | S2/S3/S4 共用訓練候選 | 43,180 題、每題 16 選項、1～3 個觀察正標集合、全部 mask 與軟標註位置 |
| `data/jeff/dev_k05.jsonl` | K=5 預生成 Dev 對照 | 5,486 筆逐筆對齊 |
| `data/jeff/dev_k16.jsonl` | K=16 主要 Dev 實驗 | 5,486 筆逐筆對齊 |
| `data/jeff/dev_k17.jsonl` | K=17 分組 Dev 實驗 | 5,486 筆逐筆對齊 |

Token 長度已用指定 Jeff tokenizer 對完整主要 Train／Dev 做過檢查：最長約 3,577／3,578 tokens，都低於 8,192 上限。這次稽核另外檢查所有文字、欄位、標註與 Prompt 的一致性。真正模型 forward/反傳／顯存還沒有實測。

### 來源資料異常

- Train 主訴欄 8 個 `U+007F`（DEL）控制字元，Dev 主訴 1 個，共 9 個。稽核報告只存原始 row index/欄名/Unicode codepoint，不輸出病歷文字；由於不修改原始 benchmark，保持照原資料輸入。建議再做「保留／移除控制字元」的敏感度比較，正式主表維持保留設定。
- Train 主訴空值 14 筆、四診空值 3 筆；Dev 主訴空值 3 筆、四診空值 1 筆。全部原始紀錄保留，沒有全空病歷。
- 官方 Train 有 252 組同一完整臨床輸入但金標相異，共涉及 820 列；S2～S4 依原 Train 觀察集合學習。這**不是已經由醫師確認的多個臨床真值**。
- 任何 Test 的分析結果不得參與選超參數、K、訓練資料建構、檢索索引或模型挑選。

## 與論文／上游公式再次對齊的關鍵點

1. **TCM-SD（CCL 2022）**：分類主要以 chief complaint + medical history 當模型特徵；原論文的「五候選」設定可能事先提供金標，與本研究真實 BM25 top-K 是不同的候選任務。投稿要清楚標示三個臨床欄位與兩個欄位的條件差異。原文：https://aclanthology.org/2022.ccl-1.80/
2. **Jeff（j-geuter/Jeff，v1.3.0）**：模型原生 16-slot Decision Head，Prompt 使用固定 SemIf 系統訊息與 `state/question/options`，不啟用 reasoning。K>16 先依官方 `decide_wide` 分組再選各組冠軍；分組後的輸出為由分組機率轉換的對數分數，不能當成單次原始 logits。官方來源：https://github.com/j-geuter/Jeff/tree/v1.3.0
3. **Loss 假設**：Masked CE 忽略其他已觀察正標；Set Loss 邊際化集合中的各正標；Uniform Soft CE 額外強制等權。S2/S3/S4 共用候選可隔離 Loss；S1 候選與 S3 不同，S1→S3 的差異不可全歸因於 Loss。
4. **候選數效益**：K=16 的官方 Dev BM25 micro recall 約 98.60%、macro recall 約 89.91%，只是 Jeff 在限定候選中的潛在上限；不等於 Jeff Top-1。大 K 可能因選項干擾、分組步數上升而降低真實準確率。
5. **論文基準可比性**：TCM-SD 原論文的 148 類 Accuracy/Macro-F1 應當成歷史對照；BM25 + Jeff 的檢索－決策主表應同時列出 BM25 Top-1、Jeff zero-shot、S1～S4 及每 K 的候選 Recall，上述所有方法使用相同官方 Dev。
6. **資料洩漏**：Train/Dev 完整臨床文字重疊 529 筆；可另以預先固定的重疊／非重疊分層，報告敏感度，但不能刪除官網原始測試分母，也不能稱為患者完全獨立泛化。
7. **長尾證型**：Train 148 類最少 7 筆、最多 7,380 筆；Dev 有許多僅 2～5 筆支援的類別，Macro-F1 必須報每類 support、信賴區間及多 seed 的波動。
8. **Token 預算公平性**：1 epoch 約 30.3M 個 Train prompt tokens（不含 padding）。只報 optimizer updates 會掩蓋序列長度差異，應同時報病歷處理筆數、實際有效 token、累積 batch、訓練時間/顯存與 seeds。設定每方法相同 step、病例曝光與 seed。
9. **新發現的 H200 因果遮罩驗收**：已加 `jeff_attention.py` 的合成問題前綴一致性測試，且真實模型載入入口會先跑遮罩檢查，避免 Gemma 4 注意力實作讓後續 token 影響早期隱藏狀態。合成 CPU 測試與真實 H200 model check 是兩個不同層級；後者仍待 H200。
10. **安全重現**：只在 H200 上完成正式 Step-0 logits parity、1 update + checkpoint reload parity、BF16 LoRA 梯度、GPU 記憶體峰值後才啟動完整訓練。正式 Test 應延後到所有 Dev 決策凍結後一次執行。

## 已完成的新輸入消融（重建 BM25，非 Jeff）

完整官方 43,180 Train 建索引，5,486 Dev 計分：

| Feature | BM25 Top-1 | Macro-F1 | Recall@16 | Macro Recall@16 |
|---|---:|---:|---:|---:|
| 主訴＋病史 | 78.67299% | 0.532755 | 98.46883% | 89.51964% |
| 主訴＋病史＋四診 | 79.58440% | 0.548496 | 98.59643% | 89.91165% |

完整檔：`experiments/bm25_chief_history_vs_four_diagnoses_2026-11-12.json`。兩個方法在完整 Dev 同時比較，未做顯著性或多重比較檢定；保留負面或不顯著結果。

## 需要明確保留的研究限制

全量「看資料」可以客觀確認原檔和衍生檔的每筆結構、欄位、金標、候選集合、正標 mask 與來源可追溯；**不能在沒有臨床醫師逐案裁決前替 54,152 筆病歷核發醫療真值**。目前尚未人工判定同義證型、語義近重複、可接受多正解與病史記載品質。官方單金標就是此研究可量化的 benchmark ground truth，臨床有效性另作限制。

## 禁止直接啟動大規模訓練的條件

在 H200 上以下任一未通過均不能執行長訓練：全部資料再次逐筆驗證、Jeff 模型 revision 與 tokenizer/head 版本檢查、真正因果遮罩 prefix check、官方原版 16 logits 與新 LoRA Step-0 parity、第一次 backward 的 adapter+head 梯度、一步更新後 checkpoint/optimizer 無損續訓、GPU peak VRAM 與足夠磁碟空間。
