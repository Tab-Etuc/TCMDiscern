# 研究資料清冊

TCMDiscern 的研究資料僅放在本機 `data/`；這個目錄整體不進 Git。資料授權仍遵守各原始提供者的條款。

## 核心資料：TCM-SD

官方來源：[Borororo/ZY-BERT](https://github.com/Borororo/ZY-BERT)，資料集採 **CC BY-NC-SA 4.0**，非商業學術研究使用。

| 本機位置 | 筆數 | 用途 |
| --- | ---: | --- |
| `data/tcm-sd/train.json` | 43,180 | 訓練及 Train-only 檢索庫 |
| `data/tcm-sd/dev.json` | 5,486 | 模型選擇、調參、Adaptive K 驗證 |
| `data/tcm-sd/test.json` | 5,486 | 最後一次正式評分 |
| `data/tcm-sd/syndrome_knowledge.json` | 1,027 | 候選證型知識描述；僅使用公開、不含患者答案的內容 |
| `data/tcm-sd/syndrome_vocab.txt` | 148 | 官方證型詞表 |
| `data/TCM_SD_train_dev.zip` | — | 官方 Train/Dev/Knowledge 原始壓縮包 |

官方 `.json` 的病例檔為 JSON Lines。欄位包括：

- 輸入：`chief_complaint`（主訴）、`description`（病情描述）、`detection`（四診資訊）
- 證型金標：`norm_syndrome`；原始標籤：`syndrome`
- 其他欄位：`user_id`、`lcd_id`、`lcd_name`。疾病代碼與病名不能未經確認就當作西醫 ICD/診斷；患者 ID 僅用於重複情況的記錄，不做資料排除或重新切分。

**輸入建構一律只選擇診斷前欄位。** 檢索資料、LoRA prompt 不帶入 `norm_syndrome`、`syndrome`、`lcd_id`、`lcd_name` 或 `user_id`。

## 補充資料：獨立評估／後續擴展

| 本機路徑 | 來源 | 使用限制 |
| --- | --- | --- |
| `data/secondary/syndrome819/cases.json` | 使用者既有 819 筆／78 類多標籤研究集 | 只作分開的研究評估；標籤空間與 TCM-SD 不同，不納入主訓練 |
| `data/secondary/mtcmb/{diagnosis,msdd}.jsonl` | [MTCMB](https://github.com/Wayyuanyuan/MTCMB) | 含診斷相關評測，但其任務與 TCM-SD 148 類不同；需保留各自評分規則 |
| `data/secondary/robust-sdt/*.jsonl`、`longitudinal.json` | [TCM-RobustSDT](https://github.com/DYJG-research/TCM-RobustSDT) | 供日後魯棒性與跨資料集測試；不可混用原有金標 |
| `data/secondary/best4sdt/cases.json` | [TCM-BEST4SDT](https://github.com/DYJG-research/TCM-BEST4SDT) | 多任務評估資料；不能直接當作 148 類單標籤準確率 |

補充資料由既有 `CMU_AITCM` 的本機快取複製，來源檔案未更動；`syndrome819` 來自使用者的本機研究資料。這些檔案不會隨 Git clone 自動取得。資料授權與使用範圍應依各來源專案的最新規定確認。

避免引入 TCM-SD 同源的 `syndrome-48665.json` SFT 副本或 `TCMSD100` 作為「獨立資料集」，因為它們和 TCM-SD 高度重疊。

## 操作

```bash
# 新環境重新取得官方 TCM-SD
uv run python scripts/fetch_tcmsd.py

# 從既有本機 CMU_AITCM 及研究資料複製補充資料
uv run python scripts/sync_existing_data.py

# 產生不含病例原文的摘要統計
uv run python scripts/audit_datasets.py
```

稽核結果在 `data/reports/tcmsd_audit.json`，補充資料清單在 `data/reports/secondary_sources.json`，均不進 Git。

## 已發現的資料風險

2026-11-11 對官方資料實測：

- 54,152 筆，Train／Dev／Test：43,180／5,486／5,486，148 類，證型知識 1,027 條。
- 完整四診文字在 Train 與 Test 之間重複 **553 筆**，其中 **23 筆**的標準證型與 Train 不一致。
- Train 與 Dev 之間完整四診文字重複 **529 筆**，其中 **20 筆**標籤衝突。
- `user_id` 對照 Train 在 Dev 有 19 筆、Test 有 20 筆重複患者代碼所屬的病例；僅記錄，**不排除任何病例**。
- 長尾明顯：Train 最多類別 7,380 筆，最少 7 筆。應報告 Macro-F1 與長尾分層成績。

**定稿（2026-11-11）：唯一正式基準為官方原始切分，Train 43,180／Dev 5,486／Test 5,486。零刪除、零排除、零重新切分，標籤衝突照官方標註保留。** 上述重複情況只作描述性稽核，論文如實揭露可能的樂觀偏差；不另建去重 benchmark。正式 Test 保留到模型定案後一次性評估。詳見 [研究定稿](research_protocol.md)。
