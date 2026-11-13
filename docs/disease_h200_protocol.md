# H200 疾病條件化 Jeff + BM25 實驗定稿

## 研究問題

- A：由病例文字預測 148 類證型（原始官方逐列分類基準）。
- B：由病例文字加已知中醫病名，預測單一證型（疾病條件化）。
- C：由病例文字預測中醫病名與證型的有序配對（未知病名聯合分類）。
- 主要新方法：B + Jeff-LoRA + 同疾病 OOF 困難負例成對排序損失。

官方 Train 43,180、Dev 5,486、Test 5,486 筆資料列全部使用；保留原始官方切分、原始標籤與每筆病例，採逐列 Top-1 和 Macro-F1。

## 一鍵執行

在有足夠磁碟、H200 及官方模型存取權限的 Linux 主機上，Clone 含這些檔案的專案後：

```bash
cd TCMDiscern
bash run_disease_h200.sh plan
bash run_disease_h200.sh prepare
bash run_disease_h200.sh pilot
bash run_disease_h200.sh full --confirm
bash run_disease_h200.sh status
bash run_disease_h200.sh test --confirm
```

執行 `full` 會完整跑三個原始 Jeff Zero-shot Dev、四種兩 epoch 的 LoRA（各三種子）、每組 Dev 評估及固定選模；`test` 會依 Dev 選擇的每種方法代表 checkpoint，在完整官方 Test 逐列評估，並產生比較表所需的 JSON。執行 `full --dry-run` 可以先查核全部命令。若要指定已選定的代表模型：`test --confirm --selected-run disease-conditioned_hard-seed20261112-e2`（需先完成該 Seed 的 Dev 評估）。

專案使用現有 `run_h200.sh` 安裝受釘選的 Python 3.12、Jeff v1.3.0、原版 Jeff-1.1-Large、CUDA 13 相依套件及 GPU Pilot，另外使用 `run_disease_h200.sh` 的完整執行與 checkpoint 續跑。GPU 硬體需要依原版 Jeff 配置提供足夠顯存及 NVIDIA R580 或更新 Driver。模型下載授權由官方 Hugging Face 機制處理。

## H200 訓練細節

| 參數 | 固定值 |
|---|---|
| 研究骨幹 | jgeuter/Jeff-1.1-Large，原版 Head |
| 候選數 | K=5；A–E，從原版 16 槽 Head 取前 5 槽 |
| 候選檢索 | CJK bigram BM25，k1=1.5，b=0.75，min_df=2 |
| 候選聚合 | 每標籤取最高 BM25 病例分數 |
| OOF | 原 Train 五折；每例使用其他四折索引 |
| LoRA | rank=32，alpha=64 |
| 精度 | BF16 backbone、FP32 Head |
| LoRA LR / Head LR | 2e-5 / 1e-4 |
| Batch / 梯度累積 | 一例一 forward，64 次累積才更新 |
| Weight decay / 梯度裁剪 | 0.01 / 1.0 |
| 訓練 | 2 epoch × 675 updates = 1350 updates |
| Seeds | 20261112、20261113、20261114 |
| 成對損失 | CE + 0.2 × softplus(logsumexp(至多兩個疾病內高排名負例) − gold_logit + 0.2) |
| 選模 | 三種子 Dev Macro-F1 平均優先，Top-1 平均次之；各方法代表 checkpoint 由 Dev 選定 |

四種主要訓練：`global`、`conditioned`、`conditioned_hard`、`pair`。每組三種子共 **12 次完整兩 epoch LoRA**。所有對照使用一樣的 43,180 筆 Train 及更新次數；已知病名任務 B 使用病例真實 `lcd_name` 作為事前給定條件，另外與只用病名的 Train 標籤先驗比較。

每筆 Train 題的金標僅存於 supervision metadata。Dev/Test 題只有病例證據與真正檢索出的五候選。模型從候選選擇首位答案。

## 產出與重現

- `data/disease_study/train_{global,conditioned,pair}_k5.jsonl`：43,180 筆 Train supervision。
- `data/disease_study/dev_{mode}_prompts.jsonl`：Jeff 評估 Prompt，不含 gold。
- `data/disease_study/dev_{mode}_labels.jsonl`：對應 gold 與候選，用於評分。
- `experiments/disease_study_dev_baselines.json`：CPU BM25 及病名先驗實測。
- `data/runs/disease-*/`：LoRA、Head、optimizer、scheduler 及續跑 checkpoint。
- `experiments/disease_study_dev_{mode}_{run}.json`：Jeff 各組逐次 Dev 評估。
- `experiments/disease_study_dev_selection.json`：三種子選模與最終 Test 使用的模型。
- `experiments/disease_study_test_{mode}_{run}.json`：Test 結果，於選模後生成。
- `data/disease_study/*_logits.jsonl`：逐筆模型預測與 logits。

## 已完成的 CPU Dev 結果

| BM25 方法 | Top-1 | Macro-F1 | Oracle@5 |
|---|---:|---:|---:|
| Global | 79.5844% | 54.8496% | 95.4612% |
| Conditioned | 81.3161% | 57.4349% | 96.0992% |
| Pair | 52.6066% | 11.9925% | 84.2326% |

提供病名的 Conditioned BM25 較 Global BM25 的 Top-1 提高 1.7317 個百分點；Macro-F1 提高 2.5853 個百分點。Pair 為較大的病名—證型配對標籤空間，還需同時報病名、證型及完整配對的個別 Accuracy。

目前 H200 GPU Zero-shot／LoRA 成績留待實際執行，實驗論文中對應表格保持空白。
