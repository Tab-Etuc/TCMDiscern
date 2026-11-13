# TCM-SD 三任務 H200 實驗規格（2026-11-13 定稿）

本文件是正式實驗優先規範。IEEE 論文草稿暫停修改。

## 一、研究問題及資料

使用官方 Train 43,180、Dev 5,486、Test 5,486 列，保留原始順序及每筆標註。

| 任務 | 輸入 | 輸出 | Train 標籤數 |
|---|---|---|---:|
| A 證型 | 主訴＋現病史＋四診 | norm_syndrome | 148 |
| B 中醫病名 | 主訴＋現病史＋四診 | lcd_name | 403 |
| C 病證配對 | 主訴＋現病史＋四診 | (lcd_name,norm_syndrome) | 1,782 |

三項任務一律以各自的 BM25 Top-16 候選進行 Jeff-1.1-Large 決策，逐列只輸出一個答案。B、C 的「標籤很多」是長尾多類別問題。病例層級多診斷集合的多標籤任務為次要延伸，評分及訓練目標分開定義。

## 二、已實際完成的 CPU Dev 基準

| 面向 | 證型 A | 病名 B | 配對 C |
|---|---:|---:|---:|
| Train 類別數 | 148 | 403 | 1,782 |
| Train 僅 1 筆的類別 | 0 | 89 | 748 |
| Train 不超過 10 筆的類別 | 14 | 242 | 1,444 |
| Dev 只出現在 Dev 的類別 | 0 | 30 | 155 |
| Dev 未見於 Train 的金標列 | 0 | 35 | 167 |
| BM25 Top-1 | 79.58% | 60.74% | 52.61% |
| BM25 Macro-F1（Train∪Dev 標籤宇宙） | 54.85% | 23.14% | 11.99% |
| Oracle@16 | 98.60% | 97.45% | 91.10% |

Train-only 五折 OOF 需在訓練題人工放入金標的列數：病名 621、配對 2,794；僅作訓練監督，Dev/Test 候選完全不補金標。這是可量化的 Train/Inference 候選差距，須列入分析。

資料位於：
- `experiments/tcmsd_joint_label_space_dev.json`
- `experiments/tcmsd_joint_candidate_allocations_dev.json`
- `experiments/joint16_dev_prep.json`

## 三、必要的實驗矩陣

| 代號 | 設定 | 訓練工作量 | 目的 |
|---|---|---|---|
| A0/B0/C0 | BM25 Top1、Recall@1/5/16、Macro-F1 | CPU | 同領域檢索底線 |
| A1/B1/C1 | Released Jeff Zero-shot，K16 | 各完整 Dev 推論 | 單次決策基準 |
| A2 | Jeff 證型 S1：one-hot CE | 3 seeds × 2 epochs | 中醫領域 LoRA 效果 |
| A3 | Jeff 證型 S2：Masked CE | 3 × 2 | 同標註集合但中立其他觀察正標 |
| A4 | Jeff 證型 S3：Set Loss | 3 × 2 | 多正解弱監督 |
| B2 | Jeff 病名 CE，K16 | 3 × 2 | 403 類長尾分類 |
| C2 | Jeff 配對 CE，K16 | 3 × 2 | 直接 1,782 類聯合診斷 |
| C3 | B2 病名＋A2/A4 證型 logits 因子化聯合 | 無新訓練、2 次推論 | 階層式診斷與直接聯合的比較 |
| C4 | C3 ＋ Train 病證相容性約束 | 無新訓練 | 一致性與未見配對 trade-off |

最完整配置共 15 次兩 Epoch 的 LoRA 訓練（30 model-epochs）。分階段進行，第一個 Seed 的 A2/A3/A4/B2/C2 先完成，再決定其餘 Seed；必要時保留負面結果。

**C3/C4 是必要強基準。** 單獨訓練病名與證型，使用獨立 Jeff logits 組成 `(d,s)`。無約束版本可輸出未在 Train 配對中出現的組合；Train 配對相容性約束版本只能輸出已見組合。兩者都需要與直接 C2 比較，且正式聯合推論不使用真實病名。

## 四、候選策略消融

已跑過的候選生成 CPU 試驗，全部固定 K16：

| 策略 | Pair Oracle@16 |
|---|---:|
| 直接 Pair BM25 Top16 | 91.10% |
| 2 個病名×8 個證型 | 85.82% |
| 4 個病名×4 個證型 | 89.68% |
| 8 個病名×2 個證型 | 87.91% |
| 8 直接＋8 階層 | 91.10% |
| 8 直接＋8 獨立病名／證型交叉組合 | 90.94% |

主候選採直接 Pair Top16；候選多樣性作長尾消融。對 Train 支援不超過 10 筆的配對，8 直接＋8 交叉候選的 Recall 52.62%，直接 Pair Top16 為 49.65%；應同時報整體與長尾的取捨。

## 五、實驗前必需的公平控制

- A/B/C 的索引及標籤分開，病例輸入只含主訴、現病史及四診，不把 `lcd_name` 當作 B/C 的已知答案。
- S2 與 S3 必須共用相同的多觀察正標候選、隨機種子、更新數與選項排列。S1 作單金標基準。
- Train 每筆只檢索其他四折病例。正式 Dev/Test 使用全 Train 索引，候選完全不注入金標。
- 原始 K16 SemIf Prompt、Jeff Decision Head、BF16、關閉 reasoning，全部保持一致。
- 每 Epoch 43,180 筆、梯度累積64、675 次 optimizer updates；兩個 Epoch 共1350 updates。LoRA rank32/alpha64，模型 LR 2e-5，Head LR 1e-4，weight decay 0.01，40 steps warmup。
- Seeds：20261112、20261113、20261114。每個方法保存 Epoch1 與 Epoch2，Dev 以三 Seed 平均 Accuracy、Macro-F1 平手決定 Epoch；Test 固定第一 Seed。
- 同一套 148 類既有論文結果只和 A 的官方 Test 比較。B/C 標籤數更多，需與相同標籤宇宙的基準比較。

## 六、每項任務完整評分

必要：Top-1 Accuracy、固定標籤宇宙的 Macro-F1、Oracle@16、在候選內的條件 Top1、BM25 2～16 救回數、BM25 第一名原本正確卻改錯數、淨改善、每 Seed／Epoch／平均與標準差、Latency median/P95、GPU 訓練及決策耗時。

長尾：按 Train 類別支援數分組 0（未見）、1、2–5、6–10、11–50、51–100、>100，各層 support/Recall/F1 與候選 Oracle。未見類別無法從純 Train 病例索引召回，評分與標籤數須單獨說清楚。

配對 C 額外報：Pair Top1、Disease Top1、Syndrome Top1、完整配對 Macro-F1、已見／未見配對、無約束組合與 Train 相容性限制的取捨、1次（C2）或2次（C3）的 Jeff forward 推論成本。

若做真正多標籤 D，需另外建立病例層級標註集合，使用 sigmoid/BCE、Dev 閾值校準，報 Micro-F1、Macro-F1、Set Precision/Recall/F1、Exact Match。由於每列原始標籤只有一個，D 的集合有效性與樣本數需單獨列出，不能和 A/B/C 的逐列分類指標混算。

## 七、H200 執行（已接通實驗程式）

證型（A）：
```bash
bash run_h200.sh plan
bash run_h200.sh prepare
bash run_h200.sh pilot
bash run_h200.sh screen --confirm-screen
bash run_h200.sh full --confirm-full
```

病名、配對（B/C）：
```bash
bash run_joint16_h200.sh plan
bash run_joint16_h200.sh prepare
bash run_joint16_h200.sh pilot
bash run_joint16_h200.sh screen --confirm
bash run_joint16_h200.sh full --confirm
bash run_joint16_h200.sh test --confirm
```

主要程式：`scripts/prepare_joint16.py`、`src/tcmdiscern/joint_study16.py`、`scripts/evaluate_joint16.py`、`scripts/joint16_h200.py`、`scripts/select_joint16.py`。所有臨床資料、候選題目、逐筆 logits、Checkpoint 都存於被 Git 忽略的 `data/`，公開報告只存彙總。

C3/C4 模型間分數融合、真正多標籤 D、Qwen3 Embedding 等強分類器比較為下一階段待實作的正式實驗項目；目前已實測的只有 BM25 CPU 基線。

## 八、優先順序與停止規則

1. 已完成 CPU 基線、長尾與 K16 上限；先實測 H200 原版 Jeff Zero-shot 與一個 Step、16 updates 的梯度/速度 Pilot。
2. 第一 Seed 跑 A2/A3/A4/B2/C2 的兩個 Epoch，保存完整 Dev logits 供後續免 GPU 的融合、排名先驗、保守切換分析。
3. C2 若在 Top1/Macro-F1 同時落後 Pair BM25 與 C3，仍記錄結果，將追加 Seeds 轉為次要；C2 若有整體或長尾收益，補三 Seed 作確認。
4. 最終候選與方法在 Dev 固定後，才跑官方 Test。

既有醫療編碼研究已有 BM25＋Retrieve/Rerank，因此學術主張應著重超長尾病名、病證結構化推論、未見標籤與精確候選上限分解，以及與零新增模型參數的因子化基準公平比較。
