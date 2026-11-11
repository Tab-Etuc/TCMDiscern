# Jeff-1.1-Large 中醫辨證微調實作決策

**決策日：2026-11-11｜資料準備已完成（43,180 Train 五折 OOF／129,540 Jeff 監督題）；模型 LoRA 訓練尚未啟動。**

## 結論與優先順序

主要方法：**TCM-SD → BM25 互異證型候選 → Jeff-1.1-Large（Gemma 4 31B-it backbone）→ 中醫 LoRA + Decision Head → 證型選擇**。

- **第一優先：** 從公開的 `jgeuter/Jeff-1.1-Large` 權重繼續做中醫領域適應，凍結已存在的合併 backbone 權重，在其上新增可訓練 LoRA，並微調既有 Decision Head。
- **對照基準：** BM25 Top-1、未額外微調的 Jeff-1.1-Large。若資源允許，再對照原始 Gemma 4 31B-it 的同規模 LoRA 微調；此對照非本階段必要條件。
- **唯一正式資料：** TCM-SD 官方 Train 43,180／Dev 5,486／Test 5,486，148 類。**全部原始病例保留，不去重、不排除、不重新切分、不改標籤**。不混用 819 筆研究集。
- **K 比較：** 全部 **K=1～35**，每 K 使用相同完整 Dev、相同 BM25 候選排名及同一 Jeff checkpoint。K=1 直接使用 BM25 第一名，作為基準；只有 K≥2 需要真正的選項決策。
- **已完成／未完成：** BM25 K=1～35、Train 5-fold OOF BM25、Jeff 格式監督題目（K=5／16／混合 2～16）與逐筆完整性檢查**已完成**。Jeff LoRA、Decision Head 領域適應及端到端 Dev/Test 尚未執行；不得將候選 Recall@K 宣稱為 Jeff 的 Top-1。

## 原作者可驗證的架構與技術限制

資料來源：
- [Jeff 官方 README](https://github.com/j-geuter/Jeff/blob/main/README.md)
- [Decision Head 實作](https://github.com/j-geuter/Jeff/blob/main/src/jeff/head.py)
- [Prompt 格式](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py)
- [模型推論與 >16 選項分組](https://github.com/j-geuter/Jeff/blob/main/src/jeff/model.py)
- [Jeff-1.1-Large 公開權重](https://huggingface.co/jgeuter/Jeff-1.1-Large)
- [研究報告](https://arxiv.org/abs/2610.17205)

Jeff 使用最後提示詞位置的 hidden state `h`，將 Gemma 的 A～P 答案字母讀出與可訓練修正層合併：

`logits = softcap(h @ W0.T) + A @ ((h - mu) / sigma) + b`

其中 `W0` 為語言模型原字母 logits 對應權重，`mu/sigma` 是已發布 head 內的特徵標準化統計，`A/b` 是修正層。**本次從已發布 Jeff 繼續訓練時應載入作者的既有 Decision Head、起始權重及 Prompt 模板，不將其重新隨機初始化。**

已發布的 Jeff 程式庫主要提供**服務／推論與 Head 載入程式**；在公開 main 樹中未見完整訓練腳本，因此以下微調流程屬於 **依原作者架構實作的 TCM 領域適應**，不能聲稱已直接重現作者私有的訓練流程。

公開權重已包含合併後的 Jeff 調整效果。**本次新增 LoRA 應附加在該發布權重上**，而非假設仍能取得可分離的原作者 LoRA；微調後需要與 Decision Head 一同保存並重新評估機率校準。作者的原始 `serving_rule.json` 溫度是原任務驗證集所調，不應直接當成中醫病例上的最佳校準。

## Jeff/SemIf 病例格式（已實作、已驗證）

Jeff 原作者使用的每道題目格式是 `{"id","state","question","options"}`。正式模型會透過官方 `jeff.prompting.messages` 將其轉成固定 System Message 與一個 JSON User Message，欄位依序為 `evidence`（病例）、`criterion`（判斷問題）、`options`（A～P 選項）。最後必須呼叫模型自己的 `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`。**格式和 Prompt 一致，才可直接沿用 Jeff 已學會的多選題決策習慣。**

| TCM-SD 原始欄位 | Jeff 的 `state/evidence` |
|---|---|
| `chief_complaint` | 保留主訴文字 |
| `description` | 保留病情描述 |
| `detection` | 保留四診文字 |
| `norm_syndrome` | **只當 Train 監督標籤，不能進入病例 evidence** |
| `syndrome`、`lcd_id`、`lcd_name`、`user_id` | **不進入 evidence** |

**來源程式：** [Jeff prompting.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py)，[Jeff model.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/model.py)。

TCMDiscern 已新增無 GPU 依賴的轉換器 [`src/tcmdiscern/jeff_format.py`](../src/tcmdiscern/jeff_format.py)，含 `decision_row`、`train_example`、`semif_messages` 與 `render_jeff_prompt`。

### 訓練示意

以下僅為**虛構病例與示意選項**，不代表實際 TCM-SD 標註：

```json
{
  "row": {
    "id": "train-000001",
    "state": {
      "chief_complaint": "咳嗽",
      "description": "痰多胸悶",
      "detection": "苔白膩、脈滑"
    },
    "question": "根據患者主訴、病情與四診資訊，選出最符合的中醫證型。",
    "options": [
      {"id": "風寒襲肺", "description": "風寒襲肺"},
      {"id": "痰濕蘊肺", "description": "痰濕蘊肺"},
      {"id": "肺氣虛", "description": "肺氣虛"}
    ],
    "qtype": "choice"
  },
  "target_index": 1,
  "target_letter": "B"
}
```

`row` 才是輸入 Jeff 的物件；`target_index/target_letter` 是**TCMDiscern 自訂的訓練資料標籤封裝**，只交給 Cross-Entropy Loss 使用，不能序列化進 Prompt。Jeff 上游公開 main 提供推論格式與 Head，未釋出原作者完整訓練資料輸出檔案規格，因此不能把這個外層封裝宣稱為上游官方訓練檔格式。

原作者的選項按照 A～P 依序編號；每個訓練 epoch 都要重新打亂選項，**同時重新計算正確答案的位置**。目前轉換器支援直接決策的 K=2～16；K=17～35 輸出完整 Jeff Row，再交給作者 `Decider.decide_wide` 分組決策，**不可直接以 17/35 個選項呼叫 `prompting.messages`**。K=1 直接採 BM25 第一名。

### 真實 Dev 格式已產生並核對

```bash
uv run python scripts/export_jeff_dev.py --k 5
uv run python scripts/export_jeff_dev.py --k 17
```

已完整產出本機：

- `data/jeff/dev_k05.jsonl`：官方 Dev **5,486** 筆，K=5。
- `data/jeff/dev_k17.jsonl`：官方 Dev **5,486** 筆，K=17；後續必須由官方 Jeff 作分組推論。

兩份檔案只有病例查詢與 BM25 候選，不含 Dev `gold_labels` 或 `target_index`，並受 `.gitignore` 的 `data/` 規則保護。直接用從 GitHub 下載的**官方 `prompting.py`**比對了 20 筆真實 Dev 的 K=2、5、16：System Message、User JSON 及 chat-template 參數**完全一致**；K=17 的 Row 也通過官方 `validate_row(max_options=256)`。

**Train 全量 43,180 筆的五折 OOF 已實際完成，且已匯出三套 Jeff 訓練題目（K=5／16／混合 2～16），共 129,540 筆，逐筆 QA 全部通過。** 詳見 [完整資料產生、OOF 分數及驗證報告](jeff_train_data.md)。OOF 只排除相同原始資料列；跨折 6,054 筆文字完全相同的病例依定稿保留並揭露。

## 訓練題目如何建立

針對每筆官方 Train 病例的真實標籤 `y`：

1. 僅用 `chief_complaint`、`description`、`detection` 建立病例描述，絕不在 Prompt 輸入原始證型答案、`lcd_id/lcd_name` 或 `user_id`。
2. 在 **Train 內部 5-fold Out-of-Fold** 建 BM25 候選。每次只用其他四折生成候選，避免同一列當自己的檢索結果；原始官方 Train/Dev/Test 切分完全不變。
3. 從 OOF BM25 前 50 個互異證型候選中排除與官方 `norm_syndrome` **完全相同**的正解，取排序最前面的 K−1 個標籤作 **Hard Negatives**。148 類使用官方正規化標籤；語意近義但標籤不同者的額外同義性檢核尚未執行，需記入研究限制，後續可再做獨立消融。
4. **訓練階段**刻意放入 1 個正解 `y`，並搭配 K−1 個錯誤標籤（例如 K=5 為 1+4）。每個 epoch 隨機打亂 A～P 選項位置，避免答案位置捷徑；負例採樣種子、候選清單和正確位置留本機紀錄。
5. 以 **Cross-Entropy** 訓練：正確候選的 Head logit 應最高。以可回溯的 LoRA + Head optimizer/scheduler、warmup、梯度累積、有效 batch、序列長度、訓練步數、checkpoint 和隨機種子記錄完整設定；先小規模實際測 GPU 記憶體與 loss 下降，再定具體參數。

訓練時的「正解強制放入選項」是多選題監督資料建立方式；**正式 Dev/Test 的真實候選保持 BM25 輸出原樣**，絕不把金標補進候選，也不從 Dev/Test 取得訓練負例。

### K=17～35：原作者的 16 選項上限

Jeff 原生 Head 只有 A～P **16 個選項 slot**。公開推論程式對更多候選採 **balanced grouping + final winners**：

- K=2～16：1 次候選決策 forward pass。
- K=17：分組 9、8；先各選一位優勝者，再選最後一位，約 **3 次 forward passes**。
- K=35：分 12、11、12 三組；各選一位優勝者，再在三位中選最後一位，約 **4 次 forward passes**。

**不能只把 17／35 選項塞進 16 個 slot 然後取錯誤的結果。** 首輪微調先用 2～16 的多種 K 訓練 Head；K=17～35 使用與原作者相同的分組推論做完整 Dev 評估。若要訓練跨組決策或改成 35-slot Head，必須明確標為另一項方法消融，另外驗證選項校準與訓練策略。

## 最小可審核實驗流程

1. 確認模型版本／授權、可存取 H200 的 bf16 訓練環境與顯存，從 Jeff-1.1-Large 的固定 revision 載入模型、Tokenizer、Head；先跑一次純 Jeff Zero-shot Dev 以檢查建模管線。
2. **已完成：** Train 5-fold OOF BM25 候選生成、43,180 筆全部覆蓋與候選資料檢核；三套 Jeff 監督資料共 129,540 筆全部完成逐筆 QA。原始病例有 1 筆臨床文字包含金標字串，跨折完全重複文字 6,054 筆查詢，均保留並揭露。詳見 [訓練資料紀錄](jeff_train_data.md)。
3. 第一輪：Jeff 已發布權重 + 新增 LoRA + 既有 Decision Head，負例採 BM25 硬負例、訓練 K 在 2～16 混合抽樣；訓練時隨機打亂選項位置。
4. Dev 比較全部 K=1～35，儲存每 K 的 **Top-1、Macro-F1、候選 Recall@K、條件準確率、輸入 tokens、推論耗時、峰值顯存**，以及選項數量 >16 時的實際 forward 次數。
5. 消融：未額外微調 Jeff vs Jeff LoRA；Hard Negatives vs Random Negatives（同一訓練預算）；可補充混合負例。決定最佳 K 與 checkpoint 時只看 Dev；官方 Test 5,486 筆最後一次性評估。
6. 確認相較 BM25 Top-1 **79.58%**、Macro-F1 **54.85%** 的差異；除了模型準確率，也比較端到端效能、長尾類別。模型和數據未跑出之前不得宣稱提升。

## 發表與授權

Jeff 專案程式碼標為 Apache-2.0，公開**模型權重**標為 CC BY-NC 4.0；TCM-SD 上游 README 標為 CC BY-NC-SA 4.0。研究使用、公開 checkpoint、商用部署或散布衍生權重之前，需依各自授權、引用與使用條款審查。

**現階段只定稿方法與訓練規則；本文件不代表模型已下載、微調完成或取得 Jeff 的中醫分數。**
