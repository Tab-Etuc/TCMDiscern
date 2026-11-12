# Jeff K=16：H200 前本機檢查結果（2026-11-12）

**結論：TCMDiscern 的 CPU 資料產生、S1～S4 Loss 數值檢查、全量資料 QA、官方 Tokenizer 及評分流程 Smoke Test 已完成；大型模型尚未推論／微調。**

正式依據：[多觀察標註與 Jeff 研究定稿](jeff_ambiguity_final_protocol_2026-11-12.md)。這次只補完原規範的執行前驗收，**不更改 S3 Set Loss 為主要假設、S1/S2/S4 對照組，也不讀 Test 做選模**。

## 已在本機跑完

| 項目 | 真正執行結果 | 彙總資料 |
|---|---|---|
| 官方 Train 與候選 | 43,180 筆、固定 K=16、Group-aware 5-fold OOF | [原 OOF / Group-aware](../experiments/bm25_train_text_grouped_2026-11-11.json) |
| S1 安全單金標 CE | 已有 43,180 筆 `epoch02`，已過濾 797 個確定矛盾負例 | [S1 QA](../experiments/jeff_grouped_filtered_k16_verification_2026-11-12.json) |
| **S2/S3/S4 多正解共用訓練資料** | **新建 43,180 筆**、每題 16 個候選、全部逐筆 QA 通過 | [輸出統計](../experiments/jeff_multipos_k16_epoch03_2026-11-12.json)／[全量 QA](../experiments/jeff_multipos_k16_verified_epoch03_2026-11-12.json) |
| S1～S4 解析梯度 | NumPy 穩定性、有限差分、選項 permutation、單標籤退化均通過 | `src/tcmdiscern/jeff_losses.py`、單元測試 |
| **CPU PyTorch autograd** | **4 種 Loss、共 7 種設定通過**；最大梯度差約 **1.4×10⁻⁷** | [Torch CPU 結果](../experiments/jeff_cpu_autograd_loss_2026-11-12.json) |
| 新訓練題的 Jeff Tokenizer | **43,180/43,180**；中位數 658、最長 3,577 tokens，超過 8,192：**0** | [Token 長度](../experiments/jeff_multipos_k16_tokenizer_2026-11-12.json) |
| Dev 評分流程 Dry Run | 以**模擬輸出**固定選 BM25 第一名，完整 5,486 Dev 精準重現 **79.58% Top-1 / 54.85% Macro-F1** | [評分 QA](../experiments/jeff_k16_eval_dryrun_2026-11-12.json) |
| 正負候選曝光 | 44,004 次正例、646,876 次負例；148 類負例／正例比中位 53.69、最高 864.11 | [曝光報告](../experiments/jeff_multipos_exposure_2026-11-12.json) |

S2（Masked CE）、S3（Set Loss）、S4（Uniform Soft-target CE）**共用相同的 43,180 筆 K=16 題目**，完整原始 Train 保留，其中 **42,360 筆只有 1 個觀察金標、816 筆有 2 個、4 筆有 3 個**。三組在訓練時只切換 Loss；`positive_indices`、`positive_mask`、`target_distribution` 以及原始 `target_index` 均存於模型 `row` **外面**，不進入 Jeff 看到的 Prompt。A～P 的選項打亂後，監督索引與 Mask 同步重排。

### 不應誤讀的地方

- 本機評分通過是 **BM25 rank-1 的模擬 logits**，沒有 Jeff 模型參與。79.58%／54.85% 仍是 **BM25** 成績，沒有新的 Jeff Zero-shot 或 LoRA 準確率。
- CPU PyTorch 使用 **2.9.1+cpu** 測數學 Loss 與 autograd；**尚未**檢查目標 H200 上的 Jeff 原生 Decision Head、PEFT LoRA、BF16 Backbone、最後有效 token、梯度累積與正確 checkpoint。H200 需依模型相容的固定版本另測。
- 使用 `P(x)` 的不同官方標註集合，目前稱**觀察到的標註歧義**；未經中醫師重新審查，不能宣稱全部證型都同時臨床正確。
- 多正解會更換 S1 部分病例的候選組成。因此比較 S1→S3 的差異包含候選與 Loss 效果；**S2／S3／S4 的候選完全相同**，才可以公平比較不同 Loss。

## 本機可重現程式

以下是本次**實際已跑**過的指令。程式使用不覆寫政策，對已存在的 `epoch03`／報表再次執行會拒絕；需要建立新資料排列時，用未使用的 `--epoch`。逐筆病例與答案檔均只保存在 git-ignored `data/`，不提交 GitHub。

```bash
# Train-only Group-aware 5-fold OOF 已事先完成
uv run python -m scripts.export_jeff_multipos --epoch 3
uv run python -m scripts.verify_jeff_multipos --epoch 3

# 使用原版 Jeff pinned tokenizer；沒有下載 31B 權重
# 這一輪已完成，結果保存在 experiments/jeff_multipos_k16_tokenizer_2026-11-12.json

# 在隔離的 CPU Torch 環境測自動微分，不更動專案基礎依賴
uv run --with 'torch==2.9.1+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  python -m scripts.check_jeff_torch_cpu

# 多正解負例曝光分析、評估接線 Dry Run
uv run python -m scripts.audit_jeff_exposure
uv run python -m scripts.check_jeff_eval_path

uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

單獨的本機程式碼入口：

- `src/tcmdiscern/jeff_multi.py`：相同臨床文字的 Train-only 觀察標註集合＋多正解候選組成。
- `src/tcmdiscern/jeff_losses.py`：NumPy 數值／梯度參考；延遲引入 torch 的可微 Loss。
- `src/tcmdiscern/jeff_scoring.py`：真實 Jeff 16 個 logits 的讀取／評分；檢查 case_id、來源候選順序、Top-1、Macro-F1、rescue／broken。
- `scripts/export_jeff_multipos.py`：正解候選、獨立標籤與 epoch seed，防覆寫。
- `scripts/verify_jeff_multipos.py`：43,180 筆逐筆完整性、來源、BM25 負例、Prompt 欄位與機率分布。

## H200 才能完成的阻擋項目

1. **Jeff Zero-shot 真實 Dev 5,486 筆：** 模型與 Decision Head 原生 runner，輸出每筆 16 logits；先驗證讀取、token padding、最後有效 hidden state、step-0 logits parity、每筆輸出是否屬於原本 BM25 候選。
2. **訓練器 Step-0：** 預訓練合併權重＋發布版 FP32 Decision Head、BF16 Backbone、另加 trainable LoRA；先完成反向傳播、非零梯度、顯存峰值、optimizer、checkpoint 存取後結果一致，再全量微調。
3. **公平主實驗：** S1 單金標、S2 Masked、S3 Set、S4 Soft CE，控制相同 43,180 Train 與 token/step/seed 預算；Dev 選模，Test 最後一次正式評估。多 epoch 需動態重排候選及目標，避免反覆把 `epoch03` 當作每輪重排的資料。
4. **模型成效與論文驗收：** Macro-F1、Top-1、148 類支持度、BM25 rescue／broken、同輸入異標分析、Train↔Dev 重複群組分層、配對 group bootstrap、延遲及顯存。若 S3 無實際改善，保留負面結果。

**本機環境：RTX 4060 Laptop 8GB、系統磁碟剩餘約 11GB。** 31B 合併模型權重約 60GB 以上，這台本機不應下載或載入整個 Jeff-1.1-Large；H200 是正式模型驗證／訓練環境。
