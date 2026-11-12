# Jeff-1.1-Large：H200 執行手冊與安全驗收

**2026-11-12｜H200 相關程式已準備，尚未在 H200 實際跑模型。** 主訓練方法仍依 [標註歧義研究定稿](jeff_ambiguity_final_protocol_2026-11-12.md)：S1 安全單金標、S2 Masked CE、S3 Set Loss（主假說）、S4 Soft-target CE。所有病例、官方 Train/Dev/Test 切分、K=16、官方 Jeff/SemIf Prompt 均不改。

## 先決條件與真正的 GPU 入口

本機 RTX 4060 Laptop 8GB 僅能跑 CPU／小模型測試。**本專案沒有連線 H200 或實測 31B Step-0**；任何從本手冊取得的模型分數與顯存結果，都必須以 H200 的真實執行紀錄為準。

- 官方 Jeff 上游 **[v1.3.0](https://github.com/j-geuter/Jeff/tree/v1.3.0)**；避免直接用 main，因為模型卡對應 v1.3.0 且上游會更新程式。
- 模型 **[jgeuter/Jeff-1.1-Large](https://huggingface.co/jgeuter/Jeff-1.1-Large)**，指定 release `970c21af246a44c5527f7109fca129ab6015c941`，原始合併 Gemma 4 31B-it BF16 權重與 `head.safetensors`，已發布 LoRA 已**合併**至 Backbone，不再額外注入 release 的 `adapter/`。
- Jeff 官方 v1.3.0 的測試依賴：Python 3.12、torch 2.13.0/CUDA 13.0、transformers 5.17.0、safetensors 0.8.0、huggingface_hub 1.32.0、accelerate 1.15.0。PEFT **0.21.1** 已在本機 CPU 小模型執行四種 Loss、七種 LoRA 投影層、Optimizer 與 Checkpoint 恢復驗證；**CUDA 版本相容性仍須在 H200 實測，正式實驗全程釘選同一版本**。
- H200 需要足夠顯存與磁碟（原版 BF16 權重約 62 GB）；**單次 GPU 推論可用不等於完整微調可用**，實際 Forward＋Backward／Optimizer 佔用需在 Step-0 測量。
- 原始資料、逐筆訓練題及預測檔均在 `data/`（Git 忽略），必須透過受控通道提供到 H200，公開 GitHub 只有程式與 aggregate 統計。檔案校驗用具體列數、欄位及分布，不引入自製 SHA 簽章／版本驗證。

依官方 repo 的 requirements 在 H200 的獨立環境安裝：

```bash
git clone --branch v1.3.0 https://github.com/j-geuter/Jeff /opt/Jeff
# 以下在已建立的 Python 3.12 虛擬環境中，沿用官方 CUDA wheel 指引
python -m pip install -r /opt/Jeff/requirements.txt
python -m pip install -e /opt/Jeff 'peft==0.21.1'
python -m pip install -e /path/to/TCMDiscern
hf download jgeuter/Jeff-1.1-Large \
  --revision 970c21af246a44c5527f7109fca129ab6015c941 \
  --local-dir /models/Jeff-1.1-Large
```

上面為**待在 H200 執行的指令**，目前尚未執行，也沒有對模型權重產生任何訓練。

## 1. 執行順序：先官方 Zero-shot

```bash
cd /path/to/TCMDiscern
python scripts/preflight_jeff_training.py --model-dir /models/Jeff-1.1-Large

# 只跑兩筆官方 Dev，檢查原始 Model/Head/Prompt/Logits 的真正 GPU 路徑
python -m scripts.run_jeff_zeroshot \
  --model-dir /models/Jeff-1.1-Large --limit 2

# 只有小型測試成功，再完整評估 5,486 筆 Dev
python -m scripts.run_jeff_zeroshot \
  --model-dir /models/Jeff-1.1-Large
```

- 使用 Jeff 官方 `JeffModel(..., reasoning=False, long_input="refuse")` 的 `decider.decide(row)`；直接讀取原始 16 個 logits，**不改模型，不重複打開已合併 Adapter**。
- 確認 `data/jeff/dev_k16.jsonl` 5,486 筆與原始 BM25 候選 **ID／順序完全相同**，不補 Dev 金標，任何 Prompt 截斷立即報錯。
- 輸出 `data/jeff/dev_zero_shot_k16_full.jsonl`（私有、不提交 Git），完成後再根據官方金標產生 `experiments/jeff_zero_shot_k16_dev.json` 的聚合 Accuracy、Macro-F1、BM25 rescued/broken、推論秒數。
- `--limit 2` 只做環境 Smoke Test，不會創建正式模型成績。輸出路徑採不覆寫政策。

## 2. 執行順序：Head＋LoRA 真正 GPU Step-0

```bash
# 只做原模型 vs 加入新 LoRA 的輸出一致性檢查，不更新模型
python -m scripts.train_jeff_lora \
  --model-dir /models/Jeff-1.1-Large \
  --method s3

# 明確要求才做「一筆 Train、一次更新、存檔、重載」的最低限度訓練測試
python -m scripts.train_jeff_lora \
  --model-dir /models/Jeff-1.1-Large \
  --method s3 --train-step
```

- 已實作：官方 `Decider.hidden` 取得最後有效 Token、發布版 FP32 `head.delta`、凍結原 31B 權重；新增 LoRA 預設 **rank=32／alpha=64**，作用於 q/k/v/o/gate/up/down **七種原作者目標模組**。要求注入前後 16 個原始 logit 差異不超過 **1e-3**。
- `--train-step` 才執行**一個** Update，測 LoRA 與 Head 均得到有效梯度、Gradient Checkpointing、有限 Loss、Optimizer、Adapter/Head Checkpoint 重載。新多步入口 `run_jeff_finetune.py` 同樣會先執行原版 Step-0 parity，任何失敗即終止。
- 此程式只驗收模型輸出與最小梯度，不會自動完整訓練 43,180 筆，不能當作 S3 已完成實驗。
- 真正測過的是 [CPU 小模型全部七種 LoRA 投影](../experiments/jeff_toy_peft_all_projections_2026-11-12.json)及 [Optimizer／Scheduler／RNG 無損續訓](../experiments/jeff_cpu_resume_2026-11-12.json)；**Jeff-1.1-Large 在 H200 的實際一致性仍待驗證**。

## 3. 完整 LoRA 訓練前的剩餘關卡

`train_jeff_lora.py` 為受控 Step-0＋單次更新檢查；`run_jeff_finetune.py` 為多步訓練入口，載入完整 43,180 筆 Train、每 epoch 打亂病例及選項並同步重排全部標籤，僅更新 LoRA 與已發布 Head correction。**預設只跑 1 個 Optimizer update；H200 尚未實測。** 現已實作每個完整更新邊界的 Snapshot：儲存 Adapter、Head、Optimizer、Scheduler、Torch RNG 和下個 Train 游標，並可用 `--resume` 接續。最後不足累積批次的病例按實際批大小縮放 Loss，確保不漏資料。已用真正 CPU PEFT 執行中斷→恢復，第二次更新的 16 個 logits **差異為 0**。

已補好 `src/tcmdiscern/jeff_epoch.py`：每個 epoch 重新打亂選項並**同步重排**原金標、觀察標註集合、Mask、Soft-target 機率；不修改初始訓練檔且不讀 Dev/Test 答案。已在 CPU 以 **S1＋多正解各 43,180 題×兩輪**實測共 **172,720 題全部通過**（[報表](../experiments/jeff_full_epoch_shuffle_2026-11-12.json)），正式訓練迴圈也已呼叫相同函式。

### 真正多步訓練與微調後 Dev

~~~bash
# 預設僅執行一個 optimizer step，獨立 run-id
python -m scripts.run_jeff_finetune \
  --model-dir /models/Jeff-1.1-Large \
  --method s3 --run-id s3-step1 --steps 1 --epochs 1 --accumulation 1

# 400 updates 僅是示例；H200 Step-0 通過後再決定實際預算
python -m scripts.run_jeff_finetune \
  --model-dir /models/Jeff-1.1-Large \
  --method s3 --run-id s3-seed1-steps400 \
  --steps 400 --epochs 1 --accumulation 8 --seed 20261112 \
  --checkpoint-every 100

# 400 update 的續訓示例：必須重複指定完全相同設定與 run-id
python -m scripts.run_jeff_finetune \
  --model-dir /models/Jeff-1.1-Large \
  --method s3 --run-id s3-seed1-steps400 \
  --steps 400 --epochs 1 --accumulation 8 --seed 20261112 \
  --checkpoint-every 100 \
  --resume data/runs/s3-seed1-steps400/resume-0000100

# 從新建的 JeffModel 載入 adapter/Head，先比對合成 Probe 再評估完整 Dev
python -m scripts.run_jeff_zeroshot \
  --model-dir /models/Jeff-1.1-Large \
  --checkpoint data/runs/s3-step1/resume-0000001 \
  --run-id s3-step1-dev --limit 2
~~~

**注意：400 updates × accumulation 8 只會處理 3,200 筆當輪 Train 病例，是小規模實測，不得聲稱完整訓練 43,180 筆。** 完整一個 epoch 並使用 accumulation 8 需 `ceil(43180/8)=5398` 個 updates；實際步數與 epoch／Seed 必須在各模型間一致。S1 使用單標籤安全資料，S2/S3/S4 共用多答案資料。訓練輸出在被 Git 忽略的 `data/runs/`；包括 Adapter、Head、Optimizer、Scheduler、Torch RNG 和合成驗證 logits，無完整 31B 權重備份。正式 Dev 評分使用原始 BM25 前 16、不補金標，對照 Top-1、Macro-F1、rescue/broken；目前沒有 H200 真實結果。

**額外修正：** Checkpoint Manifest 的原版模型欄位統一為 `model_revision`，與可續訓檔的實際寫入格式一致；新評分端在載入模型前核對全量 Train、S1～S4 方法與正確的資料來源，避免正確訓練的 Adapter 因欄位不同而被拒絕。

**停止條件：** 若發布版 Head 缺檔、模型載入不完整、Jeff v1.3.0 parity 未通過、GPU OOM、LoRA／Head 無梯度、Checkpoints 重載結果不同，立即停止；不能宣稱完成領域微調或使用現有 BM25 79.58% 充當 Jeff 成績。
