# TCMDiscern：IEEE 方法搜尋與聚合假說實測

日期：2026-11-16。完成來源核對、數學檢查與五折 Train OOF CPU 診斷；新語義方法尚未訓練。

**結論：不把現有 RC-BM25 公式升格為 J-BHI 主方法。** Top-3 值得保留為基準，但其改善未證明類別極值偏差是主要原因。本輪增加分數統計與病人／文字群組控制，未得到穩定的額外收益。

下一個最值得驗證的候選是**候選條件化的語義證據似然模型**：先學習病例文字能否區分當前候選證型，再把檢索截尾、類別支持數及病例群組納入機率模型。這是研究設計，尚未證實有效、原創或足以獲 IEEE 接受。若只實作 BM25 分數版，目前結果不支持優先追加運算。

## 1. 範圍與來源

- 僅用官方 TCM-SD 原始病例與標註；不引入外部病例、不要求醫師或新人工標註。原有證型知識及教材可作固定知識。
- 既有病人＋相同文字傳遞分組：Fit 30,226、Cal 6,477，固定自然 BM25 Top-16，不注入正解。
- 本輪只載入官方 Train 病例檔；未評估 Train-Audit、官方 Dev 或 Test。Dev 數字來自既有彙總。
- Cal、Dev 及先前的 Train-Audit 都已有研究層級探索，不能重新稱作全新獨立測試。
- 原始完整 Train 每類 7–7,380 筆；本次實際 Fit 索引每類 **4–5,150** 筆。
- 不修改論文正文，不將新設計填成已完成結果。

核對過 TCMDiscern 的 pooling 原報告／程式、線性分類器結果、BWR 結案、Jeff/SGD 互補診斷、clinical-factor-lattice 完整 pilot、最新 README 及 TCM-SD-only 範圍文件。

## 2. 原公式的三個問題

### 自適應 k 可能嚴格退化為 max

原式：
\[
S_c=\max_k\{\mu_{c,k}-\lambda\widehat\sigma_{c,k}/
\sqrt{N_{\mathrm{eff},c,k}+\epsilon}-\gamma R_{c,k}\}.
\]
若權重非負、\(\lambda,\gamma\ge0\)、單筆變異數設為零，且 \(R_{c,k}\ge R_{c,1}\)，則
\[
\mu_{c,k}-\lambda\widehat\sigma_{c,k}/\sqrt{N_{\mathrm{eff}}+\epsilon}
-\gamma R_{c,k}\le s_{c,1}-\gamma R_{c,1}.
\]
右側恰為 k=1 的值，故 k=1 必達最大（可能並列）。若採不偏樣本變異數，k=1 的變異數則未定義。必須先定義小樣本不確定性和選取懲罰，不能只寫 max over k。

### ESS 不能直接辨識相關病例

\[
(\sum_jw_j)^2/\sum_jw_j^2
\]
只反映權重集中度。三份相同病例若等權，結果仍為 3。

在固定權重、同邊際變異數、已知相關矩陣 \(C\) 的簡化模型，
\[
N_{\mathrm{eff}}=(\mathbf1^\top w)^2/(w^\top Cw).
\]
\(C=I\) 時回到原式；\(C=\mathbf1\mathbf1^\top\) 時為 1。這是既有變異數恆等式，不是新演算法；文字 cosine 也不能直接當作分數相關矩陣。排序選出的 Top-k 本身相依，不能直接由樣本標準差／根號 ESS 得到有效信賴下界。

### 極值偏差尚未被識別為本資料的主要原因

共同連續背景及 i.i.d. 下，最大值 CDF 是 \(F(s)^{n_c}\)。真實資料各類的文字背景、模板、異質性及先驗不同。近鄰不平衡理論不能直接推出 BM25 應按類別大小扣分。

Top-3 改善也可能來自減少錯標或模板偶合的影響。需固定查詢與計分器，對 Train 記憶庫做預先指定的群組數／重複次數干預，才能區分機制；重新估計 IDF 的影響須另作對照。

## 3. 本輪實測

五折 GroupKFold 的每筆 Fit 查詢都排除其病人／文字群組。五折及 Cal 的全部候選順序均與原始 BM25 Top-16 完全一致。原 max、Top-3、Top-5 的 Top-1 和固定 148 類 Macro-F1 均重現。

學習對照使用普通 HistGradientBoostingClassifier：100 iterations、15 leaves、learning rate .07、min_samples_leaf 40、L2 10、early_stopping=False、seed 20261116。參數在看到本輪 Cal 結果前固定。用所有 Fit OOF 訓練、Cal 評估，未用 Cal 擬合。輸入含 Top-1/2/3/5/8 平均、競爭候選差距、病例／群組數、查詢尺度，至多 25 維；不讀語義表示或 Jeff logits。**這是診斷基準，不是新演算法。**

| 方法 | Cal Top-1 % | Macro-F1（148 類）% | 原 Train-tail 11 類 F1 % |
|---|---:|---:|---:|
| BM25 max | 79.111 | 50.644 | 27.273 |
| 固定 Top-3 mean | 82.260 | 52.723 | 31.818 |
| 病人／同文字群組 Top-3 | 82.152 | 52.594 | 31.818 |
| OOF 學習原始分數分布 | 82.384 | 51.412 | 27.273 |
| OOF 學習群組分數分布 | 82.476 | 52.091 | 27.273 |
| OOF 聯合分數分布 | 82.446 | 52.381 | 27.273 |
| 既有平衡 TF-IDF SGD，同 Fit/Cal、Top-16 | 84.298 | 54.796 | — |

- 聯合分數分布 vs Top-3：救回 57、改錯 45，淨多 12 筆；Top-1 **+0.185 pp**，Macro-F1 **-0.342 pp**。
- Cal 病人／文字群組 bootstrap 2,000 次，準確率差異的描述性 95% CI **[-0.138,+0.539] pp**。僅條件於已訓練模型，未校正研究層級多次探索。
- 群組 Top-3 vs 原 Top-3：救回 3、改錯 10。此明確去重控制無額外收益；不代表所有語義近重複／相依模型皆無效。
- 固定 Top-5：81.967%／49.218%。Macro 下降也包含大量 Mid 類，不能全歸於最稀有 11 類。
- 原 pooling 報告按 **Fit 支持數**分組；本輪用完整 Train 固定 **43/94/11** 類，總體指標一致，但分層值不可混用。固定 Tail-11 的 max→Top-3 是 27.273→31.818%，不是原 Fit-tail 的 34.329→34.418%。
- Cal Oracle@16 = 98.286%，僅是候選命中上限。

因此保留 Top-3 與上述控制，不將這輪小幅、犧牲 Macro 的變化包裝成 RC-BM25 成功。

## 4. 最接近的外部文獻

本輪非系統性回顧；下表區分全文方法與僅查到摘要的來源。

| 文獻／查閱範圍 | 對原創性的限制 |
|---|---|
| [Garg et al., Stat 2025](https://onlinelibrary.wiley.com/doi/10.1002/sta4.70110)，出版頁與摘要；全文入口回到摘要 | 不平衡近鄰與條件下 Bayes-risk consistency 已有先例，未核讀證明全文，不能把定理轉用到 BM25。 |
| [Ilse et al., ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html)，PDF 方法 | 可學習加權 MIL pooling 已成熟。 |
| [DSMIL, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf)，公開方法 | 最大 instance 與集合訊號的聯合聚合已存在。 |
| [TransMIL, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)，PDF | 已處理 correlated MIL；考慮 instance 相依性不是空白。 |
| [Arampatzis et al., SIGIR 2009](https://e.humanities.uva.nl/publications/2009/aram_wher09b.pdf)，PDF | 截尾分數混合模型與自適應檢索 cutoff 已有研究。 |
| [Zamani et al., ICTIR 2022](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1452)，PDF | 檢索條件化重排及 cutoff 聯合最佳化已有理論。 |
| [Hashimoto et al., 2026 preprint](https://arxiv.org/html/2609.31712v1)，HTML 方法 | 已對 attention instance 與 adaptive reference 的雙重選取作統計推論；其 Gaussian／已知變異數假設不能直接套 BM25。 |
| [REAR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.321/)，正式摘要與方法節選 | retrieved-document relevance、可靠度頭與抗噪訓練已有先例。 |
| [CHR, 2026 preprint v2](https://arxiv.org/html/2604.04593v2)，HTML 方法 | 目標／混淆診斷的正負檢索及自適應懲罰已有先例。 |
| [SD-MKEK, AI in Medicine 2026](https://www.sciencedirect.com/science/article/pii/S0933365726000485)，摘要及方法概述 | 同 TCM-SD 已有標籤引導對比、知識融合與 cross-attention；未取得完整實作。 |
| [Agent-GNN, Digital Chinese Medicine 2026](https://www.sciencedirect.com/science/article/pii/S2589377726000157)，出版摘要與倉庫既有全文核讀 | 同來源資料的長尾圖模型已存在；清理、標籤合併與 F1 分組未完全對齊，不拿其 Test 與本輪 Cal/Dev 排 SOTA。 |

## 5. 唯一建議的研究候選：候選條件化的語義證據似然

問題改為：**這批支持病例的內容，在真正屬於 c 與誤選 c 的情況下，各有多常見？**

單看 BM25 分數的版本不優先。先加入能判斷病例與候選診斷相容性的表示，再檢驗統計建模是否增加額外價值。

### 最小實作

1. 固定自然 Top-16，每類從不同病人／文字群組取得少量支持病例；保存出處。訓練 query 全用真正 OOF，正式評估仍保留全部官方病例列。
2. 共享小型中文 encoder 編碼 query、支持病例、原有證型定義，產生語義標記 \(z(q,d,c)\)。不為每個尾類建立獨立大型頭。以 Train 證型標籤端到端學習，單筆 support 的可靠性是潛變數，不能當成既有醫學金標。
3. 從 OOF 真候選及**自然 Top-16 錯候選**學正／誤選分布，不只使用簡單隨機負例。尾類參數向跨類共享分布收縮，強度只由 Train 決定。
4. 模型知道支持群組數、可見的排序前幾筆，以及進入 shortlist 的機制。更多抽樣機會不自動等於更多診斷證據。
5. 輸出候選排序／校準結果與病例來源；未提及症狀保留 unknown。無新醫師標註時，不宣稱 clinical certificate 或注意力等於忠實推理。

### 數學起點：標準 likelihood，而非新定理

令 \(H_1:Y=c,\ H_0:Y\ne c\)，\(M_c\) 是支持群組數。每群先以查詢無關規則取固定代表。
觀察按 BM25 由高至低的前 r 個群組
\(E_c=\{(s_{c,j},z_{c,j})\}_{j=1}^r\)。
在**群組層條件 i.i.d.、分數連續**的簡化模型中，\(f_h(s,z)\) 是聯合密度，
\(F_h(s)\) 是分數邊際 CDF：
\[
p_h(E_c\mid M_c)=
\frac{M_c!}{(M_c-r)!}
F_h(s_{c,r})^{M_c-r}
\prod_{j=1}^r f_h(s_{c,j},z_{c,j}).
\]
正／誤選對數似然比：
\[
\Lambda_c=
\sum_{j=1}^r\log\frac{f_1(s_{c,j},z_{c,j})}{f_0(s_{c,j},z_{c,j})}
+(M_c-r)\log\frac{F_1(s_{c,r})}{F_0(s_{c,r})}.
\]
後項計入未進入前 r 的支持群組；階乘在比值中抵消。
這是標準 order-statistic likelihood 的應用，**公式本身不主張原創**。

若還條件化進入 Top-16 事件 \(A:s_{c,1}\ge\tau_c\)，在同一模型下需把密度除以
\(1-F_h(\tau_c)^{M_c}\)，且搭配**條件於 A 的先驗**。
也可直接對完整自然候選集學條件 likelihood。
不能使用已條件化的模型後，再任意疊加第二次 selection penalty。

真實 BM25 有零值與並列，需要離散／混合模型或區間 likelihood。
群組之間仍可能相關，須檢驗殘餘相依，不能直接宣稱精確誤差控制。
只使用單類訊號估計的 one-vs-rest 分數，也不自動構成一致的多類後驗；
正式實作需聯合候選條件化與校準。

**待證實的貢獻**是自然錯候選的語義背景、群組截尾證據及長尾共享參數的聯合估計。
在本輪核讀文獻中未找到完全相同設計；有限搜尋不構成原創性證明。
它與 REAR/CHR 的區別必須靠同訊息、同容量對照建立，不能靠更換名稱。

### 必須成立的性質

- 複製同群同內容病例不應增加證據；此性質只能在群組規則及 tokenizer/IDF/encoder 固定時主張。若重建 IDF，需另處理整條路徑。
- 單筆高品質 support 可以有效，但單筆樣本變異數零不等於確定性。
- 語義表示若沒有額外可辨識訊號，likelihood 不會憑空提高準確率。
- log-loss、class balancing、p-value 都不能直接等同 Macro-F1 最優或診斷機率。

## 6. 投入／停止的具體規則

下列是本研究的投資判準，不是 IEEE 的統一錄用門檻。

1. 同一 encoder／預算比較直接 148 類分類、普通 pairwise reranker、ABMIL/DSMIL、
   Top-3、普通 learned pooling、無截尾、無群組、無共享收縮的版本。
   本輪 HGB 與平衡 TF-IDF SGD 必留。
2. 先量測語義表示的增量，再比較完整 likelihood 與**使用相同語義表示的普通聚合器**。
   若增益全部來自 encoder，不能歸功於統計聚合。
3. 用原 Train 作記憶庫群組抽樣及複製干預；維持 query、標籤不變。
   移除真正有用的獨立病例不必保持預測不變，不能對所有索引變化強加 invariance。
4. 報固定 148 類 Macro-F1、43/94/11 分層、rescue/break、候選覆蓋、group bootstrap、
   多個 seed、完整流程延遲。正式 Test 最後一次使用。
5. 先跑小型 encoder 的受限 pilot。可預先把可重複的 +1–2 pp Macro-F1、
   Top-1 無實質下降、核心消融有效設為擴大投入條件，不能寫成預期成果。
6. Cal 同協定基準為平衡 SGD 84.298/54.796；既有 Dev 強基準
   SGD+Jeff S1 為 85.071/60.960。兩個切分不得互相比高低。
7. 只有 Accuracy 改善而 Macro 下降、與普通語義 reranker 無差異、
   或只能在單一 seed／索引規模有效，就不升級完整 31B 微調。
8. 不設外部病例或醫師審閱為必要條件；單資料集成果只支持 TCM-SD 範圍內的方法效用，
   不宣稱外院／前瞻臨床效度。

## 7. 產物與重現

本機論文倉庫：
- experiments/tcmdiscern/bm25_profile_probe_2026-11-16.json：完整彙總。
- experiments/tcmdiscern/probe_bm25_evidence_profiles.py：相同腳本快照；
  執行時放到 TCMDiscern 的 scripts/，不在論文倉庫直接執行。
- docs/ieee-method-search-2026-11-16.md：本文件。

遠端 TCMDiscern：
- scripts/probe_bm25_evidence_profiles.py
- experiments/bm25_profile_probe_2026-11-16.json
- data/bm25_profile_probe_2026-11-16/：私有特徵快取及逐筆預測，不加入論文倉庫。
- docs/bm25_evidence_method_search_2026-11-16.md

在 TCMDiscern 目錄依序執行：
~~~bash
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py 0
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py 1
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py 2
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py 3
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py 4
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py cal
uv run --no-sync python scripts/probe_bm25_evidence_profiles.py evaluate
uv run --no-sync ruff check scripts/probe_bm25_evidence_profiles.py
~~~

本輪六份特徵建構、學習／評估及 Ruff 檢查均完成。結果檔記錄資料、分割與最終腳本 SHA-256。
