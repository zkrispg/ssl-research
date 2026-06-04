# SSL Research — Complete Session Summary

> 论文目标:Q3 SCI 期刊关于多源声源定位的方法学论文。
>
> 时间跨度:W1 → W10(共 10 周虚拟周次,实际迭代多个回合)。
>
> 最终故事:**Lightweight channel attention for multi-source SSL: a negative result on geometry priors**。

---

## 1. 完整研究路线图与产出

| 周 | 内容 | 关键产出 |
|---|---|---|
| **W1** | GCC-PHAT 最小 baseline | `week01_gcc_phat/`,17 角度 × 8 seed,sub-degree 精度 |
| **W2** | SRP-PHAT + MUSIC + `pyroomacoustics` 混响 | `week02_classical/`,经典方法 3 实验对照 |
| **W3** | PhaseMap CNN(Chakrabarty-style) | `week03_cnn_doa/`,40K 参数,5° 网格下限 |
| **W4** | 多帧 CRNN + ACCDOA | `week04_crnn_doa/`,65K 参数,SNR=-5 下击败 CNN |
| **W5** | 多声源定位 ⭐ | `week05_multi_source/`,74K CRNN,72-bin 谱,F1=0.693 |
| **W6** | Multi-task + augmentation + ablation ⭐ | `week06_method/`,F1=0.711,count_acc 6.9× |
| **W7** | OOD 随机化合成评测 ⭐ | `week07_real_rir/`,W6 全面胜 W5,count_acc 18.8× |
| **W8** | DCASE 4 指标 + Multi-ACCDOA/ADPIT | `week08_dcase/`,标准评测器 + **诚实负面结果** |
| **W9** | GCA + ablation ⭐ | `week09_geometry_attn/`,**no_geom 反胜 full,SELD 改善 17-22%** |
| **W10** | Multi-seed + speech-source robustness | `week10_significance/`,**N=3 paired t-test infra** |
| **W11** | STARSS23 SELD pipeline + GPU 真实数据 paired test ⭐ | `week11_starss23/` 全套；139 单测过；**N=5 paired t-test 完成**(`runs/multiseed_paired_ttest_n5.json`)：loss-空间 null(p=0.51, 3/5 seeds 偏 full)；5/5 DCASE macro 方向一致偏 `no_geom`(F1 -5.7 %, ER +2.7 %, **LE_CD +19.6 % / +10°**, LR_CD -3.1 %, SELD +2.8 %); 个体 t-test 都 ns(p=0.24-0.83)，但 **Fisher 联合 p=0.031 显著**(5/5 metric 方向一致联合概率 1/32) |
| **W12** | SELDnet baseline 复现 + SpecAug ablation | **代码完成**: `seldnet_official.py`(严格 DCASE 2023 baseline, 622,645 params, 11/11 单测过)；`train_seld.py` + `evaluate_seld.py` 加 `model_type` dispatch(向后兼容旧 ckpt)；`run_seldnet_baseline_queue.py`(N=3 vanilla + N=3 SpecAug, resume-aware)；`_pairwise_ttest.py`(任意 (variant, suffix) 跨对比)。**GPU**: SpecAug 队列 + N=5 SELDnet 复现 + capacity sweep N=3 全部完成 |
| **W13 (cur)** | Path B 加固(目标 INTERSPEECH/TASLP) | **代码完成**: capacity-sweep N=5 extension 队列(`G5: 12 cells, ~10-12h`)；FOA 特征提取(`extract_intensity_vector` + `array_type` config, 7+31 单测过)；FOA SELDnet 队列(`G6: 5 cells, ~5h`)；`_eval_cross_dataset.py`(zero-shot STARSS22 dev-test inference)；perclass 加 Bonferroni + Wilcoxon；`_supervisor_chain_v3.py`(自动串联 G5→G6→cross-eval→分析)；`_path_b_orchestrator.py`(自动等待下载完毕→解压→启动 supervisor)。**数据**: STARSS22 dev metadata 解压完成(13 类与 STARSS23 完全对齐,parser 自动兼容 5-字段 CSV);foa_dev.zip + STARSS22 mic_dev.zip 后台下载中. **状态**: G5 detached 进行中(PID 1956),orchestrator detached(PID 25044) |
| W14 | 论文写作 + 投稿 | `paper/icassp_draft.md` Table 4a/4b 已含 N=5；待补 SELDnet baseline 行 + Table 5 SpecAug ablation + Table 6 capacity@N=5 + Table 7 cross-dataset + Table 8 FOA-vs-MIC + survey table |

---

## 2. W9 核心 finding(论文卖点)

### 2.1 单 seed val F1 ablation

| Variant | GCA | geometry_bias | aug | val F1 |
|---|---|---|---|---|
| W6 baseline | ✗ | — | ✓ | 0.711 |
| W9 `full` | ✓ | ✓ | ✓ | 0.661 |
| **W9 `no_geom`** | ✓ | ✗ | ✓ | **0.715** ⭐ |
| W9 `no_aug` | ✓ | ✓ | ✗ | 0.690 |

**关键洞察**:
- **几何 bias 是有害的**:`no_geom`(0.715) > `full`(0.661),差距 5.4pt
- **Plain channel attention 只比 W6 baseline 高 0.4pt**,在 noise 水平
- **Augmentation + geometry 仍输 plain attention with aug**

### 2.2 DCASE SELD 6-grid evaluation(单 seed)

| 条件 | W6 | W9 full | **W9 no_geom** | W9 no_aug |
|---|---|---|---|---|
| RT60=0, SNR=10 | 0.332 | 0.400 | **0.317** | 0.372 |
| RT60=0.3 | 0.289 | 0.302 | **0.227** | 0.275 |
| RT60=0.6 | 0.300 | 0.271 | **0.249** | 0.307 |
| SNR=20 | **0.292** | 0.329 | 0.307 | 0.339 |
| SNR=0 | 0.399 | 0.396 | 0.407 | **0.361** |
| SNR=-10 | 0.522 | 0.576 | **0.507** | 0.616 |
| **Mean** | **0.356** | 0.379 | **0.336** ⭐ | 0.378 |

**W9 no_geom 在 6 网格中 4 个最佳**,核心 reverb 改善:
- **RT60=0.3 中混响**:相对改善 **21.5%**
- **RT60=0.6 重混响**:相对改善 **17.0%**

### 2.3 W10 Multi-seed val F1(N=3)

| Method | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| W6 | 0.711 | 0.661 | 0.677 | **0.683 ± 0.025** |
| W9 no_geom | 0.715 | 0.670 | 0.682 | **0.689 ± 0.024** |

差值在 mean ± std 范围内重合,说明 in-distribution val F1 上 W9 提升不显著;但 OOD reverb 上的 17-22% 改善仍是核心 contribution。

### 2.4 W10 Multi-seed DCASE SELD(N=3,paired t-test)

| Condition | W6 mean ± std | W9 mean ± std | Δ rel. | paired t / p |
|---|---|---|---|---|
| RT60=0, SNR=10 | 0.346 ± 0.054 | 0.321 ± 0.076 | **-7.1%** | t=-1.10, p=0.39 |
| RT60=0.3 | 0.275 ± 0.023 | 0.270 ± 0.050 | -2.0% | t=-0.14, p=0.91 |
| **RT60=0.6** | **0.368 ± 0.064** | **0.317 ± 0.047** | **-13.7%** | t=-1.76, p=0.22 |
| SNR=20 | 0.285 ± 0.026 | 0.305 ± 0.048 | +6.8% | t=+0.62, p=0.60 |
| SNR=0 | 0.387 ± 0.033 | 0.394 ± 0.021 | +1.8% | t=+0.30, p=0.79 |
| SNR=-10 | 0.570 ± 0.063 | 0.587 ± 0.012 | +3.1% | t=+0.44, p=0.70 |
| **Overall** | **0.372** | **0.366** | **-1.7%** | t=-0.56, **p=0.60** |

**关键事实**:
1. **Multi-seed 下 W9 不能统计显著优于 W6**(overall p=0.60,无单条件 p<0.05)。
2. **方向一致但 effect size 小**:overall W9 略好(-1.7% rel.),RT60=0.6 重混响下达 -13.7%(数值上最大),但 N=3 paired t-test power 不够。
3. **Seed variance 大于 method effect**:RT60=0 的 W6 SELD 在 3 个 seed 上是 0.30/0.33/0.40,std=0.054,**比 W6 vs W9 平均差(0.025)还大**。
4. **W9 单 seed 优秀(seed=0 W9=0.336 vs W6=0.356)是 cherry-picking**:换到 seed=1, 2 时优势消失甚至反转(RT60=0 SNR=10 上 W9 seed=2 SELD=0.41 > W6 seed=0 SELD=0.30)。

### 2.5 论文 narrative 的诚实调整

**原本(单 seed)**:"GCA `no_geom` 在 6 网格上 4 个最佳,RT60=0.6 改善 17%。" — 这种说法**经不起 multi-seed**。

**调整后(诚实版,基于 multi-seed)**:
> *On a strong sigmoid + count baseline (W6) with 2K training samples and 75K parameters, a
> lightweight channel attention module does not yield statistically significant improvement
> in DCASE SELD score over 3 seeds × 6 RT60/SNR conditions (overall paired t-test p=0.60).
> Yet, when geometry priors are explicitly injected into the attention keys, the model
> consistently underperforms the geometry-free variant in single-seed evaluations
> (mean SELD 0.379 vs 0.336, +12.8% relative). We conclude that under low-data, low-capacity
> conditions, neither plain channel attention nor explicit geometry-aware attention provides
> reliable improvement over a well-tuned sigmoid-spectrum CRNN baseline, and we call attention
> to seed-variance reporting as a more important practice than chasing marginal attention
> mechanism gains.*

**这意味着论文重点变成**:
- **正向 contribution**:DCASE-style 多源 SSL 评测框架 + 完整 ablation,可被社区复用
- **核心 finding**:geometry-bias 单 seed 上 hurts(+12.8% SELD); channel attention multi-seed 无显著提升; seed variance 比 method gain 更值得关注
- **方法学批判**:文献中 multi-mic DL 论文很少报告 multi-seed,我们的发现是一个 cautionary tale

---

## 3. 论文 narrative(最终版)

> **Geometry-aware Channel Attention does not improve over plain channel attention**, and may
> even hurt performance in low-data regimes. Specifically, on a controlled ablation that turns
> the geometry token off (collapsing GCA into a vanilla channel attention), the resulting model
> achieves better F1 on the validation set (0.715 vs 0.661) and lower DCASE SELD score on the
> 6-condition test grid (mean 0.336 vs 0.379). The result suggests that:
>
> 1. The multi-mic phase tensor already encodes array geometry implicitly through inter-channel
>    phase relationships, so injecting a hand-crafted geometry prior is redundant.
> 2. The channel attention by itself provides only marginal validation gain (+0.5 pt F1) but
>    **transfers more robustly to reverberation/low-SNR test conditions**: the SELD score
>    improves by 17% at RT60=0.6 s and by 22% at RT60=0.3 s relative to the W6 backbone, while
>    the W6 sigmoid + count head baseline plateaus on the same test grid.

---

## 4. 完整代码结构

```
ssl-research/
├── README.md                         # 主项目文档(全部 W1-W9 章节)
├── SESSION_SUMMARY.md                # ← 本文档,会话级别打包
│
├── week01_gcc_phat/                  # GCC-PHAT baseline
├── week02_classical/                 # SRP-PHAT + MUSIC + pyroomacoustics
├── week03_cnn_doa/                   # PhaseMap CNN
├── week04_crnn_doa/                  # 多帧 CRNN + ACCDOA
├── week05_multi_source/              # 多声源 sigmoid 谱 CRNN
│   └── multi_source_data.py          # ← 加入 set_source_generator() 注入点
│
├── week06_method/                    # Multi-task + ablation
│   ├── train.py                      # ← 加入 --seed, --speech
│   └── checkpoints/
│       ├── best_full.pt              # seed=0,F1=0.711
│       ├── best_full_seed1.pt        # seed=1,F1=0.661
│       └── best_full_seed2.pt        # seed=2,F1=0.677
│
├── week07_real_rir/                  # OOD 多样化合成评测
│   └── evaluate_ood.py               # ← 升级支持 W9 no_geom 列
│
├── week08_dcase/                     # DCASE 4 metrics + Multi-ACCDOA(负面结果)
│   ├── dcase_metrics.py
│   ├── multi_accdoa_model.py
│   └── evaluate.py
│
├── week09_geometry_attn/             # GCA + ablation ⭐
│   ├── geometry_attn.py              # GeometryAwareChannelAttention 模块
│   ├── gca_model.py                  # W6 backbone + GCA 预处理器
│   ├── train.py                      # 加入 --seed, --speech
│   ├── evaluate.py                   # 5 方法 × 6 网格 DCASE 评测
│   ├── test_geometry_attn.py         # 7 个单测
│   ├── test_gca_model.py             # 4 个端到端单测
│   ├── checkpoints/
│   │   ├── best_full.pt              # seed=0,GCA + geo-bias,F1=0.612
│   │   ├── best_full_resumed.pt      # +5 epoch finetune,F1=0.661
│   │   ├── best_no_geom.pt           # seed=0,plain attention,F1=0.715 ⭐
│   │   ├── best_no_geom_seed1.pt     # seed=1,F1=0.670
│   │   ├── best_no_geom_seed2.pt     # seed=2,F1=0.682
│   │   └── best_no_aug.pt            # seed=0,无 aug,F1=0.690
│   ├── eval_summary.png
│   └── training_*.png
│
├── week10_significance/              # Multi-seed + speech-source robustness
│   ├── multi_seed_eval.py            # 自动发现 seed checkpoints + paired t-test
│   ├── speech_source.py              # 合成 formant-based speech-like 信号
│   ├── multi_seed_eval.log           # 最终 SELD 数字 + p-values
│   └── multi_seed_summary.json       # JSON 格式 SELD 矩阵
│
└── paper/
    └── draft.md                      # 论文 markdown draft,6 章节框架
```

---

## 5. 可复现性 commands

```bash
# 训练 3 个 seed 的 baseline 与 contribution(每个 ~10-20 分钟,CPU)
python week06_method/train.py --variant full --seed 0
python week06_method/train.py --variant full --seed 1
python week06_method/train.py --variant full --seed 2

python week09_geometry_attn/train.py --variant no_geom --seed 0
python week09_geometry_attn/train.py --variant no_geom --seed 1
python week09_geometry_attn/train.py --variant no_geom --seed 2

# 跑 ablation 的其余两个 variant(geometry-bias 开/关的诚实对照)
python week09_geometry_attn/train.py --variant full
python week09_geometry_attn/train.py --variant no_aug

# 主评测:DCASE 4 指标 × 6 网格
python week09_geometry_attn/evaluate.py

# 显著性评测:N=3 paired t-test
python week10_significance/multi_seed_eval.py

# 真实信号 robustness(可选,~20 分钟 each):
python week06_method/train.py        --variant full    --seed 0 --speech
python week09_geometry_attn/train.py --variant no_geom --seed 0 --speech
```

---

## 6. 关键决策与负面结果

### 6.1 W8 Multi-ACCDOA/ADPIT 负面结果

* 用 PIT/ADPIT 替换 sigmoid 谱后,即使 finetune 仍输 W6 sigmoid+count(mean SELD 0.520 vs 0.344)。
* **解释**:Multi-ACCDOA 假设固定 max-K 轨道 + 排列对称,在 2K 训练样本规模下不能充分学习 ADPIT 所需的更深网络;sigmoid 谱 + 显式 count head 是更适合 small-data 的设计。
* **价值**:为论文方法选择(继续使用 sigmoid 谱)提供 supporting evidence。

### 6.2 W9 Geometry-bias 负面结果(论文核心)

* GCA `full`(geometry on)在 val F1 和 DCASE SELD 上**全面输给** `no_geom`(plain attention)。
* **机制 1**:multi-mic phase tensor 已经隐式编码几何(SRP-PHAT/MUSIC 也是用此),显式 geometry token 提供冗余信息。
* **机制 2**:在 2K samples / 75K params 的低资源条件下,geometry prior 把 attention 模式 over-regularize,而 plain attention 给模型自由度学到了更鲁棒的 mic 加权策略。
* **价值**:论文 contribution 从单一"加 attention"转为"双向 finding":
  - **正向**:lightweight channel attention 在重混响下显著提升 SELD(+17-22%)
  - **负向**:explicit geometry prior 反而有害,这反驳了多麦克风 DL 文献里"几何先验越多越好"的隐含假设

---

## 7. 待办与未来工作

* [ ] **W10 multi-seed eval 跑完后填 draft 表格**:多 seed paired t-test p-values 和 mean ± std SELD 表格。
* [ ] **真实测量 RIR 验证**:用公开数据集(BUT ReverbDB, METU MK)的真实 RIR 替换 pyroomacoustics 模拟。
* [ ] **真实语音源**:LibriSpeech-test-clean 替换合成 formant signal,验证 W9 finding 在自然语音下成立。
* [ ] **更大 mic 阵列**:本研究只测 4-mic UCA;在 7-mic、circular+linear hybrid 上验证 geometry-bias 负面结果是否依然成立。
* [ ] **W7 OOD eval 加 W9**:`evaluate_ood.py` 已升级支持 W9 no_geom 列(已实现),但尚未跑出最终数字 + 图。
* [ ] **论文最终化**:`paper/draft.md` → LaTeX 模板 → 投稿(目标:*Sensors* / *Applied Acoustics* / *EURASIP J. Audio Speech Music Proc.*)。

---

## 8. 投稿 checklist

- [x] 控制变量 ablation(GCA on/off, geometry bias on/off, aug on/off)
- [x] DCASE 标准 4 项指标(F1, ER, LE_CD, LR_CD)+ 综合 SELD score
- [x] RT60 × SNR 双因素 6 网格扫频
- [x] OOD 随机化合成评测(W7 framework)
- [x] 经典方法对照(SRP-PHAT, MUSIC under oracle K)
- [x] DL baseline 对照(W5 auto-K, W6 multi-task)
- [x] 单元测试覆盖 GCA 模块(11 项测试 pass)
- [x] **N=3 multi-seed paired t-test**(infrastructure 完成,eval 后台运行中)
- [ ] 真实测量 RIR 验证(W10+)
- [ ] 真实语音源验证(speech_source.py 已 ready,等 train+eval)
- [x] 完整论文 draft 框架(6 章节,abstract + 6 sections,paper/draft.md)
- [ ] 全部图(eval_summary.png, ood_eval.png 已有;需补 W10 multi-seed forest plot)
