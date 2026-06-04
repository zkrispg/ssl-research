# 论文研读：Sound Event Detection and Localization with Distance Estimation

## 标题

Sound Event Detection and Localization with Distance Estimation
（联合距离估计的声音事件检测与定位）

## 作者

Daniel Aleksander Krause, Archontis Politis, Annamaria Mesaros
（Faculty of Information Technology and Communication Sciences, Tampere University, Finland）

通讯：daniel.krause@tuni.fi · archontis.politis@tuni.fi · annamaria.mesaros@tuni.fi

## 时间

- arXiv 首次提交：2024 年 3 月 18 日（v1），v2 于 EUSIPCO 接收后更新
- 会议正式发表：2024 年 8 月 26–30 日，会议论文集

## 来源

- **会议**：32nd European Signal Processing Conference (EUSIPCO 2024)，Lyon, France
- **会场**：Microphone Array Processing and Spatial Audio 分会场（Wednesday, Aug 28, 2024，poster WE1.PA3.9）
- **页码**：pp. 286–290
- **DOI**：10.23919/EUSIPCO63174.2024.10715220
- **arXiv 链接**：[arXiv:2403.11827](https://arxiv.org/abs/2403.11827)
- **本文角色**：作为 DCASE 2024 Challenge Task 3（带距离估计的 SELD）的官方 baseline 设计依据，被 `partha2409/DCASE2024_seld_baseline` 仓库直接引用并实现

## 背景

经典 SELD（Sound Event Localization and Detection）任务由 Adavanne et al.（JSTSP 2018）提出：在多通道音频里联合识别"事件类别 + 方向角 (DOA)"，但**只覆盖二维球面方向，不估距离**。这意味着系统无法回答"声源离听者多远"——而在机器人导航、增强现实、辅助听觉等场景里，距离信息对场景理解至关重要。

距离估计方面，已有工作多集中在双耳（binaural）格式 + 离散类别分桶（< 4 m 的几个段位），与 SELD 长期分立研究。Kushwaha et al.（WASPAA 2023）在 mic-array 上做过单独的距离估计，但**没人把 distance 和完整的 SELD 任务（detection + DOA）做成统一框架训练**。

DCASE 2024 Challenge 把任务推进到 "3D SELD"——同时输出（类别、DOA、距离），这正是本文要解决的设计问题。

## 贡献

1. **提出 Multi-ACCDDOA 表示**：在 Multi-ACCDOA（Shimada 2022）基础上把 3 维 DOA 向量扩成 4 维（x, y, z, distance），让单网络分支直接学到"激活 + 方向 + 距离"。
2. **系统比较两种范式**：单任务 (Multi-ACCDDOA) vs 多任务 (Multi-task with独立 SDE 分支)，配 4 种损失函数（MSE, MAE, MSPE, MAPE），形成完整 6 行对照表。
3. **跨两种音频格式做对比**：Ambisonics（FOA）vs Binaural，证明距离估计比 DOA 估计在双耳格式下退化更小。
4. **奠定 DCASE 2024 baseline**：Multi-ACCDDOA + MSE 被官方仓库选为推荐配置，13.1 % F 20° 成为整个 challenge 的对照基线。
5. **指出 SELD 与 SDE 的损失函数张力**：SED/DOA 适合 MSE，SDE 适合 MAE，单一损失函数无法两端皆优——为后续混合损失研究指明方向。

## 方法

### 特征

| 格式 | 通道 (CH) | 时间帧 (T) | 频率 (F) | Pooling P |
| --- | --- | --- | --- | --- |
| Ambisonics (FOA) | **7** = 4 log-mel + 3 Intensity Vector | 250 | 64 | [4, 4, 2] |
| Binaural | **4** = 2 log-mel mean + sin/cos IPD + ILD | 250 | 512 | [8, 8, 4] |

STFT 用 Hamming 窗，长度 40 ms，50 % overlap，得到 512 频率 bin。
FOA 的 IV (Intensity Vector) 是从 STFT 域计算 3 个空间分量并 mel-pool。
每段输入 250 帧（≈ 5 s），fs = 24 kHz，hop = 20 ms。

### 模型架构（CRNN + MHSA）

```
Input (CH × 250 × F)
   │
   ▼
Conv2D × 3   ─ 128 filter，kernel 3×3，BN，MaxPool 频率维 [4,4,2]，时间维仅第 1 层 pool 5×
   │  → 输出形状 (128, 50, 4)，频率维降到 4
   ▼
BiGRU × 2   ─ hidden 128
   │
   ▼
MHSA × 2    ─ 8 heads（来自 Sudarsanam et al. DCASE 2021）
   │
   ▼
FC head     ─ 128 → C_q
```

### 输出表示与损失

**方案 I — Multi-ACCDDOA（推荐）**

- $y_{nct} = [a_{nct} R_{nct},\ D_{nct}]$，$n=$ track 索引，$c=$ 类别，$t=$ 帧
- 维度：a, D ∈ ℝ^{N×C×T}, R ∈ ℝ^{3×N×C×T}，∥R_{nct}∥ = 1
- N = 3 tracks，C = 13 classes，输出 156 维向量（线性激活）
- 训练用 ADPIT（Auxiliary Duplicating Permutation Invariant Training）
- 损失：13 种 track 排列里取最小（MSE 或 MAE）

**方案 II — Multi-task (MT)**

- 两个分支：分支 1 = ACCDOA (39 维, tanh)，分支 2 = SDE (13 维, ReLU)
- 总损失 = ℒ₁ + ℒ₂，分支 2 可单独换 MSE/MAE/MSPE/MAPE

### 训练设置

- 优化器：Adam
- Epochs：250 with patience 75
- 框架：PyTorch
- 数据：STARSS23（90 train + 78 test 真实片段）+ 1200 个用 FSD50k 合成的 1 分钟 mixture（同最高 polyphony=3）
- Binaural 数据用 spaudiopy 库通过 HRTF MLS decoding 从 Ambisonics 转换得到

## 实验

### 实验设置

- **数据集**：STARSS23（Sony-TAU Realistic Spatial Soundscapes 2023），FOA 4 通道；Binaural 由 FOA 转换得来
- **评估指标**：DCASE 2023 Task 3 metrics（ER、F₁ with ±20° 角度门、DOAE、Recall）+ 距离误差（米，MAE between gt vs pred）
- **匹配**：1 秒段 + Hungarian 算法 + micro-averaging
- **置信区间**：jackknife 估计，α = 0.05

### 对照表（6 行 × 2 数据格式）

每个组合都跑 4 种 distance loss × 2 种范式（MT / Multi-ACCDDOA）。SELD loss 固定 MSE（防止距离值放大对方向训练的影响）。

## 结果

### Table III — Ambisonics 上的全部结果

| 范式 | SELD loss | Dist. loss | ER | **F₁ [%]** | DOA error [°] | Recall [%] | Dist. error [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-task | MSE | MSE | 0.63 | 41.4 | 22.5 | 61.0 | 0.95 |
| Multi-task | MSE | MAE | 0.64 | 43.6 | 21.6 | 41.1 | 0.93 |
| Multi-task | MSE | MSPE | 0.63 | 44.1 | 23.2 | 64.7 | 0.89 |
| Multi-task | MSE | MAPE | 0.65 | 43.5 | 22.0 | 64.5 | 0.88 |
| **Multi-ACCDDOA** | **MSE** | — | 0.65 | **44.2** | 22.9 | **68.4** | 0.92 |
| Multi-ACCDDOA | MAE | — | 0.86 | 21.5 | **17.7** | 19.1 | **0.74** |

### Table IV — Binaural 上的全部结果

| 范式 | SELD loss | Dist. loss | ER | F₁ [%] | DOA error [°] | Recall [%] | Dist. error [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-task | MSE | MSE | 0.82 | 20.0 | 41.1 | 45.6 | 1.02 |
| Multi-task | MSE | MAE | 0.85 | 16.5 | 38.6 | 36.7 | 1.04 |
| Multi-task | MSE | MSPE | 0.85 | 19.3 | 38.9 | 38.9 | 1.01 |
| Multi-task | MSE | MAPE | 0.87 | 18.5 | 38.1 | 42.2 | 0.98 |
| **Multi-ACCDDOA** | **MSE** | — | 0.87 | **21.1** | 39.7 | **48.0** | 0.99 |
| Multi-ACCDDOA | MAE | — | 0.97 | 5.4 | 44.5 | 16.3 | **0.75** |

### 主要发现

1. **Multi-ACCDDOA + MSE 在 SELD 主指标 (F₁, Recall) 上最优**：FOA 上 F₁ = 44.2 %，Recall = 68.4 %，且仍允许同类多源建模——优于所有 MT 方案。
2. **MAE 损失对距离友好但毁掉检测**：FOA 上把距离误差从 0.92 m 降到 0.74 m，但 F₁ 从 44.2 % 跌到 21.5 %。说明距离估计的优化目标和 SELD 的优化目标存在张力。
3. **距离估计在 Binaural 上退化最小**：DOA error 从 22° 飙到 39°（≈ 70 % 退化），但距离误差只从 0.92 m 涨到 0.99 m（≈ 8 % 退化）。说明 binaural 也能做距离估计——为单声道/双耳设备的 SELD 应用打开窗口。
4. **MSPE/MAPE（相对距离损失）改善 distance 但对 SELD 几乎无影响**：在 MT 方案下 distance error 降到 0.88 m，F₁ 仍维持 43.5–44.1 %，整体最均衡的 MT 配置。
5. **官方 challenge baseline 选择**：作者推荐 Multi-ACCDDOA + MSE 作为 challenge 起点；DCASE 2024 仓库（partha2409）实现的就是这一支。

### 与 challenge 仓库 README 数字的关系

仓库 README 里的 STARSS23 dev-test 数字（FOA F 20° = 13.1 %, DOAE_CD = 36.9°, RDE = 0.33）和论文 Table III（F₁ = 44.2 %）看起来差距巨大，原因是：

| 项目 | 论文 Table III | 仓库 README |
| --- | --- | --- |
| 评估粒度 | 1 秒 segment | 帧级 (frame-level) |
| metric 版本 | 2023 metric（仅角度阈值） | 2024 metric（角度 + 距离 RDE < 1.0） |
| 训练数据 | STARSS23 + 1200 synthetic | 仅 STARSS23 finetune from synthetic checkpoint |
| Average | micro | macro |

两组数字评的是同一个模型，只是协议不同。复现时要对齐协议才能比较。

## 评价与对我们工作的启示

**优点**：

- 把 SELD 从 2D 推到 3D 的工作量大但路径清晰；Multi-ACCDDOA 的扩展设计优雅。
- 实验对照足够完整（4 种 dist loss × 2 范式 × 2 格式），结论 robust。
- 直接通过 DCASE 2024 challenge 推动整个领域转向 3D SELD。

**局限**：

- N = 1 没多 seed 配对统计，所有数字是单次训练 + jackknife CI；难以验证 "Multi-ACCDDOA + MSE 比 MT 显著好" 的统计强度。
- Multi-ACCDDOA 的合理性在 SED 主指标上确实最优，但和 MT 方案的差距很小（44.2 % vs 44.1 %）——这点论文没强调。
- 没做容量 ablation，MHSA 块数 / GRU hidden 等都用一组超参跑下来。

**对我们论文的意义**：

1. 这篇是我们 paper 里 *Modern Baseline* 的标准引用；声明"我们复现了 EUSIPCO 2024 baseline 并在其架构上做几何先验消融"是 reviewer 接受的姿态。
2. 它的 Multi-ACCDDOA 表示 + ADPIT loss 是我们要继承的损失——这样我们的 GCA / no_geom / full ablation 就和 modern baseline 在同一损失/输出框架里直接对比。
3. **它的 self-attention 块（来自 Sudarsanam 2021）正是我们 GCA 想替换的对照模块**——把 GCA 与 MHSA 在同一 baseline 上替换/并联，形成"Standard MHSA vs Geometry-Aware Attention"的清晰对比。这是 ICASSP/INTERSPEECH 论文的标准 framing。
4. 论文承认 "SELD 与 SDE 损失张力" 这个 future work 留白——我们的工作不需要解决距离问题，专注于 GCA 是否能改进 SELD 主指标即可。
5. 复现目标：跑出 README 报的 F 20° = 13.1 %，DOAE = 36.9°，RDE = 0.33（在 60 epoch 微调 + finetune from synthetic ckpt 协议下）。我们目前 seed=0 跑到 epoch 0 已 F=2.0 %、DOA error=31°——曲线还在上升期。

## 引用 (BibTeX)

```bibtex
@inproceedings{Krause2024EUSIPCO,
  author    = {Krause, Daniel Aleksander and Politis, Archontis and Mesaros, Annamaria},
  title     = {Sound Event Detection and Localization with Distance Estimation},
  booktitle = {Proceedings of the 32nd European Signal Processing Conference (EUSIPCO)},
  pages     = {286--290},
  year      = {2024},
  address   = {Lyon, France},
  doi       = {10.23919/EUSIPCO63174.2024.10715220},
  note      = {arXiv:2403.11827}
}
```
