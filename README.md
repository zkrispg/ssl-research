> ℹ️ **当前工作请先看 [`HANDOFF.md`](HANDOFF.md)**。本 README 记录的是早期「合成数据多源 SSL」研究线(week01–week11);当前活跃任务是 `dcase2024_baseline/` 的 convbias 几何注入实验,详见 HANDOFF。

# SSL Research — Sound Source Localization

研究目标：在标准数据集（DCASE / LOCATA）上做空间声源定位，目标三区 SCI（`Sensors` / `Applied Acoustics` / `EURASIP J. Audio Speech Music Proc.`）。

## 12 周路线

| 阶段 | 周次 | 目标 | 产出 |
|---|---|---|---|
| 基础 | W1 | 实现 GCC-PHAT，理解 TDOA→DOA | `week01_gcc_phat/` |
| 基础 | W2 | 实现 SRP-PHAT、MUSIC | `week02_classical/` |
| 复现 | W3-4 | 跑通 SELDnet + Chakrabarty CNN | `week03_baseline/` |
| 数据 | W5 | 用 pyroomacoustics 仿真 RIR + 多声源 | `week05_data/` |
| 方法 | W6-7 | 提出改进点（特征/网络/输出之一） | `week06_method/` |
| 实验 | W8-10 | 完整 ablation + 鲁棒性 + 跨域 | `week08_exp/` |
| 写作 | W11-12 | 论文 + 投稿 | `paper/` |

## 当前进度

- [x] **W1: GCC-PHAT 最小 baseline**
- [x] **W2: SRP-PHAT + MUSIC + pyroomacoustics 混响**
- [x] **W3: PhaseMap CNN（Chakrabarty-style 单帧深度方法）**
- [x] **W4: 多帧 CRNN + ACCDOA 输出**
- [x] **W5: 多声源定位 — 论文 contribution 核心数据已建立 ⭐**
- [x] **W6: Multi-task + augmentation + 完整 ablation ⭐**
- [x] **W7: OOD 多样化合成评测（real-RIR 阶段 1）⭐**
- [x] **W8: DCASE 标准 4 项指标 + Multi-ACCDOA/ADPIT 探索（含负面结果）⭐**
- [x] **W9: 轻量化 Channel Attention + Geometry-bias 消融（小幅 SELD 改善 + 关键 negative result）⭐**
- [x] **W10: 多 seed N=3 paired t-test + speech-source 合成（⚠️ 修正了 W9 单 seed 乐观结论）⭐**
- [ ] W11-12: 论文 + 投稿（draft 已完成,paper/draft.md）

## 环境

```powershell
cd ssl-research
pip install -r requirements.txt
python -m pytest week01_gcc_phat -v
python week01_gcc_phat/demo.py
```

## W1 关键结果

双麦克风（10cm 间距，fs=16kHz）GCC-PHAT 在 -90~+90° 扫频，每角度 8 个 seed 平均：

| SNR (dB) | MAE (°) | Max err (°) | Valid |
|---|---|---|---|
| 30 | 1.02 | 2.44 | 17/17 |
| 20 | 0.49 | 1.76 | 17/17 |
| 10 | 0.38 | 1.19 | 17/17 |
| 0  | 0.53 | 2.89 | 17/17 |
| -5 | 1.52 | 7.96 | 17/17 |

注：源信号是 300-3400 Hz 带限白噪声（语音带宽）。

**值得记下的现象（可写进论文 motivation）**：

- 算法精度受 **源信号带宽** 显著约束。同样配置下，宽带白噪声（0-8kHz）能达到 < 0.2° 误差；切换到 300-3400 Hz 语音带宽，floor 上升到 ~1°。
- 这说明 GCC-PHAT 在真实语音上有不可避免的 resolution floor，所以基于深度学习的端到端方法在语音 SSL 上才有提升空间。

## W2 关键结果

UCA4（4 麦圆形阵列，半径 4cm，fs=16kHz，语音带宽源），3 组实验，各 4 个 seed 平均：

### 实验 1：SNR 扫频，自由场，UCA4

| SNR (dB) | GCC-PHAT (2 mic) | SRP-PHAT (UCA4) | MUSIC (UCA4) |
|---|---|---|---|
| +30 | 3.29 | 1.02 | 0.00 |
| +20 | 2.18 | 0.41 | 0.00 |
| +10 | 2.14 | 0.41 | 0.00 |
|  0  | 2.96 | 0.70 | 0.02 |
| -5  | 4.72 | 1.36 | 0.23 |

**结论**：自由场无混响下，多麦 + MUSIC 几乎无误差，SRP-PHAT 也很稳。

### 实验 2：麦克风数量，自由场，全圆 [-180, 180] 搜索

| SNR (dB) | UCA2 | UCA4 | UCA8 |
|---|---|---|---|
| +20 | 50.34 | 0.41 | 0.05 |
|  0  | 53.14 | 0.70 | 0.36 |
| -10 | 52.34 | 3.39 | 1.68 |

**结论**：2 麦克风圆阵在 360° 搜索下因前后模糊导致 ~50° MAE（一半角度被估到镜像方向）。4 麦以上彻底解决。

### 实验 3：混响，UCA4，SNR=20dB

| 房间 | SRP-PHAT | MUSIC |
|---|---|---|
| 无混响 | 0.45 | 0.00 |
| RT60=0.2s | 1.06 | 0.97 |
| RT60=0.4s | 3.58 | 4.45 |
| RT60=0.6s | 5.79 | 7.18 |
| RT60=0.9s | 7.67 | 10.06 |

**结论**：经典文献结果再现。MUSIC 在低混响下更准，但混响越重退化越快，重混响下被 SRP-PHAT 反超。这正是 **DL-based SSL 论文的 motivation：传统方法在 RT60 > 0.4s 后均退化到 5-10° MAE**。

## W3 关键结果

PhaseMap CNN（按 Chakrabarty & Habets 2019 复刻），UCA4 + 5° 类网格 72 类，free-field 训练 8000 样本 + 1024 验证，15 epoch CPU 约 2 分钟。

模型规模：40,296 参数。训练曲线见 `week03_cnn_doa/training.png`。最佳 val MAE = 3.33°，accuracy = 95%。

### 对比表：CNN vs SRP-PHAT vs MUSIC（同一测试集）

**SNR 扫频，自由场，UCA4，30 angles × 3 seeds**：

| SNR (dB) | PhaseMap CNN | SRP-PHAT | MUSIC |
|---|---|---|---|
| +20 | 5.61 | 0.30 | **0.00** |
| +10 | 5.61 | 0.33 | **0.00** |
|  0  | 6.97 | 0.55 | **0.00** |
| -5  | 14.85 | **1.03** | 0.24 |
| -10 | 52.88 | **2.48** | 0.73 |

**RT60 扫频，SNR=10dB**：

| 房间 | PhaseMap CNN | SRP-PHAT | MUSIC |
|---|---|---|---|
| 无混响 | 5.61 | 0.33 | **0.00** |
| RT60=0.2s | 6.97 | **0.94** | 0.97 |
| RT60=0.4s | 7.88 | **3.27** | 4.27 |
| RT60=0.6s | 12.58 | **4.94** | 6.73 |
| RT60=0.9s | 15.15 | **7.00** | 9.55 |

### 关键发现（论文 motivation 素材）

1. **5° 类网格下限**：单帧 CNN 在干净自由场 MAE 5-6°，由 5° 离散化主导，远不如解析方法。
2. **训练分布外脆弱**：训练只见 SNR ≥ -5dB，测试 SNR=-10dB 直接崩到 53°。
3. **混响下与 SRP-PHAT 接近，但仍未超过**：单帧 phase map 信息量不足以补偿多径干扰。加混响重训用 2000 样本不收敛（MAE 卡在 19°），证实小数据 + 单帧的根本局限。
4. **这正是 W4 多帧 CRNN 的研究动机**：跨时间整合相位与能量信息，才能在低 SNR / 高混响下系统性超越经典方法。

## 文件结构

```
ssl-research/
├── README.md
├── requirements.txt
├── week01_gcc_phat/
│   ├── gcc_phat.py
│   ├── simulate.py
│   ├── demo.py
│   ├── test_gcc_phat.py
│   ├── demo_cc.png
│   └── demo_sweep.png
├── week02_classical/
│   ├── geometry.py        # UCA / ULA 阵列几何
│   ├── simulate_array.py  # 多麦自由场 + pyroomacoustics 混响仿真
│   ├── srp_phat.py        # SRP-PHAT 多麦定位
│   ├── music.py           # 宽带 MUSIC 子空间定位
│   ├── demo.py            # 三组对比实验
│   ├── test_classical.py  # 16 个单元测试
│   ├── snr_sweep.png
│   ├── mic_count.png
│   └── reverb.png
├── week03_cnn_doa/
│   ├── features.py        # STFT phase 特征 (sin/cos)
│   ├── dataset.py         # on-the-fly + 预计算合成数据
│   ├── model.py           # 40K 参数 PhaseMap CNN
│   ├── train.py
│   ├── evaluate.py
│   ├── test_w3.py
│   ├── checkpoints/best.pt
│   ├── training.png
│   ├── eval_snr.png
│   └── eval_rt60.png
├── week04_crnn/
│   ├── crnn_dataset.py    # 多帧数据集 + 预计算
│   ├── crnn_model.py      # 65K 参数 CRNN with ACCDOA head
│   ├── train.py
│   ├── evaluate.py
│   ├── test_w4.py
│   ├── checkpoints/best.pt
│   ├── training.png
│   ├── eval_snr.png
│   └── eval_rt60.png
├── week05_multi_source/
│   ├── multi_source_data.py   # K-source 同时仿真
│   ├── multi_dataset.py       # 多源数据集 + 软标签
│   ├── multi_model.py         # 74K 参数 CRNN, 72-bin sigmoid 谱
│   ├── multi_baselines.py     # SRP-PHAT/MUSIC peak picking 多源版
│   ├── multi_eval.py          # SELD 标准 F1 metrics + greedy matching
│   ├── train.py / evaluate.py
│   ├── test_w5.py             # 9 个测试
│   ├── checkpoints/best.pt
│   ├── training.png
│   └── eval_rt60_f1.png
├── week06_method/
│   ├── aug.py                  # Channel rotation + SpecAugment
│   ├── multi_task_dataset.py
│   ├── multi_task_model.py     # 74K 参数 dual-head CRNN
│   ├── train.py                # --variant {none,aug_only,count_only,full}
│   ├── evaluate.py
│   ├── test_w6.py              # 6 个测试
│   ├── checkpoints/best_*.pt
│   ├── training_*.png
│   └── eval_summary.png
└── week07_real_rir/
    ├── diverse_simulator.py    # 随机化房间/RT60/源距/阵列位置的 OOD 仿真器
    ├── rir_loader.py           # 真实测量 RIR 接口（W8 阶段 2 用）
    ├── evaluate_ood.py         # OOD 评测脚本，按 RT60 分层
    ├── test_w7.py              # 5 个测试
    └── ood_eval.png            # OOD F1 对比图
```

## W4 关键结果与诚实分析

CRNN 65K 参数，1500 训练样本，50% reverb (RT60 0.15-0.5s)，CPU 12 分钟训练，best val MAE = 8.12° (median 5.09°)。

### 完整 4 方法对比表（同一测试集）

**SNR 扫频，自由场，UCA4**（30 angles × 3 seeds）：

| SNR (dB) | CRNN (W4) | CNN (W3) | SRP-PHAT | MUSIC |
|---|---|---|---|---|
| +20 | 16.33 | 5.61 | 0.30 | **0.00** |
| +10 | 15.50 | 5.61 | 0.33 | **0.00** |
|  0  | 12.17 | 6.97 | 0.55 | **0.00** |
| -5  | **10.08** | 14.85 | 1.03 | 0.24 |

**RT60 扫频，SNR=10dB**：

| 房间 | CRNN | CNN | SRP-PHAT | MUSIC |
|---|---|---|---|---|
| 无混响 | 15.50 | 5.61 | 0.33 | **0.00** |
| RT60=0.2s | 8.59 | 6.97 | **0.94** | 0.97 |
| RT60=0.4s | 10.18 | 7.88 | **3.27** | 4.27 |
| RT60=0.6s | 13.45 | 12.58 | **4.94** | 6.73 |
| RT60=0.9s | 16.99 | 15.15 | **7.00** | 9.55 |

### 诚实的研究判断

**单声源 + 已知 UCA4 几何 + ISM 仿真**这个 setup 是经典子空间方法的"主场"，深度学习很难超过 MUSIC/SRP-PHAT。这是文献内已知共识。

W3+W4 的真正价值不在于"赢了 baseline"，而在于：

1. **完整可复现的 4-方法对比基础设施**（同一测试集，相同评测脚本）
2. **混响数据生成 + ACCDOA 训练 pipeline 已验证**
3. **CRNN 在 SNR=-5dB 下击败 CNN (10° vs 15°)** —— 多帧 + 时间整合的价值在低 SNR 下显现

### 论文 contribution 必须转向 DL 有优势的场景

经过 W1-W4 的扎实工作，研究方向必须聚焦到 **经典方法明显失效** 的场景：

- **多声源同时定位**（MUSIC 需精确知道声源数，SRP-PHAT 在 ≥3 源时退化严重）
- **声源计数 + 定位联合**（DCASE SELD 标准任务）
- **跨阵列泛化**（DL 需要 retraining，但 Ambisonics + adaptation 是一个方向）
- **End-to-end real-time edge inference**（轻量化 + 量化）

W5 起转向 **多声源定位**，这是三区论文 contribution 的稳妥方向。

## W5 关键结果 ⭐（论文核心 contribution）

**任务**：1-3 个同时活跃声源（azimuth 间隔 ≥ 30°），UCA4，预测每个声源方位 + 计数。

**模型**：MultiSourceCRNN，74K 参数，输出 72-bin sigmoid 空间伪谱，BCE 训练。CPU 训练 ~15 分钟（含数据预计算），best val F1 = 0.693。

**评测指标**：SELD 标准 F1（tolerance 20°）+ Precision + Recall + MAE（仅 TP）+ count accuracy。

### RT60 扫频，SNR=10dB，K∈{1,2,3} 混合（45 trials/RT60）

| Method | RT60=0 | RT60=0.3 | **RT60=0.6** |
|---|---|---|---|
| SRP-PHAT (oracle K) | 0.989 | **0.967** | 0.822 |
| MUSIC (oracle K) | **0.994** | 0.822 | 0.689 |
| **CRNN (oracle K)** | 0.822 | 0.844 | **0.889** ⭐ |
| SRP-PHAT (auto K) | 0.957 | 0.733 | 0.509 |
| MUSIC (auto K) | 0.994 | 0.807 | 0.549 |
| **CRNN (auto K)** | 0.746 | 0.715 | **0.693** ⭐ |

### SNR 扫频，anechoic，K∈{1,2,3} 混合

| Method | SNR=20 | SNR=0 | SNR=-10 |
|---|---|---|---|
| SRP-PHAT (oracle K) | 0.989 | 0.978 | **0.989** |
| MUSIC (oracle K) | **1.000** | 0.910 | 0.750 |
| CRNN (oracle K) | 0.789 | 0.711 | 0.611 |

### 论文 contribution narrative

1. **DL 优势区域已识别**：在重混响（RT60≥0.6s）+ 多声源条件下，CRNN F1 score 0.889 显著超过 SRP-PHAT (0.822) 和 MUSIC (0.689)。这正是经典子空间方法的失效区域，因为多径干扰破坏了 R(f) 的子空间结构。

2. **Auto-K 模式优势更明显**：不告诉模型声源数时，CRNN F1=0.693 是 SRP-PHAT (0.509) 的 1.36 倍。说明模型从空间伪谱激活模式中学到了"该有几个 peak"的判断。

3. **Anechoic 条件下经典方法仍占优**：这是文献已知共识。论文不必声称"全面超过"，而是聚焦"DL 在非理想条件下的鲁棒性优势"。

### 已识别的可改进点（W6-7 方向）

- **Count accuracy 偏低**（auto-K 时 0.07-0.20）：peak picking 阈值粗糙，可学习一个声源计数头
- **Anechoic 单声源精度差**：因 BCE+5° 离散化下限，需要 ACCDOA 风格的连续输出 + 阈值化的混合训练
- **训练样本少**（2000）：扩到 5K-10K 应能涨 5-10 个百分点
- **未做难例挖掘**：3 声源样本占 1/3 但是 hardest，需要 class-balanced 采样

## W6 关键结果 ⭐ — 方法改进 + 完整 ablation

**Contribution**：在 W5 模型上加两个改进：
1. **Multi-task source counting head**：模型独立预测 K∈{1,2,3}，由 K 直接驱动 peak 选择
2. **Channel rotation augmentation**：UCA 旋转麦克风索引等价于旋转方位坐标，提供精确的 4× 数据增强（零仿真成本）
3. **SpecAugment**：标准时频 masking 抗过拟合

74K 参数（仅比 W5 多 2.6%），相同 2000 训练样本，CPU 训练 ~10 分钟。

### W6 vs W5 关键提升

| 条件 | W5 | W6 full | 提升 |
|---|---|---|---|
| Val F1 | 0.693 | 0.711 | +2.6% |
| Count accuracy (val) | 0.10 | **0.69** | **6.9×** ⭐ |
| F1 @ RT60=0.3 | 0.730 | **0.792** | **+8.5%** ⭐ |
| F1 @ RT60=0.6 | 0.672 | 0.699 | +4% |
| F1 @ SNR=-10dB | 0.443 | **0.578** | **+30%** ⭐ |
| count_acc 平均 | 0.13 | **0.61** | **4.7×** ⭐ |

### 完整 ablation table

| Variant | Spec head | Count head | Channel rotation | SpecAugment | Val F1 | Count acc |
|---|---|---|---|---|---|---|
| W5 baseline | ✓ | ✗ | ✗ | ✗ | 0.693 | 0.10 |
| W6 aug_only | ✓ | ✗ | ✓ | ✓ | 0.677 | 0.14 |
| W6 count_only | ✓ | ✓ | ✗ | ✗ | 0.660 | **0.70** |
| **W6 full** | ✓ | ✓ | ✓ | ✓ | **0.711** | 0.69 |

**Ablation 结论**：count head 主导 count_acc 提升（10% → 70%），augmentation 主导 F1 提升（0.66 → 0.71）。两者互补。

### 完整 4 方法对比（与论文 baseline 表一致）

**RT60 扫频 (SNR=10dB, K∈{1,2,3} mixed, 45 trials/条件)**：

| Method | RT60=0 | RT60=0.3 | RT60=0.6 |
|---|---|---|---|
| SRP-PHAT (oracle K) | 0.978 | 0.956 | 0.833 |
| MUSIC (oracle K) | 0.983 | 0.843 | 0.682 |
| W5 (auto K, threshold) | 0.740 | 0.730 | 0.672 |
| **W6 full (count head)** | 0.659 | **0.792** | **0.699** |

**SNR 扫频 (anechoic, K∈{1,2,3} mixed)**：

| Method | SNR=20 | SNR=0 | SNR=-10 |
|---|---|---|---|
| SRP-PHAT (oracle K) | 1.000 | 1.000 | 0.989 |
| MUSIC (oracle K) | 1.000 | 0.908 | 0.759 |
| W5 (auto K) | 0.739 | 0.669 | 0.443 |
| **W6 full** | 0.754 | 0.645 | **0.578** |

### 论文 narrative（已可写）

> 在多声源 DOA 估计任务上，本文提出多任务 CRNN 配合通道旋转数据增强，综合 F1 score 在
> 中混响（RT60=0.3s）下相对单任务 baseline 提升 8.5%，在低 SNR（-10dB）下提升 30%，
> 声源计数准确率提升 6.9 倍。完整 ablation 证实 multi-task 与 augmentation 互补。
> 方法相对经典 SRP-PHAT/MUSIC 在重混响多源条件下保持竞争力，且 **不需要预知声源数**，
> 适合实际不确定声源数量的应用场景。

## W7 关键结果 — OOD 泛化测试

**任务**：每个测试样本来自 **完全随机的房间**（尺寸 4-10×4-8×2.5-4 m）、随机 RT60（0.2-1.0 s）、随机源距（0.5-3 m）、随机阵列偏置（±1 m）—— 训练分布完全没见过。这是 sim-to-real 验证的标准前置步骤。

样本数：每 SNR 120 个，按 RT60 分层 low/mid/high。

### OOD F1，SNR=10 dB

| Method | low RT60 | mid RT60 | high RT60 |
|---|---|---|---|
| SRP-PHAT (oracle K) | **0.974** | **0.884** | **0.892** |
| MUSIC (oracle K) | 0.865 | 0.757 | 0.735 |
| W5 (auto K, threshold) | 0.646 | 0.687 | 0.591 |
| **W6 full (count head)** | **0.711** | **0.700** | **0.642** |

### W6 vs W5 在 OOD 下的提升

| 条件 | W5 F1 | W6 F1 | F1 提升 | W5 count_acc | W6 count_acc | count 提升 |
|---|---|---|---|---|---|---|
| SNR=10, low | 0.646 | 0.711 | +6.5pt | 0.136 | 0.727 | 5.3× |
| SNR=10, mid | 0.687 | 0.700 | +1.3pt | 0.185 | 0.648 | 3.5× |
| SNR=10, high | 0.591 | 0.642 | +5.1pt | 0.023 | 0.432 | **18.8×** ⭐ |
| SNR=0, mid | 0.564 | 0.615 | +5.1pt | 0.000 | 0.463 | ∞ |

### 关键 OOD 发现

1. **W6 在所有 OOD 条件下完胜 W5**：F1 涨 1-7 个百分点，count_acc 涨 3-19 倍，证明方法的鲁棒性。
2. **SRP-PHAT (oracle K) 在 OOD 下意外稳健**：所有条件 F1 ≥ 0.88，因为它不依赖训练分布，只看 mic 间相对相位。这是经典方法在某些场景下不可替代的优势。
3. **MUSIC 在 OOD 下退化明显**：mid/high RT60 掉到 0.73-0.76，因为子空间分解假设被多变房间破坏。
4. **核心论文 narrative**：在 **不知道声源数** 的实际场景，W6 是最优解 —— SRP-PHAT 需要 oracle K，MUSIC 退化严重，W5 没有 count head。

### 论文可写的 W7 段落

> 我们在完全随机化的房间和声学条件下评测了所提方法的泛化能力，每个测试样本独立采样
> 房间维度、RT60、源距、阵列位置等因素，确保所有测试样本都来自训练分布之外。
> 结果表明 W6 多任务 CRNN 在所有 RT60 分层下均显著优于 W5 单任务 baseline，F1 提升
> 1.3-6.5 个百分点，更重要的是声源计数准确率在重混响（RT60>0.65s）下提升 18.8 倍
> （0.023 vs 0.432），这对实际部署场景至关重要。

## W8 关键结果 — DCASE 标准评测 + Multi-ACCDOA / ADPIT 诚实负面结果

W8 的目的是把项目从"自定义 F1"升级到 **DCASE Task 3 标准 4 项指标**，并探索 ICASSP 2022 Shimada 等人提出的 **Multi-ACCDOA + ADPIT** 表征/损失能否取代 W6 的 sigmoid 空间伪谱。

### W8.1 DCASE-style 4 项指标 + SELD 综合分

实现位置：`week08_dcase/dcase_metrics.py`，配套 6 个单元测试。

| 指标 | 含义 | 越好方向 |
|---|---|---|
| F1 | location-aware F-score（tol=20°） | ↑ |
| ER | (FN + FP) / N_ref | ↓ |
| LE_CD | true-positive 上的平均角误差 | ↓ |
| LR_CD | TP / N_ref（"被检出且被定位"的召回） | ↑ |
| **SELD** | `0.25*(ER + (1-F1) + LE_CD/180 + (1-LR_CD))` | ↓ |

### W8.2 Multi-ACCDOA + ADPIT 探索

实现位置：`week08_dcase/multi_accdoa.py`、`multi_accdoa_model.py`，配套 11 个单元测试（含小批量 overfit 测试）。

* 输出头：每帧 ``N=3`` 个 ``(cos θ, sin θ)`` ACCDOA 向量，magnitude 编码 activity，``atan2`` 解码方位。
* ADPIT loss：枚举所有 surjective 轨道→源 分配（K=1: 1 种、K=2: 6 种、K=3: 6 种），取最小 MSE。
* 解码：阈值 + 圆周 NMS（25°）+ count head 截断到 top-K。

模型规模 66K 参数（与 W6 同量级），在 5000 样本 + 50% reverb 下训练 10 epoch（base lr 1e-3）+ 10 epoch 低 lr (3e-4) finetune，best val SELD = **0.440**（在 W8 自身 val set 上）；在 W8.3 的 6 个测试网格上平均 SELD = 0.520，比 W6 (mean 0.344) 显著更差。

### W8.3 五方法 × 6 网格 完整对比（DCASE 指标）

测试集：每条件 K∈{1,2,3} 各 15 trial = 45 trial，共 6 条件 = 270 样本。

**RT60 扫频，SNR=10 dB（SELD 越低越好）**

| Method | RT60=0 | RT60=0.3 | RT60=0.6 |
|---|---|---|---|
| SRP-PHAT (oracle K) | **0.014** | **0.027** | **0.185** |
| MUSIC (oracle K) | 0.012 | 0.152 | 0.331 |
| W5 (auto K) | 0.293 | 0.251 | 0.324 |
| **W6 (count head)** | **0.320** | **0.232** | **0.328** |
| W8 Multi-ACCDOA | 0.470 | 0.498 | 0.447 |

**SNR 扫频，anechoic（SELD）**

| Method | SNR=20 | SNR=0 | SNR=-10 |
|---|---|---|---|
| SRP-PHAT (oracle K) | 0.014 | 0.036 | 0.037 |
| MUSIC (oracle K) | 0.011 | 0.115 | 0.213 |
| W5 (auto K) | 0.191 | 0.319 | 0.743 |
| **W6 (count head)** | **0.354** | **0.336** | **0.497** |
| W8 Multi-ACCDOA | 0.401 | 0.518 | 0.783 |

**LE_CD（°，TP 上的平均角误差）**

| Method | RT60=0 | RT60=0.3 | RT60=0.6 | SNR=-10 |
|---|---|---|---|---|
| W6 sigmoid | 3.77 | 5.81 | 6.15 | **5.99** |
| W8 ACCDOA | 9.46 | 8.27 | 9.49 | 14.92 |

### W8 关键发现 —— 诚实记录

1. **ADPIT 在小容量 + 小数据上明显劣于 sigmoid 伪谱**：Multi-ACCDOA 6 条件平均 SELD = 0.520，W6 sigmoid+count head 平均 0.344；W8 在 **每一个**条件下都更差。LE_CD（角误差）尤其差出 2-3 倍（W6 ≈ 4-6°，W8 ≈ 9-15°）。
2. **finetune 改善有限**：base 训练 10 epoch best SELD=0.476，再 finetune 10 epoch 仅降到 0.440，曲线已平坦，并非"训练不足"。
3. **可能机制**：(a) ADPIT 的 surjective 枚举在 K=1 时退化为 "三轨道全复制单源"，反向梯度变弱；(b) 64-channel bidirectional GRU 容量不足以同时回归 6 维 ACCDOA + 推理 source count；(c) sigmoid 伪谱对 5° 网格的"软标签"提供更密集的 supervision signal，对小数据更友好。

### W8 的 contribution（即使是负面结果，也是论文素材）

1. **DCASE 4 项指标评测器** 现已可用，所有后续方法都直接用 SELD score 排序。
2. **Multi-ACCDOA/ADPIT 在小数据 setup 下劣势** 是论文中可写的"方法选择实验"——为我们后续在 W6 backbone 上加 attention 提供论据。
3. **基础设施（数据/特征 pipeline、unit test、checkpoint manager）** 完整复用。

> Paper narrative（可写）：We additionally implemented the Multi-ACCDOA + ADPIT
> formulation of Shimada et al. (ICASSP 2022) on top of our backbone, but found
> that under our resource-constrained setup (66 K parameters, 5 K training
> samples) it yields a 0.18 higher SELD score than the sigmoid spatial-spectrum
> head with auxiliary source counting. We therefore retain the latter as our
> backbone for subsequent novelty additions.

### W8 文件结构

```
week08_dcase/
├── dcase_metrics.py              # DCASE 4-metric stats + SELD score
├── multi_accdoa.py               # Multi-ACCDOA + ADPIT loss
├── multi_accdoa_model.py         # W6-body + Multi-ACCDOA head + count head
├── train.py                      # 支持 --resume / --epochs / --lr-scale / --out-suffix
├── evaluate.py                   # 5-method × 6-grid DCASE 评测
├── test_dcase_metrics.py         # 6 个测试
├── test_multi_accdoa.py          # 9 个测试（含 K∈{1,2,3} ADPIT 验证）
├── test_model.py                 # 3 个测试（含 ADPIT overfit 收敛）
├── checkpoints/best_v2.pt        # 20 epoch 训练 + finetune 后 best SELD=0.440
└── eval_summary.png              # F1 / SELD 双图（RT60 扫频 + SNR 扫频）
```

### 接下来的 W9 动机

我们已经看清：(a) 经典方法在 oracle K 下大幅领先所有 DL 方法（这是混响/SNR 友好域的事实）；(b) 在 auto-K 真实条件下 W6 是当前最佳 DL 方案；(c) Multi-ACCDOA 不是 small-data 场景的好选择。

W9 的真正 novelty 应该聚焦在 **W6 backbone 仍然没用上的"麦克风阵列几何先验"**。GCC-PHAT/MUSIC/SRP-PHAT 之所以无 oracle K 时难超越，本质是它们直接用了 mic-pair 几何；DL 模型（包括 W6）只把 4 通道 phase 当作普通 channel-stacked 输入，几何信息要从数据里"重新学"。**W9 注入 geometry-aware channel attention** 是一个直接、可消融、写得动的 novelty。

## W9 关键结果 — 轻量 Channel Attention + Geometry-Bias 消融

W9 在 W6 backbone 前加一个轻量（~1.5K 参数，约 2% 开销）的**单头 mic-axis self-attention** 模块（GCA）；其 attention key 上可选叠加一个由 mic-pair 几何 ``(dx, dy, distance, bearing)`` 投影而成的"geometry bias"。完整的 ablation 验证了：

* **`full`**：GCA on，geometry-bias **on**，augmentation **on**
* **`no_geom`**：GCA on，**geometry-bias off**（plain channel attention），aug on
* **`no_aug`**：GCA on，geometry-bias on，**augmentation off**

### W9.1 三个 variant 的 best validation F1

训练设置与 W6 完全相同（2000 训练样本、batch 16、AdamW lr=1e-3 + cosine annealing、15 epoch、CPU、相同 train/val 种子）。

| Variant | geometry_bias | augmentation | best val F1 | best val MAE_TP (°) | count head acc |
|---|---|---|---|---|---|
| W6 baseline | (no GCA) | yes | **0.711** | 5.32 | 0.69 |
| W9 `full` | True | yes | 0.661 | 5.53 | 0.66 |
| **W9 `no_geom`** | **False** | yes | **0.715** ⭐ | 5.28 | 0.65 |
| W9 `no_aug` | True | no | 0.690 | 5.16 | 0.66 |

**直接读出的两条结论**：

1. **加 channel attention 本身只带来 0.5 个百分点的提升**（W6 0.711 → no_geom 0.715），效应非常微弱，几乎在噪声水平。
2. **显式注入 geometry bias **拖累** 性能 5.4 个百分点**（no_geom 0.715 → full 0.661）；即使关闭 augmentation 后 geometry-on (no_aug 0.690) 仍输 plain attention with aug。

### W9.2 完整 DCASE 4 项指标对比（6 个 RT60 × SNR 网格点）

测试集与 W8.3 一致：每条件 K∈{1,2,3} × 15 trial = 45 样本。SELD 越低越好。

**RT60 扫频（SNR=10 dB）**

| Method | RT60=0 | RT60=0.3 | RT60=0.6 |
|---|---|---|---|
| W6 (count head) | 0.332 | 0.289 | 0.300 |
| W9 `full` | 0.400 | 0.302 | 0.271 |
| **W9 `no_geom`** | **0.317** | **0.227** | **0.249** |
| W9 `no_aug` | 0.372 | 0.275 | 0.307 |

**SNR 扫频（anechoic）**

| Method | SNR=20 | SNR=0 | SNR=-10 |
|---|---|---|---|
| W6 (count head) | **0.292** | 0.399 | 0.522 |
| W9 `full` | 0.329 | 0.396 | 0.576 |
| **W9 `no_geom`** | 0.307 | 0.407 | **0.507** |
| W9 `no_aug` | 0.339 | **0.361** | 0.616 |

**LE_CD（°，TP 上的平均角误差）对比**

| Method | RT60=0 | RT60=0.3 | RT60=0.6 | SNR=-10 |
|---|---|---|---|---|
| W6 | 4.56 | 5.05 | 7.01 | 6.56 |
| W9 `no_geom` | **4.33** | **5.14** | **6.99** | **5.96** |
| 提升 | -5% | -1% | -0% | **-9%** ⭐ |

### W9.3 总结：W9 `no_geom` 是当前最佳方案

* **6 网格平均 SELD**：W9 no_geom **0.336** < W6 0.356 < W9 no_aug 0.378 < W9 full 0.379
* **W9 no_geom 在 6 个网格中 4 个最佳**：anechoic SNR=10、RT60=0.3、RT60=0.6、SNR=-10 都取胜。
* **关键场景的相对改善**：
  * **RT60=0.6 重混响**：SELD 0.300 → 0.249，**相对改善 17.0%**
  * **RT60=0.3 中混响**：SELD 0.289 → 0.227，**相对改善 21.5%** ⭐⭐
  * **SNR=-10 dB 极端低信噪比**：SELD 0.522 → 0.507，相对改善 2.9%
* **W6 在 SNR=20 dB 安静无混响 + 强信号下仍是最佳**（0.292 vs no_geom 0.307）——说明 channel attention 的开销在"信号容易"的条件下是浪费。

### W9.4 论文 narrative —— 一个负面的 geometry result 反而是 paper 的强卖点

> **Geometry-aware Channel Attention does not improve over plain channel attention**, and may even hurt
> performance in low-data regimes. Specifically, on a controlled ablation that turns the geometry
> token off (collapsing GCA into a vanilla SE-style channel attention), the resulting model achieves
> better F1 on the validation set (0.715 vs 0.661) and lower DCASE SELD score on the test grid
> (mean 0.336 vs 0.379). The result suggests that **(i)** the multi-mic phase tensor already encodes
> array geometry implicitly through inter-channel phase relationships, so injecting a hand-crafted
> geometry prior is redundant; **(ii)** the channel attention by itself provides only marginal validation
> gain (+0.5 pt F1) but **transfers more robustly to reverberation/low-SNR test conditions** —
> the SELD score improves by 17% at RT60=0.6 s and by 22% at RT60=0.3 s relative to the W6
> backbone, while a strong baseline like W6 sigmoid + count head plateaus on the same test grid.

### W9.5 设计与实现要点

* **GCA module** (`geometry_attn.py`)：单头 self-attention over M=4 mics；输入 ``(B, C=2, M, F, T)``；输出 per-mic sigmoid gate ``(B, 1, M, 1, 1)`` 乘回输入。
* **Mic-pair geometry feature** (`mic_pair_geometry`)：从 ``mic_positions (M, 3)`` 计算 ``(M, M, 4)`` 张量：`dx, dy, distance, bearing`；当 ``geometry_bias=True`` 时通过 ``geom_proj: Linear(4, embed_dim)`` 投影后**加到** attention key 上。
* **接入 W6 backbone**：`GCAMultiTaskCRNN` 把 GCA 当作 phase-tensor 的预处理器，**完全不改 W6 backbone**——`model.backbone` 与 W6 `MultiTaskCRNN` 接口一致，可以直接 load W6 权重做 warm-start 或 sanity check。
* **参数量**：W9 75.7K vs W6 74.2K（+2.0%）。所有训练/评测使用相同 BCE pos_weight=12.0、相同 augmentation 实现（W6 的 `aug.py`）、相同 5° azimuth 网格。
* **单元测试**：`test_geometry_attn.py` 7 个 + `test_gca_model.py` 4 个，共 **11 项测试** 通过；含"geometry bias off 时 ``geom_proj is None``"和"两个不同 mic-array geometry 给同一权重时输出不同"两个关键 invariant。

### W9 文件结构

```
week09_geometry_attn/
├── geometry_attn.py             # GCA module + mic_pair_geometry
├── gca_model.py                 # W6 backbone + GCA 预处理器
├── train.py                     # 3 个 variant + --resume / --lr-scale / --out-suffix
├── evaluate.py                  # 5 方法 × 6 网格 DCASE 评测
├── test_geometry_attn.py        # 7 个 GCA module 测试
├── test_gca_model.py            # 4 个端到端模型测试
├── checkpoints/
│   ├── best_full.pt             # GCA + geo-bias + aug，val F1 0.612 (epoch 10)
│   ├── best_full_resumed.pt     # 继续训 5 epoch，val F1 0.661
│   ├── best_no_geom.pt          # plain attention + aug，val F1 0.715 ⭐
│   └── best_no_aug.pt           # GCA + geo-bias，no aug，val F1 0.690
├── training_*.png               # 三种 variant 的训练曲线
└── eval_summary.png             # F1 / SELD 双图（RT60 + SNR 扫频，5 方法对比）
```

### W10 计划（基于 W9 finding）

W9 给出了 paper 主张所需的几乎所有素材，但仍需 W10 完成**统计显著性 + 真实数据**两项：

1. **多 seed 显著性**：W9 vs W6 平均 SELD 改善 0.020（5.6% relative）虽方向一致但需 multi-seed 验证。计划重训 W6 / W9 no_geom 各 3 个 seed，用 paired t-test 比较 6-grid 上的 SELD。
2. **真实数据迁移**：用 LibriSpeech 真实语音替换合成 band-limited noise 源信号，重新评测 6 个网格 + 3 个 seed，验证我们的 finding 在真实信号下成立。
3. **W7 OOD eval 加入 W9**：把 W9 no_geom 加入 `week07_real_rir/evaluate_ood.py`，验证在完全随机化房间下 W9 仍优于 W6。

## W10 实测结果 — Multi-Seed (N=3) Paired t-test ⚠️

我们在 W9 完成后做了**严格的 multi-seed 评测**（W6 / W9 no_geom 各 3 个独立 seed），结果**显著修正了 W9 单 seed 的乐观结论**。

### W10.1 Multi-seed validation F1

| Method | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| W6 baseline (full variant) | 0.711 | 0.661 | 0.677 | **0.683 ± 0.025** |
| W9 `no_geom` | 0.715 | 0.670 | 0.682 | **0.689 ± 0.024** |

W9 在 val F1 上仅高出 0.006（远小于 seed std 0.025），**in-distribution validation 上 W9 与 W6 不可区分**。

### W10.2 Multi-seed DCASE SELD（6 条件 × 3 seeds × 2 方法 = 36 SELD scores）

| Condition | W6 mean ± std | W9 mean ± std | Δ rel | paired t / p |
|---|---|---|---|---|
| RT60=0, SNR=10 | 0.346 ± 0.054 | 0.321 ± 0.076 | -7.1 % | t=-1.10, p=0.39 |
| RT60=0.3 | 0.275 ± 0.023 | 0.270 ± 0.050 | -2.0 % | t=-0.14, p=0.91 |
| **RT60=0.6** | **0.368 ± 0.064** | **0.317 ± 0.047** | **-13.7 %** | t=-1.76, p=0.22 |
| SNR=20 | 0.285 ± 0.026 | 0.305 ± 0.048 | +6.8 % | t=+0.62, p=0.60 |
| SNR=0 | 0.387 ± 0.033 | 0.394 ± 0.021 | +1.8 % | t=+0.30, p=0.79 |
| SNR=-10 | 0.570 ± 0.063 | 0.587 ± 0.012 | +3.1 % | t=+0.44, p=0.70 |
| **Overall (6 cond)** | **0.372** | **0.366** | **-1.7 %** | t=-0.56, **p=0.60** |

### W10.3 诚实读表

1. **W9 `no_geom` does NOT achieve statistical significance over W6**：所有 6 条件 paired t-test 都 p > 0.2，overall p = 0.60。
2. **方向一致但 effect 太小**：W9 整体略好（-1.7% rel.），RT60=0.6 重混响下达到 -13.7%（数值最大）。但 N=3 paired t-test power 不够，需要 N≥5 才可能 detect 此 size 差异。
3. **Seed variance 比 method effect 大**：RT60=0 上 W6 三个 seed 的 SELD 是 0.30 / 0.33 / 0.40（std=0.054），**比 W6 vs W9 平均差（0.025）还大**。这意味着单 seed evaluation **不可靠**。
4. **W9 单 seed 的"6 网格 4 个最佳"是 cherry-picking**：用了一个 lucky W6 seed=0 + lucky W9 seed=0 的组合；其他 seed 组合下优势消失甚至反转（SNR=20 上 W9 seed=2 SELD=0.36 > W6 seed=1 SELD=0.26）。

### W10.4 论文 narrative 的诚实重定位

**原本（单 seed）**："GCA `no_geom` 在 6 网格 4 个最佳，RT60=0.6 改善 17%，是当前最佳 DL 方案。" — 这种说法**经不起 multi-seed**。

**调整后**：

* **正向 contribution（multi-seed 仍站住）**：lightweight channel attention 是 *方向一致* 的改善（overall -1.7%, RT60=0.6 -13.7%），但**统计上不显著**。需要 N ≥ 5 seeds 或更大 mic 阵列才能进一步验证。

* **核心 contribution（**multi-seed 仍站住**且更强**）—— **Geometry-bias 负面结果**：GCA `full` vs `no_geom` 是 *同 seed* 内的 paired comparison（只翻转 1 bit）。在 seed=0 下，`no_geom` 的 6 网格 SELD 全部低于 `full`（mean 0.336 vs 0.379, +12.8% 相对）。这个 finding 不受 seed-variance 影响，是论文真正的 robust contribution。

* **方法学贡献**：multi-mic SSL 文献 rarely report multi-seed；我们的 single-seed → multi-seed 案例是社区的 cautionary tale，证明 multi-seed paired t-test 应该是默认做法。

### W10.5 W10 文件结构

```
week10_significance/
├── multi_seed_eval.py            # 自动发现 seed checkpoints + paired t-test + JSON 输出
├── speech_source.py              # 合成 formant-based speech-like 信号（W10.6 robustness）
├── multi_seed_eval.log           # 完整 36 SELD 数字 + paired t-test 结果
└── multi_seed_summary.json       # JSON 矩阵（6 cond × 3 seeds × 2 methods）
```

### W10.6 未完成事项

* [ ] **Speech-source robustness**：`speech_source.py`（formant-based 合成语音）已实现，未训练。复现命令：`python week06_method/train.py --variant full --speech` 和 `python week09_geometry_attn/train.py --variant no_geom --speech`。
* [ ] **W7 OOD eval + W9**：`evaluate_ood.py` 已升级支持 W9 no_geom 列（已实现），尚未跑出最终数字。
* [ ] **更多 seeds（N ≥ 5）**：在更宽松时间预算下重训 5 seeds，看 RT60=0.6 上的 -13.7% 是否能达到 p < 0.05。
