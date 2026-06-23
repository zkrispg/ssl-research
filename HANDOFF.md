# 项目接手清单(HANDOFF)

> 给在**新机器上接手本项目的 AI 助手 / 研究者**看。读完这一份 + `README.md` 即可恢复工作。
> 最后更新:2026-06-23。

---

## 0. 一句话项目状态

正在为一篇 **SELD(声音事件定位与检测)方法学论文**(目标期刊 *Applied Acoustics*)收敛"**几何先验注入 × 时序架构** 交互效应"的稳健性实验和投稿稿件。核心代码与论文已用 git + Git LFS 备份到私有仓库 `github.com/zkrispg/ssl-research`。**截至 2026-06-23,主要 revision 实验已全部跑完:FOA convbias 补充 task 184–187 的 ablate/GPU rerun 已完成,GCA extremes 已补到 n=10,Conformer tuning sweep 也已完成。当前最稳妥的论文表述是:GCA 的作用强依赖架构与模态;Conformer 是 neutral-to-weak-help middle point;`doae` 选模虽然能压低角度误差,但会显著牺牲 F20/SELD,不能作为主工作点。**

---

## 1. 两条研究线(别混淆)

仓库里有两套并存的工作,**当前活跃的是第 2 条**:

| 线 | 位置 | 内容 | 状态 |
|---|---|---|---|
| ① 合成多源 SSL | `week01_*` ~ `week11_*`、`README.md`、`SESSION_SUMMARY.md` | 自建合成数据上的 GCC-PHAT/MUSIC/CRNN + "几何 channel attention 是负面结果"的故事 | 早期工作,draft 已写,**非当前重点** |
| ② **DCASE 真实数据 + 几何注入 × 架构(当前)** | **`dcase2024_baseline/`** | 在官方 DCASE2024 SELD baseline 上,研究几何先验注入对不同时序架构(CRNN/Conformer/Transformer)的不同影响 | **进行中**,见第 3 节 |

⚠️ `README.md` 和 `SESSION_SUMMARY.md` 描述的是第 ① 条线,**不要据此理解当前任务**。当前任务以本文件第 3 节为准。

---

## 2. 当前实验的科学逻辑

**已有发现(GCA 这第一种注入机制)**:几何先验通过 *Geometry-aware Channel Attention*(GCA,加在 attention key 上的几何 bias)注入时,效果随**时序架构**变化:

- `task 130` FOA + **CRNN** → 几何先验 **HELPS**(有帮助)
- `task 171` FOA + **Conformer** → 中间(neutral)
- `task 151` FOA + **Transformer** / `task 141` MIC + Transformer → 几何先验 **HURTS**(有害)

即 **"helps → neutral → harms" 随架构从 CRNN→Conformer→Transformer 变化**。

**当前回答的问题**:这个排序是 GCA 这一种注入方式的偶然,还是更普适?于是引入**第二种机制不同的注入方式 `convbias`**:把几何描述子用一个 `Linear(G → nb_cnn2d_filt, bias=False)` 投影,作为**逐通道的加性 bias** 加到第一层卷积特征图上(实现见 `dcase2024_baseline/seldnet_model.py` 的 `per_channel_geometry_vector` + `_maybe_add_geom_bias`)。新结果显示:FOA+Conformer 在两轮 convbias 中都基本 neutral(`+0.21°`/`+0.31°`),但 FOA+Transformer 从 ablate 的 harm(`+9.55°`)变为 GPU-logged rerun 的 help(`-2.89°`);MIC+Transformer 端 convbias 也与 GCA 反号(`-3.79°` vs `+5.70°`)。因此论文表述必须是"主 GCA 结论成立;第二注入机制显示机制/训练选择敏感性",不能写成跨机制稳健复现。

每个实验都有一对 variant,**唯一区别是几何信息开/关**,参数量严格相等:
- `full` = 几何先验 ON
- `no_geom` = 同样的投影层但几何输入置零(matched-capacity 对照)

---

## 3. 任务定义与训练进度

任务在 `dcase2024_baseline/parameters.py` 里按 `argv` 数字定义。convbias 实验:

| task | 模态 | 架构 | variant | 对应 GCA 任务 | 进度(目标 3 seeds) |
|---|---|---|---|---|---|
| 180 | FOA | CRNN | full | 130 | ✅ 3/3 |
| 181 | FOA | CRNN | no_geom | — | ✅ 3/3 |
| 182 | MIC | Transformer | full | 141 | ✅ 3/3 |
| 183 | MIC | Transformer | no_geom | — | 🔄 2/3(seed2 在 3050Ti 上跑) |
| 184 | FOA | Conformer | full | 171 | ✅ ablate 3/3 + GPU-logged 3/3 |
| 185 | FOA | Conformer | no_geom | — | ✅ ablate 3/3 + GPU-logged 3/3 |
| 186 | FOA | Transformer | full | 151 | ✅ ablate 3/3 + GPU-logged 3/3 |
| 187 | FOA | Transformer | no_geom | — | ✅ ablate 3/3 + GPU-logged 3/3 |

> 180–183 是在旧 3050Ti 上跑的(MIC+FOA 混合);184–187 是 **FOA-only 补充**,专门迁到 RTX 5060 跑,用来补全"FOA 模态、固定模态只变架构"的干净对照(CRNN=180/181, Conformer=184/185, Transformer=186/187)。

**n=3 + GPU-logged 重跑后的信号**:FOA+Conformer 稳定 neutral:ablate `+0.21°`,GPU rerun `+0.31°`。FOA+Transformer 不稳定:ablate `+9.55°`(harm),GPU rerun `-2.89°`(help)。MIC+Transformer convbias 为 `-3.79°`,也与 GCA `+5.70°` 反号。因此 convbias 应写成"第二注入机制暴露敏感性/边界条件",不是 A-plan 的稳健复现。

### 3.1 2026-06-17 GCA Conformer deterministic runner 结果

训练机上的 **GCA + Conformer 确定性补跑** 已完成，用于把 GCA 表里的 Conformer 行整理成最终可引用版本:

- runner: `dcase2024_baseline/_run_gca_conformer_deterministic_local.ps1`
- 任务矩阵: `161/162/171/172 × seed0..4`，共 20 个 train/test 单元
- 完成时间: `2026-06-17 04:32:55`
- 当前进度: **20/20 test log 已完整落盘**
- 状态文件: `runs/gca_conformer_det_seld_20260614_234047_status.txt`
- 最终汇总: `runs/gca_conformer_det_seld_final.md` / `.csv` / `.json`
- 复跑方式: 直接重跑同一个 `.ps1`；脚本会用 test log 完整性跳过已完成单元

最终 Conformer GCA 结果:

| pair | 含义 | seeds | 方向 |
|---|---|---:|---|
| 161-162 | MIC + Conformer, GCA full minus no_geom | 0..4 | ΔDOAE=-1.83°, ΔSELD=-0.0266: weak full advantage |
| 171-172 | FOA + Conformer, GCA full minus no_geom | 0..4 | ΔDOAE=-0.57°, ΔSELD=-0.0510: near-neutral / weak full advantage |

论文表述应保持克制: Conformer 不是 Transformer 那种明确 harm，也不是 CRNN 那种强 help；最稳妥写法是 **neutral to weak-help middle point**。

### 3.2 2026-06-20 GCA extremes n=10 结果

训练机上的 **key extremes n=10 补种子** 已完成,用于回答"核心极端点在更大 seed 数下是否依然成立":

- runner: `dcase2024_baseline/_run_gca_extremes_n10_local.ps1`
- 任务矩阵: `130/131/141/142 × seed5..9`,与既有 `seed0..4` 合并成 `n=10`
- 完成时间: `2026-06-20 13:52:48`
- 最终汇总: `runs/gca_extremes_n10_summary.md` / `.csv` / `.json`

最终 n=10 配对结果:

| label | pair | n | delta SELD | delta F20 | delta DOAE | p(DOAE) | 解释 |
|---|---|---:|---:|---:|---:|---:|---|
| FOA_CRNN | 130-131 | 10 | +0.002 | -0.690 | +2.275 | 0.027 | full 比 no_geom 更差,且 DOAE 已到显著 |
| MIC_Transformer | 141-142 | 10 | +0.007 | +0.333 | -2.316 | 0.311 | full 方向上略好,但统计上不显著 |

这组结果的重要含义不是"几何先验普遍有效",而是 **GCA 的收益/伤害依赖具体架构与模态**。对论文写法来说,`130/131` 这组必须从"稳健帮助"改为"在 n=10 下 full 反而显著更差";`141/142` 则最多写成弱趋势,不能写成确定性帮助。

### 3.3 2026-06-23 Conformer tuning sweep 结果

训练机上的 **Conformer operating-point pilot** 已完成,用于判断 Conformer 的弱表现是不是学习率/选模指标导致的假象:

- runner: `dcase2024_baseline/_run_conformer_tuning_sweep_local.ps1`
- 任务矩阵: `171/172 × seed0,1 × lr(1e-3,5e-4,3e-4) × best_metric(seld,doae)`,共 24 个 train/test 单元
- 完成时间: `2026-06-23 02:10:30`
- 最终汇总:
  - `runs/conformer_tune_seld_lr1em3_summary.*`
  - `runs/conformer_tune_seld_lr5em4_summary.*`
  - `runs/conformer_tune_seld_lr3em4_summary.*`
  - `runs/conformer_tune_doae_lr1em3_summary.*`
  - `runs/conformer_tune_doae_lr5em4_summary.*`
  - `runs/conformer_tune_doae_lr3em4_summary.*`

可直接用于论文的结论:

- `best_metric=seld` 明显优于 `best_metric=doae`,因为后者虽然把 `DOAE` 压到 `24.5–30.1°`,但 `F20` 会掉到 `0–3%`,`SELD` 也会劣化到 `0.82–0.97`
- 在 `seld` 选模里,`lr=3e-4` 是最稳妥的工作点:
  - `171`: `SELD 0.583`, `F20 8.00%`, `DOAE 37.80°`
  - `172`: `SELD 0.589`, `F20 8.30%`, `DOAE 39.10°`
  - 配对差: `delta SELD = -0.007`, `delta DOAE = -1.300`,方向上是 full 略好,但 `n=2` 下不显著
- `lr=5e-4` 也能工作,但 seed 间波动更大;`lr=1e-3` 的绝对指标整体更差

因此当前最稳妥的 Conformer 叙事是: **Conformer 不是几何注入的强受益者,但在合理工作点(`seld + lr=3e-4`)下表现稳定且接近 neutral-to-weak-help;使用 `doae` 做 early-stop 会造成检测能力塌缩,不应作为主结果工作点。**

---

## 4. 环境搭建(新机器)

### RTX 5060(Blackwell)专用 —— 必读
Blackwell 架构需要 **PyTorch ≥ 2.7 + CUDA 12.8**,否则 GPU 不被识别:

```bash
# 1) 先单独装 torch(cu128 轮子)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# 2) 再装其余依赖
pip install -r dcase2024_baseline/requirements_5060.txt
# 3) 验证 GPU 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

依赖清单见 `dcase2024_baseline/requirements_5060.txt`(numpy/scipy/librosa/h5py 等,Python 3.10)。

### MacBook(仅看代码/写论文,不训练)
直接 `pip install -r requirements.txt`(CPU 版),可跑分析脚本和单元测试,但不要在 Mac 上训练 DCASE 模型(太慢)。

---

## 5. 数据(GitHub 里没有,需自备)

git 仓库**只含代码 + 论文 + checkpoint(LFS)**,不含数据集。新机器需要:

1. **原始数据集**(公开,重新下载):
   - DCASE2024 SELD dev set(FOA + MIC),来自 DCASE 官网 / Zenodo
   - 当前 convbias 实验**只用 FOA**,所以最少只需 FOA dev 音频
2. **提取特征**:数据集放好后,运行 baseline 的特征提取脚本生成 `seld_feat_label/`(`.npy`)。**不要试图备份/传输这些特征(~60GB),它们可重新生成。**
3. 期望目录布局(放在仓库外或仓库内被 `.gitignore` 的位置):
   ```
   <数据根>/foa_dev/          原始 FOA 音频(4 通道 wav)
   <数据根>/metadata_dev/     标注 csv
   <数据根>/seld_feat_label/  提取的特征+标签(运行提取脚本生成)
   ```
   路径在 `dcase2024_baseline/parameters.py` 的 `dataset_dir` / `feat_label_dir` 配置。

> 数据集为什么大、各目录含义:见与本仓库无关的说明,简言之 = 4 通道空间音频 + FOA/MIC 双格式 + 未压缩特征。

### 预训练 baseline 权重(已在 LFS 里)
convbias 用 finetune 模式,依赖两个 baseline 预训练权重(已随仓库 LFS 下载):
- `dcase2024_baseline/3_1_dev_split0_multiaccdoa_foa_model.h5`(FOA)
- `dcase2024_baseline/6_1_dev_split0_multiaccdoa_mic_gcc_model.h5`(MIC)

---

## 6. 怎么跑

所有命令在 `dcase2024_baseline/` 下执行。

### 单次训练(手动)
```bash
# 用法: python train_seldnet.py <task_id> <job_name> <seed>
python train_seldnet.py 184 ablate_seed0 0
```
checkpoint 输出到 `models_audio/<task>_ablate_seed<N>_..._model.h5`,测试结果到 `results_audio/`。

### 批量跑 5060 上的 FOA 补充(184–187 × 3 seeds = 12 runs)
```powershell
# 已写好编排脚本,带断点续跑(auto-skip 已完成的 run)
./_run_convbias_foa.ps1
```

### 烟雾测试(确认 task 配置能正确实例化模型)
```bash
python _smoke_convbias.py
```

### 跨注入统计分析(全部 seed 跑完后)
```bash
python _path_c_crossinject.py
```
它复用 `_path_c_2x2_dissociation.py` 的日志解析与统计函数(paired t-test、Cohen's dz、bootstrap 95% CI),对比 convbias 的 ΔDOAE 与 GCA,输出 JSON + Markdown。

---

## 7. 待办(按优先级)

1. **[已完成]** RTX 5060 GPU-logged 重跑 `gpu_20260609_153817` 已完成,日志在 `runs/gpu_20260609_153817_driver.out.log`,GPU 监控在 `runs/gpu_20260609_153817_nvidia_smi.csv`。
2. **[已完成]** 184–187 × 3 seeds 的 ablate 结果已汇总到 `outputs/convbias_foa_184_187_summary.md`;GPU rerun 审计见 `outputs/convbias_gpu_rerun_audit.md`。
3. **[已完成]** GCA extremes 已补到 `n=10`,汇总见 `runs/gca_extremes_n10_summary.md`;Conformer tuning sweep 已完成,汇总见 `runs/conformer_tune_*_summary.*`。
4. **[写作进行中]** 把 `n=10` robustness check 和 Conformer tuning 结论写回 Applied Acoustics 稿件,更新实验表格/讨论/限制部分,并确认主工作点采用 `seld + lr=3e-4` 而不是 `doae` early-stop。
5. **[写作进行中]** 编译 PDF、检查版式、压缩图表和完善投稿材料。
6. **[可选]** 若要继续研究 convbias,应先做 deterministic runner(排序文件列表、cuDNN deterministic、固定 checkpoint 选择指标),再补更大 seed;否则不要把 convbias Transformer 方向作为主论文证据。

---

## 8. git / LFS 工作流(新机器)

```bash
# 克隆(含 checkpoint,约 530MB LFS)
git clone https://github.com/zkrispg/ssl-research.git
# 只要代码、跳过 LFS 大文件:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/zkrispg/ssl-research.git

# 日常备份(改了代码/论文/新 checkpoint 后)
git add -A && git commit -m "描述改动" && git push
```

- `*.h5` 已配置走 **Git LFS**(见 `.gitattributes`)。新机器需先 `git lfs install`。
- `.gitignore` 已排除数据集、`.npy` 特征、`venv/`、`results_audio/` 等大文件。
- LFS 免费额度 1GB 存储 + 1GB/月流量,别一个月全量克隆超过 2 次。

### 关于内嵌仓库的历史说明
`dcase2024_baseline/` 和 `external/seld-dcase2022/` 原本是从上游 clone 的独立 git 仓库;为纳入统一备份,它们的内嵌 `.git` 已重命名为 `.git_upstream_backup/`(被 ignore,保留在本地)。当前它们的内容已作为普通文件在主仓库里。上游地址:
- `dcase2024_baseline` ← `https://github.com/partha2409/DCASE2024_seld_baseline.git`
- `external/seld-dcase2022` ← `https://github.com/sharathadavanne/seld-dcase2022.git`

---

## 9. 关键文件速查

| 文件 | 作用 |
|---|---|
| `dcase2024_baseline/parameters.py` | 所有实验配置(task 180–187 在此定义) |
| `dcase2024_baseline/seldnet_model.py` | SELD 模型;含 GCA 与 convbias 注入实现 |
| `dcase2024_baseline/train_seldnet.py` | 训练入口 |
| `dcase2024_baseline/_run_convbias_foa.ps1` | 184–187 批量训练编排(5060 用) |
| `dcase2024_baseline/_run_convbias.ps1` | 180–183 批量训练编排(已基本跑完) |
| `dcase2024_baseline/_smoke_convbias.py` | task 配置烟雾测试 |
| `dcase2024_baseline/_path_c_crossinject.py` | convbias × GCA 跨注入统计分析 |
| `dcase2024_baseline/requirements_5060.txt` | 5060 依赖清单 |
| `paper/` | 论文稿件与图表 |
