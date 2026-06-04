# 项目接手清单(HANDOFF)

> 给在**新机器上接手本项目的 AI 助手 / 研究者**看。读完这一份 + `README.md` 即可恢复工作。
> 最后更新:2026-06-04。

---

## 0. 一句话项目状态

正在为一篇 **SELD(声音事件定位与检测)方法学论文**(目标期刊 *Applied Acoustics*)补做"**几何先验注入 × 时序架构** 交互效应"的稳健性实验。核心代码与论文已用 git + Git LFS 备份到私有仓库 `github.com/zkrispg/ssl-research`。**当前唯一进行中的任务:把 convbias 这第二种几何注入机制,在 FOA 模态的三种架构上各跑 3 个 seed(task 184–187),验证"helps→neutral→harms"的架构序是否稳健。**

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

**当前要回答的问题**:这个排序是 GCA 这一种注入方式的偶然,还是更普适?于是引入**第二种机制不同的注入方式 `convbias`**:把几何描述子用一个 `Linear(G → nb_cnn2d_filt, bias=False)` 投影,作为**逐通道的加性 bias** 加到第一层卷积特征图上(实现见 `dcase2024_baseline/seldnet_model.py` 的 `per_channel_geometry_vector` + `_maybe_add_geom_bias`)。如果 convbias 也呈现同样的 helps→harms 架构序,则结论稳健(可写成论文的"跨注入机制鲁棒性"小节 = A-plan)。

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
| 184 | FOA | Conformer | full | 171 | ⬜ 0/3 ← **待 5060 跑** |
| 185 | FOA | Conformer | no_geom | — | ⬜ 0/3 ← **待 5060 跑** |
| 186 | FOA | Transformer | full | 151 | ⬜ 0/3 ← **待 5060 跑** |
| 187 | FOA | Transformer | no_geom | — | ⬜ 0/3 ← **待 5060 跑** |

> 180–183 是在旧 3050Ti 上跑的(MIC+FOA 混合);184–187 是 **FOA-only 补充**,专门迁到 RTX 5060 跑,用来补全"FOA 模态、固定模态只变架构"的干净对照(CRNN=180/181, Conformer=184/185, Transformer=186/187)。

**preliminary 信号(n=2 时,待 n=3 复核)**:MIC+Transformer 端 convbias 的方向疑似与 GCA 反号(convbias≈−4.52° vs GCA≈+5.70° DOAE),所以补 n=3 很关键。

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

1. **[进行中]** 3050Ti 上 183 seed2 跑完(无需人工干预,批处理会自动收尾)。
2. **[5060 主任务]** 跑 184–187 × 3 seeds(`_run_convbias_foa.ps1`),约 12 个 run。
3. **[分析]** 全部 n=3 齐后跑 `_path_c_crossinject.py`,确认 convbias 是否复现 GCA 的 helps→neutral→harms 架构序;重点看 MIC+Transformer 端方向在 n=3 下是否稳住(n=2 时疑似反号)。
4. **[写作]** 按 A-plan 把结论写进论文:新增"跨注入机制鲁棒性"小节 + 一张"GCA vs convbias × 架构"对照表;叙事从"架构决定几何先验有效性"升级为"注入机制 × 架构的交互效应"。

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
