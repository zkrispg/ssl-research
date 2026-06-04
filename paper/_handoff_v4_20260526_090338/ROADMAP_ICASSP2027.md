# ICASSP 2027 冲刺路线图

**目标**：ICASSP 2027（CCF-B）  
**ddl**：~2026 年 9 月初  
**预算**：4-6 周实验 + 写作（共 ~8 周，留 4 周缓冲）  
**当前**：2026-05-14（剩余 ~16 周）

---

## 总体阶段

```
[Phase 1] 第 1-2 周：基础重建（数据 + 现代 baseline 框架）
[Phase 2] 第 3-4 周：实验 + 调优
[Phase 3] 第 5 周：消融 + 多 seed
[Phase 4] 第 6 周：补充实验 + 论文初稿
[Phase 5] 第 7-8 周：润色 + 投稿前检查
[Buffer] 剩余 8 周机动 / 应对意外
```

---

## Phase 1: 基础重建（第 1-2 周）

### Week 1：STARSS23 + SELDnet pipeline

| 天 | 任务 | 产出 | 状态 |
|---|---|---|---|
| Day 1 | 解析 STARSS23 metadata，生成 ACCDOA 标签 | `week11_starss23/seld_labels.py` + 单测 | ⏳ |
| Day 2 | log-mel + GCC-PHAT 特征提取 | `seld_features.py` + 单测 | ⏳ |
| Day 3 | STARSS23 Dataset 类（懒加载） | `starss_dataset.py` | ⏳ |
| Day 4 | SELD 模型架构（W6 backbone + 3 heads） | `seld_model.py` | ⏳ |
| Day 5 | 训练循环 + 3D class-coupled ADPIT | `train_seld.py` | ⏳ |
| Day 6 | SELDnet baseline 集成到 pipeline | `seld_baseline.py` | ⏳ |
| Day 7 | dev set 跑通 + 第一组 baseline 数字 | 数据表 | ⏳ |

**Week 1 目标**：在 STARSS23 上拿到 W6 vs W9 vs SELDnet 的初步对比。

### Week 2：GCA 集成 + 多 seed 训练

| 天 | 任务 | 产出 |
|---|---|---|
| Day 8-9 | 把 W9 GCA 模块挂到 SELD 模型，跑 `full` / `no_geom` 两个 variant | 2 个 checkpoint |
| Day 10-11 | 在 STARSS23 上 N=3 seed 训练（3 模型 × 3 seed = 9 个 run） | 9 个 checkpoint |
| Day 12 | DCASE 评估脚本扩展，计算 4 metrics + SELD score | `evaluate_seld.py` |
| Day 13-14 | 跑 STARSS23 dev/eval 评估，填 Table 2 | Table 2 数据 |

---

## Phase 2: 实验 + 调优（第 3-4 周）

### Week 3：合成数据 GPU 重跑 + baseline 补全

| 任务 | 工作量 |
|---|---|
| 用 GPU 重跑合成数据 W6/W9 multi-seed（之前 CPU 数字要更新） | 1 天 |
| 把 SELDnet baseline 在合成数据上也跑一遍 | 1 天 |
| 填 Table 1（合成数据完整版） | 0.5 天 |
| 调整 W9 GCA 超参（如果 STARSS 表现差） | 1-2 天 |
| 跑 ablation：augmentation on/off, attention head 数 | 1 天 |

### Week 4：补强实验 + 鲁棒性

| 任务 | 工作量 |
|---|---|
| OOD 测试：STARSS23 dev-train (Sony) → dev-test (TAU) cross-room | 1 天 |
| Multi-seed t-test 在 STARSS23 上的版本 | 0.5 天 |
| 注意力可视化（Figure 2 用） | 1 天 |
| 收集所有训练日志，整理成可重现的 config 文件 | 1 天 |
| **Buffer**：处理意外 bug | 1.5 天 |

---

## Phase 3: 消融 + 多 seed（第 5 周）

### Week 5：完整消融矩阵

最终消融矩阵：

| 维度 | 取值 | 数据集 |
|---|---|---|
| Method | {SELDnet, W6, W9 full, W9 no_geom} | × |
| Seed | {0, 1, 2} | × |
| Dataset | {Synthetic, STARSS23} | = 24 runs |

如有时间加：
- N=5 seed（更稳的 t-test）
- 阵列尺寸 ablation（4 mic vs 8 mic）

---

## Phase 4: 论文初稿（第 6 周）

| 任务 | 工作量 |
|---|---|
| 把 30 页 `draft.md` 压缩到 4 页 ICASSP 格式 | 1.5 天 |
| 重写 Related Work（加 2024-2025 引用：GI-DOAEnet, CST-Former, AuralNet, Neural-SRP, SWeC, DCASE 2024 winner USTC） | 1 天 |
| 画 Figure 1（架构图：backbone + GCA + heads） | 0.5 天 |
| 画 Figure 2（punchline：配对消融 Δ across conditions/datasets） | 0.5 天 |
| 画 Figure 3（可选：注意力权重热图） | 0.5 天 |
| 整理 References（~25 篇） | 0.5 天 |
| 转 LaTeX（用 ICASSP 模板） | 1 天 |
| Self-review 第一遍 | 0.5 天 |

---

## Phase 5: 润色 + 投稿（第 7-8 周）

| 任务 | 工作量 |
|---|---|
| 找导师 / 同学读，收一轮反馈 | 1 周 |
| 改稿：清楚的 contributions、流畅的转折、所有 TODO 填齐 | 2-3 天 |
| 检查 ICASSP 提交清单：4 页限制、figure 分辨率、anonymity、cite format | 0.5 天 |
| 最终 proofread + 上传 | 0.5 天 |

---

## 关键里程碑

| 时间点 | 里程碑 |
|---|---|
| **Week 1 末** | STARSS23 跑通，第一组 baseline 数字（哪怕是 quick test） |
| **Week 2 末** | Table 2（STARSS23 完整结果）填满 |
| **Week 4 末** | 所有实验数据出齐，论文还没动笔但消融完整 |
| **Week 6 末** | 论文初稿（English, LaTeX）完成 |
| **Week 8 末** | 投稿就绪 |

---

## 必做项（CRITICAL — 没有就投不了 ICASSP）

🔴 **这四件事必须完成**：

1. STARSS23 数据 pipeline（loader, features, labels）
2. W9 SELD 适配（3D class-coupled ACCDOA + ADPIT）
3. SELDnet baseline 集成
4. 在 STARSS23 上 N=3 seed × 3 模型完整跑

---

## 重要项（IMPORTANT — 没有论文会弱）

🟠 **强烈建议完成**：

5. GPU 重跑合成数据 multi-seed（~1 小时，验证可重现）
6. 合成 Table 1 加 SELDnet baseline
7. 重写 Related Work（加 2024-2025 引用）

---

## 锦上添花（NICE TO HAVE）

🟢 **时间够再做**：

8. distance estimation（DCASE 2024 task variant）
9. 注意力可视化 figure
10. CST-Former 第二个现代 baseline（额外 2-3 天）
11. N=5 seed（更稳的 t-test）

---

## 风险登记 & 缓解策略

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| STARSS23 上 W9 `no_geom` **不再**优于 `full`（即合成上的 finding 不在真实数据上 hold） | 🟡 中 | 🔴 大 | 即便如此可写"synthetic vs real divergence"也是有趣发现 |
| 4-mic UCA 子采样失真严重 | 🟡 中 | 🟠 中 | 备选：用 SELDnet 的 GCC 特征（不需要纯 UCA 假设） |
| 训练 9 个 model × 多数据集时间超预算 | 🟢 低 | 🟠 中 | GPU 充足，每个 model 9 分钟 ≈ 1 小时全跑完 |
| GI-DOAEnet 复现差异大 | 🟡 中 | 🟢 小 | 不打算复现，只 cite 作为 supportive evidence |
| ICASSP 模板 / LaTeX 学习曲线 | 🟢 低 | 🟢 小 | 模板成熟，半天足够 |

---

## 论文核心消息（**别忘了**）

> 在 low-resource（75K 参数 / 2K 样本）多源 SSL 中，通过注意力偏置注入的手工几何先验**降低**了在合成 UCA-4 和真实 STARSS23 上的性能。配对消融把这个效应隔离到几何路径。结合多 seed 配对检验，这呼吁一个更谨慎的默认设置：**当相位在输入里，几何就已经在那了。**

---

## 当前进度（2026-05-14）

✅ 已完成：
- 项目搬到 D 盘 + 权限修复
- Python venv + CUDA torch 2.6.0
- 6 个 train.py GPU 化
- 2-epoch GPU 烟测通过（RTX 3050 Ti）
- 文献综述（GI-DOAEnet, CST-Former, AuralNet, DCASE 2024 winner）
- ICASSP 论文新大纲（`paper/icassp_draft.md`）
- SELDnet 官方代码 clone（`external/seld-dcase2022/`）
- STARSS23 metadata 下载 + 解压
- STARSS23 mic_dev.zip 后台下载中（10.8 %, ETA ~2 小时）

🔄 进行中：
- STARSS23 数据下载

⏳ 下一步：
- Day 1：写 `seld_labels.py`（不需要等音频，metadata 已就位）

---

## 文件位置速查

| 内容 | 路径 |
|---|---|
| 论文新大纲 | `D:\ssl-research\paper\icassp_draft.md` |
| 论文长版（保留） | `D:\ssl-research\paper\draft.md` |
| 路线图（本文件） | `D:\ssl-research\paper\ROADMAP_ICASSP2027.md` |
| SELDnet 代码 | `D:\ssl-research\external\seld-dcase2022\` |
| STARSS23 数据 | `D:\ssl-research\data\STARSS23\` |
| 已有训练代码 | `D:\ssl-research\week03..week09\` |
| 待写 SELD 模块 | `D:\ssl-research\week11_starss23\`（未创建） |
| GI-DOAEnet 论文（友军） | `agent-tools/4b42247e-...txt` |
