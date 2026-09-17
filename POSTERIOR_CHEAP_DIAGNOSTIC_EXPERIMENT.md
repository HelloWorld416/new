# EgoJEPA-MP：FutureModePosterior 低成本补充诊断实验

> 目标：**在不重新训练模型、不修改 FutureModePosterior、不修改 Past Mode Prior、不修改 trajectory loss 的前提下**，利用当前已经训练好的 K=4 checkpoint，进一步判断 mode 1/2 “有几何价值但几乎不被 posterior / prior argmax 选中”的根因到底是：
>
> 1. posterior 只是排序/置信度不够；
> 2. posterior 对 mode 1/2 的 soft probability 本身就接近 0，属于真正的 mode collapse；
> 3. oracle mode 通常只是 posterior rank-2，说明问题主要是 calibration / margin；
> 4. 或者 posterior 对 trajectory-specialized mode 的语义根本没有形成稳定对应。
>
> 本实验只做诊断，不改变任何模型权重。

---

## 1. 背景与当前已知结果

当前 JAAD 诊断已经得到：

```text
Validation:
Oracle ADE     = 18.869
Posterior ADE  = 22.605
Prior ADE      = 25.339
G_mode         = 3.736
G_prior        = 2.733

Test:
Oracle ADE     = 16.802
Posterior ADE  = 20.298
Prior ADE      = 21.907
G_mode         = 3.496
G_prior        = 1.609
```

并且测试集上：

```text
prior ↔ posterior agreement = 91.96%
posterior ↔ oracle agreement = 61.13%
```

已有关键发现：

- posterior 与 prior 的 argmax 只选择 mode 0 / mode 3；
- oracle 在验证集约 35.20%、测试集约 37.27% 的样本上选择 mode 1 / mode 2；
- mode 1 / 2 因而不是“完全无效的候选”，它们在大量样本上具有实际几何价值；
- 但目前还不知道 posterior 是“知道但没排第一”，还是“几乎完全不给 mode 1/2 概率”。

本实验只回答这个问题。

---

# 2. 必须新增的四组诊断

## 2.1 Oracle mode 的 posterior probability

对每个样本 i：

```math
k_i^{oracle}=\arg\min_k ADE_{i,k}
```

定义：

```math
q_i^{oracle}=q_i[k_i^{oracle}]
```

即：

> FutureModePosterior 给“几何上真正最优的 trajectory mode”分配了多少 probability。

至少汇总：

```text
mean(q_oracle)
median(q_oracle)
p25(q_oracle)
p75(q_oracle)
```

并分别统计：

```text
all
oracle_mode = 0
oracle_mode = 1
oracle_mode = 2
oracle_mode = 3
crossing = 0
crossing = 1
```

重点关注：

```text
oracle_mode in {1,2}
```

的 `q_oracle`。

### 判读

如果 mode 1/2 的 oracle 样本中：

```text
q_oracle 仍经常在 0.20 ~ 0.35 左右
```

说明 posterior 可能**知道 mode 1/2 有可能，只是 argmax 排序/置信度不够**。

如果：

```text
q_oracle 接近 0
```

说明 posterior 对这些有效 trajectory modes 基本没有识别能力，属于更强的 mode semantics / collapse 问题。

---

## 2.2 Soft mode usage 与 Hard mode usage

必须分别统计两种 mode usage。

### Hard usage

```math
u_k^{hard}
=
\frac{1}{N}\sum_i \mathbf{1}[\arg\max_j q_{i,j}=k]
```

也就是 posterior argmax 的使用频率。

### Soft usage

```math
u_k^{soft}
=
\frac{1}{N}\sum_i q_{i,k}
```

也就是 posterior probability mass 的平均占比。

输出：

| Mode | Hard usage | Soft usage | Oracle usage |
|---|---:|---:|---:|
| 0 | ... | ... | ... |
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |

其中 oracle usage 定义为：

```math
u_k^{oracle}
=
\frac{1}{N}\sum_i \mathbf{1}[k_i^{oracle}=k]
```

### 判读

#### 情况 A：hard collapse，但 soft 没 collapse

例如：

```text
hard usage:
mode0 = 60%
mode1 = 0%
mode2 = 0%
mode3 = 40%

soft usage:
mode0 = 35%
mode1 = 18%
mode2 = 17%
mode3 = 30%
```

说明：

> posterior 并没有真正忽略 mode 1/2，只是概率 margin / temperature / ranking 导致它们几乎从不成为 argmax。

此时下一步优先考虑 calibration / sharpening / mode alignment，而不是直接判定 prototype collapse。

#### 情况 B：soft usage 也接近 0

例如：

```text
mode1 soft usage ≈ 0
mode2 soft usage ≈ 0
```

说明：

> posterior 的表示或 prototype 本身没有覆盖 mode 1/2，对应真正的 mode collapse / semantic mismatch。

此时才更有理由进入 Plan Transformer posterior、prototype stabilization 等结构性改造。

---

## 2.3 Oracle mode 在 posterior 中的 rank

对每个样本，将 q 从高到低排序。

定义：

```math
rank_i^{oracle}
=
rank\big(q_i[k_i^{oracle}]\big)
```

取值：

```text
1, 2, 3, ..., K
```

K=4 时必须报告：

```text
oracle_rank_1_rate
oracle_rank_2_rate
oracle_rank_3_rate
oracle_rank_4_rate
```

并分别报告：

```text
all
oracle_mode=0
oracle_mode=1
oracle_mode=2
oracle_mode=3
crossing=0
crossing=1
```

### 重点判读

如果 mode 1/2 的 oracle 样本大量满足：

```text
oracle rank = 2
```

说明 posterior 对这些 mode 有明显感知，但 argmax margin 不够。

如果大量落在：

```text
rank = 3 or 4
```

则说明 posterior semantics 与 trajectory modes 的对应关系更差。

---

## 2.4 每个 mode 的 oracle frequency 与几何价值

对每个 mode k，统计：

```text
posterior_argmax_frequency
prior_argmax_frequency
oracle_frequency
```

以及当该 mode 成为 oracle 时，它相比 prior top-1 trajectory 的平均改进：

```math
\Delta ADE_k
=
\mathbb{E}[ADE_{prior}-ADE_{oracle}\mid k^{oracle}=k]
```

同步统计：

```math
\Delta FDE_k
=
\mathbb{E}[FDE_{prior}-FDE_{oracle}\mid k^{oracle}=k]
```

输出：

| Mode | Posterior argmax % | Prior argmax % | Oracle % | Mean q_oracle | Mean oracle rank | ΔADE vs prior | ΔFDE vs prior |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | ... | ... | ... | ... | ... | ... | ... |
| 1 | ... | ... | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... | ... |

### 关键问题

需要明确回答：

> mode 1/2 是在很多样本上都稳定有价值，还是仅在少数样本中偶尔成为 oracle？

如果 mode 1/2：

```text
oracle frequency 高
且 ΔADE 明显大
```

但 posterior/prior argmax frequency 接近 0，则说明当前 mode utilization 存在明显浪费。

---

# 3. 建议增加的补充指标

这些不是本实验的核心，但实现成本很低，建议一起输出。

## 3.1 Posterior margin

定义 posterior top-1 与 top-2 probability margin：

```math
margin_i=q_i^{(1)}-q_i^{(2)}
```

其中：

```text
q^(1) = 最大 posterior probability
q^(2) = 第二大 posterior probability
```

统计：

```text
mean_margin
median_margin
```

并对：

```text
posterior correct vs oracle
posterior incorrect vs oracle
```

分别报告。

如果错误样本的 margin 很小，说明 posterior 主要是低置信度排序问题。

如果错误样本 margin 很大，说明 posterior 是高置信度地选错 mode，问题更偏语义错配。

---

## 3.2 Oracle probability gap

定义：

```math
gap_i^{oracle}=q_i^{max}-q_i[k_i^{oracle}]
```

其中：

```math
q_i^{max}=\max_k q_{i,k}
```

报告：

```text
mean_oracle_prob_gap
median_oracle_prob_gap
```

以及 mode 1/2 子集。

---

## 3.3 Prior 对 mode 1/2 的 soft usage

虽然本实验重点诊断 posterior，但建议同时输出：

```math
\bar\pi_k=\frac{1}{N}\sum_i\pi_{i,k}
```

用于区分：

- prior 只是跟着 posterior collapse；
- 还是 prior 比 posterior collapse 更严重。

---

# 4. 分组要求

必须至少报告：

```text
all
crossing=0
crossing=1
```

如果现有样本字段支持，推荐再按：

```text
oracle_mode
posterior_correct / posterior_wrong
prior_correct / prior_wrong
```

分组。

不要为了新增分组修改数据协议。

---

# 5. 数据与 checkpoint

## 5.1 不重新训练

必须直接复用当前完成 JAAD 诊断的同一个 checkpoint。

优先使用当前实验中已经确认的：

```text
JAAD epoch 88 best.pt
```

如果实际路径与命名不同，以当前诊断脚本已经使用的 checkpoint 为准。

不要重新训练，不重新选 checkpoint。

---

## 5.2 Split 顺序

先运行：

```text
validation
```

确认诊断逻辑与样本统计合理。

再运行：

```text
test
```

不能基于 test 结果修改诊断定义。

---

# 6. Codex 实现原则

## 6.1 优先复用已有 mode-selection diagnostic

仓库中已经有：

```text
MODE_SELECTION_DIAGNOSTIC_EXPERIMENT.md
```

并且已有诊断代码能得到：

```text
k_oracle
k_post
k_prior
ADE_per_mode
FDE_per_mode
q
pi
```

Codex 应优先在现有 mode-selection diagnostic 基础上扩展，而不是重新写一套独立 forward / metric 流程。

---

## 6.2 不允许修改模型 forward 语义

本实验不得：

- 修改 FutureModePosterior；
- 修改 mode prototype；
- 修改 Past Mode Prior；
- 修改 trajectory head；
- 修改训练 loss；
- 修改 checkpoint；
- 修改 mode index；
- 加 temperature re-scaling 后再做主统计。

所有 q / pi 必须是 checkpoint 原始输出。

---

## 6.3 不重复实现 ADE / FDE

继续复用现有：

```text
bbox decode
ADE/FDE
per-mode trajectory error
```

保证与前一轮 diagnostic 完全一致。

---

# 7. 推荐代码改动

优先扩展现有：

```text
src/egojepa_diffusion/diagnose_mode_selection.py
```

如果实际代码路径不同，扩展当前已实现 mode-selection diagnostic 的脚本。

建议新增 helper：

```python
compute_posterior_oracle_diagnostics(...)
```

输入至少包括：

```text
posterior_probs q       [B, K]
prior_probs pi           [B, K]
ADE_per_mode             [B, K]
FDE_per_mode             [B, K]
crossing_label           [B]
```

返回 sample-level：

```text
oracle_mode
posterior_mode
prior_mode
q_oracle
oracle_rank
posterior_margin
oracle_prob_gap
```

以及 dataset-level aggregation。

---

# 8. 必须输出的文件

建议继续使用：

```text
outputs/<experiment>/mode_selection_diagnostics/<split>/
```

并新增：

```text
posterior_diagnostics_summary.json
posterior_diagnostics_samples.csv
posterior_diagnostics_report.md
```

---

## 8.1 summary.json

至少包含：

```json
{
  "num_samples": 0,
  "num_modes": 4,
  "q_oracle_mean": 0.0,
  "q_oracle_median": 0.0,
  "posterior_margin_mean": 0.0,
  "oracle_prob_gap_mean": 0.0,
  "oracle_rank_rates": {
    "rank1": 0.0,
    "rank2": 0.0,
    "rank3": 0.0,
    "rank4": 0.0
  },
  "per_mode": {
    "0": {},
    "1": {},
    "2": {},
    "3": {}
  }
}
```

并包含：

```text
all
crossing=0
crossing=1
```

分组结果。

---

## 8.2 samples.csv

每个样本至少保存：

```text
sample_id
video_id（若有）
pedestrian_id（若有）
crossing_label
oracle_mode
posterior_mode
prior_mode
q_oracle
oracle_rank
posterior_margin
oracle_prob_gap
posterior_correct_vs_oracle
prior_correct_vs_oracle
q_0
q_1
q_2
q_3
pi_0
pi_1
pi_2
pi_3
ade_mode_0
ade_mode_1
ade_mode_2
ade_mode_3
fde_mode_0
fde_mode_1
fde_mode_2
fde_mode_3
```

如果 K 不是 4，则动态生成 mode 列。

---

## 8.3 report.md

报告必须至少包含以下表格。

### 表 1：Soft / Hard / Oracle mode usage

| Mode | Posterior Hard % | Posterior Soft % | Prior Soft % | Oracle % |
|---|---:|---:|---:|---:|

### 表 2：Oracle posterior confidence

| Mode | Oracle Samples | Mean q_oracle | Median q_oracle | Mean Oracle Rank | ΔADE vs Prior |
|---|---:|---:|---:|---:|---:|

### 表 3：Oracle rank distribution

| Group | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---:|---:|---:|---:|

至少包括：

```text
all
oracle mode 1
oracle mode 2
crossing=0
crossing=1
```

---

# 9. 推荐可视化

如果已有 matplotlib 依赖，可低成本增加：

```text
posterior_soft_vs_hard_usage.png
oracle_rank_distribution.png
q_oracle_by_mode.png
posterior_margin_correct_vs_wrong.png
```

可选再加：

```text
q_oracle_hist_mode1.png
q_oracle_hist_mode2.png
```

如果绘图会引入新依赖，则可以跳过，不影响实验完成。

---

# 10. 单元测试

至少新增以下测试。

## 10.1 q_oracle 正确性

人工构造：

```text
q = [[0.1, 0.7, 0.1, 0.1]]
ADE = [[5.0, 3.0, 1.0, 4.0]]
```

应得到：

```text
oracle_mode = 2
q_oracle = 0.1
```

---

## 10.2 Oracle rank 正确性

上述例子中：

```text
q order = mode1 > mode0/mode2/mode3
```

为避免 tie 导致定义不稳定，单测必须使用无并列概率的 q。

例如：

```text
q = [0.15, 0.55, 0.20, 0.10]
ADE oracle = mode2
```

应得到：

```text
oracle_rank = 2
```

---

## 10.3 Soft usage 正确性

构造多样本 q，验证：

```text
soft_usage = mean(q, dim=0)
```

hard usage 必须来自 `argmax(q)`，两者不能混淆。

---

## 10.4 Oracle frequency 正确性

验证：

```text
oracle_usage = bincount(argmin(ADE_per_mode)) / N
```

---

## 10.5 K=1 sanity

K=1 时应满足：

```text
soft_usage = [1.0]
hard_usage = [1.0]
oracle_usage = [1.0]
q_oracle = 1.0
oracle_rank = 1
posterior_margin 可定义为 NaN 或 1.0，但必须统一并写入文档
```

---

# 11. 推荐的结果解释规则

## Case A：mode 1/2 的 soft usage 不低，但 hard usage≈0

且：

```text
oracle rank 多为 2
q_oracle 中等
posterior margin 较小
```

结论：

> posterior 并未真正 collapse，主要问题是 ranking / confidence / calibration，以及 posterior 与 trajectory mode 的边界不够清晰。

下一步优先：

```text
posterior↔trajectory alignment
trajectory routing sharpening
```

而不是立即换大 posterior 网络。

---

## Case B：mode 1/2 的 soft usage≈0

同时：

```text
q_oracle≈0
oracle rank 多为 3/4
```

结论：

> posterior 对有效 trajectory modes 基本没有建模，属于真正的 mode discovery / prototype collapse / semantics mismatch。

下一步优先：

```text
Plan Transformer Posterior
更强 temporal encoding
prototype stabilization
```

---

## Case C：mode 1/2 soft usage 有一定质量，但 q_oracle 仍低

说明 posterior probability mass 虽然没有完全 collapse，但并没有分配给“对应真实 oracle trajectory mode”。

结论：

> 更像 mode identity / trajectory semantics mismatch，而不是简单 mode utilization 问题。

下一步优先：

```text
posterior ↔ trajectory responsibility alignment
```

---

## Case D：posterior 错误样本 margin 很大

即：

```text
posterior 高置信度选错 mode
```

结论：

> 不是简单 temperature 问题，posterior semantics 本身有偏差。

不要只做 temperature tuning。

---

## Case E：posterior 错误样本 margin 普遍很小

结论：

> posterior 多数时候只是决策边界模糊；后续可以优先尝试 mode alignment、sharpening 或更稳定的 prototype，而不是先大改网络。

---

# 12. 实验完成标准

只有以下内容全部完成，本实验才算结束：

- [ ] 直接加载当前 JAAD K=4 checkpoint，不重新训练；
- [ ] validation 与 test 都运行；
- [ ] 输出 posterior hard usage；
- [ ] 输出 posterior soft usage；
- [ ] 输出 prior soft usage；
- [ ] 输出 oracle usage；
- [ ] 输出 q_oracle；
- [ ] 输出 oracle rank distribution；
- [ ] 输出 posterior margin；
- [ ] 输出 oracle probability gap；
- [ ] 输出 per-mode ΔADE / ΔFDE；
- [ ] crossing=0/1 分组完成；
- [ ] mode 1/2 子集单独报告；
- [ ] `posterior_diagnostics_summary.json` 生成；
- [ ] `posterior_diagnostics_samples.csv` 生成；
- [ ] `posterior_diagnostics_report.md` 生成；
- [ ] 单元测试通过；
- [ ] 不修改模型权重；
- [ ] 不基于 test 结果调任何参数。

---

# 13. 给 Codex 的最终执行指令

> 当前任务只是在已有 `MODE_SELECTION_DIAGNOSTIC_EXPERIMENT` 基础上补充一个低成本 FutureModePosterior 诊断，不重新训练模型，也不修改任何模型结构或 loss。使用与上一轮 JAAD mode-selection diagnostic 完全相同的 K=4 checkpoint，优先扩展现有 `diagnose_mode_selection.py`。对每个样本已有的 `q`、`pi`、`ADE_per_mode`、`FDE_per_mode`、`k_oracle`、`k_post`、`k_prior`，新增计算：`q_oracle = q[k_oracle]`、posterior hard usage、posterior soft usage、prior soft usage、oracle usage、oracle mode posterior rank、posterior top1-top2 margin、oracle probability gap，以及每个 mode 成为 oracle 时相对 prior top-1 的平均 ΔADE / ΔFDE。分别对 all、crossing=0、crossing=1、oracle_mode=0/1/2/3 分组汇总。重点单独报告 mode 1/2。不要使用 temperature recalibration 或任何新 posterior 结构；所有统计必须基于 checkpoint 原始 q / pi。先运行 validation，再原样运行 test。最终生成 `posterior_diagnostics_summary.json`、`posterior_diagnostics_samples.csv`、`posterior_diagnostics_report.md`，并补充相应单元测试。实验完成后不要自动修改模型，只给出数据与判读。