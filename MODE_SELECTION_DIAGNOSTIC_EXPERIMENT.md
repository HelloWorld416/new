# EgoJEPA-MP：Mode Selection Diagnostic Experiment

> 目标：**先不修改 FutureModePosterior、Past Mode Prior、trajectory loss 或 JEPA 训练策略**，只对当前已经训练好的 K-mode 模型做诊断，回答当前 `minADE@4 < top-1 ADE` 的 gap 主要来自哪里。
>
> 本实验对应后续改造前的第一优先级验证：把误差拆成 **mode-definition / routing gap** 与 **past-prior selection gap**，再决定下一步应优先改 FutureModePosterior、Past Mode Prior，还是 trajectory specialization。

---

## 1. 核心问题

当前模型已经观察到：

```text
minADE@4 < top-1 ADE
```

这说明 K 个候选轨迹中存在比 prior top-1 更准确的轨迹，但仅凭这一结果无法判断问题出在：

1. **FutureModePosterior 定义的 mode 与真实轨迹几何并没有对齐**；
2. **FutureModePosterior 的 mode 是合理的，但 Past Mode Prior 没有从历史 context 选中它**；
3. 两者都存在问题；
4. 或者 mode selection 已经较好，真正瓶颈是 K 个候选本身的表达能力。

本实验只做诊断，不改模型结构，不重新设计 loss，不基于测试集调参。

---

## 2. 三个必须计算的轨迹指标

对每个样本 `i`、每个 mode `k`，先使用**现有 bbox 解码与 ADE 实现**计算：

```math
ADE_{i,k}
```

然后定义三个 mode index。

### 2.1 Oracle mode

使用真实未来轨迹，仅用于诊断：

```math
k_i^{oracle}=\arg\min_k ADE_{i,k}
```

对应：

```math
ADE_{oracle}=\frac{1}{N}\sum_i \min_k ADE_{i,k}
```

这应当与当前实现的 `minADE@K` 完全一致。

> 注意：必须是 **先对每个样本在 K 个 mode 中取最小 ADE，再对样本求均值**，不能先对每个 mode 求 dataset mean ADE 再取最小值。

---

### 2.2 Future-posterior mode

使用训练时的 FutureModePosterior：

```math
k_i^{post}=\arg\max_k q_i(k\mid Y_{future})
```

其中 `q` 必须来自当前 checkpoint 中的 FutureModePosterior，并使用该样本的 **future motion target** 计算。

对应：

```math
ADE_{posterior}
=
\frac{1}{N}\sum_i ADE_{i,k_i^{post}}
```

这个指标只用于诊断，因为它使用了真实 future 信息，**不是部署指标**。

---

### 2.3 Past-prior mode

使用真正测试时可用的 Past Mode Prior：

```math
k_i^{prior}=\arg\max_k \pi_i(k\mid C_{past})
```

对应：

```math
ADE_{prior}
=
\frac{1}{N}\sum_i ADE_{i,k_i^{prior}}
```

它应该与当前 top-1 trajectory ADE 的定义一致。

---

## 3. 两个关键 gap

### 3.1 Mode-definition / routing gap

```math
G_{mode}=ADE_{posterior}-ADE_{oracle}
```

含义：

> 即使已经看到了真实 future，FutureModePosterior 选择的 behavioral mode 距离“几何上最适合这条 future 的 mode”还有多远。

如果 `G_mode` 大，优先怀疑：

- FutureModePosterior 太弱；
- temporal pooling 丢失 future dynamics；
- prototype / posterior mode 语义不稳定；
- posterior mode 与 trajectory specialization 没有对齐；
- 当前 trajectory loss 让多个 mode 同时向同一 GT 收缩。

---

### 3.2 Prior-selection gap

```math
G_{prior}=ADE_{prior}-ADE_{posterior}
```

含义：

> 在当前 FutureModePosterior 定义下，Past Mode Prior 从过去 context 恢复 realized future mode 的能力有多强。

如果 `G_prior` 大且为正，优先怀疑：

- `context[:,0]` + MLP 的 prior readout 不够；
- prior 没充分利用 visual / motion / ego tokens；
- prior 与 posterior 的训练目标不稳定；
- posterior 本身跨 epoch 漂移，使 prior 在追逐移动目标。

`G_prior` **也可能为负数**。这不应被当成程序错误：如果 prior 选择的轨迹几何上反而比 posterior argmax 更准，说明 posterior mode semantics 与 trajectory accuracy 没有很好对齐。

---

### 3.3 恒等关系

实现时验证：

```math
ADE_{prior}-ADE_{oracle}=G_{mode}+G_{prior}
```

数值误差范围内应成立。

---

## 4. FDE 同步计算

ADE 是本实验的主判据，但同步报告 FDE，避免只看平均轨迹误差。

对每个样本和 mode 计算：

```math
FDE_{i,k}
```

使用**同一套由 ADE 定义的 mode index**报告：

```text
FDE_oracle_by_ADE
FDE_posterior
FDE_prior
```

同时可以额外报告真正的：

```text
minFDE@K
```

但要明确：`minFDE@K` 使用的 oracle mode 可能与 `minADE@K` 不同，不能混为一个选择规则。

---

## 5. 必须增加的 agreement 指标

除了三个 ADE，还需要直接检查 mode index 是否一致。

### 5.1 Posterior vs Oracle agreement

```math
A_{post-oracle}
=
\frac{1}{N}\sum_i
\mathbf{1}[k_i^{post}=k_i^{oracle}]
```

### 5.2 Prior vs Posterior agreement

```math
A_{prior-post}
=
\frac{1}{N}\sum_i
\mathbf{1}[k_i^{prior}=k_i^{post}]
```

### 5.3 Prior vs Oracle agreement

```math
A_{prior-oracle}
=
\frac{1}{N}\sum_i
\mathbf{1}[k_i^{prior}=k_i^{oracle}]
```

同时输出三个 `K x K` confusion matrix：

```text
posterior_mode vs oracle_mode
prior_mode vs posterior_mode
prior_mode vs oracle_mode
```

---

## 6. Posterior / Prior uncertainty 指标

对每个样本记录：

```math
H(q_i)=-\sum_k q_{i,k}\log(q_{i,k}+\epsilon)
```

```math
H(\pi_i)=-\sum_k \pi_{i,k}\log(\pi_{i,k}+\epsilon)
```

以及：

```text
posterior_max_prob = max(q)
prior_max_prob     = max(pi)
```

汇总：

```text
posterior_entropy_mean
prior_entropy_mean
posterior_max_prob_mean
prior_max_prob_mean
```

可选但推荐额外输出：

```math
ADE_{post-exp}
=
\frac{1}{N}\sum_i\sum_k q_{i,k}ADE_{i,k}
```

```math
ADE_{prior-exp}
=
\frac{1}{N}\sum_i\sum_k \pi_{i,k}ADE_{i,k}
```

这两个 expected ADE 只作为校准补充，不替代三个主指标。

---

## 7. 按 crossing label 分组

JAAD / PIE 的 crossing 分布差异很大，因此至少分别报告：

```text
all
crossing=0
crossing=1
```

每组都计算：

```text
ADE_oracle
ADE_posterior
ADE_prior
G_mode
G_prior
posterior-oracle agreement
prior-posterior agreement
prior-oracle agreement
```

不要使用测试集的这些结果调阈值或训练超参数；分组分析仅用于机制诊断。

---

## 8. 实验数据与 checkpoint

### 8.1 第一优先级

直接使用当前已经完成训练并产生 `minADE@4` 与 top-1 ADE 的 **同一个 K=4 checkpoint**。

不要为了本诊断重新训练模型。

### 8.2 Split

先跑：

```text
validation split
```

用于机制判断。

然后原样跑：

```text
test split
```

只作为确认，不基于 test 结果修改实验定义。

### 8.3 Checkpoint 一致性

必须保证三种轨迹：

```text
oracle
posterior-selected
prior-selected
```

全部来自**同一个 checkpoint、同一次 forward 产生的 K 个 trajectory modes**。

禁止跨 checkpoint 拼接。

---

## 9. Codex 实现要求

### 9.1 先读代码，不假设函数名

Codex 先定位当前 EgoJEPA 代码中的：

1. multimodal forward 输出；
2. `mode_probs / prior_probs`；
3. FutureModePosterior 输出 `q` 的位置；
4. K 条 bbox trajectory 的输出；
5. bbox decode 函数；
6. ADE / FDE 实现；
7. 当前 `minADE@K` 与 top-1 ADE 实现；
8. validation / test dataloader；
9. checkpoint loading；
10. sample id / video id / pedestrian id / crossing label 可用字段。

**不要复制一套新的几何解码或 ADE 实现。** 必须复用现有项目中的 decode 与 metric 代码，避免诊断指标与正式指标定义不一致。

---

### 9.2 推荐新增脚本

优先新增一个独立诊断入口，例如：

```text
src/egojepa_diffusion/diagnose_mode_selection.py
```

如果项目结构不同，可放在现有 diagnostics 模块附近。

CLI 至少支持：

```text
--config
--checkpoint
--split val|test
--output-dir
```

可选：

```text
--dataset jaad|pie
--batch-size
--num-workers
```

不要改正常 `evaluate.py` 的默认行为，除非只增加 backward-compatible 的可选辅助函数。

---

### 9.3 Forward 规则

对每个 batch：

1. 使用过去输入正常计算 context；
2. 得到 Past Mode Prior：`pi`；
3. 得到 K 组 predicted future latent；
4. 解码得到 K 条未来 bbox；
5. 使用 future GT bbox 计算每个 mode 的 ADE / FDE；
6. 在 `torch.no_grad()` 下，用当前 FutureModePosterior 和 **future motion target** 计算 `q`；
7. 分别得到 `k_oracle`、`k_post`、`k_prior`；
8. 累积 sample-level diagnostics。

关键限制：

> `q` 只能用于 `posterior-selected` 诊断，绝不能进入 prior/top-1 的预测路径。

---

## 10. 必须验证 mode index 对齐

当前设计假设 posterior prototype 的 `k` 与 decoder 的 mode embedding `k` 是同一个 mode identity。

Codex 必须检查代码确认：

```text
posterior q[..., k]
mode embedding k
decoded trajectory k
```

三者 index 一致。

如果当前实现中存在 reorder / sorting / filtering / top-k 重排，必须在诊断前统一索引并在报告中说明。

如果无法证明 index 一致，停止实验并先报告该问题，不要输出误导性的 gap。

---

## 11. 当前 minADE@K 实现一致性检查

Codex 必须增加一个断言或测试，验证：

```math
current\_minADE@K
\approx
\frac{1}{N}\sum_i\min_k ADE_{i,k}
```

若不一致，先检查是否出现：

- 先 dataset mean 再 mode min；
- batch-wise min 后错误平均；
- bbox / center 坐标不一致；
- normalized coordinate 与 pixel coordinate 混用；
- trajectory mask / padding 处理不一致。

在解决定义一致性前，不继续解释 gap。

---

## 12. 输出文件

建议输出到：

```text
outputs/<experiment>/mode_selection_diagnostics/<split>/
```

至少生成：

### 12.1 `summary.json`

格式示例：

```json
{
  "dataset": "JAAD",
  "split": "val",
  "checkpoint": "...",
  "num_modes": 4,
  "num_samples": 571,
  "ade_oracle": 0.0,
  "ade_posterior": 0.0,
  "ade_prior": 0.0,
  "g_mode": 0.0,
  "g_prior": 0.0,
  "fde_oracle_by_ade": 0.0,
  "fde_posterior": 0.0,
  "fde_prior": 0.0,
  "min_fde_at_k": 0.0,
  "agreement_post_oracle": 0.0,
  "agreement_prior_post": 0.0,
  "agreement_prior_oracle": 0.0,
  "posterior_entropy_mean": 0.0,
  "prior_entropy_mean": 0.0,
  "posterior_max_prob_mean": 0.0,
  "prior_max_prob_mean": 0.0
}
```

并包含 `crossing=0/1` 分组结果。

---

### 12.2 `samples.csv`

每个样本至少保存：

```text
sample_id
video_id（若有）
pedestrian_id（若有）
crossing_label
oracle_mode
posterior_mode
prior_mode
ade_oracle
ade_posterior
ade_prior
fde_oracle_by_ade
fde_posterior
fde_prior
posterior_entropy
prior_entropy
posterior_max_prob
prior_max_prob
q_0 ... q_K-1
pi_0 ... pi_K-1
ade_mode_0 ... ade_mode_K-1
fde_mode_0 ... fde_mode_K-1
```

如果某些 ID 当前 dataset 不提供，不要伪造，留空并在报告中注明。

---

### 12.3 `report.md`

自动生成一个便于人工阅读的 Markdown 报告，至少包含：

| Dataset/Split | Oracle ADE | Posterior ADE | Prior ADE | G_mode | G_prior | Post↔Oracle | Prior↔Post | Prior↔Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

以及 crossing=0/1 子表。

Markdown 中数学公式统一使用 GitHub-safe 的 `math` fenced block，不使用 `$$`、`\operatorname` 等已知存在渲染兼容问题的写法。

---

## 13. 建议增加的可视化

不是本实验完成的硬要求，但若工作量很小，推荐生成：

```text
confusion_prior_vs_posterior.png
confusion_posterior_vs_oracle.png
confusion_prior_vs_oracle.png
```

以及：

```text
prior_entropy_vs_selection_error.png
posterior_entropy_vs_mode_gap.png
```

其中 selection error 可定义为样本级：

```math
ADE_{i,prior}-ADE_{i,oracle}
```

mode gap 可定义为：

```math
ADE_{i,post}-ADE_{i,oracle}
```

不要为画图引入新的模型依赖。

---

## 14. 单元测试

至少新增以下测试。

### 14.1 Per-sample oracle 语义

构造 2 个样本、3 个 modes，使两个样本的最佳 mode 不同，验证：

```text
mean(min_per_sample_ADE)
```

而不是：

```text
min(mean_per_mode_ADE)
```

---

### 14.2 Mode selection

人工构造：

```text
q
pi
ADE_per_mode
```

验证：

```text
oracle_mode = argmin ADE
posterior_mode = argmax q
prior_mode = argmax pi
```

及三个 ADE 聚合结果正确。

---

### 14.3 Gap identity

验证数值上：

```math
(ADE_{prior}-ADE_{oracle})
\approx
G_{mode}+G_{prior}
```

---

### 14.4 Future leakage

确保改变 future motion / future bbox 时：

- `posterior_mode` 可以变化；
- `prior_mode` 在 past 输入不变时不能变化。

该测试用于防止 future 信息错误进入 Past Mode Prior 路径。

---

### 14.5 K=1 sanity check

当 `K=1` 时应满足：

```text
ADE_oracle == ADE_posterior == ADE_prior
G_mode == 0
G_prior == 0
all agreements == 1
```

允许浮点误差。

---

## 15. 统计稳定性

如果当前 checkpoint 只有单种子，本诊断先对该 checkpoint 完成，不要求为了诊断重新训练多种子。

但对样本级指标推荐使用 bootstrap 估计 95% CI，优先按：

```text
video
或 pedestrian identity
```

分组重采样，而不是把高度重叠的滑动窗口全部视为独立样本。

若现有数据结构不方便实现 grouped bootstrap，可先输出 point estimate，并在报告里明确“未计算分组置信区间”。不要临时引入错误的 IID window bootstrap 冒充视频级置信区间。

---

## 16. 结果判读规则

完成实验后按下面逻辑解释。

### Case A：`G_mode` 明显大，`G_prior` 小

主要瓶颈：

```text
FutureModePosterior / trajectory routing / trajectory specialization
```

下一步优先验证：

- Plan Transformer posterior；
- attention pooling；
- prototype 稳定性；
- sharpened / hard posterior-guided trajectory routing。

---

### Case B：`G_mode` 小，`G_prior` 明显大

主要瓶颈：

```text
Past Mode Prior
```

下一步优先验证：

- full-context prior；
- mode-query cross-attention；
- visual / motion / ego context 的 mode-specific evidence readout；
- prior/posterior target stability。

---

### Case C：两者都明显大

说明：

```text
future mode semantics 尚未稳定
+
past mode inference 也没有学好
```

先解决 FutureModePosterior / trajectory alignment，再增强 prior，避免 prior 去拟合一个不稳定 target。

---

### Case D：两者都小，但 `ADE_oracle` 仍然较差

说明 selection 不是主要瓶颈，下一步应检查：

```text
K 个 trajectory modes 本身的表达能力
future latent quality
trajectory head capacity
JEPA representation quality
```

而不是继续增强 prior。

---

### Case E：`G_prior < 0`

说明 prior 选择的 trajectory 几何上反而优于 FutureModePosterior 的 argmax mode。

这通常意味着：

```text
FutureModePosterior 学到的 behavioral mode
并没有和 trajectory-specialized mode 充分对齐
```

此时不要简单把 prior 视为“更好”，而应重点检查 posterior mode semantics 与 trajectory loss routing。

---

## 17. 完成标准

本实验只有在以下条件全部满足时才算完成：

- [ ] 不改变现有模型结构或训练 loss；
- [ ] 不重新训练即可对当前 K=4 checkpoint 运行；
- [ ] 验证当前 `minADE@K` 定义正确；
- [ ] 确认 posterior / decoder / trajectory mode index 对齐；
- [ ] 输出 `ADE_oracle`；
- [ ] 输出 `ADE_posterior`；
- [ ] 输出 `ADE_prior`；
- [ ] 输出 `G_mode`；
- [ ] 输出 `G_prior`；
- [ ] 输出三种 agreement；
- [ ] 输出 posterior/prior entropy；
- [ ] 输出 crossing=0/1 分组；
- [ ] 输出 FDE 对应诊断；
- [ ] 输出 `summary.json`；
- [ ] 输出 `samples.csv`；
- [ ] 输出 `report.md`；
- [ ] K=1 sanity test 通过；
- [ ] future leakage test 通过；
- [ ] val 先完成，再原样运行 test；
- [ ] 不使用 test 结果调参。

---

# 18. 给 Codex 的最终执行指令

> 先不要实现 Plan Transformer Posterior、Mode-query Cross Attention Prior、sharpened trajectory routing、A1/A2 pretraining 或新的 checkpoint selection。当前任务仅对现有已经训练好的 K-mode EgoJEPA 做 mode-selection diagnostic。先阅读现有模型、evaluation、metrics、dataset、diagnostics 和 checkpoint loading 代码，确认 FutureModePosterior 的 `q`、Past Mode Prior 的 `pi`、K 个 trajectory outputs 和 mode index 的对应关系。复用现有 bbox decode 与 ADE/FDE 实现，新增独立诊断脚本，计算 per-sample `k_oracle = argmin_k ADE`、`k_post = argmax_k q`、`k_prior = argmax_k pi`，并汇总 `ADE_oracle`、`ADE_posterior`、`ADE_prior`、`G_mode = ADE_posterior - ADE_oracle`、`G_prior = ADE_prior - ADE_posterior`。同步报告 FDE、三种 mode agreement、posterior/prior entropy、crossing=0/1 分组结果。首先验证现有 `minADE@K` 与 mean-of-per-sample-min 定义一致，并确认 q / mode embedding / trajectory index 对齐；如果任一项不成立，停止并报告问题，不继续解释结果。实验必须能直接加载当前 K=4 checkpoint，不改模型权重，不重新训练。先运行 validation，再原样运行 test，禁止基于 test 调参。最终输出 `summary.json`、`samples.csv`、`report.md` 和相应单元测试结果。
