# EgoJEPA-MP：下一阶段五步实验计划

> 本文面向 Codex 执行，基于当前 JAAD K=4 模型及已完成的 mode-selection / posterior cheap diagnostics。
>
> 当前已知现象：
>
> - `minADE@4 < top-1 ADE`，说明 K 个候选中存在更准确轨迹；
> - 测试集 `prior ↔ posterior` 一致率约 91.96%，说明 Past Mode Prior 已较好拟合当前 posterior；
> - 但 `posterior ↔ oracle` 仅约 61.13%；
> - oracle 最优模式为 mode 1/2 的样本占测试集约 37.27%，但 mode 1/2 合计 posterior soft usage 仅约 0.87%；
> - 在这些样本中，oracle mode 约 74.10% 被 posterior 排到第 3/4 位；
> - 因此当前首要问题不是单纯 prior 不够强，而是 **FutureModePosterior 与 trajectory mode semantics 严重错位，并伴随 posterior 概率塌缩**。
>
> Step 1 已进一步发现：原 soft responsibility 的平均最大概率仅约 `0.262`，非常接近 K=4 均匀分布的 `0.25`；训练后 mode 1/2 soft usage 虽恢复到约 50%，但测试集 `posterior↔oracle` 反而下降，说明该 soft target 主要把 posterior 拉平，并未建立可靠的 trajectory-mode 语义。
>
> 因此在原五个主步骤之间新增 **Step 1B：Responsibility Target Validation**。Step 1B 是进入 Step 2 或 Step 3 之前的强制门槛。

---

# 总体原则

## 必须保持

- `K=4` 暂时固定；
- JAAD 作为第一验证数据集；
- 当前 epoch 88 checkpoint 作为基线；
- 所有新实验使用新的输出目录；
- 不覆盖现有 checkpoint；
- 不基于测试集调参；
- 每一步先看 validation，再按冻结配置跑 test；
- 每一步至少保存 `summary.json`、训练日志、验证指标、测试指标和关键 diagnostics；
- 复用现有 bbox decode、ADE/FDE、intent evaluation 与 mode diagnostics 实现；
- GitHub Markdown 数学统一使用 `math` fenced block，避免 `$$` 与不兼容宏。

## 主判断指标

所有步骤持续报告：

```text
ADE_oracle
ADE_posterior
ADE_prior
G_mode = ADE_posterior - ADE_oracle
G_prior = ADE_prior - ADE_posterior
posterior↔oracle agreement
prior↔posterior agreement
prior↔oracle agreement
minADE@4
top1 ADE
minFDE@4
top1 FDE
posterior hard usage
posterior soft usage
posterior entropy
prior entropy
crossing F1 / AUC / AP
```

特别关注 mode 1/2 的：

```text
hard usage
soft usage
oracle frequency
posterior rank
posterior probability
when-oracle ADE gain
```

---

# Step 1：冻结现有 Decoder，只训练 Posterior Alignment

## 1.1 目的

先回答最关键的问题：

> 当前 FutureModePosterior 的架构本身是否有能力学会“哪个现有 trajectory mode 对这条 future 最有用”？

这一步不改 Plan Encoder，不改 Past Mode Prior，不改 trajectory decoder，不改 task head。

如果只靠一个 alignment target 就能让 posterior 学会利用 mode 1/2，那么说明当前 posterior 架构未必是首要瓶颈，真正缺失的是 posterior 与 trajectory mode 之间的 grounding。

---

## 1.2 冻结模块

加载当前 K=4 checkpoint 后冻结：

```text
visual/context encoder
motion/context encoder
ego projection
fusion transformer
future predictor / decoder
mode embeddings
trajectory head
intention head
Past Mode Prior
EMA target encoders
Dual-target projectors（若本实验不需要）
```

只允许训练：

```text
FutureModePosterior
posterior prototypes（若当前实现属于该模块）
```

Codex 必须打印并保存 trainable parameter list，确保没有误解冻。

---

## 1.3 构造 trajectory responsibility

对每个样本、每个 mode，使用**现有 trajectory output 与 GT future trajectory** 计算 mode error。

优先使用与训练轨迹头一致的 encoded-box SmoothL1：

```math
E_k = SmoothL1(B^{(k)}, B^{gt})
```

同时保留基于 pixel ADE 的诊断版本：

```math
E^{ADE}_k = ADE(B^{(k)}, B^{gt})
```

原 Step 1 定义 soft trajectory responsibility：

```math
r_k=
\frac{\exp(-E_k/\tau_r)}
{\sum_j \exp(-E_j/\tau_r)}
```

原始搜索：

```text
tau_r = 0.5, 1.0, 2.0
```

调参仅使用 validation。

必须对 `r` 使用 stop-gradient。

---

## 1.4 Alignment loss

Posterior 输出：

```math
q_k=q(M=k\mid Y_{future})
```

训练：

```math
L_{align}
=
-\sum_k r_k \log(q_k + \epsilon)
```

或者等价 KL：

```math
L_{align}=KL(stopgrad(r)\,\|\,q)
```

第一版优先 cross-entropy 形式，简单稳定。

---

## 1.5 训练配置建议

只训练 posterior，因此训练应很轻量：

```text
epochs: 5-20
optimizer: AdamW
lr: 1e-4 / 3e-4 / 1e-3 做小范围验证
weight_decay: 1e-4
grad_clip: 1.0
```

不要改其他模型权重。

---

## 1.6 必须比较

Baseline：当前 epoch 88 checkpoint，不训练。

Alignment-only：冻结 decoder，仅训练 posterior。

主表：

| Method | Posterior↔Oracle | G_mode | mode1/2 soft usage | mode1/2 hard usage | Posterior ADE | Oracle ADE |
|---|---:|---:|---:|---:|---:|---:|
| current | ... | ... | ... | ... | ... | ... |
| + posterior alignment | ... | ... | ... | ... | ... | ... |

---

## 1.7 Step 1 当前结论

Step 1 已完成，关键现象包括：

- 原 soft responsibility 在所选温度下平均最大概率约 `0.262`，接近四模式均匀分布 `0.25`；
- mode 1/2 soft usage 从约 1% 恢复到约 50%，但这更像被均匀 target 拉平，而不是学会正确语义；
- 测试集 `posterior↔oracle` 未提升，反而下降；
- 新增的 mode 1/2 hard selection 中，只有少部分与 oracle 一致。

因此 Step 1 **不能证明 simple posterior capacity insufficient，也不能证明 alignment idea 无效**。首先需要验证更有判别力的 responsibility target。

---

# Step 1B：Responsibility Target Validation —— R1 Hard Oracle vs R2 Label-Smoothed Oracle

> **强制执行。** 在进入 Step 2 的 sharpened routing 或 Step 3 的 Plan Transformer 之前，必须完成本步。
>
> 本步只回答一个问题：
>
> **在 decoder、future representation、trajectory heads 全部冻结的情况下，当前 simple FutureModePosterior / prototypes 是否有能力从 future representation 识别“哪个既有 trajectory mode 对该 future 最有用”？**

---

## 1B.1 为什么需要 Step 1B

Step 1 的 soft responsibility：

```math
r_k \propto \exp(-E_k/\tau_r)
```

在当前 loss scale 下几乎均匀，导致 target 本身没有足够判别信息。

对于 K=4：

```text
uniform max probability = 0.25
observed mean max responsibility ≈ 0.262
```

因此本步不再继续搜索同一种 softmax temperature，而直接比较两种清晰、可解释的 responsibility：

```text
R1: Hard oracle
R2: Label-smoothed oracle
```

不比较第三种 normalized-soft responsibility，避免同时引入新的尺度设计。

---

## 1B.2 冻结规则

加载与 Step 1 相同的 JAAD epoch 88 K=4 checkpoint。

必须冻结：

```text
visual/context encoder
motion/context encoder
ego projection
fusion transformer
future predictor / decoder
mode embeddings
trajectory head
intention head
Past Mode Prior
EMA target encoders
dual-target projectors
```

只允许训练：

```text
FutureModePosterior 当前可训练参数
posterior prototypes
```

当前实现若 FutureModePosterior 实际只有 prototypes 可训练，则保持这一事实，不要额外增加 MLP、Transformer 或 classifier。

必须保存：

```text
trainable_parameters.txt
```

并断言其余参数在训练前后逐位或容差范围内不变。

---

## 1B.3 Oracle mode 的统一定义

本步 responsibility 必须以**轨迹几何 oracle**为基准，而不是 encoded-box SmoothL1 的 argmin，避免训练目标与最终诊断 `ADE_oracle` 定义不一致。

对每个样本：

```math
k^{oracle}
=
\arg\min_k ADE(B^{(k)},B^{gt})
```

其中 ADE 必须复用现有正式 pixel-center ADE 实现。

要求：

```text
mean_i ADE(i, k_oracle(i)) == 当前 minADE@4
```

在浮点误差范围内成立。

如果存在完全相同 ADE 的 tie：

- 使用稳定的最小 index tie-break；
- 记录 tie rate；
- 不随机选 mode。

---

## 1B.4 R1：Hard Oracle Responsibility

R1 使用 one-hot trajectory oracle：

```math
r_k^{R1}
=
\mathbf{1}[k=k^{oracle}]
```

Alignment loss：

```math
L_{R1}
=
-\log(q_{k^{oracle}}+\epsilon)
```

等价于对 oracle mode 做 4 类交叉熵。

### R1 的意义

R1 是一个**capacity upper-bound diagnostic**：

> 如果只训练当前 posterior/prototypes，连明确的 oracle mode label 都无法从 future representation 学出来，那么下一步才有充分理由怀疑 simple pooling / prototype classifier 的表达能力不足，并进入 Step 3。

R1 不是最终方法默认 supervision，也不是 inference-time 可用信息；它只用于诊断和 grounding feasibility。

---

## 1B.5 R2：Label-Smoothed Oracle Responsibility

R2 保留 oracle 的主语义，但避免绝对 one-hot target。

对于 K=4：

```math
r_{k^{oracle}}^{R2}=1-\epsilon_{ls}
```

其他 mode：

```math
r_{k\neq k^{oracle}}^{R2}
=
\frac{\epsilon_{ls}}{K-1}
```

默认首先测试：

```text
epsilon_ls = 0.10
```

validation 可额外比较：

```text
epsilon_ls = 0.05, 0.10, 0.20
```

但**只能在 validation 选择 epsilon**，选定后固定到 test。

Alignment loss：

```math
L_{R2}
=
-\sum_k r_k^{R2}\log(q_k+\epsilon)
```

### R2 的意义

R2 回答：

> 在明确告诉 posterior 哪个 trajectory mode 是最优的同时，保留少量不确定性，是否比硬 one-hot 有更好的 val→test 泛化与概率校准？

---

## 1B.6 公平实验矩阵

必须比较三个设置：

| ID | Responsibility | Decoder | Posterior architecture | Trainable params |
|---|---|---|---|---|
| B0 | none，原 epoch88 | frozen | current simple | 0 |
| R1 | Hard oracle | frozen | current simple | posterior/prototypes only |
| R2 | Label-smoothed oracle | frozen | current simple | posterior/prototypes only |

R1 与 R2 必须使用：

```text
相同初始化 checkpoint
相同 optimizer
相同 epoch budget
相同 lr search space
相同 batch size
相同 validation selection rule
```

不要让 R2 因额外超参获得更多 test-time selection 机会。

---

## 1B.7 训练超参数

建议保持 Step 1 的轻量设置：

```text
epochs: 20
optimizer: AdamW
lr candidates: 1e-4, 3e-4, 1e-3
weight_decay: 1e-4
grad_clip: 1.0
```

每组仅使用 validation 选择最佳 epoch / lr。

推荐主选模指标按以下优先级：

1. `ADE_posterior` 最低；
2. 若接近，则 `posterior↔oracle agreement` 更高；
3. 再看 mode usage 是否合理。

禁止使用 test 结果选择 R1/R2 或超参。

---

## 1B.8 必须输出的核心指标

validation 和 test 都必须输出：

```text
ADE_oracle
ADE_posterior
G_mode
posterior↔oracle agreement
posterior hard usage per mode
posterior soft usage per mode
posterior entropy
posterior max-prob mean
q_oracle mean
q_oracle median
oracle rank distribution
mode1/2 hard usage
mode1/2 soft usage
mode1/2 oracle frequency
mode1/2 q_oracle
```

另外输出：

```text
crossing=0 分组
crossing=1 分组
```

重点比较：

```text
R1 vs R2 的 val→test 泛化差
R1/R2 是否真正把 mode1/2 的 hard usage 拉向其 oracle frequency
q_oracle 是否显著提高
posterior-oracle agreement 是否显著提高
G_mode 是否显著下降
```

---

## 1B.9 必须新增“语义恢复”表

报告：

| Method | Oracle ADE | Posterior ADE | G_mode | Post↔Oracle | mode1/2 Oracle Freq | mode1/2 Hard Usage | mode1/2 Soft Usage | q_oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | ... | ... | ... | ... | ... | ... | ... | ... |
| R1 Hard oracle | ... | ... | ... | ... | ... | ... | ... | ... |
| R2 Label-smoothed | ... | ... | ... | ... | ... | ... | ... | ... |

其中 `Oracle ADE` 因 decoder 冻结，理论上三行应一致；若不一致，先检查实现，不解释结果。

---

## 1B.10 Step 1B 的判定逻辑

### Case A：R1 明显成功，R2 同样成功或更稳

典型表现：

```text
posterior↔oracle 大幅提升
G_mode 明显下降
ADE_posterior 明显靠近 ADE_oracle
mode1/2 hard usage 接近其 oracle frequency
q_oracle 显著提高
```

结论：

> 当前 future representation + simple posterior/prototype capacity 基本足够；Step 1 失败的主因是 soft responsibility target 无判别力。

下一步：

```text
优先进入 Step 2
```

正式研究 alignment + sharpened trajectory routing。

主方法候选责任目标优先考虑 R2，因为它比 hard oracle 更平滑；R1 保留为 capacity upper-bound / 消融。

---

### Case B：R1 成功，但 R2 明显弱于 R1

结论：

> 当前 posterior 可以学 trajectory semantics，但对 target softness 较敏感。

下一步：

- 仍可进入 Step 2；
- Step 2 先以 R1 或更小 `epsilon_ls` 的 R2 做受控实验；
- 不要立即升级 Plan Transformer。

---

### Case C：R1 都无法明显提高 posterior↔oracle / 降低 G_mode

结论：

> 在 frozen representation 下，当前 simple posterior/prototype 无法可靠解码 trajectory oracle mode。

下一步：

```text
优先进入 Step 3：Temporal Plan Encoder
```

此时再引入更强 future temporal representation 才有实验证据支持。

不要先进入 Step 2 的 sharpened routing，因为 routing target 本身尚未被 posterior 学会。

---

### Case D：validation 大幅改善但 test 明显退化

结论：

> posterior capacity 可能存在，但当前 oracle supervision / prototype-only fitting 泛化不足。

下一步：

- 比较 R2 是否优于 R1；
- 检查 train/val/test oracle mode distribution；
- 检查视频级 bootstrap CI；
- 不根据 test 重新调 epsilon/lr；
- 暂不进入复杂联合训练。

---

## 1B.11 必须新增的单元测试

```text
test_hard_oracle_responsibility_one_hot
test_label_smoothed_responsibility_sum_to_one
test_label_smoothed_oracle_has_max_probability
test_oracle_mode_matches_minade_mode
test_r1_r2_alignment_stop_gradient_target
test_step1b_only_posterior_trainable
test_frozen_decoder_outputs_identical_before_after
test_oracle_ade_identical_across_b0_r1_r2
```

K=1 sanity：

```text
R1 responsibility = [1]
R2 responsibility = [1]
G_mode = 0
posterior↔oracle = 1
```

---

## 1B.12 Step 1B 输出目录

建议：

```text
outputs/<base_exp>/step1b_responsibility/
  B0_baseline/
  R1_hard_oracle/
  R2_label_smooth_eps005/
  R2_label_smooth_eps010/
  R2_label_smooth_eps020/
  comparison.json
  comparison.md
```

每个训练目录保存：

```text
config_resolved.yaml
train.log
best_posterior.pt
val_summary.json
test_summary.json
mode_diagnostics_val.json
mode_diagnostics_test.json
```

只有 validation 选中的 R1 / R2 配置才允许跑 test。

---

# Step 2：正式加入 Posterior–Trajectory Alignment，并改为 Sharpened Trajectory Routing

> **仅当 Step 1B 证明至少 R1 或 R2 可以显著学习 trajectory oracle semantics 时，优先执行本步。**
>
> 若 Step 1B 的 R1 都失败，则先跳到 Step 3，再回来做 Step 2。

## 2.1 目的

解决当前软 q-weighted trajectory loss 可能形成的 mode averaging / positive feedback collapse：

```text
q_k 低
→ 该 mode trajectory supervision 弱
→ specialization 更差
→ posterior 更不愿选择该 mode
→ q_k 更低
```

---

## 2.2 保留 alignment

联合训练时加入：

```math
L_{align}=CE(stopgrad(r),q)
```

其中 `r` 不再默认使用 Step 1 的近均匀 softmax responsibility。

优先依据 Step 1B 结果选择：

```text
R2 Label-smoothed oracle 作为主候选
R1 Hard oracle 作为上界/消融
```

总 task loss 中先使用小权重：

```text
lambda_align = 0.05 / 0.1 / 0.25
```

只在 validation 搜索。

---

## 2.3 Sharpened posterior routing

当前 trajectory loss 若为：

```math
L_{traj}=\sum_k q_k L_k
```

改成：

```math
\tilde{q}_k
=
\frac{q_k^{1/\tau_{traj}}}
{\sum_j q_j^{1/\tau_{traj}}}
```

并：

```math
L_{traj}
=
\sum_k stopgrad(\tilde{q}_k)L_k
```

其中 `L_k` 仍使用当前 encoded-box SmoothL1。

---

## 2.4 Temperature annealing

建议：

```text
训练前期: tau_traj = 1.0
训练中期: tau_traj = 0.5
训练后期: tau_traj = 0.2
```

第一版可按训练进度线性或分段变化。

必须通过配置可关闭：

```yaml
trajectory_routing:
  type: soft | sharpened
  tau_start: 1.0
  tau_mid: 0.5
  tau_end: 0.2
```

---

## 2.5 可选 hard posterior routing 对照

只作为补充消融：

```math
k^*=\arg\max_k q_k
```

```math
L_{traj}=L_{k^*}
```

注意这不是传统 best-of-K/WTA，因为 mode 由 posterior representation 定义，而不是 `argmin ADE`。

不要把 hard routing 作为第一默认方案。

---

## 2.6 必须监测 mode diversity

至少增加：

```text
pairwise endpoint distance
pairwise trajectory ADE distance
pairwise latent cosine similarity
mode-wise oracle frequency
mode-wise hard usage
```

防止 sharpen 之后只形成数值上分离但不合理的候选。

---

## 2.7 Step 2 成功标准

相对 Step 1B / baseline：

- `minADE@4` 不恶化，最好进一步下降；
- `G_mode` 继续下降；
- mode 1/2 oracle frequency 与 posterior usage 更接近；
- trajectory diversity 增强或至少保持；
- top-1 ADE 改善或不明显恶化；
- intention F1/AUC/AP 不出现明显负迁移。

---

# Step 3：如果简单 Posterior 能力不足，则升级为 Temporal Plan Encoder

## 3.1 触发条件

优先在以下情况执行：

- Step 1B 中 R1 hard oracle 都无法明显降低 `G_mode`；
- mode 1/2 hard usage 仍无法跟随 oracle frequency；
- oracle mode 仍大量 rank 3/4；
- 当前 future representation / simple pooling 无法解码 trajectory oracle semantics。

如果 Step 1B 已经很好，则 Step 3 作为架构消融而不是必选主路线。

---

## 3.2 新 Posterior 结构

将当前简单 temporal pooling 替换为：

```text
future motion EMA tokens [B,30,256]
        ↓
LayerNorm
        ↓
Temporal positional embedding
        ↓
2-layer Transformer encoder
        ↓
learnable PLAN query
        ↓
attention pooling over 30 future tokens
        ↓
plan embedding [B,256]
        ↓
mode prototypes
        ↓
posterior q [B,K]
```

不要改 visual target，不要把 future image 引入 mode posterior。

---

## 3.3 模块建议

新增例如：

```text
FuturePlanEncoder
```

配置：

```yaml
future_mode_posterior:
  type: simple_pool | plan_transformer
  layers: 2
  hidden_dim: 256
  heads: 8
  dropout: 0.1
  pooling: plan_query
```

必须保留旧 `simple_pool` 路径用于公平对照。

---

## 3.4 Prototype 稳定性

第一版不要同时引入复杂 VQ/Sinkhorn/Gumbel。

但必须记录：

```text
prototype cosine drift per epoch
hard mode usage
soft mode usage
posterior entropy
mode permutation / semantic drift diagnostics
```

如果 prototype 明显漂移，可新增 EMA prototype 作为下一层消融，但不要与 Plan Transformer 首次实验同时混入。

---

## 3.5 Step 3 核心对照

| Posterior | Responsibility | Routing | 目的 |
|---|---|---|---|
| simple_pool | R1/R2，frozen decoder | unchanged | Step 1B capacity |
| simple_pool | best R1/R2 | sharpened | Step 2 |
| plan_transformer | same best R1/R2 | sharpened | Step 3 |

判断 Plan Transformer 是否真的额外降低 `G_mode`，而不是仅靠更有信息量的 responsibility 已经解决问题。

---

# Step 4：Posterior 稳定后，再升级 Past Mode Prior

## 4.1 为什么必须放到第四步

当前 prior↔posterior 一致率已经很高，因此在 posterior 错位时优先增强 prior 没有意义。

只有 posterior 的 mode 1/2 真正被利用后，才重新测：

```math
G_{prior}=ADE_{prior}-ADE_{posterior}
```

如果新的 `G_prior` 仍明显，特别是 crossing subgroup 较大，再升级 prior。

---

## 4.2 Baseline Prior

保留当前：

```text
context[:,0]
→ MLP
→ K logits
```

作为对照。

---

## 4.3 新 Prior：Mode-Query Cross Attention

使用 K 个 mode embeddings 作为 query，读取完整 context：

```text
K mode queries [K,D]
        ↓
Cross Attention
        ↓
context memory [B,33,D]
        ↓
mode-specific context [B,K,D]
        ↓
shared/small MLP scorer
        ↓
prior logits [B,K]
```

数学形式：

```math
h_k=CrossAttn(e_k,C,C)
```

```math
s_k=f(h_k)
```

```math
\pi=softmax(s)
```

---

## 4.4 重点观察 crossing subgroup

必须单独报告：

```text
G_prior all
G_prior crossing=0
G_prior crossing=1
prior↔posterior agreement
prior↔oracle agreement
top1 ADE / FDE
intent F1 / AUC / AP
```

如果新 prior 只改善 prior↔posterior agreement，但不改善 `G_prior` / top1 ADE，则不能宣称有效。

---

# Step 5：将 Pretraining 拆成 A1 Mode Discovery + A2 Prior Fitting

## 5.1 背景

当前预训练中：

```text
JEPA reconstruction loss
+ prior KL
+ usage regularization
```

被一个仍在漂移的 posterior target 共同优化，导致：

- prior 在追逐 moving target；
- prior KL 可能主导 checkpoint total loss；
- `best.pt` 可能错误选择早期 epoch；
- representation quality 与 prior fitting quality 被混成一个选模问题。

因此最后一步将二者解耦。

---

## 5.2 Stage A1：Representation + Future Mode Discovery

训练：

```text
online encoders
fusion
future predictor
FutureModePosterior
mode prototypes
mode embeddings
dual-target projectors
```

不训练或冻结 Past Mode Prior。

Loss 建议：

```math
L_{A1}
=
L_{dualJEPA}
+
\lambda_{align}L_{align}
+
\lambda_{usage}L_{usage}
```

其中 alignment responsibility 必须沿用 Step 1B / Step 2 已验证的定义，不能恢复到近均匀的原 Step 1 soft target。

如果 alignment 仅在下游阶段可稳定计算，也可以先不在纯预训练 A1 使用；Codex 必须根据当前数据流确认 trajectory outputs 在 A1 是否可用，再决定是否启用。

A1 选模：

```text
主要依据 val JEPA / dual-target reconstruction
同时要求 mode 不 collapse
```

保存：

```text
best_repr.pt
```

禁止使用 prior KL 参与 `best_repr` 评分。

---

## 5.3 Stage A2：Prior Fitting

加载并冻结 `best_repr.pt` 中：

```text
context representation
FutureModePosterior
prototypes
mode semantics
```

只训练：

```text
Past Mode Prior
```

目标：

```math
L_{A2}=KL(stopgrad(q)\,\|\,\pi)
```

可先完全冻结 context，若 prior 明显不足，再做小范围：

```text
unfreeze final fusion layer
或增加 adapter
```

但必须作为独立消融。

A2 选模：

```text
val prior KL / NLL
prior↔posterior agreement
G_prior（诊断）
```

保存：

```text
best_prior.pt
```

如果工程上需要一个完整模型文件，则保存完整 state_dict，但明确其 representation 权重来自 best_repr。

---

# 推荐执行顺序与停止条件

## 顺序

```text
Step 1（已完成，发现 soft responsibility 近均匀）
→ Step 1B（必须：R1 Hard oracle vs R2 Label-smoothed oracle）
→ Step 2（仅当 Step 1B 证明 simple posterior 能学 trajectory semantics）
→ Step 3（若 R1 都失败，优先于 Step 2；否则作为结构消融）
→ Step 4
→ Step 5
```

## 不允许跳过的依赖

- Step 2 不得继续使用 Step 1 的近均匀 soft responsibility；
- Step 2 必须以 Step 1B 验证过的 R1/R2 结果为依据；
- Step 4 不能在 posterior 仍严重 collapse 时作为主实验；
- Step 5 必须基于已确定的 posterior/routing 设计；
- 不要同时修改 posterior architecture、trajectory routing、prior architecture 和 pretraining schedule 后只报一个结果。

---

# 实验编号建议

> 为避免与 Step 1B 的 responsibility 名称 `R1/R2` 冲突，完整模型实验编号改用 `E*`。

| ID | Posterior | Responsibility / Alignment | Traj Routing | Prior | Pretrain |
|---|---|---|---|---|---|
| E0 | simple | off | current soft-q | state-token MLP | current joint |
| E1-R1 | simple | Hard oracle，decoder frozen | unchanged | frozen | no retrain decoder |
| E1-R2 | simple | Label-smoothed oracle，decoder frozen | unchanged | frozen | no retrain decoder |
| E2 | simple | best R1/R2 | sharpened | state-token MLP | current |
| E3 | Plan Transformer | best R1/R2 | sharpened | state-token MLP | current |
| E4 | best posterior | best alignment | sharpened | mode-query cross-attn | current |
| E5 | best posterior | best alignment | sharpened | best prior | A1 + A2 split |

主论文版本应由 E0→E1-R1/E1-R2→E2→E3→E4→E5 的证据逐步决定，不预设 E5 一定最好。

---

# 每一步统一输出

建议输出目录：

```text
outputs/<exp_name>/
  config_resolved.yaml
  train.log
  val_metrics.json
  test_metrics.json
  mode_diagnostics_val.json
  mode_diagnostics_test.json
  report.md
```

报告必须包含：

```text
ADE_oracle
ADE_posterior
ADE_prior
G_mode
G_prior
minADE@4
top1 ADE
posterior↔oracle
prior↔posterior
prior↔oracle
mode-wise usage
mode-wise oracle frequency
mode-wise ADE gain
crossing/non-crossing subgroup
intent F1/AUC/AP
```

---

# 必须新增 / 保留的测试

Codex 每一步都要保持已有 diagnostics tests 通过，并增加：

```text
test_hard_oracle_responsibility_one_hot
test_label_smoothed_responsibility_sum_to_one
test_label_smoothed_oracle_has_max_probability
test_oracle_mode_matches_minade_mode
test_r1_r2_alignment_stop_gradient_target
test_step1b_only_posterior_trainable
test_frozen_decoder_outputs_identical_before_after
test_oracle_ade_identical_across_b0_r1_r2
test_sharpened_routing_normalization
test_sharpened_routing_temperature_effect
test_hard_routing_optional_path
test_plan_transformer_output_shape
test_mode_query_prior_output_shape
test_a1_prior_frozen
test_a2_repr_frozen
test_best_repr_selection_ignores_prior_kl
```

额外 sanity：

```text
K=1 时 R1/R2 responsibility 都退化为 [1]
K=1 时 routing/align 不改变单模式语义
posterior q / decoder mode / trajectory mode index 始终一致
future 信息绝不能进入 inference-time prior path
```

---

# 给 Codex 的最终执行指令

> 当前 Step 1 已完成，并发现原 soft trajectory responsibility 的平均最大概率约 0.262，接近 K=4 均匀分布 0.25；因此不要直接进入 sharpened routing，也不要立即升级 Plan Transformer。下一任务是强制执行 Step 1B。加载同一个 JAAD K=4 epoch 88 checkpoint，冻结 decoder、context、mode embeddings、trajectory/intention heads、Past Mode Prior、EMA teacher 和 dual projectors，只训练当前 FutureModePosterior / prototypes。统一以 per-sample pixel ADE 的 argmin 定义 `k_oracle`，并验证其聚合结果等于当前 minADE@4。比较两种 responsibility：R1 Hard oracle，使用 one-hot `k_oracle` 做 4 类交叉熵；R2 Label-smoothed oracle，默认 epsilon=0.10，并只在 validation 可比较 0.05/0.10/0.20。R1 与 R2 使用相同 checkpoint、optimizer、epoch budget、lr 搜索空间和选模规则。validation 选择后才运行 test。必须报告 ADE_oracle、ADE_posterior、G_mode、posterior↔oracle、q_oracle、oracle rank、每 mode hard/soft usage、mode1/2 oracle frequency 与 usage，以及 crossing/non-crossing 分组。若 R1 明显成功，则说明 simple posterior capacity 基本足够，Step 1 失败主要来自 responsibility 无判别力，下一步进入 Step 2；若 R1 都无法明显降低 G_mode，则优先进入 Step 3 Temporal Plan Encoder，不要先做 sharpened routing。后续 Step 2 也不得恢复使用原 Step 1 的近均匀 soft responsibility，应以 Step 1B 验证过的 R1/R2 为基础。每个实验使用独立输出目录，不覆盖原 checkpoint，不允许基于 test 调参。
