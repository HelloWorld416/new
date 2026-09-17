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
> 下一阶段严格按下面五步顺序执行。不要把五步一次性混合实现，否则无法判断每个改动的因果贡献。

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

定义 soft trajectory responsibility：

```math
r_k=
\frac{\exp(-E_k/\tau_r)}
{\sum_j \exp(-E_j/\tau_r)}
```

建议初始尝试：

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

## 1.7 Step 1 判定标准

### 若明显成功

满足多数条件：

- `G_mode` 明显下降；
- posterior↔oracle agreement 明显上升；
- mode 1/2 soft/hard usage 恢复；
- `ADE_posterior` 向 `ADE_oracle` 靠近；
- oracle/minADE@4 不变（因为 decoder 冻结）；

则进入 Step 2，并**保留简单 posterior 架构**。

### 若基本学不动

例如：

- mode1/2 soft usage 仍接近 0；
- posterior↔oracle 几乎不变；
- `G_mode` 几乎不降；

则记录为“simple posterior capacity insufficient”，在 Step 3 优先升级 Plan Encoder。

---

# Step 2：正式加入 Posterior–Trajectory Alignment，并改为 Sharpened Trajectory Routing

> 仅当 Step 1 证明 alignment 有效时，优先执行本步；若 Step 1 明确失败，可先跳到 Step 3，再回来做 Step 2。

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
L_{align}=KL(stopgrad(r)\,\|\,q)
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

相对 Step 1 / baseline：

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

- Step 1 alignment-only 无法明显降低 `G_mode`；
- mode 1/2 soft usage 仍严重塌缩；
- oracle mode 仍大量 rank 3/4；
- future motion pooling 无法表达 crossing timing / stop-go / acceleration pattern。

如果 Step 1 已经很好，则 Step 3 作为架构消融而不是必选主路线。

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

| Posterior | Alignment | Routing | 目的 |
|---|---|---|---|
| simple_pool | off | current | baseline |
| simple_pool | on | sharpened | Step 2 |
| plan_transformer | on | sharpened | Step 3 |

判断 Plan Transformer 是否真的额外降低 `G_mode`，而不是仅靠 alignment 已经解决问题。

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
Step 1
→ Step 2
→ Step 3（仅当 Step 1/2 表明 posterior capacity 不足，或作为后续结构消融）
→ Step 4
→ Step 5
```

## 不允许跳过的依赖

- Step 4 不能在 posterior 仍严重 collapse 时作为主实验；
- Step 5 必须基于已确定的 posterior/routing 设计；
- 不要同时修改 posterior architecture、trajectory routing、prior architecture 和 pretraining schedule 后只报一个结果。

---

# 实验编号建议

| ID | Posterior | Alignment | Traj Routing | Prior | Pretrain |
|---|---|---|---|---|---|
| R0 | simple | off | current soft-q | state-token MLP | current joint |
| R1 | simple | alignment-only, decoder frozen | unchanged | frozen | no retrain decoder |
| R2 | simple | on | sharpened | state-token MLP | current |
| R3 | Plan Transformer | on | sharpened | state-token MLP | current |
| R4 | best posterior | on | sharpened | mode-query cross-attn | current |
| R5 | best posterior | on | sharpened | best prior | A1 + A2 split |

主论文版本应由 R0→R1/R2→R3→R4→R5 的证据逐步决定，不预设 R5 一定最好。

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
test_alignment_responsibility_normalization
test_alignment_stop_gradient_target
test_alignment_only_freezing
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
K=1 时 routing/align 不改变单模式语义
posterior q / decoder mode / trajectory mode index 始终一致
future 信息绝不能进入 inference-time prior path
```

---

# 给 Codex 的最终执行指令

> 当前任务不是一次性重构整个 EgoJEPA，而是严格按五步实验顺序推进，并在每一步根据验证结果决定下一步。Step 1 先加载当前 JAAD K=4 epoch 88 checkpoint，冻结 decoder、context、trajectory/intention head、prior 与 EMA teacher，只训练 FutureModePosterior，用当前 K 条 trajectory 对 GT 的误差构造 soft trajectory responsibility，并训练 posterior alignment，验证当前 posterior 架构是否能恢复 mode 1/2。Step 2 若 alignment 有效，则正式加入小权重 posterior-trajectory alignment，并把当前 raw q-weighted trajectory loss 改成带温度退火的 sharpened posterior routing，继续监测 mode diversity、G_mode、minADE@4 与 intention 指标。Step 3 只有在 simple posterior 容量不足时，或作为结构消融，将 simple temporal pooling 升级为 2-layer temporal Plan Transformer + learnable plan-query attention pooling，并与 simple posterior 在相同 alignment/routing 条件下公平比较。Step 4 在 posterior 稳定后重新测 G_prior；若仍明显，尤其 crossing subgroup 较大，则把 state-token MLP prior 升级为 K-mode-query cross-attention over full context，并以 G_prior/top1 ADE 而非仅 prior↔posterior agreement 判断有效性。Step 5 最后把 pretraining 拆为 A1 representation + future mode discovery 与 A2 frozen-posterior prior fitting，分别保存 best_repr.pt 和 best_prior.pt，禁止 prior KL 再主导 representation checkpoint selection。每一步必须使用独立配置和输出目录，先跑 validation 再固定配置跑 test，不允许基于 test 调参，也不要把五步改动一次性合并后只跑一个结果。