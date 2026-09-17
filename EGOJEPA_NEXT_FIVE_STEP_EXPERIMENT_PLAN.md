# EgoJEPA-MP：下一阶段五步实验计划（Step 1B 后更新版）

> 本文面向 Codex 执行，基于当前 JAAD `K=4` 模型、已完成的 mode-selection diagnostics、posterior cheap diagnostics、Step 1 与 Step 1B 结果。
>
> **当前主结论已经变化：暂时不应优先升级 Posterior 架构。** Step 1B 证明仅训练现有 Posterior 的 prototype 参数，就可以显著学习一部分 trajectory-relevant mode semantics。因此下一阶段应先把已经有效的 posterior–trajectory grounding 与 trajectory specialization 闭环训练起来，再根据 `G_mode` 是否 plateau 决定是否进入 Plan Transformer。

---

# 0. 当前证据与最新结论

## 0.1 原始基线

JAAD 测试集基线：

```text
Posterior ADE          = 20.298
G_mode                 = 3.496
Posterior↔Oracle       = 61.13%
mode 1/2 hard usage    = 0%
mode 1/2 soft usage    = 0.87%
```

同时已经确认：

- `minADE@4 < top-1 ADE`，说明 K 个候选中存在真实有价值的 alternative futures；
- oracle 最优模式为 mode 1/2 的样本占测试集约 `37.27%`；
- 原 posterior 几乎不选择 mode 1/2；
- 原 prior 与 posterior 高度一致，因此此前增强 prior 并不能解决 posterior 自身的语义错位。

---

## 0.2 Step 1 结论

原 soft responsibility：

```math
r_k \propto \exp(-E_k/\tau_r)
```

在当前 loss scale 下过于平坦，平均最大 responsibility 约 `0.262`，非常接近 K=4 均匀分布的 `0.25`。

结果虽然把 mode 1/2 soft usage 拉到约 50%，但 test `posterior↔oracle` 没有改善，说明主要是在**拉平 posterior**，并没有建立可靠的 trajectory-mode semantics。

因此：

> 原 Step 1 soft responsibility 不再作为后续主方案使用。

---

## 0.3 Step 1B 最新结果

Step 1B 比较：

```text
R1 = Hard oracle responsibility
R2 = Label-smoothed oracle responsibility
```

仅训练现有 Posterior 的 `1,024` 个 prototype 参数，其他权重全部冻结。

JAAD 测试集结果：

| 指标 | 原始 B0 | R1 Hard oracle | R2 Label-smoothed |
|---|---:|---:|---:|
| Posterior ADE ↓ | 20.298 | **19.253** | 19.329 |
| G_mode ↓ | 3.496 | **2.451** | 2.527 |
| Posterior↔Oracle ↑ | 61.13% | **66.76%** | 65.42% |
| mode 1/2 hard usage | 0% | **23.32%** | 16.35% |
| mode 1/2 soft usage | 0.87% | **38.61%** | 36.64% |

R1 使 `G_mode` 相对基线缩小约 `29.9%`，并且 Posterior ADE 有稳定改善。

但恢复仍不完整：

```text
mode 1 recall ≈ 6.94%
```

收益主要来自 mode 2，不能据此认为四个 trajectory modes 已经全部形成稳定可识别语义。

### Step 1B 最终结论

```text
1. 当前 fixed future representation 并不是完全无法支持 mode 1/2。
2. 当前 simple Posterior / prototype 至少具有部分 trajectory-mode 识别能力。
3. 原 Step 1 失败的重要原因是 responsibility target 缺乏区分度。
4. 现在有足够证据优先进入 Step 2，而不是立即进入 Plan Transformer。
5. R1 当前优于 R2，因此 Step 2 主方案先使用 R1；R2 继续作为主要消融。
```

---

# 1. 总体原则

## 1.1 必须保持

- `K=4` 暂时固定；
- JAAD 作为第一验证数据集；
- epoch 88 checkpoint 保留为最初基线 B0；
- Step 2 从 **Step 1B validation 选出的 R1 最优 posterior** 开始；
- 所有新实验使用新的输出目录；
- 不覆盖历史 checkpoint；
- 所有超参数只用 validation 选择；
- test 只运行 validation 已冻结的配置；
- 不基于 test 结果重新选择 lr / epoch / temperature / loss weight；
- 每次结构变化后都重新执行 mode diagnostic；
- 复用现有 bbox decode、ADE/FDE、intent metrics 和 diagnostics；
- GitHub Markdown 数学统一使用 `math` fenced block。

---

## 1.2 后续统一主指标

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
posterior entropy
prior entropy
intent F1 / AUC / AP
```

从 Step 2 开始必须额外按 mode 分开报告：

```text
mode0 oracle frequency
mode1 oracle frequency
mode2 oracle frequency
mode3 oracle frequency

mode0 posterior recall / precision
mode1 posterior recall / precision
mode2 posterior recall / precision
mode3 posterior recall / precision

mode-wise hard usage
mode-wise soft usage
mode-wise q_oracle
mode-wise ADE gain when oracle
```

不能再只报告 `mode1/2 combined usage`，因为 Step 1B 已经发现 mode 2 的恢复会掩盖 mode 1 仍然接近失效的问题。

---

# 2. Step 1 / Step 1B：已完成

## 2.1 Step 1 状态

```text
DONE
```

结论：原 soft responsibility 近似均匀，不再作为主 responsibility。

## 2.2 Step 1B 状态

```text
DONE
```

主结果：R1 Hard oracle 当前优于 R2 Label-smoothed oracle。

后续：

```text
Step 2 主责任目标 = R1 Hard oracle
Step 2 对照责任目标 = R2 Label-smoothed oracle
```

不要重新搜索原 Step 1 的 `tau_r` soft target。

---

# 3. Step 2A：从 R1 最优 Posterior 开始做短暂 Warm-up

## 3.1 目的

Step 1B 已经证明 prototype-only posterior 能吸收 trajectory oracle 语义，但 mode 1 仍明显不足。

Step 2A 的目的不是再做一次完整超参搜索，而是在正式解冻 trajectory branch 前，让 validation 选出的 R1 posterior 在完全冻结 decoder 的条件下稳定数个 epoch，避免一开始同时让 posterior 与 trajectory modes 一起移动。

---

## 3.2 初始化

加载：

```text
base model weights      = 原 epoch 88 checkpoint
posterior/prototypes    = Step 1B validation 选出的 R1 best_posterior
```

必须验证：

```text
future predictor / decoder 与 epoch88 完全一致
trajectory head 与 epoch88 完全一致
mode embeddings 与 epoch88 完全一致
context encoders 与 epoch88 完全一致
```

---

## 3.3 冻结规则

冻结：

```text
visual/context encoder
motion/context encoder
ego projection
fusion
future predictor / decoder
mode embeddings
trajectory head
intention head
Past Mode Prior
EMA target encoders
dual-target projectors
```

仅训练：

```text
FutureModePosterior / prototypes
```

---

## 3.4 Responsibility

Step 2A 主方案使用 R1：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}= -\log(q_{k^{oracle}}+\epsilon)
```

R2 保留为平行消融，但不替代 R1 主线。

---

## 3.5 训练建议

```text
epochs: 2-3
lr: 使用 Step 1B validation 已选中的 R1 lr
optimizer: 沿用 Step 1B
不重新做大范围 lr search
```

如果 warm-up 后 validation `G_mode` 明显恶化，则回到 Step 1B best posterior，不继续使用 warm-up 权重。

---

# 4. Step 2B：Posterior Alignment + Sharpened Trajectory Routing 联合训练

## 4.1 这是下一阶段的主实验

核心目标：

> 让 posterior 的 future mode semantics 与 trajectory specialization 共同形成闭环，而不是继续让所有 trajectory modes 被同一个 GT 通过 raw soft-q 权重同时拉近。

当前问题可以描述为：

```text
q_k 低
→ mode k 获得的 trajectory supervision 弱
→ specialization 不稳定
→ posterior 更少选择 mode k
→ q_k 进一步降低
```

Step 2B 要打破这个正反馈环。

---

## 4.2 解冻模块

从 Step 2A 最优状态开始。

解冻：

```text
FutureModePosterior / prototypes
future predictor / mode-conditioned decoder
mode embeddings
trajectory head
intention head
```

继续冻结：

```text
visual/context encoder
motion/context encoder
ego projection
fusion transformer
Past Mode Prior
EMA target encoders
```

Dual-target projectors 是否训练取决于当前 Stage C 数据流；若本实验没有 JEPA anchoring，则继续冻结。

Codex 必须保存：

```text
trainable_parameters.txt
frozen_parameters.txt
```

---

## 4.3 Posterior alignment

主方案继续使用 R1 Hard oracle：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}= -\log(q_{k^{oracle}}+\epsilon)
```

总 loss 中：

```text
lambda_align candidates = 0.05 / 0.1 / 0.25
```

只允许 validation 选择。

R2 使用 Step 1B validation 选出的 label smoothing 配置作为消融：

```math
L_{align}^{R2}=-\sum_k r_k^{R2}\log(q_k+\epsilon)
```

不要恢复 Step 1 的近均匀 soft responsibility。

---

## 4.4 Sharpened trajectory routing

当前 raw routing：

```math
L_{traj}=\sum_k q_k L_k
```

改为：

```math
\tilde q_k
=
\frac{q_k^{1/\tau_{traj}}}
{\sum_j q_j^{1/\tau_{traj}}}
```

```math
L_{traj}
=
\sum_k stopgrad(\tilde q_k)L_k
```

其中：

```math
L_k=SmoothL1(B^{(k)},B^{gt})
```

必须 stop-gradient routing weight，避免 trajectory regression loss 通过 `q` 自行操纵 mode probability。

---

## 4.5 推荐 temperature schedule

第一轮使用保守 schedule：

```text
前 25% epoch:  tau_traj = 1.0
25%-60%:       tau_traj = 0.5
后 40%:        tau_traj = 0.25
```

配置必须可切回原 soft routing：

```yaml
trajectory_routing:
  type: soft | sharpened
  tau_start: 1.0
  tau_mid: 0.5
  tau_end: 0.25
```

第一轮不要直接 hard routing，因为 Step 1B 后 `posterior↔oracle` 仍只有约 `66.76%`。

---

## 4.6 推荐联合 task loss

```math
L_{step2}
=
L_{traj}^{sharp}
+
\lambda_{align}L_{align}
+
0.5L_{intent}
```

如果当前 implementation 中还存在其他已经验证过且不可移除的 task regularizer，保持不变并在 resolved config 中明确记录。

本步不要同时加入新的 Plan Transformer、Cross-Attention Prior 或 A1/A2 pretraining split。

---

## 4.7 必须监测 trajectory specialization

新增 / 保留：

```text
pairwise endpoint distance
pairwise trajectory distance
pairwise latent cosine similarity
mode-wise oracle frequency
mode-wise posterior recall
mode-wise posterior precision
mode-wise hard usage
mode-wise soft usage
```

重点防止两种失败：

```text
A. 所有 mode 再次收敛成几乎同一条轨迹
B. mode 被人为推开，但出现不合理、无物理意义的候选
```

---

## 4.8 Step 2B 成功标准

相对 Step 1B R1：

```text
R1 test Posterior ADE = 19.253
R1 test G_mode        = 2.451
R1 test Post↔Oracle   = 66.76%
mode1 recall          ≈ 6.94%
```

期望：

- `G_mode` 继续明显下降；
- `ADE_posterior` 继续向 `ADE_oracle` 靠近；
- `posterior↔oracle` 继续提升；
- mode 1 recall 明显高于当前约 6.94%；
- mode-wise hard usage 更接近各自 oracle frequency；
- `minADE@4` 不恶化，最好继续下降；
- trajectory diversity 不 collapse；
- intention F1/AUC/AP 不出现明显负迁移。

特别注意：

> Step 2B 判断主指标是 `G_mode` 与 mode-wise recall，而不是旧 prior 的 top-1 ADE。

Posterior semantics 已经变化，旧 prior 尚未重新拟合，因此旧 `ADE_prior/top1 ADE` 此时只能作为参考，不能作为否定 Step 2 的主要依据。

---

# 5. Step 2C：联合训练后强制重新运行 Mode Diagnostic

Step 2B 完成后，不允许直接进入 Prior 改造。

必须重新运行与之前相同定义的：

```text
ADE_oracle
ADE_posterior
ADE_prior
G_mode
G_prior
posterior↔oracle
prior↔posterior
prior↔oracle
mode-wise oracle frequency
mode-wise posterior recall / precision
mode-wise hard / soft usage
q_oracle
oracle rank
crossing / non-crossing subgroup
```

## 5.1 主要判断

### Case A：`G_mode` 明显继续下降，四个 mode recall 均改善

结论：

```text
simple posterior + correct grounding + sharpened routing 基本足够
```

下一步：

```text
Step 3 降级为架构消融，可先进入 Step 4A
```

### Case B：总体 G_mode 改善，但 mode 1 recall 仍长期极低

结论：

```text
simple posterior 在部分 trajectory semantics 上存在 plateau
```

下一步优先执行 Step 3。

### Case C：G_mode 几乎不再下降，posterior↔oracle plateau

下一步执行 Step 3。

### Case D：minADE@4 恶化明显

说明 trajectory specialization 训练本身破坏候选质量。

先检查：

```text
lambda_align
tau schedule
trajectory loss scale
mode diversity
```

不要直接进入 Step 3 / Step 4。

---

# 6. Step 3：仅在 Simple Posterior Plateau 时升级 Temporal Plan Encoder

## 6.1 触发条件

只有满足以下之一才把 Step 3 作为主实验：

- Step 2C 后 `G_mode` 仍明显较大且改善有限；
- mode 1 recall 仍接近失效；
- posterior↔oracle 明显 plateau；
- diagnostics 显示 simple pooled future representation 无法区分关键 temporal behavior。

如果 Step 2 已经解决大部分 `G_mode`，Step 3 只作为论文架构消融。

---

## 6.2 Plan Encoder 结构

将当前 simple posterior 的 future pooling 替换为：

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

不要把 future image 引入 posterior。

---

## 6.3 公平比较

必须保持与最佳 simple posterior 实验一致：

```text
same responsibility target
same alignment weight search space
same sharpened routing schedule
same downstream training budget
same checkpoint selection rule
```

只改变：

```text
simple future pooling → temporal Plan Encoder
```

---

## 6.4 成功标准

Plan Transformer 必须在相同 grounding / routing 条件下进一步改善：

```text
G_mode
Posterior ADE
posterior↔oracle
mode-wise recall
尤其 mode1 recall
```

如果只增加参数但提升很小，则优先保留 simple posterior 作为最终模型。

---

# 7. Step 4A：Posterior 稳定后，先重新训练原 State-Token MLP Prior

## 7.1 为什么必须先做 4A

Posterior semantics 已经经过 Step 1B / Step 2（以及可能的 Step 3）改变。

旧 Past Mode Prior 学的是旧 posterior，因此：

> 不能直接拿旧 prior 的 `G_prior` 来证明 state-token MLP 能力不足。

必须先在**新的固定 posterior semantics**下重新训练最简单 prior。

---

## 7.2 冻结规则

加载当前最佳 posterior + decoder checkpoint 后冻结：

```text
context encoders
fusion
future predictor / decoder
mode embeddings
trajectory head
intention head
FutureModePosterior
prototypes
EMA teachers
```

只训练当前 baseline prior：

```text
context[:,0]
→ MLP
→ K logits
```

---

## 7.3 Prior target

固定 posterior：

```math
q(M\mid Y_{future})
```

prior：

```math
\pi(M\mid C_{past})
```

训练：

```math
L_{prior}=KL(stopgrad(q)\,\|\,\pi)
```

如果当前最优 posterior 使用 R1-grounded semantics，仍然让 prior 拟合 posterior probability `q`，不要直接让 inference-time prior 看 oracle label。

---

## 7.4 Step 4A 重新评估

训练完成后重新计算：

```text
ADE_posterior
ADE_prior
G_prior
prior↔posterior
prior↔oracle
top1 ADE/FDE
```

并单独报告：

```text
G_prior crossing=0
G_prior crossing=1
```

### 如果重新训练后的 simple prior 已经足够好

例如：

```text
G_prior 明显下降
crossing subgroup 不再突出
```

则：

> Step 4B Cross-Attention Prior 可降级为结构消融，不应强行加入主模型。

---

# 8. Step 4B：只有原 Prior 重训后仍不足，才升级 Mode-Query Cross Attention

## 8.1 触发条件

Step 4A 后仍存在：

```text
G_prior 明显较大
或 crossing=1 的 G_prior 明显高于总体
```

才执行本步。

---

## 8.2 新 Prior

使用 K 个 mode embeddings 作为 query 读取完整 context：

```text
K mode queries [K,D]
        ↓
Cross Attention
        ↓
context memory [B,33,D]
        ↓
mode-specific context [B,K,D]
        ↓
shared/small scorer
        ↓
prior logits [B,K]
```

形式：

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

## 8.3 公平对照

比较：

```text
P0 = retrained state-token MLP prior
P1 = mode-query cross-attention prior
```

必须使用：

```text
same fixed posterior
same fixed decoder
same train/val split
same optimization budget
same selection rule
```

成功标准不能只看 prior↔posterior agreement，而要看：

```text
G_prior
top1 ADE/FDE
prior↔oracle
crossing subgroup
intent F1/AUC/AP
```

---

# 9. Step 5：最后再把 Pretraining 拆成 A1 Mode Discovery + A2 Prior Fitting

只有 Posterior、trajectory routing、Prior 结构都基本确定后，才重构 pretraining schedule。

---

## 9.1 A1：Representation + Future Mode Discovery

训练：

```text
online encoders
fusion
future predictor
FutureModePosterior
prototypes
mode embeddings
dual-target projectors
```

Past Mode Prior 冻结或不参与 optimizer。

核心 loss：

```math
L_{A1}
=
L_{dualJEPA}
+
\lambda_{align}L_{align}
+
\lambda_{usage}L_{usage}
```

其中 responsibility / alignment 必须沿用 Step 1B / Step 2 已验证定义。

如果纯 pretrain 数据流中无法稳定使用 trajectory alignment，则允许 A1 只使用：

```math
L_{A1}=L_{dualJEPA}+\lambda_{usage}L_{usage}
```

但必须在报告中说明。

---

## 9.2 A1 checkpoint selection

保存：

```text
best_repr.pt
```

选模主要依据：

```text
val JEPA / dual-target reconstruction
+ mode non-collapse diagnostics
```

禁止 prior KL 参与 `best_repr` 评分。

---

## 9.3 A2：Prior Fitting

加载并冻结：

```text
best_repr representation
FutureModePosterior
prototypes
mode semantics
```

只训练最终选定的 Past Mode Prior。

目标：

```math
L_{A2}=KL(stopgrad(q)\,\|\,\pi)
```

保存：

```text
best_prior.pt
```

A2 选模：

```text
val prior KL / NLL
prior↔posterior
G_prior diagnostics
```

---

# 10. 最终执行顺序（当前唯一推荐顺序）

> 本节覆盖文档中所有旧的“推荐顺序”描述。Codex 后续按此顺序执行。

```text
Step 1       DONE
Step 1B      DONE

→ Step 2A
  从 Step 1B validation 选出的 R1 best posterior 开始
  冻结 decoder，posterior warm-up 2-3 epoch

→ Step 2B
  R1 Hard Oracle alignment
  + sharpened trajectory routing
  + trajectory/intention joint specialization

→ Step 2C
  强制重新运行完整 mode diagnostic
  核心检查 G_mode、mode-wise recall、minADE@4、diversity

→ Step 3（条件执行）
  只有 simple posterior plateau 时升级 Temporal Plan Encoder
  如果 Step 2 已解决大部分 G_mode，则 Step 3 仅做结构消融

→ Step 4A
  固定新 posterior semantics
  先重新训练原 state-token MLP Prior
  重新测 G_prior

→ Step 4B（条件执行）
  只有 Step 4A 后 G_prior 仍明显，特别是 crossing subgroup 仍差时
  才升级 Mode-Query Cross-Attention Prior

→ Step 5
  最后将 pretraining 拆成：
  A1 Representation / Mode Discovery
  + A2 Fixed-Posterior Prior Fitting
```

---

# 11. 关键停止条件

## Step 2 → Step 3

如果 Step 2C 后：

```text
G_mode 明显下降
mode1 recall 明显恢复
posterior↔oracle 持续提升
minADE@4 不恶化
```

则不要把 Plan Transformer 当作必需模块。

如果：

```text
G_mode plateau
或 mode1 recall 仍接近失效
```

才进入 Step 3。

---

## Step 4A → Step 4B

如果重新训练后的 state-token MLP prior 已使：

```text
G_prior 明显降低
crossing subgroup gap 可接受
```

则 Cross-Attention Prior 只做消融。

只有 simple prior 重训后仍不足，才执行 Step 4B。

---

# 12. 更新后的实验编号

| ID | Posterior | Alignment | Traj Routing | Prior | Pretrain | 作用 |
|---|---|---|---|---|---|---|
| E0 | original simple | off | current soft-q | old state-token MLP | current | 原始基线 |
| E1-R1 | simple | Hard oracle | decoder frozen | frozen | none | Step 1B capacity |
| E1-R2 | simple | Label-smoothed | decoder frozen | frozen | none | Step 1B smoothing |
| E2A | simple R1-best | Hard oracle | unchanged | frozen | none | posterior warm-up |
| E2B | simple | Hard oracle | sharpened | frozen old prior | current downstream | 主 specialization 实验 |
| E3 | Plan Transformer | same alignment | same sharpened | frozen old prior | current | 条件架构实验 |
| E4A | best posterior | fixed | fixed | retrained state-token MLP | current | prior 公平重训 |
| E4B | best posterior | fixed | fixed | mode-query cross-attn | current | 条件 prior 升级 |
| E5 | final posterior | final | final | final prior | A1 + A2 | 最终 pretrain 重构 |

---

# 13. 每一步统一输出

```text
outputs/<exp_name>/
  config_resolved.yaml
  train.log
  trainable_parameters.txt
  val_metrics.json
  test_metrics.json
  mode_diagnostics_val.json
  mode_diagnostics_test.json
  report.md
```

所有 `report.md` 至少包含：

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
mode-wise oracle frequency
mode-wise posterior recall
mode-wise posterior precision
mode-wise hard/soft usage
mode-wise ADE gain
crossing/non-crossing subgroup
intent F1/AUC/AP
```

---

# 14. 必须新增 / 保留的测试

```text
test_hard_oracle_responsibility_one_hot
test_label_smoothed_responsibility_sum_to_one
test_oracle_mode_matches_minade_mode
test_alignment_stop_gradient_target
test_step2a_only_posterior_trainable
test_step2b_trainable_modules
test_sharpened_routing_normalization
test_sharpened_routing_temperature_effect
test_routing_weights_stop_gradient
test_modewise_metrics_consistency
test_plan_transformer_output_shape
test_step4a_only_prior_trainable
test_mode_query_prior_output_shape
test_a1_prior_frozen
test_a2_repr_frozen
test_best_repr_selection_ignores_prior_kl
```

额外 sanity：

```text
K=1 时 responsibility = [1]
K=1 时 sharpened routing = [1]
posterior mode index / decoder mode index / trajectory mode index 始终一致
future information 绝不能进入 inference-time prior path
old prior 在 Step 2 后只作参考，不作为 Step 2 成败的主要判据
```

---

# 15. 给 Codex 的最新最终执行指令

> Step 1 和 Step 1B 已完成。Step 1B 已证明当前 simple posterior / prototype 在 trajectory oracle supervision 下可以明显改善 mode semantics：测试 Posterior ADE 从 20.298 降至 19.253，G_mode 从 3.496 降至 2.451，posterior↔oracle 从 61.13% 提升到 66.76%，因此当前不要优先升级 Plan Transformer。下一步严格执行 Step 2A→Step 2B→Step 2C。Step 2A 从 validation 选出的 R1 Hard-oracle best posterior 开始，保持 decoder/context/prior 冻结，只让 posterior 再稳定 2-3 epoch。Step 2B 解冻 future predictor/decoder、mode embeddings、trajectory head、intention head 和 posterior，继续冻结 context encoder、fusion、Past Mode Prior 与 EMA teacher；使用 R1 Hard-oracle posterior alignment 作为主方案，并加入 stop-gradient 的 sharpened posterior trajectory routing，温度按 1.0→0.5→0.25 分阶段退火。R2 Label-smoothed responsibility 只做主要消融，不恢复 Step 1 的近均匀 soft responsibility。Step 2B 完成后强制执行 Step 2C mode diagnostic，重点检查 G_mode、posterior↔oracle、每个 mode 的 recall/precision、mode1 recall、minADE@4 与 trajectory diversity。只有 simple posterior 在这些指标上 plateau 时才进入 Step 3 Temporal Plan Encoder；如果 Step 2 已经解决大部分 G_mode，Step 3 降级为结构消融。Posterior semantics 稳定后进入 Step 4A：先固定 posterior/decoder，重新训练原 state-token MLP Prior，再重新测 G_prior；只有重训后的 simple prior 仍明显不足，特别是 crossing subgroup G_prior 仍大时，才进入 Step 4B Mode-Query Cross-Attention Prior。最后再执行 Step 5，将 pretraining 拆成 A1 representation/mode discovery 与 A2 fixed-posterior prior fitting，分别保存 best_repr.pt 和 best_prior.pt，禁止 prior KL 再参与 representation checkpoint selection。所有超参只在 validation 选择，test 只运行 validation 已冻结的配置，不允许根据 test 结果回调参数。
