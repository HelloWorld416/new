# EgoJEPA-MP：下一阶段实验计划（Step 1B 后最终执行顺序）

> 本文面向 Codex 执行，基于当前 JAAD `K=4` 模型、已完成的 mode-selection diagnostics、posterior cheap diagnostics、Step 1 与 Step 1B。
>
> **最新结论：暂时不优先升级 Posterior 架构。** Step 1B 已证明：仅训练当前 FutureModePosterior 的 prototype 参数，就可以让 posterior 学到一部分 trajectory-relevant mode semantics，并显著降低 `G_mode`。因此下一阶段首先闭环训练 posterior–trajectory grounding 与 trajectory specialization；只有在 simple posterior 明显 plateau 时，才进入 Temporal Plan Encoder。

---

# 0. 当前证据与结论

## 0.1 原始 JAAD 基线

测试集：

```text
Posterior ADE          = 20.298
G_mode                 = 3.496
Posterior↔Oracle       = 61.13%
mode 1/2 hard usage    = 0%
mode 1/2 soft usage    = 0.87%
```

同时已经确认：

- `minADE@4 < top-1 ADE`，说明 K 个候选中存在真正有价值的 alternative futures；
- oracle 最优模式为 mode 1/2 的样本占测试集约 `37.27%`；
- 原 posterior 几乎从不选择 mode 1/2；
- 原 prior 与 posterior 高度一致，因此此前优先增强 prior 没有意义。

---

## 0.2 Step 1 结论

原 soft responsibility：

```math
r_k \propto \exp(-E_k/\tau_r)
```

在当前 trajectory loss scale 下过于平坦，平均最大 responsibility 约 `0.262`，几乎等于 K=4 均匀分布的 `0.25`。

因此 Step 1 主要是在把 posterior 拉平，并没有建立可靠 trajectory semantics。

> 原 Step 1 soft responsibility 不再作为主 responsibility 使用。

---

## 0.3 Step 1B 结果

Step 1B 比较：

```text
R1 = Hard oracle responsibility
R2 = Label-smoothed oracle responsibility
```

只训练当前 Posterior 的 `1,024` 个 prototype 参数，其余模型全部冻结。

JAAD 测试集：

| 指标 | 原始 B0 | R1 Hard oracle | R2 Label-smoothed |
|---|---:|---:|---:|
| Posterior ADE ↓ | 20.298 | **19.253** | 19.329 |
| G_mode ↓ | 3.496 | **2.451** | 2.527 |
| Posterior↔Oracle ↑ | 61.13% | **66.76%** | 65.42% |
| mode 1/2 hard usage | 0% | **23.32%** | 16.35% |
| mode 1/2 soft usage | 0.87% | **38.61%** | 36.64% |

R1 使 `G_mode` 缩小约 `29.9%`，说明当前 fixed future representation + simple prototype posterior 具备一定 trajectory-mode 识别能力。

但恢复仍不完整：

```text
mode 1 recall ≈ 6.94%
```

收益主要来自 mode 2，因此不能认为四个 trajectory modes 已经全部形成稳定语义。

### Step 1B 最终判断

```text
1. 当前 Posterior 架构不是首先需要替换的模块。
2. posterior–trajectory grounding 确实有效。
3. R1 当前优于 R2，因此后续主线使用 R1。
4. 下一步进入 Step 2，而不是直接进入 Plan Transformer。
5. 后续所有诊断必须按 mode 单独报告，不能继续只看 mode1/2 合并统计。
```

---

# 1. 总体实验原则

## 1.1 固定条件

- `K=4` 暂时固定；
- JAAD 作为第一验证数据集；
- epoch 88 checkpoint 保留为原始基线 B0；
- Step 2 从 **Step 1B validation 选出的 R1 best posterior** 初始化；
- 每个阶段使用独立配置和独立输出目录；
- 不覆盖历史 checkpoint；
- 所有超参数只使用 validation 选择；
- test 只运行 validation 已冻结的配置；
- 不基于 test 调 lr / epoch / temperature / loss weight；
- 每次 posterior / routing / prior 发生变化后重新运行 mode diagnostic；
- 复用现有 bbox decode、ADE/FDE、intent metrics 与 diagnostics；
- GitHub Markdown 数学统一使用 `math` fenced block。

---

## 1.2 后续统一主指标

每一步持续报告：

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

从 Step 2 开始必须按 mode 分开报告：

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

特别注意：mode 2 的恢复不能掩盖 mode 1 仍然失效的问题。

---

# 2. Step 1 / Step 1B：已完成

```text
Step 1  = DONE
Step 1B = DONE
```

后续固定：

```text
主 responsibility = R1 Hard oracle
主要消融          = R2 Label-smoothed oracle
```

不要继续搜索原 Step 1 的近均匀 soft responsibility。

---

# 3. Step 2A：R1 Posterior 短 Warm-up

## 3.1 目的

在正式让 trajectory branch 一起移动之前，先让 validation 选出的 R1 posterior 稳定少量 epoch，降低 posterior 与 decoder 同时漂移的风险。

---

## 3.2 初始化

加载：

```text
base model weights   = epoch 88 checkpoint
posterior/prototypes = Step 1B validation 选出的 R1 best posterior
```

必须验证以下权重与 epoch 88 完全一致：

```text
future predictor / decoder
trajectory head
mode embeddings
context encoders
Past Mode Prior
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

## 3.4 Responsibility 与 loss

主方案使用 R1：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}=-\log(q_{k^{oracle}}+\epsilon)
```

R2 仅保留为平行消融。

---

## 3.5 训练建议

```text
epochs: 2-3
lr: 使用 Step 1B validation 已选中的 R1 lr
optimizer: 沿用 Step 1B
不重新做大范围 lr search
```

如果 validation `G_mode` 相对 Step 1B best 明显恶化，则直接回退 Step 1B best posterior，不使用 warm-up 权重进入 Step 2B。

---

# 4. Step 2B：Posterior Alignment + Sharpened Trajectory Routing

## 4.1 当前主实验

目标：

> 把 FutureModePosterior 的 mode semantics 与 trajectory specialization 真正闭环训练，而不是继续让所有 trajectory modes 被同一个 GT 通过 raw soft-q 权重同时拉近。

当前可能存在的反馈环：

```text
q_k 低
→ mode k 获得的 trajectory supervision 弱
→ specialization 不稳定
→ posterior 更少选择 mode k
→ q_k 进一步降低
```

---

## 4.2 解冻模块

从 Step 2A 最优状态或 Step 1B R1 best 开始。

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

如果本阶段没有 JEPA anchoring，则 dual-target projectors 继续冻结。

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
L_{align}=-\log(q_{k^{oracle}}+\epsilon)
```

候选：

```text
lambda_align = 0.05 / 0.1 / 0.25
```

只在 validation 选择。

R2 使用 Step 1B validation 选出的 label smoothing 参数作为主要消融。

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
L_{traj}=\sum_k stopgrad(\tilde q_k)L_k
```

其中：

```math
L_k=SmoothL1(B^{(k)},B^{gt})
```

必须 stop-gradient routing weights，避免 trajectory regression loss 直接通过 `q` 操纵 mode probability。

---

## 4.5 Temperature schedule

第一轮使用保守 schedule：

```text
前 25% epoch:  tau_traj = 1.0
25%-60%:       tau_traj = 0.5
后 40%:        tau_traj = 0.25
```

配置必须支持切回原始 soft routing：

```yaml
trajectory_routing:
  type: soft | sharpened
  tau_start: 1.0
  tau_mid: 0.5
  tau_end: 0.25
```

第一轮不要直接 hard routing，因为 Step 1B `posterior↔oracle` 仍只有约 `66.76%`。

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

本阶段不要同时加入：

```text
Plan Transformer
Cross-Attention Prior
A1/A2 pretraining split
新的 backbone
Diffusion
```

---

## 4.7 必须监测 trajectory specialization

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

重点防止：

```text
A. 所有 mode 再次收敛成几乎同一条轨迹
B. mode 被人为拉开，但出现不合理候选
```

---

## 4.8 Step 2B 成功标准

相对 Step 1B R1：

```text
Posterior ADE = 19.253
G_mode        = 2.451
Post↔Oracle   = 66.76%
mode1 recall  ≈ 6.94%
```

期望：

- `G_mode` 继续明显下降；
- `ADE_posterior` 继续向 `ADE_oracle` 靠近；
- `posterior↔oracle` 继续提升；
- mode 1 recall 明显高于 6.94%；
- mode-wise hard usage 更接近各自 oracle frequency；
- `minADE@4` 不恶化，最好进一步下降；
- trajectory diversity 不 collapse；
- intention F1/AUC/AP 不出现明显负迁移。

---

# 5. Step 2C：重新执行三层 Mode Diagnostic

Step 2B 完成后，必须重新计算：

```math
ADE_{oracle}
```

```math
ADE_{posterior}
```

```math
ADE_{prior}
```

以及：

```math
G_{mode}=ADE_{posterior}-ADE_{oracle}
```

```math
G_{prior}=ADE_{prior}-ADE_{posterior}
```

同时报告：

```text
posterior↔oracle
prior↔posterior
prior↔oracle
per-mode recall / precision
per-mode hard / soft usage
crossing/non-crossing subgroup
minADE@4 / top1 ADE
```

### 重要解释规则

Step 2 后 Posterior semantics 已经变化，而 Past Mode Prior 仍然是旧 prior，因此：

> **Step 2 的主判断指标是 `G_mode`、Posterior ADE 和 per-mode recall。旧 prior 的 top-1 ADE 不能用于否定 Step 2。**

旧 prior 此时只作为“语义漂移后 prior 失配程度”的诊断。

---

# 6. Step 3：仅在 simple Posterior Plateau 时升级 Temporal Plan Encoder

## 6.1 触发条件

只有出现下面一种或多种情况时，Step 3 才从“可选消融”升级为“必要改造”：

- Step 2 后 `G_mode` 下降仍有限；
- mode 1 recall 仍然很低；
- posterior↔oracle 明显 plateau；
- oracle mode 仍大量排 posterior rank 3/4；
- simple pooled future representation 无法识别 crossing timing / stop-go / acceleration pattern。

如果 Step 2 已经很好，则 Step 3 只作为架构消融。

---

## 6.2 新 Posterior 结构

将当前 simple pooling 替换为：

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

不要把 future image 引入 mode posterior。

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

必须保留 `simple_pool` 做公平对照。

---

## 6.3 核心对照

| Posterior | Alignment | Routing | 目的 |
|---|---|---|---|
| simple_pool | R1 | sharpened | Step 2 主模型 |
| plan_transformer | R1 | sharpened | Step 3 |

判断：Plan Transformer 是否在相同 responsibility 与 routing 下进一步降低 `G_mode`。

---

# 7. Step 4A：Posterior 稳定后，先重新训练原 State-Token MLP Prior

## 7.1 为什么必须先做 4A

Posterior semantics 经 Step 2/3 修改后，旧 prior 不再是公平对照。

因此不能直接把旧 prior 的失败归因于：

```text
context[:,0] + MLP 能力不足
```

必须先让相同架构重新拟合**新的稳定 posterior**。

---

## 7.2 冻结规则

冻结：

```text
context representation
future predictor / decoder
mode embeddings
trajectory head
intention head
FutureModePosterior / prototypes
EMA teacher
```

只训练：

```text
现有 Past Mode Prior:
context[:,0] → MLP → K logits
```

---

## 7.3 Prior target

```math
L_{prior}=KL(stopgrad(q)\,\|\,\pi)
```

或沿用当前 prior NLL/CE 实现，只要 target 是已经冻结的 posterior。

---

## 7.4 重新测

```text
G_prior all
G_prior crossing=0
G_prior crossing=1
prior↔posterior
prior↔oracle
top1 ADE / FDE
intent F1 / AUC / AP
```

只有此时才能公平判断 state-token MLP 是否真的不足。

---

# 8. Step 4B：只有原 Prior 仍不足，才升级 Mode-Query Cross Attention Prior

## 8.1 触发条件

重新训练原 Prior 后，如果：

- `G_prior` 仍明显；
- 尤其 crossing subgroup `G_prior` 仍高；
- prior↔posterior / prior↔oracle 仍明显不足；

才升级 prior 架构。

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
shared/small MLP scorer
        ↓
prior logits [B,K]
```

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

## 8.3 判断标准

新 prior 必须至少改善：

```text
G_prior
或 top1 ADE/FDE
```

仅仅提升 `prior↔posterior agreement` 不能证明有效。

特别关注 crossing subgroup。

---

# 9. Step 5：最后拆分 Pretraining 为 A1 + A2

只有 Posterior、trajectory routing 和 Prior 架构都稳定后，才修改预训练阶段。

---

## 9.1 A1：Representation + Future Mode Discovery

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

冻结或不训练 Past Mode Prior。

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

若 trajectory alignment 在纯 pretraining 数据流中不可稳定定义，可把 alignment 留到后续 task stage；Codex 必须根据当前实现确认后再启用。

A1 checkpoint selection：

```text
主要依据 val JEPA / dual-target reconstruction
同时要求 mode 不 collapse
```

保存：

```text
best_repr.pt
```

禁止 prior KL 参与 `best_repr` selection。

---

## 9.2 A2：Frozen-Posterior Prior Fitting

加载 `best_repr.pt` 并冻结：

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

A2 checkpoint selection：

```text
val prior KL / NLL
prior↔posterior agreement
G_prior（诊断）
```

保存：

```text
best_prior.pt
```

这样解决此前 `prior KL` 主导 representation checkpoint、导致早期 epoch 被错误选中的问题。

---

# 10. 最终执行顺序（Codex 必须遵守）

```text
Step 1 / Step 1B              DONE
        ↓
Step 2A  R1 posterior short warm-up
        ↓
Step 2B  R1 alignment + sharpened trajectory routing
        ↓
Step 2C  rerun full mode diagnostic
        ↓
Step 3   ONLY IF simple posterior plateaus
         Temporal Plan Encoder
        ↓
Step 4A  retrain original state-token MLP prior
        ↓
Step 4B  ONLY IF 4A still insufficient
         Mode-Query Cross-Attention Prior
        ↓
Step 5   A1 representation/mode discovery
         + A2 frozen-posterior prior fitting
```

### 强制依赖关系

- 不允许跳过 Step 2C 就判断是否需要 Step 3；
- 不允许拿旧 prior 的表现直接证明新 Posterior 方案失败；
- 不允许跳过 Step 4A 直接宣称 state-token MLP prior 不够；
- 不允许 posterior semantics 仍在变化时执行正式 Prior 架构比较；
- Step 5 必须基于已经确定的 posterior/routing/prior 设计。

---

# 11. 推荐实验编号

| ID | Posterior | Alignment | Traj Routing | Prior | Pretrain |
|---|---|---|---|---|---|
| E0 | simple | off | current soft-q | old state-token MLP | current joint |
| E1B | simple | R1 prototype-only | unchanged | frozen | no decoder retrain |
| E2 | simple | R1 | sharpened | old prior frozen | current |
| E3 | Plan Transformer | R1 | sharpened | old prior frozen | current |
| E4A | best posterior | fixed | fixed | retrained state-token MLP | current |
| E4B | best posterior | fixed | fixed | mode-query cross-attn | current |
| E5 | best posterior | fixed | sharpened | best prior | A1 + A2 split |

R2 Label-smoothed responsibility 始终作为主要 responsibility ablation，不作为默认主线。

---

# 12. 每一步统一输出

```text
outputs/<exp_name>/
  config_resolved.yaml
  train.log
  trainable_parameters.txt
  frozen_parameters.txt
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
per-mode recall / precision
mode-wise usage
mode-wise oracle frequency
mode-wise ADE gain
crossing/non-crossing subgroup
intent F1/AUC/AP
```

---

# 13. 必须新增 / 保留的测试

Codex 每一步必须保持已有 diagnostics tests 通过，并确保：

```text
test_hard_oracle_responsibility
test_label_smoothed_responsibility
test_alignment_stop_gradient_target
test_alignment_only_freezing
test_sharpened_routing_normalization
test_sharpened_routing_temperature_effect
test_routing_weights_stop_gradient
test_modewise_metrics
test_step2_old_prior_not_used_for_selection
test_plan_transformer_output_shape
test_state_token_prior_retrain_freezing
test_mode_query_prior_output_shape
test_a1_prior_frozen
test_a2_repr_frozen
test_best_repr_selection_ignores_prior_kl
```

Sanity：

```text
K=1 时 routing / alignment 不改变单模式语义
posterior q / decoder mode / trajectory mode index 始终一致
future 信息绝不能进入 inference-time prior path
旧 checkpoint/config 仍可加载
```

---

# 14. 给 Codex 的最终执行指令

> Step 1 与 Step 1B 已完成。当前不要立即升级 Posterior 架构。先从 Step 1B validation 选出的 R1 Hard-oracle best posterior 出发执行 Step 2A：保持 decoder/context/prior/task heads 冻结，只让 Posterior/prototypes 短暂稳定 2-3 epoch；若 validation G_mode 变差则回退 Step 1B best。随后执行 Step 2B：解冻 Posterior、future predictor/mode-conditioned decoder、mode embeddings、trajectory head 与 intention head，继续冻结 context encoders、fusion、Past Mode Prior 与 EMA teacher；使用 R1 Hard-oracle posterior alignment，并将当前 raw q-weighted trajectory loss 改为 stop-gradient sharpened routing，temperature 按 1.0 → 0.5 → 0.25 分阶段退火。只在 validation 搜索 lambda_align，不基于 test 调参。Step 2B 后必须执行 Step 2C 完整 mode diagnostic，重点看 G_mode、Posterior ADE、posterior↔oracle、每个 mode 的 recall/precision、mode 1 recall、minADE@4 与 trajectory diversity。旧 Past Prior 在 posterior semantics 变化后只能作为诊断，不能用其 top1 ADE 否定 Step 2。只有 simple posterior 在 Step 2 后明显 plateau 时，才执行 Step 3 Temporal Plan Encoder；否则 Step 3 仅作为结构消融。Posterior 稳定后执行 Step 4A：冻结其余模块并重新训练原 `context[:,0] -> MLP` prior，重新测 G_prior；只有重新训练后的简单 Prior 仍不足，尤其 crossing subgroup G_prior 仍高，才执行 Step 4B Mode-Query Cross-Attention Prior。最后执行 Step 5，将 pretraining 拆成 A1 representation/future-mode discovery 与 A2 frozen-posterior prior fitting，分别保存 best_repr.pt 和 best_prior.pt，禁止 prior KL 再参与 representation checkpoint selection。所有阶段使用独立输出目录、保留历史 checkpoint、先 validation 后 test，并逐阶段提交结果，不允许把多个结构改动一次性混在一个实验里。
