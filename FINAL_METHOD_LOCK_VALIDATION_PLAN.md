# EgoJEPA-MP：最终方法锁定与证实验收计划

> 本文用于 Codex 执行最终方法锁定。目标不是继续探索新模块，而是把目前已经得到正证据的方法收敛成一个**可复现、可解释、可部署、可做论文主结果**的固定版本，并通过最小但充分的实验确认它值得锁定。
>
> 除非本文明确写明，否则不要再新增网络结构、loss、routing、prior、JEPA 训练阶段或新的超参数搜索。

---

# 0. 当前已经得到的关键证据

## 0.1 Step 2：Posterior alignment + sharpened routing 有明确正收益

JAAD 测试集，Step 1B R1 → Step 2 R1：

| 指标 | Step 1B R1 | Step 2 R1 |
|---|---:|---:|
| Posterior ADE ↓ | 19.253 | **18.517** |
| G_mode ↓ | 2.451 | **1.504** |
| Posterior↔Oracle ↑ | 66.76% | **77.75%** |
| Mode 1 recall ↑ | 6.94% | **56.60%** |
| minADE@4 ↓ | **16.802** | 17.013 |
| Intention AP ↑ | **0.232** | 0.179 |

解释：

- trajectory-grounded posterior alignment 有效；
- sharpened routing 明显改善 mode semantics 与 posterior selection；
- 但候选质量与 intention AP 有小幅代价，因此最终方法不能再继续把 routing 做得更硬。

---

## 0.2 Step 4A：简单 state-token MLP Prior 重新拟合后恢复部署选择能力

JAAD 测试集：

```text
KL(q||pi):            2.7372 → 0.1334
G_prior:              11.1958 → 3.3379
Top1 ADE:             29.7125 → 21.8546
Top1 FDE:             59.5575 → 46.2423
Prior↔Posterior:      38.87% → 78.82%
Intention F1:         0.2440 → 0.2696
Intention AP:         0.1789 → 0.2154
```

解释：

- posterior semantics 改变后，prior 必须重新拟合；
- 简单 MLP prior 已能恢复大部分语义匹配；
- 部署性能仍存在 `G_prior`，但这不是继续堆 prior 结构的充分理由。

---

## 0.3 Step 4B：Mode-query Cross-Attention Prior 为负结果

相对 Step 4A：

```text
G_prior:              3.3379 → 3.9377
Top1 ADE:             21.8546 → 22.4543
Top1 FDE:             46.2423 → 47.5202
Prior↔Posterior:      78.82% → 75.60%
Crossing G_prior:     8.7922 → 9.2235
Crossing Top1 ADE:    29.7657 → 30.1970
```

因此最终方法：

> **不使用 Cross-Attention Prior。**

---

## 0.4 Step 5A：Predictability-aware regularization 未证明有效

`lambda_pred=0.05` 相比 `lambda_pred=0`：

```text
G_mode:       1.0846 → 1.0846
G_prior:      3.9244 → 4.0007
G_total:      5.0090 → 5.0853
Top1 ADE:     22.0215 → 22.0979
Crossing ADE: 27.7705 → 27.7705
```

因此最终方法：

> **不使用 predictability-aware regularization。**

---

## 0.5 Step 5D：A1/A2 二次 JEPA 拆分未证明有效

A1 没有选中任何训练后的 representation，`best_repr.pt` 实际退回 epoch 0；A2 只带来极小 ADE 变化，同时 AP 下降。

因此最终方法：

> **不执行 downstream 后的二次 JEPA re-alignment，也不把 Step 5D 作为最终训练流程。**

---

# 1. 最终候选方法：必须锁定的组件

最终候选方法固定为：

```text
现有 JEPA future representation
+
K = 4 mode conditioning
+
R1 Hard-Oracle Posterior Alignment
+
Sharpened Soft Trajectory Routing
+
现有简单 FutureModePosterior / prototypes
+
State-Token MLP Past Mode Prior
+
Posterior semantics 稳定后单独 refit Prior
```

数学上，下游 trajectory specialization 固定使用：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}=-\log(q_{k^{oracle}}+\epsilon)
```

sharpened routing：

```math
\tilde q_k=
\frac{q_k^{1/\tau}}
{\sum_j q_j^{1/\tau}}
```

```math
L_{traj}=\sum_k stopgrad(\tilde q_k)L_k
```

其中：

```math
L_k=SmoothL1(B^{(k)},B^{gt})
```

最终 task loss 沿用 Step 2 已验证配置：

```math
L_{task}=L_{traj}+\lambda_{align}L_{align}+0.5L_{intent}
```

Past Prior 单独拟合：

```math
L_{prior}=KL(stopgrad(q)\,\|\,\pi)
```

---

# 2. 最终固定超参数

## 2.1 Innovation-specific 参数

使用已经验证过的 Step 2 / Step 4A 配置，不再重新大范围搜索：

```yaml
num_modes: 4
responsibility: hard_oracle
lambda_align: 0.1
trajectory_routing:
  type: sharpened
  tau_start: 1.0
  tau_mid: 0.5
  tau_end: 0.25
```

Prior：

```yaml
prior:
  type: state_token_mlp
```

JAAD prior refit 使用已验证配置：

```text
lr = 3e-4
max_epochs = 20
checkpoint selection = minimum validation prior KL
```

如 PIE 原项目已有 dataset-specific optimizer / LR 配置，可以继续使用既有 dataset config；但**本方法新增的 K、alignment、routing schedule、prior architecture 不允许因 PIE 测试结果而改变。**

---

# 3. 最终明确排除的模块

以下内容不得进入最终主模型：

```text
Mode-query Cross-Attention Prior
predictability-aware lambda_pred
Posterior-guided hard assignment
Plan Transformer Posterior（除非仅作为补充消融）
Step 5D downstream 后二次 JEPA re-alignment
A1/A2 作为最终主训练流程
原 Step 1 近均匀 soft responsibility
Diffusion
VQ / Gumbel / Sinkhorn
额外 ROI attention
新的 backbone
```

Codex 不要为了追求更高测试指标重新启用上述失败或未证实模块。

---

# 4. 最终训练 / 推理流程必须固定

> **最终训练顺序必须固定为：**
>
> ```text
> Stage A: JEPA pretraining
>     ↓
> Stage B: K=4 multimodal trajectory / intention specialization
>          + FutureModePosterior training-time assignment
>          + R1 Hard-Oracle alignment
>          + sharpened trajectory routing
>     ↓
> Freeze posterior semantics + candidate generator
>     ↓
> Stage C: Past Mode Prior refit
>     ↓
> Inference: past-only prior selects top-1 candidate
> ```
>
> **FutureModePosterior 与 Past Mode Prior 不允许从头同时联合训练。**
>
> 原因：Posterior 的职责是先利用真实 future 建立稳定的 trajectory-mode semantics；Prior 只能在这些 semantics 稳定后，再学习从过去观测近似它。此前实验已证明，Posterior semantics 改变后如果 Prior 不重新拟合，Top1 ADE 会明显恶化。

---

## 4.1 Stage A：JEPA representation pretraining

首先运行现有已经验证的 JEPA pretraining，得到基础 future representation checkpoint。

这一阶段的职责是：

```text
past visual / motion / ego context
→ future latent representation
```

Stage A 不负责最终 trajectory mode selection，也不训练最终 Past Mode Prior 去预测尚未稳定的 downstream modes。

Codex 必须从代码中确认并记录：

```text
JEPA pretrain checkpoint
pretraining loss 实际组成
best checkpoint selection rule
online / EMA target 初始化与更新方式
```

最终方法**不执行 downstream specialization 后的二次 JEPA re-alignment**；Step 5D 已证明该流程没有稳定正收益。

---

## 4.2 Stage B：FutureModePosterior + K=4 trajectory / intention specialization

加载 Stage A 的 JEPA checkpoint，进入最终多模态 specialization。

这一阶段生成：

```text
K = 4 future latent hypotheses
→ K trajectories
→ K mode-conditioned intention predictions
```

### FutureModePosterior 的训练位置

**FutureModePosterior 必须在这一阶段训练。**

它使用真实 future 信息：

```math
q(M\mid Y_{future})
```

并通过 R1 Hard-Oracle trajectory grounding 学习 mode semantics：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}=-\log(q_{k^{oracle}}+\epsilon)
```

其作用是：

```text
训练期 assignment / matching
+
定义 K 个 trajectory hypotheses 的稳定语义
+
为 sharpened trajectory routing 提供 q
```

Posterior 不是 inference-time 模块。

### Sharpened trajectory routing

使用：

```math
\tilde q_k=
\frac{q_k^{1/\tau}}
{\sum_j q_j^{1/\tau}}
```

```math
L_{traj}=\sum_k stopgrad(\tilde q_k)L_k
```

其中：

```math
L_k=SmoothL1(B^{(k)},B^{gt})
```

固定 temperature schedule：

```yaml
tau_start: 1.0
tau_mid: 0.5
tau_end: 0.25
```

最终 task loss：

```math
L_{task}
=
L_{traj}
+
\lambda_{align}L_{align}
+
0.5L_{intent}
```

固定：

```text
lambda_align = 0.1
```

### Stage B 中 Past Mode Prior 的规则

**Past Mode Prior 不参与 Posterior semantics 的形成。**

主流程中：

```text
Past Mode Prior = frozen / not optimized
```

不得让：

```text
prior KL
predictability regularization
prior gradient
```

反向塑造 FutureModePosterior、trajectory decoder 或 mode embeddings。

只有 Stage B 完成并选定最终 specialization checkpoint 后，才进入 Prior refit。

---

## 4.3 Stage B 结束后的冻结点

Stage B 完成后，必须将以下内容视为已经固定的 mode semantics：

```text
context encoders
FutureModePosterior / prototypes
mode embeddings
future predictor / decoder
trajectory head
intention head
```

进入下一阶段前保存并记录：

```text
final_specialization_checkpoint
Posterior ADE
minADE@4
G_mode
Posterior↔Oracle
mode-wise recall / precision
```

这些权重在 Stage C 中全部冻结。

---

## 4.4 Stage C：Past Mode Prior 单独 refit

**Past Mode Prior 只在 Stage B 的 trajectory modes / Posterior semantics 稳定后训练。**

Prior 只使用过去 context：

```math
\pi(M\mid X_{past})
```

固定 Posterior 作为 teacher：

```math
L_{prior}
=
KL(stopgrad(q)\,\|\,\pi)
```

冻结：

```text
context encoders
FutureModePosterior / prototypes
mode embeddings
future predictor / decoder
trajectory head
intention head
EMA target encoders
```

仅训练：

```text
state-token MLP Past Mode Prior
```

JAAD 已验证配置：

```text
lr = 3e-4
max_epochs = 20
checkpoint selection = minimum validation prior KL
```

Stage C 的目的仅是：

> 把已经稳定的 future-aware mode semantics 映射为 past-only mode probabilities。

不得在此阶段重新修改 candidate trajectories 或 Posterior semantics。

---

## 4.5 为什么 Posterior 与 Prior 必须分阶段

最终方法明确采用：

```text
Posterior first
→ semantics fixed
→ Prior refit
```

而不是：

```text
Posterior + Prior joint training from scratch
```

已有实验依据：

1. Step 2 改变 Posterior semantics 后，旧 Prior 的 Top1 ADE 一度显著恶化；
2. Step 4A 在固定 Posterior 后单独 refit Prior，可显著恢复 Prior↔Posterior 一致率和 Top1 ADE；
3. Step 5A 让 past predictability 反向约束 Posterior 没有带来额外收益；
4. 因此最终不采用双向耦合或 Prior 反向塑造 Posterior。

最终 teacher-student 方向固定为：

```math
Y_{future}
\rightarrow
q(M\mid Y_{future})
\rightarrow
\pi(M\mid X_{past})
```

其中第二个箭头只发生在 Stage C 的 Prior refit 中。

---

## 4.6 正式 inference

测试时：

```text
FutureModePosterior = disabled
future GT            = unavailable
oracle assignment    = disabled
```

只使用：

```math
k^{top1}=\arg\max_k \pi_k(X_{past})
```

正式轨迹输出：

```math
B^{top1}=B^{(k^{top1})}
```

正式部署指标：

```text
Top1 ADE
Top1 FDE
box_ADE
box_FDE
final_IoU
intention F1 / AUC / AP
```

以下只作为训练机制 / 表征分析，不能作为部署指标：

```text
Posterior ADE
minADE@4
Posterior↔Oracle
Oracle mode
```

---

## 4.7 Checkpoint lineage 必须固定

最终报告必须明确给出：

```text
Stage A:
  JEPA pretrain checkpoint

Stage B:
  JEPA checkpoint
  → K=4 specialization checkpoint
  → final Posterior / trajectory-mode semantics

Stage C:
  frozen Stage B checkpoint
  → refit state-token MLP Prior
  → final deployable checkpoint
```

禁止：

```text
跨不同实验目录拼接未记录权重
用测试集选择 Stage B epoch
用测试集选择 Prior epoch
在 Prior refit 后重新开启 Posterior / decoder 训练
```

---

# 5. 最终方法锁定前必须完成的实验

## Experiment F0：实现与 checkpoint lineage 审计

Codex 首先检查：

```text
K=4 index mapping
posterior q[k]
mode embedding[k]
trajectory[k]
prior pi[k]
```

必须保持同一 mode index。

同时验证：

```text
future information 不进入 prior / inference path
R1 oracle target 只用于训练 posterior alignment
routing weight 对 q stop-gradient
Step 4A prior refit 时其余权重完全冻结
```

输出：

```text
final_method_audit.md
trainable_parameters_step2.txt
trainable_parameters_prior.txt
checkpoint_lineage.json
```

---

## Experiment F1：JAAD 单种子 clean reproduction

目的：证明最终方法不是由历史目录中多次实验拼接出来的偶然结果。

使用**新的输出目录**，从固定初始化重新完整执行：

```text
固定 JEPA initialization
→ Step 2 final specialization
→ Step 4A MLP prior refit
→ validation
→ test
```

不得复用历史 Step 2 / Step 4A 的最终权重作为 clean reproduction 的终点。

### 需要复现的参考量级

历史结果仅作为 sanity reference，不作为硬性要求：

```text
Step 2 Posterior ADE ≈ 18.517
Step 2 G_mode ≈ 1.504
Step 2 Posterior↔Oracle ≈ 77.75%
Step 2 minADE@4 ≈ 17.013
Step 4A Top1 ADE ≈ 21.855
Step 4A Top1 FDE ≈ 46.242
Step 4A G_prior ≈ 3.338
```

如果 clean reproduction 与历史结果差异很大，先检查实现 / checkpoint lineage / seed，不要直接修改方法。

---

## Experiment F2：最终核心消融

只保留最必要的四行：

| ID | Alignment | Routing | Prior refit | 目的 |
|---|---|---|---|---|
| A0 | off | original soft-q | original/current | 原多模态 baseline |
| A1 | R1 hard-oracle | original soft-q | refit MLP | alignment 单独贡献 |
| A2 | R1 hard-oracle | sharpened | old/unfitted prior | representation/mode specialization |
| A3 | R1 hard-oracle | sharpened | **refit MLP** | **最终方法** |

要求：

- 同一初始化；
- 同一训练预算；
- 相同数据划分；
- 相同 checkpoint 规则；
- 不跨行拼 checkpoint。

主结论必须来自 A3 相对 A0/A1/A2 的成对比较。

---

# 6. 最终方法必须报告的三层 ADE 分解

对 A3 同时报告：

```math
ADE_{oracle}=minADE@4
```

```math
ADE_{posterior}
```

```math
ADE_{prior}=Top1\ ADE
```

以及：

```math
G_{mode}=ADE_{posterior}-ADE_{oracle}
```

```math
G_{prior}=ADE_{prior}-ADE_{posterior}
```

```math
G_{total}=ADE_{prior}-ADE_{oracle}
```

这三层必须清楚区分：

```text
Oracle / minADE@4      = 候选集合能力上限
Posterior ADE          = future-aware mode semantics 诊断
Prior Top1 ADE         = 真正 inference-time 性能
```

---

# 7. JAAD 多随机种子确认

在 F1 单种子 clean reproduction 通过后，至少对最终 A3 和核心 baseline A0 做：

```text
seed = 42 / 43 / 44
```

报告：

```text
mean ± std
```

至少包括：

```text
Top1 ADE / FDE
minADE@4 / minFDE@4
Posterior ADE
G_mode
G_prior
Intent F1
Intent AP
```

如果资源有限：

```text
A3 必须 3 seeds
A0 至少已有可比 baseline；若无 3 seeds，则明确标注单种子限制
```

禁止从三个种子中只选最好测试结果。

---

# 8. PIE 跨数据集确认

JAAD final method 锁定后，将**同一方法设计**迁移到 PIE。

允许：

```text
沿用 PIE 原本 dataset-specific optimizer / batch / LR 设置
```

不允许基于 PIE test 重新调整：

```text
K
lambda_align
responsibility type
routing type
routing temperature schedule
prior architecture
```

PIE 至少报告：

```text
Top1 ADE / FDE
minADE@4 / minFDE@4
Posterior ADE
G_mode
G_prior
Prior↔Posterior
Posterior↔Oracle
Intent F1 / AUC / AP
```

如果 PIE 不复现 JAAD 的收益，不得静默调方法；应作为 cross-dataset limitation 分析。

---

# 9. Crossing / Non-crossing 分组必须保留

最终 JAAD 和 PIE 都至少报告：

```text
all
crossing=0
crossing=1
```

每组：

```text
Top1 ADE / FDE
Posterior ADE
G_mode
G_prior
mode-wise recall / precision
```

原因：已有 JAAD 证据表明 crossing 的 prior selection 明显更困难，不能只用总体平均掩盖该问题。

---

# 10. 最终方法锁定成功标准

满足以下条件后，停止继续加模块，并标记方法为 `LOCKED`。

## 10.1 必须满足

- [ ] clean reproduction 可完整从固定初始化跑通；
- [ ] future leakage test 通过；
- [ ] K=4 mode index 一致性测试通过；
- [ ] Step 2 的 `G_mode` 相对原 baseline 有明确改善；
- [ ] mode 1 不再维持原来的近死亡状态；
- [ ] prior refit 明显恢复 posterior 语义匹配；
- [ ] A3 inference 只依赖 past；
- [ ] final Top1 ADE/FDE 不依赖 posterior/oracle future 信息；
- [ ] JAAD 结果至少能稳定复现当前量级；
- [ ] 3-seed 最终方法没有出现完全相反的结论；
- [ ] PIE 用相同方法设计完成一次完整验证。

## 10.2 不要求满足

以下不是锁定必要条件：

```text
Posterior↔Oracle 达到 100%
G_prior 接近 0
Crossing ADE 必须显著优于所有历史版本
minADE@4 必须优于所有实验
Intention AP 必须在 Step 2 specialization 阶段达到最高
```

最终方法允许保留清楚、可解释的 limitation。

---

# 11. 最终不再继续探索的条件

以下情况不要再新增复杂结构：

- Cross-Attention Prior 已为负结果；
- predictability regularization 已未证明有效；
- Step 5D 二次 JEPA 已未证明有效；
- posterior-guided hard routing 缺乏必要性且可能放大错误 assignment；
- simple Posterior 已经通过 Step 1B / Step 2 得到明确正证据。

因此最终阶段的目标从：

```text
继续追求更复杂模型
```

切换为：

```text
复现
多种子
跨数据集
消融
机制分析
论文证据完善
```

---

# 12. 最终输出文件

Codex 最终创建：

```text
outputs/final_method_lock/
  method_config_locked.yaml
  checkpoint_lineage.json
  final_method_audit.md
  jaad_seed42/
  jaad_seed43/
  jaad_seed44/
  pie/
  ablations/
  final_results.json
  final_results.csv
  FINAL_METHOD_REPORT.md
```

`FINAL_METHOD_REPORT.md` 至少包含：

1. 最终固定方法；
2. 被排除模块及对应负结果；
3. JAAD clean reproduction；
4. JAAD multi-seed；
5. PIE transfer；
6. A0/A1/A2/A3 消融；
7. Oracle / Posterior / Prior 三层 ADE 分解；
8. crossing/non-crossing 分析；
9. intention 结果；
10. 参数量、训练时间、推理时间；
11. 当前 limitation；
12. `LOCKED / NOT LOCKED` 最终状态。

---

# 13. 必须保留 / 新增的测试

至少保证：

```text
test_k4_mode_index_alignment
test_no_future_leakage_in_prior
test_no_future_leakage_in_inference
test_oracle_only_used_for_training_alignment
test_routing_weights_stop_gradient
test_sharpened_routing_schedule
test_step2_trainable_modules
test_prior_refit_freezing
test_top1_uses_prior_only
test_minade_is_per_sample_oracle
test_checkpoint_lineage_consistency
test_locked_config_reproducible
```

K=1 sanity 继续保留。

---

# 14. 给 Codex 的最终执行指令

> 当前任务是“最终证实并锁定方法”，不是继续做结构探索。最终候选固定为：现有 JEPA future representation + K=4 mode conditioning + R1 Hard-Oracle Posterior Alignment + sharpened soft trajectory routing + simple FutureModePosterior/prototypes + state-token MLP Past Mode Prior，并在 posterior semantics 稳定后单独 refit prior。不要启用 Cross-Attention Prior、predictability-aware regularization、posterior-guided hard routing、Plan Transformer、Step 5D 二次 JEPA re-alignment 或其他新模块。首先审计并保存完整 checkpoint lineage、mode index mapping、future leakage 与 trainable parameter freezing；随后用新的输出目录从固定初始化做 JAAD clean reproduction，完整执行 final Step 2 specialization 与 Step 4A prior refit。完成 A0/A1/A2/A3 最小核心消融，并严格区分 `minADE@4`、Posterior ADE 与真正部署的 Prior Top1 ADE。单种子复现通过后，对最终方法至少运行 JAAD seeds 42/43/44，报告 mean±std，不从多个种子中挑最好结果。之后把完全相同的创新设计迁移到 PIE，只允许沿用已有 dataset-specific optimizer 设置，不允许根据 PIE test 改 K、alignment、routing schedule 或 prior architecture。最后输出 `method_config_locked.yaml`、`checkpoint_lineage.json`、`final_results.json/csv` 与 `FINAL_METHOD_REPORT.md`；只有当 clean reproduction、多种子、无 future leakage、核心消融和 PIE transfer 全部完成后才标记方法为 `LOCKED`。
