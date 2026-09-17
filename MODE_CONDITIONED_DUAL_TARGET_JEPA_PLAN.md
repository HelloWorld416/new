# EgoJEPA-MP：Mode-Conditioned Dual-Target JEPA 改造计划

## 0. 研究目标

当前 EgoJEPA 使用单一确定性 future latent：

```math
C \rightarrow \hat{Z}_{1:T}
```

其中历史视觉、运动与自车信息融合为 context，30 个 future query 预测唯一的未来潜在序列。该设计面对同一历史对应多个合理未来时，会产生一个关键问题：deterministic JEPA 为了降低预测难度，可能倾向于弱化或过滤那些“不可由过去唯一确定、但对下游轨迹和意图很重要”的未来差异。

本轮改造的目标是把模型改造成：

```math
C \rightarrow p(M \mid C)
```

以及：

```math
(C, M_k) \rightarrow \hat{Z}^{(k)}_{1:T}, \qquad k=1,\dots,K
```

其中：

- `C`：历史视觉 + 运动 + ego context；
- `M`：latent pedestrian future plan / behavioral mode；
- `K`：未来行为模式数，第一版默认 `K=4`；
- 每个 mode 产生一组 `[30, 256]` future latent；
- 同一组 mode-conditioned future latent 同时支持轨迹与过街意图预测。

最终核心思想：

> JEPA 继续负责学习稳定、可预测的 future structure；latent future plan 显式保留同一历史条件下仍然不确定的行为分支，使多模态未来从 representation level 就存在，而不是只在最后的轨迹头中额外采样。

---

## 1. 目标输出接口

新模型在 multimodal 模式下应支持：

```text
mode_probabilities: [B, K]
future_latents:     [B, K, 30, 256]
boxes:              [B, K, 30, 4]
intent_logits:      [B, K]
```

最终意图概率：

```math
p(y=1 \mid C)=\sum_k \pi_k \sigma(l_k)
```

默认部署轨迹：

```math
k^{*}=\arg\max_k \pi_k
```

使用 top-1 mode 轨迹作为 K=1-equivalent 结果，同时保留 K 条候选轨迹用于 `minADE@K / minFDE@K`。

---

# Phase 0：冻结并核验当前 baseline

在改算法之前，先检查并记录当前实现，重点阅读：

- `src/egojepa_diffusion/models/model.py`
- `src/egojepa_diffusion/models/layers.py`
- `src/egojepa_diffusion/pretrain.py`
- `src/egojepa_diffusion/train.py`
- `src/egojepa_diffusion/training.py`
- `src/egojepa_diffusion/data/dataset.py`
- `src/egojepa_diffusion/evaluate.py`
- `src/egojepa_diffusion/metrics.py`
- `src/egojepa_diffusion/diagnostics.py`
- JAAD / PIE 配置文件

需要确认：

1. 当前 summed target 的实现位置；
2. 30-query future predictor 的输入输出 shape；
3. EMA teacher 更新方式；
4. trajectory head 与 intention temporal-attention head；
5. Stage A / B / C 的冻结与解冻逻辑；
6. evaluation 当前 K=1 路径；
7. checkpoint 和旧 config 的加载逻辑。

### 验收标准

- 旧配置无需新增字段即可运行；
- 旧 checkpoint 仍能加载；
- 旧 K=1 forward / evaluation 行为保持不变；
- 新增一个最小 smoke test，先固定 baseline。

---

# Phase 1：Dual-Target JEPA

## 1.1 当前问题

当前 target：

```math
z^{*}=norm(v^{*}+m^{*})
```

视觉未来和运动未来在监督前已经相加，可能丢失模态专属信息。

## 1.2 改造

保留两个独立 EMA target：

```math
V^{*}\in\mathbb{R}^{B\times30\times256}
```

```math
M^{*}\in\mathbb{R}^{B\times30\times256}
```

同一个 predicted future latent 通过两个轻量 projector：

```math
q_v(\hat{Z}), \qquad q_m(\hat{Z})
```

分别拟合：

```math
L_{dual}^{(k)}
=
\lambda_v d(q_v(\hat{Z}^{(k)}),V^{*})
+
\lambda_m d(q_m(\hat{Z}^{(k)}),M^{*})
```

`d` 第一版继续使用 normalized cosine distance。

## 1.3 配置

必须保留旧路径：

```yaml
jepa:
  target_type: summed
```

新增：

```yaml
jepa:
  target_type: dual
  dual_target:
    visual_weight: 1.0
    motion_weight: 1.0
    projector_hidden_dim: 256
```

第一阶段只做 `K=1`，先单独验证 dual target。

---

# Phase 2：Future Plan Posterior

这是本轮改造的核心。

训练阶段拥有真实 future motion，因此可让未来本身推断“该样本最终属于哪一种行为模式”。

## 2.1 FutureModePosterior

输入：

```text
future_motion_target: [B, 30, 256]
```

首先做 temporal pooling：

```math
h_f=Pool(M^{*}_{1:30})
```

维护 K 个 learnable prototype：

```math
e_1,\dots,e_K
```

soft assignment：

```math
q_k
=
softmax\left(\frac{sim(h_f,e_k)}{\tau}\right)
```

输出：

```text
posterior_logits: [B, K]
posterior_probs:  [B, K]
```

## 2.2 默认配置

```yaml
jepa:
  multimodal:
    enabled: true
    num_modes: 4
    temperature: 0.1
    mode_source: motion_target
    assignment: soft
```

## 2.3 重要约束

- Stage A 不使用 crossing label 定义 mode；
- mode 默认只由 future motion teacher 决定；
- 不使用 future image 来主导 mode assignment，避免 prototype 主要编码背景、天气或纹理；
- mode 是 latent behavior structure，不等同于人为定义的 crossing / non-crossing。

---

# Phase 3：Past Mode Prior

新增 `ModePrior`，从过去 context 预测 future mode distribution：

```math
\pi=p(M \mid C)
```

第一版建议直接使用 context 的 state token：

```text
context[:, 0]
→ LayerNorm
→ Linear
→ GELU
→ Linear(K)
```

输出：

```text
prior_logits: [B, K]
prior_probs:  [B, K]
```

训练目标：

```math
L_{prior}
=
D_{KL}(sg[q(M \mid Y)] \;\|\; p(M \mid C))
```

含义：

- `q(M|Y)`：真实未来告诉模型最终发生了哪类 future behavior；
- `p(M|C)`：历史观测预测各 future plan 的可能性。

模型不再被要求从 past 直接恢复“唯一正确未来”。

---

# Phase 4：Mode-Conditioned Future Predictor

不要复制 K 个 decoder。

保留当前共享 future predictor，只增加 learnable mode embedding：

```python
mode_embeddings = nn.Embedding(K, D)
```

当前 query `Q_t` 改成：

```math
Q_t^{(k)}=Q_t+e_k
```

然后共享 decoder：

```math
\hat{Z}^{(k)}=P(C,Q^{(k)})
```

最终：

```text
future_latents: [B, K, T, D]
```

工程上建议 reshape：

```text
context: [B,S,D]
→ repeat K
→ [B*K,S,D]

queries: [B,K,T,D]
→ [B*K,T,D]

decoder
→ [B*K,T,D]
→ reshape
→ [B,K,T,D]
```

这样参数量基本不随 K 增长。

配置：

```yaml
jepa:
  multimodal:
    enabled: true
    num_modes: 4
    condition_type: query_add
```

---

# Phase 5：新的 Stage-A JEPA Loss

对每个 mode 计算 dual-target loss `L_dual^(k)`，由 future posterior 加权：

```math
L_{multiJEPA}
=
\sum_{k=1}^{K} q_k L_{dual}^{(k)}
```

再加入 past mode prior：

```math
L_{prior}=D_{KL}(sg[q] \;\|\; \pi)
```

以及轻量 mode usage regularization `L_usage`。

最终：

```math
L_A
=
L_{multiJEPA}
+
\lambda_p L_{prior}
+
\lambda_u L_{usage}
```

默认建议：

```yaml
lambda_mode_prior: 0.5
lambda_mode_usage: 0.01
```

## 5.1 Stage A 不使用意图标签

主模型的 Stage A 必须保持 JEPA-only：

- 不把 crossing label 作为 mode supervision；
- 不把 intention BCE 加进主预训练 loss；
- 之后可单独做 `+ pretrain intention supervision` 消融。

如果无 intention label 的 Stage A 仍然能学出与未来 crossing / motion structure 相关的 latent modes，这会成为更强的 representation evidence。

---

# Phase 6：Mode Collapse 监控与轻量正则

第一版不要引入 Sinkhorn、VQ、Gumbel。

首先记录 batch-level posterior：

```math
\bar{q}_k=\frac{1}{B}\sum_b q_{bk}
```

计算：

```math
effective\_modes=\exp(H(\bar{q}))
```

至少记录：

```text
mode_usage
mode_entropy
prior_entropy
posterior_entropy
effective_num_modes
prior_posterior_kl
per_mode_jepa_loss
visual_jepa_loss
motion_jepa_loss
```

如果出现明显 dead mode，再加入 minimum-usage penalty，例如：

```math
L_{usage}
=
\sum_k \max(0,m-\bar{q}_k)^2
```

原则：

> 只防止 mode 完全死亡，不强制 K 个 mode 等频。

---

# Phase 7：Stage B — 固定表示训练任务头

Stage B 保留当前“固定表示可读出性”的研究目的。

冻结：

```text
context encoder
future predictor
mode embeddings
mode prior
future posterior
EMA teacher
dual projectors（若仅用于 JEPA）
```

训练：

```text
trajectory head
intention temporal attention
intention classifier
```

这样可以回答：

> multimodal JEPA 预训练出来的 future representation 本身，是否已经支持轨迹与意图读出？

---

# Phase 8：Multimodal Trajectory Head

轨迹头在所有 mode 间共享：

```math
\hat{B}^{(k)}=H_{traj}(\hat{Z}^{(k)})
```

输出：

```text
boxes_all: [B, K, 30, 4]
```

训练使用 future posterior：

```math
L_{traj}
=
\sum_k q_k SmoothL1(\hat{B}^{(k)},B^{gt})
```

第一版不使用纯 best-of-K：

```math
\min_k L_k
```

因为主研究问题要求 mode assignment 由 future representation 定义，而不是由 trajectory error 临时选择。

---

# Phase 9：Mode-Conditioned Intention Prediction

每个 mode latent 通过同一个 intention head：

```math
l_k=H_{intent}(\hat{Z}^{(k)})
```

得到：

```text
intent_logits_all: [B, K]
```

最终 mixture crossing probability：

```math
p_{mix}=\sum_k \pi_k \sigma(l_k)
```

## 9.1 Mixture supervision

```math
L_{intent}^{mix}=BCE(p_{mix},y)
```

## 9.2 Realized-mode supervision

```math
L_{intent}^{assigned}
=
\sum_k q_k BCE(l_k,y)
```

最终：

```math
L_{intent}
=
L_{intent}^{mix}
+
\eta L_{intent}^{assigned}
```

默认：

```yaml
task:
  intention:
    assigned_mode_weight: 0.5
```

重要：

> 不要简单地把同一个 crossing label 强行监督给全部 K 个 mode，否则会破坏 mode specialization。

---

# Phase 10：Stage C — Predictor Adaptation + Weak JEPA Anchoring

Stage C 建议冻结：

```text
context encoder
EMA teacher
future posterior
```

训练：

```text
future predictor
mode embeddings
mode prior
trajectory head
intention head
dual projectors
```

Loss：

```math
L_C
=
L_{traj}
+
0.5L_{intent}
+
\beta L_{multiJEPA}
+
\gamma L_{prior}
```

初始建议：

```yaml
training:
  stage_c:
    jepa_weight: 0.1
    mode_prior_weight: 0.1
```

同时必须支持：

```yaml
training:
  stage_c:
    jepa_weight: 0.0
```

用于消融。

该模块回答：

> downstream trajectory / intention supervision 是否会再次把预训练学出的 multimodal future structure 压平？

---

# Phase 11：Dataset / DataLoader 条件加载

Codex 必须检查 Stage C 当前是否加载 future RGB。

要求：当 `stage_c.jepa_weight > 0` 时才需要 future image / future teacher target。

若：

```text
stage_c.jepa_weight == 0
```

则不应引入额外 future RGB IO 与显存消耗。

保持 backward compatibility。

---

# Phase 12：Inference API

建议统一返回：

```python
{
    "mode_probs":        [B, K],
    "future_latents":    [B, K, 30, 256],
    "boxes_all":         [B, K, 30, 4],
    "intent_logits_all": [B, K],
    "intent_prob":       [B],
    "top1_mode":         [B],
    "boxes_top1":        [B, 30, 4],
}
```

其中：

```math
k^{*}=\arg\max_k \pi_k
```

`boxes_top1` 用于和当前 deterministic K=1 baseline 公平比较。

---

# Phase 13：评价指标

## 13.1 Top-1 deployment metrics

基于最高 prior probability mode：

```text
ADE_top1
FDE_top1
box_ADE_top1
box_FDE_top1
final_IoU_top1
```

## 13.2 Multimodal oracle metrics

额外报告：

```text
minADE@K
minFDE@K
```

必须明确：

> `ADE_top1` 与 `minADE@K` 是两种不同指标，不能用后者和旧 K=1 ADE 直接混报为“提升”。

## 13.3 Intention metrics

继续：

```text
F1
Precision
Recall
ROC-AUC
Accuracy
Balanced Accuracy
```

新增：

```text
PR-AUC / Average Precision
```

---

# Phase 14：Representation Diagnostics

`diagnostics.py` 至少新增：

1. 每个 mode 的使用比例；
2. posterior entropy；
3. prior entropy；
4. prior / posterior KL；
5. effective number of modes；
6. 每个 mode 平均 future endpoint；
7. 每个 mode displacement 分布；
8. 每个 mode velocity change；
9. 每个 mode crossing positive ratio；
10. 每个 mode trajectory ADE；
11. trajectory / intention task gradient cosine。

重点检查：

```math
M \leftrightarrow y_{cross}
```

但理想情况不是：

```text
mode 0 = non-cross
mode 1 = cross
```

而应该表现为：

- crossing 内仍有多个 motion modes；
- non-crossing 内也可能有多个 motion modes；
- mode 主要反映 future behavioral dynamics，而不是退化成二分类器。

---

# Phase 15：配置建议

```yaml
jepa:
  target_type: dual

  multimodal:
    enabled: true
    num_modes: 4
    temperature: 0.1
    mode_source: motion_target
    assignment: soft
    condition_type: query_add

  dual_target:
    visual_weight: 1.0
    motion_weight: 1.0
    projector_hidden_dim: 256

  mode_prior:
    weight: 0.5

  usage_regularization:
    enabled: true
    weight: 0.01

training:
  stage_c:
    jepa_weight: 0.1
    mode_prior_weight: 0.1

task:
  intention:
    assigned_mode_weight: 0.5
```

旧配置必须继续支持：

```yaml
jepa:
  target_type: summed
  multimodal:
    enabled: false
```

并且行为与当前 deterministic baseline 一致。

---

# Phase 16：实验矩阵

至少执行：

| ID | Target | Future Mode | Stage-C JEPA |
|---|---|---:|---:|
| M0 | summed | K=1 | 0 |
| M4 | dual | K=1 | 0 |
| M5 | summed | K=4 | 0 |
| M6 | dual | K=4 | 0 |
| M7 | dual | K=4 | 0.1 |

再做：

```text
K = 2 / 4 / 8
```

回答三个问题：

1. dual target 是否有价值；
2. multimodal future representation 是否有价值；
3. Stage-C weak JEPA anchoring 是否减少 representation drift。

---

# Phase 17：Codex Commit 计划

建议拆成四个独立 commit。

## C1 — Dual Target

```text
feat: add dual-target JEPA with legacy compatibility
```

内容：

- 拆分 visual / motion target；
- dual projectors；
- dual JEPA loss；
- legacy summed target 保留；
- K=1 测试。

## C2 — Latent Future Modes

```text
feat: add future mode posterior and mode-conditioned predictor
```

内容：

- `FutureModePosterior`；
- prototypes；
- `ModePrior`；
- mode embeddings；
- shared decoder 的 K-mode forward；
- mode diagnostics 基础统计。

## C3 — Downstream Tasks

```text
feat: add multimodal trajectory and intention training with JEPA anchoring
```

内容：

- posterior-weighted trajectory loss；
- mixture intention；
- assigned-mode intention loss；
- Stage B freezing；
- Stage C weak JEPA anchoring。

## C4 — Evaluation & Diagnostics

```text
feat: add multimodal evaluation diagnostics and tests
```

内容：

- top-1 trajectory；
- minADE@K / minFDE@K；
- AP / PR-AUC；
- mode entropy / usage；
- representation diagnostics；
- 完整测试。

---

# Phase 18：第一版不要加入的内容

为了保证实验可归因，第一版不要同时加入：

```text
diffusion
VQ-VAE
Sinkhorn
Gumbel hard sampling
复杂 contrastive loss
ROI / bbox-guided visual attention
新的 backbone
future ego action conditioning
predictable-core / residual 双 decoder
```

第一版核心只回答：

```math
Deterministic\ JEPA
\rightarrow
Mode\text{-}Conditioned\ JEPA
```

是否改善 trajectory + intention，并保留更好的 multimodal future representation。

---

# Phase 19：单元测试与验收标准

至少增加：

```text
test_dual_target_shapes
test_multimodal_shapes
test_mode_probabilities_sum_to_one
test_teacher_stop_gradient
test_future_posterior_stop_gradient_where_required
test_stage_a_no_intention_loss
test_stage_b_freezing
test_stage_c_freezing
test_k1_backward_compatibility
test_mode_weighted_trajectory_loss
test_intention_mixture_probability
test_top1_selection
test_minade_k
```

额外验收：

```text
旧配置 forward 输出与当前模型一致
K=1 新路径与 deterministic tensor shape 一致
K=4 可完成小 batch forward + backward
posterior / prior 无 NaN 或 Inf
mode_probs 每行和为 1
future_latents = [B,K,30,256]
boxes_all = [B,K,30,4]
intent_logits_all = [B,K]
Stage A 不用 intention label 计算 loss
Stage B 冻结 representation / predictor
Stage C 只有指定模块 require_grad=True
EMA teacher 无梯度
evaluation 同时支持 top1 ADE 和 minADE@K
旧 checkpoint / config 不因新字段报错
```

---

# Phase 20：给 Codex 的最终执行指令

> 基于当前 EgoJEPA main 分支，实现一个 backward-compatible 的 Mode-Conditioned Dual-Target JEPA。不要重构与本任务无关的代码。首先阅读并总结 `models/model.py`、`models/layers.py`、`pretrain.py`、`train.py`、`training.py`、`dataset.py`、`evaluate.py`、`metrics.py`、`diagnostics.py` 以及 JAAD/PIE 配置，确认当前 summed target、30-query future predictor、EMA teacher、三阶段冻结策略和任务头实现。然后按四个独立阶段实现：C1 Dual Target；C2 Future Mode Posterior + Past Mode Prior + mode-conditioned shared decoder；C3 multimodal trajectory/intention heads and Stage-C weak JEPA anchoring；C4 evaluation, diagnostics and tests。保持旧 summed-target、K=1 路径完全可运行。Stage A 不使用 intention label；future mode posterior 默认仅从 future motion EMA representation 推断；past branch 学习 mode distribution；K 个 mode 必须共享同一个 future decoder，通过 learnable mode embedding 调制 future queries。不要引入 diffusion、VQ、Gumbel、Sinkhorn、ROI attention、新 backbone 或无关重构。每个阶段完成后先运行 unit tests 和小 batch forward/backward，确保没有 NaN，并输出改动文件、tensor shape、参数冻结情况和测试结果，再进入下一阶段。

---

# 研究定位总结

这轮创新不是简单地“给轨迹预测增加 K 个候选”，而是针对 deterministic JEPA 在 multimodal future 下可能产生的 representation bias：

1. **Dual Target**：避免 visual / motion future 在 target 相加时过早丢失模态专属信息；
2. **Future-inferred Latent Plan**：用真实 future motion 自监督地发现行为模式；
3. **Past Mode Prior**：让历史输入预测未来行为模式的分布，而不是唯一未来；
4. **Mode-conditioned JEPA**：给定 mode 后，再预测相对确定的 future latent；
5. **Shared Trajectory + Intention Readout**：轨迹和意图都从同一组 mode-conditioned latent 中读取；
6. **Weak JEPA Anchoring**：避免 downstream supervision 把预训练阶段学到的 multimodal future structure 再次压平。

核心研究问题可概括为：

> **How to preserve decision-relevant multimodality in predictive JEPA representations for unified pedestrian trajectory and intention prediction?**
