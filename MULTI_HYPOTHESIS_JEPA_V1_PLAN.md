# EgoJEPA-MH v1：Multiple-Hypothesis Latent JEPA 第一版改造计划

> 本文基于 `HelloWorld416/EgoJEPA` 当前 `main` 分支代码制定，目标是让 Codex 在**尽量少改现有框架**的前提下，把当前 deterministic future latent regression 改造成内部 K 个 latent hypotheses，但最终 JAAD / PIE 仍然只输出 **K=1 trajectory**。
>
> 第一版只研究**轨迹预测**。不把意图预测作为优化目标，不引入 diffusion、flow matching、Past Mode Prior、FutureModePosterior、RL/REINFORCE、VQ、Gumbel 或新的 backbone。
>
> 核心研究问题：
>
> > 当前 JEPA 只预测一个 future latent，可能把 one-to-many future 压成平均表示。是否可以通过多个内部 latent hypotheses + winner-take-all future grounding，让 JEPA 保留多种合理未来，同时学习一个 past-conditioned probability head，在测试时只选一条轨迹，从而改善正式 K=1 ADE/FDE？

---

# 0. 当前 main 分支基线事实

Codex 开始修改前必须先核对当前 `main`，不要假设历史实验代码仍存在。

当前已确认的关键实现：

## 0.1 模型

文件：

```text
src/egojepa_diffusion/models/model.py
src/egojepa_diffusion/models/layers.py
```

当前 deterministic 路径：

```text
past visual + past motion + ego
        ↓
context memory [B, 33, 256]
        ↓
LatentPredictor
        ↓
future latent [B, 30, 256]
        ↓
trajectory_head
        ↓
future encoded boxes [B, 30, 4]
```

`LatentPredictor` 当前只有一组 learnable future queries：

```python
self.queries: [1, T, D]
```

`sample()` 当前明确只允许：

```text
K = 1
```

## 0.2 JEPA target

当前 teacher target：

```math
z^*
=
normalize(
normalize(z^v)
+
normalize(z^m)
)
```

当前 JEPA loss：

```math
L_{JEPA}
=
1
-
cos(\hat Z, Z^*)
```

文件：

```text
src/egojepa_diffusion/pretrain.py
```

当前 pretraining：

```text
30 epochs
AdamW
lr = 3e-4
best checkpoint = minimum validation JEPA loss
EMA target momentum = 0.996
```

## 0.3 下游 trajectory loss

当前：

```math
L_{traj}
=
SmoothL1(\hat B, B^{gt})
```

文件：

```text
src/egojepa_diffusion/training.py
```

第一版新实验不使用 intention loss 作为优化项。

---

# 1. 第一版方法总览

最终结构：

```text
Past Context C
    ↓
Shared K-hypothesis Latent Predictor
    ↓
Z^(1), Z^(2), ..., Z^(K)
    ↓
Shared Trajectory Head
    ↓
B^(1), B^(2), ..., B^(K)
    ↓
Training:
  Best-of-K assignment → winner k*
  winner JEPA + winner trajectory regression
  probability head learns to rank winner
    ↓
Inference:
  probability p_1...p_K
  argmax probability
    ↓
ONE final trajectory
```

固定第一版：

```text
K = 4
```

正式 JAAD / PIE 测试仍然：

```text
K = 1 output
```

`minADE@4` 只作为内部诊断，不作为正式主结果。

---

# 2. 术语修正：使用概率，不使用“单 latent 信息熵”

不要为每个 latent 定义所谓：

```text
latent_entropy_k
```

来表示出现概率。

正确变量：

```text
score/logit:      s_k
probability:      p_k
sample entropy:   H(p)
mode surprisal:   -log p_k
```

概率：

```math
p_k
=
\frac{\exp(s_k / \tau_p)}
{\sum_j \exp(s_j / \tau_p)}
```

整个 hypothesis distribution 的熵：

```math
H(p)
=
-\sum_k p_k \log(p_k + \epsilon)
```

解释：

- `p_k`：第 k 个 latent hypothesis 的相对未来 plausibility；
- `H(p)`：当前样本 future ambiguity / uncertainty；
- 测试时选择最大 `p_k`，不是最大 entropy。

---

# 3. Stage A：保持当前 deterministic JEPA pretraining 不变

第一版不要直接重写 JEPA pretraining。

继续运行当前：

```text
pretrain.py
```

得到 deterministic JEPA checkpoint：

```text
best.pt
```

原因：

1. 当前 main 分支 JEPA pretraining 路径已经稳定；
2. 第一版先单独验证“multi-hypothesis latent specialization”是否有价值；
3. 避免同时修改 pretraining + downstream，导致无法归因。

因此：

```text
Stage A = current main branch JEPA pretraining
```

完全保持：

```math
L_{JEPA}^{base}
=
1-cos(\hat Z,Z^*)
```

---

# 4. Stage B：把 deterministic latent predictor 扩展为 K=4 latent hypotheses

## 4.1 不复制 K 个 decoder

保留一个 shared Transformer decoder。

新增：

```python
mode_embeddings = nn.Embedding(K, D)
```

当前 query：

```math
Q_t
```

改成：

```math
Q_t^{(k)}
=
Q_t + e_k
```

然后：

```math
Z^{(k)}
=
P(C,Q^{(k)})
```

输出：

```text
future_latents: [B, K, T, D]
```

其中第一版：

```text
K = 4
T = 30
D = 256
```

工程实现建议：

```text
[B, K, T, D]
→ flatten B*K
→ shared Transformer decoder
→ reshape [B, K, T, D]
```

旧 deterministic path 必须可通过配置保留。

---

# 5. K 个 latent 共用一个 trajectory decoder

继续使用当前：

```text
trajectory_head
```

不要复制 K 个 trajectory heads。

```math
B^{(k)}
=
H_{traj}(Z^{(k)})
```

输出：

```text
trajectory_all: [B, K, 30, 4]
```

共享 head 可以确保差异主要来自 latent hypothesis，而不是 K 个独立 decoder 参数。

---

# 6. Best-of-K Assignment

训练时真实 future 只用于确定：

> 当前样本应该主要监督哪一个 latent hypothesis。

## 6.1 Assignment metric

第一版使用 pixel center ADE 做 assignment：

```math
E_k^{assign}
=
ADE(B^{(k)},B^{gt})
```

然后：

```math
k^*
=
\arg\min_k E_k^{assign}
```

注意：

- ADE 只用于 assignment；
- `argmin` / winner index 必须 stop-gradient；
- 不允许用 GT 在测试时选择 trajectory。

## 6.2 为什么 assignment 与 regression loss 分开

winner 的真正 regression loss 继续使用当前项目稳定的 encoded-box SmoothL1：

```math
L_{traj}^{WTA}
=
SmoothL1(
B_{enc}^{(k^*)},
B_{enc}^{gt}
)
```

所以：

```text
pixel ADE   → decide winner
SmoothL1    → optimize winner
```

---

# 7. Winner-Takes-JEPA

这是第一版最关键的 JEPA 改动。

不能对 K 个 latent 都使用同一个 JEPA target：

```math
\sum_k d(Z^{(k)}, Z^*)
```

否则所有 hypotheses 会重新 collapse 到同一个 future latent。

因此只让 winner 匹配真实 future teacher target：

```math
L_{JEPA}^{WTA}
=
1
-
cos(
Z^{(k^*)},
Z^*
)
```

其他 hypotheses：

```text
不接受当前样本同一个 JEPA target 的直接拉近
```

这使得：

> 每个训练样本只把最适合它的 latent hypothesis 拉向真实 future representation。

方法的核心从：

```text
single deterministic JEPA regression
```

变成：

```text
multiple-choice JEPA regression
```

---

# 8. Probability / Ranking Head

## 8.1 第一版 scorer 输入

每个 hypothesis 都得到一个 score：

```math
s_k=g(C,Z^{(k)})
```

为了保持第一版简单：

Past summary：

```math
c = LN(C[:,0])
```

Future hypothesis summary：

```math
z_k
=
MeanPool(Z^{(k)})
```

融合：

```math
h_k
=
[c,z_k,c\odot z_k,|c-z_k|]
```

shared MLP：

```math
s_k
=
MLP(h_k)
```

K 个 hypotheses **共享同一个 scorer 参数**。

输出：

```text
scores: [B, K]
probabilities: [B, K]
```

---

# 9. 第一版不使用 RL，直接用 Winner Classification

用户原始想法是：

> winner reward，losers punishment，通过强化学习优化 hypothesis probability。

第一版不使用 REINFORCE。

原因：训练时已经计算全部 K 条轨迹，因此 winner label 是完整可用监督：

```math
k^*
=
\arg\min_k ADE_k
```

直接训练：

```math
L_{score}
=
-\log(p_{k^*}+\epsilon)
```

softmax 会自动：

```text
increase winner probability
decrease loser probabilities
```

已经实现“奖励 winner、惩罚 losers”，但比 REINFORCE 更稳定、方差更低、实验更容易解释。

真正 RL 只作为未来可选扩展，不进入第一版。

---

# 10. 可选 Expected-Risk Loss：第一版仅作为消融

如果 winner CE 有效，再额外测试：

```math
L_{risk}
=
\sum_k
p_k
\cdot
stopgrad(E_k^{assign})
```

组合：

```math
L_{score}^{+risk}
=
L_{score}
+
\lambda_r L_{risk}
```

但第一版主模型：

```text
lambda_r = 0
```

不要一开始同时加。

---

# 11. 防止 hypothesis collapse

WTA 可能产生：

```text
一个 hypothesis 早期赢得更多
→ 得到更多监督
→ 更容易继续赢
→ 其他 hypotheses 死亡
```

因此第一版需要：

## 11.1 必须记录

每个 epoch：

```text
winner_count_mode0
winner_count_mode1
winner_count_mode2
winner_count_mode3

winner_frequency
mean_probability_per_mode
sample_entropy_mean
pairwise_latent_cosine
pairwise_trajectory_distance
```

## 11.2 轻量 balance regularizer

只在出现明显 collapse 时启用。

推荐：

```math
\bar p_k
=
\frac1B\sum_i p_{i,k}
```

```math
L_{balance}
=
KL(U_K\,\|\,\bar p)
```

第一版默认：

```text
lambda_balance = 0.01
```

必须允许：

```text
lambda_balance = 0
```

做消融。

注意：

> balance 只约束 batch-level usage，不能强迫每个样本的 probability 接近 uniform。

---

# 12. Stage B 总 loss

第一版 trajectory-only specialization：

```math
L_B
=
\lambda_J L_{JEPA}^{WTA}
+
\lambda_T L_{traj}^{WTA}
+
\lambda_S L_{score}
+
\lambda_B L_{balance}
```

第一轮建议：

```yaml
lambda_jepa: 1.0
lambda_trajectory: 1.0
lambda_score: 0.1
lambda_balance: 0.01
```

不要同时引入 intention loss。

必须分别记录：

```text
loss_jepa_wta
loss_traj_wta
loss_score
loss_balance
loss_total
```

如果 score loss 梯度干扰 candidate generation，Stage C 将单独 refit scorer，因此 Stage B 中 `lambda_score` 保持小权重。

---

# 13. Stage B 参数更新范围

从 Stage A JEPA best checkpoint 初始化。

第一版建议训练：

```text
latent_predictor
mode_embeddings
trajectory_head
probability scorer
```

冻结：

```text
visual_online
motion encoder
ego_projection
fusion
visual_target
motion_target
```

这样第一版只回答：

> 在固定 JEPA context representation 下，多 latent hypotheses 是否能缓解 deterministic future regression。

如果验证有效，再单独做“解冻最后 fusion/predictor”的后续实验。

---

# 14. Stage C：冻结 hypotheses，单独 refit probability scorer

Stage B 完成后冻结：

```text
context encoder
latent predictor
mode embeddings
trajectory head
EMA target encoders
```

只训练：

```text
probability scorer
```

target 仍然是：

```math
k^*
=
\arg\min_k ADE(B^{(k)},B^{gt})
```

loss：

```math
L_C
=
-\log(p_{k^*}+\epsilon)
```

目的：

> 在 candidate set 固定后，让 scorer 学习稳定的 hypothesis ranking，而不是和不断变化的 trajectories 同时追逐。

checkpoint selection：

```text
minimum validation scorer CE
```

同时报告 validation Top1 ADE，但不使用 test 选 checkpoint。

---

# 15. Inference：正式仍然只输出 K=1

测试时：

```text
不使用 GT
不使用 Best-of-K oracle
不使用 minADE 选择
```

流程：

```text
Past
 ↓
C
 ↓
Z^(1)...Z^(4)
 ↓
B^(1)...B^(4)
 ↓
p_1...p_4
 ↓
argmax p_k
 ↓
ONE final trajectory
```

```math
k^{top1}
=
\arg\max_k p_k
```

```math
B^{final}
=
B^{(k^{top1})}
```

正式主指标：

```text
Top1 ADE
Top1 FDE
box_ADE
box_FDE
final_IoU
```

内部 diagnostics：

```text
minADE@4
minFDE@4
winner/oracle frequency
Top1↔Oracle agreement
probability entropy
```

不能把 `minADE@4` 当最终正式性能。

---

# 16. Entropy 只作为 uncertainty diagnostic

对每个测试样本：

```math
H_i
=
-\sum_k p_{i,k}\log(p_{i,k}+\epsilon)
```

至少分析：

```text
entropy vs Top1 ADE
entropy vs Top1 FDE
entropy vs oracle gap
entropy quartile trajectory error
```

希望验证：

```text
higher entropy
→ more ambiguous future
→ larger trajectory prediction error
```

如果没有相关性，不影响主方法是否成立，但不能宣称 entropy 是可靠 uncertainty estimate。

---

# 17. 配置建议

新增配置，默认关闭以保持旧 main 完全兼容：

```yaml
multi_hypothesis:
  enabled: false
  num_hypotheses: 4

  assignment:
    metric: center_ade

  wta:
    jepa: true
    trajectory: true

  scorer:
    enabled: true
    type: context_hypothesis_mlp
    hidden_dim: 256
    temperature: 1.0

  loss:
    jepa_weight: 1.0
    trajectory_weight: 1.0
    score_weight: 0.1
    balance_weight: 0.01

  scorer_refit:
    enabled: true
    epochs: 20
    learning_rate: 0.0003
```

旧配置：

```yaml
multi_hypothesis:
  enabled: false
```

必须保持 current deterministic behavior。

---

# 18. 代码修改范围

## 18.1 `models/layers.py`

修改：

```text
LatentPredictor
```

支持可选：

```text
K mode embeddings
[B,K,T,D] output
```

同时保留旧：

```text
[B,T,D]
```

path。

---

## 18.2 `models/model.py`

新增/修改：

```text
predict_future_latents_multi()
predict_trajectories_multi()
HypothesisScorer
predict_hypothesis_probabilities()
sample() support internal K but external top1
```

旧 `predict()` 与 K=1 path 必须向后兼容。

第一版 trajectory-only 实验不要删除 intention module，只是不把 intention loss 放入新 experiment。

---

## 18.3 `training.py`

新增：

```text
per_mode_trajectory_error()
select_wta_winner()
gather_winner_latent()
gather_winner_trajectory()
wta_trajectory_loss()
hypothesis_score_loss()
balance_loss()
```

所有 winner index 必须 detach / no-grad。

---

## 18.4 `train.py`

增加 multi-hypothesis trajectory-only training path：

```text
Stage B specialization
Stage C scorer refit
```

不要破坏 current deterministic train path。

---

## 18.5 `evaluate.py` / `metrics.py`

新增：

```text
Top1 selected by probability
minADE@K diagnostic
minFDE@K diagnostic
Top1↔Oracle agreement
winner frequency
entropy statistics
```

正式 JSON 明确区分：

```text
top1_ade
top1_fde
minade_k_diagnostic
minfde_k_diagnostic
```

---

# 19. 第一版必须完成的实验

## M0：当前 main deterministic baseline

完全不改。

报告：

```text
ADE
FDE
```

---

## M1：K=4 + WTA trajectory only

```text
K=4 latent hypotheses
WTA trajectory
no WTA JEPA
no scorer ranking
```

目的：

> 单纯多 latent trajectory specialization 是否形成有价值候选。

报告：

```text
minADE@4
candidate diversity
winner usage
```

---

## M2：K=4 + WTA JEPA + WTA trajectory

加入：

```math
L_{JEPA}^{WTA}
```

目的：

> Multiple-choice JEPA 本身是否比只在 trajectory head 做多候选更有价值。

---

## M3：M2 + probability scorer

完整第一版：

```text
WTA JEPA
+
WTA trajectory
+
winner CE scorer
+
light balance
+
Stage C scorer refit
```

正式比较：

```text
M0 deterministic K=1 ADE/FDE
vs
M3 probability-selected K=1 ADE/FDE
```

只有 M3 正式 K=1 优于 M0，才能支持：

> internal multimodal latent hypotheses improve deterministic trajectory prediction.

---

# 20. 成功标准

第一版方法有效需要至少满足：

- [ ] `minADE@4 < M0 ADE`，证明内部多个 hypotheses 确实覆盖更多 future；
- [ ] K=4 winner usage 不是长期单 mode 独占；
- [ ] WTA JEPA 不导致所有 latent collapse；
- [ ] scorer Top1 明显优于随机 mode 选择；
- [ ] **M3 Top1 ADE/FDE 至少一个主指标稳定优于 M0，另一个不明显恶化**；
- [ ] test selection 完全不使用 GT；
- [ ] entropy 可以正常计算且无 NaN；
- [ ] old deterministic config 仍可复现。

如果：

```text
minADE@4 明显改善
但 Top1 ADE 不改善
```

则结论只能是：

> latent multimodality 有潜力，但 probability ranking 未能转化为 K=1 收益。

不能把 minADE@4 作为最终方法提升。

---

# 21. 第一版不要做的内容

暂时不要加入：

```text
REINFORCE / PPO / actor-critic
conditional flow matching
diffusion
FutureModePosterior
Past Mode Prior
intention multi-task loss
Plan Transformer
cross-attention scorer
K=8 / K=16 搜索
full backbone unfreeze
复杂 contrastive diversity loss
```

先把最核心因果链跑干净：

```text
deterministic JEPA
→ multiple internal latent hypotheses
→ WTA JEPA specialization
→ probability ranking
→ K=1 trajectory improvement
```

---

# 22. 必须增加的测试

至少：

```text
test_multi_hypothesis_shapes
test_k1_backward_compatibility
test_wta_assignment_uses_per_sample_ade
test_wta_assignment_stop_gradient
test_wta_trajectory_only_updates_winner
test_wta_jepa_only_matches_winner_to_target
test_shared_trajectory_head_across_modes
test_probability_sum_to_one
test_scorer_shared_across_modes
test_balance_loss_batch_level
test_stage_c_only_scorer_trainable
test_top1_uses_probability_not_gt
test_minade_k_is_diagnostic_only
test_entropy_finite
```

并保留现有 tests 全部通过。

---

# 23. 输出目录建议

```text
outputs/mh_jepa_v1/
  M0_deterministic/
  M1_wta_traj/
  M2_wta_jepa_traj/
  M3_wta_jepa_traj_score/
  ablations/
  diagnostics/
  MH_JEPA_V1_REPORT.md
```

报告至少包含：

1. current main baseline；
2. K=4 latent architecture；
3. WTA assignment；
4. WTA JEPA；
5. WTA trajectory；
6. probability scorer；
7. entropy diagnostics；
8. M0/M1/M2/M3；
9. Top1 vs minADE@4 明确区分；
10. JAAD / PIE K=1 最终结果。

---

# 24. 给 Codex 的最终执行指令

> 基于 `HelloWorld416/EgoJEPA` 当前 `main` 分支做第一版 Multiple-Hypothesis JEPA 实验。不要从历史多模态分支拷贝复杂 Posterior/Prior 代码，也不要引入 RL。首先审计当前 `model.py`、`layers.py`、`pretrain.py`、`training.py`、`train.py`、`evaluate.py`、`metrics.py` 与 configs，确认 deterministic baseline 的 tensor shape、JEPA target、EMA、trajectory loss 与 checkpoint 规则。Stage A 保持当前 deterministic JEPA pretraining 完全不变。Stage B 从其 best checkpoint 初始化，把 `LatentPredictor` 扩展成共享 decoder + K=4 mode embeddings，输出 `[B,K,30,256]` future latent hypotheses，共享现有 trajectory head 解码 K 条轨迹。训练时使用 per-sample pixel center ADE 选 `k*=argmin ADE_k`，winner index stop-gradient；只对 winner 计算 encoded-box SmoothL1，并只让 winner latent 对当前 JEPA future target 计算 cosine JEPA loss，避免所有 K 个 latent 被同一 target 拉塌。新增共享 probability scorer `s_k=g(C,Z_k)`，用 winner CE `-log p_{k*}` 学习 hypothesis probability；不要把单个 probability 称为 entropy，entropy 只对完整 `p_1...p_K` 分布计算。加入轻量 batch-level balance regularizer 防止 dead modes，但必须可关闭。Stage C 固定 candidate generator，只 refit scorer。测试时严禁使用 GT/minADE 选轨迹，只允许 `argmax p_k` 输出一条 K=1 trajectory；`minADE@4` 只作为 diagnostic。完成 M0 deterministic、M1 WTA trajectory、M2 WTA JEPA+trajectory、M3 full scorer 四组实验，并以 M3 正式 K=1 ADE/FDE 是否优于 M0 作为方法是否值得继续的主要判断标准。所有新增行为必须由配置控制，旧 deterministic main config 必须完全可运行。
