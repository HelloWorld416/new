# EgoJEPA-MP：Hypothesis Plausibility Scoring 最终方法锁定与证实验收计划

> 本文基于现有 `FINAL_METHOD_LOCK_VALIDATION_PLAN.md` 进行方法重构，但只修改两个核心概念：
>
> 1. **Past Mode Prior 不再作为“mode classifier”**，而是降级/重定义为 **Hypothesis Plausibility Scorer (HPS)**：在 K 个已经生成的 future hypotheses 之间给出相对 plausibility 分数。
> 2. **FutureModePosterior 不再被解释为需要在测试时恢复的“真实 future mode teacher”**，而是重新定位为 **训练阶段的 matching / assignment mechanism**，仅用于把真实 future 与 K 个预测 hypotheses 对齐并形成 specialization。
>
> 本文目标不是继续增加复杂网络，而是把方法从“Past 先预测 mode，再选择 future”改成：
>
> ```text
> Past Context
>     ↓
> Generate K Future Latent Hypotheses
>     ↓
> Decode K Trajectories / K Intention Hypotheses
>     ↓
> Score each hypothesis by plausibility
>     ↓
> Keep the full K-set; use top-1 only when benchmark requires K=1
> ```
>
> 训练期真实 future 只负责 **matching / grounding**；测试期不需要 FutureModePosterior，也不要求模型从过去恢复一个“真实离散 mode”。

---

# 0. 为什么需要这次重定位

当前实验已经证明：

- K=4 的候选轨迹本身有价值，`minADE@4 < top1 ADE`；
- Step 2 的 posterior alignment + sharpened routing 能显著改善 trajectory mode specialization；
- Future-aware Posterior 可以更接近 oracle candidate；
- Step 4A 的 state-token MLP Prior 可以恢复一部分 top-1 selection；
- 但 Step 4B 更复杂的 Cross-Attention Prior 没有改善；
- predictability-aware regularization 也没有解决 residual `G_prior`。

因此当前证据更支持下面的解释：

> **future mode 不是 past observation 的确定标签。**

同一个 past context 可能对应多个合理 future。要求：

```math
X_{past} \rightarrow M_{true}
```

并不总是一个可识别的问题。

因此最终方法不再强调“预测真实 mode”，而强调：

> **从 past 生成多个 future hypotheses，并评估每个 hypothesis 在当前 context 下的相对 plausibility。**

---

# 1. 最终方法的角色重新定义

## 1.1 Multimodal Future Latents：真正承载 future modes

保留 K=4 mode-conditioned future latent：

```math
Z_k=P(C,e_k), \qquad k=1,\dots,K
```

其中：

- `C`：past visual + motion + ego context；
- `e_k`：第 k 个 hypothesis embedding；
- `Z_k`：第 k 个 future latent hypothesis。

轨迹：

```math
B_k=H_{traj}(Z_k)
```

意图：

```math
l_k=H_{intent}(Z_k)
```

因此最终定义：

> **future mode = one generated future latent / trajectory hypothesis。**

不再把 mode 解释成一个必须被 past 单独分类出来的隐藏标签。

---

## 1.2 FutureModePosterior：训练期 Matching / Assignment Mechanism

代码层面可以暂时保留 `FutureModePosterior` 类名以兼容 checkpoint，但论文和最终方法中统一解释为：

> **Future Assignment Posterior / training-time matcher**

它只在训练时使用真实 future：

```math
q_k=q(k\mid Y_{future})
```

主要职责：

1. 把真实 future 与 K 个 hypotheses 建立 assignment；
2. 形成 trajectory specialization；
3. 为 sharpened routing 提供训练期 routing weights；
4. 监测 assignment quality。

它**不是测试阶段的 mode predictor**，也不再作为“Past Scorer 必须蒸馏的真实 mode teacher”。

### Hard-oracle grounding

继续使用已经得到正证据的 R1：

```math
k^{oracle}=\arg\min_k ADE(B_k,B^{gt})
```

```math
L_{assign}=-\log(q_{k^{oracle}}+\epsilon)
```

这表示：

> FutureModePosterior 学的是“真实 future 应匹配到哪个当前 hypothesis”。

而不是学习一个独立于候选轨迹的行为类别。

---

## 1.3 Hypothesis Plausibility Scorer：替代 Past Mode Prior

原设计：

```text
past state token
→ MLP
→ K mode logits
→ “预测真实 future mode”
```

改为：

```text
past context + generated future hypothesis k
→ shared lightweight scorer
→ plausibility score s_k
```

定义：

```math
s_k=g(C,Z_k)
```

然后：

```math
\pi_k=\frac{\exp(s_k)}{\sum_j\exp(s_j)}
```

这里的 `pi_k` 解释为：

> **给定当前 past context，第 k 个已经生成的 future hypothesis 有多 plausible。**

而不是：

> 第 k 个 mode 是唯一真实 mode 的概率。

---

# 2. Hypothesis Plausibility Scorer 的最小实现

为了避免再次堆复杂结构，第一版使用轻量共享 scorer。

## 2.1 输入

Past summary：

```math
c=LN(C[:,0])
```

Future hypothesis summary：

```math
z_k=Pool(Z_k)
```

第一版优先使用 mean pooling 或已有 attention pooling，不新增 Transformer。

然后融合：

```math
h_k=[c, z_k, c\odot z_k, |c-z_k|]
```

shared scorer：

```math
s_k=MLP(h_k)
```

所有 K 个 hypotheses 使用同一个 MLP，避免每个 mode 单独参数化。

建议：

```text
LayerNorm
→ Linear
→ GELU
→ Dropout
→ Linear(1)
```

如果现有代码改动成本较大，可先使用简化版：

```math
h_k=LN(W_c c + W_z z_k)
```

```math
s_k=w^T h_k
```

但必须保证 scorer 显式看到 candidate-specific `Z_k`，不能只看 state token 输出 K 类 logits。

---

# 3. Scorer 的训练目标

## 3.1 默认 target：trajectory-grounded assignment

继续使用已经验证过的 R1 hard oracle：

```math
k^{oracle}=\arg\min_k ADE(B_k,B^{gt})
```

Scorer loss：

```math
L_{score}
=
-\log(\pi_{k^{oracle}}+\epsilon)
```

注意：

- `k^{oracle}` 由 GT future 计算，只用于训练；
- oracle selection 必须 stop-gradient；
- `L_score` 不反向修改 GT matching；
- 测试时完全不计算 oracle。

这与旧 prior KL 的区别是：

> scorer 不是从 past 单独恢复一个抽象 mode ID，而是在**已经生成的 K 个 future hypotheses 上做相对排序**。

---

## 3.2 FutureModePosterior 与 Scorer 不做双向 KL

最终主模型不使用：

```math
KL(q\|\pi)
```

作为核心方法。

也不使用：

```math
KL(\pi\|q)
```

原因：

- Posterior 的职责是 training assignment；
- Scorer 的职责是 inference-time hypothesis ranking；
- 两者可以共享同一个 trajectory-grounded assignment target，但不需要互相定义语义。

可选补充消融可以测试：

```text
hard-oracle scorer CE
vs
stopgrad(q) soft distillation
```

但主方法默认使用 hard-oracle scorer CE，减少 moving-target 依赖。

---

# 4. Trajectory Specialization 保持 Step 2 已验证方案

FutureModePosterior 继续提供训练期 q：

```math
q_k=q(k\mid Y_{future})
```

使用 sharpened routing：

```math
\tilde q_k
=
\frac{q_k^{1/\tau}}
{\sum_j q_j^{1/\tau}}
```

trajectory loss：

```math
L_{traj}
=
\sum_k stopgrad(\tilde q_k)L_k
```

其中：

```math
L_k=SmoothL1(B_k,B^{gt})
```

最终 specialization loss：

```math
L_{spec}
=
L_{traj}
+
\lambda_{assign}L_{assign}
+
0.5L_{intent}
```

继续使用已经验证的：

```yaml
num_modes: 4
responsibility: hard_oracle
lambda_assign: 0.1
trajectory_routing:
  type: sharpened
  tau_start: 1.0
  tau_mid: 0.5
  tau_end: 0.25
```

不要加入 posterior-guided hard routing。

---

# 5. Scorer 训练阶段

trajectory hypotheses 与 assignment semantics 稳定后，冻结：

```text
context encoders
FutureModePosterior / assignment prototypes
mode embeddings
future predictor / decoder
trajectory head
intention head
```

只训练：

```text
Hypothesis Plausibility Scorer
```

训练输入仍然只使用：

```text
past context
+
由 past context 生成的 K 个 future latent hypotheses
```

GT future 只用于构造 training target `k_oracle`。

建议：

```text
optimizer: AdamW
lr: 3e-4 作为 JAAD 起点
max_epochs: 20
checkpoint selection: minimum validation scorer CE / NLL
```

同时报告 validation top-1 ADE，但不使用 test 选 scorer checkpoint。

---

# 6. Intention Prediction 改为 hypothesis-set aggregation

测试时不要先强制选一个 hypothesis 再做 intention。

每个 future latent 输出：

```math
p_k^{cross}=\sigma(l_k)
```

Scorer 给出 plausibility：

```math
\pi_k
```

最终 crossing probability：

```math
p^{cross}
=
\sum_k \pi_k p_k^{cross}
```

这样 intention 利用完整 multimodal future set，而不是依赖单一 top-1 hypothesis。

必须保留对照：

```text
top1-mode intention
vs
plausibility-weighted mixture intention
```

最终主方法优先使用 mixture，前提是 validation F1/AP 不劣于 top1。

---

# 7. 最终 inference 定义

测试时完全不使用：

```text
FutureModePosterior
future GT
oracle mode
assignment target
```

推理流程：

```text
Past observations
    ↓
Context C
    ↓
Generate K future latent hypotheses Z_1...Z_K
    ↓
Decode K trajectories B_1...B_K
    ↓
Hypothesis Plausibility Scorer g(C,Z_k)
    ↓
pi_1...pi_K
```

正式 multimodal 输出：

```text
{B_k, pi_k}_{k=1..K}
```

如果 benchmark 要求 K=1：

```math
k^{top1}=\arg\max_k \pi_k
```

```math
B^{top1}=B_{k^{top1}}
```

这只是：

> MAP / highest-plausibility hypothesis

不能表述为：

> 模型预测出了真实 future mode。

---

# 8. 正式评价指标重新分层

## 8.1 Multimodal capability

主报告：

```text
minADE@4
minFDE@4
trajectory diversity
mode-wise oracle frequency
```

表示 K 个候选集合是否覆盖真实未来。

---

## 8.2 Hypothesis ranking

报告：

```text
Top1 ADE
Top1 FDE
Scorer NLL / CE
Top1↔Oracle agreement
Oracle rank under scorer
pi_oracle
```

这里不再使用术语：

```text
G_prior
Prior↔Posterior
Past Mode Prior
```

建议改名：

```text
G_score = Top1 ADE - minADE@4
Scorer↔Oracle agreement
Hypothesis ranking gap
```

---

## 8.3 Training assignment diagnostics

FutureModePosterior 只报告：

```text
Assignment ADE
Assignment↔Oracle agreement
G_assign = Assignment ADE - minADE@4
mode-wise assignment recall / precision
```

这些只属于训练机制分析，不属于 inference 指标。

---

# 9. 新的三层 ADE 分解

最终方法建议改成：

```math
ADE_{oracle}=minADE@4
```

```math
ADE_{assign}
```

```math
ADE_{score}=Top1\ ADE
```

定义：

```math
G_{assign}=ADE_{assign}-ADE_{oracle}
```

```math
G_{score}=ADE_{score}-ADE_{oracle}
```

其中：

```text
Oracle/minADE@4 = candidate set coverage
Assignment ADE  = training-time matcher quality
Scorer Top1 ADE = inference-time ranking quality
```

不再把 `Assignment ADE → Scorer ADE` 解释成 teacher-student mode classification gap。

---

# 10. 最终候选方法

最终候选固定为：

```text
Existing JEPA future representation
+
K=4 multimodal future latent hypotheses
+
R1 Hard-Oracle Training Assignment
+
Simple FutureModePosterior used only as matcher/router
+
Sharpened Soft Trajectory Routing
+
Lightweight Hypothesis Plausibility Scorer g(C, Z_k)
+
Plausibility-weighted intention aggregation
```

---

# 11. 明确排除的旧解释 / 模块

最终主方法不再使用以下表述或设计：

```text
“Past Prior predicts the true future mode”
“Posterior is a mode teacher that must be distilled into Prior”
KL(q||pi) as the core inference bridge
Mode-query Cross-Attention Prior
predictability-aware lambda_pred
posterior-guided hard trajectory assignment
Plan Transformer Posterior（除非补充消融）
Step 5D downstream 后二次 JEPA re-alignment
Diffusion
VQ / Gumbel / Sinkhorn
```

---

# 12. Codex 必须完成的实现审计

## F0：No-future-leakage audit

确认 scorer 输入只来自：

```text
C = past context
Z_k = model-generated future hypothesis from past
```

不得输入：

```text
future image
future GT box
FutureModePosterior output q
oracle mode
```

测试时 FutureModePosterior 必须完全不执行。

---

## F1：Hypothesis index consistency

必须确认：

```text
mode embedding k
future latent Z_k
trajectory B_k
intention logit l_k
scorer score s_k
```

全部保持同一个 k。

FutureModePosterior assignment 也必须对应同一 index，仅在训练期使用。

---

# 13. 最小核心实验

## H0：旧 classifier Prior baseline

保留历史 Step 4A：

```text
state-token MLP
→ K logits
→ top1 trajectory
```

只作为 baseline，不再作为最终方法。

---

## H1：Hypothesis Plausibility Scorer

固定 Step 2 candidate generator，训练：

```text
g(C,Z_k)
→ K plausibility scores
```

用 hard-oracle target。

比较：

```text
Top1 ADE/FDE
Scorer↔Oracle
G_score
Intention F1/AP
```

H1 必须直接与 Step 4A H0 比较。

---

## H2：Scorer input ablation

至少比较：

```text
context-only MLP      = 旧 Prior
hypothesis-only       = g(Z_k)
context+hypothesis    = g(C,Z_k)
```

目标是证明：

> candidate-aware scoring 比单纯 past mode classification 更合理/有效。

若 `context+hypothesis` 没有优于 context-only，则不能宣称 scorer formulation 得到实证支持。

---

## H3：Intention aggregation

比较：

```text
top1 hypothesis intention
vs
sum_k pi_k * p_k(cross)
```

用 validation 决定最终主结果采用哪种。

---

# 14. Clean reproduction

新的最终候选必须在全新输出目录从固定初始化完整跑：

```text
JEPA initialization
→ K=4 specialization
→ training-time assignment alignment
→ sharpened routing
→ freeze generator
→ train plausibility scorer
→ validation
→ test
```

不得直接把历史 scorer / prior 权重拼进最终结果。

---

# 15. JAAD 多随机种子与 PIE

完成单种子验证后：

```text
JAAD seeds = 42 / 43 / 44
```

至少报告：

```text
Top1 ADE/FDE
minADE@4/minFDE@4
Assignment ADE
G_assign
G_score
Scorer↔Oracle
Intent F1/AP
```

随后以相同方法迁移 PIE。

允许沿用 dataset-specific optimizer / batch / LR，但不允许根据 PIE test 修改：

```text
K
assignment type
lambda_assign
routing schedule
scorer architecture
```

---

# 16. 方法锁定标准

只有满足以下条件才标记新版方法为 `LOCKED`：

- [ ] K=4 candidate set 能稳定复现 multimodal coverage；
- [ ] R1 assignment + sharpened routing 的正收益可复现；
- [ ] FutureModePosterior 完全从 inference path 移除；
- [ ] scorer 无 future leakage；
- [ ] `g(C,Z_k)` 至少不劣于旧 context-only Prior，并最好显著改善 top1 ranking；
- [ ] mixture intention 不造成明显负迁移；
- [ ] JAAD 多种子趋势一致；
- [ ] PIE 以相同设计完成一次验证；
- [ ] 所有正式 inference 指标只依赖 past 与生成 hypotheses。

如果 scorer 没有优于旧 Prior：

> 保留“set-valued multimodal future representation + training matcher”作为主要创新，但不要宣称 hypothesis-aware scoring 有额外收益。

---

# 17. 最终输出文件

Codex 创建：

```text
outputs/final_hypothesis_scoring_lock/
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
  FINAL_HYPOTHESIS_SCORING_REPORT.md
```

报告至少包含：

1. 新的角色定义；
2. FutureModePosterior 作为 training matcher 的实现；
3. Hypothesis Plausibility Scorer；
4. H0/H1/H2/H3 消融；
5. minADE@4 / Assignment ADE / Scorer Top1 ADE 三层分解；
6. intention mixture；
7. JAAD multi-seed；
8. PIE transfer；
9. 无 future leakage 审计；
10. `LOCKED / NOT LOCKED`。

---

# 18. 必须新增 / 更新的测试

至少：

```text
test_assignment_posterior_training_only
test_assignment_not_used_in_inference
test_scorer_has_no_future_input
test_scorer_receives_candidate_specific_latent
test_scorer_shared_across_hypotheses
test_hypothesis_index_alignment
test_top1_uses_scorer_only
test_minade_is_per_sample_oracle
test_assignment_oracle_only_used_for_training
test_routing_weights_stop_gradient
test_intention_mixture_weights_sum_to_one
test_locked_config_reproducible
```

---

# 19. 给 Codex 的最终执行指令

> 基于当前 `FINAL_METHOD_LOCK_VALIDATION_PLAN.md` 的已验证组件，把方法重构为“Generate K future hypotheses, then score them”，而不是“Predict a future mode, then generate/select it”。保留 K=4、R1 hard-oracle training assignment、simple FutureModePosterior/prototypes 和 sharpened routing，但把 FutureModePosterior 重新定位为仅训练阶段使用的 matching / assignment mechanism，测试时不得执行。删除“Past Mode Prior predicts the true mode”的方法解释，不再以 `KL(q||pi)` 作为核心 inference bridge。新增轻量 Hypothesis Plausibility Scorer `s_k=g(C,Z_k)`，其中 scorer 必须同时看到 past context 和 candidate-specific future latent，所有 hypotheses 共享 scorer 参数。默认使用 hard-oracle candidate assignment 监督 scorer 的 relative plausibility；训练期 GT 只用于生成 target，测试期 scorer 输入只能来自 past 与由 past 生成的 hypotheses。正式 multimodal 输出为 `{B_k, pi_k}`，benchmark 要求 K=1 时才用 `argmax pi_k` 输出 top1。意图默认尝试使用 `sum_k pi_k * p_k(cross)` 做 hypothesis-set aggregation，并与 top1 intention 做 validation 对照。首先完成 no-future-leakage 与 index alignment 审计，然后做 H0 旧 context-only Prior、H1 context+hypothesis scorer、H2 scorer input ablation、H3 intention aggregation；若 H1/H2 不能证明 candidate-aware scorer 至少不劣于旧 Prior，则不要强行把 scorer 作为最终贡献。最后完成 clean reproduction、JAAD 3 seeds、PIE transfer，并只在所有正式 inference 指标完全 past-only 时标记方法为 LOCKED。
