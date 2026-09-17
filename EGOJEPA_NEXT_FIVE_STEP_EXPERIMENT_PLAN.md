# EgoJEPA-MP：下一阶段实验计划（Step 4B 后更新版）

> 本文面向 Codex 执行，基于 JAAD `K=4` 当前主线实验。Step 1 / 1B / 2A / 2B / 2C / 4A / 4B 已完成。
>
> **最新结论：不再继续堆叠 Prior 容量。** Step 4A 证明重新拟合原 state-token MLP prior 是必要的；Step 4B 的 mode-query cross-attention prior 在测试集上整体和 crossing 子集均退化，因此不替换 Step 4A。下一阶段的核心问题转为：**如何让 future mode 既具有 trajectory discrimination，又能从 past context 被可靠预测。**

---

# 0. 已完成实验与当前证据

## 0.1 Step 1B：Posterior trajectory grounding 有效

JAAD 测试集：

| 指标 | 原始 B0 | Step 1B R1 Hard oracle |
|---|---:|---:|
| Posterior ADE ↓ | 20.298 | 19.253 |
| G_mode ↓ | 3.496 | 2.451 |
| Posterior↔Oracle ↑ | 61.13% | 66.76% |
| mode 1/2 hard usage | 0% | 23.32% |

结论：当前 simple posterior 并非完全没有能力识别 trajectory-relevant mode，R1 hard-oracle alignment 是有效方向。

---

## 0.2 Step 2：Posterior alignment + sharpened routing 有效，但存在副作用

JAAD 测试集：

| 指标 | Step 1B R1 | Step 2 R1 |
|---|---:|---:|
| Posterior ADE ↓ | 19.253 | **18.517** |
| G_mode ↓ | 2.451 | **1.504** |
| Posterior↔Oracle ↑ | 66.76% | **77.75%** |
| mode 1 recall ↑ | 6.94% | **56.60%** |
| minADE@4 ↓ | **16.802** | 17.013 |
| Intention AP ↑ | **0.232** | 0.179 |

结论：

```text
posterior–trajectory semantics 明显改善；
但 candidate quality 有小幅回退；
意图 AP 出现负迁移；
旧 prior 因 posterior semantics 改变而严重失配。
```

因此 Step 2 后首先需要重新拟合 prior，而不是直接判断 deployment top-1。

---

## 0.3 Step 4A：重新拟合原 state-token MLP Prior 有效

只更新原 prior 的 `67,332` 个参数，其余权重全部冻结。

JAAD 测试集：

| 指标 | 拟合前 | Step 4A MLP 拟合后 |
|---|---:|---:|
| KL(q||π) ↓ | 2.7372 | **0.1334** |
| G_prior ↓ | 11.1958 | **3.3379** |
| Top1 ADE ↓ | 29.7125 | **21.8546** |
| Top1 FDE ↓ | 59.5575 | **46.2423** |
| Prior↔Posterior ↑ | 38.87% | **78.82%** |
| Intention F1 ↑ | 0.2440 | **0.2696** |
| Intention AP ↑ | 0.1789 | **0.2154** |

但：

```text
Crossing Top1 ADE: 29.586 → 29.766，无改善
总体 Top1 ADE 21.855 与原始 baseline 21.907 基本持平
```

说明：

> 旧 prior 的语义失配基本修复，但仍存在明显 residual selection gap，尤其 crossing 样本。

---

## 0.4 Step 4B：Cross-Attention Prior 不支持替换 Step 4A

新 prior 使用冻结的 mode embeddings，通过 cross-attention 读取完整 past context，其余模块全部冻结。

JAAD 测试集：

| 指标 | Step 4A MLP | Step 4B Cross-attention |
|---|---:|---:|
| G_prior ↓ | **3.3379** | 3.9377 |
| Top1 ADE ↓ | **21.8546** | 22.4543 |
| Top1 FDE ↓ | **46.2423** | 47.5202 |
| Prior↔Posterior ↑ | **78.82%** | 75.60% |
| Crossing G_prior ↓ | **8.7922** | 9.2235 |
| Crossing Top1 ADE ↓ | **29.7657** | 30.1970 |

paired video bootstrap 也支持 Step 4B 的退化不是纯随机波动。

### Step 4B 最终结论

```text
1. 不使用 Step 4B 替换 Step 4A。
2. 保留 state-token MLP Prior 作为当前 Prior baseline。
3. 当前问题不再优先解释为“Prior 没看到足够完整 context”。
4. 更可能的问题是：部分 posterior future modes 事后可区分，但仅凭 past observation 难以预测。
5. 下一阶段应改变 mode discovery 的目标，而不是继续增加 Prior 容量。
```

---

# 1. 下一阶段核心科学问题

当前 Posterior 学的是：

```math
q(M\mid Y_{future})
```

它越来越擅长回答：

> “看到真实未来以后，这条 future 最适合哪个 trajectory mode？”

而 Prior 需要回答：

```math
\pi(M\mid X_{past})
```

即：

> “只看过去，哪个 mode 最可能发生？”

这两件事不天然等价。

当前实验表明：

```text
trajectory-discriminative mode ≠ necessarily past-predictable mode
```

因此下一阶段要把 mode discovery 改成：

```text
Good future mode
= trajectory-discriminative
+ past-predictable
```

而不是继续单方面增强 Posterior 或 Prior。

---

# 2. 最新执行顺序

严格按以下顺序执行：

```text
Step 5-0  Prior 可预测性诊断
    ↓
Step 5A   Predictability-aware Mode Discovery（小权重双向一致性）
    ↓
Step 5A-Diagnostic  同时检查 G_mode / G_prior / candidate quality / intention
    ↓
Step 5B   若 5A 有效：Predictability-aware joint specialization
    ↓
Step 5C   冻结新 mode semantics，重新拟合 Step 4A MLP Prior
    ↓
Step 5D   最终再拆分 Pretraining：A1 mode discovery + A2 prior fitting
```

### 暂停项

```text
Step 3 Plan Transformer        = 暂停，降为可选结构消融
Step 4B Cross-attention Prior  = 负结果，不再作为主线
更大 Prior                    = 暂停
K=2/8 sensitivity             = 暂停
Diffusion / 新 backbone       = 不进入当前阶段
```

---

# 3. Step 5-0：先诊断哪些 future modes 从 Past 最难预测

## 3.1 目的

在改变训练目标前，先确认 `G_prior` 主要来自哪些 mode / crossing 子群。

使用当前：

```text
Posterior / decoder = Step 2 最优 checkpoint
Prior               = Step 4A state-token MLP best checkpoint
```

不训练模型，只跑 validation / test diagnostics。

---

## 3.2 必须按 mode 报告

```text
posterior mode frequency
prior hard usage
prior recall per posterior mode
prior precision per posterior mode
prior↔posterior confusion matrix
prior↔oracle confusion matrix
G_prior per posterior mode
ADE_prior per posterior mode
posterior entropy per mode
prior entropy per mode
```

并对：

```text
all
crossing=0
crossing=1
```

分别报告。

---

## 3.3 重点回答

1. crossing 样本的大 `G_prior` 是否集中在特定 mode？
2. 哪些 posterior modes 虽然 trajectory oracle 价值高，但 prior recall 极低？
3. 这些 mode 是否具有高 posterior confidence、低 prior confidence？
4. 是否存在 posterior 过于细分、但 past context 无法分辨的 mode pair？

输出：

```text
step5_prior_predictability_summary.json
step5_prior_predictability_samples.csv
step5_prior_predictability_report.md
```

---

# 4. Step 5A：Predictability-aware Mode Discovery

## 4.1 目的

让 posterior mode 不只满足 trajectory discrimination，还受到来自 past prior 的轻量 predictability regularization。

核心原则：

> 不允许 predictability loss 压过 trajectory semantics，否则 posterior 可能退化为“最容易从 past 预测”的粗粒度 mode。

因此从 Step 2 + Step 4A 的现有模型开始，只做小权重实验。

---

## 4.2 初始化

加载：

```text
Posterior / decoder / trajectory head = Step 2 validation best
Prior                                = Step 4A MLP validation best
context encoders                     = 对应 Step 2 frozen weights
```

第一阶段保持 trajectory candidate 本身固定，先只测试 mode semantics 能否在两种约束之间找到更好的平衡。

---

## 4.3 第一阶段冻结规则

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
EMA target encoders
```

训练：

```text
FutureModePosterior / prototypes
Step 4A state-token MLP Prior
```

不要使用 Step 4B cross-attention prior。

---

## 4.4 三个 loss

### A. Trajectory semantic alignment

继续使用已经验证有效的 R1 Hard oracle：

```math
k^{oracle}=\arg\min_k ADE(B^{(k)},B^{gt})
```

```math
L_{align}=-\log(q_{k^{oracle}}+\epsilon)
```

它保证 posterior 仍然与 trajectory modes 对齐。

---

### B. Prior fitting loss

只更新 Prior，使其拟合当前 posterior：

```math
L_{prior\_fit}
=
KL(stopgrad(q)\,\|\,\pi)
```

---

### C. Predictability regularizer

反向约束 posterior，使其不要定义完全无法从 past 推断的 mode：

```math
L_{predict}
=
KL(q\,\|\,stopgrad(\pi))
```

注意：

```text
L_prior_fit 只给 Prior 梯度
L_predict   只给 Posterior 梯度
```

这样避免一个 KL 同时让两边互相追逐造成实现含义不清。

---

## 4.5 总 loss

```math
L_{5A}
=
L_{align}
+
\lambda_{fit}L_{prior\_fit}
+
\lambda_{pred}L_{predict}
```

建议：

```text
lambda_fit  = 1.0
lambda_pred = 0.01 / 0.05 / 0.1
```

只用 validation 选择 `lambda_pred`。

第一轮不要超过 `0.1`，避免 posterior 被 prior 过度牵引。

---

## 4.6 关键防 collapse 监控

必须持续报告：

```text
posterior hard / soft usage per mode
posterior entropy
prior entropy
posterior↔oracle
prior↔posterior
mode-wise oracle frequency
mode-wise posterior recall
mode-wise prior recall
```

如果出现：

```text
G_prior 下降
但 G_mode 明显上升
```

说明 predictability weight 过大，不能视为成功。

---

# 5. Step 5A-Diagnostic：双 Gap 联合判定

Step 5A 完成后重新计算：

```math
G_{mode}=ADE_{posterior}-ADE_{oracle}
```

```math
G_{prior}=ADE_{prior}-ADE_{posterior}
```

以及：

```text
ADE_oracle
ADE_posterior
ADE_prior
minADE@4
top1 ADE
posterior↔oracle
prior↔posterior
prior↔oracle
mode-wise posterior recall
mode-wise prior recall
crossing G_mode
crossing G_prior
crossing top1 ADE
intention F1/AUC/AP
```

### Step 5A 成功标准

相对 Step 2 + Step 4A：

```text
1. G_prior 明显下降；
2. G_mode 不能明显恶化，最好同步下降；
3. posterior↔oracle 保持或提升；
4. prior↔posterior 提升；
5. minADE@4 不恶化；
6. crossing G_prior 有实质改善；
7. intention AP 不进一步明显下降。
```

核心判断不是单独最小化某一个 gap，而是改善：

```math
G_{total}=G_{mode}+G_{prior}=ADE_{prior}-ADE_{oracle}
```

同时保留 candidate quality。

---

# 6. Step 5B：Predictability-aware Joint Specialization

> 仅当 Step 5A 证明 predictability-aware mode discovery 有效时执行。

## 6.1 目的

在 Step 5A 已找到更可预测的 mode semantics 后，再允许 trajectory branch 一起适配，使新的 mode semantics 与 trajectory specialization 闭环。

---

## 6.2 解冻

训练：

```text
FutureModePosterior / prototypes
Step 4A MLP Prior
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
fusion
EMA target encoders
```

---

## 6.3 Loss

保留 Step 2 已验证的 sharpened routing：

```math
L_{traj}^{sharp}=\sum_k stopgrad(\tilde q_k)L_k
```

联合：

```math
L_{5B}
=
L_{traj}^{sharp}
+
\lambda_{align}L_{align}
+
\lambda_{fit}L_{prior\_fit}
+
\lambda_{pred}L_{predict}
+
0.5L_{intent}
```

`lambda_pred` 优先沿用 Step 5A validation 选出的值，不重新大范围搜索。

---

## 6.4 成功标准

必须同时看：

```text
G_mode
G_prior
G_total
minADE@4
top1 ADE
crossing top1 ADE
posterior/prior mode-wise recall
intention AP/F1
trajectory diversity
```

不能为了降低 `G_prior` 而牺牲大量 `minADE@4` 或 `G_mode`。

---

# 7. Step 5C：冻结新 Mode Semantics，重新做最终 MLP Prior Fitting

Step 5B 完成后：

冻结：

```text
Posterior
prototypes
trajectory decoder
mode embeddings
context encoders
任务头
```

只训练：

```text
Step 4A state-token MLP Prior
```

目标：

```math
L_{5C}=KL(stopgrad(q)\,\|\,\pi)
```

目的：把“mode discovery 本身的质量”和“最后 prior 是否充分拟合”重新分离。

输出最终：

```text
best_mode_semantics.pt
best_prior.pt
```

---

# 8. Step 5D：最后再拆分 Pretraining A1 / A2

只有 Step 5A–5C 证明 predictability-aware mode semantics 确实有价值后，才回到完整 JEPA pretraining 流程重新设计 checkpoint selection。

---

## 8.1 A1：Representation + Predictability-aware Mode Discovery

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

Past Prior 不作为 representation checkpoint 的主选模项。

A1 核心目标：

```text
JEPA future representation
+ trajectory-relevant mode discovery
+ weak past-predictability constraint
```

保存：

```text
best_repr.pt
```

### A1 选模禁止项

```text
禁止用 prior KL 主导 best_repr 评分
```

必须以 JEPA / representation quality 为主，同时设置 mode-collapse 与 predictability diagnostics 作为约束。

---

## 8.2 A2：Frozen Mode Semantics Prior Fitting

冻结：

```text
best_repr representation
FutureModePosterior
prototypes
mode semantics
```

只训练：

```text
state-token MLP Prior
```

保存：

```text
best_prior.pt
```

如果 A2 后仍有 residual `G_prior`，记录为 observation-to-future-mode 的不可约 ambiguity 候选，而不是继续无止境增加 Prior capacity。

---

# 9. Plan Transformer 的新定位

Plan Transformer 暂时**不作为下一步主线**。

只有在以下条件出现时重新启用：

```text
1. Step 5A 在多个 lambda_pred 下都无法兼顾 G_mode 和 G_prior；
2. 某些 mode 的 posterior recall 仍显著低；
3. simple pooled future representation 无法表达关键 temporal pattern；
4. 已排除 responsibility / predictability objective 的问题。
```

此时再比较：

```text
simple posterior
vs
Temporal Plan Encoder posterior
```

并保持完全相同的 alignment / predictability / routing 条件。

---

# 10. Step 4B Cross-Attention Prior 的定位

Step 4B 作为**负结果保留**：

```text
Step 4A MLP Prior          = 当前主 Prior
Step 4B Cross-attn Prior   = 不采用
```

不要删除相关代码和结果，保留用于实验记录和论文 ablation / negative finding。

---

# 11. 新的实验编号建议

| ID | Mode Discovery | Traj Routing | Prior | 目的 |
|---|---|---|---|---|
| E0 | Step 2 posterior | sharpened | old prior | posterior semantics baseline |
| E1 | Step 2 posterior | sharpened | Step 4A MLP refit | current deployment baseline |
| E2 | predictability-aware | frozen candidates | MLP joint fit | Step 5A |
| E3 | predictability-aware | sharpened joint | MLP joint fit | Step 5B |
| E4 | E3 semantics frozen | fixed | MLP refit | Step 5C |
| E5 | predictability-aware from pretrain | sharpened | A2 MLP | Step 5D |

Step 4B Cross-attention prior 单独作为负对照，不进入主编号序列。

---

# 12. 每一步统一输出

至少保存：

```text
config_resolved.yaml
trainable_parameters.txt
frozen_parameters.txt
train.log
val_metrics.json
test_metrics.json
mode_diagnostics_val.json
mode_diagnostics_test.json
report.md
```

从 Step 5 开始，报告必须同时包含：

```text
G_mode
G_prior
G_total = G_mode + G_prior
ADE_oracle
ADE_posterior
ADE_prior
minADE@4
top1 ADE
mode-wise posterior recall
mode-wise prior recall
crossing G_mode
crossing G_prior
crossing top1 ADE
intent F1/AUC/AP
```

---

# 13. 必须新增 / 保留的测试

Codex 必须保持已有 36+ 项相关测试继续通过，并新增：

```text
test_predictability_loss_stopgrad_prior
test_prior_fit_loss_stopgrad_posterior
test_step5a_only_posterior_and_prior_trainable
test_predictability_weight_zero_matches_previous_semantics
test_no_future_leakage_into_prior
test_g_total_identity
test_step5c_only_prior_trainable
test_cross_attention_prior_not_selected_as_default
```

Sanity：

```text
lambda_pred = 0 时，Step 5A 应退化为 posterior alignment + prior fitting，不得改变数学语义；
Past Prior inference 绝不能读取 future motion / future bbox / posterior target；
K=1 时 G_mode / G_prior routing 退化应保持一致性。
```

---

# 14. 给 Codex 的最新执行指令

> Step 1/1B/2/4A/4B 已完成。不要继续增强 Cross-Attention Prior，也不要直接进入 Plan Transformer。当前保留 Step 4A state-token MLP Prior 作为 Prior baseline，并把 Step 4B 记录为负结果。下一步首先执行 Step 5-0：对 Step 2 Posterior + Step 4A Prior 做 mode-wise / crossing-wise predictability diagnostics，定位 residual G_prior 来自哪些 future modes。然后执行 Step 5A：保持 trajectory candidates 和 context frozen，只训练 FutureModePosterior/prototypes 与 Step 4A MLP Prior，同时使用三项损失：R1 hard-oracle `L_align` 保证 trajectory semantics，`KL(stopgrad(q)||pi)` 只训练 Prior，`KL(q||stopgrad(pi))` 以小权重约束 Posterior 使 mode 更 past-predictable；`lambda_pred` 仅在 validation 搜索 `0.01/0.05/0.1`。Step 5A 后必须同时检查 G_mode、G_prior、G_total、minADE@4、crossing G_prior 和 intention AP，禁止仅以 prior↔posterior agreement 选模。只有 Step 5A 能在不明显破坏 G_mode/candidate quality 的情况下降低 G_prior，才进入 Step 5B：解冻 mode-conditioned decoder、mode embeddings、trajectory/intention heads，保留 Step 2 sharpened routing，做 predictability-aware joint specialization。Step 5B 后执行 Step 5C：冻结新的 mode semantics，仅重新拟合 state-token MLP Prior，得到最终 best_prior。最后才执行 Step 5D，把完整 JEPA pretraining 拆成 A1 representation + predictability-aware mode discovery 和 A2 frozen-mode prior fitting，并禁止 prior KL 再主导 representation checkpoint selection。所有超参数只使用 validation，test 只运行冻结后的配置。