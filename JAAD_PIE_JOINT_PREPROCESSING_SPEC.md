# JAAD / PIE 联合轨迹预测与意图预测数据预处理规范

## 0. 文档目的

本文档用于指导后续 Codex **重新划分并预处理 JAAD 与 PIE 数据集**，为当前基于 JEPA 的统一轨迹预测（trajectory prediction）与意图预测（intention prediction）方法生成一套稳定、可复现、无数据泄漏的训练 / 验证 / 测试数据。

本轮数据重构的核心原则是：

> **不要再分别构建“轨迹预测数据集”和“意图预测数据集”。**
>
> 对每个 pedestrian track 先按照数据集官方 split 划分，再按照轨迹预测任务的标准时间窗口生成统一样本。每个样本都包含相同的过去 15 帧与未来 45 帧；轨迹预测和 JEPA 使用所有有效样本，意图预测只在该样本具有合法意图监督时通过 `intent_mask=1` 参与 loss。

统一样本形式：

```text
past 15 frames (0.5 s)  ->  future 45 frames (1.5 s)
          |                         |
          |                         +-- future trajectory target
          |
          +-- context for JEPA predictor
                                      |
                                      +-- intention target (masked when unavailable)
```

模型层面的目标是：

```math
X_{1:15}
\rightarrow
\hat Z_{future}
\rightarrow
\left(
\hat Y_{16:60}^{trajectory},
\hat I^{intent}
\right)
```

数据层面必须保证 **trajectory target 与 intention target 来自同一个 sample / 同一个 pedestrian / 同一个 observation end time**。

---

# 1. 全局强制约束

以下规则对 PIE 与 JAAD 都适用。

## 1.1 先划分，再滑窗

必须按照 dataset 官方 video / set split 先划分 train / val / test，然后在各自 split 内生成 pedestrian tracks 和 sliding windows。

禁止：

```text
all tracks -> sliding windows -> random train/val/test
```

必须：

```text
official video/set split
        -> pedestrian tracks
        -> filter
        -> sliding windows
        -> manifest
```

这样可以保证：

- 同一个 video 不会同时出现在 train 与 test；
- 同一个 pedestrian 的高度重叠窗口不会跨 split；
- JEPA future target 不会产生隐式数据泄漏。

---

## 1.2 统一时间设置

两个数据集均为 30 FPS。

统一使用：

```yaml
fps: 30
observe_length: 15
predict_length: 45
window_length: 60
observe_seconds: 0.5
predict_seconds: 1.5
```

每个 canonical sample 的 raw frame 数必须严格为：

```text
15 observed + 45 future = 60 frames
```

注意：

- 预处理阶段保存完整的 45 帧 future；
- 如果当前 EgoJEPA 模型内部使用 30 个 future query 或其它时间下采样，必须在 **dataset/model adapter** 中显式完成；
- 不允许在预处理时把 45 帧偷偷压缩成 30 帧，否则后续 trajectory baseline 无法公平复用同一份数据。

---

## 1.3 最小轨迹长度

统一采用：

```yaml
min_track_size: 61
```

原因：

- 与 PIEPredict / 常用 JAAD-PIE trajectory preprocessing 保持一致；
- canonical window 本身需要 60 帧；
- 保留 61 帧门槛也兼容后续使用 bbox displacement / velocity 时删除第一帧的实现。

只有满足：

```math
L_{track} \ge 61
```

的 pedestrian track 才进入 window generation。

---

## 1.4 不做 preprocessing-level class balancing

预处理阶段：

- 不复制 positive sample；
- 不删除 negative sample；
- 不对 train / val / test 做 intention class balancing；
- 不改变 trajectory 原始数据分布。

若训练时存在类别不平衡，使用：

- weighted BCE；
- focal loss；
- sampler；
- class weight；

在训练代码中处理。

**validation 与 test 永远保持原始分布。**

---

## 1.5 ground-truth intention 不能作为 trajectory 输入

本项目的目标是从共同 future latent 中联合预测 trajectory 与 intention。

因此：

```text
GT intention -> trajectory decoder
```

是禁止的。

例如 PIEPredict 经典 baseline 会将 `intention_prob` 作为 trajectory decoder input，这种设置不能用于我们的主方法，因为推理阶段不存在 ground-truth intention，而且会破坏“统一 latent future prediction”的方法定义。

允许的输入：

- observed RGB / pedestrian context；
- observed bbox / center / motion；
- ego-motion；
- scene context；
- observed behavior cues（如果实验设计明确允许）。

禁止的输入：

- future trajectory；
- future RGB；
- GT `intention_prob`；
- GT `crossing`；
- `critical_point` / `decision_point` 本身作为模型 feature。

这些 future / intent 字段只用于 target 构建、mask 与 evaluation。

---

# 2. 统一输出格式

建议 Codex 生成两个独立 preprocessing 脚本：

```text
scripts/preprocess_pie_joint.py
scripts/preprocess_jaad_joint.py
```

以及一个公共验证脚本：

```text
scripts/validate_joint_manifests.py
```

建议最终目录：

```text
processed_data/
├── pie/
│   ├── manifests/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── split_definition.json
│   ├── statistics.json
│   └── statistics.md
└── jaad/
    ├── manifests/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    ├── split_definition.json
    ├── statistics.json
    └── statistics.md
```

原则上 **不要复制原始图片**。

manifest 中保存原始 image path，dataset loader 在训练时读取图片。若后续已有 feature cache，可额外生成 cache，但 canonical manifest 必须与 feature cache 解耦。

---

# 3. Canonical sample schema

每一行 JSONL 表示一个 60-frame joint sample。

建议至少包含：

```json
{
  "sample_id": "pie_set01_video_0001_1_1_12_f000123",
  "dataset": "pie",
  "split": "train",

  "set_id": "set01",
  "video_id": "video_0001",
  "ped_id": "1_1_12",

  "start_frame": 123,
  "obs_end_frame": 137,
  "future_start_frame": 138,
  "future_end_frame": 182,

  "obs_frame_ids": [],
  "future_frame_ids": [],

  "obs_image_paths": [],
  "future_image_paths": [],

  "obs_bbox_abs": [],
  "future_bbox_abs": [],
  "obs_bbox_norm": [],
  "future_bbox_norm": [],

  "obs_center_abs": [],
  "future_center_abs": [],
  "obs_center_norm": [],
  "future_center_norm": [],

  "ego_motion": {},

  "intent_soft": null,
  "intent_binary": null,
  "intent_mask": 0,

  "crossing_attr": null,
  "crossing_point": null,
  "critical_point": null,
  "decision_point": null
}
```

要求：

- `obs_frame_ids` 长度 = 15；
- `future_frame_ids` 长度 = 45；
- `obs_bbox_abs` shape = `[15, 4]`；
- `future_bbox_abs` shape = `[45, 4]`；
- bbox 格式统一为：
  ```text
  [x1, y1, x2, y2]
  ```
- center 统一为：
  ```text
  [(x1+x2)/2, (y1+y2)/2]
  ```

如果某个 dataset 不存在某字段，使用 `null`，不要伪造。

---

# 4. 坐标处理

canonical manifest 同时保留 **绝对坐标** 和 **图像尺寸归一化坐标**。

绝对坐标：

```math
B_t^{abs}=[x_1,y_1,x_2,y_2]
```

归一化坐标：

```math
B_t^{norm}
=
[x_1/W, y_1/H, x_2/W, y_2/H]
```

center 同理：

```math
C_t^{norm}=[c_x/W, c_y/H]
```

不要在 preprocessing 阶段只保留 displacement。

如训练需要 velocity / displacement：

```math
\Delta B_t=B_t-B_{t-1}
```

在 dataset loader 中动态构造，保留 canonical absolute coordinates 作为唯一 ground truth。

---

# 5. PIE 单独预处理规范

# 5.1 官方 split

使用当前 PIE data interface 可复现的 fixed default split：

```yaml
train:
  - set01
  - set02
  - set04

val:
  - set05
  - set06

test:
  - set03
```

对应 video 数参考：

| Split | Sets | Videos |
|---|---|---:|
| Train | set01 + set02 + set04 | 23 |
| Val | set05 + set06 | 11 |
| Test | set03 | 19 |
| Total | set01–set06 | 53 |

**禁止重新 random split pedestrian IDs。**

必须将实际使用的 set 列表写入：

```text
processed_data/pie/split_definition.json
```

---

# 5.2 PIE trajectory track 获取

使用 PIE 官方 annotation / `pie_data.py` 的 trajectory 语义。

建议参数等价于：

```python
data_opts = {
    "fstride": 1,
    "sample_type": "all",
    "height_rng": [0, float("inf")],
    "squarify_ratio": 0,
    "data_split_type": "default",
    "seq_type": "trajectory",
    "min_track_size": 61,
}
```

trajectory 数据至少读取：

- `image`
- `pid`
- `bbox`
- `center`
- `occlusion`
- `obd_speed`
- `gps_speed`
- `heading_angle`
- `intention_prob`
- pedestrian attributes 中的：
  - `exp_start_point`
  - `critical_point`
  - `crossing_point`
  - `crossing`

如项目后续需要，可额外保存：

- yaw / roll / pitch；
- acceleration；
- gyroscope；
- traffic annotations。

但不要因为额外字段缺失而丢掉合法 trajectory sample。

---

# 5.3 PIE trajectory sliding window

PIE 轨迹任务沿用：

```yaml
track_overlap: 0.5
observe_length: 15
predict_length: 45
stride: 7
```

注意不要重新通过浮点公式推导 stride，直接固定：

```python
PIE_STRIDE = 7
```

对于长度为 `L` 的有效 track：

```python
for start in range(0, L - 60 + 1, 7):
    window = track[start:start + 60]
```

每个 window：

```python
obs = window[:15]
future = window[15:60]
```

trajectory supervision：

```text
traj_mask = 1
```

对所有生成出的 PIE window 都成立。

---

# 5.4 PIE intention 定义

PIE 的 intention **不是 future crossing action 的简单同义词**。

PIE 官方每个 pedestrian track 提供一个：

```text
intention_prob in [0, 1]
```

它表示 human observers 对该 pedestrian 在 action 发生前是否“想要过街”的平均判断。

因此保留两种 target：

```python
intent_soft = intention_prob
intent_binary = int(intention_prob > 0.5)
```

不要只保存 binary label。

---

# 5.5 PIE intention 时间有效区间

`intention_prob` 只在官方 human-intention experiment 对应的 pre-action 区间具有严格语义。

使用：

- `exp_start_point`
- `critical_point`

定义 intention supervision mask。

设 sample 的 observation：

```text
obs_start_frame
obs_end_frame
```

则：

```python
intent_mask = int(
    obs_start_frame >= exp_start_point
    and obs_end_frame <= critical_point
)
```

如果：

```text
intent_mask == 1
```

保存：

```python
intent_soft = intention_prob
intent_binary = int(intention_prob > 0.5)
```

如果：

```text
intent_mask == 0
```

则建议：

```json
{
  "intent_soft": null,
  "intent_binary": null,
  "intent_mask": 0
}
```

注意：

- trajectory / JEPA loss 仍使用该 sample；
- 只是 intention loss 不参与；
- 不要因为 intention 无效而删除 trajectory sample。

---

# 5.6 PIE joint sample 逻辑

最终 PIE 一个 track 可以同时产生两类 sample：

### 类型 A：有意图监督

```text
JEPA       ✓
Trajectory ✓
Intention  ✓
```

### 类型 B：无意图监督

```text
JEPA       ✓
Trajectory ✓
Intention  ✗
```

训练 loss：

```math
\mathcal L
=
\lambda_J \mathcal L_{JEPA}
+
\lambda_T \mathcal L_{traj}
+
\lambda_I M_I \mathcal L_{intent}
```

其中：

```math
M_I \in \{0,1\}
```

---

# 5.7 PIE 参考统计与实际统计要求

文献 / 当前 fixed split 中常见的 track-level 参考值：

| Split | Reference tracks |
|---|---:|
| Train | 880 |
| Val | 243 |
| Test | 719 |
| Total | 1842 |

**这些数字只能作为 sanity-check，不允许硬编码。**

Codex 必须从当前本地 annotation 版本重新统计：

1. videos；
2. raw pedestrian tracks；
3. tracks with `L >= 61`；
4. total joint windows；
5. intent-valid windows；
6. intent-positive windows；
7. intent-negative windows；
8. unique intent-labeled pedestrians；
9. soft intention probability 分布：
   - mean；
   - std；
   - min；
   - max；
   - 0–0.25；
   - 0.25–0.5；
   - 0.5–0.75；
   - 0.75–1.0。

若实际 track 数与参考值不同：

- 不要强制删数据来匹配；
- 在 `statistics.md` 中记录差异；
- 输出当前 annotation / interface 的版本信息。

---

# 6. JAAD 单独预处理规范

# 6.1 官方 default split

JAAD 必须使用官方：

```text
split_ids/default/
```

中的：

```text
train.txt
val.txt
test.txt
```

禁止使用：

- `all_videos` split；
- `high_visibility` split；
- random split。

当前 default split 的 video 数参考：

| Split | Videos |
|---|---:|
| Train | 177 |
| Val | 29 |
| Test | 117 |
| Total | 323 |

Codex 应直接读取官方 txt 文件，而不是在代码中手抄 video ID。

将最终 video IDs 写入：

```text
processed_data/jaad/split_definition.json
```

---

# 6.2 JAAD trajectory track 获取

trajectory 主数据使用：

```yaml
subset: default
sample_type: all
seq_type: trajectory
data_split_type: default
fstride: 1
min_track_size: 61
```

等价目标：

```python
data_opts = {
    "fstride": 1,
    "sample_type": "all",
    "subset": "default",
    "height_rng": [0, float("inf")],
    "squarify_ratio": 0,
    "data_split_type": "default",
    "seq_type": "trajectory",
    "min_track_size": 61,
}
```

JAAD 官方接口中：

- 排除 `people` / 非 pedestrian-of-interest 类 track；
- `sample_type="all"` 保留有 behavior annotation 和无 behavior annotation 的合法 pedestrian；
- trajectory task 不应只限制为 behavior subset。

读取至少：

- `image`
- `pid`
- `bbox`
- `center`
- `occlusion`
- ego vehicle action / motion；
- pedestrian attributes（若存在）：
  - `crossing`
  - `crossing_point`
  - `decision_point`

---

# 6.3 JAAD trajectory sliding window

JAAD 轨迹任务采用：

```yaml
track_overlap: 0.8
observe_length: 15
predict_length: 45
stride: 3
```

**必须显式固定：**

```python
JAAD_STRIDE = 3
```

不要写：

```python
int((1 - 0.8) * 15)
```

作为最终唯一实现，因为浮点表示可能在某些环境中产生 `2.999...` 后被 `int()` 截断为 2。

对于长度为 `L` 的有效 track：

```python
for start in range(0, L - 60 + 1, 3):
    window = track[start:start + 60]
```

然后：

```python
obs = window[:15]
future = window[15:60]
```

所有 JAAD joint windows：

```text
traj_mask = 1
```

---

# 6.4 JAAD 的 behavior annotation 与 trajectory annotation 必须分开理解

JAAD 中不是所有 pedestrian 都有完整 behavior / intention annotation。

因此：

- trajectory / JEPA：使用所有满足条件的 `JAAD_all` tracks；
- intention：只对具有合法 behavior attributes 的 pedestrian 启用。

最重要的规则：

> **没有 behavior annotation != negative intention。**

禁止：

```python
if no_behavior_annotation:
    intent_binary = 0
```

必须：

```python
if no_behavior_annotation:
    intent_binary = None
    intent_mask = 0
```

这些 sample 仍用于：

```text
JEPA       ✓
Trajectory ✓
Intention  ✗
```

---

# 6.5 JAAD intention label 定义

对具有 behavior attributes 的 pedestrian，使用官方 track-level `crossing` attribute：

```text
crossing =  1 : observed crossing
crossing =  0 : has crossing intention / relevant pedestrian but did not cross
crossing = -1 : irrelevant / no crossing intention
```

为了与 PIE 的“crossing intention”语义对齐，本项目统一将 JAAD intention 定义为：

```python
if crossing == -1:
    intent_binary = 0
elif crossing in {0, 1}:
    intent_binary = 1
else:
    intent_binary = None
```

即：

```math
I^{JAAD}
=
\mathbf 1[crossing \neq -1]
```

注意：

- 这不是 `will_cross = 1[crossing == 1]`；
- 本项目预测的是 crossing intention，而不是 future crossing action label；
- 后续如果需要额外评估 “will cross / not cross”，应新增独立 `crossing_action_binary` 字段，不得覆盖 `intent_binary`。

建议 manifest 同时保存：

```json
{
  "crossing_attr": -1,
  "intent_binary": 0
}
```

方便后续分析。

---

# 6.6 JAAD intention 时间有效区间

JAAD 官方 intention sequence 使用 `decision_point` 截止。

因此对具有 behavior annotation 的 pedestrian，设：

```python
intent_mask = int(
    decision_point is not None
    and decision_point >= 0
    and obs_end_frame <= decision_point
)
```

如果 observation 已经晚于 `decision_point`：

```text
intent_mask = 0
```

但该 sample 仍然用于 trajectory / JEPA。

如果 behavior annotation 缺失：

```text
intent_mask = 0
```

如果 `decision_point` 异常或缺失：

```text
intent_mask = 0
```

同时在 statistics 中记录 missing / invalid count。

---

# 6.7 JAAD joint sample 逻辑

JAAD 每个 window 最终分成：

### 类型 A：behavior track 且位于 decision point 之前

```text
JEPA       ✓
Trajectory ✓
Intention  ✓
```

### 类型 B：behavior track 但 observation 已晚于 decision point

```text
JEPA       ✓
Trajectory ✓
Intention  ✗
```

### 类型 C：没有 behavior annotation

```text
JEPA       ✓
Trajectory ✓
Intention  ✗
```

不要为了 joint task 只保留 Type A。

---

# 6.8 JAAD 参考统计与实际统计要求

JAAD default split 下常见的 track-level 参考统计：

| Split | JAAD_all tracks | JAAD_beh tracks |
|---|---:|---:|
| Train | 1355 | 324 |
| Val | 202 | 48 |
| Test | 1023 | 276 |
| Total | 2580 | 648 |

这些同样只作为 sanity-check，不得硬编码。

Codex 必须实际统计：

1. videos；
2. raw `JAAD_all` tracks；
3. raw behavior tracks；
4. `L >= 61` 的 all tracks；
5. `L >= 61` 的 behavior tracks；
6. total joint windows；
7. windows from behavior tracks；
8. intent-valid windows；
9. intention positive windows；
10. intention negative windows；
11. `crossing=-1/0/1` 的 track 数；
12. `crossing=-1/0/1` 的 window 数；
13. missing / invalid decision point 数。

若当前 annotation 版本与参考值不同，记录而不是人为修正。

---

# 7. 统一 window 生成伪代码

建议抽象公共函数。

```python
def generate_joint_windows(
    track,
    observe_length,
    predict_length,
    stride,
    min_track_size=61,
):
    assert observe_length == 15
    assert predict_length == 45

    window_length = observe_length + predict_length  # 60

    if len(track["frame_ids"]) < min_track_size:
        return []

    samples = []

    for start in range(
        0,
        len(track["frame_ids"]) - window_length + 1,
        stride,
    ):
        end = start + window_length

        sample = slice_track(track, start, end)

        assert len(sample["frame_ids"]) == 60

        sample["obs"] = slice_sample(sample, 0, 15)
        sample["future"] = slice_sample(sample, 15, 60)

        samples.append(sample)

    return samples
```

PIE：

```python
stride = 7
```

JAAD：

```python
stride = 3
```

---

# 8. sample_id 规则

必须保证稳定、确定、可重复生成。

PIE 建议：

```text
pie_{set_id}_{video_id}_{ped_id}_s{start_frame}
```

JAAD 建议：

```text
jaad_{video_id}_{ped_id}_s{start_frame}
```

例如：

```text
pie_set01_video_0001_1_1_12_s00123
jaad_video_0042_0_42_3b_s00315
```

同一份 annotation 重跑预处理后，`sample_id` 必须完全一致。

---

# 9. JEPA 相关 target 准备原则

canonical manifest 保留完整 future raw information：

```text
future_image_paths: 45
future_bbox:        45
future_center:      45
```

训练阶段：

- context encoder 只读取 observed 15 帧；
- target encoder 可读取 future 45 帧对应的 target modality；
- predictor 从 past latent 预测 future latent；
- future raw annotation 绝不作为 predictor input。

如果当前模型未来 target 使用：

```text
visual future latent
motion future latent
```

则两个 target 必须由同一个 canonical future 45-frame window 生成。

不要再让 visual target 和 motion target 来自不同时间切片。

---

# 10. intention head 的统一接口

最终 dataset loader 对两个数据集统一返回：

```python
batch["intent_binary"]  # float tensor, missing position can temporarily fill 0
batch["intent_mask"]    # float tensor, 0 or 1
```

PIE 额外：

```python
batch["intent_soft"]
batch["intent_soft_mask"]
```

推荐：

```python
intent_soft_mask = intent_mask
```

训练计算时必须：

```python
intent_loss = (
    per_sample_intent_loss * intent_mask
).sum() / max(intent_mask.sum(), 1)
```

禁止让 `intent_mask=0` 的伪填充值参与 BCE。

---

# 11. trajectory head 的统一接口

两个数据集统一输出：

```python
batch["obs_bbox"]
batch["future_bbox"]
batch["obs_center"]
batch["future_center"]
```

shape：

```text
obs_bbox:      [B, 15, 4]
future_bbox:   [B, 45, 4]
obs_center:    [B, 15, 2]
future_center: [B, 45, 2]
```

训练使用 normalized bbox 或 displacement 均可，但 evaluation 必须能够恢复到 canonical absolute / normalized coordinates。

---

# 12. 数据统计文件

每个 dataset 必须自动输出：

```text
statistics.json
statistics.md
```

`statistics.json` 建议结构：

```json
{
  "dataset": "pie",
  "protocol": {
    "fps": 30,
    "observe_length": 15,
    "predict_length": 45,
    "min_track_size": 61,
    "overlap": 0.5,
    "stride": 7
  },
  "splits": {
    "train": {
      "videos": 0,
      "raw_tracks": 0,
      "valid_tracks_ge_61": 0,
      "joint_windows": 0,
      "intent_labeled_tracks": 0,
      "intent_valid_windows": 0,
      "intent_positive_windows": 0,
      "intent_negative_windows": 0
    },
    "val": {},
    "test": {}
  }
}
```

JAAD 额外统计：

```json
{
  "behavior_tracks": 0,
  "crossing_attr_minus1_tracks": 0,
  "crossing_attr_0_tracks": 0,
  "crossing_attr_1_tracks": 0,
  "invalid_decision_point_tracks": 0
}
```

PIE 额外统计：

```json
{
  "soft_intent_mean": 0.0,
  "soft_intent_std": 0.0,
  "invalid_exp_or_critical_tracks": 0
}
```

---

# 13. statistics.md 最终表格要求

PIE：

| Split | Videos | Raw Tracks | Valid Tracks >=61 | Joint Windows | Intent-valid Windows | Intent+ | Intent- |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | | | | | | | |
| Val | | | | | | | |
| Test | | | | | | | |
| Total | | | | | | | |

JAAD：

| Split | Videos | All Tracks | Beh Tracks | Valid Tracks >=61 | Joint Windows | Intent-valid Windows | Intent+ | Intent- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | | | | | | | | |
| Val | | | | | | | | |
| Test | | | | | | | | |
| Total | | | | | | | | |

再附：

### JAAD crossing attribute distribution

| Split | crossing=-1 | crossing=0 | crossing=1 |
|---|---:|---:|---:|
| Train | | | |
| Val | | | |
| Test | | | |

### PIE soft intention distribution

| Split | [0,.25) | [.25,.5] | (.5,.75] | (.75,1] |
|---|---:|---:|---:|---:|
| Train | | | | |
| Val | | | | |
| Test | | | | |

---

# 14. 数据完整性检查

`validate_joint_manifests.py` 必须执行以下检查。

## 14.1 split 无泄漏

检查：

```text
train_video_ids ∩ val_video_ids  = ∅
train_video_ids ∩ test_video_ids = ∅
val_video_ids   ∩ test_video_ids = ∅
```

同时检查 `sample_id` 全局唯一。

---

## 14.2 时间长度

每个 sample：

```python
assert len(obs_frame_ids) == 15
assert len(future_frame_ids) == 45
assert len(obs_bbox_abs) == 15
assert len(future_bbox_abs) == 45
```

并检查 frame id 单调递增。

如果 annotation 原始 track 中间存在非连续 frame：

- 不要无声插值；
- 默认丢弃跨越 frame gap 的 window；
- statistics 中记录 `dropped_non_contiguous_windows`。

canonical sample 必须满足：

```python
frame_ids[i + 1] == frame_ids[i] + 1
```

因为当前 protocol 假设 30 FPS 连续时间。

---

## 14.3 bbox 合法性

每个 bbox：

```python
0 <= x1 < x2 <= image_width
0 <= y1 < y2 <= image_height
```

允许极少数 annotation 越界时：

- 先统计；
- 默认 clip 到 image boundary；
- 保留 `bbox_was_clipped=true`；
- 不要无声修改。

---

## 14.4 intention mask

PIE：

```python
if intent_mask == 1:
    assert exp_start_point <= obs_start_frame
    assert obs_end_frame <= critical_point
    assert intent_soft is not None
    assert intent_binary in {0, 1}
```

JAAD：

```python
if intent_mask == 1:
    assert has_behavior_annotation
    assert obs_end_frame <= decision_point
    assert crossing_attr in {-1, 0, 1}
    assert intent_binary in {0, 1}
```

JAAD 无 behavior annotation：

```python
assert intent_mask == 0
```

---

# 15. 不允许出现的 preprocessing 行为

Codex 实现时明确禁止以下做法：

1. 不允许对所有 sample random split；
2. 不允许 pedestrian-level random split 替代官方 video/set split；
3. 不允许先生成 window 后再 split；
4. 不允许把 JAAD 无行为标注 pedestrian 当作 intent negative；
5. 不允许把 `crossing == 0` 自动定义成 “no intention”；
6. 不允许把 PIE `crossing` action 替代 `intention_prob` 作为主 intention label；
7. 不允许为了 intention task 删除没有 intention label 的 trajectory samples；
8. 不允许把 GT intention 输入 trajectory decoder；
9. 不允许 train/val/test 使用不同 bbox normalization；
10. 不允许对 val/test 做 class balancing；
11. 不允许对 split 内 track 重排后导致 sample_id 不稳定；
12. 不允许用随机 seed 决定核心数据划分；
13. 不允许为了匹配文献参考 sample 数而人工删样本；
14. 不允许 silently interpolate frame gaps；
15. 不允许在 canonical preprocessing 中把 45 future frames 改成当前模型的 30 query 数。

---

# 16. Codex 实现任务清单

后续 Codex 根据本文档改数据时，按以下顺序执行。

## Phase 1：只读检查

先检查项目现有：

```text
dataset.py
data loader
JAAD config
PIE config
preprocess scripts
training.py
evaluate.py
```

确认目前：

- split 在哪里定义；
- overlap 在哪里定义；
- observation / prediction length 在哪里定义；
- intention label 当前如何生成；
- 当前是否存在 trajectory 与 intention 两套独立数据；
- 当前模型是否使用 ground-truth intention 作为输入。

先输出检查结果，不要立即大改模型。

---

## Phase 2：实现 canonical preprocessors

实现：

```text
preprocess_pie_joint.py
preprocess_jaad_joint.py
```

目标：

- 只依赖官方 annotation；
- deterministic；
- 重跑结果完全一致；
- 输出统一 JSONL manifest；
- 自动输出 statistics。

---

## Phase 3：实现 validation

实现：

```text
validate_joint_manifests.py
```

所有 assertion 通过后才允许修改 training dataset loader。

---

## Phase 4：适配 dataset loader

旧 loader 改成从 canonical manifest 读取。

必须返回至少：

```python
{
    "obs_images": ...,
    "future_images": ...,

    "obs_bbox": ...,
    "future_bbox": ...,

    "obs_center": ...,
    "future_center": ...,

    "intent_binary": ...,
    "intent_mask": ...,

    "dataset_name": ...,
    "sample_id": ...
}
```

PIE 可额外返回：

```python
"intent_soft"
```

---

## Phase 5：保持 baseline 可比性

重新划分数据后：

- 先跑简单 constant velocity / linear trajectory baseline；
- 再跑当前旧模型；
- 再跑新 JEPA 模型。

确认数据改造本身没有导致 metric 计算错误。

不要在同一 commit 同时大改：

- preprocessing；
- model architecture；
- loss；
- evaluation metric。

优先先锁定数据。

---

# 17. 验收标准

预处理重构完成必须满足：

### Split

- PIE = `set01+set02+set04 / set05+set06 / set03`；
- JAAD = official `default` video split；
- 无 video 泄漏。

### Time

- 每个 sample 15 observed；
- 每个 sample 45 future；
- PIE stride = 7；
- JAAD stride = 3。

### Trajectory

- 所有 joint windows 都有 trajectory target；
- bbox / center 可恢复；
- future 45 帧完整保存。

### Intention

PIE：

- 使用 `intention_prob`；
- 保存 soft + binary；
- 只在 `exp_start_point -> critical_point` 范围启用 `intent_mask`。

JAAD：

- trajectory 使用 all valid tracks；
- intention 只监督 behavior-annotated tracks；
- `crossing=-1 -> intent=0`；
- `crossing in {0,1} -> intent=1`；
- observation 晚于 `decision_point` 后 `intent_mask=0`。

### JEPA

- observed 与 future 来自同一个 canonical sample；
- future target 不作为 predictor input；
- intention label 不作为 trajectory input；
- 无 intention label 的样本仍参与 JEPA + trajectory training。

### Reproducibility

- 两次运行 manifest 的 sample_id 集合完全一致；
- statistics 完全一致；
- 所有数据统计由代码自动生成，不手填最终 window 数。

---

# 18. 论文实验章节后续应报告的最终统计

等 preprocessing 跑完后，论文不要只报告原始 pedestrian 数。

至少报告：

| Dataset | Split | Videos | Valid Tracks | Joint Windows | Intent-supervised Windows | Obs | Pred | Stride |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PIE | Train | 23 | AUTO | AUTO | AUTO | 15 | 45 | 7 |
| PIE | Val | 11 | AUTO | AUTO | AUTO | 15 | 45 | 7 |
| PIE | Test | 19 | AUTO | AUTO | AUTO | 15 | 45 | 7 |
| JAAD | Train | 177 | AUTO | AUTO | AUTO | 15 | 45 | 3 |
| JAAD | Val | 29 | AUTO | AUTO | AUTO | 15 | 45 | 3 |
| JAAD | Test | 117 | AUTO | AUTO | AUTO | 15 | 45 | 3 |

其中所有 `AUTO` 必须来自最终 preprocessing 实际运行结果。

---

# 19. 参考官方实现

PIE：

- Dataset / annotations:
  - https://github.com/aras62/PIE
- PIE data interface:
  - https://github.com/aras62/PIE/blob/master/utilities/pie_data.py
- PIEPredict:
  - https://github.com/aras62/PIEPredict

JAAD：

- JAAD:
  - https://github.com/ykotseruba/JAAD
- JAAD 2.0 data interface:
  - https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/jaad_data.py
- Pedestrian Action Prediction Benchmark:
  - https://github.com/ykotseruba/PedestrianActionBenchmark

实现时优先以当前下载的数据集 annotation 和官方 data interface 为准；论文中的 track 数只作为 sanity-check。

---

# 20. 最终数据设计总结

最终不要得到：

```text
trajectory_dataset/
intent_dataset/
```

而是分别得到：

```text
PIE canonical joint dataset
JAAD canonical joint dataset
```

每一个 sample 都是：

```text
15-frame past context
        |
        v
joint future prediction
        |
        +------------------+
        |                  |
        v                  v
45-frame trajectory     intention
                       (masked)
```

训练时：

```math
\mathcal L
=
\lambda_{JEPA}\mathcal L_{JEPA}
+
\lambda_{traj}\mathcal L_{traj}
+
\lambda_{intent}M_{intent}\mathcal L_{intent}
```

这套数据 protocol 的目的不是简单做 multi-task learning，而是保证 **trajectory 与 intention 在数据定义上就描述同一个 future event**，从而与当前 JEPA “在共享未来隐空间中统一建模运动演化与行为意图”的方法动机保持一致。
