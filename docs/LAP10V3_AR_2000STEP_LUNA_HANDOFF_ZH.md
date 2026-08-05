# LAP10V3 AR-2000：真实动作结果对齐训练方案（Luna 交付版）

## 1. 目标与命名

本轮训练命名为：

```text
Aligned Results Training（AR）
```

固定任务：

```text
move_pillbottle_pad（任务 14）
```

AR 的目标是直接利用 RoboTwin 采集轨迹中的真实 EEF action chunk，优化：

```text
三视角 DINO 特征 + 当前 EEF
        ↓
LAP10V3 条件序列 [B,284,768]
        ↓
Action Expert
        ↓
预测 action chunk
        ↕ 直接对齐
数据集真实 action chunk
```

本轮不是 VLM 蒸馏，不允许加载或使用：

- Qwen/VLM；
- VLM teacher-condition cache；
- Teacher Action Expert；
- teacher velocity；
- VLM token MSE/cosine/structure loss。

本轮只使用采集数据中的真实 action 作为监督。旧的 VLM teacher cache 仅保留在磁盘，AR 代码不得读取。

## 2. 当前基线与本轮初始化

初始化 checkpoint：

```text
outputs/lap10v3_expert_joint_task14_2000step/
└── lap10v3_expert_step0002000.pt
```

该 checkpoint 包含：

- 联合训练后的 LAP10V3；
- 联合训练后的 Action Expert。

此前联合训练共 2000 optimizer step，effective batch 为 4，累计访问约 8000 个样本，约等于旧 8192 样本集合的 0.98 epoch。此前训练为混合目标：

```text
L_old
= flow loss（真实 action）
+ VLM token alignment losses
```

AR 必须保留其中已经有效的模型权重，但重新初始化 optimizer/scheduler，并删除所有 VLM 对齐项。

当前闭环基线：

| 场景 | 成功数 |
|---|---:|
| Clean | 1/5 |
| Randomized | 3/5 |
| 合计 | 4/10 |

AR-2000 必须与该 4/10 基线比较，不能只比较训练 loss。

## 3. 现有数据与缓存：禁止重新预处理

现有缓存已经覆盖完整 train split，不需要重新运行 DINO、读取原始视频或生成 VLM teacher。

### 3.1 Train split

```text
480 条训练轨迹
├── randomized：440
└── clean：       40

训练时刻样本：24,140
```

现有缓存：

| 内容 | 路径 | Shard 数 | 样本数 |
|---|---|---:|---:|
| 主视角/Stage 1 特征与 EEF | `cache/lap_stage1_task14/train` | 189 | 24,140 |
| 左右腕 DINO 特征 | `cache/lap_stage1_task14_wrist/train` | 189 | 24,140 |
| 真实 action chunk | `cache/lap8_phase1_task14_actions/train` | 189 | 24,140 |

前 188 个 shard 各 128 个样本，最后一个 shard 有 76 个样本：

```text
188 × 128 + 76 = 24,140
```

最后一个 shard 已确认：

```text
vision_t           [76,256,768]
vision_t1          [76,256,768]
z_idm              [76,1,32]
state_t            [76,16]
state_t1           [76,16]
vision_left_t      [76,256,768]
vision_right_t     [76,256,768]
actions            [76,50,32]
actions_mask       [76,50,32]
```

RoboTwin 30 Hz、1.2 秒 horizon 下，每个样本有 36 个有效 action token；`actions_mask` 是唯一有效性依据，禁止把 50 个位置全部计入 loss。

### 3.2 数据读取修改

现有 `tools/train_lap10_alignment.py` 中的 `LAP10Dataset` 使用：

```python
def __len__(self):
    return len(self.teacher)
```

因此旧联合训练被 8192 条 teacher cache 截断。AR 必须新增不含 teacher 的 dataset，例如：

```text
ARActionDataset
└── 直接包装 Phase1Dataset
    ├── feature cache
    ├── wrist cache
    └── action cache
```

AR dataset 的长度必须严格为：

```text
24,140
```

`__getitem__` 只返回：

```text
vision_t
vision_left_t
vision_right_t
state_t
actions
actions_mask
```

`vision_t1`、`z_idm` 和 `state_t1` 可以存在于缓存中，但本轮不得作为 teacher action 或 VLM teacher 使用。

### 3.3 数据分布

AR-2000 第一轮保持 manifest 的原始打乱顺序和自然比例：

```text
randomized：约 91%
clean：      约 9%
```

本轮不做 clean 过采样，避免同时改变过多变量。关键阶段加权在 loss 内在线计算，不生成新缓存。

## 4. 模型输入输出和冻结边界

### 4.1 固定前向链路

```text
visual = stack(main, left_wrist, right_wrist)
visual.shape = [B,3,256,768]

state_t.shape = [B,16]
actions.shape = [B,50,32]
actions_mask.shape = [B,50,32]
```

LAP10V3：

```text
visual + state_t
    ↓
LAP6（冻结）
    ├── z_lap [B,1,32] → LaWM
    └── scene tokens
    ↓
LAP7–10
    ↓
cond_lap [B,284,768]
```

LaWM：

```text
h_t = 主视角 DINO tokens [B,256,768]
z_lap = LAP6 输出
h_t1 = LaWM(h_t, z_lap)
```

Action Expert：

```text
h_t
h_t1
h_lap = cond_lap
state = zeros [B,32]
state_mask = zeros [B,32]
actions / actions_mask
action_hz = 30
embodiment_id = 1
```

官方 RoboTwin flow config 为 `use_state=false`，因此 Expert 使用 zero state 是与官方配置一致的；真实 EEF 已经输入 LAP10V3。

### 4.2 LAP10V3 代码映射

实现位置：

```text
starVLA/model/lap_stage2.py
└── class LAP10V3
```

模块边界：

```text
LAP10V3
├── lap6                         # Stage 1 trunk，永久冻结
├── content_queries
├── role_embeddings
├── view_embeddings
├── latent_to_memory
├── state_to_memory
├── blocks[0]                    # LAP7
├── blocks[1]                    # LAP8
├── blocks[2]                    # LAP9
├── blocks[3]                    # LAP10
├── residual_norm
├── residual_head
├── residual_scale
└── teacher_position_mean        # 固定 buffer，不训练
```

注意：`teacher_position_mean` 只是已经存在于 LAP10V3 checkpoint 中的固定输出模板，不是本轮 teacher，不需要 teacher cache，也不得更新。

## 5. AR-2000 两阶段训练

总训练：

```text
2000 optimizer step
FP32
双卡 DDP
effective batch = 16
```

数据访问量：

```text
2000 × 16 = 32,000
32,000 / 24,140 = 1.33 epoch
```

### 5.1 AR-A：step 1–1000

目的：固定当前 LAP token 分布，让联合后的 Expert 先稳定适配真实 action。

| 模块 | 状态 |
|---|---|
| LAP6 | 冻结、eval |
| LAP7–8 和底层 LAP 分支 | 冻结、eval |
| LAP9–10 和输出头 | DDP 注册，但 AR-A 前向使用 `no_grad/detach`，有效冻结 |
| LaWM | 冻结、eval |
| Action Expert（除 `enc_vlm`） | 更新、train |
| `enc_vlm` | 冻结、未使用 |

AR-A 不允许 LAP 使用 view dropout；LAP10V3 必须以确定性 eval 模式生成条件。建议 `view_dropout=0.0`。

学习率：

```text
Expert LR：1.0e-6
step 1–100：linear warmup
step 101–1000：保持或缓慢 cosine 衰减
```

### 5.2 AR-B：step 1001–2000

目的：只允许最靠近结果接口的两层和输出头通过真实 action loss 做小幅校正。

| 模块 | 状态 |
|---|---|
| LAP6 | 冻结、eval |
| content/role/view embeddings | 冻结 |
| latent/state projections | 冻结 |
| LAP7–8：`blocks[0:2]` | 冻结 |
| LAP9–10：`blocks[2:4]` | 更新 |
| `residual_norm/head/scale` | 更新 |
| `teacher_position_mean` | 固定 buffer |
| LaWM | 冻结、eval |
| Action Expert（除 `enc_vlm`） | 更新 |

学习率：

```text
LAP9–10 + output LR：1.0e-6
Expert LR：            5.0e-7
AR-B LAP warmup：      50 step
最终最低 LR：          各自初始 LR 的 0.1
```

禁止解冻 LAP6、LAP7–8 或底层 query/embedding/projection。

### 5.3 Optimizer 与渐进解冻

使用 AdamW：

```text
betas：       PyTorch 默认值
eps：         PyTorch 默认值
weight_decay：0.01
grad clip：   1.0
```

建议在训练开始、DDP 包装之前固定最终参数集合，并创建两个参数组：

1. Expert 参数组；
2. AR-B LAP 参数组（`blocks[2:4]`、`residual_norm/head/scale`）。

AR-B LAP 参数从进程启动时即设置 `requires_grad=True` 并注册给 DDP，但在 step 1–1000：

- LAP 条件前向必须包在 `torch.no_grad()` 中，传给 Expert 的 `cond_lap` 必须 `detach()`；
- LAP 参数组有效 LR 为 0。

从 step 1001 开始：

- 只对规定的 AR-B LAP 参数恢复普通带梯度前向，不再 `no_grad/detach`；
- 使用独立的 50-step warmup/cosine schedule；
- 保留 Expert 的 optimizer moments，不重新创建整个 optimizer。

DDP 必须使用：

```text
find_unused_parameters=True
```

原因是 AR-A 中已注册的 LAP9–10/output 参数不会进入反向图，AR-B 才开始使用。禁止在 DDP 构造完成后改变参与同步的参数集合；禁止通过 step 1001 临时把未注册参数从 `requires_grad=False` 改成 `True`。

## 6. AR 损失：只对齐真实动作

### 6.1 Flow matching 基础

对真实动作 `a`、噪声 `eps` 和 flow time `t`：

```text
x_t = (1 - t) * eps + t * a
velocity_target = a - eps
```

Action Expert 预测：

```text
velocity_pred
```

从同一次前向的预测速度还原动作估计：

```text
action_pred = x_t + (1 - t) * velocity_pred
```

同一个 micro-batch 的 flow、action reconstruction 和 delta loss 必须复用完全相同的 `eps`、`t`、`x_t` 和 `velocity_pred`，禁止为三个 loss 分别重新采样。

### 6.2 总损失

```text
L_AR
= 1.0 * L_flow_weighted
+ 0.5 * L_action_reconstruction
+ 0.1 * L_action_delta
```

所有 loss 必须应用 `actions_mask`。归一化分母使用有效加权元素之和，禁止包含 padding。

### 6.3 动作维度映射

真实有效动作维度为前 16 维：

```text
左臂：
0:3   xyz
3:7   quaternion
7     gripper

右臂：
8:11  xyz
11:15 quaternion
15    gripper
```

Flow loss 的维度权重：

| 维度 | 权重 |
|---|---:|
| 左臂 xyz（0:3） | 2.0 |
| 左臂 quaternion（3:7） | 1.0 |
| 左夹爪（7） | 4.0 |
| 右臂全部有效维度（8:16） | 1.0 |
| padding（16:32） | 0，由 mask 决定 |

### 6.4 关键时刻权重

缓存中的左夹爪动作已经归一化到约 `[-1,1]`。本任务中已核实：

```text
-1：张开
+1：闭合
```

根据 GT 左夹爪序列 `g = actions[...,7]` 在线构造时间权重：

```text
普通时刻：                 1.0
闭合/松开 transition ±4： 4.0
闭合后的保持/提起时刻：    3.0
```

transition 可通过相邻差分检测：

```text
abs(g[t] - g[t-1]) > 0.1
```

闭合保持区间可使用：

```text
g[t] > 0.8
```

当维度权重与时间权重同时存在时，元素权重为二者乘积，并在最终 loss 中按权重和重新归一化。

### 6.5 Action reconstruction loss

第一版 AR 直接重点约束当前已观察到的失败维度：左臂 xyz 和左夹爪。

```text
L_action_reconstruction
= SmoothL1(action_pred[...,0:3], actions[...,0:3])
+ 2.0 * SmoothL1(action_pred[...,7], actions[...,7])
```

仍需应用关键时刻权重和 mask。

Quaternion 和右臂仍由 weighted flow loss 监督。第一版不要直接对 quaternion 分量做普通 delta loss，避免 quaternion 符号等价问题。

### 6.6 Action delta loss

只比较相邻有效 token 的左臂 xyz 变化：

```text
delta_pred = action_pred[:,1:,0:3] - action_pred[:,:-1,0:3]
delta_gt   = actions[:,1:,0:3]     - actions[:,:-1,0:3]

L_action_delta = SmoothL1(delta_pred, delta_gt)
```

pair mask：

```text
pair_valid = actions_mask[:,1:,0:3] & actions_mask[:,:-1,0:3]
```

该项用于抑制 EEF chunk 内的不必要跳变，但不得用“预测动作自身尽量平滑”替代 GT delta 对齐，否则会把真实快速抓取动作过度平滑。

## 7. Flow Head 代码修改约束

当前实现：

```text
starVLA/model/framework/vlas/flowmatching_expert.py
└── ConditionalFlowMatchingHead.forward
```

当前 `forward` 只返回 scalar flow loss。AR 需要获得：

```text
pred_velocity
velocity_target
x_t
time
actions_mask
```

实现必须保持官方/旧训练调用向后兼容。推荐新增一个训练专用、可选返回结构，例如：

```python
forward(..., return_training_details=False, element_weights=None)
```

默认 `False` 时：

- 返回值和旧逻辑完全不变；
- 原有训练和推理测试不得受影响。

AR 调用 `True` 时返回 dict：

```text
loss_flow
pred_velocity
velocity_target
x_t
time
```

不要在 AR 训练脚本中复制整份 300 多行 flow forward；应在 flow head 内以向后兼容方式暴露已有中间量，避免两套实现漂移。

## 8. 建议新增文件和输出命名

新增训练脚本：

```text
tools/train_lap10v3_ar_ddp.py
```

可复用：

```text
tools/train_lap10v3_expert_joint_ddp.py
tools/train_lap10_alignment.py
tools/train_lap8_phase1.py
```

输出目录：

```text
outputs/lap10v3_ar_expert_task14_2000step/
```

Checkpoint：

```text
lap10v3_ar_step0000500.pt
lap10v3_ar_step0001000.pt
lap10v3_ar_step0001500.pt
lap10v3_ar_step0002000.pt
```

日志：

```text
logs/lap10v3_ar_expert_task14_2000step/train.log
```

## 9. Checkpoint 必须保存的内容

每个 checkpoint 至少包含：

```text
lap10v3
expert
optimizer
scheduler 或 scheduler state
global_step
phase                 # ar_a 或 ar_b
args/config
trainable_parameter_names
loss_weights
cache_paths
source_joint_checkpoint
```

建议同时保存 Python、NumPy、Torch CPU/CUDA RNG state，保证必要时可精确恢复。

step 1000 checkpoint 必须在 AR-A 完成、AR-B 解冻前保存；不能把已经解冻更新过的参数误标为 step 1000。

## 10. 日志指标

每 10 step 记录：

```text
step / phase
loss_total
loss_flow_weighted
loss_action_reconstruction
loss_action_delta
left_xyz_recon_mae
left_gripper_recon_mae
gripper_event_mae
expert_grad_norm
lap9_10_grad_norm
total_grad_norm
expert_lr
lap_lr
samples_seen
effective_epoch
step_time
ETA
peak_cuda_memory
```

梯度要求：

```text
AR-A：
Expert grad > 0
全部 LAP grad = 0 / None（包括已注册但由 no_grad/detach 隔离的 LAP9–10/output）

AR-B：
Expert grad > 0
LAP9–10 和 residual output grad > 0
LAP6、LAP7–8、底层 embeddings/projections grad = 0 / None
```

日志中不得出现 teacher MSE、teacher cosine、teacher structure 或 VLM velocity 等指标。

## 11. 训练参数和启动命令

固定参数：

```text
precision：             FP32
world size：            2
per-device batch：      2
gradient accumulation： 4
effective batch：       16
steps：                 2000
AR-B start：            1001
warmup：                100
AR-B LAP warmup：       50
weight decay：          0.01
grad clip：             1.0
save every：            500
log every：             10
view dropout：          0.0
```

建议命令：

```bash
cd /data/pxchen/LaWAM
mkdir -p logs/lap10v3_ar_expert_task14_2000step \
         outputs/lap10v3_ar_expert_task14_2000step

CUDA_VISIBLE_DEVICES=6,7 \
/home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tools/train_lap10v3_ar_ddp.py \
  --joint-checkpoint outputs/lap10v3_expert_joint_task14_2000step/lap10v3_expert_step0002000.pt \
  --feature-cache cache/lap_stage1_task14 \
  --wrist-cache cache/lap_stage1_task14_wrist \
  --action-cache cache/lap8_phase1_task14_actions \
  --steps 2000 \
  --phase-b-start 1001 \
  --batch-size 2 \
  --grad-accumulation 4 \
  --expert-lr-a 1e-6 \
  --expert-lr-b 5e-7 \
  --lap-lr-b 1e-6 \
  --warmup-steps 100 \
  --phase-b-warmup-steps 50 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --view-dropout 0.0 \
  --save-every 500 \
  --log-every 10 \
  --output-dir outputs/lap10v3_ar_expert_task14_2000step \
  |& tee logs/lap10v3_ar_expert_task14_2000step/train.log
```

若 GPU 6、7 被占用，只允许改 `CUDA_VISIBLE_DEVICES`，不能擅自改变 world size、batch 或 effective batch。

实时日志：

```bash
tail -f /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

## 12. Smoke test：正式训练前必须通过

### 12.1 静态检查

1. Dataset 长度严格为 24,140；
2. 三套缓存 shard 数均为 189；
3. 最后一个 shard 均为 76 个样本；
4. AR 进程参数和模块名中不存在 Qwen/VLM teacher；
5. 不打开 `cache/lap10_task14_vlm_teacher_8192`。

### 12.2 单卡 2-step smoke

至少检查：

```text
loss 全部 finite
actions_mask 有效 token 数为 36
action_pred shape = [B,50,32]
cond_lap shape = [B,284,768]
h_t / h_t1 shape 合法
无 NaN/Inf
```

### 12.3 双卡 20-step smoke

```text
CUDA_VISIBLE_DEVICES=6,7
world=2
effective batch=16
DDP 无 unused-parameter error
DDP 使用 find_unused_parameters=True
rank 间 loss all-reduce 正常
显存稳定，无持续增长
```

AR-A 梯度审计必须证明 LAP 全冻结。可用一个专门的 `--audit-gradients` 模式列出所有出现非零梯度的参数名。

### 12.4 阶段切换 smoke

使用短配置令 AR-B 在 step 3 开始，检查：

- step 2 checkpoint 仍是全 LAP 冻结状态；
- step 3 仅 LAP9–10/output head 出现梯度；
- optimizer 没有重建导致 Expert moments 丢失；
- LAP LR 从 AR-B warmup 起点正确变化。

这里的“全 LAP 冻结”指 AR-A 的有效梯度状态；LAP9–10/output 已在 DDP 中注册，但必须通过 `no_grad/detach` 保持 `grad=None`。

## 13. 离线验证

不得仅用训练 loss 判断模型。

对固定 validation subset、固定 noise 和固定 flow time，至少比较：

```text
source joint step2000
AR step500
AR step1000
AR step1500
AR step2000
```

指标：

- weighted flow loss；
- 左臂 xyz action reconstruction MAE；
- 左夹爪 reconstruction MAE；
- 夹爪 transition 附近 MAE；
- 左臂 xyz delta MAE；
- 固定 noise 下完整采样 action 与 GT 的分组误差。

AR step 1000 是 Expert-only；AR step 2000 是 LAP9–10 + Expert。必须单独报告两者，不能只报告最后 checkpoint。

## 14. RoboTwin 闭环验证口径

AR 训练结束后，至少测试：

```text
AR step1000
AR step2000
```

固定设置：

```text
任务：              move_pillbottle_pad
精度：              FP32
VLM：               不加载
diffusion steps：    10
replan steps：       36
action ensemble：    关闭
Clean：              5 条
Randomized：         5 条
```

固定 seeds：

```text
Clean：
100002, 100005, 100006, 100007, 100008

Randomized：
100001, 100002, 100003, 100004, 100006
```

选择 checkpoint 的第一指标：

```text
max min(clean_successes, randomized_successes)
```

若相同，再比较总成功数、左臂 xyz 误差和夹爪 transition 误差。禁止默认选择最后一步。

## 15. 验收标准

代码链路验收：

1. 完全不加载 VLM、Teacher Expert、teacher-condition cache；
2. 训练数据长度为 24,140；
3. loss 只来自真实 actions/actions_mask；
4. AR-A 和 AR-B 梯度边界正确；
5. FP32 双卡 effective batch 16；
6. 四个 checkpoint 均可在无 VLM 服务中恢复推理；
7. 训练和验证无 NaN/Inf/OOM。

实验验收：

```text
最低要求：AR 最佳 checkpoint 超过当前 4/10
阶段目标：clean 和 randomized 均至少 4/5
开发目标：固定 5+5 达到 10/10
```

若 AR step1000 已明显优于 step2000，说明解冻 LAP9–10 导致泛化退化，应选 step1000，并停止继续解冻。若两者均未超过 4/10，不得直接增加 step，必须先检查分组动作误差和闭环失败视频。

## 16. Luna 交付清单

Luna 完成后应交付：

1. `tools/train_lap10v3_ar_ddp.py`；
2. flow head 的向后兼容中间量返回修改及相应测试；
3. 单卡 smoke 日志；
4. 双卡 20-step smoke 日志；
5. 阶段切换与梯度审计日志；
6. 正式 AR-2000 日志；
7. step 500/1000/1500/2000 checkpoint；
8. 离线 validation 对比表和 loss/action-error 曲线；
9. step1000 与 step2000 的 RoboTwin 5+5 结果和视频；
10. 最佳 checkpoint 选择结论。

不得以“总 loss 下降”代替上述交付。
