# LAP10V3 AR-2000 训练会话完整记录

> 文档类型：实验执行记录 / 可复现实验审计记录
> 记录日期：2026-08-05（Asia/Hong_Kong）
> 状态快照：2026-08-05 20:35 HKT（AR-2000 已完成，闭环待测）
> 工作目录：`/data/pxchen/LaWAM`
> 关联交付方案：`docs/LAP10V3_AR_2000STEP_LUNA_HANDOFF_ZH.md`
> 当前实验：`LAP10V3_AR_2000`，Aligned Results（AR）训练
> 重要性：本文件记录本次会话实际发生的事情；如果“交付方案”和“实际实现”存在差异，以本文件的“实际实现”章节为准。

---

## 1. 一页摘要

本次会话的目标，是按照 AR（Aligned Results）方案启动一次 **不依赖 VLM 蒸馏、直接使用 RoboTwin 真实 action chunk 监督** 的 LAP10V3 + Action Expert 训练。

实际执行结果：

1. 新增 AR 训练入口 `tools/train_lap10v3_ar_ddp.py`。
2. 保留并复用已有三视角 DINO 特征、EEF state 和 action cache，没有重新做数据预处理。
3. 加载此前 LAP10V3 + Action Expert 联合训练的 2000-step checkpoint。
4. 加载 Stage-1 LaWM 权重，但 LaWM 只作为冻结的未来视觉特征生成器，不参与更新。
5. 不加载 Qwen/VLM，不读取 VLM teacher cache，不加载 teacher Action Expert，不计算 token 蒸馏损失。
6. 双卡 FP32 smoke 测试通过：
   - AR-A 阶段 LAP 梯度为 0、Action Expert 梯度非零；
   - AR-B 阶段 LAP9–10 梯度出现、Action Expert 仍然更新；
   - 无 OOM、NaN、shape mismatch 或 action mask 错误。
7. 正式训练已脱离会话后台运行，当前 PID 为 `4189881`，使用 GPU 4、5。
8. 截至本状态快照，正式训练已经完成 step 2000，已保存 step 2000 checkpoint；AR-B 正常完成，闭环待测。

当前正式运行目录：

```text
/data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
/data/pxchen/LaWAM/outputs/lap10v3_ar_expert_task14_2000step/
```

实时查看：

```bash
tail -f /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

---

## 2. 实验背景和本轮问题定义

### 2.1 前序模型和失败现象

此前已有一个 LAP10V3 + Action Expert 联合训练 checkpoint：

```text
outputs/lap10v3_expert_joint_task14_2000step/
└── lap10v3_expert_step0002000.pt
```

此前模型的 RoboTwin 小规模闭环结果记录为：

| 场景 | 成功数 |
|---|---:|
| clean | 1/5 |
| randomized | 3/5 |
| 合计 | 4/10 |

前序联合训练使用过真实 flow loss，同时还包含 VLM token 对齐相关目标。因而即使其离线 loss 较低，也不能证明 Action Expert 已经学会了当前 LAP 条件下的真实动作映射。

本次 AR 训练的目的，是把监督口径改成：

```text
当前三视角观测 + 当前 EEF
        ↓
LAP10V3 条件 token
        ↓
LaWM / Expert 条件链路
        ↓
Action Expert
        ↓
直接对齐 RoboTwin 采集的真实 action chunk
```

本轮不再使用 VLM 的 284 token 作为目标，也不再试图通过 token MSE、cosine 或 structure loss 间接拟合 VLM。

### 2.2 “AR”在本会话中的严格含义

AR = **Aligned Results**，本会话中的定义是：

```text
真实采集动作监督
而不是
VLM/teacher 输出蒸馏
```

因此以下内容明确禁止进入 AR 训练图：

- Qwen-VL / VLM 在线推理；
- VLM teacher-condition cache；
- teacher Action Expert；
- teacher velocity；
- VLM token MSE；
- VLM token cosine loss；
- token structure loss；
- 任何以 VLM 284 token 为标签的 loss。

需要特别区分一个容易误解的点：旧 LAP10V3 checkpoint 内部保存了一个名为 `teacher_position_mean` 的 buffer。它是旧模型初始化时保存的固定 284×768 输出模板，当前只作为常数输出基线，不会读取 teacher cache、不计算 teacher loss，也不会更新。它不能被解释成当前 AR 训练正在使用 VLM teacher。

---

## 3. 数据集和缓存

### 3.1 任务和 split

本轮固定使用 RoboTwin 任务 14：

```text
move_pillbottle_pad
```

已有训练 split：

```text
480 条训练轨迹
├── randomized：440 条
└── clean：40 条
```

本轮没有重新划分数据、没有重新生成缓存、没有重新读取原始视频。训练使用完整 train cache，而不是此前旧联合训练中被 teacher cache 截断的 8192 样本子集。

### 3.2 实际使用的三个缓存

| 内容 | 实际路径 | 训练作用 |
|---|---|---|
| 主视角 / Stage-1 特征和 EEF | `cache/lap_stage1_task14/train` | 主视角 DINO token、`state_t`，同时提供 LAP 输入所需缓存字段 |
| 左右腕 DINO 特征 | `cache/lap_stage1_task14_wrist/train` | 与主视角按样本索引同步的左右腕 token |
| 真实 action chunk | `cache/lap8_phase1_task14_actions/train` | AR 唯一动作监督来源 |

每个目录有 189 个 shard：

```text
188 × 128 + 76 = 24,140 samples
```

正式进程启动时打印并核实：

```text
[data] 24,140 samples (35.4 GiB) resident in CPU RAM
```

主视角特征的 resident 部分约 35.4 GiB；左右腕特征和 action cache 也被预加载，因此两个 rank 会各自持有一份 CPU 数据副本。正式运行期间每个训练 rank 的主存占用约 86 GiB，总体仍在当前机器可承受范围内。

### 3.3 样本张量形状

特征和 state 的典型形状：

```text
vision_t        [B,256,768]
vision_t1       [B,256,768]   # 本轮不使用
vision_left_t   [B,256,768]
vision_right_t  [B,256,768]
z_idm           [B,1,32]      # 本轮不作为 teacher 使用
state_t         [B,16]
state_t1        [B,16]        # 本轮不使用
```

动作缓存：

```text
actions         [B,50,32]
actions_mask    [B,50,32]
```

在 RoboTwin 30 Hz、1.2 秒动作 horizon 下，有效位置为：

```text
前 36 个时间 token
每个 token 的前 16 个动作维度
```

也就是说，真实有效部分是 36×16；其余 50×32 中的 padding 必须由 `actions_mask` 排除。AR 代码没有把 50 个 token 或 32 个维度全部当作有效监督。

### 3.4 Action 维度口径

缓存前 16 维对应双臂 EEF action：

```text
左臂 xyz：        [0:3]
左臂 quaternion： [3:7]
左夹爪：          [7]
右臂 xyz：        [8:11]
右臂 quaternion： [11:15]
右夹爪：          [15]
```

已检查缓存中的左夹爪归一化约定：

```text
-1：张开
+1：闭合
```

---

## 4. 实际模型链路

### 4.1 输入

AR 脚本从 `Phase1Dataset` 读取数据，但只取以下字段：

```text
main  = vision_t
left  = vision_left_t
right = vision_right_t
state = state_t
action = actions
mask   = actions_mask
```

三视角输入在脚本内堆叠为：

```python
visual = torch.stack([main, left, right], dim=1)
visual.shape == [B,3,256,768]
```

### 4.2 LAP10V3

实现位置：

```text
starVLA/model/lap_stage2.py
└── class LAP10V3
```

模块组成：

```text
LAP10V3
├── lap6                         # Stage-1 trunk，永久冻结
├── content_queries              # 284 queries
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
└── teacher_position_mean        # 固定 buffer
```

LAP10V3 输出：

```text
z_lap       [B,1,32]       # LAP6 latent，供 LaWM 生成 h_t1
scene_lap6  [B,? ,768]    # LAP6 scene token
cond_lap    [B,284,768]   # 给 Action Expert 的 LAP 条件
```

其中 `cond_lap` 的 284 token 是 LAP10V3 的条件序列；它不是 VLM 输出，也不是本轮要对齐的 teacher label。

### 4.3 LaWM

LaWM 权重来自 Stage-1 checkpoint：

```text
outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt
```

LaWM 初始化所需 release 文件：

```text
latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt
latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml
```

实际加载过程：

```text
z_lap = frozen LAP6(visual, state)["z_lap"]
h_t1 = LaWM(main, z_lap)
```

LaWM：

- 以 FP32 加载；
- `eval()`；
- `requires_grad=False`；
- 不在 optimizer 中；
- 不产生 VLM teacher loss；
- 只为 Action Expert 提供冻结的未来视觉条件 `h_t1`。

这意味着本轮虽然保留了 LaWM 这条世界模型路径，但训练对象不是 LaWM，也不是 VLM 蒸馏。

### 4.4 Action Expert

Action Expert 结构由官方 RoboTwin Action Expert cache 构造：

```text
cache/lap8_phase1_official_action_expert.pt
```

随后覆盖此前联合训练 checkpoint 中的 Expert 权重：

```text
outputs/lap10v3_expert_joint_task14_2000step/lap10v3_expert_step0002000.pt
└── expert
```

实际输入：

```text
h_t       = 主视角 DINO token [B,256,768]
h_t1_star = 冻结 LaWM 输出
h_vlm     = None
h_lap     = LAP10V3 cond_lap [B,284,768]
state     = zeros [B,32]
state_mask= zeros [B,32]
action_hz = 30
embodiment_id = 1
actions / actions_mask = 真实缓存动作
```

官方 flow config 使用 `use_state=false`，所以 Action Expert 的 `state` 传入 zero tensor 是遵循官方 Expert 接口，而不是丢失了当前 EEF：真正的 EEF 已经进入 LAP10V3 的 `state_t`。

Action Expert 内部的 `enc_vlm` 参数仍然存在于官方 checkpoint 的 state dict 中，但在本轮：

- `h_vlm=None`；
- `enc_vlm` 参数冻结；
- 不进入有效训练路径；
- 不会产生梯度。

因此不能把“Action Expert checkpoint 中存在 enc_vlm 参数”误读成“本轮加载了 VLM”。

---

## 5. 初始化权重和冻结边界

### 5.1 初始化来源

| 模块 | 初始化来源 | 本轮是否更新 |
|---|---|---|
| LAP6 | `lap10v3_expert_step0002000.pt` 内的 `lap10v3.lap6.*` | 否 |
| LAP7–LAP10 / residual head | 同一联合 checkpoint 的 `lap10v3.*` | AR-A 否，AR-B 只更新 LAP9–10 和 output head |
| LaWM | Stage-1 checkpoint 的 `lawm_decoder`，结合 lam release 构造 | 否 |
| Action Expert | 官方 Expert 架构 + 联合 checkpoint 的 `expert.*` | 是，除 enc_vlm |
| `teacher_position_mean` | LAP10V3 checkpoint buffer | 否，固定 buffer |

### 5.2 AR-A：step 1–1000

目的：先在不改变 LAP 条件分布的前提下，让 Action Expert 适应真实 action target。

| 模块 | 实际状态 |
|---|---|
| LAP6 | `eval`、冻结 |
| LAP7–8 | 冻结 |
| LAP9–10 | 在 DDP 启动时注册，但前向使用 `no_grad`，输出 detach，实际梯度为 0 |
| content/role/view embedding | 冻结 |
| latent/state projection | 冻结 |
| residual head | AR-A 前向不回传 |
| LaWM | `eval`、冻结 |
| Action Expert 除 enc_vlm | `train`、更新 |
| enc_vlm | 冻结且不使用 |

AR-A 日志中应看到：

```text
lap_grad=0.000e+00
expert_grad>0
```

### 5.3 AR-B：step 1001–2000

目的：只让最靠近 Action Expert 接口的 LAP 后端做小幅真实动作校正，避免重新扰动 LAP6、LAP7–8 和 token/embedding 分布。

| 模块 | 实际状态 |
|---|---|
| LAP6 | 冻结、eval |
| content/role/view embedding | 冻结 |
| latent/state projection | 冻结 |
| blocks[0:2]（LAP7–8） | 冻结 |
| blocks[2:4]（LAP9–10） | 更新 |
| residual_norm | 更新 |
| residual_head | 更新 |
| residual_scale | 更新 |
| teacher_position_mean | 固定 buffer |
| LaWM | 冻结 |
| Action Expert 除 enc_vlm | 更新 |

AR-B 日志中应看到：

```text
lap_grad>0
expert_grad>0
```

### 5.4 DDP 参数注册策略

不能在 DDP 包装以后临时增加未注册参数。实际脚本采取的安全策略是：

1. 启动时就将 AR-B 的 LAP9–10/output 参数设置为 `requires_grad=True`；
2. 在 DDP 中使用 `find_unused_parameters=True`；
3. AR-A 前向时对 LAP 分支使用 `no_grad` 和 `detach`，并给 LAP 参数组 LR=0；
4. AR-B 从 step 1001 起恢复普通梯度前向和独立 LR；
5. 保留同一个 Action Expert optimizer，不清除其 AdamW moments。

这保证了 AR-A 的 LAP 参数虽然已注册，但不会被更新；同时 AR-B 可以安全打开反向路径。

---

## 6. AR 损失实际实现

### 6.1 Flow matching 输出

本次修改了：

```text
starVLA/model/framework/vlas/flowmatching_expert.py
└── ConditionalFlowMatchingHead.forward
```

向后兼容地增加了两个可选参数：

```python
return_training_details: bool = False
element_weights: Optional[torch.Tensor] = None
```

默认调用仍然返回原来的 scalar flow loss。AR 调用 `return_training_details=True` 后，额外得到：

```text
loss
pred_velocity
velocity_target
x_t
time
action_pred
valid_weights
```

其中：

```text
x_t = (1-t) * noise + t * actions
velocity_target = actions - noise
action_pred = x_t + (1-t) * pred_velocity
```

这样 flow loss、action reconstruction 和 delta loss 使用同一次 forward 生成的 `x_t`、`time` 和 `pred_velocity`，没有复制整份 flow head，也没有为不同 loss 重复采样。

### 6.2 总损失

实际脚本的总损失为：

```text
L_AR = 1.0 * L_flow_weighted
     + 0.5 * L_action_reconstruction
     + 0.1 * L_action_delta
```

所有项均使用有效 action mask；padding 不进入分母。

### 6.3 Flow 的维度和事件加权

代码中的维度权重：

```text
左臂 xyz [0:3]：2.0
左臂 quaternion [3:7]：1.0
左夹爪 [7]：4.0
右臂 [8:16]：1.0
padding [16:32]：由 actions_mask 排除
```

代码在线检查左夹爪相邻差分：

```python
abs(actions[...,7][:,1:] - actions[...,7][:,:-1]) > 0.1
```

然后构造时间权重：

```text
普通时刻：1.0
夹爪 transition 及其 ±4 个 token 邻域：4.0
闭合保持（g > 0.8）：至少 3.0
```

最终 flow 的 element weight 是：

```text
时间权重 × 动作维度权重 × actions_mask
```

并按有效加权元素总和重新归一化。

### 6.4 Action reconstruction 和 delta

代码实际计算：

```text
rec_xyz  = SmoothL1(action_pred[...,0:3], actions[...,0:3])
rec_grip = SmoothL1(action_pred[...,7],   actions[...,7])
L_action_reconstruction = 0.75 * rec_xyz + 0.25 * rec_grip
```

两个分量均使用时间权重和有效 mask。

左臂 xyz 的动作变化约束为：

```text
delta_pred = action_pred[:,1:,0:3] - action_pred[:,:-1,0:3]
delta_gt   = actions[:,1:,0:3]     - actions[:,:-1,0:3]
L_action_delta = SmoothL1(delta_pred, delta_gt)
```

仅在相邻两个 token 都有效时计算 delta。

### 6.5 一个需要保留的实现差异

交付方案正文中曾用概念性形式写过“左夹爪相对 xyz 的额外 2 倍权重”。本次实际运行的代码使用的是：

```text
L_action_reconstruction = 0.75 * rec_xyz + 0.25 * rec_grip
```

也就是说，本次正式运行的 reconstruction 分支不是“`rec_xyz + 2×rec_grip`”的字面实现，而是 0.75/0.25 混合。Flow 主损失中的 gripper 维度权重 4.0 仍然生效。

这是本实验审计中必须明确记录的差异：

- 若要复现本次已经启动的结果，保持当前脚本不变；
- 若要严格执行交付方案的“左夹爪额外 2 倍 reconstruction”口径，应在下一轮实验前修改并重新启动，不应把两个结果混在一起比较。

---

## 7. 新增代码和实际改动

### 7.1 新增 AR 训练入口

```text
tools/train_lap10v3_ar_ddp.py
```

主要职责：

- 初始化分布式 NCCL；
- 加载完整 Phase1Dataset；
- 从联合 checkpoint 恢复 LAP10V3 和 Expert；
- 从 Stage-1 恢复冻结 LaWM；
- 组织三视角输入；
- 计算真实 action 加权 flow/reconstruction/delta loss；
- 执行 AR-A/AR-B 阶段切换；
- 记录 loss、MAE、梯度和 LR；
- 每 500 step 保存 checkpoint。

### 7.2 Flow Head 修改

```text
starVLA/model/framework/vlas/flowmatching_expert.py
```

改动原则：

- 默认旧调用行为不变；
- 只有 AR 显式要求时才返回中间量；
- 支持 optional element weight；
- 不复制旧 flow forward；
- 推理接口 `sample_actions_cfg` 未被 AR 改写。

### 7.3 静态检查

使用 flashwam 环境执行了 Python 编译检查：

```bash
/home/pxchen/miniconda3/envs/flashwam/bin/python -m py_compile \
  tools/train_lap10v3_ar_ddp.py \
  starVLA/model/framework/vlas/flowmatching_expert.py
```

检查通过。

注意：系统默认 Python 环境没有 torch；训练和检查均应使用：

```text
/home/pxchen/miniconda3/envs/flashwam/bin/python
/home/pxchen/miniconda3/envs/flashwam/bin/torchrun
```

---

## 8. Smoke 测试记录

### 8.1 Smoke 命令

为了同时覆盖 AR-A 和 AR-B，smoke 使用了 20 step、step 11 切换阶段：

```bash
cd /data/pxchen/LaWAM
mkdir -p logs/lap10v3_ar_smoke

CUDA_VISIBLE_DEVICES=4,5 \
/home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tools/train_lap10v3_ar_ddp.py \
  --steps 20 \
  --phase-b-start 11 \
  --batch-size 1 \
  --grad-accumulation 1 \
  --log-every 1 \
  --save-every 20 \
  --no-preload-cache \
  --output-dir outputs/lap10v3_ar_smoke \
  |& tee logs/lap10v3_ar_smoke/train.log
```

### 8.2 Smoke 结果

前 10 step 为 AR-A：

```text
lap_grad = 0.000e+00
expert_grad > 0
```

第 11 step 起切换 AR-B：

```text
step=0011  lap_grad=1.332e-01  expert_grad=4.329e-01
step=0012  lap_grad=2.086e-01  expert_grad=5.595e-01
step=0020  lap_grad=6.106e-02  expert_grad=1.982e-01
```

Smoke 期间：

- `loss` 为有限数；
- `flow/recon/delta` 均为有限数；
- `xyz_mae/grip_mae` 均为有限数；
- 没有 action mask/time-grid mismatch；
- 没有 shape mismatch；
- 没有显存溢出；
- 成功保存 `outputs/lap10v3_ar_smoke` checkpoint。

这证明 DDP、数据、LAP、LaWM、Action Expert、flow 中间量和阶段切换链路都能工作。

Smoke 曾出现一次 DDP 提示：

```text
find_unused_parameters=True was specified ... did not find any unused parameters
```

这是性能提示，不是错误。它与 AR-A/AR-B 的条件式计算有关；为了允许 AR-A 注册但不反传 LAP 参数，同时保证 AR-B 安全切换，本轮保留 `find_unused_parameters=True`。

---

## 9. 正式训练启动记录

### 9.1 GPU 选择

交付方案原本建议 GPU 6、7，但启动前检查发现：

```text
GPU 6：约 19.3 GiB 已被其他进程占用
GPU 7：空闲
```

为避免杀掉或干扰他人任务，本轮选择两个当前空闲卡：

```text
GPU 4、GPU 5
```

本次没有终止任何其他用户进程。

### 9.2 正式实际命令

```bash
cd /data/pxchen/LaWAM
mkdir -p logs/lap10v3_ar_expert_task14_2000step \
         outputs/lap10v3_ar_expert_task14_2000step

setsid env CUDA_VISIBLE_DEVICES=4,5 \
/home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tools/train_lap10v3_ar_ddp.py \
  --joint-checkpoint \
    outputs/lap10v3_expert_joint_task14_2000step/lap10v3_expert_step0002000.pt \
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
  --save-every 500 \
  --log-every 10 \
  --output-dir outputs/lap10v3_ar_expert_task14_2000step \
  > logs/lap10v3_ar_expert_task14_2000step/train.log 2>&1 \
  < /dev/null &
```

最终 torchrun 父进程：

```text
PID=4189881
PPID=1
```

PPID 为 1 表示它已经脱离当前终端会话，不会因为退出本次 Codex/SSH 会话而自动结束。

### 9.3 一次无效的后台启动尝试

最初使用普通 `nohup ... &` 得到 PID `4188851`，但该进程很快退出，日志为空。没有根据这个 PID 继续判断训练成功。

随后改用 `setsid`，并显式重定向 stdin/stdout/stderr。第二次启动产生 PID `4189881`，进程 PPID 为 1，日志正常增长。这是当前有效的正式训练进程。

---

## 10. 正式训练配置

### 10.1 优化器和 batch

```text
precision：FP32
world size：2
per-GPU batch size：2
gradient accumulation：4
effective batch：2 × 2 × 4 = 16
optimizer：AdamW
weight decay：0.01
gradient clip：1.0
```

### 10.2 学习率

AR-A（step 1–1000）：

```text
Expert 初始 LR：1e-6
warmup：100 step
LAP 有效 LR：0
```

AR-B（step 1001–2000）：

```text
Expert 初始 LR：5e-7
LAP9–10/output 初始 LR：1e-6
phase-B warmup：50 step
最低 LR 比例：0.1
```

脚本通过每 step 更新两个 optimizer parameter group 的 LR 实现阶段调度，并在 checkpoint 的 `scheduler` 字段中保存当前 step/phase 元数据。

### 10.3 数据访问量

```text
总 optimizer step：2000
每 step 有效样本访问：16
总样本访问：32,000
完整 train cache：24,140
等效 epoch：32,000 / 24,140 ≈ 1.33
```

这里的“step”指 optimizer update，不是 micro-batch 数，也不是单卡 forward 次数。

---

## 11. 正式运行观测

### 11.1 启动和加载

正式启动后，两个 rank 分别加载完整 CPU cache。主要日志：

```text
[data] 24,140 samples (35.4 GiB) resident in CPU RAM; load_time=52.6s
[AR] no VLM/teacher cache; samples=24,140 world=2 FP32
[model] LAP10V3=98,592,801 AR-B trainable=19,495,681; Expert trainable=304,830,464
```

关键参数量：

| 项目 | 参数量 |
|---|---:|
| 完整 LAP10V3 | 98,592,801 |
| AR-B 实际可更新 LAP 参数 | 19,495,681 |
| Action Expert 可更新参数 | 304,830,464 |

Action Expert 的可更新参数不包括冻结、未使用的 `enc_vlm`。

### 11.2 AR-A 早期日志

正式训练早期曾观察到：

```text
step 0001: loss=0.004103, lap_grad=0, expert_grad=1.230e-01
step 0010: loss=0.006992, lap_grad=0, expert_grad=2.855e-01
step 0100: loss=0.010985, lap_grad=0, expert_grad=3.600e-01
step 0500: loss=0.017263, lap_grad=0, expert_grad=6.403e-01
step 0510: loss=0.010911, lap_grad=0, expert_grad=3.172e-01
step 0550: loss=0.003691, lap_grad=0, expert_grad=1.543e-01
```

在 20:31 状态快照（AR-A 中段）时，最新日志约为：

```text
step 0820: loss=0.006055, lap_grad=0, expert_grad=2.164e-01
step 0830: loss=0.005722, lap_grad=0, expert_grad=2.265e-01
step 0840: loss=0.005528, lap_grad=0, expert_grad=1.378e-01
step 0850: loss=0.006427, lap_grad=0, expert_grad=3.624e-01
step 0860: loss=0.026858, lap_grad=0, expert_grad=4.972e-01
```

这些是单个 log interval 的 stochastic mini-batch 结果，不能把单点 loss 当成完整 epoch 平均 loss。当前没有出现 NaN 或持续发散信号；AR-A 中 LAP 梯度始终为零，符合预期冻结边界。step 0860 的单点上升需要等窗口趋势判断，不能仅凭一个 batch 判定塌缩。

### 11.3 step 500 checkpoint

已生成：

```text
outputs/lap10v3_ar_expert_task14_2000step/lap10v3_ar_step0000500.pt
```

文件大小约：

```text
4,059,926,101 bytes ≈ 4.06 GB
```

checkpoint 内容包括：

```text
step
phase
lap10v3
expert
optimizer
scheduler
args
source_joint_checkpoint
trainable_parameter_names
```

本轮实际已生成以下正式 checkpoint：

```text
lap10v3_ar_step0001000.pt
lap10v3_ar_step0001500.pt
lap10v3_ar_step0002000.pt
```

### 11.4 当前 GPU 和进程

AR-2000 完成时观察到：

```text
GPU 4：约 9.14 GiB
GPU 5：约 9.14 GiB
```

训练进程已经正常退出，torchrun 父进程 PID `4189881` 已结束；日志最后为 step 2000/2000，未出现异常退出。

---

## 12. 监视和故障检查命令

### 12.1 实时日志

```bash
tail -f /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

只看最近 50 行：

```bash
tail -50 /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

### 12.2 查看进程是否仍在

```bash
ps -o pid,ppid,etime,stat,pcpu,pmem,cmd \
  -p 4189881 --ppid 4189881
```

### 12.3 查看 GPU

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

只查看本实验卡：

```bash
nvidia-smi -i 4,5
```

### 12.4 判断是否进入 AR-B

```bash
rg '\[AR\]\[AR-B\]' \
  /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

进入 AR-B 后应该同时看到：

```text
phase=AR-B
lap_grad > 0
expert_grad > 0
lr_lap > 0
```

### 12.5 检查异常

```bash
rg -n 'nan|NaN|inf|Inf|Traceback|CUDA out of memory|ERROR|failed' \
  /data/pxchen/LaWAM/logs/lap10v3_ar_expert_task14_2000step/train.log
```

若命令无输出，表示目前日志中没有匹配到这些关键词；仍需结合 checkpoint 和进程状态判断训练是否完整。

### 12.6 查看保存的 checkpoint

```bash
find /data/pxchen/LaWAM/outputs/lap10v3_ar_expert_task14_2000step \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

---

## 13. 训练速度和完成时间估计

正式训练在 step 100 附近耗时约 1.1 分钟，step 500 附近耗时约 5.6 分钟，step 550 附近耗时约 6.3 分钟。当前 AR-A 主要更新 Action Expert；AR-B 需要对 LAP9–10 回传梯度，单 step 可能略慢。

因此合理估计：

```text
数据加载：约 1–2 分钟
AR-A：约 10–12 分钟
AR-B：约 12–18 分钟
checkpoint 保存和余量：约 2–3 分钟
总计：约 25–35 分钟
```

该估计不是完成保证。应以日志里的 `step=2000/2000` 和最终 checkpoint 是否存在为准。

---

## 14. 结果判断口径

本轮训练结束后不能只看最后一个 loss。至少应记录：

1. `step 500`、`step 1000`、`step 1500`、`step 2000` 的 checkpoint；
2. AR-A 和 AR-B 的 flow/reconstruction/delta 曲线；
3. LAP grad 和 Expert grad 的阶段变化；
4. 左臂 xyz MAE；
5. 左夹爪 MAE；
6. clean/randomized 的 RoboTwin 闭环成功率；
7. 与此前 4/10 基线使用相同任务、相同 seed 和相同 5+5 口径比较。

推荐后续验证最少保持：

```text
同一任务
5 条 clean
5 条 randomized
相同推理配置
相同 action horizon 和控制频率
```

如果 loss 下降但成功率不升，需要进一步检查：

- Action Expert 推理时是否确实接收 `h_lap=cond_lap`；
- 推理时的三视角顺序是否仍为主视角、左腕、右腕；
- state 是否进入 LAP10V3 而不是被置零；
- action 归一化和 gripper 符号是否与训练 cache 一致；
- 36 个有效 action token 是否被正确截取；
- 采样时是否仍然误用旧 VLM condition；
- 训练 checkpoint 是否加载了完整 `lap10v3` 和 `expert`，而不是只加载其中一支。

---

## 15. 已知限制和不得混淆的事项

### 15.1 本轮不是从零预训练

AR 使用的是已有联合 checkpoint 作为初始化：

```text
lap10v3_expert_step0002000.pt
```

所以本轮属于“真实 action 监督下的再对齐/继续训练”，不是从随机 LAP10V3 和随机 Action Expert 开始。

### 15.2 本轮不是完整解冻

AR-A 只更新 Expert。AR-B 只更新 LAP9–10/output head 和 Expert。LAP6、LAP7–8、embedding/projection 和 LaWM 都没有解冻。

### 15.3 `teacher_position_mean` 不代表当前 teacher

它是旧 checkpoint 的固定输出模板。它会参与 `cond_lap = teacher_position_mean + residual` 的前向表达，但不依赖当前 VLM，不读取旧 teacher cache，不参与 loss，不更新。

### 15.4 单点 loss 波动是预期现象

日志每 10 optimizer step 打印一次，但每次日志是当前累积 batch 的统计，不是整个 train split 的严格 epoch 平均。Flow matching 本身每次会采样 noise/time，真实轨迹的动作阶段也不同，因此单点 loss 有波动不等于模型塌缩。

应观察窗口平均、MAE、梯度、学习率和闭环成功率的联合变化。

### 15.5 工作树存在其他历史修改

本仓库在本次会话开始前已经存在很多与 LaWAM、Robotwin、部署和训练相关的用户修改。没有执行 `git reset`、`git checkout` 或其他破坏性清理。当前文档和 AR 脚本是本次会话新增内容；其他 dirty worktree 修改不应被归因于本轮 AR 实验。

---

## 16. 复现实验清单

要重新复现本次**实际运行**，至少需要满足：

```text
[ ] 工作目录为 /data/pxchen/LaWAM
[ ] flashwam 环境可用
[ ] 主视角 cache 存在且长度为 24,140
[ ] 左右腕 cache 存在且与主视角逐样本对齐
[ ] action cache 存在且长度为 24,140
[ ] joint checkpoint step0002000 存在
[ ] Stage-1 checkpoint step0003000 存在
[ ] lam release checkpoint/yaml 存在
[ ] 使用 FP32
[ ] 使用双卡 DDP
[ ] batch-size=2
[ ] grad-accumulation=4
[ ] phase-b-start=1001
[ ] 没有传入 teacher-cache 参数
[ ] 没有启动 Qwen/VLM
[ ] 正确设置 CUDA_VISIBLE_DEVICES
[ ] 保留相同的 loss 权重和 action mask 口径
```

建议先用 smoke 命令验证：

```text
AR-A：lap_grad=0，expert_grad>0
AR-B：lap_grad>0，expert_grad>0
```

只有 smoke 通过后，才启动正式 2000 step。

---

## 17. 当前结论

截至本记录：

- AR 训练代码链路已经实现并通过双卡 smoke；
- 正式训练已经启动且脱离终端运行；
- 数据量和张量形状已经核实；
- 本轮没有加载或使用 VLM teacher；
- AR-A 梯度边界符合设计；
- step 500 checkpoint 已保存；
- AR-2000 训练已经完成并保存 step 2000 checkpoint；
- 最终闭环成功率尚未得到，不能提前宣称 AR-2000 已经改善 RoboTwin 成功率；
- 正式结论仍需同口径 RoboTwin 5 clean + 5 randomized 测试。

本文件应与以下文件一起保存：

```text
docs/LAP10V3_AR_2000STEP_LUNA_HANDOFF_ZH.md
docs/LAP10V3_AR_2000STEP_SESSION_RECORD_ZH.md
tools/train_lap10v3_ar_ddp.py
logs/lap10v3_ar_expert_task14_2000step/train.log
outputs/lap10v3_ar_expert_task14_2000step/
```

---

## 18. 全部实验结果总账本

本记录之外的前序训练、失败对照、官方 VLM 对照、LAP8 闭环、LAP10V3 joint 4/10，以及早期 A/B/C 结果，统一汇总在：

[LAP10V3_EXPERIMENT_LEDGER_ZH.md](LAP10V3_EXPERIMENT_LEDGER_ZH.md)

该账本明确区分工程成功、离线学习成功和闭环任务成功，并记录了早期 A/B/C 结果与后续独立 no-VLM 0/10 结果之间需要复核的口径差异。
### 11.5 AR-2000 最终训练结果

最终日志：

~~~text
step=2000/2000
phase=AR-B
loss=0.013269
flow=0.011800
recon=0.002929
delta=0.000049
xyz_mae=0.012205
grip_mae=0.044784
samples=32000
elapsed=23.8m
~~~

AR-A 中 LAP 梯度为零，AR-B 中 LAP9–10/output 与 Expert 都有梯度。训练工程结果为成功；Robotwin 闭环结果仍待测。
