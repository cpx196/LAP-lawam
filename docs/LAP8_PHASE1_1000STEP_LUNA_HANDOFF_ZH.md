# LAP8 Phase 1（1000 Step）训练交付方案

## 1. 本阶段目标

本阶段训练一个不依赖 VLM 的专用任务语义接口，使 LAP8 仅根据三个当前
视角和当前双臂 EEF 状态，为官方 Action Expert 生成动作条件 token。

任务固定为：

```text
move_pillbottle_pad
```

Phase 1 是接口预热，不追求最终成功率。完成标准是：

1. 无需加载 Qwen/VLM 即可完成端到端 flow-matching 前向和反向；
2. LAP8 新增分支获得稳定、非零梯度；
3. flow loss 有下降趋势，8 个条件 token 不坍缩；
4. 为下一阶段 LAP8、LaWM、Action Expert 联合微调提供非随机初始化。

本方案中的 Phase 1 不等同于先前的 Stage 1：

```text
已完成 Stage 1：LAP6 + LaWM 联合训练
当前 Phase 1：只训练 LAP8 新增的 Expert 条件分支
后续 Phase 2：LAP8 + LaWM + Action Expert 联合微调
```

## 2. 重要接口说明

官方 Action Expert 原来接收 Qwen 输出的完整：

```text
h_vlm [B,L,2048] -> enc_vlm -> [B,L,768]
```

原模型虽然在 VLM 序列内插入了 8 个 flow query，但送入 Expert 的仍是完整
VLM hidden sequence。本方案并非逐 token 复刻原始接口，而是主动使用一个
单任务信息瓶颈：

```text
cond_lap [B,8,768]
```

这样做的依据是：

- 任务语言固定，不需要重复编码长文本；
- 当前视觉和预测子目标已分别通过 `h_t`、`h_t1` 提供给 Expert；
- 8 个 LAP token 专门描述动作语义；
- Expert cross-attention 支持可变长度条件，无须与原 VLM 序列长度对齐。

## 3. 输入、监督与输出

### 3.1 LAP8 推理输入

```text
主视角当前帧 DINO token      [B,256,768]
左腕当前帧 DINO token        [B,256,768]
右腕当前帧 DINO token        [B,256,768]
当前双臂 EEF state           [B,16]
```

三个视角堆叠为：

```text
lap_visual [B,3,256,768]
```

三个视角分别来自 RoboTwin：

```text
video.cam_high
video.cam_left_wrist
video.cam_right_wrist
```

### 3.2 训练监督

```text
actions       [B,50,32] FP32
actions_mask  [B,50,32] bool
```

官方配置为 30 Hz、`horizon_sec=1.2`，所以前 36 个时间步有效。RoboTwin
双臂 EEF action 为 16 维，故前 16 个动作维度有效；其余为 padding。

有效 action 结构：

```text
left xyz + left quaternion + left gripper
+ right xyz + right quaternion + right gripper
= 16 dimensions
```

位置和夹爪使用官方 RoboTwin SFT 的 min-max statistics；四元数保持原值；
左右夹爪沿用官方 invert 语义。

### 3.3 LAP8 输出

```text
z_lap      [B,1,32]   -> LaWM
cond_lap   [B,8,768]  -> Action Expert
```

## 4. 模型结构和参数状态

```text
三视角 DINO + 当前 EEF
             |
         LAP6（冻结、eval）
       /                       \
z_lap [B,1,32]          scene_6 [B,8,768]
       |                       |
LaWM（冻结、eval）       + task embedding
       |                 + Linear(z_lap)
h_t1 [B,256,768]               |
                            Block 7
                               |
                            Block 8
                               |
                       Linear + LayerNorm
                               |
                       cond_lap [B,8,768]
                               |
       h_t + h_t1 + cond_lap --+
                               |
                    Action Expert（冻结、eval）
                               |
                    flow-matching velocity loss
```

LAP8 的实测参数量：

| 模块 | 参数量 | Phase 1 状态 |
|---|---:|---|
| LAP6 | 59,715,104 | 冻结 |
| Block 7～8、映射和任务参数 | 19,521,792 | 训练 |
| LAP8 合计 | 79,236,896 | 其中 19.52M 可训练 |
| LaWM | 228,286,720 | 冻结 |
| Action Expert | 约 307M | 冻结 |
| DINO | 使用离线缓存，不分配 | 冻结 |
| Qwen/VLM | 不加载 | 不参与 |

冻结 Action Expert 参数时，不能用 `torch.no_grad()` 包裹 Expert 前向。
必须保留 Expert 输出到 `cond_lap` 的输入梯度：

```text
flow loss
  -> frozen Action Expert computation graph
  -> cond_lap
  -> lap_to_expert
  -> Block 8
  -> Block 7
  -> latent_to_expert / task_embedding
```

LAP6、LaWM 和 Expert 参数梯度应为零；LAP8 新增分支梯度应非零。

## 5. 初始化和现有资产

### 5.1 Stage 1 初始化

```text
outputs/lap_stage1_task14_3view_joint_3000step/
stage1_phase2_step0003000.pt
```

从该 checkpoint 加载：

```text
lap             -> LAP6
lawm_decoder    -> LaWM
```

### 5.2 官方 Action Expert

源 checkpoint：

```text
results/Checkpoints/robotwin/lawam_robotwin_sft_release/
final_model/pytorch_model.pt
```

已经提取成不包含 Qwen 的独立权重：

```text
cache/lap8_phase1_official_action_expert.pt
```

该文件约 1.14 GiB，包含 `policy_backend.flow.*`、flow config 和
`action_horizon=50`。

### 5.3 特征和动作缓存

```text
主视角及 Stage 1 张量：cache/lap_stage1_task14
左右腕视角：           cache/lap_stage1_task14_wrist
动作序列：             cache/lap8_phase1_task14_actions
```

动作缓存已按 Stage 1 manifest 对齐生成：train/val/test 共 217 个 shard。

训练集：

```text
480 episodes
24,140 anchor samples
```

## 6. 损失和正则化

主要损失使用官方 Conditional Flow Matching：

```text
z ~ N(0,I)
x_tau = (1-tau)z + tau * action
target_velocity = action - z
L_flow = masked_MSE(pred_velocity, target_velocity)
```

总损失：

```text
L_total = L_flow + 0.01 * L_diversity
```

`L_diversity` 约束 Block 8 输出的 8 个 token，降低全部 token 坍缩为相同
表示的风险。

正则化：

```text
腕部 view dropout：          0.2
semantic-condition dropout：0.0
weight decay：               0.05
global gradient clip：       1.0
```

Phase 1 必须令 semantic-condition dropout 为 0。因为 Expert 冻结时，如果
把完整 `cond_lap` 置零，该 micro-step 对 LAP8 不产生有效梯度。

## 7. 1000-Step 双卡训练配置

```text
GPU：physical 6,7
strategy：2-card DDP
precision：FP32（不使用 autocast / GradScaler）
per-GPU batch size：1
gradient accumulation：8
global effective batch：2 * 1 * 8 = 16

steps：1000
样本使用次数：1000 * 16 = 16,000
等效 epoch：16,000 / 24,140 = 0.663

optimizer：AdamW
learning rate：1.0e-4
warmup：200 steps
scheduler：cosine decay
minimum LR：1.0e-5
weight decay：0.05
gradient clip：1.0
view dropout：0.2
condition dropout：0.0
diversity weight：0.01
```

1000 step 只覆盖约 0.66 个 epoch，定位为 Phase 2 前的短接口预热。若 loss
尚未明显下降，也不在 Phase 1 盲目延长，而是在检查仿真表现后决定是否直接
进入全量联合微调。

保存 checkpoint：

```text
step 250
step 500
step 750
step 1000
```

## 8. Luna 正式启动命令

工作目录：

```bash
cd /data/pxchen/LaWAM
```

启动前确认：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tail -2
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
NO_ALBUMENTATIONS_UPDATE=1 \
/home/pxchen/miniconda3/envs/flashwam/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  tools/train_lap8_phase1.py \
  --mode train \
  --steps 1000 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --preload-cache \
  --lr 1e-4 \
  --warmup-steps 200 \
  --view-dropout 0.2 \
  --condition-dropout 0 \
  --diversity-weight 0.01 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --log-every 10 \
  --save-every 250 \
  --output-dir outputs/lap8_phase1_task14_1000step
```

建议由 Luna 使用 `setsid`、`tmux` 或其他现有任务管理方式保存 stdout/stderr，
不要依赖断开后会结束的前台 shell。

## 9. 训练监控和验收

日志至少监控：

```text
loss
flow
div
grad_before_clip
grad_after_clip
cond_cos
learning rate
step time / ETA
```

链路验收条件：

1. `loss`、`flow` 为有限值，无 NaN/Inf；
2. 新增 LAP8 分支 grad 持续非零；
3. LAP6、LaWM、Action Expert 不在 optimizer 中；
4. `cond_lap` 为 `[B,8,768]`；
5. `h_t`、`h_t1` 均为 `[B,256,768]`；
6. action/mask 均为 `[B,50,32]`，每个样本有 `36*16=576` 个有效值；
7. 训练进程中没有 Qwen/VLM 权重分配；
8. step 250、500、750、1000 checkpoint 均可读取。

成功率不作为单独的 Phase 1 训练停止条件。建议分别用 step 500 和 step 1000
在 RoboTwin clean/randomized 上做小规模对比，再决定 Phase 2 的初始化点。

## 10. 代码位置

```text
LAP8：
starVLA/model/lap_stage2.py

Action Expert 的原生 h_lap 接口：
starVLA/model/framework/vlas/flowmatching_expert.py

缓存、权重提取和训练入口：
tools/train_lap8_phase1.py
```

## 11. Phase 2 边界

本文件不启动 Phase 2。建议后续“全量微调”定义为：

```text
LAP6 + Block 7～8 + LaWM + Action Expert：解冻
DINO：继续冻结并复用缓存
VLM：不存在
```

Phase 2 需要在 flow loss 外保留 Stage 1 的 latent/world consistency loss，
防止 LAP6 和 LaWM 在 action loss 下发生灾难性漂移。

## 12. 交付前 Smoke 验证结果

执行日期：2026-08-05。正式 1000-step 训练未启动。

### 12.1 单卡端到端 Smoke

配置：GPU 7、FP32、batch 1、gradient accumulation 1、2 optimizer steps，
不预加载缓存。

结果：

```text
LAP8 total:       79,236,896
LAP8 trainable:   19,521,792
LaWM frozen:     228,286,720
Expert frozen:   306,405,632
VLM:             not-loaded

step 1:
loss=0.288910 flow=0.286399 div=0.251061 grad=2.0588

step 2:
loss=0.296101 flow=0.293279 div=0.282197 grad=8.9642
```

两步 loss 均为有限值，LAP8 梯度非零，checkpoint 成功保存。两步样本不同且
flow time/noise 随机，因此不可用两点 loss 的升降判断收敛。

### 12.2 双卡 DDP Smoke

首次测试发现项目导入链会提前创建 distributed process group，训练入口再次
初始化会报：

```text
ValueError: trying to initialize the default process group twice
```

训练入口已改为幂等初始化：仅在 `dist.is_initialized()==False` 时调用
`init_process_group`。

修复后使用 GPU 6、7、FP32、每卡 batch 1、gradient accumulation 2，完成
1 个同步 optimizer step：

```text
world_size=2
effective_batch=4
loss=0.320549
flow=0.318155
div=0.239433
grad_before_clip=13.4455
step_time=1.06s
checkpoint saved
```

该测试确认：

1. 两卡 torchrun 启动和 NCCL 同步正常；
2. 梯度累积及最后一个 micro-step 的 DDP 同步正常；
3. 无 VLM 的 `h_lap [B,8,768]` Expert 接口正常；
4. LAP8 -> LaWM -> frozen Expert -> flow loss -> LAP8 的反向链路正常；
5. checkpoint 保存正常。

正式配置每步有 8 个 micro-step。按 smoke 粗略线性估计，1000 step 的纯计算
时间约 70 分钟；加上双 rank 大缓存预加载和四次 checkpoint 保存，建议按
约 1.3～1.8 小时预留。实际 ETA 以正式训练第 10～20 step 的日志为准。

## 13. 正式 1000-Step 完成记录

执行日期：2026-08-05。

正式命令已在 GPU 6、7 上完成 `1000/1000` step。训练过程中没有加载 VLM，
两个 DDP worker 正常完成梯度同步和 checkpoint 保存。

最终文件：

```text
outputs/lap8_phase1_task14_1000step/lap8_phase1_step0001000.pt
```

训练日志：

```text
logs/lap8_phase1_task14_1000step/train.log
```

最终记录：

```text
step=1000/1000
loss=0.017117
flow=0.014958
div=0.215905
grad_before_clip=0.0696
grad_after_clip=0.0696
lr=1.000e-05
```

关键过程点：

```text
step 1：    flow=0.188891
step 100：  flow=0.067842
step 250：  flow=0.014440，checkpoint 已保存
step 500：  flow=0.011319，checkpoint 已保存
step 750：  flow=0.030501，checkpoint 已保存
step 1000： flow=0.014958，最终 checkpoint 已保存
```

loss 在后半程保持有限值并持续出现较低 flow loss，梯度从 warmup 初期的
clip 区域下降到约 `0.05～0.15`，未出现 NaN/Inf 或进程异常退出。最终
checkpoint 文件大小约 452 MiB，可直接作为后续 Phase 2 的 LAP8 初始化。
从进程启动到最终 checkpoint 保存的实际墙钟时间约 15 分钟，其中主缓存双
rank 预加载约 172 秒；正式计算阶段单 step 约 0.5～0.7 秒。
