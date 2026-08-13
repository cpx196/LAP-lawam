# LaWAM / SEC284 项目 Handoff（2026-08-12 / 01）

> 记录时间：2026-08-12 00:16 HKT  
> 本地仓库：`/data/pxchen/LaWAM`  
> 云端仓库：`cpx196/LAP-lawam`  
> Git 快照：以仓库 `main` 最新提交为准（本次更新将包含 2000-step 训练及 10 个 clean 测评结果）。
> 当前主题：LAP6 + SEC284-L 替代在线 VLM condition，并通过冻结 Action Expert 的行为监督改善闭环效果。

## 1. 当前状态摘要

本轮已经完成 SEC284-L 的模型实现、全量 VLM condition teacher cache、纯表征蒸馏、冻结 Action Expert 的 behavior KD、LAP6 无 VLM 闭环测试、固定 instruction 的真实 VLM shadow trace，以及全量 10-step Expert inference-grid cache。

此前已完成一轮 1000-step inference-grid KD；当前 Expert grid-KD 续训和部署测评也已完成：

- 2000-step Expert grid-KD 输出：`outputs/sec284_expert_grid_kd_2000step`
- 最终日志：`outputs/sec284_expert_grid_kd_2000step/train.log`
- 部署 checkpoint：`step-001000_deploy.pt`、`step-001500_deploy.pt`、`step-002000_deploy.pt`
- 当前经验上的部署 best：500-step checkpoint（需以固定离线验证进一步确认）

但当前训练曲线**不能直接证明已经收敛，也不能直接证明没有学习**。主要原因是每个 batch 随机抽一个 denoising grid step，日志中的 `grid_kd` 混合了不同难度的 k；同时当前 total loss 被 representation loss 主导，grid 项只占很小的标量比例。此外，当前 grid trainer 没有沿用旧 behavior-KD trainer 中的显式 dynamic loss，这是一个需要优先核查的实现差异。

另一个关键纠正是：SEC284 clean episode 并不是“完全没抓住”。从视频看，它能够找到瓶子、闭合夹爪、接触并短暂操控瓶子，之后动作轨迹不稳定，将瓶子碰倒/打翻。因此不能再把 clean 失败简单归因于 gripper，也不能只凭 gripper 误差较大就断言它是因果瓶颈。

## 2. 已确定的项目边界

### 2.1 只使用 LAP6

后续不再使用 LAP8 相关输出或 LAP8 作为基线模块。此前历史排查曾短暂接到 LAP8，但用户已明确要求改回 LAP6。

当前目标组合是：

```text
三视角观测
  -> 冻结 DINO latent
  -> LAP6 + LaWM 视觉未来分支
  -> SEC284-L 语义 condition 分支（不调用在线 VLM）
  -> 冻结 Action Expert
  -> action chunk
```

真实 VLM 只作为 teacher、诊断基线和离线缓存构建工具，不作为最终部署时的在线模块。

### 2.2 SEC284 暂不输入 action / state / EEF

SEC284-L 当前输入只有三视角 DINO latent，不接收：

- action；
- EEF；
- proprio/state；
- LAP6 输出；
- LaWM 输出；
- 运行时自由文本 instruction。

Action Expert 原本就有独立的 state/proprio 通道，因此“SEC284 不输入 state”不代表整个策略完全没有机器人状态。

### 2.3 固定单任务和固定语义

当前只针对任务：

```text
move_pillbottle_pad
```

固定详细 instruction：

```text
Use the left arm to pick and place the orange bottle for pills or liquid onto the pad.
```

SEC284 内部使用 learned queries 表达固定任务语义。这样做是有意限制泛化范围：当前约 77M 的模型不要求理解任意自然语言，也不要求跨任务泛化。

RoboTwin 运行时可能生成随机 instruction；这与当前 SEC284 的固定语义契约不一致。因此 shadow teacher 必须强制使用上述固定 instruction，原始环境 instruction 只能保存为元数据，不能拿来与 SEC284 condition 做直接一一比较。

## 3. SEC284-L 设计

### 3.1 输入输出契约

输入：

```text
DINO latent: [B, 3, 256, 768]
```

三个视角分别是场景主视角和两个腕部视角。模型直接消费冻结 DINO 特征，不在 SEC284 内重新编码原始 RGB。

输出：

```text
semantic condition: [B, 284, 768]
```

这个输出对齐真实 VLM 投影后的 hidden-state/condition，供 Action Expert 使用。它不是自然语言 token，也不是 action。

### 3.2 模型规模

- 层数：8
- hidden width：768
- attention heads：12
- FFN width：3072
- 参数量：约 76,624,896，即约 76.6M / 77M
- 项目统一名称：`SEC284-L`
- 不再使用 `bridge` 作为模块名称

主要实现：

```text
starVLA/model/sec284.py
```

### 3.3 模型能力边界

当前 SEC284-L 的目标不是成为通用 VLM，而是对固定任务、固定语义，把三视角视觉 latent 映射成 Action Expert 可消费的 condition。

因此评价重点应是：

1. condition 是否保留 teacher 的样本间动态变化；
2. 冻结 Action Expert 是否对 SEC condition 产生接近真实 VLM 的速度场；
3. 最终闭环 episode 是否成功。

仅看 token MSE 或 cosine 不足以判断部署效果。

## 4. 数据和缓存

### 4.1 全量对齐数据

当前不是只有少量 episode 或 9 个 trace 点。已有完整离线训练集：

| Split | 样本数 |
|---|---:|
| train | 24,140 |
| val | 1,749 |
| test | 1,749 |

主要缓存：

| 内容 | 路径 | 约占空间 |
|---|---|---:|
| 主视角/基础特征 | `cache/lap_stage1_task14` | 41G |
| wrist 特征 | `cache/lap_stage1_task14_wrist` | 41G |
| VLM condition teacher | `cache/sec284_task14_teacher` | 12G |
| GT actions | `cache/lap8_phase1_task14_actions` | 212M |

注意：actions 缓存路径保留了历史 `lap8` 命名，但这不表示当前模型使用 LAP8 模块。接手者不要仅根据目录名误判模型接线。

### 4.2 VLM condition teacher cache

VLM teacher 输入是真实三视角帧加固定 instruction，离线输出投影后的 condition：

```text
[284, 768]
```

teacher cache 已核对 train/val/test 数量，与数据集一致。

相关工具：

```text
tools/build_sec284_teacher_cache.py
tools/build_sec284_teacher_stats.py
tools/sec284_data.py
```

### 4.3 10-step Expert inference-grid cache

缓存路径：

```text
cache/sec284_task14_inference_grid/train
```

状态：

- 96 个 shards；
- 24,140 个唯一训练样本；
- index 0–24139；
- 无缺失；
- 无重复；
- 约 1.5G。

每个样本保存：

```text
teacher_x_inputs:   [10, 50, 32]
teacher_velocities: [10, 50, 32]
```

含义：真实 VLM condition 下，Action Expert 在官方 10-step flow inference 轨迹的每个离散 denoising 位置，其输入状态 `x_k` 以及 teacher velocity `v_k`。

它不是“让 gripper 单独学习”的缓存，也不是额外收集的 10 个 episode。它是在全量 24,140 个离线样本上，把已有 observation/state/action 对齐数据送入 teacher Expert，缓存其整个 10-step 推理网格。

构建工具：

```text
tools/build_sec284_inference_grid_cache_ddp.py
```

构建时已经修复过两个问题：

1. 初版 shard 维度错误地按 `[step, batch, ...]` 保存，已改为 `[batch, step, ...]`；
2. Accelerate `PartialState` 与手动 DDP 重复初始化，已加 `not dist.is_initialized()` 防护。

## 5. 已完成训练

### 5.1 纯 representation distillation

checkpoint：

```text
outputs/sec284_l_bs32_3000step/step-003000.pt
```

配置：

- 3000 steps；
- local batch size 32；
- 4 GPU；
- 目标是拟合真实 VLM condition。

这轮说明 SEC284 能较好拟合 teacher 的平均 token 表征，但仅凭 representation loss 不能保证冻结 Action Expert 的行为等价。

### 5.2 Frozen Expert behavior KD

checkpoint：

```text
outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt
```

日志：

```text
outputs/sec284_frozen_expert_behavior_kd_2000step/train.log
```

配置：

- 2000 steps；
- local batch size 32；
- GPU 1、3、4、5；
- 只有 SEC284 可训练；
- LAP6、LaWM、Action Expert 全部冻结。

旧版主要损失：

```text
L_total = L_repr + lambda_dynamic * L_dynamic + lambda_behavior * L_behavior
```

其中：

- `L_repr`：condition 的 whitened MSE、raw MSE、cosine 对齐；
- `L_dynamic`：显式约束样本间动态变化，避免只拟合均值或发生 condition collapse；
- `L_behavior`：teacher/student 在同一个随机 flow time、同一个插值状态和同一 RNG 条件下，匹配冻结 Action Expert 输出的 velocity；
- official rollout flow loss 在这轮主要用于监控，不等价于直接优化完整 10-step inference trajectory。

训练末段参考指标：

- representation loss 约 0.0620；
- raw condition loss 约 0.0507；
- behavior KD 约 0.00295；
- std ratio 约 0.924。

这些数值必须结合固定验证集和闭环结果解释，不能单独宣称成功。

## 6. 闭环与诊断结果

### 6.1 LAP6 + SEC284，无在线 VLM

使用 behavior-KD step-2000 checkpoint：

```text
results/eval_runs/sec284_behavior_kd_step2000_lap6_no_vlm_paired_seed0_1x
```

结果：

| 场景 | 成功率 | rollout 长度 |
|---|---:|---:|
| clean | 0/1 | 400 steps |
| randomized | 0/1 | 400 steps |

一加一测试不能估计统计成功率，但足以验证接线，并暴露具体失败模式。

### 6.2 同 seed 的 LAP6 + 真实 VLM

```text
results/eval_runs/sec284_lap6_paired_seed0_1x
```

结果：

| 场景 | 成功率 |
|---|---:|
| clean | 1/1 |
| randomized | 1/1 |

这说明同一 Action Expert 和 LAP6 主体在对应 seed 上本来具有完成任务的能力，SEC284 condition 替换仍是主要差异来源之一。

### 6.3 对 clean 失败模式的关键纠正

此前曾根据数值 trace 把问题概括成“gripper 没抓住”。查看实际图片/视频后，这个说法不准确：

1. 机器人能够找到瓶子；
2. 夹爪发生闭合；
3. 确实与瓶子接触并短暂操控；
4. 后续轨迹不稳定，把瓶子碰倒/打翻；
5. 最终未完成放置。

因此 clean episode 的核心问题更像**抓取后的轨迹稳定性、姿态控制或阶段切换误差**，而不是单纯的抓取触发失败。

randomized episode 更接近定位/接触失败，但只有一个样本，不能过度归纳。

结论：

- gripper error 与失败相关，不代表它是唯一因果因素；
- 手工给 gripper 4 倍权重只是启发式，目前证据不足；
- 后续必须按任务阶段拆分误差：接近、闭合、提起、搬运、放置；
- 应同时检查 xyz、旋转、gripper，不应只盯 gripper sign。

### 6.4 固定 instruction 的真实 VLM shadow trace

真实 VLM 控制环境，SEC284 只在 shadow 分支计算，不影响真实 rollout。server 强制 teacher 使用 SEC284 的固定详细 instruction，同时保存环境随机 instruction 作为 metadata。

运行结果：

```text
results/eval_runs/sec284_step2000_shadow_trace_fixed_seed0_1x
```

- clean：1/1，5 次 replan；
- randomized：1/1，4 次 replan；
- 总计 9 个 trace points。

这 9 个点是在线成功轨迹诊断，不是全量训练数据。

聚合指标：

| 场景 | condition MSE | cosine | action MSE | grid velocity MSE | gripper sign |
|---|---:|---:|---:|---:|---:|
| clean | 0.116944 | 0.915296 | 0.011049 | 0.034429 | 0.941667 |
| randomized | 0.129082 | 0.906149 | 0.011025 | 0.031110 | 0.944444 |

worst grid call：

- clean call 4：0.137789；
- randomized call 3：0.113340。

输出：

```text
results/eval_runs/sec284_step2000_shadow_trace_fixed_seed0_1x/shadow_trace_metrics.png
results/eval_runs/sec284_step2000_shadow_trace_fixed_seed0_1x/shadow_trace_metrics.json
```

实现：

```text
deployment/model_server/server_policy_lap6_sec284_shadow_trace.py
tools/analyze_sec284_shadow_traces.py
```

trace 提示 velocity error 会在 10-step denoising 后段增大，但由于 clean 实际已经抓到又碰倒，不能再把某一维的峰值直接解释成单一因果瓶颈。

## 7. 当前 inference-grid KD 训练

### 7.1 目标

旧 `L_behavior` 只在随机 flow time / 插值状态匹配 Expert velocity。当前 grid KD 改为从真实 teacher 10-step inference trajectory 中取 `x_k`，要求 student condition 在同一个 `x_k` 和对应 flow step 上复现 teacher velocity。

冻结项：

- DINO；
- LAP6；
- LaWM；
- Action Expert；
- 真实 VLM teacher。

可训练项：

- SEC284-L。

当前实现的损失：

```text
L_total = L_repr + lambda_grid * L_grid
```

`L_repr` 目前调用基础 `sec284_distillation_loss`，包含：

- whitened condition MSE；
- raw condition MSE；
- cosine loss。

`L_grid` 是 teacher/student velocity MSE，当前手工维度权重：

- 普通维度：1x；
- xyz 维度 `[0, 1, 2, 8, 9, 10]`：2x；
- gripper 维度 `[7, 15]`：4x。

`lambda_grid` 不是固定常数，而是根据 condition 分支和 grid 分支对 SEC 参数产生的 gradient norm，使用目标比例 0.25 和 EMA 自适应。训练中常见值约 0.03–0.045。

### 7.2 单 grid-step 等价优化

最初每个 batch 对 10 个 step 全部做可微 Expert forward，local batch 32 时 OOM。第一步 PyTorch peak 约 12.20 GiB，同时某张卡还有外部进程占用约 9.62G，第二步失败。

现实现每个 batch 随机选择一个缓存 grid index `k`，只执行一次 Expert forward：

```text
num_inference_steps = 1
flow_total_steps = 10
flow_step_offset = k
forced_x_inputs = teacher_x[k:k+1]
```

已经对 k=0、4、9 与完整 10-step 路径做过数值等价验证：

```text
max diff = 0
mean diff = 0
```

这不是近似修改 flow 定义，而是在 teacher `x_k` 已离线缓存的前提下，只反向当前抽中的那一个 velocity evaluation，从而把峰值显存降到约 9.18 GiB / rank。

相关实现：

```text
tools/train_sec284_full_inference_grid_kd_ddp.py
tools/run_sec284_full_inference_grid_kd_1000step.sh
starVLA/model/framework/vlas/flowmatching_expert.py
```

`flowmatching_expert.py` 增加了：

- 可选 trace；
- `forced_x_inputs`；
- 可微训练 wrapper；
- `flow_total_steps`；
- `flow_step_offset`。

### 7.3 2026-08-12 00:16 的历史快照（已完成）

```text
tmux session: sec284_grid_1000
GPU: 3,4,5,6
local batch: 32
global batch: 128
steps: 1000
save interval: 250
LR: 3e-6 -> 3e-7
warmup: 50 steps
```

2026-08-12 00:16 HKT 快照：

```text
step=410/1000
total=0.048827
repr=0.048687
grid_kd=0.003770
lambda_grid=0.03734
condition_grad_cos=0.0084
sec_grad=0.0640
lr=2.151e-6
step_time=18.55s
peak_cuda_rank0=9.18GiB
```

注意：这是当时的活动训练快照，已被后续 output-primary 2000-step 和 Expert-only 2000-step 结果 supersede；保留原值用于审计，不应再作为当前进度。

### 7.4 当前不能直接从 loss 得出的结论

从 step 1 到 410，`grid_kd` 大致在 0.0026–0.0046 之间震荡，`repr` 大致在 0.048–0.049。曲线没有肉眼明显的单调收敛趋势，但现在还不能直接断言训练崩溃，原因包括：

1. 不同 log point 抽中的 `k` 不同，各 grid step 难度不同；
2. 日志打印的 `grid_step` 只是最后一个 batch 的 k，并不能表示整个 10-step 窗口的组成；
3. total 几乎由 `L_repr` 主导；按当前数值，`lambda_grid * L_grid` 约 0.00014，只占 total 的约 0.3%；
4. 缺少固定 held-out batch、按 k 拆分的 validation；
5. train loss 还混合了数据样本难度和随机 k 的变化。

所以当前最重要的不是继续肉眼看混合 train loss，而是建立固定验证。

### 7.5 当前实现风险

当前 grid trainer 与旧 frozen-Expert behavior trainer 有一个关键差异：

```text
当前 L_repr 没有显式加入旧版 L_dynamic。
```

虽然基础 representation loss 有 whitened/raw/cosine，但这不能自动证明样本间 dynamic variance 会保持。旧训练专门监控过 dynamic R² 和 std ratio；当前 run 若不做同一套验证，可能在 grid 行为损失改善时损伤 condition 的动态变化。

另一个风险是手工动作维度权重：clean 视频已经否定“gripper 完全没抓住”的简单解释，因此 gripper 4x 的依据不充分。更稳妥的替代方案是按 teacher 每个 action/velocity 维度的统计方差做归一化，必要时再加入一个较小的 gripper sign 辅助项，而不是直接把 gripper MSE 放大 4 倍。

## 8. 建议的下一步

### 8.1 先评估，不先凭曲线改方向

当时建议对以下两个 checkpoint 使用完全相同的固定 held-out 样本；该建议仍适用于当前 behavior-KD、output-primary 和 500-step Expert 候选：

```text
baseline:
outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt

grid KD:
outputs/sec284_inference_grid_kd_1000step/step-000250.pt
```

后续应加入 output-primary 2000 和 Expert-only 500/1000/1500/2000。

验证至少输出：

1. 每个 grid step k=0..9 的 velocity MSE；
2. xyz、rotation、gripper 分组误差；
3. condition raw MSE、cosine；
4. dynamic R²；
5. std ratio；
6. 同一固定验证集的均值和方差；
7. 不同任务阶段或动作相位的误差。

只有固定验证显示 inference-grid error 相对 baseline 确实下降，且 dynamic 指标没有退化，才值得做下一轮 1+1 闭环。

### 8.2 修改 loss 的建议顺序

如果固定验证确认当前 grid KD 无明显改善：

1. 恢复显式 `L_dynamic`，防止 SEC condition 的样本间变化被 representation 均值主导；
2. 把手工 `xyz=2x / gripper=4x` 改成按 teacher 维度方差归一化的 velocity loss；
3. 记录并平衡每个 k，而不是完全随机混合后只看总均值；
4. 为 grid branch 设置最低有效权重或先固定 lambda 做 ablation，确认自适应 lambda 没有把行为监督压得过小；
5. 如果仍需 gripper sign loss，只使用小权重，并通过失败视频验证其必要性；
6. 加固定 validation 和 best-checkpoint 选择，再决定是否继续训练。

不建议在没有 fixed validation 的情况下仅因为 train loss 抖动就盲目扩到 1 万步。

### 8.3 闭环验证顺序

建议顺序：

1. 固定离线 validation；
2. 对同一个 clean seed 做 1 次闭环；
3. 对同一个 randomized seed 做 1 次闭环；
4. 同 seed 对照真实 VLM；
5. 查看完整视频，按接近/抓取/搬运/放置阶段标记失败；
6. 只有 1+1 接线和行为合理后，再扩成功率样本数。

## 9. 运行和监控命令

查看日志：

```bash
tail -n 80 -F /data/pxchen/LaWAM/outputs/sec284_inference_grid_kd_1000step/train.log
```

进入 tmux：

```bash
tmux attach -t sec284_grid_1000
```

查看 session：

```bash
tmux ls
```

查看 GPU：

```bash
nvidia-smi
```

如用户明确决定停止，优先发送 Ctrl-C 让训练脚本有机会正常退出：

```bash
tmux send-keys -t sec284_grid_1000 C-c
```

不要根据本文件中的 PID 杀进程；PID 会变化，tmux session 和输出目录才是稳定标识。

## 10. 代码和脚本索引

### 模型与 loss

```text
starVLA/model/sec284.py
starVLA/model/framework/vlas/flowmatching_expert.py
tools/sec284_data.py
```

### Teacher cache / stats

```text
tools/build_sec284_teacher_cache.py
tools/build_sec284_teacher_stats.py
tools/build_sec284_inference_grid_cache_ddp.py
tools/run_sec284_teacher_cache_parallel.sh
tools/tail_sec284_teacher_cache.sh
```

### 训练

```text
tools/train_sec284_distill_ddp.py
tools/train_sec284_frozen_expert_ddp.py
tools/train_sec284_inference_grid_kd.py
tools/train_sec284_full_inference_grid_kd_ddp.py
tools/run_sec284_frozen_expert_2000step.sh
tools/run_sec284_full_inference_grid_kd_1000step.sh
```

### 离线分析

```text
tools/eval_sec284_distill.py
tools/plot_sec284_training.py
tools/probe_sec284_sample.py
tools/probe_sec284_variance.py
tools/analyze_sec284_shadow_traces.py
```

### LAP6 部署和闭环

```text
deployment/model_server/server_policy_lap6_sec284_no_vlm.py
deployment/model_server/server_policy_lap6_sec284_shadow_trace.py
tools/run_lap6_paired_1x.sh
tools/run_lap6_sec284_no_vlm_paired_1x.sh
tools/run_lap6_sec284_shadow_trace_paired_1x.sh
tools/run_sec284_robotwin_1x.sh
tools/run_sec284_paired_baselines_1x.sh
```

### 设计文档

```text
README.md
docs/PROJECT_OVERVIEW_ZH.md
docs/VLM_REPLACEMENT_JOINT_TRAINING_DESIGN_ZH.md
timeline.md
```

## 11. 文档和仓库注意事项

当前 worktree 有未提交修改和新增文件。它们属于本轮或用户已有工作，接手时不得使用 `git reset --hard`、`git checkout --` 等方式清理。

当前主要未提交内容包括：

- SEC284 模型、训练、缓存、评估与部署脚本；
- LAP6 + SEC284 no-VLM/shadow server；
- flow expert 的 trace/forced-x/step-offset 支持；
- README、项目概览、timeline 和 joint-training 设计文档修改；
- `docs/assets/` 中的图。

现有 `README.md`、`docs/PROJECT_OVERVIEW_ZH.md` 和 `docs/VLM_REPLACEMENT_JOINT_TRAINING_DESIGN_ZH.md` 部分段落仍以旧 behavior KD 或早期少量 shadow trace 为背景。全量 grid cache 已建成，clean 失败模式也已纠正；后续应把这些文档与本 handoff 的事实统一，尤其删除或改写“gripper 是唯一主要瓶颈”的表述。

## 12. 接手者必须避免的误区

1. 不要再接 LAP8；当前唯一有效主线是 LAP6。
2. 不要把 actions cache 的历史路径名误认为当前在使用 LAP8 模块。
3. 不要用 RoboTwin 随机 instruction 的真实 VLM condition 与固定-query SEC284 做直接比较。
4. 不要把 9 个 shadow trace points 当成全部数据；全量训练集是 24,140。
5. 不要把 inference-grid cache 解释成 gripper 专用训练。
6. 不要只看 cosine/MSE 就判断闭环一定成功。
7. 不要只看混合随机 k 的 train loss 判断是否收敛。
8. 不要根据 clean trace 的 gripper 误差就断言“没抓住”；视频已经表明抓到后又碰倒。
9. 不要在没有 dynamic R²/std ratio 验证时默认 condition 不会塌缩。
10. 不要在没有固定验证 ablation 时盲目延长到 1 万步。
11. 不要静默停止当前 active tmux 训练；除非用户明确要求，先评估并报告。
12. 不要清理或覆盖 dirty worktree 中的用户修改。

## 13. 当前待决策事项

当前最需要用户/接手者做的决策不是“要不要直接全量解冻”，而是：

1. 立即补一个固定 held-out、按 k 拆分的验证脚本；
2. 下一轮是否从 500-step 经验候选加入简单 anchor；
3. 是否用 teacher 方差归一化替代手工 gripper 4x；
4. grid checkpoint 只有在离线固定验证优于 baseline 后，是否再做同 seed 1+1 闭环。

推荐优先级是：**固定验证 > 修正 loss/日志 > 1+1 闭环 > 扩大成功率测试 > 考虑解冻或长训**。

## 14. 2026-08-12 下午更新：Expert grid-KD 续训与 clean 闭环

### 14.1 续训配置和状态

在 500-step Expert grid-KD checkpoint 的基础上继续训练到总计 2000 step：

```text
tmux: sec284_expert_grid_2000
output: outputs/sec284_expert_grid_kd_2000step
log: outputs/sec284_expert_grid_kd_2000step/train.log
GPU: 1,3,4,5
world=4, local_batch=8, global_batch=32
LR=1e-7, uniform inference-grid KD, enc_vlm frozen
```

续训脚本使用 `--start-step 500 --steps 1500`，因此日志中的 step 是绝对 step（501/2000 到 2000/2000），grid step 仍按 10 个 inference-grid step 循环。500-step checkpoint 没有保存 AdamW optimizer 状态，续训时重新初始化 optimizer，但加载的是 500-step Expert 权重。

训练已完成 `2000/2000`；最终日志显示 `grid_kd=0.000908`，并已生成 1500/2000 的部署 checkpoint。

### 14.2 train loss 与闭环结果

当前训练 loss 是 forced teacher `x_k` 上的 uniform velocity MSE，只是局部 teacher-forcing 指标，不是闭环成功目标。日志统计显示：

```text
step 1-500      mean grid_kd ≈ 0.000905
step 501-1000   mean grid_kd ≈ 0.000861
step 1001-1500  mean grid_kd ≈ 0.000893
step 1501-2000  mean grid_kd ≈ 0.000871
```

下降幅度很小且之后基本平台化。它不能保证 gripper 时序、接触、抬升和误差累积在真实 rollout 中变好。

使用完全相同的 LAP6 + SEC284 no-VLM、clean task、seed 100002、400 steps、replan=36 做 1+1 闭环：

| Expert checkpoint | clean success | 视频观察 |
|---|---:|---|
| 500 step | 0/1 | 能接触并夹住瓶子，后段有短暂抬升/搬运尝试，但最终失稳 |
| 1000 step | 0/1 | 接触后更早失稳，瓶子较快被碰倒，未形成稳定抬升 |
| 1500 step | 0/1 | 未形成成功闭环 |
| 2000 step | 0/1 | 未形成成功闭环 |

结果目录：

```text
results/eval_runs/sec284_expert_grid_kd_step500_clean_seed0_1x/
results/eval_runs/sec284_expert_grid_kd_step1000_clean_seed0_1x/
results/eval_runs/sec284_expert_grid_kd_step1500_clean_seed0_1x/
results/eval_runs/sec284_expert_grid_kd_step2000_clean_seed0_1x/
```

本轮没有跑 randomized。当前经验上的 best checkpoint 是 500 step；不能按总 `grid_kd` 单调选择 checkpoint。1000 step 的退化提示当前 loss 与下游存在 objective mismatch，可能是关键 gripper/抬升维度被均匀平均，或全量 Expert 继续更新后偏离 500-step 的有效策略。

### 14.3 部署 checkpoint 和新增脚本

训练保存的 `step-000500.pt` / `step-001000.pt` 只包含 `expert` state dict；部署 loader 还需要原始 Expert 的 `flow_cfg` 和 `action_horizon`。因此额外生成了：

```text
outputs/sec284_expert_grid_kd_500step/step-000500_deploy.pt
outputs/sec284_expert_grid_kd_2000step/step-001000_deploy.pt
```

本轮新增或更新：

```text
tools/train_sec284_expert_grid_kd_ddp.py
tools/run_sec284_expert_grid_kd_continue_2000step.sh
tools/run_lap6_sec284_no_vlm_clean_1x.sh
tools/run_lap6_sec284_no_vlm_paired_1x.sh
```

`run_lap6_sec284_no_vlm_clean_1x.sh` 支持通过 `ROBOTWIN_TEST_NUM` 批量跑固定 clean episode，并支持通过 `ROBOTWIN_STEP_LIMIT_OVERRIDE` 覆盖 episode 步数。

### 14.4 后续建议

1. 不把任何 grid checkpoint 按训练 loss 直接标成部署成功；
2. 以 500-step 作为经验候选，对比固定 held-out 和同 seed 闭环；
3. 若后续 checkpoint 仍不如 500，下一轮从 500 重启，优先尝试一个简单的 500-step anchor；
4. 增加 held-out、按 `k=0..9` 拆分的 grid validation，并单独记录 gripper/抬升相关动作误差。

## 15. 2026-08-12 当前进度：500-step clean 10x

为确认 500-step checkpoint 的稳定性，使用与此前一致的 LAP6 + SEC284 no-VLM、`move_pillbottle_pad`、`demo_clean`、replan=36，固定跑 10 个 episode；本轮不跑 randomized：

```text
checkpoint: outputs/sec284_expert_grid_kd_500step/step-000500_deploy.pt
output: results/eval_runs/sec284_expert_grid_kd_step500_clean10
timestamp: 20260812_162222
```

结果：**0/10（0%）**，10 个 episode 均执行到 400/400 步。seed 为 100002、100005、100006、100007、100008、100009、100010、100011、100013、100015。严格成功率仍为 0%，但这不否定视频中观察到的接触、夹持和短暂抬升迹象；后续仍应结合动作阶段指标和固定 held-out 验证选择 checkpoint，不能仅按 total/grid train loss 选择。

正式结果与视频：

```text
results/eval_runs/sec284_expert_grid_kd_step500_clean10/clean/lawam_robotwin_sft_release__demo_clean/20260812_162222/tasks/move_pillbottle_pad/summary.json
results/eval_runs/sec284_expert_grid_kd_step500_clean10/clean/lawam_robotwin_sft_release__demo_clean/20260812_162222/tasks/move_pillbottle_pad/episode*.mp4  # 本地视频，未提交
```

---

## 16. 2026-08-13 当前归档

### 16.1 当前训练和评测状态

- `sec284_output_kd_primary_2000step` 已完成；末段 `repr=0.062496`、`grid_kd=0.000812`、`std_ratio=0.9209`。
- Expert-only grid-KD 已完成总计 2000 step；500/1000/1500/2000 clean 1+1 均为 `0/1`，500-step clean 10x 为 `0/10`。
- 纯表示 held-out 仍以 `cosine=0.955959`、`dynamic R²=0.724421`、`std ratio=0.8701` 为唯一完整 test 口径；新下游 checkpoint 尚未完成同口径 held-out。
- 2026-08-13 在同一 randomized seed `100001` 下，原始 VLM 与 LAP6+官方 VLM 均成功（`1/1`，139/141 steps）；这是环境/主干对照，不是 SEC284 成功结果。

### 16.2 原始证据提交范围

本次同步会将 SEC284 的训练日志、评测日志、JSON/JSONL、`meta.json`、`run.log` 和 `_result.txt` 按原路径加入 Git；视频、checkpoint、feature/teacher cache、二进制 shadow trace 和图片不加入。完整路径索引见 [SEC284 当前状态与原始证据索引](SEC284_CURRENT_STATUS_2026-08-13_ZH.md)。

本文件记录的是 2026-08-13 的最新交接状态；训练进程已结束，后续应从固定 held-out 验证和 500-step 对照开始。
