# LaWAM × RoboTwin 项目总览

> 更新：2026-08-05（Asia/Hong_Kong）  
> 工作目录：`/data/pxchen/LaWAM`  
> 当前范围：RoboTwin 单任务 `move_pillbottle_pad`（任务 14）  
> 时间线：仓库根目录 [timeline.md](../timeline.md)

## 1. 目标与当前结论

本项目探索在 RoboTwin 中移除 Qwen3-VL-2B 的在线推理，以较小的视觉/状态条件模块替代其对 LaWM 和 Action Expert 的条件输入，同时保持单任务闭环能力。

当前最可靠的结论如下：

1. 三视角 LAP 已能从当前 DINO 特征和双臂 EEF state 产生有效 latent action，并可稳定驱动冻结 LaWM。
2. 将 VLM 的 284 个语义 token 替换为 LAP10V3 token 后，离线 token/action 指标可以改善，但不能据此推断闭环成功。
3. 当前最佳无 VLM 闭环基线是 T7：clean `1/5`、randomized `3/5`、总计 `4/10`。
4. 直接真实动作 AR 微调（T8）得到 clean `1/5`、randomized `0/5`、总计 `1/10`。AR-A step 1000 与 AR-B step 2000 相同，问题已定位到 Expert-only AR 更新，而不是 LAP9–10 解冻。
5. 正在运行的 FlowOnly 实验从 T7 恢复、冻结 LAP/LaWM、只更新 Expert，并删除自定义 action 加权、reconstruction 和 delta loss，用于验证此前退化是否来自训练目标而非模型容量。

## 2. 任务、数据与评估口径

### 2.1 固定任务

```text
任务名：move_pillbottle_pad
目标：抓取药瓶并放置到蓝色垫上
机器人：Aloha-AgileX 双臂
控制：双臂 EEF absolute pose + gripper，16 个有效维度
相机：主视角、左腕、右腕，均为 640×480 RGB
```

### 2.2 训练数据

训练缓存来自 RoboTwin 官方 `robotwin_merged` 数据集的任务 14：

| 项目 | 数量 |
|---|---:|
| 训练轨迹 | 480 |
| randomized 轨迹 | 440 |
| clean 轨迹 | 40 |
| 训练时刻样本 | 24,140 |
| action horizon | 36 token（30 Hz，约 1.2 秒） |

每个训练样本由当前三视角 DINO token、当前归一化 EEF state 和未来 action chunk 组成。样本按轨迹每 3 帧取 anchor，因此相邻 action chunk 大量重叠；24,140 是训练时刻数，不是 24,140 条独立轨迹。

### 2.3 闭环评估

所有可比较结果使用下列固定口径：

```text
RoboTwin task：move_pillbottle_pad
clean：5 条
randomized：5 条
replan_steps：36
skip_get_obs_within_replan：true
precision：FP32
```

训练 loss、离线 MAE 和 token MSE 都是诊断指标；唯一的任务指标是 `summary.json` 中的 RoboTwin 成功数。

## 3. 当前模型链路与代码映射

```text
三视角 RGB + 当前 EEF state
        ↓
DINO / LAM vision encoder（冻结）
        ↓
三视角 DINO token [B,3,256,768]
        ├── LAP6 → z_lap [B,1,32] → LaWM → h_t1（视觉子目标）
        └── LAP10V3 → cond_lap [B,284,768]
                                      ↓
当前主视角 h_t + h_t1 + cond_lap → Action Expert → 36-step action chunk
```

| 概念模块 | 当前实现 | 作用 |
|---|---|---|
| LAP6 | `starVLA/model/lap_stage1.py` | 三视角/EEF 到 32-D latent action；输出供 LaWM 使用 |
| LaWM | `latent_action_model/`，由 Stage-1 checkpoint 加载 | 根据主视角与 `z_lap` 预测未来视觉特征 `h_t1` |
| LAP10V3 | `starVLA/model/lap_stage2.py` | 生成 284×768 的 Expert 条件 token |
| Action Expert | `starVLA/model/framework/vlas/flowmatching_expert.py` | 通过 conditional flow matching 生成 action chunk |
| 无 VLM 部署 | `deployment/model_server/server_policy_lap8_no_vlm.py` | DINO → LAP/LaWM/Expert 的 FP32 RoboTwin 服务 |

重要边界：LAP6 的 `z_lap` 是 LaWM 的输入；LAP10V3 的 `cond_lap` 是 Action Expert 的语义条件。两条链路在代码与训练中分开维护。

## 4. 实验脉络

完整训练日志、checkpoint 和 RoboTwin `summary.json` 保留在本地工作区的 `logs/`、`outputs/` 与 `results/eval_runs/` 下；本页只记录可追踪的项目结论。

| ID | 实验 | 核心变化 | 结论 |
|---|---|---|---|
| T0 | Stage-1 单视角 LAP60M | IDM latent + LaWM + EEF | 离线 latent 学习有效 |
| T1 | Stage-1 三视角联合 | 加入主/左腕/右腕 DINO token | 三视角训练链路成功 |
| T2 | LAP8 Phase 1 | 8 token 无 VLM 条件 | 闭环 `0/10` |
| T3 | LAP10 alignment | 284 token 对齐 VLM 条件 | 离线有效，不足以保证闭环 |
| T4/T5 | LAP10V2 scratch/unified | 从零学习 token 条件 | 出现/接近塌缩 |
| T6 | LAP10V3 scratch | 四层 284-token 条件器 | 离线有效，闭环不稳定 |
| T7 | LAP10V3 + Expert joint | flow + token alignment | 闭环 `4/10`，当前无 VLM baseline |
| T8 | AR-2000 | 全量真实 action；无 teacher | 闭环 `1/10`，低于 T7 |
| T9（进行中） | FlowOnly-1000 | T7 初始化；冻结 LAP；仅官方 flow | 验证 AR 退化来源 |

## 5. 已定位的风险

### 5.1 不应归因于的数据问题

- 训练集中 randomized 占 `22,110 / 24,140 = 91.5%`，不是 randomized 样本不足。
- Action cache 由三视角 feature cache 的同一 manifest、同一 `episode/base_index` 生成，未发现视觉/action 乱序证据。
- LaWM 与 LAP6 在 AR-A 中冻结；AR-A 已退化，因此它们不是此次退化的直接原因。

### 5.2 仍需处理的学习问题

1. 成功示范不包含策略犯错后的恢复状态，闭环误差会使部署状态偏离训练分布。
2. 训练中的随机 flow 中间状态误差不同于部署时 10-step flow 积分后的动作误差。
3. 原 AR 对夹爪事件、闭合阶段和左臂 xyz 做强加权，可能优化了局部动作数值而损害落点、松爪和撤离时机。
4. `replan_steps=36` 会连续执行约 1.2 秒动作；精确放置中的小偏差可能在下一次观测前放大。
5. 5+5 是快速诊断口径，仍有流采样随机性；重要结论应使用固定 policy noise 或更大样本复核。

## 6. 当前运行实验：T9 FlowOnly-1000

```text
初始化：T7 step 2000
训练：单卡 GPU7，FP32
step：1000
effective batch：16（batch 2 × gradient accumulation 8）
可训练模块：Action Expert，enc_vlm 冻结且未使用
冻结模块：LAP6–LAP10V3、LaWM
loss：官方、未加权的 flow matching loss；仍使用 actions_mask
学习率：2e-7，50-step warmup，cosine decay
保存：step 250 / 500 / 750 / 1000
```

日志与输出：

```text
logs/lap10v3_ar_flowonly_task14_1000step/train.log
outputs/lap10v3_ar_flowonly_task14_1000step/
```

选择 checkpoint 的原则不是最低训练 loss，而是先在 T7 曾成功的 randomized seeds（`100001`、`100004`、`100006`）上做闭环筛选，再运行完整 5+5。

## 7. 文档索引

| 文档 | 用途 |
|---|---|
| [timeline.md](../timeline.md) | 按时间排序的项目进展与当前状态 |
| 本页 | 当前结构、数据口径、实验结论与下一步 |

## 8. 推荐的后续顺序

1. 完成 T9 FlowOnly-1000，保留所有 250-step checkpoint。
2. 使用同一 RoboTwin seed 和固定 flow sampling 进行 checkpoint 筛选。
3. 若 T9 能保住或超过 T7 的 `4/10`，再考虑小幅解冻 Expert 的局部模块；不要先解冻 LAP。
4. 若 T9 仍退化，优先缩短 replan interval 并增加纠错/偏移状态数据，而不是盲目扩大 LAP 或延长训练。
