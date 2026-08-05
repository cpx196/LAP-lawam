# 单任务 60M LAP：Stage 1 训练方案

> 目标任务：`move_pillbottle_pad`（将药瓶放到蓝色目标垫）
> 文档范围：只讨论 Stage 1；核心模块只使用 **IDM、LAP、LaWM** 三个名称。
> 核心决定：**IDM 冻结作为教师，LAP 约 60M 并从零训练，LaWM 从已有权重初始化后受约束地联合训练。**

## 1. 目标与边界

Stage 1 的目标是训练一个单任务 LAP，使其只根据当前视觉和当前 EEF 状态预测 latent action，并利用 LaWM 预测未来视觉特征。

训练完成后的在线路径为：

```text
当前图像 I_t + 当前 EEF 状态 s_t
             ↓
       冻结视觉特征提取
             ↓
        F_t + s_t
             ↓
          60M LAP
             ↓
    z_lap + scene_tokens
             ↓
           LaWM
             ↓
     预测未来视觉特征 F_hat_t1
```

Stage 1 明确不做以下事情：

- 不输入历史帧；
- 不输入历史 EEF；
- 不把真实 action 序列作为 LAP 输入；
- 不输入语言；
- 不输入外部 task ID 或 task token；
- 不训练 Action Head；
- 不以 Stage 1 的离线指标代替最终闭环成功率。

当前只有一个任务，任务语义隐式保存在 LAP 的参数和 learned queries 中。

## 2. 模块职责

### 2.1 IDM

IDM 在训练时看到真实的当前与未来视觉特征：

```text
IDM(F_t, F_t1) -> z_idm
```

其中：

```text
F_t:     [B, 256, 768]
F_t1:    [B, 256, 768]
z_idm:   [B, 1, 32]
```

IDM 的职责是产生 latent-action 教师目标。IDM 全程冻结，训练结束后从部署模型中删除。

### 2.2 LAP

LAP 只看到当前信息：

```text
LAP(F_t, s_t) -> z_lap, scene_tokens
```

其中：

```text
F_t:            [B, 256, 768]
s_t:            [B, 16]
z_lap:          [B, 1, 32]
scene_tokens:   [B, 8, 768]
```

`z_lap` 用于驱动 LaWM；`scene_tokens` 保留当前场景、目标、机械臂状态和任务阶段信息，供 Stage 2 替代原来的大模型条件 token。

### 2.3 LaWM

LaWM 接受当前视觉特征和 latent action：

```text
LaWM(F_t, z) -> F_hat_t1
```

训练时同时运行两条路径：

```text
LaWM(F_t, z_idm) -> F_hat_t1_teacher
LaWM(F_t, z_lap) -> F_hat_t1_student
```

第一条路径约束 LaWM 不偏离已有 latent 空间；第二条路径让 LaWM 适应 LAP 的预测误差。

## 3. 训练样本定义

RoboTwin 数据为 30 Hz。当前 RoboTwin 策略使用约 1.2 秒的 action chunk，两帧视觉端点对应帧偏移 `[0, 35]`。

每个 Stage 1 样本定义为：

```text
当前帧：t
未来帧：t1 = min(t + 35, episode_last_frame)
```

样本内容：

```text
I_t, s_t, I_t1, s_t1, episode_index, frame_index, domain
```

其中：

- `I_t`、`I_t1` 使用主视角 `cam_high`；
- `s_t`、`s_t1` 是 16 维双臂绝对 EEF 状态；
- `domain` 为 `clean` 或 `randomized`；
- `s_t1` 只作为辅助训练目标，不是 LAP 输入；
- episode 尾部不足 35 帧时使用最终帧，形成任务完成后的停止/no-op样本。

EEF 维度顺序：

```text
左臂 xyz(3) + quaternion(4) + gripper(1)
右臂 xyz(3) + quaternion(4) + gripper(1)
```

## 4. 数据范围与划分

任务 `move_pillbottle_pad` 的数据范围：

```text
randomized: episode 6500-6999，共 500 条
clean:      episode 25650-25699，共 50 条
```

按 episode 分层划分：

| Split | Randomized | Clean | 合计 |
|---|---:|---:|---:|
| Train | 440 | 40 | 480 |
| Validation | 30 | 5 | 35 |
| Offline Test | 30 | 5 | 35 |

约束：

1. 先按 episode 划分，再生成帧对；
2. 同一 episode 不得跨 split；
3. 固定随机种子并保存 episode manifest；
4. clean 与 randomized 分别报告指标；
5. 最终使用未见过的 RoboTwin 仿真种子做在线测试。

推荐帧采样：

- 非末端区域每隔 3 帧取一个 anchor；
- 末端 35 帧单独采样，保证完成与停止阶段不缺失；
- 按归一化轨迹进度分桶采样，避免长轨迹或某一阶段占比过高；
- 第一版保持 randomized:clean 的自然比例约 10:1；若 clean 验证明显较差，再将 clean 采样比例提高到 15%-20%。

## 5. 60M LAP 结构

### 5.1 设计原则

LAP 不重新学习像素级视觉编码，而是在冻结视觉特征上完成：

- 目标药瓶识别；
- 蓝色目标垫定位；
- 左右臂选择；
- 当前 EEF 状态融合；
- 抓取、搬运、放置、释放阶段判断；
- latent action 预测；
- 为 Stage 2 保留紧凑场景 token。

LAP 不先把 DINO 的 768 维 token 统一压缩到 256 或 384 维，而是让每个 Cross-Attention 层直接读取完整 768 维特征，避免单次投影形成过强信息瓶颈。

### 5.2 输入编码

视觉输入保持：

```text
F_t: [B, 256, 768]
```

EEF 使用训练集统计量进行逐维标准化，然后编码：

```text
16 -> 256 -> 768
```

得到一个状态 token：

```text
e_state: [B, 1, 768]
```

四元数输入前重新归一化到单位长度。状态标准化统计量只能由训练 split 计算。

### 5.3 Scene queries

初始化 8 个 learned scene queries：

```text
Q_scene: [B, 8, 768]
```

与状态 token 拼接：

```text
Q_0 = concat(Q_scene, e_state)
Q_0: [B, 9, 768]
```

这些 query 不设置人工角色标签，但允许模型分别承载物体、目标、左右臂、夹爪状态、任务阶段和干扰物等信息。

### 5.4 六层融合块

LAP 使用 6 个 Transformer fusion blocks：

```text
hidden_dim = 768
attention_heads = 12
head_dim = 64
ffn_dim = 3072
dropout = 0.1
```

每层依次执行：

1. 9 个 query/state token 之间的 Self-Attention；
2. query 对 256 个 DINO token 的 Cross-Attention；
3. FFN；
4. Pre-LayerNorm 和残差连接。

形式化表示：

```text
Q'      = Q + SelfAttention(LN(Q))
Q''     = Q' + CrossAttention(LN(Q'), LN(F_t), LN(F_t))
Q_next  = Q'' + FFN(LN(Q''))
```

状态 token 每层都参与 Self-Attention，因此 EEF 信息不会只在入口处注入一次。

### 5.5 Latent pooling

使用一个 learned latent query 对最终的 8 个 scene tokens 和状态 token 做一次 Attention Pooling：

```text
q_latent -> CrossAttention(Q_final) -> h_latent
```

输出头：

```text
LayerNorm(768)
Linear(768, 32)
```

最终得到：

```text
z_lap: [B, 1, 32]
```

Stage 2 使用的场景输出为：

```text
scene_tokens = Q_final[:, :8, :]
scene_tokens: [B, 8, 768]
```

### 5.6 参数量预算

| 组成 | 预计参数量 |
|---|---:|
| 6 层 Self-Attention + Cross-Attention + FFN | 约 56.7M |
| Latent Attention Pooling | 约 2.36M |
| EEF MLP、queries、LayerNorm、32维输出头 | 约 0.25M |
| LAP Core 合计 | **约 59.3M** |

实现误差允许在 57M-62M 之间，但需要记录最终精确参数量。

## 6. 初始化策略

主方案：

| 模块 | 初始化 | Stage 1 状态 |
|---|---|---|
| IDM | 已有权重 | 全程冻结 |
| LAP | 随机初始化 | 训练 |
| LaWM | 已有权重 | 先冻结，后小学习率联合训练 |

不从零训练 IDM 和 LaWM。原因：

- 当前单任务只有 550 条轨迹；
- LaWM 约 228M，完整从零训练容易过拟合；
- IDM 只需要作为稳定教师，没有重新训练的必要；
- 本项目目标是替换昂贵的 VLM 路径，不是放弃已有的视觉动力学先验。

LaWM 初始化候选：

1. 通用 Stage 1 权重；
2. 已适配 RoboTwin 的 LaWM 权重。

在正式训练前，用冻结 IDM 产生的 `z_idm` 比较两种 LaWM 初始化的验证集未来特征误差，选择误差更低者。预期 RoboTwin 适配版本更优，但以实测为准。

## 7. 离线教师缓存

冻结视觉特征提取和 IDM，预先缓存：

```text
F_t
F_t1
z_idm
s_t
s_t1
episode_index
frame_index
domain
terminal_mask
```

建议：

- `F_t`、`F_t1` 使用 BF16；
- `z_idm` 可用 FP16/BF16，计算损失时转 FP32；
- 缓存中记录模型权重 hash、图像预处理配置和数据 manifest hash；
- 缓存不得跨不同图像预处理设置复用；
- 第一版关闭像素级增强，与现有 RoboTwin 配置保持一致。

## 8. 损失函数

### 8.1 Latent 蒸馏损失

```text
L_latent = MSE(z_lap, stop_gradient(z_idm))
```

同时记录 cosine similarity，但第一版不把 cosine loss 加入主损失。

### 8.2 Student world loss

```text
F_hat_t1_student = LaWM(F_t, z_lap)
L_world_student = MSE(F_hat_t1_student, F_t1)
```

该损失直接检验 LAP latent 是否能驱动正确的未来预测。

### 8.3 Teacher world 保持损失

```text
F_hat_t1_teacher = LaWM(F_t, z_idm)
L_world_teacher = MSE(F_hat_t1_teacher, F_t1)
```

该损失在联合训练 LaWM 时防止 LAP 与 LaWM 共同漂移到一套失去原有含义的私有 latent 空间。

### 8.4 EEF 辅助损失

增加一个仅训练时使用的小型辅助头：

```text
[z_lap, s_t] -> s_hat_t1
```

损失组成：

- 左右臂 xyz：Smooth L1；
- 左右臂 quaternion：旋转 geodesic loss；
- 左右 gripper：MSE 或 BCE，按实际归一化定义选择。

该辅助头在部署时删除，不计入最终 LAP 参数预算。

### 8.5 Scene token 防坍缩损失

对 8 个 scene tokens 加轻量的去相关/多样性约束：

```text
L_div = mean(off_diagonal(cosine_similarity(scene_tokens)))^2
```

目标是避免 8 个 scene tokens 全部收敛到相同表示。该项权重必须很小，不能压过 latent 与 world 目标。

### 8.6 总损失

第一版权重：

```text
L_total =
    1.00 * L_latent
  + 1.00 * L_world_student
  + 0.25 * L_world_teacher
  + 0.10 * L_eef
  + 0.01 * L_div
```

每个损失必须单独记录。若梯度范数相差超过一个数量级，再调整权重，不能只根据 loss 数值大小盲目调整。

## 9. 训练阶段

### 9.1 Phase 0：数据与教师基线

目标：先确定数据、IDM 和 LaWM 初始化正确。

操作：

1. 生成 episode manifest；
2. 生成训练/验证/测试 split；
3. 构建帧对并缓存教师输出；
4. 统计 `z_idm` 的均值、标准差、协方差和有效秩；
5. 测量 `LaWM(F_t, z_idm)` 的验证误差；
6. 比较两种 LaWM 初始化，选择 teacher-world 误差更低者；
7. 检查 clean/randomized 和左右臂样本比例。

该阶段不训练任何模块。

### 9.2 Phase 1：LAP warm-up

冻结：

```text
IDM
LaWM
```

训练：

```text
LAP
EEF辅助头
```

虽然 LaWM 冻结，`L_world_student` 的梯度仍通过 LaWM 回传到 LAP。

推荐配置：

```yaml
steps: 8000-12000
optimizer: AdamW
lap_lr: 2.0e-4
weight_decay: 5.0e-2
betas: [0.9, 0.95]
warmup_steps: 800
scheduler: cosine
effective_batch_size: 128
precision: bf16-mixed
gradient_clip_norm: 1.0
dropout: 0.1
eval_interval: 250
```

停止条件：

- 验证 `L_latent` 连续多次不改善；或
- student-world/teacher-world 比例已满足验收标准；或
- 出现明显训练下降、验证恶化。

### 9.3 Phase 2：LAP + LaWM 后层联合训练

冻结：

```text
IDM
LaWM前部
```

训练：

```text
LAP
LaWM最后4层及输出归一化/投影
EEF辅助头
```

推荐配置：

```yaml
steps: 4000-6000
lap_lr: 5.0e-5
lawm_lr: 1.0e-5
weight_decay: 1.0e-2
warmup_steps: 300
scheduler: cosine
effective_batch_size: 128
precision: bf16-mixed
gradient_clip_norm: 1.0
eval_interval: 250
```

必须保留 `L_world_teacher`，否则不允许联合更新 LaWM。

### 9.4 Phase 3：可选的完整 LaWM 微调

仅当以下条件同时满足时启动：

1. `z_lap` 已很好逼近 `z_idm`；
2. student-world 误差仍明显高于 teacher-world；
3. Phase 2 没有明显过拟合；
4. 误差分析显示问题来自 LaWM 对 student latent 的敏感性。

推荐配置：

```yaml
steps: 1500-3000
lap_lr: 2.0e-5
lawm_lr: 3.0e-6
effective_batch_size: 128
precision: bf16-mixed
gradient_clip_norm: 1.0
```

若 Phase 2 已达标，跳过 Phase 3。

## 10. 验证指标与验收标准

### 10.1 Latent 指标

记录：

```text
MSE(z_lap, z_idm)
cosine(z_lap, z_idm)
z_lap逐维均值和标准差
z_lap协方差有效秩
```

初步目标：

- 验证集平均 cosine similarity 不低于 0.90；
- `z_lap` 方差不应坍缩为接近零；
- clean 与 randomized 的 latent 指标分别报告。

### 10.2 World 指标

记录：

```text
teacher_world_mse
student_world_mse
student_world_mse / teacher_world_mse
future token cosine similarity
```

初步验收线：

```text
student_world_mse <= 1.10 * teacher_world_mse
```

若 teacher-world 本身在某些样本上很差，需要单独标记，不把这类误差全部归因于 LAP。

### 10.3 EEF 指标

分别报告：

- 左右臂 xyz 终点误差，单位厘米；
- quaternion 角度误差，单位度；
- gripper 状态准确率；
- 工作臂选择准确率；
- 抓取、搬运、放置、末端阶段分桶误差。

### 10.4 Latent 敏感性

验证时至少比较：

```text
正确 z_lap
batch 内打乱后的 z_lap
全零 z
z_idm
```

要求正确 `z_lap` 的 world loss 明显低于打乱和全零 latent。初步要求打乱 latent 至少使 world loss 上升 20%。若差异很小，说明 LaWM 正在忽略 latent。

### 10.5 Scene token 有效性

记录：

- 8 个 scene tokens 的两两 cosine similarity；
- token 协方差有效秩；
- 打乱 EEF 状态后 scene tokens 和 `z_lap` 的变化；
- 遮挡目标药瓶或目标垫后相应 token 的变化。

Stage 1 只要求 scene tokens 不坍缩且对视觉/EEF输入敏感；它们是否足以替代 Action Head 的原条件，要在 Stage 2 用动作损失和闭环成功率验证。

## 11. 泛化评估

本方案保留的是单任务内部泛化，而不是跨任务和开放语言泛化。

需要验证：

- 未见药瓶位置；
- 未见目标垫位置；
- 左右臂切换；
- 五种药瓶外观；
- 未见背景、光照、桌面高度和干扰物组合；
- 未见仿真随机种子。

最终在线评估建议：

```text
demo_clean:      100 个新随机种子
demo_randomized: 100 个新随机种子
```

Stage 1 本身不输出真实机器人 action，因此在线成功率要在 Stage 2 接入 Action Head 后测量。Stage 1 只负责通过离线指标判断是否值得进入 Stage 2。

## 12. 必做消融

主模型是约 60M LAP。至少保留以下消融：

| 实验 | 目的 |
|---|---|
| 10M LAP | 判断 60M 容量是否确有收益 |
| 60M LAP 去掉 EEF 输入 | 定量判断当前状态的重要性 |
| 60M LAP + 冻结 LaWM | 判断联合训练是否必要 |
| 60M LAP + 联合 LaWM | 主方案 |
| 打乱/全零 latent | 判断 LaWM 是否真正使用 latent |

不把“IDM、LAP、LaWM全部从零训练”作为主实验，只可做短程失败对照。

## 13. Checkpoint 内容

Stage 1 训练 checkpoint 保存：

```text
LAP权重
LaWM权重或相对初始化的delta
EEF标准化统计量
数据split manifest
训练配置
模型结构配置
教师缓存版本/hash
验证指标
```

Stage 1 部署包保留：

```text
冻结视觉特征提取
LAP
LaWM
EEF标准化统计量
```

删除：

```text
IDM
EEF辅助预测头
教师缓存
未来图像输入
```

## 14. 进入 Stage 2 的 Go/No-Go 条件

只有同时满足以下条件才进入 Stage 2：

1. `z_lap` 验证 cosine similarity 达到约 0.90 或以上；
2. student-world 误差不高于 teacher-world 的约 1.10 倍；
3. 打乱 latent 后 world loss 至少上升约 20%；
4. clean 和 randomized 均未出现 latent 坍缩；
5. EEF 打乱实验能显著改变工作臂和任务阶段预测；
6. scene tokens 不坍缩，并对药瓶、目标垫和 EEF 状态敏感；
7. Phase 2 联合训练未明显破坏 teacher-world 基线。

若未达到条件，优先按以下顺序诊断：

```text
检查时间对齐和数据split
检查IDM教师输出
检查LaWM teacher-world基线
检查LAP训练是否欠拟合
检查单帧输入是否存在不可消除的未来歧义
最后才考虑继续增大LAP
```

## 15. 实施顺序

推荐执行顺序：

1. 生成任务14的 episode manifest；
2. 实现帧对采样和 train/val/test 划分；
3. 生成冻结教师缓存；
4. 验证 IDM 和 LaWM teacher-world 基线；
5. 实现约 59-60M LAP；
6. 完成单 batch shape、梯度和 loss 单元测试；
7. 先对一个 batch 过拟合，确认模型与损失可学习；
8. 运行 Phase 1；
9. 通过敏感性检查后运行 Phase 2；
10. 根据验证结果决定是否运行 Phase 3；
11. 固化 Stage 1 checkpoint，进入 Stage 2。

## 附录 A：代码映射

本文主体只使用 IDM、LAP、LaWM 三个概念名称。实现时的映射由代码层处理：

| 概念模块 | 当前项目中的来源 |
|---|---|
| IDM | 已有 latent-action 逆动力学编码与 bottleneck |
| LAP | 新增约 60M 当前状态 latent-action predictor |
| LaWM | 已有 latent-conditioned future-feature decoder |

视觉特征提取保持冻结，作为三者共享的输入预处理，不改变本文的模块划分。

## 附录 B：当前单卡 FP32 启动方案

### B.1 已完成缓存

任务 14 的训练缓存位于：

```text
cache/lap_stage1_task14/train
```

当前共有 189 个 shard、24,140 个训练样本、约 35.4 GiB。训练默认通过
`--preload-cache` 将它们一次性顺序读入 CPU 内存；本机可用内存足够。不要在
全局随机采样的同时只保留单个 shard，否则每个样本都可能触发约 200 MiB 的
反序列化，GPU 会长期等待磁盘。

### B.2 Phase 1 启动命令

```bash
cd /data/pxchen/LaWAM
CUDA_VISIBLE_DEVICES=7 python tools/train_lap_stage1.py \
  --mode train \
  --phase 1 \
  --cache-dir cache/lap_stage1_task14 \
  --steps 10000 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --log-every 20 \
  --save-every 1000 \
  --output-dir outputs/lap_stage1_task14
```

训练阶段从原始教师 checkpoint 中只加载 LaWM（228,286,720 参数）；DINO 和
IDM 已经完成缓存生成，不再分配到 GPU。LAP 为 59,712,800 参数。两项优化都
不改变 `z_idm -> LAP 蒸馏` 和 `LaWM(F_t, z) -> F_{t+1}` 的损失定义。

### B.3 本机吞吐基线（2026-08-04）

```text
缓存预载：27.8--31.2 秒
CPU 峰值常驻内存：约 39.7 GiB
预载后随机读取：约 59,000 samples/s
FP32、batch=1、grad_accumulation=8：稳态约 0.58--0.69 秒/step
```

因此 Phase 1 的 10,000 步纯训练时间约 1.7--1.9 小时；考虑首次预载、每
1,000 步保存 checkpoint 和波动，建议按约 2 小时安排。日志会输出
`step_time` 和 `eta`，应以运行 20--100 步后的 ETA 为最终依据。

若迁移到 CPU 内存不足 45 GiB 的机器，可使用 `--no-preload-cache`，但必须
同时改用 shard-aware sampler；当前全局随机 sampler 下不建议关闭预载。
