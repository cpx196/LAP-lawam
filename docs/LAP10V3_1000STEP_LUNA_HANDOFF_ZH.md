# LAP10V3：284-Token 防塌缩训练技术方案（Luna 交付版）

## 1. 目标与实验边界

本方案用于训练一个不依赖 VLM 推理的单任务 Expert 条件模块。任务固定为：

```text
move_pillbottle_pad（任务 14）
```

模型根据当前时刻的三视角 DINO 特征和双臂 EEF state，生成与官方 VLM
Expert 条件接口对齐的：

```text
cond_lap [B,284,768]
```

本轮正式训练固定为 **1000 step、FP32、双卡**。本轮只验证 LAP10V3 能否：

1. 保持 284 个输出位置的独立身份，不发生 token collapse；
2. 拟合官方 VLM 经 `Action Expert.enc_vlm` 投影后的 284-token 条件；
3. 在冻结官方 Action Expert 时维持较低 flow-matching loss；
4. 为后续 RoboTwin 闭环测试和 Action Expert 联合微调提供有效初始化。

本轮不允许：

- 加载 LAP8 的两个 `expert_fusion` block；
- 加载旧 LAP10/LAP10V2 的 `interface_fusion`、output queries 或输出层；
- 更新 LAP6、LaWM 或官方 Action Expert；
- 把同一个 task/latent embedding 广播相加到全部 284 个 query；
- 使用“逼近单位矩阵”的旧 diversity loss。

## 2. 已确认的问题与定量证据

旧 LAP10V2 的前两层输出发生了 token collapse。其日志长期满足：

```text
diversity loss ≈ 0.9964
```

现有 `diversity_loss` 对归一化 token 的 Gram 矩阵与单位矩阵计算 MSE。若
284 个 token 完全相同，该损失理论值为：

```text
283 / 284 = 0.99648
```

因此该日志几乎等价于 284 个 token 完全塌缩。塌缩不是输出为零，而是同一
样本内：

```text
token_1 ≈ token_2 ≈ ... ≈ token_284
```

对现有 8192 条 teacher cache 的统计结果如下：

| 统计项 | 数值 |
|---|---:|
| teacher token 平均非对角 cosine | 0.2829 |
| teacher 对旧 identity-diversity 目标的损失 | 0.1710 |
| 每个位置使用独立固定均值的 MSE | 0.2195 |
| 每个样本复制单个平均 token 的 MSE | 0.5081 |
| 全局固定 token 的 MSE | 0.5219 |
| 旧 LAP10 最终 alignment MSE | 0.1674 |
| 塌缩 LAP10V2 最终 alignment MSE | 约 0.70～0.75 |

结论：284 个 VLM token 是有序、有位置角色的条件序列。Action Expert 不会
为该条件序列额外添加位置编码，因此 LAP10V3 必须把 token 身份直接编码在
输出数值中。

## 3. 冻结模块与可训练模块

### 3.1 加载并冻结

| 模块 | 初始化来源 | 状态 |
|---|---|---|
| LAP6 | Stage 1 三视角联合训练 checkpoint | 冻结、eval |
| LaWM | Stage 1 checkpoint | 冻结、eval |
| Action Expert | RoboTwin 官方 release | 冻结、eval |
| DINO | 已离线生成特征，不进入当前训练图 | 不加载 |

LAP6 checkpoint：

```text
outputs/lap_stage1_task14_3view_joint_3000step/stage1_phase2_step0003000.pt
```

若从 LAP8 checkpoint 中抽取 LAP6，必须只加载以 `lap6.` 开头的 state dict。
禁止向 LAP10V3 加载任何其他 LAP8 参数。

LaWM 和 Expert 的现有加载路径沿用：

```text
cache/lap8_phase1_official_action_expert.pt
```

### 3.2 从头训练

以下参数全部重新初始化：

- 284 个 content queries；
- 284 个 role/position embeddings；
- 四个 Transformer Decoder block；
- 三视角 Expert 分支 view embeddings；
- `z_lap -> 768` memory-token projection；
- `EEF 16 -> 768` memory-token projection；
- dynamic residual 输出头和 residual scale/gate。

teacher position mean 是训练集统计 buffer，不是旧模型权重，不参与梯度更新。

## 4. 输入、输出与数据

### 4.1 输入

```text
主视角 DINO              [B,256,768]
左腕视角 DINO            [B,256,768]
右腕视角 DINO            [B,256,768]
当前双臂 EEF state       [B,16]
```

LAP6 冻结前向产生：

```text
scene_lap6               [B,8,768]
z_lap                    [B,1,32]
```

### 4.2 Teacher 与动作监督

```text
teacher_condition        [B,284,768]
actions                  [B,50,32]
actions_mask             [B,50,32]
```

现有缓存：

```text
cache/lap_stage1_task14/train
cache/lap_stage1_task14_wrist/train
cache/lap8_phase1_task14_actions/train
cache/lap10_task14_vlm_teacher_8192/train
```

训练期间不加载 Qwen/VLM，只读取 teacher cache。

## 5. Teacher 位置统计

新增脚本建议命名：

```text
tools/build_lap10v3_teacher_stats.py
```

仅使用 teacher train split，逐 shard 以 FP64 累加、最终保存 FP32：

```python
position_mean = teacher_condition.mean(dim=sample_dimension)  # [284,768]
position_std  = teacher_condition.std(dim=sample_dimension)   # [284,768]
```

输出：

```text
cache/lap10_task14_vlm_teacher_8192/position_stats.pt
```

文件至少包含：

```python
{
    "position_mean": Tensor[284,768],
    "position_std": Tensor[284,768],
    "num_samples": 8192,
    "source_cache": "...",
}
```

生成后必须复核：

```text
constant_per_position_MSE ≈ 0.2195
position_mean RMS         ≈ 0.7010
```

数值明显不符时停止，不进入训练。

## 6. LAP10V3 架构

建议在以下文件新增独立类，不要继续修改 LAP8/LAP10V2：

```text
starVLA/model/lap_stage2.py
class LAP10V3(nn.Module)
```

### 6.1 Memory 序列

三个视角 DINO token 加入独立 view embedding 后展平：

```text
visual_memory             [B,768,768]
scene_lap6                [B,8,768]
z_token = Linear(z_lap)   [B,1,768]
state_token = MLP(EEF)    [B,1,768]
```

拼接为：

```text
memory [B,778,768]
    = [visual_memory, scene_lap6, z_token, state_token]
```

`z_lap` 和 state 必须作为独立 memory token，禁止广播加到 284 个 query。
单任务配置不使用 task embedding。

### 6.2 284 个有身份的 Query

维护两组参数：

```text
content_queries   [284,768]
role_embeddings   [284,768]
```

两者均随机初始化；推荐：

```python
Normal(mean=0, std=0.02)
```

每一个 Decoder block 在计算 attention query/key 时重新注入
`role_embeddings`，但不能把 role embedding 每层累计到 content residual：

```python
q_role = norm(q) + role_embedding
q = q + self_attention(query=q_role, key=q_role, value=norm(q))

q_role = norm(q) + role_embedding
q = q + cross_attention(query=q_role, key=memory, value=memory)

q = q + ffn(norm(q))
```

这样 role embedding 决定“第几个 token”，content residual 表示当前场景内容。

### 6.3 四层 Decoder

固定配置：

```text
layers      = 4
width       = 768
heads       = 12
FFN width   = 3072
dropout     = 0.0（前300 step）
```

四层都直接 cross-attend 完整 `memory [B,778,768]`。模型中不存在先生成 8 个
Expert token、再扩展为 284 个 token 的接口。

### 6.4 位置模板加动态残差

注册 teacher mean：

```python
self.register_buffer("teacher_position_mean", position_mean[None])
```

四层输出经过 residual head：

```text
dynamic_residual [B,284,768]
```

最终条件：

```python
cond_lap = teacher_position_mean + residual_scale * dynamic_residual
```

建议 `residual_scale` 初始为 `0.01～0.1`，但必须保证第一步梯度能够传到四层
Decoder；不要把 gate 精确初始化为零。输出头使用小权重初始化。

未训练时 `cond_lap` 应接近 position mean，因此首批 alignment MSE 应约为
`0.2195`，而不是约 `1.7`。这是最重要的链路检查之一。

### 6.5 输出字典

至少返回：

```python
{
    "z_lap":             Tensor[B,1,32],
    "scene_lap6":        Tensor[B,8,768],
    "dynamic_residual":  Tensor[B,284,768],
    "cond_lap":          Tensor[B,284,768],
}
```

## 7. 损失设计

### 7.1 Token MSE

```python
loss_mse = mse(cond_lap, teacher_condition)
```

### 7.2 Token cosine

```python
loss_cos = 1 - cosine_similarity(cond_lap, teacher_condition, dim=-1).mean()
```

### 7.3 Teacher-structure Gram loss

归一化后计算 token-token Gram：

```python
pred_n = normalize(cond_lap, dim=-1)
teacher_n = normalize(teacher_condition, dim=-1)
gram_pred = pred_n @ pred_n.transpose(1, 2)
gram_teacher = teacher_n @ teacher_n.transpose(1, 2)
loss_gram = mse(gram_pred, gram_teacher)
```

禁止继续使用：

```python
mse(gram_pred, identity_matrix)
```

真实 teacher token 并不相互正交，identity 不是正确目标。

### 7.4 Flow loss

使用冻结 LaWM 生成 `h_t1`，冻结 Expert 计算现有 flow-matching loss。阶段一不
加入总损失；阶段二逐步加入。

### 7.5 总损失

Step 1～300：

```text
loss = 1.0 * loss_mse
     + 0.1 * loss_cos
     + 0.1 * loss_gram
```

Step 301～350：

```text
flow_weight = (step - 300) / 50
loss = alignment_loss + flow_weight * loss_flow
```

Step 351～1000：

```text
loss = alignment_loss + 1.0 * loss_flow
```

## 8. 训练日程：总计 1000 Step

### 8.1 Phase A：位置身份与视觉残差学习

```text
step              1～300
flow weight       0
view dropout      0
更新              LAP10V3 全部新参数
冻结              LAP6、LaWM、Expert
```

目标：先让 284 个 query 形成稳定的位置角色和 teacher token 结构，不让有噪声
的 flow loss 干扰防塌缩学习。

### 8.2 Phase B：加入动作相关监督

```text
step              301～1000
flow weight       301～350 从0线性升到1，之后保持1
view dropout      301～350 从0线性升到0.1，之后保持0.1
更新              LAP10V3 全部新参数
冻结              LAP6、LaWM、Expert
```

### 8.3 优化参数

```text
precision                 FP32
optimizer                 AdamW
peak learning rate        1e-4
weight decay              0.05
warmup                    200 step
schedule                  cosine decay to 1e-5
gradient clip             1.0
GPU                       0,1
batch per GPU             1
gradient accumulation     4
effective global batch    8
total step                1000
```

双卡预加载会复制两份约 80GB 级别的 CPU resident 数据，预计总占用约
170GB，机器容量可承受。不要使用四卡预加载。

## 9. 训练前必须通过的测试

### 9.1 State-dict 隔离测试

启动日志必须明确打印：

```text
init=scratch_v3
loaded_lap6_keys=<数量>
loaded_post_lap6_keys=0
```

对加载结果做断言：

```text
只允许 LAP6 key 从 checkpoint 加载
四层 Decoder 与 query/residual 参数不得出现在 loaded key 列表
```

### 9.2 Shape smoke test

```text
memory                  [B,778,768]
cond_lap                [B,284,768]
teacher_condition       [B,284,768]
Expert attention mask   [B,284]
```

执行一次前向、反向、optimizer step，确认所有新参数获得有限梯度。

### 9.3 32 样本过拟合测试

选择固定的 32 条样本，建议 clean/randomized 各 16 条：

```text
view dropout = 0
flow weight  = 0
训练 300～500 step
```

通过标准：

- alignment MSE 明显低于 `0.10`；
- Gram loss 持续下降；
- 284 token 平均非对角 cosine 不接近 1；
- 所有 trainable parameter 有有限梯度；
- 同一个样本的 284 token 不得近似完全相等。

不通过时禁止启动正式 1000-step 作业。

## 10. 正式训练监控与停止条件

每 10 step 记录：

```text
loss_total
loss_mse
loss_cos
loss_gram
loss_flow
pred_rms / teacher_rms
pred_offdiag_cos / teacher_offdiag_cos
pred_identity_div（只监控，不加入 loss）
gradient norm
learning rate
step time
GPU peak memory
```

健康参考：

- 初始 alignment MSE 应接近 `0.2195`；
- teacher off-diagonal cosine 应约为 `0.283`；
- `pred_offdiag_cos` 不应长期高于 `0.9`；
- 旧 identity-div 指标不应再次长期停在 `0.9964`；
- flow 加入后的短暂波动允许，但 alignment 不应永久退化。

立即停止条件：

- NaN/Inf；
- step 50 后 `pred_offdiag_cos > 0.95` 且没有下降；
- step 50 后 `pred_identity_div > 0.90` 且没有下降；
- 初始 MSE 明显高于 `0.30`，说明 position mean 未正确接入；
- 新分支无梯度，或冻结模块出现梯度；
- GPU OOM/CPU OOM。

保存 checkpoint：

```text
step 100（早期诊断）
step 300（Phase A 完成）
step 500
step 750
step 1000
```

## 11. 正式训练验收口径

最低验收要求：

1. 没有 token collapse；
2. held-out alignment MSE 优于固定位置均值基线 `0.2195`；
3. 目标是达到或优于旧 LAP10 的约 `0.167`；
4. predicted Gram 与 teacher Gram 的误差持续下降；
5. flow loss 保持稳定，不依靠更新 Expert 降低；
6. 三视角任一非主视角 dropout 后输出仍保持有限、稳定；
7. checkpoint 能在完全不加载 VLM 的进程中恢复并前向。

注意：训练集单批 loss 达标不等于闭环成功。完成 1000 step 后，先做固定
held-out 离线对比，再决定是否进行 RoboTwin 5 clean + 5 randomized 测试。

离线对比至少包括：

```text
VLM condition vs LAP10V3 condition：MSE、cosine、Gram error
同噪声下 Expert velocity/action 差异
左右 xyz、quaternion、gripper 分组 MAE 和 max error
```

不要只报告包含大量 padding/common 维度的全局 action cosine。

## 12. 建议代码与输出路径

新增或修改：

```text
starVLA/model/lap_stage2.py                 # 新增 LAP10V3
tools/build_lap10v3_teacher_stats.py        # teacher 位置统计
tools/train_lap10v3_ddp.py                  # 双卡、两阶段、1000 step
tools/compare_vlm_lap10v3_conditions.py     # held-out 离线对比
```

日志：

```text
logs/lap10v3_task14_1000step/train.log
```

输出：

```text
outputs/lap10v3_task14_1000step/
    lap10v3_step0000100.pt
    lap10v3_step0000300.pt
    lap10v3_step0000500.pt
    lap10v3_step0000750.pt
    lap10v3_step0001000.pt
```

checkpoint 必须保存：

```text
LAP10V3 state dict
teacher position mean 或其 stats 文件校验信息
optimizer / scheduler
global step / phase
全部训练参数
LAP6、LaWM、Expert 来源路径
init_mode=scratch_v3
代码版本或 git commit（若可用）
```

## 13. Luna 执行顺序

严格按以下顺序执行：

1. 生成并核验 teacher position statistics；
2. 实现独立 `LAP10V3`，不得复用 LAP8/LAP10V2 后四层权重；
3. 完成 state-dict 隔离测试和 shape smoke test；
4. 完成 32 样本过拟合测试；
5. 过拟合测试通过后，在 GPU 0、1 启动正式 1000-step FP32 训练；
6. 训练至 step 300 时核查防塌缩指标，再自动进入 Phase B；
7. step 1000 后执行 held-out 离线对比；
8. 离线验收通过后再提交 RoboTwin 闭环测试建议，不自动启动闭环测试。

本方案的首要判据不是总 loss，而是：

```text
284 个位置是否保持不同身份，并且这种结构是否与 teacher 一致。
```
