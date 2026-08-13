# LaWAM / RoboTwin 实验总账本（Task 14）

> 记录日期：2026-08-05
> 工作目录：/data/pxchen/LaWAM
> 目的：把本条实验链上的训练、离线诊断、闭环成功和失败结果集中记录，避免将不同模型、不同监督口径和不同推理路由混为一谈。

---

## 1. 阅读规则

本账本将结果分成三类：

1. **工程成功**：进程正常结束、checkpoint 保存成功。
2. **离线学习成功**：loss、MAE、latent 或 token 指标改善。
3. **闭环任务成功**：RoboTwin 的 summary.json 中明确记录成功。

工程成功不等于离线学习成功；离线学习成功也不等于闭环任务成功。

“待测”表示训练已结束或正在进行，但没有可靠的同口径闭环结果；“失败”必须说明是训练失败、闭环失败，还是测试口径未通过。

---

## 2. 训练实验主表

| ID | 实验 | 主要监督 | 数据量 / step | 最终指标 | 工程状态 | 判断 |
|---|---|---|---:|---|---|---|
| T0 | Stage-1 单视角 LAP60M | IDM latent、LaWM world、EEF | 24,140 / 3,000 | test latent cosine 约 0.8983；EEF position MAE 约 0.01179 m | 完成 | 离线成功 |
| T1 | Stage-1 三视角联合 | 三视角 latent + LaWM + EEF | 24,140 / 3,000 | loss 0.26903；latent 0.03865；EEF 0.05757 | 完成 | 训练链路成功 |
| T2 | LAP8 Phase 1 | 真实 action flow；无 VLM | 24,140 / 1,000 | loss 0.017117；flow 0.014958；diversity 0.215905 | 完成 | 接口预热成功，闭环失败 |
| T3 | LAP10 alignment | VLM teacher cache + 284 token alignment + flow | 8,192 / 1,000 | loss 0.184732；align MSE 0.167401 | 完成 | 离线对齐成功，不能代替闭环 |
| T4 | LAP10V2 scratch | 从零拟合 teacher token | 8,192 / 1,000 | loss 0.821989；align MSE 0.744432；diversity 0.996454 | 完成 | 失败，表示塌缩 |
| T5 | LAP10V2 unified | 统一结构拟合 teacher token | 8,192 / 1,000 | loss 0.773176；align MSE 0.696766；diversity 0.996431 | 完成 | 略好但仍失败 |
| T6 | LAP10V3 scratch | 四层后端拟合 teacher token | 8,192 / 1,000 | loss 0.189358；align MSE 0.159223；structure 0.025719 | 完成 | 离线有效，闭环不稳定 |
| T7 | LAP10V3 + Expert joint | 真实 flow + teacher alignment | 8,192 / 2,000 | loss 0.026365；flow 0.007548；align MSE 0.158732 | 完成 | 闭环 4/10，旧 baseline |
| T8 | LAP10V3 AR-2000 | 真实 action only；无 VLM/teacher cache | 24,140 / 2,000 | final loss 0.013269；flow 0.011800；AR-B LAP/Expert 梯度均非零 | 完成 | 闭环 1/10，低于 T7 |
| T9 | FlowOnly-1000 | T7 初始化；LAP/LaWM 冻结；仅官方 flow | 24,140 / 1,000 | final flow 0.004960；128 样本 action MSE 0.003641 | 完成 | 闭环 1/6，未超过 T7 同 seed 2/6 |

证据路径：

~~~text
T0  outputs/lap_stage1_task14/eval_step3000.json
T1  outputs/lap_stage1_task14_3view_joint_3000step/train.log
T2  logs/lap8_phase1_task14_1000step/train.log
T3  logs/lap10_alignment_task14_1000step/train.log
T4  logs/lap10v2_scratch_task14_1000step/train.log
T5  logs/lap10v2_unified_task14_1000step/train.log
T6  logs/lap10v3_task14_1000step/train.log
T7  logs/lap10v3_expert_joint_task14_2000step/train.log
T8  logs/lap10v3_ar_expert_task14_2000step/train.log
~~~

---

## 3. T0/T1：Stage-1 表征学习

### T0：单视角 LAP60M

最终 test eval 的关键指标：

~~~text
latent_cosine       0.898334
eef_position_mae    0.011790 m
eef_gripper_mae     0.056967
eef_quaternion_err  3.9723 deg
shuffle_increase    47.496%
zero_increase       32.976%
~~~

train split 的 latent MSE 约 0.03612、latent cosine 约 0.91395。结果说明 LAP latent 对视觉和 EEF 输入敏感，且不是简单常数。但 Stage-1 不产生最终 Action Expert action，不能直接报告 Robotwin 成功率。

### T1：三视角联合

三视角日志中 lap_grad 和 lawm_grad 都非零，说明 LAP 和 LaWM 都实际参与更新。最后日志点：

~~~text
step=3000
loss=0.26903
latent=0.03865
eef=0.05757
diversity=0.26339
~~~

结论：三视角训练链路成功；该日志没有对应的闭环成功率，因此不能写成“任务完成”。

---

## 4. T2：LAP8 Phase 1

训练配置为完整 24,140 train samples，1000 optimizer step，LaWM 冻结，Action Expert 直接接 LAP8 条件，无 VLM。

最终训练日志：

~~~text
loss=0.017117
flow=0.014958
diversity=0.215905
cond_rms=0.997476
~~~

工程状态：正常结束并保存 step 1000 checkpoint，无 NaN、Inf 或异常退出。

对应闭环：

~~~text
clean：      0/5
randomized： 0/5
合计：       0/10
~~~

证据目录：

~~~text
results/eval_runs/lap8_phase1_no_vlm_5x_clean_random_fp32/
~~~

结论：T2 说明“低 flow loss + 训练链路正常”不足以证明 LAP 已替代 VLM。失败属于闭环失败，不是训练进程失败。

---

## 5. T3：LAP10 alignment

T3 使用 VLM teacher cache 和 284 token 对齐损失。最终训练日志：

~~~text
loss       0.184732
flow       0.005360
align_mse  0.167401
align_cos  0.096450
diversity  0.232594
~~~

8 个 held-out 样本的匹配条件诊断：

~~~text
token_cosine mean  0.905334
token_mse mean      0.164559
matched_action_cos  0.999111
matched_action_mse  0.001300
~~~

这证明在 matched VLM subgoal 和相同 diffusion noise 下，动作可以高度一致；不能证明替换 VLM 后，在重规划、观测变化和 randomized 场景中仍然稳定。

---

## 6. T4/T5：LAP10V2 失败对照

### T4：scratch

~~~text
loss       0.821989
align_mse  0.744432
align_cos  0.474411
diversity  0.996454
~~~

### T5：unified

~~~text
loss       0.773176
align_mse  0.696766
align_cos  0.473762
diversity  0.996431
~~~

两次实验 diversity 都接近 1，说明 token 表示没有形成有效结构；loss 停在约 0.77–0.82，不能称为可用收敛。T5 比 T4 数值略低，但没有解决核心问题。这两次应作为失败对照保留。

---

## 7. T6：LAP10V3 scratch

最终指标：

~~~text
loss       0.189358
flow       0.015315
align_mse  0.159223
align_cos  0.122477
structure  0.025719
~~~

相比 T4/T5，离线 token 和 structure 指标明显改善，说明模型确实学到了可优化的表示。但此前 Robotwin 运行不能稳定完成任务。

该实验是关键反例：

~~~text
离线 alignment loss 较低
不等于
closed-loop task success 较高
~~~

可能涉及条件分布、重规划时序、Expert 接口、动作 chunk 偏移和推理路由，而不只是训练 loss。

---

## 8. T7：LAP10V3 + Action Expert joint

T7 最终训练日志：

~~~text
loss        0.026365
flow        0.007548
align_mse   0.158732
align_cos   0.121849
structure   0.025358
lap_grad    0.0556
expert_grad 0.2365
~~~

对应 RoboTwin 5+5：

~~~text
clean：      1/5 = 20%
randomized： 3/5 = 60%
合计：       4/10 = 40%
~~~

证据：

~~~text
results/eval_runs/lap10v3_expert_joint_2000step_5x_clean_random_fp32/
~~~

T7 是当前 AR 的初始化 checkpoint 和旧 baseline。它的 loss 很低，但成功率只有 40%，进一步说明不能用 loss 单独作为最终评价。

---

## 9. T8：AR-2000 当前主实验

T8 的严格口径：

~~~text
无 Qwen/VLM
无 VLM teacher cache
无 teacher Action Expert
无 token MSE/cosine/structure distillation
完整 24,140 train samples
真实 action flow + reconstruction + delta
AR-A step 1–1000：Expert 先适配，LAP 有效冻结
AR-B step 1001–2000：解冻 LAP9–10/output head
~~~

截至本账本更新时：

~~~text
step 2000 checkpoint 已保存
AR-B 已完成
lap_grad > 0
expert_grad > 0
无 NaN/Inf/OOM
~~~

证据：

~~~text
logs/lap10v3_ar_expert_task14_2000step/train.log
outputs/lap10v3_ar_expert_task14_2000step/
~~~

T8 已使用 step 2000 checkpoint 按与 T7 相同的 5 clean + 5 randomized seeds 完成闭环评估：clean 1/5，randomized 0/5，合计 1/10。唯一成功样本为 clean seed 100002，149 steps；其余 9 条均运行至 400 steps 后失败。

该结果比 T7 的 4/10 下降 3 个成功样本，尤其 randomized 从 3/5 降至 0/5。它说明 AR 的训练 loss 收敛和真实 action 监督本身不足以保证闭环性能；AR 训练还改变了成功样本分布（T7 clean 成功 seed 为 100006，T8 为 100002）。

---

## 10. RoboTwin 闭环结果总表

### 10.1 有 summary.json 证据的结果

| 配置 | clean | randomized | 合计 | 解释 |
|---|---:|---:|---:|---|
| 官方 VLM baseline | 5/5 | 5/5 | 10/10 | 独立 5+5 参考上限 |
| LAP8 Phase 1 no-VLM | 0/5 | 0/5 | 0/10 | 闭环失败 |
| T7 LAP10V3 + Expert joint | 1/5 | 3/5 | 4/10 | 部分成功，旧 baseline |
| 独立 no-VLM ablation | 0/5 | 0/5 | 0/10 | 闭环失败 |
| T8 AR-2000 | 1/5 | 0/5 | 1/10 | 低于 T7；离线收敛未转化为闭环提升 |

证据目录：

~~~text
官方 / no-VLM：
results/eval_runs/lap10_no_vlm_vs_official_5x_clean_random_fp32/

LAP8：
results/eval_runs/lap8_phase1_no_vlm_5x_clean_random_fp32/

T7：
results/eval_runs/lap10v3_expert_joint_2000step_5x_clean_random_fp32/

T8：
results/eval_runs/lap10v3_ar_2000step_5x_clean_random_fp32/
~~~

### 10.2 早期 A/B/C ablation 结果

目录：

~~~text
results/eval_runs/lap_lawm_ablation/abc_5x_clean_random_fp32/
~~~

summary.json 记录：

~~~text
A baseline：    clean 5/5，randomized 5/5
B official LAP：clean 5/5，randomized 5/5
C joint LAP：   clean 5/5，randomized 5/5
~~~

这些结果必须保留，但不能未经审计直接和后面的独立 no-VLM 0/10 合并为一个结论。两组结果可能使用了不同的 checkpoint、输入路由、replan 配置、seed 或 policy bridge。后续正式比较必须统一：

~~~text
checkpoint
task config
clean/randomized seed
replan 频率
action horizon
推理 dtype
VLM/LAP 条件路由
~~~

因此本账本将 A/B/C 标为“日志记录为成功，条件待复核”，而不是无条件宣布模型 100% 成功。

### 10.3 T9 FlowOnly 配对子集

~~~text
T9 FlowOnly step 1000：clean 0/3，randomized 1/3，合计 1/6
T7 相同 seeds：       clean 1/3，randomized 1/3，合计 2/6
AR-A step 1000 相同 seeds：clean 1/3，randomized 0/3，合计 1/6
~~~

T9 恢复了 T7 的 randomized seed `100001`，却丢失了 T7 的 clean seed `100006`。这说明 3+3 总成功数之外，成功 seed 的转移也必须被记录。

128 个 validation 样本使用相同 flow noise 和 10-step sampler 的离线检查：

~~~text
                    T7          T9 FlowOnly
action MSE16        0.003820    0.003641
left xyz MAE        0.015983    0.015917
left gripper MAE    0.044364    0.042425
final left xyz MAE  0.014724    0.014677
~~~

T9 在这些离线指标上略优于 T7，但闭环没有提升，构成“离线 action/flow 误差改善不等于闭环改善”的直接证据。

---

## 11. 实验结论的正确表述

目前可以确定：

~~~text
1. Stage-1 LAP 学到了有效视觉/EEF latent。
2. LAP8 Phase 1 工程成功，但无 VLM 闭环为 0/10。
3. LAP10V2 scratch/unified 出现明显塌缩。
4. LAP10V3 scratch 离线 alignment 改善，但闭环不稳定。
5. LAP10V3 + Expert joint 完成训练，闭环为 4/10。
6. 官方 VLM 独立 5+5 为 10/10，独立 no-VLM 为 0/10。
7. AR-2000 已完成训练和同口径闭环测试，结果为 clean 1/5、randomized 0/5、合计 1/10，低于 T7 的 4/10。
8. FlowOnly-1000 完成，离线 action MSE 略优于 T7，但同 seed RoboTwin 3+3 仅 1/6，未超过 T7 的 2/6。
~~~

目前不能确定：

~~~text
1. 早期 A/B/C 的 10/10 是否与独立 no-VLM 测试完全同口径。
2. 为什么 AR-2000 在 randomized 上从 T7 的 3/5 退化到 0/5。
3. 需要怎样的时序、闭环或 rollout 监督，才能把离线 action loss 的改善转化为闭环成功率提升。
4. 独立 SEC284 是否能在不扰动 LAP6→LaWM 路径的前提下，替代 VLM 给 Expert 的 284-token 条件。
~~~

---

## 12. 后续追加规则

每次新实验完成后，必须记录：

~~~text
实验 ID
日期和时间
训练脚本和完整参数
checkpoint 路径
VLM/teacher 是否使用
数据量、有效 epoch 和 dtype
最终 loss 分解
clean/randomized 成功率
日志和 summary.json 路径
成功/失败/未完成判断
与上一个 baseline 的差异
~~~

如果只完成训练但还没有闭环测试，状态写“训练完成，闭环待测”；如果环境或进程异常，另写“工程失败”，不要直接写成“模型失败”。

---

## 13. 2026-08-13：SEC284 当前阶段归档

LAP10/LAP10V3 的历史实验记录继续保留在本账本中；当前主线已转为 LAP6 + SEC284。最新状态、原始日志和 JSON 证据索引见：[SEC284 当前状态与原始证据索引](SEC284_CURRENT_STATUS_2026-08-13_ZH.md)。

摘要：

~~~text
SEC284 表征 held-out：raw MSE 0.060777，cosine 0.955959，dynamic R² 0.724421，std ratio 0.8701。
Frozen behavior-KD、output-primary inference-grid KD 已完成；Expert-only grid-KD 从 500 续训至总计 2000 step。
Expert-only 500/1000/1500/2000 的 clean 1+1 均为 0/1；500-step clean 10x 为 0/10。
2026-08-13 同一 randomized seed 100001 下，原始 VLM 与 LAP6+官方 VLM 各为 1/1。
~~~

500-step 只是视频动作质量上的经验候选，不是已验证成功模型；同 seed VLM 对照用于排除“随机环境必然失败”，不代表 SEC284 成功。
