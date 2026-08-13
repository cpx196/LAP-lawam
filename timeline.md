# LaWAM × RoboTwin Timeline

> 本文件记录本工作区内与 RoboTwin 无 VLM 路线有关的可复现实验节点。  
> 详细指标、日志和证据目录请见 [docs/LAP10V3_EXPERIMENT_LEDGER_ZH.md](docs/LAP10V3_EXPERIMENT_LEDGER_ZH.md)。

## 2026-08-05：RoboTwin 数据与 Stage-1

- 确认单任务研究对象为 `move_pillbottle_pad`（Task 14）。
- 建立 train split：480 条轨迹（440 randomized、40 clean），共 24,140 个当前观测—未来 action chunk 样本。
- 完成 Stage-1 单视角 LAP60M（T0）：离线 latent cosine 约 `0.8983`，证明视觉/EEF 到 latent action 的表征有效。
- 完成 Stage-1 三视角联合训练（T1）：主视角、左腕、右腕特征均进入 LAP；LAP 与 LaWM 均有梯度。

## 2026-08-05：替代 VLM 条件的探索

- 完成 LAP8 Phase 1（T2）：以 8 个 LAP token 替代 VLM 条件。训练正常但 RoboTwin 闭环为 `0/10`。
- 完成 LAP10 alignment（T3）：将条件长度扩展至 284 token 并对齐 VLM 条件。离线 token 对齐改善，但不能直接带来闭环成功。
- 完成 LAP10V2 scratch/unified（T4/T5）：从零拟合 284 token 时出现明显塌缩，未进入有效闭环。
- 完成 LAP10V3 scratch（T6）：四层 284-token 条件器恢复可用离线对齐，闭环仍不稳定。

## 2026-08-05：当前无 VLM baseline

- 完成 LAP10V3 + Action Expert 联合训练（T7）。
- 同口径 RoboTwin 5 clean + 5 randomized：
  - clean：`1/5`
  - randomized：`3/5`
  - 合计：`4/10`
- T7 作为后续真实 action 微调和 FlowOnly 训练的初始化 checkpoint。

## 2026-08-05：AR（真实 action）训练与诊断

- 完成 AR-2000（T8）：全量 24,140 样本、无 VLM teacher、无 token alignment。
- AR-A（step 1–1000）只更新 Action Expert；AR-B（step 1001–2000）再更新 LAP9–10/output head。
- step 2000 闭环结果：clean `1/5`、randomized `0/5`、合计 `1/10`。
- 单独评估 AR-A step 1000：同样为 clean `1/5`、randomized `0/5`、合计 `1/10`。
- 诊断结论：退化已经发生在 Expert-only 的 AR-A，不是 LAP9–10 解冻造成；下一步应检验 AR 的自定义 action 加权与辅助 loss。

## 2026-08-05：FlowOnly-1000（T9）

- 从 T7 恢复。
- 全冻结 LAP 与 LaWM，只更新 Action Expert。
- 不使用 VLM、teacher cache、token alignment、夹爪/时刻权重、reconstruction loss 或 delta loss。
- 仅使用 actions_mask 下的官方标准 flow matching loss。
- 单卡 FP32，1000 step，effective batch 16，保存 step 250/500/750/1000。
- 1000 step 正常结束，耗时约 12.8 分钟，无 NaN/OOM；最终 flow loss 约 `0.004960`，但移动平均未呈现明显持续下降。
- RoboTwin 3 clean + 3 randomized：clean `0/3`，randomized `1/3`，合计 `1/6`。
- 同一 3+3 seed 子集上，T7 为 `2/6`，AR-A step 1000 为 `1/6`。FlowOnly 没有超过 T7，但成功 seed 发生了变化。
- 固定同一 flow noise 的 128 样本离线比较中，FlowOnly 的 action MSE 从 T7 的 `0.003820` 小幅降至 `0.003641`，但闭环成功率未提高。
- 结论：当前主要矛盾是离线 action/flow 代理指标与闭环成功不一致，而不是训练未更新或 loss 未降低。

## 2026-08-05：架构决策——解耦 LAP6 与 Expert 条件分支

- 后续不再沿用 LAP7–10 作为 LAP6 的串联后端。
- 保留已验证的 `LAP6 → 32-D latent action → LaWM` 路径。
- 另建独立的 Expert 条件模块 `SEC284`：284 个 task-specific learned query 直接 cross-attend 三视角 DINO token，输出 `[B,284,768]`。
- SEC284 不接收 EEF、`z_lap` 或 LAP6 scene token；EEF 由 Action Expert 的独立 proprioception 通道接收。
- 当前阶段先独立蒸馏官方 VLM 在 Expert 输入边界处的 284-token condition，不引入 action、EEF、LAP6/LaWM 或 Expert loss。

## 2026-08-11：SEC284 设计定稿

- VLM 替代模块统一命名为 `SEC284`（Semantic Expert Conditioner, 284 tokens），不再使用其他模块名称。
- 当前单任务输入合同固定为三视角 DINO latent；任务语义由 284 个 learned queries 固化，不增加运行时 language encoder；输出固定为 `[B,284,768]`，兼容官方 Action Expert。
- 明确禁止 `teacher_position_mean + residual`、EEF shortcut 和 LAP6 latent shortcut。
- 本轮固定使用 SEC284-L：8 层、hidden 768、12 heads、FFN 3072、约 76.6M 参数，不再并行训练 B/XL 规格。
- 日志复核确认已有约 39M 的 LAP10/LAP10V3 条件头只达到约 `0.159` condition MSE；新训练先增加 64/256-sample 可记忆性测试，再使用完整 `24,140/1,749/1,749` train/val/test teacher cache。
- 本阶段只以 held-out VLM condition 指标选择 checkpoint；达到表示验收门槛后再单独设计 Expert 接入和部署。
- 完整方案见 [SEC284 设计文档](docs/VLM_REPLACEMENT_JOINT_TRAINING_DESIGN_ZH.md)。

## 2026-08-11：SEC284 冻结 Expert 第一阶段

- 纯表示蒸馏 3000 step 得到 raw MSE `0.060777`、cosine `0.955959`、dynamic R² `0.724421`、std ratio `0.8701`。
- 固定 LAP6、LaWM、Action Expert，只更新 SEC284；action 仅作 flow/velocity 监督，不作为 SEC284 输入。
- 使用相同 noise/time 的 teacher/student Expert velocity KD，加表示锚点、动态方差约束及 `0.25` 归一化 flow loss。
- 训练配置为 2000 step、每 500 step 保存、local batch size 32；四卡 global batch 128，两卡回退 global batch 64。

## 2026-08-12：SEC284 Expert inference-grid KD

- 固定 SEC284、LAP6、LaWM 和 `enc_vlm`，只训练官方 Action Expert；使用 uniform inference-grid velocity KD，4 卡 `1,3,4,5`，local batch 8、global batch 32、学习率 `1e-7`。
- 首轮 500 step 完成，保存 `outputs/sec284_expert_grid_kd_500step/step-000500.pt`；训练 `grid_kd` 从约 `0.00104` 的早期区间降到约 `0.00090`，但该指标不是闭环成功指标。
- 从 500 step 续训到总计 2000 step；续训使用绝对 step offset，已正常完成并保存 1000/1500/2000 部署 checkpoint。
- 固定 LAP6 + SEC284 no-VLM、clean、seed `100002` 的 1+1 闭环：500 step 和 1000 step 均为 `0/1`，但视频显示 500 step 已出现接触、夹持和短暂抬升/搬运尝试；1000 step 更早失稳并碰倒瓶子。
- 当前经验上的 best checkpoint 是 500 step。结论再次确认：不要按总 train loss 单独选择 checkpoint；teacher-forcing velocity MSE 与真实闭环存在 objective mismatch。randomized 本轮暂不评估。
- 详细状态、输出目录和后续建议见 [SEC284 grid-KD handoff](docs/HANDOFF_2026-08-12_01_SEC284_GRID_KD_ZH.md)。

## 2026-08-12～13：SEC284 结果归档与同 seed 随机对照

- SEC284 表征 held-out 结果归档：raw MSE `0.060777`、whitened MSE `0.025876`、cosine `0.955959`、dynamic R² `0.724421`、std ratio `0.8701`。这表示公共任务语义接近，但跨画面动态仍比 VLM 小约 13%。
- Frozen behavior-KD 末段 batch 指标约为 raw MSE `0.050708`、behavior KD `0.002951`、std ratio `0.9241`；output-primary 2000-step 末段 `repr=0.062496`、`grid_kd=0.000812`、`std_ratio=0.9209`。两者均需固定 held-out 验证，不能只凭训练日志选 checkpoint。
- Expert-only inference-grid KD 从 500 续训到总计 2000 step。500/1000/1500/2000 的 clean 1+1 均为 `0/1`；500-step clean 10x 为 `0/10`。500-step 视频中仍观察到接触、夹持和短暂抬升/搬运迹象，因此保留为经验候选而非成功模型。
- 真实 VLM 成功轨迹 shadow trace 的 clean 平均 action MSE 为 `0.011049`、grid velocity MSE 为 `0.034429`、gripper sign agreement 为 `0.941667`；少数重规划点的 flow 后段误差显著放大。
- 2026-08-13 在同一 randomized seed `100001` 下，原始 VLM 和 LAP6+官方 VLM 各为 `1/1`（139/141 steps），说明随机环境不是该任务失败的充分原因；该结果不是 SEC284 成功率。
- 原始 SEC284 训练/评测日志、JSON/JSONL 和 RoboTwin元数据已按原路径归档到 Git；视频、checkpoint、cache、二进制 trace 和图片仍排除。详见 [SEC284 当前状态与原始证据索引](docs/SEC284_CURRENT_STATUS_2026-08-13_ZH.md)。

## 后续维护规则

每次新增训练或 RoboTwin 评估后，追加：

1. 日期与实验 ID；
2. 初始化 checkpoint、可训练模块和主要 loss；
3. 训练是否正常结束；
4. 同口径 clean/randomized 成功率；
5. 下一步决策及其依据。

不要把训练 loss 或 token 对齐指标单独写成“任务成功”。
