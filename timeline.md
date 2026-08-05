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
- 另建独立的 Expert 条件模块（暂称 `SEC284`）：284 个 learned query 直接 cross-attend 三视角的 768 个 DINO token，输出 `[B,284,768]`。
- SEC284 可以接收 EEF 和 LAP6 latent 作为辅助条件，但不应只依赖 32-D latent，以免丢失场景位置、物体状态和操作阶段信息。
- 训练计划：先使用离线缓存的 VLM 284-token 表示预训练 SEC284，再使用真实 action 监督小学习率联合调整 SEC284 与 Expert 后层。部署时不加载 VLM。

## 后续维护规则

每次新增训练或 RoboTwin 评估后，追加：

1. 日期与实验 ID；
2. 初始化 checkpoint、可训练模块和主要 loss；
3. 训练是否正常结束；
4. 同口径 clean/randomized 成功率；
5. 下一步决策及其依据。

不要把训练 loss 或 token 对齐指标单独写成“任务成功”。
