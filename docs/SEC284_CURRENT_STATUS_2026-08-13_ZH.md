# SEC284 当前状态与原始证据索引

> 更新时间：2026-08-13（Asia/Hong_Kong）
> 仓库：`cpx196/LAP-lawam`
> 任务：RoboTwin `move_pillbottle_pad`（Task 14）

## 1. 一句话结论

SEC284 已经能够学习固定任务下 VLM condition 的公共语义和大部分样本动态，但 condition 的动态残差仍不完整，并会在 Action Expert 的 flow 后段放大。当前 Expert grid-KD 的经验 best 是 500 step；严格 clean 10x 仍为 `0/10`，因此不能把训练 loss 或 cosine 单独写成闭环成功。

## 2. 当前架构和边界

```text
三视角 DINO latent ── LAP6 ── LaWM ── visual subgoal ─┐
                                                       ├─ Action Expert ── action chunk
三视角 DINO latent ── SEC284 ── [B,284,768] condition ┘
```

- SEC284-L：8 层、hidden 768、12 heads、FFN 3072，约 76.6M 参数。
- SEC284 只读取三视角 DINO latent；不输入 EEF、action、LAP6 输出、LaWM 输出或运行时语言。
- LAP6、LaWM、Action Expert 和在线 VLM 的冻结边界以各训练脚本和日志为准。
- 当前所有 SEC284 训练均针对固定 instruction：

```text
Use the left arm to pick and place the orange bottle for pills or liquid onto the pad.
```

## 3. 表征差异：SEC284 vs VLM

来自 1,749 个 held-out test 样本的 `step-003000.pt`：

| 指标 | 数值 | 解释 |
|---|---:|---|
| raw condition MSE | 0.060777 | 原始 `[284,768]` condition 误差 |
| whitened MSE | 0.025876 | 按 teacher 维度尺度归一后的误差 |
| token cosine | 0.955959 | 整体方向相似，公共任务语义已学到 |
| mean-only baseline MSE | 0.220543 | 始终输出平均 condition 的基线 |
| shuffle-teacher MSE | 0.387414 | 样本错配后误差，证明输出保留样本对应信息 |
| dynamic R² | 0.724421 | 相比 mean-only，约 72.4% 的跨样本动态被恢复 |
| teacher cross-sample std RMS | 0.466675 | VLM 的跨样本动态幅度 |
| SEC284 cross-sample std RMS | 0.406054 | SEC284 的跨样本动态幅度 |
| student/teacher std ratio | 0.870100 | 动态幅度约比 VLM 小 13% |

这说明 SEC284 不是完全塌缩，但存在向平均 condition 收缩的问题。cosine 高主要反映固定任务语义接近；动态 R² 和 std ratio 才反映“是否随当前画面正确改变”。

## 4. 三类训练结果

### 4.1 Frozen-Expert behavior KD（SEC284 更新）

```text
outputs/sec284_frozen_expert_behavior_kd_2000step/step-002000.pt
```

训练末段日志：representation total 约 `0.062031`，raw MSE `0.050708`，whitened MSE `0.021833`，behavior KD `0.002951`，动态 std ratio `0.924146`。这是训练 batch 统计，不等价于 held-out 或闭环成功率。

### 4.2 Output-primary inference-grid KD（SEC284 更新）

```text
outputs/sec284_output_kd_primary_2000step/step-002000.pt
```

训练末段：`repr=0.062496`、`grid_kd=0.000812`、`std_ratio=0.9209`。2000 step 的日志显示 grid 项已平台化；当前尚未对该 checkpoint 重跑完整 held-out `k=0..9` 验证，因此不能据此证明它优于 behavior-KD。

### 4.3 Expert inference-grid KD（Expert 更新）

从 500 step checkpoint 继续到总计 2000 step：

```text
outputs/sec284_expert_grid_kd_500step/train.log
outputs/sec284_expert_grid_kd_2000step/train.log
```

采样 grid step 的训练日志均值：

| step 区间 | `grid_kd` 均值 |
|---|---:|
| 1–500 | 0.000905 |
| 501–1000 | 0.000861 |
| 1001–1500 | 0.000893 |
| 1501–2000 | 0.000871 |

这是 teacher-forcing 的局部 velocity MSE，不是完整 rollout 指标；因此当前以视频观察和固定评测选择 500 step，而不是按最低 loss 选择。

## 5. 闭环结果

所有以下 SEC284 评测均为 LAP6 + SEC284 no-VLM、`replan=36`、`move_pillbottle_pad`，视频未纳入 Git，但本页引用的 JSON、`eval.log`、`run.log`、`meta.json` 和 `_result.txt` 已保留。

| Expert checkpoint | 评测 | 结果 | 备注 |
|---|---|---:|---|
| 500 | clean 1x | 0/1 | 有接触、夹持、短暂抬升/搬运尝试 |
| 500 | clean 800-step 1x | 0/1 | 执行到 800/800 |
| 1000 | clean 1x | 0/1 | 更早失稳 |
| 1500 | clean 1x | 0/1 | 未成功 |
| 2000 | clean 1x | 0/1 | 未成功 |
| 500 | clean 10x | **0/10** | 10 个 episode 均执行到 400/400 |

500-step clean 10x 的 seeds 为：`100002, 100005, 100006, 100007, 100008, 100009, 100010, 100011, 100013, 100015`。

正式汇总：

```text
results/eval_runs/sec284_expert_grid_kd_step500_clean10/clean/lawam_robotwin_sft_release__demo_clean/20260812_162222/tasks/move_pillbottle_pad/summary.json
```

## 6. VLM shadow trace 的动作层差异

在真实 VLM 成功控制的轨迹上，SEC284 只做 shadow，不影响 teacher rollout；固定 instruction 下 clean 5 个 replan、randomized 4 个 replan：

| 指标 | clean 均值 | randomized 均值 |
|---|---:|---:|
| condition MSE | 0.116944 | 0.129082 |
| condition cosine | 0.915296 | 0.906149 |
| action MSE | 0.011049 | 0.011025 |
| XYZ MSE | 0.001826 | 0.001059 |
| gripper sign agreement | 0.941667 | 0.944444 |
| 10-step grid velocity MSE | 0.034429 | 0.031110 |

最大偏差发生在少数重规划点：clean call 4 的 velocity MSE 为 `0.137789`、gripper sign agreement `0.791667`；randomized call 3 的 velocity MSE 为 `0.113340`、gripper sign agreement `0.819444`。按 flow step 看，误差从前段向后段放大，说明平均 condition 相似不等于 Expert rollout 行为等价。

## 7. 2026-08-13 同 seed 随机环境对照

在 `demo_randomized`、seed `100001`、`replan=36` 下，原始 VLM 和 LAP6（仍使用官方 VLM condition，仅替换 LAP/LaWM 路径）均成功：

| 变体 | 结果 | steps |
|---|---:|---:|
| original LaWAM | 1/1 | 139 |
| LAP6 + official VLM | 1/1 | 141 |

这不是 SEC284 的成功结果，但说明该 seed/随机环境并非必然失败；后续 SEC284 randomized 评测应使用同一 seed 和同一服务端口径做对照。

## 8. 原始数据保留策略

本次提交保留以下原始文本数据（按原工作区路径）：

- `outputs/sec284_*/` 下 SEC284 训练日志、评估 JSON/JSONL 和 teacher-cache 构建日志；
- `results/eval_runs/sec284*/` 下 SEC284 评测的 `eval.log`、server 日志、`meta.json`、`run.log`、`summary.json` 和 `_result.txt`；
- 外部模型对照日志 `EXTERNAL_MODEL_ROBOTWIN_RANDOM_LOG_2026-08-12.md`。

以下内容不提交：RoboTwin 视频（`*.mp4`）、模型 checkpoint（`*.pt`）、大型 feature/teacher cache、二进制 shadow trace、图片和训练曲线 PNG。它们仍保留在本地工作区，路径和生成命令记录在对应文档中。

## 9. 当前下一步

1. 对 behavior-KD、output-primary 和 500-step Expert checkpoint 使用完全相同的 held-out 样本，按 `k=0..9` 报告 velocity MSE、动作维度误差、dynamic R² 和 std ratio。
2. 只有固定验证优于 baseline 后，才把某个 checkpoint 作为下一轮同 seed 闭环候选。
3. 继续把抓取、抬升、搬运和放置分阶段统计；不要只用总体 cosine 或总 train loss 选模型。
