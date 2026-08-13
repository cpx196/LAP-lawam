# 外部模型 RoboTwin randomized 对照日志

日期：2026-08-12  
用途：为 LaWAM 后续 RoboTwin 评估记录本地 TurboVLA、LingBot-VA 的 randomized 运行情况。

## 结论摘要

本次严格复核得到的结果是：

| 模型 / 流程 | 配置 | 任务数 | 成功数 | 成功率 |
|---|---|---:|---:|---:|
| TurboVLA 本地已有运行 | `demo_randomized` | 5 | 0 | 0/5 |
| LingBot-VA 旧版运行 | `demo_randomized` | 2 个完成，1 个中断，1 个未开始 | 0 | 0/2（不作为完整 4-task 结果） |
| LingBot-VA 严格复核 | `demo_randomized`，seed 0，即 RoboTwin `stseed-10000` | 4 | 4 | 4/4 |
| LingBot-VA 严格 clean 对照 | `demo_clean`，同 seed | 4 | 4 | 4/4 |

当前最重要的判断：randomized 环境本身不是必然导致 LingBot-VA 失败。恢复原版客户端结算、使用严格推理环境并将 VAE/T5 改为 GPU 临时计算后，LingBot-VA 在同一组四个任务上 clean 和 randomized 都完成了 `4/4`。因此，之前 LingBot-VA 的 `0` 结果不能直接归因于模型能力，至少包含部署链路和环境版本差异。

这不是论文规模的 benchmark：每个任务只运行 1 个 episode，且 TurboVLA 与 LingBot-VA 的历史运行并非完全同一套服务端实现、checkpoint 记录和 seed 记录。下面的结果用于定位问题和保留实验轨迹，不应当解读为模型总体成功率。

## 任务和环境

严格 LingBot-VA 复核使用了四个共同任务：

- `adjust_bottle`
- `beat_block_hammer`
- `blocks_ranking_rgb`
- `blocks_ranking_size`

客户端配置：

- RoboTwin：`/data/pxchen/RoboTwin`
- RoboTwin commit：`2eeec322d95799f537cbfe5f291a8220d965ccb8`
- LingBot-VA 原版客户端逻辑，结算使用 `TASK_ENV.eval_success`
- server GPU：5；client GPU：6
- seed 参数：`--seed 0`，对应 `stseed-10000`
- 每个 task：`--test_num 1`
- instruction：客户端日志显示为 `seen`

strict LingBot-VA 推理环境：

- worktree：`/data/pxchen/lingbot-va-strict`
- Python：3.10.16
- PyTorch：2.9.0+cu126
- Transformers：4.55.2
- Diffusers：0.36.0
- checkpoint：`/data/pxchen/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- attention：`attn_mode=torch`
- VAE/T5：按需搬到 GPU 计算，完成后 offload 回 CPU；Transformer 常驻 GPU 5

randomized 配置日志明确显示：

```text
Messy Table: True
Random Background: True
Random Light: True
Random Table Height: 0.03
Random Head Camera Distance: 0
```

## LingBot-VA 严格复核结果

### randomized

结果目录：

`/data/pxchen/lingbot-va-strict/results/randomized_same_seed_gpu_offload2_20260812/stseed-10000/`

| 任务 | 结果 | 视频 |
|---|---|---|
| `adjust_bottle` | `1/1` | `visualization/adjust_bottle/0_Carefully_grab_the_green_bottle_with_indented_base_head-up_True.mp4` |
| `beat_block_hammer` | `1/1` | `visualization/beat_block_hammer/0_Pick_the_medium-sized_metal_hammer_from_the_table_and_strike_True.mp4` |
| `blocks_ranking_rgb` | `1/1` | `visualization/blocks_ranking_rgb/0_Arrange_red_block,_green_block,_and_blue_block_from_left_to_right_using_the_left_arm,_the_right_arm,_and_the_left_arm_in_the_order_red,_green,_blue._True.mp4` |
| `blocks_ranking_size` | `1/1` | `visualization/blocks_ranking_size/0_Take_large_block_with_the_left_arm,_medium_block_with_the_left_arm,_and_small_block_with_the_right_arm_to_the_middle._True.mp4` |

对应的 metrics 文件位于：

`/data/pxchen/lingbot-va-strict/results/randomized_same_seed_gpu_offload2_20260812/stseed-10000/metrics/<task>/res.json`

四个文件均为：

```json
{"succ_num": 1.0, "total_num": 1.0, "succ_rate": 1.0}
```

### clean 同 seed 对照

结果目录：

`/data/pxchen/lingbot-va-strict/results/clean_same_seed_gpu_offload2_20260812/stseed-10000/`

四个任务均为 `1/1`，对应 metrics 文件和视频均已保存。clean 与 randomized 使用相同的 `--seed 0` 起点；randomized 只切换了 RoboTwin 的 `demo_randomized` 环境配置。

## TurboVLA 本地 randomized 结果

已有结果目录：

`/data/pxchen/RoboTwin/eval_result/turbovla_random_5task/`

配置为 `demo_randomized`，共 5 个任务：

| 任务 | `_result.txt` 结果 |
|---|---:|
| `adjust_bottle` | `0.0` |
| `beat_block_hammer` | `0.0` |
| `blocks_ranking_rgb` | `0.0` |
| `blocks_ranking_size` | `0.0` |
| `click_alarmclock` | `0.0` |

因此这次本地 TurboVLA randomized 记录为：`0/5`。

对应视频位于各 task 目录下的：

`model2robotwin_interface/demo_randomized/turbovla_random_5task/<timestamp>/episode0.mp4`

TurboVLA 结果文件中记录的 instruction type 为 `seen`。该结果是已有本地运行记录；当前日志没有将它与 strict LingBot-VA 复核完全统一到同一 GPU 分配、同一服务端实现和同一 seed 证据链，因此这里只记录观测结果，不据此做严格模型排名。

## LingBot-VA 旧版 randomized 结果

旧结果目录：

`/data/pxchen/lingbot-va/results/lingbotva_robotwin_same4_random_20260812/`

旧流程中已完成并落盘的两个任务：

| 任务 | 结果 |
|---|---:|
| `adjust_bottle` | `0/1` |
| `beat_block_hammer` | `0/1` |

`blocks_ranking_rgb` 只有运行日志和部分过程，没有最终 `Success rate`；`blocks_ranking_size` 没有开始。因此旧流程不能记为完整的 `0/4`，准确表述是“已完成的 2 个任务为 `0/2`，其余任务未完成”。

旧流程不适合作为模型能力基线，原因包括：

1. 客户端不是当前恢复的原版结算逻辑，曾加入额外的 settle/tail 处理。
2. 旧环境中 Transformers 版本为 5.2.0，而仓库推理要求为 4.55.2。
3. 服务端和客户端代码包含临时部署修改，不能与官方原版路径直接等同。
4. 旧流程不是完整的四任务对照，且后续任务被中止。

## 现阶段定位结论

- TurboVLA：本地 randomized 5-task 运行确实是 `0/5`，需要单独继续核对其 checkpoint、action mapping、推理频率和客户端接口；当前不能用 LingBot-VA 的严格复核结果替代 TurboVLA 的诊断。
- LingBot-VA：旧版 `0` 结果主要暴露了部署链路不一致；在 strict 环境、原版客户端结算、GPU offload 后，四个共同 randomized 任务全部成功。
- 环境随机化：本次 `demo_randomized` 确认开启了 clutter/background/light/table-height 随机化，不是仅仅换背景。
- LaWAM 后续对比：建议所有模型至少统一记录 `task_config`、seed、instruction type、checkpoint、server/client commit、每 task 的 episode 数、metrics 路径和视频路径；在达到相同 episode 数前，不把这些 smoke test 汇总成论文意义上的成功率。

## 原始日志索引

- Strict LingBot-VA randomized logs：`/data/pxchen/lingbot-va-strict/results/randomized_same_seed_gpu_offload2_20260812/*.log`
- Strict LingBot-VA clean logs：`/data/pxchen/lingbot-va-strict/results/clean_same_seed_gpu_offload2_20260812/*.log`
- TurboVLA 5-task result logs：`/data/pxchen/TurboVLA/robotwin_eval_5plus5.log`、`/data/pxchen/TurboVLA/robotwin_eval_5plus5_release_policy.log`
- TurboVLA randomized result files：`/data/pxchen/RoboTwin/eval_result/turbovla_random_5task/`
- LingBot-VA old randomized logs：`/data/pxchen/lingbot-va/results/lingbotva_robotwin_same4_random_20260812/*.log`

## 2026-08-13：LaWAM 同任务 randomized seed 对照补充

这不是外部模型结果，而是为 SEC284 诊断补充的同任务控制实验。在 `move_pillbottle_pad`、`demo_randomized`、seed `100001`、`replan=36` 下：

| 变体 | 结果 | steps |
|---|---:|---:|
| 原始 LaWAM（官方 VLM） | `1/1` | 139 |
| LAP6 + 官方 VLM | `1/1` | 141 |

结果原始文件位于：

```text
results/eval_runs/sec284_failed_random_seed100001_original_vs_lap6_1x_20260813/
```

该对照只说明这个 randomized seed 和主干链路具备成功可能，不应与 SEC284 no-VLM 的 `0/10` clean 结果合并成同一模型成功率。
