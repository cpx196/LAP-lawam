# SEC284-L：VLM Condition 蒸馏与冻结 Expert 联合训练设计

> 状态：纯表示蒸馏、冻结 Expert behavior-KD、output-primary inference-grid KD 和 Expert-only grid-KD 均已完成；当前进入固定 held-out 验证阶段
>
> 更新：2026-08-13（Asia/Hong_Kong）
>
> 当前任务：RoboTwin `move_pillbottle_pad`（任务 14）
>
> 当前阶段范围：SEC284 仍只读取三视角 DINO latent；action 仅作为冻结 Expert 的 flow 监督，不进入 SEC284

## 0. 2026-08-11 冻结 Expert 第一阶段

纯表示蒸馏 3000 step checkpoint：

```text
outputs/sec284_l_bs32_3000step/step-003000.pt
```

离线结果为 raw MSE `0.060777`、cosine `0.955959`、dynamic R²
`0.724421`、跨样本 std ratio `0.8701`。表示条件已达到接入门槛，但闭环
`LAP6 + SEC284` 的 1+1 排查没有成功，因此进入冻结下游的行为对齐阶段。

训练图固定为：

```text
三视角 DINO ── SEC284 ── student condition ─┐
                                             ├─ frozen Action Expert ─ velocity KD / flow
缓存 VLM condition ─ teacher condition ─────┘

三视角 DINO + EEF ─ frozen LAP6 ─ z_lap ─ frozen LaWM ─ future visual
缓存 action ───────────────────────────────────────────── flow supervision
```

只有 SEC284 的 `76,624,896` 个参数可训练；LAP6、LaWM、Action Expert 均为
`eval()` 且 `requires_grad=False`。梯度允许穿过冻结 Expert 回到 SEC284。
teacher 与 student Expert 前向恢复相同 CPU/CUDA RNG state，保证两支使用
完全相同的 flow noise 和 time，velocity 差异只来自 condition。

第一阶段 loss 为：

```text
L_repr = L_white + 0.5 L_raw + 0.05 L_token_cos
         + 0.1 (L_cross_sample_cos + L_log_std_ratio)

L_behavior = masked_MSE(v_student, v_teacher)
g_repr = ||dL_repr / d(condition_SEC)||
g_behavior = ||dL_behavior / d(condition_SEC)||
lambda_behavior = EMA(clamp(0.25 * g_repr / g_behavior, 1e-3, 10), 0.9)

L_total = L_repr + warmup(lambda_behavior, 100 steps) * L_behavior
```

梯度范数在 SEC284 输出 condition 上计算并跨 rank 汇总；这样 behavior 对
SEC284 的梯度目标为表示梯度的 25%，而不是按 loss 数值做不可靠的等权归一化。
真实 flow loss 第一阶段只记录、不参与反传。训练配置固定为 2000 step、每 500
step 保存、local batch size 32、AdamW `lr=1e-5`、100-step warmup、cosine
衰减到 `1e-6`、BF16 autocast。四卡时 global batch 128；两卡回退时 global
batch 64，不使用梯度累积。

实现文件：

```text
tools/train_sec284_frozen_expert_ddp.py
tools/run_sec284_frozen_expert_2000step.sh
```

下文第 1～14 节保留纯表示蒸馏阶段的原始设计合同，作为本阶段的初始化与
表示锚点依据。

## 1. 最终决定

本阶段只实现一个模型规格：**SEC284-L**。

```text
三视角 DINO latent [B,3,256,768]
                 │
                 ▼
SEC284-L
  - 284 task-specific learned queries
  - hidden width 768
  - 8 decoder blocks
  - 12 attention heads
  - FFN width 3072
  - 约 76.6M 参数
                 │
                 ▼
student condition [B,284,768]
                 │
                 ▼
拟合 fixed-prompt VLM teacher condition [B,284,768]
```

当前阶段明确不包含：

- action/action chunk；
- flow matching、velocity 或 trajectory loss；
- EEF/current state；
- LAP6、`z_lap` 或 LAP scene token；
- LaWM/future visual subgoal；
- Action Expert 前向或反向；
- 在线 language encoder；
- RoboTwin 闭环训练。

Action Expert 兼容性与闭环部署属于 SEC284-L 表示蒸馏完成后的独立阶段，不参与本阶段 checkpoint 选择。

## 2. 已有实验依据

SEC284-L 不是凭空提出。已有 LAP10/LAP10V3 已经验证过约 39M 的 284-query 条件器：

| 实验 | 条件头参数 | 训练量 | condition 结果 | 闭环记录 |
|---|---:|---:|---|---:|
| T3 LAP10 | 38.64M | 8192 samples，1000 steps，effective batch 8 | train MSE `0.1674`；8-sample MSE `0.1646` | `0/10` |
| T6 LAP10V3 | 38.88M | 8192 samples，1000 steps，effective batch 8 | MSE `0.1592`，student RMS `0.7427` | 不稳定 |
| T7 LAP10V3 + Expert | SEC 38.88M + Expert 304.83M trainable | 2000 steps | condition MSE `0.1587` | `4/10` |

teacher 每位置均值的 MSE 为约 `0.2195`。T6 相对该均值模板只解释约 `27.5%` 的动态方差，而且已有训练约等于一个有效 epoch。日志不能证明 39M 已充分训练，但足以说明不应再默认使用更小的 18M～24M 条件器。

本设计固定使用约 77M 的 SEC284-L，并通过小样本可记忆性测试与正式 episode-held-out 曲线区分：

- 实现/对齐错误；
- 优化不足；
- 模型容量不足；
- DINO 表示对 teacher condition 的可预测性不足。

## 3. 固定任务与 teacher 指令

现有 teacher cache 的全部 8192 个样本使用同一条详细指令：

```text
Use the left arm to pick and place the orange bottle for pills or liquid onto the pad.
```

这条指令在样本间是常量，不提供逐样本动态信息。当前单任务 SEC284-L 因此不运行 text encoder，也不读取 language embedding；固定任务语义由 284 个 learned queries 吸收。

teacher cache 必须继续记录完整 instruction 和 prompt hash，保证所有 target 来自相同 teacher 任务定义。若未来扩展多任务，另行设计 `Q_task[task_id,284,768]`；自由文本泛化不属于本阶段目标。

## 4. 模型代码设计

### 4.1 文件

新建：

```text
starVLA/model/sec284.py
```

包含：

```python
@dataclass(frozen=True)
class SEC284Config:
    num_views: int = 3
    tokens_per_view: int = 256
    vision_dim: int = 768
    model_dim: int = 768
    output_dim: int = 768
    num_queries: int = 284
    num_layers: int = 8
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.0


class SEC284DecoderBlock(nn.Module):
    ...


class SEC284L(nn.Module):
    ...
```

配置固定为 L 规格。CLI 可以读取这些值用于 checkpoint 自描述，但正式训练不开放 B/XL 模型切换，避免实验口径漂移。

### 4.2 Forward 合同

```python
class SEC284L(nn.Module):
    def forward(
        self,
        visual_tokens: torch.Tensor,      # [B,3,256,768]
        view_mask: torch.Tensor | None = None,  # [B,3], True=有效
    ) -> torch.Tensor:                    # [B,284,768]
        ...
```

强制校验：

- rank 必须为 4；
- view/token/width 必须严格为 `[3,256,768]`；
- `view_mask` 缺省时三个视角全部有效；
- 每个样本至少一个有效视角；
- 输出必须是有限值且严格为 `[B,284,768]`。

函数签名中不允许出现 `state`、`actions`、`z_lap`、`scene_tokens` 或 language 输入。

### 4.3 输入编码

参数：

```text
view_embeddings:  [3,768]
patch_embeddings: [256,768]
input_norm:        LayerNorm(768)
```

计算：

```python
x = visual_tokens
x = x + view_embeddings[None, :, None, :]
x = x + patch_embeddings[None, None, :, :]
memory = input_norm(x).reshape(B, 768, 768)
memory_padding_mask = (~view_mask)[:, :, None].expand(B, 3, 256).reshape(B, 768)
```

虽然 DINO token 已包含自身位置编码，额外的 patch/view embedding 用于明确标识跨视角和 patch 索引；是否保留 patch embedding应在 64-sample 可记忆性测试中验证，但正式主配置固定后不得在同一 run 中变化。

本阶段不做 view dropout、random crop 或 token masking，因为目标是先建立确定性的 teacher condition 拟合上限。

### 4.4 Task-specific queries

```text
task_queries: [284,768]
初始化：Normal(mean=0, std=0.02)
```

每个 query 同时承担：

- 固定 `move_pillbottle_pad` 任务先验；
- 对应 teacher condition 的输出位置索引。

不使用 `teacher_position_mean` 初始化，不把 teacher 均值加到输出，也不使用 `mean + residual` 路径。模型必须从视觉与 learned queries 直接生成完整 condition。

### 4.5 Decoder block

每层使用 pre-norm：

```python
q = q + self_attn(norm1(q))
q = q + cross_attn(norm2(q), memory, memory, memory_padding_mask)
q = q + ffn(norm3(q))
```

具体配置：

```text
层数：8
hidden：768
heads：12
head dim：64
FFN：Linear(768,3072) → GELU → Linear(3072,768)
dropout：0.0
weight sharing：无
attention implementation：PyTorch SDPA/MultiheadAttention，优先启用 fused kernel
```

输出头：

```python
condition = output_proj(output_norm(q))
# output_norm = LayerNorm(768)
# output_proj = Linear(768,768)
```

`output_proj` 使用 Xavier 初始化，bias 置零。不使用小 residual scale，避免重新形成 position-mean shortcut。

### 4.6 参数预算

8 个 decoder blocks 共 `75,614,208` 参数，加上 queries、view/patch embeddings、norm 和输出投影，按本文配置总参数应为 `76,624,896`（约 `76.6M`）。

单元测试要求：

```text
76,000,000 <= parameter_count <= 78,000,000
trainable_count == parameter_count
```

## 5. Teacher cache 设计

### 5.1 现有数据划分

使用 `cache/lap_stage1_task14/manifest.json` 的 episode 级固定划分：

| split | episodes | samples |
|---|---:|---:|
| train | 480（440 randomized + 40 clean） | 24,140 |
| val | 35（30 randomized + 5 clean） | 1,749 |
| test | 35（30 randomized + 5 clean） | 1,749 |

现有 `cache/lap10_task14_vlm_teacher_8192` 只包含前 8192 个 train samples，不能作为 SEC284-L 正式数据集。

### 5.2 新缓存路径与脚本

新建：

```text
tools/build_sec284_teacher_cache.py
cache/sec284_task14_teacher/
  metadata.json
  train/shard-*.pt
  val/shard-*.pt
  test/shard-*.pt
```

每个 shard：

```python
{
    "teacher_condition": Float16Tensor[N,284,768],
    "teacher_mask": BoolTensor[N,284],
    "episode_id": Int64Tensor[N],
    "base_index": Int64Tensor[N],
}
```

teacher condition 必须是官方 VLM hidden state 经 `Action Expert.enc_vlm` 投影后、实际进入 Expert cross-attention 的 `[284,768]` condition。

`metadata.json` 必须记录：

```text
fixed instruction + prompt hash
official policy checkpoint path + SHA256
policy config SHA256
manifest SHA256
camera preprocessing fingerprint
split/sample counts
shape [284,768]
dtype float16
shard size
creation command + git commit
```

cache builder 不读取 action cache，不输出 state/action 字段。

### 5.3 Teacher stats

新建：

```text
tools/build_sec284_teacher_stats.py
cache/sec284_task14_teacher/train_stats.pt
```

只用 train split 计算：

```python
position_mean:     [284,768]
position_variance: [284,768]
position_std:      [284,768]
global_mse_mean_baseline: scalar
teacher_rms: scalar
```

val/test 不得参与 stats。

## 6. Dataset 代码设计

新建：

```text
tools/sec284_data.py
```

`SEC284Dataset(split)` 只返回：

```python
{
    "visual_tokens": FloatTensor[3,256,768],
    "view_mask": BoolTensor[3],
    "teacher_condition": FloatTensor[284,768],
    "teacher_mask": BoolTensor[284],
    "episode_id": int,
    "base_index": int,
    "domain": str,
}
```

三视角输入来自：

```text
main:  cache/lap_stage1_task14/<split>/vision_t
left:  cache/lap_stage1_task14_wrist/<split>/vision_left_t
right: cache/lap_stage1_task14_wrist/<split>/vision_right_t
```

Dataset 初始化时必须逐 shard 校验样本数、`episode_id/base_index` 和 manifest 顺序；发现错位立即报错，不允许按最短长度截断。

## 7. Loss 选择

### 7.1 选择原则

已有 LAP10V3 使用 raw MSE、cosine 和 structure loss，但 position mean 主导且动态方差拟合不足。本版本：

- 保留 raw MSE，确保绝对数值和尺度正确；
- 增加有界方差加权 MSE，避免高方差/静态位置结构完全主导；
- 保留小权重 cosine，约束方向；
- 不使用 structure/Gram loss：已有 T6/T7 没有显示它带来进一步 condition 改善，且 `284×284` 关系矩阵开销高；
- 不使用 batch variance/covariance loss：小 batch 下不稳定，方差加权 MSE已经直接监督逐样本动态误差；
- 不使用 action、flow、velocity 或 Expert loss。

### 7.2 Masked raw MSE

```python
error2 = (student.float() - teacher.float()).square()
mask = teacher_mask[:, :, None].float()
L_raw = (error2 * mask).sum() / (mask.sum() * 768)
```

### 7.3 有界方差加权 MSE

从 train-only `position_variance` 构造固定权重：

```python
mean_var = position_variance.mean()
w = mean_var / (position_variance + 1e-4)
# 用二分找到全局 scale，使 clamp(scale*w, 0.25, 4.0) 的均值为 1；
# 不能先 clamp 再直接除均值，否则会重新突破边界。
w = clamp(scale * w, min=0.25, max=4.0)

L_white = (error2 * w[None] * mask).sum() / (
    (w[None] * mask).sum()
)
```

权重限制在 `[0.25,4.0]`，避免极低方差维度放大浮点噪声；再归一化到均值 1，使 loss 量级可解释。

### 7.4 Token cosine loss

```python
cos = F.cosine_similarity(student.float(), teacher.float(), dim=-1)
L_cos = ((1.0 - cos) * teacher_mask.float()).sum() / teacher_mask.sum()
```

### 7.5 最终 loss

固定为：

```text
L_total = 1.00 * L_white
        + 0.50 * L_raw
        + 0.05 * L_cos
```

不设置随训练阶段变化的 loss 权重，保证 overfit、train 和 validation 使用同一目标。日志必须同时输出未加权与加权后的三项。

## 8. 训练入口与配置

新建：

```text
tools/train_sec284_distill_ddp.py
```

### 8.1 默认配置

```text
model: SEC284-L, 8×768, FFN 3072
precision: BF16 autocast；loss/statistics 用 FP32
optimizer: AdamW
betas: (0.9, 0.95)
weight_decay: 0.01
peak_lr: 1e-4
min_lr: 1e-5
warmup: 5% optimizer steps
scheduler: cosine
grad_clip: 1.0
effective_batch: 32
epochs: 10
minimum_epochs_before_early_stop: 5
early_stop_patience: 2 validation epochs
seed: 42
dropout/view_dropout/augmentation: 0
```

在 24,140 train samples、effective batch 32 下，每个 epoch 约 755 optimizer steps，10 epochs 约 7,550 steps。CLI 以 `--epochs` 为主，不用固定的 `--steps=1000` 旧口径。

推荐首个正式 run：

```bash
torchrun --nproc-per-node=2 tools/train_sec284_distill_ddp.py \
  --teacher-cache cache/sec284_task14_teacher \
  --feature-cache cache/lap_stage1_task14 \
  --wrist-cache cache/lap_stage1_task14_wrist \
  --epochs 10 \
  --batch-size 2 \
  --grad-accumulation 8 \
  --lr 1e-4 \
  --output-dir outputs/sec284_l_task14_distill
```

两卡 × batch 2 × accumulation 8 = effective batch 32。若实际显存允许，只调整 per-device batch 与 accumulation，effective batch 保持 32。

### 8.2 Validation 与 checkpoint 选择

每个 epoch 完整运行 val 1,749 samples。checkpoint 排序：

1. 最低 `val/L_total`；
2. 若差异小于 0.5%，选择更低 `val/L_raw`；
3. 不按 train loss 或最后一步自动选模型。

每个 epoch 保存轻量评估记录；只长期保留 best、last 和最近两个 epoch checkpoint。

### 8.3 Checkpoint schema

```python
{
    "format_version": 1,
    "model_name": "SEC284-L",
    "sec284": model.state_dict(),
    "config": asdict(SEC284Config()),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "global_step": step,
    "best_val_total": best,
    "teacher_stats_sha256": ...,
    "teacher_cache_fingerprint": ...,
    "manifest_sha256": ...,
    "git_commit": ...,
    "args": vars(args),
}
```

正式导出的推理 checkpoint 只保留 `format_version/model_name/sec284/config/fingerprints`。

## 9. 训练前可记忆性测试

正式训练前必须运行两个固定子集：

### 9.1 64-sample overfit

- 取 train manifest 固定前 64 个样本；
- 不 shuffle、不增强；
- 训练到 5,000 steps 或收敛；
- 目标：raw MSE `< 0.01`、token cosine `> 0.99`。

若未达到，先检查：cache 对齐、query/teacher 位置、mask、学习率、输出初始化和梯度，不进入完整训练。

### 9.2 256-sample overfit

- 固定 256 个样本；
- 训练到 10,000 steps 或收敛；
- 目标：raw MSE `< 0.03`、token cosine `> 0.97`。

若 64 通过而 256 明显失败，才把问题记录为可能的容量/优化瓶颈。当前已经选定 SEC284-L，不在本阶段临时增加 XL。

## 10. 离线评估

新建：

```text
tools/eval_sec284_distill.py
```

对 train/val/test 分别报告：

```text
raw MSE / MAE
bounded-whitened MSE
token cosine mean/std/p05
student RMS / teacher RMS / RMS ratio
student cross-sample std / teacher std / std ratio
mean-only baseline MSE
dynamic R2 = 1 - raw_MSE / mean-only_MSE
按 clean/randomized 分组的全部指标
按 main/left/right view ablation 的指标
condition shuffle 指标
```

因果检查：

1. 同一视觉重复前向必须确定性一致。
2. 打乱 batch 内视觉后，condition MSE 必须明显恶化。
3. 分别遮挡三个视角，记录每个视角贡献。
4. SEC284-L 必须显著优于 train-position-mean baseline。
5. val/test 使用 train-only mean/variance，禁止重算。

本阶段不报告 action MSE、velocity MSE、flow loss 或闭环成功率作为训练结果。

## 11. 表示蒸馏验收门槛

进入后续 Expert 兼容性阶段前，必须同时满足：

1. 64/256-sample overfit gate 通过。
2. val raw MSE 明确低于旧 LAP10V3 的约 `0.159` 水平。
3. 目标值：val raw MSE `<= 0.10`、token cosine `>= 0.95`。
4. val dynamic R2 `>= 0.50`，且显著高于 mean-only baseline。
5. student/teacher 跨样本 std ratio 位于 `[0.85,1.15]`，排除静态模板塌缩。
6. randomized 与 clean 都满足改进，不允许总体指标被单一 domain 掩盖。
7. condition shuffle 和视觉遮挡产生可解释的明显退化。
8. test 只在模型和阈值固定后运行一次，不用于调参。

若未达标，当前阶段只允许调整优化、数据对齐或 SEC284-L 内部实现；不得通过引入 action、EEF、LAP6 或 Expert loss 绕过 condition 拟合问题。

## 12. 测试设计

新建：

```text
tests/test_sec284.py
tests/test_sec284_data.py
tests/test_sec284_loss.py
```

覆盖：

- output shape/dtype/device；
- 参数量在 76M～78M；
- 无 state/action/language 参数；
- view mask 正确屏蔽整个视角；
- 全视角被屏蔽时报错；
- eval mode 重复前向确定性；
- masked loss 分母正确；
- variance weight 范围和均值归一化；
- shard/manifest/teacher key 对齐；
- checkpoint round-trip 后输出一致；
- BF16 forward + FP32 loss 无 NaN/Inf。

## 13. 实施顺序

```text
P0  构建完整 train/val/test teacher cache 与 train-only stats
P1  实现 SEC284-L、loss、dataset 和单元测试
P2  运行 64-sample overfit gate
P3  运行 256-sample overfit gate
P4  完整 10-epoch DDP 训练与每 epoch validation
P5  固定 best checkpoint，运行 test 和视角/condition 因果诊断
P6  汇总结果后再决定是否进入 Action Expert 兼容性阶段
```

当前设计在 P5 完成前不实现 policy server，不启动 RoboTwin 闭环，不加入任何 action 相关 loss。

## 14. 纯表示阶段代码项

- `starVLA/model/sec284.py`
- `tools/build_sec284_teacher_cache.py`
- `tools/build_sec284_teacher_stats.py`
- `tools/sec284_data.py`
- `tools/train_sec284_distill_ddp.py`
- `tools/eval_sec284_distill.py`
- `tests/test_sec284.py`
- `tests/test_sec284_data.py`
- `tests/test_sec284_loss.py`

其中 SEC284 模型、缓存、训练和评估工具已经实现；测试项仍需补齐。本文件是代码实现合同。任何 shape、输入源、loss 权重、数据 split 或验收阈值的改变都应先更新本文档并记录原因。

## 15. 下游诊断与 inference-grid KD（2026-08-11）

表示蒸馏与 frozen-Expert 单时刻 behavior KD 已完成后，闭环 `LAP6 + SEC284`
在 clean/randomized 的配对 1+1 上均失败；相同 seed 的 `LAP6 + real VLM`
均成功。因此当前瓶颈已定位到 SEC284 condition 在 Expert 敏感方向上的误差，而不是
LAP6、任务难度或 Expert 本身。

### 15.1 固定 language 合同

SEC284 不接收 language，teacher cache 固定绑定以下详细指令：

```text
Use the left arm to pick and place the orange bottle for pills or liquid onto the pad.
```

RoboTwin 在线评测会随机生成不同措辞，导致真实 VLM condition 长度随 tokenization
变化（实测出现 279/282，而固定指令为 284）。因此 SEC284 的正式对照、cache 和 KD
必须统一使用上述固定指令；随机在线 instruction 只记录为 metadata，不作为 SEC284
输入。不得对不同长度、不同语义的 condition 直接做逐 token MSE。

### 15.2 成功 teacher rollout 诊断

`tools/run_lap6_sec284_shadow_trace_paired_1x.sh` 使用真实 VLM 动作控制环境，同时让
SEC284 在相同 observation、相同 LAP6/LaWM future、相同初始 flow noise 下运行影子分支。
固定指令对照结果：clean `1/1`、randomized `1/1`，共保存 9 个重规划点。

聚合结果：

| split | condition cosine | action MSE | grid velocity MSE | gripper sign agreement | 最大偏差重规划点 |
|---|---:|---:|---:|---:|---:|
| clean | 0.9153 | 0.01105 | 0.03443 | 0.9417 | 4 |
| randomized | 0.9061 | 0.01102 | 0.03111 | 0.9444 | 3 |

两条轨迹的最大误差都位于抓取附近，且主要由 gripper 维度贡献；teacher-grid velocity
误差总体随 10-step flow 后半段放大。由此不再把 condition cosine 当作唯一优化目标。
原始 trace、JSON 聚合和曲线位于：

```text
results/eval_runs/sec284_step2000_shadow_trace_fixed_seed0_1x/
```

### 15.3 新损失

新增 `tools/train_sec284_inference_grid_kd.py`，仅更新 SEC284，Action Expert 全冻结：

```text
L_total = L_repr + lambda_grid * L_grid
```

- `L_repr`：沿用 bounded-whitened MSE + raw MSE + cosine，锚定固定指令的 284-token
  teacher condition，防止为了少量动作点破坏全局表征。
- `L_grid`：在真实成功 teacher rollout 的 10-step flow 网格上，随机抽一个 step，固定
  teacher `x_t`，拟合 Expert velocity。每次只保留一个 grid step 的反向图，以控制显存；
  长期均匀抽样覆盖全部 10 步。
- 动作维权重：双臂 XYZ 为 `2x`，双 gripper 为 `4x`，其余为 `1x`。这是由抓取阶段
  的实测误差确定，不是任意增加 loss。
- `lambda_grid`：根据 `L_repr` 与 `L_grid` 对 SEC condition 的梯度范数自动标定，目标
  grid/repr 梯度比为 `0.25`，EMA `0.9`，裁剪到 `[1e-3, 10]`。

单步 smoke 已通过：`grid_kd=0.007251`、自适应 `lambda_grid=0.635`、峰值显存
`3.02 GiB`（不含卡上其他进程），证明可反传且没有 OOM。当前仅 9 个重规划点，正式
训练前应扩充多 seed 的成功 teacher rollout cache，避免对 1+1 轨迹记忆化。

## 16. 2026-08-13 完成状态与验证边界

本设计中的表示蒸馏和下游训练均已实际完成，但训练日志不能替代固定验证：

| 阶段 | checkpoint / 日志 | 当前证据 |
|---|---|---|
| 纯表示蒸馏 | `outputs/sec284_l_bs32_3000step/` | held-out cosine `0.955959`、dynamic R² `0.724421`、std ratio `0.8701` |
| Frozen behavior-KD | `outputs/sec284_frozen_expert_behavior_kd_2000step/` | 末段 batch std ratio `0.9241`，behavior KD `0.002951` |
| Output-primary grid-KD | `outputs/sec284_output_kd_primary_2000step/` | 末段 `repr=0.062496`、`grid_kd=0.000812`、std ratio `0.9209` |
| Expert-only grid-KD | `outputs/sec284_expert_grid_kd_2000step/` | 500→2000 已完成；500-step 为经验候选 |

Expert-only checkpoint 的固定 clean 结果为：500/1000/1500/2000 均为 `0/1`，500-step 额外 clean 10x 为 `0/10`。因此本设计不把任何一个 checkpoint 标记为“闭环成功”。当前必须补齐：

1. behavior-KD、output-primary 和 500-step Expert 的同一 held-out 样本评估；
2. 按 `k=0..9` 拆分的 velocity MSE、XYZ、gripper 和 flow 后段误差；
3. dynamic R²、std ratio、shuffle-teacher 和 mean-only baseline；
4. 同一 seed、同一 `replan=36` 的 clean/randomized 闭环。

原始训练日志、评测日志、JSON/JSONL、`meta.json`、`run.log` 和 `_result.txt` 已按原路径归档；视频、checkpoint、cache 和二进制 trace 不提交。完整索引见 [SEC284 当前状态与原始证据索引](SEC284_CURRENT_STATUS_2026-08-13_ZH.md)。
