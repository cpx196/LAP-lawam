# LAP-LaWAM: RoboTwin 无 VLM 轻量机器人策略研究

[![Repository](https://img.shields.io/badge/GitHub-cpx196%2FLAP--lawam-181717?logo=github)](https://github.com/cpx196/LAP-lawam)
[![Task](https://img.shields.io/badge/RoboTwin-move__pillbottle__pad-2563eb)](docs/PROJECT_OVERVIEW_ZH.md)
[![Status](https://img.shields.io/badge/status-research-orange)](timeline.md)
[![Upstream](https://img.shields.io/badge/upstream-RLinf%2FLaWAM-6b7280)](https://github.com/RLinf/LaWAM)

这是 **cpx196** 基于 [RLinf/LaWAM](https://github.com/RLinf/LaWAM) 开发的个人研究仓库，而不是 LaWAM 官方发布仓库。项目聚焦于 RoboTwin 单任务环境，用轻量 LAP 和 SEC284 替代 Qwen3-VL-2B 的在线推理，同时保留 LaWM 的视觉子目标能力。

## 研究目标

当前固定任务为 `move_pillbottle_pad`：Aloha-AgileX 双臂机器人将药瓶抓取并放到蓝色垫上。策略运行时输入包含主视角、左腕视角、右腕视角以及当前 16-D 双臂 EEF state；任务语义固定在 SEC284 的 learned queries 中，不运行语言编码器。

目标架构：

```text
三视角 RGB ──> frozen DINO tokens
        ├── LAP6(+ EEF) → 32-D latent action → LaWM → visual subgoal
        └── SEC284(task-specific learned queries) → 284×768 VLM-compatible condition

visual subgoal + current vision + SEC284 condition + 独立 EEF
        ↓
Action Expert → 36-step action chunk
```

- `LAP6 → LaWM` 保留已验证的 latent-action 分支。
- `SEC284` 是计划中与 LAP6 并行的单任务专用模块，以三视角 DINO latent 为动态输入，以 284 个 learned queries 固化任务先验，输出与官方 VLM condition 同形状的 `[B,284,768]` hidden state。
- SEC284 不读取 EEF、`z_lap` 或 LAP6 token；EEF 只进入 Action Expert 的 proprioception 通道。
- SEC284-L 已完成 3000-step 固定指令 VLM condition 蒸馏；冻结 Expert 的 behavior-KD、output-primary inference-grid KD 和 Expert-only grid-KD 也已完成。当前经验上的 Expert best 是 500 step，但 clean 10x 仍为 `0/10`，不能按 cosine 或 train loss 宣称闭环成功。

## 当前进展

| 实验 | 关键变化 | RoboTwin 闭环结果 | 结论 |
|---|---|---:|---|
| Stage-1 三观测 LAP | DINO + EEF 预测 latent action | 离线表征测试 | LAP6/LaWM 链路有效 |
| LAP8 no-VLM | 8 token 直接替代 VLM | `0/10` | 条件容量/接口不足 |
| T7 LAP10V3 + Expert | 284 token + Expert 联合训练 | `4/10` | 当前最佳无 VLM baseline |
| T8 AR-2000 | 真实 action 监督 | `1/10` | 离线 loss 改善未转化为闭环改善 |
| T9 FlowOnly-1000 | 只训练 Expert，仅官方 flow loss | `1/6` | 未超过 T7 同 seed 的 `2/6` |
| SEC284 表征蒸馏 | 三视角 DINO → `[B,284,768]` | cosine `0.955959`；dynamic R² `0.724421` | 公共语义接近，但动态残差仍有缺口 |
| SEC284 frozen behavior-KD | 冻结 LAP6/LaWM/Expert，只更新 SEC284 | batch std ratio 约 `0.924` | 行为代理改善，未证明闭环 |
| SEC284 output/grid-KD | 分别更新 SEC284 或 Expert 的 inference-grid velocity | 500-step 经验 best；clean `0/10` | teacher-forcing loss 与闭环存在 mismatch |

训练 loss、token MSE 和 action MAE 只作为离线诊断；最终指标是固定 seed 和 policy flow noise 的 RoboTwin 闭环成功率。

## 项目文档

- [项目总览](docs/PROJECT_OVERVIEW_ZH.md)
- [SEC284 当前状态与原始证据索引（2026-08-13）](docs/SEC284_CURRENT_STATUS_2026-08-13_ZH.md)
- [SEC284-L：VLM condition 蒸馏与冻结 Expert 联合训练设计](docs/VLM_REPLACEMENT_JOINT_TRAINING_DESIGN_ZH.md)
- [实验时间线](timeline.md)
- [LAP10V3 实验总账本](docs/LAP10V3_EXPERIMENT_LEDGER_ZH.md)
- [Stage-1 60M 训练方案](docs/LAP_STAGE1_60M_TRAINING_PLAN_ZH.md)
- [LAP10V3 AR-2000 会话与实验记录](docs/LAP10V3_AR_2000STEP_SESSION_RECORD_ZH.md)

## 仓库结构

```text
starVLA/model/lap_stage1.py       LAP6 / Stage-1 模型
starVLA/model/lap_stage2.py       LAP8、LAP10 和 LAP10V3 实验模型
tools/                            数据缓存、训练、离线分析与评估脚本
deployment/model_server/          无 VLM 和 A/B/C 消融政策服务
examples/Robotwin/                RoboTwin bridge 与闭环评估接口
docs/                             训练方案、交付文档和实验账本
```

模型 checkpoint、数据集、DINO/Qwen 权重、feature cache、视频和二进制 trace 仍不存储在 Git 仓库中；本次更新将可审计的 SEC284 训练日志、评测日志、JSON/JSONL、`meta.json` 和 `_result.txt` 按原路径保留在仓库中。原始证据清单见 [SEC284 当前状态](docs/SEC284_CURRENT_STATUS_2026-08-13_ZH.md)。

## 原始 LaWAM

本项目的基础模型与工程来自：

- 原始仓库：[RLinf/LaWAM](https://github.com/RLinf/LaWAM)
- 论文：[LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies](https://arxiv.org/abs/2606.15768)
- 原始项目页：[rlinf.github.io/LaWAM](https://rlinf.github.io/LaWAM/)
- 官方权重：[Hugging Face collection](https://huggingface.co/collections/jialei02/lawam-checkpoints)
- RoboTwin 数据：[robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged)

原始 LaWAM 使用 VLM 条件和 LaWM 视觉子目标生成 action chunk。原始方法架构如下：

<p align="center">
  <img src="./assets/lawam_overview.png" alt="Original LaWAM method overview" width="95%">
</p>

## 环境安装

克隆本个人研究仓库，然后创建 LaWAM 训练环境：

```bash
git clone git@github.com:cpx196/LAP-lawam.git LaWAM
cd LaWAM

conda create -n lawam python=3.10 -y
conda activate lawam

pip install -U pip
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
```

如需同步官方 LaWAM 的新变更，可以将其添加为只读 `upstream`：

```bash
git remote add upstream https://github.com/RLinf/LaWAM.git
git fetch upstream
```

如果本地 CUDA/PyTorch 与 `flash-attn==2.8.3` 不兼容，请安装匹配的 `flash-attn` wheel，然后重新运行 `pip install -e .`。

Quick import check:

```bash
python - <<'PY'
import torch
import starVLA
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
PY
```

## Model Preparation

All commands in this section and the training sections assume the current
directory is the `LaWAM` repository root. The original LaWAM baseline needs:

- Base VLM:
  [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- LAM vision encoder:
  [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- LaWM/LAM checkpoint and config:
  [lawam_lam](https://huggingface.co/jialei02/lawam_lam)

For the LAP no-VLM route, Qwen is only required when generating offline teacher
targets or running the official baseline. No-VLM deployment loads DINO, LAP6,
LaWM, the Expert-condition module and Action Expert checkpoints without loading
Qwen online.

Downloadable resources used by the released configs:

| Type | Resource | Used for | Local path expected by examples/configs |
| --- | --- | --- | --- |
| Base VLM weights | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Training and inference | `results/Checkpoints/qwen3_weights` |
| DINOv3 vision encoder weights | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | LAM feature extraction | `weights/dinov3-vitb16-pretrain-lvd1689m` |
| LAM checkpoint/config | [lawam_lam](https://huggingface.co/jialei02/lawam_lam) | Training and inference | `latent_action_model/logs/dino_large_vae/lam_release` |
| LaWAM pretraining checkpoint | [lawam_pretrain](https://huggingface.co/jialei02/lawam_pretrain) | LIBERO/RoboTwin SFT initialization | `results/Checkpoints/pretrain/lawam_pretrain` |
| LIBERO SFT checkpoint | [lawam_libero_sft_release](https://huggingface.co/jialei02/lawam_libero_sft_release) | LIBERO benchmark inference | `results/Checkpoints/libero/lawam_libero_sft_release` |
| RoboTwin SFT checkpoint | [lawam_robotwin_sft_release](https://huggingface.co/jialei02/lawam_robotwin_sft_release) | RoboTwin evaluation | `results/Checkpoints/robotwin/lawam_robotwin_sft_release` |
| LIBERO SFT dataset | [libero_merged_no_noops_20hz](https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz) | LIBERO SFT | `dataset/libero_merged_no_noops_20hz` |
| RoboTwin SFT dataset | [robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged) | RoboTwin SFT | `dataset/robotwin_merged` |

Download Qwen3-VL into the path recorded by the provided configs:

```bash
mkdir -p results/Checkpoints/qwen3_weights

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir results/Checkpoints/qwen3_weights
```

Download DINOv3 into the path used by the LAM YAML config:

```bash
mkdir -p weights/dinov3-vitb16-pretrain-lvd1689m

hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitb16-pretrain-lvd1689m
```

Download the LaWM/LAM checkpoint and YAML config into the paths recorded by the
provided configs:

```bash
hf download jialei02/lawam_lam \
  --local-dir latent_action_model/logs/dino_large_vae/lam_release
```

The policy server loads Qwen3-VL and LAM from the checkpoint config, then the
LAM YAML loads DINOv3 through `model.vision_model_id`. If your downloaded LAM
YAML still points to a Hugging Face model id or an unavailable absolute path,
set it to:

```yaml
model:
  vision_model_id: weights/dinov3-vitb16-pretrain-lvd1689m
```

## Inference

Inference uses two environments:

- the `lawam` environment above for policy loading and serving;
- a separate simulator environment for LIBERO or RoboTwin.

Run LIBERO first if you only need one smoke test. RoboTwin setup is separate and
usually heavier.

### LIBERO Inference

#### 1. Install The LIBERO Simulator

Install LIBERO in a separate environment following the official repository:

https://github.com/Lifelong-Robot-Learning/LIBERO

Example layout:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO

# Create the LIBERO simulator environment with Python 3.10, then install
# LIBERO following the official instructions.
conda create -n libero python=3.10 -y
conda activate libero

# Then set:
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero_env/bin/python
```

After completing the official LIBERO installation, install the MuJoCo version
used by this repository in the Python 3.10 LIBERO simulator environment:

```bash
conda activate <libero_env>
pip install mujoco==3.3.2
```

#### 2. Run LIBERO Benchmark

Set the policy checkpoint path. Use a released LIBERO checkpoint if available
from [lawam_libero_sft_release](https://huggingface.co/jialei02/lawam_libero_sft_release),
or a checkpoint produced by [LIBERO SFT](#libero-sft).

```bash
cd LaWAM
conda activate lawam

hf download jialei02/lawam_libero_sft_release \
  --local-dir results/Checkpoints/libero/lawam_libero_sft_release

export CKPT_PATH=results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero_env/bin/python
export STAR_VLA_PYTHON="$(which python)"

SUITES="libero_10 libero_goal libero_object libero_spatial" \
NUM_TRIALS_PER_TASK=50 \
NUM_WORKERS=4 \
GPU_IDS="0 1 2 3" \
OUTPUT_ROOT=results/eval_runs/libero \
LIBERO_CKPT_ALIAS=lawam_libero_sft \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT_PATH"
```

Outputs are saved under:

```text
results/eval_runs/libero/<ckpt_alias>/<run_tag>/
  run_meta.json
  suites/<suite_name>/eval.log
```

### RoboTwin Inference

#### 1. Install The RoboTwin Simulator

Install RoboTwin in a separate environment following the official repository:

https://github.com/RoboTwin-Platform/RoboTwin

Example layout:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git ../RoboTwin

# Create and install the RoboTwin simulator environment following the official
# RoboTwin instructions. Then set:
export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python
```

After completing the official RoboTwin installation, install the extra packages
used by this repository in the RoboTwin simulator environment:

```bash
conda activate <robotwin_env>
pip install \
  accelerate==1.5.2 \
  json-numpy==2.1.1 \
  websockets==15.0.1 \
  msgpack==1.1.2 \
  rich==14.2.0 \
  omegaconf==2.3.0
```

#### 2. Run RoboTwin Evaluation

Use the auto evaluation entrypoint for RoboTwin runs. It starts the LaWAM
policy server, launches RoboTwin workers, and writes a resumable run directory.

```bash
cd LaWAM
conda activate lawam

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python

hf download jialei02/lawam_robotwin_sft_release \
  --local-dir results/Checkpoints/robotwin/lawam_robotwin_sft_release

# Single-task smoke test.
ROBOTWIN_TASKS=lift_pot \
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt \
  demo_clean
```

Full RoboTwin benchmark:

```bash
cd LaWAM
conda activate lawam

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python

ROBOTWIN_EVAL_ROOT=results/eval_runs/robotwin \
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt \
  demo_clean
```

Outputs are saved under:

```text
results/eval_runs/robotwin/<ckpt_alias>__<task_config>/<run_tag>/
  tasks/<task_name>/run.log
  tasks/<task_name>/summary.json
```

## SFT Training

SFT training uses the same Qwen3-VL and LAM files prepared in
[Model Preparation](#model-preparation). It also needs:

- LaWAM pretraining checkpoint:
  [lawam_pretrain](https://huggingface.co/jialei02/lawam_pretrain)
- benchmark-specific SFT data

Download the pretraining checkpoint:

```bash
mkdir -p results/Checkpoints/pretrain/lawam_pretrain/final_model

hf download jialei02/lawam_pretrain \
  --local-dir results/Checkpoints/pretrain/lawam_pretrain
```

All training is launched through `train_lawam.sh` for a single node
or `train_lawam_distributed.sh` for multi-node jobs. Extra arguments
are forwarded to OmegaConf, so config fields can be overridden with
`--a.b.c value`.

### LIBERO SFT

#### 1. Download LIBERO SFT Data

The preprocessed LIBERO SFT dataset is available at:

[libero_merged_no_noops_20hz](https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz)

This dataset is derived from the public
[IPEC-COMMUNITY/libero-benchmark-dataset](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset)
release. Compared with the public source, this release merges the four LIBERO
subsets and converts the data to LeRobot 3.0 format.

Download it under the unified dataset root used by the provided configs
(`dataset/`) with the directory name expected by `data_mix: libero`:

```bash
mkdir -p dataset

hf download jialei02/libero_merged_no_noops_20hz \
  --repo-type dataset \
  --local-dir dataset/libero_merged_no_noops_20hz
```

Expected layout:

```text
dataset/
  libero_merged_no_noops_20hz/
    meta/
    data/
    videos/
```

#### 2. Launch LIBERO SFT

```bash
cd LaWAM
conda activate lawam

bash train_lawam.sh \
  --run_id libero_sft_from_pretrain
```

The output checkpoint is written under:

```text
results/Checkpoints/libero/<timestamp>+<run_id>/
```

### RoboTwin SFT

#### 1. Download RoboTwin SFT Data

The preprocessed RoboTwin SFT dataset is available at:

[robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged)

This dataset uses RoboTwin EEF actions and is derived from the lingbot-va
release, specifically
[robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot/tree/main/lerobot_robotwin_eef_aug_500/beat_block_hammer-aloha-agilex_randomized_500-1000).
Compared with that public source, this release converts the data to LeRobot 3.0
format.

The provided RoboTwin SFT config uses `data_mix: robotwin_merged`, so download
the dataset under `dataset/robotwin_merged`:

```bash
mkdir -p dataset

hf download jialei02/robotwin_merged \
  --repo-type dataset \
  --local-dir dataset/robotwin_merged
```

Expected layout:

```text
dataset/
  robotwin_merged/
    meta/
    data/
    videos/
```

#### 2. Launch RoboTwin SFT

Important RoboTwin SFT settings:

- Reproducing the paper results requires a global batch size of 1024. The
  effective global batch size is
  `per_device_batch_size * total_num_gpus * gradient_accumulation_steps`.
  Adjust `datasets.vla_data.per_device_batch_size` in
  `starVLA/config/training/train_robotwin.yaml` for your GPU memory and GPU
  count. If you do not have enough GPUs, increase
  `trainer.gradient_accumulation_steps` to keep the global batch size at 1024.
- For debugging, a 30k-step RoboTwin SFT run is usually enough to reach around
  80% of the reported performance. You can set
  `--trainer.max_train_steps 30000` for a shorter debug run.

```bash
cd LaWAM
conda activate lawam

bash train_lawam.sh \
  starVLA/config/training/train_robotwin.yaml \
  --run_id robotwin_sft_from_pretrain
```

The output checkpoint is written under:

```text
results/Checkpoints/robotwin/<timestamp>+<run_id>/
```

For multi-node training, use `train_lawam_distributed.sh` with the
same config:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_host> MASTER_PORT=29500 \
bash train_lawam_distributed.sh \
  starVLA/config/training/train_robotwin.yaml
```

Run the same command on every node and set `NODE_RANK` accordingly.

## Checkpoint Notes

Training checkpoints are regular PyTorch `.pt` files that include the model
state and the merged training config. Evaluation scripts use the checkpoint
config to recover dataset statistics, action normalization, Qwen3-VL source,
and LAM source. When moving checkpoints across machines, make sure these paths
are valid in the new environment.

- LIBERO checkpoints should use `datasets.vla_data.data_mix: libero`.
- RoboTwin EEF checkpoints should use `datasets.vla_data.data_mix:
  robotwin_merged` or another supported RoboTwin EEF mixture.
- Official VLM baseline runs require `framework.qwenvl.base_vlm` to point to
  Qwen3-VL-2B-Instruct or a local copy; LAP no-VLM policy servers do not load it.
- `framework.action_model.lam_ckpt_path` and
  `framework.action_model.lam_yaml_path` must point to a matching LAM checkpoint
  and YAML config.

## Citation

```bibtex
@misc{chen2026lawam,
  title = {LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies},
  author = {Chen, Jialei and Wang, Kai and Chen, Kang and Chen, Shuaihang and Gao, Feng and Tang, Wenhao and Li, Zhiyuan and Liu, Weilin and Yao, Zhuyu and Li, Boxun and Xu, Yuanbo and Yu, Chao},
  journal = {arXiv preprint arXiv:2606.15768},
  year = {2026},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
}
```

## Acknowledgements

LAP-LaWAM is maintained by `cpx196` as a research derivative of
[RLinf/LaWAM](https://github.com/RLinf/LaWAM). The upstream code retains its
original authorship and MIT license. This project also builds on StarVLA,
LeRobot, Qwen-VL, DINO, LIBERO and RoboTwin.
