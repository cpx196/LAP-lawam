# LaWAM 边缘端环境说明

本文档记录当前服务器上可以运行 LaWAM / Three_Cubes_1 1k 微调 checkpoint 的主要环境，用于在边缘端复现推理环境。

## 推荐基础环境

- OS: Linux x86_64，推荐 Ubuntu 20.04/22.04。
- Python: 3.10，当前可用环境为 `Python 3.10.16`。
- CUDA: 当前环境为 CUDA 12.6。
- PyTorch: 当前环境为 `torch==2.8.0+cu126`。
- GPU: 当前测试机器为 `NVIDIA GeForce RTX 4090 D`。

边缘端如果是 Jetson / Thor / Orin 一类 ARM64 设备，PyTorch、torchvision、flash-attn 通常不能直接用 x86_64 wheel，需要按 NVIDIA JetPack / L4T 对应版本安装。

## 当前服务器环境导出

包内包含：

```text
requirements.txt
requirements_flashwam_freeze.txt
```

- `requirements.txt` 是整理过的仓库主要依赖。
- `requirements_flashwam_freeze.txt` 是当前 `flashwam` conda 环境的完整 `pip freeze`，更接近真实可运行环境。

## 关键依赖

最核心的依赖包括：

```text
torch==2.8.0+cu126
torchvision==0.23.0+cu126
transformers==5.2.0
tokenizers==0.22.2
huggingface_hub==1.7.1
datasets==3.6.0
diffusers==0.36.0
accelerate==1.14.0
numpy==1.26.4
pandas==2.3.3
pyarrow==25.0.0
omegaconf==2.3.0
einops==0.8.2
timm==1.0.22
opencv-python-headless==4.11.0.86
av==17.1.0
Pillow==12.2.0
matplotlib==3.10.9
rich==14.2.0
tqdm==4.68.4
websockets==16.1
json-numpy==2.1.1
lerobot==0.3.3
```

当前环境还安装了：

```text
flash_attn==2.8.3.post1
```

如果边缘端不能安装 flash-attn，原则上可以先尝试不用 flash-attn 跑通链路，但 VLM 推理延迟会明显变高。

## x86_64 GPU 服务器安装参考

如果目标边缘端也是普通 x86_64 + NVIDIA GPU，可以参考：

```bash
conda create -n lawam_edge python=3.10 -y
conda activate lawam_edge

pip install --upgrade pip setuptools wheel

# 根据目标 CUDA 版本选择 PyTorch。当前服务器是 cu126。
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

cd LaWAM
pip install -r requirements.txt

# 可选：加速 Qwen/VLM attention
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

如果 `requirements.txt` 中某些包版本和 PyTorch/CUDA 不兼容，优先保持 PyTorch、torchvision、transformers、flash-attn 这几项一致。

## Jetson / Thor / ARM64 注意事项

如果目标是 NVIDIA Jetson/Thor 这类 ARM64 边缘端：

1. 不建议直接执行 x86_64 服务器上的完整 `pip freeze`。
2. 先安装和 JetPack/L4T 匹配的 PyTorch / torchvision。
3. flash-attn 可能需要源码编译，或者暂时禁用。
4. `opencv-python-headless`、`av`、`torchvision` 这类包可能需要系统依赖或平台专用 wheel。
5. 若显存紧张，优先测试：
   - `bf16` / `fp16` 推理；
   - 减小图像分辨率；
   - 关闭不必要的 benchmark/eval 组件；
   - 只加载推理必须路径。

## 权重和配置

本次下游微调 checkpoint：

```text
results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/final_model/pytorch_model.pt
```

对应配置：

```text
results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/config.yaml
results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/dataset_statistics.json
```

包内也包含：

```text
latent_action_model/logs/dino_large_vae/lam_release/
```

用于满足配置中的 LaWM/LAM release 路径依赖。

## 需要额外确认的外部权重

当前打包重点是代码和 Three_Cubes_1 下游微调后的 full checkpoint。若边缘端代码路径仍尝试按 `config.yaml` 读取外部基础权重，需要确认以下路径是否存在或修改为实际路径：

```text
results/Checkpoints/qwen3_weights
weights/dinov3-vitb16-pretrain-lvd1689m
```

如果 full checkpoint 已能完整恢复模型，上述路径可能只在构图/processor 初始化阶段使用；第一次迁移到边缘端时建议保留同名目录结构，减少路径问题。

## 快速检查命令

```bash
conda activate lawam_edge
cd LaWAM

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```
如果要验证 LaWAM 代码是否能被 import：

```bash
PYTHONPATH=$PWD python - <<'PY'
from starVLA.model.framework.base_framework import baseframework
print("LaWAM import OK")
PY
```
