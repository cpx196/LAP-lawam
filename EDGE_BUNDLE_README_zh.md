# LaWAM Three_Cubes_1 1k 微调边缘端包说明

这个压缩包用于把当前 LaWAM 代码、依赖说明、LaWM/LAM release 文件，以及 Three_Cubes_1 下游微调后的完整 checkpoint 搬到边缘端做推理验证。

## 主要内容

- `LaWAM/starVLA/`：LaWAM / StarVLA 推理和训练相关源码。
- `LaWAM/latent_action_model/`：latent action model 相关源码和 `lam_release` 权重。
- `LaWAM/examples/`：LIBERO / RoboTwin 等示例接口代码。
- `LaWAM/tools/`：本次调试新增/使用过的诊断脚本。
- `LaWAM/requirements.txt`：仓库原始依赖文件。
- `LaWAM/requirements_flashwam_freeze.txt`：服务器当前 `flashwam` 环境的 `pip freeze` 导出。
- `LaWAM/pyproject.toml`：Python 项目配置。
- `LaWAM/README.md`：官方 README。
- `LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/final_model/pytorch_model.pt`：Three_Cubes_1 1000 step 下游微调后的完整权重。
- `LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/config.yaml`：该 checkpoint 对应配置。
- `LaWAM/results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/dataset_statistics.json`：该 checkpoint 对应 action/state 归一化统计。

## 注意

- 这个包没有包含 Three_Cubes_1 dataset，也没有包含 Qwen3-VL base 权重目录和 DINOv3 权重目录。
- 下游微调后的 `pytorch_model.pt` 是 full checkpoint，已经包含 LaWAM 主要模型权重；但边缘端代码如果仍按 config 路径检查 `lam_ckpt_path`，包内也带了 `latent_action_model/logs/dino_large_vae/lam_release/`。
- 如果边缘端缺少 Qwen/DINO 等外部基础权重路径，需要按目标机器实际路径修改 `config.yaml` 里的相关字段。

## 本次 checkpoint

- run id: `three_cubes_1_1k_8gpu_lawm_action`
- checkpoint path: `results/Checkpoints/three_cubes/0728_144659+three_cubes_1_1k_8gpu_lawm_action/final_model/pytorch_model.pt`
- 训练步数: `1000`
- 训练数据: `Three_Cubes_1`
