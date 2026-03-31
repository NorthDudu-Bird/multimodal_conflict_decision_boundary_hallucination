# Qwen2-VL Setup and Smoke Test Report

## 1. 当前 cv_proj 环境检查结果
- Python: `3.11.2`
- 基础深度学习栈: `torch 2.6.0+cu126`, `torchvision 0.21.0+cu126`
- CUDA: `True`
- GPU: `NVIDIA GeForce RTX 4080 Laptop GPU`
- GPU 显存: `11.99 GB`
- BF16 支持: `True`
- `nvidia-smi` 显示驱动版本 `577.00`，CUDA Runtime `12.9`
- 环境检查日志已写入 `logs/env_check_cv_proj.txt`
- 详细环境报告已写入 `reports/vlm_env_check_report.md`

## 2. 安装或修复了哪些依赖
- 直接写入 `D:\anaconda3\envs\cv_proj\Lib\site-packages` 时出现权限拒绝，因此没有破坏原有 `torch` 环境。
- 采用了项目内本地依赖覆盖层：`vendor/cv_proj_sitepackages/`
- 已补齐的关键依赖:
  - `transformers 4.51.3`
  - `accelerate 1.8.1`
  - `safetensors 0.6.2`
  - `bitsandbytes 0.49.2`
  - `qwen-vl-utils 0.0.14`
- 继续复用 `cv_proj` 中已有依赖:
  - `huggingface_hub 0.34.4`
  - `pillow 11.1.0`
  - `pandas 2.2.2`
  - `tqdm 4.67.1`
  - `sentencepiece 0.2.1`
- 安装日志已写入 `logs/env_install_log.txt`

## 3. GPU / CUDA 是否满足要求
- 当前机器可以进行 Qwen2-VL GPU 推理。
- 但 `12 GB` 显存对 `Qwen2-VL-7B-Instruct` 的全精度或半精度加载偏紧张，不建议直接走 BF16/FP16 全量加载。
- 在当前机器上，`4-bit` 量化是可运行且更稳妥的方案。

## 4. Qwen2-VL-7B 是否成功部署
- 成功。
- 模型已下载到本地目录: `models/qwen2_vl_7b/`
- 目录下已包含 `5` 个 `safetensors` 分片和完整 tokenizer / processor 配置。
- 模型下载日志已写入 `logs/qwen2vl_model_download.log`
- 该模型为公开仓库，本次下载未额外要求 Hugging Face 登录。

## 5. 最终采用的加载方式
- 最终采用: `4-bit NF4 量化`
- 具体方式:
  - `bitsandbytes 4-bit`
  - `bnb_4bit_quant_type='nf4'`
  - `bnb_4bit_use_double_quant=True`
  - 计算 dtype 使用 `bfloat16`
  - `device_map='auto'`
- 原因:
  - 当前 GPU 仅约 `12 GB` 显存
  - 单样本真实探针已证明 `4-bit` 可以成功加载并推理
  - 该方案比直接 BF16/FP16 更适合当前硬件

## 6. 冒烟测试是否成功
- 成功。
- 已运行脚本: `scripts/run_qwen2vl_smoke_test.py`
- 成功完成前 `20` 条样本推理。
- 结果文件已写入 `data/metadata/qwen2vl_smoke_test_results.csv`
- 运行日志已写入 `logs/qwen2vl_smoke_test.log`

## 7. 成功推理了多少条
- `20 / 20` 条成功完成
- `status=ok` 的记录数为 `20`
- 当前结果文件未出现中途报错终止

## 8. 当前存在的风险或限制
- `cv_proj` 环境目录本身不可写，因此补包采用了项目内依赖覆盖层，而不是把新包装进 conda 环境目录。
- `12 GB` 显存仍属于 7B 多模态模型的边界配置；若后续你把 `batch_size` 调高，OOM 风险会明显上升。
- 下载阶段 `huggingface_hub` 曾提示未安装 `hf_xet`，但最终未阻碍模型完整下载。
- 冒烟测试已出现明显“语言牵引/视觉幻觉”现象，例如 `2592_S0` 输出中出现了与图像不符的额外描述；这属于实验目标现象，不是脚本故障。
- 目前结果中的 `label / language_consistent / vision_consistent / ambiguous` 仍为空，等待你后续人工标注或进一步自动判别。

## 9. 下一步如何跑完整 200 条实验
- 可直接使用 `scripts/run_qwen2vl_batch.py`
- 默认输出文件: `data/metadata/qwen2vl_batch_results.csv`
- 默认支持:
  - 跳过已完成 `sample_id`
  - 断点续跑
  - 分批写入 CSV
- 推荐先保持保守参数:
  - `batch_size=1`
  - `max_new_tokens=96`
  - `temperature=0.0`
- 建议运行命令:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\anaconda3\envs\cv_proj\python.exe' scripts\run_qwen2vl_batch.py --batch-size 1 --max-new-tokens 96 --temperature 0.0
```

## 10. 本次新增或落地的文件
- `scripts/_local_deps.py`
- `scripts/check_vlm_env.py`
- `scripts/qwen2vl_runtime.py`
- `scripts/run_qwen2vl_smoke_test.py`
- `scripts/run_qwen2vl_batch.py`
- `logs/env_check_cv_proj.txt`
- `logs/env_install_log.txt`
- `logs/qwen2vl_model_download.log`
- `logs/qwen2vl_single_probe.log`
- `logs/qwen2vl_smoke_test.log`
- `reports/vlm_env_check_report.md`
- `reports/qwen2vl_setup_and_smoke_test_report.md`
- `data/metadata/qwen2vl_smoke_test_results.csv`
- `models/qwen2_vl_7b/`

## 11. 当前结论
- 你的当前环境经过最小修复后，已经能够支撑这轮论文实验。
- `Qwen2-VL-7B-Instruct` 在当前机器上并非“完全不可行”，但应固定使用 `4-bit` 量化并避免增大 batch。
- 你现在可以直接开始完整 `200` 条实验。
