# 多模态冲突决策边界实验仓库

本项目用于复现实验主题：

“跨模态冲突下多模态大模型语言主导偏置的实证研究：基于分级冲突强度的决策边界分析”

当前仓库已经完成以下核心流程：

- 基于 COCO `val2017` 构建 `no-dog` 样本池
- 人工筛选并确认 `50` 张不含狗的图片
- 为每张图片扩展 `S0 / S1 / S2 / S3` 四级存在性冲突 prompt
- 生成 `200` 条实验输入表
- 在本地部署 `Qwen2-VL-7B-Instruct`
- 以 `4-bit NF4` 量化方式完成最小可用冒烟测试

## 目录结构

```text
.
├─ data/
│  ├─ coco/                            COCO 原始数据与标注
│  ├─ metadata/                        样本表、prompt 表、模型输出表
│  ├─ previews/                        数据预览图
│  └─ selected_images/no_dog_sample_50 50 张 no-dog 图片
├─ scripts/                            数据准备、环境检查、推理与批量实验脚本
├─ reports/                            数据准备与实验说明报告
├─ logs/                               运行日志
├─ models/                             本地模型目录（默认已加入 .gitignore）
├─ vendor/                             本地依赖覆盖层（默认已加入 .gitignore）
├─ requirements.txt                    Python 依赖清单
├─ ENVIRONMENT.md                      环境复现说明
└─ README.md                           项目说明
```

## 当前关键文件

- `scripts/prepare_no_dog_subset.py`
  用于从 COCO 中筛选并导出 no-dog 样本。
- `scripts/generate_existence_conflict_prompt_table.py`
  用于将 50 张图片扩展为四级冲突实验输入。
- `scripts/check_vlm_env.py`
  用于检查 Python、Torch、CUDA、GPU 和 Qwen2-VL 依赖是否齐全。
- `scripts/run_qwen2vl_smoke_test.py`
  用于运行前 20 条样本的冒烟测试。
- `scripts/run_qwen2vl_batch.py`
  用于运行完整 200 条实验，支持断点续跑。
- `scripts/chat_qwen2vl.py`
  用于和本地 `Qwen2-VL-7B-Instruct` 做交互式对话，支持附图。
- `data/metadata/no_dog_sample_50.csv`
  50 张已确认 no-dog 图片的基础元数据表。
- `data/metadata/no_dog_sample_50_prompt_levels.csv`
  四级冲突实验输入表，共 200 条。
- `data/metadata/qwen2vl_smoke_test_results.csv`
  已完成的 20 条冒烟测试输出。

## 环境与模型

本仓库当前验证通过的环境组合：

- Python `3.11.2`
- PyTorch `2.6.0+cu126`
- torchvision `0.21.0+cu126`
- CUDA 可用
- GPU：`NVIDIA GeForce RTX 4080 Laptop GPU 12GB`

验证通过的模型配置：

- 模型：`Qwen/Qwen2-VL-7B-Instruct`
- 加载方式：`4-bit NF4`
- 计算 dtype：`bfloat16`

更详细的环境说明见：

- `ENVIRONMENT.md`
- `reports/vlm_env_check_report.md`
- `reports/qwen2vl_setup_and_smoke_test_report.md`

## 快速开始

### 1. 安装环境

```powershell
conda create -n cv_proj python=3.11.2 -y
conda activate cv_proj
pip install -r requirements.txt
```

### 2. 下载模型

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-VL-7B-Instruct', local_dir='models/qwen2_vl_7b')"
```

### 3. 检查环境

```powershell
python scripts/check_vlm_env.py
```

### 4. 运行 20 条冒烟测试

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_qwen2vl_smoke_test.py
```

### 5. 运行完整 200 条实验

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_qwen2vl_batch.py --batch-size 1 --max-new-tokens 96 --temperature 0.0
```

## 对话脚本

交互式启动：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/chat_qwen2vl.py
```

一轮式文本调用：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/chat_qwen2vl.py --prompt "你好，请用一句话介绍你自己。"
```

带图一轮式调用：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/chat_qwen2vl.py --image data/selected_images/no_dog_sample_50/000000002592.jpg --prompt "请描述这张图里有什么。"
```

交互模式支持以下命令：

- `/image <path>`：给下一轮消息附图
- `/image clear`：清除待发送图片
- `/clear`：清空历史对话
- `/status`：查看当前状态
- `/help`：显示帮助
- `/quit`：退出

## 当前进展

目前已经完成：

- no-dog 样本准备
- 四级冲突 prompt 生成
- Qwen2-VL-7B 本地部署
- 20 条冒烟测试跑通
- 结果表与批量实验脚本落地

当前尚未默认纳入 Git 的本地目录：

- `models/`
- `vendor/`

原因是它们分别包含大体积模型权重和本地依赖覆盖层，更适合在复现时本地生成，而不是直接版本化。

## 备注

- 在 `12 GB` 显存上，推荐固定使用 `4-bit` 量化并保持 `batch_size=1`
- 当前实验目标是观察模型在 `S0 → S3` 冲突增强过程中是否出现语言主导偏置、误报和视觉幻觉
- 冒烟测试结果中的 `label / language_consistent / vision_consistent / ambiguous` 字段可用于后续人工标注
