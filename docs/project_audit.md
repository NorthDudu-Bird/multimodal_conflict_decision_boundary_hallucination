# 最终清理审计

## 保留的子系统

- `data/`：平衡评测集与 Stanford appendix 清单
- `data_external/`：VCoR 选择记录与来源元数据
- `prompts/c0_c4/`、`prompts/a1_a2/`：正式提示表
- `scripts/`：论文主线实际调用的构建、推理、解析、分析和复现校验脚本
- `results/`：当前论文正式输出
- `deliverables/`：最终 ZIP 交付包

## 本次下线或删除的旧内容

- `archive/`
- `analysis/`
- `outputs/`
- 旧预览与 review 目录
- 旧 `current` 配置树
- 旧 `current` prompt 树
- 旧 Stanford-only 并列主流程
- 旧可视化预览页面和旧 review pack
- `decision boundary / threshold / LDI / RPE` 一类不再使用的旧包装

## 清理原则

- 只保留能直接服务当前论文主线复现与投稿交付的内容
- 删除会把读者重新带回旧叙事或旧流水线的入口
- 把结果可信度建立在主实验统计、prompt 变体控制、parser audit 和 source sanity check 上，而不是靠大叙事扩展

## 当前公开入口

- 项目总览：`README.md`
- 快速入口：`START_HERE.md`
- 复现说明：`docs/reproduction.md`
- 冻结实验计划：`docs/experiment_plan.md`
- 官方脚本：
  - `scripts/build_dataset.py`
  - `scripts/run_baseline_c0.py`
  - `scripts/run_main_c0_c4.py`
  - `scripts/run_aux_a1_a2.py`
  - `scripts/run_robustness_c3_prompt_variants.py`
  - `scripts/generate_parser_audit.py`
  - `scripts/make_figures.py`
  - `scripts/verify_reproducibility.py`

## 结论边界

当前仓库支持的最终结论是一个窄而保守的 empirical claim：

- 视觉证据在本任务中总体占主导；
- `LLaVA-1.5-7B` 只在原始强误导开放式模板下出现有限而显著的 conflict-aligned 偏移；
- 该偏移对 wording 敏感，因此不能写成稳定的跨模板规律。
