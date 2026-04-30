# 当前论文冻结实验计划

## 研究目标

本文只研究一个窄问题：在“车身主颜色”任务中，当文本提示给出错误颜色时，视觉语言模型的输出更受视觉证据还是语言提示驱动。

这不是方法创新论文，也不扩展成模型规模效应、跨任务普适规律或决策边界理论。

## 冻结主线

- 正文数据：`data/balanced_eval_set/final_manifest.csv`
- 模型：`LLaVA-1.5-7B`、`Qwen2-VL-7B-Instruct`、`InternVL2-8B`
- 正文实验：
  - `C0` 基线
  - `C0-C4` 主实验
  - `A1/A2` 辅助实验
- 加固实验：
  - `C3` prompt wording 变体控制
  - parser 映射审查
  - appendix 来源分层 sanity check

## 当前结论边界

- 三模型在 `C0` 下均保持视觉忠实。
- 在原始强误导开放式模板下，`LLaVA-1.5-7B` 在 `C3` 与 `C4` 出现有限但显著的 conflict-aligned 行为。
- `Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 基本保持视觉一致。
- `C3` prompt-variant 控制表明该现象对 wording 敏感，因此更稳妥的表述是“有限、模板敏感的语言偏差”，而不是稳定的跨模板规律。

## 正文与附录分工

- 正文：
  - 平衡评测集
  - `C0`
  - `C0-C4`
  - 三模型比较
  - `A1/A2`
- 附录：
  - 按 `source_dataset` 分层的 sanity check
  - parser alias 审查样本
  - 必要的代表性案例

## 明确不再扩展

- 新属性任务
- 新模型堆表
- 模型规模效应分析
- 阈值曲线或复杂建模
- 旧 Stanford-only 并列主流程
