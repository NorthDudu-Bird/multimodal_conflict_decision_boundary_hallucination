# 25-File Paper Writing Pack

这个包专门为“论文写作阶段最多只能上传 25 个文件”的场景制作。

## 包内文件数

- 总文件数：23
- 其中辅助引导文件：2
- 其中核心项目文件：21

## 适用场景

- 让 GPT 帮你写论文的 `Abstract`
- 让 GPT 帮你写 `Results` / `Discussion` / `Limitations`
- 让 GPT 检查当前表述是否夸大
- 让 GPT 基于已有结果润色，而不是改研究方向

## 推荐上传顺序

如果平台一次只能传少量文件，建议优先上传：

1. `README.md`
2. `GPT_PROMPT_TEMPLATE.md`
3. `results/final_result_summary.md`
4. `results/results_discussion_summary.md`
5. `results/main/table1_main_metrics.csv`
6. `results/main/main_stats_summary.md`
7. `results/main/figure2_conflict_aligned_rates.png`
8. `results/robustness/prompt_variant_summary.md`
9. `results/parser/label_mapping_audit.md`
10. `results/appendix/stanford_core_sanity_check.md`

如果可以一次性上传完整包，则直接按当前目录全部上传即可，总数不超过 25。

## 结构说明

- `docs/`：冻结实验设计与复现实验说明
- `data/`：正式评测集 manifest 与摘要
- `results/main/`：正文主实验核心结果
- `results/auxiliary/`：辅助实验压缩版结果
- `results/robustness/`：prompt wording 鲁棒性控制
- `results/parser/`：输出解析可靠性审查
- `results/appendix/`：数据来源附录 sanity check

## 当前写作边界

当前最稳妥的论文口径是：

- 三模型在 `C0` 基线下均保持视觉忠实。
- 在原始强误导开放式模板下，`LLaVA-1.5-7B` 在 `C3` 和 `C4` 出现有限但显著的 conflict-aligned 行为。
- `Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 基本保持视觉一致。
- `C3` prompt-variant 控制表明该现象对 wording 敏感，因此不能写成稳定的跨模板规律。

## 不建议让 GPT 做的事

- 不要让 GPT 重写研究问题
- 不要扩展到新任务、新模型、新理论叙事
- 不要把当前结果包装成“普遍规律”
- 不要忽略 robustness / parser / source sanity check 给出的结论边界
