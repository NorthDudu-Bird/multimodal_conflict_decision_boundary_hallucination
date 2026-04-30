# Strengthening Audit Summary

## 保留的实验

- 保留原始正文主线：`C0-C4` 主实验、三模型、300 张平衡评测集不变。
- 保留 `A1/A2` 作为辅助实验，不把它升级成正文主证据。
- 保留原结论中的“视觉优先为主、偏差集中在特定模型和强误导条件”的基本框架。

## 修改的部分

- `results/main/table1_main_metrics.csv` 与 `.md` 已升级为论文表，显式加入 `other_wrong`，并在 `conflict_aligned` 单元格上加入显著性标记。
- `results/main/figure2_conflict_aligned_rates.png` 已重画为更接近论文风格的版本，使用 `C0-C4` 短标签、正式模型名和 CI 误差条。
- `results/appendix/stanford_core_sanity_check.csv` 与 `.md` 已从“旧 Stanford-only vs balanced overall”改成“最终 balanced set 内部按 `source_dataset` 分层”的附录 sanity check。
- `results/final_result_summary.md` 已更新为包含主实验统计、prompt robustness、parser audit 与 appendix sanity check 的综合摘要。

## 新增的部分

- `results/main/main_key_tests.csv`
- `results/main/main_stats_summary.md`
- `results/robustness/prompt_variant_metrics.csv`
- `results/robustness/prompt_variant_exact_tests.csv`
- `results/robustness/prompt_variant_summary.md`
- `results/parser/label_mapping_audit.md`
- `results/parser/ambiguous_outputs_sample.csv`
- `results/results_discussion_summary.md`
- `results/reproducibility_comparison.csv`
- `results/reproducibility_audit.md`

## 可以保留的结论

- 三模型在 `C0` 基线下均保持完美视觉一致性。
- 在原始主实验模板下，`LLaVA-1.5-7B` 仅在强误导开放式提示中出现有限 conflict-aligned 行为，其中 `C3` 最明显、`C4` 次之。
- 在原始主实验模板下，`Qwen2-VL-7B-Instruct` 与 `InternVL2-8B` 基本保持视觉一致，未表现出与 `LLaVA-1.5-7B` 相同量级的偏移。
- 数据来源附录检查显示：`LLaVA-1.5-7B` 的 `C3` 效应在 `StanfordCars` 与 `VCoR` 中方向一致，幅度有差异，但不推翻主实验结论。
- parser audit 显示当前主实验结论不依赖复杂后处理；主实验输出全部处于基础单标签范围内。
- 全量重跑后的 `results/reproducibility_audit.md` 显示 canonical 结果与锁定快照一致，因此当前数字具备可复现性支撑。

## 必须降级的结论

- 不能再把 `C3` 下观察到的 `LLaVA-1.5-7B` 偏移写成“对等强度 prompt wording 稳定成立”的现象。
- 更稳妥的写法应是：在原始 `C3` 模板下可观察到显著但有限的语言偏差；当提示 wording 改写后，该效应明显减弱，甚至消失。
- 因此不应把当前结果上升为更宽泛的“模型普遍语言偏置”或“稳定跨模板规律”。
