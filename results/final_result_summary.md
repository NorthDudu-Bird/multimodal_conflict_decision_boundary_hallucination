# 当前结果摘要

## 数据集

- 正式评测集：`data/balanced_eval_set/final_manifest.csv`
- 总样本数：300
- 颜色分布：`red / blue / green / yellow / black / white` 各 50
- 来源构成：StanfordCars 93，VCoR 207

## 主实验 C0-C4

- 三模型在 `C0` 均保持 100.00% 忠实率，`conflict_aligned=0/300`。
- `LLaVA-1.5-7B` 在 `C3` 的冲突一致率为 9.00% [6.26%, 12.78%]，`C4` 为 3.33% [1.82%, 6.03%]。
- `Qwen2-VL-7B-Instruct` 仅在 `C3/C4` 各出现 1 例 conflict-aligned 输出；`InternVL2-8B` 在 `C0-C4` 中未出现 conflict-aligned 输出。
- 主实验 `refusal / other_wrong / parse_error` 全部为 0，因此主推断聚焦在 `conflict_aligned`。

## 关键统计

- LLaVA-1.5-7B 在 C3 相对 `C0` 的 conflict-aligned 增幅达到 Holm 显著 (raw p=1.49e-08, Holm p=8.94e-08)。
- LLaVA-1.5-7B 在 C4 相对 `C0` 的 conflict-aligned 增幅达到 Holm 显著 (raw p=0.0020, Holm p=0.0098)。
- 在 C3，LLaVA-1.5-7B 的 conflict-aligned rate 显著高于 Qwen2-VL-7B-Instruct (raw p=2.16e-07, Holm p=1.08e-06)。
- 在 C3，LLaVA-1.5-7B 的 conflict-aligned rate 显著高于 InternVL2-8B (raw p=1.49e-08, Holm p=8.94e-08)。
- 在 C4，LLaVA-1.5-7B 的 conflict-aligned rate 显著高于 Qwen2-VL-7B-Instruct (raw p=0.0117, Holm p=0.0352)。
- 在 C4，LLaVA-1.5-7B 的 conflict-aligned rate 显著高于 InternVL2-8B (raw p=0.0020, Holm p=0.0078)。

## Prompt Variant Robustness

- `LLaVA-1.5-7B` | Original C3: conflict-aligned=9.00% [6.26%, 12.78%].
- `LLaVA-1.5-7B` | C3-v2: conflict-aligned=1.67% [0.71%, 3.84%].
- `LLaVA-1.5-7B` | C3-v3: conflict-aligned=0.00% [0.00%, 1.26%].
- Prompt-variant robustness details are summarized in `results/robustness/prompt_variant_summary.md`.
- 模板鲁棒性控制显示原始 `C3` 效应并不稳定：`LLaVA-1.5-7B` 从 Original C3 的 9.00% 降到 C3-v2 的 1.67%，并在 C3-v3 降到 0.00%。
- 因此正文应把该现象写成“对原始强误导模板敏感的有限语言偏差”，而不应写成对所有等强度 wording 都稳定成立。

## 解析与附录检查

- 解析规则审查见 `results/parser/label_mapping_audit.md`，别名抽样复核见 `results/parser/ambiguous_outputs_sample.csv`。
- 数据来源 sanity check 已改为最终平衡集内部按 `source_dataset` 分层，见 `results/appendix/stanford_core_sanity_check.md`。

## 关键文件

- Table 1：`results/main/table1_main_metrics.csv`
- Figure 2：`results/main/figure2_conflict_aligned_rates.png`
- Main stats summary：`results/main/main_stats_summary.md`
- Main key tests：`results/main/main_key_tests.csv`
- Prompt robustness：`results/robustness/prompt_variant_summary.md`
- Parser audit：`results/parser/label_mapping_audit.md`
- Appendix sanity check：`results/appendix/stanford_core_sanity_check.md`

## Phase 2 Full Strengthening Addendum

- Per-color split tightens the original LLaVA C3 interpretation: the 27 C3 flips are not dispersed across all six colors; 20/27 are `white->black`, with smaller `black->white` and `blue->red` contributions. Across LLaVA C3/C4, 33/37 main flip row-events are in the achromatic black/white family.
- The completed visual clarity audit reviewed 42 target flip rows and 42 matched faithful controls. Target rows are mostly inspectable (`clear`=38/42), but visual confound flags are more common in targets (11/42) than controls (4/42). This reduces, but does not eliminate, the "images are hard" alternative explanation.
- Prompt factorization shows that false-text form matters. LLaVA reaches 32.00% under title/prefix framing and 16.33% under no-correction presupposition; Qwen is most affected by no-correction presupposition (34.00%), and InternVL2 by title/prefix framing (36.00%).
- Answer-format control shows that original LLaVA C3 is format-sensitive: free-answer C3 is 2.33%, multiple-choice C3 is 1.33%, and yes/no false-claim acceptance is 1.33%. 20/27 original C3 flip rows are not reproduced by all three formal controls.
- Multi-turn persuasion is an appendix-level extension. LLaVA remains near zero across MT1/MT2/MT3 (0.33%, 0.33%, 0.00%), while InternVL2 rises sharply in MT2/MT3 (21.33%/74.67%). This changes the boundary of the paper, not the frozen single-turn mainline.
