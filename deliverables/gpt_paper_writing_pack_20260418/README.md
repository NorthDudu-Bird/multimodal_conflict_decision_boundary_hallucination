# GPT Paper Writing Pack

这个压缩包用于继续论文写作，内容已经对齐当前最新版实验主线。

## 建议上传给 GPT 的顺序

1. 先上传本压缩包中的核心文件
2. 再把 `GPT_PROMPT_TEMPLATE.md` 中的总控提示词发给 GPT
3. 按需继续发送“正文写作”“辅助实验写作”两段后续提示

## 包内重点文件

- `docs/experiment_plan.md`
- `results/final_result_summary.md`
- `results/main/table1_main_metrics.md`
- `results/main/main_summary.md`
- `results/main/main_exact_tests.csv`
- `results/auxiliary/table3_aux_metrics.md`
- `results/auxiliary/aux_summary.md`
- `results/auxiliary/aux_exact_tests.csv`
- `results/main/figure2_conflict_aligned_rates.png`
- `results/appendix/dataset_distribution.png`

## 说明

- 正文主线只围绕最终平衡评测集
- Stanford-only 相关内容仅作数据集构建说明或 appendix sanity check
- `results/auxiliary/` 是正式辅助实验目录
- `results/aux/` 不再作为默认引用路径
