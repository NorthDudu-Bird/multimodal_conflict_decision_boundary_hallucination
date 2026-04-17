# rerun_summary

## 数据版本
- Stanford-only 对照：`data/processed/stanford_cars/primary_core_stanford_only.csv`
- Expanded balanced：`data/processed/stanford_cars/primary_expanded_balanced_with_vcor.csv`

## Rerun 完成情况
- Stanford-only：三模型 smoke / primary / auxiliary 全部完成。
- Expanded balanced：三模型 smoke / primary / auxiliary 全部完成。
- Auxiliary answer-space compliance 已输出。
- 所有 parse review CSV 均为 0 行，说明本轮没有额外 parse-error 复核积压。

## Primary 主结论
- Qwen2-VL-7B：expanded 下仍接近零 hallucination，仅 `C3` 与 `C4` 各出现 1/300 的 `conflict_aligned`。
- LLaVA-1.5-7B：主效应模式保持不变，hallucination 峰值仍出现在 `C3_presupposition_correction_allowed`。
- InternVL2-8B：Stanford-only 与 expanded 两版均维持 0 `conflict_aligned`。

## Expanded 主实验峰值
- qwen2vl7b: 峰值条件=C3_presupposition_correction_allowed, conflict_aligned=0.33%, faithful=99.67%
- llava15_7b: 峰值条件=C3_presupposition_correction_allowed, conflict_aligned=9.00%, faithful=91.00%
- internvl2_8b: 峰值条件=C0_neutral, conflict_aligned=0.00%, faithful=100.00%

## 稳健性判断
- 扩色后，Primary 的核心排序没有改变：LLaVA 对语言框架最敏感，且最强触发条件仍是 `C3_presupposition_correction_allowed`；Qwen 仅有极低水平响应；InternVL2 基本稳定。
- 因此 expanded balanced 版本构成对主结论的 robustness strengthening，而不是推翻。
