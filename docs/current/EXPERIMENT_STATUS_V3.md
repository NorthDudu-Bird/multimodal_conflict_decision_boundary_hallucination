# Experiment Status: V3 Expanded Strict-Colors Rerun

## What Changed in V3

Compared with the previous strict-colors rerun, this v3 refresh makes five substantive changes:

1. The formal strict-color dataset was cleaned again after manual review, and ten ambiguous images were explicitly excluded in the manifest rather than silently skipped downstream.
2. The analysis set was expanded from the earlier 98 included images to 140 included images using additional manually screened clean samples from the processed Stanford Cars pool.
3. The faithful rule was kept strictly exact across the mainline pipeline: `parsed_label == true_color`.
4. Auxiliary analysis was extended with answer-space compliance outputs.
5. All currently integrated models were rerun on the same new v3 prompts and parsed with the same strict rules.

## Newly Excluded Manual-Review Images

The following ten images are explicitly removed from formal v3 analysis:

- `test_01079`
- `test_02823`
- `test_04952`
- `test_05175`
- `test_06328`
- `test_06708`
- `test_06787`
- `train_04054`
- `train_07422`
- `train_01346`

These rows are recorded in:

- `data/processed/stanford_cars/excluded_manual_review_v3.csv`

and are marked in the expanded final manifest with:

- `include_in_analysis = no`
- `exclusion_reason = ambiguous_after_manual_review`

## New V3 Data and Prompt Files

The main v3 files are:

- `data/processed/stanford_cars/final_primary_manifest_v4_expanded.csv`
- `data/processed/stanford_cars/final_auxiliary_manifest_v4_expanded.csv`
- `data/processed/stanford_cars/excluded_records_v4_expanded.csv`
- `analysis/current/color_distribution_v4_expanded.csv`
- `analysis/current/color_distribution_v4_expanded.md`
- `prompts/current/primary_prompts_v3.csv`
- `prompts/current/auxiliary_prompts_v3.csv`
- `prompts/current/smoke_prompts_v3.csv`

## New Analysis Outputs in V3

Primary outputs now emphasize descriptive rates plus exact confidence intervals for:

- `HR` / conflict-aligned rate
- faithful rate
- other-wrong rate

Auxiliary outputs retain the same outcome summaries and now also add:

- `answer_space_compliance_rate`
- `in_space_conflict_aligned`
- `out_of_space_faithful`
- `other_answer_space_behavior`

The main auxiliary compliance files are:

- `analysis/current/auxiliary_v3/answer_space_compliance_metrics.csv`
- `analysis/current/auxiliary_v3/answer_space_behavior_breakdown.csv`
- `analysis/current/auxiliary_v3/plots/auxiliary_answer_space_compliance.png`
- `analysis/current/auxiliary_v3/plots/auxiliary_answer_space_breakdown.png`

These files are also copied into:

- `analysis/current/cross_model_v3/`

## Model Rerun Status

The following currently integrated models completed the v3 rerun:

- `qwen2vl7b`
- `llava15_7b`
- `internvl2_8b`

Completed output directories:

- `outputs/current/qwen2vl7b_smoke_v3/`
- `outputs/current/qwen2vl7b_primary_v3/`
- `outputs/current/qwen2vl7b_auxiliary_v3/`
- `outputs/current/llava15_7b_smoke_v3/`
- `outputs/current/llava15_7b_primary_v3/`
- `outputs/current/llava15_7b_auxiliary_v3/`
- `outputs/current/internvl2_8b_smoke_v3/`
- `outputs/current/internvl2_8b_primary_v3/`
- `outputs/current/internvl2_8b_auxiliary_v3/`

## Recommended Files to Open First

If you want the fastest path into the v3 evidence, start here:

1. `docs/current/results_ready_summary_v3.md`
2. `analysis/current/primary_v3/analysis_summary.md`
3. `analysis/current/auxiliary_v3/analysis_summary.md`
4. `analysis/current/auxiliary_v3/answer_space_compliance_metrics.csv`
5. `analysis/current/cross_model_v3/README.md`
6. `reports/current/strict_colors_multimodel_results_viewer.html`

## Current Interpretation Status

The v3 rerun sharpens the same qualitative split seen earlier:

- primary open-answer conflict alignment remains concentrated in LLaVA, especially under presuppositional framing
- Qwen and InternVL remain much more stable under the primary open-answer task
- auxiliary conditions continue to act as strong frame-following probes rather than simple extensions of the primary task

The one unresolved dataset limitation is class imbalance: `green` and `yellow` remain scarce even after expansion. For that reason, v3 is best read as a prompt-mechanism study with stricter data cleaning, not as a clean test of color-category differences.
