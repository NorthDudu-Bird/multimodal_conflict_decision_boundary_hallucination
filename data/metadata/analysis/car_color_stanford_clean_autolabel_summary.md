# Car Color Attribute Conflict Auto-Label Summary

## Files
- input prompt csv: data/metadata/prompts/car_color_attribute_conflict_stanford_clean_s0_s7_30x8.csv
- input runtime csv: data/metadata/outputs_raw/qwen2vl7b_stanford_clean_runtime.csv
- prelabeled output: data/metadata/outputs_labeled/qwen2vl7b_stanford_clean_prelabeled.csv
- manual review list: data/metadata/outputs_labeled/qwen2vl7b_stanford_clean_manual_review.csv
- final labeled template: data/metadata/outputs_labeled/qwen2vl7b_stanford_clean_final_labeled.csv

## Overall Counts
- input rows: 240
- faithful: 112
- hallucination: 60
- needs_manual_review: 68
- manual review rows: 74

## By Prompt Level
- S0: faithful=17, hallucination=2, needs_manual_review=11
- S1: faithful=17, hallucination=2, needs_manual_review=11
- S2: faithful=17, hallucination=2, needs_manual_review=11
- S3: faithful=19, hallucination=1, needs_manual_review=10
- S4: hallucination=30
- S5: faithful=18, hallucination=3, needs_manual_review=9
- S6: faithful=17, hallucination=3, needs_manual_review=10
- S7: faithful=7, hallucination=17, needs_manual_review=6

## Main Manual-Review Reasons
- other_positive_color: 62
- no_stable_rule_hit: 5
- positive_conflict_color: 5
- positive_true_color: 1
- true_and_conflict_both_positive: 1
