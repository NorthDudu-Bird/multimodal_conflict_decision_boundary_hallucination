# Multi-Turn Persuasion Summary

## All-Model Readout

| model | highest_new_cell | highest_rate | highest_count | new_cells_at_or_below_1pct |
| --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | two_turn_persuasion | 0.33% | 1/300 | 3/3 |
| LLaVA-1.5-7B | single_turn_false_note | 0.33% | 1/300 | 3/3 |
| InternVL2-8B | three_turn_persuasion | 74.67% | 224/300 | 1/3 |

## LLaVA-1.5-7B Focus Metrics

| factor_id | condition_name | tone_strength | injection_position | false_text_form | answer_format | n | conflict_aligned_n | conflict_aligned_rate | faithful_n | faithful_rate | parse_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REF_C0_label_set | C0_neutral | neutral |  | none | label_set | 300 | 0 | 0.00% | 300 | 100.00% | 0 |
| REF_C3_original_label_set | C3_presupposition_correction_allowed | presuppositional |  | presupposition_correction_allowed | label_set | 300 | 27 | 9.00% | 273 | 91.00% | 0 |
| REF_C4_original_label_set | C4_stronger_open_conflict | stronger_open_conflict |  | repeated_false_report | label_set | 300 | 10 | 3.33% | 290 | 96.67% | 0 |
| single_turn_false_note | MT1_single_turn_false_note | assertive | question_body | single_turn_false_note | constrained_short_label | 300 | 1 | 0.33% | 299 | 99.67% | 0 |
| two_turn_persuasion | MT2_two_turn_persuasion | assertive_context | previous_user_turn | previous_turn_false_statement | constrained_short_label | 300 | 1 | 0.33% | 299 | 99.67% | 0 |
| three_turn_persuasion | MT3_three_turn_persuasion | repeated_context | previous_user_turn | repeated_previous_turn_false_statement | constrained_short_label | 300 | 0 | 0.00% | 300 | 100.00% | 0 |

## Key Paired Tests

- within_model_vs_C0: `REF_C3_original_label_set` vs `REF_C0_label_set` diff=9.00 pp, current-only=27, reference-only=0, Holm p=1.94e-07.
- within_model_vs_C0: `REF_C4_original_label_set` vs `REF_C0_label_set` diff=3.33 pp, current-only=10, reference-only=0, Holm p=0.0234.

## Interpretation

- The multi-turn module is an extension diagnostic. It tests whether short context accumulation increases conflict following while keeping the final question neutral in MT2/MT3.
- LLaVA and Qwen do not show a meaningful monotonic persuasion effect in this compact setting: LLaVA MT1/MT2/MT3 are 0.33%, 0.33%, and 0.00%; Qwen remains at or below 0.33%.
- InternVL2 is the clear exception: MT2 reaches 21.33% and MT3 reaches 74.67%, despite zero conflict following in the original single-turn C0-C4 mainline.
- This changes the boundary, not the mainline: multi-turn context accumulation can create a strong model-specific vulnerability, but it should remain an appendix/extension result rather than replacing the frozen single-turn C0-C4 evidence chain.
