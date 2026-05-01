# Answer Format Control Summary

## All-Model Readout

| model | highest_new_cell | highest_rate | highest_count | new_cells_at_or_below_1pct |
| --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | yesno_false_claim | 1.00% | 3/300 | 8/8 |
| LLaVA-1.5-7B | free_c3_presupposition | 2.33% | 7/300 | 5/8 |
| InternVL2-8B | free_c4_stronger_open | 0.33% | 1/300 | 8/8 |

## LLaVA-1.5-7B Focus Metrics

| factor_id | condition_name | tone_strength | injection_position | false_text_form | answer_format | n | conflict_aligned_n | conflict_aligned_rate | faithful_n | faithful_rate | parse_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REF_C0_label_set | C0_neutral | neutral |  | none | label_set | 300 | 0 | 0.00% | 300 | 100.00% | 0 |
| REF_C3_original_label_set | C3_presupposition_correction_allowed | presuppositional |  | presupposition_correction_allowed | label_set | 300 | 27 | 9.00% | 273 | 91.00% | 0 |
| REF_C4_original_label_set | C4_stronger_open_conflict | stronger_open_conflict |  | repeated_false_report | label_set | 300 | 10 | 3.33% | 290 | 96.67% | 0 |
| free_neutral | FC1_free_neutral | neutral | none | none | free_color_word | 300 | 0 | 0.00% | 298 | 99.33% | 0 |
| free_c3_presupposition | FC2_free_c3 | presuppositional | question_body | presupposition | free_color_word | 300 | 7 | 2.33% | 292 | 97.33% | 0 |
| free_c4_stronger_open | FC3_free_c4 | stronger_open_conflict | question_body | repeated_false_report | free_color_word | 300 | 1 | 0.33% | 298 | 99.33% | 0 |
| multiple_choice_neutral | FC4_mc_neutral | neutral | none | none | multiple_choice_label | 300 | 0 | 0.00% | 299 | 99.67% | 0 |
| multiple_choice_c3_presupposition | FC5_mc_c3 | presuppositional | question_body | presupposition | multiple_choice_label | 300 | 4 | 1.33% | 296 | 98.67% | 0 |
| multiple_choice_c4_stronger_open | FC6_mc_c4 | stronger_open_conflict | question_body | repeated_false_report | multiple_choice_label | 300 | 1 | 0.33% | 299 | 99.67% | 0 |
| yesno_false_claim | FC7_yesno_false_claim | assertive | question_body | direct_false_claim | yes_no | 300 | 4 | 1.33% | 296 | 98.67% | 0 |
| yesno_report_correct | FC8_yesno_report_correct | stronger_open_conflict | question_body | repeated_false_report | yes_no | 300 | 0 | 0.00% | 300 | 100.00% | 0 |

## Key Paired Tests

- within_model_vs_C0: `REF_C3_original_label_set` vs `REF_C0_label_set` diff=9.00 pp, current-only=27, reference-only=0, Holm p=4.47e-07.

## Interpretation

- The format-control module separates false-text effects from answer-format effects. Yes/no rows measure acceptance of a false claim, not color-label production, and must not be pooled with C0-C4 label-set rates without that caveat.
- For LLaVA, original label-set C3 is 9.00%, while matched free color-word C3 is 2.33%, multiple-choice C3 is 1.33%, and yes/no false-claim acceptance is 1.33%. The original open label-set framing therefore appears to amplify the observed shift.
- Qwen and InternVL2 remain near zero across most formal answer formats, so answer format does not create a broad cross-model conflict-following effect in this diagnostic.
- Use A1/A2 only as auxiliary stress tests; they are not formal answer-format-control cells.
