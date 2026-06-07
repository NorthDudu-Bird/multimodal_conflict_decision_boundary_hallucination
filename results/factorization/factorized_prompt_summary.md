# Factorized Prompt Summary

## All-Model Readout

| model | highest_new_cell | highest_rate | highest_count | new_cells_at_or_below_1pct |
| --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | presupposition_no_correction | 34.00% | 102/300 | 3/5 |
| LLaVA-1.5-7B | title_prefix_framing | 32.00% | 96/300 | 3/5 |
| InternVL2-8B | title_prefix_framing | 36.00% | 108/300 | 2/5 |

## LLaVA-1.5-7B Focus Metrics

| factor_id | condition_name | tone_strength | injection_position | false_text_form | answer_format | n | conflict_aligned_n | conflict_aligned_rate | faithful_n | faithful_rate | parse_error_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REF_C0_label_set | C0_neutral | neutral |  | none | label_set | 300 | 0 | 0.00% | 300 | 100.00% | 0 |
| REF_C3_original_label_set | C3_presupposition_correction_allowed | presuppositional |  | presupposition_correction_allowed | label_set | 300 | 27 | 9.00% | 273 | 91.00% | 0 |
| REF_C3_v2_wording | C3_presupposition_correction_allowed | assertive | question_body | quoted_or_reported_claim | label_set | 300 | 5 | 1.67% | 294 | 98.00% | 0 |
| REF_C3_v3_wording | C3_presupposition_correction_allowed | assertive | question_body | explicit_prompt_says | label_set | 300 | 0 | 0.00% | 299 | 99.67% | 0 |
| REF_C4_original_label_set | C4_stronger_open_conflict | stronger_open_conflict |  | repeated_false_report | label_set | 300 | 10 | 3.33% | 290 | 96.67% | 0 |
| quoted_claim_user | F1_quoted_claim | assertive | question_body | quoted_false_claim | constrained_short_label | 300 | 2 | 0.67% | 297 | 99.00% | 0 |
| indirect_hint_user | F2_indirect_hint | suggestive | question_body | indirect_hint | constrained_short_label | 300 | 1 | 0.33% | 298 | 99.33% | 0 |
| title_prefix_framing | F3_title_prefix | assertive | title_prefix | title_or_prefix | constrained_short_label | 300 | 96 | 32.00% | 203 | 67.67% | 0 |
| presupposition_no_correction | F4_presupposition_no_correction | presuppositional | question_body | presupposition_without_correction_cue | constrained_short_label | 300 | 49 | 16.33% | 251 | 83.67% | 0 |
| previous_turn_false_context | F5_previous_turn_context | assertive | previous_user_turn | previous_turn_false_statement | constrained_short_label | 300 | 0 | 0.00% | 300 | 100.00% | 0 |

## Key Paired Tests

- within_model_vs_C0: `REF_C3_original_label_set` vs `REF_C0_label_set` diff=9.00 pp, current-only=27, reference-only=0, Holm p=3.28e-07.
- within_model_vs_C0: `REF_C4_original_label_set` vs `REF_C0_label_set` diff=3.33 pp, current-only=10, reference-only=0, Holm p=0.0371.
- within_model_vs_C0: `presupposition_no_correction` vs `REF_C0_label_set` diff=16.33 pp, current-only=49, reference-only=0, Holm p=8.53e-14.
- within_model_vs_C0: `title_prefix_framing` vs `REF_C0_label_set` diff=32.00 pp, current-only=96, reference-only=0, Holm p=6.31e-28.

## Interpretation

- The factorization module should be read as a prompt-form diagnostic. Compare the original C3 reference to quoted, indirect, title/prefix, presuppositional, and previous-turn forms; do not describe the result as a general prompt-engineering law.
- The strongest new factors are not the quoted or indirect hints. They are title/prefix framing and presupposition without an explicit correction cue, with model-specific ordering: Qwen is most affected by no-correction presupposition, while LLaVA and InternVL2 peak under title/prefix framing.
- For LLaVA, original C3 is 9.00%, but title/prefix is 32.00% and no-correction presupposition is 16.33%. The original C3 is therefore a meaningful mainline cell, not the upper bound of prompt susceptibility.
- Quoted claims and indirect hints stay near zero for most models. This argues against a simple 'any false text works' story and supports a narrower account about framing, presupposition, and correction affordances.
