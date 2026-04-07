# Method-Ready Text: Strict Colors

## Dataset Curation

We used the existing Stanford Cars clean subset already curated in the repository as the base image pool for the car-color experiment. The strict-colors rerun started from the current 120-image Stanford Cars main subset and then applied an additional color-boundary restriction designed to reduce ambiguity in coarse body-color naming. Only images whose reviewed true color belonged to the set `{red, blue, green, yellow, black, white}` were retained for the formal main analysis. Images labeled `silver`, `gray`, or `other` were preserved in audit files but removed from the formal primary and auxiliary reruns. We additionally retained the existing exclusion of a small number of dark, low-saturation boundary cases whose visible body color remained unstable under the coarse label inventory. The resulting strict-color manifest is stored in `data/processed/stanford_cars/final_primary_manifest_strict_colors.csv`, with excluded rows documented in `data/processed/stanford_cars/excluded_records_strict_colors.csv` and `data/processed/stanford_cars/excluded_due_to_ambiguous_colors.csv`.

## Final Strict Subset

The base main subset contained 120 images. The strict-colors rerun excluded 19 images because their reviewed true color was `silver`, `gray`, or `other`, and excluded 3 further low-saturation boundary cases. The final strict-color analysis set therefore contained 98 images. The retained reviewed true-color distribution was: 40 red, 27 black, 14 white, 13 blue, 2 green, and 2 yellow.

## Conflict-Color Construction

Each included image was paired with a single misleading conflict color such that `conflict_color != true_color` and the conflict label also belonged to the strict retained color set. The conflict mapping used in the current rerun was: `red -> blue`, `blue -> red`, `green -> yellow`, `yellow -> red`, `black -> white`, and `white -> black`. Excluded colors such as `silver` and `gray` were never used as formal conflict labels in the strict prompt tables.

## Prompt Design

The strict-colors rerun preserved the current restructured prompt design rather than reverting to the older `S0-S7` intensity scheme. The primary study compared five open-answer conditions: `C0_neutral`, `C1_weak_suggestion`, `C2_false_assertion_open`, `C3_presupposition_correction_allowed`, and `C4_stronger_open_conflict`. Two additional manipulations were analyzed separately as auxiliary conditions: `A1_forced_choice_red_family` and `A2_counterfactual_assumption`.

For the primary study, models were instructed to output exactly one label from the formal strict inventory `{red, blue, green, yellow, black, white, other}`. `silver` and `gray` were intentionally removed from the formal primary answer space. Auxiliary prompts continued to use restricted conflict-family answer spaces around the currently assigned conflict color. On the final strict subset, this yielded 490 primary prompt rows and 196 auxiliary prompt rows per full multimodel rerun.

## Multi-Model Inference

We evaluated three vision-language models using the same sequential full-precision inference pipeline: `Qwen/Qwen2-VL-7B-Instruct`, `llava-hf/llava-1.5-7b-hf`, and `OpenGVLab/InternVL2-8B`. Models were run one at a time with batch size 1, with CPU offload where necessary, so that all conditions, parsing rules, and statistics were matched across models.

## Output Parsing

Raw outputs were parsed into structured fields including `parsed_label`, `parse_success`, `parse_method`, and `outcome_type`. The parser first attempted exact one-label extraction from the expected answer inventory. When a model emitted a recognizable but non-standard color word, the parser could still recover the label as a parsable non-standard output. However, faithful evaluation was defined strictly by exact canonical match to the reviewed true label:

- `faithful`: `parsed_label == true_color`
- `conflict_aligned`: `parsed_label == conflict_color`
- `other_wrong`: parsing succeeded but the label matched neither `true_color` nor `conflict_color`
- `refusal_or_correction`: the model did not follow the required one-label format and instead semantically refused or explicitly corrected the prompt
- `parse_error`: no reliable label could be extracted

No compatibility rule was used for `silver`, `gray`, or `white`. In particular, `silver -> gray` was not treated as faithful.

## Evaluation Metrics

The main outcome of interest was hallucination rate, defined strictly as the proportion of rows with `outcome_type == conflict_aligned`. For the primary study, we additionally report `RPE(c) = HR(c) - HR(C0)` for each condition `c`, and `LDI = mean(HR(C2), HR(C3), HR(C4)) - mean(HR(C0), HR(C1))`. For the auxiliary study, we report `FSS = HR(A1) - HR(C2)` using the matched primary `C2_false_assertion_open` rows for each model.

## Statistical Analysis

For each model-condition cell, we report exact hallucination rates, exact 95% confidence intervals, and outcome-type counts. Cross-model analyses were conducted on the shared image set. The preferred inferential model was a mixed-effects logistic regression predicting `is_conflict_aligned` from model, condition, and their interaction, with a random intercept for `image_id`. Exact confidence intervals and bootstrap summaries were retained to stabilize interpretation under rare-event settings, especially for models and conditions with zero conflict-aligned outputs.
