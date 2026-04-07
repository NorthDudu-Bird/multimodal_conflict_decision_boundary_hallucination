# Method-Ready Text: V3 Expanded Strict-Colors Rerun

## Dataset Scope and Cleaning

The v3 rerun starts from the repository's current strict-colors main analysis subset and then applies one more round of manual cleaning before re-running all models. The allowed formal color set remains:

- `red`
- `blue`
- `green`
- `yellow`
- `black`
- `white`

No `silver`, `gray`, or `other` rows are allowed back into the formal main analysis.

The prior strict-colors formal subset contained 98 included images. We then removed ten images that remained too unstable after manual review:

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

These rows are now explicitly marked with `include_in_analysis = no` and `exclusion_reason = ambiguous_after_manual_review` in the new v3 manifests rather than being skipped only at runtime. The dedicated exclusion audit file is `data/processed/stanford_cars/excluded_manual_review_v3.csv`.

After removing those ten rows, the pre-expansion strict subset contained 93 included images. We then expanded the clean analysis set by manually screening additional candidates from the processed Stanford Cars clean pool while preserving the same strict-color and clean-sample rules. The final expanded v3 analysis set contains 140 included images, stored in:

- `data/processed/stanford_cars/final_primary_manifest_v4_expanded.csv`
- `data/processed/stanford_cars/final_auxiliary_manifest_v4_expanded.csv`

The expanded manifest also keeps excluded rows for auditability. `data/processed/stanford_cars/excluded_records_v4_expanded.csv` summarizes all v3 exclusions.

## Clean-Sample Rules

Newly added images were screened under the same clean-subset standards used for the strict-color rerun:

- the target vehicle remains visually primary
- the car body color is dominated by a single coarse color
- no obvious two-tone paint, wrap, or livery pattern drives the decision
- glare or reflections do not dominate the body-color judgment
- the crop keeps the main body visible enough for coarse color recognition
- the scene does not contain a large misleading same-color background object
- the image does not remain high-ambiguity after manual review

We explicitly did not relax these rules just to reach a larger sample count. The target expansion range was 150-200 images, but the final clean set stopped at 140 because acceptable `green` and `yellow` examples remained genuinely scarce after manual review, and five additional expansion candidates were later removed after further ambiguity passes. The v3 color distribution file is:

- `analysis/current/color_distribution_v4_expanded.csv`
- `analysis/current/color_distribution_v4_expanded.md`

Final included color counts are:

- `red = 56`
- `blue = 21`
- `green = 2`
- `yellow = 7`
- `black = 29`
- `white = 25`

## Conflict-Color Construction

Each retained image is paired with exactly one misleading conflict color, with `conflict_color != true_color` and both labels restricted to the strict formal color inventory. The v3 prompts continue the current strict-colors mapping rather than reverting to older prompt designs.

## Prompt Families

V3 preserves the current prompt families and only refreshes the sample set:

- Primary open-answer conditions:
  - `C0_neutral`
  - `C1_weak_suggestion`
  - `C2_false_assertion_open`
  - `C3_presupposition_correction_allowed`
  - `C4_stronger_open_conflict`
- Auxiliary framework/probe conditions:
  - `A1_forced_choice_red_family`
  - `A2_counterfactual_assumption`

The regenerated v3 prompt tables are:

- `prompts/current/primary_prompts_v3.csv`
- `prompts/current/auxiliary_prompts_v3.csv`
- `prompts/current/smoke_prompts_v3.csv`

All models use the same final 140-image set for both primary and auxiliary prompts. The ten manually excluded ambiguous images do not appear in any v3 prompt table.

## Parsing and the Strict Faithful Definition

The formal v3 faithful rule is intentionally strict and unique across the pipeline:

- `faithful` iff `parsed_label == true_color`

No looser compatibility rule is used. In particular:

- `silver`, `gray`, and `white` are not treated as mutually compatible
- `black` and `blue` are never treated as "close enough"
- no legacy approximate-color mapping counts as faithful

The parser writes the standard outcome categories:

- `conflict_aligned`
- `faithful`
- `other_wrong`
- `refusal_or_correction`
- `parse_error`

For the current v3 formal runs, all three models remained fully parseable under the one-label format, so interpretation centers on `conflict_aligned`, `faithful`, and `other_wrong`.

## Auxiliary Answer-Space Compliance

V3 adds one auxiliary-only interpretive metric: answer-space compliance. For each auxiliary prompt, we record:

- `in_allowed_answer_space = true` if the model output falls inside that prompt's explicitly allowed answer set
- `in_allowed_answer_space = false` otherwise

This is not a new main hypothesis. It is an explanatory probe for the auxiliary results: auxiliary prompts are deliberately stronger framing conditions, so the compliance metric helps separate two behaviors:

- staying inside the prompt-imposed answer space and aligning with the false frame
- escaping the allowed answer space and returning the visually faithful color instead

The auxiliary analysis therefore reports, for each model and auxiliary condition:

- answer-space compliance rate
- `in-space + conflict_aligned`
- `out-of-space + faithful`
- other behavior, if any

## Models and Inference

The v3 rerun evaluates the same three currently integrated multimodal models:

- `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-1.5-7b-hf`
- `OpenGVLab/InternVL2-8B`

All models were rerun on the v3 prompt tables with the same parsing rules and the same downstream analysis scripts. Smoke, primary, and auxiliary inference were executed sequentially so that only one model needed to be resident at a time.

## Statistical Reporting

Primary remains the main experiment and is interpreted as open-answer language-bias sensitivity. For each model-condition cell we report:

- `HR` = conflict-aligned rate
- faithful rate
- other-wrong rate
- exact 95% confidence intervals

Auxiliary remains a secondary probe of strong framing / frame-following behavior. It reports the same outcome rates plus answer-space compliance summaries.

Because several primary model-condition cells remain at or near zero conflict-aligned outputs, descriptive rates and exact confidence intervals remain the main statistical presentation. The mixed-effects logistic model is retained as a secondary cross-model summary when it converges, but we do not rely on unstable rare-event model estimates as the sole basis for interpretation. Color-level comparisons are treated cautiously because the final color distribution remains uneven, especially for `green` and `yellow`; the study focus remains prompt mechanism rather than color-to-color hypothesis testing.
