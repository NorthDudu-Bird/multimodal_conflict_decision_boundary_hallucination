# Main Results Paper-Ready Summary

## Baseline Visual Faithfulness

The main results should start from the neutral `C0` condition. All three models answer the primary body color faithfully for all 300 images under `C0`, with `conflict_aligned=0/300` and no refusals, parse errors, or other-wrong outputs. This establishes that the task is not dominated by baseline color-recognition failure under the frozen six-color setting.

## Conflict Conditions

Under the original C0-C4 prompt templates, Qwen2-VL-7B-Instruct and InternVL2-8B remain visually consistent. Qwen2-VL-7B-Instruct has only one conflict-aligned output in `C3` and one in `C4`; InternVL2-8B has none across C0-C4.

LLaVA-1.5-7B shows the only clear conditional shift. It produces `27/300` conflict-aligned outputs in `C3` and `10/300` in `C4`. These increases are significant in paired exact tests against the same model's `C0` baseline, and LLaVA is significantly higher than the stable comparison models under the original `C3` and `C4` conditions.

## Paired Interpretation

The strengthened paired analysis makes the inference more direct. Every C1-C4 answer is compared to the same model's answer on the same image under C0. Because C0 is fully faithful, the LLaVA `C3` result can be written as `27/300` image-level flips from faithful C0 answers to the false prompt color, and `C4` as `10/300` such flips.

This is stronger than reporting condition-level percentages alone: it rules out image-pool composition as the explanation for the main change and ties the observed shift to the false-text condition applied to the same visual evidence.

## Boundary From Wording Robustness

The prompt-boundary module limits the claim. Under semantically related C3 rewrites, LLaVA-1.5-7B drops from Original C3 `27/300` to C3-v2 `5/300` and C3-v3 `0/300`. The new wording variants no longer produce significant LLaVA-vs-stable-model differences after Holm correction.

Therefore the paper should not state that LLaVA has a stable strong-conflict behavior across templates. The safer conclusion is that the original strong misleading open template induces a limited but significant conflict-aligned shift in LLaVA-1.5-7B, and that this shift is wording-sensitive.

## Claims To Retain

- The three models are visually faithful under neutral prompting on this controlled task.
- LLaVA-1.5-7B shows limited significant conflict-aligned behavior under the original `C3` and `C4` templates.
- Qwen2-VL-7B-Instruct and InternVL2-8B are stable visual-consistency controls in the same evaluation.
- Same-image paired flips support a conditional-shift interpretation for LLaVA under the original templates.

## Claims To Avoid

- Do not claim that VLMs generally prioritize language over vision.
- Do not claim a cross-task or cross-attribute law.
- Do not infer model-size effects from these three checkpoints.
- Do not treat C3 wording variants as evidence of robustness; they are evidence that the main claim must be bounded.
- Do not use A1/A2 as primary evidence for the C0-C4 causal chain.

## Phase 2 Writing Placement

Use the Phase 2 results to strengthen, not replace, the main C0-C4 story. The main-text evidence chain should remain: balanced 300-image dataset, C0 visual fidelity, C0-C4 conflict conditions, same-image paired flips, and C3 wording boundary.

Recommended main-text secondary results are: per-color split showing `white->black` concentration, answer-format control showing reduced conflict following under free/multiple-choice/yes-no formats, and a compact prompt-factorization paragraph showing that title/prefix framing and no-correction presupposition are the strongest false-text forms.

Recommended appendix results are: completed visual clarity audit, full factorization tables, multi-turn persuasion, failure taxonomy/casebook, gatekeeping protocol, parser/source/reproducibility details.

Updated main conclusion: in a visually clear single-attribute car-color task, all three models are faithful under neutral prompting; LLaVA shows a limited, significant, same-image conflict-following shift under the original strong misleading templates; that shift is model-, wording-, answer-format-, and color-pair-sensitive; multi-turn context can induce a separate model-specific vulnerability, especially for InternVL2, but it should not be folded into the single-turn mainline.
