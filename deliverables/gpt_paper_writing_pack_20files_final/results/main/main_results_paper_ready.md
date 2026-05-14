# Main Results Paper-Ready Summary

## A. Primary Evaluation

The main Results section should begin with the neutral C0 condition. All three models
answer the primary body color faithfully for all 300 images under C0, with
`conflict_aligned=0/300` and no refusals, parse errors, or other-wrong outputs. This
establishes that the task is not dominated by baseline color-recognition failure.

Under the C0-C4 conflict conditions, Qwen2-VL-7B-Instruct and InternVL2-8B remain
visually consistent in the primary template family. Qwen has only one conflict-following
output in C3 and one in C4; InternVL2 has none across C0-C4.

LLaVA-1.5-7B shows the only clear primary shift. It produces `27/300` conflict-following
outputs in C3 and `10/300` in C4. These increases are significant in paired exact tests
against the same model's C0 baseline, and LLaVA is significantly higher than the stable
comparison models under C3 and C4.

## Same-Image Paired Interpretation

Every C1-C4 answer is compared to the same model's answer on the same image under C0.
Because C0 is fully faithful, the LLaVA C3 result can be written as `27/300` image-level
flips from faithful C0 answers to the false prompt color, and C4 as `10/300` such flips.

This is stronger than reporting condition-level percentages alone because it ties the
observed shift to changed prompting over identical visual evidence.

## C. Robustness And Controlled Diagnostics

C3 wording robustness limits the claim. Under semantically related C3 rewrites,
LLaVA-1.5-7B drops from canonical C3 `27/300` to C3-v2 `5/300` and C3-v3 `0/300`.
The wording variants no longer produce significant LLaVA-vs-stable-model differences
after Holm correction.

Per-color split should be reported as a secondary diagnostic: LLaVA C3 flips are
concentrated in `white -> black` (`20/27`), and C4 flips are also concentrated in
`white -> black` (`8/10`). This means the 9.00% C3 result is not a uniform color effect.

Answer-format control and prompt factorization can be summarized briefly in the main
text, with details in appendix. They show that the observed shift depends on response
format and false-text form.

## What To Retain

- The three models are visually faithful under neutral prompting on this controlled task.
- LLaVA-1.5-7B shows limited significant conflict-following under canonical C3 and C4.
- Qwen2-VL-7B-Instruct and InternVL2-8B are stable in the primary C0-C4 template family.
- Same-image paired flips support a conditional-shift interpretation for LLaVA.
- Controlled diagnostics show the effect is local and conditional.

## What To Avoid

- Do not claim that VLMs generally prioritize language over vision.
- Do not claim a cross-task or cross-attribute law.
- Do not infer model-size effects from these three checkpoints.
- Do not treat C3 wording variants as evidence of robustness; they bound the claim.
- Do not use A1/A2 as primary evidence for the C0-C4 causal chain.
- Do not promote multi-turn results into the primary single-turn evaluation.
