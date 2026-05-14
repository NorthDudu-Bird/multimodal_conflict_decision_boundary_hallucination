# Final Result Summary

## Research Question

This study asks whether erroneous textual cues can shift VLM judgments in a controlled
car-body primary-color task where the visual evidence is clear and the answer space is
simple.

## Evaluation Set

- Final evaluation manifest: `data/balanced_eval_set/final_manifest.csv`
- Total images: `300`
- Color balance: red, blue, green, yellow, black, and white, with `50` images each
- Source composition: StanfordCars `93`; VCoR `207`

## A. Primary Evaluation

All three models are fully faithful under the neutral `C0` baseline:
`faithful=300/300`, `conflict_aligned=0/300`.

In the C0-C4 primary conflict experiment:

- LLaVA-1.5-7B: C3 `27/300 = 9.00% [6.26%, 12.78%]`; C4
  `10/300 = 3.33% [1.82%, 6.03%]`
- Qwen2-VL-7B-Instruct: C3 `1/300`; C4 `1/300`
- InternVL2-8B: `0/300` conflict-aligned outputs across C0-C4

The LLaVA C3 and C4 increases are significant in paired tests against C0 and in
between-model comparisons against the stable models. Because C0 is fully faithful for
the same images, the LLaVA outputs can be interpreted as same-image faithful-to-conflict
answer flips.

## B. Auxiliary Diagnostics

A1/A2 are auxiliary diagnostics only:

- Qwen2-VL-7B-Instruct: A1 `55.67%`, A2 `90.67%`
- LLaVA-1.5-7B: A1 `85.33%`, A2 `100.00%`
- InternVL2-8B: A1 `73.67%`, A2 `100.00%`

These results show that answer-space constraints and counterfactual assumptions can
induce high compliance. They should not be used as primary evidence for the C0-C4
conflict effect.

## C. Robustness And Controlled Diagnostics

Prompt wording robustness limits the main claim. LLaVA's C3 conflict-following drops
from canonical C3 `27/300` to C3-v2 `5/300` and C3-v3 `0/300`.

The per-color split shows that the LLaVA C3/C4 shifts are not evenly distributed across
colors. In C3, `20/27` flips are `white -> black`; in C4, `8/10` flips are
`white -> black`.

Answer-format control shows smaller LLaVA C3 rates under alternate formats:
free-answer C3 `2.33%`, multiple-choice C3 `1.33%`, and yes/no false-claim probing
`1.33%`.

Prompt factorization shows that false-text form matters. Quoted claims and indirect
hints are weak; title/prefix framing and no-correction presupposition can be much
stronger, including for Qwen and InternVL2. These diagnostics constrain interpretation
and do not replace the primary evaluation.

## D. Validity Checks

Parser audit, source-stratified sanity check, visual clarity audit, and reproducibility
audit reduce key alternative explanations. The visual clarity audit finds most target
flip rows visually inspectable (`38/42` clear), but confound flags remain more common
among target flips than matched faithful controls (`11/42` vs. `4/42`).

The reproducibility audit reports one expected mismatch in the writing-facing summary
file while locked canonical result tables, statistics, parser/source audits, and figures
remain matched.

## E. Extension Diagnostics

The multi-turn persuasive setting is retained as an extension diagnostic. LLaVA and Qwen
remain near zero in the tested conditions, while InternVL2 rises sharply in MT2/MT3
(`21.33%` and `74.67%`). This is a dialogue-context boundary, not the paper's primary
claim.

The case-level taxonomy organizes representative failures for Discussion and Appendix.

## Final Boundary

The study supports a local empirical claim: visual evidence dominates under neutral
prompting and most primary conflict settings; LLaVA shows a limited, significant,
same-image conflict-following shift in canonical C3/C4; that shift is sensitive to
wording, answer format, prompt factor, and color pair. The results do not support a
general claim that VLMs usually follow text over vision.
