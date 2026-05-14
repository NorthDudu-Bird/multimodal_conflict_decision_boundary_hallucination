# Integrated Experiment Summary

This summary organizes the full study by experimental function. It is intended as the
compact writing-facing overview for the final paper.

## A. Primary Evaluation

The primary evaluation uses a 300-image, six-color balanced car-body primary-color set
with 50 images each for red, blue, green, yellow, black, and white. The model set is
fixed to LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, and InternVL2-8B.

Under the neutral C0 baseline, all three models are visually faithful on all 300 images.
This establishes that the task is not dominated by baseline color-recognition failure.

In the C0-C4 primary conflict experiment, LLaVA-1.5-7B is the only model with a clear
conflict-following shift: `27/300` in C3 and `10/300` in C4. Qwen2-VL-7B-Instruct has
`1/300` in C3 and `1/300` in C4. InternVL2-8B has `0/300` conflict-following outputs
across C0-C4.

Because the same 300 images are used across C0-C4 and C0 is fully faithful, the LLaVA
C3/C4 findings can be written as same-image faithful-to-conflict paired flips rather
than independent aggregate differences.

## B. Auxiliary Diagnostics

A1/A2 are auxiliary diagnostics, not primary evidence. They test whether restricted
answer spaces and counterfactual assumptions can induce compliance-like behavior.

Compliance rates are high in A2 and nontrivial in A1:

- Qwen2-VL-7B-Instruct: A1 `55.67%`, A2 `90.67%`
- LLaVA-1.5-7B: A1 `85.33%`, A2 `100.00%`
- InternVL2-8B: A1 `73.67%`, A2 `100.00%`

These results belong in an auxiliary or appendix section. They should not be folded into
the causal interpretation of the primary C0-C4 conflict effect.

## C. Robustness And Controlled Diagnostics

### C3 Wording Robustness

LLaVA's primary C3 result is wording-sensitive. The C3 conflict-following count drops
from `27/300` in the canonical C3 wording to `5/300` in C3-v2 and `0/300` in C3-v3.
This prevents the paper from claiming a stable cross-wording law.

### Per-Color Split

The LLaVA C3 effect is not uniform across colors. Among the 27 C3 flips, `20` are
`white -> black`, `3` are `black -> white`, and `4` are `blue -> red`. C4 is similarly
concentrated: `8/10` flips are `white -> black`, and `2/10` are `black -> white`.

This supports a narrower conclusion: the observed shift is a conditional prompt effect
with strong color-pair concentration, especially in achromatic white/black routes.

### Answer-Format Control

The canonical LLaVA C3 format remains the largest single-turn format result in this
diagnostic set. Free C3 is `7/300` (`2.33%`), multiple-choice C3 is `4/300` (`1.33%`),
and yes/no false-claim probing is `4/300` (`1.33%`). This supports a cautious statement
that answer format and response framing can amplify or reduce conflict-following.

### Prompt Factorization

Prompt factorization shows that false-text form matters. Quoted claims and indirect
hints are near zero across models. Stronger framing is more consequential:

- LLaVA: title/prefix framing `32.00%`; no-correction presupposition `16.33%`
- Qwen: no-correction presupposition `34.00%`
- InternVL2: title/prefix framing `36.00%`

These results should be used as controlled diagnostics for boundary and attribution,
not as a new prompt-engineering mainline.

## D. Validity Checks

Parser audit supports that the primary results are not an artifact of complex label
mapping. Source-stratified sanity checks reduce the concern that the finding is driven
by a single source subset. Reproducibility audit shows that locked canonical tables and
figures remain stable, with the only mismatch coming from writing-facing summary text.

The visual clarity audit covers 42 target flip rows and 42 matched faithful controls.
Target rows are mostly clear (`38/42`), similar to controls (`39/42`), which reduces the
claim that the flip set is simply unreadable. However, visual confound flags are more
common in target rows (`11/42`) than controls (`4/42`), so lighting, reflections,
background color, and multi-car interference remain limitations.

## E. Extension Diagnostics

The multi-turn persuasive setting is an extension diagnostic. LLaVA and Qwen remain
near zero in the tested multi-turn conditions, while InternVL2 rises to `21.33%` in MT2
and `74.67%` in MT3. This result should be reported as a separate dialogue-context
boundary, not as a replacement for the single-turn primary evaluation.

The case-level failure taxonomy helps organize the discussion into prompt-following
flips, color-pair concentration, visual-clarity flagged cases, format/factor-induced
shifts, multi-turn-induced cases, and source/style-sensitive cases. It is useful for
Discussion and Appendix, not as a new theory.

## Final Paper Boundary

The full study supports a narrow empirical claim: visual evidence dominates under
neutral prompting and most primary conflict conditions; LLaVA shows a limited but
significant same-image conflict-following shift under canonical C3/C4; that shift is
wording-, format-, factor-, and color-pair-sensitive; and extension diagnostics identify
additional boundary cases without changing the primary evaluation story.
