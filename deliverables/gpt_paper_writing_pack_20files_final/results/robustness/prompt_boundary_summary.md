# Prompt Wording Boundary-Control Summary

## Role

- This module limits the claim boundary. It is not used to expand the paper into a cross-template law.
- The relevant question is whether the original `C3` effect survives semantically close wording changes.

## Metrics

- Qwen2-VL-7B-Instruct | Original C3: conflict_aligned=1/300 (0.33% [0.06%, 1.86%]); faithful=299/300 (99.67%).
- Qwen2-VL-7B-Instruct | C3-v2: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=300/300 (100.00%).
- Qwen2-VL-7B-Instruct | C3-v3: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=300/300 (100.00%).
- LLaVA-1.5-7B | Original C3: conflict_aligned=27/300 (9.00% [6.26%, 12.78%]); faithful=273/300 (91.00%).
- LLaVA-1.5-7B | C3-v2: conflict_aligned=5/300 (1.67% [0.71%, 3.84%]); faithful=294/300 (98.00%).
- LLaVA-1.5-7B | C3-v3: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=299/300 (99.67%).
- InternVL2-8B | Original C3: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=300/300 (100.00%).
- InternVL2-8B | C3-v2: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=300/300 (100.00%).
- InternVL2-8B | C3-v3: conflict_aligned=0/300 (0.00% [0.00%, 1.26%]); faithful=300/300 (100.00%).

## Key Tests

- LLaVA-1.5-7B: C3-v2 vs Original C3: diff=-7.33 pp, left-only=0, right-only=22, raw p=4.77e-07, Holm p=2.38e-06, Holm significant=yes.
- LLaVA-1.5-7B: C3-v3 vs Original C3: diff=-9.00 pp, left-only=0, right-only=27, raw p=1.49e-08, Holm p=8.94e-08, Holm significant=yes.
- Original C3: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct: diff=8.67 pp, left-only=27, right-only=1, raw p=2.16e-07, Holm p=1.08e-06, Holm significant=yes.
- Original C3: LLaVA-1.5-7B vs InternVL2-8B: diff=9.00 pp, left-only=27, right-only=0, raw p=1.49e-08, Holm p=8.94e-08, Holm significant=yes.
- C3-v2: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct: diff=1.67 pp, left-only=5, right-only=0, raw p=0.0625, Holm p=0.2500, Holm significant=no.
- C3-v2: LLaVA-1.5-7B vs InternVL2-8B: diff=1.67 pp, left-only=5, right-only=0, raw p=0.0625, Holm p=0.2500, Holm significant=no.
- C3-v3: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct: diff=0.00 pp, left-only=0, right-only=0, raw p=1.0000, Holm p=1.0000, Holm significant=no.
- C3-v3: LLaVA-1.5-7B vs InternVL2-8B: diff=0.00 pp, left-only=0, right-only=0, raw p=1.0000, Holm p=1.0000, Holm significant=no.
- LLaVA-1.5-7B: Original C3 vs C0: diff=9.00 pp, left-only=27, right-only=0, raw p=1.49e-08, Holm p=4.47e-08, Holm significant=yes.
- LLaVA-1.5-7B: C3-v2 vs C0: diff=1.67 pp, left-only=5, right-only=0, raw p=0.0625, Holm p=0.1250, Holm significant=no.
- LLaVA-1.5-7B: C3-v3 vs C0: diff=0.00 pp, left-only=0, right-only=0, raw p=1.0000, Holm p=1.0000, Holm significant=no.

## Interpretation

Original `C3` remains the strongest observation: LLaVA-1.5-7B shows 27/300 conflict-aligned outputs (9.00%).
Under revised wording, the same model drops to 5/300 in `C3-v2` and 0/300 in `C3-v3`. The within-model decreases from Original C3 to both variants are Holm-significant.
The new wording variants no longer show Holm-significant LLaVA-vs-stable-model differences. Therefore, the paper should state that the observed language-aligned behavior is template-sensitive, not stable across equivalent wording variants.

## Paper Paragraph

The prompt wording control provides an explicit upper bound on the main claim. Although LLaVA-1.5-7B shows a significant conflict-aligned shift under the original `C3` wording, semantically similar rewrites substantially weaken or remove the effect. This pattern supports a conservative interpretation: the observed shift is a limited, wording-sensitive behavior in one model under the original strong misleading template, rather than evidence for a stable cross-template rule.
