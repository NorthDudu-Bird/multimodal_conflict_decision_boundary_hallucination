# Main Experiment Statistical Summary

- Primary inference target: `conflict_aligned` only.
- Wilson 95% confidence intervals are reported for all main-condition proportions.
- Holm correction was applied separately within the six `within_model_vs_C0` tests and the six `cross_model_same_condition` tests.
- Main-experiment `refusal`, `other_wrong`, and `parse_error` counts were all zero across every model-condition cell.

## Condition Labels
- `C0`: neutral prompt.
- `C1`: weak suggestion.
- `C2`: false assertion, open answer.
- `C3`: presupposition with correction allowed.
- `C4`: stronger open conflict framing.

## Key Tests

| family | comparison | left rate | right rate | diff | left-only | right-only | raw p | Holm p | Holm sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within-model vs C0 | Qwen2-VL-7B-Instruct: C0 vs C3 | 0.33% | 0.00% | 0.33 pp | 1 | 0 | 1.0000 | 1.0000 | no |
| within-model vs C0 | Qwen2-VL-7B-Instruct: C0 vs C4 | 0.33% | 0.00% | 0.33 pp | 1 | 0 | 1.0000 | 1.0000 | no |
| within-model vs C0 | LLaVA-1.5-7B: C0 vs C3 | 9.00% | 0.00% | 9.00 pp | 27 | 0 | 1.49e-08 | 8.94e-08 | yes |
| within-model vs C0 | LLaVA-1.5-7B: C0 vs C4 | 3.33% | 0.00% | 3.33 pp | 10 | 0 | 0.0020 | 0.0098 | yes |
| within-model vs C0 | InternVL2-8B: C0 vs C3 | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |
| within-model vs C0 | InternVL2-8B: C0 vs C4 | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |
| cross-model same condition | C3: Qwen2-VL-7B-Instruct vs LLaVA-1.5-7B | 0.33% | 9.00% | -8.67 pp | 1 | 27 | 2.16e-07 | 1.08e-06 | yes |
| cross-model same condition | C3: Qwen2-VL-7B-Instruct vs InternVL2-8B | 0.33% | 0.00% | 0.33 pp | 1 | 0 | 1.0000 | 1.0000 | no |
| cross-model same condition | C3: LLaVA-1.5-7B vs InternVL2-8B | 9.00% | 0.00% | 9.00 pp | 27 | 0 | 1.49e-08 | 8.94e-08 | yes |
| cross-model same condition | C4: Qwen2-VL-7B-Instruct vs LLaVA-1.5-7B | 0.33% | 3.33% | -3.00 pp | 1 | 10 | 0.0117 | 0.0352 | yes |
| cross-model same condition | C4: Qwen2-VL-7B-Instruct vs InternVL2-8B | 0.33% | 0.00% | 0.33 pp | 1 | 0 | 1.0000 | 1.0000 | no |
| cross-model same condition | C4: LLaVA-1.5-7B vs InternVL2-8B | 3.33% | 0.00% | 3.33 pp | 10 | 0 | 0.0020 | 0.0078 | yes |

## Interpretation
- LLaVA-1.5-7B showed a significant increase in conflict-aligned rate from C0 to C3 (raw p=1.49e-08, Holm p=8.94e-08).
- LLaVA-1.5-7B showed a significant increase in conflict-aligned rate from C0 to C4 (raw p=0.0020, Holm p=0.0098).
- Under C3, LLaVA-1.5-7B had a significantly higher conflict-aligned rate than Qwen2-VL-7B-Instruct (raw p=2.16e-07, Holm p=1.08e-06).
- Under C3, LLaVA-1.5-7B had a significantly higher conflict-aligned rate than InternVL2-8B (raw p=1.49e-08, Holm p=8.94e-08).
- Under C4, LLaVA-1.5-7B had a significantly higher conflict-aligned rate than Qwen2-VL-7B-Instruct (raw p=0.0117, Holm p=0.0352).
- Under C4, LLaVA-1.5-7B had a significantly higher conflict-aligned rate than InternVL2-8B (raw p=0.0020, Holm p=0.0078).
