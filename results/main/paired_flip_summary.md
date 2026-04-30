# Same-Image Paired Flip Analysis

## Paired Structure Check

- Passed: for each model, `C0-C4` use the same 300 `image_id` values with one row per model-condition-image cell.
- Passed: current main outputs contain only `faithful` and `conflict_aligned` outcome states; `other_wrong`, `refusal`, and `parse_error` transitions are all zero.

## Derived Metrics

| model | condition | n_pairs | faithful_to_faithful_n | faithful_to_conflict_aligned_n | answer_flip_rate | faithful_retention_rate | net_conflict_shift | p_value_exact_mcnemar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | C1 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |
| Qwen2-VL-7B-Instruct | C2 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |
| Qwen2-VL-7B-Instruct | C3 | 300 | 299 | 1 | 0.33% | 99.67% | 0.33% | 1 |
| Qwen2-VL-7B-Instruct | C4 | 300 | 299 | 1 | 0.33% | 99.67% | 0.33% | 1 |
| LLaVA-1.5-7B | C1 | 300 | 299 | 1 | 0.33% | 99.67% | 0.33% | 1 |
| LLaVA-1.5-7B | C2 | 300 | 297 | 3 | 1.00% | 99.00% | 1.00% | 0.25 |
| LLaVA-1.5-7B | C3 | 300 | 273 | 27 | 9.00% | 91.00% | 9.00% | 1.49e-08 |
| LLaVA-1.5-7B | C4 | 300 | 290 | 10 | 3.33% | 96.67% | 3.33% | 0.001953 |
| InternVL2-8B | C1 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |
| InternVL2-8B | C2 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |
| InternVL2-8B | C3 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |
| InternVL2-8B | C4 | 300 | 300 | 0 | 0.00% | 100.00% | 0.00% | 1 |

## Exact Paired Tests

- LLaVA-1.5-7B: C0 vs C3: current-only=27, C0-only=0, raw p=1.49e-08, Holm p=1.788e-07.
- LLaVA-1.5-7B: C0 vs C4: current-only=10, C0-only=0, raw p=0.001953, Holm p=0.02148.

## Paper-Ready Interpretation

The main experiment is a same-image paired evaluation: every conflict prompt is compared against the same model's answer to the same image under `C0`. Because all three models are fully faithful under `C0`, conflict-aligned outputs under later conditions can be read as image-level answer flips from a visually faithful baseline rather than as differences between unrelated image pools.
For LLaVA-1.5-7B, `C3` produced 27/300 faithful-to-conflict flips and `C4` produced 10/300. Qwen2-VL-7B-Instruct produced only one such flip in each of `C3` and `C4`, while InternVL2-8B produced none.
This paired framing is stronger than reporting condition-level rates alone because it ties each changed answer to the exact same visual evidence. It supports a narrow claim of conditional language-induced shifts in one model/template setting, not a broad claim that text generally dominates vision.
