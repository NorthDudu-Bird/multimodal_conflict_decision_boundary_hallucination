# Prompt Variant Robustness Summary

## Metrics
- Qwen2-VL-7B-Instruct | Original C3: conflict_aligned=0.33% [0.06%, 1.86%]; faithful=99.67% [98.14%, 99.94%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- Qwen2-VL-7B-Instruct | C3-v2: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- Qwen2-VL-7B-Instruct | C3-v3: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- LLaVA-1.5-7B | Original C3: conflict_aligned=9.00% [6.26%, 12.78%]; faithful=91.00% [87.22%, 93.74%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- LLaVA-1.5-7B | C3-v2: conflict_aligned=1.67% [0.71%, 3.84%]; faithful=98.00% [95.71%, 99.08%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.33% [0.06%, 1.86%]; n=300
- LLaVA-1.5-7B | C3-v3: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=99.67% [98.14%, 99.94%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.33% [0.06%, 1.86%]; n=300
- InternVL2-8B | Original C3: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- InternVL2-8B | C3-v2: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300
- InternVL2-8B | C3-v3: conflict_aligned=0.00% [0.00%, 1.26%]; faithful=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; other_wrong=0.00% [0.00%, 1.26%]; n=300

## Exact Tests

| family | comparison | left rate | right rate | diff | left-only | right-only | raw p | Holm p | Holm sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within_model_vs_original | Qwen2-VL-7B-Instruct: Original C3 vs C3-v2 | 0.00% | 0.33% | -0.33 pp | 0 | 1 | 1.0000 | 1.0000 | no |
| within_model_vs_original | Qwen2-VL-7B-Instruct: Original C3 vs C3-v3 | 0.00% | 0.33% | -0.33 pp | 0 | 1 | 1.0000 | 1.0000 | no |
| within_model_vs_original | LLaVA-1.5-7B: Original C3 vs C3-v2 | 1.67% | 9.00% | -7.33 pp | 0 | 22 | 4.77e-07 | 2.38e-06 | yes |
| within_model_vs_original | LLaVA-1.5-7B: Original C3 vs C3-v3 | 0.00% | 9.00% | -9.00 pp | 0 | 27 | 1.49e-08 | 8.94e-08 | yes |
| within_model_vs_original | InternVL2-8B: Original C3 vs C3-v2 | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |
| within_model_vs_original | InternVL2-8B: Original C3 vs C3-v3 | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |
| cross_model_same_variant | Original C3: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct | 9.00% | 0.33% | 8.67 pp | 27 | 1 | 2.16e-07 | 1.08e-06 | yes |
| cross_model_same_variant | Original C3: LLaVA-1.5-7B vs InternVL2-8B | 9.00% | 0.00% | 9.00 pp | 27 | 0 | 1.49e-08 | 8.94e-08 | yes |
| cross_model_same_variant | C3-v2: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct | 1.67% | 0.00% | 1.67 pp | 5 | 0 | 0.0625 | 0.2500 | no |
| cross_model_same_variant | C3-v2: LLaVA-1.5-7B vs InternVL2-8B | 1.67% | 0.00% | 1.67 pp | 5 | 0 | 0.0625 | 0.2500 | no |
| cross_model_same_variant | C3-v3: LLaVA-1.5-7B vs Qwen2-VL-7B-Instruct | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |
| cross_model_same_variant | C3-v3: LLaVA-1.5-7B vs InternVL2-8B | 0.00% | 0.00% | 0.00 pp | 0 | 0 | 1.0000 | 1.0000 | no |

## LLaVA Variant vs C0 Check

- Original C3 vs C0: diff=9.00 pp, raw p=1.49e-08, Holm p=4.47e-08, Holm significant=yes.
- C3-v2 vs C0: diff=1.67 pp, raw p=0.0625, Holm p=0.1250, Holm significant=no.
- C3-v3 vs C0: diff=0.00 pp, raw p=1.0000, Holm p=1.0000, Holm significant=no.

## Conclusion

- 结论：当前现象对原模板敏感，不应写成稳定规律。
- 解释：至少一个新 wording 下，LLaVA 不再稳定高于两个视觉稳定模型，因此结论需要降级为模板依赖现象。
