# Appendix Sanity Check: Source-Stratified Comparison

This appendix-only sanity check splits the final balanced evaluation set by `source_dataset` (`StanfordCars` vs `VCoR`) and reports only `C0` and `C3`.

| model | condition | source_dataset | n | conflict_aligned | faithful |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct | C0 | StanfordCars | 93 | 0/93 (0.00% [0.00%, 3.97%]) | 93/93 (100.00% [96.03%, 100.00%]) |
| Qwen2-VL-7B-Instruct | C0 | VCoR | 207 | 0/207 (0.00% [0.00%, 1.82%]) | 207/207 (100.00% [98.18%, 100.00%]) |
| Qwen2-VL-7B-Instruct | C3 | StanfordCars | 93 | 0/93 (0.00% [0.00%, 3.97%]) | 93/93 (100.00% [96.03%, 100.00%]) |
| Qwen2-VL-7B-Instruct | C3 | VCoR | 207 | 1/207 (0.48% [0.09%, 2.69%]) | 206/207 (99.52% [97.31%, 99.91%]) |
| LLaVA-1.5-7B | C0 | StanfordCars | 93 | 0/93 (0.00% [0.00%, 3.97%]) | 93/93 (100.00% [96.03%, 100.00%]) |
| LLaVA-1.5-7B | C0 | VCoR | 207 | 0/207 (0.00% [0.00%, 1.82%]) | 207/207 (100.00% [98.18%, 100.00%]) |
| LLaVA-1.5-7B | C3 | StanfordCars | 93 | 13/93 (13.98% [8.35%, 22.46%]) | 80/93 (86.02% [77.54%, 91.65%]) |
| LLaVA-1.5-7B | C3 | VCoR | 207 | 14/207 (6.76% [4.07%, 11.03%]) | 193/207 (93.24% [88.97%, 95.93%]) |
| InternVL2-8B | C0 | StanfordCars | 93 | 0/93 (0.00% [0.00%, 3.97%]) | 93/93 (100.00% [96.03%, 100.00%]) |
| InternVL2-8B | C0 | VCoR | 207 | 0/207 (0.00% [0.00%, 1.82%]) | 207/207 (100.00% [98.18%, 100.00%]) |
| InternVL2-8B | C3 | StanfordCars | 93 | 0/93 (0.00% [0.00%, 3.97%]) | 93/93 (100.00% [96.03%, 100.00%]) |
| InternVL2-8B | C3 | VCoR | 207 | 0/207 (0.00% [0.00%, 1.82%]) | 207/207 (100.00% [98.18%, 100.00%]) |

## Interpretation
- `C0` remained perfectly faithful for all three models across both source groups, so there is no source-specific baseline collapse.
- For LLaVA-1.5-7B under `C3`, conflict-aligned behavior was higher on `StanfordCars` (13/93 (13.98% [8.35%, 22.46%])) than on `VCoR` (14/207 (6.76% [4.07%, 11.03%])), but the direction remained the same in both sources; Fisher exact p=0.0510.
- Qwen2-VL-7B-Instruct and InternVL2-8B remained visually faithful across both source groups, so the main conclusion is not driven by a single data source.
- This analysis is appendix-only: it checks whether the core trend is source-dependent, not whether source becomes a new main experimental factor.
