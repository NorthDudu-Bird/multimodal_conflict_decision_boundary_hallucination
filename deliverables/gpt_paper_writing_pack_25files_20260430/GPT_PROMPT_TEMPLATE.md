You are helping write an empirical paper from a frozen project repository.

Research question: whether visual-language models are affected by false text prompts in a clear single-attribute car-body primary color task.

Strict boundaries:

1. Do not change the research direction.
2. Do not add models, tasks, or a benchmark survey narrative.
3. Do not claim that VLMs generally have language bias or that text generally dominates vision.
4. Do not infer model-scale effects.
5. Keep A1/A2 as auxiliary diagnostics only.
6. Treat prompt wording robustness as a boundary-control result, not as evidence of cross-template stability.

Current defensible conclusion:

- All three models are visually faithful under `C0`.
- LLaVA-1.5-7B shows limited but significant conflict-aligned behavior under the original strong misleading open templates, especially `C3` and secondarily `C4`.
- Qwen2-VL-7B-Instruct and InternVL2-8B remain largely visually consistent in C0-C4.
- Same-image paired analysis supports interpreting the LLaVA result as image-level answer flips from faithful C0 outputs.
- C3 wording variants substantially weaken or remove the LLaVA effect, so the conclusion must be wording-sensitive and local.

Recommended first response after reading the pack:

1. Summarize the current reliable conclusion in 8-12 sentences.
2. List 3-5 overclaims to avoid.
3. Propose a Results structure centered on C0, C0-C4 paired flips, prompt-boundary control, and auxiliary diagnostics.

Use the following files as primary writing anchors:

- `results/final_result_summary.md`
- `results/main/main_results_paper_ready.md`
- `results/main/paired_flip_summary.md`
- `results/robustness/prompt_boundary_summary.md`
- `results/auxiliary/aux_role_note.md`
- `results/threats_to_validity_summary.md`
