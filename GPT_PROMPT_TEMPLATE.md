You are helping write an empirical paper from a frozen project repository.

Research question: whether visual-language models are affected by false text prompts in a clear single-attribute car-body primary color task.

Strict boundaries:

1. Do not change the research direction.
2. Do not add models, tasks, or a benchmark survey narrative.
3. Do not claim that VLMs generally have language bias or that text generally dominates vision.
4. Do not infer model-scale effects.
5. Keep A1/A2 as auxiliary diagnostics only.
6. Treat prompt wording robustness as a boundary-control result, not as evidence of cross-template stability.
7. Treat Phase 2 analyses as attribution, boundary, extension, or threat-reduction diagnostics. They strengthen the main claim but do not replace the C0-C4 mainline.

Current defensible conclusion:

- All three models are visually faithful under `C0`.
- Under the original single-turn C0-C4 templates, LLaVA-1.5-7B shows limited but significant conflict-aligned behavior, especially `C3` and secondarily `C4`.
- Qwen2-VL-7B-Instruct and InternVL2-8B remain largely visually consistent in the original C0-C4 template family.
- Same-image paired analysis supports interpreting the LLaVA result as image-level answer flips from faithful C0 outputs.
- C3 wording variants substantially weaken or remove the LLaVA effect, so the conclusion must be wording-sensitive and local.
- Phase 2 further narrows the interpretation: the LLaVA C3/C4 flips are color-pair-sensitive, concentrated especially in `white->black`; answer format and false-text framing matter; completed visual clarity audit reduces but does not eliminate image-level threats; multi-turn persuasion is an appendix-style extension, with a strong InternVL2-specific vulnerability in MT2/MT3.

Recommended first response after reading the pack:

1. Summarize the current reliable conclusion in 8-12 sentences.
2. List 3-5 overclaims to avoid.
3. Propose a Results structure centered on C0, C0-C4 paired flips, prompt-boundary control, and the most important Phase 2 attribution checks.
4. Explicitly separate mainline evidence, secondary attribution, appendix extension diagnostics, and threats-to-validity checks.

Use the following files as primary writing anchors:

- `results/final_result_summary.md`
- `docs/final_writing_interface_note.md`
- `results/main/main_results_paper_ready.md`
- `results/main/paired_flip_summary.md`
- `results/robustness/prompt_boundary_summary.md`
- `results/color_split/color_split_summary.md`
- `results/format_control/format_control_summary.md`
- `results/factorization/factorized_prompt_summary.md`
- `results/auxiliary/aux_role_note.md`
- `results/threats_to_validity_summary.md`

Do not overclaim:

- Do not write a general VLM language-bias paper.
- Do not write a prompt-engineering paper.
- Do not write a multi-turn persuasion paper.
- Do not describe the 9% LLaVA C3 effect as uniform across colors.
- Do not treat A1/A2 or Phase 2 factorized prompts as the primary causal evidence for the original C0-C4 result.
