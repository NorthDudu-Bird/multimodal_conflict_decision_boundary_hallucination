You are helping write an empirical paper from a frozen project repository.

Research question:

> In a visually clear car-body primary-color task, can erroneous textual cues shift
> visual-language model judgments away from the image evidence?

Write the paper as one integrated experiment system, organized by experimental function:

1. Primary evaluation: balanced evaluation set, C0 baseline, C0-C4 main experiment,
   and same-image paired flips.
2. Auxiliary diagnostics: A1/A2.
3. Robustness and controlled diagnostics: C3 wording robustness, per-color split,
   answer-format control, and prompt factorization.
4. Validity checks: parser audit, source-stratified sanity check, visual clarity audit,
   and reproducibility audit.
5. Extension diagnostics: multi-turn persuasive setting and case-level failure taxonomy.

Strict boundaries:

1. Do not change the research direction.
2. Do not add models, tasks, or a benchmark-survey narrative.
3. Do not claim that VLMs generally have language bias or that text generally dominates vision.
4. Do not infer model-scale effects.
5. Keep A1/A2 as auxiliary diagnostics only.
6. Treat per-color, format-control, and prompt-factorization results as controlled
   diagnostics, not as a new mainline.
7. Treat multi-turn persuasion and case taxonomy as extension diagnostics, not as the
   central story.
8. Treat visual clarity audit as a validity check that reduces, but does not eliminate,
   image-difficulty concerns.
9. Treat StanfordCars/VCoR as a mixed-source car-image evaluation set. Do not turn
   source identity into a primary benchmark axis; keep it as an appendix sanity check
   and limitation.

Current defensible conclusion:

- All three models are visually faithful under `C0`.
- In the primary single-turn C0-C4 evaluation, LLaVA-1.5-7B shows limited but significant
  conflict-following in `C3` and secondarily `C4`.
- Qwen2-VL-7B-Instruct and InternVL2-8B remain largely visually consistent in the
  primary C0-C4 template family.
- Same-image paired analysis supports interpreting the LLaVA result as image-level
  answer flips from faithful C0 outputs.
- C3 wording robustness, per-color split, answer-format control, and prompt factorization
  show that the LLaVA effect is local and conditional rather than a uniform law.
- Factorized and multi-turn diagnostics show important boundary cases for Qwen and
  InternVL2, so do not claim global robustness.

Use these as primary writing anchors:

- `docs/final_writing_interface_note.md`
- `results/integrated_experiment_summary.md`
- `results/final_result_summary.md`
- `results/main/main_results_paper_ready.md`
- `results/main/paired_flip_summary.md`
- `results/robustness/prompt_boundary_summary.md`
- `results/auxiliary/aux_role_note.md`
- `results/threats_to_validity_summary.md`

For Nature-style follow-up work, use `results/final_result_summary.md` as a
writing-facing synthesis, not as a locked experimental artifact. The blocking
reproducibility gate is the canonical manifests, prompts, parsed outputs, metrics,
figures, parser/source audits, and statistical outputs.

Before drafting, produce:

1. A short statement of the final paper claim.
2. A Results outline using the five functional layers above.
3. A list of overclaims to avoid.
