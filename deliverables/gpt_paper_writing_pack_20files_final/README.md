# Multimodal Conflict Decision Boundary Hallucination

This repository supports an empirical behavior-analysis paper on one controlled question:

> In a visually clear car-body primary-color task, can erroneous textual cues shift
> visual-language model color judgments away from the image evidence?

The study is intentionally local. It is not a method paper, not a broad benchmark
leaderboard, not a model-scale analysis, not a general color-perception paper, and not
a general claim that language dominates vision in VLMs.

## Unified Experiment System

The paper-facing materials are organized by experimental function.

| Layer | Modules | Role in the paper |
| --- | --- | --- |
| A. Primary evaluation | balanced evaluation set; C0 baseline; C0-C4 main experiment; same-image paired flips | Main evidence chain |
| B. Auxiliary diagnostics | A1/A2 | Answer-space and counterfactual-assumption stress tests |
| C. Robustness and controlled diagnostics | C3 wording robustness; per-color split; answer-format control; prompt factorization | Attribution and boundary controls |
| D. Validity checks | parser audit; source-stratified sanity check; visual clarity audit; reproducibility audit | Threat reduction |
| E. Extension diagnostics | multi-turn persuasive setting; case-level failure taxonomy | Appendix-style extensions and case interpretation |

## Defensible Conclusion

- All three models are visually faithful under `C0` on the 300-image balanced set.
- In the primary single-turn C0-C4 evaluation, LLaVA-1.5-7B shows a limited but
  statistically significant same-image conflict-following shift in `C3` and `C4`.
- Qwen2-VL-7B-Instruct and InternVL2-8B are essentially stable in the primary C0-C4
  template family.
- Same-image paired flips ground the LLaVA shift in identical visual evidence rather
  than independent image pools.
- C3 wording robustness, per-color split, answer-format control, and prompt
  factorization show that the effect is wording-, format-, factor-, and color-pair
  sensitive.
- The LLaVA `C3` effect is not uniform across colors: `20/27` C3 flips are
  `white -> black`; C4 similarly concentrates in `white -> black` (`8/10`).
- Visual clarity audit reduces, but does not eliminate, the alternative explanation
  that flip rows are visually difficult images.
- Multi-turn persuasion is retained only as an extension diagnostic, not as the paper's
  main story.

## Source Handling

The final evaluation set is treated as one mixed-source car-image set rather than as
two source-specific benchmarks. StanfordCars and VCoR both provide real vehicle images
with a clearly visible principal car and inspectable car-body primary color after
cropping. Source identity is retained for appendix sanity checks and limitations, but
it is not a primary experimental factor in the manuscript argument.

## Key Outputs

- Final writing interface: `docs/final_writing_interface_note.md`
- Final writing pack note: `docs/final_writing_pack_note.md`
- Integrated experiment summary: `results/integrated_experiment_summary.md`
- Final result summary: `results/final_result_summary.md`
- Main paper-ready result note: `results/main/main_results_paper_ready.md`
- Main metrics: `results/main/table1_main_metrics.csv`
- Main key tests: `results/main/main_key_tests.csv`
- Paired flips: `results/main/paired_flip_summary.md`
- Auxiliary diagnostics: `results/auxiliary/aux_role_note.md`
- Wording robustness: `results/robustness/prompt_boundary_summary.md`
- Per-color split: `results/color_split/color_split_summary.md`
- Answer-format control: `results/format_control/format_control_summary.md`
- Prompt factorization: `results/factorization/factorized_prompt_summary.md`
- Visual clarity audit: `results/audit/visual_clarity_audit_summary.md`
- Gatekeeping protocol: `results/gatekeeping/gatekeeping_summary.md`
- Validity and scope audit: `results/validity_and_scope_audit_summary.md`
- Final 20-file writing pack: `deliverables/gpt_paper_writing_pack_20files_final.zip`

## Reproduction Entry Points

Canonical evaluation and summaries:

```bash
python scripts/analyze_results.py
python scripts/generate_parser_audit.py
python scripts/make_figures.py
python scripts/verify_reproducibility.py
```

`results/final_result_summary.md` is a writing-facing integrated summary and may differ
from the locked snapshot without blocking the reproducibility gate. The gate remains
blocking for experimental manifests, prompts, parsed outputs, result tables, figures,
parser/source audits, and statistical outputs.

Derived and controlled diagnostics:

```bash
python scripts/generate_paired_flip_analysis.py
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_visual_clarity_audit.py
python scripts/generate_color_split_analysis.py
python scripts/generate_visual_clarity_completed_audit.py
python scripts/run_controlled_diagnostics.py --family factorization
python scripts/run_controlled_diagnostics.py --family format_control
python scripts/run_controlled_diagnostics.py --family multiturn
python scripts/generate_integrated_synthesis.py
python scripts/build_final_writing_pack.py
```

## Writing Boundaries

- Do not write this as a general VLM language-bias paper.
- Do not write this as a prompt-engineering paper.
- Do not write this as a multi-turn persuasion paper.
- Do not use A1/A2 as primary evidence for the C0-C4 conflict effect.
- Do not describe the LLaVA `C3` 9.00% result as a uniform color effect.
- Do not claim that visual ambiguity has been completely eliminated.
- Do not claim that Qwen or InternVL2 are globally robust to all misleading-text forms.
