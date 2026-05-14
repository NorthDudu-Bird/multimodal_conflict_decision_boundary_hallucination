# Final Writing Interface Note

This is the final paper-writing interface. It presents the repository as one integrated
empirical study organized by experimental function, not by development history.

## Standard Paper Positioning

The paper asks whether erroneous textual cues can shift VLM judgments in a controlled
car-body primary-color task where visual evidence is clear and the answer space is
simple. The contribution is empirical characterization and evaluation-style behavior
analysis.

Do not frame the paper as:

- a new method;
- a broad VLM benchmark;
- a general language-bias claim;
- a prompt-engineering paper;
- a multi-turn persuasion paper;
- a general color-perception paper.

## Final Claim Boundary

The safest paper claim is:

> On a 300-image six-color balanced car-body color evaluation set, all three models are
> faithful under neutral prompting. In the primary single-turn C0-C4 evaluation,
> LLaVA-1.5-7B shows a limited but statistically significant same-image
> conflict-following shift in C3/C4, while Qwen2-VL-7B-Instruct and InternVL2-8B are
> essentially stable in that primary template family. Controlled diagnostics show that
> the effect is conditional on wording, answer format, prompt factor, and color pair,
> especially the `white -> black` route. Validity checks reduce parser, source,
> reproducibility, and visual-clarity threats, but they do not justify a broad
> text-over-vision claim.

## Dataset Source Framing

Write the 300-image evaluation set as a mixed-source car-image set. Do not make
StanfordCars versus VCoR a main comparison or a separate benchmark axis. Both sources
are used as real vehicle-image inputs after cropping, and source identity is retained
only for appendix sanity checks and limitation language. The source-stratified check
reduces the concern that the main pattern is confined to one source, but it does not
turn source into a primary experimental factor.

## Final Paper Structure Mapping

| Paper layer | Modules | Main files | How to write it |
| --- | --- | --- | --- |
| A. Primary evaluation | balanced evaluation set; C0 baseline; C0-C4; same-image paired flips | `balanced_eval_set_summary.json`; `table1_main_metrics.csv`; `main_key_tests.csv`; `paired_flip_summary.md` | Core Results. This is the primary evidence chain. |
| B. Auxiliary diagnostics | A1/A2 | `table3_aux_metrics.csv`; `aux_role_note.md` | Short auxiliary subsection or appendix. Stress tests only. |
| C. Robustness and controlled diagnostics | C3 wording robustness; per-color split; answer-format control; prompt factorization | `prompt_boundary_summary.md`; `color_split_summary.md`; `format_control_summary.md`; `factorized_prompt_summary.md` | Main-text secondary diagnostics, with full details in appendix. |
| D. Validity checks | parser audit; source sanity; visual clarity audit; reproducibility audit | `label_mapping_audit.md`; `stanford_core_sanity_check.md`; `visual_clarity_audit_summary.md`; `reproducibility_audit.md` | Methods, limitations, and appendix threat reduction. |
| E. Extension diagnostics | multi-turn persuasive setting; case-level failure taxonomy | `multiturn_summary.md`; `casebook.md`; `failure_taxonomy_definition.md` | Appendix or brief discussion extension. |

## Terminology

| Term | Use this meaning | Avoid |
| --- | --- | --- |
| faithful | Parsed answer matches the true car-body primary color. | Treating it as broad semantic correctness. |
| conflict_aligned / conflict_following | Parsed answer follows the false prompt color in a conflict condition. | Calling every case hallucination. |
| answer flip / paired flip | Same-image transition from faithful C0 to conflict-following under a conflict prompt. | Aggregate difference without paired grounding. |
| auxiliary diagnostics | A1/A2 stress answer-space and counterfactual-assumption compliance. | Main evidence for the primary C0-C4 effect. |
| controlled diagnostics | Wording, color, format, and factor analyses that explain boundaries. | A new prompt-engineering mainline. |
| validity checks | Parser, source, visual clarity, and reproducibility checks. | Decisive proof that all threats are removed. |
| extension diagnostics | Multi-turn and case taxonomy analyses. | The central paper contribution. |

## Key Integrated Results

- Primary evaluation: C0 is fully faithful for all three models. LLaVA has `27/300`
  conflict-following outputs in C3 and `10/300` in C4; Qwen has `1/300` in C3 and C4;
  InternVL2 has `0/300` in C0-C4.
- Paired flips: because C0 is faithful, the LLaVA C3/C4 outputs can be written as
  same-image faithful-to-conflict flips.
- Wording robustness: LLaVA C3 drops from `27/300` to `5/300` and then `0/300` under
  C3 wording variants.
- Per-color split: LLaVA C3 flips are concentrated in `white -> black` (`20/27`), with
  smaller `black -> white` (`3/27`) and `blue -> red` (`4/27`) routes. C4 is also
  concentrated in `white -> black` (`8/10`).
- Answer format: LLaVA C3 is smaller under free answer (`2.33%`), multiple choice
  (`1.33%`), and yes/no false-claim probing (`1.33%`) than under the primary C3 format.
- Prompt factorization: quoted and indirect hints are weak; title/prefix framing and
  no-correction presupposition can be much stronger, including for Qwen and InternVL2.
- Visual clarity audit: target flip rows are mostly visually inspectable (`38/42` clear),
  but confound flags are more common among targets than controls (`11/42` vs. `4/42`).
- Multi-turn extension: InternVL2 shows a large dialogue-context vulnerability in MT2/MT3,
  while this remains separate from the primary single-turn evaluation.

## Overclaims To Avoid

- Do not claim a general VLM language bias.
- Do not claim text generally overrides vision.
- Do not claim the LLaVA C3 effect is uniform across colors.
- Do not claim Qwen or InternVL2 are globally robust to all misleading-text designs.
- Do not make A1/A2, prompt factorization, or multi-turn persuasion the primary evidence chain.
- Do not claim the visual clarity audit fully rules out visual ambiguity.

## Recommended Upload Order

1. `README.md`
2. `GPT_PROMPT_TEMPLATE.md`
3. `docs/final_writing_interface_note.md`
4. `docs/experiment_plan.md`
5. `docs/reproduction.md`
6. `data/metadata/balanced_eval_set/balanced_eval_set_summary.json`
7. `results/final_result_summary.md`
8. `results/results_discussion_summary.md`
9. `results/main/table1_main_metrics.csv`
10. `results/main/main_key_tests.csv`
11. `results/main/main_results_paper_ready.md`
12. `results/main/paired_flip_summary.md`
13. `results/main/figure2_conflict_aligned_rates.png`
14. `results/auxiliary/table3_aux_metrics.csv`
15. `results/auxiliary/aux_role_note.md`
16. `results/robustness/prompt_boundary_summary.md`
17. `results/parser/label_mapping_audit.md`
18. `results/appendix/stanford_core_sanity_check.md`
19. `results/reproducibility_audit.md`
20. `results/integrated_experiment_summary.md`
