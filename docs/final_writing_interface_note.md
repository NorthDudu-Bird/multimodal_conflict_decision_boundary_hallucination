# Final Writing Interface Note

This file is the final handoff interface for drafting the paper. It consolidates the
paper stance, evidence hierarchy, safe wording, and upload order after the Phase 2
strengthening pass.

## Final Paper Stance

The paper is an empirical behavior-analysis study of whether erroneous textual cues can
shift visual-language model judgments in a controlled car-body primary-color task.
It is not a method paper, not a general language-bias benchmark, not a prompt-engineering
paper, and not a multi-turn persuasion paper.

The safest main claim is:

> In a visually clear, six-color balanced car-body color task, all three models are
> faithful under neutral prompting. Under the original single-turn C0-C4 conflict
> templates, LLaVA-1.5-7B shows limited but statistically significant same-image
> conflict-following in C3/C4, while Qwen2-VL-7B-Instruct and InternVL2-8B are
> essentially stable in that original template family. Phase 2 diagnostics show that
> the effect is conditional on wording, answer format, prompt factor, and color pair,
> rather than a uniform text-over-vision phenomenon.

## Unified Wording Table

| Term | Use This Meaning | Do Not Write |
| --- | --- | --- |
| faithful | The parsed answer matches the true car-body primary color. | "correct" without saying the task is color-label matching. |
| conflict_aligned / conflict_following | The parsed answer follows the false prompt color in a conflict condition. | "hallucination" as the default label for every case. |
| answer flip / paired flip | Same-image transition from faithful in C0 to conflict-aligned under a conflict prompt. | Independent aggregate difference without paired grounding. |
| prompt wording sensitivity | The effect changes sharply across C3 wording variants and factorized prompts. | Stable cross-template law. |
| auxiliary diagnostics | A1/A2 stress answer-space and counterfactual-assumption compliance. | Main evidence for C0-C4 conflict effects. |
| threat reduction / gatekeeping | Checks that reduce alternative explanations. | New headline contribution or new benchmark claim. |

## Evidence Placement

Main-text primary evidence:

1. Dataset balance and visual task definition: 300 images, six colors, 50 per color.
2. C0-C4 main experiment across LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, and InternVL2-8B.
3. Same-image paired flip analysis relative to C0.
4. C3 wording boundary: LLaVA C3 original drops from 27 flips to 5 and then 0 under variants.

Main-text secondary attribution:

1. Per-color split: LLaVA's original C3/C4 effect is concentrated, especially in
   `white -> black`, rather than uniform across colors.
2. Answer-format control: label-set/open original C3 is larger than free, multiple-choice,
   or yes/no variants, so the main effect is partly format/framing-sensitive.
3. Prompt factorization: title/prefix framing and no-correction presupposition can produce
   larger shifts, including for Qwen and InternVL2, but this is a diagnostic extension and
   must not replace the original C0-C4 mainline.

Appendix or extension diagnostics:

1. Visual clarity audit: use as threat reduction against "the images are unreadable."
2. Multi-turn persuasion: report as an extension only, especially InternVL2 MT2/MT3.
3. Case-level taxonomy and gatekeeping details.
4. Parser audit, source-stratified sanity check, reproducibility audit.

## Non-Negotiable Boundaries

- Do not claim that VLMs generally prioritize language over vision.
- Do not claim that all three models are unstable in the original C0-C4 template family.
- Do not claim that Qwen/InternVL2 are globally robust to misleading text; Phase 2
  factorization and multi-turn diagnostics show important exceptions.
- Do not describe A1/A2 as main evidence for the C0-C4 conflict effect.
- Do not write the color split as a general theory of color perception; it is a local
  attribution check for the car-body color task.
- Do not write the format-control result as a pure answer-format causal effect; task
  wording and response schema change together.
- Do not write the visual audit as fully eliminating all visual ambiguity. It is a
  single-reviewer, local visual review that reduces but does not eliminate that threat.
- Do not promote multi-turn results into the paper's central story.

## Key Phase 2 Corrections To The Mainline

- The 9.00% LLaVA C3 effect is not a uniform color effect. It is concentrated in a small
  set of color pairs: 20/27 C3 flips are `white -> black`, with 3/27 `black -> white`
  and 4/27 `blue -> red`.
- LLaVA C4 has the same concentration pattern at a smaller scale: 8/10 flips are
  `white -> black`, and 2/10 are `black -> white`.
- Qwen and InternVL2 can be shifted by specific factorized or multi-turn prompts, so the
  stable-model statement must be scoped to the original single-turn C0-C4 template family
  and the C3 wording-boundary family.
- Visual audit supports the claim that target flips are usually visually reviewable
  (38/42 clear), but confound flags are somewhat more common in flip targets than controls
  (11/42 vs. 4/42). This belongs in limitations/threat reduction.
- Reproducibility currently has one expected mismatch: `results/final_result_summary.md`
  differs from the locked snapshot because it now includes Phase 2 writing-facing text.
  Canonical result tables and figures remain matched.

## Recommended Upload Order For GPT Drafting

Upload the final 20-file pack in this order:

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
20. `results/phase2_final_summary.md`

This set is intentionally compact. It includes one Phase 2 synthesis file rather than
all Phase 2 raw tables, so GPT receives the final interpretation without being pulled
into a new benchmark or prompt-engineering story.
