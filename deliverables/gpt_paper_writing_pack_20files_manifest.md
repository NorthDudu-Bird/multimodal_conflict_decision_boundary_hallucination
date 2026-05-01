# GPT Paper Writing Pack - 20 Files Final

- Zip: `deliverables/gpt_paper_writing_pack_20files_final.zip`
- Count: 20 selected project files, excluding this manifest.
- Purpose: compact, writing-facing handoff that covers the final empirical study without dragging GPT into raw-result sprawl.

## Files And Purposes

| Upload order | File | Purpose |
| --- | --- | --- |
| 1 | `README.md` | Top-level project scope, frozen empirical-paper boundary, and result map. |
| 2 | `GPT_PROMPT_TEMPLATE.md` | Drafting instructions and explicit overclaim boundaries for downstream GPT writing. |
| 3 | `docs/final_writing_interface_note.md` | Final writing interface: evidence hierarchy, terminology, caveats, and upload order. |
| 4 | `docs/experiment_plan.md` | Original C0-C4/A1-A2 experimental design and main protocol framing. |
| 5 | `docs/reproduction.md` | Reproduction and rerun instructions, including Phase 2 diagnostic scripts. |
| 6 | `data/metadata/balanced_eval_set/balanced_eval_set_summary.json` | Dataset balance summary: 300 images, six colors, source composition. |
| 7 | `results/final_result_summary.md` | Writing-facing final result summary with Phase 2 addendum. |
| 8 | `results/results_discussion_summary.md` | Discussion-ready synthesis and limitation language. |
| 9 | `results/main/table1_main_metrics.csv` | Canonical main C0-C4 metrics across all three models. |
| 10 | `results/main/main_key_tests.csv` | Canonical statistical tests for the main experiment. |
| 11 | `results/main/main_results_paper_ready.md` | Paper-ready main experiment narrative and table/figure references. |
| 12 | `results/main/paired_flip_summary.md` | Same-image C0-to-conflict paired flip interpretation. |
| 13 | `results/main/figure2_conflict_aligned_rates.png` | Main conflict-aligned rate figure for paper drafting. |
| 14 | `results/auxiliary/table3_aux_metrics.csv` | A1/A2 auxiliary diagnostic metrics. |
| 15 | `results/auxiliary/aux_role_note.md` | Boundary note keeping A1/A2 as auxiliary diagnostics, not main evidence. |
| 16 | `results/robustness/prompt_boundary_summary.md` | C3 wording robustness and template-sensitivity summary. |
| 17 | `results/parser/label_mapping_audit.md` | Parser reliability audit and label-mapping notes. |
| 18 | `results/appendix/stanford_core_sanity_check.md` | Source-stratified sanity check for StanfordCars core images. |
| 19 | `results/reproducibility_audit.md` | Canonical reproducibility audit and Phase 2 summary-drift interpretation. |
| 20 | `results/phase2_final_summary.md` | Compact A-G Phase 2 synthesis covering all new diagnostics. |

## Selection Logic

The pack prioritizes project boundary, reproduction, dataset balance, canonical main results, A1/A2 auxiliary diagnostics, wording robustness, parser/source/reproducibility checks, and one compact Phase 2 synthesis. It intentionally does not include every raw Phase 2 CSV because the downstream writing task needs the final interpretation and boundaries, not a new benchmark-style data dump.

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
20. `results/phase2_final_summary.md`
