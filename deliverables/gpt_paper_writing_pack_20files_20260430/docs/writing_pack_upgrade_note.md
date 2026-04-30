# Writing Pack Upgrade Note

## What Changed

The writing interface was upgraded from a compact result bundle into stronger paper-writing packs. The default 25-file pack keeps more tables and figures. The compact 20-file pack is for platforms with a stricter upload cap and still preserves all major evidence categories through consolidated summaries.

- same-image paired flip summary;
- main results paper-ready summary;
- C3 wording boundary-control summary;
- auxiliary role note;
- formal threats-to-validity summary;
- visual clarity audit readme.

## Recommended 20-File Upload Pack

If the platform allows only 20 files, use:

- `deliverables/gpt_paper_writing_pack_20files_20260430.zip`

The 20-file pack intentionally drops only redundant or lower-priority files:

- `docs/experiment_plan.md` and `docs/reproduction.md` are represented by `README.md`, `docs/strengthening_master_plan.md`, and `results/reproducibility_audit.md`.
- `results/main/main_key_tests.csv` is represented by the key-test table in `results/main/main_stats_summary.md`.
- `results/robustness/prompt_boundary_metrics.csv` is represented by the metric bullets in `results/robustness/prompt_boundary_summary.md`.
- `results/main/figure2_conflict_aligned_rates.png` is represented by `table1_main_metrics.csv` and the paper-ready summaries.

## Recommended Manual Upload Order

1. `README.md`
2. `GPT_PROMPT_TEMPLATE.md`
3. `results/final_result_summary.md`
4. `results/main/main_results_paper_ready.md`
5. `results/main/paired_flip_summary.md`
6. `results/main/table1_main_metrics.csv`
7. `results/main/main_stats_summary.md`
8. `results/robustness/prompt_boundary_summary.md`
9. `results/auxiliary/aux_role_note.md`
10. `results/threats_to_validity_summary.md`
11. `results/audit/visual_clarity_audit_readme.md`
12. `results/parser/label_mapping_audit.md`
13. `results/appendix/stanford_core_sanity_check.md`
14. `results/reproducibility_audit.md`

## Writing Boundaries

Use the paired analysis to strengthen the main inference, not to broaden the claim. Use prompt-boundary results to limit the conclusion, not to claim robustness. Use A1/A2 only as auxiliary diagnostics. Use parser/source/reproducibility/visual-clarity materials as threats-to-validity support.

The safest final wording remains: visual evidence dominates overall in this task, while LLaVA-1.5-7B shows a limited, significant, and wording-sensitive conflict-aligned shift under the original strong misleading open template.
