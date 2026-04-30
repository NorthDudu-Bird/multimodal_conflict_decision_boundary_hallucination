# Writing Pack Upgrade Note

## What Changed

The writing interface was upgraded from a compact result bundle into a stronger paper-writing pack. The new pack keeps the 25-file cap while adding the highest-value strengthening artifacts:

- same-image paired flip summary;
- main results paper-ready summary;
- C3 wording boundary-control summary;
- auxiliary role note;
- formal threats-to-validity summary;
- visual clarity audit readme.

## Recommended Upload Order

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
