# Gatekeeping Summary

This pipeline organizes the main and phase-2 evidence as a controlled diagnostic evaluation. It does not turn the paper into a broad benchmark leaderboard.

| gate_id | gate | role | evidence | alternative_explanation_reduced | verdict |
| --- | --- | --- | --- | --- | --- |
| Gate 1 | Dataset validity / balance | main evidence chain | 300 reviewed images; six true colors balanced at 50 each; source composition StanfordCars=93 and VCoR=207. | Uneven color priors or accidental single-source dataset construction. | Pass; keep task local to car-body primary color. |
| Gate 2 | Neutral visual fidelity (C0) | main evidence chain | C0 faithful=900/900; conflict_aligned=0/900. | Baseline color-recognition failure under neutral prompting. | Pass; conflict outputs can be interpreted against a faithful visual baseline. |
| Gate 3 | Parser reliability | main evidence chain | Main C0-C4 parse_error=0; parser audit remains documented in results/parser/label_mapping_audit.md. | Parser inflation of conflict-aligned counts. | Pass for mainline; yes/no phase-2 rows are separately normalized. |
| Gate 4 | Source robustness | appendix threat reduction | LLaVA C3 split: StanfordCars 13/93 (13.98%); VCoR 14/207 (6.76%); C0 remains faithful in both sources. | Effect exists only because one source is visually invalid. | Partial pass; direction persists, magnitude differs and should not be overinterpreted. |
| Gate 5 | Visual clarity validity | appendix threat reduction | Targets clear=38/42, controls clear=39/42; any confound targets=11/42, controls=4/42. | Flip cases are simply unreadable or systematically occluded. | Mostly pass; residual reflection/lighting/background confounds remain visible. |
| Gate 6 | Wording boundary | main boundary evidence | LLaVA C3 original=27/300 (9.00%); C3-v2=5/300; C3-v3=0/300. | Original C3 reflects a stable cross-wording law. | Boundary set; the effect is wording-sensitive. |
| Gate 7 | Color / format / factorization diagnostics | secondary attribution | LLaVA original C3/C4 flips concentrate in 33/37 achromatic black/white rows; LLaVA free/MC/yes-no C3 rates are 2.33%, 1.33%, 1.33%; factor peaks include LLaVA title/prefix 32.00%, Qwen no-correction presupposition 34.00%, InternVL title/prefix 36.00%. | The 9% LLaVA result is uniform across colors, formats, and false-text forms. | Boundary tightened; prompt form, answer form, and color pair all matter. |
| Gate 8 | Multi-turn extension | appendix extension | InternVL2 MT2=64/300 (21.33%), MT3=224/300 (74.67%); LLaVA MT3=0/300. | Single-turn stability necessarily implies dialogue stability. | Extension only; multi-turn vulnerability is strong for InternVL2 but not a replacement for C0-C4. |

## Main vs Appendix Placement

- Main chain: Gates 1-3, same-image C0-C4 flips, Gate 6 wording boundary.
- Secondary attribution in Results: per-color split, answer-format control, and the most compact prompt-factorization comparison from Gate 7.
- Appendix and threats: source robustness, completed visual clarity audit, multi-turn extension, parser/reproducibility details, and full gate table.
