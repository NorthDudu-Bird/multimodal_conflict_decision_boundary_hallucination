# Strengthening Master Plan

## Purpose

This plan records the completed strengthening path for the frozen empirical paper. It keeps the research question unchanged: whether VLMs are affected by false text prompts when the visual evidence is a clear single-attribute car-body color judgment.

The strengthening work is designed to improve rigor, interpretability, reproducibility, and paper writability without adding models, adding tasks, or expanding the claim into a broad language-bias story.

## Current Data Flow

| Stage | Script or source | Canonical output | Downstream role |
| --- | --- | --- | --- |
| Balanced dataset | `scripts/build_dataset.py` | `data/balanced_eval_set/final_manifest.csv`; balanced metadata; prompt CSVs | Defines the 300-image six-color evaluation set and C0-C4/A1-A2 prompt rows. |
| C0 baseline | `scripts/run_baseline_c0.py` | `results/baseline/*` | Confirms basic visual color recognition under neutral prompting. |
| Main C0-C4 experiment | `scripts/run_main_c0_c4.py`; `scripts/analyze_results.py` | `results/main/main_combined_parsed_results.csv`; `main_condition_metrics.csv`; `main_exact_tests.csv` | Primary evidence chain for conflict-aligned behavior. |
| Paper tables and figures | `scripts/make_figures.py` | `table1_main_metrics.csv`; `main_key_tests.csv`; `figure2_conflict_aligned_rates.png`; source appendix | Paper-facing summaries of main metrics, key tests, and source sanity check. |
| C3 wording variants | `scripts/run_robustness_c3_prompt_variants.py`; `scripts/analyze_prompt_variant_robustness.py` | `results/robustness/prompt_variant_*` | Boundary-control evidence for wording sensitivity. |
| Parser audit | `scripts/generate_parser_audit.py` | `results/parser/label_mapping_audit.md`; `ambiguous_outputs_sample.csv` | Checks whether parser mappings inflate the main finding. |
| Reproducibility audit | `scripts/verify_reproducibility.py` | `results/reproducibility_audit.md`; `reproducibility_comparison.csv` | Confirms locked canonical artifacts match the rerun snapshot. |

## Main Risks And Mitigations

| Risk | Why it matters | Mitigation | Output | Rerun class | Paper section |
| --- | --- | --- | --- | --- | --- |
| Attribution is not explicit enough | Condition-level rates can look like independent image pools. | Make same-image paired transitions explicit and quantify faithful-to-conflict flips from C0. | `results/main/paired_*`; `figure_paired_flip_rates.png` | Direct derived analysis | Methods, Results |
| Prompt variables are entangled | Original C3 could be a wording artifact rather than a stable behavior. | Reframe C3 variants as a boundary-control module and report weakened/disappearing effects. | `results/robustness/prompt_boundary_*`; `figure_prompt_boundary.png` | Direct derived analysis | Results, Discussion |
| Task may be visually ambiguous | Reviewers may ask whether conflict cases are simply harder images. | Build a human-review manifest and gallery for LLaVA original C3 conflict cases plus matched faithful controls. | `results/audit/visual_clarity_*` | Derived audit infrastructure | Appendix, Threats |
| Parser may inflate conflict-aligned counts | Alias/mention logic can create false positives. | Keep parser audit separate and emphasize main outputs are base single-label answers. | `results/parser/label_mapping_audit.md`; `results/threats_to_validity_summary.md` | Existing audit synthesis | Threats |
| Source composition may drive the effect | Final set mixes StanfordCars and VCoR. | Use source-stratified sanity check only as appendix evidence. | `results/appendix/stanford_core_sanity_check.md` | Existing derived analysis | Appendix, Threats |
| Reproducibility may be unclear | Empirical papers need artifact stability. | Reference locked snapshot audit and keep derived scripts separate from inference. | `results/reproducibility_audit.md`; new scripts | Existing audit plus derived scripts | Methods, Appendix |
| A1/A2 may be overused | Large A1/A2 numbers can be mistaken for primary evidence. | Reposition A1/A2 as auxiliary diagnostics: answer-space and counterfactual compliance stress tests. | `results/auxiliary/aux_role_note.md`; `aux_interpretation_summary.md` | Derived summary | Auxiliary Results, Discussion |

## New Strengthening Outputs

| Output | Source | Role |
| --- | --- | --- |
| `scripts/generate_paired_flip_analysis.py` | New derived script | Generates same-image transition counts, flip metrics, paired exact tests, and a paired-flip figure. |
| `results/main/paired_transition_counts.csv` | `main_combined_parsed_results.csv` | Full 5x5 state transition table for C0 vs C1-C4 by model. |
| `results/main/paired_flip_metrics.csv` | `main_combined_parsed_results.csv` | `answer_flip_rate`, `faithful_retention_rate`, `conflict_following_rate`, net shift, and discordant counts. |
| `results/main/paired_flip_key_tests.csv` | `main_combined_parsed_results.csv` | Exact paired McNemar tests for conflict-aligned shifts. |
| `results/main/paired_flip_summary.md` | Paired outputs | Direct Methods/Results wording for same-image inference. |
| `results/main/main_results_paper_ready.md` | Main metrics, paired summary, boundary summary | Main results narrative with claim contraction. |
| `scripts/generate_prompt_boundary_analysis.py` | New derived script | Repackages C3 variants as claim-boundary control. |
| `results/robustness/prompt_boundary_*` | Robustness combined parsed outputs | Boundary metrics, tests, summary, and figure. |
| `results/auxiliary/aux_role_note.md` | Existing auxiliary outputs | Prevents A1/A2 overclaiming. |
| `results/audit/visual_clarity_audit_manifest.csv` | LLaVA original C3 parsed outputs | Human-review checklist for image clarity and task validity. |
| `results/threats_to_validity_summary.md` | Existing audits plus visual audit | Formal threats-to-validity module. |
| `docs/writing_pack_upgrade_note.md` | New writing interface | Explains the upgraded 25-file writing pack. |

## Execution Order

1. Preserve pre-existing work with `git stash push -u -m "pre-strengthening-existing-deliverables"` and create `paper-strengthening-complete-20260430`.
2. Add derived analysis scripts only; do not call model inference scripts.
3. Generate paired flip, prompt boundary, and visual clarity audit outputs from existing parsed results.
4. Add paper-ready result summaries and formal threat/auxiliary role notes.
5. Update writing interfaces: `README.md`, `GPT_PROMPT_TEMPLATE.md`, `deliverables/README.md`, and the new writing-pack note.
6. Build the new capped 25-file writing pack with `scripts/build_writing_pack.py`.
7. Validate script compilation, expected counts, image paths, and pack file count.

## Decisions Locked

| Decision | Locked value |
| --- | --- |
| New model runs | None. |
| New tasks or attributes | None. |
| Primary inference target | Main C0-C4 conflict-aligned behavior, interpreted through same-image paired flips. |
| Boundary control | C3 wording variants limit the claim; they do not expand the story. |
| A1/A2 role | Auxiliary diagnostics only. |
| Remaining conclusion | Visual consistency dominates overall; LLaVA-1.5-7B shows limited significant conflict-aligned behavior only under the original strong misleading open template, and the effect is wording-sensitive. |
