# Reproduction Guide

This guide reproduces the integrated experiment system by module. It separates primary
evaluation, auxiliary diagnostics, controlled diagnostics, validity checks, and extension
diagnostics.

## Environment

Use the repository configuration and model paths already defined in `configs/` and
`scripts/utils/paper_mainline_utils.py`. The fixed model set is:

- LLaVA-1.5-7B
- Qwen2-VL-7B-Instruct
- InternVL2-8B

The fixed evaluation set is `data/balanced_eval_set/final_manifest.csv`.

## A. Primary Evaluation

Canonical C0-C4 parsed outputs and metrics are stored under `results/main/` and
`results/baseline/`. Paper-facing tables and figures can be regenerated with:

```bash
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/generate_paired_flip_analysis.py
```

Key outputs:

- `results/main/table1_main_metrics.csv`
- `results/main/main_key_tests.csv`
- `results/main/figure2_conflict_aligned_rates.png`
- `results/main/paired_transition_counts.csv`
- `results/main/paired_flip_metrics.csv`
- `results/main/paired_flip_summary.md`

Expected primary checks:

- C0 faithful: `300/300` for all three models
- LLaVA C3 conflict-following: `27/300`
- LLaVA C4 conflict-following: `10/300`
- `results/main/main_key_tests.csv`: 12 rows

## B. Auxiliary Diagnostics

A1/A2 outputs are under `results/auxiliary/`.

Key files:

- `results/auxiliary/table3_aux_metrics.csv`
- `results/auxiliary/aux_role_note.md`
- `results/auxiliary/aux_interpretation_summary.md`

Expected compliance checks:

- Qwen A1 `55.67%`, A2 `90.67%`
- LLaVA A1 `85.33%`, A2 `100.00%`
- InternVL2 A1 `73.67%`, A2 `100.00%`

## C. Robustness And Controlled Diagnostics

Derived controlled diagnostics:

```bash
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_color_split_analysis.py
```

Local controlled diagnostic inference:

```bash
python scripts/run_controlled_diagnostics.py --family factorization --limit 12
python scripts/run_controlled_diagnostics.py --family format_control --limit 12
python scripts/run_controlled_diagnostics.py --family multiturn --limit 12
```

Full controlled diagnostic runs:

```bash
python scripts/run_controlled_diagnostics.py --family factorization
python scripts/run_controlled_diagnostics.py --family format_control
python scripts/run_controlled_diagnostics.py --family multiturn
python scripts/generate_integrated_synthesis.py
```

Key outputs:

- C3 wording robustness: `results/robustness/prompt_boundary_summary.md`
- Per-color split: `results/color_split/color_split_summary.md`
- Answer-format control: `results/format_control/format_control_summary.md`
- Prompt factorization: `results/factorization/factorized_prompt_summary.md`

Expected row-count checks:

- Color split main metrics: 90 rows
- Color paired metrics: 72 rows
- Factorization combined parsed results: 9000 rows
- Format-control combined parsed results: 9900 rows

## D. Validity Checks

Parser and source checks:

```bash
python scripts/generate_parser_audit.py
python scripts/verify_reproducibility.py
python scripts/generate_visual_clarity_audit.py
python scripts/generate_visual_clarity_completed_audit.py
```

Key outputs:

- `results/parser/label_mapping_audit.md`
- `results/parser/ambiguous_outputs_sample.csv`
- `results/appendix/stanford_core_sanity_check.md`
- `results/audit/visual_clarity_audit_manifest_completed.csv`
- `results/audit/visual_clarity_audit_summary.md`
- `results/reproducibility_audit.md`

Expected checks:

- `results/parser/ambiguous_outputs_sample.csv`: 27 rows
- `results/appendix/stanford_core_sanity_check.csv`: 12 rows
- Completed visual clarity audit manifest: 84 rows
- Completed visual clarity audit: no missing image paths

The reproducibility audit treats `results/final_result_summary.md` as a non-blocking
writing-facing integrated summary. Locked experimental manifests, prompts, result tables,
figures, parser/source audits, and statistical outputs remain under the blocking gate.

## E. Extension Diagnostics

Extension outputs are retained under:

- `results/multiturn/multiturn_summary.md`
- `results/case_analysis/failure_taxonomy_definition.md`
- `results/case_analysis/failure_taxonomy_counts.csv`
- `results/case_analysis/casebook.md`

Expected multi-turn combined parsed results: 5400 rows.

## Final Writing Pack

Regenerate the final 20-file writing package with:

```bash
python scripts/build_final_writing_pack.py
```

Outputs:

- `deliverables/gpt_paper_writing_pack_20files_final.zip`
- `deliverables/gpt_paper_writing_pack_20files_final/`
- `deliverables/gpt_paper_writing_pack_20files_manifest.md`

## Source Handling For Manuscript Writing

The final 300-image evaluation set should be written as a mixed-source car-image set,
not as two separate source-specific benchmarks. StanfordCars and VCoR both contribute
real vehicle images with a clearly visible principal car and inspectable car-body color
after cropping. Source identity is therefore retained for sanity checks and limitations,
but it is not a primary experimental factor.
