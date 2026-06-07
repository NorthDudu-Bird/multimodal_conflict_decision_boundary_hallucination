# Reproduction Guide

This guide covers the stored primary evaluation, auxiliary diagnostics, controlled
diagnostics, and validity checks.

## Environment

Install `requirements.txt`, download the three model checkpoints, and configure their
local paths through `configs/experiment.yaml`. The evaluated model families are:

- LLaVA-1.5-7B
- Qwen2-VL-7B-Instruct
- InternVL2-8B

The fixed evaluation manifest is `data/balanced_eval_set/final_manifest.csv`.

## Primary Evaluation

Run:

```powershell
python scripts/analyze_results.py
python scripts/make_figures.py
python scripts/generate_paired_flip_analysis.py
```

Key outputs are stored in `results/baseline/` and `results/main/`, including condition
metrics, exact tests, paired transitions, paired-flip metrics, and plots.

## Auxiliary Diagnostics

A1/A2 outputs are stored in `results/auxiliary/`. Regenerate them with:

```powershell
python scripts/run_aux_a1_a2.py --skip-build
```

## Robustness and Controlled Diagnostics

Regenerate derived analyses:

```powershell
python scripts/generate_prompt_boundary_analysis.py
python scripts/generate_color_split_analysis.py
```

Run controlled diagnostic inference:

```powershell
python scripts/run_controlled_diagnostics.py --family factorization
python scripts/run_controlled_diagnostics.py --family format_control
python scripts/run_controlled_diagnostics.py --family multiturn
```

Outputs are stored in `results/robustness/`, `results/color_split/`,
`results/factorization/`, `results/format_control/`, and `results/multiturn/`.

## Validity Checks

Run:

```powershell
python scripts/generate_parser_audit.py
python scripts/generate_visual_clarity_audit.py
python scripts/generate_visual_clarity_completed_audit.py
python scripts/verify_reproducibility.py
```

Key artifacts:

- `results/parser/label_mapping_audit.md`
- `results/parser/ambiguous_outputs_sample.csv`
- `results/appendix/stanford_core_sanity_check.md`
- `results/audit/visual_clarity_audit_manifest_completed.csv`
- `results/audit/visual_clarity_audit_summary.md`
- `results/reproducibility_audit.md`

The reproducibility gate compares canonical manifests, prompts, parsed outputs,
metrics, figures, parser audits, and appendix checks against a local locked snapshot.
Runtime logs, raw timing information, and machine-specific metadata are excluded from
the gate.
