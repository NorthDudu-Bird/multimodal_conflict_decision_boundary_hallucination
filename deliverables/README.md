# Deliverables

This directory contains paper-writing and review handoff packages.

## Current Recommended Package

- `gpt_paper_writing_pack_20files_final.zip`
  - Final recommended package for GPT paper drafting.
  - Contains exactly 20 selected files, excluding the manifest.
  - Covers project scope, reproduction, dataset balance, canonical main results,
    A1/A2 auxiliary diagnostics, wording robustness, parser/source/reproducibility
    checks, and the compact Phase 2 synthesis.
- `gpt_paper_writing_pack_20files_manifest.md`
  - Explains why each file was selected.
  - Provides the recommended GPT upload order.

## Historical Packages

Older packages are kept only as historical artifacts if present. They are not the
current writing interface:

- `gpt_paper_writing_pack_20260418.zip`
- `gpt_experiment_check_pack_20260418.zip`
- `gpt_paper_writing_pack_20files_20260430.zip`
- `gpt_paper_writing_pack_25files_20260430.zip`

## Regeneration

To regenerate the current final package, run:

```bash
python scripts/build_final_writing_pack.py
```

Do not use the old 20260430 pack builders as the final writing entrypoint.
