# Writing Pack Upgrade Note

This note records the final writing-pack status after Phase 2. Older 20260430 20-file
and 25-file packs are historical artifacts only; they are no longer the recommended
paper-writing interface.

## Current Recommended Pack

Use:

- `deliverables/gpt_paper_writing_pack_20files_final.zip`
- `deliverables/gpt_paper_writing_pack_20files_manifest.md`

The final 20-file pack is deliberately compact. It includes the project boundary,
reproduction entrypoint, dataset balance, canonical main metrics and tests, A1/A2
auxiliary diagnostics, prompt wording boundary, parser/source/reproducibility checks,
and one compact Phase 2 synthesis.

## Why Not The Old 25-File Pack?

The old 25-file pack predates Phase 2 and still treats several materials as
infrastructure rather than completed diagnostics. It should not be used as the current
writing input because it does not carry the final color-split, factorization,
format-control, multi-turn, visual-audit, and gatekeeping boundaries.

## Final Upload Order

Follow the order in:

- `docs/final_writing_interface_note.md`
- `deliverables/gpt_paper_writing_pack_20files_manifest.md`

## Writing Boundaries

Use Phase 2 to narrow and strengthen the original mainline, not to replace it. The
paper remains a controlled empirical study of car-body primary-color conflict prompts.
A1/A2 remain auxiliary diagnostics. Factorization and multi-turn results are boundary
and extension diagnostics, not a new prompt-engineering or persuasion paper.

The safest final wording remains: visual evidence dominates overall in this task; under
the original single-turn C0-C4 templates, LLaVA-1.5-7B shows a limited, significant,
same-image conflict-following shift in C3/C4; this shift is wording-, format-, and
color-pair-sensitive, especially concentrated in `white -> black`; Qwen2-VL-7B-Instruct
and InternVL2-8B are stable in the original single-turn template family, but not
globally immune to every factorized or multi-turn misleading-text design.
