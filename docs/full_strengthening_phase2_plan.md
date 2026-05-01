# Full Strengthening Phase 2 Plan

## Branch And Starting State

- Working branch for this phase: `exp/full_strengthening_color_prompt_multiturn`.
- Starting branch before switch: `paper-strengthening-complete-20260430`.
- Pre-existing dirty worktree state preserved: deleted files under `deliverables/gpt_paper_writing_pack_25files_20260430*`.
- These deliverables deletions are not part of this phase and must not be restored or discarded unless explicitly requested.

## Frozen Research Boundary

This phase keeps the paper as an empirical behavior-analysis study of false text prompts in a clear single-attribute car-body primary-color task. It does not add models, add attribute tasks, infer model-scale effects, propose a method, or convert the work into a broad benchmark leaderboard.

The retained central claim is intentionally narrow: visual evidence dominates overall in this controlled task, while LLaVA-1.5-7B shows limited, significant, same-image conflict-aligned shifts under the original strong misleading C3/C4 templates. New analyses are allowed only as attribution, boundary control, diagnostic extension, or threat reduction.

## Module Plan

| Module | Type | Data dependency | New prompts | New inference | Outputs |
| --- | --- | --- | --- | --- | --- |
| Color split | Direct derived | `results/main/main_combined_parsed_results.csv` | No | No | `results/color_split/*` |
| Visual clarity completed audit | Direct derived + image review | main results, prompt variants, images | No | No | `results/audit/visual_clarity_audit_manifest_completed.csv`; gallery; reviewer instructions; summary |
| Prompt factorization | New diagnostic inference | balanced manifest and existing C0/C3/C4/C3 variants | Yes | Yes | `prompts/factorization/factorized_prompt_prompts.csv`; `results/factorization/*` |
| Answer format control | New diagnostic inference | balanced manifest and existing C0/C3/C4 | Yes | Yes | `prompts/format_control/format_control_prompts.csv`; `results/format_control/*` |
| Multi-turn persuasion | New diagnostic inference | balanced manifest and runtime context support | Yes | Yes | `prompts/multiturn/multiturn_prompts.csv`; `results/multiturn/*` |
| Failure taxonomy | Direct derived synthesis | all available main + diagnostic outputs | No | No | `results/case_analysis/*` |
| Gatekeeping protocol | Direct derived synthesis | all canonical and phase2 summaries | No | No | `docs/gatekeeping_protocol.md`; `results/gatekeeping/*` |

## Implementation Decisions

- Existing canonical directories are not overwritten except for writing-facing summaries intentionally updated after phase2 synthesis.
- New inference modules write under separate result directories and use separate prompt CSVs.
- Runtime receives optional `context_turns_json` support. Rows without this field retain the existing single-turn behavior.
- System-prompt injection is excluded because the current runtime has no uniform system-message path across Qwen2-VL, LLaVA, and InternVL2.
- New prompt metadata fields are carried through CSVs: `diagnostic_family`, `factor_id`, `tone_strength`, `injection_position`, `false_text_form`, `answer_format`, `response_schema`, and `context_turns_json`.

## Rerun Classes

- Direct derived only: color split, audit pack, taxonomy, gatekeeping, writing summaries.
- Local inference reruns: factorization, format control, multi-turn.
- Full canonical rerun trigger: only if the dataset builder, existing C0-C4 prompt generator, or parser semantics for existing canonical outputs are changed. This phase avoids those changes.

## Validation

- Run smoke tests with `--limit 12` for new inference modules before full module runs.
- Confirm row counts, parse status, figure files, image paths, and phase2 summary existence.
- Run Python compilation checks for all touched scripts.
- Run `scripts/verify_reproducibility.py` to confirm unchanged canonical artifacts remain reproducible.

## Known Early Boundary

A pre-implementation read-only check found that LLaVA original C3 flips are concentrated in a small set of color-pair routes: `white -> black` dominates, with smaller `black -> white` and `blue -> red` contributions. The color-split module must report this directly and, if confirmed, the paper should tighten the conclusion from a generic color-task prompt effect to a conditional model/template/color-pair-sensitive shift.
