# Prompt Factorization Plan

## Purpose

This diagnostic module decomposes the false-text intervention without turning the paper into a prompt-engineering study. It asks which false-prompt forms are most associated with conflict-aligned answers in the same car-body primary-color task.

## Design

The module combines existing reference cells with five new prompt-factor cells.

Existing reference cells:

- `C0_neutral`
- original `C3_presupposition_correction_allowed`
- original `C4_stronger_open_conflict`
- `C3_v2`
- `C3_v3`

New factorized cells:

- `F1_quoted_claim`: quoted false statement in the user prompt.
- `F2_indirect_hint`: indirect false-color hint.
- `F3_title_prefix`: title/prefix framing with the false color.
- `F4_presupposition_no_correction`: presupposition without an explicit correction cue.
- `F5_previous_turn_context`: false-color context in a previous user turn, followed by a neutral final question.

## Factors

- Tone strength: suggestive, assertive, presuppositional, stronger open conflict.
- Injection position: question body, title/prefix, previous turn.
- False-text form: direct false statement, indirect hint, presupposition, quoted claim.
- Result requirement: one-label color output from the same six-color-plus-other space.

System-prompt injection is excluded because the current model runtime does not expose a uniform system-message interface across Qwen2-VL, LLaVA, and InternVL2.

## Analysis

Report faithful and conflict-aligned rates by model and factor cell. Compare each cell to the same model's C0 baseline using same-image paired tests when the image set matches. Interpret effects as local prompt-form diagnostics, not as a broad prompt-engineering benchmark.

