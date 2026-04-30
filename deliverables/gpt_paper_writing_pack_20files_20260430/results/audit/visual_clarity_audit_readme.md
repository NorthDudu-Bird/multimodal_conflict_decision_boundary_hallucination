# Visual Clarity Audit Readme

## Purpose

This audit checks whether the LLaVA-1.5-7B original `C3` conflict-aligned cases might be visually harder or less task-valid than matched faithful controls.

## Scope

- Conflict-aligned cases: 27
- Matched faithful controls: 27
- Matching priority: exact `true_color + source_dataset`, then `true_color`, then source-balanced deterministic fill.
- Human review fields are intentionally blank so the audit can be completed independently.

## Role In The Paper

- This module is threat reduction, not a new main experiment.
- It helps address whether the observed conflict-aligned cases are concentrated in visually ambiguous examples.
- It should not be used to claim a new visual difficulty factor unless a completed human audit supports that claim.

## File Checks

- Image paths checked: 54
- Missing image paths: 0
