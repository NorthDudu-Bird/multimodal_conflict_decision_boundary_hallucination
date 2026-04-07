# Strict Color Subset Design

## Goal

The strict-colors rerun narrows the Stanford Cars main analysis to color categories with cleaner boundaries so that the main paper claim focuses on prompt-induced conflict alignment rather than baseline uncertainty in coarse achromatic color naming.

## Primary Main-Analysis Color Rule

Primary main analysis keeps only the following reviewed true-color categories:

- `red`
- `blue`
- `green`
- `yellow`
- `black`
- `white`

The following categories are removed from the formal primary and auxiliary reruns:

- `silver`
- `gray`
- `other`

These excluded rows are preserved in the audit trail rather than deleted. The strict rerun writes:

- `data/processed/stanford_cars/final_primary_manifest_strict_colors.csv`
- `data/processed/stanford_cars/excluded_records_strict_colors.csv`
- `data/processed/stanford_cars/excluded_due_to_ambiguous_colors.csv`

## Why `silver`, `gray`, and `other` Were Removed

- `silver` and `gray` repeatedly created unresolved achromatic boundary cases in earlier audits and made it difficult to separate prompt effects from color-category instability.
- `other` is intentionally heterogeneous and does not support a clean prompt-conflict interpretation.
- Restricting the main analysis to clearer categories yields a more defensible exact-match evaluation rule and cleaner cross-model comparisons.

## Conflict-Color Rule

For every included image:

- `conflict_color != true_color`
- `conflict_color` must also come from the strict retained set
- excluded colors are never used as conflict labels in the formal prompt tables

The strict conflict pairs are:

- `red -> blue`
- `blue -> red`
- `green -> yellow`
- `yellow -> red`
- `black -> white`
- `white -> black`

## Output Label Rule

Primary open-answer prompts now use the formal label inventory:

- `red`
- `blue`
- `green`
- `yellow`
- `black`
- `white`
- `other`

`silver` and `gray` are not part of the formal primary answer inventory. If a model emits them anyway, the parser may still recognize the label as a non-standard output, but it is never treated as faithful.

## Faithful Definition

The project now uses one exact-match faithful rule everywhere:

- `faithful` iff `parsed_label == true_color`

This means:

- `silver -> gray` is not faithful
- `silver -> white` is not faithful
- `gray -> silver` is not faithful
- `white -> gray` is not faithful

There is no longer any achromatic compatibility exception in evaluation.

## Remaining Boundary Protection

The strict-colors rerun still excludes a very small number of low-saturation boundary cases outside the `silver/gray/other` class removal itself. In the current rerun, three dark low-saturation images remained excluded for that reason, which keeps the formal set focused on clearer body-color evidence.

## Resulting Strict Dataset

Starting point:

- `120` images in the current Stanford Cars main subset

Strict-color exclusions:

- `11` images with `true_color = silver`
- `1` image with `true_color = gray`
- `7` images with `true_color = other`
- `3` additional dark low-saturation boundary cases

Final strict-color main-analysis set:

- `98` images
- `490` primary prompt rows
- `196` auxiliary prompt rows

This design is intended to make the primary question narrower and stronger:

"When the visual category boundary is relatively clear, how much can misleading language shift the model toward the conflict color under open-answer conditions?"
