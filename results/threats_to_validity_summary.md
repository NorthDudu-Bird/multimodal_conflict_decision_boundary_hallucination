# Threats To Validity Summary

## Parser Reliability

The parser audit checks whether label mapping or ambiguous outputs inflate
conflict-following counts. The primary C0-C4 outputs are simple color-label answers, and
the main conclusion does not rely on broad alias expansion.

Remaining boundary: parser reliability supports the reported counts, but the paper
should still describe the parsing rule and include the audit in appendix.

## Source Composition

The final evaluation set combines StanfordCars and VCoR images. Source-stratified sanity
checks reduce the concern that the main effect is a single-source artifact.

Remaining boundary: source checks are sanity checks, not a new source-generalization
benchmark.

## Visual Clarity

The visual clarity audit reviews target flip rows and matched faithful controls. Most
target rows are visually inspectable (`38/42` clear), similar to controls (`39/42`
clear). This reduces the alternative explanation that the flip set is globally
unreadable.

Remaining boundary: visual confound flags are more frequent among target rows (`11/42`)
than controls (`4/42`). Reflection, lighting, background color, and multi-car
interference remain local limitations.

## Wording, Format, And Factor Dependence

C3 wording robustness, answer-format control, and prompt factorization show that the
effect is not a stable response to all misleading-text prompts. The paper should write
the LLaVA C3/C4 effect as conditional and bounded.

Remaining boundary: controlled diagnostics explain the observed behavior; they should
not be written as a broad prompt-engineering theory.

## Color-Pair Concentration

Per-color analysis shows that LLaVA C3/C4 flips concentrate strongly in achromatic
black/white routes, especially `white -> black`.

Remaining boundary: the 9.00% C3 result should not be described as a uniform color
effect or as a general color-perception finding.

## Reproducibility

Locked canonical result tables, statistical tests, parser/source audits, and figures
match the reproducibility snapshot. The only mismatch is a writing-facing summary file
that was reorganized into the final integrated paper structure.

Remaining boundary: readers should distinguish stable result artifacts from evolving
paper-facing summaries.

## Extension Diagnostics

The multi-turn persuasive setting shows a strong InternVL2 vulnerability under repeated
previous-turn false context. This is a separate dialogue-context diagnostic.

Remaining boundary: multi-turn results should not be folded into the primary single-turn
C0-C4 conclusion.
