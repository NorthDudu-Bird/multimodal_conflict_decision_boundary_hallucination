# Validity And Control Plan

This document records how the study reduces alternative explanations while keeping the
research question fixed.

## Controlled Study Boundary

The study does not add models, tasks, or attributes beyond the frozen car-body
primary-color setting. It evaluates a fixed set of three VLMs on a fixed 300-image
balanced evaluation set.

## Validity Risks And Controls

| Risk | Control | Paper role |
| --- | --- | --- |
| Baseline color-recognition failure | C0 neutral baseline | Primary evaluation |
| Independent image-pool artifact | Same-image paired flips | Primary evaluation |
| Wording artifact | C3 wording robustness | Controlled diagnostic |
| Color-pair concentration | Per-color split and color-pair matrix | Controlled diagnostic |
| Answer-format dependence | Free, multiple-choice, and yes/no format controls | Controlled diagnostic |
| False-text form dependence | Prompt factorization | Controlled diagnostic |
| Parser inflation | Parser audit and ambiguous-output sample | Validity check |
| Source/style artifact | Source-stratified sanity check | Validity check |
| Image difficulty | Visual clarity audit with matched controls | Validity check |
| Artifact drift | Reproducibility audit | Validity check |
| Dialogue-context sensitivity | Multi-turn persuasive setting | Extension diagnostic |

## Writing Guidance

Use controls to narrow the claim rather than broaden it. The paper can state that the
observed LLaVA C3/C4 shift is local, paired, and statistically detectable, but also
wording-, format-, factor-, and color-pair-sensitive. It should not state that VLMs
generally follow language over vision.
