# Integrated Experiment System Plan

This document describes the final paper-facing experiment system by function.

## Study Boundary

The study evaluates whether erroneous textual cues shift VLM judgments in a controlled
car-body primary-color task. It keeps the dataset, models, and task fixed:

- Dataset: 300 images, six balanced colors
- Models: LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, InternVL2-8B
- Task: identify the primary body color of the main car

## Functional Layers

| Layer | Purpose | Modules | Outputs |
| --- | --- | --- | --- |
| A. Primary evaluation | Establish the main behavioral effect | balanced set; C0 baseline; C0-C4; same-image paired flips | `results/main/*`; `results/baseline/*` |
| B. Auxiliary diagnostics | Stress answer-space and assumption compliance | A1/A2 | `results/auxiliary/*` |
| C. Robustness and controlled diagnostics | Attribute and bound the effect | C3 wording; per-color split; format control; prompt factorization | `results/robustness/*`; `results/color_split/*`; `results/format_control/*`; `results/factorization/*` |
| D. Validity checks | Reduce alternative explanations | parser; source; visual clarity; reproducibility | `results/parser/*`; `results/appendix/*`; `results/audit/*`; `results/reproducibility_audit.md` |
| E. Extension diagnostics | Explore adjacent regimes and case interpretation | multi-turn; case taxonomy | `results/multiturn/*`; `results/case_analysis/*` |

## Main Claim Constraint

The paper should present the primary evidence chain first and use other modules to
interpret, bound, or validate that chain. The controlled diagnostics are important, but
they are not separate main claims. The extension diagnostics are useful for discussion
and appendix, but they should not redirect the paper.

## Expected Paper Placement

- Methods: dataset construction, model set, C0-C4 prompting, parser, paired design
- Results: C0 baseline, C0-C4 metrics, paired flips, C3 wording robustness
- Secondary Results: per-color split, answer-format control, prompt factorization
- Auxiliary Results or Appendix: A1/A2
- Validity and Limitations: parser, source, visual clarity, reproducibility
- Appendix or Discussion Extension: multi-turn setting and failure taxonomy
