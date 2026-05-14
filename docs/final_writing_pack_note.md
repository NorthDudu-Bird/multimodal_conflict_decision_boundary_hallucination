# Final Writing Pack Note

The final writing package is:

- `deliverables/gpt_paper_writing_pack_20files_final.zip`
- `deliverables/gpt_paper_writing_pack_20files_manifest.md`

The package is compact by design. It presents the study as one integrated experiment
system organized by experimental function:

1. Primary evaluation
2. Auxiliary diagnostics
3. Robustness and controlled diagnostics
4. Validity checks
5. Extension diagnostics

## Why The Pack Is Limited To 20 Files

The pack is meant for GPT-based paper drafting, not raw data inspection. It includes the
files needed to write the paper's Methods, Results, Discussion, and Limitations while
avoiding redundant raw outputs.

The package includes:

- project scope and drafting instructions;
- reproduction guide;
- dataset balance summary;
- primary metrics, tests, paired flips, and figure;
- A1/A2 auxiliary diagnostic metrics;
- C3 wording robustness;
- parser, source, and reproducibility checks;
- one integrated summary covering controlled, validity, and extension diagnostics.

## Current Writing Boundary

The paper should be written as a controlled empirical study of car-body primary-color
conflict prompts. A1/A2 are auxiliary diagnostics. Per-color split, answer-format
control, and prompt factorization are controlled diagnostics. Visual clarity audit is a
validity check. Multi-turn persuasion and failure taxonomy are extension diagnostics.

The safe conclusion is local: visual evidence dominates under neutral prompting and most
primary conflict settings; LLaVA shows a limited same-image conflict-following shift in
canonical C3/C4; the shift is conditional on wording, format, factor, and color pair.
