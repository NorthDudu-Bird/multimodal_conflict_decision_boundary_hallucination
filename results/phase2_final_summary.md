# Phase 2 Final Summary

Phase 2 strengthens the original empirical paper without changing the frozen research
question: in a controlled car-body primary-color task, do erroneous textual cues shift
VLM judgments away from clear visual evidence?

## Final QA Snapshot

| Module | Output checked | QA result | Writing placement |
| --- | --- | --- | --- |
| Per-color split | 90 color metric rows, 72 paired-flip rows, color matrix and figures | Pass | Main-text secondary attribution |
| Visual clarity audit | 84 reviewed rows, 42 target flips and 42 matched controls | Basic pass | Threat reduction / appendix |
| Prompt factorization | 9,000 parsed rows, zero parse errors | Pass with boundary caution | Main text for selected factors; details in appendix |
| Answer-format control | 9,900 parsed rows, one parse error | Pass with comparability caution | Main-text secondary attribution or compact appendix |
| Multi-turn persuasion | 5,400 parsed rows, zero parse errors | Pass as extension | Appendix / extension diagnostic |
| Failure taxonomy | Counts, definitions, casebook, gallery | Pass | Discussion support / appendix |
| Gatekeeping protocol | Eight gates with evidence and threat mapping | Pass | Methods framing plus appendix details |

## A. Per-Color Split

The original LLaVA C3 effect is not uniform across colors. Among 27 same-image
faithful-to-conflict flips, 20 are `white -> black`, 3 are `black -> white`, and 4 are
`blue -> red`. C4 shows the same pattern at lower magnitude: 8/10 flips are
`white -> black`, and 2/10 are `black -> white`.

This requires a narrower conclusion: the observed 9.00% C3 shift is a conditional
template effect with strong color-pair concentration, especially in achromatic
`white -> black` cases. It should not be written as a uniform color effect or as a broad
law of visual-language conflict.

## B. Visual Clarity Audit

The completed audit covers 42 target flip rows and 42 matched faithful controls.
Target flips are usually visually reviewable: 38/42 are marked clear, compared with
39/42 controls. Confound flags are more frequent among targets than controls
(11/42 vs. 4/42), mostly involving reflections, lighting/shadow, background bias, or
multi-car interference.

The audit supports the limited claim that the flip set is not dominated by unreadable
images. It does not fully rule out local visual ambiguity, and it should be written as
single-reviewer threat reduction rather than a standalone human study.

## C. Prompt Factorization

The factorized diagnostics show that not all erroneous-text forms behave alike.
Quoted claims and indirect hints are near zero across models. Stronger framing can
matter: LLaVA reaches 32.00% under title/prefix false-color framing and 16.33% under
no-correction presupposition; Qwen reaches 34.00% under no-correction presupposition;
InternVL2 reaches 36.00% under title/prefix framing.

This changes the wording around model stability. Qwen and InternVL2 are stable in the
original single-turn C0-C4 template family, but they are not globally immune to all
misleading-text designs. Factorization should be used to explain which prompt factors
make the conflict stronger, not to reposition the whole paper as prompt engineering.

## D. Answer-Format Control

The LLaVA original C3 label-set framing remains the largest single-turn format result
in this module: 27/300 (9.00%). More constrained variants are smaller: free C3 is
7/300 (2.33%), multiple-choice C3 is 4/300 (1.33%), and yes/no false-claim probing is
4/300 (1.33%).

This supports a cautious statement that the original open label-set framing amplifies
conflict-following relative to these alternatives. It should not be described as a
pure answer-format effect because wording and response schema also vary.

## E. Multi-Turn Persuasion

LLaVA and Qwen remain close to zero in the compact multi-turn setting, while InternVL2
shows a large extension-only vulnerability: 64/300 (21.33%) in MT2 and 224/300
(74.67%) in MT3.

This is credible as an exploratory diagnostic because parse errors are zero and the
row counts match the design. It should remain appendix/extension material. It does not
replace the single-turn mainline, and it should mention that InternVL2 context was
implemented through the available runtime serialization rather than as a new canonical
main experiment.

## F. Case-Level Failure Taxonomy

The taxonomy is data-driven from actual Phase 2 outputs and audit labels. The most
important categories for discussion are prompt-following flips, color-pair
concentration, visual-clarity flagged cases, and format/factor-induced shifts.
Multi-turn-induced and source/style-sensitive cases are useful extension categories,
but they should not become new headline claims.

## G. Gatekeeping Protocol

The gatekeeping protocol turns the study into a controlled diagnostic pipeline:
dataset validity, C0 visual fidelity, parser reliability, source robustness, visual
clarity, wording boundary, format/factorization diagnostics, and multi-turn extension.
It helps organize Methods, Results, and Appendix, but it should not be marketed as a
general-purpose benchmark.

## Final Conclusion Boundary

Phase 2 does not overturn the original main conclusion. It makes the conclusion more
precise: visual evidence dominates under neutral prompting and most original conflict
conditions; LLaVA shows a limited but significant original C3/C4 conflict-following
effect; that effect is concentrated in specific color pairs and is sensitive to wording,
format, and prompt factor. Qwen and InternVL2 are stable in the original single-turn
template family, but Phase 2 shows that particular factorized or multi-turn settings
can induce large shifts, so no global robustness claim is allowed.
