# Results And Discussion Summary

## Primary Evaluation

The primary evaluation starts from a strong visual baseline. On the 300-image balanced
set, all three models are faithful under C0. This supports the interpretation that the
task is visually tractable under neutral prompting.

Under the C0-C4 conflict conditions, Qwen2-VL-7B-Instruct and InternVL2-8B remain
essentially visually consistent in the primary template family. Qwen has one
conflict-following output in C3 and one in C4; InternVL2 has none across C0-C4.
LLaVA-1.5-7B shows the only clear primary shift: C3 is `27/300 = 9.00%`, and C4 is
`10/300 = 3.33%`.

Same-image paired flips strengthen this interpretation. Because every conflict prompt
is paired with the same model's C0 answer to the same image, and because C0 is fully
faithful, the LLaVA C3/C4 outputs can be described as image-level transitions from
faithful visual answers to the false prompt color.

## Auxiliary Diagnostics

A1/A2 should be read as auxiliary diagnostics. They show that restricted answer spaces
and counterfactual-assumption wording can induce high compliance, but they do not replace
the primary C0-C4 evidence chain.

## Robustness And Controlled Diagnostics

C3 wording robustness constrains the main claim. LLaVA drops from canonical C3 `27/300`
to C3-v2 `5/300` and C3-v3 `0/300`. The result should therefore be written as limited
and wording-sensitive, not as stable across equivalent prompts.

The per-color split further narrows the interpretation. The LLaVA C3 shift is heavily
concentrated in `white -> black` (`20/27`), with smaller `black -> white` and
`blue -> red` contributions. C4 shows the same achromatic concentration at lower
magnitude. This prevents a uniform color-effect interpretation.

Answer-format control and prompt factorization show that misleading text is not a single
monolithic intervention. The canonical C3 format is stronger for LLaVA than free,
multiple-choice, or yes/no probes. Factorized prompts reveal that title/prefix framing
and no-correction presupposition can be much stronger than quoted claims or indirect
hints, including for Qwen and InternVL2. These results are controlled diagnostics, not a
new prompt-engineering story.

## Validity Checks

Parser audit, source-stratified sanity checks, reproducibility audit, and visual clarity
audit support the reliability of the primary interpretation. The visual audit suggests
that target flip rows are usually inspectable, but confound flags are more common among
target rows than controls. This belongs in limitations: image difficulty is reduced as a
global explanation but not fully eliminated as a local factor.

## Extension Diagnostics

The multi-turn persuasive setting tests a different interaction regime. LLaVA and Qwen
remain near zero, while InternVL2 becomes highly susceptible under repeated previous-turn
false context. This is best written as an extension diagnostic and should not be folded
into the primary single-turn result.

The failure taxonomy helps organize representative cases for Discussion. It should be
used to explain the observed patterns, not to propose a new theory.

## Discussion Boundary

The final discussion should emphasize a narrow empirical finding: in a controlled
car-body color task, visual evidence dominates under neutral prompting and most primary
conflict settings, while LLaVA shows a limited same-image conflict-following shift under
canonical C3/C4. The controlled diagnostics show that this shift is model-, wording-,
format-, factor-, and color-pair-sensitive. This does not justify a general claim that
VLMs prioritize language over vision.
