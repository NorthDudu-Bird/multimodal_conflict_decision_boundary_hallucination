# Validity And Scope Audit Summary

## Scope Retained

- The task remains primary car-body color recognition.
- The dataset remains the 300-image six-color balanced evaluation set.
- The model set remains LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, and InternVL2-8B.
- The primary evaluation remains C0-C4 with same-image paired interpretation.
- A1/A2 remain auxiliary diagnostics.

## Scope Narrowed

- The LLaVA C3/C4 finding is not written as a general VLM language-bias result.
- The C3 finding is not written as stable across wordings.
- The 9.00% LLaVA C3 result is not written as a uniform color effect.
- Qwen and InternVL2 are not described as globally robust to all misleading-text forms.
- Multi-turn results are not used as the primary evidence chain.

## Validity Checks

- Parser audit supports that primary results are not driven by broad alias mapping.
- Source-stratified sanity checks reduce the concern that the finding is a single-source artifact.
- Visual clarity audit reduces the "images are unreadable" alternative explanation but
  leaves local reflection, lighting, background, and multi-car confounds as limitations.
- Reproducibility audit confirms the stability of locked canonical result tables and
  figures, with only writing-facing summary text differing from the snapshot.

## Final Writing Instruction

Write the study as a controlled empirical evaluation with a primary evidence chain and
supporting diagnostics. Do not flatten all modules into equal main claims.
