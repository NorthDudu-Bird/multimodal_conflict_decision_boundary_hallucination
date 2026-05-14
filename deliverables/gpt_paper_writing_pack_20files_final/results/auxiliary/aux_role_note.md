# Auxiliary A1/A2 Role Note

`A1/A2` should be described as auxiliary diagnostics, not as the main evidence chain.

`A1_forced_choice_red_family` is best interpreted as an answer-space stress test. It restricts the available answers to the false color family and asks whether the model complies with that restricted space or resists it by producing the visually faithful color outside the offered set.

`A2_counterfactual_assumption` is best interpreted as a counterfactual-assumption or compliance stress test. It asks the model to operate under an assumed false color family, so high conflict-aligned or compliance rates are expected and should not be treated as the same phenomenon as open-answer C0-C4 conflict following.

## What A1/A2 Can Support

- They show that models can be pushed toward the false color family when answer space or counterfactual assumptions are made stronger.
- They diagnose model compliance under constrained or assumption-heavy prompts.
- They help explain why prompt format matters for this task.

## What A1/A2 Cannot Support

- They do not prove that false text overrides visual evidence under the main open-answer task.
- They should not be pooled with C0-C4 metrics.
- They should not be used as the primary evidence for LLaVA's original `C3/C4` shift.
- They do not justify a broad claim that the evaluated VLMs generally follow language over vision.

## Recommended Paper Placement

Mention A1/A2 after the main C0-C4 and wording-boundary results. Use them as auxiliary diagnostics showing that stronger prompt constraints can induce high compliance, while keeping the primary conclusion anchored to C0-C4 paired flips and C3 wording sensitivity.
