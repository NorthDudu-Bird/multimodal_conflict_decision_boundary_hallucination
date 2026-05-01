# Multi-Turn Persuasion Plan

## Purpose

This appendix-style diagnostic tests whether short context accumulation changes the false-prompt effect. It must not replace the main C0-C4 evidence chain or turn the paper into a dialogue hallucination study.

## Design

All conditions use the same 300-image balanced set and the same three frozen models.

- `MT1_single_turn_false_note`: one user prompt contains a false color note and asks for the image-based primary body color.
- `MT2_two_turn_persuasion`: a previous user turn states the false color, then the final user turn asks the neutral color question.
- `MT3_three_turn_persuasion`: two previous context turns repeat/strengthen the false color, then the final user turn asks the same neutral color question.

The final question is held constant for MT2/MT3 to isolate context accumulation from final-question wording.

## Analysis

Report faithful and conflict-aligned rates by model and condition, plus paired tests from MT1 to MT2/MT3 and from C0 where applicable. Treat any increase as an extension diagnostic only.

