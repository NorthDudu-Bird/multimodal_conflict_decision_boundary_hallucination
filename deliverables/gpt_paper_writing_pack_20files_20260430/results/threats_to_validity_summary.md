# Threats To Validity Summary

## Threat 1: Parser-Induced Inflation

Parser rules could inflate conflict-aligned counts if color aliases or embedded color words are mapped too aggressively.

Current evidence reduces this risk. The main C0-C4 outputs remain in the base single-label regime, with no parse errors, refusals, or other-wrong outputs. The parser audit in `results/parser/label_mapping_audit.md` shows that ambiguous alias handling is only relevant to auxiliary outputs, not to the main conflict-aligned counts.

Remaining boundary: future prompts that elicit phrases such as "metallic blue" or composite color descriptions would need additional parser review. In this paper, parser behavior is a threat-control issue rather than a new experimental factor.

## Threat 2: Source-Specific Driving Effect

The final balanced set combines StanfordCars and VCoR examples, so a reviewer could ask whether the effect is driven by one source.

The source-stratified appendix check in `results/appendix/stanford_core_sanity_check.md` reduces this concern. Under `C0`, all three models remain faithful in both sources. Under `C3`, the LLaVA-1.5-7B direction is consistent across StanfordCars and VCoR, although the magnitude differs.

Remaining boundary: source is not randomized as an experimental factor, and the paper should not overinterpret source differences. This check supports robustness of direction, not a new source-comparison claim.

## Threat 3: Non-Reproducible Artifacts

Empirical conclusions are weak if the reported tables and summaries cannot be reproduced from locked artifacts.

The reproducibility audit in `results/reproducibility_audit.md` reports that all checked canonical manifests, prompts, parsed outputs, metrics, summaries, parser-audit files, and appendix files matched the locked snapshot. The new paired, prompt-boundary, and visual-clarity outputs are derived from existing canonical parsed outputs and do not call model inference.

Remaining boundary: raw runtime logs and timing metadata remain non-canonical and are not part of the reproducibility gate. This is acceptable because the paper claims are based on parsed outputs and metrics, not on runtime behavior.

## Threat 4: Prompt-Template Specificity

The observed LLaVA shift could depend on one exact prompt wording.

The prompt-boundary module in `results/robustness/prompt_boundary_summary.md` directly supports this limitation. LLaVA-1.5-7B drops from Original C3 `27/300` to C3-v2 `5/300` and C3-v3 `0/300`; the new variants no longer show Holm-significant LLaVA-vs-stable-model differences.

Remaining boundary: the paper should treat wording sensitivity as a claim limiter. It should not present the original C3 effect as a stable cross-template phenomenon.

## Threat 5: Image-Level Validity And Visual Clarity

Conflict-aligned cases might be visually harder, affected by reflections, background color, occlusion, or multiple-car interference.

The new visual clarity audit infrastructure in `results/audit/visual_clarity_audit_readme.md` and `results/audit/visual_clarity_audit_manifest.csv` addresses this as a reviewable threat. It extracts all 27 LLaVA original C3 conflict-aligned cases and 27 matched faithful controls, preserving image paths, source, true color, false prompt color, and model output.

Remaining boundary: the manifest is infrastructure for human review, not a completed human-rating result. It should be cited as threat-reduction support and appendix material, not as a new main experiment unless manually completed later.

## Overall Boundary

These controls support a narrow empirical conclusion: the task is visually faithful under C0; LLaVA-1.5-7B shows limited significant conflict-aligned behavior under the original strong misleading C3/C4 templates; Qwen2-VL-7B-Instruct and InternVL2-8B remain largely visually consistent; and the LLaVA effect is wording-sensitive. They do not support a broad claim about general VLM language dominance.
