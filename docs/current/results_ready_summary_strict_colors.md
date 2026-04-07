# Results-Ready Summary: Strict Colors

## Final Strict Dataset

The strict-colors rerun started from the current 120-image Stanford Cars main subset and retained 98 images for the formal main analysis. Nineteen rows were excluded because their reviewed true color was `silver`, `gray`, or `other`, and three additional dark low-saturation boundary cases remained excluded. The retained color distribution was 40 red, 27 black, 14 white, 13 blue, 2 green, and 2 yellow. This produced 490 primary prompt rows and 196 auxiliary prompt rows per model.

## Primary Conditions

The primary open-answer pattern remained strongly model-dependent. Qwen2-VL-7B and InternVL2-8B again showed zero conflict-aligned outputs in all five primary conditions on the strict subset. LLaVA-1.5-7B remained the only model with clear prompt sensitivity under open-answer conditions, and the strongest effect again occurred in `C3_presupposition_correction_allowed`. On the strict subset, LLaVA reached 14 conflict-aligned outputs out of 98 rows in `C3` (`HR = 0.1429`, 95% CI `[0.0804, 0.2281]`). LLaVA's other primary rates were `C1 = 0.0102`, `C2 = 0.0102`, and `C4 = 0.0408`, while `C0` remained at zero. The corresponding `LDI` for LLaVA was `0.0595` with bootstrap interval `[0.0289, 0.0935]`.

Under the strict exact-match rule, no model produced any `parse_error` or `refusal_or_correction` rows in the formal primary analysis. Across all three models and all five primary conditions combined, the strict primary run contained 1,429 faithful rows, 20 conflict-aligned rows, and 21 other-wrong rows. No `gray` or `silver` predictions appeared in the strict rerun outputs, so the exact-match faithful rule did not require any edge-case handling at runtime.

## Auxiliary Conditions

The auxiliary manipulations remained much stronger than the open-answer primary conditions. In `A1_forced_choice_red_family`, hallucination rates were `0.4796` for Qwen2-VL-7B, `0.9694` for LLaVA-1.5-7B, and `0.5714` for InternVL2-8B. In `A2_counterfactual_assumption`, Qwen2-VL-7B reached `0.8061`, while both LLaVA-1.5-7B and InternVL2-8B reached `1.0000`. The resulting auxiliary sensitivity statistic `FSS = HR(A1) - HR(C2)` was `0.4796` for Qwen2-VL-7B, `0.9592` for LLaVA-1.5-7B, and `0.5714` for InternVL2-8B.

## Cross-Model Interpretation

The cross-model interpretation did not collapse after removing ambiguous color classes. The main qualitative conclusion became, if anything, sharper: open-answer conflict alignment is not a universal behavior across models, but when it does appear, it remains concentrated in LLaVA-1.5-7B and especially under presuppositional framing. The primary mixed-effects logistic model still converged and showed the strongest interaction for `LLaVA × C3`, with estimated odds ratio approximately `20.79` (95% interval from the fitted coefficient: `[10.42, 41.47]`) relative to the Qwen baseline framing.

## Change Relative to the Earlier Reviewed-Truth V2 Run

Relative to the earlier reviewed-truth `v2` run, the strict-colors rerun reduced the primary dataset from 104 images to 98 and removed all `silver`, `gray`, and `other` rows from the formal main analysis. This change also removed the need for the previous achromatic compatibility interpretation and made `faithful` an exact-match category only.

The substantive conclusions were stable in two ways. First, Qwen2-VL-7B and InternVL2-8B still showed zero primary hallucination rates across all five open-answer conditions. Second, LLaVA-1.5-7B remained the only model with a clear open-answer vulnerability. In fact, the LLaVA `C3` effect became larger after strict filtering, increasing from `0.0769` in the earlier 104-image run to `0.1429` in the strict 98-image run. The auxiliary results changed only slightly in magnitude: Qwen2-VL-7B and InternVL2-8B decreased modestly in `A1`, LLaVA remained near ceiling in `A1`, and both LLaVA and InternVL2-8B remained at ceiling in `A2`.

## Files to Cite

The main strict-color result files are:

- `analysis/current/strict_colors_primary/model_condition_metrics.csv`
- `analysis/current/strict_colors_primary/summary_metrics.csv`
- `analysis/current/strict_colors_primary/analysis_summary.md`
- `analysis/current/strict_colors_auxiliary/model_condition_metrics.csv`
- `analysis/current/strict_colors_auxiliary/summary_metrics.csv`
- `analysis/current/strict_colors_cross_model/README.md`
