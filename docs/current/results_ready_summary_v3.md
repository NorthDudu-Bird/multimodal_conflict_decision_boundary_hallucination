# Results-Ready Summary: V3 Expanded Strict-Colors Rerun

## Final V3 Dataset

The previous strict-colors formal analysis used 98 included images. V3 now removes ten manually flagged ambiguous cases, leaving 93 images before expansion, and then expands the clean subset to 140 included images without relaxing the cleaning standard. The final included color distribution is:

- `red = 56`
- `blue = 21`
- `green = 2`
- `yellow = 7`
- `black = 29`
- `white = 25`

This did not fully reach the aspirational 150-200 range, but it did pass the "do not lower the clean standard just to add samples" requirement. The most constrained classes remained `green` and `yellow`.

Per model, the refreshed v3 rerun contains:

- `700` primary rows (`140 images x 5 conditions`)
- `280` auxiliary rows (`140 images x 2 conditions`)

## Primary: Open-Answer Language Bias

The main open-answer result remains sharply model-dependent. Qwen2-VL-7B and InternVL2-8B stayed highly stable under all five primary conditions, with zero conflict-aligned outputs in every condition on the expanded v3 set. Their primary results are therefore dominated by faithful responses plus a small number of other-wrong labels:

- `Qwen2-VL-7B`: HR = `0.0000` in all primary conditions; faithful rate `1.0000` throughout
- `InternVL2-8B`: HR = `0.0000` in all primary conditions; faithful rate `1.0000` in every primary condition

LLaVA-1.5-7B remains the only model showing a clear open-answer vulnerability to misleading framing. The strongest effect again appears in `C3_presupposition_correction_allowed`, where LLaVA reaches:

- `HR = 0.1286` with 95% CI `[0.0780, 0.1956]`
- faithful rate `0.8714`

Its other primary conflict-aligned rates are much smaller:

- `C0_neutral = 0.0000`
- `C1_weak_suggestion = 0.0071`
- `C2_false_assertion_open = 0.0071`
- `C4_stronger_open_conflict = 0.0429`

So the primary conclusion remains narrow and stable: open-answer conflict alignment is not a general property of all tested models, and the only clear v3 open-answer effect is concentrated in LLaVA under presuppositional framing.

## Auxiliary: Strong Framing / Frame Following

The auxiliary probe remains much stronger than the primary open-answer setting. Here the models are pushed into a more constrained answer format or explicit false-frame setup, and conflict alignment rises sharply:

- `Qwen2-VL-7B`
  - `A1_forced_choice_red_family`: HR `0.4786`
  - `A2_counterfactual_assumption`: HR `0.8000`
- `LLaVA-1.5-7B`
  - `A1_forced_choice_red_family`: HR `0.9714`
  - `A2_counterfactual_assumption`: HR `1.0000`
- `InternVL2-8B`
  - `A1_forced_choice_red_family`: HR `0.5929`
  - `A2_counterfactual_assumption`: HR `1.0000`

This keeps the auxiliary interpretation clear: under stronger linguistic framing, all three models can be pulled toward the false premise, but the degree of compliance varies substantially by model and by auxiliary condition.

## Answer-Space Compliance

V3 adds an auxiliary-only answer-space compliance analysis to clarify whether models are following the auxiliary prompt frame or escaping it. The key rates are:

- `Qwen2-VL-7B`
  - `A1` compliance = `0.4786`
  - `A2` compliance = `0.8000`
- `LLaVA-1.5-7B`
  - `A1` compliance = `0.9714`
  - `A2` compliance = `1.0000`
- `InternVL2-8B`
  - `A1` compliance = `0.5929`
  - `A2` compliance = `1.0000`

The compliance tables show a very clean behavioral split in this rerun:

- in-space responses are overwhelmingly `conflict_aligned`
- out-of-space responses are overwhelmingly `faithful`
- no model produced a meaningful residual "other behavior" tail in the auxiliary runs

This strengthens the interpretation that auxiliary is measuring framework obedience under a strong prompt scaffold, rather than the same mechanism as the open-answer primary task.

## Which Models Look Most Stable

Under the primary open-answer setting:

- most stable: `InternVL2-8B`
- also highly stable: `Qwen2-VL-7B`
- least stable: `LLaVA-1.5-7B`, especially in `C3`

Under the stronger auxiliary framing:

- most frame-compliant: `LLaVA-1.5-7B`
- also strongly frame-compliant in `A2`: `InternVL2-8B`
- most resistant, but still clearly affected: `Qwen2-VL-7B`

## Recommended Result Files

The highest-value v3 result files to cite first are:

- `analysis/current/primary_v3/analysis_summary.md`
- `analysis/current/auxiliary_v3/analysis_summary.md`
- `analysis/current/auxiliary_v3/answer_space_compliance_metrics.csv`
- `analysis/current/auxiliary_v3/answer_space_behavior_breakdown.csv`
- `analysis/current/cross_model_v3/README.md`
- `analysis/current/color_distribution_v4_expanded.md`

For image-level inspection, use the current preview entry:

- `reports/current/strict_colors_multimodel_results_viewer.html`
