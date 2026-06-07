# A1/A2 Auxiliary Experiment Summary

- qwen2vl7b | A1_forced_choice_red_family: faithful=44.33% [38.82%, 49.99%]; conflict_aligned=55.67% [50.01%, 61.18%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=55.67% [50.01%, 61.18%]
- qwen2vl7b | A2_counterfactual_assumption: faithful=9.33% [6.54%, 13.16%]; conflict_aligned=90.67% [86.84%, 93.46%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=90.67% [86.84%, 93.46%]
- llava15_7b | A1_forced_choice_red_family: faithful=14.67% [11.11%, 19.12%]; conflict_aligned=85.33% [80.88%, 88.89%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=85.33% [80.88%, 88.89%]
- llava15_7b | A2_counterfactual_assumption: faithful=0.00% [0.00%, 1.26%]; conflict_aligned=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=100.00% [98.74%, 100.00%]
- internvl2_8b | A1_forced_choice_red_family: faithful=26.33% [21.67%, 31.59%]; conflict_aligned=73.67% [68.41%, 78.33%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=73.67% [68.41%, 78.33%]
- internvl2_8b | A2_counterfactual_assumption: faithful=0.00% [0.00%, 1.26%]; conflict_aligned=100.00% [98.74%, 100.00%]; refusal=0.00% [0.00%, 1.26%]; n=300; compliance=100.00% [98.74%, 100.00%]

## Exact Paired Proportion Tests
- within_model_A2_vs_A1 | qwen2vl7b | conflict_aligned | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.3500, p_exact=0.000000, discordant=(106, 1)
- within_model_A2_vs_A1 | qwen2vl7b | answer_space_compliance | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.3500, p_exact=0.000000, discordant=(106, 1)
- within_model_A2_vs_A1 | llava15_7b | conflict_aligned | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.1467, p_exact=0.000000, discordant=(44, 0)
- within_model_A2_vs_A1 | llava15_7b | answer_space_compliance | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.1467, p_exact=0.000000, discordant=(44, 0)
- within_model_A2_vs_A1 | internvl2_8b | conflict_aligned | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.2633, p_exact=0.000000, discordant=(79, 0)
- within_model_A2_vs_A1 | internvl2_8b | answer_space_compliance | A2_counterfactual_assumption minus A1_forced_choice_red_family: diff=0.2633, p_exact=0.000000, discordant=(79, 0)
- cross_model_same_condition | conflict_aligned | condition=A1_forced_choice_red_family | qwen2vl7b minus llava15_7b: diff=-0.2967, p_exact=0.000000, discordant=(16, 105)
- cross_model_same_condition | answer_space_compliance | condition=A1_forced_choice_red_family | qwen2vl7b minus llava15_7b: diff=-0.2967, p_exact=0.000000, discordant=(16, 105)
- cross_model_same_condition | conflict_aligned | condition=A1_forced_choice_red_family | qwen2vl7b minus internvl2_8b: diff=-0.1800, p_exact=0.000001, discordant=(35, 89)
- cross_model_same_condition | answer_space_compliance | condition=A1_forced_choice_red_family | qwen2vl7b minus internvl2_8b: diff=-0.1800, p_exact=0.000001, discordant=(35, 89)
- cross_model_same_condition | conflict_aligned | condition=A1_forced_choice_red_family | llava15_7b minus internvl2_8b: diff=0.1167, p_exact=0.002045, discordant=(79, 44)
- cross_model_same_condition | answer_space_compliance | condition=A1_forced_choice_red_family | llava15_7b minus internvl2_8b: diff=0.1167, p_exact=0.002045, discordant=(79, 44)
- cross_model_same_condition | conflict_aligned | condition=A2_counterfactual_assumption | qwen2vl7b minus llava15_7b: diff=-0.0933, p_exact=0.000000, discordant=(0, 28)
- cross_model_same_condition | answer_space_compliance | condition=A2_counterfactual_assumption | qwen2vl7b minus llava15_7b: diff=-0.0933, p_exact=0.000000, discordant=(0, 28)
- cross_model_same_condition | conflict_aligned | condition=A2_counterfactual_assumption | qwen2vl7b minus internvl2_8b: diff=-0.0933, p_exact=0.000000, discordant=(0, 28)
- cross_model_same_condition | answer_space_compliance | condition=A2_counterfactual_assumption | qwen2vl7b minus internvl2_8b: diff=-0.0933, p_exact=0.000000, discordant=(0, 28)
- cross_model_same_condition | conflict_aligned | condition=A2_counterfactual_assumption | llava15_7b minus internvl2_8b: diff=0.0000, p_exact=1.000000, discordant=(0, 0)
- cross_model_same_condition | answer_space_compliance | condition=A2_counterfactual_assumption | llava15_7b minus internvl2_8b: diff=0.0000, p_exact=1.000000, discordant=(0, 0)
