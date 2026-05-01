# Color-Split Main Experiment Analysis

## Role

This analysis checks whether the main conflict-aligned effect is distributed across the six balanced true-color classes or driven by a small subset of color pairs. It is an attribution and boundary-control module, not a new color-perception paper.

## Key Findings

- LLaVA C3: 27 total flips; largest route is `white -> black` with 20/27 (74.07%).
- LLaVA C4: 10 total flips; largest route is `white -> black` with 8/10 (80.00%).
- C0 remains fully faithful in every true-color stratum for all three models.
- The LLaVA effect is therefore not evenly dispersed across the six colors. It is concentrated mainly in the achromatic `white -> black` route, with smaller contributions from `black -> white` and `blue -> red` under C3.
- This pattern supports a contracted interpretation: the phenomenon is best described as template sensitivity plus partial color-pair vulnerability, not a general color-task shift and not mainly a neighboring-hue confusion effect.

## LLaVA Focus: C0/C3/C4 By True Color

| condition | true_color | false_prompt_pair | n | faithful_n | faithful_rate | conflict_aligned_n | conflict_aligned_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | black | black->white | 50 | 50 | 100.00% | 0 | 0.00% |
| C0 | blue | blue->red | 50 | 50 | 100.00% | 0 | 0.00% |
| C0 | green | green->yellow | 50 | 50 | 100.00% | 0 | 0.00% |
| C0 | red | red->blue | 50 | 50 | 100.00% | 0 | 0.00% |
| C0 | white | white->black | 50 | 50 | 100.00% | 0 | 0.00% |
| C0 | yellow | yellow->red | 50 | 50 | 100.00% | 0 | 0.00% |
| C3 | black | black->white | 50 | 47 | 94.00% | 3 | 6.00% |
| C3 | blue | blue->red | 50 | 46 | 92.00% | 4 | 8.00% |
| C3 | green | green->yellow | 50 | 50 | 100.00% | 0 | 0.00% |
| C3 | red | red->blue | 50 | 50 | 100.00% | 0 | 0.00% |
| C3 | white | white->black | 50 | 30 | 60.00% | 20 | 40.00% |
| C3 | yellow | yellow->red | 50 | 50 | 100.00% | 0 | 0.00% |
| C4 | black | black->white | 50 | 48 | 96.00% | 2 | 4.00% |
| C4 | blue | blue->red | 50 | 50 | 100.00% | 0 | 0.00% |
| C4 | green | green->yellow | 50 | 50 | 100.00% | 0 | 0.00% |
| C4 | red | red->blue | 50 | 50 | 100.00% | 0 | 0.00% |
| C4 | white | white->black | 50 | 42 | 84.00% | 8 | 16.00% |
| C4 | yellow | yellow->red | 50 | 50 | 100.00% | 0 | 0.00% |

## LLaVA Paired Flips By True Color

### C3

| true_color | false_prompt_color | pair_family | n_pairs | faithful_to_conflict_aligned_n | conflict_following_rate | paired_discordant_current_only | paired_discordant_c0_only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| black | white | achromatic_black_white | 50 | 3 | 6.00% | 3 | 0 |
| blue | red | red_blue | 50 | 4 | 8.00% | 4 | 0 |
| green | yellow | green_yellow | 50 | 0 | 0.00% | 0 | 0 |
| red | blue | red_blue | 50 | 0 | 0.00% | 0 | 0 |
| white | black | achromatic_black_white | 50 | 20 | 40.00% | 20 | 0 |
| yellow | red | yellow_red | 50 | 0 | 0.00% | 0 | 0 |

### C4

| true_color | false_prompt_color | pair_family | n_pairs | faithful_to_conflict_aligned_n | conflict_following_rate | paired_discordant_current_only | paired_discordant_c0_only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| black | white | achromatic_black_white | 50 | 2 | 4.00% | 2 | 0 |
| blue | red | red_blue | 50 | 0 | 0.00% | 0 | 0 |
| green | yellow | green_yellow | 50 | 0 | 0.00% | 0 | 0 |
| red | blue | red_blue | 50 | 0 | 0.00% | 0 | 0 |
| white | black | achromatic_black_white | 50 | 8 | 16.00% | 8 | 0 |
| yellow | red | yellow_red | 50 | 0 | 0.00% | 0 | 0 |

## Pair-Family Summary

| condition | pair_family | n_pairs | faithful_to_conflict_aligned_n | conflict_following_rate |
| --- | --- | --- | --- | --- |
| C3 | achromatic_black_white | 100 | 23 | 23.00% |
| C3 | green_yellow | 50 | 0 | 0.00% |
| C3 | red_blue | 100 | 4 | 4.00% |
| C3 | yellow_red | 50 | 0 | 0.00% |
| C4 | achromatic_black_white | 100 | 10 | 10.00% |
| C4 | green_yellow | 50 | 0 | 0.00% |
| C4 | red_blue | 100 | 0 | 0.00% |
| C4 | yellow_red | 50 | 0 | 0.00% |

## Paper Boundary

The original 9% LLaVA C3 effect should not be written as a uniformly distributed susceptibility across all colors. The more accurate wording is that a limited same-image conflict-following shift appears under the original strong misleading templates, and the shift is strongly concentrated in specific true-color/false-color pairings, especially `white -> black`.
