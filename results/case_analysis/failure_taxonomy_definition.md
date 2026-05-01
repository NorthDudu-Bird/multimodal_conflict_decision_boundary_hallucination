# Failure Taxonomy Definition

This taxonomy is descriptive and data-driven. Categories are diagnostic tags and may overlap; they are not proposed as a new theory of VLM behavior.

| category_id | definition | decision_rule | paper_use |
| --- | --- | --- | --- |
| prompt_following_flip | A same-image LLaVA C3/C4 row where C0 was faithful and the conflict prompt answer equals the false prompt color. | `model_key=llava15_7b`, condition in original C3/C4, `is_conflict_aligned=True`, paired C0 faithful. | Core case-level evidence for conditional conflict following. |
| color_pair_concentration | Prompt-following flips concentrated in a specific true-color/false-color route rather than spread across all six colors. | Tag prompt-following flips by `true_color -> conflict_color`, with achromatic black/white and `white->black` reported separately. | Limits the 9% result: not a uniform color-class effect. |
| visual_clarity_flagged | Flip/control rows where the completed audit notes moderate/strong reflection, shadow, background color, salience, occlusion, or multi-car interference. | Any completed audit field enters a moderate/strong/present or lower-salience state. | Threat-to-validity discussion, not a new main effect. |
| format_compliance_sensitive | Original C3 flip cases that do not remain conflict-aligned under matched free-answer, multiple-choice, and yes/no format controls. | LLaVA original C3 conflict row; same `image_id` is non-conflict in all three formal C3/false-claim probes. | Supports answer-format dependence of the original C3 effect. |
| multiturn_induced | Conflict-following rows appearing only after short previous-turn false context while the final question remains neutral. | InternVL2 `two_turn_persuasion` or `three_turn_persuasion` with `phase2_is_conflict_aligned=True`. | Appendix extension showing multi-turn vulnerability can be model-specific. |
| source_style_sensitive_candidate | A source-stratified difference in flip rate that is visible but not sufficient to make source a new main experimental factor. | Report LLaVA C3 conflict rates separately for StanfordCars and VCoR. | Appendix caveat: direction persists across sources, magnitude varies. |

## Counts

| category_id | sample_count | unique_image_count | scope | primary_role | note |
| --- | --- | --- | --- | --- | --- |
| prompt_following_flip | 37 | 28 | LLaVA original C3/C4 same-image flips | main evidence | Rows are same-image flips from faithful C0 to the false prompt color. |
| color_pair_concentration | 33 | 24 | Achromatic black/white route among LLaVA C3/C4 flips | boundary/attribution | 28 row-events are specifically white->black. |
| visual_clarity_flagged | 11 | 6 | Completed audit target rows with at least one visual confound flag | threat reduction | Target rows 11/42 flagged; controls 4/42 flagged. |
| format_compliance_sensitive | 20 | 20 | Original LLaVA C3 flips not reproduced by free C3, MC C3, or yes/no false-claim probes | format boundary | This is a format-sensitivity tag, not proof that format alone caused each individual flip. |
| multiturn_induced | 288 | 224 | InternVL2 MT2/MT3 conflict-following rows with neutral final question | extension diagnostic | Short multi-turn context creates a strong InternVL2-specific effect absent from its original C0-C4 rows. |
| source_style_sensitive_candidate | 27 | 27 | LLaVA C3 flips split by StanfordCars/VCoR | appendix threat check | StanfordCars: 13/93 (13.98%); VCoR: 14/207 (6.76%) |

## Boundary

The most important categories for the main story are `prompt_following_flip`, `color_pair_concentration`, and `format_compliance_sensitive`. `visual_clarity_flagged`, `multiturn_induced`, and `source_style_sensitive_candidate` are threat-reduction or appendix categories.
