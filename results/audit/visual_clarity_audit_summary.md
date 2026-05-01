# Visual Clarity Audit Summary

## Scope

- Target conflict-flip rows reviewed: 42.
- Matched faithful-control rows reviewed: 42.
- Target rows include all LLaVA original C3 flips, all LLaVA original C4 flips, and all LLaVA C3-v2 remaining flips.
- Repeated images across prompt conditions are retained as separate target rows; visual-quality annotations are kept consistent by `image_id`.

## Target Modules

| audit_source_module | target_n |
| --- | --- |
| main_original_C3 | 27 |
| main_original_C4 | 10 |
| wording_variant_C3_v2 | 5 |

## Target Color Routes

| audit_source_module | true_color | false_prompt_color | target_n |
| --- | --- | --- | --- |
| main_original_C3 | black | white | 3 |
| main_original_C3 | blue | red | 4 |
| main_original_C3 | white | black | 20 |
| main_original_C4 | black | white | 2 |
| main_original_C4 | white | black | 8 |
| wording_variant_C3_v2 | black | white | 1 |
| wording_variant_C3_v2 | blue | red | 1 |
| wording_variant_C3_v2 | white | black | 3 |

## Review Field Distributions

- Target visual clarity: `clear`=38, `moderate`=4.
- Control visual clarity: `clear`=39, `moderate`=3.
- Target body-color salience: `high`=37, `medium`=5.
- Control body-color salience: `high`=39, `medium`=3.
- Target rows with any flagged visual confound: 11/42.
- Control rows with any flagged visual confound: 4/42.

## Interpretation

The completed audit should be read as a task-validity check. If the fields remain mostly `clear` and high-salience, the main LLaVA flips are less plausibly explained by globally unreadable images. If any moderate/strong confounds are present, they should be discussed as remaining image-level threats rather than hidden.

Current completed annotations do not by themselves create a new visual-difficulty result. They support the narrower claim that the reviewed flip cases are inspectable car-body color examples, while leaving room for residual effects from reflections, lighting, background color, or dataset style.
