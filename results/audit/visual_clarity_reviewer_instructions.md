# Visual Clarity Audit Reviewer Instructions

## Purpose

Review whether conflict-flip images are visually valid examples for primary body-color judgment, and whether obvious visual confounds are more common in flip cases than in matched faithful controls.

## Rating Fields

- `audit_visual_clarity`: `clear`, `moderate`, or `low`.
- `audit_body_color_salience`: `high`, `medium`, or `low`.
- `audit_specular_reflection`: `none_minor`, `moderate`, or `strong`.
- `audit_shadow_or_night_effect`: `none_minor`, `moderate`, or `strong`.
- `audit_background_color_bias`: `none_minor`, `moderate`, or `strong`.
- `audit_multi_car_interference`: `none`, `minor`, or `present`.
- `audit_occlusion`: `none`, `minor`, or `moderate`.
- `audit_notes`: short free-text note; mention if the image is reviewable but not ideal.

## Use In Paper

Use this as a threat-reduction audit. It should not become a new main experiment unless independently reviewed by additional annotators.
