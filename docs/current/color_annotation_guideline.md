# Color Annotation Guideline

## Scope

This guideline defines how to assign ground-truth labels for the v2 Stanford Cars car-color experiment.

The target of annotation is:

- the `main car`
- the `primary body color`
- based on the cropped image and, when needed, the original image for context

The annotation task is not about trim, lights, windows, wheels, background objects, or secondary accents.

## Label Set

Annotators must use exactly one coarse label when the image is usable:

- `white`
- `black`
- `gray`
- `silver`
- `red`
- `blue`
- `green`
- `yellow`
- `other`

## What Counts as the True Color

Use the dominant body-panel color of the main car.

Include:

- hood
- doors
- roof
- trunk
- main side panels

Ignore:

- headlights and taillights
- license plates
- stickers and decals
- mirrors when they are a different accent color
- windows and windshield reflections
- tires and rims
- background objects

## When to Exclude a Sample

Mark `include_in_primary_main_analysis = no` when any of the following applies:

- the main car is not clearly the dominant object
- the crop does not show enough of the body
- the body is genuinely two-tone or multi-color
- strong glare or reflection makes the body color unreliable
- heavy shadow makes the color indeterminate
- the visible color changes substantially across the body
- the annotator cannot confidently map the body color to the coarse label set

If the vehicle is clearly outside the coarse label set but still usable, use `other` rather than excluding it. In the paper-facing primary analysis, `other` is still excluded later, but it should remain annotated in the audit trail.

## Files Used in V2

The audit and annotation workflow centers on:

- `data/processed/stanford_cars/final_analysis_manifest.csv`
- `annotation/annotator_a.csv`
- `annotation/annotator_b.csv`
- `annotation/adjudication_template.csv`
- `annotation/reviewed_truth_current.csv`
- `annotation/provisional_truth_current.csv`

## Annotator Workflow

### Annotator A and Annotator B

Each annotator fills one file:

- `annotation/annotator_a.csv`
- `annotation/annotator_b.csv`

Required columns include:

- `image_id`
- `cropped_path`
- `true_color`
- `annotator_label`
- `include_in_primary_main_analysis`
- `annotation_status`
- `notes`

### Adjudication

Use:

- `annotation/adjudication_template.csv`

Resolve disagreements and then update the reviewed truth table that the pipeline reads by default:

- `annotation/reviewed_truth_current.csv`

The config also reserves:

- `annotation/adjudicated_truth.csv`

for future paper-final adjudicated truth.

## Recommended Decision Rules

- Prefer the body color that occupies the largest visible area.
- If silver versus gray is unclear, use the overall metallic appearance as the tie-breaker.
- Downstream evaluation now treats `faithful` strictly as `parsed_label == true_color`; ambiguous `white/gray/silver` cases should therefore be excluded from the formal primary analysis rather than handled by a compatibility rule.
- If red is desaturated toward maroon or burgundy, keep `red` unless the sample is too unstable for formal analysis.
- If yellow is very gold or beige-like and cannot be assigned confidently, use `other` or exclude.
- If a very dark, low-saturation `blue` or `green` car repeatedly looks gray, blue, black, or green across reviewers or models, prefer exclusion from the formal primary set over forcing a fragile hue label.
- If the crop is borderline but still interpretable, keep the row in the audit manifest and explain the decision in `notes`.
- If a sample has a plausible alternate acceptable label, flag it for adjudication rather than silently forcing a confident label.

## Provisional, Reviewed, and Adjudicated Truth

- `provisional` mode uses the current automated or lightly reviewed table in `annotation/provisional_truth_current.csv`.
- `reviewed` mode uses `annotation/reviewed_truth_current.csv`.
- `adjudicated` truth is reserved for the future final table after double annotation and disagreement resolution.

Only reviewed or adjudicated truth should be used for paper-ready claims, and adjudicated truth should be preferred once available.
