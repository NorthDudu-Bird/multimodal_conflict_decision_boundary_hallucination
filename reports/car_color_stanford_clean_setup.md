# Stanford Cars Clean Car-Color Experiment Setup

## 1. Pipeline Entry
- Data preparation script: `scripts/generate_car_color_stanford_clean_table.py`
- Inference entry: `scripts/run_qwen2vl_batch.py`
- Runtime merge: `scripts/export_qwen2vl_raw_results.py`
- Label summary: `scripts/auto_label_car_color_attribute_conflict_outputs.py`

## 2. What Changed
- The old COCO-derived car images are not touched.
- A new Stanford Cars raw mirror is downloaded into `data/raw/stanford_cars/`.
- A clean subset of bbox-guided main-car crops is created under `data/processed/stanford_cars/`.
- A smaller experiment sample is exported in the same CSV-driven format used by the current car-color pipeline.

## 3. Raw Dataset Source
- Hugging Face dataset mirror: `iharabukhouski/stanford_cars`
- Mirror target: `data/raw/stanford_cars`
- Note: the historical Stanford download URLs used by torchvision now return 404, so this run uses the mirror while preserving the original Stanford Cars folder structure and annotations.

## 4. Clean-Subset Heuristics
- bbox area ratio >= 0.10
- bbox width >= 90
- bbox height >= 60
- crop fill ratio >= 0.42
- color confidence >= 0.66
- background complexity <= 0.78
- aspect ratio in [0.85, 5.20]
- keep top `320` records after filtering, balanced by split when possible

## 5. Image Processing
- bbox-guided crop with padding via existing crop helper
- resize crops to short edge = `256`
- export clean crops to `data/processed/stanford_cars/clean_crops`

## 6. Experiment Compatibility
- Experiment sample size: `30`
- Prompt design: same S0-S7 conflict ladder, but wording now explicitly asks for the main car body color and to ignore the background
- Downstream interface preserved: sample CSV + prompt CSV + runtime CSV + labeling CSV

## 7. Counts
- Raw records scanned: **16185**
- Candidate pool scored with image heuristics: **1600**
- Clean subset kept: **320**
- Clean subset split counts: **{'test': 160, 'train': 160}**
- Experiment sample size: **30**
- Prompt rows: **240**
- Experiment color distribution: **{'black': 6, 'blue': 6, 'green': 2, 'orange': 2, 'red': 7, 'white': 6, 'yellow': 1}**
