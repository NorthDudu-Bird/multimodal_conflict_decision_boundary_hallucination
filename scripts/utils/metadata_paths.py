#!/usr/bin/env python
"""Shared metadata paths and compatibility helpers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SAMPLES_DIR = METADATA_DIR / "samples"
PROMPTS_DIR = METADATA_DIR / "prompts"
OUTPUTS_RAW_DIR = METADATA_DIR / "outputs_raw"
OUTPUTS_LABELED_DIR = METADATA_DIR / "outputs_labeled"
ANALYSIS_DIR = METADATA_DIR / "analysis"
ARCHIVES_DIR = ROOT / "archives"
LEGACY_PREVIOUS_DESIGN_DIR = ARCHIVES_DIR / "legacy_previous_design"
LEGACY_STANFORD_CARS_DIR = LEGACY_PREVIOUS_DESIGN_DIR / "car_color_stanford_clean_s0_s7" / "metadata"

LEGACY_NO_DOG_ALL_CSV = METADATA_DIR / "no_dog_all.csv"
LEGACY_NO_DOG_FILTERED_CSV = METADATA_DIR / "no_dog_filtered_candidates.csv"
LEGACY_NO_DOG_SAMPLE_50_CSV = METADATA_DIR / "no_dog_sample_50.csv"
LEGACY_NO_DOG_STRESS_SUBSET_10_CSV = METADATA_DIR / "no_dog_stress_subset_10.csv"
LEGACY_BASELINE_PROMPTS_CSV = METADATA_DIR / "no_dog_sample_50_prompt_levels.csv"
LEGACY_STRESS_PROMPTS_CSV = METADATA_DIR / "no_dog_stress_conflict_prompt_levels.csv"
LEGACY_SMOKE_RAW_CSV = METADATA_DIR / "qwen2vl_smoke_test_results.csv"
LEGACY_BASELINE_RUNTIME_CSV = METADATA_DIR / "qwen2vl_7b_full_results_runtime.csv"
LEGACY_BASELINE_RAW_CSV = METADATA_DIR / "qwen2vl_7b_full_results_raw.csv"
LEGACY_BASELINE_PRELABELED_CSV = METADATA_DIR / "qwen2vl_7b_full_results_prelabeled.csv"
LEGACY_BASELINE_MANUAL_REVIEW_CSV = METADATA_DIR / "qwen2vl_7b_manual_review_priority.csv"
LEGACY_BASELINE_FINAL_LABELED_CSV = METADATA_DIR / "qwen2vl_7b_labeled_template.csv"

NO_DOG_ALL_CSV = SAMPLES_DIR / "no_dog_all.csv"
NO_DOG_FILTERED_CSV = SAMPLES_DIR / "no_dog_filtered_candidates.csv"
NO_DOG_SAMPLE_50_CSV = SAMPLES_DIR / "no_dog_sample_50.csv"
NO_DOG_STRESS_SUBSET_10_CSV = SAMPLES_DIR / "no_dog_stress_subset_10.csv"
BASELINE_PROMPTS_CSV = PROMPTS_DIR / "baseline_existence_conflict_50x4.csv"
BASELINE_PROMPTS_EN_CSV = PROMPTS_DIR / "baseline_existence_conflict_50x4_en.csv"
STRESS_PROMPTS_CSV = PROMPTS_DIR / "stress_existence_conflict_10x4.csv"
SMOKE_RAW_CSV = OUTPUTS_RAW_DIR / "qwen2vl7b_smoke_raw.csv"
BASELINE_RUNTIME_CSV = OUTPUTS_RAW_DIR / "qwen2vl7b_baseline_runtime.csv"
BASELINE_RAW_CSV = OUTPUTS_RAW_DIR / "qwen2vl7b_baseline_raw.csv"
STRESS_RAW_CSV = OUTPUTS_RAW_DIR / "qwen2vl7b_stress_raw.csv"
BASELINE_PRELABELED_CSV = OUTPUTS_LABELED_DIR / "qwen2vl7b_baseline_prelabeled.csv"
BASELINE_MANUAL_REVIEW_CSV = OUTPUTS_LABELED_DIR / "qwen2vl7b_baseline_manual_review.csv"
BASELINE_FINAL_LABELED_CSV = OUTPUTS_LABELED_DIR / "qwen2vl7b_baseline_final_labeled.csv"
STANFORD_CARS_SAMPLE_CSV = LEGACY_STANFORD_CARS_DIR / "samples" / "car_color_stanford_clean_sample_30.csv"
STANFORD_CARS_REVIEW_CSV = LEGACY_STANFORD_CARS_DIR / "samples" / "car_color_stanford_clean_review.csv"
STANFORD_CARS_PROMPTS_CSV = LEGACY_STANFORD_CARS_DIR / "prompts" / "car_color_attribute_conflict_stanford_clean_s0_s7_30x8.csv"
STANFORD_CARS_SMOKE_RUNTIME_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_raw" / "qwen2vl7b_stanford_clean_smoke_runtime.csv"
STANFORD_CARS_RUNTIME_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_raw" / "qwen2vl7b_stanford_clean_runtime.csv"
STANFORD_CARS_RAW_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_raw" / "qwen2vl7b_stanford_clean_raw.csv"
STANFORD_CARS_PRELABELED_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_labeled" / "qwen2vl7b_stanford_clean_prelabeled.csv"
STANFORD_CARS_MANUAL_REVIEW_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_labeled" / "qwen2vl7b_stanford_clean_manual_review.csv"
STANFORD_CARS_FINAL_LABELED_CSV = LEGACY_STANFORD_CARS_DIR / "outputs_labeled" / "qwen2vl7b_stanford_clean_final_labeled.csv"
STANFORD_CARS_AUTOLABEL_SUMMARY_MD = LEGACY_STANFORD_CARS_DIR / "analysis" / "car_color_stanford_clean_autolabel_summary.md"
STANFORD_CARS_SANITY_JSON = LEGACY_STANFORD_CARS_DIR / "analysis" / "car_color_stanford_clean_sanity_check.json"
STANFORD_CARS_RUN_SUMMARY_MD = LEGACY_STANFORD_CARS_DIR / "analysis" / "car_color_stanford_clean_run_summary.md"


def ensure_metadata_dirs() -> None:
    for path in [METADATA_DIR, SAMPLES_DIR, PROMPTS_DIR, OUTPUTS_RAW_DIR, OUTPUTS_LABELED_DIR, ANALYSIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
