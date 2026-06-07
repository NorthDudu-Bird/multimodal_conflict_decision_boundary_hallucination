#!/usr/bin/env python
"""Prepare Stanford Cars assets for the restructured prompt-mechanism comparison study."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.bootstrap.generate_car_color_stanford_clean_table import (
    choose_kept_rows,
    ensure_raw_dataset,
    export_clean_crops,
    load_records,
    score_records,
    select_candidate_records,
)
from scripts.utils.restructured_experiment_utils import (
    AUXILIARY_CONDITIONS,
    PRIMARY_CONDITIONS,
    build_logger,
    canonicalize_color,
    condition_lookup,
    conflict_color_for,
    ensure_dirs,
    expected_output_map,
    expected_output_space,
    json_dumps,
    load_config,
    prompt_text_for,
    read_rows,
    relative_str,
    repo_path,
    write_rows,
)


ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stanford Cars clean subset, annotation templates, and restructured prompt tables.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--truth-mode", choices=["provisional", "reviewed"], default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--refresh-clean-subset", action="store_true")
    return parser.parse_args()


def manifest_fieldnames() -> list[str]:
    return [
        "image_id",
        "split",
        "class_id",
        "class_name",
        "file_name",
        "source_image_path",
        "original_path",
        "cropped_path",
        "width",
        "height",
        "original_width",
        "original_height",
        "bbox",
        "bbox_xywh",
        "bbox_area_ratio",
        "bbox_width",
        "bbox_height",
        "aspect_ratio",
        "crop_box",
        "crop_width",
        "crop_height",
        "crop_fill_ratio",
        "target_short_edge",
        "cropped_width",
        "cropped_height",
        "true_color",
        "estimated_color",
        "preliminary_color_guess",
        "color_confidence",
        "needs_manual_review",
        "foreground_dominant_share",
        "background_complexity",
        "quality_score",
        "keep",
        "drop",
        "keep_or_drop",
        "keep_reason",
        "passed_clean_filters",
        "filter_fail_reasons",
        "crop_source",
        "mask_source",
        "focus_pixel_count",
        "score_summary",
        "notes",
        "manual_exclusion_reason",
        "duplicate_group_id",
        "duplicate_canonical_image_id",
    ]


def subset_fieldnames() -> list[str]:
    return [
        "image_id",
        "split",
        "class_id",
        "class_name",
        "file_name",
        "original_path",
        "cropped_path",
        "width",
        "height",
        "quality_score",
        "color_confidence",
        "background_complexity",
        "foreground_dominant_share",
        "preliminary_color_guess",
        "selection_rank",
        "selection_bucket",
        "selection_notes",
        "split_or_subset",
    ]


def truth_fieldnames() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "provisional_true_color",
        "acceptable_true_colors",
        "truth_mode",
        "include_in_analysis",
        "review_notes",
    ]


def annotation_fieldnames() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "provisional_true_color",
        "annotator_label",
        "include_in_formal_analysis",
        "annotation_status",
        "notes",
    ]


def reviewed_truth_template_fields() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "reviewed_true_color",
        "acceptable_true_colors",
        "include_in_formal_analysis",
        "review_status",
        "review_notes",
    ]


def manual_exclusion_fieldnames() -> list[str]:
    return ["image_id", "exclude_from_subset", "reason", "notes"]


def prompt_fieldnames() -> list[str]:
    return [
        "sample_id",
        "image_id",
        "file_name",
        "image_path",
        "original_image_path",
        "width",
        "height",
        "experiment_type",
        "dataset_name",
        "target_object",
        "attribute_type",
        "prompt_level",
        "prompt_code",
        "condition_family",
        "condition_name",
        "condition_index",
        "truth_mode",
        "split_or_subset",
        "true_color",
        "acceptable_true_colors",
        "conflict_color",
        "prompt_text",
        "expected_output_space",
        "expected_output_map",
        "model_name",
        "model_output",
        "label",
        "language_consistent",
        "vision_consistent",
        "ambiguous",
        "notes",
    ]


def average_hash(image_path: Path, size: int = 16) -> int:
    image = Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(image.getdata())
    mean_value = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel >= mean_value else 0)
    return bits


def difference_hash(image_path: Path, size: int = 16) -> int:
    image = Image.open(image_path).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = list(image.getdata())
    bits = 0
    for y in range(size):
        row_start = y * (size + 1)
        for x in range(size):
            left = pixels[row_start + x]
            right = pixels[row_start + x + 1]
            bits = (bits << 1) | (1 if left >= right else 0)
    return bits


def load_manual_exclusions(config: dict) -> dict[str, dict[str, str]]:
    manual_csv = repo_path(config["annotation"]["manual_exclusion_csv"])
    if not manual_csv.exists():
        return {}
    exclusions: dict[str, dict[str, str]] = {}
    for row in read_rows(manual_csv):
        if str(row.get("exclude_from_subset", "")).strip().lower() not in {"1", "true", "yes", "y"}:
            continue
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        exclusions[image_id] = row
    return exclusions


def annotate_duplicate_clusters(rows: list[dict]) -> list[dict]:
    kept_rows = [row for row in rows if str(row.get("keep", "0")) == "1" and str(row.get("cropped_path", "")).strip()]
    by_signature: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in kept_rows:
        image_path = repo_path(row["cropped_path"])
        if not image_path.exists():
            continue
        signature = (average_hash(image_path), difference_hash(image_path))
        by_signature[signature].append(row)

    cluster_index = 0
    for cluster_rows in by_signature.values():
        if len(cluster_rows) <= 1:
            continue
        cluster_index += 1
        cluster_rows.sort(
            key=lambda item: (
                float(item.get("quality_score", 0.0) or 0.0),
                float(item.get("color_confidence", 0.0) or 0.0),
                str(item.get("image_id", "")),
            ),
            reverse=True,
        )
        canonical_image_id = cluster_rows[0]["image_id"]
        cluster_id = f"dup_cluster_{cluster_index:03d}"
        for row in cluster_rows:
            row["duplicate_group_id"] = cluster_id
            row["duplicate_canonical_image_id"] = canonical_image_id
    return rows


def load_or_build_manifest(config: dict, logger, refresh_clean_subset: bool, skip_download: bool) -> list[dict]:
    dataset_cfg = config["dataset"]
    manifest_csv = repo_path(dataset_cfg["clean_subset_manifest_csv"])
    clean_crops_dir = repo_path(dataset_cfg["clean_crops_dir"])

    if manifest_csv.exists() and not refresh_clean_subset:
        rows = read_rows(manifest_csv)
        keep_count = sum(1 for row in rows if str(row.get("keep", "0")) == "1")
        if keep_count >= int(dataset_cfg["clean_subset_size"]):
            logger.info("Reusing existing clean-subset manifest with %s kept rows: %s", keep_count, relative_str(manifest_csv))
            return annotate_duplicate_clusters(augment_manifest_rows(rows))

    logger.info("Rebuilding clean subset manifest and crops from Stanford Cars raw data.")
    ensure_raw_dataset(logger=logger, skip_download=skip_download)
    records = load_records()
    candidate_records = select_candidate_records(records, candidate_pool_size=min(int(dataset_cfg["candidate_pool_size"]), len(records)))
    all_rows = score_records(
        candidate_records,
        target_short_edge=int(dataset_cfg["target_short_edge"]),
        num_workers=int(dataset_cfg["num_workers"]),
        logger=logger,
    )
    kept_ids = set(choose_kept_rows(all_rows, clean_subset_size=int(dataset_cfg["clean_subset_size"])))
    kept_rows = [row for row in all_rows if row["image_id"] in kept_ids]
    export_clean_crops(kept_rows, target_short_edge=int(dataset_cfg["target_short_edge"]), logger=logger)

    clean_crops_dir.mkdir(parents=True, exist_ok=True)
    augmented_rows = annotate_duplicate_clusters(augment_manifest_rows(all_rows, kept_ids=kept_ids))
    write_rows(manifest_csv, manifest_fieldnames(), augmented_rows)
    logger.info("Wrote refreshed manifest: %s", relative_str(manifest_csv))
    return augmented_rows


def augment_manifest_rows(rows: list[dict], kept_ids: set[str] | None = None) -> list[dict]:
    augmented_rows: list[dict] = []
    for row in rows:
        current = dict(row)
        keep_flag = str(current.get("keep", ""))
        drop_flag = str(current.get("drop", ""))
        if kept_ids is not None:
            keep_flag = "1" if current["image_id"] in kept_ids else "0"
            drop_flag = "0" if keep_flag == "1" else "1"
        keep_or_drop = "keep" if keep_flag == "1" else "drop"
        current["keep"] = keep_flag
        current["drop"] = drop_flag
        current["keep_or_drop"] = current.get("keep_or_drop", keep_or_drop) or keep_or_drop
        preliminary_color = canonicalize_color(str(current.get("estimated_color", "") or current.get("true_color", "")))
        current["preliminary_color_guess"] = preliminary_color
        current["true_color"] = preliminary_color
        current.setdefault("original_path", current.get("source_image_path", ""))
        current.setdefault("manual_exclusion_reason", "")
        current.setdefault("duplicate_group_id", "")
        current.setdefault("duplicate_canonical_image_id", "")
        augmented_rows.append(current)
    return augmented_rows


def sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("quality_score", 0.0) or 0.0),
        float(row.get("color_confidence", 0.0) or 0.0),
        float(row.get("foreground_dominant_share", 0.0) or 0.0),
    )


def select_main_subset(manifest_rows: list[dict], subset_size: int, subset_name: str, manual_exclusions: dict[str, dict[str, str]]) -> list[dict]:
    kept_rows = [row for row in manifest_rows if str(row.get("keep", "0")) == "1" and str(row.get("cropped_path", "")).strip()]
    candidate_rows: list[dict] = []
    for row in kept_rows:
        image_id = row["image_id"]
        canonical_image_id = str(row.get("duplicate_canonical_image_id", "")).strip()
        if canonical_image_id and canonical_image_id != image_id:
            row["manual_exclusion_reason"] = row.get("manual_exclusion_reason", "") or f"duplicate_of:{canonical_image_id}"
            continue
        if image_id in manual_exclusions:
            row["manual_exclusion_reason"] = manual_exclusions[image_id].get("reason", "manual_exclusion")
            continue
        candidate_rows.append(row)
    if len(candidate_rows) < subset_size:
        raise RuntimeError(f"Only found {len(candidate_rows)} eligible rows after exclusions, fewer than requested subset_size={subset_size}.")

    by_color: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        by_color[row["preliminary_color_guess"]].append(row)
    for color_rows in by_color.values():
        color_rows.sort(key=sort_key, reverse=True)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    available_colors = sorted(by_color)
    guaranteed_quota: dict[str, int] = {}
    for color in available_colors:
        available = len(by_color[color])
        if available <= 0:
            continue
        if color == "silver":
            guaranteed_quota[color] = 1
        else:
            guaranteed_quota[color] = min(2, available)

    for color in available_colors:
        target = guaranteed_quota.get(color, 0)
        for rank in range(target):
            candidate = by_color[color][rank]
            if candidate["image_id"] in selected_ids:
                continue
            enriched = dict(candidate)
            enriched["selection_bucket"] = f"guaranteed_{color}"
            selected.append(enriched)
            selected_ids.add(candidate["image_id"])

    remaining_slots = max(0, subset_size - len(selected))
    if remaining_slots > 0:
        remainder: list[dict] = []
        for color in available_colors:
            quota = guaranteed_quota.get(color, 0)
            remainder.extend(by_color[color][quota:])
        remainder.sort(key=sort_key, reverse=True)
        for row in remainder[:remaining_slots]:
            if row["image_id"] in selected_ids:
                continue
            enriched = dict(row)
            enriched["selection_bucket"] = "quality_ranked_fill"
            selected.append(enriched)
            selected_ids.add(row["image_id"])

    selected.sort(key=sort_key, reverse=True)
    for rank, row in enumerate(selected[:subset_size], start=1):
        row["selection_rank"] = rank
        row["selection_notes"] = (
            f"subset={subset_name}; color={row['preliminary_color_guess']}; "
            f"quality_score={row['quality_score']}; bucket={row['selection_bucket']}"
        )
        row["split_or_subset"] = subset_name
    return selected[:subset_size]


def build_subset_rows(selected_rows: list[dict], subset_name: str) -> list[dict]:
    rows: list[dict] = []
    for row in selected_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "file_name": row["file_name"],
                "original_path": row.get("original_path", row.get("source_image_path", "")),
                "cropped_path": row["cropped_path"],
                "width": row.get("cropped_width", row.get("width", "")),
                "height": row.get("cropped_height", row.get("height", "")),
                "quality_score": row["quality_score"],
                "color_confidence": row["color_confidence"],
                "background_complexity": row["background_complexity"],
                "foreground_dominant_share": row["foreground_dominant_share"],
                "preliminary_color_guess": row["preliminary_color_guess"],
                "selection_rank": row["selection_rank"],
                "selection_bucket": row["selection_bucket"],
                "selection_notes": row["selection_notes"],
                "split_or_subset": subset_name,
            }
        )
    return rows


def build_truth_rows(subset_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in subset_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "provisional_true_color": row["preliminary_color_guess"],
                "acceptable_true_colors": "",
                "truth_mode": "provisional",
                "include_in_analysis": "1",
                "review_notes": "Seeded from preliminary_color_guess; replace in reviewed mode after dual annotation.",
            }
        )
    return rows


def build_annotation_rows(subset_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in subset_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "provisional_true_color": row["preliminary_color_guess"],
                "annotator_label": "",
                "include_in_formal_analysis": "",
                "annotation_status": "",
                "notes": "",
            }
        )
    return rows


def build_reviewed_truth_template_rows(subset_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in subset_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "reviewed_true_color": "",
                "acceptable_true_colors": "",
                "include_in_formal_analysis": "",
                "review_status": "",
                "review_notes": "",
            }
        )
    return rows


def build_adjudication_rows(subset_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in subset_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "provisional_true_color": row["preliminary_color_guess"],
                "annotator_a_label": "",
                "annotator_b_label": "",
                "adjudicated_label": "",
                "include_in_formal_analysis": "",
                "adjudication_status": "",
                "notes": "",
            }
        )
    return rows


def load_truth_rows(config: dict, subset_rows: list[dict], truth_mode: str) -> list[dict]:
    annotation_cfg = config["annotation"]
    provisional_truth_csv = repo_path(annotation_cfg["provisional_truth_csv"])
    reviewed_truth_csv = repo_path(annotation_cfg["reviewed_truth_csv"])
    subset_by_id = {row["image_id"]: row for row in subset_rows}

    if truth_mode == "provisional":
        return [row for row in read_rows(provisional_truth_csv) if row["image_id"] in subset_by_id]

    if not reviewed_truth_csv.exists():
        raise FileNotFoundError(
            f"Reviewed truth mode requested, but reviewed truth file does not exist: {relative_str(reviewed_truth_csv)}"
        )

    loaded_rows = read_rows(reviewed_truth_csv)
    truth_rows: list[dict] = []
    for row in loaded_rows:
        image_id = row.get("image_id", "")
        if image_id not in subset_by_id:
            continue
        reviewed_true_color = canonicalize_color(row.get("reviewed_true_color", ""))
        include_flag = str(row.get("include_in_formal_analysis", "")).strip().lower()
        if not reviewed_true_color or reviewed_true_color == "other" and not row.get("reviewed_true_color", "").strip():
            continue
        if include_flag not in {"1", "true", "yes", "y"}:
            continue
        truth_rows.append(
            {
                "image_id": image_id,
                "cropped_path": row.get("cropped_path", subset_by_id[image_id]["cropped_path"]),
                "original_path": row.get("original_path", subset_by_id[image_id]["original_path"]),
                "provisional_true_color": reviewed_true_color,
                "acceptable_true_colors": row.get("acceptable_true_colors", ""),
                "truth_mode": "reviewed",
                "include_in_analysis": "1",
                "review_notes": row.get("review_notes", ""),
            }
        )
    return truth_rows


def build_prompt_rows(subset_rows: list[dict], truth_rows: list[dict], truth_mode: str, subset_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    subset_by_id = {row["image_id"]: row for row in subset_rows}
    condition_by_name = condition_lookup()
    all_rows: list[dict] = []
    primary_rows: list[dict] = []
    auxiliary_rows: list[dict] = []

    for truth_row in truth_rows:
        image_id = truth_row["image_id"]
        subset_row = subset_by_id[image_id]
        true_color = canonicalize_color(truth_row["provisional_true_color"])
        conflict_color = conflict_color_for(true_color)
        acceptable_true_colors = str(truth_row.get("acceptable_true_colors", "") or "").strip()
        for condition in PRIMARY_CONDITIONS + AUXILIARY_CONDITIONS:
            condition_name = condition["condition_name"]
            prompt_row = {
                "sample_id": f"{image_id}_{condition_name}",
                "image_id": image_id,
                "file_name": subset_row["file_name"],
                "image_path": subset_row["cropped_path"],
                "original_image_path": subset_row["original_path"],
                "width": subset_row["width"],
                "height": subset_row["height"],
                "experiment_type": "car_color_prompt_mechanism_comparison",
                "dataset_name": "stanford_cars_clean",
                "target_object": "main_car",
                "attribute_type": "primary_body_color",
                "prompt_level": condition_name,
                "prompt_code": condition_name,
                "condition_family": condition["condition_family"],
                "condition_name": condition_name,
                "condition_index": condition["condition_index"],
                "truth_mode": truth_mode,
                "split_or_subset": subset_name,
                "true_color": true_color,
                "acceptable_true_colors": acceptable_true_colors,
                "conflict_color": conflict_color,
                "prompt_text": prompt_text_for(condition_name, conflict_color),
                "expected_output_space": json_dumps(expected_output_space(condition_name, conflict_color)),
                "expected_output_map": json_dumps(expected_output_map(condition_name, conflict_color)),
                "model_name": "",
                "model_output": "",
                "label": "",
                "language_consistent": "",
                "vision_consistent": "",
                "ambiguous": "",
                "notes": condition_by_name[condition_name]["description"],
            }
            all_rows.append(prompt_row)
            if condition["condition_family"] == "primary":
                primary_rows.append(prompt_row)
            else:
                auxiliary_rows.append(prompt_row)
    return all_rows, primary_rows, auxiliary_rows


def write_summary(config: dict, manifest_rows: list[dict], subset_rows: list[dict], truth_rows: list[dict], prompt_rows: list[dict], primary_rows: list[dict], auxiliary_rows: list[dict], truth_mode: str) -> None:
    dataset_cfg = config["dataset"]
    annotation_cfg = config["annotation"]
    metadata_cfg = config["metadata"]
    summary_dir = repo_path(metadata_cfg["summaries_dir"])
    ensure_dirs([summary_dir])

    color_counts = Counter(row["preliminary_color_guess"] for row in subset_rows)
    summary = {
        "manifest_rows": len(manifest_rows),
        "clean_subset_count": sum(1 for row in manifest_rows if str(row.get("keep", "0")) == "1"),
        "manual_exclusion_count": sum(1 for row in manifest_rows if str(row.get("manual_exclusion_reason", "")).strip()),
        "duplicate_noncanonical_count": sum(
            1
            for row in manifest_rows
            if str(row.get("duplicate_canonical_image_id", "")).strip()
            and str(row.get("duplicate_canonical_image_id", "")).strip() != str(row.get("image_id", "")).strip()
        ),
        "main_subset_count": len(subset_rows),
        "truth_mode": truth_mode,
        "truth_row_count": len(truth_rows),
        "primary_prompt_count": len(primary_rows),
        "auxiliary_prompt_count": len(auxiliary_rows),
        "total_prompt_count": len(prompt_rows),
        "main_subset_color_distribution": dict(sorted(color_counts.items())),
    }
    summary_json = summary_dir / f"stanford_cars_restructured_prepare_{truth_mode}.json"
    summary_md = summary_dir / f"stanford_cars_restructured_prepare_{truth_mode}.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    processed_summary = {
        "clean_subset_count": summary["clean_subset_count"],
        "manual_exclusion_count": summary["manual_exclusion_count"],
        "duplicate_noncanonical_count": summary["duplicate_noncanonical_count"],
        "main_subset_count": summary["main_subset_count"],
        "truth_mode": truth_mode,
        "manifest_csv": relative_str(repo_path(dataset_cfg["clean_subset_manifest_csv"])),
        "main_subset_csv": relative_str(repo_path(dataset_cfg["main_subset_csv"])),
        "annotation_dir": relative_str(repo_path(annotation_cfg["dir"])),
        "prompt_tables": {
            "all_conditions": relative_str(repo_path(metadata_cfg["prompts_dir"]) / f"stanford_cars_restructured_{truth_mode}_all_conditions.csv"),
            "primary_conditions": relative_str(repo_path(metadata_cfg["prompts_dir"]) / f"stanford_cars_restructured_{truth_mode}_primary_conditions.csv"),
            "auxiliary_conditions": relative_str(repo_path(metadata_cfg["prompts_dir"]) / f"stanford_cars_restructured_{truth_mode}_auxiliary_conditions.csv"),
        },
        "summary_files": {
            "prepare_json": relative_str(summary_json),
            "prepare_md": relative_str(summary_md),
        },
        "main_subset_color_distribution": dict(sorted(color_counts.items())),
    }
    repo_path(dataset_cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)
    (repo_path(dataset_cfg["processed_dir"]) / "clean_subset_summary.json").write_text(
        json.dumps(processed_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Stanford Cars Restructured Preparation Summary",
        "",
        f"- truth_mode: {truth_mode}",
        f"- clean_subset_count: {summary['clean_subset_count']}",
        f"- manual_exclusion_count: {summary['manual_exclusion_count']}",
        f"- duplicate_noncanonical_count: {summary['duplicate_noncanonical_count']}",
        f"- main_subset_count: {summary['main_subset_count']}",
        f"- primary_prompt_count: {summary['primary_prompt_count']}",
        f"- auxiliary_prompt_count: {summary['auxiliary_prompt_count']}",
        "",
        "## Main Subset Color Distribution",
    ]
    for color, count in sorted(color_counts.items()):
        lines.append(f"- {color}: {count}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    truth_mode = args.truth_mode or config["annotation"]["truth_mode"]

    logs_dir = ROOT / "logs"
    ensure_dirs([logs_dir])
    logger = build_logger("prepare_stanford_cars_restructured", logs_dir / f"prepare_stanford_cars_restructured_{truth_mode}.log")
    logger.info("Preparing Stanford Cars restructured assets with truth_mode=%s", truth_mode)

    dataset_cfg = config["dataset"]
    annotation_cfg = config["annotation"]
    metadata_cfg = config["metadata"]
    subset_name = dataset_cfg["subset_name"]
    manual_exclusions = load_manual_exclusions(config)

    ensure_dirs(
        [
            repo_path(dataset_cfg["processed_dir"]),
            repo_path(annotation_cfg["dir"]),
            repo_path(metadata_cfg["prompts_dir"]),
            repo_path(metadata_cfg["summaries_dir"]),
        ]
    )

    manifest_rows = load_or_build_manifest(
        config=config,
        logger=logger,
        refresh_clean_subset=args.refresh_clean_subset,
        skip_download=args.skip_download,
    )
    write_rows(repo_path(dataset_cfg["clean_subset_manifest_csv"]), manifest_fieldnames(), manifest_rows)
    subset_rows = build_subset_rows(
        select_main_subset(
            manifest_rows,
            subset_size=int(dataset_cfg["main_subset_size"]),
            subset_name=subset_name,
            manual_exclusions=manual_exclusions,
        ),
        subset_name=subset_name,
    )

    subset_csv = repo_path(dataset_cfg["main_subset_csv"])
    provisional_truth_csv = repo_path(annotation_cfg["provisional_truth_csv"])
    reviewed_truth_template_csv = repo_path(annotation_cfg["reviewed_truth_template_csv"])
    annotator_a_csv = repo_path(annotation_cfg["annotator_a_csv"])
    annotator_b_csv = repo_path(annotation_cfg["annotator_b_csv"])
    adjudication_template_csv = repo_path(annotation_cfg["adjudication_template_csv"])

    write_rows(subset_csv, subset_fieldnames(), subset_rows)
    write_rows(provisional_truth_csv, truth_fieldnames(), build_truth_rows(subset_rows))
    write_rows(annotator_a_csv, annotation_fieldnames(), build_annotation_rows(subset_rows))
    write_rows(annotator_b_csv, annotation_fieldnames(), build_annotation_rows(subset_rows))
    write_rows(reviewed_truth_template_csv, reviewed_truth_template_fields(), build_reviewed_truth_template_rows(subset_rows))
    write_rows(
        adjudication_template_csv,
        [
            "image_id",
            "cropped_path",
            "original_path",
            "provisional_true_color",
            "annotator_a_label",
            "annotator_b_label",
            "adjudicated_label",
            "include_in_formal_analysis",
            "adjudication_status",
            "notes",
        ],
        build_adjudication_rows(subset_rows),
    )

    truth_rows = load_truth_rows(config=config, subset_rows=subset_rows, truth_mode=truth_mode)
    if not truth_rows:
        raise RuntimeError(f"No truth rows available for truth_mode={truth_mode}.")
    prompt_rows, primary_rows, auxiliary_rows = build_prompt_rows(
        subset_rows=subset_rows,
        truth_rows=truth_rows,
        truth_mode=truth_mode,
        subset_name=subset_name,
    )

    prompts_dir = repo_path(metadata_cfg["prompts_dir"])
    all_prompt_csv = prompts_dir / f"stanford_cars_restructured_{truth_mode}_all_conditions.csv"
    primary_prompt_csv = prompts_dir / f"stanford_cars_restructured_{truth_mode}_primary_conditions.csv"
    auxiliary_prompt_csv = prompts_dir / f"stanford_cars_restructured_{truth_mode}_auxiliary_conditions.csv"

    write_rows(all_prompt_csv, prompt_fieldnames(), prompt_rows)
    write_rows(primary_prompt_csv, prompt_fieldnames(), primary_rows)
    write_rows(auxiliary_prompt_csv, prompt_fieldnames(), auxiliary_rows)
    write_summary(
        config=config,
        manifest_rows=manifest_rows,
        subset_rows=subset_rows,
        truth_rows=truth_rows,
        prompt_rows=prompt_rows,
        primary_rows=primary_rows,
        auxiliary_rows=auxiliary_rows,
        truth_mode=truth_mode,
    )

    result = {
        "manifest_csv": relative_str(repo_path(dataset_cfg["clean_subset_manifest_csv"])),
        "subset_csv": relative_str(subset_csv),
        "provisional_truth_csv": relative_str(provisional_truth_csv),
        "reviewed_truth_template_csv": relative_str(reviewed_truth_template_csv),
        "annotator_a_csv": relative_str(annotator_a_csv),
        "annotator_b_csv": relative_str(annotator_b_csv),
        "adjudication_template_csv": relative_str(adjudication_template_csv),
        "all_prompt_csv": relative_str(all_prompt_csv),
        "primary_prompt_csv": relative_str(primary_prompt_csv),
        "auxiliary_prompt_csv": relative_str(auxiliary_prompt_csv),
        "truth_mode": truth_mode,
        "subset_count": len(subset_rows),
        "primary_prompt_count": len(primary_rows),
        "auxiliary_prompt_count": len(auxiliary_rows),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
