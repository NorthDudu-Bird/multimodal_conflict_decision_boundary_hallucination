#!/usr/bin/env python
"""Prepare the current strict-colors Stanford Cars manifest, annotation tables, and prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import (
    build_logger,
    canonicalize_color,
    conflict_color_for,
    ensure_dirs,
    expected_output_map,
    expected_output_space,
    excluded_primary_labels,
    get_conditions,
    get_color_policy,
    json_dumps,
    load_config,
    normalize_bool,
    primary_main_analysis_labels,
    primary_main_analysis_label_set,
    prompt_text_for,
    read_rows,
    relative_str,
    repo_path,
    write_rows,
)


ROOT = REPO_ROOT
BOOTSTRAP_DIR = ROOT / "scripts" / "data_prep" / "bootstrap"

# These images repeatedly produced non-conflict alternative colors because the visible body color
# is very dark / low-saturation, making the coarse hue label unstable for the main analysis.
DARK_LOW_SATURATION_COLOR_EXCLUSION_IDS = {
    "test_03269": "dark_low_saturation_color_boundary_case",
    "train_01310": "dark_low_saturation_color_boundary_case",
    "train_02408": "dark_low_saturation_color_boundary_case",
}

# These images remain in the reviewed-truth set for now, but the refreshed strict-colors outputs show
# repeated residual non-conflict mistakes on them. Keep them analyzable while surfacing them
# as the highest-priority rows for the next round of human re-checking.
POST_REFRESH_REVIEWER_CHECK_IDS = {
    "test_01497": "persistent_silver_white_boundary_after_v2_refresh",
    "train_01446": "persistent_silver_white_boundary_after_v2_refresh",
    "test_04952": "residual_black_vehicle_misread_after_v2_refresh",
    "test_03234": "residual_black_vehicle_misread_after_v2_refresh",
    "train_01346": "residual_black_vehicle_misread_after_v2_refresh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the current strict-colors Stanford Cars manifests and prompt tables.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--truth-source",
        choices=["auto", "provisional", "reviewed", "adjudicated"],
        default="auto",
        help="Truth source for the current main analysis. 'auto' prefers adjudicated, then reviewed, then provisional.",
    )
    return parser.parse_args()


def final_manifest_fieldnames() -> list[str]:
    return [
        "image_id",
        "split",
        "class_id",
        "class_name",
        "file_name",
        "original_path",
        "cropped_path",
        "true_color",
        "conflict_color",
        "truth_source",
        "truth_status",
        "acceptable_true_colors",
        "review_status",
        "review_notes",
        "preliminary_color_guess",
        "selection_rank",
        "selection_bucket",
        "selection_tier",
        "quality_score",
        "color_group",
        "candidate_source",
        "prior_issue_flag",
        "reviewer_check_needed",
        "include_in_analysis",
        "include_in_primary_main_analysis",
        "include_in_v2_auxiliary_analysis",
        "exclusion_reason",
        "notes",
    ]


def excluded_fieldnames() -> list[str]:
    return final_manifest_fieldnames()


def ambiguous_excluded_fieldnames() -> list[str]:
    return [
        "image_id",
        "true_color",
        "exclusion_reason",
        "original_path",
        "cropped_path",
    ]


def prompt_fieldnames() -> list[str]:
    return [
        "sample_id",
        "image_id",
        "file_name",
        "image_path",
        "original_image_path",
        "experiment_type",
        "dataset_name",
        "target_object",
        "attribute_type",
        "truth_source",
        "true_color",
        "acceptable_true_colors",
        "conflict_color",
        "condition_family",
        "condition_name",
        "condition_index",
        "prompt_template_version",
        "prompt_text",
        "expected_output_space",
        "expected_output_map",
        "include_in_primary_main_analysis",
        "notes",
    ]


def annotation_fieldnames() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "current_truth",
        "current_truth_source",
        "acceptable_true_colors",
        "include_in_primary_main_analysis",
        "prior_issue_flag",
        "reviewer_check_needed",
        "annotator_label",
        "include_in_formal_analysis",
        "annotation_status",
        "notes",
    ]


def adjudication_fieldnames() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "current_truth",
        "current_truth_source",
        "acceptable_true_colors",
        "include_in_primary_main_analysis",
        "prior_issue_flag",
        "reviewer_check_needed",
        "annotator_a_label",
        "annotator_b_label",
        "adjudicated_label",
        "include_in_formal_analysis",
        "adjudication_status",
        "notes",
    ]


def current_truth_fieldnames() -> list[str]:
    return [
        "image_id",
        "cropped_path",
        "original_path",
        "current_truth",
        "truth_source",
        "acceptable_true_colors",
        "include_in_primary_main_analysis",
        "prior_issue_flag",
        "reviewer_check_needed",
        "notes",
    ]


def expanded_selection_enabled(config: dict) -> bool:
    return bool((config.get("expanded_selection") or {}).get("enabled", False))


def yes_no(value: object) -> str:
    return "yes" if normalize_bool(value) else "no"


def load_expanded_manual_review_rows(config: dict) -> list[dict[str, str]]:
    expanded_cfg = config.get("expanded_selection") or {}
    review_csv = expanded_cfg.get("manual_review_csv")
    if not review_csv:
        raise KeyError("expanded_selection.manual_review_csv is required when expanded_selection.enabled=true")
    rows = read_rows(repo_path(review_csv))
    if not rows:
        raise RuntimeError(f"No manual review rows were found in {review_csv}.")
    return rows


def load_reviewed_truth_override_rows(config: dict) -> list[dict[str, str]]:
    expanded_cfg = config.get("expanded_selection") or {}
    review_csv = expanded_cfg.get("current_reviewed_truth_csv")
    if not review_csv:
        return []
    review_path = repo_path(review_csv)
    if not review_path.exists():
        return []
    return read_rows(review_path)


def load_clean_subset_by_id(config: dict) -> dict[str, dict[str, str]]:
    rows = read_rows(repo_path(config["dataset"]["clean_subset_manifest_csv"]))
    return {row["image_id"]: row for row in rows}


def load_base_manifest_rows(config: dict) -> list[dict[str, str]]:
    expanded_cfg = config.get("expanded_selection") or {}
    base_manifest_csv = expanded_cfg.get("base_manifest_csv")
    if not base_manifest_csv:
        raise KeyError("expanded_selection.base_manifest_csv is required when expanded_selection.enabled=true")
    return read_rows(repo_path(base_manifest_csv))


def load_manual_subset_exclusions(config: dict) -> dict[str, dict[str, str]]:
    manual_csv = repo_path(config["annotation"]["manual_exclusion_csv"])
    if not manual_csv.exists():
        return {}
    rows = read_rows(manual_csv)
    return {
        row["image_id"]: row
        for row in rows
        if normalize_bool(row.get("exclude_from_subset", "0")) and row.get("image_id", "")
    }


def clean_subset_metric_note(row: dict[str, str]) -> str:
    return (
        f"quality_score={row.get('quality_score', '')}; "
        f"color_confidence={row.get('color_confidence', '')}; "
        f"background_complexity={row.get('background_complexity', '')}; "
        f"foreground_dominant_share={row.get('foreground_dominant_share', '')}"
    )


def build_expanded_base_row(base_row: dict[str, str]) -> dict[str, str]:
    row = dict(base_row)
    row["truth_status"] = row.get("truth_status", "reviewed_base_strict_colors")
    row["selection_tier"] = row.get("selection_tier", "base_reviewed_strict_colors")
    row["color_group"] = row.get("color_group", row.get("true_color", ""))
    row["candidate_source"] = row.get("candidate_source", "strict_colors_base_manifest")
    row["include_in_analysis"] = "yes"
    row["include_in_primary_main_analysis"] = "yes"
    row["include_in_v2_auxiliary_analysis"] = "yes"
    row["exclusion_reason"] = ""
    row["notes"] = str(row.get("notes", "") or "").strip()
    return row


def build_expanded_manual_exclusion_row(
    image_id: str,
    base_manifest_by_id: dict[str, dict[str, str]],
    clean_subset_by_id: dict[str, dict[str, str]],
    reviewed_truth_by_id: dict[str, dict[str, str]],
    exclusion_reason: str,
    config: dict,
) -> dict[str, str]:
    base_row = dict(base_manifest_by_id.get(image_id, {}))
    clean_row = clean_subset_by_id.get(image_id, {})
    reviewed_row = reviewed_truth_by_id.get(image_id, {})
    source_row = base_row or clean_row or reviewed_row
    true_color = canonicalize_color(
        reviewed_row.get("current_truth", "")
        or base_row.get("true_color", "")
        or clean_row.get("true_color", "")
    )
    conflict_color = ""
    if true_color and true_color in primary_main_analysis_label_set(config):
        conflict_color = conflict_color_for(true_color, config=config)
    note_parts = [
        str(base_row.get("notes", "") or "").strip(),
        str(reviewed_row.get("notes", "") or "").strip(),
        str(clean_row.get("notes", "") or "").strip(),
    ]
    return {
        "image_id": image_id,
        "split": source_row.get("split", ""),
        "class_id": source_row.get("class_id", ""),
        "class_name": source_row.get("class_name", ""),
        "file_name": source_row.get("file_name", f"{image_id}.jpg"),
        "original_path": source_row.get("original_path", ""),
        "cropped_path": source_row.get("cropped_path", ""),
        "true_color": true_color,
        "conflict_color": conflict_color,
        "truth_source": "manual_review_v3",
        "truth_status": "excluded_after_manual_review_v3",
        "acceptable_true_colors": reviewed_row.get("acceptable_true_colors", ""),
        "review_status": "manual_review_excluded_v3",
        "review_notes": reviewed_row.get("notes", ""),
        "preliminary_color_guess": clean_row.get("preliminary_color_guess", clean_row.get("true_color", "")),
        "selection_rank": "",
        "selection_bucket": "manual_review_excluded_v3",
        "selection_tier": "manual_review_excluded_v3",
        "quality_score": clean_row.get("quality_score", ""),
        "color_group": true_color,
        "candidate_source": "manual_review_v3",
        "prior_issue_flag": "yes" if normalize_bool(reviewed_row.get("prior_issue_flag", "0")) else "no",
        "reviewer_check_needed": "yes",
        "include_in_analysis": "no",
        "include_in_primary_main_analysis": "no",
        "include_in_v2_auxiliary_analysis": "no",
        "exclusion_reason": exclusion_reason,
        "notes": " | ".join(part for part in note_parts if part),
    }


def build_expanded_included_row(
    clean_row: dict[str, str],
    review_row: dict[str, str],
    selection_rank: int,
    config: dict,
) -> dict[str, str]:
    review_label = canonicalize_color(review_row.get("review_label", clean_row.get("true_color", "")))
    return {
        "image_id": clean_row["image_id"],
        "split": clean_row.get("split", ""),
        "class_id": clean_row.get("class_id", ""),
        "class_name": clean_row.get("class_name", ""),
        "file_name": clean_row.get("file_name", f"{clean_row['image_id']}.jpg"),
        "original_path": clean_row.get("original_path", clean_row.get("source_image_path", "")),
        "cropped_path": clean_row.get("cropped_path", ""),
        "true_color": review_label,
        "conflict_color": conflict_color_for(review_label, config=config),
        "truth_source": "manual_review_v3",
        "truth_status": "manual_expansion_include_v3",
        "acceptable_true_colors": "",
        "review_status": "manual_expansion_include_v3",
        "review_notes": review_row.get("review_reason", ""),
        "preliminary_color_guess": clean_row.get("preliminary_color_guess", clean_row.get("true_color", "")),
        "selection_rank": str(selection_rank),
        "selection_bucket": f"expanded_{review_label}",
        "selection_tier": review_row.get("notes", "manual_expansion_include_v3"),
        "quality_score": clean_row.get("quality_score", ""),
        "color_group": review_label,
        "candidate_source": "clean_subset_manual_review_v3",
        "prior_issue_flag": "no",
        "reviewer_check_needed": "no",
        "include_in_analysis": "yes",
        "include_in_primary_main_analysis": "yes",
        "include_in_v2_auxiliary_analysis": "yes",
        "exclusion_reason": "",
        "notes": " | ".join(
            part
            for part in [
                "manual_expansion_include_v3",
                review_row.get("review_reason", ""),
                clean_subset_metric_note(clean_row),
            ]
            if part
        ),
    }


def write_color_distribution_summary(config: dict, expanded_stats: dict[str, object]) -> None:
    dataset_cfg = config["dataset"]
    csv_path = repo_path(
        dataset_cfg.get("color_distribution_csv", "data/metadata/balanced_eval_set/legacy_color_distribution.csv")
    )
    md_path = repo_path(
        dataset_cfg.get("color_distribution_md", "data/metadata/balanced_eval_set/legacy_color_distribution.md")
    )
    ensure_dirs([csv_path.parent, md_path.parent])

    rows: list[dict[str, object]] = []
    color_order = primary_main_analysis_labels(config)
    stage_keys = [
        ("strict_colors_before_manual_v3_exclusion", expanded_stats["before_manual_counts"]),
        ("post_manual_v3_exclusion_pre_expansion", expanded_stats["after_manual_counts"]),
        ("expanded_final_v4", expanded_stats["final_counts"]),
    ]
    for stage_name, counts in stage_keys:
        for color in color_order:
            rows.append(
                {
                    "stage": stage_name,
                    "color_group": color,
                    "count": int(counts.get(color, 0)),
                }
            )

    rows.extend(
        [
            {"stage": "summary", "color_group": "excluded_manual_review_v3", "count": int(expanded_stats["manual_excluded_count"])},
            {"stage": "summary", "color_group": "expanded_added_count", "count": int(expanded_stats["expanded_added_count"])},
            {"stage": "summary", "color_group": "final_total", "count": int(expanded_stats["final_total"])},
        ]
    )
    write_rows(csv_path, ["stage", "color_group", "count"], rows)

    target_min = int((config.get("expanded_selection") or {}).get("target_total_min", 150))
    target_max = int((config.get("expanded_selection") or {}).get("target_total_max", 200))
    reached_target = target_min <= int(expanded_stats["final_total"]) <= target_max
    lines = [
        "# Expanded Strict-Color Distribution V4",
        "",
        f"- strict_colors_before_manual_v3_exclusion_total: {expanded_stats['before_manual_total']}",
        f"- post_manual_v3_exclusion_pre_expansion_total: {expanded_stats['after_manual_total']}",
        f"- expanded_final_total: {expanded_stats['final_total']}",
        f"- manual_review_excluded_count: {expanded_stats['manual_excluded_count']}",
        f"- expanded_added_count: {expanded_stats['expanded_added_count']}",
        f"- target_range_150_200_reached: {'yes' if reached_target else 'no'}",
        "",
        "## Color Counts",
    ]
    for color in color_order:
        lines.append(
            f"- {color}: before={expanded_stats['before_manual_counts'].get(color, 0)}, "
            f"post_manual={expanded_stats['after_manual_counts'].get(color, 0)}, "
            f"expanded_final={expanded_stats['final_counts'].get(color, 0)}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_expanded_final_manifest(config: dict, logger) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    expanded_cfg = config.get("expanded_selection") or {}
    base_manifest_rows = load_base_manifest_rows(config)
    base_manifest_by_id = {row["image_id"]: row for row in base_manifest_rows}
    clean_subset_by_id = load_clean_subset_by_id(config)
    reviewed_truth_rows = load_reviewed_truth_override_rows(config)
    reviewed_truth_by_id = {row["image_id"]: row for row in reviewed_truth_rows}
    manual_review_rows = load_expanded_manual_review_rows(config)
    manual_subset_exclusions = load_manual_subset_exclusions(config)
    manual_exclude_ids = {
        image_id
        for image_id in expanded_cfg.get("exclude_after_manual_review_ids", [])
        if image_id
    }
    review_excluded_ids = {
        row["image_id"]
        for row in reviewed_truth_rows
        if row.get("image_id", "") and not normalize_bool(row.get("include_in_primary_main_analysis", "0"))
    }
    additional_guard_ids = set(DARK_LOW_SATURATION_COLOR_EXCLUSION_IDS)

    included_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []

    before_manual_included = [
        row for row in base_manifest_rows if yes_no(row.get("include_in_primary_main_analysis", "0")) == "yes"
    ]
    before_manual_counts = Counter(row.get("true_color", "") for row in before_manual_included)

    guard_exclusion_ids = set(manual_subset_exclusions) | review_excluded_ids | additional_guard_ids
    base_included_rows = [
        row
        for row in before_manual_included
        if row["image_id"] not in manual_exclude_ids and row["image_id"] not in guard_exclusion_ids
    ]
    for row in base_included_rows:
        included_rows.append(build_expanded_base_row(row))

    manual_excluded_rows: list[dict[str, str]] = []
    for image_id in sorted(manual_exclude_ids):
        manual_row = build_expanded_manual_exclusion_row(
            image_id=image_id,
            base_manifest_by_id=base_manifest_by_id,
            clean_subset_by_id=clean_subset_by_id,
            reviewed_truth_by_id=reviewed_truth_by_id,
            exclusion_reason="ambiguous_after_manual_review",
            config=config,
        )
        manual_excluded_rows.append(manual_row)
        excluded_rows.append(dict(manual_row))

    include_review_rows = [
        row
        for row in manual_review_rows
        if row.get("decision", "").strip().lower() == "include" and row.get("image_id", "") not in manual_exclude_ids
    ]
    exclude_review_rows = [
        row
        for row in manual_review_rows
        if row.get("decision", "").strip().lower() == "exclude" and row.get("image_id", "") not in manual_exclude_ids
    ]

    added_rank = 0
    for review_row in include_review_rows:
        image_id = review_row.get("image_id", "")
        if not image_id:
            continue
        clean_row = clean_subset_by_id.get(image_id)
        if clean_row is None:
            raise KeyError(f"Manual include row references image_id={image_id}, but it is missing from clean_subset_manifest.csv")
        if image_id in guard_exclusion_ids:
            raise ValueError(f"Manual include row references image_id={image_id}, but that image is globally excluded by a guard rule.")
        added_rank += 1
        included_rows.append(build_expanded_included_row(clean_row=clean_row, review_row=review_row, selection_rank=added_rank, config=config))

    for review_row in exclude_review_rows:
        image_id = review_row.get("image_id", "")
        if not image_id:
            continue
        clean_row = clean_subset_by_id.get(image_id, {})
        review_label = canonicalize_color(review_row.get("review_label", clean_row.get("true_color", "")))
        conflict_color = ""
        if review_label and review_label in primary_main_analysis_label_set(config):
            conflict_color = conflict_color_for(review_label, config=config)
        excluded_rows.append(
            {
                "image_id": image_id,
                "split": clean_row.get("split", ""),
                "class_id": clean_row.get("class_id", ""),
                "class_name": clean_row.get("class_name", ""),
                "file_name": clean_row.get("file_name", f"{image_id}.jpg"),
                "original_path": clean_row.get("original_path", clean_row.get("source_image_path", "")),
                "cropped_path": clean_row.get("cropped_path", ""),
                "true_color": review_label,
                "conflict_color": conflict_color,
                "truth_source": "manual_review_v3",
                "truth_status": "manual_expansion_excluded_v3",
                "acceptable_true_colors": "",
                "review_status": "manual_expansion_excluded_v3",
                "review_notes": review_row.get("review_reason", ""),
                "preliminary_color_guess": clean_row.get("preliminary_color_guess", clean_row.get("true_color", "")),
                "selection_rank": "",
                "selection_bucket": "manual_expansion_excluded_v3",
                "selection_tier": review_row.get("notes", ""),
                "quality_score": clean_row.get("quality_score", ""),
                "color_group": review_label,
                "candidate_source": "clean_subset_manual_review_v3",
                "prior_issue_flag": "no",
                "reviewer_check_needed": "no",
                "include_in_analysis": "no",
                "include_in_primary_main_analysis": "no",
                "include_in_v2_auxiliary_analysis": "no",
                "exclusion_reason": review_row.get("review_reason", "manual_expansion_excluded_v3"),
                "notes": " | ".join(
                    part for part in [review_row.get("notes", ""), clean_subset_metric_note(clean_row) if clean_row else ""] if part
                ),
            }
        )

    included_rows.sort(key=lambda item: (item.get("color_group", ""), item["image_id"]))
    excluded_rows.sort(key=lambda item: item["image_id"])
    manifest_rows = list(included_rows) + list(excluded_rows)

    after_manual_counts = Counter(row.get("true_color", "") for row in base_included_rows)
    final_counts = Counter(row.get("true_color", "") for row in included_rows)
    stats = {
        "before_manual_total": len(before_manual_included),
        "before_manual_counts": dict(before_manual_counts),
        "after_manual_total": len(base_included_rows),
        "after_manual_counts": dict(after_manual_counts),
        "expanded_added_count": len(include_review_rows),
        "manual_excluded_count": len(manual_excluded_rows),
        "final_total": len(included_rows),
        "final_counts": dict(final_counts),
    }
    logger.info("Expanded manifest stats: %s", json_dumps(stats))
    return manifest_rows, excluded_rows, stats


def maybe_prepare_subset(config_path: Path | None, logger) -> None:
    subset_csv = repo_path(load_config(config_path)["dataset"]["main_subset_csv"])
    if subset_csv.exists():
        return

    logger.info("Main subset CSV missing; regenerating current restructured assets first.")
    command = [sys.executable, str(BOOTSTRAP_DIR / "prepare_stanford_cars_restructured.py")]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    subprocess.run(command, check=True, cwd=ROOT)


def choose_truth_source(config: dict, requested: str) -> tuple[str, Path]:
    annotation_cfg = config["annotation"]
    sources = {
        "provisional": repo_path(annotation_cfg["provisional_truth_csv"]),
        "reviewed": repo_path(annotation_cfg["reviewed_truth_csv"]),
        "adjudicated": repo_path(annotation_cfg["v2_adjudicated_truth_csv"]),
    }

    if requested != "auto":
        return requested, sources[requested]

    for source_name in ["adjudicated", "reviewed", "provisional"]:
        if sources[source_name].exists():
            return source_name, sources[source_name]
    raise FileNotFoundError("No available truth table was found for provisional/reviewed/adjudicated modes.")


def load_current_truth_rows(config: dict, truth_source: str) -> list[dict[str, str]]:
    subset_rows = read_rows(repo_path(config["dataset"]["main_subset_csv"]))
    subset_by_id = {row["image_id"]: row for row in subset_rows}

    if truth_source == "provisional":
        truth_rows = read_rows(repo_path(config["annotation"]["provisional_truth_csv"]))
        current_rows: list[dict[str, str]] = []
        for row in truth_rows:
            image_id = row.get("image_id", "")
            if image_id not in subset_by_id:
                continue
            current_rows.append(
                {
                    "image_id": image_id,
                    "cropped_path": row.get("cropped_path", subset_by_id[image_id]["cropped_path"]),
                    "original_path": row.get("original_path", subset_by_id[image_id]["original_path"]),
                    "current_truth": canonicalize_color(row.get("provisional_true_color", "")),
                    "acceptable_true_colors": row.get("acceptable_true_colors", ""),
                    "include_in_formal_analysis": row.get("include_in_analysis", "1"),
                    "review_status": "provisional_auto_seed",
                    "review_notes": row.get("review_notes", ""),
                }
            )
        return current_rows

    if truth_source == "reviewed":
        truth_rows = read_rows(repo_path(config["annotation"]["reviewed_truth_csv"]))
        current_rows = []
        for row in truth_rows:
            image_id = row.get("image_id", "")
            if image_id not in subset_by_id:
                continue
            current_rows.append(
                {
                    "image_id": image_id,
                    "cropped_path": row.get("cropped_path", subset_by_id[image_id]["cropped_path"]),
                    "original_path": row.get("original_path", subset_by_id[image_id]["original_path"]),
                    "current_truth": canonicalize_color(row.get("reviewed_true_color", "")),
                    "acceptable_true_colors": row.get("acceptable_true_colors", ""),
                    "include_in_formal_analysis": row.get("include_in_formal_analysis", "1"),
                    "review_status": row.get("review_status", ""),
                    "review_notes": row.get("review_notes", ""),
                }
            )
        return current_rows

    truth_rows = read_rows(repo_path(config["annotation"]["v2_adjudicated_truth_csv"]))
    current_rows = []
    for row in truth_rows:
        image_id = row.get("image_id", "")
        if image_id not in subset_by_id:
            continue
        current_rows.append(
            {
                "image_id": image_id,
                "cropped_path": row.get("cropped_path", subset_by_id[image_id]["cropped_path"]),
                "original_path": row.get("original_path", subset_by_id[image_id]["original_path"]),
                "current_truth": canonicalize_color(row.get("adjudicated_label", "")),
                "acceptable_true_colors": row.get("acceptable_true_colors", ""),
                "include_in_formal_analysis": row.get("include_in_formal_analysis", "1"),
                "review_status": row.get("adjudication_status", ""),
                "review_notes": row.get("notes", ""),
            }
        )
    return current_rows


def collect_prior_issue_ids(config: dict) -> set[str]:
    if not get_color_policy(config)["exclude_historical_issue_ids"]:
        return set()

    prior_issue_ids: set[str] = set()
    outputs_to_check = [
        repo_path(config["outputs"]["main_dir"]) / "primary_parsed_results.csv",
        repo_path(config["outputs"]["auxiliary_dir"]) / "auxiliary_parsed_results.csv",
    ]
    for csv_path in outputs_to_check:
        if not csv_path.exists():
            continue
        for row in read_rows(csv_path):
            if row.get("outcome_type", "") in {"other_wrong", "parse_error", "refusal_or_correction"}:
                prior_issue_ids.add(row.get("image_id", ""))
    return {image_id for image_id in prior_issue_ids if image_id}


def should_exclude_from_primary(config: dict, row: dict[str, str], prior_issue_ids: set[str]) -> tuple[bool, str]:
    allowed_true_colors = primary_main_analysis_label_set(config)
    true_color = row["true_color"]
    conflict_color = row["conflict_color"]
    acceptable = str(row.get("acceptable_true_colors", "") or "").strip()
    review_status = str(row.get("review_status", "") or "").lower()

    if true_color not in allowed_true_colors:
        return True, f"ambiguous_true_color_excluded:{true_color or 'missing'}"
    if not normalize_bool(row.get("include_in_formal_analysis", "1")):
        return True, "truth_table_marked_excluded"
    if not conflict_color or conflict_color == true_color:
        return True, "invalid_conflict_color"
    if conflict_color not in allowed_true_colors:
        return True, "conflict_color_outside_allowed_primary_set"
    if row["image_id"] in DARK_LOW_SATURATION_COLOR_EXCLUSION_IDS:
        return True, DARK_LOW_SATURATION_COLOR_EXCLUSION_IDS[row["image_id"]]
    if acceptable:
        return True, "acceptable_alternate_color_boundary_case"
    if "boundary" in review_status:
        return True, "reviewed_boundary_case"
    if row["image_id"] in prior_issue_ids:
        return True, "historical_nonfaithful_or_parse_issue"
    return False, ""


def build_final_manifest(config: dict, truth_source: str, logger) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    subset_rows = read_rows(repo_path(config["dataset"]["main_subset_csv"]))
    subset_by_id = {row["image_id"]: row for row in subset_rows}
    current_truth_rows = load_current_truth_rows(config, truth_source=truth_source)
    truth_by_id = {row["image_id"]: row for row in current_truth_rows}
    prior_issue_ids = collect_prior_issue_ids(config)

    manifest_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    for image_id, subset_row in subset_by_id.items():
        truth_row = truth_by_id.get(image_id)
        if truth_row is None:
            logger.warning("Skipping image_id=%s because no current truth row is available.", image_id)
            continue

        true_color = canonicalize_color(truth_row.get("current_truth", ""))
        conflict_color = ""
        if true_color:
            conflict_color = conflict_color_for(true_color, config=config)

        row = {
            "image_id": image_id,
            "split": subset_row.get("split", ""),
            "class_id": subset_row.get("class_id", ""),
            "class_name": subset_row.get("class_name", ""),
            "file_name": subset_row.get("file_name", ""),
            "original_path": truth_row.get("original_path", subset_row.get("original_path", "")),
            "cropped_path": truth_row.get("cropped_path", subset_row.get("cropped_path", "")),
            "true_color": true_color,
            "conflict_color": conflict_color,
            "truth_source": truth_source,
            "acceptable_true_colors": truth_row.get("acceptable_true_colors", ""),
            "review_status": truth_row.get("review_status", ""),
            "review_notes": truth_row.get("review_notes", ""),
            "preliminary_color_guess": subset_row.get("preliminary_color_guess", ""),
            "selection_rank": subset_row.get("selection_rank", ""),
            "selection_bucket": subset_row.get("selection_bucket", ""),
            "quality_score": subset_row.get("quality_score", ""),
            "include_in_formal_analysis": truth_row.get("include_in_formal_analysis", "1"),
            "prior_issue_flag": "yes" if image_id in prior_issue_ids or "review" in str(truth_row.get("review_status", "")).lower() and "single_review_confirmed" not in str(truth_row.get("review_status", "")).lower() else "no",
            "reviewer_check_needed": "yes" if image_id in prior_issue_ids or true_color == "other" or str(truth_row.get("acceptable_true_colors", "")).strip() else "no",
            "include_in_primary_main_analysis": "yes",
            "include_in_v2_auxiliary_analysis": "yes",
            "exclusion_reason": "",
            "notes": "",
        }

        if image_id in POST_REFRESH_REVIEWER_CHECK_IDS:
            row["reviewer_check_needed"] = "yes"

        exclude_primary, exclusion_reason = should_exclude_from_primary(config, row, prior_issue_ids=prior_issue_ids)
        if exclude_primary:
            row["include_in_primary_main_analysis"] = "no"
            row["include_in_v2_auxiliary_analysis"] = "no"
            row["exclusion_reason"] = exclusion_reason

        note_parts = []
        if row["prior_issue_flag"] == "yes":
            note_parts.append("prior_issue_flag")
        if row["reviewer_check_needed"] == "yes":
            note_parts.append("reviewer_check_needed")
        review_notes = str(row.get("review_notes", "") or "").strip()
        if review_notes:
            note_parts.append(review_notes)
        if image_id in POST_REFRESH_REVIEWER_CHECK_IDS:
            note_parts.append(POST_REFRESH_REVIEWER_CHECK_IDS[image_id])
        row["notes"] = " | ".join(note_parts)

        manifest_rows.append(row)
        if row["include_in_primary_main_analysis"] != "yes":
            excluded_rows.append(dict(row))

    manifest_rows.sort(key=lambda item: (item["include_in_primary_main_analysis"] != "yes", item["image_id"]))
    excluded_rows.sort(key=lambda item: item["image_id"])
    return manifest_rows, excluded_rows


def build_ambiguous_excluded_rows(config: dict, manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ambiguous_labels = set(excluded_primary_labels(config))
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if row.get("include_in_primary_main_analysis") == "yes":
            continue
        if row.get("true_color") not in ambiguous_labels:
            continue
        rows.append(
            {
                "image_id": row["image_id"],
                "true_color": row["true_color"],
                "exclusion_reason": row["exclusion_reason"],
                "original_path": row["original_path"],
                "cropped_path": row["cropped_path"],
            }
        )
    rows.sort(key=lambda item: item["image_id"])
    return rows


def build_annotation_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "current_truth": row["true_color"],
                "current_truth_source": row["truth_source"],
                "acceptable_true_colors": row["acceptable_true_colors"],
                "include_in_primary_main_analysis": row["include_in_primary_main_analysis"],
                "prior_issue_flag": row["prior_issue_flag"],
                "reviewer_check_needed": row["reviewer_check_needed"],
                "annotator_label": "",
                "include_in_formal_analysis": "",
                "annotation_status": "",
                "notes": row["notes"],
            }
        )
    return rows


def build_adjudication_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "current_truth": row["true_color"],
                "current_truth_source": row["truth_source"],
                "acceptable_true_colors": row["acceptable_true_colors"],
                "include_in_primary_main_analysis": row["include_in_primary_main_analysis"],
                "prior_issue_flag": row["prior_issue_flag"],
                "reviewer_check_needed": row["reviewer_check_needed"],
                "annotator_a_label": "",
                "annotator_b_label": "",
                "adjudicated_label": "",
                "include_in_formal_analysis": "",
                "adjudication_status": "",
                "notes": row["notes"],
            }
        )
    return rows


def build_current_truth_rows(manifest_rows: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if row["truth_source"] != source_name:
            continue
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "current_truth": row["true_color"],
                "truth_source": row["truth_source"],
                "acceptable_true_colors": row["acceptable_true_colors"],
                "include_in_primary_main_analysis": row["include_in_primary_main_analysis"],
                "prior_issue_flag": row["prior_issue_flag"],
                "reviewer_check_needed": row["reviewer_check_needed"],
                "notes": row["notes"],
            }
        )
    return rows


def build_truth_snapshot_rows(config: dict, manifest_rows: list[dict[str, str]], truth_source: str) -> list[dict[str, str]]:
    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    truth_rows = load_current_truth_rows(config=config, truth_source=truth_source)
    rows: list[dict[str, str]] = []
    for truth_row in truth_rows:
        image_id = truth_row["image_id"]
        manifest_row = manifest_by_id.get(image_id)
        if manifest_row is None:
            continue
        rows.append(
            {
                "image_id": image_id,
                "cropped_path": manifest_row["cropped_path"],
                "original_path": manifest_row["original_path"],
                "current_truth": canonicalize_color(truth_row["current_truth"]),
                "truth_source": truth_source,
                "acceptable_true_colors": truth_row.get("acceptable_true_colors", ""),
                "include_in_primary_main_analysis": manifest_row["include_in_primary_main_analysis"],
                "prior_issue_flag": manifest_row["prior_issue_flag"],
                "reviewer_check_needed": manifest_row["reviewer_check_needed"],
                "notes": manifest_row["notes"],
            }
        )
    return rows


def build_prompt_rows(config: dict, manifest_rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    dataset_cfg = config["dataset"]
    prompts_cfg = config["prompts"]
    dataset_name = str(dataset_cfg.get("dataset_name", "stanford_cars_clean"))
    experiment_type = str(dataset_cfg.get("experiment_type", "car_color_prompt_mechanism_comparison_v2"))
    prompt_template_version = (
        str(prompts_cfg.get("v2_primary_prompt_template_version", "primary_conditions_v2"))
        if family == "primary"
        else str(prompts_cfg.get("v2_auxiliary_prompt_template_version", "auxiliary_conditions_v2"))
    )
    prompt_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        include_flag = "include_in_primary_main_analysis" if family == "primary" else "include_in_v2_auxiliary_analysis"
        if row.get(include_flag) != "yes":
            continue
        for condition in get_conditions(condition_version="v2", family=family):
            condition_name = str(condition["condition_name"])
            prompt_rows.append(
                {
                    "sample_id": f"{row['image_id']}_{condition_name}",
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "image_path": row["cropped_path"],
                    "original_image_path": row["original_path"],
                    "experiment_type": experiment_type,
                    "dataset_name": dataset_name,
                    "target_object": "the main car",
                    "attribute_type": "primary body color",
                    "truth_source": row["truth_source"],
                    "true_color": row["true_color"],
                    "acceptable_true_colors": row["acceptable_true_colors"],
                    "conflict_color": row["conflict_color"],
                    "condition_family": family,
                    "condition_name": condition_name,
                    "condition_index": str(condition["condition_index"]),
                    "prompt_template_version": prompt_template_version,
                    "prompt_text": prompt_text_for(condition_name, row["conflict_color"], condition_version="v2", config=config),
                    "expected_output_space": json_dumps(
                        expected_output_space(condition_name, row["conflict_color"], condition_version="v2", config=config)
                    ),
                    "expected_output_map": json_dumps(
                        expected_output_map(condition_name, row["conflict_color"], condition_version="v2", config=config)
                    ),
                    "include_in_primary_main_analysis": row["include_in_primary_main_analysis"],
                    "notes": condition["description"],
                }
            )
    return prompt_rows


def build_smoke_rows(primary_rows: list[dict[str, str]], auxiliary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    primary_image_ids: list[str] = []
    for row in primary_rows:
        if row["image_id"] not in primary_image_ids:
            primary_image_ids.append(row["image_id"])
        if len(primary_image_ids) >= 2:
            break

    smoke_ids = set(primary_image_ids)
    smoke_rows = [row for row in primary_rows + auxiliary_rows if row["image_id"] in smoke_ids]
    smoke_rows.sort(key=lambda item: (item["image_id"], item["condition_name"]))
    return smoke_rows


def write_summary(config: dict, manifest_rows: list[dict[str, str]], excluded_rows: list[dict[str, str]], primary_prompt_rows: list[dict[str, str]], auxiliary_prompt_rows: list[dict[str, str]], truth_source: str) -> None:
    summaries_dir = repo_path(config["metadata"]["summaries_dir"])
    ensure_dirs([summaries_dir])
    color_policy = get_color_policy(config)
    included_rows = [row for row in manifest_rows if row["include_in_primary_main_analysis"] == "yes"]
    excluded_counter = Counter(row["exclusion_reason"] for row in excluded_rows)
    included_counter = Counter(row["true_color"] for row in included_rows)
    ambiguous_excluded = build_ambiguous_excluded_rows(config, manifest_rows)
    summary = {
        "color_policy_variant": color_policy["variant_name"],
        "truth_source": truth_source,
        "manifest_rows": len(manifest_rows),
        "included_primary_rows": len(included_rows),
        "excluded_primary_rows": len(excluded_rows),
        "ambiguous_color_excluded_rows": len(ambiguous_excluded),
        "allowed_primary_labels": list(color_policy["primary_main_analysis_labels"]),
        "excluded_primary_labels": list(color_policy["excluded_primary_labels"]),
        "included_primary_color_distribution": dict(sorted(included_counter.items())),
        "excluded_reason_counts": dict(sorted(excluded_counter.items())),
        "primary_prompt_rows": len(primary_prompt_rows),
        "auxiliary_prompt_rows": len(auxiliary_prompt_rows),
    }

    basename = str(config.get("metadata", {}).get("prepare_summary_basename", "stanford_cars_v2_prepare_summary"))
    summary_json = summaries_dir / f"{basename}.json"
    summary_md = summaries_dir / f"{basename}.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stanford Cars V2 Preparation Summary",
        "",
        f"- color_policy_variant: {color_policy['variant_name']}",
        f"- truth_source: {truth_source}",
        f"- manifest_rows: {len(manifest_rows)}",
        f"- included_primary_rows: {len(included_rows)}",
        f"- excluded_primary_rows: {len(excluded_rows)}",
        f"- ambiguous_color_excluded_rows: {len(ambiguous_excluded)}",
        f"- primary_prompt_rows: {len(primary_prompt_rows)}",
        f"- auxiliary_prompt_rows: {len(auxiliary_prompt_rows)}",
        "",
        "## Allowed Primary Labels",
    ]
    for label in color_policy["primary_main_analysis_labels"]:
        lines.append(f"- {label}")
    lines.extend(["", "## Excluded Primary Labels"])
    for label in color_policy["excluded_primary_labels"]:
        lines.append(f"- {label}")
    lines.extend(["", "## Included Primary Color Distribution"])
    for color, count in sorted(included_counter.items()):
        lines.append(f"- {color}: {count}")
    lines.extend(["", "## Exclusion Reasons"])
    for reason, count in sorted(excluded_counter.items()):
        lines.append(f"- {reason}: {count}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_truth_snapshot_rows_from_manifest(manifest_rows: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        rows.append(
            {
                "image_id": row["image_id"],
                "cropped_path": row["cropped_path"],
                "original_path": row["original_path"],
                "current_truth": row["true_color"],
                "truth_source": source_name,
                "acceptable_true_colors": row.get("acceptable_true_colors", ""),
                "include_in_primary_main_analysis": row["include_in_primary_main_analysis"],
                "prior_issue_flag": row.get("prior_issue_flag", "no"),
                "reviewer_check_needed": row.get("reviewer_check_needed", "no"),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs([ROOT / "logs"])
    logger = build_logger("prepare_stanford_cars_multimodel_v2", ROOT / "logs" / "prepare_stanford_cars_multimodel_v2.log")
    maybe_prepare_subset(args.config, logger=logger)

    truth_source, truth_path = choose_truth_source(config, requested=args.truth_source)
    logger.info("Using truth_source=%s from %s", truth_source, relative_str(truth_path))

    dataset_cfg = config["dataset"]
    prompts_cfg = config["prompts"]
    annotation_cfg = config["annotation"]
    ensure_dirs(
        [
            repo_path(dataset_cfg["processed_dir"]),
            repo_path(prompts_cfg["dir"]),
            repo_path(annotation_cfg["v2_dir"]),
        ]
    )

    expanded_stats: dict[str, object] = {}
    if expanded_selection_enabled(config):
        manifest_rows, excluded_rows, expanded_stats = build_expanded_final_manifest(config=config, logger=logger)
        truth_source = "final_v3_curated_manual_review"
        truth_path = repo_path((config.get("expanded_selection") or {}).get("manual_review_csv", ""))
    else:
        manifest_rows, excluded_rows = build_final_manifest(config=config, truth_source=truth_source, logger=logger)
    ambiguous_excluded_rows = build_ambiguous_excluded_rows(config, manifest_rows)
    primary_prompt_rows = build_prompt_rows(config=config, manifest_rows=manifest_rows, family="primary")
    auxiliary_prompt_rows = build_prompt_rows(config=config, manifest_rows=manifest_rows, family="auxiliary")
    smoke_rows = build_smoke_rows(primary_rows=primary_prompt_rows, auxiliary_rows=auxiliary_prompt_rows)

    final_manifest_csv = repo_path(dataset_cfg["final_manifest_csv"])
    final_auxiliary_manifest_csv = repo_path(dataset_cfg.get("final_auxiliary_manifest_csv", dataset_cfg["final_manifest_csv"]))
    excluded_csv = repo_path(dataset_cfg["excluded_records_csv"])
    excluded_manual_review_csv = repo_path(dataset_cfg["excluded_manual_review_csv"]) if dataset_cfg.get("excluded_manual_review_csv") else None
    ambiguous_excluded_csv = repo_path(dataset_cfg["ambiguous_color_excluded_csv"]) if dataset_cfg.get("ambiguous_color_excluded_csv") else None
    write_rows(final_manifest_csv, final_manifest_fieldnames(), manifest_rows)
    write_rows(final_auxiliary_manifest_csv, final_manifest_fieldnames(), manifest_rows)
    write_rows(excluded_csv, excluded_fieldnames(), excluded_rows)
    if excluded_manual_review_csv is not None:
        manual_excluded_rows = [row for row in excluded_rows if row.get("exclusion_reason") == "ambiguous_after_manual_review"]
        write_rows(excluded_manual_review_csv, excluded_fieldnames(), manual_excluded_rows)
    if ambiguous_excluded_csv is not None:
        write_rows(ambiguous_excluded_csv, ambiguous_excluded_fieldnames(), ambiguous_excluded_rows)
    write_rows(repo_path(prompts_cfg["v2_primary_csv"]), prompt_fieldnames(), primary_prompt_rows)
    write_rows(repo_path(prompts_cfg["v2_auxiliary_csv"]), prompt_fieldnames(), auxiliary_prompt_rows)
    write_rows(repo_path(prompts_cfg["v2_smoke_csv"]), prompt_fieldnames(), smoke_rows)

    write_rows(repo_path(annotation_cfg["v2_annotator_a_csv"]), annotation_fieldnames(), build_annotation_rows(manifest_rows))
    write_rows(repo_path(annotation_cfg["v2_annotator_b_csv"]), annotation_fieldnames(), build_annotation_rows(manifest_rows))
    write_rows(repo_path(annotation_cfg["v2_adjudication_template_csv"]), adjudication_fieldnames(), build_adjudication_rows(manifest_rows))
    write_rows(
        repo_path(annotation_cfg["v2_provisional_truth_csv"]),
        current_truth_fieldnames(),
        build_truth_snapshot_rows_from_manifest(manifest_rows=manifest_rows, source_name=truth_source),
    )
    write_rows(
        repo_path(annotation_cfg["v2_reviewed_truth_csv"]),
        current_truth_fieldnames(),
        build_truth_snapshot_rows_from_manifest(manifest_rows=manifest_rows, source_name=truth_source),
    )

    write_summary(
        config=config,
        manifest_rows=manifest_rows,
        excluded_rows=excluded_rows,
        primary_prompt_rows=primary_prompt_rows,
        auxiliary_prompt_rows=auxiliary_prompt_rows,
        truth_source=truth_source,
    )
    if expanded_stats:
        write_color_distribution_summary(config=config, expanded_stats=expanded_stats)

    included_rows = [row for row in manifest_rows if row["include_in_primary_main_analysis"] == "yes"]
    result = {
        "truth_source": truth_source,
        "truth_path": relative_str(truth_path),
        "final_manifest_csv": relative_str(final_manifest_csv),
        "final_auxiliary_manifest_csv": relative_str(final_auxiliary_manifest_csv),
        "excluded_records_csv": relative_str(excluded_csv),
        "excluded_manual_review_csv": relative_str(excluded_manual_review_csv) if excluded_manual_review_csv else "",
        "ambiguous_color_excluded_csv": relative_str(ambiguous_excluded_csv) if ambiguous_excluded_csv else "",
        "annotation_dir": relative_str(repo_path(annotation_cfg["v2_dir"])),
        "primary_prompt_csv": relative_str(repo_path(prompts_cfg["v2_primary_csv"])),
        "auxiliary_prompt_csv": relative_str(repo_path(prompts_cfg["v2_auxiliary_csv"])),
        "smoke_prompt_csv": relative_str(repo_path(prompts_cfg["v2_smoke_csv"])),
        "included_primary_rows": len(included_rows),
        "excluded_primary_rows": len(excluded_rows),
        "ambiguous_color_excluded_rows": len(ambiguous_excluded_rows),
    }
    if expanded_stats:
        result["expanded_stats"] = expanded_stats
    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Prepared current strict-colors manifest and prompts: %s", json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
