#!/usr/bin/env python
"""Build rigorous Stanford-only and VCoR-supplemented manifests plus prompt tables."""

from __future__ import annotations

import argparse
import json
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
    get_conditions,
    json_dumps,
    load_config,
    normalize_bool,
    primary_main_analysis_labels,
    prompt_text_for,
    read_rows,
    relative_str,
    repo_path,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current primary manifests for Stanford-only and VCoR-balanced versions.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def formal_manifest_fieldnames() -> list[str]:
    return [
        "image_id",
        "file_name",
        "source_dataset",
        "source_split",
        "source_path",
        "original_path",
        "cropped_path",
        "true_color",
        "conflict_color",
        "truth_source",
        "review_status",
        "review_notes",
        "include_in_primary_main_analysis",
        "include_in_v2_auxiliary_analysis",
        "selection_version",
        "selection_bucket",
        "selection_rank",
        "exclusion_reason",
        "notes",
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
        "source_dataset",
        "notes",
    ]


def core_exclusion_rows_fieldnames() -> list[str]:
    return ["image_id", "true_color", "exclusion_reason", "notes"]


def dataset_counts(manifest_rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in manifest_rows:
        if row.get("include_in_primary_main_analysis") == "yes":
            counter[canonicalize_color(row.get("true_color", ""))] += 1
    return counter


def strict_core_rows(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    builder_cfg = config.get("dataset_builder", {}) or {}
    base_manifest_csv = repo_path(builder_cfg.get("base_manifest_csv", "data/processed/stanford_cars/final_primary_manifest_strict_colors.csv"))
    exclude_ids = set(builder_cfg.get("core_manual_exclude_ids", []))
    allowed_colors = set(primary_main_analysis_labels(config))
    base_rows = read_rows(base_manifest_csv)
    kept_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []

    for row in base_rows:
        if row.get("include_in_primary_main_analysis") != "yes":
            continue
        image_id = row.get("image_id", "")
        color = canonicalize_color(row.get("true_color", ""))
        if image_id in exclude_ids:
            excluded_rows.append(
                {
                    "image_id": image_id,
                    "true_color": color,
                    "exclusion_reason": "latest_manual_ambiguity_exclusion",
                    "notes": row.get("notes", ""),
                }
            )
            continue
        if color not in allowed_colors:
            continue
        kept_rows.append(
            {
                "image_id": image_id,
                "file_name": row.get("file_name", f"{image_id}.jpg"),
                "source_dataset": "StanfordCars",
                "source_split": row.get("split", ""),
                "source_path": row.get("original_path", ""),
                "original_path": row.get("original_path", ""),
                "cropped_path": row.get("cropped_path", ""),
                "true_color": color,
                "conflict_color": conflict_color_for(color, config=config),
                "truth_source": row.get("truth_source", "reviewed"),
                "review_status": row.get("review_status", "reviewed"),
                "review_notes": row.get("review_notes", ""),
                "include_in_primary_main_analysis": "yes",
                "include_in_v2_auxiliary_analysis": "yes",
                "selection_version": "primary_core_stanford_only",
                "selection_bucket": "stanford_core",
                "selection_rank": row.get("selection_rank", ""),
                "exclusion_reason": "",
                "notes": row.get("notes", ""),
            }
        )

    kept_rows.sort(key=lambda item: item["image_id"])
    excluded_rows.sort(key=lambda item: item["image_id"])
    return kept_rows, excluded_rows


def normalize_review_decision(row: dict[str, str]) -> str:
    explicit = str(row.get("decision", "") or "").strip().lower()
    if explicit in {"include", "keep", "yes"}:
        return "include"
    if explicit in {"exclude", "drop", "no"}:
        return "exclude"
    if normalize_bool(row.get("keep", "")):
        return "include"
    if normalize_bool(row.get("drop", "")):
        return "exclude"
    return ""


def build_vcor_selected_and_rejected(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    vcor_cfg = config.get("vcor", {}) or {}
    review_csv = repo_path(vcor_cfg.get("candidate_review_csv", "data_external/vcor_selected/candidate_review.csv"))
    if not review_csv.exists():
        raise FileNotFoundError(
            f"Missing VCoR review CSV: {review_csv}. Run stage_vcor_dataset.py and build_vcor_candidate_pool.py first."
        )

    allowed_colors = set(primary_main_analysis_labels(config))
    selected_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    for row in read_rows(review_csv):
        decision = normalize_review_decision(row)
        color = canonicalize_color(row.get("assigned_true_color", ""))
        if color not in allowed_colors:
            continue
        base = {
            "image_id": row.get("image_id", ""),
            "source_dataset": "VCoR",
            "split": row.get("split", ""),
            "source_path": row.get("source_path", ""),
            "staged_path": row.get("staged_path", row.get("source_path", "")),
            "assigned_true_color": color,
            "file_name": Path(row.get("staged_path", row.get("source_path", ""))).name,
            "rejection_reason": row.get("rejection_reason", ""),
            "reviewer_note": row.get("reviewer_note", ""),
        }
        if decision == "include":
            selected_rows.append(base)
        elif decision == "exclude":
            rejected_rows.append(base)

    if not selected_rows:
        raise RuntimeError("No VCoR rows were marked for inclusion. Fill in data_external/vcor_selected/candidate_review.csv first.")
    return selected_rows, rejected_rows


def vcor_manifest_row(review_row: dict[str, str], selection_rank: int, config: dict) -> dict[str, str]:
    color = canonicalize_color(review_row["assigned_true_color"])
    return {
        "image_id": review_row["image_id"],
        "file_name": review_row["file_name"],
        "source_dataset": "VCoR",
        "source_split": review_row.get("split", ""),
        "source_path": review_row["source_path"],
        "original_path": review_row["source_path"],
        "cropped_path": review_row["staged_path"],
        "true_color": color,
        "conflict_color": conflict_color_for(color, config=config),
        "truth_source": "manual_review_vcor",
        "review_status": "manual_review_vcor_include",
        "review_notes": review_row.get("reviewer_note", ""),
        "include_in_primary_main_analysis": "yes",
        "include_in_v2_auxiliary_analysis": "yes",
        "selection_version": "primary_expanded_balanced_with_vcor",
        "selection_bucket": f"vcor_{color}",
        "selection_rank": str(selection_rank),
        "exclusion_reason": "",
        "notes": review_row.get("reviewer_note", ""),
    }


def build_prompt_rows(config: dict, manifest_rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    dataset_cfg = config["dataset"]
    prompts_cfg = config["prompts"]
    dataset_name = str(dataset_cfg.get("dataset_name", "primary_expanded_balanced_with_vcor"))
    experiment_type = str(dataset_cfg.get("experiment_type", "car_color_prompt_mechanism_comparison_vcor_balanced"))
    prompt_template_version = (
        str(prompts_cfg.get("v2_primary_prompt_template_version", "primary_conditions_v2_strict_colors"))
        if family == "primary"
        else str(prompts_cfg.get("v2_auxiliary_prompt_template_version", "auxiliary_conditions_v2_strict_colors"))
    )
    include_flag = "include_in_primary_main_analysis" if family == "primary" else "include_in_v2_auxiliary_analysis"
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if row.get(include_flag) != "yes":
            continue
        for condition in get_conditions(condition_version="v2", family=family):
            condition_name = str(condition["condition_name"])
            rows.append(
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
                    "acceptable_true_colors": "",
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
                    "source_dataset": row["source_dataset"],
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def build_smoke_rows(primary_rows: list[dict[str, str]], auxiliary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    smoke_ids: list[str] = []
    for row in primary_rows:
        if row["image_id"] not in smoke_ids:
            smoke_ids.append(row["image_id"])
        if len(smoke_ids) >= 2:
            break
    smoke_id_set = set(smoke_ids)
    rows = [row for row in primary_rows + auxiliary_rows if row["image_id"] in smoke_id_set]
    rows.sort(key=lambda item: (item["image_id"], item["condition_name"]))
    return rows


def write_count_tables(config: dict, core_rows: list[dict[str, str]], expanded_rows: list[dict[str, str]], rejected_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    builder_cfg = config.get("dataset_builder", {}) or {}
    comparison_csv = repo_path(builder_cfg.get("comparison_csv", "analysis/current/vcor_balanced_dataset_version_comparison.csv"))
    comparison_md = repo_path(builder_cfg.get("comparison_md", "analysis/current/vcor_balanced_dataset_version_comparison.md"))
    ensure_dirs([comparison_csv.parent, comparison_md.parent])
    core_counts = dataset_counts(core_rows)
    expanded_counts = dataset_counts(expanded_rows)
    rows = []
    for color in primary_main_analysis_labels(config):
        rows.append(
            {
                "color": color,
                "stanford_only_core": int(core_counts.get(color, 0)),
                "expanded_balanced_with_vcor": int(expanded_counts.get(color, 0)),
                "vcor_additions": int(expanded_counts.get(color, 0) - core_counts.get(color, 0)),
            }
        )
    write_rows(comparison_csv, ["color", "stanford_only_core", "expanded_balanced_with_vcor", "vcor_additions"], rows)

    rejection_counts = Counter(canonicalize_color(row.get("assigned_true_color", "")) for row in rejected_rows)
    lines = [
        "# Dataset Version Comparison",
        "",
        f"- stanford_only_core_total: {sum(core_counts.values())}",
        f"- expanded_balanced_with_vcor_total: {sum(expanded_counts.values())}",
        f"- vcor_selected_total: {sum(expanded_counts.values()) - sum(core_counts.values())}",
        f"- vcor_rejected_total: {len(rejected_rows)}",
        "",
        "## Per-Color Counts",
    ]
    for row in rows:
        lines.append(
            f"- {row['color']}: core={row['stanford_only_core']}, expanded={row['expanded_balanced_with_vcor']}, "
            f"vcor_additions={row['vcor_additions']}"
        )
    lines.extend(["", "## Rejected VCoR Counts"])
    for color in primary_main_analysis_labels(config):
        lines.append(f"- {color}: {int(rejection_counts.get(color, 0))}")
    comparison_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison_csv, comparison_md


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    builder_cfg = config.get("dataset_builder", {}) or {}
    log_path = args.log_path or repo_path(builder_cfg.get("log_path", "logs/build_primary_vcor_balanced_manifests.log"))
    ensure_dirs([log_path.parent])
    logger = build_logger("build_primary_vcor_balanced_manifests", log_path)

    core_rows, core_excluded_rows = strict_core_rows(config)
    selected_vcor_rows: list[dict[str, str]] = []
    rejected_vcor_rows: list[dict[str, str]] = []
    expanded_rows: list[dict[str, str]] = []
    if not args.core_only:
        selected_vcor_rows, rejected_vcor_rows = build_vcor_selected_and_rejected(config)
        vcor_rows = [vcor_manifest_row(row, selection_rank=index, config=config) for index, row in enumerate(selected_vcor_rows, start=1)]
        expanded_rows = list(core_rows) + vcor_rows
        expanded_rows.sort(key=lambda item: (item["source_dataset"], item["true_color"], item["image_id"]))

    dataset_cfg = config["dataset"]
    prompts_cfg = config["prompts"]
    vcor_cfg = config.get("vcor", {}) or {}

    core_manifest_csv = repo_path(dataset_cfg.get("core_manifest_csv", "data/processed/stanford_cars/primary_core_stanford_only.csv"))
    expanded_manifest_csv = repo_path(dataset_cfg["final_manifest_csv"])
    expanded_aux_csv = repo_path(dataset_cfg.get("final_auxiliary_manifest_csv", dataset_cfg["final_manifest_csv"]))
    core_excluded_csv = repo_path(dataset_cfg.get("core_excluded_csv", "data/processed/stanford_cars/primary_core_excluded_latest_manual_review.csv"))
    selected_manifest_csv = repo_path(vcor_cfg.get("selected_manifest_csv", "data_external/vcor_selected/selected_manifest.csv"))
    rejected_manifest_csv = repo_path(vcor_cfg.get("rejected_manifest_csv", "data_external/vcor_selected/rejected_manifest.csv"))

    ensure_dirs([core_manifest_csv.parent, expanded_manifest_csv.parent, selected_manifest_csv.parent, rejected_manifest_csv.parent])
    write_rows(core_manifest_csv, formal_manifest_fieldnames(), core_rows)
    write_rows(core_excluded_csv, core_exclusion_rows_fieldnames(), core_excluded_rows)
    comparison_csv = None
    comparison_md = None
    primary_prompt_rows: list[dict[str, str]] = []
    auxiliary_prompt_rows: list[dict[str, str]] = []
    smoke_rows: list[dict[str, str]] = []
    if not args.core_only:
        write_rows(expanded_manifest_csv, formal_manifest_fieldnames(), expanded_rows)
        write_rows(expanded_aux_csv, formal_manifest_fieldnames(), expanded_rows)
        write_rows(
            selected_manifest_csv,
            ["image_id", "source_dataset", "split", "source_path", "staged_path", "assigned_true_color", "reviewer_note"],
            selected_vcor_rows,
        )
        write_rows(
            rejected_manifest_csv,
            ["image_id", "source_dataset", "split", "source_path", "assigned_true_color", "rejection_reason", "reviewer_note"],
            rejected_vcor_rows,
        )

        primary_prompt_rows = build_prompt_rows(config=config, manifest_rows=expanded_rows, family="primary")
        auxiliary_prompt_rows = build_prompt_rows(config=config, manifest_rows=expanded_rows, family="auxiliary")
        smoke_rows = build_smoke_rows(primary_rows=primary_prompt_rows, auxiliary_rows=auxiliary_prompt_rows)
        comparison_csv, comparison_md = write_count_tables(config=config, core_rows=core_rows, expanded_rows=expanded_rows, rejected_rows=rejected_vcor_rows)
    else:
        primary_prompt_rows = build_prompt_rows(config=config, manifest_rows=core_rows, family="primary")
        auxiliary_prompt_rows = build_prompt_rows(config=config, manifest_rows=core_rows, family="auxiliary")
        smoke_rows = build_smoke_rows(primary_rows=primary_prompt_rows, auxiliary_rows=auxiliary_prompt_rows)

    write_rows(repo_path(prompts_cfg["v2_primary_csv"]), prompt_fieldnames(), primary_prompt_rows)
    write_rows(repo_path(prompts_cfg["v2_auxiliary_csv"]), prompt_fieldnames(), auxiliary_prompt_rows)
    write_rows(repo_path(prompts_cfg["v2_smoke_csv"]), prompt_fieldnames(), smoke_rows)

    payload = {
        "core_manifest_csv": relative_str(core_manifest_csv),
        "core_excluded_csv": relative_str(core_excluded_csv),
        "stanford_only_core_counts": dict(dataset_counts(core_rows)),
        "core_only": args.core_only,
        "primary_prompt_csv": relative_str(repo_path(prompts_cfg["v2_primary_csv"])),
        "auxiliary_prompt_csv": relative_str(repo_path(prompts_cfg["v2_auxiliary_csv"])),
        "smoke_prompt_csv": relative_str(repo_path(prompts_cfg["v2_smoke_csv"])),
    }
    if not args.core_only:
        payload.update(
            {
                "expanded_manifest_csv": relative_str(expanded_manifest_csv),
                "expanded_auxiliary_manifest_csv": relative_str(expanded_aux_csv),
                "selected_manifest_csv": relative_str(selected_manifest_csv),
                "rejected_manifest_csv": relative_str(rejected_manifest_csv),
                "comparison_csv": relative_str(comparison_csv),
                "comparison_md": relative_str(comparison_md),
                "expanded_counts": dict(dataset_counts(expanded_rows)),
            }
        )
    logger.info("Built manifests: %s", json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
