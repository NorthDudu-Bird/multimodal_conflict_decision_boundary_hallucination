#!/usr/bin/env python
"""Build a deterministic VCoR candidate pool and contact sheets for manual review."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import (
    build_logger,
    canonicalize_color,
    ensure_dirs,
    load_config,
    normalize_bool,
    primary_main_analysis_labels,
    read_rows,
    relative_str,
    repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VCoR candidate pool for strict manual review.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overfetch-factor", type=int, default=4)
    parser.add_argument("--min-per-color", type=int, default=24)
    parser.add_argument("--target-per-color", type=int, default=25)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def load_core_counts(config: dict) -> Counter[str]:
    builder_cfg = config.get("dataset_builder", {}) or {}
    base_manifest_csv = repo_path(builder_cfg.get("base_manifest_csv", "data/processed/stanford_cars/final_primary_manifest_strict_colors.csv"))
    exclude_ids = set(builder_cfg.get("core_manual_exclude_ids", []))
    rows = read_rows(base_manifest_csv)
    allowed = set(primary_main_analysis_labels(config))
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("include_in_primary_main_analysis") != "yes":
            continue
        if row.get("image_id") in exclude_ids:
            continue
        color = canonicalize_color(row.get("true_color", ""))
        if color in allowed:
            counts[color] += 1
    return counts


def sample_inventory_rows(config: dict, overfetch_factor: int, min_per_color: int, target_per_color: int) -> tuple[list[dict[str, str]], list[dict[str, int]], Counter[str]]:
    vcor_cfg = config.get("vcor", {}) or {}
    inventory_csv = repo_path(vcor_cfg.get("inventory_csv", "data_external/vcor_selected/vcor_inventory.csv"))
    rows = read_rows(inventory_csv)
    allowed = primary_main_analysis_labels(config)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        color = canonicalize_color(row.get("assigned_true_color", ""))
        if color in set(allowed):
            grouped[color].append(row)

    core_counts = load_core_counts(config)
    candidate_rows: list[dict[str, str]] = []
    planning_rows: list[dict[str, int]] = []
    for color in allowed:
        deficit = max(0, target_per_color - int(core_counts.get(color, 0)))
        requested = max(min_per_color if deficit > 0 else 0, deficit * max(1, overfetch_factor))
        available = len(grouped[color])
        selected = grouped[color][: min(requested, available)]
        planning_rows.append(
            {
                "color": color,
                "core_count": int(core_counts.get(color, 0)),
                "target_per_color": target_per_color,
                "deficit": deficit,
                "requested_candidates": requested,
                "available_candidates": available,
                "selected_candidates": len(selected),
            }
        )
        for index, row in enumerate(selected, start=1):
            candidate = dict(row)
            candidate["candidate_rank"] = str(index)
            candidate_rows.append(candidate)
    return candidate_rows, planning_rows, core_counts


def candidate_fieldnames() -> list[str]:
    return [
        "candidate_id",
        "candidate_rank",
        "image_id",
        "source_dataset",
        "split",
        "assigned_true_color",
        "source_path",
        "staged_path",
        "keep",
        "drop",
        "decision",
        "rejection_reason",
        "reviewer_note",
    ]


def stage_candidate_images(candidate_rows: list[dict[str, str]], stage_root: Path) -> list[dict[str, str]]:
    staged_rows: list[dict[str, str]] = []
    for row in candidate_rows:
        source_path = repo_path(row["source_path"])
        color = canonicalize_color(row["assigned_true_color"])
        staged_dir = stage_root / color
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_dir / f"{row['image_id']}{source_path.suffix.lower()}"
        if staged_path.resolve() != source_path.resolve():
            shutil.copy2(source_path, staged_path)
        staged_row = dict(row)
        staged_row["candidate_id"] = row["image_id"]
        staged_row["staged_path"] = relative_str(staged_path)
        staged_row["keep"] = ""
        staged_row["drop"] = ""
        staged_row["decision"] = ""
        staged_row["rejection_reason"] = ""
        staged_row["reviewer_note"] = ""
        staged_rows.append(staged_row)
    return staged_rows


def draw_contact_sheet(rows: list[dict[str, str]], color: str, output_path: Path) -> None:
    if not rows:
        return
    thumb_w = 240
    thumb_h = 160
    padding = 16
    text_h = 36
    cols = 4
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new(
        "RGB",
        (cols * (thumb_w + padding) + padding, rows_n * (thumb_h + text_h + padding) + padding),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, row in enumerate(rows):
        x = padding + (idx % cols) * (thumb_w + padding)
        y = padding + (idx // cols) * (thumb_h + text_h + padding)
        image = Image.open(repo_path(row["staged_path"])).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        canvas.paste(image, (x + (thumb_w - image.width) // 2, y))
        draw.text((x, y + thumb_h + 6), f"{color} | {row['image_id']}", fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    vcor_cfg = config.get("vcor", {}) or {}
    stage_root = repo_path(vcor_cfg.get("candidate_pool_dir", "data_external/vcor_selected/candidate_pool"))
    review_csv = repo_path(vcor_cfg.get("candidate_review_csv", "data_external/vcor_selected/candidate_review.csv"))
    planning_csv = repo_path(vcor_cfg.get("candidate_plan_csv", "data_external/vcor_selected/candidate_plan.csv"))
    contact_dir = repo_path(vcor_cfg.get("candidate_contact_dir", "logs/vcor_candidate_review"))
    log_path = args.log_path or repo_path(vcor_cfg.get("candidate_log_path", "logs/build_vcor_candidate_pool.log"))
    ensure_dirs([stage_root, review_csv.parent, planning_csv.parent, contact_dir, log_path.parent])
    logger = build_logger("build_vcor_candidate_pool", log_path)

    sampled_rows, planning_rows, core_counts = sample_inventory_rows(
        config=config,
        overfetch_factor=args.overfetch_factor,
        min_per_color=args.min_per_color,
        target_per_color=args.target_per_color,
    )
    if not sampled_rows:
        raise RuntimeError("No VCoR candidate rows were selected. Stage the dataset first and verify the VCoR inventory.")

    staged_rows = stage_candidate_images(sampled_rows, stage_root=stage_root)
    for color in primary_main_analysis_labels(config):
        color_rows = [row for row in staged_rows if canonicalize_color(row["assigned_true_color"]) == color]
        if color_rows:
            draw_contact_sheet(color_rows, color=color, output_path=contact_dir / f"{color}_candidate_sheet.jpg")

    with review_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=candidate_fieldnames())
        writer.writeheader()
        for row in staged_rows:
            writer.writerow({field: row.get(field, "") for field in candidate_fieldnames()})

    with planning_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "color",
                "core_count",
                "target_per_color",
                "deficit",
                "requested_candidates",
                "available_candidates",
                "selected_candidates",
            ],
        )
        writer.writeheader()
        for row in planning_rows:
            writer.writerow(row)

    payload = {
        "review_csv": relative_str(review_csv),
        "planning_csv": relative_str(planning_csv),
        "candidate_pool_dir": relative_str(stage_root),
        "contact_dir": relative_str(contact_dir),
        "core_counts": dict(core_counts),
        "candidate_counts": dict(Counter(canonicalize_color(row["assigned_true_color"]) for row in staged_rows)),
    }
    logger.info("VCoR candidate pool complete: %s", payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
