#!/usr/bin/env python
"""Build a human-review manifest for visual clarity and task-validity checks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.utils.paper_mainline_utils import load_bool_results, write_markdown


AUDIT_COLUMNS = [
    "audit_visual_clarity",
    "audit_specular_reflection",
    "audit_background_color_bias",
    "audit_multi_car_interference",
    "audit_occlusion",
    "audit_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual clarity audit infrastructure.")
    parser.add_argument("--main-csv", type=Path, default=REPO_ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "audit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick_controls(conflict_df: pd.DataFrame, faithful_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = faithful_df.copy()
    pool["_rand"] = [rng.random() for _ in range(len(pool))]
    pool = pool.sort_values(["_rand", "image_id"]).reset_index(drop=True)
    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    match_strategy: dict[str, str] = {}

    def take(candidates: pd.DataFrame, n: int, strategy: str) -> None:
        nonlocal selected_parts, selected_ids, match_strategy
        if n <= 0:
            return
        candidates = candidates[~candidates["image_id"].isin(selected_ids)].head(n).copy()
        if candidates.empty:
            return
        for image_id in candidates["image_id"].astype(str):
            match_strategy[image_id] = strategy
            selected_ids.add(image_id)
        selected_parts.append(candidates)

    strata = conflict_df.groupby(["true_color", "source_dataset"], observed=False).size().reset_index(name="needed")
    for row in strata.to_dict("records"):
        candidates = pool[(pool["true_color"] == row["true_color"]) & (pool["source_dataset"] == row["source_dataset"])]
        take(candidates, int(row["needed"]), "matched_true_color_and_source")

    selected_count = sum(len(part) for part in selected_parts)
    remaining = len(conflict_df) - selected_count
    if remaining > 0:
        color_needs = conflict_df["true_color"].value_counts().to_dict()
        selected_so_far = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=pool.columns)
        selected_color_counts = selected_so_far["true_color"].value_counts().to_dict() if not selected_so_far.empty else {}
        for color, needed_total in color_needs.items():
            deficit = int(needed_total) - int(selected_color_counts.get(color, 0))
            if deficit > 0:
                candidates = pool[pool["true_color"] == color]
                take(candidates, deficit, "matched_true_color_after_source_shortage")

    selected_count = sum(len(part) for part in selected_parts)
    remaining = len(conflict_df) - selected_count
    if remaining > 0:
        source_needs = conflict_df["source_dataset"].value_counts().to_dict()
        selected_so_far = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=pool.columns)
        selected_source_counts = selected_so_far["source_dataset"].value_counts().to_dict() if not selected_so_far.empty else {}
        for source, needed_total in source_needs.items():
            deficit = int(needed_total) - int(selected_source_counts.get(source, 0))
            if deficit > 0:
                candidates = pool[pool["source_dataset"] == source]
                take(candidates, deficit, "matched_source_after_color_shortage")

    selected_count = sum(len(part) for part in selected_parts)
    remaining = len(conflict_df) - selected_count
    if remaining > 0:
        take(pool, remaining, "deterministic_remaining_fill")

    controls = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["image_id"]).head(len(conflict_df)).copy()
    controls["match_strategy"] = controls["image_id"].astype(str).map(match_strategy).fillna("deterministic_remaining_fill")
    return controls


def build_manifest(main_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    c3 = main_df[
        (main_df["model_key"] == "llava15_7b")
        & (main_df["condition_name"] == "C3_presupposition_correction_allowed")
    ].copy()
    conflict = c3[c3["is_conflict_aligned"]].copy().sort_values(["true_color", "source_dataset", "image_id"])
    faithful = c3[c3["is_faithful"]].copy()
    controls = pick_controls(conflict, faithful, seed)

    conflict["audit_group"] = "conflict_aligned_case"
    conflict["match_strategy"] = "target_case"
    controls["audit_group"] = "matched_faithful_control"
    combined = pd.concat([conflict, controls], ignore_index=True, sort=False)
    combined = combined.sort_values(["audit_group", "true_color", "source_dataset", "image_id"]).reset_index(drop=True)

    manifest = pd.DataFrame(
        {
            "sample_id": combined["sample_id"],
            "image_path": combined["image_path"],
            "model": combined["model_name"],
            "condition": combined["condition_name"],
            "true_color": combined["true_color"],
            "false_prompt_color": combined["conflict_color"],
            "model_output": combined["raw_output"],
            "source_dataset": combined["source_dataset"],
            "is_conflict_aligned": combined["is_conflict_aligned"],
            "audit_group": combined["audit_group"],
            "parsed_label": combined["parsed_label"],
            "match_strategy": combined["match_strategy"],
        }
    )
    for column in AUDIT_COLUMNS:
        manifest[column] = ""
    return manifest


def rel_from_audit(path_text: str) -> str:
    return "../../" + path_text.replace("\\", "/")


def write_gallery(manifest: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Visual Clarity Audit Gallery",
        "",
        "This gallery indexes the conflict-aligned LLaVA original `C3` cases and matched faithful controls. It is an audit aid, not a model result.",
        "",
    ]
    for group, group_df in manifest.groupby("audit_group", observed=False):
        lines.extend([f"## {group}", ""])
        for row in group_df.to_dict("records"):
            image_rel = rel_from_audit(str(row["image_path"]))
            lines.extend(
                [
                    f"### {row['sample_id']}",
                    "",
                    f'<img src="{image_rel}" width="220">',
                    "",
                    f"- true_color: `{row['true_color']}`",
                    f"- false_prompt_color: `{row['false_prompt_color']}`",
                    f"- model_output: `{row['model_output']}`",
                    f"- source_dataset: `{row['source_dataset']}`",
                    f"- match_strategy: `{row['match_strategy']}`",
                    "",
                ]
            )
    write_markdown(output_path, "\n".join(lines))


def write_readme(manifest: pd.DataFrame, output_path: Path) -> None:
    conflict_n = int((manifest["audit_group"] == "conflict_aligned_case").sum())
    control_n = int((manifest["audit_group"] == "matched_faithful_control").sum())
    missing_paths = [path for path in manifest["image_path"] if not (REPO_ROOT / str(path)).exists()]
    lines = [
        "# Visual Clarity Audit Readme",
        "",
        "## Purpose",
        "",
        "This audit checks whether the LLaVA-1.5-7B original `C3` conflict-aligned cases might be visually harder or less task-valid than matched faithful controls.",
        "",
        "## Scope",
        "",
        f"- Conflict-aligned cases: {conflict_n}",
        f"- Matched faithful controls: {control_n}",
        "- Matching priority: exact `true_color + source_dataset`, then `true_color`, then source-balanced deterministic fill.",
        "- Human review fields are intentionally blank so the audit can be completed independently.",
        "",
        "## Role In The Paper",
        "",
        "- This module is threat reduction, not a new main experiment.",
        "- It helps address whether the observed conflict-aligned cases are concentrated in visually ambiguous examples.",
        "- It should not be used to claim a new visual difficulty factor unless a completed human audit supports that claim.",
        "",
        "## File Checks",
        "",
        f"- Image paths checked: {len(manifest)}",
        f"- Missing image paths: {len(missing_paths)}",
    ]
    if missing_paths:
        lines.extend(["", "## Missing Paths", ""])
        lines.extend([f"- `{path}`" for path in missing_paths])
    write_markdown(output_path, "\n".join(lines))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_df = load_bool_results(args.main_csv)
    manifest = build_manifest(main_df, args.seed)
    manifest.to_csv(args.output_dir / "visual_clarity_audit_manifest.csv", index=False, encoding="utf-8-sig")
    write_gallery(manifest, args.output_dir / "visual_clarity_gallery.md")
    write_readme(manifest, args.output_dir / "visual_clarity_audit_readme.md")
    missing = [path for path in manifest["image_path"] if not (REPO_ROOT / str(path)).exists()]
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "visual_clarity_audit_manifest.csv"),
                "gallery": str(args.output_dir / "visual_clarity_gallery.md"),
                "readme": str(args.output_dir / "visual_clarity_audit_readme.md"),
                "rows": int(len(manifest)),
                "missing_image_paths": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
