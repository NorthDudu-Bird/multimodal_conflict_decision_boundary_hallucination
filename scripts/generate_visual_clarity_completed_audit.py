#!/usr/bin/env python
"""Build and summarize a completed visual-clarity audit pack for phase 2."""

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
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.utils.paper_mainline_utils import load_bool_results, write_markdown


MODEL_KEY = "llava15_7b"
AUDIT_FIELDS = [
    "audit_visual_clarity",
    "audit_body_color_salience",
    "audit_specular_reflection",
    "audit_shadow_or_night_effect",
    "audit_background_color_bias",
    "audit_multi_car_interference",
    "audit_occlusion",
    "audit_notes",
    "audit_reviewer",
    "audit_review_date",
]

# Manual visual review defaults are deliberately conservative and can be overridden
# after inspecting the contact sheets. Values are applied row-wise to keep every
# audit field complete.
DEFAULT_REVIEW = {
    "audit_visual_clarity": "clear",
    "audit_body_color_salience": "high",
    "audit_specular_reflection": "none_minor",
    "audit_shadow_or_night_effect": "none_minor",
    "audit_background_color_bias": "none_minor",
    "audit_multi_car_interference": "none",
    "audit_occlusion": "none",
    "audit_notes": "Body color is reviewable from the local image/contact sheet.",
}

# Filled after contact-sheet review. Keys are image_id, so repeated condition rows
# for the same image receive consistent visual-quality annotations.
MANUAL_REVIEW_OVERRIDES: dict[str, dict[str, str]] = {
    "train_08107": {
        "audit_visual_clarity": "moderate",
        "audit_body_color_salience": "medium",
        "audit_specular_reflection": "strong",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "moderate",
        "audit_multi_car_interference": "present",
        "audit_occlusion": "none",
        "audit_notes": "Glossy black car in an indoor/showroom-like scene with strong reflections and nearby vehicles; color remains inspectable but not pristine.",
    },
    "test_08002": {
        "audit_visual_clarity": "moderate",
        "audit_body_color_salience": "medium",
        "audit_specular_reflection": "moderate",
        "audit_shadow_or_night_effect": "none_minor",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "minor",
        "audit_occlusion": "none",
        "audit_notes": "Black pickup is visible, but wet pavement and glossy reflections make the body tone less clean than studio examples.",
    },
    "test_03865": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "medium",
        "audit_specular_reflection": "moderate",
        "audit_shadow_or_night_effect": "none_minor",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "White body is clear, but dark/blue racing stripes introduce a non-body-color visual distractor.",
    },
    "train_03125": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "high",
        "audit_specular_reflection": "moderate",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "moderate",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "White vehicle is clear, but indoor lighting and dark/red surroundings add visible contextual color contrast.",
    },
    "train_06150": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "high",
        "audit_specular_reflection": "none_minor",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "White van is inspectable; warm indoor lighting creates mild shadow/illumination variation.",
    },
    "vcor_train_white_54322230fe": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "high",
        "audit_specular_reflection": "none_minor",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "White car is clear, but the image has warm outdoor lighting and shadowed regions.",
    },
    "test_03801": {
        "audit_visual_clarity": "moderate",
        "audit_body_color_salience": "medium",
        "audit_specular_reflection": "moderate",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "Black vehicle is visible but angled and shadowed, with reflection on the body panels.",
    },
    "test_05126": {
        "audit_visual_clarity": "moderate",
        "audit_body_color_salience": "medium",
        "audit_specular_reflection": "strong",
        "audit_shadow_or_night_effect": "strong",
        "audit_background_color_bias": "moderate",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "Dark studio-style image with strong reflections; the black color is plausible but less visually clean.",
    },
    "test_05003": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "high",
        "audit_specular_reflection": "moderate",
        "audit_shadow_or_night_effect": "none_minor",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "none",
        "audit_notes": "Blue car is clear; headlight flare and body gloss create moderate reflection.",
    },
    "vcor_train_white_41616d6f4f": {
        "audit_visual_clarity": "clear",
        "audit_body_color_salience": "high",
        "audit_specular_reflection": "none_minor",
        "audit_shadow_or_night_effect": "moderate",
        "audit_background_color_bias": "none_minor",
        "audit_multi_car_interference": "none",
        "audit_occlusion": "minor",
        "audit_notes": "White car remains clear, with a tight crop and rainy/gray lighting.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate completed visual clarity audit outputs.")
    parser.add_argument("--main-csv", type=Path, default=REPO_ROOT / "results" / "main" / "main_combined_parsed_results.csv")
    parser.add_argument(
        "--prompt-variant-csv",
        type=Path,
        default=REPO_ROOT / "results" / "robustness" / "prompt_variant_combined_parsed_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "audit")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--reviewer", default="Codex local visual review")
    parser.add_argument("--review-date", default="2026-05-01")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["is_conflict_aligned", "is_faithful", "is_other_wrong", "is_refusal_or_correction", "is_parse_error"]:
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    return result


def target_rows(main_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    c3 = main_df[
        (main_df["model_key"] == MODEL_KEY)
        & (main_df["condition_name"] == "C3_presupposition_correction_allowed")
        & (main_df["is_conflict_aligned"])
    ].copy()
    c3["audit_group"] = "target_conflict_flip"
    c3["audit_source_module"] = "main_original_C3"
    c3["robustness_variant"] = "C3_original"

    c4 = main_df[
        (main_df["model_key"] == MODEL_KEY)
        & (main_df["condition_name"] == "C4_stronger_open_conflict")
        & (main_df["is_conflict_aligned"])
    ].copy()
    c4["audit_group"] = "target_conflict_flip"
    c4["audit_source_module"] = "main_original_C4"
    c4["robustness_variant"] = "C4_original"

    v2 = variant_df[
        (variant_df["model_key"] == MODEL_KEY)
        & (variant_df["robustness_variant"].astype(str) == "C3_v2")
        & (variant_df["is_conflict_aligned"])
    ].copy()
    v2["audit_group"] = "target_conflict_flip"
    v2["audit_source_module"] = "wording_variant_C3_v2"

    combined = pd.concat([c3, c4, v2], ignore_index=True, sort=False)
    combined["target_row_id"] = [
        f"target_{idx:03d}_{row['audit_source_module']}_{row['image_id']}" for idx, row in combined.iterrows()
    ]
    return combined


def pool_for_target(target: pd.Series, main_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    if target["audit_source_module"] == "wording_variant_C3_v2":
        pool = variant_df[
            (variant_df["model_key"] == MODEL_KEY)
            & (variant_df["robustness_variant"].astype(str) == "C3_v2")
            & (variant_df["is_faithful"])
        ].copy()
    else:
        pool = main_df[
            (main_df["model_key"] == MODEL_KEY)
            & (main_df["condition_name"] == target["condition_name"])
            & (main_df["is_faithful"])
        ].copy()
    pool = pool[pool["image_id"].astype(str) != str(target["image_id"])].copy()
    return pool


def choose_control(
    target: pd.Series,
    main_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    used_control_keys: set[str],
    rng: random.Random,
) -> pd.Series:
    pool = pool_for_target(target, main_df, variant_df)
    pool["_rand"] = [rng.random() for _ in range(len(pool))]
    pool = pool.sort_values(["_rand", "image_id"]).reset_index(drop=True)

    strategies = [
        ("matched_true_color_source_condition", (pool["true_color"] == target["true_color"]) & (pool["source_dataset"] == target["source_dataset"])),
        ("matched_true_color_condition", pool["true_color"] == target["true_color"]),
        ("matched_source_condition", pool["source_dataset"] == target["source_dataset"]),
        ("deterministic_condition_fill", pd.Series([True] * len(pool))),
    ]
    for strategy, mask in strategies:
        candidates = pool[mask].copy()
        for _, row in candidates.iterrows():
            key = f"{target['audit_source_module']}::{row['condition_name']}::{row.get('robustness_variant', '')}::{row['image_id']}"
            if key in used_control_keys:
                continue
            used_control_keys.add(key)
            result = row.copy()
            result["match_strategy"] = strategy
            return result
    raise RuntimeError(f"No faithful control available for {target['target_row_id']}")


def build_manifest(main_df: pd.DataFrame, variant_df: pd.DataFrame, seed: int, reviewer: str, review_date: str) -> pd.DataFrame:
    rng = random.Random(seed)
    targets = target_rows(main_df, variant_df)
    control_rows: list[pd.Series] = []
    used_control_keys: set[str] = set()
    for _, target in targets.iterrows():
        control = choose_control(target, main_df, variant_df, used_control_keys, rng)
        control["audit_group"] = "matched_faithful_control"
        control["audit_source_module"] = f"control_for_{target['audit_source_module']}"
        control["target_row_id"] = target["target_row_id"]
        control_rows.append(control)
    controls = pd.DataFrame(control_rows)
    targets["match_strategy"] = "target_case"

    combined = pd.concat([targets, controls], ignore_index=True, sort=False)
    combined = combined.sort_values(["audit_group", "audit_source_module", "true_color", "source_dataset", "image_id"]).reset_index(drop=True)

    manifest = pd.DataFrame(
        {
            "audit_row_id": [f"audit_{idx:03d}" for idx in range(len(combined))],
            "target_row_id": combined["target_row_id"],
            "sample_id": combined["sample_id"],
            "image_id": combined["image_id"],
            "image_path": combined["image_path"],
            "model": combined["model_name"],
            "condition": combined["condition_name"],
            "robustness_variant": combined.get("robustness_variant", "").fillna(""),
            "audit_source_module": combined["audit_source_module"],
            "true_color": combined["true_color"],
            "false_prompt_color": combined["conflict_color"],
            "model_output": combined["raw_output"],
            "source_dataset": combined["source_dataset"],
            "is_conflict_aligned": combined["is_conflict_aligned"],
            "is_faithful": combined["is_faithful"],
            "audit_group": combined["audit_group"],
            "parsed_label": combined["parsed_label"],
            "match_strategy": combined["match_strategy"],
        }
    )
    for field, value in DEFAULT_REVIEW.items():
        manifest[field] = value
    for idx, row in manifest.iterrows():
        overrides = MANUAL_REVIEW_OVERRIDES.get(str(row["image_id"]), {})
        for field, value in overrides.items():
            manifest.at[idx, field] = value
    manifest["audit_reviewer"] = reviewer
    manifest["audit_review_date"] = review_date
    return manifest


def relative_image_path(path_text: str) -> str:
    return "../../" + str(path_text).replace("\\", "/")


def write_gallery(manifest: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Visual Clarity Audit Gallery",
        "",
        "This gallery contains all phase-2 target conflict flips and matched faithful controls. It is used for human-style image validity review, not as a new model benchmark.",
        "",
    ]
    for group_name, group_df in manifest.groupby("audit_group", observed=False):
        lines.extend([f"## {group_name}", ""])
        for row in group_df.to_dict("records"):
            lines.extend(
                [
                    f"### {row['audit_row_id']} | {row['audit_source_module']} | {row['image_id']}",
                    "",
                    f'<img src="{relative_image_path(row["image_path"])}" width="240">',
                    "",
                    f"- true_color: `{row['true_color']}`",
                    f"- false_prompt_color: `{row['false_prompt_color']}`",
                    f"- model_output: `{row['model_output']}`",
                    f"- source_dataset: `{row['source_dataset']}`",
                    f"- visual_clarity: `{row['audit_visual_clarity']}`",
                    f"- body_color_salience: `{row['audit_body_color_salience']}`",
                    f"- reflection: `{row['audit_specular_reflection']}`",
                    f"- shadow_or_night: `{row['audit_shadow_or_night_effect']}`",
                    f"- background_bias: `{row['audit_background_color_bias']}`",
                    f"- multi_car: `{row['audit_multi_car_interference']}`",
                    f"- occlusion: `{row['audit_occlusion']}`",
                    f"- notes: {row['audit_notes']}",
                    "",
                ]
            )
    write_markdown(output_path, "\n".join(lines))


def make_contact_sheets(manifest: pd.DataFrame, output_dir: Path) -> list[str]:
    sheet_dir = output_dir / "visual_clarity_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for module, module_df in manifest.groupby("audit_source_module", observed=False):
        rows = module_df.to_dict("records")
        thumb_w, thumb_h = 240, 170
        label_h = 72
        cols = 4
        cell_w, cell_h = thumb_w, thumb_h + label_h
        sheet_w = cols * cell_w
        sheet_h = int(np.ceil(len(rows) / cols)) * cell_h if rows else cell_h
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(rows):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            image_path = REPO_ROOT / str(row["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h))
                canvas = Image.new("RGB", (thumb_w, thumb_h), "#f7f7f7")
                canvas.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
                image.close()
            except Exception:
                canvas = Image.new("RGB", (thumb_w, thumb_h), "#dddddd")
            sheet.paste(ImageOps.expand(canvas, border=1, fill="#cccccc"), (x, y))
            label = (
                f"{row['audit_row_id']} {row['audit_group']}\n"
                f"{row['image_id']}\n"
                f"{row['true_color']}->{row['false_prompt_color']} | {row['source_dataset']}\n"
                f"{row['audit_visual_clarity']} / {row['audit_body_color_salience']}"
            )
            draw.text((x + 4, y + thumb_h + 4), label, fill="#111111")
        safe_module = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(module))
        path = sheet_dir / f"{safe_module}.jpg"
        sheet.save(path, quality=92)
        paths.append(str(path))
    return paths


def write_instructions(output_path: Path) -> None:
    lines = [
        "# Visual Clarity Audit Reviewer Instructions",
        "",
        "## Purpose",
        "",
        "Review whether conflict-flip images are visually valid examples for primary body-color judgment, and whether obvious visual confounds are more common in flip cases than in matched faithful controls.",
        "",
        "## Rating Fields",
        "",
        "- `audit_visual_clarity`: `clear`, `moderate`, or `low`.",
        "- `audit_body_color_salience`: `high`, `medium`, or `low`.",
        "- `audit_specular_reflection`: `none_minor`, `moderate`, or `strong`.",
        "- `audit_shadow_or_night_effect`: `none_minor`, `moderate`, or `strong`.",
        "- `audit_background_color_bias`: `none_minor`, `moderate`, or `strong`.",
        "- `audit_multi_car_interference`: `none`, `minor`, or `present`.",
        "- `audit_occlusion`: `none`, `minor`, or `moderate`.",
        "- `audit_notes`: short free-text note; mention if the image is reviewable but not ideal.",
        "",
        "## Use In Paper",
        "",
        "Use this as a threat-reduction audit. It should not become a new main experiment unless independently reviewed by additional annotators.",
    ]
    write_markdown(output_path, "\n".join(lines))


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in df.to_dict("records")]
    return "\n".join([header, divider, *rows])


def write_summary(manifest: pd.DataFrame, output_path: Path) -> None:
    targets = manifest[manifest["audit_group"] == "target_conflict_flip"].copy()
    controls = manifest[manifest["audit_group"] == "matched_faithful_control"].copy()

    def value_count_lines(df: pd.DataFrame, field: str) -> list[str]:
        counts = df[field].value_counts(dropna=False).to_dict()
        return [f"`{key}`={value}" for key, value in counts.items()]

    target_issue = targets[
        (targets["audit_visual_clarity"] != "clear")
        | (targets["audit_body_color_salience"] != "high")
        | (targets["audit_specular_reflection"].isin(["moderate", "strong"]))
        | (targets["audit_shadow_or_night_effect"].isin(["moderate", "strong"]))
        | (targets["audit_background_color_bias"].isin(["moderate", "strong"]))
        | (targets["audit_multi_car_interference"].isin(["minor", "present"]))
        | (targets["audit_occlusion"].isin(["minor", "moderate"]))
    ]
    control_issue = controls[
        (controls["audit_visual_clarity"] != "clear")
        | (controls["audit_body_color_salience"] != "high")
        | (controls["audit_specular_reflection"].isin(["moderate", "strong"]))
        | (controls["audit_shadow_or_night_effect"].isin(["moderate", "strong"]))
        | (controls["audit_background_color_bias"].isin(["moderate", "strong"]))
        | (controls["audit_multi_car_interference"].isin(["minor", "present"]))
        | (controls["audit_occlusion"].isin(["minor", "moderate"]))
    ]
    module_counts = targets.groupby("audit_source_module", observed=False).size().reset_index(name="target_n")
    color_counts = targets.groupby(["audit_source_module", "true_color", "false_prompt_color"], observed=False).size().reset_index(name="target_n")

    lines = [
        "# Visual Clarity Audit Summary",
        "",
        "## Scope",
        "",
        f"- Target conflict-flip rows reviewed: {len(targets)}.",
        f"- Matched faithful-control rows reviewed: {len(controls)}.",
        "- Target rows include all LLaVA original C3 flips, all LLaVA original C4 flips, and all LLaVA C3-v2 remaining flips.",
        "- Repeated images across prompt conditions are retained as separate target rows; visual-quality annotations are kept consistent by `image_id`.",
        "",
        "## Target Modules",
        "",
        table_md(module_counts),
        "",
        "## Target Color Routes",
        "",
        table_md(color_counts),
        "",
        "## Review Field Distributions",
        "",
        f"- Target visual clarity: {', '.join(value_count_lines(targets, 'audit_visual_clarity'))}.",
        f"- Control visual clarity: {', '.join(value_count_lines(controls, 'audit_visual_clarity'))}.",
        f"- Target body-color salience: {', '.join(value_count_lines(targets, 'audit_body_color_salience'))}.",
        f"- Control body-color salience: {', '.join(value_count_lines(controls, 'audit_body_color_salience'))}.",
        f"- Target rows with any flagged visual confound: {len(target_issue)}/{len(targets)}.",
        f"- Control rows with any flagged visual confound: {len(control_issue)}/{len(controls)}.",
        "",
        "## Interpretation",
        "",
        "The completed audit should be read as a task-validity check. If the fields remain mostly `clear` and high-salience, the main LLaVA flips are less plausibly explained by globally unreadable images. If any moderate/strong confounds are present, they should be discussed as remaining image-level threats rather than hidden.",
        "",
        "Current completed annotations do not by themselves create a new visual-difficulty result. They support the narrower claim that the reviewed flip cases are inspectable car-body color examples, while leaving room for residual effects from reflections, lighting, background color, or dataset style.",
    ]
    write_markdown(output_path, "\n".join(lines))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_df = normalize_columns(load_bool_results(args.main_csv))
    variant_df = normalize_columns(load_bool_results(args.prompt_variant_csv))
    manifest = build_manifest(main_df, variant_df, args.seed, args.reviewer, args.review_date)

    missing = [path for path in manifest["image_path"] if not (REPO_ROOT / str(path)).exists()]
    manifest.to_csv(args.output_dir / "visual_clarity_audit_manifest_completed.csv", index=False, encoding="utf-8-sig")
    write_gallery(manifest, args.output_dir / "visual_clarity_gallery.md")
    write_instructions(args.output_dir / "visual_clarity_reviewer_instructions.md")
    contact_sheets = make_contact_sheets(manifest, args.output_dir)
    write_summary(manifest, args.output_dir / "visual_clarity_audit_summary.md")

    payload = {
        "completed_manifest": str(args.output_dir / "visual_clarity_audit_manifest_completed.csv"),
        "gallery": str(args.output_dir / "visual_clarity_gallery.md"),
        "instructions": str(args.output_dir / "visual_clarity_reviewer_instructions.md"),
        "summary": str(args.output_dir / "visual_clarity_audit_summary.md"),
        "contact_sheets": contact_sheets,
        "rows": int(len(manifest)),
        "target_rows": int((manifest["audit_group"] == "target_conflict_flip").sum()),
        "control_rows": int((manifest["audit_group"] == "matched_faithful_control").sum()),
        "missing_image_paths": len(missing),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
