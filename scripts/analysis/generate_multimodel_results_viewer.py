#!/usr/bin/env python
"""Generate a static HTML preview for the current multimodel Stanford Cars results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_CSV = ROOT / "analysis" / "current" / "strict_colors_primary" / "primary_combined_parsed_results.csv"
DEFAULT_AUXILIARY_CSV = ROOT / "analysis" / "current" / "strict_colors_auxiliary" / "auxiliary_combined_parsed_results.csv"
DEFAULT_MANIFEST_CSV = ROOT / "data" / "processed" / "stanford_cars" / "final_primary_manifest_strict_colors.csv"
DEFAULT_OUTPUT_HTML = ROOT / "reports" / "current" / "strict_colors_multimodel_results_viewer.html"

VIEWER_PRESETS = {
    "v2": {
        "page_title": "V2 Multimodel Results Viewer",
        "page_subtitle": "Used to preview Stanford Cars v2 multimodel results, with image-level grouping and filters for model, family, condition, outcome, color, and keywords.",
        "links": {
            "primarySummary": ROOT / "analysis" / "archived" / "v2_primary" / "analysis_summary.md",
            "auxiliarySummary": ROOT / "analysis" / "archived" / "v2_auxiliary" / "analysis_summary.md",
            "primaryPlot": ROOT / "analysis" / "archived" / "v2_primary" / "plots" / "primary_cross_model_comparison.png",
            "primaryOutcomePlot": ROOT / "analysis" / "archived" / "v2_primary" / "plots" / "primary_outcome_distribution.png",
            "auxiliaryPlot": ROOT / "analysis" / "archived" / "v2_auxiliary" / "plots" / "auxiliary_cross_model_comparison.png",
            "auxiliaryOutcomePlot": ROOT / "analysis" / "archived" / "v2_auxiliary" / "plots" / "auxiliary_outcome_distribution.png",
            "primaryMetrics": ROOT / "analysis" / "archived" / "v2_primary" / "model_condition_metrics.csv",
            "auxiliaryMetrics": ROOT / "analysis" / "archived" / "v2_auxiliary" / "model_condition_metrics.csv",
            "crossModelReadme": ROOT / "analysis" / "archived" / "v2_cross_model" / "README.md",
            "finalManifest": ROOT / "data" / "processed" / "stanford_cars" / "archived" / "final_analysis_manifest.csv",
            "excludedRecords": ROOT / "data" / "processed" / "stanford_cars" / "archived" / "excluded_records.csv",
            "methodDoc": ROOT / "docs" / "archived" / "method_ready_text_v2.md",
            "resultsDoc": ROOT / "docs" / "archived" / "results_ready_summary_v2.md",
        },
    },
    "strict_colors": {
        "page_title": "Strict Colors Multimodel Results Viewer",
        "page_subtitle": "Preview of the strict-colors Stanford Cars rerun, using the narrowed primary color set and exact-match faithful evaluation.",
        "links": {
            "primarySummary": ROOT / "analysis" / "current" / "strict_colors_primary" / "analysis_summary.md",
            "auxiliarySummary": ROOT / "analysis" / "current" / "strict_colors_auxiliary" / "analysis_summary.md",
            "primaryPlot": ROOT / "analysis" / "current" / "strict_colors_primary" / "plots" / "primary_cross_model_comparison.png",
            "primaryOutcomePlot": ROOT / "analysis" / "current" / "strict_colors_primary" / "plots" / "primary_outcome_distribution.png",
            "auxiliaryPlot": ROOT / "analysis" / "current" / "strict_colors_auxiliary" / "plots" / "auxiliary_cross_model_comparison.png",
            "auxiliaryOutcomePlot": ROOT / "analysis" / "current" / "strict_colors_auxiliary" / "plots" / "auxiliary_outcome_distribution.png",
            "primaryMetrics": ROOT / "analysis" / "current" / "strict_colors_primary" / "model_condition_metrics.csv",
            "auxiliaryMetrics": ROOT / "analysis" / "current" / "strict_colors_auxiliary" / "model_condition_metrics.csv",
            "crossModelReadme": ROOT / "analysis" / "current" / "strict_colors_cross_model" / "README.md",
            "finalManifest": ROOT / "data" / "processed" / "stanford_cars" / "final_primary_manifest_strict_colors.csv",
            "excludedRecords": ROOT / "data" / "processed" / "stanford_cars" / "excluded_records_strict_colors.csv",
            "ambiguousExcluded": ROOT / "data" / "processed" / "stanford_cars" / "excluded_due_to_ambiguous_colors.csv",
            "methodDoc": ROOT / "docs" / "current" / "method_ready_text_strict_colors.md",
            "resultsDoc": ROOT / "docs" / "current" / "results_ready_summary_strict_colors.md",
            "designDoc": ROOT / "docs" / "current" / "strict_color_subset_design.md",
        },
    },
    "strict_colors_v3": {
        "page_title": "Strict Colors V3 Multimodel Results Viewer",
        "page_subtitle": "Preview of the expanded v3 strict-colors rerun, using the latest manually cleaned analysis set and auxiliary answer-space compliance outputs.",
        "links": {
            "primarySummary": ROOT / "analysis" / "current" / "primary_v3" / "analysis_summary.md",
            "auxiliarySummary": ROOT / "analysis" / "current" / "auxiliary_v3" / "analysis_summary.md",
            "primaryPlot": ROOT / "analysis" / "current" / "primary_v3" / "plots" / "primary_cross_model_comparison.png",
            "primaryOutcomePlot": ROOT / "analysis" / "current" / "primary_v3" / "plots" / "primary_outcome_distribution.png",
            "auxiliaryPlot": ROOT / "analysis" / "current" / "auxiliary_v3" / "plots" / "auxiliary_cross_model_comparison.png",
            "auxiliaryOutcomePlot": ROOT / "analysis" / "current" / "auxiliary_v3" / "plots" / "auxiliary_outcome_distribution.png",
            "primaryMetrics": ROOT / "analysis" / "current" / "primary_v3" / "model_condition_metrics.csv",
            "auxiliaryMetrics": ROOT / "analysis" / "current" / "auxiliary_v3" / "model_condition_metrics.csv",
            "crossModelReadme": ROOT / "analysis" / "current" / "cross_model_v3" / "README.md",
            "finalManifest": ROOT / "data" / "processed" / "stanford_cars" / "final_primary_manifest_v4_expanded.csv",
            "excludedRecords": ROOT / "data" / "processed" / "stanford_cars" / "excluded_records_v4_expanded.csv",
            "ambiguousExcluded": ROOT / "data" / "processed" / "stanford_cars" / "excluded_manual_review_v3.csv",
            "methodDoc": ROOT / "docs" / "current" / "method_ready_text_v3.md",
            "resultsDoc": ROOT / "docs" / "current" / "results_ready_summary_v3.md",
            "designDoc": ROOT / "docs" / "current" / "EXPERIMENT_STATUS_V3.md",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML preview for the current multimodel Stanford Cars results.")
    parser.add_argument("--viewer-mode", choices=sorted(VIEWER_PRESETS.keys()), default="strict_colors")
    parser.add_argument("--primary-csv", type=Path, default=DEFAULT_PRIMARY_CSV)
    parser.add_argument("--auxiliary-csv", type=Path, default=DEFAULT_AUXILIARY_CSV)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--page-title", default=None)
    parser.add_argument("--page-subtitle", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def clean_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def load_manifest_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {clean_value(row.get("image_id")): row for row in read_rows(path)}


def to_rel(output_html: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), output_html.parent.resolve())).as_posix()


def to_root_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def normalize_row(
    row: dict[str, str],
    source_name: str,
    output_html: Path,
    manifest_map: dict[str, dict[str, str]],
) -> dict[str, object]:
    image_id = clean_value(row.get("image_id"))
    manifest_row = manifest_map.get(image_id, {})

    image_path = clean_value(row.get("image_path")) or clean_value(manifest_row.get("cropped_path"))
    original_path = clean_value(row.get("original_image_path")) or clean_value(manifest_row.get("original_path"))
    notes = clean_value(row.get("notes")) or clean_value(manifest_row.get("notes"))

    try:
        condition_index = int(clean_value(row.get("condition_index")) or "999")
    except ValueError:
        condition_index = 999

    return {
        "sample_id": clean_value(row.get("sample_id")),
        "image_id": image_id,
        "file_name": clean_value(row.get("file_name")) or clean_value(manifest_row.get("file_name")),
        "image_path": image_path,
        "image_href": to_rel(output_html, to_root_path(image_path)) if image_path else "",
        "original_image_path": original_path,
        "original_image_href": to_rel(output_html, to_root_path(original_path)) if original_path else "",
        "experiment_type": clean_value(row.get("experiment_type")),
        "dataset_name": clean_value(row.get("dataset_name")),
        "target_object": clean_value(row.get("target_object")),
        "attribute_type": clean_value(row.get("attribute_type")),
        "truth_source": clean_value(row.get("truth_source")) or clean_value(manifest_row.get("truth_source")),
        "true_color": clean_value(row.get("true_color")) or clean_value(manifest_row.get("true_color")),
        "acceptable_true_colors": clean_value(row.get("acceptable_true_colors")) or clean_value(manifest_row.get("acceptable_true_colors")),
        "conflict_color": clean_value(row.get("conflict_color")) or clean_value(manifest_row.get("conflict_color")),
        "condition_family": clean_value(row.get("condition_family")) or source_name,
        "condition_name": clean_value(row.get("condition_name")),
        "condition_index": condition_index,
        "prompt_template_version": clean_value(row.get("prompt_template_version")),
        "prompt_text": clean_value(row.get("prompt_text")),
        "expected_output_space": clean_value(row.get("expected_output_space")),
        "expected_output_map": clean_value(row.get("expected_output_map")),
        "include_in_primary_main_analysis": clean_value(row.get("include_in_primary_main_analysis"))
        or clean_value(manifest_row.get("include_in_primary_main_analysis")),
        "include_in_v2_auxiliary_analysis": clean_value(manifest_row.get("include_in_v2_auxiliary_analysis")),
        "notes": notes,
        "model_key": clean_value(row.get("model_key")),
        "model_name": clean_value(row.get("model_name")),
        "checkpoint_name": clean_value(row.get("checkpoint_name")),
        "precision": clean_value(row.get("precision")),
        "device": clean_value(row.get("device")),
        "device_map": clean_value(row.get("device_map")),
        "batch_size": clean_value(row.get("batch_size")),
        "elapsed_seconds": clean_value(row.get("elapsed_seconds")),
        "raw_output": clean_value(row.get("raw_output")),
        "status": clean_value(row.get("status")),
        "error": clean_value(row.get("error")),
        "normalized_output": clean_value(row.get("normalized_output")),
        "parsed_label": clean_value(row.get("parsed_label")),
        "parse_success": clean_value(row.get("parse_success")),
        "parse_method": clean_value(row.get("parse_method")),
        "one_label_only": clean_value(row.get("one_label_only")),
        "correction_detected": clean_value(row.get("correction_detected")),
        "refusal_detected": clean_value(row.get("refusal_detected")),
        "outcome_type": clean_value(row.get("outcome_type")),
        "is_conflict_aligned": clean_value(row.get("is_conflict_aligned")),
        "is_faithful": clean_value(row.get("is_faithful")),
        "is_other_wrong": clean_value(row.get("is_other_wrong")),
        "in_allowed_answer_space": clean_value(row.get("in_allowed_answer_space")),
        "is_refusal_or_correction": clean_value(row.get("is_refusal_or_correction")),
        "is_parse_error": clean_value(row.get("is_parse_error")),
        "parse_notes": clean_value(row.get("parse_notes")),
        "source_name": source_name,
        "manifest_review_status": clean_value(manifest_row.get("review_status")),
        "manifest_review_notes": clean_value(manifest_row.get("review_notes")),
        "preliminary_color_guess": clean_value(manifest_row.get("preliminary_color_guess")),
        "prior_issue_flag": clean_value(manifest_row.get("prior_issue_flag")),
        "reviewer_check_needed": clean_value(manifest_row.get("reviewer_check_needed")),
        "exclusion_reason": clean_value(manifest_row.get("exclusion_reason")),
        "selection_bucket": clean_value(manifest_row.get("selection_bucket")),
        "selection_rank": clean_value(manifest_row.get("selection_rank")),
        "quality_score": clean_value(manifest_row.get("quality_score")),
        "class_name": clean_value(manifest_row.get("class_name")),
        "split": clean_value(manifest_row.get("split")),
    }


def load_all_rows(
    primary_csv: Path,
    auxiliary_csv: Path,
    manifest_csv: Path,
    output_html: Path,
) -> list[dict[str, object]]:
    manifest_map = load_manifest_map(manifest_csv)
    rows: list[dict[str, object]] = []

    if primary_csv.exists():
        rows.extend(normalize_row(row, "primary", output_html, manifest_map) for row in read_rows(primary_csv))
    if auxiliary_csv.exists():
        rows.extend(normalize_row(row, "auxiliary", output_html, manifest_map) for row in read_rows(auxiliary_csv))

    if not rows:
        raise FileNotFoundError("No parsed result CSV could be loaded for the requested viewer mode.")

    model_rank = {"qwen2vl7b": 0, "llava15_7b": 1, "internvl2_8b": 2}
    family_rank = {"primary": 0, "auxiliary": 1}
    rows.sort(
        key=lambda row: (
            str(row["image_id"]),
            model_rank.get(str(row["model_key"]), 99),
            family_rank.get(str(row["condition_family"]), 99),
            int(row["condition_index"]),
            str(row["sample_id"]),
        )
    )
    return rows


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    model_counts = Counter(str(row["model_key"]) or "unknown" for row in rows)
    family_counts = Counter(str(row["condition_family"]) or "unknown" for row in rows)
    condition_counts = Counter(str(row["condition_name"]) or "unknown" for row in rows)
    outcome_counts = Counter(str(row["outcome_type"]) or "unknown" for row in rows)
    true_color_counts = Counter(str(row["true_color"]) or "unknown" for row in rows)
    return {
        "total_records": len(rows),
        "unique_images": len({str(row["image_id"]) for row in rows}),
        "model_counts": dict(sorted(model_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "true_color_counts": dict(sorted(true_color_counts.items())),
    }


def build_links(output_html: Path, viewer_mode: str) -> dict[str, str]:
    links = VIEWER_PRESETS[viewer_mode]["links"]
    return {key: to_rel(output_html, path) for key, path in links.items() if path.exists()}


HTML_TEMPLATE = ""
HTML_TEMPLATE += r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #f2efe7;
      --panel: #fffdf8;
      --line: #ddd6c8;
      --ink: #1f2937;
      --muted: #667085;
      --shadow: 0 16px 34px rgba(31, 41, 55, 0.08);
      --primary: #0f766e;
      --aux: #1d4ed8;
      --conflict: #b42318;
      --faithful: #0f766e;
      --wrong: #9a6700;
      --refusal: #7c3aed;
      --parse: #475467;
      --qwen: #14532d;
      --llava: #7c2d12;
      --intern: #1e40af;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 24%),
        radial-gradient(circle at top right, rgba(29,78,216,0.08), transparent 22%),
        var(--bg);
    }
    a { color: inherit; }
    .page { max-width: 1680px; margin: 0 auto; padding: 28px 24px 56px; }
    .hero {
      background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(251,246,237,0.98));
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 24px 26px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }
    h1 { margin: 0 0 8px; font-size: 31px; }
    .subtitle { margin: 0; color: var(--muted); line-height: 1.75; }
    .linkbar { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
    .linkbar a {
      display: inline-flex; align-items: center; padding: 8px 12px;
      border-radius: 999px; border: 1px solid var(--line);
      background: rgba(255,255,255,0.84); text-decoration: none; font-size: 14px;
    }
    .stats {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px; margin-top: 18px;
    }
    .stat {
      border: 1px solid var(--line); border-radius: 18px;
      background: rgba(255,255,255,0.86); padding: 14px 16px;
    }
    .stat .k { color: var(--muted); font-size: 13px; margin-bottom: 4px; }
    .stat .v { font-size: 15px; line-height: 1.6; word-break: break-word; }
    .toolbar {
      position: sticky; top: 0; z-index: 12;
      display: flex; flex-wrap: wrap; gap: 10px; align-items: end;
      padding: 14px 16px; margin-bottom: 18px;
      background: rgba(242,239,231,0.92); backdrop-filter: blur(10px);
      border: 1px solid var(--line); border-radius: 18px;
    }
    .toolbar label {
      display: flex; flex-direction: column; gap: 6px;
      font-size: 13px; color: var(--muted);
    }
    .toolbar select, .toolbar input {
      min-width: 150px; padding: 9px 10px; border-radius: 10px;
      border: 1px solid var(--line); background: white; font-size: 14px;
    }
    .toolbar .grow { flex: 1 1 320px; }
    .toolbar button {
      padding: 9px 12px; border-radius: 10px; border: 1px solid var(--line);
      background: white; cursor: pointer; font-size: 14px; font-weight: 700;
    }
    .toolbar .count { margin-left: auto; font-size: 14px; color: var(--muted); }
    .groups { display: flex; flex-direction: column; gap: 18px; }
    .group {
      border: 1px solid var(--line); border-radius: 22px;
      background: var(--panel); box-shadow: var(--shadow); overflow: hidden;
    }
    .group-top {
      display: grid; grid-template-columns: 320px 1fr; gap: 18px;
      padding: 18px; border-bottom: 1px solid var(--line);
    }
    .thumb-wrap {
      display: flex; align-items: center; justify-content: center;
      min-height: 220px; background: #e7ebf0; border-radius: 18px; overflow: hidden;
    }
    .thumb { width: 100%; height: 260px; object-fit: cover; display: block; }
    .thumb-missing { color: var(--muted); font-size: 14px; text-align: center; padding: 16px; }
    .meta { display: flex; flex-direction: column; gap: 8px; justify-content: center; }
    .group-title { font-size: 22px; font-weight: 700; }
    .meta-line { color: var(--muted); line-height: 1.7; word-break: break-word; }
    .meta-line a { text-decoration: underline; }
    .records { display: flex; flex-direction: column; gap: 14px; padding: 18px; }
    .record {
      border: 1px solid var(--line); border-radius: 18px;
      background: rgba(255,255,255,0.88); padding: 14px;
    }
    .record-head {
      display: flex; justify-content: space-between; gap: 12px;
      align-items: center; margin-bottom: 12px;
    }
    .record-left {
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    }
    .record-right { color: var(--muted); font-size: 13px; }
    .badge {
      display: inline-flex; align-items: center; padding: 4px 10px;
      border-radius: 999px; font-size: 12px; font-weight: 700;
      background: #f3f4f6; color: #111827;
    }
    .badge.primary { background: rgba(15,118,110,0.14); color: var(--primary); }
    .badge.auxiliary { background: rgba(29,78,216,0.12); color: var(--aux); }
    .badge.faithful { background: rgba(15,118,110,0.14); color: var(--faithful); }
    .badge.conflict_aligned { background: rgba(180,35,24,0.12); color: var(--conflict); }
    .badge.other_wrong { background: rgba(154,103,0,0.15); color: var(--wrong); }
    .badge.refusal_or_correction { background: rgba(124,58,237,0.12); color: var(--refusal); }
    .badge.parse_error { background: rgba(71,84,103,0.14); color: var(--parse); }
    .badge.qwen2vl7b { background: rgba(20,83,45,0.12); color: var(--qwen); }
    .badge.llava15_7b { background: rgba(124,45,18,0.12); color: var(--llava); }
    .badge.internvl2_8b { background: rgba(30,64,175,0.12); color: var(--intern); }
    .badge.parsed { background: rgba(31,41,55,0.08); color: #0f172a; }
    .sample-id { color: var(--muted); font-size: 13px; }
    .record-grid {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 12px; margin-bottom: 12px;
    }
    .section-title {
      font-size: 12px; font-weight: 700; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
    }
    .block {
      border: 1px solid var(--line); border-radius: 14px; padding: 12px;
      background: rgba(255,255,255,0.92); line-height: 1.7;
      font-size: 14px; white-space: pre-wrap; word-break: break-word;
    }
    .prompt { background: rgba(29,78,216,0.05); }
    .output { background: rgba(15,118,110,0.05); }
    .meta-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;
    }
    .meta-card {
      border: 1px solid var(--line); border-radius: 14px;
      background: rgba(255,255,255,0.9); padding: 10px 12px;
    }
    .meta-card .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; }
    .meta-card .v { font-size: 14px; line-height: 1.6; word-break: break-word; }
    .empty {
      border: 1px dashed var(--line); border-radius: 16px; padding: 18px;
      color: var(--muted); background: rgba(255,255,255,0.72);
    }
    @media (max-width: 920px) {
      .group-top { grid-template-columns: 1fr; }
      .record-grid { grid-template-columns: 1fr; }
      .thumb { height: 220px; }
    }
    @media (max-width: 720px) {
      .page { padding: 18px 12px 28px; }
      .hero { padding: 20px; border-radius: 18px; }
      h1 { font-size: 26px; }
      .toolbar .count { width: 100%; margin-left: 0; }
      .record-head { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>__PAGE_TITLE__</h1>
      <p class="subtitle">__PAGE_SUBTITLE__</p>
      <div class="linkbar" id="linkbar"></div>
      <div class="stats" id="stats"></div>
    </section>

    <section class="toolbar">
      <label>Model
        <select id="modelFilter"><option value="all">all</option></select>
      </label>
      <label>Study Family
        <select id="familyFilter"><option value="all">all</option></select>
      </label>
      <label>Condition
        <select id="conditionFilter"><option value="all">all</option></select>
      </label>
      <label>Outcome
        <select id="outcomeFilter"><option value="all">all</option></select>
      </label>
      <label>True Color
        <select id="trueColorFilter"><option value="all">all</option></select>
      </label>
      <label>Conflict Color
        <select id="conflictColorFilter"><option value="all">all</option></select>
      </label>
      <label>Parse
        <select id="parseFilter">
          <option value="all">all</option>
          <option value="1">parse_success = 1</option>
          <option value="0">parse_success = 0</option>
        </select>
      </label>
      <label class="grow">Search
        <input id="searchInput" type="text" placeholder="image_id, sample_id, model, parsed label, raw output">
      </label>
      <button id="resetBtn" type="button">Reset</button>
      <div class="count" id="visibleCount"></div>
    </section>

    <section class="groups" id="groups"></section>
  </div>

  <script>
    const RECORDS = __RECORDS_JSON__;
    const SUMMARY = __SUMMARY_JSON__;
    const LINKS = __LINKS_JSON__;

    const modelFilter = document.getElementById('modelFilter');
    const familyFilter = document.getElementById('familyFilter');
    const conditionFilter = document.getElementById('conditionFilter');
    const outcomeFilter = document.getElementById('outcomeFilter');
    const trueColorFilter = document.getElementById('trueColorFilter');
    const conflictColorFilter = document.getElementById('conflictColorFilter');
    const parseFilter = document.getElementById('parseFilter');
    const searchInput = document.getElementById('searchInput');
    const resetBtn = document.getElementById('resetBtn');
    const visibleCount = document.getElementById('visibleCount');
    const groupsEl = document.getElementById('groups');
    const statsEl = document.getElementById('stats');
    const linkbarEl = document.getElementById('linkbar');

    const modelRank = { qwen2vl7b: 0, llava15_7b: 1, internvl2_8b: 2 };
    const familyRank = { primary: 0, auxiliary: 1 };

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function optionList(values) {
      return Array.from(new Set(values.filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b)));
    }

    function populateSelect(selectEl, values) {
      for (const value of optionList(values)) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }
    }

    function renderLinkbar() {
      const items = [
        ['Primary summary', LINKS.primarySummary],
        ['Auxiliary summary', LINKS.auxiliarySummary],
        ['Primary plot', LINKS.primaryPlot],
        ['Primary outcomes', LINKS.primaryOutcomePlot],
        ['Auxiliary plot', LINKS.auxiliaryPlot],
        ['Auxiliary outcomes', LINKS.auxiliaryOutcomePlot],
        ['Primary metrics CSV', LINKS.primaryMetrics],
        ['Auxiliary metrics CSV', LINKS.auxiliaryMetrics],
        ['Cross-model index', LINKS.crossModelReadme],
        ['Final manifest', LINKS.finalManifest],
        ['Excluded records', LINKS.excludedRecords],
        ['Ambiguous exclusions', LINKS.ambiguousExcluded],
        ['Method doc', LINKS.methodDoc],
        ['Results doc', LINKS.resultsDoc],
        ['Design doc', LINKS.designDoc],
      ].filter(item => item[1]);
      linkbarEl.innerHTML = items
        .map(item => `<a href="${escapeHtml(item[1])}" target="_blank" rel="noopener noreferrer">${escapeHtml(item[0])}</a>`)
        .join('');
    }
"""

HTML_TEMPLATE += r"""
    function countBy(records, key) {
      const counts = new Map();
      for (const record of records) {
        const value = String(record[key] || '');
        counts.set(value, (counts.get(value) || 0) + 1);
      }
      return Array.from(counts.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    }

    function formatCounts(records, key) {
      return countBy(records, key).map(item => `${item[0]}=${item[1]}`).join(', ') || 'none';
    }

    function renderStats(records) {
      const uniqueImages = new Set(records.map(record => record.image_id));
      statsEl.innerHTML = [
        ['Visible Records', `${records.length}`],
        ['Visible Images', `${uniqueImages.size}`],
        ['Models', formatCounts(records, 'model_key')],
        ['Outcomes', formatCounts(records, 'outcome_type')],
        ['Families', formatCounts(records, 'condition_family')],
        ['True Colors', formatCounts(records, 'true_color')],
      ].map(item => `
        <div class="stat">
          <div class="k">${escapeHtml(item[0])}</div>
          <div class="v">${escapeHtml(item[1])}</div>
        </div>
      `).join('');
    }

    function matches(record) {
      const search = searchInput.value.trim().toLowerCase();
      if (modelFilter.value !== 'all' && record.model_key !== modelFilter.value) return false;
      if (familyFilter.value !== 'all' && record.condition_family !== familyFilter.value) return false;
      if (conditionFilter.value !== 'all' && record.condition_name !== conditionFilter.value) return false;
      if (outcomeFilter.value !== 'all' && record.outcome_type !== outcomeFilter.value) return false;
      if (trueColorFilter.value !== 'all' && record.true_color !== trueColorFilter.value) return false;
      if (conflictColorFilter.value !== 'all' && record.conflict_color !== conflictColorFilter.value) return false;
      if (parseFilter.value !== 'all' && String(record.parse_success) !== parseFilter.value) return false;

      if (!search) return true;
      const haystack = [
        record.image_id,
        record.sample_id,
        record.model_key,
        record.model_name,
        record.condition_name,
        record.true_color,
        record.conflict_color,
        record.parsed_label,
        record.raw_output,
        record.prompt_text,
        record.notes,
        record.manifest_review_notes,
      ].join(' ').toLowerCase();
      return haystack.includes(search);
    }

    function formatMaybe(value, fallback='-') {
      const text = String(value ?? '').trim();
      return text ? escapeHtml(text) : fallback;
    }

    function renderRecord(record) {
      const elapsed = record.elapsed_seconds ? `${Number(record.elapsed_seconds).toFixed(2)}s` : '-';
      return `
        <article class="record">
          <div class="record-head">
            <div class="record-left">
              <span class="badge ${escapeHtml(record.model_key)}">${escapeHtml(record.model_key)}</span>
              <span class="badge ${escapeHtml(record.condition_family)}">${escapeHtml(record.condition_family)}</span>
              <span class="badge ${escapeHtml(record.outcome_type)}">${escapeHtml(record.outcome_type)}</span>
              <span class="badge parsed">${formatMaybe(record.parsed_label)}</span>
              <span class="sample-id">${escapeHtml(record.sample_id)}</span>
            </div>
            <div class="record-right">elapsed=${escapeHtml(elapsed)} | parse_method=${formatMaybe(record.parse_method)}</div>
          </div>

          <div class="record-grid">
            <div>
              <div class="section-title">Prompt</div>
              <div class="block prompt">${formatMaybe(record.prompt_text)}</div>
            </div>
            <div>
              <div class="section-title">Raw Output</div>
              <div class="block output">${formatMaybe(record.raw_output)}</div>
            </div>
          </div>

          <div class="meta-grid">
            <div class="meta-card"><div class="k">Condition</div><div class="v">${formatMaybe(record.condition_name)}</div></div>
            <div class="meta-card"><div class="k">True / Conflict</div><div class="v">${formatMaybe(record.true_color)} / ${formatMaybe(record.conflict_color)}</div></div>
            <div class="meta-card"><div class="k">Model</div><div class="v">${formatMaybe(record.model_name)}</div></div>
            <div class="meta-card"><div class="k">Checkpoint</div><div class="v">${formatMaybe(record.checkpoint_name)}</div></div>
            <div class="meta-card"><div class="k">Precision / Device</div><div class="v">${formatMaybe(record.precision)} / ${formatMaybe(record.device)}</div></div>
            <div class="meta-card"><div class="k">Status</div><div class="v">${formatMaybe(record.status)}</div></div>
            <div class="meta-card"><div class="k">Parse Success</div><div class="v">${formatMaybe(record.parse_success)} | one_label_only=${formatMaybe(record.one_label_only)}</div></div>
            <div class="meta-card"><div class="k">Allowed Answer Space</div><div class="v">${formatMaybe(record.in_allowed_answer_space)}</div></div>
            <div class="meta-card"><div class="k">Expected Output Space</div><div class="v">${formatMaybe(record.expected_output_space)}</div></div>
          </div>
        </article>
      `;
    }

    function renderGroup(imageId, records) {
      const sorted = records.slice().sort((a, b) => {
        const modelDiff = (modelRank[a.model_key] ?? 99) - (modelRank[b.model_key] ?? 99);
        if (modelDiff !== 0) return modelDiff;
        const familyDiff = (familyRank[a.condition_family] ?? 99) - (familyRank[b.condition_family] ?? 99);
        if (familyDiff !== 0) return familyDiff;
        return (Number(a.condition_index) || 999) - (Number(b.condition_index) || 999);
      });
      const first = sorted[0];
      const outcomeSummary = formatCounts(sorted, 'outcome_type');
      const cropLink = first.image_href ? `<a href="${escapeHtml(first.image_href)}" target="_blank" rel="noopener noreferrer">crop</a>` : '';
      const originalLink = first.original_image_href ? `<a href="${escapeHtml(first.original_image_href)}" target="_blank" rel="noopener noreferrer">original</a>` : '';
      const linkSummary = [cropLink, originalLink].filter(Boolean).join(' | ') || '-';
      const thumbHtml = first.image_href
        ? `<a href="${escapeHtml(first.image_href)}" target="_blank" rel="noopener noreferrer"><img class="thumb" src="${escapeHtml(first.image_href)}" alt="${escapeHtml(first.image_id)}"></a>`
        : `<div class="thumb-missing">Image preview missing</div>`;

      return `
        <section class="group">
          <div class="group-top">
            <div class="thumb-wrap">${thumbHtml}</div>
            <div class="meta">
              <div class="group-title">${escapeHtml(imageId)}</div>
              <div class="meta-line">class=${formatMaybe(first.class_name)} | split=${formatMaybe(first.split)} | truth=${formatMaybe(first.truth_source)}</div>
              <div class="meta-line">true_color=${formatMaybe(first.true_color)} | conflict_color=${formatMaybe(first.conflict_color)} | acceptable_true_colors=${formatMaybe(first.acceptable_true_colors)}</div>
              <div class="meta-line">prior_issue_flag=${formatMaybe(first.prior_issue_flag)} | reviewer_check_needed=${formatMaybe(first.reviewer_check_needed)} | include_in_primary_main_analysis=${formatMaybe(first.include_in_primary_main_analysis)}</div>
              <div class="meta-line">review_status=${formatMaybe(first.manifest_review_status)} | preliminary_color_guess=${formatMaybe(first.preliminary_color_guess)}</div>
              <div class="meta-line">links=${linkSummary}</div>
              <div class="meta-line">outcomes in current view: ${escapeHtml(outcomeSummary)}</div>
              <div class="meta-line">notes=${formatMaybe(first.notes)}${first.manifest_review_notes ? ' | review_notes=' + escapeHtml(first.manifest_review_notes) : ''}</div>
            </div>
          </div>
          <div class="records">${sorted.map(renderRecord).join('')}</div>
        </section>
      `;
    }
"""

HTML_TEMPLATE += r"""
    function render() {
      const filtered = RECORDS.filter(matches);
      const grouped = new Map();
      for (const record of filtered) {
        if (!grouped.has(record.image_id)) grouped.set(record.image_id, []);
        grouped.get(record.image_id).push(record);
      }

      const imageIds = Array.from(grouped.keys()).sort((a, b) => String(a).localeCompare(String(b)));
      visibleCount.textContent = `${filtered.length} records / ${imageIds.length} images`;
      renderStats(filtered);

      if (!filtered.length) {
        groupsEl.innerHTML = '<div class="empty">当前筛选条件下没有匹配记录。</div>';
        return;
      }

      groupsEl.innerHTML = imageIds.map(imageId => renderGroup(imageId, grouped.get(imageId))).join('');
    }

    function resetFilters() {
      modelFilter.value = 'all';
      familyFilter.value = 'all';
      conditionFilter.value = 'all';
      outcomeFilter.value = 'all';
      trueColorFilter.value = 'all';
      conflictColorFilter.value = 'all';
      parseFilter.value = 'all';
      searchInput.value = '';
      render();
    }

    populateSelect(modelFilter, RECORDS.map(record => record.model_key));
    populateSelect(familyFilter, RECORDS.map(record => record.condition_family));
    populateSelect(conditionFilter, RECORDS.map(record => record.condition_name));
    populateSelect(outcomeFilter, RECORDS.map(record => record.outcome_type));
    populateSelect(trueColorFilter, RECORDS.map(record => record.true_color));
    populateSelect(conflictColorFilter, RECORDS.map(record => record.conflict_color));

    [modelFilter, familyFilter, conditionFilter, outcomeFilter, trueColorFilter, conflictColorFilter, parseFilter]
      .forEach(el => el.addEventListener('change', render));
    searchInput.addEventListener('input', render);
    resetBtn.addEventListener('click', resetFilters);

    renderLinkbar();
    renderStats(RECORDS);
    render();
  </script>
</body>
</html>
"""


def generate_html(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    links: dict[str, str],
    page_title: str,
    page_subtitle: str,
) -> str:
    return (
        HTML_TEMPLATE.replace("__RECORDS_JSON__", json.dumps(rows, ensure_ascii=False))
        .replace("__SUMMARY_JSON__", json.dumps(summary, ensure_ascii=False))
        .replace("__LINKS_JSON__", json.dumps(links, ensure_ascii=False))
        .replace("__PAGE_TITLE__", page_title)
        .replace("__PAGE_SUBTITLE__", page_subtitle)
    )


def main() -> None:
    args = parse_args()
    preset = VIEWER_PRESETS[args.viewer_mode]
    page_title = str(args.page_title or preset["page_title"])
    page_subtitle = str(args.page_subtitle or preset["page_subtitle"])
    rows = load_all_rows(args.primary_csv, args.auxiliary_csv, args.manifest_csv, args.output_html)
    summary = build_summary(rows)
    links = build_links(args.output_html, viewer_mode=args.viewer_mode)
    html = generate_html(rows, summary, links, page_title=page_title, page_subtitle=page_subtitle)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
