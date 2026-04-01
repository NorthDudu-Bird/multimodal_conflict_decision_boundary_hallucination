#!/usr/bin/env python
"""Merge resumable runtime outputs with the source prompt table."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from metadata_paths import (
    BASELINE_PROMPTS_CSV,
    BASELINE_RAW_CSV,
    BASELINE_RUNTIME_CSV,
    LEGACY_BASELINE_PROMPTS_CSV,
    LEGACY_BASELINE_RUNTIME_CSV,
    resolve_existing_path,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_CSV = resolve_existing_path(BASELINE_PROMPTS_CSV, LEGACY_BASELINE_PROMPTS_CSV)
DEFAULT_RUNTIME_CSV = resolve_existing_path(BASELINE_RUNTIME_CSV, LEGACY_BASELINE_RUNTIME_CSV)
DEFAULT_OUTPUT_CSV = BASELINE_RAW_CSV
LEGACY_FIELDS = {"decision_label", "error_type", "is_language_led_bias"}


def merge_fieldnames(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for field in group:
            if not field or field in seen or field in LEGACY_FIELDS:
                continue
            merged.append(field)
            seen.add(field)
    return merged


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [field for field in (reader.fieldnames or []) if field not in LEGACY_FIELDS]
        rows = []
        for row in reader:
            rows.append({field: row.get(field, "") for field in fieldnames})
        return fieldnames, rows


def pick_preferred_row(existing: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    if existing is None:
        return candidate

    existing_status = (existing.get("status", "") or "").strip().lower()
    candidate_status = (candidate.get("status", "") or "").strip().lower()
    existing_ok = existing_status in {"", "ok", "success"}
    candidate_ok = candidate_status in {"", "ok", "success"}

    if candidate_ok and not existing_ok:
        return candidate
    if candidate_ok and existing_ok:
        return candidate
    if not candidate_ok and not existing_ok:
        return candidate
    return existing


def export_raw_results(source_csv: Path, runtime_csv: Path, output_csv: Path) -> dict[str, object]:
    source_fieldnames, source_rows = read_csv_rows(source_csv)
    runtime_fieldnames: list[str] = []
    runtime_rows: list[dict[str, str]] = []
    if runtime_csv.exists():
        runtime_fieldnames, runtime_rows = read_csv_rows(runtime_csv)

    runtime_by_id: dict[str, dict[str, str]] = {}
    for row in runtime_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            continue
        runtime_by_id[sample_id] = pick_preferred_row(runtime_by_id.get(sample_id), row)

    output_rows: list[dict[str, str]] = []
    missing_ids: list[str] = []
    status_counter: Counter[str] = Counter()

    merged_fieldnames = merge_fieldnames(
        source_fieldnames,
        runtime_fieldnames,
        [
            "sample_id",
            "image_id",
            "file_name",
            "image_path",
            "prompt_level",
            "prompt_text",
            "model_name",
            "raw_output",
            "status",
            "error",
        ],
    )

    for source_row in source_rows:
        sample_id = source_row.get("sample_id", "")
        runtime_row = runtime_by_id.get(sample_id, {})
        merged_row = dict(source_row)
        for field in runtime_fieldnames:
            value = runtime_row.get(field, "")
            if value != "" or field not in merged_row:
                merged_row[field] = value

        if not runtime_row:
            merged_row["status"] = "missing_result"
            merged_row["error"] = "No runtime result found for sample_id"
            missing_ids.append(sample_id)
        else:
            merged_row.setdefault("status", runtime_row.get("status", ""))
            merged_row.setdefault("error", runtime_row.get("error", ""))

        status_counter[(merged_row.get("status", "") or "").strip().lower() or "blank"] += 1
        output_rows.append(merged_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in merged_fieldnames})

    return {
        "source_rows": len(source_rows),
        "runtime_rows": len(runtime_rows),
        "output_rows": len(output_rows),
        "status_counts": dict(status_counter),
        "missing_ids": missing_ids,
        "output_csv": str(output_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the formal raw-results table for the first Qwen2-VL round.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--runtime-csv", type=Path, default=DEFAULT_RUNTIME_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_raw_results(args.source_csv, args.runtime_csv, args.output_csv)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
