#!/usr/bin/env python
"""Parse one-label car-color outputs for the restructured prompt-mechanism study."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.restructured_experiment_utils import (
    build_alias_lookup,
    build_logger,
    canonicalize_color,
    classify_exact_outcome,
    clean_label_text,
    detect_color_mentions,
    is_primary_condition,
    json_dumps,
    load_config,
    looks_like_correction,
    looks_like_refusal,
    read_rows,
    relative_str,
    repo_path,
    write_rows,
)


ROOT = REPO_ROOT
OK_STATUSES = {"", "ok", "success"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse raw runtime rows into structured one-label outcomes.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def output_fieldnames(source_fields: list[str]) -> list[str]:
    extra_fields = [
        "normalized_output",
        "allowed_answer_space",
        "parsed_label",
        "parse_success",
        "parse_method",
        "one_label_only",
        "in_allowed_answer_space",
        "correction_detected",
        "refusal_detected",
        "outcome_type",
        "is_conflict_aligned",
        "is_faithful",
        "is_other_wrong",
        "is_refusal_or_correction",
        "is_parse_error",
        "parse_notes",
    ]
    merged = list(source_fields)
    for field in extra_fields:
        if field not in merged:
            merged.append(field)
    return merged


def load_output_map(row: dict[str, str]) -> dict[str, str]:
    raw = row.get("expected_output_map", "") or "{}"
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected_output_map must be a dict JSON string, got: {raw}")
    return {str(key): str(value) for key, value in parsed.items()}


def load_output_space(row: dict[str, str]) -> list[str]:
    raw = row.get("expected_output_space", "") or "[]"
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"expected_output_space must be a list JSON string, got: {raw}")
    return [str(item) for item in parsed]


def in_allowed_answer_space(normalized_output: str, output_space: list[str]) -> bool:
    if not normalized_output:
        return False
    allowed_normalized = {clean_label_text(label) for label in output_space}
    return normalized_output in allowed_normalized


def exact_single_label(cleaned_output: str, alias_lookup: dict[str, str]) -> tuple[str, bool]:
    if cleaned_output in alias_lookup:
        return alias_lookup[cleaned_output], True
    return "", False


def pick_single_label_from_mentions(
    row: dict[str, str],
    mention_counts: Counter[str],
    correction_detected: bool,
) -> tuple[str, str]:
    if not mention_counts:
        return "", ""
    if len(mention_counts) == 1:
        return next(iter(mention_counts)), "single_mentioned_label"

    true_color = row.get("true_color", "")
    conflict_color = row.get("conflict_color", "")
    if correction_detected:
        if true_color and true_color in mention_counts:
            return true_color, "correction_true_color"
        non_conflict = [label for label in mention_counts if label != conflict_color]
        if len(non_conflict) == 1:
            return non_conflict[0], "correction_non_conflict_label"
    return "", ""


def classify_parsed_row(row: dict[str, str]) -> dict[str, str]:
    status = (row.get("status", "") or "").strip().lower()
    raw_output = row.get("raw_output", "") or row.get("model_output", "") or ""
    normalized_output = clean_label_text(raw_output)
    correction_detected = looks_like_correction(raw_output)
    refusal_detected = looks_like_refusal(raw_output)
    true_color = row.get("true_color", "")
    conflict_color = row.get("conflict_color", "")
    condition_name = row.get("condition_name", "")

    base = {
        "normalized_output": normalized_output,
        "allowed_answer_space": "",
        "parsed_label": "",
        "parse_success": "0",
        "parse_method": "",
        "one_label_only": "0",
        "in_allowed_answer_space": "0",
        "correction_detected": "1" if correction_detected else "0",
        "refusal_detected": "1" if refusal_detected else "0",
        "outcome_type": "parse_error",
        "is_conflict_aligned": "0",
        "is_faithful": "0",
        "is_other_wrong": "0",
        "is_refusal_or_correction": "0",
        "is_parse_error": "1",
        "parse_notes": "",
    }

    if status not in OK_STATUSES:
        base["parse_notes"] = f"runtime_status={row.get('status', '')}; error={row.get('error', '')}".strip("; ")
        return base

    if not normalized_output:
        base["parse_notes"] = "empty_output"
        return base

    output_space = load_output_space(row)
    output_map = load_output_map(row)
    alias_lookup = build_alias_lookup(output_map)
    base["allowed_answer_space"] = json_dumps(output_space)
    base["in_allowed_answer_space"] = "1" if in_allowed_answer_space(normalized_output, output_space) else "0"
    allowed_canonical_labels = {canonicalize_color(value) for value in output_map.values()}
    parsed_label, matched = exact_single_label(normalized_output, alias_lookup)
    parse_method = "exact_single_label" if matched else ""
    one_label_only = matched

    if not matched:
        mention_counts = detect_color_mentions(raw_output, alias_lookup)
        parsed_label, parse_method = pick_single_label_from_mentions(row, mention_counts, correction_detected)
        if parsed_label:
            matched = True
        else:
            parsed_label = ""

    if matched:
        base["parsed_label"] = parsed_label
        base["parse_success"] = "1"
        base["parse_method"] = parse_method
        base["one_label_only"] = "1" if one_label_only else "0"
        nonstandard_label = parsed_label not in allowed_canonical_labels
        if nonstandard_label:
            base["parse_notes"] = "parsed_nonstandard_color_label"

        if (correction_detected and not one_label_only and condition_name != "C0_neutral") or (refusal_detected and not one_label_only):
            base["outcome_type"] = "refusal_or_correction"
            base["is_refusal_or_correction"] = "1"
            base["is_parse_error"] = "0"
            base["parse_notes"] = "label parsed, but output corrected/refused the prompt rather than following one-label formatting."
            return base

        outcome_type = classify_exact_outcome(parsed_label, true_color, conflict_color)
        if outcome_type == "faithful":
            base["outcome_type"] = "faithful"
            base["is_faithful"] = "1"
            base["is_parse_error"] = "0"
            return base

        if outcome_type == "conflict_aligned":
            base["outcome_type"] = "conflict_aligned"
            base["is_conflict_aligned"] = "1"
            base["is_parse_error"] = "0"
            return base

        base["outcome_type"] = "other_wrong"
        base["is_other_wrong"] = "1"
        base["is_parse_error"] = "0"
        return base

    if refusal_detected or (correction_detected and is_primary_condition(condition_name)):
        base["outcome_type"] = "refusal_or_correction"
        base["is_refusal_or_correction"] = "1"
        base["is_parse_error"] = "0"
        base["parse_notes"] = "no reliable label parsed, but the output refused or explicitly corrected the prompt."
        return base

    base["parse_notes"] = "no_reliable_label_parsed"
    return base


def build_outputs(rows: list[dict[str, str]], output_csv: Path, review_csv: Path, summary_md: Path) -> dict[str, object]:
    parsed_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    outcome_counter: Counter[str] = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        parsed = dict(row)
        parsed.update(classify_parsed_row(row))
        parsed_rows.append(parsed)
        outcome_type = parsed.get("outcome_type", "")
        condition_name = parsed.get("condition_name", "")
        outcome_counter[outcome_type] += 1
        by_condition[condition_name][outcome_type] += 1
        if outcome_type in {"parse_error", "refusal_or_correction"}:
            review_rows.append(parsed)

    write_rows(output_csv, output_fieldnames(list(rows[0].keys()) if rows else []), parsed_rows)
    write_rows(review_csv, output_fieldnames(list(rows[0].keys()) if rows else []), review_rows)

    lines = [
        "# Restructured Car-Color Parse Summary",
        "",
        f"- input_rows: {len(rows)}",
        f"- parsed_rows: {len(parsed_rows)}",
        f"- review_rows: {len(review_rows)}",
        "",
        "## Outcome Counts",
    ]
    for outcome_type, count in sorted(outcome_counter.items()):
        lines.append(f"- {outcome_type}: {count}")
    lines.extend(["", "## By Condition"])
    for condition_name in sorted(by_condition):
        parts = ", ".join(f"{outcome}={count}" for outcome, count in sorted(by_condition[condition_name].items()))
        lines.append(f"- {condition_name}: {parts}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "input_rows": len(rows),
        "review_rows": len(review_rows),
        "outcome_counts": dict(outcome_counter),
        "output_csv": relative_str(output_csv),
        "review_csv": relative_str(review_csv),
        "summary_md": relative_str(summary_md),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    log_path = args.log_path or repo_path(config["outputs"]["main_dir"]) / "parse.log"
    logger = build_logger("parse_restructured_car_color_outputs", log_path)
    logger.info("Parsing raw results: %s", args.input_csv)

    rows = read_rows(args.input_csv)
    summary = build_outputs(rows, output_csv=args.output_csv, review_csv=args.review_csv, summary_md=args.summary_md)
    logger.info("Parse summary: %s", json_dumps(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
